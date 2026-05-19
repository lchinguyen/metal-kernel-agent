import Foundation
import Metal

private let expectedElementCount = 4096

private struct SoftmaxParams {
    var elementCount: UInt32
    var rowCount: UInt32
}

private struct VariantConfig {
    let name: String
    let kernelName: String
    let threadgroupSize: Int
}

private struct MetalRunResult: Codable {
    let variant_name: String
    let device_name: String
    let kernel_name: String
    let threadgroup_size: Int
    let thread_execution_width: Int
    let max_total_threads_per_threadgroup: Int
    let wall_latencies_ms: [Double]
    let gpu_latencies_ms: [Double]
    let output: [Float]
}

private enum RunnerError: Error, CustomStringConvertible {
    case usage(String)
    case fileRead(String)
    case invalidBinarySize(String)
    case unknownVariant(String)
    case noMetalDevice
    case commandQueueCreation
    case libraryCreation(String)
    case functionCreation(String)
    case pipelineCreation(String)
    case bufferCreation(String)
    case commandBufferCreation
    case encoderCreation
    case jsonEncoding

    var description: String {
        switch self {
        case let .usage(message):
            return message
        case let .fileRead(message):
            return message
        case let .invalidBinarySize(message):
            return message
        case let .unknownVariant(name):
            return "Unknown Metal Softmax variant '\(name)'."
        case .noMetalDevice:
            return "Metal device unavailable. MTLCreateSystemDefaultDevice() returned nil."
        case .commandQueueCreation:
            return "Failed to create Metal command queue."
        case let .libraryCreation(message):
            return "Failed to compile Metal library: \(message)"
        case let .functionCreation(name):
            return "Failed to create Metal function '\(name)'."
        case let .pipelineCreation(message):
            return "Failed to create compute pipeline: \(message)"
        case let .bufferCreation(label):
            return "Failed to create Metal buffer for \(label)."
        case .commandBufferCreation:
            return "Failed to create command buffer."
        case .encoderCreation:
            return "Failed to create compute command encoder."
        case .jsonEncoding:
            return "Failed to encode JSON result."
        }
    }
}

private let variantConfigs: [String: VariantConfig] = [
    "shared_256": VariantConfig(name: "shared_256", kernelName: "softmax_256_shared_f32", threadgroupSize: 256),
    "simd_256": VariantConfig(name: "simd_256", kernelName: "softmax_256_simd_f32", threadgroupSize: 256),
]

private func parseArguments() throws -> (variant: VariantConfig, kernelPath: String, inputPath: String, warmupIters: Int, timedIters: Int) {
    let arguments = Array(CommandLine.arguments.dropFirst())
    var values: [String: String] = [:]
    var index = 0
    while index < arguments.count {
        let argument = arguments[index]
        guard argument.hasPrefix("--"), index + 1 < arguments.count else {
            throw RunnerError.usage("Expected flag/value pairs, received '\(argument)'.")
        }
        values[argument] = arguments[index + 1]
        index += 2
    }

    guard
        let variantName = values["--variant"],
        let variant = variantConfigs[variantName],
        let kernelPath = values["--kernel"],
        let inputPath = values["--input-bin"],
        let warmupString = values["--warmup-iters"],
        let warmupIters = Int(warmupString),
        let timedString = values["--timed-iters"],
        let timedIters = Int(timedString)
    else {
        if let variantName = values["--variant"], variantConfigs[variantName] == nil {
            throw RunnerError.unknownVariant(variantName)
        }
        throw RunnerError.usage(
            "Usage: softmax_metal_helper.swift --variant <shared_256|simd_256> --kernel <path> --input-bin <path> --warmup-iters <int> --timed-iters <int>"
        )
    }

    return (variant, kernelPath, inputPath, warmupIters, timedIters)
}

private func loadFloatArray(from path: String) throws -> [Float] {
    let url = URL(fileURLWithPath: path)
    let data: Data
    do {
        data = try Data(contentsOf: url)
    } catch {
        throw RunnerError.fileRead("Failed to read \(path): \(error.localizedDescription)")
    }

    guard data.count % MemoryLayout<Float>.stride == 0 else {
        throw RunnerError.invalidBinarySize("File \(path) length \(data.count) is not a multiple of 4 bytes.")
    }

    return data.withUnsafeBytes { rawBuffer in
        Array(rawBuffer.bindMemory(to: Float.self))
    }
}

private func makeSharedBuffer(device: MTLDevice, floats: [Float], label: String) throws -> MTLBuffer {
    let byteCount = floats.count * MemoryLayout<Float>.stride
    guard
        let buffer = floats.withUnsafeBytes({
            device.makeBuffer(bytes: $0.baseAddress!, length: byteCount, options: .storageModeShared)
        })
    else {
        throw RunnerError.bufferCreation(label)
    }
    return buffer
}

private func makeZeroedBuffer(device: MTLDevice, floatCount: Int, label: String) throws -> MTLBuffer {
    let byteCount = floatCount * MemoryLayout<Float>.stride
    guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
        throw RunnerError.bufferCreation(label)
    }
    return buffer
}

private func encodeAndRun(
    commandQueue: MTLCommandQueue,
    pipelineState: MTLComputePipelineState,
    inputBuffer: MTLBuffer,
    outputBuffer: MTLBuffer,
    paramsBuffer: MTLBuffer,
    threadgroupSize: Int,
    rowCount: Int
) throws -> (wallLatencyMs: Double, gpuLatencyMs: Double) {
    let wallStart = DispatchTime.now().uptimeNanoseconds
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
        throw RunnerError.commandBufferCreation
    }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw RunnerError.encoderCreation
    }

    encoder.setComputePipelineState(pipelineState)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(outputBuffer, offset: 0, index: 1)
    encoder.setBuffer(paramsBuffer, offset: 0, index: 2)

    let threadsPerThreadgroup = MTLSize(width: threadgroupSize, height: 1, depth: 1)
    let threadgroupsPerGrid = MTLSize(width: rowCount, height: 1, depth: 1)
    encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    encoder.endEncoding()

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    let wallEnd = DispatchTime.now().uptimeNanoseconds
    let wallLatencyMs = Double(wallEnd - wallStart) / 1_000_000.0

    let gpuLatencyMs: Double
    if commandBuffer.gpuEndTime > commandBuffer.gpuStartTime {
        gpuLatencyMs = (commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1_000.0
    } else {
        gpuLatencyMs = 0.0
    }

    return (wallLatencyMs, gpuLatencyMs)
}

private func readOutput(from outputBuffer: MTLBuffer, floatCount: Int) -> [Float] {
    let pointer = outputBuffer.contents().bindMemory(to: Float.self, capacity: floatCount)
    let buffer = UnsafeBufferPointer(start: pointer, count: floatCount)
    return Array(buffer)
}

private func main() throws {
    let config = try parseArguments()
    let input = try loadFloatArray(from: config.inputPath)

    guard input.count % expectedElementCount == 0 else {
        throw RunnerError.invalidBinarySize("Input element count must be a multiple of \(expectedElementCount), received \(input.count).")
    }
    let rowCount = input.count / expectedElementCount

    guard let device = MTLCreateSystemDefaultDevice() else {
        throw RunnerError.noMetalDevice
    }
    guard let commandQueue = device.makeCommandQueue() else {
        throw RunnerError.commandQueueCreation
    }

    let kernelSource: String
    do {
        kernelSource = try String(contentsOfFile: config.kernelPath, encoding: .utf8)
    } catch {
        throw RunnerError.fileRead("Failed to read kernel source \(config.kernelPath): \(error.localizedDescription)")
    }

    let library: MTLLibrary
    do {
        library = try device.makeLibrary(source: kernelSource, options: nil)
    } catch {
        throw RunnerError.libraryCreation(error.localizedDescription)
    }

    guard let function = library.makeFunction(name: config.variant.kernelName) else {
        throw RunnerError.functionCreation(config.variant.kernelName)
    }

    let pipelineState: MTLComputePipelineState
    do {
        pipelineState = try device.makeComputePipelineState(function: function)
    } catch {
        throw RunnerError.pipelineCreation(error.localizedDescription)
    }

    let inputBuffer = try makeSharedBuffer(device: device, floats: input, label: "input")
    let outputBuffer = try makeZeroedBuffer(device: device, floatCount: input.count, label: "output")

    var params = SoftmaxParams(
        elementCount: UInt32(expectedElementCount),
        rowCount: UInt32(rowCount)
    )
    guard let paramsBuffer = device.makeBuffer(
        bytes: &params,
        length: MemoryLayout<SoftmaxParams>.stride,
        options: .storageModeShared
    ) else {
        throw RunnerError.bufferCreation("params")
    }

    for _ in 0..<config.warmupIters {
        _ = try encodeAndRun(
            commandQueue: commandQueue,
            pipelineState: pipelineState,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            paramsBuffer: paramsBuffer,
            threadgroupSize: config.variant.threadgroupSize,
            rowCount: rowCount
        )
    }

    var wallLatenciesMs: [Double] = []
    wallLatenciesMs.reserveCapacity(config.timedIters)
    var gpuLatenciesMs: [Double] = []
    gpuLatenciesMs.reserveCapacity(config.timedIters)
    for _ in 0..<config.timedIters {
        let result = try encodeAndRun(
            commandQueue: commandQueue,
            pipelineState: pipelineState,
            inputBuffer: inputBuffer,
            outputBuffer: outputBuffer,
            paramsBuffer: paramsBuffer,
            threadgroupSize: config.variant.threadgroupSize,
            rowCount: rowCount
        )
        wallLatenciesMs.append(result.wallLatencyMs)
        gpuLatenciesMs.append(result.gpuLatencyMs)
    }

    let result = MetalRunResult(
        variant_name: config.variant.name,
        device_name: device.name,
        kernel_name: config.variant.kernelName,
        threadgroup_size: config.variant.threadgroupSize,
        thread_execution_width: pipelineState.threadExecutionWidth,
        max_total_threads_per_threadgroup: pipelineState.maxTotalThreadsPerThreadgroup,
        wall_latencies_ms: wallLatenciesMs,
        gpu_latencies_ms: gpuLatenciesMs,
        output: readOutput(from: outputBuffer, floatCount: input.count)
    )

    let encoder = JSONEncoder()
    guard let jsonData = try? encoder.encode(result) else {
        throw RunnerError.jsonEncoding
    }
    FileHandle.standardOutput.write(jsonData)
}

do {
    try main()
} catch {
    fputs("\(error)\n", stderr)
    exit(1)
}
