import Foundation
import Metal

private let expectedElementCount = 4096
private let threadgroupSize = 256

private struct RMSNormParams {
    var elementCount: UInt32
    var rowCount: UInt32
    var eps: Float
}

private struct ModeConfig {
    let name: String
}

private struct MetalRunResult: Codable {
    let mode: String
    let device_name: String
    let primary_kernel_name: String
    let secondary_kernel_name: String?
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
    case unknownMode(String)
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
        case let .unknownMode(name):
            return "Unknown fused RMSNorm mode '\(name)'."
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

private let modeConfigs: [String: ModeConfig] = [
    "standalone_rmsnorm": ModeConfig(name: "standalone_rmsnorm"),
    "separate_rmsnorm_residual": ModeConfig(name: "separate_rmsnorm_residual"),
    "fused_rmsnorm_residual": ModeConfig(name: "fused_rmsnorm_residual"),
]

private func parseArguments() throws -> (mode: ModeConfig, kernelPath: String, inputPath: String, weightPath: String, residualPath: String, eps: Float, warmupIters: Int, timedIters: Int) {
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
        let modeName = values["--mode"],
        let mode = modeConfigs[modeName],
        let kernelPath = values["--kernel"],
        let inputPath = values["--input-bin"],
        let weightPath = values["--weight-bin"],
        let residualPath = values["--residual-bin"],
        let epsString = values["--eps"],
        let eps = Float(epsString),
        let warmupString = values["--warmup-iters"],
        let warmupIters = Int(warmupString),
        let timedString = values["--timed-iters"],
        let timedIters = Int(timedString)
    else {
        if let modeName = values["--mode"], modeConfigs[modeName] == nil {
            throw RunnerError.unknownMode(modeName)
        }
        throw RunnerError.usage(
            "Usage: fused_rmsnorm_residual_helper.swift --mode <standalone_rmsnorm|separate_rmsnorm_residual|fused_rmsnorm_residual> --kernel <path> --input-bin <path> --weight-bin <path> --residual-bin <path> --eps <float> --warmup-iters <int> --timed-iters <int>"
        )
    }

    return (mode, kernelPath, inputPath, weightPath, residualPath, eps, warmupIters, timedIters)
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

private func encodeKernel(
    encoder: MTLComputeCommandEncoder,
    pipelineState: MTLComputePipelineState,
    rowCount: Int,
    buffers: [(MTLBuffer, Int)]
) {
    encoder.setComputePipelineState(pipelineState)
    for (buffer, index) in buffers {
        encoder.setBuffer(buffer, offset: 0, index: index)
    }
    let threadsPerThreadgroup = MTLSize(width: threadgroupSize, height: 1, depth: 1)
    let threadgroupsPerGrid = MTLSize(width: rowCount, height: 1, depth: 1)
    encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
}

private func encodeAndRun(
    mode: String,
    commandQueue: MTLCommandQueue,
    rmsnormPipeline: MTLComputePipelineState,
    residualPipeline: MTLComputePipelineState,
    fusedPipeline: MTLComputePipelineState,
    inputBuffer: MTLBuffer,
    weightBuffer: MTLBuffer,
    residualBuffer: MTLBuffer,
    scratchBuffer: MTLBuffer,
    outputBuffer: MTLBuffer,
    paramsBuffer: MTLBuffer,
    rowCount: Int
) throws -> (wallLatencyMs: Double, gpuLatencyMs: Double) {
    let wallStart = DispatchTime.now().uptimeNanoseconds
    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
        throw RunnerError.commandBufferCreation
    }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw RunnerError.encoderCreation
    }

    switch mode {
    case "standalone_rmsnorm":
        encodeKernel(
            encoder: encoder,
            pipelineState: rmsnormPipeline,
            rowCount: rowCount,
            buffers: [
                (inputBuffer, 0),
                (weightBuffer, 1),
                (outputBuffer, 2),
                (paramsBuffer, 3),
            ]
        )
    case "separate_rmsnorm_residual":
        encodeKernel(
            encoder: encoder,
            pipelineState: rmsnormPipeline,
            rowCount: rowCount,
            buffers: [
                (inputBuffer, 0),
                (weightBuffer, 1),
                (scratchBuffer, 2),
                (paramsBuffer, 3),
            ]
        )
        encodeKernel(
            encoder: encoder,
            pipelineState: residualPipeline,
            rowCount: rowCount,
            buffers: [
                (scratchBuffer, 0),
                (residualBuffer, 1),
                (outputBuffer, 2),
                (paramsBuffer, 3),
            ]
        )
    case "fused_rmsnorm_residual":
        encodeKernel(
            encoder: encoder,
            pipelineState: fusedPipeline,
            rowCount: rowCount,
            buffers: [
                (inputBuffer, 0),
                (weightBuffer, 1),
                (residualBuffer, 2),
                (outputBuffer, 3),
                (paramsBuffer, 4),
            ]
        )
    default:
        throw RunnerError.unknownMode(mode)
    }

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
    let weight = try loadFloatArray(from: config.weightPath)
    let residual = try loadFloatArray(from: config.residualPath)

    guard input.count % expectedElementCount == 0 else {
        throw RunnerError.invalidBinarySize("Input element count must be a multiple of \(expectedElementCount), received \(input.count).")
    }
    guard weight.count == expectedElementCount else {
        throw RunnerError.invalidBinarySize("Weight element count must be \(expectedElementCount), received \(weight.count).")
    }
    guard residual.count == input.count else {
        throw RunnerError.invalidBinarySize("Residual element count must match input element count.")
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

    guard let rmsnormFunction = library.makeFunction(name: "rmsnorm_256_shared_f32") else {
        throw RunnerError.functionCreation("rmsnorm_256_shared_f32")
    }
    guard let residualFunction = library.makeFunction(name: "residual_add_f32") else {
        throw RunnerError.functionCreation("residual_add_f32")
    }
    guard let fusedFunction = library.makeFunction(name: "fused_rmsnorm_residual_256_shared_f32") else {
        throw RunnerError.functionCreation("fused_rmsnorm_residual_256_shared_f32")
    }

    let rmsnormPipeline: MTLComputePipelineState
    let residualPipeline: MTLComputePipelineState
    let fusedPipeline: MTLComputePipelineState
    do {
        rmsnormPipeline = try device.makeComputePipelineState(function: rmsnormFunction)
        residualPipeline = try device.makeComputePipelineState(function: residualFunction)
        fusedPipeline = try device.makeComputePipelineState(function: fusedFunction)
    } catch {
        throw RunnerError.pipelineCreation(error.localizedDescription)
    }

    let inputBuffer = try makeSharedBuffer(device: device, floats: input, label: "input")
    let weightBuffer = try makeSharedBuffer(device: device, floats: weight, label: "weight")
    let residualBuffer = try makeSharedBuffer(device: device, floats: residual, label: "residual")
    let scratchBuffer = try makeZeroedBuffer(device: device, floatCount: input.count, label: "scratch")
    let outputBuffer = try makeZeroedBuffer(device: device, floatCount: input.count, label: "output")

    var params = RMSNormParams(
        elementCount: UInt32(expectedElementCount),
        rowCount: UInt32(rowCount),
        eps: config.eps
    )
    guard let paramsBuffer = device.makeBuffer(
        bytes: &params,
        length: MemoryLayout<RMSNormParams>.stride,
        options: .storageModeShared
    ) else {
        throw RunnerError.bufferCreation("params")
    }

    for _ in 0..<config.warmupIters {
        _ = try encodeAndRun(
            mode: config.mode.name,
            commandQueue: commandQueue,
            rmsnormPipeline: rmsnormPipeline,
            residualPipeline: residualPipeline,
            fusedPipeline: fusedPipeline,
            inputBuffer: inputBuffer,
            weightBuffer: weightBuffer,
            residualBuffer: residualBuffer,
            scratchBuffer: scratchBuffer,
            outputBuffer: outputBuffer,
            paramsBuffer: paramsBuffer,
            rowCount: rowCount
        )
    }

    var wallLatenciesMs: [Double] = []
    var gpuLatenciesMs: [Double] = []
    wallLatenciesMs.reserveCapacity(config.timedIters)
    gpuLatenciesMs.reserveCapacity(config.timedIters)
    for _ in 0..<config.timedIters {
        let result = try encodeAndRun(
            mode: config.mode.name,
            commandQueue: commandQueue,
            rmsnormPipeline: rmsnormPipeline,
            residualPipeline: residualPipeline,
            fusedPipeline: fusedPipeline,
            inputBuffer: inputBuffer,
            weightBuffer: weightBuffer,
            residualBuffer: residualBuffer,
            scratchBuffer: scratchBuffer,
            outputBuffer: outputBuffer,
            paramsBuffer: paramsBuffer,
            rowCount: rowCount
        )
        wallLatenciesMs.append(result.wallLatencyMs)
        gpuLatenciesMs.append(result.gpuLatencyMs)
    }

    let primaryKernelName: String
    let secondaryKernelName: String?
    switch config.mode.name {
    case "standalone_rmsnorm":
        primaryKernelName = "rmsnorm_256_shared_f32"
        secondaryKernelName = nil
    case "separate_rmsnorm_residual":
        primaryKernelName = "rmsnorm_256_shared_f32"
        secondaryKernelName = "residual_add_f32"
    case "fused_rmsnorm_residual":
        primaryKernelName = "fused_rmsnorm_residual_256_shared_f32"
        secondaryKernelName = nil
    default:
        throw RunnerError.unknownMode(config.mode.name)
    }

    let result = MetalRunResult(
        mode: config.mode.name,
        device_name: device.name,
        primary_kernel_name: primaryKernelName,
        secondary_kernel_name: secondaryKernelName,
        threadgroup_size: threadgroupSize,
        thread_execution_width: rmsnormPipeline.threadExecutionWidth,
        max_total_threads_per_threadgroup: rmsnormPipeline.maxTotalThreadsPerThreadgroup,
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
