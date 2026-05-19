#include <metal_stdlib>

using namespace metal;

struct RMSNormParams {
    uint element_count;
    uint row_count;
    float eps;
};

kernel void rmsnorm_256_shared_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RMSNormParams& params [[buffer(3)]],
    uint row_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float partial_sums[256];
    threadgroup float inv_rms;

    if (row_id >= params.row_count) {
        return;
    }

    uint row_offset = row_id * params.element_count;

    float local_sum = 0.0f;
    for (uint index = tid; index < params.element_count; index += threads_per_group) {
        float value = input[row_offset + index];
        local_sum += value * value;
    }

    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float mean_square = partial_sums[0] / static_cast<float>(params.element_count);
        inv_rms = rsqrt(mean_square + params.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < params.element_count; index += threads_per_group) {
        output[row_offset + index] = input[row_offset + index] * inv_rms * weight[index];
    }
}

kernel void residual_add_f32(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RMSNormParams& params [[buffer(3)]],
    uint row_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    if (row_id >= params.row_count) {
        return;
    }

    uint row_offset = row_id * params.element_count;
    for (uint index = tid; index < params.element_count; index += threads_per_group) {
        uint offset = row_offset + index;
        output[offset] = input[offset] + residual[offset];
    }
}

kernel void fused_rmsnorm_residual_256_shared_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* residual [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant RMSNormParams& params [[buffer(4)]],
    uint row_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float partial_sums[256];
    threadgroup float inv_rms;

    if (row_id >= params.row_count) {
        return;
    }

    uint row_offset = row_id * params.element_count;

    float local_sum = 0.0f;
    for (uint index = tid; index < params.element_count; index += threads_per_group) {
        float value = input[row_offset + index];
        local_sum += value * value;
    }

    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float mean_square = partial_sums[0] / static_cast<float>(params.element_count);
        inv_rms = rsqrt(mean_square + params.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < params.element_count; index += threads_per_group) {
        uint offset = row_offset + index;
        output[offset] = residual[offset] + (input[offset] * inv_rms * weight[index]);
    }
}

kernel void rmsnorm_128_shared_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RMSNormParams& params [[buffer(3)]],
    uint row_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float partial_sums[128];
    threadgroup float inv_rms;

    if (row_id >= params.row_count) {
        return;
    }

    uint row_offset = row_id * params.element_count;

    float local_sum = 0.0f;
    for (uint index = tid; index < params.element_count; index += threads_per_group) {
        float value = input[row_offset + index];
        local_sum += value * value;
    }

    partial_sums[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        float mean_square = partial_sums[0] / static_cast<float>(params.element_count);
        inv_rms = rsqrt(mean_square + params.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < params.element_count; index += threads_per_group) {
        output[row_offset + index] = input[row_offset + index] * inv_rms * weight[index];
    }
}

kernel void rmsnorm_128_simd_f32(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant RMSNormParams& params [[buffer(3)]],
    uint row_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    threadgroup float simdgroup_sums[4];
    threadgroup float inv_rms;

    if (row_id >= params.row_count) {
        return;
    }

    uint row_offset = row_id * params.element_count;

    float local_sum = 0.0f;
    for (uint index = tid; index < params.element_count; index += threads_per_group) {
        float value = input[row_offset + index];
        local_sum += value * value;
    }

    float simd_total = simd_sum(local_sum);
    if (lane_id == 0) {
        simdgroup_sums[simdgroup_id] = simd_total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint simdgroup_count = threads_per_group / threads_per_simdgroup;
    if (simdgroup_id == 0) {
        float block_sum = lane_id < simdgroup_count ? simdgroup_sums[lane_id] : 0.0f;
        float total_sum = simd_sum(block_sum);
        if (lane_id == 0) {
            float mean_square = total_sum / static_cast<float>(params.element_count);
            inv_rms = rsqrt(mean_square + params.eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < params.element_count; index += threads_per_group) {
        output[row_offset + index] = input[row_offset + index] * inv_rms * weight[index];
    }
}
