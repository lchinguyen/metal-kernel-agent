#include <metal_stdlib>

using namespace metal;

struct SoftmaxParams {
    uint element_count;
    uint row_count;
};

kernel void softmax_256_shared_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint row_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float partial_max[256];
    threadgroup float partial_sum[256];
    threadgroup float row_max;
    threadgroup float row_sum;

    if (row_id >= params.row_count) {
        return;
    }

    uint row_offset = row_id * params.element_count;

    float local_max = -INFINITY;
    for (uint index = tid; index < params.element_count; index += threads_per_group) {
        local_max = max(local_max, input[row_offset + index]);
    }

    partial_max[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_max[tid] = max(partial_max[tid], partial_max[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        row_max = partial_max[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint index = tid; index < params.element_count; index += threads_per_group) {
        local_sum += exp(input[row_offset + index] - row_max);
    }

    partial_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            partial_sum[tid] += partial_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        row_sum = partial_sum[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < params.element_count; index += threads_per_group) {
        output[row_offset + index] = exp(input[row_offset + index] - row_max) / row_sum;
    }
}

kernel void softmax_256_simd_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint row_id [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint lane_id [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint threads_per_simdgroup [[threads_per_simdgroup]]
) {
    threadgroup float simdgroup_max[8];
    threadgroup float simdgroup_sum[8];
    threadgroup float row_max;
    threadgroup float row_sum;

    if (row_id >= params.row_count) {
        return;
    }

    uint row_offset = row_id * params.element_count;

    float local_max = -INFINITY;
    for (uint index = tid; index < params.element_count; index += threads_per_group) {
        local_max = max(local_max, input[row_offset + index]);
    }

    float simd_max_value = simd_max(local_max);
    if (lane_id == 0) {
        simdgroup_max[simdgroup_id] = simd_max_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint simdgroup_count = threads_per_group / threads_per_simdgroup;
    if (simdgroup_id == 0) {
        float block_max = lane_id < simdgroup_count ? simdgroup_max[lane_id] : -INFINITY;
        float total_max = simd_max(block_max);
        if (lane_id == 0) {
            row_max = total_max;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint index = tid; index < params.element_count; index += threads_per_group) {
        local_sum += exp(input[row_offset + index] - row_max);
    }

    float simd_sum_value = simd_sum(local_sum);
    if (lane_id == 0) {
        simdgroup_sum[simdgroup_id] = simd_sum_value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_id == 0) {
        float block_sum = lane_id < simdgroup_count ? simdgroup_sum[lane_id] : 0.0f;
        float total_sum = simd_sum(block_sum);
        if (lane_id == 0) {
            row_sum = total_sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint index = tid; index < params.element_count; index += threads_per_group) {
        output[row_offset + index] = exp(input[row_offset + index] - row_max) / row_sum;
    }
}
