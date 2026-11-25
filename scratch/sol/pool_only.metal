#include <metal_stdlib>
#ifdef H3_METAL_HAS_TENSOR
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
using namespace metal;
#ifdef H3_METAL_HAS_TENSOR
using namespace mpp::tensor_ops;
#endif

inline float h3_bf16_to_f32(ushort value) {
    return as_type<float>(uint(value) << 16);
}

inline ushort h3_f32_to_bf16(float value) {
    uint bits = as_type<uint>(value);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return ushort(bits >> 16);
}

inline float4 h3_bf16x4_to_f32(ushort4 value) {
    return as_type<float4>(uint4(value) << 16);
}

inline ushort4 h3_f32x4_to_bf16(float4 value) {
    uint4 bits = as_type<uint4>(value);
    bits += uint4(0x7fffu) + ((bits >> 16) & uint4(1u));
    return ushort4(bits >> 16);
}

struct linear_args {
    uint rows;
    uint input_dim;
    uint output_dim;
    uint has_bias;
};

struct int8_quant_args {
    uint rows;
    uint columns;
    float clip;
};

struct int8_head_major_quant_args {
    uint rows;
    uint padded_rows;
    uint heads;
    uint head_dim;
    float clip;
};

struct int8_group_quant_args {
    uint rows;
    uint columns;
    uint group_size;
    uint groups;
};

struct sol_pool_args { uint sequence, heads, block, video_start, video_stop; };
kernel void h3_sol_block_pool_bf16(
        device const ushort *query [[buffer(0)]],
        device const ushort *key [[buffer(1)]],
        device const ushort *value [[buffer(2)]],
        device ushort *pooled [[buffer(3)]],
        constant sol_pool_args &args [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint3 posn [[thread_position_in_threadgroup]]) {
    uint tid = posn.x;
    uint i = group.x, h = group.y, which = group.z;
    uint nqb = (args.sequence + args.block - 1u) / args.block;
    if (i >= nqb || h >= args.heads) return;
    device const ushort *src = which == 0u ? query : (which == 1u ? key : value);
    uint row0 = i * args.block, row1 = min(row0 + args.block, args.sequence);
    uint lo = max(row0, args.video_start), hi = min(row1, args.video_stop);
    uint count = hi > lo ? hi - lo : 0u;
    size_t out = ((size_t)which * args.heads + h) * nqb * 128u + (size_t)i * 128u;
    if (count == 0u) { if (tid < 128u) pooled[out + tid] = 0u; return; }
    float sum = 0.0f;
    for (uint row = lo; row < hi; ++row)
        sum += h3_bf16_to_f32(src[((size_t)row * args.heads + h) * 128u + tid]);
    pooled[out + tid] = h3_f32_to_bf16(sum / (float)count);
}
