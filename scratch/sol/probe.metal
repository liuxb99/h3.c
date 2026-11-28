// probe2: read args AND compute lo/hi/count/nqb inside kernel, write as ints (bf16 can't hold 1536)
#include <metal_stdlib>
using namespace metal;
struct probe_args { uint sequence, heads, block, video_start, video_stop; };
kernel void h3_probe2(const constant probe_args &args [[buffer(0)]],
                      device uint *out [[buffer(1)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid == 0) {
        uint nqb = (args.sequence + args.block - 1u) / args.block;
        uint i = 2;
        uint row0 = i * args.block, row1 = min(row0 + args.block, args.sequence);
        uint lo = max(row0, args.video_start), hi = min(row1, args.video_stop);
        out[0] = args.sequence; out[1] = args.heads; out[2] = args.block;
        out[3] = args.video_start; out[4] = args.video_stop;
        out[5] = nqb; out[6] = row0; out[7] = row1; out[8] = lo; out[9] = hi;
        out[10] = (uint)(hi > lo ? hi - lo : 0u);
    }
}
