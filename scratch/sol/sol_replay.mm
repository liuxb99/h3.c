// sol_replay.mm — run REAL sol pipeline kernels on dumped q/k/v, compare vs exact CPU reference.
// Usage: SOL_SRC=h3_shaders.metal Q=q.bin K=k.bin V=v.bin ./sol_replay seq heads video_start video_stop tau
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

static float bf2f(uint16_t b) { union { uint32_t u; float f; } v; v.u = (uint32_t)b << 16; return v.f; }
static uint16_t *read_bf16(const char *p, size_t &n) {
    FILE *f = fopen(p, "rb"); if (!f) { printf("open fail %s\n", p); exit(1); }
    fseek(f, 0, SEEK_END); n = ftell(f) / 2; fseek(f, 0, SEEK_SET);
    uint16_t *b = (uint16_t *)malloc(n * 2); fread(b, 2, n, f); fclose(f); return b;
}

struct sol_pool_args { uint32_t sequence, heads, block, video_start, video_stop; };
struct sol_scores_args { uint32_t nqb, heads, video_first, video_last, video_start, video_stop; float scale, tau; };
struct sol_attn_args { uint32_t sequence, heads, block, nqb, video_start, video_stop, video_first, video_last; float scale; };

int main(int argc, char **argv) {
    @autoreleasepool {
        uint32_t sequence = atoi(argv[1]), heads = atoi(argv[2]);
        uint32_t vs = atoi(argv[3]), ve = atoi(argv[4]);
        float tau = atof(argv[5]);
        uint32_t block = 128, nqb = (sequence + block - 1) / block;
        size_t nq;
        uint16_t *q = read_bf16(getenv("Q"), nq), *k = read_bf16(getenv("K"), nq), *v = read_bf16(getenv("V"), nq);
        size_t rows = nq / (heads * 128); if (rows < sequence) sequence = rows;
        float scale = 1.0f / sqrtf(128.0f);

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        NSString *srcPath = [NSString stringWithUTF8String:getenv("SOL_SRC")];
        NSString *src = [NSString stringWithContentsOfFile:srcPath encoding:NSUTF8StringEncoding error:nil];
        if (!src) { printf("no src\n"); return 1; }
        // kernel functions need structs declared identically — prepend nothing (shaders file self-contained)
        MTLCompileOptions *opt = [MTLCompileOptions new];
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
        if (!lib) { printf("compile: %s\n", [[err localizedDescription] UTF8String]); return 2; }
        id<MTLFunction> fpool = [lib newFunctionWithName:@"h3_sol_block_pool_bf16"];
        id<MTLFunction> fscores = [lib newFunctionWithName:@"h3_sol_block_scores_bf16"];
        id<MTLFunction> fselect = [lib newFunctionWithName:@"h3_sol_block_select_bf16"];
        id<MTLFunction> fattn = [lib newFunctionWithName:@"h3_sol_attn_bf16"];
        if (!fpool || !fscores || !fselect || !fattn) { printf("missing kernel fns\n"); return 2; }
        id<MTLBuffer> bq = [dev newBufferWithBytes:q length:sequence * heads * 256 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bk = [dev newBufferWithBytes:k length:sequence * heads * 256 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bv = [dev newBufferWithBytes:v length:sequence * heads * 256 options:MTLResourceStorageModeShared];
        size_t pooledN = (size_t)3 * heads * nqb * 128;
        id<MTLBuffer> bpool = [dev newBufferWithLength:pooledN * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bscore = [dev newBufferWithLength:(size_t)heads * nqb * nqb * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bth = [dev newBufferWithLength:(size_t)heads * nqb * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bso = [dev newBufferWithLength:(size_t)heads * (nqb + 1) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsi = [dev newBufferWithLength:(size_t)heads * nqb * nqb * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bout = [dev newBufferWithLength:sequence * heads * 256 options:MTLResourceStorageModeShared];

        sol_pool_args pa = { sequence, heads, block, vs, ve };
        sol_scores_args sa = { nqb, heads, (vs + block - 1) / block, ve / block, vs, ve, scale, tau };
        if (sa.video_last < sa.video_first) sa.video_last = sa.video_first;
        sol_attn_args aa = { sequence, heads, block, nqb, vs, ve, sa.video_first, sa.video_last, scale };

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        [e setComputePipelineState:[dev newComputePipelineStateWithFunction:fpool error:&err]];
        [e setBuffer:bq offset:0 atIndex:0]; [e setBuffer:bk offset:0 atIndex:1];
        [e setBuffer:bv offset:0 atIndex:2]; [e setBuffer:bpool offset:0 atIndex:3];
        [e setBytes:&pa length:sizeof(pa) atIndex:4];
        [e dispatchThreadgroups:(MTLSizeMake(nqb, heads, 3)) threadsPerThreadgroup:(MTLSizeMake(128, 1, 1))];
        [e setComputePipelineState:[dev newComputePipelineStateWithFunction:fscores error:&err]];
        [e setBuffer:bpool offset:0 atIndex:0]; [e setBuffer:bscore offset:0 atIndex:1];
        [e setBuffer:bth offset:0 atIndex:2]; [e setBytes:&sa length:sizeof(sa) atIndex:3];
        [e dispatchThreadgroups:(MTLSizeMake(heads, (nqb + 127) / 128, 1)) threadsPerThreadgroup:(MTLSizeMake(128, 1, 1))];
        [e setComputePipelineState:[dev newComputePipelineStateWithFunction:fselect error:&err]];
        [e setBuffer:bscore offset:0 atIndex:0]; [e setBuffer:bth offset:0 atIndex:1];
        [e setBuffer:bso offset:0 atIndex:2]; [e setBuffer:bsi offset:0 atIndex:3];
        [e setBytes:&sa length:sizeof(sa) atIndex:4];
        [e dispatchThreadgroups:(MTLSizeMake(heads, 1, 1)) threadsPerThreadgroup:(MTLSizeMake(1024, 1, 1))];
        [e setComputePipelineState:[dev newComputePipelineStateWithFunction:fattn error:&err]];
        [e setBuffer:bq offset:0 atIndex:0]; [e setBuffer:bk offset:0 atIndex:1];
        [e setBuffer:bv offset:0 atIndex:2]; [e setBuffer:bout offset:0 atIndex:3];
        [e setBytes:&aa length:sizeof(aa) atIndex:4];
        [e setBuffer:bpool offset:0 atIndex:5]; [e setBuffer:bscore offset:0 atIndex:6];
        [e setBuffer:bso offset:0 atIndex:7]; [e setBuffer:bsi offset:0 atIndex:8];
        [e dispatchThreadgroups:(MTLSizeMake(nqb, heads, 1)) threadsPerThreadgroup:(MTLSizeMake(128, 1, 1))];
        [e endEncoding];
        [cmd commit]; [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) { printf("cmd: %s\n", [[[cmd error] localizedDescription] UTF8String]); return 3; }

        uint16_t *gpool = (uint16_t *)bpool.contents;
        float *gscore = (float *)bscore.contents, *gth = (float *)bth.contents;
        uint32_t *gso = (uint32_t *)bso.contents, *gsi = (uint32_t *)bsi.contents;
        uint16_t *gout = (uint16_t *)bout.contents;

        // ---- CPU reference with exact same semantics ----
        auto atq = [&](uint32_t r, uint32_t h, uint32_t d) { return bf2f(q[((size_t)r * heads + h) * 128 + d]); };
        auto atk = [&](uint32_t r, uint32_t h, uint32_t d) { return bf2f(k[((size_t)r * heads + h) * 128 + d]); };
        auto atv = [&](uint32_t r, uint32_t h, uint32_t d) { return bf2f(v[((size_t)r * heads + h) * 128 + d]); };
        std::vector<float> cpooled(pooledN), cscores((size_t)heads * nqb * nqb, 0.f), cth((size_t)heads * nqb, 1e30f);
        for (int which = 0; which < 3; which++)
            for (uint32_t h = 0; h < heads; h++) for (uint32_t i = 0; i < nqb; i++) {
                uint32_t r0 = i * block, r1 = std::min(r0 + block, sequence);
                uint32_t lo = std::max(r0, vs), hi = std::min(r1, ve), cnt = hi > lo ? hi - lo : 0;
                for (uint32_t d = 0; d < 128; d++) {
                    double s = 0;
                    for (uint32_t r = lo; r < hi; r++)
                        s += which == 0 ? atq(r, h, d) : (which == 1 ? atk(r, h, d) : atv(r, h, d));
                    cpooled[(((size_t)which * heads + h) * nqb + i) * 128 + d] = cnt ? (float)(s / cnt) : 0.f;
                }
            }
        double worst_pool = 0;
        for (size_t idx = 0; idx < pooledN; idx++) {
            double d = fabs(bf2f(gpool[idx]) - cpooled[idx]); if (d > worst_pool) worst_pool = d;
        }
        printf("POOL worst abs diff %.5f (ref meanabs %.5f, gpu meanabs %.5f)\n", worst_pool,
               [&]{ double s = 0; for (auto x : cpooled) s += fabs(x); return s / pooledN; }(),
               [&]{ double s = 0; for (size_t idx = 0; idx < pooledN; idx++) s += fabs(bf2f(gpool[idx])); return s / pooledN; }());
        // scores: dots over window j
        for (uint32_t h = 0; h < heads; h++) for (uint32_t i = 0; i < nqb; i++) {
            uint32_t b0 = i * block, b1 = b0 + block;
            bool inter = b0 < ve && b1 > vs && sa.video_last > sa.video_first;
            bool fully = b0 >= vs && b1 <= ve;
            if (!inter) continue;
            double mean = 0, var = 0; uint32_t cnt = sa.video_last - sa.video_first;
            for (int pass = 0; pass < 2; pass++) {
                double sum = 0, sum2 = 0;
                for (uint32_t j = sa.video_first; j < sa.video_last; j++) {
                    double dot = 0;
                    for (uint32_t d = 0; d < 128; d++)
                        dot += (double)cpooled[((size_t)h * nqb + i) * 128 + d] * cpooled[((size_t)h * nqb + j) * 128 + d];
                    dot *= scale;
                    if (pass) cscores[((size_t)h * nqb + i) * nqb + j] = (float)dot;
                    sum += dot; sum2 += dot * dot;
                }
                mean = sum / cnt; var = std::max(sum2 / cnt - mean * mean, 0.0);
            }
            cth[(size_t)h * nqb + i] = fully ? (float)(mean + tau * sqrt(var)) : 1e30f;
        }
        double worst_s = 0, worst_t = 0;
        for (uint32_t h = 0; h < heads; h++) for (uint32_t i = 0; i < nqb; i++) for (uint32_t j = sa.video_first; j < sa.video_last; j++) {
            double d = fabs(gscore[((size_t)h * nqb + i) * nqb + j] - cscores[((size_t)h * nqb + i) * nqb + j]);
            if (d > worst_s) worst_s = d;
        }
        for (size_t idx = 0; idx < (size_t)heads * nqb; idx++) { double d = fabs(gth[idx] - cth[idx]); if (d > worst_t) worst_t = d; }
        printf("SCORES worst %.5f (gpu meanabs %.5f) TH worst %.5f\n", worst_s,
               [&]{ double s = 0; for (uint32_t h = 0; h < heads; h++) for (uint32_t i = 0; i < nqb; i++) for (uint32_t j = sa.video_first; j < sa.video_last; j++) s += fabs(gscore[((size_t)h * nqb + i) * nqb + j]); return s / ((double)heads * nqb * (sa.video_last - sa.video_first)); }(),
               worst_t);
        // selection counts
        double worst_a = 0, worst_v = 0; long bad_a = 0, bad_v = 0, selcnt = 0, selrows = 0;
        uint32_t samples[3] = {0, 64, 127};
        for (uint32_t h = 0; h < heads; h++) for (uint32_t i = 0; i < nqb; i++) {
          for (uint32_t si = 0; si < 3; si++) { uint32_t r = i * block + samples[si]; if (r >= sequence) continue;
            {
                bool target = r >= vs && r < ve;
                double acc[128], lsum = 0, m = -INFINITY;
                for (int d = 0; d < 128; d++) acc[d] = 0;
                for (uint32_t j = 0; j < nqb; j++) {
                    bool vblock = j >= sa.video_first && j < sa.video_last;
                    bool exact = !target || !vblock;
                    if (!exact) {
                        uint32_t base = gso[(size_t)h * (nqb + 1) + i], end = gso[(size_t)h * (nqb + 1) + i + 1];
                        for (uint32_t s2 = base; s2 < end; s2++) if (gsi[((size_t)h * nqb + i) * nqb + s2] == j) { exact = true; break; }
                    }
                    if (!vblock || exact) { selrows++; if (exact && vblock) selcnt++; }
                    if (exact) {
                        uint32_t k0 = j * block, k1 = std::min(k0 + block, sequence);
                        for (uint32_t kc = k0; kc < k1; kc += 64) {
                            uint32_t kn = std::min(64u, k1 - kc);
                            std::vector<float> sc(kn); float nm = (float)m;
                            for (uint32_t t = 0; t < kn; t++) {
                                double dot = 0;
                                for (uint32_t d = 0; d < 128; d++) dot += (double)atq(r, h, d) * atk(kc + t, h, d);
                                sc[t] = (float)(dot * scale); if (sc[t] > nm) nm = sc[t];
                            }
                            double em = exp(m - nm), ladd = 0;
                            std::vector<double> ps(kn);
                            for (uint32_t t = 0; t < kn; t++) { ps[t] = exp(sc[t] - nm); ladd += ps[t]; }
                            for (int d = 0; d < 128; d++) {
                                double pa = 0;
                                for (uint32_t t = 0; t < kn; t++) pa += ps[t] * atv(kc + t, h, d);
                                acc[d] = acc[d] * em + pa;
                            }
                            lsum = lsum * em + ladd; m = nm;
                        }
                    } else {
                        float s2 = cscores[((size_t)h * nqb + i) * nqb + j];
                        float nm = std::max((float)m, s2); double em = exp(m - nm), p = exp(s2 - nm);
                        for (int d = 0; d < 128; d++)
                            acc[d] = acc[d] * em + p * bf2f(gpool[(((size_t)2 * heads + h) * nqb + j) * 128 + d]);
                        lsum = lsum * em + p; m = nm;
                    }
                }
                for (int d = 0; d < 128; d++) {
                    float ref = (float)(acc[d] / lsum);
                    float got = bf2f(gout[((size_t)r * heads + h) * 128 + d]);
                    double dd = fabs(got - ref);
                    if (r < vs) { if (dd > worst_a) worst_a = dd; if (dd > 0.5) bad_a++; }
                    else { if (dd > worst_v) worst_v = dd; if (dd > 0.5) bad_v++; }
                }
            }
          }
        }
        printf("ATTN vs ref: prefix worst %.4f (bad %ld) video worst %.4f (bad %ld) | selected video blocks %ld, video-KV candidate blocks %ld\n",
               worst_a, bad_a, worst_v, bad_v, selcnt, selrows);
        return 0;
    }
}
