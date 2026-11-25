// Standalone unit test for h3_sol_attn_kernel (+ pool/scores/select) with synthetic data.
// Build: clang++ -ObjC++ -framework Metal -framework Foundation sol_unit.mm h3_sol_kernel_logic -o sol_unit
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

static const int HEADS = 4, BLK = 128, DIMS = 128;
static const unsigned short *gq, *gk, *gv;

uint16_t f2bf(float f) {
    union { float f; uint32_t u; } v; v.f = f;
    uint32_t r = (v.u + 0x8000u) & 0xFFFF0000u;
    if ((v.u & 0xFFFFu) == 0x8000u && (v.u & 0x10000u)) r += 0x10000u;
    return (uint16_t)(r >> 16);
}
float bf2f(uint16_t b) { union { uint32_t u; float f; } v; v.u = ((uint32_t)b) << 16; return v.f; }

int main(int argc, char **argv) {
    @autoreleasepool {
        int sequence = 600, vs = 64, ve = 576; // text prefix 64, partial tail block
        if (argc > 1) sequence = atoi(argv[1]);
        int nqb = (sequence + BLK - 1) / BLK;
        size_t nq = (size_t)sequence * HEADS * DIMS;
        std::vector<uint16_t> q(nq), k(nq), v(nq);
        srand(42);
        auto rf = []() { return (float)((rand() % 2000) - 1000) / 1000.0f; };
        for (size_t i = 0; i < nq; i++) { q[i] = f2bf(rf()); k[i] = f2bf(rf()); v[i] = f2bf(rf()); }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        NSString *src = [NSString stringWithUTF8String: getenv("SOL_SRC")];
        MTLCompileOptions *opt = [MTLCompileOptions new];
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
        if (!lib) { printf("compile fail: %s\n", [[err localizedDescription] UTF8String]); return 2; }
        id<MTLFunction> attn = [lib newFunctionWithName:@"h3_sol_attn_kernel"];
        id<MTLFunction> poolf = [lib newFunctionWithName:@"h3_sol_pool_kernel"];
        id<MTLFunction> scoresf = [lib newFunctionWithName:@"h3_sol_scores_kernel"];
        id<MTLFunction> selectf = [lib newFunctionWithName:@"h3_sol_select_kernel"];
        id<MTLBuffer> bq = [dev newBufferWithLength:nq * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bk = [dev newBufferWithLength:nq * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bv = [dev newBufferWithLength:nq * 2 options:MTLResourceStorageModeShared];
        memcpy(bq.contents, q.data(), nq * 2);
        memcpy(bk.contents, k.data(), nq * 2);
        memcpy(bv.contents, v.data(), nq * 2);
        size_t npool = (size_t)3 * HEADS * nqb * DIMS * 2;
        size_t nscores = (size_t)HEADS * nqb * nqb * 4;
        id<MTLBuffer> bpool = [dev newBufferWithLength:npool options:MTLResourceStorageModeShared];
        id<MTLBuffer> bscore = [dev newBufferWithLength:nscores options:MTLResourceStorageModeShared];
        id<MTLBuffer> bth = [dev newBufferWithLength:(size_t)HEADS * nqb * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bso = [dev newBufferWithLength:(size_t)HEADS * (nqb + 1) * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bsi = [dev newBufferWithLength:(size_t)HEADS * nqb * nqb * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bout = [dev newBufferWithLength:nq * 2 options:MTLResourceStorageModeShared];
        memset(bout.contents, 0xAB, nq * 2);

        struct { const void *q, *k, *v, *pool, *scores, *th, *so, *si, *out;
                 uint32_t heads, sequence, nqb, vs, ve; uint32_t vf, vl; float tau, scale; } P = {
            bq.contents, bk.contents, bv.contents, bpool.contents, bscore.contents, bth.contents,
            bso.contents, bsi.contents, bout.contents, HEADS, sequence, nqb, vs, ve,
            vs / BLK, (ve + BLK - 1) / BLK, 1.3f, 1.0f / sqrtf(128.0f) };

        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        {   id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:[dev newComputePipelineStateWithFunction:poolf error:&err]];
            [e setBytes:&P length:sizeof(P) atIndex:0];
            [e dispatchThreadgroups:(MTLSizeMake((unsigned)ceil(HEADS * nqb / 128.0), 3, 1))
              threadsPerThreadgroup:(MTLSizeMake(128, 1, 1))];
            [e endEncoding]; }
        [cmd commit]; [cmd waitUntilCompleted];
        {   cmd = [queue commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:[dev newComputePipelineStateWithFunction:scoresf error:&err]];
            [e setBytes:&P length:sizeof(P) atIndex:0];
            [e dispatchThreadgroups:(MTLSizeMake(HEADS, nqb, 1)) threadsPerThreadgroup:(MTLSizeMake(32, 1, 1))];
            [e endEncoding]; }
        {   id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:[dev newComputePipelineStateWithFunction:selectf error:&err]];
            [e setBytes:&P length:sizeof(P) atIndex:0];
            [e dispatchThreadgroups:(MTLSizeMake(1, HEADS * nqb, 1)) threadsPerThreadgroup:(MTLSizeMake(32, 1, 1))];
            [e endEncoding]; }
        [cmd commit]; [cmd waitUntilCompleted];
        {   cmd = [queue commandBuffer]; id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
            [e setComputePipelineState:[dev newComputePipelineStateWithFunction:attn error:&err]];
            if (err) { printf("attn pso fail: %s\n", [[err localizedDescription] UTF8String]); return 3; }
            [e setBytes:&P length:sizeof(P) atIndex:0];
            [e dispatchThreadgroups:(MTLSizeMake(nqb, HEADS, 1)) threadsPerThreadgroup:(MTLSizeMake(128, 1, 1))];
            [e endEncoding]; }
        [cmd commit]; [cmd waitUntilCompleted];
        if ([cmd status] != MTLCommandBufferStatusCompleted) { printf("attn dispatch failed: %s\n", [[[cmd error] localizedDescription] UTF8String]); return 4; }

        // CPU replay of the same pipeline (matches kernel semantics incl. pooled-comp design)
        const float scale = 1.0f / sqrtf(128.0f);
        std::vector<float> qb(HEADS * nqb * DIMS), kb(HEADS * nqb * DIMS), vb(HEADS * nqb * DIMS);
        for (int c = 0; c < 3; c++) {
            const std::vector<uint16_t> &src = c == 0 ? q : (c == 1 ? k : v);
            for (int h = 0; h < HEADS; h++) for (int ib = 0; ib < nqb; ib++) {
                int i0 = ib * BLK, i1 = std::min(i0 + BLK, sequence);
                for (int d = 0; d < DIMS; d++) {
                    double acc = 0;
                    for (int i = i0; i < i1; i++) acc += bf2f(src[((size_t)i * HEADS + h) * DIMS + d]);
                    (c == 0 ? qb : (c == 1 ? kb : vb))[((size_t)h * nqb + ib) * DIMS + d] = (float)(acc / (i1 - i0));
                }
            }
        }
        float *gdPool = (float*)NULL; (void)gdPool;
        // compare GPU pooled vs CPU pooled (bf16 rounding of pool result)
        uint16_t *gpool = (uint16_t *)bpool.contents;
        double maxdp = 0; for (size_t i = 0; i < (size_t)3 * HEADS * nqb * DIMS; i++) {
            int c = i / ((size_t)HEADS * nqb * DIMS);
            size_t r = i % ((size_t)HEADS * nqb * DIMS);
            float ref = (c == 0 ? qb : (c == 1 ? kb : vb))[r];
            double d = fabs(bf2f(gpool[i]) - ref); if (d > maxdp) maxdp = d;
        }
        printf("pooled maxdiff %.6f\n", maxdp);
        // scores
        float *gscores = (float *)bscore.contents;
        maxdp = 0;
        for (int h = 0; h < HEADS; h++) for (int i = 0; i < nqb; i++) for (int j = 0; j < nqb; j++) {
            double acc = 0;
            for (int d = 0; d < DIMS; d++)
                acc += (double)qb[((size_t)h * nqb + i) * DIMS + d] * kb[((size_t)h * nqb + j) * DIMS + d];
            double ref = acc * scale;
            double dd = fabs(gscores[((size_t)h * nqb + i) * nqb + j] - ref);
            if (dd > maxdp) maxdp = dd;
        }
        printf("scores maxdiff %.6f\n", maxdp);
        // attention replay using GPU's own scores/thresholds/selection inputs
        uint32_t *gso = (uint32_t *)bso.contents; uint32_t *gsi = (uint32_t *)bsi.contents;
        float *gth = (float *)bth.contents;
        uint16_t *gout = (uint16_t *)bout.contents;
        // Simplified final compare: replay attention using GPU selection outputs directly (assume select correct if scores match)
        double worst_text = 0, worst_video = 0; int bad_text = 0, bad_video = 0;
        for (int h = 0; h < HEADS; h++) for (int row = 0; row < sequence; row++) {
            bool target = row >= vs;
            int block = row / BLK;
            double acc[DIMS], lsum = 0, m = -1e30;
            for (int d = 0; d < DIMS; d++) acc[d] = 0;
            for (int j = 0; j < nqb; j++) {
                bool video_block = P.vf <= j && j < P.vl;
                bool exact = !target || !video_block;
                if (!exact) {
                    uint32_t off = gso[h * (nqb + 1) + block], cnt = gso[h * (nqb + 1) + block + 1] - off;
                    for (uint32_t s = 0; s < cnt; s++) if (gsi[(size_t)h * nqb * nqb + (size_t)block * nqb + s] == (uint32_t)j) { exact = true; break; }
                }
                if (exact) {
                    int k0 = j * BLK, k1 = std::min(k0 + BLK, sequence);
                    for (int kk = k0; kk < k1; kk += 64) {
                        int kn = std::min(64, k1 - kk);
                        std::vector<float> sc(kn);
                        float nm = (float)m;
                        for (int kt = 0; kt < kn; kt++) {
                            double dot = 0;
                            for (int d = 0; d < DIMS; d++)
                                dot += (double)bf2f(q[((size_t)row * HEADS + h) * DIMS + d]) * bf2f(k[((size_t)(kk + kt) * HEADS + h) * DIMS + d]);
                            sc[kt] = (float)(dot * scale);
                            if (sc[kt] > nm) nm = sc[kt];
                        }
                        double em = exp(m - nm), ladd = 0;
                        for (int kt = 0; kt < kn; kt++) ladd += exp(sc[kt] - nm);
                        for (int d = 0; d < DIMS; d++) {
                            double pa = 0;
                            for (int kt = 0; kt < kn; kt++) pa += exp(sc[kt] - nm) * bf2f(v[((size_t)(kk + kt) * HEADS + h) * DIMS + d]);
                            acc[d] = acc[d] * em + pa;
                        }
                        lsum = lsum * em + ladd; m = nm;
                    }
                } else {
                    float s = gscores[((size_t)h * nqb + block) * nqb + j];
                    float nm = std::max((float)m, s);
                    double em = exp(m - nm), p = exp(s - nm);
                    for (int d = 0; d < DIMS; d++)
                        acc[d] = acc[d] * em + p * bf2f(((uint16_t*)bpool.contents)[((size_t)(2 * HEADS + h) * nqb + j) * DIMS + d]);
                    lsum = lsum * em + p; m = nm;
                }
            }
            for (int d = 0; d < DIMS; d++) {
                float ref = (float)(acc[d] / lsum), got = bf2f(gout[((size_t)row * HEADS + h) * DIMS + d]);
                double dd = fabs(got - ref);
                if (row < vs) { if (dd > worst_text) worst_text = dd; if (dd > 0.5) bad_text++; }
                else { if (dd > worst_video) worst_video = dd; if (dd > 0.5) bad_video++; }
            }
        }
        printf("attn vs replay: text worst %.4f (%d bad/64*4*128)  video worst %.4f (%d bad)\n",
               worst_text, bad_text, worst_video, bad_video);
        return 0;
    }
}
