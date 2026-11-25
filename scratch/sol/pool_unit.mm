// pool_unit — minimal: fill q/k/v with known pattern, run ONLY h3_sol_block_pool_bf16, compare.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
static float f2bf(float f) { return f; } // placeholder unused
int main() {
    @autoreleasepool {
        uint32_t sequence = 1627, heads = 4, block = 128, vs = 197, ve = 1627;
        uint32_t nqb = (sequence + block - 1) / block;
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:getenv("SOL_SRC")] encoding:NSUTF8StringEncoding error:nil];
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
        if (!lib) { printf("compile: %s\n", [[err localizedDescription] UTF8String]); return 2; }
        id<MTLFunction> fpool = [lib newFunctionWithName:@"h3_sol_block_pool_bf16"];
        size_t nqkv = (size_t)sequence * heads * 128;
        uint16_t *q = (uint16_t *)malloc(nqkv * 2), *k = (uint16_t *)malloc(nqkv * 2), *v = (uint16_t *)malloc(nqkv * 2);
        union { float f; uint32_t u; } cv;
        auto put = [&](uint16_t *b, size_t idx, float val) { cv.f = val; b[idx] = (uint16_t)(cv.u >> 16); };
        // pattern: q[row,h,d] = row + d*0.001 ; k = -row ; v = d
        for (uint32_t r = 0; r < sequence; r++) for (uint32_t h = 0; h < heads; h++) for (uint32_t d = 0; d < 128; d++) {
            size_t idx = ((size_t)r * heads + h) * 128 + d;
            put(q, idx, (float)r + d * 0.001f); put(k, idx, -(float)r); put(v, idx, (float)d * 0.01f);
        }
        id<MTLBuffer> bq = [dev newBufferWithBytes:q length:nqkv * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bk = [dev newBufferWithBytes:k length:nqkv * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bv = [dev newBufferWithBytes:v length:nqkv * 2 options:MTLResourceStorageModeShared];
        size_t pooledN = (size_t)3 * heads * nqb * 128;
        uint16_t *poolcpu = (uint16_t *)malloc(pooledN * 2);
        { union { float f; uint32_t u; } sv; sv.f = 9999.0f; uint16_t sent = (uint16_t)(sv.u >> 16); for (size_t si2 = 0; si2 < pooledN; si2++) poolcpu[si2] = sent; }
        id<MTLBuffer> bpool = [dev newBufferWithBytes:poolcpu length:pooledN * 2 options:MTLResourceStorageModeShared];
        struct { uint32_t sequence, heads, block, video_start, video_stop; } pa = { sequence, heads, block, vs, ve };
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        if (!cmd) { printf("NO COMMAND BUFFER\n"); return 7; }
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        [e setComputePipelineState:[dev newComputePipelineStateWithFunction:fpool error:&err]];
        [e setBuffer:bq offset:0 atIndex:0]; [e setBuffer:bk offset:0 atIndex:1];
        [e setBuffer:bv offset:0 atIndex:2]; [e setBuffer:bpool offset:0 atIndex:3];
        [e setBytes:&pa length:sizeof(pa) atIndex:4];
        [e dispatchThreadgroups:(MTLSizeMake(nqb, heads, 3)) threadsPerThreadgroup:(MTLSizeMake(128, 1, 1))];
        [e endEncoding];
        [cmd commit]; [cmd waitUntilCompleted];
        printf("status=%d completed=%d err=%s\n", (int)[cmd status], (int)MTLCommandBufferStatusCompleted, [[[cmd error] description] UTF8String]);
        uint16_t *g = (uint16_t *)bpool.contents;
        auto bf = [&](uint16_t b) { union { uint32_t u; float f; } x; x.u = (uint32_t)b << 16; return x.f; };
        // reference: block i, which w, head h, dim d = mean over rows lo..hi of tensor
        double worst = 0; uint32_t wi = 0, wh = 0, wd = 0, ww = 0;
        for (int w = 0; w < 3; w++) for (uint32_t h = 0; h < heads; h++) for (uint32_t i = 0; i < nqb; i++) {
            uint32_t r0 = i * block, r1 = r0 + block < sequence ? r0 + block : sequence;
            uint32_t lo = r0 > vs ? r0 : vs, hi = r1 < ve ? r1 : ve; if (hi < lo) hi = lo;
            for (uint32_t d = 0; d < 128; d++) {
                double s = 0;
                for (uint32_t r = lo; r < hi; r++) {
                    size_t idx = ((size_t)r * heads + h) * 128 + d;
                    s += w == 0 ? (double)r + d * 0.001 : (w == 1 ? -(double)r : (double)d * 0.01);
                }
                double ref = hi > lo ? s / (hi - lo) : 0;
                size_t pi = (((size_t)w * heads + h) * nqb + i) * 128 + d;
                double dd = fabs(bf(g[pi]) - ref);
                if (dd > worst) { worst = dd; wi = i; wh = h; wd = d; ww = w; }
            }
        }
        printf("POOL unit worst %.6f at which=%d block=%u h=%u d=%u\n", worst, ww, wi, wh, wd);
        // PASS verdict: |gpu-ref| <= max(0.01*|ref| + 1e-2, bf16 2-ulp) per element
        { double viol=0; uint32_t nbad=0;
          for (int w=0;w<3;w++) for (uint32_t h=0;h<heads;h++) for (uint32_t i=0;i<nqb;i++) {
            uint32_t r0=i*block, r1=r0+block<sequence?r0+block:sequence;
            uint32_t lo=r0>vs?r0:vs, hi=r1<ve?r1:ve; if(hi<lo)hi=lo;
            for (uint32_t d=0;d<128;d++){
              double s=0; for(uint32_t r=lo;r<hi;r++){size_t idx=((size_t)r*heads+h)*128+d; s+= w==0?(double)r+d*0.001:(w==1?-(double)r:(double)d*0.01);} 
              double ref=hi>lo?s/(hi-lo):0;
              size_t pi=(((size_t)w*heads+h)*nqb+i)*128+d;
              union{uint32_t u;float f;}x; x.u=(uint32_t)g[pi]<<16;
              double tol=0.01*fabs(ref)+0.02; if(fabs(x.f-ref)>tol){nbad++; if(fabs(x.f-ref)-tol>viol){viol=fabs(x.f-ref)-tol;}}}
          }
          printf("POOL %s: nbad=%u/%u worst-excess=%.6f\n", nbad? "FAIL":"PASS", nbad, 3*heads*nqb*128, viol); }
        // show a few values
        uint32_t i = 2, h = 0, d = 0;
        size_t pi = (((size_t)0 * heads + h) * nqb + i) * 128 + d;
        for (int dbg=0; dbg<13; dbg++) { printf("i=%d slot0=%.3f slot1=%.3f slot2=%.3f slot3=%.3f\n", dbg, (double)bf(g[(size_t)dbg*128]), (double)bf(g[(size_t)dbg*128+1]), (double)bf(g[(size_t)dbg*128+2]), (double)bf(g[(size_t)dbg*128+3])); }
        printf("gpu pool q blk2 h0 d0 = %.4f (expect mean rows 256..383 = %.4f)\n", bf(g[pi]), (256.0 + 383.0) / 2);
        return 0;
    }
}
// appended diagnostics via re-exec: see pool_dump below
