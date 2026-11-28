#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
struct probe_args { uint32_t sequence, heads, block, video_start, video_stop; };
int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        NSString *src = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:getenv("PROBE_SRC")] encoding:NSUTF8StringEncoding error:nil];
        NSError *err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
        if (!lib) { printf("compile: %s\n", [[err localizedDescription] UTF8String]); return 2; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"h3_probe"];
        if (!fn) { printf("no fn\n"); return 2; }
        id<MTLBuffer> bo = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
        probe_args pa = {1627, 4, 128, 197, 1627};
        printf("host sizeof=%zu\n", sizeof(probe_args));
        id<MTLCommandBuffer> cmd = [queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cmd computeCommandEncoder];
        [e setComputePipelineState:[dev newComputePipelineStateWithFunction:fn error:&err]];
        [e setBytes:&pa length:sizeof(pa) atIndex:0];
        [e setBuffer:bo offset:0 atIndex:1];
        [e dispatchThreadgroups:(MTLSizeMake(1, 1, 1)) threadsPerThreadgroup:(MTLSizeMake(32, 1, 1))];
        [e endEncoding];
        [cmd commit]; [cmd waitUntilCompleted];
        uint32_t *o = (uint32_t *)bo.contents;
        printf("gpu args: seq=%u heads=%u block=%u vs=%u ve=%u sizeof=%u\n", o[0], o[1], o[2], o[3], o[4], o[5]);
        return 0;
    }
}
