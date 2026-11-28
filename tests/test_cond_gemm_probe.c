/* P3-C micro repro: cond-GEMM shape (20x5120 @ 5120x5376 + bias) with finite
 * bf16 inputs, run under H3_NATIVE_INT8=1 vs unset; count NaN in output.
 * Build: cc into a small binary linked against LIB_OBJ. */
#include "../h3.h"
#include "../h3_gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static uint16_t *make_bf16(size_t n, unsigned seed) {
    uint16_t *v = malloc(n * 2);
    unsigned x = seed ? seed : 1u;
    for (size_t i = 0; i < n; i++) {
        x = x * 1664525u + 1013904223u;
        float f = (float)((int)(x >> 8) % 2000 - 1000) / 1000.0f; /* [-1,1) */
        uint32_t u; memcpy(&u, &f, 4);
        v[i] = (uint16_t)(u >> 16); /* truncate to bf16, always finite */
    }
    return v;
}

int main(void) {
    char error[512] = {0};
    const int rows = 20, in_dim = 5120, out_dim = 5376;
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof error);
    if (!gpu) { fprintf(stderr, "gpu: %s\n", error); return 2; }
    uint16_t *src = make_bf16((size_t)rows * in_dim, 7);
    uint16_t *w = make_bf16((size_t)out_dim * in_dim, 13);
    uint16_t *b = make_bf16((size_t)out_dim, 19);
    h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(gpu, src, (size_t)rows * in_dim);
    h3_gpu_tensor *weight = h3_gpu_tensor_from_bf16(gpu, w, (size_t)out_dim * in_dim);
    h3_gpu_tensor *bias = h3_gpu_tensor_from_bf16(gpu, b, (size_t)out_dim);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * out_dim);
    if (!input || !weight || !bias || !out) {
        fprintf(stderr, "alloc failed: %s\n", h3_gpu_error(gpu));
        return 2;
    }
    int ok;
    if (!h3_gpu_begin(gpu)) { fprintf(stderr, "begin: %s\n", h3_gpu_error(gpu)); return 2; }
    ok = h3_gpu_linear_bf16(gpu, out, input, weight, bias, (uint32_t)rows,
                            (uint32_t)in_dim, (uint32_t)out_dim);
    if (!h3_gpu_submit(gpu)) { fprintf(stderr, "submit: %s\n", h3_gpu_error(gpu)); return 2; }
    if (!ok) { fprintf(stderr, "linear failed: %s\n", h3_gpu_error(gpu)); return 2; }
    uint16_t *got = malloc((size_t)rows * out_dim * 2);
    if (!h3_gpu_tensor_read_bf16(out, got, (size_t)rows * out_dim)) {
        fprintf(stderr, "read out: %s\n", h3_gpu_error(gpu));
        return 2;
    }
    size_t nan = 0, inf = 0; double maxabs = 0;
    for (size_t i = 0; i < (size_t)rows * out_dim; i++) {
        uint32_t u = (uint32_t)got[i] << 16; float f; memcpy(&f, &u, 4);
        if (isnan(f)) nan++; else if (isinf(f)) inf++;
        else { double a = fabs(f); if (a > maxabs) maxabs = a; }
    }
    printf("NATIVE=%s linear=%s out_elements=%zu nan=%zu inf=%zu maxabs=%.4g\n",
           getenv("H3_NATIVE_INT8") ? "1" : "0", ok ? "ok" : "fail",
           (size_t)rows * out_dim, nan, inf, maxabs);
    return nan ? 1 : 0;
}
