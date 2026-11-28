/* convrot_probe — dumps the engine's dequantized BF16 for one convrot int8
 * tensor through the same call sequence the production DiT lane uses
 * (h3_weight_load_i8 + scale f32 upload + ±1 Hadamard upload +
 * h3_gpu_dequant_convrot with rowmap NULL), so the Metal kernel output can
 * be diffed against the FP32/CPU references.
 *
 * Usage: convrot_probe <shader.metal> <model_dir> <out_dir> <stem>
 *                      <tensor> <rows> <cols>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "h3_gpu.h"
#include "h3_weights.h"

static const int8_t H4[4][4] = {
    { 1, 1, 1,-1},
    { 1, 1,-1, 1},
    { 1,-1, 1, 1},
    {-1, 1, 1, 1},
};

/* Unnormalized kron-power Hadamard (kernel folds 1/sqrt(gs) via args). */
static float *build_hadamard(unsigned group) {
    float *h = calloc(group * group, sizeof(float));
    unsigned size = 4;
    for (unsigned i = 0; i < 4; i++)
        for (unsigned j = 0; j < 4; j++) h[i * 4 + j] = H4[i][j];
    while (size < group) {
        unsigned nsize = size * 4;
        float *next = malloc((size_t)nsize * nsize * sizeof(float));
        for (unsigned i = 0; i < size; i++)
            for (unsigned j = 0; j < size; j++) {
                float base = h[i * size + j];
                for (unsigned a = 0; a < 4; a++)
                    for (unsigned b = 0; b < 4; b++)
                        next[(i * 4 + a) * nsize + (j * 4 + b)] =
                            base * H4[a][b];
            }
        free(h);
        h = next;
        size = nsize;
    }
    return h;
}

int main(int argc, char **argv) {
    if (argc != 8) {
        fprintf(stderr, "usage: convrot_probe <shaders> <model_dir> <out_dir> <stem> <tensor> <rows> <cols>\n");
        return 2;
    }
    char error[512];
    h3_gpu *gpu = h3_gpu_create(argv[1], error, sizeof error);
    if (!gpu) { fprintf(stderr, "gpu: %s\n", error); return 1; }
    h3_weight_store *store = h3_weight_store_open(argv[2], error, sizeof error);
    if (!store) { fprintf(stderr, "store: %s\n", error); return 1; }
    const char *tensor = argv[5];
    uint32_t groupsize = 0;
    int convrot = 0;
    printf("probe %s: quantized=%d", tensor, h3_weight_is_quantized(store, tensor));
    if (h3_weight_quant_info(store, tensor, &groupsize, &convrot, error, sizeof error))
        printf(" convrot=%d groupsize=%u", convrot, groupsize);
    else
        printf(" quant_info: %s", error);
    printf("\n");
    if (!convrot) { fprintf(stderr, "not a convrot tensor\n"); return 1; }
    uint64_t rows = strtoull(argv[6], NULL, 10), cols = strtoull(argv[7], NULL, 10);
    uint64_t w_shape[2] = { rows, cols };
    h3_gpu_tensor *weight = h3_weight_load_i8(store, gpu, tensor, 2, w_shape,
                                              error, sizeof error);
    if (!weight) { fprintf(stderr, "load i8: %s\n", error); return 1; }
    char scale_name[1024];
    snprintf(scale_name, sizeof scale_name, "%s_scale", tensor);
    uint64_t s_shape[2] = { rows, 1 };
    h3_gpu_tensor *scale = h3_weight_load_f32(store, gpu, scale_name, 2,
                                              s_shape, error, sizeof error);
    if (!scale) { fprintf(stderr, "load scale %s: %s\n", scale_name, error); return 1; }
    float *had_host = build_hadamard(groupsize);
    h3_gpu_tensor *had = h3_gpu_tensor_from_f32(gpu, had_host,
                                                (size_t)groupsize * groupsize);
    if (!had) { fprintf(stderr, "had upload: %s\n", h3_gpu_error(gpu)); return 1; }
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * cols);
    if (!out) { fprintf(stderr, "out alloc: %s\n", h3_gpu_error(gpu)); return 1; }
    char error_before[512];
    snprintf(error_before, sizeof(error_before), "%s", h3_gpu_error(gpu));
    if (!h3_gpu_dequant_convrot(gpu, out, weight, scale, had, NULL,
                                (uint32_t)rows, (uint32_t)cols, groupsize)) {
        fprintf(stderr, "dequant: %s\n", h3_gpu_error(gpu));
        return 1;
    }
    if (!h3_gpu_submit(gpu)) {
        /* Weight uploads / dequant may auto-submit internally; a redundant
         * explicit submit then returns 0 with the fallback error string.
         * Only fail when a *new* Metal error string appeared. */
        const char *submit_error = h3_gpu_error(gpu);
        if (strcmp(submit_error, error_before) != 0) {
            fprintf(stderr, "submit: %s\n", submit_error);
            return 1;
        }
    }
    size_t bytes = 0;
    const unsigned char *bytes_ptr = h3_gpu_tensor_bytes(out, &bytes);
    if (!bytes_ptr) { fprintf(stderr, "bytes fail\n"); return 1; }
    char path[2048];
    snprintf(path, sizeof path, "%s/%s.gpu_bf16.bin", argv[3], argv[4]);
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return 1; }
    if (fwrite(bytes_ptr, 1, bytes, f) != bytes) { fprintf(stderr, "short write\n"); return 1; }
    fclose(f);
    printf("wrote %s (%zu bytes)\n", path, bytes);
    h3_weight_store_free(store);
    return 0;
}
