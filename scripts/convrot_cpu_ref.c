/* convrot_cpu_ref — CPU fp32 reference for Comfy convrot int8 dequant.
 *
 * Third independent implementation of the upstream spec (Python/torch and
 * numpy are the other two; this C pass uses its own fp32 accumulation loop
 * written straight from the documented math):
 *   out[j] = bf16_rne( sum_k ( q[k] * scale_row ) * H[j][k] )
 * with H = kron-power of upstream h4 =
 *   [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]   (conference-type,
 * NOT Sylvester), normalized by 1/sqrt(groupsize).
 *
 * Usage: convrot_cpu_ref <dir> <stem> <rows> <cols> <groupsize>
 * Reads <dir>/<stem>.q.bin (int8) + <dir>/<stem>.scale.bin (f32 rows),
 * writes <dir>/<stem>.cpu_bf16.bin (uint16).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

static const int8_t H4[4][4] = {
    { 1, 1, 1,-1},
    { 1, 1,-1, 1},
    { 1,-1, 1, 1},
    {-1, 1, 1, 1},
};

/* Build normalized kron-power Hadamard, mirroring torch.kron(h, h4) growth:
   result[(i*4+a),(j*4+b)] = h_prev[i][j] * h4[a][b]. */
static float *build_hadamard(unsigned group) {
    float *h = calloc(group * group, sizeof(float));
    float *next;
    unsigned size = 4;
    for (unsigned i = 0; i < 4; i++)
        for (unsigned j = 0; j < 4; j++) h[i * 4 + j] = H4[i][j];
    while (size < group) {
        unsigned nsize = size * 4;
        next = malloc((size_t)nsize * nsize * sizeof(float));
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
    float inv = 1.0f / (float)((group == 256) ? 16 : (group == 64) ? 8 :
                               group == 16 ? 4 : 2);
    for (size_t i = 0; i < (size_t)group * group; i++) h[i] *= inv;
    return h;
}

static uint16_t bf16_rne(float value) {
    uint32_t bits;
    memcpy(&bits, &value, 4);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: convrot_cpu_ref <dir> <stem> <rows> <cols> <groupsize>\n");
        return 2;
    }
    const char *dir = argv[1], *stem = argv[2];
    unsigned long rows = strtoul(argv[3], NULL, 10);
    unsigned long cols = strtoul(argv[4], NULL, 10);
    unsigned group = (unsigned)strtoul(argv[5], NULL, 10);
    char path[2048];
    snprintf(path, sizeof path, "%s/%s.q.bin", dir, stem);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return 1; }
    int8_t *q = malloc(rows * cols);
    if (fread(q, 1, rows * cols, f) != rows * cols) { fprintf(stderr, "short q\n"); return 1; }
    fclose(f);
    snprintf(path, sizeof path, "%s/%s.scale.bin", dir, stem);
    f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return 1; }
    float *scale = malloc(rows * sizeof(float));
    if (fread(scale, sizeof(float), rows, f) != rows) { fprintf(stderr, "short scale\n"); return 1; }
    fclose(f);
    snprintf(path, sizeof path, "%s/%s.cpu_bf16.bin", dir, stem);
    f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "open %s failed\n", path); return 1; }
    const float *h = build_hadamard(group);
    uint16_t *row_out = malloc(cols * sizeof(uint16_t));
    float *group_in = malloc(group * sizeof(float));
    for (unsigned long r = 0; r < rows; r++) {
        float s = scale[r];
        const int8_t *row = q + r * cols;
        for (unsigned long g = 0; g < cols / group; g++) {
            const int8_t *src = row + (size_t)g * group;
            for (unsigned k = 0; k < group; k++)
                group_in[k] = (float)src[k] * s;   /* upstream: scale before rotation */
            uint16_t *dst = row_out + (size_t)g * group;
            for (unsigned j = 0; j < group; j++) {
                const float *hrow = h + (size_t)j * group;
                float acc = 0.0f;
                for (unsigned k = 0; k < group; k++)
                    acc += group_in[k] * hrow[k];
                dst[j] = bf16_rne(acc);
            }
        }
        if (fwrite(row_out, sizeof(uint16_t), cols, f) != cols) { fprintf(stderr, "write fail\n"); return 1; }
    }
    fclose(f);
    printf("cpu_ref %s: %lu x %lu gs %u ok\n", stem, rows, cols, group);
    return 0;
}
