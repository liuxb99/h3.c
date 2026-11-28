/* GGUF v3 container reader against the pruned MiniMax H3 IQ1_S asset
 * (R2 acceptance): header/KV/tensor-table parse, byte-exact data-region
 * continuity, spot reads of a known F32 table and an IQ1_S tensor head,
 * and rejection of ggml type values outside the sd.cpp fork enum.
 *
 *   ./h3_gguf_tests [file.gguf]
 */
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../h3_gguf.h"

#ifndef DEFAULT_GGUF_PATH
#define DEFAULT_GGUF_PATH \
    "/Volumes/SSD2T/models/MiniMax-H3-gguf/minimax_h3_fl2va_pruned-IQ1_S.gguf"
#endif

#define ADALN_NAME "model.diffusion_model.adaln_t_table"
#define QKV0_NAME "model.diffusion_model.blocks.0.attn.qkv_proj.weight"

/* IEEE binary16 -> binary32 (exact for all finite bit patterns). */
static float h3_f16_to_f32(uint16_t bits) {
    uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    uint32_t exponent = (uint32_t)(bits >> 10) & 0x1Fu;
    uint32_t fraction = (uint32_t)bits & 0x3FFu;
    uint32_t pattern;
    if (exponent == 0) {
        if (fraction == 0) pattern = sign; /* +/- zero */
        else { /* subnormal: renormalize */
            exponent = 127 - 15 + 1;
            while ((fraction & 0x400u) == 0) { fraction <<= 1; exponent--; }
            fraction &= 0x3FFu;
            pattern = sign | (exponent << 23) | (fraction << 13);
        }
    } else if (exponent == 0x1Fu) {
        pattern = sign | 0x7F800000u | ((uint32_t)fraction << 13); /* inf/NaN */
    } else {
        pattern = sign | ((exponent + 112u) << 23) | ((uint32_t)fraction << 13);
    }
    float value;
    memcpy(&value, &pattern, sizeof(value));
    return value;
}

typedef struct {
    const char *name;
    uint64_t offset;
    uint64_t nbytes;
} sortable_tensor;

static int compare_by_offset(const void *left, const void *right) {
    const sortable_tensor *a = (const sortable_tensor *)left;
    const sortable_tensor *b = (const sortable_tensor *)right;
    if (a->offset < b->offset) return -1;
    if (a->offset > b->offset) return 1;
    return 0;
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : DEFAULT_GGUF_PATH;
    char error[512];
    int checks = 0, fails = 0;

    h3_gguf_file store;
    if (!h3_gguf_open(path, &store, error, sizeof(error))) {
        printf("FAIL open: %s\n", error);
        return 1;
    }
    checks++;
    if (store.version != 3) {
        printf("FAIL version: got %u, expected 3\n", store.version);
        fails++;
    }
    if (store.tensor_count != 532 || store.n_tensors != 532) {
        printf("FAIL n_tensors: parsed %zu declared %llu, expected 532\n",
               store.tensor_count, (unsigned long long)store.n_tensors);
        fails++;
    }
    checks++;
    if (store.kv_count != 0 || store.n_kv != 0) {
        printf("FAIL n_kv: parsed %zu declared %llu, expected 0\n",
               store.kv_count, (unsigned long long)store.n_kv);
        fails++;
    }
    if (store.alignment != 32 || store.data_start != 45504) {
        printf("FAIL layout: alignment=%u data_start=%llu (expected 32/45504)\n",
               store.alignment, (unsigned long long)store.data_start);
        fails++;
    }
    printf("header: v%u tensors=%zu kv=%zu file=%llu metadata_end=%llu data_start=%llu\n",
           store.version, store.tensor_count, store.kv_count,
           (unsigned long long)store.file_size, (unsigned long long)store.metadata_end,
           (unsigned long long)store.data_start);

    /* Type census (matches the R1 GGUF audit). */
    checks++;
    size_t count_iq1s = 0, count_f32 = 0, count_f16 = 0, count_bf16 = 0, count_other = 0;
    for (size_t index = 0; index < store.tensor_count; index++) {
        const h3_gguf_tensor *tensor = h3_gguf_tensor_at(&store, index);
        if (!tensor) { fails++; break; }
        switch (tensor->ggml_type) {
            case 0: count_f32++; break;
            case 1: count_f16++; break;
            case 19: count_iq1s++; break;
            case 30: count_bf16++; break;
            default: count_other++; break;
        }
    }
    if (count_iq1s != 208 || count_f32 != 10 || count_f16 != 102 ||
        count_bf16 != 212 || count_other != 0) {
        printf("FAIL type census: iq1_s=%zu f32=%zu f16=%zu bf16=%zu other=%zu "
               "(expected 208/10/102/212/0)\n",
               count_iq1s, count_f32, count_f16, count_bf16, count_other);
        fails++;
    }
    printf("census: iq1_s=%zu bf16=%zu f16=%zu f32=%zu\n",
           count_iq1s, count_bf16, count_f16, count_f32);

    /* Data-region continuity: sorted by offset, first block starts at 0,
       every tensor starts exactly where the previous ended, and the union
       of all tensors covers file_size - data_start with no slack. */
    checks++;
    sortable_tensor *sorted = malloc(store.tensor_count * sizeof(*sorted));
    uint64_t total_bytes = 0;
    if (!sorted) {
        printf("FAIL continuity: out of memory\n");
        fails++;
    } else {
        for (size_t index = 0; index < store.tensor_count; index++) {
            const h3_gguf_tensor *tensor = h3_gguf_tensor_at(&store, index);
            sorted[index].name = tensor->name;
            sorted[index].offset = tensor->offset;
            sorted[index].nbytes = tensor->nbytes;
            total_bytes += tensor->nbytes;
        }
        qsort(sorted, store.tensor_count, sizeof(*sorted), compare_by_offset);
        int contiguous = 1;
        uint64_t expect = 0;
        for (size_t index = 0; index < store.tensor_count; index++) {
            if (sorted[index].offset != expect) {
                printf("FAIL continuity: tensor %s offset %llu, expected %llu\n",
                       sorted[index].name, (unsigned long long)sorted[index].offset,
                       (unsigned long long)expect);
                contiguous = 0;
                break;
            }
            expect = sorted[index].offset + sorted[index].nbytes;
        }
        if (contiguous && total_bytes != store.file_size - store.data_start) {
            printf("FAIL continuity: Σ tensors %llu != data region %llu\n",
                   (unsigned long long)total_bytes,
                   (unsigned long long)(store.file_size - store.data_start));
            contiguous = 0;
        }
        if (!contiguous) fails++;
        if (contiguous) {
            printf("continuity: %zu tensors packed 0..%llu == data region %llu\n",
                   store.tensor_count, (unsigned long long)expect,
                   (unsigned long long)(store.file_size - store.data_start));
        }

        /* Last tensor by offset: its final byte must sit at EOF. */
        checks++;
        const sortable_tensor *last = &sorted[store.tensor_count - 1];
        uint64_t last_end_absolute = store.data_start + last->offset + last->nbytes;
        printf("last tensor: %s %llu bytes, ends at absolute byte %llu (file size %llu)\n",
               last->name, (unsigned long long)last->nbytes,
               (unsigned long long)(last_end_absolute - 1),
               (unsigned long long)store.file_size);
        if (last_end_absolute != store.file_size) {
            printf("FAIL tail byte: last tensor ends at %llu, expected EOF %llu\n",
                   (unsigned long long)last_end_absolute,
                   (unsigned long long)store.file_size);
            fails++;
        } else {
            unsigned char *tail = malloc((size_t)last->nbytes);
            if (!tail || !h3_gguf_read_tensor_raw(&store, last->name, tail,
                                                  (size_t)last->nbytes,
                                                  error, sizeof(error))) {
                printf("FAIL tail read: %s\n", error);
                fails++;
            } else {
                unsigned char final_byte = tail[last->nbytes - 1];
                printf("tail byte: read last %llu bytes of %s ok, final raw byte 0x%02x\n",
                       (unsigned long long)last->nbytes, last->name, final_byte);
                checks++;
            }
            free(tail);
        }
        free(sorted);
    }

    /* F32 spot read: adaln_t_table is [8, 1025] f32, all values finite. */
    checks++;
    const h3_gguf_tensor *adaln = h3_gguf_find(&store, ADALN_NAME);
    if (!adaln) {
        printf("FAIL lookup: %s missing\n", ADALN_NAME);
        fails++;
    } else if (adaln->ggml_type != 0 || adaln->ndim != 2 ||
               adaln->dims[0] != 8 || adaln->dims[1] != 1025 ||
               adaln->nbytes != 1025ull * 8 * 4) {
        printf("FAIL adaln meta: type=%u dims=%llu x %llu bytes=%llu\n",
               adaln->ggml_type, (unsigned long long)adaln->dims[0],
               (unsigned long long)adaln->dims[1], (unsigned long long)adaln->nbytes);
        fails++;
    } else {
        float *table = malloc((size_t)adaln->nbytes);
        if (!table || !h3_gguf_read_tensor_raw(&store, ADALN_NAME, table,
                                               (size_t)adaln->nbytes,
                                               error, sizeof(error))) {
            printf("FAIL adaln read: %s\n", error);
            fails++;
        } else {
            size_t values = (size_t)(adaln->nbytes / 4);
            size_t nonfinite = 0;
            float min_value = 0.0f, max_value = 0.0f;
            for (size_t index = 0; index < values; index++) {
                if (!isfinite(table[index])) { nonfinite++; continue; }
                if (index == 0 || table[index] < min_value) min_value = table[index];
                if (index == 0 || table[index] > max_value) max_value = table[index];
            }
            printf("adaln_t_table: %zu f32 values read, %zu non-finite, range [%.6f, %.6f]\n",
                   values, nonfinite, min_value, max_value);
            if (nonfinite != 0) {
                printf("FAIL adaln finite: %zu non-finite floats\n", nonfinite);
                fails++;
            }
        }
        free(table);
    }

    /* IQ1_S spot read: blocks.0.attn.qkv_proj.weight.  The first two bytes
       of the first block_iq1_s are the fp16 scale d; it must decode to a
       positive finite value. */
    checks++;
    const h3_gguf_tensor *qkv = h3_gguf_find(&store, QKV0_NAME);
    if (!qkv) {
        printf("FAIL lookup: %s missing\n", QKV0_NAME);
        fails++;
    } else if (qkv->ggml_type != 19 || qkv->ndim != 2 ||
               qkv->dims[0] != 5376 || qkv->dims[1] != 21504) {
        printf("FAIL qkv meta: type=%u dims=%llu x %llu\n",
               qkv->ggml_type, (unsigned long long)qkv->dims[0],
               (unsigned long long)qkv->dims[1]);
        fails++;
    } else {
        uint64_t expected_bytes;
        if (!h3_ggml_row_bytes(19, qkv->elements, &expected_bytes,
                               error, sizeof(error)) ||
            expected_bytes != qkv->nbytes) {
            printf("FAIL qkv size: row_bytes says %llu (%s), tensor says %llu\n",
                   (unsigned long long)expected_bytes, error,
                   (unsigned long long)qkv->nbytes);
            fails++;
        } else {
            unsigned char *head = malloc((size_t)qkv->nbytes);
            if (!head || !h3_gguf_read_tensor_raw(&store, QKV0_NAME, head,
                                                  (size_t)qkv->nbytes,
                                                  error, sizeof(error))) {
                printf("FAIL qkv read: %s\n", error);
                fails++;
            } else {
                uint16_t scale_bits = (uint16_t)((uint16_t)head[0] | ((uint16_t)head[1] << 8));
                float scale = h3_f16_to_f32(scale_bits);
                printf("qkv iq1_s: %llu bytes read, first block scale bits 0x%04x -> %.6f\n",
                       (unsigned long long)qkv->nbytes, scale_bits, scale);
                if (!isfinite(scale) || !(scale > 0.0f)) {
                    printf("FAIL qkv scale: first iq1_s block d=%.6f, expected > 0\n", scale);
                    fails++;
                }
            }
            free(head);
        }
    }

    /* Unsupported ggml types: anything outside the fork enum must be
       rejected by name, including type 45+ and the retired holes. */
    checks++;
    uint64_t scratch_bytes;
    int rejected = 0, accepted_wrong = 0;
    const uint32_t bad_types[] = { 4, 5, 31, 32, 33, 36, 37, 38, 45, 46, 100, 65535 };
    for (size_t index = 0; index < sizeof(bad_types) / sizeof(bad_types[0]); index++) {
        error[0] = '\0';
        if (h3_ggml_row_bytes(bad_types[index], 256, &scratch_bytes,
                              error, sizeof(error))) {
            printf("FAIL type reject: ggml type %u accepted\n", bad_types[index]);
            accepted_wrong++;
        } else if (!strstr(error, "unknown")) {
            printf("FAIL type reject: ggml type %u rejected without naming it: %s\n",
                   bad_types[index], error);
            accepted_wrong++;
        } else {
            rejected++;
        }
    }
    /* Divisibility is enforced too: 100 elements cannot fill 256-wide IQ1_S
       blocks, and plain types must still be accepted. */
    if (!h3_ggml_row_bytes(19, 100, &scratch_bytes, error, sizeof(error)))
        rejected++;
    else {
        printf("FAIL iq1_s divisibility: 100 elements accepted\n");
        accepted_wrong++;
    }
    if (!h3_ggml_row_bytes(0, 8200, &scratch_bytes, error, sizeof(error)) ||
        scratch_bytes != 8200 * 4) {
        printf("FAIL f32 row bytes sanity\n");
        accepted_wrong++;
    }
    if (rejected != 13 || accepted_wrong != 0) {
        printf("FAIL type gating: %d properly rejected, %d wrong\n", rejected, accepted_wrong);
        fails++;
    } else {
        printf("type gating: %d unknown/ill-formed requests properly rejected\n", rejected);
    }

    h3_gguf_close(&store);
    printf("\nRESULT checks=%d fails=%d\n", checks, fails);
    return fails ? 1 : 0;
}
