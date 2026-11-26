/* Native A8W8 acceptance tests (Phase 3).
 *
 * Section 1: bit-exact CPU oracle for h3_rotate_quantize_bf16_int8_rows.
 *   The oracle reproduces the kernel's exact float operation order:
 *   - digit 0: the four component formulas as written in the kernel,
 *   - digits 1-3: ((v + p1) + p2) - p3 with XOR partners per stage,
 *   - clipped-absmax row scale with the 1/16 fold on the stored scale,
 *   - rint (ties-to-even) + clamp(-127, 127) quantization.
 *   int8 codes must match byte-for-byte; scales bit-for-bit.
 *
 * Section 2: pairing identity for the native convrot GEMM on a synthetic
 *   256-group convrot layer (random int8 weights + per-row scales):
 *   native A8W8 output must match the dequantized reference within the
 *   int8 quantization budget (this catches wrong rotation matrix, wrong
 *   transpose, or a missing/duplicated 1/16 fold, each of which shows up
 *   as an O(1) or 16x error, not a rounding error).
 */
#include "h3_gpu.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    GROUP = 256,
    MAX_COLUMNS = 14336,
    MAX_ORACLE_ROWS = 384
};

static int failures;

static void fail(const char *message) {
    fprintf(stderr, "FAIL tests/test_native_rotquant.c: %s\n", message);
    failures++;
}

static void require(int condition, const char *message) {
    if (!condition) fail(message);
}

static uint64_t rng_state = 0x9E3779B97F4A7C15ULL;

static uint64_t splitmix64(void) {
    rng_state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = rng_state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/* bf16 code -> exact float (left shift by 16). */
static float bf16_to_f32(uint16_t code) {
    uint32_t bits = (uint32_t)code << 16;
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static float rand_bf16_value(uint16_t *code_out) {
    uint64_t raw = splitmix64();
    uint32_t sign = (uint32_t)(raw & 1u);
    uint32_t exponent = 100u + (uint32_t)((raw >> 1) % 14u); /* 2^-27..2^0 */
    uint32_t mantissa = (uint32_t)(raw >> 17) & 0x7fu;
    uint16_t code = (uint16_t)((sign << 15) | (exponent << 7) | mantissa);
    if (code_out) *code_out = code;
    return bf16_to_f32(code);
}

/* CPU oracle: exact stage composition in the kernel's operation order. */
static void rotate_row_cpu(const float *in, float *out, uint32_t columns) {
    for (uint32_t group = 0; group * GROUP < columns; group++) {
        const float *src = in + group * GROUP;
        float *dst = out + group * GROUP;
        float stage[GROUP];
        for (uint32_t lane = 0; lane < GROUP / 4; lane++) {
            float v0 = src[lane * 4 + 0], v1 = src[lane * 4 + 1];
            float v2 = src[lane * 4 + 2], v3 = src[lane * 4 + 3];
            stage[lane * 4 + 0] = ((v0 + v1) + v2) - v3;
            stage[lane * 4 + 1] = ((v0 + v1) - v2) + v3;
            stage[lane * 4 + 2] = ((v0 - v1) + v2) + v3;
            stage[lane * 4 + 3] = ((-v0 + v1) + v2) + v3;
        }
        for (uint32_t digit = 2; digit <= 4; digit += 2) {
            float next[GROUP];
            for (uint32_t e = 0; e < GROUP; e++) {
                next[e] = ((stage[e] + stage[e ^ (1u << digit)])
                           + stage[e ^ (2u << digit)])
                          - stage[e ^ (3u << digit)];
            }
            memcpy(stage, next, sizeof stage);
        }
        for (uint32_t e = 0; e < GROUP; e++) {
            dst[e] = ((stage[e] + stage[e ^ (1u << 6)])
                      + stage[e ^ (2u << 6)])
                     - stage[e ^ (3u << 6)];
        }
    }
}

static int32_t quant_rint(float value) {
    if (!(value < 0.0f) && !(value >= 0.0f)) return 0; /* NaN -> saturates */
    if (value > 2147483000.0f) return 127;
    if (value < -2147483000.0f) return -127;
    double rounded = rint((double)value);
    if (rounded > 127.0) return 127;
    if (rounded < -127.0) return -127;
    return (int32_t)rounded;
}

/* Quantizes expected scale/code for one row into caller buffers. */
static void oracle_row(const float *rotated, uint32_t columns, float clip,
                       float *scale_out, int8_t *code_out) {
    float max_abs = 0.0f;
    for (uint32_t column = 0; column < columns; column++) {
        float magnitude = fabsf(rotated[column]);
        if (magnitude > max_abs) max_abs = magnitude;
    }
    float clipped_max = max_abs * clip;
    float scale = clipped_max > 0.0f ? clipped_max / 127.0f : 1.0f / 127.0f;
    float inverse = clipped_max > 0.0f ? 127.0f / clipped_max : 127.0f;
    *scale_out = scale * 0.0625f;
    for (uint32_t column = 0; column < columns; column++)
        code_out[column] = (int8_t)quant_rint(rotated[column] * inverse);
}

static void fill_adversarial_bf16(uint16_t *values, uint32_t rows,
                                  uint32_t columns) {
    for (uint32_t row = 0; row < rows; row++)
        for (uint32_t column = 0; column < columns; column++)
            rand_bf16_value(&values[row * columns + column]);
    /* Row 0: all zeros (clipped==0 branch: scale 1/127 * 1/16, zero codes).*/
    memset(values, 0, (size_t)columns * sizeof values[0]);
    /* Row 1: one nonzero element, rest zero (outlier-dominated row). */
    if (rows > 1) {
        uint16_t code;
        rand_bf16_value(&code);
        values[columns + 5] = code;
        values[columns + 200] = code;
    }
    /* Row 2: alternating +-ulp near 1.0 (digit-stage cancellation tests). */
    if (rows > 2)
        for (uint32_t column = 0; column < columns; column++)
            values[2u * columns + column] =
                (uint16_t)(((column & 1u) << 15) | 0x3F80u);
    /* Row 3: near-largest safe bf16 (exponent 245: rotation x256 stays
       finite, exercising the 1/16 fold landing entirely on the scale). */
    if (rows > 3)
        for (uint32_t column = 0; column < columns; column++)
            values[3u * columns + column] = (uint16_t)0x7AFFu;
}

static void dump_fail_row(const uint16_t *host, uint32_t rows,
                          uint32_t columns, uint32_t row) {
    char name[96];
    FILE *f;
    snprintf(name, sizeof name, "rowfail_%ux%u_%u.bin", rows, columns, row);
    f = fopen(name, "wb");
    if (f) {
        fwrite(host + (size_t)row * columns, sizeof *host, columns, f);
        fclose(f);
        fprintf(stderr, "dumped failing row -> %s\n", name);
    }
}

static void test_shape(h3_gpu *gpu, uint32_t rows, uint32_t columns,
                       float clip) {
    uint32_t dispatch_rows = (rows + 127u) & ~127u;
    size_t elements = (size_t)dispatch_rows * columns;
    uint16_t *host = malloc((size_t)rows * columns * sizeof *host);
    float *rotated = malloc(columns * sizeof *rotated);
    float *f32_row = malloc(columns * sizeof *f32_row);
    int8_t *expected_codes = malloc(elements);
    float *expected_scales = malloc(dispatch_rows * sizeof *expected_scales);
    h3_gpu_tensor *input, *output, *scales;
    int8_t *got_codes = malloc(elements);
    float *got_scales = malloc(dispatch_rows * sizeof *got_scales);
    uint32_t row, column;
    int failures_before = failures;
    if (!host || !rotated || !f32_row || !expected_codes || !got_codes ||
        !expected_scales || !got_scales) {
        fail("oracle allocation failed");
        return;
    }
    fill_adversarial_bf16(host, rows, columns);
    /* Compute the full expected tensor before touching the GPU so a
       mismatch report can name the exact row/column. */
    for (row = 0; row < dispatch_rows; row++) {
        if (row < rows) {
            for (column = 0; column < columns; column++)
                f32_row[column] =
                    bf16_to_f32(host[(size_t)row * columns + column]);
            rotate_row_cpu(f32_row, rotated, columns);
            oracle_row(rotated, columns, clip, &expected_scales[row],
                       expected_codes + (size_t)row * columns);
        } else {
            /* Padding rows: scale stays 1/127 (unfolded), codes zero. */
            expected_scales[row] = 1.0f / 127.0f;
            memset(expected_codes + (size_t)row * columns, 0, columns);
        }
    }
    input = h3_gpu_tensor_from_bf16(gpu, host, (size_t)rows * columns);
    output = h3_gpu_tensor_new_i8(gpu, elements);
    scales = h3_gpu_tensor_new_f32(gpu, dispatch_rows);
    require(input && output && scales, "tensor allocation failed");
    require(h3_gpu_begin(gpu), "begin command stream (shape test)");
    if (!h3_gpu_quantize_bf16_int8_rotated_rows(gpu, output, scales, input,
                                                rows, dispatch_rows, columns,
                                                clip)) {
        fprintf(stderr, "FAIL rot-quant dispatch rows=%u cols=%u: %s\n",
                rows, columns, h3_gpu_error(gpu));
        failures++;
        goto cleanup;
    }
    if (!h3_gpu_submit(gpu)) {
        fail("submit failed (shape test)");
        goto cleanup;
    }
    if (!h3_gpu_tensor_read_i8(output, got_codes, elements) ||
        h3_gpu_tensor_read_f32_range(scales, 0, got_scales, dispatch_rows) !=
            1) {
        fail("readback failed");
        goto cleanup;
    }
    for (row = 0; row < dispatch_rows; row++) {
        int scale_mismatch =
            memcmp(&got_scales[row], &expected_scales[row],
                   sizeof got_scales[0]) != 0;
        if (scale_mismatch) {
            fprintf(stderr,
                    "FAIL rot-quant rows=%u cols=%u clip=%g row=%u scale"
                    " got=%.9g want=%.9g\n",
                    rows, columns, clip, row, got_scales[row],
                    expected_scales[row]);
            if (row < rows)
                dump_fail_row(host, rows, columns, row);
            failures++;
        }
        if (memcmp(got_codes + (size_t)row * columns,
                   expected_codes + (size_t)row * columns, columns)) {
            uint32_t first = 0;
            const int8_t *want = expected_codes + (size_t)row * columns;
            const int8_t *got = got_codes + (size_t)row * columns;
            while (first < columns && got[first] == want[first]) first++;
            fprintf(stderr,
                    "FAIL rot-quant rows=%u cols=%u clip=%g row=%u col=%u"
                    " got=%d want=%d\n",
                    rows, columns, clip, row, first, got[first], want[first]);
            failures++;
        }
    }
    printf("shape rows=%u cols=%u clip=%g %s\n", rows, columns, clip,
           failures != failures_before ? "checked-with-failures"
                                       : "bit-exact OK");
cleanup:
    h3_gpu_tensor_free(input);
    h3_gpu_tensor_free(output);
    h3_gpu_tensor_free(scales);
    free(host);
    free(rotated);
    free(f32_row);
    free(expected_codes);
    free(got_codes);
}

/* ---- Section 2: pairing identity on a synthetic convrot layer ---- */

static double compare_against_reference(const uint16_t *got_bf16,
                                        const double *reference,
                                        uint32_t rows, uint32_t columns,
                                        const char *label) {
    double sum_square_error = 0.0, sum_square_reference = 0.0, max_error = 0.0,
           max_reference = 0.0;
    for (uint32_t index = 0; index < rows * columns; index++) {
        double got = bf16_to_f32(got_bf16[index]);
        double error = got - reference[index];
        sum_square_error += error * error;
        sum_square_reference += reference[index] * reference[index];
        if (fabs(error) > max_error) max_error = fabs(error);
        if (fabs(reference[index]) > max_reference)
            max_reference = fabs(reference[index]);
    }
    double rel_rms = sqrt(sum_square_error / (sum_square_reference + 1e-30));
    printf("%s: rel_rms=%.6f max_abs_err=%.6g max_abs_ref=%.6g\n", label,
           rel_rms, max_error, max_reference);
    return rel_rms;
}

static void test_pairing(h3_gpu *gpu) {
    enum { ROWS = 128, OUTPUTS = 256 };
    uint16_t *host_a = malloc((size_t)ROWS * GROUP * sizeof *host_a);
    int8_t *host_q = malloc((size_t)OUTPUTS * GROUP);
    float *host_sw = malloc(OUTPUTS * sizeof *host_sw);
    float *host_had = malloc(GROUP * GROUP * sizeof *host_had);
    double *reference = malloc((size_t)ROWS * OUTPUTS * sizeof *reference);
    float *row_w_eff = malloc(GROUP * sizeof *row_w_eff);
    uint16_t *out_a = malloc((size_t)ROWS * OUTPUTS * sizeof *out_a);
    uint16_t *out_b = malloc((size_t)ROWS * OUTPUTS * sizeof *out_b);
    h3_gpu_tensor *a, *q, *sw, *had, *w_eff, *result_a, *result_b, *qact,
        *sact;
    uint32_t row, column, chunk;
    if (!host_a || !host_q || !host_sw || !host_had || !reference ||
        !row_w_eff || !out_a || !out_b) {
        fail("pairing allocation failed");
        return;
    }
    for (row = 0; row < ROWS; row++)
        for (column = 0; column < GROUP; column++)
            rand_bf16_value(&host_a[row * GROUP + column]);
    for (row = 0; row < OUTPUTS; row++) {
        for (column = 0; column < GROUP; column++) {
            int32_t code = (int32_t)(splitmix64() % 255u) - 127;
            host_q[row * GROUP + column] = (int8_t)code;
        }
        host_sw[row] = bf16_to_f32((uint16_t)(0x3800u + (splitmix64() & 0x7fu)));
    }
    /* host_had := exact matrix of the GPU quant kernel, probed columnwise
       with unit vectors through the CPU oracle (bit-exact per the shape
       section). This removes any basis-ordering ambiguity between the
       kron builder and the kernel's stage indexing. */
    {
        static float probe_in[GROUP], probe_out[GROUP];
        for (chunk = 0; chunk < GROUP; chunk++) {
            for (row = 0; row < GROUP; row++)
                probe_in[row] = (row == (size_t)chunk) ? 1.0f : 0.0f;
            rotate_row_cpu(probe_in, probe_out, GROUP);
            for (row = 0; row < GROUP; row++)
                host_had[(size_t)row * GROUP + chunk] = probe_out[row];
        }
    }
    /* Double-precision reference: out[i][j] = sum_k a[i][k] *
     * sw[j] * (1/16) * sum_c q[j][c] * H[k][c]. */
    for (row = 0; row < ROWS; row++) {
        double a_row[GROUP];
        for (column = 0; column < GROUP; column++)
            a_row[column] = bf16_to_f32(host_a[row * GROUP + column]);
        for (uint32_t output = 0; output < OUTPUTS; output++) {
            double sum = 0.0;
            for (column = 0; column < GROUP; column++) {
                row_w_eff[column] = 0.0f;
                for (chunk = 0; chunk < GROUP; chunk++)
                    row_w_eff[column] +=
                        (float)host_q[output * GROUP + chunk] *
                        host_had[(size_t)column * GROUP + chunk];
                row_w_eff[column] *= host_sw[output] * 0.0625f;
                sum += a_row[column] * (double)row_w_eff[column];
            }
            reference[row * OUTPUTS + output] = sum;
        }
    }
    a = h3_gpu_tensor_from_bf16(gpu, host_a, (size_t)ROWS * GROUP);
    q = h3_gpu_tensor_from_i8(gpu, host_q, (size_t)OUTPUTS * GROUP);
    sw = h3_gpu_tensor_from_f32(gpu, host_sw, OUTPUTS);
    had = h3_gpu_tensor_from_f32(gpu, host_had, GROUP * GROUP);
    w_eff = h3_gpu_tensor_new_bf16(gpu, (size_t)OUTPUTS * GROUP);
    result_a = h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * OUTPUTS);
    result_b = h3_gpu_tensor_new_bf16(gpu, (size_t)ROWS * OUTPUTS);
    qact = h3_gpu_tensor_new_i8(gpu, (size_t)ROWS * GROUP);
    sact = h3_gpu_tensor_new_f32(gpu, ROWS);
    require(a && q && sw && had && w_eff && result_a && result_b && qact &&
                sact,
            "pairing tensor allocation failed");
    require(h3_gpu_begin(gpu), "begin command stream (path A)");
    if (!h3_gpu_dequant_convrot(gpu, w_eff, q, sw, had, NULL, OUTPUTS, GROUP,
                                GROUP)) {
        fprintf(stderr, "FAIL dequant_convrot: %s\n", h3_gpu_error(gpu));
        failures++;
        goto pairing_cleanup;
    }
    if (!h3_gpu_linear_bf16(gpu, result_a, a, w_eff, NULL, ROWS, GROUP,
                            OUTPUTS)) {
        fprintf(stderr, "FAIL linear_bf16: %s\n", h3_gpu_error(gpu));
        failures++;
        goto pairing_cleanup;
    }
    if (!h3_gpu_submit(gpu)) fail("submit failed (path A)");
    if (!h3_gpu_tensor_read_bf16(result_a, out_a, (size_t)ROWS * OUTPUTS))
        fail("path A readback failed");
    require(setenv("H3_NATIVE_INT8", "1", 1) == 0, "setenv failed");
    require(h3_gpu_begin(gpu), "begin command stream (path B)");
    if (!h3_gpu_linear_int8_bf16(gpu, result_b, qact, sact, a, q, sw, ROWS,
                                 GROUP, OUTPUTS, 0)) {
        fprintf(stderr, "FAIL linear_int8_bf16 native: %s\n",
                h3_gpu_error(gpu));
        failures++;
        unsetenv("H3_NATIVE_INT8");
        goto pairing_cleanup;
    }
    unsetenv("H3_NATIVE_INT8");
    if (!h3_gpu_submit(gpu)) fail("submit failed (path B)");
    if (!h3_gpu_tensor_read_bf16(result_b, out_b, (size_t)ROWS * OUTPUTS))
        fail("path B readback failed");

    double rel_a = compare_against_reference(out_a, reference, ROWS, OUTPUTS,
                                             "A dequant+bf16 vs fp64 oracle");
    /* Discriminate quantizer output vs GEMM pairing: replay the engine's
       own quantized activation (dumped to host) against the raw int8
       weight on the host in plain linear order. If this matches the
       engine's GEMM output, the quantizer/scales are fine and the
       pairing (swizzle/permutation) is the problem. */
    int8_t *dump_codes = malloc((size_t)ROWS * GROUP);
    float *dump_scales = malloc(ROWS * sizeof *dump_scales);
    double *predicted = malloc((size_t)ROWS * OUTPUTS * sizeof *predicted);
    float *cpu_rotated = malloc(GROUP * sizeof *cpu_rotated);
    float *cpu_row = malloc(GROUP * sizeof *cpu_row);
    if (dump_codes && dump_scales && predicted && cpu_rotated && cpu_row &&
        h3_gpu_tensor_read_i8(qact, dump_codes, (size_t)ROWS * GROUP) &&
        h3_gpu_tensor_read_f32_range(sact, 0, dump_scales, ROWS) == 1) {
        /* Direct check: dequantized engine activations vs CPU oracle of
           the same (Ma)/16 rotation the pairing identity assumes. */
        double q_err = 0.0, q_signal = 0.0;
        uint32_t worst_row = 0, worst_col = 0;
        double worst_delta = 0.0;
        for (row = 0; row < ROWS; row++) {
            for (column = 0; column < GROUP; column++)
                cpu_row[column] =
                    bf16_to_f32(host_a[(size_t)row * GROUP + column]);
            rotate_row_cpu(cpu_row, cpu_rotated, GROUP);
            double scale_ref = 0.0;
            for (column = 0; column < GROUP; column++) {
                double m = fabs(cpu_rotated[column]) * 0.0625;
                if (m > scale_ref * 127.0) scale_ref = m / 127.0;
            }
            for (column = 0; column < GROUP; column++) {
                double got = (double)dump_codes[row * GROUP + column] *
                             (double)dump_scales[row];
                double want = (double)cpu_rotated[column] * 0.0625;
                q_err += (got - want) * (got - want);
                q_signal += want * want;
                if (fabs(got - want) > worst_delta) {
                    worst_delta = fabs(got - want);
                    worst_row = row;
                    worst_col = column;
                }
            }
            (void)scale_ref;
        }
        printf("qa dump vs CPU oracle (Ma)/16: rel_rms=%.6f row=%u col=%u "
               "max_delta=%.6g\n",
               sqrt(q_err / (q_signal + 1e-300)), worst_row, worst_col,
               worst_delta);
        for (row = 0; row < ROWS; row++)
            for (uint32_t output = 0; output < OUTPUTS; output++) {
                double sum = 0.0;
                for (column = 0; column < GROUP; column++)
                    sum += (double)dump_codes[row * GROUP + column] *
                           (double)host_q[output * GROUP + column];
                predicted[row * OUTPUTS + output] =
                    sum * (double)dump_scales[row] * (double)host_sw[output];
            }
        double rel_pred =
            compare_against_reference(out_b, predicted, ROWS, OUTPUTS,
                                      "B engine output vs host GEMM replay");
        if (rel_pred < 0.01)
            printf("verdict: quantizer+scales OK; GEMM pairing mismatch\n");
        else
            printf("verdict: engine GEMM output differs from its own "
                   "quantized input replay\n");
    } else {
        fail("GEMM replay dump failed");
    }
    double rel_b = compare_against_reference(out_b, reference, ROWS, OUTPUTS,
                                             "B native A8W8 vs fp64 oracle");
    /* Path A vs path B share the oracle scale; compare via the oracle-free
       difference is not exact, so bound each against the oracle instead. */
    if (rel_a > 0.015) fail("dequant reference path deviates from oracle");
    if (rel_b > 0.05) fail("native A8W8 deviates from oracle beyond budget");
    printf("pairing: rel_a=%.6f rel_b=%.6f (budget a<=0.015 b<=0.05)\n",
           rel_a, rel_b);
pairing_cleanup:
    h3_gpu_tensor_free(a);
    h3_gpu_tensor_free(q);
    h3_gpu_tensor_free(sw);
    h3_gpu_tensor_free(had);
    h3_gpu_tensor_free(w_eff);
    h3_gpu_tensor_free(result_a);
    h3_gpu_tensor_free(result_b);
    h3_gpu_tensor_free(qact);
    h3_gpu_tensor_free(sact);
    free(host_a);
    free(host_q);
    free(host_sw);
    free(host_had);
    free(reference);
    free(row_w_eff);
    free(out_a);
    free(out_b);
    free(dump_codes);
    free(dump_scales);
    free(predicted);
}

int main(int argc, char **argv) {
    const int pairing_only = argc > 1 && !strcmp(argv[1], "--pairing-only");
    const int shape2048 = argc > 1 && !strcmp(argv[1], "--shape-2048");
    char error[256];
    h3_gpu *gpu = h3_gpu_create("h3_shaders.metal", error, sizeof error);
    if (!gpu) {
        fprintf(stderr, "FAIL tests/test_native_rotquant.c: %s\n", error);
        return 1;
    }
    if (!h3_gpu_is_m5(gpu)) {
        printf("skip: native rot-quant path requires M5 NAX hardware\n");
        h3_gpu_free(gpu);
        return 0;
    }
    /* Bit-exact contract across shapes covering: single wave (1 block),
       dead-pair waves (cols=14336 -> 57 blocks vs 56), padding rows,
       and the production FC2 activation shape 2048x7168. */
    if (shape2048) {
        test_shape(gpu, 2048, 7168, 1.0f);
    } else if (!pairing_only) {
        test_shape(gpu, 1, 256, 1.0f);
        test_shape(gpu, 7, 256, 1.0f);
        test_shape(gpu, 128, 14336, 1.0f);
        test_shape(gpu, 384, 512, 1.0f);
        test_shape(gpu, 2048, 7168, 1.0f);
        test_shape(gpu, 384, 512, 0.999f);
    }
    if (!shape2048)
        test_pairing(gpu);
    h3_gpu_free(gpu);
    if (failures) {
        fprintf(stderr, "tests/test_native_rotquant.c: %d failures\n",
                failures);
        return 1;
    }
    printf("native rot-quant oracle + pairing identity: PASS\n");
    return 0;
}
