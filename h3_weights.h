#ifndef H3_WEIGHTS_H
#define H3_WEIGHTS_H

#include "h3_gpu.h"
#include "h3_safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_weight_store h3_weight_store;

/* Open every safetensors header in a component directory without reading
 * tensor payloads. */
h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size);
void h3_weight_store_free(h3_weight_store *store);
size_t h3_weight_store_shards(const h3_weight_store *store);

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header);

/* Validate an exact BF16 shape, allocate a shared Metal buffer, and read the
 * payload directly into that buffer with no intermediate host allocation. */
h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);
/* True when name is registered as one-byte quantized payload (I8/U8)
   and carries its comfy_quant sidecar in the same store. */
int h3_weight_is_quantized(const h3_weight_store *store, const char *name);
/* Parse the comfy_quant sidecar for a quantized weight: gate on
   format int8_tensorwise, report whether convrot rotation was applied
   and the Hadamard groupsize.  Fails (0 + error) for unsupported
   sidecar semantics. */
int h3_weight_quant_info(const h3_weight_store *store, const char *name,
                         uint32_t *groupsize, int *convrot,
                         char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_i8(const h3_weight_store *store, h3_gpu *gpu,
                                 const char *name, int ndim,
                                 const uint64_t *shape,
                                 char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_u8(const h3_weight_store *store, h3_gpu *gpu,
                                 const char *name, int ndim,
                                 const uint64_t *shape,
                                 char *error, size_t error_size);
/* 2D strict-BF16 consumer entry point that also accepts convrot/int8
   quantized payloads: materializes BF16 via the verified convrot
   dequant when the checkpoint quantized the tensor. */
h3_gpu_tensor *h3_weight_load_bf16_auto(const h3_weight_store *store,
                                        h3_gpu *gpu, const char *name,
                                        uint64_t rows, uint64_t columns,
                                        char *error, size_t error_size);

#endif
