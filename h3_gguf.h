/* GGUF v1-v3 container reader for native IQ1_S assets (R2).
 *
 * Two distinct type domains must not be confused:
 *  - Key-value scalar types follow the llama.cpp GGUF file specification
 *    (authoritative for the container): UINT8=0, INT8=1, UINT16=2, INT16=3,
 *    UINT32=4, INT32=5, FLOAT32=6, BOOL=7, STRING=8, ARRAY=9, UINT64=10,
 *    INT64=11, FLOAT64=12.  (BOOL8 is not part of the spec; BOOL is a 1-byte
 *    0/1 value.)
 *  - Tensor element types follow the *sd.cpp ggml fork* enum
 *    (~/tools/stable-diffusion.cpp ggml/include/ggml.h @ e239e4b, upstream
 *    pin 13b9d92): F32=0, F16=1, Q8_0=8, IQ1_S=19, IQ1_M=29, BF16=30,
 *    ... — deliberately NOT the llama.cpp mainline numbering.
 *
 * Tensor dims are stored in GGUF order (fastest-varying dimension first,
 * i.e. reversed relative to the safetensors "shape" convention the engine
 * already uses).  Tensor offsets are relative to the aligned data region
 * start (general.alignment, default 32).  Reader mirrors h3_safetensors.c:
 * pread only, error-buffer API, C11, zero dependencies. */
#ifndef H3_GGUF_H
#define H3_GGUF_H

#include <stddef.h>
#include <stdint.h>

#define H3_GGUF_MAX_DIMS 8
#define H3_GGUF_DEFAULT_ALIGNMENT 32u

/* GGUF value types (llama.cpp GGUF spec numbering). */
typedef enum {
    H3_GGUF_KV_UINT8 = 0,
    H3_GGUF_KV_INT8 = 1,
    H3_GGUF_KV_UINT16 = 2,
    H3_GGUF_KV_INT16 = 3,
    H3_GGUF_KV_UINT32 = 4,
    H3_GGUF_KV_INT32 = 5,
    H3_GGUF_KV_FLOAT32 = 6,
    H3_GGUF_KV_BOOL = 7,
    H3_GGUF_KV_STRING = 8,
    H3_GGUF_KV_ARRAY = 9,
    H3_GGUF_KV_UINT64 = 10,
    H3_GGUF_KV_INT64 = 11,
    H3_GGUF_KV_FLOAT64 = 12
} h3_gguf_kv_type;

typedef struct {
    char *key;
    uint32_t type;             /* h3_gguf_kv_type */
    union {
        uint8_t unsigned8;
        int8_t signed8;
        uint16_t unsigned16;
        int16_t signed16;
        uint32_t unsigned32;
        int32_t signed32;
        uint64_t unsigned64;
        int64_t signed64;
        float float32;
        double float64;
        int boolean;
        char *string;
    } value;
    /* Only meaningful when type == H3_GGUF_KV_ARRAY: element type (never
       ARRAY itself — nested arrays are rejected) and element count.  Array
       contents are validated for bounds but their values are not retained:
       weight data lives in the tensor table, and KV arrays in practice hold
       small metadata (vocabulary lists are read by the tokenizer path). */
    uint32_t array_type;
    uint64_t array_count;
} h3_gguf_kv;

typedef struct {
    char *name;                /* full GGUF key, e.g. model.diffusion_model.* */
    uint32_t ggml_type;        /* sd.cpp fork ggml_type enum value */
    const char *type_name;     /* traits table name, e.g. "iq1_s" */
    int ndim;
    uint64_t dims[H3_GGUF_MAX_DIMS];
    uint64_t elements;         /* product of dims */
    uint64_t nbytes;           /* ggml_row_size equivalent:
                                  (elements / block_size) * type_size */
    uint64_t offset;           /* bytes from data_start */
} h3_gguf_tensor;

typedef struct {
    char *path;
    uint32_t version;          /* 1..3 accepted */
    uint64_t file_size;
    uint64_t n_tensors;        /* as declared in the header */
    uint64_t n_kv;
    uint32_t alignment;        /* general.alignment or the default 32 */
    uint64_t metadata_end;     /* offset just past the tensor info table */
    uint64_t data_start;       /* metadata_end rounded up to alignment */
    h3_gguf_kv *kvs;
    size_t kv_count;
    h3_gguf_tensor *tensors;
    size_t tensor_count;
} h3_gguf_file;

int h3_gguf_open(const char *path, h3_gguf_file *store,
                 char *error, size_t error_size);
void h3_gguf_close(h3_gguf_file *store);

/* Enumeration: index in file order (tensor info table order). */
const h3_gguf_tensor *h3_gguf_tensor_at(const h3_gguf_file *store, size_t index);
const h3_gguf_tensor *h3_gguf_find(const h3_gguf_file *store, const char *name);
const h3_gguf_kv *h3_gguf_kv_find(const h3_gguf_file *store, const char *key);

uint64_t h3_gguf_tensor_elements(const h3_gguf_tensor *tensor);
uint64_t h3_gguf_tensor_bytes(const h3_gguf_tensor *tensor);

/* Reads the full raw payload of tensor `name` (packed quantized blocks, no
   interpretation) into dst; capacity must be >= the tensor's byte size. */
int h3_gguf_read_tensor_raw(const h3_gguf_file *store, const char *name,
                            void *dst, size_t capacity,
                            char *error, size_t error_size);

/* ggml_row_size equivalent for the sd.cpp fork enum: bytes for `elements`
   contiguous elements of `ggml_type`.  Fails (with a message naming the
   type) for unknown/retired ggml types, including every value > 44. */
int h3_ggml_row_bytes(uint32_t ggml_type, uint64_t elements,
                      uint64_t *bytes, char *error, size_t error_size);
int h3_ggml_type_info(uint32_t ggml_type, uint32_t *block_elements,
                      uint32_t *block_bytes);

#endif
