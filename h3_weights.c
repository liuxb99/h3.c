#include "h3_weights.h"

#include <dirent.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct h3_weight_store {
    h3_st_header *headers;
    size_t count;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int safetensors_name(const char *name) {
    static const char suffix[] = ".safetensors";
    size_t length = strlen(name);
    return length > sizeof(suffix) - 1 &&
           strcmp(name + length - (sizeof(suffix) - 1), suffix) == 0;
}

static int compare_paths(const void *left, const void *right) {
    const char *const *a = left;
    const char *const *b = right;
    return strcmp(*a, *b);
}

static void free_paths(char **paths, size_t count) {
    if (!paths) return;
    for (size_t index = 0; index < count; index++) free(paths[index]);
    free(paths);
}

h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size) {
    if (!directory || !*directory) {
        fail(error, error_size, "weight directory is required");
        return NULL;
    }
    DIR *stream = opendir(directory);
    if (!stream) {
        fail(error, error_size, "cannot open weight directory: %s", directory);
        return NULL;
    }
    char **paths = NULL;
    size_t count = 0;
    size_t capacity = 0;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (!safetensors_name(entry->d_name)) continue;
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 8;
            char **grown = realloc(paths, next * sizeof(*grown));
            if (!grown) {
                fail(error, error_size, "out of memory listing weight shards");
                closedir(stream);
                free_paths(paths, count);
                return NULL;
            }
            paths = grown;
            capacity = next;
        }
        size_t length = strlen(directory) + strlen(entry->d_name) + 2;
        paths[count] = malloc(length);
        if (!paths[count]) {
            fail(error, error_size, "out of memory resolving a weight shard");
            closedir(stream);
            free_paths(paths, count);
            return NULL;
        }
        snprintf(paths[count], length, "%s/%s", directory, entry->d_name);
        count++;
    }
    closedir(stream);
    if (!count) {
        fail(error, error_size, "no safetensors shards in %s", directory);
        free(paths);
        return NULL;
    }
    qsort(paths, count, sizeof(*paths), compare_paths);
    h3_weight_store *store = calloc(1, sizeof(*store));
    if (!store) {
        fail(error, error_size, "out of memory creating weight store");
        free_paths(paths, count);
        return NULL;
    }
    store->headers = calloc(count, sizeof(*store->headers));
    if (!store->headers) {
        fail(error, error_size, "out of memory allocating weight headers");
        free(store);
        free_paths(paths, count);
        return NULL;
    }
    store->count = count;
    for (size_t index = 0; index < count; index++) {
        char detail[384];
        if (!h3_st_read_header(paths[index], &store->headers[index], detail,
                               sizeof(detail))) {
            fail(error, error_size, "%s", detail);
            free_paths(paths, count);
            h3_weight_store_free(store);
            return NULL;
        }
    }
    free_paths(paths, count);
    return store;
}

void h3_weight_store_free(h3_weight_store *store) {
    if (!store) return;
    for (size_t index = 0; index < store->count; index++) {
        h3_st_free_header(&store->headers[index]);
    }
    free(store->headers);
    free(store);
}

size_t h3_weight_store_shards(const h3_weight_store *store) {
    return store ? store->count : 0;
}

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header) {
    if (header) *header = NULL;
    if (!store || !name) return NULL;
    for (size_t index = 0; index < store->count; index++) {
        const h3_st_tensor *tensor = h3_st_find(&store->headers[index], name);
        if (tensor) {
            if (header) *header = &store->headers[index];
            return tensor;
        }
    }
    /* Comfy-Org single-file checkpoints drop the language-model prefix
       and nest the visual tower one level less: try those spellings as
       aliases so the directory layout and the single-file layout share
       one consumer code path. */
    char alias[256];
    static const struct { const char *from, *to; } aliases[] = {
        { "model.language_model.", "model." },
        { "model.visual.", "visual." },
        { "model.", "model.language_model." },
        { "visual.", "model.visual." },
    };
    for (size_t a = 0; a < sizeof(aliases) / sizeof(aliases[0]); a++) {
        size_t flen = strlen(aliases[a].from);
        if (strncmp(name, aliases[a].from, flen)) continue;
        if (snprintf(alias, sizeof(alias), "%s%s", aliases[a].to, name + flen)
                >= (int)sizeof(alias)) break;
        for (size_t index = 0; index < store->count; index++) {
            const h3_st_tensor *tensor = h3_st_find(&store->headers[index], alias);
            if (tensor) {
                if (header) *header = &store->headers[index];
                return tensor;
            }
        }
        break;
    }
    return NULL;
}

int h3_weight_is_quantized(const h3_weight_store *store, const char *name) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    char sidecar[1024];
    if (!tensor || (tensor->dtype != H3_DTYPE_I8 &&
                    tensor->dtype != H3_DTYPE_U8)) return 0;
    /* Comfy files key sidecars by stem: <stem>.weight + <stem>_scale +
       <stem>_quant...  actually <stem>.weight_scale / <stem>.comfy_quant;
       accept the appended spelling too. */
    if (strlen(name) > 7 && !strcmp(name + strlen(name) - 7, ".weight")) {
        char stem[1024];
        snprintf(stem, sizeof stem, "%.*s",
                 (int)(strlen(name) - 7), name);
        snprintf(sidecar, sizeof sidecar, "%s.comfy_quant", stem);
        if (h3_weight_find(store, sidecar, &header)) return 1;
    }
    snprintf(sidecar, sizeof sidecar, "%s.comfy_quant", name);
    return h3_weight_find(store, sidecar, &header) != NULL;
}

static int sidecar_name_for(const char *name, char *sidecar, size_t size) {
    size_t length = strlen(name);
    if (length > 7 && !strcmp(name + length - 7, ".weight")) {
        snprintf(sidecar, size, "%.*s.comfy_quant", (int)(length - 7), name);
        if (length + 14 >= size) return -1;
        return 1;
    }
    snprintf(sidecar, size, "%s.comfy_quant", name);
    return 0;
}

int h3_weight_quant_info(const h3_weight_store *store, const char *name,
                         uint32_t *groupsize, int *convrot,
                         char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = NULL;
    char sidecar[1024];
    int tried = sidecar_name_for(name, sidecar, sizeof sidecar);
    tensor = h3_weight_find(store, sidecar, &header);
    if (!tensor && tried == 1) {
        snprintf(sidecar, sizeof sidecar, "%s.comfy_quant", name);
        tensor = h3_weight_find(store, sidecar, &header);
    }
    if (!tensor) {
        fail(error, error_size, "quantized weight lacks sidecar: %s", name);
        return 0;
    }
    if (tensor->dtype != H3_DTYPE_U8 && tensor->dtype != H3_DTYPE_I8) {
        fail(error, error_size, "sidecar %s has dtype %s", sidecar,
             h3_dtype_name(tensor->dtype));
        return 0;
    }
    uint64_t bytes = tensor->data_end - tensor->data_begin;
    if (!bytes || bytes > 4096) {
        fail(error, error_size, "sidecar %s has %llu bytes", sidecar,
             (unsigned long long)bytes);
        return 0;
    }
    char buffer[4097];
    FILE *stream = fopen(header->path, "rb");
    if (!stream) {
        fail(error, error_size, "cannot open %s: %s", header->path,
             strerror(errno));
        return 0;
    }
    size_t read = 0;
    if (fseeko(stream, (off_t)tensor->file_offset, SEEK_SET) == 0)
        read = fread(buffer, 1, (size_t)bytes, stream);
    fclose(stream);
    if (read != (size_t)bytes) {
        fail(error, error_size, "cannot read sidecar %s", sidecar);
        return 0;
    }
    buffer[bytes] = '\0';
    /* Gate: this engine only consumes Comfy int8_tensorwise payloads. */
    if (!strstr(buffer, "\"format\"") || !strstr(buffer, "int8_tensorwise")) {
        fail(error, error_size, "unsupported quant format for %s: %s", name,
             buffer);
        return 0;
    }
    int convrot_true = strstr(buffer, "\"convrot\": true") ||
                       strstr(buffer, "\"convrot\":true");
    uint32_t group = 0;
    const char *key = strstr(buffer, "convrot_groupsize");
    if (key) {
        key = strchr(key, ':');
        if (key) group = (uint32_t)strtoul(key + 1, NULL, 10);
    }
    if (convrot_true) {
        if (!group || (group & (group - 1)) || group > 256) {
            fail(error, error_size,
                 "convrot sidecar for %s lacks a usable groupsize: %s", name,
                 buffer);
            return 0;
        }
        *convrot = 1;
        *groupsize = group;
        return 1;
    }
    *convrot = 0;
    *groupsize = 0;
    return 1;
}

static h3_gpu_tensor *load_tensor(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape, h3_dtype dtype,
                                  char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(store, name, &header);
    if (!tensor) {
        fail(error, error_size, "required weight is absent: %s", name);
        return NULL;
    }
    if (tensor->dtype != dtype || tensor->ndim != ndim) {
        fail(error, error_size, "weight %s has dtype/rank %s/%d, expected %s/%d",
             name, h3_dtype_name(tensor->dtype), tensor->ndim,
             h3_dtype_name(dtype), ndim);
        return NULL;
    }
    uint64_t elements = 1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        if (tensor->shape[dimension] != shape[dimension]) {
            fail(error, error_size, "weight %s shape mismatch at dimension %d",
                 name, dimension);
            return NULL;
        }
        if (shape[dimension] && elements > UINT64_MAX / shape[dimension]) {
            fail(error, error_size, "weight %s shape overflows", name);
            return NULL;
        }
        elements *= shape[dimension];
    }
    if (elements > SIZE_MAX) {
        fail(error, error_size, "weight %s is too large for this process", name);
        return NULL;
    }
    h3_gpu_tensor *result;
    if (dtype == H3_DTYPE_BF16) {
        result = h3_gpu_tensor_load_bf16(gpu, header->path, tensor->file_offset,
                                         (size_t)elements);
    } else if (dtype == H3_DTYPE_F32) {
        result = h3_gpu_tensor_load_f32(gpu, header->path, tensor->file_offset,
                                        (size_t)elements);
    } else {
        /* One-byte quantized payload: I8 convrot weights and U8 packed
           fp4 weights upload as raw bytes into an I8-buffered tensor. */
        result = h3_gpu_tensor_load_i8(gpu, header->path, tensor->file_offset,
                                       (size_t)elements);
    }
    if (!result) {
        fail(error, error_size, "cannot load %s: %s", name, h3_gpu_error(gpu));
    }
    return result;
}

h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_BF16,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_F32,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_i8(const h3_weight_store *store, h3_gpu *gpu,
                                 const char *name, int ndim,
                                 const uint64_t *shape,
                                 char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_I8,
                       error, error_size);
}

h3_gpu_tensor *h3_weight_load_u8(const h3_weight_store *store, h3_gpu *gpu,
                                 const char *name, int ndim,
                                 const uint64_t *shape,
                                 char *error, size_t error_size) {
    return load_tensor(store, gpu, name, ndim, shape, H3_DTYPE_U8,
                       error, error_size);
}

/* Build the transposed Hadamard used by the convrot dequant kernel:
   out[j * groupsize + k] = H[k][j], with H built by iterative
   Walsh-Paley (Sylvester) expansion from the 4x4 core.  Identical
   construction to the per-block path in h3_dit.c. */
static float *hadamard_transposed(uint32_t groupsize) {
    static const float h4[4][4] = {
        {1.0f, 1.0f, 1.0f, -1.0f}, {1.0f, 1.0f, -1.0f, 1.0f},
        {1.0f, -1.0f, 1.0f, 1.0f}, {-1.0f, 1.0f, 1.0f, 1.0f}};
    size_t width = 1;
    float *had = malloc((size_t)groupsize * groupsize * sizeof(*had));
    if (!had) return NULL;
    had[0] = 1.0f;
    while (width < groupsize) {
        float *grown = malloc(width * width * 16 * sizeof(*grown));
        if (!grown) { free(had); return NULL; }
        for (size_t i = 0; i < width; i++)
            for (size_t j = 0; j < width; j++)
                for (int k = 0; k < 4; k++)
                    for (int l = 0; l < 4; l++)
                        grown[(i * 4 + k) * width * 4 + j * 4 + l] =
                            had[i * width + j] * h4[k][l];
        free(had);
        had = grown;
        width *= 4;
    }
    for (size_t j = 0; j < groupsize; j++)
        for (size_t k = j + 1; k < groupsize; k++) {
            float hold = had[j * groupsize + k];
            had[j * groupsize + k] = had[k * groupsize + j];
            had[k * groupsize + j] = hold;
        }
    return had;
}

h3_gpu_tensor *h3_weight_load_bf16_auto(const h3_weight_store *store,
                                        h3_gpu *gpu, const char *name,
                                        uint64_t rows, uint64_t columns,
                                        char *error, size_t error_size) {
    /* 2D weight that the checkpoint may store either as BF16 or as
       group-quantized int8 (convrot sidecar).  Strict-BF16 consumers
       (AdaLN schedule) use this so a convrot-int8 checkpoint loads
       without a separate materialization pass; unquantized tensors
       take the original strict path untouched. */
    if (error && error_size) error[0] = '\0';
    if (!h3_weight_is_quantized(store, name))
        { uint64_t shape[] = {rows, columns};
          return h3_weight_load_bf16(store, gpu, name, 2, shape, error,
                                     error_size); }
    uint32_t groupsize = 0;
    int convrot = 0;
    if (!h3_weight_quant_info(store, name, &groupsize, &convrot, error,
                              error_size))
        return NULL;
    char scale_name[1024];
    if (strlen(name) > 7 && !strcmp(name + strlen(name) - 7, ".weight"))
        snprintf(scale_name, sizeof scale_name, "%.*s.weight_scale",
                 (int)(strlen(name) - 7), name);
    else
        snprintf(scale_name, sizeof scale_name, "%s.weight_scale", name);
    uint64_t weight_shape[] = {rows, columns};
    uint64_t scale_shape[] = {rows, 1};
    h3_gpu_tensor *weight = h3_weight_load_i8(store, gpu, name, 2,
                                              weight_shape, error,
                                              error_size);
    if (!weight) return NULL;
    h3_gpu_tensor *scale = h3_weight_load_f32(store, gpu, scale_name, 2,
                                              scale_shape, error,
                                              error_size);
    if (!scale) { h3_gpu_tensor_free(weight); return NULL; }
    h3_gpu_tensor *had = NULL, *out = NULL;
    if (convrot) {
        if (groupsize < 4 || (groupsize & (groupsize - 1)) ||
            (groupsize & 3) || columns % groupsize) {
            if (error && error_size)
                snprintf(error, error_size,
                         "%s: convrot group %u incompatible with %llu columns",
                         name, groupsize, (unsigned long long)columns);
            goto failed;
        }
        float *had_cpu = hadamard_transposed(groupsize);
        if (!had_cpu) {
            if (error && error_size)
                snprintf(error, error_size,
                         "out of memory building Hadamard %u", groupsize);
            goto failed;
        }
        had = h3_gpu_tensor_from_f32(gpu, had_cpu,
                                     (size_t)groupsize * groupsize);
        free(had_cpu);
        if (!had) {
            if (error && error_size)
                snprintf(error, error_size, "cannot upload Hadamard %u: %s",
                         groupsize, h3_gpu_error(gpu));
            goto failed;
        }
    }
    out = h3_gpu_tensor_new_bf16(gpu, (size_t)rows * columns);
    if (!out) {
        if (error && error_size)
            snprintf(error, error_size, "out of memory materializing %s",
                     name);
        goto failed;
    }
    int ok = convrot
        ? h3_gpu_dequant_convrot(gpu, out, weight, scale, had, NULL,
                                 (uint32_t)rows, (uint32_t)columns,
                                 groupsize)
        : h3_gpu_dequant_int8_rows(gpu, out, weight, scale, (uint32_t)rows,
                                   (uint32_t)columns);
    if (!ok) {
        if (error && error_size)
            snprintf(error, error_size, "%s dequant: %s", name,
                     h3_gpu_error(gpu));
        goto failed;
    }
    h3_gpu_tensor_free(scale);
    h3_gpu_tensor_free(weight);
    if (had) h3_gpu_tensor_free(had);
    return out;
failed:
    if (out) h3_gpu_tensor_free(out);
    if (had) h3_gpu_tensor_free(had);
    h3_gpu_tensor_free(scale);
    h3_gpu_tensor_free(weight);
    return NULL;
}
