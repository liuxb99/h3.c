/* GGUF v1-v3 container reader (R2).  Mirrors h3_safetensors.c conventions:
 * pread-only, error-buffer API, no third-party dependencies, C11.
 *
 * The per-type block table below was measured from the built libggml-base of
 * ~/tools/stable-diffusion.cpp (commit e239e4b, upstream pin 13b9d92) using
 * ggml_blck_size()/ggml_type_size() for every enum value — the file we read
 * was written by that same ggml fork, so ggml_row_size semantics are exactly
 * (elements / blck_size) * type_size with no per-row grid extras. */
#include "h3_gguf.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define H3_GGUF_MAX_METADATA (256u * 1024u * 1024u)
#define H3_GGUF_MAX_GGML_TYPE 44u

typedef struct {
    uint32_t ggml_type;
    uint32_t blck_size;   /* elements per block */
    uint32_t type_size;   /* bytes per block */
    const char *name;
} h3_ggml_traits_entry;

/* sd.cpp fork ggml_type numbering; retired enum values (4, 5, 31-33, 36-38)
   are absent and therefore rejected by every lookup. */
static const h3_ggml_traits_entry h3_ggml_traits[] = {
    { 0, 1, 4, "f32" },
    { 1, 1, 2, "f16" },
    { 2, 32, 18, "q4_0" },
    { 3, 32, 20, "q4_1" },
    { 6, 32, 22, "q5_0" },
    { 7, 32, 24, "q5_1" },
    { 8, 32, 34, "q8_0" },
    { 9, 32, 36, "q8_1" },
    { 10, 256, 84, "q2_k" },
    { 11, 256, 110, "q3_k" },
    { 12, 256, 144, "q4_k" },
    { 13, 256, 176, "q5_k" },
    { 14, 256, 210, "q6_k" },
    { 15, 256, 292, "q8_k" },
    { 16, 256, 66, "iq2_xxs" },
    { 17, 256, 74, "iq2_xs" },
    { 18, 256, 98, "iq3_xxs" },
    { 19, 256, 50, "iq1_s" },
    { 20, 32, 18, "iq4_nl" },
    { 21, 256, 110, "iq3_s" },
    { 22, 256, 82, "iq2_s" },
    { 23, 256, 136, "iq4_xs" },
    { 24, 1, 1, "i8" },
    { 25, 1, 2, "i16" },
    { 26, 1, 4, "i32" },
    { 27, 1, 8, "i64" },
    { 28, 1, 8, "f64" },
    { 29, 256, 56, "iq1_m" },
    { 30, 1, 2, "bf16" },
    { 34, 256, 54, "tq1_0" },
    { 35, 256, 66, "tq2_0" },
    { 39, 32, 17, "mxfp4" },
    { 40, 64, 36, "nvfp4" },
    { 41, 128, 18, "q1_0" },
    { 42, 64, 18, "q2_0" },
    { 43, 1, 1, "f8_e4m3" },
    { 44, 1, 1, "f8_e5m2" },
};

static const h3_ggml_traits_entry *h3_ggml_traits_for(uint32_t ggml_type) {
    for (size_t index = 0; index < sizeof(h3_ggml_traits) / sizeof(h3_ggml_traits[0]); index++) {
        if (h3_ggml_traits[index].ggml_type == ggml_type)
            return &h3_ggml_traits[index];
    }
    return NULL;
}

/* Bytes of each scalar KV value on disk (STRING and ARRAY are variable). */
static const uint32_t h3_gguf_scalar_bytes[13] = {
    1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8
};

typedef struct {
    const char *path;
    int descriptor;
    unsigned char *buffer;
    size_t capacity;
    size_t filled;
    size_t at;
    int eof;
    uint64_t file_size;
    char *error;
    size_t error_size;
} h3_gguf_cursor;

static int h3_gguf_fail(h3_gguf_cursor *cursor, const char *message) {
    if (cursor->error && cursor->error_size)
        snprintf(cursor->error, cursor->error_size, "%s", message);
    return 0;
}

static int h3_gguf_failf(h3_gguf_cursor *cursor, const char *format, ...) {
    if (cursor->error && cursor->error_size) {
        va_list arguments;
        va_start(arguments, format);
        vsnprintf(cursor->error, cursor->error_size, format, arguments);
        va_end(arguments);
    }
    return 0;
}

/* Reads more of the metadata region into the growth buffer.
   Returns 1 on progress, 0 at clean EOF, -1 on I/O or cap failure. */
static int h3_gguf_fill(h3_gguf_cursor *cursor) {
    if (cursor->filled == cursor->capacity) {
        size_t next = cursor->capacity + cursor->capacity / 2 + 4096;
        if (next > H3_GGUF_MAX_METADATA) next = H3_GGUF_MAX_METADATA;
        if (next <= cursor->capacity) {
            h3_gguf_fail(cursor, "GGUF metadata exceeds reader cap");
            return -1;
        }
        unsigned char *grown = realloc(cursor->buffer, next);
        if (!grown) {
            h3_gguf_fail(cursor, "out of memory reading GGUF metadata");
            return -1;
        }
        cursor->buffer = grown;
        cursor->capacity = next;
    }
    ssize_t count = pread(cursor->descriptor, cursor->buffer + cursor->filled,
                          cursor->capacity - cursor->filled, (off_t)cursor->filled);
    if (count < 0)
        return h3_gguf_failf(cursor, "%s: %s", cursor->path, strerror(errno)) ? -1 : -1;
    if (count == 0) {
        cursor->eof = 1;
        return 0;
    }
    cursor->filled += (size_t)count;
    return 1;
}

static int h3_gguf_ensure(h3_gguf_cursor *cursor, size_t bytes) {
    while (cursor->filled < cursor->at + bytes) {
        int result = h3_gguf_fill(cursor);
        if (result == 1) continue;
        if (result == 0)
            return h3_gguf_fail(cursor, "unexpected end of GGUF metadata");
        return 0;
    }
    return 1;
}

static int h3_gguf_read_bytes(h3_gguf_cursor *cursor, void *dst, size_t bytes) {
    if (!h3_gguf_ensure(cursor, bytes)) return 0;
    if (dst) memcpy(dst, cursor->buffer + cursor->at, bytes);
    cursor->at += bytes;
    return 1;
}

static int h3_gguf_read_u32(h3_gguf_cursor *cursor, uint32_t *result) {
    unsigned char bytes[4];
    if (!h3_gguf_read_bytes(cursor, bytes, sizeof(bytes))) return 0;
    *result = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
              ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return 1;
}

static int h3_gguf_read_u64(h3_gguf_cursor *cursor, uint64_t *result) {
    unsigned char bytes[8];
    if (!h3_gguf_read_bytes(cursor, bytes, sizeof(bytes))) return 0;
    uint64_t value = 0;
    for (unsigned index = 0; index < 8; index++)
        value |= (uint64_t)bytes[index] << (index * 8);
    *result = value;
    return 1;
}

/* Copies a GGUF string out of the buffer; returns a malloc'd NUL-terminated
   copy.  The length field is sanity-bound by the file size. */
static char *h3_gguf_read_string(h3_gguf_cursor *cursor) {
    uint64_t length;
    if (!h3_gguf_read_u64(cursor, &length)) return NULL;
    if (length > cursor->file_size) {
        h3_gguf_failf(cursor, "%s: string length %llu exceeds file size",
                      cursor->path, (unsigned long long)length);
        return NULL;
    }
    if (!h3_gguf_ensure(cursor, (size_t)length)) return NULL;
    char *value = malloc((size_t)length + 1);
    if (!value) {
        h3_gguf_fail(cursor, "out of memory reading GGUF string");
        return NULL;
    }
    memcpy(value, cursor->buffer + cursor->at, (size_t)length);
    value[length] = '\0';
    cursor->at += (size_t)length;
    return value;
}

static int h3_gguf_append_kv(h3_gguf_file *store, const h3_gguf_kv *entry) {
    if (store->kv_count == (size_t)store->n_kv) return 0; /* count overrun */
    h3_gguf_kv *grown = realloc(store->kvs,
                                (store->kv_count + 1) * sizeof(*grown));
    if (!grown) return -1;
    grown[store->kv_count++] = *entry;
    store->kvs = grown;
    return 1;
}

static int h3_gguf_parse_kvs(h3_gguf_cursor *cursor, h3_gguf_file *store) {
    for (uint64_t index = 0; index < store->n_kv; index++) {
        h3_gguf_kv entry;
        memset(&entry, 0, sizeof(entry));
        entry.key = h3_gguf_read_string(cursor);
        if (!entry.key) return 0;
        if (!entry.key[0]) {
            free(entry.key);
            return h3_gguf_failf(cursor, "%s: empty KV key at index %llu",
                                 cursor->path, (unsigned long long)index);
        }
        uint32_t value_type;
        if (!h3_gguf_read_u32(cursor, &value_type)) {
            free(entry.key);
            return 0;
        }
        entry.type = value_type;
        int ok;
        switch (value_type) {
            case H3_GGUF_KV_UINT8: {
                unsigned char raw;
                ok = h3_gguf_read_bytes(cursor, &raw, 1) ? (entry.value.unsigned8 = raw, 1) : 0;
                break;
            }
            case H3_GGUF_KV_INT8: {
                signed char raw;
                ok = h3_gguf_read_bytes(cursor, &raw, 1) ? (entry.value.signed8 = raw, 1) : 0;
                break;
            }
            case H3_GGUF_KV_UINT16: {
                unsigned char raw[2];
                ok = h3_gguf_read_bytes(cursor, raw, 2);
                if (ok) entry.value.unsigned16 =
                    (uint16_t)((uint16_t)raw[0] | ((uint16_t)raw[1] << 8));
                break;
            }
            case H3_GGUF_KV_INT16: {
                unsigned char raw[2];
                ok = h3_gguf_read_bytes(cursor, raw, 2);
                if (ok) entry.value.signed16 = (int16_t)((uint16_t)raw[0] | ((uint16_t)raw[1] << 8));
                break;
            }
            case H3_GGUF_KV_UINT32:
                ok = h3_gguf_read_u32(cursor, &entry.value.unsigned32);
                break;
            case H3_GGUF_KV_INT32: {
                uint32_t raw;
                ok = h3_gguf_read_u32(cursor, &raw);
                if (ok) entry.value.signed32 = (int32_t)raw;
                break;
            }
            case H3_GGUF_KV_FLOAT32: {
                uint32_t raw;
                ok = h3_gguf_read_u32(cursor, &raw);
                if (ok) memcpy(&entry.value.float32, &raw, sizeof(raw));
                break;
            }
            case H3_GGUF_KV_BOOL: {
                unsigned char raw;
                ok = h3_gguf_read_bytes(cursor, &raw, 1);
                if (ok && raw > 1)
                    ok = h3_gguf_failf(cursor, "%s: KV %s: GGUF bool value %u not 0/1",
                                       cursor->path, entry.key, raw);
                if (ok) entry.value.boolean = raw;
                break;
            }
            case H3_GGUF_KV_STRING:
                entry.value.string = h3_gguf_read_string(cursor);
                ok = entry.value.string ? 1 : 0;
                break;
            case H3_GGUF_KV_UINT64:
                ok = h3_gguf_read_u64(cursor, &entry.value.unsigned64);
                break;
            case H3_GGUF_KV_INT64: {
                uint64_t raw;
                ok = h3_gguf_read_u64(cursor, &raw);
                if (ok) entry.value.signed64 = (int64_t)raw;
                break;
            }
            case H3_GGUF_KV_FLOAT64: {
                uint64_t raw;
                ok = h3_gguf_read_u64(cursor, &raw);
                if (ok) memcpy(&entry.value.float64, &raw, sizeof(raw));
                break;
            }
            case H3_GGUF_KV_ARRAY: {
                uint32_t element_type;
                uint64_t count;
                ok = h3_gguf_read_u32(cursor, &element_type);
                if (ok && element_type == H3_GGUF_KV_ARRAY)
                    ok = h3_gguf_failf(cursor, "%s: KV %s: nested GGUF arrays unsupported",
                                       cursor->path, entry.key);
                if (ok && element_type > H3_GGUF_KV_FLOAT64)
                    ok = h3_gguf_failf(cursor, "%s: KV %s: array element type %u unknown",
                                       cursor->path, entry.key, element_type);
                if (ok) ok = h3_gguf_read_u64(cursor, &count);
                if (!ok) break;
                entry.array_type = element_type;
                entry.array_count = count;
                if (element_type == H3_GGUF_KV_STRING) {
                    for (uint64_t element = 0; element < count; element++) {
                        char *unused = h3_gguf_read_string(cursor);
                        if (!unused) { ok = 0; break; }
                        free(unused);
                    }
                } else {
                    uint64_t total;
                    if (count > UINT64_MAX / h3_gguf_scalar_bytes[element_type])
                        ok = h3_gguf_failf(cursor, "%s: KV %s: array byte count overflow",
                                           cursor->path, entry.key);
                    else {
                        total = count * h3_gguf_scalar_bytes[element_type];
                        ok = h3_gguf_read_bytes(cursor, NULL, (size_t)total);
                    }
                }
                break;
            }
            default:
                ok = h3_gguf_failf(cursor, "%s: KV %s: GGUF value type %u unknown",
                                   cursor->path, entry.key, value_type);
                break;
        }
        if (!ok) {
            free(entry.key);
            free(entry.value.string);
            return 0;
        }
        int appended = h3_gguf_append_kv(store, &entry);
        if (appended != 1) {
            free(entry.key);
            free(entry.value.string);
            if (appended == -1) h3_gguf_fail(cursor, "out of memory indexing GGUF KV pairs");
            else h3_gguf_fail(cursor, "GGUF KV count overrun");
            return 0;
        }
    }
    return 1;
}

static int h3_gguf_parse_tensors(h3_gguf_cursor *cursor, h3_gguf_file *store) {
    size_t capacity = 0;
    for (uint64_t index = 0; index < store->n_tensors; index++) {
        h3_gguf_tensor tensor;
        memset(&tensor, 0, sizeof(tensor));
        tensor.name = h3_gguf_read_string(cursor);
        if (!tensor.name) return 0;
        if (!tensor.name[0]) {
            free(tensor.name);
            return h3_gguf_failf(cursor, "%s: empty tensor name at index %llu",
                                 cursor->path, (unsigned long long)index);
        }
        uint32_t ndim, ggml_type;
        if (!h3_gguf_read_u32(cursor, &ndim)) { free(tensor.name); return 0; }
        if (ndim > H3_GGUF_MAX_DIMS) {
            int reported = h3_gguf_failf(cursor, "tensor %s: rank %u exceeds supported %d",
                                         tensor.name, ndim, H3_GGUF_MAX_DIMS);
            free(tensor.name);
            return reported;
        }
        for (uint32_t dimension = 0; dimension < ndim; dimension++) {
            if (!h3_gguf_read_u64(cursor, &tensor.dims[dimension])) {
                free(tensor.name);
                return 0;
            }
        }
        tensor.ndim = (int)ndim;
        if (!h3_gguf_read_u32(cursor, &ggml_type)) { free(tensor.name); return 0; }
        const h3_ggml_traits_entry *traits = h3_ggml_traits_for(ggml_type);
        if (!traits) {
            int reported = h3_gguf_failf(cursor,
                                         "tensor %s: ggml type %u is unknown (sd.cpp fork enum "
                                         "supports 0..%u with retired values excluded)",
                                         tensor.name, ggml_type, H3_GGUF_MAX_GGML_TYPE);
            free(tensor.name);
            return reported;
        }
        tensor.ggml_type = ggml_type;
        tensor.type_name = traits->name;
        if (!h3_gguf_read_u64(cursor, &tensor.offset)) { free(tensor.name); return 0; }

        uint64_t elements = 1;
        for (int dimension = 0; dimension < tensor.ndim; dimension++) {
            uint64_t dim = tensor.dims[dimension];
            if (dim && elements > UINT64_MAX / dim) {
                int reported = h3_gguf_failf(cursor, "tensor %s: element count overflow",
                                             tensor.name);
                free(tensor.name);
                return reported;
            }
            elements *= dim;
        }
        tensor.elements = elements;
        if (elements % traits->blck_size != 0) {
            int reported = h3_gguf_failf(cursor,
                                         "tensor %s: %llu elements not divisible by %u-element %s blocks",
                                         tensor.name, (unsigned long long)elements,
                                         traits->blck_size, traits->name);
            free(tensor.name);
            return reported;
        }
        uint64_t blocks = elements / traits->blck_size;
        if (blocks > UINT64_MAX / traits->type_size) {
            int reported = h3_gguf_failf(cursor, "tensor %s: byte count overflow", tensor.name);
            free(tensor.name);
            return reported;
        }
        tensor.nbytes = blocks * traits->type_size;

        if (store->tensor_count == capacity) {
            size_t next = capacity ? capacity * 2 : 64;
            h3_gguf_tensor *grown = realloc(store->tensors, next * sizeof(*grown));
            if (!grown) {
                free(tensor.name);
                h3_gguf_fail(cursor, "out of memory indexing GGUF tensors");
                return 0;
            }
            store->tensors = grown;
            capacity = next;
        }
        store->tensors[store->tensor_count++] = tensor;
    }
    return 1;
}

int h3_gguf_open(const char *path, h3_gguf_file *store,
                 char *error, size_t error_size) {
    if (!path || !store) return 0;
    memset(store, 0, sizeof(*store));
    if (error && error_size) error[0] = '\0';

    int descriptor = open(path, O_RDONLY);
    if (descriptor < 0) {
        if (error && error_size)
            snprintf(error, error_size, "%s: %s", path, strerror(errno));
        return 0;
    }
    struct stat status;
    if (fstat(descriptor, &status) != 0 || status.st_size < 12) {
        if (error && error_size)
            snprintf(error, error_size, "%s: invalid file", path);
        close(descriptor);
        return 0;
    }

    h3_gguf_cursor cursor;
    memset(&cursor, 0, sizeof(cursor));
    cursor.path = path;
    cursor.descriptor = descriptor;
    cursor.file_size = (uint64_t)status.st_size;
    cursor.error = error;
    cursor.error_size = error_size;
    cursor.capacity = 65536;
    if (cursor.capacity > cursor.file_size)
        cursor.capacity = (size_t)cursor.file_size;
    cursor.buffer = malloc(cursor.capacity);
    store->path = strdup(path);
    if (!cursor.buffer || !store->path) {
        if (error && error_size) snprintf(error, error_size, "out of memory");
        free(cursor.buffer);
        h3_gguf_close(store);
        close(descriptor);
        return 0;
    }

    unsigned char magic[4];
    int ok = h3_gguf_read_bytes(&cursor, magic, 4);
    if (ok && memcmp(magic, "GGUF", 4) != 0)
        ok = h3_gguf_failf(&cursor, "%s: not a GGUF file (magic %c%c%c%c)",
                           path, magic[0], magic[1], magic[2], magic[3]);
    uint32_t version = 0;
    if (ok) ok = h3_gguf_read_u32(&cursor, &version);
    if (ok && (version < 1 || version > 3))
        ok = h3_gguf_failf(&cursor, "%s: unsupported GGUF version %u", path, version);
    if (!ok) goto fail;
    store->version = version;
    store->file_size = cursor.file_size;

    /* GGUF v1 declares counts as uint32; v2/v3 as uint64. */
    uint64_t n_tensors = 0, n_kv = 0;
    if (version == 1) {
        uint32_t wide_tensors = 0, wide_kv = 0;
        ok = h3_gguf_read_u32(&cursor, &wide_tensors) && h3_gguf_read_u32(&cursor, &wide_kv);
        n_tensors = wide_tensors;
        n_kv = wide_kv;
    } else {
        ok = h3_gguf_read_u64(&cursor, &n_tensors) && h3_gguf_read_u64(&cursor, &n_kv);
    }
    if (!ok) goto fail;
    /* Plausibility bounds: each tensor info is >= 29 bytes on disk, each KV
       pair >= 16; guard against absurd counts before walking the table. */
    if (n_tensors > cursor.file_size / 24 || n_kv > cursor.file_size / 16) {
        h3_gguf_failf(&cursor, "%s: implausible header counts (n_tensors=%llu, n_kv=%llu)",
                      path, (unsigned long long)n_tensors, (unsigned long long)n_kv);
        goto fail;
    }
    store->n_tensors = n_tensors;
    store->n_kv = n_kv;

    if (!h3_gguf_parse_kvs(&cursor, store)) goto fail;
    if (!h3_gguf_parse_tensors(&cursor, store)) goto fail;

    store->metadata_end = cursor.at;
    store->alignment = H3_GGUF_DEFAULT_ALIGNMENT;
    for (size_t index = 0; index < store->kv_count; index++) {
        const h3_gguf_kv *kv = &store->kvs[index];
        if (strcmp(kv->key, "general.alignment") != 0) continue;
        if (kv->type != H3_GGUF_KV_UINT32) {
            h3_gguf_failf(&cursor, "general.alignment: expected uint32, got GGUF type %u",
                          kv->type);
            goto fail;
        }
        uint32_t alignment = kv->value.unsigned32;
        if (alignment == 0 || (alignment & (alignment - 1)) != 0 || alignment > (1u << 24)) {
            h3_gguf_failf(&cursor, "general.alignment %u is not a power of two in 1..2^24",
                          alignment);
            goto fail;
        }
        store->alignment = alignment;
    }
    uint64_t mask = (uint64_t)store->alignment - 1;
    if (store->metadata_end > UINT64_MAX - mask) {
        h3_gguf_fail(&cursor, "data region alignment overflow");
        goto fail;
    }
    store->data_start = (store->metadata_end + mask) & ~mask;
    if (store->data_start > cursor.file_size) {
        h3_gguf_failf(&cursor, "%s: data region starts at %llu past file end %llu",
                      path, (unsigned long long)store->data_start,
                      (unsigned long long)cursor.file_size);
        goto fail;
    }
    uint64_t data_available = cursor.file_size - store->data_start;
    for (size_t index = 0; index < store->tensor_count; index++) {
        const h3_gguf_tensor *tensor = &store->tensors[index];
        if (tensor->offset > data_available ||
            tensor->nbytes > data_available - tensor->offset) {
            h3_gguf_failf(&cursor,
                          "tensor %s: data range %llu+%llu exceeds the %llu-byte data region",
                          tensor->name, (unsigned long long)tensor->offset,
                          (unsigned long long)tensor->nbytes,
                          (unsigned long long)data_available);
            goto fail;
        }
    }

    free(cursor.buffer);
    close(descriptor);
    return 1;

fail:
    free(cursor.buffer);
    close(descriptor);
    h3_gguf_close(store);
    return 0;
}

void h3_gguf_close(h3_gguf_file *store) {
    if (!store) return;
    for (size_t index = 0; index < store->kv_count; index++) {
        if (store->kvs[index].type == H3_GGUF_KV_STRING)
            free(store->kvs[index].value.string);
        free(store->kvs[index].key);
    }
    free(store->kvs);
    for (size_t index = 0; index < store->tensor_count; index++)
        free(store->tensors[index].name);
    free(store->tensors);
    free(store->path);
    memset(store, 0, sizeof(*store));
}

const h3_gguf_tensor *h3_gguf_tensor_at(const h3_gguf_file *store, size_t index) {
    if (!store || index >= store->tensor_count) return NULL;
    return &store->tensors[index];
}

const h3_gguf_tensor *h3_gguf_find(const h3_gguf_file *store, const char *name) {
    if (!store || !name) return NULL;
    for (size_t index = 0; index < store->tensor_count; index++)
        if (!strcmp(store->tensors[index].name, name)) return &store->tensors[index];
    return NULL;
}

const h3_gguf_kv *h3_gguf_kv_find(const h3_gguf_file *store, const char *key) {
    if (!store || !key) return NULL;
    for (size_t index = 0; index < store->kv_count; index++)
        if (!strcmp(store->kvs[index].key, key)) return &store->kvs[index];
    return NULL;
}

uint64_t h3_gguf_tensor_elements(const h3_gguf_tensor *tensor) {
    return tensor ? tensor->elements : 0;
}

uint64_t h3_gguf_tensor_bytes(const h3_gguf_tensor *tensor) {
    return tensor ? tensor->nbytes : 0;
}

int h3_gguf_read_tensor_raw(const h3_gguf_file *store, const char *name,
                            void *dst, size_t capacity,
                            char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!store || !store->path || !name || (!dst && capacity != 0)) {
        if (error && error_size)
            snprintf(error, error_size, "invalid GGUF tensor read request");
        return 0;
    }
    const h3_gguf_tensor *tensor = h3_gguf_find(store, name);
    if (!tensor) {
        if (error && error_size)
            snprintf(error, error_size, "%s: tensor not found in GGUF index", name);
        return 0;
    }
    if (capacity < (size_t)tensor->nbytes) {
        if (error && error_size) {
            snprintf(error, error_size, "%s: expected %llu bytes, buffer capacity %zu",
                     name, (unsigned long long)tensor->nbytes, capacity);
        }
        return 0;
    }
    if (tensor->nbytes == 0) return 1;
    int descriptor = open(store->path, O_RDONLY);
    if (descriptor < 0) {
        if (error && error_size)
            snprintf(error, error_size, "%s: %s", store->path, strerror(errno));
        return 0;
    }
    size_t done = 0;
    while (done < (size_t)tensor->nbytes) {
        size_t remaining = (size_t)tensor->nbytes - done;
        size_t chunk = remaining < (size_t)(1u << 30) ? remaining : (size_t)(1u << 30);
        ssize_t count = pread(descriptor, (unsigned char *)dst + done, chunk,
                              (off_t)(store->data_start + tensor->offset + done));
        if (count <= 0) {
            if (error && error_size) {
                snprintf(error, error_size, "%s: cannot read %s: %s", store->path, name,
                         count < 0 ? strerror(errno) : "unexpected end of file");
            }
            close(descriptor);
            return 0;
        }
        done += (size_t)count;
    }
    close(descriptor);
    return 1;
}

int h3_ggml_row_bytes(uint32_t ggml_type, uint64_t elements,
                      uint64_t *bytes, char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!bytes) return 0;
    const h3_ggml_traits_entry *traits = h3_ggml_traits_for(ggml_type);
    if (!traits) {
        if (error && error_size) {
            snprintf(error, error_size,
                     "ggml type %u is unknown (sd.cpp fork enum supports 0..%u "
                     "with retired values excluded)",
                     ggml_type, H3_GGUF_MAX_GGML_TYPE);
        }
        return 0;
    }
    if (elements % traits->blck_size != 0) {
        if (error && error_size) {
            snprintf(error, error_size,
                     "%llu elements not divisible by %u-element %s blocks",
                     (unsigned long long)elements, traits->blck_size, traits->name);
        }
        return 0;
    }
    uint64_t blocks = elements / traits->blck_size;
    if (blocks > UINT64_MAX / traits->type_size) {
        if (error && error_size) snprintf(error, error_size, "byte count overflow");
        return 0;
    }
    *bytes = blocks * traits->type_size;
    return 1;
}

int h3_ggml_type_info(uint32_t ggml_type, uint32_t *block_elements,
                      uint32_t *block_bytes) {
    const h3_ggml_traits_entry *traits = h3_ggml_traits_for(ggml_type);
    if (!traits) return 0;
    if (block_elements) *block_elements = traits->blck_size;
    if (block_bytes) *block_bytes = traits->type_size;
    return 1;
}
