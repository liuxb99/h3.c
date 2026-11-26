#ifndef H3_INTERNAL_H
#define H3_INTERNAL_H

#include "h3.h"
#include "h3_host.h"

#include <stdarg.h>

struct h3_dit;
struct h3_video_vae_decoder;

struct h3_ctx {
    char *model_dir;
    /* Directory (relative to model_dir) that actually holds the FL2VA DiT
     * weight shards for the detected layout: "FL2VA/transformer" for the
     * original split layout, "." for the Comfy single-file layout where the
     * quantized DiT lives at the directory root.  Generation must resolve
     * the DiT store through this field — a hardcoded "FL2VA/transformer"
     * silently resolves symlinks to the *original* BF16 checkpoint and
     * stranded the native convrot path with "lacks sidecar" failures
     * (2026-11-26 root cause). */
    const char *dit_weights_relative;
    int single_file_layout;
    char error[512];
    h3_device_info device;
    h3_model_info model;
    int cache_enabled;
    char *conditioning_key;
    size_t conditioning_tokens;
    size_t conditioning_width;
    uint16_t *conditioning_values;
    uint8_t *conditioning_tags;
    float *conditioning_video_rows;
    size_t conditioning_video_elements;
    float *conditioning_audio_rows;
    size_t conditioning_audio_elements;
    h3_layout_ref *conditioning_references;
    size_t conditioning_reference_count;
    int conditioning_present;
    char *dit_key;
    struct h3_dit *dit;
    char *video_decoder_key;
    struct h3_video_vae_decoder *video_decoder;
};

void h3_set_error(h3_ctx *ctx, const char *format, ...)
    __attribute__((format(printf, 2, 3)));

#endif
