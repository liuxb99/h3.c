/* Loader-only regression: engine's own weight store against real model dirs.
   Runs in seconds — no TE load, no video, no GPU.
   Semantic ground truth is the checkpoint's own header declarations:
   every convrot tensor must resolve via h3_weight_quant_info to the group
   size the header sidecar declares (families: 256; adaln_proj: 64).
   The comfy conversion dropped the three adaln_single tensors (they exist
   only in the original checkpoint) — assert they are absent, so a future
   requant run notices when they come back quantized.
   Prints shard paths and every mismatch; exits 1 on any deviation. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "../h3_weights.h"

static const struct { const char *family; int group; } families[] = {
    { "adaln_proj.linear", 64 },
    { "attn.out_proj", 256 },
    { "attn.qkv_proj", 256 },
    { "mlp.fc1", 256 },
    { "mlp.fc2", 256 },
};

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1]
        : "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8";
    char error[512];
    h3_weight_store *store = h3_weight_store_open(dir, error, sizeof error);
    if (!store) {
        fprintf(stderr, "store_open FAILED: %s\n", error);
        return 1;
    }
    printf("dir=%s shards=%zu\n", dir, h3_weight_store_shards(store));
    int fails = 0, checks = 0;
    const struct { int lo, hi; } ranges[] = { {0, 49}, {0, 2} };
    const char *first_paths[3] = { NULL, NULL, NULL };
    for (size_t r = 0; r < 2; r++) {
        for (int b = ranges[r].lo; b <= ranges[r].hi; b++) {
            for (size_t f = 0; f < sizeof(families)/sizeof(families[0]); f++) {
                char name[256], side[300];
                snprintf(name, sizeof name, "blocks.%d.%s.weight", b,
                         families[f].family);
                snprintf(side, sizeof side, "blocks.%d.%s.comfy_quant", b,
                         families[f].family);
                const h3_st_header *hd = NULL;
                const h3_st_tensor *pt = h3_weight_find(store, name, &hd);
                const h3_st_tensor *sc = h3_weight_find(store, side, &hd);
                if (b == 0 && f < 3 && hd && !first_paths[f])
                    first_paths[f] = hd->path;
                uint32_t group = 0; int convrot = 0;
                int ok = h3_weight_quant_info(store, name, &group, &convrot,
                                              error, sizeof error);
                checks++;
                if (!pt) {
                    if (ok) {
                        fails++;
                        printf("FAIL %s: quant_info ok but payload MISSING\n",
                               name);
                    }
                    continue; /* original bf16 checkpoints: nothing to gate */
                }
                if (pt->dtype != 3 && pt->dtype != 2) {
                    fails++;
                    printf("FAIL %s: payload dtype %d\n", name, pt->dtype);
                    continue;
                }
                if (pt->dtype == 2) /* plain bf16: no convrot expected */
                    continue;
                if (!ok) {
                    fails++;
                    printf("FAIL %s: payload=%s sidecar=%s err=%s\n", name,
                           pt ? (pt->dtype == 3 ? "I8" : "other") : "MISSING",
                           sc ? "present" : "MISSING", error);
                } else if (!convrot || group != (uint32_t)families[f].group) {
                    fails++;
                    printf("FAIL %s: convrot=%d group=%u expected=%d\n", name,
                           convrot, group, families[f].group);
                }
            }
        }
        if (ranges[r].hi == 49) break; /* only blocks 0-49 for DiT */
    }
    for (size_t f = 0; f < 3; f++)
        if (first_paths[f]) printf("shard via %s: %s\n",
            families[f].family, first_paths[f]);
    /* The two non-"blocks." convrot tensors the native path needs, plus the
       three the comfy conversion left out of the int8 file entirely. */
    static const char *present[] = {
        "adaln_single.emb.timestep_embedder.linear_1.weight",
        "adaln_single.emb.timestep_embedder.linear_2.weight",
        "adaln_single.linear.weight",
    };
    for (size_t e = 0; e < 3; e++) {
        const h3_st_header *hd = NULL;
        const h3_st_tensor *pt = h3_weight_find(store, present[e], &hd);
        int quantized = pt && (pt->dtype == 3 || pt->dtype == 1);
        uint32_t group = 0; int convrot = 0;
        int ok = h3_weight_quant_info(store, present[e], &group, &convrot,
                                      error, sizeof error);
        checks++;
        if (quantized && !ok) {
            fails++;
            printf("FAIL(absent-check) %s: payload I8 but %s\n",
                   present[e], error);
        }
        printf("adaln_single %s: payload=%s quant_info=%s\n", present[e],
               pt ? "present" : "absent", ok ? "ok" : "no");
    }
    printf("RESULT checks=%d fails=%d\n", checks, fails);
    h3_weight_store_free(store);
    return fails ? 1 : 0;
}
