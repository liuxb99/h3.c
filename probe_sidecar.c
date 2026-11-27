/* Probe: does h3_weight_quant_info see comfy_quant sidecars in the flat dir? */
#include "h3_weights.h"
#include <stdio.h>
#include <string.h>

int main(void) {
    char err[1024] = {0};
    const char *dirs[] = {
        "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8",
        "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/FL2VA/transformer",
    };
    const char *names[] = {
        "blocks.0.attn.out_proj.weight",
        "blocks.0.mlp.fc2.weight",
        "token_refiner.blocks.0.mlp.fc1.weight",
    };
    for (size_t d = 0; d < 2; ++d) {
        h3_weight_store *store = h3_weight_store_open(dirs[d], err, sizeof err);
        printf("DIR %s -> %s shards=%zu\n", dirs[d], store ? "OK" : err,
               store ? h3_weight_store_shards(store) : 0);
        if (!store) continue;
        for (size_t i = 0; i < 3; ++i) {
            const h3_st_header *h = NULL;
            const h3_st_tensor *t = h3_weight_find(store, names[i], &h);
            printf("  %s payload=%s\n", names[i],
                   t ? h3_dtype_name(t->dtype) : "MISSING");
            uint32_t group = 0;
            int convrot = 0;
            err[0] = 0;
            int ok = h3_weight_quant_info(store, names[i], &group, &convrot,
                                          err, sizeof err);
            printf("    quant_info ok=%d group=%u convrot=%d err=%s\n", ok,
                   group, convrot, ok ? "" : err);
        }
        h3_weight_store_free(store);
    }
    return 0;
}
