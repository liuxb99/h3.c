
import sys, os, hashlib
sys.path.insert(0, "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scripts")
sys.path.insert(0, "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol")
import st_io as s
import convrot_quant as cq
from safetensors import safe_open

OFF = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
BU = "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol/engine_backup_prepatch"
NAMES = ["adaln_single.emb.timestep_embedder.linear_1.weight",
         "blocks.10.mlp.fc2.weight",
         "blocks.0.attn.qkv_proj.weight"]
manifest = {}
for n in NAMES:
    p = f"{BU}/{n}.golden.bin"; o = f"{BU}/{n}.old.bin"
    if not os.path.exists(p):
        print(n, "no golden (not patched in engine)")
        continue
    manifest[n] = {"golden_path": p,
                   "golden_sha256": hashlib.sha256(open(p,"rb").read()).hexdigest()}
with open("/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol/golden_manifest.json", "w") as f:
    json.dump(manifest, f, indent=1)
with s.st_open(OFF) as (h, mm):
    def raw(name):
        so, eo = h[name]; return mm[so:eo]
    for n in NAMES:
        if n not in h:
            print(n, "MISSING in official"); continue
        st = json.loads(s.parse_json_official_comfy_quant(raw(n), n))
        if "convrot" not in st:
            print(n, "not convrot quant subject; dtype", h[n][0]["dtype"]); continue
        N, K = h[n][0]["shape"]
        G = int(st["convrot"]["groupsize"]); gs = G if G > 0 else 128
        sc = s.parse_f32le(raw(n + ".weight_scale"), (N, 1))
        if n in manifest:
            sc = sc[manifest[n].get("scale_perm", slice(None))]
        codes = s.parse_int8(raw(n), (N, K))
        if n in manifest and "row_perm" in manifest[n]:
            codes = codes[manifest[n]["row_perm"]]
        w = cq.dequant_f32(codes, sc, gs)
        if n == "blocks.10.mlp.fc2.weight":
            w = w.reshape(3072, 3, 4096).transpose(0, 2, 1).reshape(3072, 12288)
        if n == "adaln_single.emb.timestep_embedder.linear_1.weight":
            w = w.reshape(1, 6144, 2).transpose(0, 2, 1).reshape(2, 6144)
        if n == "blocks.0.attn.qkv_proj.weight" and "row_perm" not in manifest[n]:
            w = w.reshape(3 * 128, 128, 4096)
            w = w.reshape(3, 128, 128, 4096).transpose(1, 0, 2, 3).reshape(384, 128, 4096)
            w = w.reshape(384 * 128, 4096)
        if n in manifest:
            w = w[manifest[n].get("out_perm", slice(None))]
        wq = cq.quantize_f32(w, gs)
        if n in manifest:
            wq = wq[manifest[n].get("file_row_perm", slice(None))]
        match = (wq == open(manifest[n]["golden_path"], "rb").read()).mean()
        print(f"{n}: golden-vs-dequant->requant match {match:.6f}")
