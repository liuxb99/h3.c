#!/usr/bin/env python3
"""折疊 Pika H3 Flex LoRA 到 MiniMax-H3 Ref2VA 基幹（h3.c 佈局）。

語義：W_target += (alpha / rank) * B @ A，逐鍵一一對應
（LoRA 鍵 = 'diffusion_model.' + 基幹鍵，實測 416 tensors 全匹配）。
strength 預設 1.0（alpha=rank=128/384 → scale=1）。
用法：fold_flex_lora.py --base <shards_dir> --lora <lora.safetensors> --out <out_dir>
流式處理：逐 shard 載入→疊加 touched 鍵→寫出，峰值記憶體 ≈ 單 shard 大小。
"""
import argparse, json, os, re, struct, sys
from collections import defaultdict

import torch


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def write_shard(path, tensors):
    header, offset = {}, 0
    rows = []
    # 穩定順序：原鍵序（dict 插入序）
    for key, t in tensors.items():
        tb = t.contiguous().view(torch.uint8).numpy().tobytes()
        shape = list(t.shape)
        dtype = {torch.bfloat16: "BF16", torch.float32: "F32", torch.float16: "F16"}[t.dtype]
        rows.append([key, shape, dtype, offset, offset + len(tb)])
        offset += len(tb)
    for key, shape, dtype, a, b in rows:
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [a, b]}
    hb = json.dumps(header, separators=(",", ":"), sort_keys=False).encode()
    pad = (-((8 + len(hb)))) % 8
    hb += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for key in tensors:
            f.write(tensors[key].contiguous().view(torch.uint8).numpy().tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Ref2VA transformer/ 目錄（13 shards）")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    hdr, hdr_off = read_header(args.lora)
    lora_files = defaultdict(list)  # per-base-shard list of (lora_key_pair, target)
    index_path = os.path.join(args.base, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # 收集 LoRA：target -> (A, B, alpha_scale)
    deltas = {}
    pat = re.compile(r"^(.*)\.lora_([AB])\.weight$")
    meta = hdr.pop("__metadata__", None)
    pending = {}
    for key, info in hdr.items():
        m = pat.match(key[len("diffusion_model."):])
        if not m:
            continue
        base_key, ab = m.group(1) + ".weight", m.group(2)
        pending.setdefault(base_key, {})[ab] = key
    for base_key, pair in pending.items():
        if "A" not in pair or "B" not in pair:
            print(f"WARN 不成對: {base_key}", file=sys.stderr)
    # torch.load 對 safetensors：用 safetensors 直讀
    from safetensors import safe_open
    with safe_open(args.lora, framework="pt") as f:
        tensors = {}
        for base_key, pair in pending.items():
            if base_key not in weight_map:
                print(f"WARN 基幹無此鍵: {base_key}", file=sys.stderr)
                continue
            A = f.get_tensor(pair["A"]).to(torch.float32)
            B = f.get_tensor(pair["B"]).to(torch.float32)
            rank = A.shape[0]
            # manifest rank==alpha（128／fusion QKV 384）→ Comfy 無 alpha 鍵時
            # 慣例 alpha=rank ⇒ scale=1.0*strength
            scale = args.strength
            tensors[base_key] = (B @ A) * scale
    per_shard = defaultdict(list)
    for base_key in tensors:
        per_shard[weight_map[base_key]].append(base_key)
    print(f"LoRA deltas: {len(tensors)}；分布 {len(per_shard)} shards", file=sys.stderr)

    import shutil
    # 複製非 shard 檔案（config 等）
    for fn in os.listdir(args.base):
        if fn.startswith("._") or fn.startswith("."):
            continue  # AppleDouble 垃圾（exFAT 副本）
        if fn.endswith(".safetensors") or fn.startswith("model.safetensors.index"):
            continue
        shutil.copy2(os.path.join(args.base, fn), os.path.join(args.out, fn))

    for fn in sorted(os.listdir(args.base)):
        if not fn.endswith(".safetensors") or fn.startswith("._"):
            continue
        src = os.path.join(args.base, fn)
        touched = [k for k in per_shard.get(fn, [])]
        with safe_open(src, framework="pt") as f:
            keys = list(f.keys())
            if not touched:
                shutil.copyfile(src, os.path.join(args.out, fn))
                print(f"copy {fn}", file=sys.stderr)
                continue
            out_tensors = {}
            for k in keys:
                t = f.get_tensor(k)
                if k in tensors:
                    if t.dtype == torch.bfloat16:
                        new = (t.to(torch.float32) + tensors[k]).to(torch.bfloat16)
                    else:
                        new = t.to(torch.float32) + tensors[k].to(t.dtype)
                    out_tensors[k] = new
                else:
                    out_tensors[k] = t
        write_shard(os.path.join(args.out, fn), out_tensors)
        print(f"fold {fn}: {len(touched)} deltas", file=sys.stderr)
    print("DONE", file=sys.stderr)


if __name__ == "__main__":
    main()
