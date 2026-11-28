# Battery: recover & pin rotate_stages_f32 / qkv_grouped_perm semantics for
# scripts/qkv_layout_gate.py (mid-refactor casualties). Ground truth = the
# frozen official contract file itself: dequant(official codes, official
# scale) must reproduce HF bf16 golden rows up to int8 noise.
import importlib.machinery, importlib.util, json, math, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPTS = os.path.join(REPO, "scripts")
PYC = os.path.join(SCRIPTS, "__pycache__", "convrot_quant.cpython-313.pyc")
OFFICIAL = ("/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/"
            "minimax_h3_ref2va_int8_convrot.safetensors")
HF_DIR = "/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer"

sys.path.insert(0, SCRIPTS)
import st_io  # noqa: E402
import convrot_quant as cq_new  # noqa: E402  (new 02:53 matrix module)

# ---- load old 00:41 module from pyc (sourceless) ----
loader = importlib.machinery.SourcelessFileLoader("cq_old", PYC)
spec = importlib.util.spec_from_loader("cq_old", loader)
cq_old = importlib.util.module_from_spec(spec)
loader.exec_module(cq_old)
names = sorted(n for n in dir(cq_old) if not n.startswith("__"))
print("OLD module exports:", names)

out = {"old_exports": names, "checks": {}}

# ---- B1: old vs new quant line ----
def cmp_quant(N, K, gs, seed, tag):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(N, K, dtype=torch.float32, generator=g).to(torch.bfloat16)
    c_new, s_new = cq_new.quant_tensor_f32line(w, gs, mode="int8")
    fn_old = cq_old.quant_tensor_f32line
    try:
        c_old, s_old = fn_old(w, gs, mode="int8")
    except TypeError:
        c_old, s_old = fn_old(w, gs)
    code_eq = bool(torch.equal(c_new, c_old))
    scale_eq = bool(torch.equal(s_new.view(-1), s_old.view(-1)))
    frac = float((c_new.numpy() != c_old.numpy()).mean())
    return {"tag": tag, "shape": [N, K, gs], "codes_identical": code_eq,
            "scale_identical": scale_eq, "code_mismatch_frac": frac}

b1 = [cmp_quant(512, 1024, 64, 1, "rand 512x1024 gs64"),
      cmp_quant(512, 1024, 256, 2, "rand 512x1024 gs256"),
      cmp_quant(3840, 3072, 256, 3, "rand 3840x3072 gs256"),
      cmp_quant(2048, 2048, 256, 4, "rand 2048x2048 gs256")]
out["checks"]["B1_old_vs_new_quant_line"] = b1

# ---- load official + HF golden ----
def load_source(path):
    """simplish WeightSource: dict name -> (file, hdr_entry), files opened."""
    files = {}
    if os.path.isdir(path):
        d = path
        shards = [os.path.join(d, f) for f in sorted(os.listdir(d))
                  if f.endswith(".safetensors")]
    else:
        shards = [path]
    key_file = {}
    for s in shards:
        h, f, mm, ds = st_io.st_open(s)
        files[s] = (h, mm, ds)
        for k in h:
            if k != "__metadata__":
                key_file[k] = s
    return files, key_file

off_files, off_keys = load_source(OFFICIAL)
hf_files, hf_keys = load_source(HF_DIR)

def read_f32(files, key_file, name, dtype_hint=None):
    s = key_file[name]
    h, mm, ds = files[s]
    e = h[name]
    a, b = e["data_offsets"]
    buf = mm[ds + a: ds + b]
    if e["dtype"] == "BF16":
        u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32)
        return (u << 16).view(np.float32)
    if e["dtype"] == "F32":
        return np.frombuffer(buf, dtype=np.float32)
    if e["dtype"] == "I8":
        return np.frombuffer(buf, dtype=np.int8)
    raise SystemExit(f"unsupported {e['dtype']} {name}")

def scale_of(files, key_file, stem):
    s = key_file[stem + ".weight_scale"]
    h, mm, ds = files[s]
    e = h[s] if s in h else None
    ent = h[stem + ".weight_scale"]
    a, b = ent["data_offsets"]
    return np.frombuffer(mm[ds + a: ds + b], dtype=np.float32).reshape(-1, 1)

def gs_of(files, key_file, stem):
    s = key_file[stem + ".comfy_quant"]
    h, mm, ds = files[s]
    ent = h[stem + ".comfy_quant"]
    a, b = ent["data_offsets"]
    cj = json.loads(bytes(mm[ds + a: ds + b]))
    return int(cj["convrot_groupsize"])

def builder_grouped_map(N):
    assert N % 384 == 0
    hpc = N // 384
    j = np.arange(N)
    kind, unit, sub = j // (hpc * 128), (j // 128) % hpc, j % 128
    return (unit * 3 + kind) * 128 + sub

def cos_rel(a, b):
    a = a.reshape(-1); b = b.reshape(-1)
    dot = float((a * b).sum()); aa = float((a * a).sum()); bb = float((b * b).sum())
    c = dot / max(1e-30, math.sqrt(aa * bb))
    rr = math.sqrt(max(0.0, aa + bb - 2 * dot)) / max(1e-30, math.sqrt(bb))
    return c, rr

_all_comfy = sorted(k[: -len(".comfy_quant")] for k in off_keys
                    if k.endswith(".comfy_quant"))
def _pick(pred, default=None):
    hits = [s for s in _all_comfy if pred(s)]
    return hits[0] if hits else default
SUBJECTS = [_pick(lambda s: "adaln" in s),
            _pick(lambda s: s.endswith("mlp.fc2")),
            _pick(lambda s: s.endswith("mlp.fc1")),
            _pick(lambda s: "attn.to_out" in s or "attn.out_proj" in s),
            _pick(lambda s: "qkv" in s)]
SUBJECTS = [s for s in SUBJECTS if s]
print("subjects:", SUBJECTS)

# rotation variants (take (n,g,gs) float32 3D, return same shape)
_h_cache = {}
def _h(gs):
    if gs not in _h_cache:
        _h_cache[gs] = cq_new.build_hadamard(gs, torch.float32)
    return _h_cache[gs]
def v_matrix_norm(x):
    n, g, gs = x.shape
    return x @ _h(gs).T                       # = dequant, NORMALIZED
def v_matrix_raw(x):
    n, g, gs = x.shape
    return x @ (_h(gs) * math.sqrt(gs)).T    # unnormalized stage sum
def v_old(x):
    try:
        return cq_old.rotate_stages_f32(x)
    except TypeError as t:
        return {"TypeError": str(t)[:120]}

# ---- B2: dequant ground truth with official codes vs HF golden ----
b2 = []
for stem in SUBJECTS:
    subj = stem + ".weight"
    gs = gs_of(off_files, off_keys, stem)
    codes = read_f32(off_files, off_keys, subj).view(np.int8)
    scale = scale_of(off_files, off_keys, stem)
    ent = off_files[off_keys[subj]][0][subj]
    N, K = int(ent["shape"][0]), int(np.prod(ent["shape"][1:]))
    codes = codes.reshape(N, K)
    assert scale.shape == (N, 1), (stem, scale.shape)
    is_qkv = "qkv" in stem and N % 3 == 0
    if subj not in hf_keys:
        b2.append({"subject": stem, "error": "no HF golden"})
        continue
    # golden in HF order
    golden = read_f32(hf_files, hf_keys, subj).reshape(N, K)
    if is_qkv:
        src = builder_grouped_map(N)
        golden = golden[src]                      # -> grouped file order
    x = torch.from_numpy(codes.astype(np.float32) * scale)
    for tag, fn in (("matrix_norm", v_matrix_norm), ("matrix_raw/sqrt", None),
                    ("old_pyc", v_old)):
        try:
            if fn is None:
                r = v_matrix_raw(x.reshape(N, K // gs, gs)).numpy().reshape(N, K) / math.sqrt(gs)
                label = "matrix_raw_div_sqrt (gate formula w/ matrix stage)"
            else:
                r3 = fn(x.reshape(N, K // gs, gs))
                if isinstance(r3, dict):
                    b2.append({"subject": stem, "variant": tag, "error": r3["TypeError"]})
                    continue
                r = r3.numpy().reshape(N, K)
                label = f"{tag}"
                if tag == "old_pyc":
                    # gate consumes rotate_stages_f32 with /sqrt(gs): try both
                    for div in (1.0, 1.0 / math.sqrt(gs)):
                        c, rr = cos_rel(r * div, golden)
                        b2.append({"subject": stem, "gs": gs, "variant": f"old_pyc_div{div:g}",
                                   "cos": round(c, 6), "relrms": round(rr, 6)})
                    continue
            c, rr = cos_rel(r, golden)
            rec = {"subject": stem, "gs": gs, "variant": label,
                   "cos": round(c, 6), "relrms": round(rr, 6)}
            b2.append(rec)
        except Exception as e:
            b2.append({"subject": stem, "variant": tag, "error": repr(e)[:150]})
out["checks"]["B2_dequant_groundtruth"] = b2

# ---- B3: old qkv_grouped_perm vs builder formula ----
b3 = []
for N in (3840, 384, 11520):
    ar = np.arange(N, dtype=np.int64)
    try:
        old_map = np.asarray(cq_old.qkv_grouped_perm(ar))
        mine = builder_grouped_map(N)
        b3.append({"N": N, "identical": bool(np.array_equal(old_map, mine)),
                   "first_divergence": int(np.argmax(old_map != mine)) if not np.array_equal(old_map, mine) else -1})
    except Exception as e:
        b3.append({"N": N, "error": repr(e)[:150]})
out["checks"]["B3_qkv_perm"] = b3

# ---- B4: full requant of real HF tensors via NEW line == official codes ----
b4 = []
for stem in SUBJECTS[:5]:
    subj = stem + ".weight"
    if subj not in hf_keys:
        b4.append({"subject": stem, "error": "no HF golden"})
        continue
    gs = gs_of(off_files, off_keys, stem)
    ent = off_files[off_keys[subj]][0][subj]
    N, K = int(ent["shape"][0]), int(np.prod(ent["shape"][1:]))
    golden = read_f32(hf_files, hf_keys, subj).reshape(N, K)
    codes_off = read_f32(off_files, off_keys, subj).view(np.int8).reshape(N, K)
    c_new, s_new = cq_new.quant_tensor_f32line(torch.from_numpy(golden), gs, mode="int8")
    c_new = c_new.numpy(); s_new = s_new.reshape(-1, 1)
    is_qkv = "qkv" in subj and N % 384 == 0
    if is_qkv:
        c_new = c_new[builder_grouped_map(N)]
        s_new = s_new[builder_grouped_map(N)]
    s_off = scale_of(off_files, off_keys, stem)
    b4.append({"subject": stem, "gs": gs,
               "code_mismatch_frac_vs_official": float((c_new != codes_off).mean()),
               "scale_max_abdiff": float(np.abs(s_new - s_off).max())})
out["checks"]["B4_requant_official"] = b4

print(json.dumps(out, indent=1))
with open(os.path.join(HERE, "battery_rotperm.json"), "w") as f:
    json.dump(out, f, indent=1)
