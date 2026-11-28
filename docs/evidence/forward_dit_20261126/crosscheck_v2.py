#!/usr/bin/env python3
"""A4 crosscheck v2: ComfyX 的 OWN kernel 路徑（comfy_kitchen 0.2.42 installed wheel，
ComfyX 出貨同款）對同一 int8 convrot ckpt + 同一 engine taps。

v1 誤用 w4a4 packed API；本 ckpt 實為 TensorWiseINT8Layout(convrot=True, per-channel
scale, row-major, 8-bit codes)——即 sidecar {format=int8_tensorwise, convrot, convrot_groupsize}
（見 docs/design/2026-11-25_convrot量化語意定案.md）。v2 全部改走 ck 自家
dequantize_int8_convrot_weight_dtype（torch.ops，編譯 kernel，非 eager python）。
"""
import glob, json, math, os, struct, mmap
import numpy as np, torch
import comfy_kitchen as ck
from comfy_kitchen.tensor.int8 import TensorWiseINT8Layout as L

M = "/Users/cyliu99/.hermes/cache/scratch/fulldit_dump"
CKPT = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
EV = os.path.join(M, "evidence")
HID, HEADS, HD, FFN, INNER, ROPE = 5376, 56, 128, 14336, 7168, 48
S = 183
_BF = torch.bfloat16
_f = open(CKPT, "rb"); _n = struct.unpack("<Q", _f.read(8))[0]
hdr = json.loads(_f.read(_n)); _mm = mmap.mmap(_f.fileno(), 0, access=mmap.ACCESS_READ); _DS = 8 + _n

def raw(name, dtype):
    i = hdr[name]; a, b = i["data_offsets"]
    return torch.frombuffer(bytearray(_mm[_DS+a:_DS+b]), dtype=dtype).reshape(hdr[name]["shape"])

def meta(name):
    k = name + ".comfy_quant"
    if k not in hdr: return None
    i = hdr[k]; a, b = i["data_offsets"]
    return json.loads(_mm[_DS+a:_DS+b].decode())

def tapf(n, dt=np.dtype("<u2")):
    p = sorted(glob.glob(os.path.join(M, n + "_*.raw")))[0]
    arr = np.frombuffer(open(p, "rb").read(), dtype=dt).copy()
    t = torch.from_numpy(arr)
    return t.view(_BF) if dt == np.dtype("<u2") else t

def rel(a, b):
    a = a.reshape(-1).double(); b = b.reshape(-1).double()
    return float((a-b).norm() / b.norm().clamp_min(1e-30))

def their_dequant(name):
    """weights through comfy_kitchen's own compiled convrot dequant (their Params)."""
    q = raw(name + ".weight", torch.int8)
    s = raw(name + ".weight_scale", torch.float32)
    m = meta(name + ".weight") or meta(name) or {}
    gs = int(m.get("convrot_groupsize", 256))
    params = L.Params(scale=s, orig_dtype=torch.float32, orig_shape=tuple(q.shape),
                      is_weight=True, convrot=True, convrot_groupsize=gs,
                      transposed=False, tiled=False)
    return L.dequantize(q, params), m, gs

def interleave_qkv(w):
    """ckpt grouped q|k|v rows -> engine per-head interleaved rows
    (h3_dit.c:390-410). Same map as replay_full.ckpt_w applies."""
    N = w.shape[0]; hp = N // 384; r = torch.arange(N)
    idx = ((r // 128 % 3) * hp + r // 128 // 3) * 128 + r % 128
    return w[idx]

res = {}
BL = "blocks.49"

# ---- 1. their compiled dequant vs OUR independent ckpt-dequant (b0 anchors) ----
import replay_full as REF  # our independent eager reference module (its ckpt_w = own dequant math)
res["format_meta"] = {b: (meta(b+".weight") or {}).get("format") for b in ("blocks.0", BL)}
anchor = {}
for tag, nm in (("fc1", "blocks.0.mlp.fc1.weight"), ("qkv", "blocks.0.attn.qkv_proj.weight")):
    w_theirs, m0, gs = their_dequant("blocks.0.mlp.fc1" if tag=="fc1" else "blocks.0.attn.qkv_proj")
    w_ours = REF.ckpt_w(nm)
    if tag == "qkv":
        w_theirs = interleave_qkv(w_theirs)  # theirs stays grouped; engine/ours interleaved
    anchor[tag] = dict(theirs_vs_ours=rel(w_theirs, w_ours),
                       theirs_vs_engine_tap=rel(w_theirs, tapf(f"b0_s0_{tag}_w").float()),
                       gs=gs, meta=m0)
res["b0_dequant_anchor"] = anchor

# ---- 2. their dequant(b49 fc1) x engine's real activation vs engine fc1_out tap ----
w49, m49, gs49 = their_dequant(BL+".mlp.fc1")
res["dequant_b49_fc1_meta"] = dict(meta=m49, shape=list(w49.shape))
x = tapf("b49_s0_mlp_adaln_in").reshape(S, HID).float()
out_theirs = x @ w49.t()
gt = tapf("b49_s0_fc1_out").reshape(S, 2*FFN).float()
res["ck_dequant_gemm_vs_engine_fc1tap"] = rel(out_theirs, gt)
# scale sanity: max |codes| should reach ~127 somewhere per row (per-channel scale usage)
q49 = raw(BL+".mlp.fc1.weight", torch.int8)
amx = q49.abs().amax(1).float()
res["b49_fc1_codes_perrow_absmax"] = dict(min=float(amx.min()), median=float(amx.median()), max=float(amx.max()), note="per-row scale saturation check: expect ~127")

# ---- 3. their fused rms_rope_split_half on engine's real pre-norm qkv buffer ----
qkvbuf = tapf("b0_s0_qkv_pre_norm")  # bf16, per-head interleaved [S, 3*56*128]
freqs_cos = tapf("b0_rope_cos_table", np.dtype("<f4")).view(S, ROPE)
freqs_sin = tapf("b0_rope_sin_table", np.dtype("<f4")).view(S, ROPE)
q4 = qkvbuf.reshape(S, HEADS, 3, HD)[:, :, 0, :].unsqueeze(0).contiguous()
k4 = qkvbuf.reshape(S, HEADS, 3, HD)[:, :, 1, :].unsqueeze(0).contiguous()
qw = tapf("b0_s0_q_norm_w").reshape(HD); kw = tapf("b0_s0_k_norm_w").reshape(HD)
# 6D freqs_cis matrix per comfy/ldm/ideogram4/model.py:31-37 (_split_half_rope_matrix):
# stack((cos_lo, neg_sin, sin, cos_hi), -1) -> [S,48,2,2] -> [1,S,1,48,2,2]
fc6 = torch.stack((freqs_cos, -freqs_sin, freqs_sin, freqs_cos), -1) \
        .reshape(S, ROPE, 2, 2).unsqueeze(0).unsqueeze(2).contiguous().float()
try:
    qq, kk = ck.rms_rope_split_half(q4.clone(), k4.clone(), fc6, qw, kw,
                                    epsilon=1e-5, rot_dim=2*ROPE)
    res["rope_kernel_q_rel"] = rel(qq[0].reshape(S, HEADS*HD), tapf("b0_s0_q_post_qknorm_rope"))
    res["rope_kernel_k_rel"] = rel(kk[0].reshape(S, HEADS*HD), tapf("b0_s0_k_post_qknorm_rope"))
    res["rope_freqs_layout"] = "stack(cos,-sin,sin,cos)->[1,S,1,48,2,2]"
except Exception as e:
    res["rope_kernel_errors"] = {"matrix6d": str(e)[:400]}

# ---- 4. full block 49 step-0 through THEIR dequant weights vs engine taps ----
t = tapf("b49_s0_adaln_mod_table").view(9, 6*HID)  # modulation table is PER-BLOCK
rm = torch.from_numpy(np.frombuffer(open(sorted(glob.glob(os.path.join(M, "b49_s0_row_map_*.raw")))[0], "rb").read(), dtype="<u4").copy()).reshape(-1).long()
mod = t[rm]
sh_a, sc_a, gt_a, sh_m, sc_m, gt_m = (mod[:, a*HID:(a+1)*HID] for a in range(6))
nw1 = tapf("b49_s0_norm1_w").reshape(HID); nw2 = tapf("b49_s0_norm2_w").reshape(HID)
qw49 = tapf("b49_s0_q_norm_w").reshape(HD); kw49 = tapf("b49_s0_k_norm_w").reshape(HD)
rc = tapf("b49_s0_rope_cos_bf16").float().reshape(S, ROPE)
rs = tapf("b49_s0_rope_sin_bf16").float().reshape(S, ROPE)
w_qkv = interleave_qkv(their_dequant(BL+".attn.qkv_proj")[0])
w_out, _, _ = their_dequant(BL+".attn.out_proj")
w_fc2, _, _ = their_dequant(BL+".mlp.fc2")
def rms_f(x, w): 
    xf = x.float(); return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * w.float()
hidden = tapf("b49_s0_hidden_in").reshape(S, HID)
line = {}
attn_in = (rms_f(hidden, nw1).float()*(1+sc_a.float())+sh_a.float()).to(_BF)
line["attn_adaln_out"] = rel(attn_in, tapf("b49_s0_attn_adaln_out"))
qkv = (attn_in.float() @ w_qkv.t()).to(_BF)
line["qkv_pre_norm"] = rel(qkv, tapf("b49_s0_qkv_pre_norm"))
g = qkv.view(S, HEADS, 3, HD)
rn = lambda z: z.float()*torch.rsqrt(z.float().pow(2).mean(-1, keepdim=True) + 1e-5)
qn = (rn(g[:,:,0]) * qw49.view(1,1,HD)).to(_BF); kn = (rn(g[:,:,1]) * kw49.view(1,1,HD)).to(_BF)
c = rc.reshape(S, ROPE); s2 = rs.reshape(S, ROPE)
def ropeh(z):
    z = z.float(); a = z[..., :ROPE]; b = z[..., ROPE:2*ROPE]; tt = z[..., 2*ROPE:]
    return torch.cat([a*c - b*s2, b*c + a*s2, tt], -1).to(_BF)
qr = torch.cat([ropeh(qn[:, j]) for j in range(HEADS)], -1)
kr = torch.cat([ropeh(kn[:, j]) for j in range(HEADS)], -1)
line["q_post_qknorm_rope"] = rel(qr, tapf("b49_s0_q_post_qknorm_rope"))
qh = qr.view(S, HEADS, HD).permute(1,0,2).float(); kh = kr.view(S, HEADS, HD).permute(1,0,2).float()
vh = g[:,:,2].reshape(S, INNER).view(S, HEADS, HD).permute(1,0,2).float()
attn = (torch.softmax(qh @ kh.transpose(-2,-1)/math.sqrt(HD), -1) @ vh).permute(1,0,2).reshape(S, INNER).to(_BF)
line["attention_out"] = rel(attn, tapf("b49_s0_attention_out"))
oo = (attn.float() @ w_out.t()).to(_BF)
line["out_proj_out"] = rel(oo, tapf("b49_s0_out_proj_out"))
h1 = (hidden.float() + (gt_a.float()*oo.float()).to(_BF).float()).to(_BF)
line["post_attn_hidden"] = rel(h1, tapf("b49_s0_post_attn_hidden"))
m_in = (rms_f(h1, nw2).float()*(1+sc_m.float())+sh_m.float()).to(_BF)
line["mlp_adaln_in"] = rel(m_in, tapf("b49_s0_mlp_adaln_in"))
f1 = (m_in.float() @ w49.t()).to(_BF)
line["fc1_out"] = rel(f1, gt)
# gate-first-half split per comfyx ops.py:1044
gate, up = f1.float()[:, :FFN], f1.float()[:, FFN:]
act = (gate*torch.sigmoid(gate)*up).to(_BF)
line["swiglu_out"] = rel(act, tapf("b49_s0_swiglu_out"))
f2 = (act.float() @ w_fc2.t()).to(_BF)
line["fc2_out"] = rel(f2, tapf("b49_s0_fc2_out"))
bout = (h1.float() + (gt_m.float()*f2.float()).to(_BF).float()).to(_BF)
line["block_out"] = rel(bout, tapf("b49_s0_block_out"))
res["block49_theirs_vs_engine"] = line

with open(os.path.join(EV, "crosscheck_v2.json"), "w") as f:
    json.dump(res, f, default=str, indent=1)
print(json.dumps(res, default=str, indent=1)[:6000])
