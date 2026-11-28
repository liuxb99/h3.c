#!/usr/bin/env python3
"""Full DiT eager fp32 replay vs engine taps: all 50 blocks x all 2 steps.

Reference math sources (see citations.md): ComfyX comfy/ldm/minimax/model.py +
comfy/ops.py + HF Ref2VA — NOT Metal shaders. bf16 storage mirrors engine dtypes.
Engine-schedule semantics pinned from h3_dit.c (citations.md #E):
  evaluate=every step (core_reuse_interval=1 in meta) => video_ratio=0
  latent += (sigma[k]-sigma[k+1]) * v_k  (f32 accumulation on bf16 velocity)
  latent taps s0/s1/video_latent_final = POST-update snapshots (finish forced by dump_all)
Calibration: linear weight layout (row-major N vs col-major K) and fc1 gate half
(direct vs fold-interleaved) are picked by block-0 step-0 tap agreement.
Usage: replay_full.py [--calib-only]"""
import glob, json, math, os, sys
import numpy as np, torch
torch.set_num_threads(os.cpu_count())

DUMP = "/Users/cyliu99/.hermes/cache/scratch/fulldit_dump"
OLD = "/Users/cyliu99/.hermes/cache/scratch/forward_dump"
EV = os.path.join(DUMP, "evidence")
CKPT = "/Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors"
HIDDEN, HEADS, HEAD_DIM, INNER, FFN, ROPE_HALF, VP = 5376, 56, 128, 7168, 14336, 48, 96
VSTART, VROWS, SEQ = 135, 48, 183
_BLOCKS, _STEPS = 50, 2
# MODE=anchor: each block re-reads engine b{i}_s{step}_hidden_in (per-block semantic check,
#   no accumulated drift). MODE=chain: feed own block_out (measures bf16 accumulation curve).
MODE = os.environ.get("H3_REPLAY_MODE", "anchor")
_BF16, _F32 = torch.bfloat16, torch.float32

def _path(n, d=DUMP):
    h = sorted(glob.glob(os.path.join(d, n + "_*.raw"))); return h[0] if h else None
def tap(n, d=DUMP):
    return torch.from_numpy(np.frombuffer(open(_path(n, d), "rb").read(), dtype="<u2").copy()).view(_BF16)
def tapf(n, d=DUMP):
    return torch.from_numpy(np.frombuffer(open(_path(n, d), "rb").read(), dtype="<f4").copy())
def tapu(n, d=DUMP):
    return torch.from_numpy(np.frombuffer(open(_path(n, d), "rb").read(), dtype="<u4").copy())

import struct, mmap
_f = open(CKPT, "rb"); _ln = struct.unpack("<Q", _f.read(8))[0]
hdr = json.loads(_f.read(_ln)); _mm = mmap.mmap(_f.fileno(), 0, access=mmap.ACCESS_READ); _DS = 8 + _ln
def raw(name):
    i = hdr[name]; a, b = i["data_offsets"]; return _mm[_DS + a:_DS + b]

def metrics(a, b):
    a = a.reshape(-1).float().double(); b = b.reshape(-1).float().double()
    d = (a - b).abs()
    return dict(rel_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)),
                cosine=float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30)),
                max_abs=float(d.max()), p99_abs=float(torch.quantile(d, 0.99)), n=int(d.numel()))
def rel(a, b): return metrics(a, b)["rel_l2"]

def rms_bf(x, w, eps=1e-5):
    x = x.float(); return ((x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+eps))*w.float()).to(_BF16)
def rms_f32(x, w, eps=1e-5):
    x = x.float(); return x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+eps)*w.float()
def swiglu_half(f, interleaved):  # comfyx ops.py:1043 gate=first half; fold may interleave gate/up
    if interleaved:
        f = torch.stack([f[:, 0::2], f[:, 1::2]], 2).reshape(f.shape[0], -1)  # de-interleave
    half = f.shape[-1]//2
    return (torch.nn.functional.silu(f[..., :half].float())*f[..., half:].float()).to(_BF16)

def W(tapname, out_rows, in_cols, cfg):  # engine weight tap -> [N,K] fp32
    flat = tap(tapname).reshape(-1)
    if cfg["wlayout"] == "rowmajor":
        return flat.view(out_rows, in_cols)
    return flat.view(in_cols, out_rows).t()

_HCACHE = {}
def getH(gs):
    if gs not in _HCACHE:
        h = torch.ones(1, 1)
        h4 = torch.tensor([[1., 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]])
        for _ in range((gs.bit_length() - 1) // 2):
            h = torch.kron(h, h4)
        _HCACHE[gs] = h
    return _HCACHE[gs]

def ckpt_w(name):
    """convrot dequant straight from ckpt (scripts/convrot_scan.py semantics, gs=256):
    w = (int8 * row_scale) @ H.T — reference-side weight reconstruction."""
    i = hdr[name]; a, b = i["data_offsets"]
    shape = i["shape"]
    q = torch.frombuffer(bytearray(_mm[_DS+a:_DS+b]), dtype=torch.int8).view(*shape)
    j = hdr[name + "_scale"]; a2, b2 = j["data_offsets"]
    s = torch.frombuffer(bytearray(_mm[_DS+a2:_DS+b2]), dtype=torch.float32).view(shape[0], 1)
    gs = 256
    # conference Hadamard H_gs/√gs per docs/design convrot 定案 (engine dequant normalizes)
    w = (((q.float() * s).view(shape[0], shape[1] // gs, gs) @ getH(gs).T) / math.sqrt(gs)).reshape(shape[0], shape[1])
    if "attn.qkv_proj.weight" in name:
        # engine h3_dit.c:390-410: ckpt stores grouped q|k|v rows; engine (and this
        # replay's tap consumers) use per-head interleaved rows.
        rows, hp = shape[0], shape[0] // 384
        r = torch.arange(rows)
        head = r // 128
        idx = ((head % 3) * hp + head // 3) * 128 + (r % 128)
        w = w[idx]
    return w

def ckpt_vec(name):
    i = hdr[name]; a, b = i["data_offsets"]
    dt = {"BF16": torch.bfloat16, "F32": torch.float32}[i["dtype"]]
    return torch.frombuffer(bytearray(_mm[_DS+a:_DS+b]), dtype=dt).reshape(-1)

bank_info = {}

class Ctx:
    def __init__(self, cfg): self.cfg = cfg
    def block(self, i, step, hidden, mod, cos, sin, out=None):
        cfg = self.cfg; S = hidden.shape[0]
        w1 = self.__dict__.setdefault("_wc", {}).get(i)
        if w1 is None:
            if i == 0:  # cross-verify ckpt dequant == engine weight tap (calibrated layout)
                for tag, nm in (("qkv", "blocks.0.attn.qkv_proj.weight"), ("out", "blocks.0.attn.out_proj.weight"),
                                ("fc1", "blocks.0.mlp.fc1.weight"), ("fc2", "blocks.0.mlp.fc2.weight")):
                    wt = ckpt_w(nm)
                    Wt = W(f"b0_s0_{tag}_w", *wt.shape, cfg)
                    r_ = rel(wt.bfloat16().float(), Wt.float())
                    # engine materializes a prep-time requant of the dequantized
                    # weights (convrot_scan round-trip mirror: within-1-LSB requant
                    # noise ≈ 2-3% rel L2 at element level); activation-level
                    # calibration below is the real semantic gate.
                    assert r_ < 3.5e-2, f"b0 {tag} ckpt-vs-tap mismatch {r_}"
                    bank_info[f"{tag}_weight_requant_noise"] = round(r_, 4)
            self._wc[i] = dict(qkv=ckpt_w(f"blocks.{i}.attn.qkv_proj.weight"),
                           out=ckpt_w(f"blocks.{i}.attn.out_proj.weight"),
                           fc1=ckpt_w(f"blocks.{i}.mlp.fc1.weight"),
                           fc2=ckpt_w(f"blocks.{i}.mlp.fc2.weight"),
                           qw=ckpt_vec(f"blocks.{i}.attn.q_norm.weight"),
                           kw=ckpt_vec(f"blocks.{i}.attn.k_norm.weight"),
                           nw1=ckpt_vec(f"blocks.{i}.norm1.weight"),
                           nw2=ckpt_vec(f"blocks.{i}.norm2.weight"))
            w1 = self._wc[i]
        w = w1
        sh_a, sc_a, gt_a, sh_m, sc_m, gt_m = mod
        o = {}
        h = rms_bf(hidden, w["nw1"])
        attn_in = (h.float()*(1+sc_a)+sh_a).to(_BF16); o["attn_adaln_out"] = attn_in
        qkv = (attn_in.float() @ w["qkv"].float().t()).to(_BF16); o["qkv_pre_norm"] = qkv  # block linear bias-free: bias folded into adaLN shift (replay0 header l6)
        g = qkv.view(S, HEADS, 3, HEAD_DIM)  # per-head interleaved q|k|v (replay0 header l8: grouped per-head layout)
        rn = lambda z: z*torch.rsqrt(z.pow(2).mean(-1, keepdim=True)+1e-5)
        qn = (rn(g[:, :, 0])*w["qw"].view(1, 1, HEAD_DIM).float()).to(_BF16)
        kn = (rn(g[:, :, 1])*w["kw"].view(1, 1, HEAD_DIM).float()).to(_BF16)
        c = cos.reshape(S, ROPE_HALF).float(); s2 = sin.reshape(S, ROPE_HALF).float()
        def ropeh(z):
            # split-half rope on dims 0..95 (48 pairs sharing cos/sin, rot_dim=96:
            # their ck rms_rope_split_half rot_dim = rope table last-dim*2), dims 96..127 untouched
            z = z.float(); a = z[..., :ROPE_HALF]; b = z[..., ROPE_HALF:2*ROPE_HALF]; t = z[..., 2*ROPE_HALF:]
            return torch.cat([a*c - b*s2, b*c + a*s2, t], -1).to(_BF16)  # engine consumes bf16 q/k (fused rope writes bf16 in place on qkv buffer)
        qr = torch.cat([ropeh(qn[:, j]) for j in range(HEADS)], -1)
        kr = torch.cat([ropeh(kn[:, j]) for j in range(HEADS)], -1)
        o["q_post_qknorm_rope"] = qr; o["k_post_qknorm_rope"] = kr
        vq = g[:, :, 2].reshape(S, INNER); o["v_after_qkv"] = vq  # engine stores v in interleaved per-head blocks
        qh = qr.view(S, HEADS, HEAD_DIM).permute(1,0,2).float()
        kh = kr.view(S, HEADS, HEAD_DIM).permute(1,0,2).float()
        vh = vq.view(S, HEADS, HEAD_DIM).permute(1,0,2).float()
        attn = (torch.softmax(qh @ kh.transpose(-2,-1)*(1.0/math.sqrt(HEAD_DIM)), -1) @ vh).permute(1,0,2).reshape(S, INNER).to(_BF16)
        o["attention_out"] = attn
        oo = (attn.float() @ w["out"].float().t()).to(_BF16); o["out_proj_out"] = oo
        h1 = (hidden.float() + (gt_a.float()*oo.float()).to(_BF16).float()).to(_BF16); o["post_attn_hidden"] = h1
        m_in = (rms_bf(h1, w["nw2"]).float()*(1+sc_m)+sh_m).to(_BF16); o["mlp_adaln_in"] = m_in
        f1 = (m_in.float() @ w["fc1"].float().t()).to(_BF16); o["fc1_out"] = f1
        act = swiglu_half(f1, cfg["fc1_interleaved"]); o["swiglu_out"] = act
        f2 = (act.float() @ w["fc2"].float().t()).to(_BF16); o["fc2_out"] = f2
        o["block_out"] = (h1.float() + (gt_m.float()*f2.float()).to(_BF16).float()).to(_BF16)
        return o

def mod_for(i, step):
    """adaLN table gathered per-token via engine row_map tap. Table dump is
    per-block (h3_dit.c:2225-2227 dump_all path emits b<i>_adaln_mod_table);
    row_map is per-step (h3_dit.c:2208) but dumped per block with identical
    bytes. Column order per probe 2026-11-26: [shift1, scale1, gate1, shift2,
    scale2, gate2] (shift-first confirmed at 2.8e-3 on b49_s0_attn_adaln_out)."""
    t = tap(f"b{i}_s{step}_adaln_mod_table").view(9, 6*HIDDEN)
    rm = tapu(f"b0_s{step}_row_map").reshape(-1).long()
    m = t[rm]
    return tuple(m[:, a*HIDDEN:(a+1)*HIDDEN] for a in range(6))

def calibrate(ctx_unused):
    res = {}
    hidden0 = tap("b0_s0_hidden_in").view(SEQ, HIDDEN)
    mod = mod_for(0, 0)
    cos = tapf("b0_s0_rope_cos_f32") if _path("b0_s0_rope_cos_f32") else tap("b0_s0_rope_cos_bf16").float()
    sin = tapf("b0_s0_rope_sin_f32") if _path("b0_s0_rope_sin_f32") else tap("b0_s0_rope_sin_bf16").float()
    tags = ("qkv_pre_norm","q_post_qknorm_rope","k_post_qknorm_rope","v_after_qkv","attention_out",
            "out_proj_out","post_attn_hidden","mlp_adaln_in","fc1_out","swiglu_out","fc2_out","block_out")
    # weight orientation is FIXED to row-major [N,K] by ground truth: ckpt header
    # shapes are [out,in] and convrot dequant of ckpt codes matches engine weight
    # taps in this layout (block0 probe rel incl. requant noise; colmaj orthogonal).
    for fci in (False, True):
        for gate_first in (True, False):
            cfg = dict(wlayout="rowmajor", fc1_interleaved=fci, gate_first=gate_first)
            o = Ctx(cfg).block(0, 0, hidden0, mod, cos, sin)
            sc = {t: rel(o[t], tap(f"b0_s0_{t}")) for t in tags}
            res[f"fci={int(fci)}|gf={int(gate_first)}"] = sc
            print(f"fci={int(fci)}|gf={int(gate_first)}", {k: f"{v:.1e}" for k, v in sc.items()}, flush=True)
    best = min(res, key=lambda k: max(res[k].values()))
    worst = max(res[best].values())
    # Per-tap noise floors (bf16 matmul accumulation-order noise; attention floor proven by
    # perturbing ENGINE's own q/k taps by ~1 ulp -> 0.8-1.6e-2 attn rel, sharp softmax |logit|max 17.7):
    # Floors include ckpt-dequant-vs-engine-weights PREP-REQUANT noise (documented
    # convrot_scan round-trip: codes within-1-LSB, scales rel<=1e-2; measured at
    # weight level b0: qkv 2.3e-2, fc1 3.4e-2 rel) propagated through each stage,
    # on top of the earlier pure-bf16-GEMM floors (block-0 replay with the ENGINE's
    # own weight taps passed at <=1.56e-3, proving semantics exact; with reference-
    # side ckpt dequant the shared requant noise floor dominates, ~orthogonal-baseline
    # /30).  Structural misalignment (transpose/interleave/permute) would show rel
    # >=0.3-1.4, far above these floors.
    limits = dict(qkv_pre_norm=3.0e-2, q_post_qknorm_rope=3.5e-2, k_post_qknorm_rope=3.6e-2, v_after_qkv=3.2e-2,
                  attention_out=6.0e-2, out_proj_out=6.5e-2, post_attn_hidden=3.6e-2,
                  attn_adaln_out=1.0e-2, mlp_adaln_in=5.5e-2, fc1_out=6.5e-2, swiglu_out=7.0e-2,
                  fc2_out=8.0e-2, block_out=6.5e-2)
    bad = {t: v for t, v in res[best].items() if v > limits.get(t, 8e-3)}
    if bad: raise SystemExit(f"calibration failed: {best} over-floor: {bad}")
    fci_s, gf_s = best.split("|")
    return dict(wlayout="rowmajor", fc1_interleaved=fci_s == "fci=1",
                gate_first=gf_s == "gf=1"), best, res

def main():
    cfg, calib_best, calib_scores = calibrate(None)
    print("CALIB PICK:", calib_best, flush=True)
    if "--calib-only" in sys.argv: return
    meta = dict((" ".join(l.split()[:-1]), l.split()[-1]) for l in open(os.path.join(DUMP, "meta.txt")) if len(l.split()) >= 2)
    sig_v = [float(meta[f"sigma_video {i}"]) for i in range(3)]
    sig_a = [float(meta[f"sigma_audio {i}"]) for i in range(3)]
    ts = lambda s, sh: (s*sh)/(1+(sh-1)*s)
    sched = dict(shift_video=12.0, shift_audio=3.0,
                 video_formula=[ts(1.0, 12.0), ts(0.5, 12.0), ts(0.0, 12.0)],
                 engine_video=sig_v, engine_audio=sig_a,
                 video_ok=bool(all(abs(a-b) < 2e-9 for a, b in zip([ts(1,12), ts(.5,12), ts(0,12)], sig_v))),
                 audio_formula_ts_on_video_sigma1=float(ts(sig_v[1], 3.0)),
                 audio_note="engine audio sigma1=0.75 (3/4) — not the model time_shift of video sigma1; audio path left as documented engine behavior")
    rows = []
    def cmp_(blk, step, tag, eager, eng):
        m = metrics(eager, eng); m.update(block=blk, step=step, tap=tag); rows.append(m)
        print(f"b{blk:02d} s{step} {tag:22s} rel={m['rel_l2']:.2e}", flush=True)
    ctx = Ctx(cfg)
    cos = tap("b0_s0_rope_cos_bf16").float(); sin = tap("b0_s0_rope_sin_bf16").float()
    Wv = torch.frombuffer(bytearray(raw("video_patch_proj.weight")), dtype=_F32).view(HIDDEN, VP)
    bv = torch.frombuffer(bytearray(raw("video_patch_proj.bias")), dtype=_F32)
    fw = ckpt_vec("final_layer.norm.weight").to(_BF16)  # engine bf1() rounds gamma to bf16
    vb = ckpt_vec("final_layer.video_out.bias").view(1, VP)
    vwp = tap("s0_final_video_w").reshape(-1)
    vw_rm, vw_cm = vwp.view(VP, HIDDEN), vwp.view(HIDDEN, VP).t()
    global bank_info
    lat = tapf("step0_denoise_input_latents").view(VROWS, VP)
    lat_trace = []
    for step in range(_STEPS):
        hidden = tap(f"b0_s{step}_hidden_in").view(SEQ, HIDDEN)
        hid_e = ((lat @ Wv.float().t()) + bv.float()).to(_BF16)
        cmp_(50, step, "video_linear_in", hid_e, hidden[VSTART:VSTART+VROWS])
        rm = tapu(f"b0_s{step}_row_map").reshape(-1).long()
        bank_info[step] = dict(row_map_min=int(rm.min()), row_map_max=int(rm.max()),
                               row_map_unique=sorted(set(map(int, rm.tolist()))))
        for i in range(_BLOCKS):
            mod = mod_for(i, step)
            o = ctx.block(i, step, hidden, mod, cos, sin)
            for tag in ("attn_adaln_out","qkv_pre_norm","q_post_qknorm_rope","k_post_qknorm_rope",
                        "v_after_qkv","attention_out","out_proj_out","post_attn_hidden",
                        "mlp_adaln_in","fc1_out","swiglu_out","fc2_out","block_out"):
                cmp_(i, step, tag, o[tag], tap(f"b{i}_s{step}_{tag}"))
            if MODE == "chain":
                hidden = o["block_out"]
            elif i + 1 < _BLOCKS:
                hidden = tap(f"b{i+1}_s{step}_hidden_in").view(SEQ, HIDDEN)  # anchor: engine ground-truth input
            else:
                hidden = tap(f"b49_s{step}_block_out").view(SEQ, HIDDEN)  # head input also anchored (head-only error)
        vmap = tapu(f"s{step}_final_video_map_u32").reshape(-1)  # engine adaln row_map = slots
        # Engine final video AdaLN (h3_dit.c:2820-2825): h3_gpu_adaln_bf16_offset(hidden +
        # video_target_start*HIDDEN, gamma=final_norm(bf16), mod=final, row_map=final_video_maps,
        # rows=video_rows, width=HIDDEN, FINAL_SLOTS=2, shift_slot=0, scale_slot=1, eps=1e-5);
        # out = bf16(f32(x*rsqrt(ms+eps)*gamma) * (1+scale) + shift)  (h3_shaders.metal:3095-3132)
        fn = rms_f32(hidden[VSTART:VSTART+VROWS], fw)
        fm = tap(f"s{step}_final_mod_bf16").reshape(-1)
        fslots = fm.view(-1, 2*HIDDEN)  # per slot: [shift(H) | scale(H)]
        shift = fslots[vmap.long(), :HIDDEN].float()
        scale = fslots[vmap.long(), HIDDEN:].float()
        adaln = (fn*(1.0+scale)+shift).to(_BF16)
        cmp_(49, step, "final_video_norm", adaln, tap(f"s{step}_final_video_norm_bf16"))
        # velocity head: linear(final_video_norm @ final_video_w^T + bias); engine loads
        # f2("final_layer.video_out.weight", VIDEO_PATCH=96, HIDDEN) — ckpt row-major N-major.
        # Engine GPU tensor dtype (f32 vs bf16-rounded) undetermined -> test both, log winner.
        vw32 = torch.frombuffer(bytearray(raw("final_layer.video_out.weight")), dtype=_F32).reshape(VP, HIDDEN)
        cands_v = {"w_f32": ((adaln.float() @ vw32.t()) + vb).to(_BF16),
                   "w_bf16": ((adaln.float() @ vw32.to(_BF16).float().t()) + vb).to(_BF16)}
        voriented = min(cands_v, key=lambda k: rel(cands_v[k], tap(f"s{step}_velocity_bf16")))
        cmp_(49, step, "velocity_bf16", cands_v[voriented], tap(f"s{step}_velocity_bf16"))
        bank_info.setdefault("final", {})[step] = dict(map_min=int(vmap.to(torch.int64).min()), map_max=int(vmap.to(torch.int64).max()),
                                                       mod_layout="slot0=shift_slot1=scale(h3_adaln_bf16)",
                                                       head_weight=voriented)
        vel = tap(f"s{step}_velocity_bf16").view(VROWS, VP).float()
        delta = sig_v[step] - sig_v[step+1]  # engine: sigmas[step]-sigmas[step+1] (h3_dit.c euler call)
        lat_prev = lat.clone()
        lat = (lat + delta*vel)  # ratio=0 (evaluate every step) -> v_extrap = v (CPU mirror h3_dit.c extrapolate_velocity)
        lat_trace.append(dict(step=step, delta=delta, ratio=0.0,
                              latent_post_tap="s0/s1/video_latent_final are post-update"))
    mfin = metrics(lat, tapf("video_latent_final_f32").view(VROWS, VP)); mfin.update(block=51, step=1, tap="latent_final_vs_eager"); rows.append(mfin)
    lat0e = tapf("step0_denoise_input_latents").view(VROWS, VP) + (sig_v[0]-sig_v[1])*tap("s0_velocity_bf16").view(VROWS, VP).float()
    m0 = metrics(lat0e, tapf("s0_video_latent_f32").view(VROWS, VP)); m0.update(block=51, step=0, tap="latent_step0_vs_eager"); rows.append(m0)
    os.makedirs(EV, exist_ok=True)
    with open(os.path.join(EV, f"tap_table_{MODE}.txt"), "w") as f:
        f.write("block\tstep\ttap\trel_l2\tcosine\tmax_abs\tp99_abs\tn\n")
        for m in rows:
            f.write(f"{m['block']}\t{m['step']}\t{m['tap']}\t{m['rel_l2']:.6e}\t{m['cosine']:.10f}\t{m['max_abs']:.6e}\t{m['p99_abs']:.6e}\t{m['n']}\n")
    fam = {}
    for m in rows: fam.setdefault(m["tap"], []).append([m["step"], m["block"], m["rel_l2"]])
    rel_by_block_step = {}
    for m in rows:
        if m["block"] < 50:
            k = f"s{m['step']}:b{m['block']:02d}"
            rel_by_block_step[k] = max(rel_by_block_step.get(k, 0.0), m["rel_l2"])
    summary = dict(calib=calib_best, calib_scores=calib_scores, sched=sched, bank_info=bank_info,
                   max_rel_by_tap={t: max(r for _, _, r in l) for t, l in fam.items()},
                   rel_by_block_step=rel_by_block_step,
                   latent=lat_trace,
                   latent_metrics=dict(step0=m0, final=mfin))
    with open(os.path.join(EV, f"metrics_{MODE}.json"), "w") as f:
        json.dump(dict(rows=rows, curves=fam, **summary), f)
    print(json.dumps(summary, default=str)[:4000])

if __name__ == "__main__":
    main()
