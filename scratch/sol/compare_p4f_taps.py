#!/usr/bin/env python3
# P4F tap 逐位比對：native post-patch dump vs golden eager640 taps（門檻不得放寬）
# Gate: token_refiner 相關 tap 必須逐位全同；任何偏離 ⇒ 報首個偏離 tap（pipeline 序）與量級
import os, array

GOLDEN = "/Volumes/SSD2T/h3c_evidence_20261126/p4b_eager640_taps"
NATIVE = "/Volumes/SSD2T/h3c_evidence_20261126/p4f_nativepost640"
OUT = "/Users/cyliu99/AI/h3c_mac_20261125/h3c/scratch/sol/p4f_tap_compare_verdict.txt"

# pipeline 序（refiner 閘門 tap 標 *）
PIPELINE_ORDER = [
    "e0_audio_input_0x32_f32.raw",
    "e0_video_input_0x96_f32.raw",
    "e1_text_source_20x5120_bf16.raw*",
    "e1_after_cond_20x5376_bf16.raw*",
    "e1_after_refiner0_20x5376_bf16.raw*",
    "e1_after_refiner1_20x5376_bf16.raw*",
    "e1_after_finalnorm_20x5376_bf16.raw*",
    "e0_refined_text_0x5376_bf16.raw*",
    "e0_hidden_packed_2790x5376_bf16.raw*",
    "b0_layout_positions_2790x3_f64.raw",
    "b0_rope_cos_table_2790x48_f32.raw",
    "b0_rope_sin_table_2790x48_f32.raw",
    "e2_video_patch_w_0x96_f32.raw",
    "e2_video_patch_b_1x5376_f32.raw",
    "e2_audio_patch_w_0x32_f32.raw",
    "e2_audio_patch_b_1x5376_bf16.raw",
    "e3_source_gpu_20x5376_bf16.raw",
    "e3_condition_w_gpu_20x5376_bf16.raw",
    "e3_condition_b_gpu_20x5376_bf16.raw",
]
def base(n): return n.rstrip("*")
REFINER = {base(n) for n in PIPELINE_ORDER if n.endswith("*")}

def dtype_of(p):
    for suf, dt in (("_f64.raw","f64"),("_f32.raw","f32"),("_bf16.raw","bf16"),("_u32.raw","u32")):
        if p.endswith(suf): return dt
    return "bytes"

def load_numeric(p):
    b = open(p,'rb').read()
    dt = dtype_of(p)
    if dt=="f64": a=array.array('d'); a.frombytes(b); return a
    if dt=="f32": a=array.array('f'); a.frombytes(b); return a
    if dt=="u32": a=array.array('I'); a.frombytes(b); return a
    if dt=="bf16":
        u = array.array('H'); u.frombytes(b)
        f = array.array('f'); f.frombytes(array.array('I',(x<<16 for x in u)).tobytes())
        return f
    return None

def byte_cmp(pa, pb):
    fa, fb = open(pa,'rb'), open(pb,'rb')
    off = 0; first = None; ndiff = 0; CH = 1<<22
    while True:
        ba, bb = fa.read(CH), fb.read(CH)
        if not ba and not bb: break
        if ba==bb and len(ba)==len(bb):
            off += len(ba); continue
        m = min(len(ba), len(bb))
        for i in range(m):
            if ba[i]!=bb[i]:
                ndiff += 1
                if first is None: first = off+i
        off += max(len(ba), len(bb))
        if len(ba)!=len(bb):  # size mismatch tail
            if first is None: first = off
            break
    return first, ndiff

def max_abs(pa, pb):
    da, db = open(pa,'rb').read(), open(pb,'rb').read()
    if len(da)!=len(db): return None, "len-diff %d vs %d" % (len(da), len(db))
    if da==db: return 0.0, None
    a, b2 = load_numeric(pa), load_numeric(pb)
    if a is None:  # bytes dtype: 逐位不同但無數值序
        return None, "non-numeric dtype"
    try:
        m = max((abs(x-y) for x,y in zip(a,b2)), default=0.0)
        nan = any(x!=x or y!=y for x,y in zip(a,b2))
        return m, ("NaN present" if nan else None)
    except Exception as e:
        return None, "deviation error: %s" % e

def order_taps(common):
    order, seen = [], set()
    for n in PIPELINE_ORDER:
        nn = base(n)
        if nn in common: order.append(nn); seen.add(nn)
    order += sorted(c for c in common if c not in seen)
    return order

def main():
    gset = {f for f in os.listdir(GOLDEN) if f.endswith(".raw")}
    nset = {f for f in os.listdir(NATIVE) if f.endswith(".raw")}
    common = gset & nset
    missing_in_native = sorted(gset - nset)
    lines = []
    lines.append("P4F_TAP_COMPARISON (native post-patch vs golden eager640 taps)")
    lines.append("golden=%s native=%s" % (GOLDEN, NATIVE))
    lines.append("gate: token_refiner taps bit-identical (no relaxation)")
    lines.append("golden_raw=%d native_raw=%d common=%d golden-only=%d" % (len(gset), len(nset), len(common), len(missing_in_native)))
    first_div = None; refiner_bad = []
    for n in order_taps(common):
        pa, pb = os.path.join(GOLDEN,n), os.path.join(NATIVE,n)
        first, ndiff = byte_cmp(pa, pb)
        if first is None:
            lines.append("tap=%s bit_identical=1" % n); continue
        dev, note = max_abs(pa, pb)
        dev_s = ("%.6g" % dev) if isinstance(dev,float) else str(dev)
        lines.append("tap=%s bit_identical=0 first_diff_byte_offset=%s bytes_differing>=_%d max_abs_dev=%s %s" % (n, first, ndiff, dev_s, note or ""))
        if first_div is None: first_div = (n, first, dev_s, note)
        if n in REFINER: refiner_bad.append(n)
    refiner_in_common = sorted(REFINER & common)
    refiner_missing = sorted(REFINER - common)
    verdict = "PASS"
    if refiner_missing: verdict = "FAIL"
    if refiner_bad: verdict = "FAIL"
    lines.append("refiner_taps_common=%d refiner_taps_missing=%s refiner_taps_deviated=%s" % (len(refiner_in_common), refiner_missing or "none", refiner_bad or "none"))
    lines.append("first_diverging_tap(pipeline order)=%s" % ("%s off=%s mag=%s %s" % first_div if first_div else "none — all common taps bit-identical"))
    lines.append("VERDICT=%s" % verdict)
    txt = "\n".join(lines) + "\n"
    open(OUT,"w").write(txt)
    print(txt)

if __name__ == "__main__":
    main()
