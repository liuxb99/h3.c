#!/usr/bin/env python3
# 9eef 官方發布檔歧義消解：engine vs release vs MiniMaxAI-bf16 三方歸因
# 前置：hf 下載完成且整檔 sha==9eef...（本腳本自驗 sha，不符即 FAIL 中止）
# 判讀：
#  A) engine==release 於 279 內 -> patch 忠於 release；audit 對 MiniMaxAI 比對 = 錯參考
#  B) engine!=release 於 279 內 且 release==MiniMaxAI -> patch 未忠實複製任一參考 = 真 patch 缺陷候選
#  C) off-279 偏離名單 = 基線殘留變更（condition_proj 等）
import sys, os, json, mmap, struct

REPO = '/Users/cyliu99/AI/h3c_mac_20261125/h3c'
sys.path.insert(0, os.path.join(REPO, 'scripts'))
import asset_audit as A

ENG = '/Volumes/SSD2T/models/MiniMax-H3-h3c-engine/convrot_engine_patched.safetensors'
RELA = '/Volumes/SSD2T/h3c_evidence_20261126/hf_stage_official/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors'
HFD = '/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer'
MAN = os.path.join(REPO, 'scratch/sol/engine_backup_prepatch/manifest.json')
OUT = os.path.join(REPO, 'scratch/sol/release_disambig_verdict.txt')
EXPECT_SHA = '9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9'

def hdr_base(path):
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop('__metadata__', None)
    return 8 + n, hdr

def main():
    lines, verdict_bits = [], []
    if not os.path.exists(RELA):
        print('STAGED_MISSING', RELA); return
    sha = A.stream_sha256(RELA)
    lines.append('release_sha256=%s expect=%s' % (sha, EXPECT_SHA))
    if sha != EXPECT_SHA:
        lines.append('VERDICT=FAIL(release sha mismatch)')
        open(OUT, 'w').write('\n'.join(lines) + '\n'); print('\n'.join(lines)); return
    lines.append('RELEASE_SHA_OK')
    eb, eh = hdr_base(ENG); rb, rh = hdr_base(RELA)
    hidx = A.build_hf_index(HFD)
    man = json.load(open(MAN))
    regs = [(e['offset'], e['offset'] + e['len']) for e in man['entries']]
    def in_patch(t0, t1):  # absolute file offsets overlap
        return any(not (t1 <= s or t0 >= e) for s, e in regs)
    fm = open(ENG, 'rb'); fsm = open(RELA, 'rb')
    emm = mmap.mmap(fm.fileno(), 0, access=mmap.ACCESS_READ)
    rmm = mmap.mmap(fsm.fileno(), 0, access=mmap.ACCESS_READ)
    common = sorted(set(eh) & set(rh))
    stats = {'in_patch': [0, 0], 'off_patch': [0, 0]}  # [identical, deviated]
    dev_off = []; dev_in = []
    for n in common:
        if eh[n]['dtype'] != rh[n]['dtype'] or eh[n]['shape'] != rh[n]['shape']:
            continue
        s1, e1 = eh[n]['data_offsets']; s2, e2 = rh[n]['data_offsets']
        a = emm[eb + s1: eb + e1]; b = rmm[rb + s2: rb + e2]
        ip = in_patch(eb + s1, eb + e1)
        k = 'in_patch' if ip else 'off_patch'
        if a == b: stats[k][0] += 1
        else:
            stats[k][1] += 1
            (dev_in if ip else dev_off).append(n)
    lines.append('engine_vs_release tensors_compared=%d in_patch279 identical=%d deviated=%d | off_patch identical=%d deviated=%d' % (
        sum(sum(v) for v in stats.values()), stats['in_patch'][0], stats['in_patch'][1], stats['off_patch'][0], stats['off_patch'][1]))
    lines.append('deviated_off_patch_names=%s' % (dev_off[:50]))
    lines.append('deviated_in_patch_names(count)=%d' % len(dev_in))
    # release vs MiniMaxAI on: 6 dtype-compatible non-patch + 12 patched sample
    comp_names = [n for n in common if n in hidx and rh[n]['dtype'] == hidx[n][1][n]['dtype'] and rh[n]['shape'] == hidx[n][1][n]['shape']]
    pmnames = [e['name'] for e in man['entries']]
    p12 = [n for n in pmnames if 'qkv' in n][:6] + [n for n in pmnames if 'qkv' not in n][:6]
    r2m = {'nonpatch': [0, 0], 'patch12': [0, 0]}
    for label, group in (('nonpatch', [n for n in comp_names if n not in set(pmnames)]), ('patch12', [n for n in p12 if n in rh and n in hidx])):
        for n in group:
            p, hh = hidx[n]
            s, e = rh[n]['data_offsets']; t1 = rmm[rb + s: rb + e]
            t2 = A.tensor_bytes(p, hh, n)
            r2m[label][0 if t1 == t2 else 1] += 1
    lines.append('release_vs_miniMaxAI nonpatch identical=%d deviated=%d | patch12 identical=%d deviated=%d' % (
        r2m['nonpatch'][0], r2m['nonpatch'][1], r2m['patch12'][0], r2m['patch12'][1]))
    # disambiguation
    if stats['in_patch'][1] == 0:
        lines.append('CLASS=A: engine==release on all 279 regions -> patch faithful to release; audit MiniMaxAI reference = wrong-canonical (tool wiring fix candidate)')
    elif r2m['patch12'][0] == len([n for n in p12 if n in rh and n in hidx]) and len(dev_in) > 0:
        lines.append('CLASS=B: release==MiniMaxAI on patched samples but engine deviates from release -> patch NOT faithful to any canonical = REAL patch defect candidate')
    else:
        lines.append('CLASS=MIXED: release deviates from MiniMaxAI on patched samples -> release pass-through != original (roundtrip); engine reference contract = release file only')
    lines.append('DISAMBIG_DONE')
    txt = '\n'.join(lines) + '\n'
    open(OUT, 'w').write(txt); print(txt)
    emm.close(); rmm.close(); fm.close(); fsm.close()

if __name__ == '__main__':
    main()
