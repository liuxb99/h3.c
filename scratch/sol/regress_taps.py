#!/usr/bin/env python3
# Run with: uv run --with numpy regress_taps.py   (numpy not in base env)
"""P3 gate: tap-dump sanity. FAIL (exit!=0) if any compute tap contains
NaN/Inf or if the manifest is missing/empty — a kernel that silently
skipped its dispatch (unwritten buffer = stale garbage / NaN) must never
be scored as a successful generation run.

Usage: regress_taps.py <dump_dir> [--allow <glob> ...]
  --allow: tap name globs permitted to hold non-finite values
  (e.g. native I8 weight readbacks interpreted as bf16).
"""
import glob, os, sys, fnmatch
import numpy as np

def parse(name):
    if '_bf16' in name: return np.uint16, True
    if '_f32' in name: return '<f4', False
    if '_u32' in name: return '<u4', None
    if '_i8' in name or '_w.raw' in name and False: return np.int8, None
    return None, None

def main():
    d = sys.argv[1]
    allow = []
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == '--allow':
            i += 1
            while i < len(args) and not args[i].startswith('--'):
                allow.append(args[i]); i += 1
        else: i += 1
    manifest = None
    for cand in ('manifest.txt', 'manifest.tsv'):
        if os.path.exists(os.path.join(d, cand)): manifest = cand; break
    taps = sorted(glob.glob(os.path.join(d, '*_bf16.raw')) +
                  glob.glob(os.path.join(d, '*_f32.raw')))
    if not manifest and not taps:
        print(f"FAIL: no manifest and no compute taps in {d} "
              f"(generation produced no evidence = silent failure)"); return 2
    offenders, checked = [], 0
    for t in taps:
        n = os.path.basename(t)
        if any(fnmatch.fnmatch(n, p) for p in allow): continue
        dt, isbf = parse(n)
        a = np.fromfile(t, dtype=dt)
        if a.size == 0:
            offenders.append((n, 'EMPTY')); continue
        if isbf:
            v = (a.astype(np.uint32) << 16).view(np.float32)
        else:
            v = a
        bad = int(np.isnan(v).sum()) + int(np.isinf(v).sum())
        checked += 1
        if bad:
            row = bad // 5376 + 1
            offenders.append((n, f"non-finite={bad} (~rows {row})"))
    if offenders:
        print(f"FAIL: {len(offenders)}/{checked} taps non-finite in {d}")
        for n, why in offenders[:12]: print(f"  BAD {n}: {why}")
        if len(offenders) > 12: print(f"  ... {len(offenders)-12} more")
        return 1
    print(f"PASS: {checked} taps finite, manifest={'yes' if manifest else 'no'} ({d})")
    return 0

if __name__ == '__main__':
    sys.exit(main())
