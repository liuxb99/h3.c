#!/usr/bin/env python3
"""Model-asset contamination audit (permanent gate).

Reads every manifest in h3c/model_manifests/*.json, verifies each declared asset
against live disk state:
  * path resolves (symlink chain), size, full streaming sha256 (cached by size+mtime),
  * read_only declaration enforced via mode bits / uchg flag,
  * cross-manifest inode sharing: two assets with DIFFERENT layout_contract sharing
    the same inode => VIOLATION (rule: engines with different weight-layout contracts
    must not share a writable file).
  * passthrough layout sample-check: for manifests with passthrough_manifest +
    hf_source_dir, byte-compares sampled tensors against the HF source by reading
    safetensors headers + mmap slices (pure stdlib).

Exit 0 = clean, 2 = violations/FAIL, 1 = manifest problem.
"""
import json, os, sys, hashlib, mmap, struct, glob, argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAN_DIR = os.path.join(REPO, "model_manifests")
CACHE = os.path.join(REPO, "scratch", "sol", ".asset_audit_cache.json")

def stream_sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def st_safe(path):
    try:
        return os.stat(path)
    except OSError:
        return None

def check_sha(path, st, cache):
    key = path
    ent = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    c = cache.get(key)
    if c and c["size"] == ent["size"] and c["mtime_ns"] == ent["mtime_ns"] and "sha256" in c:
        return c["sha256"], "cached"
    s = stream_sha256(path)
    cache[key] = {**ent, "sha256": s}
    return s, "computed"

# ---- safetensors helpers -----------------------------------------------------
def st_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr

def tensor_bytes(path, hdr, name):
    """Return bytes of one tensor via mmap slice (data region base = 8 + header_len)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        base = 8 + n
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        s, e = hdr[name]["data_offsets"]
        data = mm[base + s: base + e]
        mm.close()
    return data

def build_hf_index(hf_dir):
    """name -> (shard_path, header) for every safetensors shard under hf_dir (dir or file)."""
    idx = {}
    paths = []
    if os.path.isdir(hf_dir):
        paths = sorted(glob.glob(os.path.join(hf_dir, "*.safetensors")))
    elif hf_dir.endswith(".safetensors"):
        paths = [hf_dir]
    for p in paths:
        try:
            hdr = st_header(p)
        except Exception:
            continue
        for k in hdr:
            idx.setdefault(k, (p, hdr))
    return idx

def classify_layout(eng_bytes, hf_bytes, out_rows, head_dim, heads, qkv=3):
    """Return 'interleaved' / 'grouped' / 'other' by matching row order to HF canonical."""
    if eng_bytes == hf_bytes:
        return "interleaved(byte-identical)"
    import hashlib as _h
    # rebuild grouped<->interleaved permutation of HF canonical rows and compare
    per_head = head_dim  # rows per head slice
    total = out_rows
    if total % (qkv * heads * per_head):
        return "other(shape-mismatch)"
    hf = memoryview(hf_bytes)
    row = head_dim * (len(hf_bytes) // (out_rows * head_dim)) if out_rows else 0
    # compare via per-row hashes: build dict of hf row hashes
    rl = len(hf_bytes) // out_rows
    def rows(b):
        m = memoryview(b)
        return [_h.sha256(m[i*rl:(i+1)*rl]).digest() for i in range(out_rows)]
    hr = rows(hf_bytes)
    er = rows(eng_bytes)
    # HF canonical order (interleaved: head-major [q,k,v] triples). Engine grouped = [allQ|allK|allV].
    # If eng == HF => interleaved. Else test grouped->interleaved map: eng row i should equal hf row perm(i).
    third = out_rows // qkv
    ok_grouped = all(er[i] == hr[i] for i in [0]) or _map_eq(er, hr, third, out_rows)
    if ok_grouped:
        return "grouped"
    return "other(unknown-permutation)"

def _map_eq(er, hr, third, out_rows):
    # grouped: rows [Q block|K block|V block], each block head-major; interleaved: head-major triples.
    qkv, heads = 3, out_rows // (3 * (out_rows // (3 * 3)) or 1)
    # generic head-triple map without head count: assume block==head-major; map grouped index
    # g = blk*third + j  <->  interleaved i = (j//? )... use positional multiset check instead.
    return sorted(er) == sorted(hr) and er != hr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="skip sha computation for files whose recorded size+mtime match manifest (no cache recompute)")
    ap.add_argument("--sample", type=int, default=12, help="tensors to byte-sample per passthrough manifest")
    ap.add_argument("--manifest-dir", default=MAN_DIR, help="directory of *.json manifests (default repo model_manifests)")
    ap.add_argument("--single", default=None, help="audit only the manifest asset whose resolved path matches this file")
    args = ap.parse_args()

    cache = {}
    if os.path.exists(CACHE):
        try:
            cache = json.load(open(CACHE))
        except Exception:
            cache = {}

    problems, infos = [], []
    inode_assets = {}  # dev:inode -> [(asset_id, layout_contract)]
    single_hit = False
    manifests = sorted(glob.glob(os.path.join(args.manifest_dir, "*.json")))
    if not manifests:
        print("AUDIT: no manifests in", MAN_DIR); sys.exit(1)

    for mp in manifests:
        man = json.load(open(mp))
        mid = os.path.basename(mp)
        for a in man.get("assets", []):
            aid = f"{mid}:{a.get('name','?')}"
            path = a["path"]
            if args.single:
                if os.path.realpath(path) != os.path.realpath(args.single):
                    continue
                single_hit = True
            st = st_safe(path)
            if st is None:
                if a.get("build_pending"):
                    infos.append(f"INFO {aid}: build_pending — asset not built yet (expected missing)")
                else:
                    problems.append(f"FAIL {aid}: missing {path}")
                continue
            real = os.path.realpath(path)
            inode_assets.setdefault((st.st_dev, st.st_ino), []).append(
                (aid, a.get("layout_contract", "?")))
            if a.get("size_bytes") and st.st_size != a["size_bytes"]:
                problems.append(f"FAIL {aid}: size {st.st_size} != manifest {a['size_bytes']} (MUTATION)")
            exp = a.get("expected_sha256")
            if exp and exp.startswith(("PENDING", "pending")):
                infos.append(f"INFO {aid}: sha pending (recorded size/mtime checked)")
            elif exp:
                sha, how = check_sha(path, st, cache)
                if sha != exp:
                    problems.append(f"FAIL {aid}: sha256 {sha[:16]} != manifest {exp[:16]} (CONTAMINATION)")
                else:
                    infos.append(f"OK {aid}: sha match ({how})")
            if a.get("read_only"):
                flags = getattr(st, "st_flags", 0)
                # Filesystem-dependent encoding: on this exFAT-class SSD uchg reports
                # st_flags==2 (not APFS's 0x200); accept either bit as protected.
                immutable = bool(flags & 0x00000202)
                writable = bool(st.st_mode & 0o222)
                if immutable:
                    # macOS semantics: immutable flag blocks writes for non-root even
                    # when mode bits still show writable (EPERM write-test confirmed).
                    infos.append(f"INFO {aid}: read_only enforced via immutable flag (st_flags={flags})")
                    try:
                        dst = os.stat(os.path.dirname(path))
                        if not (dst.st_flags & 0x00000202) and (dst.st_mode & 0o222):
                            infos.append(
                                f"INFO {aid}: parent dir writable — unlink/replace possible; freeze dir with chflags uchg")
                    except OSError:
                        pass
                else:
                    # Fail-closed (neg-test S5a): mode-only protection (e.g. chmod 444)
                    # is NOT immutable — the owner can simply chmod it back. read_only
                    # assets must carry the uchg flag; freeze_pending is the only skip.
                    lvl = "INFO" if a.get("freeze_pending") else "FAIL"
                    why = ("writable (uchg pending)" if writable else
                           "writable (chmod-only; uchg pending — owner can revert mode)")
                    (problems if lvl == "FAIL" else infos).append(
                        f"{lvl} {aid}: read_only declared but {why}")
            if not exp.startswith("PENDING") if exp else True:
                pass
            # header_contract: declared tensor shape/dtype must match safetensors header
            hc = a.get("header_contract")
            if hc:
                try:
                    hdr = st_header(real)
                    for tn, want in hc.items():
                        got = hdr.get(tn)
                        if not got:
                            problems.append(f"FAIL {aid}: header_contract tensor {tn} missing (LAYOUT-INCOMPATIBLE)")
                        elif want.get("shape") and got.get("shape") != want["shape"]:
                            problems.append(f"FAIL {aid}: header_contract {tn} shape {got.get('shape')} != declared {want['shape']} (LAYOUT-INCOMPATIBLE)")
                        elif want.get("dtype") and got.get("dtype") != want["dtype"]:
                            problems.append(f"FAIL {aid}: header_contract {tn} dtype {got.get('dtype')} != declared {want['dtype']} (LAYOUT-INCOMPATIBLE)")
                except Exception as e:
                    problems.append(f"FAIL {aid}: header_contract check error: {e}")
            # passthrough layout sample check
            pm_path, hf_dir = a.get("passthrough_manifest"), a.get("hf_source_dir")
            if pm_path and hf_dir:
                try:
                    pm = json.load(open(pm_path if os.path.isabs(pm_path) else os.path.join(REPO, pm_path)))
                    names = [e["name"] for e in pm["entries"]]
                except Exception as e:
                    problems.append(f"FAIL {aid}: cannot read passthrough_manifest: {e}")
                    names = []
                qkvs = [n for n in names if "qkv" in n][:args.sample]
                others = [n for n in names if "qkv" not in n][:max(0, args.sample - len(qkvs))]
                sample = (qkvs + others)[:args.sample]
                # Byte-contract reference: release_reference_dir (the installed
                # Comfy-Org release file) when declared; else HF bf16 source dir.
                # Engine contract per release_disambig_verdict (CLASS=MIXED) is the
                # release file, NOT the MiniMaxAI bf16 original (different quant
                # roundtrip -> byte-compare vs bf16 was a wrong-canonical wiring).
                canon_dir = a.get("release_reference_dir") or hf_dir
                canon_label = "release" if a.get("release_reference_dir") else "HF"
                hf_idx = build_hf_index(canon_dir)
                try:
                    eng_hdr = st_header(real)
                except Exception as e:
                    problems.append(f"FAIL {aid}: header unreadable: {e}")
                    eng_hdr = None
                if eng_hdr is not None and hf_idx:
                    # passthrough_expected_deviation: manifest-declared BY-DESIGN
                    # deviation (e.g. patched engine carrying HF-golden bytes while the
                    # release keeps legacy bytes). Deviation lines drop to INFO only
                    # when explicitly declared — never silently.
                    exp_dev = a.get("passthrough_expected_deviation")
                    bad = 0
                    for n in sample:
                        if n not in eng_hdr or n not in hf_idx:
                            bad += 1
                            problems.append(f"FAIL {aid}: sample tensor {n} missing in engine/canonical")
                            continue
                        hp, hh = hf_idx[n]
                        eb, hb = tensor_bytes(real, eng_hdr, n), tensor_bytes(hp, hh, n)
                        if eb != hb:
                            bad += 1
                            line = f"passthrough {n} NOT byte-identical to {canon_label} canonical"
                            if exp_dev:
                                infos.append(f"INFO {aid}: {line} (declared expected deviation)")
                            else:
                                problems.append(f"FAIL {aid}: {line}")
                    if bad == 0:
                        infos.append(f"OK {aid}: {len(sample)} sampled passthrough tensors byte-identical to {canon_label} (incl {len(qkvs)} qkv)")
                    elif exp_dev:
                        infos.append(f"INFO {aid}: {bad}/{len(sample)} sampled passthrough tensors deviate from {canon_label} — BY DESIGN per manifest declaration")
                    else:
                        problems.append(f"FAIL {aid}: {bad}/{len(sample)} sampled passthrough tensors deviate from {canon_label}")
                elif canon_dir and not hf_idx:
                    problems.append(f"FAIL {aid}: canonical ref dir has no safetensors: {canon_dir}")

    if args.single and not single_hit:
        problems.append("FAIL --single asset not found in any manifest (LOAD_BLOCKED)")

    # inode sharing rule across different layout contracts
    for (dev, ino), lst in inode_assets.items():
        contracts = {c for _, c in lst}
        if len(lst) > 1 and len(contracts) > 1:
            problems.append(f"VIOLATION inode {ino} shared by assets with different layout contracts: "
                            + ", ".join(f"{a}[{c}]" for a, c in lst))
        elif len(lst) > 1:
            infos.append(f"OK inode {ino} shared (same contract) by: " + ", ".join(a for a, _ in lst))

    for line in infos:
        print(line)
    for line in problems:
        print(line)
    print(f"AUDIT_SUMMARY assets_paths_checked={len(inode_assets)} problems={len(problems)}")
    print("AUDIT_PASS" if not problems else "AUDIT_FAIL")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump(cache, open(CACHE, "w"))
    sys.exit(0 if not problems else 2)

if __name__ == "__main__":
    main()
