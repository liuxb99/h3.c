#!/usr/bin/env python3
"""audit_negative_tests.py — fail-closed negative test harness for
scripts/asset_audit.py + scripts/verify_ready.sh.

Sandbox: scratch/sol/audit_neg (wiped + rebuilt at every start; idempotent).
For each scenario we build tiny fake safetensors files + a sandbox manifest dir,
invoke the audit tool / load gate as a subprocess, and assert BOTH the exit code
AND an expected substring in the captured output.

Verdicts:
  PASS = tool behaved as expected (right exit code AND expected evidence line).
  GAP  = tool exited 0 on something it SHOULD have caught ("GAP: not caught").
         Reported only — the tool itself is never patched here.
  FAIL = anything else (wrong exit code, right code but missing evidence,
         control scenario falsely rejected / false positive).

Exit 0 iff every scenario behaved as expected (GAP counts as unexpected).
"""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
AUDIT = os.path.join(HERE, "asset_audit.py")
VERIFY = os.path.join(HERE, "verify_ready.sh")
SANDBOX = os.path.join(REPO, "scratch", "sol", "audit_neg")

DTYPE_ESZ = {"BF16": 2, "F16": 2, "F32": 4, "I64": 8, "U8": 1, "I32": 4, "BOOL": 1}


# ---- tiny fake safetensors builder ------------------------------------------
def fake_bytes(n, seed):
    return bytes(((seed * 101 + (i % 251) * 17 + (i >> 5)) & 0xFF) for i in range(n))


def st_blob(tensors, seed=1):
    """tensors: ordered list of (name, dtype, [rows, cols]). Returns file bytes."""
    hdr, datas, off = {}, [], 0
    for k, (name, dtype, shape) in enumerate(tensors):
        n = DTYPE_ESZ[dtype] * shape[0] * shape[1]
        hdr[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + n]}
        off += n
        datas.append(fake_bytes(n, seed + k))
    hj = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    hj += b" " * ((-len(hj)) % 8)          # pad so data starts 8-byte aligned
    return struct.pack("<Q", len(hj)) + hj + b"".join(datas)


def write_st(path, tensors, seed=1):
    blob = st_blob(tensors, seed)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(blob)
    return len(blob), hashlib.sha256(blob).hexdigest(), blob


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def scen_dir(sid):
    p = os.path.join(SANDBOX, sid)
    shutil.rmtree(p, ignore_errors=True)
    os.makedirs(p)
    return p


def write_manifest(mdir, fname, assets):
    ensure_dir(mdir)
    with open(os.path.join(mdir, fname), "w") as f:
        json.dump({"assets": assets}, f, indent=1)


def entry(name, path, size, sha, layout="layout_v1", read_only=False, header_contract=None):
    a = {"name": name, "path": path, "size_bytes": size, "expected_sha256": sha,
         "layout_contract": layout, "read_only": read_only}
    if header_contract:
        a["header_contract"] = header_contract
    return a


# ---- subprocess runners ------------------------------------------------------
def run_cmd(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None, "TIMEOUT"
    out = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
    return r.returncode, out


def run_audit(mdir, single=None):
    cmd = [sys.executable, AUDIT, "--manifest-dir", mdir]
    if single:
        cmd += ["--single", single]
    return run_cmd(cmd)


def run_verify(path, mdir):
    return run_cmd(["bash", VERIFY, path, mdir])


# ---- scenario builders -------------------------------------------------------
def s0_control_clean():
    sd = scen_dir("S0")
    ad, md = ensure_dir(sd + "/assets"), ensure_dir(sd + "/manifests")
    p = ad + "/clean_model.safetensors"
    size, sha, _ = write_st(p, [("t_a", "BF16", [4, 8]), ("t_b", "F32", [8, 8])], seed=1)
    write_manifest(md, "m0.json", [
        entry("clean", p, size, sha, layout="contract_v1", read_only=False,
              header_contract={"t_a": {"shape": [4, 8], "dtype": "BF16"}})])
    open(p + ".ready", "w").close()
    return [
        {"label": "audit_full", "run": lambda: run_audit(md), "want_rc": 0,
         "subs": ["AUDIT_PASS"]},
        {"label": "verify_ready", "run": lambda: run_verify(p, md), "want_rc": 0,
         "subs": ["LOAD_READY"]},
    ]


def s1_bad_symlink():
    sd = scen_dir("S1")
    ad, md = ensure_dir(sd + "/assets"), ensure_dir(sd + "/manifests")
    tensors = [("t_a", "BF16", [4, 8]), ("t_b", "F32", [8, 8])]
    intended = st_blob(tensors, seed=1)                       # content manifest claims
    wrong = st_blob(tensors, seed=9)                          # same size, wrong bytes
    wpath = ad + "/wrong_target.bin"
    with open(wpath, "wb") as f:
        f.write(wrong)
    link = ad + "/symlink_model.safetensors"
    os.symlink(wpath, link)
    write_manifest(md, "m1.json", [
        entry("badlink", link, len(intended), hashlib.sha256(intended).hexdigest(),
              layout="contract_v1")])
    return [
        {"label": "audit_full", "run": lambda: run_audit(md), "want_rc": 2,
         "subs": ["AUDIT_FAIL", "(CONTAMINATION)"]},
    ]


def _hardlink_scenario(sid, layout_a, layout_b, expect_rc, subs):
    sd = scen_dir(sid)
    ad, md = ensure_dir(sd + "/assets"), ensure_dir(sd + "/manifests")
    p1 = ad + "/engine_a.safetensors"
    size, sha, _ = write_st(p1, [("t_a", "BF16", [4, 8])], seed=3)
    p2 = ad + "/engine_b.safetensors"
    os.link(p1, p2)                                           # shared inode
    write_manifest(md, "m_a.json", [entry("engine_a", p1, size, sha, layout=layout_a)])
    write_manifest(md, "m_b.json", [entry("engine_b", p2, size, sha, layout=layout_b)])
    return [
        {"label": "audit_full", "run": lambda: run_audit(md), "want_rc": expect_rc,
         "subs": subs},
    ]


def s2_hardlink_cross_contract():
    return _hardlink_scenario("S2", "contract_fasth3", "contract_h3c", 2,
                              ["AUDIT_FAIL", "different layout contracts"])


def s2b_hardlink_same_contract_control():
    return _hardlink_scenario("S2b", "contract_shared", "contract_shared", 0,
                              ["AUDIT_PASS", "shared (same contract)"])


def s3_sha_mismatch():
    sd = scen_dir("S3")
    ad, md = ensure_dir(sd + "/assets"), ensure_dir(sd + "/manifests")
    p = ad + "/model.safetensors"
    size, _sha, _ = write_st(p, [("t_a", "BF16", [4, 8])], seed=4)
    write_manifest(md, "m3.json", [
        entry("wrongsha", p, size, "f" * 64, layout="contract_v1")])  # wrong expected sha
    return [
        {"label": "audit_full", "run": lambda: run_audit(md), "want_rc": 2,
         "subs": ["AUDIT_FAIL", "(CONTAMINATION)"]},
    ]


def s4_qkv_layout_incompat():
    sd = scen_dir("S4")
    ad, md = ensure_dir(sd + "/assets"), ensure_dir(sd + "/manifests")
    qkv = "model.diffusion_blocks.0.attn.qkv_weight.weight"
    p = ad + "/model.safetensors"
    size, sha, _ = write_st(p, [(qkv, "U8", [1536, 3072]), ("other.tensor", "U8", [8, 8])],
                            seed=6)                           # file header: [1536, 3072]
    write_manifest(md, "m4.json", [
        entry("qkvw", p, size, sha, layout="contract_v1",
              header_contract={qkv: {"shape": [3072, 1536], "dtype": "U8"}})])
    return [
        {"label": "audit_full", "run": lambda: run_audit(md), "want_rc": 2,
         "subs": ["AUDIT_FAIL", "LAYOUT-INCOMPATIBLE"]},
    ]


def s5_readonly_mutation():
    sd = scen_dir("S5")
    mad = ensure_dir(sd + "/assets")
    madir, mbdir = ensure_dir(sd + "/manifests_a"), ensure_dir(sd + "/manifests_b")
    # (a) chmod 444 only, no uchg flag, freeze_pending absent
    pa = mad + "/s5a.bin"
    size_a, sha_a, _ = write_st(pa, [("t_a", "BF16", [4, 8])], seed=7)
    os.chmod(pa, 0o444)
    write_manifest(madir, "m5a.json",
                   [entry("ro_a", pa, size_a, sha_a, layout="contract_v1", read_only=True)])
    # (b) mutate bytes of a read_only asset (chmod+w, append 1 byte, chmod back)
    pb = mad + "/s5b.bin"
    size_b, sha_b, _ = write_st(pb, [("t_a", "BF16", [4, 8])], seed=8)
    os.chmod(pb, 0o644)
    with open(pb, "ab") as f:
        f.write(b"\x00")
    os.chmod(pb, 0o444)
    write_manifest(mbdir, "m5b.json",
                   [entry("ro_b", pb, size_b, sha_b, layout="contract_v1", read_only=True)])
    return [
        {"label": "a_readonly_chmod444_no_uchg", "run": lambda: run_audit(madir),
         "want_rc": 2, "subs": ["AUDIT_FAIL", "read_only declared but writable"]},
        {"label": "b_readonly_mutated_bytes", "run": lambda: run_audit(mbdir),
         "want_rc": 2, "subs": ["AUDIT_FAIL", "(MUTATION)", "(CONTAMINATION)"]},
    ]


def s6_incomplete_copy():
    sd = scen_dir("S6")
    ad = ensure_dir(sd + "/assets")
    madir, mbdir = ensure_dir(sd + "/manifests_a"), ensure_dir(sd + "/manifests_b")
    # (a) truncated copy: size != manifest size_bytes
    pa = ad + "/truncated.bin"
    size_a, sha_a, blob_a = write_st(pa, [("t_a", "BF16", [4, 8]), ("t_b", "F32", [8, 8])], seed=10)
    with open(pa, "r+b") as f:
        f.truncate(len(blob_a) // 2)
    write_manifest(madir, "m6a.json", [entry("trunc", pa, size_a, sha_a, layout="contract_v1")])
    # (b) complete+verified but no .ready marker
    pb = ad + "/complete.bin"
    size_b, sha_b, _ = write_st(pb, [("t_a", "BF16", [4, 8])], seed=11)
    write_manifest(mbdir, "m6b.json", [entry("complete", pb, size_b, sha_b, layout="contract_v1")])
    return [
        {"label": "a_truncated_audit", "run": lambda: run_audit(madir), "want_rc": 2,
         "subs": ["AUDIT_FAIL", "(MUTATION)"]},
        {"label": "b_audit_single_ok", "run": lambda: run_audit(mbdir, single=pb),
         "want_rc": 0, "subs": ["AUDIT_PASS"]},
        {"label": "b_verify_no_marker", "run": lambda: run_verify(pb, mbdir),
         "want_rc": 2, "subs": ["LOAD_BLOCKED", "missing ready marker"]},
    ]


def s7_single_not_in_manifest():
    sd = scen_dir("S7")
    ad, md = ensure_dir(sd + "/assets"), ensure_dir(sd + "/manifests")
    pc = ad + "/claimed.bin"
    size_c, sha_c, _ = write_st(pc, [("t_a", "BF16", [4, 8])], seed=12)
    write_manifest(md, "m7.json", [entry("claimed", pc, size_c, sha_c, layout="contract_v1")])
    extra = ad + "/not_in_manifest.bin"
    write_st(extra, [("t_a", "BF16", [4, 8])], seed=13)
    return [
        {"label": "audit_single_unclaimed", "run": lambda: run_audit(md, single=extra),
         "want_rc": 2, "subs": ["AUDIT_FAIL", "not found in any manifest", "LOAD_BLOCKED"]},
    ]



def s8_expected_deviation():
    sd = scen_dir("S8")
    ad, cd = ensure_dir(sd + "/engine"), ensure_dir(sd + "/canonical")
    pc = cd + "/canon.safetensors"
    write_st(pc, [("t_a", "BF16", [4, 8])], seed=20)
    pe = ad + "/patched.safetensors"          # same header, different bytes
    size_e, sha_e, _ = write_st(pe, [("t_a", "BF16", [4, 8])], seed=21)
    pm = sd + "/pm.json"
    json.dump({"entries": [{"name": "t_a"}]}, open(pm, "w"))
    md_a, md_b = ensure_dir(sd + "/manifests_a"), ensure_dir(sd + "/manifests_b")
    def entry_with_flags():
        a = entry("patched", pe, size_e, sha_e, layout="contract_v1")
        a["passthrough_manifest"] = pm
        a["hf_source_dir"] = cd
        return a
    write_manifest(md_a, "m8a.json", [entry_with_flags()])                     # undeclared
    a2 = entry_with_flags()
    a2["passthrough_expected_deviation"] = "patched engine vs legacy release: BY DESIGN"
    write_manifest(md_b, "m8b.json", [a2])                                     # declared
    return [
        {"label": "a_deviation_undeclared", "run": lambda: run_audit(md_a), "want_rc": 2,
         "subs": ["AUDIT_FAIL", "NOT byte-identical", "deviate from HF"]},
        {"label": "b_deviation_declared", "run": lambda: run_audit(md_b), "want_rc": 0,
         "subs": ["AUDIT_PASS", "BY DESIGN"]},
    ]


def s9_build_pending():
    sd = scen_dir("S9")
    ensure_dir(sd + "/assets")
    md_a, md_b = ensure_dir(sd + "/manifests_a"), ensure_dir(sd + "/manifests_b")
    missing = sd + "/assets/future_model.safetensors"
    pa = entry("pending_asset", missing, 1234, "PENDING_bld", layout="contract_v1")
    pa["build_pending"] = True
    write_manifest(md_a, "m9a.json", [pa])
    write_manifest(md_b, "m9b.json", [entry("claimed_present", missing, 1234,
                                            "PENDING_x", layout="contract_v1")])
    return [
        {"label": "a_missing_declared_pending", "run": lambda: run_audit(md_a), "want_rc": 0,
         "subs": ["AUDIT_PASS", "build_pending"]},
        {"label": "b_missing_undeclared", "run": lambda: run_audit(md_b), "want_rc": 2,
         "subs": ["AUDIT_FAIL", "missing"]},
    ]


# ---- runner -------------------------------------------------------------------
SCENARIOS = [
    ("S0_control_clean", s0_control_clean),
    ("S1_bad_symlink", s1_bad_symlink),
    ("S2_hardlink_cross_contract", s2_hardlink_cross_contract),
    ("S2b_hardlink_same_contract_control", s2b_hardlink_same_contract_control),
    ("S3_sha_mismatch", s3_sha_mismatch),
    ("S4_qkv_layout_incompat", s4_qkv_layout_incompat),
    ("S5_readonly_mutation", s5_readonly_mutation),
    ("S6_incomplete_copy", s6_incomplete_copy),
    ("S7_single_not_in_manifest", s7_single_not_in_manifest),
    ("S8_expected_deviation", s8_expected_deviation),
    ("S9_build_pending", s9_build_pending),
]


def main():
    shutil.rmtree(SANDBOX, ignore_errors=True)
    os.makedirs(SANDBOX, exist_ok=True)
    rows, gap_reports, all_as_expected = [], [], True

    for sid, builder in SCENARIOS:
        try:
            checks = builder()
        except Exception as e:
            rows.append((sid, "FAIL", f"build error: {e!r}"))
            all_as_expected = False
            continue
        statuses, details = [], []
        for c in checks:
            rc, out = c["run"]()
            missing = [s for s in c["subs"] if s not in out]
            if rc == c["want_rc"] and not missing:
                statuses.append("ok")
                details.append(f"{c['label']}: rc={rc} evidence_ok")
            elif c["want_rc"] == 2 and rc == 0:
                statuses.append("gap")
                details.append(f"{c['label']}: GAP: not caught (rc=0 want=2, "
                               f"missing={','.join(missing)})")
                gap_reports.append((sid, c["label"], out))
            else:
                statuses.append("fail")
                details.append(f"{c['label']}: rc={rc} want={c['want_rc']}"
                               + (f" missing={','.join(missing)}" if missing else ""))
        verdict = ("GAP" if "gap" in statuses else
                   "FAIL" if "fail" in statuses else "PASS")
        if verdict != "PASS":
            all_as_expected = False
        rows.append((sid, verdict, "; ".join(details)))

    print(f"sandbox={SANDBOX}")
    print(f"{'scenario':<36} verdict  detail")
    for sid, verdict, detail in rows:
        print(f"{sid:<36} verdict={verdict:<4} {detail}")
    if gap_reports:
        print("--- GAP_DETAILS (exact tool stdout, tool NOT patched) ---")
        for sid, label, out in gap_reports:
            print(f"[{sid} / {label}]")
            for line in out.splitlines():
                print(f"  {line}")
    print("NEG_TESTS_PASS" if all_as_expected else "NEG_TESTS_FINAL_FAIL")
    sys.exit(0 if all_as_expected else 1)


if __name__ == "__main__":
    main()
