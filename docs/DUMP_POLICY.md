# H3C Tap-Dump Retention Policy (user-approved 2026-11-27)

## Why
2026-11-27 SSD2T (1.8T) hit 100% full: four full-set GPU tap-dump dirs accumulated
264G of raw per-tensor binary dumps after their verdicts were already archived in
P3C_STATUS.md. Full dumps must stop being the default.

## Rules
1. NO full-set dumps by default. Dumps are layered/segmented/selective:
   - Stage 1 (screen): stats-only taps — log relrms/maxabs/nan/inf per tap, write NO binary.
   - Stage 2 (locate): dump binaries ONLY for the first-diverging tap + its direct
     inputs/outputs (≤3 tensors per incident).
   - Stage 3 (deep-dive): full-set dumps only for tiny (64x64) runs, never 640+,
     deleted after the incident report lands.
2. Prefer streaming comparison: harness computes per-tap stats vs reference on the fly;
   persist only stats, the first-divergence tensor, and minimal raw evidence.
3. Evidence that is ALWAYS kept: run_log.txt, metrics.txt, manifest, SHA256, mp4,
   png, .md reports, csv. Raw dumps are disposable once verdicts are recorded.
4. Disk gate: if df usage of the evidence volume >= 85%, STOP creating new large dumps;
   run a safe cleanup pass (delete raw dumps whose verdict is archived) before resuming.
   Harness scripts check df and refuse/switch to stats-only mode.
5. Cleanup requires an explicit approve naming the target dirs; deletes always keep the
   analysis artifacts (rule 3) and never touch models/AI/ or in-use tap dirs.

## Retention tiers (evidence dirs)
- bench runs (mp4+logs+metrics): keep 60 days rolling
- tiny-run selective dumps: delete after incident report commit
- one-off full-set 640 dumps: delete same-day as verdict, keep 1 representative tap binary
- baselines pinned for regression: keep, but stats-only where possible
