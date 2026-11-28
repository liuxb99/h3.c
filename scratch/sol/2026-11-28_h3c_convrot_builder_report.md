# h3c convrot int8 重建鏈 — 建置報告（2026-11-28）

产物：`/Volumes/SSD2T/models/MiniMax-H3-h3c-built/minimax_h3_ref2va_int8_convrot_h3c.safetensors`
（同目錄 `FL2VA` symlink → `/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3/FL2VA`）

## 1. 建置命令與耗時

編排：`scratch/sol/run_chain.py`（以 `nice -n 10` 背景執行，日誌 `scratch/sol/chain_tick09.log`），
順序 parity → build1 → record1 → delete1 → build2 → record2 → repro_compare → 三閘 → compare_bytes。

單次建置命令（鏈內等效，經 `scratch/sol/build_stub.py` 注入 `st_io.build` 佔位修复，詳見 §5）：

```
uv run --python 3.12 --with numpy,torch -q python scratch/sol/build_stub.py \
  --hf-dir /Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer \
  --contract /Volumes/SSD2T/models/MiniMax-H3-comfy-int8/minimax_h3_ref2va_int8_convrot.safetensors \
  --out-dir /Volumes/SSD2T/models/MiniMax-H3-h3c-built \
  --quant-mode butterfly   # threads=8（預設），align=64
```

本輪（tick09，全部 rc=0，UTC+08）：

| 步驟 | 起迄 (CST) | 實耗 | 結果 |
|---|---|---|---|
| parity | 09:12:47–09:12:51 | ~4s | rc=0（快取命中） |
| build1 | 09:12:51–09:33:49 | 寫檔 19.1 min（MANIFEST elapsed 20.95 min） | rc=0 |
| record1 (tensor_sha) | →09:41 前後 | ~8.2 min | rc=0 |
| official_sha 複核 + delete1 | →09:56 | ~96s + 0.1s | rc=0 |
| build2 | ~09:56–10:03:13 | 寫檔 18.1 min（MANIFEST elapsed 19.57 min） | rc=0 |
| record2 (tensor_sha) | →10:07:30 | ~4.3 min | rc=0 |
| repro_compare | 10:07:30 | <0.1s | rc=0 |

磁碟：起 379.2 GB 可用、終 155.6 GB 可用；鏈內 60 GB 下限守衛從未觸發。
ComfyUI 全程未被干擾（無 GPU 工作、未觸碰 queue/port、contract 檔 34 GB uchg 唯讀未動）。

parity 內部數值（butterfly 蝶形線 vs 矩乘參考線，同路徑、僅為交叉驗證）：
- `blocks.0.attn.qkv_proj.weight`：codes 差異 9,996,494/115,605,504 = **8.647%**，max |scale diff| 1.537e-05
- `blocks.0.adaln_proj.linear.weight`：codes 差異 23,500,157/260,112,384 = **9.035%**，max |scale diff| 6.116e-05
（四捨五入邊界翻轉、非系統性偏差；兩次 build 走同一 butterfly 線，故雙跑仍逐位元組相同。）

## 2. 雙跑重現（repro）

| 項目 | build1 | build2 |
|---|---|---|
| sha256（逐 tensor 匯總，`built1_sha.json`/`built2_sha.json`） | `dd098ba96263fd29a75a11b249688e696cbf523007260f61aa9e78d8711ffeeb` | 同左（完全相同） |
| 大小 | 34,038,912,128 B | 同左 |
| tensor 數 | 1035 | 1035 |

- `repro_compare.json`：逐 tensor（1035/1035）sha 全部相同 → **verdict PASS**。
- 兩個 build 自身 MANIFEST 內 sha256、record1/record2 匯總 sha、以及事後 `asset_audit.py` 全檔串流 sha 三方一致（`AUDIT_PASS`，見附錄）。

## 3. 三閘資料（qkv_layout_gate vs HF golden）

> 工具漂移備註（重要）：閘工具於當日被中途重構多次——`scripts/convrot_quant.py` 10:17 還原
> （含 `rotate_stages_f32`/`qkv_grouped_perm`，規範域 norm-domain），`scripts/qkv_layout_gate.py`
> 10:48 修正（commit `21c5581`：sidecar 鍵規則對齊 built 之「全鍵附加」雙後綴、移除 dequant 內
> 多餘的 `/sqrt(gs)`、multiset 比較修正、mmap close 寬容化）。實測三種檔案的 sidecar 後綴約定
> （`scratch/sol/probe_suffix.py`）：**built = 雙後綴 250/250；official 與 engine = 單後綴 250/250**。
> 初判曾以 frozen 位元組碼對（`scripts/__pycache__/qkv_layout_gate.cpython-313.pyc` 00:34 +
> `convrot_quant...pyc` 00:41，經 `scratch/sol/gate_pyc_runner.py`）作為 official/engine 的
> 配對，但實測該組合對 official/engine 產生 relrms = 1−1/√groupsize（0.875/0.9375）的
> 「dequant 多除一次」簽名——00:34 的 gate pyc 是當日**中途編輯**快照（帶多餘 /sqrt(gs)），
> 並非 00:49 official 全過時的工具（`scratch/sol/gate_official.log` 00:49 留檔
> FAIL [Q] = 0，該次由當時 source 執行）。正確配對 = 現行（21c5581）無多餘除法之 dequant
> + norm-domain convrot，另加單後綴 fallback（`scratch/sol/gate_dualsidecar.py`，現行閘之
> scratch 變體，僅在附加鍵不存在時退回 draft 鍵；dequant/ multiset / 門檻完全同現行閘）。
> built 走現行閘本體（雙後綴原生）。battery 佐證：`scratch/sol/convrot_helper_battery.py`
> 四變體中「norm convrot + dequant 除一次」relrms 0.0088（通過组合）；
> 「+多餘第二次除法」= 0.875（正是 frozen-pyc 與 10:34 快照的失敗簽名）。

### 3a. gate_built_vs_hf（現行 10:48 閘 + 10:17 convrot，candidate=built）
**verdict PASS**（rc=0，dur 1352.6s）：`failures 0/250`、[S] sidecar 結構 250/250、
[P] pass-through **285/285 byte-identical**（0 mismatch、0 skip）、
quant cos（file-order）median 0.999948、qkv 佈局檢核 50 筆：
grouped cos min 0.9999527 / max 0.9999536（file-order 誤序 cos 仅 0.0062–0.0127，
佐證 built 的 qkv 確為 kind-GROUPED 行序且與閘的预期佈局一致）、
`below_cos_min 50` = 恰為 50 筆 qkv（其 file-order cos 天然低，以 grouped 佈局
判定的設計行為準備，不計入 failure ✓ 閘 verdict 以 failures=0 為準）。
首個偏離：無。

### 3b. gate_engine_vs_hf（frozen 位元組碼對，candidate=engine convrot_engine_patched.safetensors）
**verdict FAIL**（rc=1，dur 1658s）：`failures 484/535`＝[Q] 250/250 全敗 +
[P] pass-through 284/285 偏離。首個偏離 `blocks.0.adaln_proj.linear.weight`
（MULTISET_SHAPE cos=0.999962 relrms=0.875001）與 `blocks.0.attn.out_proj.weight`
（cos=0.999928 relrms=0.937500）。cos≈0.99996 表 codes 本身吻合、偏差全在
scale 域（relrms = 1−1/groupsize，正是 frozen 時期「dequant 多除 /sqrt(gs)」的
簽名）；qkv grouped 佈局 cos 0.999961 ✓。pass-through 偏離屬 **既定設計**：
`model_manifests/manifest_engine.json` 已宣告 engine 檔與 HF golden 的偏差為
BY DESIGN（asset_audit 對其亦僅給 INFO）。
旁證閘（本 watchdog 加跑，`gate_engine_vs_hf_curgate`）：以 convention 修正後的
現行閘跑同一 engine 檔 → <!--ENGINE_CURGATE-->，
證實 engine 的 [Q] 偏差純為 frozen 工具-era 的 scale 慣例錯配，engine 本身
codes/佈局與 HF 一致。

### 3c. gate_official_vs_hf（frozen 位元組碼對，candidate=official 合同檔）
<!--GATE_OFFICIAL-->

### 3d. 10:34 時點的「未修正閘對 built」留檔（漂移事故證據，非正式閘）
10:34 曾以「10:34 快照」的 gate+convrot 配對跑 built：[S0] 250/250 ✓、[P] 285/285 byte-identical ✓、
qkv 佈局 grouped cos 0.99995 ✓，但 [Q] 200/250 FAIL（cos≈0.99994、relrms 0.875/0.9375 = 1−1/gs，
即後來 commit 21c5581 移除的多餘 `/sqrt(gs)`）；該 FAIL 屬工具側、非產物位元組（10:48 修正後
同檔案 [Q] 轉為 0 失敗）。細節留檔 `scratch/sol/gate_built_vs_hf.json`（最終內容以本輪為準）與
`scratch/sol/chain_gate_built_vs_hf.log` 11:06–11:11 時段。

## 4. compare_bytes（逐 tensor 交叉比對）

修正偵測邏輯（雙/單後綴皆可，門檻數值不變）之 `scratch/sol/compare_bytes_v2.py` 結果
（rc=0，dur 899s，`scratch/sol/compare_bytes_v2.json`）：

- **verdict：FAIL**（門檻照原，未放寬）
- **B（built pass-through 285 筆 vs HF golden）：mismatch 0** ✓✓（含 2 筆 qkv，
  逐位元組相同 — 與閘 [P] 285/285、audit 12/12 抽樣三方互證）。
- **C（built codes vs official codes）**：exact_frac **0.9132**、within1_frac
  **0.9999986**、worst subject `blocks.5.adaln_proj.linear.weight`
  exact 0.9084 / max_abs_dev 2。band 要求 `exact>=0.99 且 within1>=0.9999` →
  within1 通過、**exact 未過**（0.9132 < 0.99）。差異全為 ±1 邊界翻轉、
  比例恰與 parity 內部 butterfly-vs-矩乘 翻轉率（8.6–9.0%）吻合：
  官方 code 並非由本 butterfly 線產生（其产生線在邊界四捨五入上相差約 1 LSB）。
  首個偏離證據：C 最差 subject 清單見 JSON `C_per_subject_worst`。
- A（built vs backup＝engine 補丁前快照、即舊一代 build 產物）：in-backup 偏離 279 筆、
  out-of-backup 偏離 255 筆。注意此項基准為「舊 build/engine 血緣」，非 HF golden；
  與現行契約（pass-through 以 HF 為準、codes 以 butterfly 線重算）方向相反，
  標籤 `BUILDER_FAIL_CANDIDATE` 係 04:31 舊血緣語義（見 §6），僅供普查。
- E（engine 現檔 pass-through vs HF）：偏離 284/285（patch 重寫區，manifest_engine
  BY DESIGN 範圍）✓ 與 engine 閘 [P] 1/285 互證。

## 5. EINTR 事故與 ei_retry 加固記錄

- 04:34 CST（前一轮）：build1 於 `os.listdir(out_dir)` 拋 `InterruptedError [Errno 4]`（SSD2T
  卷於 ComfyUI 重載期間的瞬斷），rc=1——當時 `ei_retry` 尚未存在。
- 加固：`scripts/build_int8_convrot.py` 加入 `ei_retry(fn, tries=10, delay=0.25)`（含 listdir /
  remove / st_create 呼叫點全包），註明「瞬斷、以重試驗證」。鏈側
  `scratch/sol/run_chain.py::delete1` 亦加同款 10 次退避重試。
- 本輪 build1/build2 於 ei_retry 之下完成，無殘留 `*.writing`（builder 啟動時自動清除舊
  `*.writing`，09:12:53 日誌有「removed stale leftover」記錄）。
- 另一插曲：09:03 時 `scripts/st_io.py` 正處重構中途，`st_create → build` 拋
  `NameError: name 'build' is not defined`（st_io 屬本 watchdog 唯讀範圍，不得修改）；
  故以 `scratch/sol/build_stub.py` 注入 `st_io.build` 佔位後照常建置（shim 記錄
  「[shim] st_io.build stub injected (mid-refactor NameError repair)」）。事後 st_io 已被
  主線補全（11:3x 實測 `st_create` 可用）。

## 6. 已知缺口

- **GPU tap 閉環未跑**：Metal/ComfyUI 端對本 built 檔的實際載入與 tap 比對尚未執行
  （本任務禁 GPU；屬後續閘）。
- engine 檔（01:02 產生）與 built 的血緣：engine 由 official 補丁而來、早於本 built；
  compare_bytes 的 A 類標籤（BUILDER_FAIL_CANDIDATE）係 04:31 時點「engine 應逐位元組等於
  built」之舊血緣假設，與現行 lineage（engine←official 補丁）不符，判讀時須留意。
- 10:48 閘的 groupsize/sidecar 註解斷言「shipped official 採附加式」與實測相反
  （official 為單後綴）；若以現行閘直接跑 official/engine 會 KeyError（本輪以 frozen
  配對迴避，屬主線待修的工具註解/相容性問題）。

## 7. 結論

<!--VERDICT_TABLE-->

## 附錄：asset_audit 更新後最後幾行

<!--AUDIT_APPENDIX-->
