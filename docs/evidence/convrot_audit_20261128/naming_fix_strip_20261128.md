# Sidecar 命名慣例修正（2026-11-28 14:1x CST）

## FACT（實測，`h3c/scratch/sol/sidecar_keys_probe.py`，僅讀 header）
| 資產 | scale 鍵 | comfy 鍵 | 判定 |
|---|---|---|---|
| official comfy-int8 | `X.weight_scale`（strip，250/250） | `X.comfy_quant`（250/250） | 生產載入慣例 |
| h3c-engine `convrot_engine_patched.safetensors` | 同 official（250/250） | 同 official（250/250） | 引擎載入器吃 strip |
| h3c-built（改名前） | `X.weight.weight_scale`（append，250/250） | `X.weight.comfy_quant` | 引擎載入器 250/250 找不到 scale |

- 另有 `scale_ratio_official_vs_built.py` 實測：official 與 built 的 scale **數值全等**（ratio=1.000000）
  → 13:11 chain 那 162 個 `R [Q] relrms≈0.94` 失敗不是 scale 漂移，是 gate 以錯誤鍵名解析 official
  （00:34 舊 pyc + append 查找雙重錯位）所致。INFERENCE：pairing 錯位路徑未经逐案復現，屬推論。

## 根因（wrapper exit 1 類問題，永久修復）
- `st_io.st_close` 在 numpy 視圖仍掛住 mmap 時 `mm.close()` → `BufferError: cannot close
  exported pointers exist` → 判定寫出後進程仍 rc≠0（chain gate 0.5s rc1 與舊 wrapper
  "GATE_EXIT=0 但外層 exit 1" 同族）。chain 日誌尾端 BufferError traceback 為直接證據。
- 修復（雙層）：
  1. `qkv_layout_gate.py codes_i8`：`np.frombuffer(...).copy()`（來源端不再洩漏掛住 mmap 的視圖）
  2. `st_io.st_close`：`mm.close()` 包 `try/except BufferError`（收尾端永不炸）
- 驗證：修復後 gate 煙霧（3 subjects）`EXIT=0` 乾淨收尾（`scratch/sol/gate_patch_smoke.log`）。

## 動作（皆未動 Strata、未動 chain 正在讀的檔案）
1. `qkv_layout_gate.py`：新增 `sidecar_key()` 雙慣例解析（strip 優先、append 相容）套用到
   `scale_f32` / `groupsize`；`[P]` census 的 comfy→subj 規則改為拼法自適應。
2. `build_int8_convrot.py`：輸出鍵改回 strip（`name+"_scale"`、`name[:-7]+".comfy_quant"`）。
3. `rename_built_sidecar_strip.py`：built 資產 **就地改名 500 鍵** 為 strip——只重寫 header JSON，
   以空格 padding 保持 header 長度與 data 起始位移完全不變 → payload 位元組與逐張量
   `data_offsets` 對照改名前備份逐筆斷言相同；重跑冪等 `VERIFY OK`（RC=0）。
   - 改名前 header 備份：`evidence/convrot_audit_20261128/built_header_before.json`
   - 檔案凍結還原 444。
4. 滿版 gate 重跑（strip 版候選 vs HF golden）+ `finalize_strip_manifest.py` 閉環
   （assert 0 failures → 全新檔 sha256 → manifest 記 `sidecar_form`/`sidecar_rename`/
   `gate_rerun4_strip`，保留 `pre_strip_rename_sha256`）——後台執行中，完成通知。

## 影響評估
- rerun3 / oracle 的 PASS 語義**不被改名作廢**：codes、scale、payload 位移與位元組未變，
  改的只是側車鍵名；gate4 重跑將以生產命名重新背書。
- 下一步（待 gate4 PASS）：以 strip 版 built 餵引擎載入器做 GPU smoke（須 /admit 閘門）。
