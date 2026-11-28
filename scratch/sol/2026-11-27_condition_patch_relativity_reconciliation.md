# 2026-11-27 條件嵌入器/_patch_卷積權重表 relRMS 0.23 / 0.27 對帳報告（read-only，未改任何 kernel）

## 0. 判定一句話

**relRMS≈0.23（condition_embedder）與 ≈0.27（patch conv）不是 kernel/dequant/旋轉/packing 的去相關錯誤，而是兩件事疊加：**
1. **轉換器（export 端）對「未走 int8 序列化」的 pass-through 張量做了有損 requant 往返**（未存 scale、未存 sidecar），condition/patch 落在 RMSrel 2–3.7% 的 requant 噪音帶〔FACT〕；
2. **比對工具指標口徑** `tap_compare2.py:80-82` 的 `relrms = max|a−b| / RMS(a)`（峰值差 ÷ 張量 RMS），把該 requant 的最大取整誤差放大成 0.2x 數字〔FACT，已逐位再現〕。
真正的「去相關」（cos≪1）只出現在另一組張量：**token_refiner.blocks.{0,1}.attn.qkv_proj.weight，cos=0.015，整張報銷**〔FACT〕——那是轉換器更深一層的 bug，且不在本次兩個數字的路徑上。

---

## 1. 指標公式校準（FACT，本次逐位再現）

| 張量 | maxabs/RMS（=tap_compare2 relrms） | 真實 RMSrel | cos |
|---|---|---|---|
| condition_proj.weight（pre-patch backup vs golden） | **0.232445**（maxabs 0.02832 / RMS 0.121837） | 0.02398 | 0.99971（recon4，2026-11-27，backup 解碼） |
| video_patch_proj.weight（convrot file vs golden） | **0.268666** | 0.021637 | ≈0.9998（recon8 行級 cos med 同級） |
| audio_patch_proj.weight（file vs golden） | **0.277431** | 0.022571 | — |

公式來源：`h3c/scratch/sol/tap_compare2.py:80-82`（mad=max|Δ|，denom=RMS(A)）。同張張量 RMS 口徑 2.2%、峰值/RMS 口徑 27%——「0.23/0.27」從一開始就不是去相關量，是 requant 峰值取整噪聲對張量 RMS 的比值。

## 2. 逐張量對帳表（condition_embedder 路徑 + patch 卷積）

數據源：golden `/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3/FL2VA/transformer`；v2 檔 `minimax_h3_ref2va_int8_convrot.safetensors`（symlink→model_native/FL2VA/transformer，16:52）；pre-patch backup `/Volumes/SSD2T/h3c_evidence_20261126/patch_backup_convrot/`。全部由 recon5/6/7/8（`~/.hermes/cache/scratch/s10qa/qa_root/`）實測。

| 張量 | 檔 dtype/shape | 現況 vs golden RMSrel | 根因歸屬 |
|---|---|---|---|
| condition_proj.weight | BF16 [5376,5120] | **0（bit-identical）**；pre-patch 0.02398 | 轉換器 requant 往返（已在 03:20 byte-patch 修復） |
| condition_proj.bias | BF16 [5376] | **0**；pre-patch 0.02034（maxabs/RMS 0.2655） | 同上；bias 本不該量化也被 requant |
| video_patch_proj.weight | F32 [5376,96] | **0.02164（仍在）** | 轉換器 requant，不在 03:20 patch 範圍 |
| video_patch_proj.bias | F32 [5376] | 0.02054（仍在） | 同上 |
| audio_patch_proj.weight | F32 [5376,32] | 0.02257（仍在） | 同上 |
| audio_patch_proj.bias | F32 [5376] | 0.03676（仍在） | 同上 |
| token_refiner.final_norm.weight | BF16 [5120] | 0；pre-patch 0.00312 | 已修 |
| final_layer.adaln_proj.linear.weight/.bias | BF16 | 0（recon7 全新 bit-equal） | 未受損或已修 |
| blocks.{0..31}.adaln_proj.linear.bias | BF16 [96×5120] | 0.014–0.039（全 32 個仍在） | 轉換器 requant（weight 全部等於 golden，bias 單獨受損——模式 FACT，機制 UNKNOWN） |
| time_embedder.proj_out.weight | BF16 | 0.0609 | 轉換器 requant |
| final_layer.video_out/.audio_out.weight | F32/BF16 | 0.0503 / 0.0784 | 轉換器 requant |
| token_refiner.blocks.{0,1}.attn.qkv_proj.weight | BF16 [21504,5376] | **RMSrel 1.40 / cos 0.0153 = 整張去相關** | 轉換器 catastrophically 失敗（見 §4） |
| token_refiner out_proj/fc1/fc2 | BF16 | 0.030–0.035（cos 0.9994，仍同域） | requant 往返 |
| 其餘 norm/qk_norm 權重（~180 個） | BF16 | 0.002–0.007 | requant 往返 |

彙總：**279 個 pass-through 張量（1.62 GB bf16/f32）全部與 golden 不同**（recon8_report.json），I8 序列化張量（250 個，含 scale）則通過 P2 契約掃描。

## 3. 九項契約逐項核對（condition/patch 路徑）

| 契約項 | 結果 | 標籤/證據 |
|---|---|---|
| 原始權重 | golden BF16（cond）/ F32（patch），index 齊全、無缺鍵 | FACT（recon8 missing_in_file=0） |
| 轉換權重 | dtype/shape 完全一致；**內容被 requant 往返改寫**（92.5% 元素 bit 不同，grid 級細步、唯一值 5108 pattern 塌縮） | FACT（recon7 ndiff 0.925；recon5 uniq-pattern 測試） |
| scale | 檔內 `weight_scale` 張量恰好 250 個 = I8 張量數；**cond/patch 完全無 scale/sidecar 鍵** → 序列化端 requant 的 scale 未留存 | FACT（recon6 header） |
| group axis | 序列化張量 sidecar groupsize {256:200, 64:50}，kernel 按 `scales[src]` 逐行取值、組內只取 amax → K 軸 per-row [N,1]，契約自洽；pass-through 端無 group 概念，但誤差剖面與「column/全域粗 scale requant」吻合（column-scale 假想 requant 誤差 0.0205 ≈ 實測 0.0216，row-scale 僅 0.006） | 檔內 FACT；轉換器軸向 INFERENCE（recon5 quant_struct 測試） |
| Hadamard 旋轉 | 引擎核 `h3_shaders.metal:1354-1392`（h3_dequant_convrot_rows）以 XOR-conv Walsh kron-h4 回转並折入 1/16 invsqrt——與 ComfyX 契約相符（本次重讀確認）；cond/patch 檔內權重直接對 golden（cos 0.9997），與任何 gs 旋轉域相關僅 0.125 → **無漏回转/多回转**；唯 token_refiner qkv 對原域、rot64、rot256 三者 cos 全部 ≈0 → 非旋轉軸問題，是值域報銷 | FACT |
| packing | I8 張量 rowmap+`scales[src]` 一致（P2 掃描 250/250 PASS，2026-11-26）；cond/patch 無 packing/rowmap | FACT |
| transpose | eager 臂 tap 與 golden **bit-identical**（recon6 `tap_vs_golden_bytes_equal=true`）→ 載入軸約定無誤 | FACT |
| bias | pass-through 端所有 bias 一併被 requant 毀損（cond b 2.0%、video b 2.1%、audio b 3.7%、adaln b 32×2-4%）；golden bias 從未量化 | FACT |
| dequant 公式 | runtime：native int8 配對僅 out_proj/fc2 且 gate `convrot && groupsize==256`（`h3_dit.c:728` 載入 `load_native_int8_pair`，call site :786/:790；gate 註解 :724-727）→ **cond/patch 在 native 模式走 bf16 直通，無 runtime dequant**；tap 與檔 bit-equal 佐證（recon6） | FACT |
| （指標） | relrms=maxabs/RMS(A) 口徑，逐位再現 0.232445/0.268666/0.277431 | FACT（`tap_compare2.py:80-82`） |

## 4. 根因判定

1. **0.23（condition）**：轉換器把「要保留 bf16 直通」的 cond 張量也做了 i8-style requant 再寫回（無 scale 序列化），產生 2.4% RMS 往返誤差；峰值取整差落在 tap 指標分母（張量 RMS）上 = 0.2324。非去相關（cos 0.9997）、非 kernel bug。該组張量已於 11-27 03:20 以 golden 位元組修回（現況 bit-identical）。〔FACT〕
2. **0.27（patch conv）**：同一機制，patch w/b 不在當時 patch 範圍，**損傷至今仍在檔內**（video/audio patch w/b，RMSrel 2.1–3.7%），runtime 直接吃這組帶噪 bias/weight。〔FACT〕
3. 引擎端契約（scale 逐行、回转核、packing/rowmap、dequant 公式、native gate）逐項核過，**無違規**；先前 native 內容崩壞根因另案（head-major 激活旋轉，P3G 已結案）。〔FACT/引用〕
4. 更大範圍：requant 毁損是**系統性**的（279 張 pass-through 全中，0.2%–8% 常態帶），且 **token_refiner qkv_proj.weight 完全去相關（cos 0.015、RMSrel≈√2）＝轉換器對該兩張的處理整個失敗**——這是 patch 後 eager 仍 off-prompt 的頭號候選根因。〔去相關程度 FACT；「這是 off-prompt 主因」INFERENCE〕

## 5. 最小修復建議（不改 kernel）

1. **byte-patch 擴圍**（零記憶體增量的最終修復）：對 recon8 清單中 RMSrel>1e-3 的 pass-through 張量，一律以 golden 位元組就地覆寫 v2 檔：32×adaln bias、video/audio patch w+b、time_embedder proj_in/out w+b、final video/audio_out w+b、token_refiner qkv_proj.weight ×2（462 MB，報銷級）（norm 級 0.2–0.6% 是否一併修，視容忍度；建議一併，總寫入 ≈1.1 GB 內）。
2. **轉換器契約修正**（治本，在 export 工具端）：不合格 int8 序列化的張量必須原 dtype 位元組直通輸出，禁止 requant 往返；若量化則必須連 sidecar 一起序列化。加一支 export 後 audit（現成 recon8.py 邏輯：逐張量 vs source，RMSrel>1e-4 即 fail）。
3. **指標修正**（1 行）：tap_compare2 併報 RMSrel 與 cos，避免 maxabs/RMS 把取整噪聲誇大成「去相關」。
4. **token_refiner qkv 深挖**（若想知道轉換器怎麼壞的）：比對 golden bf16 → Comfy 轉換管线中間产物；目前檔內值與 golden 的任何合法變換（原域/rot64/rot256）都不相關。

### bf16 保留 fallback 記憶體代價估算
- **現況增量 = 0**：這些張量本來就以 bf16/f32 存在檔內與引擎 RAM（native int8 僅 out/fc2，`h3_dit.c:786-790`），修 bytes 不改 footprint。
- **機會成本口徑**：damaged 集 1.62 GB bf16/f32；若未來把它們 int8 化（per-row scale），可省 ≈0.81 GB → 「保留 bf16 不 int8 化」= 多佔 **≈0.81 GB**（對 native peak RSS 22.7 GB 為 +3.6%，128 GB 機器可忽略）。
- 僅 condition/patch 核心 12 張的口徑：bf16/f32 ≈ 57.9 MB → int8 可省 ≈ 29 MB；adaln 全套（含 final_layer 393.9 MB）口徑：bf16 vs int8 差 ≈ 226 MB。

## 6. 未知/未驗（UNKNOWN）
- 轉換器對 pass-through 張量用的確切 quantizer/scale 軸（column-fit 較佳僅為 recon5 誤差剖面配適，INFERENCE）。
- adaln「weight 完好、bias 獨損」的機制。
- 03:20 patch 實際範圍（backup 目錄僅存 3 個 .orig.raw，與紀錄「5 張」不符）。
- token_refiner qkv 報銷的具體轉換器步驟。

產出檔：本報告 `h3c/scratch/sol/2026-11-27_condition_patch_relativity_reconciliation.md`；數據 `~/.hermes/cache/scratch/s10qa/qa_root/recon{4..8}*.{py,json,txt}`。
