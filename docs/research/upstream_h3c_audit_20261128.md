# 上游 h3.c 功能審計（2026-11-28）

Audit basis: `origin/main = 8974cc0`（antirez/h3.c，最後 commit 2026-08-11 "Clarify SSD streaming memory and speed tradeoff"）。本地 `HEAD = ce6dd1c`（feat/mac-pika-flex）。`git fetch origin` 已執行，無新物件（fork main 與上游 main 同 SHA）。所有上游行號引用皆以 `git grep -n <pat> origin/main` / `git show origin/main:<file>` 實測。

---

## A. 上游 origin/main 逐項查證

### A1. GGUF 容器讀取 —— ❌ 無
- `git grep -in "gguf|ggml" origin/main`（含 README/Makefile/tests）：**0 命中**。整個 67 檔樹中無 `h3_gguf.*`。
- 載入器只有 safetensors：`h3_weights.h:13-17`（`h3_weight_store_open`「Open every **safetensors** header in a component directory」，實作於 `h3_weights.c`，依賴 `h3_safetensors.c`）。
- `git log --grep=-i gguf origin/main`：上游全史 **0 個** commit 提及 GGUF。
- `git ls-remote --heads origin`：上游僅一支分支 `refs/heads/main`——無 GGUF/quant 實驗分支。
- 旁證：上游 dtype 表有 I8/U8 佔位（`h3_safetensors.h:12-13` `H3_DTYPE_I8/U8`；`h3_safetensors.c:185-186` parse），但 main 无任何 I8 weight 載入/解量化路徑（見 A2/A5）。

### A2. 量化型別支援 —— 僅動態 INT8；❌ 無 GGUF 型別族
- 上游量化 = **載入 BF16 後 GPU 端動態轉 int8**（非讀預量化檔）：
  - `h3_dit.c:692 quantize_block_mlp()` / `:722 quantize_block_qkv()` / `:742 quantize_block_attention_out()` 呼叫 `h3_gpu_quantize_weight_int8()`（宣告 `h3_gpu.h:313`，實作 `h3_gpu.m:2932`；調用點 `h3_dit.c:703/729/749`，觸發於 `:1260-1266`）。
  - `h3_dit.c:1260-1266`（載入時逐 block 量化）；`h3_dit.c:1703` 區註明 int8 為 M5 預設。
  - 行量化 int8 FC2 開關與驗證：`h3.c:539-563`（`use_int8_row_fc2` 需 `metal4`/M5：`:562-563 "int8 row FC2 requires an M5-class Metal 4 GPU"`）。
  - 非對稱組量化（256-thread grouped activation quantizer）：`h3.h:122-123 use_slower_grouped_quantizer`。
- `git grep -i "iq1|iq2|q2_|q4_|nvfp4" origin/main`：**0 命中**（.c/.h/.m/.metal/Makefile 全無）。上游只有 INT8（row/group scale）與 BF16/FP32，無 sub-8bit codebook 型別。
- 上游 `README.md:705-784` 整節為 int8 M5 TensorOps 路徑文件；`README.md:157,196-201` `--use-int8-row-fc2`。

### A3. Metal 壓縮權重直接 matmul —— ✅ 有（int8-only）
上游 shaders 內核**直接吃 int8 權重 + F32 row/group scale 做 matmul**（dequant 折入 accumulation），即「壓縮權重直接 matmul」語義：
- `h3_shaders.metal:2346 kernel h3_linear_int8_nax_r128`（另有 `:2399 ..._full_k14336`、`:2453 ..._r128x256_full_k14336`、`:2512 ..._local_scales_nax_r128_impl`、`:2597/2677/2796 h3_linear_int8_grouped_*`）。
- `h3_shaders.metal:2139 kernel h3_fc1_swiglu_int8_nax_r128_impl`、`:2263 h3_fc1_swiglu_int8_local_nax_r128`。
- `h3_shaders.metal:1688 h3_qkv_project_split_int8_nax_r128_morton4`、`:1755 ..._int8_rope_..._impl`、`:1943 ..._local_scales...`。
- scale 快取註解：`h3_shaders.metal:2509`「Cache the two 128-value dequantization vectors once per projection tile」。
- 量化產生端（同檔）：`:1320 h3_quantize_bf16_int8_rows`（`:1317` 註解自稱「Draw Things/ccv-style dynamic symmetric row quantization」）、`:1481/1530/1593/1637 ..._int8_groups*`。
- 融合格線+量化：`h3_dit.c:1961 h3_gpu_gate_adaln_quantize_int8(...)`（gate/AdaLN/quant 三合一）。
**但**：型別僅 int8+scale；無 iq1_s/q4/nvfp4 codebook kernel，無 GGUF payload 消費端。

### A4. pruned/摺疊 AdaLN —— ✅ 上游有「schedule 表化」，❌ 無「摺入權重檔」版
- 上游有 per-step AdaLN **預計算表**（按去重複 timestep 列）：
  - `h3_dit_schedule.h:22-30 h3_dit_schedule_precompute()`——註解「Materialize **every per-step AdaLN value**…one projection at a time, so a 498 MiB block projection is released before the next is loaded」。
  - `h3_dit_schedule.c:76-160 prepare_rows()`——timestep 列去重（video/audio/condition 同 sigma 共用一列），`:97 "out of memory allocating timestep row maps"`。
  - `h3_dit_schedule.c:163 time_embeddings()`——每唯一列跑 `time_embedder.proj_in/proj_out` matmul+SiLU（`:201-216` 「timestep input projection / SiLU / output projection / AdaLN SiLU」）；權重名 `time_embedder.proj_in.weight` 等見 `:170-178`。
  - 逐 block 調制投影 `blocks[H3_DIT_BLOCKS]` 張量表（`h3_dit_schedule.c:17-24` struct；`h3_dit_schedule.h:37-43 block()/gate_score()/final()`）。
- Block 剪枝（gate-ranked）：`h3_dit_schedule.c:420 h3_dit_schedule_prune()`；`h3.h:80-81 dit_layers`（「gate-ranked DiT residual blocks to retain. 50 exact, 45 validated, 40 aggressive」）；套用于 `h3_dit.c:1676`。
- 跨步复用 + timestep head 每步刷新：`h3.h:83-85 core_reuse`（註解「Recompute the transformer core every N denoiser steps while refreshing the timestep head each step」，欄位宣告 `h3.h:85`）。
- AdaLN 執行核存在：`h3_shaders.metal:357 h3_adaln_f32`、`:3030 h3_adaln_bf16`；融合格線/池化：`h3_dit.c:1807-1867 enter/leave_token_reduction_adaln`、`:1967 H3_DISABLE_FUSED_GATE_ADALN`。
- **差别**：上游仍在載入期跑 time_embedder + 每 block 調制 matmul（只是按唯一 timestep 列去重、結果駐留 VRAM）；無 `adaln_t_table`、無「time_embedder 摺疊成 per-block 曲線表、執行期零 matmul」概念（grep `t_table|curve` 上游 0 命中）。上游把 time_embedder 權重原樣保留（`h3_dit_schedule.c:170-178`），我們 GGUF 已將其實體摺成 `adaln_t_table`（R1 映射，d7be295）。

### A5. 影片生成完整鏈 —— ✅ 全有
- Text encoder：`h3_text_encoder.c`（`h3_text_encode_bf16` 調用於 `h3.c:1419`）、`h3_tokenizer.m`、vision `h3_vision_encoder.c`、多模態 `h3_multimodal.c`（ref2va/fl2va `h3.c:1396-1401`）。
- Scheduler：`h3_sigma_schedule`（`h3.c:1458`）+ `h3_dit_schedule.c`（A4）。Euler 更新 `h3_dit.c:2822-2832`。
- VAE：`h3_video_vae.c`、`h3_audio_vae.c`（Makefile LIB_C 全鏈列於 `Makefile:10-13`）。
- MP4 輸出：FFmpeg 子行程管線 `h3_ffmpeg.h:37 h3_ffmpeg_write_rgb24 / :43 h3_ffmpeg_write_av_rgb24_f32`（RGB24+F32 PCM 双流併發、無中間檔）；檔名生成 `h3_cli.c:428 "%s/video-%04u.mp4"`；README 全程 `-o *.mp4`。
- CLI/互動：`h3_cli.c`、`h3_terminal.c`、`main.c`。

### A6. 編譯開關矩陣 —— ❌ 無
- 單一 `Makefile`（67 檔樹無 CMakeLists）；開頭旗標寫死：`Makefile:1-8`（`CC := clang`、固定 `CFLAGS := -std=c11 -O3 ...`、固定 Metal/Accelerate frameworks）。無任何 `-D` 功能開關、無 if 型別/容器選項。
- 全部功能走**執行期**：`h3_params`（`h3.h:64-131`）+ `getenv` 鉗（`h3_dit.c:1967 H3_DISABLE_FUSED_GATE_ADALN`、`:2191 H3_DISABLE_FUSED_TOKEN_POOL_ADALN` 等）。

---

## B. 上游動態（GitHub API，2026-11-28 實查）
- `antirez/h3.c` open PRs = **52**、open issues（非 PR）= **34**。上游 main 自 2026-08-11 起 **零新 commit**——全部 PR 掛 open 未併入。
- 最近 30 個上游 commit 全在 2026-08-08…08-11（int8 M5 kernels/量化调優/SSD streaming/low-step scheduler；`--stat` 觸及 quant/loader/schedule 域者見 log 記錄：`47d5498 Add M5 int8 DiT MLP kernels` … `cae6e54 Add original-BF16 SSD-streamed DiT mode`）。無 GGUF/convrot/pruned 字样（`git log --grep` 上游史 0 命中）。
- GGUF/quant/AdaLN 相關 open PR（皆未併入、皆非 GGUF）：
  - **#99** "Optional offline INT8 DiT pack (h3_pack_int8)"（2026-11-17, ain-soph）：離線 int8 包——輸出**普通 safetensors**（`qweight` I8 + `qscale` F32）+ `adaln-table.safetensors`（預計算 2..20 步 AdaLN 表）+ `h3-int8-pack.json`。是上游系最接近我們方向的提案，但非 GGUF、僅 int8、adaln 表為引擎另存檔。
  - **#88** int8 權重磁碟快取（H3_DIT_INT8_CACHE）、**#90** AdaLN schedule 磁碟快取（H3_ADALN_CACHE_DIR）、**#97** SageAttention 式 int8 QK（掛 #86 Sol-Attn）、**#73** VAE BF16/int8 解碼。
  - **無任何 PR/issue 標題或 body 提 "GGUF"**（52 PR + 34 issue 全件 body 關鍵字掃描，gguf/ggml/iq*/fp4 0 命中）。
  - Issue **#41** "Feature request: 4bit and 8bit support"（2026-08-19）仍 open，唯一回覆僅「check forks - allready done」——上游無 4-bit 答案。
  - Issue **#28** 記錄官方「optimized int8 single-file layout」資產名 `minimax_h3_fl2va_pruned_int8_convrot.safetensors` / `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`（MiniMax HF revision 014cd40）——即 **pruned+int8+convrot 的官方資產存在（safetensors 容器）**，但上游 main 代碼無 convrot/comfy_quant/row_scale 讀取路徑（grep 0 命中）。
- 其他域：#95 LoRA fold 工具、#84 MiniMax-H3-Image、#86 Sol-Attn block-sparse、#92 renoise sampler、#43/#42 CUDA backend（issue #50 明示反 CUDA 併入）。

---

## C. 分岔地圖
- `git merge-base origin/main HEAD` = **8974cc0**（即 origin/main 本體，2026-08-11）。
- `git rev-list --count $(merge-base)..origin/main` = **0**：分岔後上游**零新 commit** → 目前無拉上游代碼壓力（但 52 個 open PR 是未來上游可能吸收的功能面，追蹤清單見 B）。
- `origin/main..HEAD` = **50 個本地 commit**，功能域分類（自動歸檔 `upstream_h3c_audit_commits_20261128.json`）：
  | 域 | 數 | 代表 commit |
  |---|---|---|
  | GGUF loader（R1–R2） | 3 | d7be295（R1 結構審計）、aa95b9f（R2 reader 697 行）、04a329d（R1–R7 收尾+HF GGUF 資產文檔） |
  | IQ1_S dequant（R3–R5 bundled） | 1 | ce6dd1c（R3 CPU bit-exact 參考 + R4 Metal 移植規格 + R5 pruned-adaln 契約） |
  | int8 convrot / native A8W8（Metal8） | 20 | f137c6c/046faff（檔案 int8 直載+F8_E4M3/I8 dtype）、58d691d（dequant 鏈 Metal）、5af4819（convrot 三方驗收）、7597ad8（rot-quant kernel 定案 h4⊗4 XOR-conv）、974ffed/8d1243f（H3_NATIVE_INT8 wiring）、1f9d773（zero-copy mmap 根因）等 |
  | sol-attn block-sparse | 5 | 7f9d549、b453376、1fe9a45 |
  | W4A8/S3 格式研究 | 2 | d8bdadf、73d6e7b |
  | 證據/對拍/gate/forensics | 13 | b9c998a、7248ed5、8080d74、9d43bf5、3083455 等 |
  | 設計/移植文檔 | 4 | 0255075、cb137f0、49cc2a5、d0c8d1e |
  | 其他（P4C/P4E 語義法證、SHA manifest 等） | 2+ | 2804797、349e871、4af2e70 |

---

## D. 結論表（每行附證據）

| # | 項目 | 分類 | 證據 |
|---|------|------|------|
| 1 | 影片生成完整鏈（text encoder/scheduler/VAE/MP4） | **上游已有可直接用** | `h3_text_encoder.c`+`h3_multimodal.c`；`h3_dit_schedule.c`（sigma+AdaLN 表）；`h3_video_vae.c`/`h3_audio_vae.c`；`h3_ffmpeg.h:36-44`；fork=origin=8974cc0，我們直接繼承 |
| 2 | gate-ranked block 剪枝 + core_reuse（步間复用） | **上游已有可直接用** | `h3.h:80-85`（dit_layers 註解 `:80-81`、欄位 `:82`；core_reuse 註解 `:83-84`、欄位 `:85`）、`h3_dit_schedule.c:420 h3_dit_schedule_prune`；我們 GGUF 的 pruned 資產與此互補（剪枝在資產層已完成） |
| 3 | safetensors 載入器 + M5 int8 動態量化全家 | **上游已有可直接用**（BF16 來源路徑） | `h3_safetensors.c:185-186`、`h3_dit.c:692-755`、`h3_gpu.h:313` |
| 4 | Metal 壓縮權重直接 matmul（int8+scale 型別） | **上游已有可直接用**（int8-only）；sub-8bit 版 **上游沒有必須自研** | 上游 `h3_shaders.metal:2346/2139/1688/2597`（int8 直接 matmul）；IQ1_S 型別 0 命中 → R4 自研 |
| 5 | AdaLN schedule 表化 + 磁碟快取概念 | **上游已有需適配**（且未併入） | 上游 in-memory 版 `h3_dit_schedule.h:26`（main 內）；磁碟化版是 **open PR #90**（未併入）；我們 R5 契約（ce6dd1c，`docs/research/r5_pruned_adaln_contract.md`：adaln_t_table lerp + per-block 96768×8 預摺曲線表 → 執行期零 matmul）比上游更激進——上游仍需 time_embedder+block 調制 matmul（`h3_dit_schedule.c:201-216`），引擎讀我們 GGUF 表要寫新載入分支，不能復用 time_embeddings() |
| 6 | GGUF 容器（v1–v3）讀取 | **上游沒有必須自研（本次已完成應保留）** | 上游 gguf/ggml 全樹 0 命中、0 commit、0 分支、0 PR/issue；我們 `h3_gguf.c`（697 行，`h3_gguf_open` h3_gguf.h:98，GGUF magic 檢查 h3_gguf.c:476-480，sd.cpp traits 表 `h3_ggml_row_bytes`）+ `tests/test_gguf_loader.c` 532 tensors fails=0（commit aa95b9f） |
| 7 | IQ1_S 解量化（CPU 參考） | **上游沒有必須自研（本次已完成應保留）** | `scripts/iq1s_cpu_ref.c` 316 行，208/208 tensors、20.04B 元素 global max_abs_diff=0 對 pinned sd.cpp golden（commit ce6dd1c + `docs/evidence/iq1s_gguf_r3_20261128/`） |
| 8 | IQ1_S Metal kernel 移植（R4） | **上游沒有必須自研**（規格已出） | 上游無 sub-8bit kernel（A3）；`docs/research/r4_metal_iq1s_port_spec.md`（ce6dd1c） |
| 9 | convrot（Hadamard-旋轉 int8）讀取/解量化 | **上游沒有必須自研（本次已完成應保留）**；注意官方**資產**存在 | 上游代碼 convrot/comfy_quant/row_scale 0 命中（main）；官方 safetensors 資產名見 issue #28 body；我們 `f137c6c`（檔案 int8 直載）、`58d691d`（dequant 鏈）、`5af4819`（三方驗收）、`7597ad8`（旋转矩阵定案 h4⊗4 XOR-conv + bit-exact oracle） |
| 10 | 離線 int8 包（免重複量化） | **上游已有需適配**（open PR，未併入，非 GGUF） | PR #99 `h3_pack_int8`（safetensors qweight/qscale + adaln-table）與 PR #88（int8 cache）——方向重疊但格式/容器不同；上游若併入可取代我們的 sidecar cache 思路，暫不併購代碼 |
| 11 | Sol-Attn block-sparse attention | **上游已有需適配**（open PR #86/#97，未併入）；我們 v1 已完成 | 我們 7f9d549/b453376（v1 A/B 慢 3.4× FAIL 已記錄 1fe9a45）；上游 PR #86 稱 23–28% 提速——接手上游版本前勿再自行迭代 v1 |
| 12 | sub-8bit（4bit/NVFP4/IQ1 家族）資產支援 | **上游沒有必須自研** | issue #41 open 無答案；PR #99 僅 int8；我們 S3/W4A8 研究（d8bdadf/73d6e7b）與 IQ1_S 線（d7be295→ce6dd1c）互補，屬無人佔位域 |

### 重點直問
1. **R2 `h3_gguf.c` 是否上游已有等價物？——無。** 上游（代碼、歷史、分支、52 個 open PR、34 個 open issue）**完全沒有 GGUF**（關鍵字 0 命中，見 A1/B）。上游系唯一離線壓縮方案是未併入的 PR #99（自製 safetensors 包，非 GGUF 容器、不含 codebook 型別）。→ **無重複造輪子問題。**
2. **R4 移植 sd.cpp Metal kernel 是否上游已有等價物？——無（同型別域才等價）。** 上游 Metal 已有自家 int8 直接 matmul kernel 族（A3），但它們只吃 int8+F32 scale 且來源是載入期 BF16→int8 動態量化；沒有任何 kernel 消費 IQ1_S/codebook payload，也沒有 sub-8bit 型別。R4 移植（iq1_s grid kernel + codebook 折入）在上游無等價物 → 不構成重複。
3. **GGUF 全域判定：上游/側路完全沒有 GGUF 支援 → 明確「無重複造輪子問題」。**（側路=52 個 open PR 全件 body 掃描亦 0 gguf 命中。）

### 重複研發可停止清單
- **停止**：再自行優化 sol-attn v1（上游 #86/#97 在跑，先追蹤）。
- **停止**：自建「AdaLN schedule 磁碟快取」獨立機制——上游 #90 同概念已在 PR 佇列；我們的 R5 摺疊表是資產層方案，不冲突，但快取層勿雙軌。
- **停止**：BF16 來源的 int8 動態量化路線再投入（上游 M5 int8 全家已是 main 預設，README:705-784 文件齊全）。
- **保留且独有**：GGUF 容器、IQ1_S 全家（R1–R4）、convrot 引擎鏈、adaln_t_table 預摺契約——上游無等價物。

---

## 附
- commit 域歸檔：`docs/research/upstream_h3c_audit_commits_20261128.json`
- 上游證據行號清單 = 本檔 A 節全部 `檔案:行 + 函數名` 引用（可 `git grep -n <pat> origin/main` 復現）。
- GitHub 快照取得時間：2026-11-28（api.github.com pulls/issues/comments 實查，無 token 限內）。
