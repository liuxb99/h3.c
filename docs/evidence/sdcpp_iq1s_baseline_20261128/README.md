# sd.cpp IQ1_S 端到端基線判定（2026-11-28）

**判定：(ii) NEED_COORDINATION_WINDOW** —— 代碼路徑支援但 companion 檔案缺席 + 記憶體閘門當前全 DENY；非 (i) 立刻可跑，亦非 (iii) loader 硬 incompat。

## 判定依據（復核引用）

### 1. companion 旗標（實名非 --text-encoder-path/--vae-path）
`docs/evidence/sdcpp_iq1s_baseline_20261128/sd_cli_help.txt`（sd.cpp pin e239e4b `--help` 全 299 行）：
- `:41` `--llm <string>` — LLM text encoder 路徑（MiniMax-H3 的 TE 走此旗標）
- `:53` `--vae <string>` — standalone VAE 路徑
- `:55` `--audio-vae <string>` — standalone audio VAE 路徑
- `--text-encoder-path` / `--vae-path` 於 help 全文 grep 計數 = 0（前手記憶的旗標名不存在，實際為上述三個）

TE/VAE 綁定碼：`~/tools/stable-diffusion.cpp/src/model/integrations/minimax_h3.cpp:11`（conditioner = `LLMEmbedder`，TE 必需）；`~/tools/stable-diffusion.cpp/src/pipeline/model_builders.cpp:161`（`vae_kind == ModelVAEKind::MINIMAX_H3` → `MiniMaxH3VAE::MiniMaxH3VideoVAERunner`，讀 `first_stage_model.*` 前綴）、`:219`（audio runner 分支）。

### 2. adaln_single 期望的真實來源（非 MiniMax-H3 路徑）
- `~/tools/stable-diffusion.cpp/src/model_loader.cpp:566-574`：`model.diffusion_model.adaln_single.*` 僅用於 **PixArt / LTXAV** 版本偵測
- `~/tools/stable-diffusion.cpp/src/name_conversion.cpp:979-985`：`adaln_single.*` → flux/LTX 內部名映射
- `~/tools/stable-diffusion.cpp/src/model/diffusion/ltxv.hpp:1357-1364`：adaln_single block 屬 LTX 模型定義

**GGUF 表化 fallback 已存在（選 (iii) 前的指定 grep，結果推翻 adaln 硬 incompat 假設）**：
- `~/tools/stable-diffusion.cpp/src/model/diffusion/minimax_h3.hpp:96` `find("adaln_t_table")`、`:516` `params["adaln_t_table"]`、`:545-546` `ggml_get_rows` 查表
- `minimax_h3.hpp:390/:457` 每 block `adaln_proj`（AdaLayerNormModulation）
- 版本偵測 `model_loader.cpp:575-577`：`video_patch_proj.weight` + `audio_patch_proj.weight` → `VERSION_MINIMAX_H3` —— 本 GGUF 兩者俱在（gguf_inventory.txt:5）

即：本 GGUF 的 `adaln_t_table [8,1025]` + `blocks.N.adaln_proj.linear.*`（inventory:8-14）正是 sd.cpp MiniMax-H3 分支期望的命名，語義**相容**而非差異。前手「adaln_single 表 vs GGUF 差異」線索係誤植（那是 LTX 偵測碼）。

### 3. 本 GGUF = DiT-only 事實
`gguf_inventory.txt`：
- `:1` 532 tensors（f32:10/f16:102/bf16:212/iq1_s:208）
- `:6` `non-DiT tensors: 0` — 無 `first_stage_model.*`（VAE）、無 TE/LLM、無 audio VAE tensors
- 檔案：`/Volumes/SSD2T/models/MiniMax-H3-gguf/minimax_h3_fl2va_pruned-IQ1_S.gguf`（3.8G，目錄內仅此一個模型檔，無 companion GGUF）

### 4. 記憶體閘門現值（admit_poll.log）
21:45–21:54 十次輪詢 **10/10 DENY**：committed 145.4–149.7GB > hard 115GB（正式 8080 線，pid 61463），wired 46–48%（Metal-pinned）。
常駐算術：DiT ~49.6G + TE（原 62G bf16 / 量化後 15.7G）+ VAE 9.7G ≈ **65–75G 常駐需求** vs 當前 free ~45G（35%）+ swap used 僅 48M —— 閘門 DENY 期間無法 Admission，需協調窗口。

## 三選一逐項否證
- **(i) 補 companion 立刻跑 — 否**：支援路徑明確（--llm/--vae/--audio-vae + MiniMaxH3 分支俱在），但本地無現成 companion GGUF（需產出/下載 TE-LLM + MiniMaxH3 VAE + audio VAE），且 admit 閘門當前 DENY。
- **(iii) loader 硬 incompat — 否**：adaln 表化在 minimax_h3.hpp 有 GGUF 分支（見上），版本偵測可命中。
- **(ii) NEED_COORDINATION_WINDOW — 是**：代码就绪，缺 (a) companion 權重檔產出/下載、(b) 記憶體窗口（+65–75G 常駐 vs free 45G，committed 已貼 hard）。

## 對 h3.c 的含義
- 端到端 sd.cpp 基線（同 GGUF、img/vid_gen 全流程對拍）**短期不可得** → 速度/畫質外部基準改用**已驗收的 ConvRot 原生影片** + 既有**每層 sd.cpp 對拍數據**（層級對拍不受 companion 缺席阻塞）。
- sd.cpp 的**數值 oracle 地位不變**（逐層 logits/中間激活對拍仍是 ground-truth 校準源）。
- 窗口協調事項：TE/VAE companion 落地（IQ 量化 TE ~15.7G 優先用）+ 8080 committed 回落至 hard 115GB 以下時再排 admission。
