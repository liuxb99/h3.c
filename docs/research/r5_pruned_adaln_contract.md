# R5 接線規格 — pruned adaln 契約（全部 FACT 摘自源碼，2026-11-28）

## sd.cpp 側（oracle：minimax_h3.hpp @e239e4b）
- `adaln_t_table`（GGUF [8,1025] F32）＝**時間曲線表**，取代被剪掉的 `time_embedder.proj_in/out`。
  消費（hpp:541-551,1070-1090）：`position = clamp(t,0,1) * (grid-1)`；
  `t_emb = lerp(table[⌊position⌋], table[⌊position⌋+1], position-⌊position⌋)`（get_rows+mul+add，fp32）。
- `AdaLayerNormModulation(time_dim=2688, hidden, expand=6, modalities=3, apply_silu=false, force_f32=true)`（hpp:311-338,390-395）：
  對每 block 本應是 `linear(t_emb) → [hidden*6*3]`。**pruned GGUF 已把此 matmul 折叠**：
  每 block 存 `blocks.N.adaln.weight [96768, 8] F16`＝8 個曲線格點位置各自的**最終 modulation 行**（96768 = 2688×36 = 2688×6×6，承 H3 雙模態 6-slot 契約；確切 chunk/modality 排列需 golden tap 驗證，見下）。
  推論（依 hpp 代數，NOT_TESTED 至 tap 為止）：
  `mod(block, step) = lerp(adaln[block][⌊pos⌋], adaln[block][⌊pos⌋+1], frac)`——**運行期零 matmul、零 time_embedder**。
- 兼容線索：sd.cpp 載入端見 `model_loader.cpp:566-581`（adaln_single 家族判型）與 name_conversion.cpp（adaln_single→t_block 映射）；H3 判型在 model_loader.cpp:582（video_patch_proj+audio_patch_proj→VERSION_MINIMAX_H3）。

## h3.c 引擎側（h3_dit_schedule.c 489 行）
- 現行：`time_embeddings()` 用 `time_embedder.proj_in(silu)→proj_out` 逐 step 產 t_emb（行 167-214），再逐 block `blocks.N.adaln_proj.linear.weight`（BF16→bf16_auto）**matmul** 產 BLOCK_OUTPUT=MODALITIES*SLOTS*HIDDEN 的每步调制行（行 242-275+）。H3_DIT_TIME_DIM=2688。
- **GGUF 模式改法（R5）**：schedule 模組加 pruned 路徑——
  1. 載入 `adaln_t_table`（F32 GPU tensor）＋每 block `adaln.weight` [96768,8]（F16→GPU，勿展開）；
  2. 每 step 依 sigma/timestep→`position`（重算 hpp:1076 公式，timestep 來源＝engine 現行 sigmas→t 映射，需與 sd.cpp layout.timesteps 對齊——**此映射是 tap 驗證點 #1**）；
  3. modulation 表列＝兩相鄰格點 lerp（一條小型 Metal kernel 或直接 CPU 每步 8×96768 f32 ≈ 3MB/step，6 步＝24MB，可忽略）；
  4. 下游 `h3_dit_schedule_block()` API 不變→block/attention token-pool 路徑零改動。

## tap 驗證點（R5 序列，逐項產 JSON）
1. timestep→position 映射 vs sd.cpp（同一 sigma 佈局，雙邊 dump 逐 step 比對）
2. t_emb lerp vs sd.cpp（CPU 對 CPU，位元級）
3. block0 mod 行 lerp vs sd.cpp forward 輸出（容差 bf16/f16 舍入≤1 ulp）→ 定案 96768 的 chunk/modality 排列
4. 其後逐層 tap（x_embedder→block0 attn→mlp→...→final）沿用 forward_dit evidence 慣例

## 記憶體注記
GGUF 模式省：time_embedder 4 鍵（~?MB）＋adaln_proj matmul 輸出常駐不變；adaln F16 表 45×1.5MB=68MB ≪ convrot 的 96768×2688 BF16 ≈ 500MB。禁整檔展開（R4 規則沿用）。
