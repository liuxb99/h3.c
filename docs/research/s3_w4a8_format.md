# Sulphur-3 (W4A8) 格式研究 — Phase 1

日期: 2026-11-27 | 來源: HF SulphurAI/Sulphur-3 (library=minimax-h3, 非 gated, 189k downloads)
模型卡: s3 = MiniMax-H3 uncensored 衍生，「run it like any regular MiniMax H3 model」；
quantized/ = 低記憶體變體；s3_corrector 可選（ComfyUI/models/h3_correctors/，節點=ComfyUI-H3-Corrector zip）。
授權: MiniMax H3 Community License。

## 倉庫檔案清單（28 files, 257.5 GB 總量，勿全倉下載）
| 檔 | 大小 | 角色 | 下載決策 |
|---|---|---|---|
| s3.safetensors | 37.46G | 正式版（RL lora 已併） | Phase 3 基準，暫緩 |
| s3_raw.safetensors | 37.46G | 無 RL lora 原版 | 暫緩 |
| s3_ref2va.safetensors | 37.46G | Ref2VA 變體（未訓練） | 暫緩（本地已有 ref2va 鏈） |
| quantized/s3_w4a8.safetensors | 10.65G | **正式 W4A8** | ✅ 下載中（SSD2T/models/SulphurAI-Sulphur-3/） |
| quantized/s3_raw_w4a8.safetensors | 10.65G | raw 版 W4A8 | header 已取；實體檔暫緩 |
| quantized/s3_raw_int8.safetensors | 19.51G | raw int8（A8W8-linear?） | 暫緩 |
| s3_corrector.safetensors | 7.44G | 可選 corrector | Phase 4 |
| s3_raw_lora.safetensors | 4.44G | s3_raw 的 LoRA 形態 | 暫緩 |
| experimental/* | 0.4–31.7G | 作者建議忽略（含 s3k8_w4a8 9G、hybrid mid int8 convrot keepB0-1-49 20.6G） | 暫緩 |

TE/VAE：s3 檔為 **DiT-only**（無 cross_attn/TE/VAE）→ 沿用本地 NVFP4 TE 62.1G(BF16) / 15.69G(NVFP4) + FL2VA/Ref2VA ✓ 零額外下載。

## W4A8 格式契約（header 實測 + ComfyUI v0.46.1/comfy_kitchen 0.2.39 官方實作交叉驗證）
- metadata: quantization=`asym_w4a8_int8`，逐層 `group_size=16, convrot_groupsize=256`。
- 每量化層四件套:
  - `weight` I8 [out, in/2] — **2 nibble/byte packed INT4**（偶數 col=低 nibble，odd=高 nibble，`_pack_codes` 契約）
  - `weight_codebook` F32 [16] — 對稱 Lloyd-Max codebook（可 frozen 固定 LUT 或層專屬）
  - `weight_s_rel` U8 [out, in/16] — 每 group(16) 一個 **FP8 e4m3** scale（safetensors 以 U8 位元組存）
  - `weight_s_channel` F32 [out] — 每 output row scale
- 官方 dequant 鏈（comfy_kitchen eager，= 可攜 golden）:
  1. unpack nibbles → unsigned code
  2. values = codebook[code]（F32）
  3. × s_rel(fp8→f32) per group → round → clamp ±127 → **INT8 grid**（s_rel 折入 grid）
  4. × s_channel per row → rotated-basis 權重（bf16）
  5. `rotate_int8_convrot_weight(G=256)` 反旋转回原基底
- 官方 linear 契約（A8 側）: x 動態 **per-token(per-row) int8** 量化 + convrot=True（x 端同 H 旋轉，G=256）
  → int8 GEMM（int32 累積）→ epilogue × act_scale × s_channel。`correction` 可選（GPTQ 殘差，有則走全 dequant 路）。
- 非量化張量原樣: adaln F16、norms BF16、patch/final F32、rope F32。結構 = 原版 H3（50 blocks、token_refiner×2、condition_proj、audio_patch_proj、adaln_t_table[1025,8]）**無架構差異** → h3.c 前端可直接重用。

## 與 h3.c native 引擎的差距分析（FACT）
1. **G 差異**: 引擎 convrot 表固定 G=128（v2 sidecar+rot 表），s3 中繼契约 G=256 → 需把 dequant/GEMM 表尺寸參數化（G 由 sidecar/metadata 讀取），或 load 時做一次基底轉換矩乘。
2. **codebook 步驟**: 引擎 v2 是線性 I8+row_scale；s3 需 nibble→codebook LUT(16)×fp8 group scale → 生成 int8 grid 後即與引擎現有「I8 grid + row_scale」語義**同型** → 可在 loader 端一次性轉成引擎 grid（含 un-rotate G256 / 重旋转 G128），native GEMM 路零改動。
3. dequant 參考路（bf16）= 同一 loader 輸出的 grid 全反量化 → 天然正確性 fallback（計畫路線 B）。
4. 尺寸效益: s3_w4a8 DiT 10.65G vs 原版 BF16 62.1G vs v2 I8 ~30G → **W4A8 為 36GB 記憶體預算目標的最大槓桿**。

## Workflow 契約（Phase 3 基準用）
- ComfyUI 標準 H3 跑法（sampler res_multistep/simple、CLIPLoader type=minimax、幀長 17n+5、TE=Qwen3VL nvfp4）。
- ComfyX 生產配方既有資產（skill comfyx-comfyui-macos pr_stage 已驗收 payload）。
- corrector 契約 + 高階 motion 指南 = S3_ComfyUI.zip（已取得）+ experimental/H3_Corrector_Advanced_*.docx（Phase 4 再讀）。

## Open questions
- s3 建議 steps/CFG/sigma shift（作者未寫死；預設=原版 H3 配方 20 步 golden；等步數對比矩陣在 Phase 4）。
- raw vs 正式 = RL LoRA 併入與否（merged）；s3_raw_w4a8 header 需對照（同 quant metadata?）。
- ComfyUI 需 ≥0.46.x + comfy_kitchen ≥0.2.39 才認 asym_w4a8_int8（本機實路徑已符合 ✓）。

## 下載完成＋全檔驗證（FACT, 2026-11-27 01:37）
- `/Volumes/SSD2T/models/SulphurAI-Sulphur-3/s3_w4a8.safetensors` 11,433,462,056 B（=10.65 GiB）；curl 單程完成且 SIZE==EXPECT；SHA256=6c496c746d380bc371e5b9ef0c81cbea4f107e8fba1ac3f6b7d821c4e96e9ae0（HF tree API 未提供 lfs sha 可比對，以 size+單程完成+header 全掃代替並記錄本 hash 為來源指紋）。
- header 全掃：quantization=asym_w4a8_int8 + `_quantization_metadata`(format_version 1.0, 逐層 layers 清單)；1156 tensors（I8 208=INT4 packed、U8 208=fp8e4m3 s_rel、F32 426、F16 102、BF16 212）；208 量化層四件套齊全；非量化=norms/adaln/patch_proj 等（與原版同型）。
- 實測 sanity：qkv codebook=對稱 16-level Lloyd-Max（±0.9807）；s_channel 0.0007–0.007 無零行；s_rel fp8 值域 22–128；weight I8 [21504, 2688]（5376 packed ÷2 ✓）。
- 下載備註：HF CDN 對該檔 HTTP/2 傳輸兩度 CANCEL(err 92)，HTTP/1.1 單程成功——後續大檔下載預設 --http1.1。
