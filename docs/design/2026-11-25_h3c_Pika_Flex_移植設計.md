# h3.c Mac 原生 Pika H3 Flex 移植設計

日期：2026-11-25（修訂 1）
主機：Mac Studio M5 Max 128GB（macOS 27.2.1）
上游基線：antirez/h3.c main @ 8974cc0（含 PR#94 整併）
分支：`feat/mac-pika-flex`
Strata 約束：8080 服務其他 Hermes 節點，不得停止／重啟；全程記憶體護欄。

---

## 1. 任務目標

在本機以 h3.c 純 C／Metal 引擎原生執行 O87 已驗收的 Pika H3 Flex 生產配方
（1344×768、124 幀、24fps、6-step），目標單鏡 wall time 與畫質超越 O87 基準
（1344×768 單鏡 305 秒），且與 Strata 共存穩定。不重複標準 ComfyUI H3／FastH3／
MPS 基準測試。

## 2. 上游能力矩陣（實讀原始碼判定，非 README）

| 能力 | h3.c main 8974cc0 狀態 | 證據 |
|---|---|---|
| Metal 原生推理（bf16） | ✅ 已實作 | h3_metal.metal、h3_dit.c |
| INT8 量化 DiT（grouped int8，自載入時量化） | ✅ 預設啟用；`--use-slower-bf16-*` 為退回開關 | h3_dit.c、CLI usage |
| M5 專屬快速路徑（qkv/fc1 矩陣乘、row-scale int8 FC2） | ✅ `--use-int8-row-fc2` | CLI usage、profile 旗標 |
| 文字編碼 Qwen3-VL TE（bf16，含 vision span 拼接） | ✅（非 NVFP4） | h3_text_encoder.c:478–643、h3_vision_encoder.c |
| 影片 VAE 解碼 | ✅ FP32 | h3_video_vae.c:122–213 |
| 音訊 VAE 解碼 | ✅ | h3_audio_vae.c |
| T2V／I2V（first-frame）／Ref2VA（--ref-image/--ref-video）／FL2VA | ✅ 原生段落布局 COND/REF_IMAGE/VIDEO | h3_dit.c prepare_projection_maps、CLI flags |
| Sigma 排程＝均勻格點＋shift(video 12/audio 3) | ✅ | h3_host.c:150–165、H3_VIDEO_SIGMA_SHIFT=12（h3_host.h:12） |
| Flex knots 排程（1000/750/500/250/100/40/0） | ❌ 無 knots 模式 | 同上 |
| Renoise（步間注入新雜訊，stochastic） | ❌ 現行 euler 確定性 | h3_dit.c:1026 起、h3.c:1576 euler_preview |
| LoRA 載入／折疊（執行期） | ❌ 主程式無 loader；tools/ 折疊範例僅 PR#95（未併入） | grep 無 Pika/adapter 特例 |
| sol-attn BlockSparse | ❌ 上游 PR#86/#97 皆 OPEN | for-each-ref 確認 65 個 PR refs |
| SelfLift 兩段式（0.4→latent upscale） | ❌ 完全缺席 | grep selflift/upscale 無實作 |
| Latent upscaler（3d conv v1） | ❌ | 同上 |
| VAE int8/sol 加速变體 | ❌（僅 fp32） | h3_video_vae.c |
| SSD streaming DiT（BF16 逐層串流） | ✅ `--ssd-streaming`（與 int8-row-fc2 互斥） | h3.c:549–554 |
| 模型常駐互動會話（!cache/!session 指令） | ✅ | h3_cli.c |
| ffmpeg H.264 封裝 | ✅ | h3_ffmpeg.c、h3_video_encoder.c |
| 逐階段 profile 計時 | ✅ `--profile`（預設開） | CLI usage |

上游 PR 生態（全部 OPEN，僅供移植評估，不得預設引用）：
#85 seed barrier 修正、#88 int8 去噪快取、#90 adaLN 快取、#91 生成續跑、
#92 renoise 排程、#93/#73/#80/#78/#81 VAE 加速、#95 LoRA 折疊工具、
#96 條件快取鍵、#97 sol-attn 後續、#98 競爭修正、#100 fp16 路徑。

## 3. 本機資產盤點（2026-11-25 實測）

| 資產 | 路徑 | 大小 | 狀態 |
|---|---|---|---|
| 基幹 FL2VA 原版 bf16（13 shards） | SSD5 掛載點（**未掛載**） | 62G | BLOCKED→改由 HF 重下 |
| 基幹 Ref2VA 原版 bf16 | /Volumes/SSD2T/models/MiniMax-H3-ref2va/transformer（下載中） | 62G | 進行中（HF 公開、免 token） |
| TE Qwen3-VL bf16 | SSD2T/…/AI-h3-metal-MiniMax-H3/text_encoder | 62G | ✅ 現成 |
| VAE（video fp32 佈局）＋audio VAE | 同上 vae/、audio_vae/ | 9.7G+0.56G | ✅ |
| tokenizer/processor | 同上 | 小檔 | ✅ |
| Pika Flex comfy LoRA（fl2va/ref2va bf16） | SSD2T/…/Pika h3 flex加速模型/models/loras/ | 1.8G×2 | ✅ 現成 |
| ComfyX int8_convrot 基幹（pruned，50 blocks 完整） | …/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors | 20G | ✅（標頭解析：932 tensors，I8+U8 convrot+F32 行尺度，blocks 0–49 完整，hidden5376/qkv21504/fc1 28672 與 h3.c 常量一致） |
| Singularity ref2va int8 變體 | 同上 | 20G | ✅ 備援 |
| turbo_8step_hd LoRA | …/loras/ | 3.6G | ✅ 備援 |
| latent upscaler 3d conv v1 fp16/bf16 | …/models/latent_upscaler/ | 0.64G×2 | ✅ 本輪自 HF LBH-123-AI 取得 |
| SolAttn／sol VAE／NVFP4 TE（Comfy 加速变體） | 未在本機 | — | 不需要（h3.c 走原生路徑等效） |

Ref2VA 模型目錄（h3.c 佈局）已組裝：SSD2T/models/MiniMax-H3-ref2va/
Ref2VA/transformer(下載中)＋text_encoder/tokenizer/vae/audio_vae/processor
符號連結＋scheduler(s) 複製。

## 4. O87 生產配方契約（解自 SSD2T mac工作流/參考生視頻加速_Mac_MPS兼容版.json）

節點圖（語義）：
- UNETLoader `minimax_h3_ref2va_int8_convrot` → LoraLoaderModelOnly `pika_h3_flex_ref2va_comfy_bf16`(strength 1.0)
- CLIPLoader `qwen3vl_32b_minimax_h3_nvfp4_awq`(minimax) ＋ ConditioningZeroOut（CFG=1 下 negative 置零）
- MiniMaxH3ReferenceToVideo 1344×768、latent 幀 124、ref sizing "match"
- BlockSparseAttention sol-attn：τ=1.3、γ=0.2、top-p 1、block 12288/256、exact_kv_and_rows
- SelfLiftH3Sampler：seed 固定、cfg 1.0、ts=3、rho 0.4、0 段、wmin 0.5、wmax 1.0、upscaler 3d conv v1 fp16、音訊開
- BasicScheduler simple、6 步、cfg 1；KSampler euler
- VAEDecode（video VAE int8_convrot）＋VAEDecodeAudio（fp32）→ VHS_VideoCombine 24fps、yuv420p、CRF 19
- 幀數公式 `max(5, round(sec*24)) + (5 - …%17)%17`（5+17n 週期）

官方 Flex 取樣契約（Pika-Labs/Pika-H3-Flex ref2va/manifest.json＋scheduler configs）：
- knots [1000,750,500,250,100,40,0]；video_shift 12.0；audio_shift 3.0；method＝renoise（stochastic_sampling=true）
- LoRA：rank 128／fusion QKV rank 384／SwiGLU 半部重排；alpha 128；strength 1.0
- 基幹 revision：MiniMaxAI/MiniMax-H3 @ 42ed227e（comfy LoRA 即針對此原版訓練）
- 官方 transformer_ref 結構＝h3.c 常量：hidden 5376、ffn 14336、50 層＋2 refiner、head 128×56、text_dim 5120 ✅ 完全相容

## 5. 移植路線圖（每裡程碑＝單變因＋固定 seed＋實測表）

- **M0 端到端基線**：Ref2VA 原版＋h3.c 原生 int8，預設線性排程 20 步→小解析度
  （640×352、22 幀）產出真實 MP4＋ffprobe＋接觸表。目的：驗證引擎、TE、VAE、
  Ref2VA 段落、封裝全鏈路。
- **M1 Flex 排程＋Renoise**：新增 knots 排程模式（shift 套用於 knots 歸一 sigma）
  ＋步間 renoise（先以 diffusers MiniMaxH3Scheduler 語義移植，再用 ComfyX 節點源碼
  校核）。6 步跑通＝配方步數契約成立。
- **M2 LoRA 折疊**：移植 PR#95 折疊工具，擴充 ComfyUI 命名
  （`lora_diffusion_model_blocks.N…`、fusion QKV rank384、SwiGLU 重排）→
  折入原版 Ref2VA 產出 flex 權重（bf16 常駐版＋h3.c 載入時 int8 量化）。
  驗收：折疊前後對 3 個 block 抽樣 GEMM 數值對比 PASS。
- **M3 SelfLift 兩段式**：新增 `h3_selflift.c`：低解析度（0.4×）去噪→latent
  3d-conv upscaler（權重＝本機 fp16 v1）→高解析度尾段 ts=3（wmin0.5/wmax1、
  rho0.4 語義按 ComfyX SelfLiftH3Sampler 源碼逐行校核）。先 640×352→1280×704，
  後 1344×768。
- **M4 sol-attn**：基線先用 h3.c 原生稠密 attention 測 1344×768；若 attention
  占比>35% 再移植 BlockSparse（PR#86/#97 語義＋τ1.3/γ0.2/12288/256/exact_kv）。
  以實測決定是否納入，不合適即記錄 FAIL 理由。
- **M5 記憶體與常駐**：TE 編碼完成後釋放策略、DiT int8 常駐＋連續生成
  （互動 !cache）、VAE bf16 化評估、swap 護欄（目前系統 swap 檔案=0，jetsam
  風險高，護線必須保守）。目標：Strata(27G)＋h3.c 峰值同時在線不觸發護欄。
- **M6 ComfyX 整合**：h3.c 註冊為 ComfyX Mac 原生後端（複用 ComfyX SQLite／佇列／
  重試／成果管理；h3.c 以子行程執行，配方參數全存 ComfyX）。不新建排程器。

## 6. 驗收矩陣

| 項 | 標準 | 判定方式 |
|---|---|---|
| 端到端生成 | 真實 MP4＋ffprobe 幀數／解析度／fps 相符 | ffprobe 實測 |
| 配方一致性 | knots/shift/renoise/6步/LoRA/upscaler 語義逐條對照 PASS／PARTIAL／FAIL | 源碼對照＋數值抽檢 |
| 畫質 | 首／中／末＋等距關鍵幀接觸表；脸部一致性、無鬼影、肢體正常 | 視覺檢查（vision） |
| 效能 | 各階段 wall time（TE／DiT／VAE／封裝）＋總時；目標 beat 305s@1344×768 | --profile 實測表 |
| 記憶體 | 峰值 RSS、swap、Strata RSS／API 延遲同場記錄 | monitor.log |
| 穩定性 | ≥20 段連續生成無崩潰、無護欄觸發 | 批次實測 |
| 共存 | Strata API 正常回應全程 | 護欄探針 |

## 7. 風險與對策

1. **convrot int8 與 h3.c grouped int8 畫質差異**：以原版 bf16→h3.c int8 為準；
   如不足，評估 ConvRot→dequant→bf16 中間路徑（權重已在機，可離線轉換）。
2. **renoise 語義**：以 ComfyX MiniMaxH3 節點源碼＋diffusers scheduler 為唯一真相，
   禁止臆造公式；抽樣與 MPS 基線同 seed 逐幀對比。
3. **無 swap 檔 jetsam 風險**：護欄閾值 swap>45G 或 used>88% 或 Strata API 連續
   失敗→立即 SIGTERM h3.c 保留證據。
4. **TE 62G 常駐＋DiT int8 31G＋VAE fp32 10G≈103G＋Strata 27G**：接近物理上限；
   必要時 --ssd-streaming（bf16）或 TE 編碼後釋放＋重載策略，以實測擇一。
5. **GitHub git-over-HTTPS 掛死**：推播改走 Data API（Contents+PR），本地 git 照常。

## 8. 實測記錄（持續追加）

- 2026-11-25 02:47 下載路徑修正（檔名補零 bug 已修，13 shards 重下中）。
- 2026-11-25 upscaler 3d conv v1 fp16/bf16 自 HF 取得（0.64G×2）。
- （待续：E0 M0 小尺寸端到端 run 数据）
