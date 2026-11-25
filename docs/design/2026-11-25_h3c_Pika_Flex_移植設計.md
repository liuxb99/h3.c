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

## v2.1 調研補充（2026-11-25 M0 前後）
- **模型盤佈局（實測）**：原版 FL2VA 全套 134G 已在 `/Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3/FL2VA/`（transformer 13 shards+TE+video_vae+audio_vae+tokenizer）；Ref2VA 62G 於 `/Volumes/SSD2T/models/MiniMax-H3-ref2va/`，以 `Ref2VA` 符號連結掛入 h3.c model dir。`./h3 --info` 全過（535 tensors×13 ✓）。
- **`--render-width/height` 現狀**：僅把 conditioning+latent canvas 縮到 render 尺寸，輸出=render 尺寸，**無任何 latent/pixel 上採**——達不到 PR#97 語義，M3 latent 上採器移植必須自建。
- **h3.c 無 sol-attn/tau 任何實作**（grep 零命中）；P2 連 CLI 標幟（`--sol-attn --tau`）一併新增。
- **LatentResizer3D 架構（ground truth 歸檔 `docs/upstream_refs/selflift/h3_upscaler.py`）**：24ch in→channels(由 conv_in 頭探測) 3d-conv；in_blocks 低解析段 → 中段 trilinear(align_corners=False) → out_blocks 高解析段 → conv_out；scale-emb=Linear(1→64)→SiLU→Linear，注入每個 ResBlockEmb3D（emb→2C 拆 scale/shift，`out_norm(h)*(1+scale)+shift`）；TemporalConv=GroupNorm→SiLU→dwconv(T,k,groups=C,pad=k//2)→pwconv(零初始化)→殘差；GroupNorm 固定 32 組；時間 chunk=32/overlap=5 線性 blend（首版可關 chunking，latent_t≤5 時本就單段）。
- **記憶體共存策略**：TE bf16 62G 常駐 + int8 DiT 31G + VAE 10G ≈ 104G > (107.5G GPU 集上限 − Strata 佔用) ⇒ **M0/M1 走 `--ssd-streaming`**（DiT 權重 SSD 直讀不入 GPU 集，720MB/s）；resident int8 產量模式需在 Strata 空閒窗口或 TE 釋放方案後驗證——不動 Strata（用戶政策）。
- **折疊工具驗證**：合成單測 PASS（delta=scale·B@A、scale=strength、rank==alpha 慣例；殘差 max 0.0156=bf16 舍入）。

## 3. 本機資產盤點（2026-11-25 實測）

| 資產 | 路徑 | 大小 | 狀態 |
|---|---|---|---|
| 基幹 FL2VA 原版 bf16（13 shards） | SSD5 掛載點（**未掛載**） | 62G | BLOCKED→改由 HF 重下 |
| 基幹 Ref2VA 原版 bf16 | /Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer（13/13 標頭健檢 PASS，見 evidence/資產溯源） | 62G | 已到位 |
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
- **M1（降級為選用實驗）**：實讀 Comfy 源碼判定：O87 MPS 配方
  BasicScheduler(simple)＋ModelSampling(shift 12)＋KSampler(euler) ≡
  均勻 sigma 格點＋shift 變形＋確定性 euler ＝ h3.c 預設排程，語義已等價 ✅。
  官方 Flex knots[1000,750,500,250,100,40,0]＋renoise（k_diffusion
  `sample_dpmpp_2s_ancestral_RF`，renoise_coeff=√(σ_{i+1}²−σ_down²α_{i+1}²/α_down²)）
  僅為官方運行變體，列為可選 A/B，不是阻塞項。
- **M2 LoRA 折疊**：移植 PR#95 折疊工具，擴充 ComfyUI 命名
  （`lora_diffusion_model_blocks.N…`、fusion QKV rank384、SwiGLU 重排）→
  折入原版 Ref2VA 產出 flex 權重（bf16 常駐版＋h3.c 載入時 int8 量化）。
  驗收：折疊前後對 3 個 block 抽樣 GEMM 數值對比 PASS。
- **M3 SelfLift 兩段式**（ground-truth：facok/comfyui-SelfLift@master 已存
  `docs/upstream_refs/selflift/`，語義逐行摘錄如下）：
  1. 低解析度latent（round(H·scale 取偶、≥2；trilinear 下採樣初始 latent；
     音訊流不動）以 `sigmas[:ts+1]` euler 去噪到 σ_{ts−1}，記錄末次評測的
     x0 預言（Eq.3 端點，其 euler 更新丟棄）。
  2. 轉換：z0_low(VAE latent 域)→雙_lift：z_lat＝外部 3d-conv upscaler 昇解析
     （latent 域 route，rho=0 預設，免 VAE 來回）；z_pix＝VAE 解碼→像素昇解析→
     VAE 重編碼（route 像素，rho>0 混合需要）；`artifact_aware_consistency_lift
     (z_lat,z_pix,rho,w_min,w_max)`      依風險圖混合。h3.c 移植順序：先 latent-route
      （rho=0，免 VAE 來回），像素 route 列後續實驗。注意：O87 生產節點參數為
      rho=0.4（混合路線，含 VAE 來回），故 rho=0 首版屬**記錄在案的偏差**，
      畫質驗收須與 O87 成片對比後決定是否補像素 route。
  3. 再噪：x=noise_scaling(σ_{ts−1}, N_{seed+1}, z0_high)→單步 euler 到 σ_ts
     →inverse_noise_scaling→高解析度以 `sigmas[ts:]` 續跑（NFE 守恆：總評測
     步數＝原排程）。
  4. 驗收：先 640×352→(scale=0.5)1280×704 小尺寸，後 1344×768 目標尺寸。
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
