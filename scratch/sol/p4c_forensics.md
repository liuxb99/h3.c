# P4C forensics — 640×352 s20 seed1 同契約兩臂對比（2026-11-27）

## 執行與版本
- 腳本 scratch/sol/chain_p4b.sh（shader symlink 統一＋版本戳 echo）；兩臂 640×352 frames25(→39f 契約) s20 seed1 串行完成（10:52–11:28）。
- 證據 EV=/Volumes/SSD2T/h3c_evidence_20261126/p4b_{eager,native}640_taps/（run_log.txt 內含 git_head、shader_sha256、binary_sha256、完整參數）。
- 注意：本輪 tap 名稱有 bug（selective 模式缺 block/step 前綴→b0_* 覆寫成最後 block/step），已修 commit 7d581c4；基礎設施 tap（e*/step0_*/rope/layout）名稱不受影響且全部有效。

## 影片語義判定（vision 逐幀）
- EAGER(原權重轉換) 640：全片 = 灰棕高頻噪點＋微弱色調不均，無貓/蝴蝶/草地/文字 → FAIL（引擎內 eager 從無有效 golden；與 11/26 綠底漸層 golden 同類）。
- NATIVE(INT8) 640：構圖完整、動作連貫的「兩位女性室內餵食黑色動物」生活片段＋底部燒錄中文 prompt 字幕（内容≈逐字 prompt）。語義 FAIL（無橘貓/草地/蝴蝶）、字幕=非預期、但時序/幾何 OK。場景與 earlier native-tiny 輸出近似 → prompt 近乎不影響場景（條件化失效特徵）。
- 兩臂 mp4 皆 640×352×39f：frames 25→39 = VAE chunks*17+5 ＋ ffmpeg `while(fc%17!=5) fc--`（h3_video_vae.c:967 / h3_ffmpeg.c:360），與 ComfyX 公式 (5-(x%17))%17 舍入一致 → 39f 屬模型契約（INFERENCE，待子代理 B 引文定案）。

## Tap 數值法醫（eager vs native，bf16 解碼 CPU）
完全一致（bit-equal）：e1_text_source(20x5120), e1_after_cond(20x5376), e3_condition_w/b/source_gpu, e0_video_input, e0_audio_input, step0_denoise_input_latents, b0_layout_positions(f64), b0_rope_cos/sin_table。
偏離鏈（按流水線順序）：
1. e1_after_refiner0 cos=0.693 ← 首偏離（輸入 after_cond 逐位相同！）
2. e1_after_refiner1 cos=0.676
3. e1_after_finalnorm(=e0_refined_text) cos=0.414, mad 78.5
4. e0_hidden_packed cos=0.9953（僅 text 20/2790 行受染）
5. step0_velocity cos=0.671
次要：e2_video/audio_patch_w/b cos≈0.9997（量化噪點級，非根因）；b0_swiglu_out(=block49 s19 覆寫名) cos≈-0.0015（DiT 輸出兩臂完全不相關——下游全髒，不能歸因 DiT kernel）。
推論：兩臂共享轉換後 condition/DiT 權重路徑（e3_condition_w_gpu 逐位相同）→ 先前 CPU 審計的 condition relRMS 0.23/0.27（轉換 vs 原始）為兩臂共享缺陷，不是臂間判別因子；refiner 才是臂間判別點 → 子代理 D 以 FP32 CPU 重放 refiner(原 HF 權重為 oracle)定位 culprit。

## 獨立 Golden（FACT，語義 PASS）
piH3 8390（官方 ComfyX 節點＋LakonLab pi_h3_4step DWD int8 adapter）同 prompt、512×288、length 39→輸出 39f、4-step、seed 1：逐幀 vision = 橘色虎斑小貓＋粉紅蝴蝶＋綠草地＋卡通風格＋晴天、無任何文字、動作連貫 → 原生引擎的 prompt 字幕化與場景偏離 = 引擎缺陷，非模型行為。證據存 EV/golden_comfy_official/（mp4 sha 88c375e8493ba65e…＋workflow＋NOTE.md）。另證 length≡5(mod17) 為官方幀契約（39 直接被接受）。

## 子代理 A 報告（已收）＋引擎繼承驗證
- relRMS 0.23/0.27 真相（FACT）：指標口徑 = max|Δ|/RMS（tap_compare2.py:80-82 峰值口徑）× pass-through 張量有損 requant 往返；真實 RMSrel≈0.02、cos 0.9997。非 kernel bug、非去相關。
- 損傷面（FACT）：convrot 檔 279 張 pass-through ≠golden；video/audio patch w+b、adaln bias 32、time/final heads 損傷仍在；**token_refiner qkv 兩張整張報銷**（RMSrel 1.40、cos 0.015，對 rot 域亦不相關）。
- 引擎繼承驗證（FACT，本次）：model_native/FL2VA/transformer 的 token_refiner 張量對 convrot 檔逐位相同（eng-convrot RMSrel=0.0000）、對 golden qkv=1.40/out_proj=0.035/norms≈0.004 → **引擎 refiner 權重直接繼承 convrot 轉換損傷**，坐實 tap 首偏離=refiner 與兩臂同 off-prompt。bf16 fallback 記憶體代價：0（本即 bf16 直存）。
- 全量掃描進行中（engine transformer 對 golden，damaged>1e-3 清單＋convrot 溯源旗標 → 產生可 patch 集）；patch 將做位元區間備份＋manifest，P4E 鏈完成後才套用以避免 mmap 混視。
- 修復契約（A 提，待批格式）：不合格 int8 序列化的張量原 dtype 直通、禁 requant 往返；export 後 recon8 邏輯 audit；指標併報 RMSrel+cos。

## P4E 三語義消融（FACT，256²×22f×2step native INT8，seed1）
- 三 prompt（corgi鋼琴／霓虹招牌 OPEN 24 HOURS／中文茶座水墨）影片全部語義 FAIL：皆為同一泛型扭曲人形＋紅腫臉，無 corgi/鋼琴/招牌/茶座，且**連字幕都沒有**（與 640 臂的字幕化不同；2-step 小契約下連文字渲染能力也不存在，對招牌正向對照解釋力有限——標記為契约過小的混杂因素）。
- tap 級判别（FACT）：跨 prompt 相似度 e1_text_source raw cos≈0.998（公共模態主導）→ 去均值後 p1~p2 0.28、p1~p3 −0.80、p2~p3 −0.80 = **TE/tokenizer 路徑在 tap 層確已產出 prompt 專屬內容**（H2「tokenizer 漂移致文本退化」進一步排除）。
- e1_after_refiner0 跨 prompt cos 塌到 0.71–0.73、finalnorm 0.39–0.41：與「報銷的 refiner qkv 把 conditioning 信號搅成垃圾」一致（金標refiner 對同 prompt 的跨語義對照待 D 的 CPU oracle 復放補齊）。
- token 數随 prompt 長度變（text_source 12/22/…行）為契約行為，非 bug。

## 子代理 B/C 報告（已收）
- (B) 幀契約 = FACT 符合官方契約：h3_host.c:15-21 align ≡5 mod17（25→39）、latent_t=((39-5)/17)*5+2=12、audio_t=round(39*40/24)=65、序列 20+130+2640=2790、VAE chunks*17+5=39；與官方 nodes_minimax_h3.py:50-63 + ldm/minimax/vae.py 三式逐字相同（17k+5 網格：22/39/56；25 非網格值，引擎向上取整是官方行為）。無需修復。
- (C) 前端差異清單：chat template 兩邊皆無（排除）、TE 第50層 unnormalized 相同（排除）、modality tags/attention 無 mask/cond pin timestep/shift 同式（排除）。實際差異：D1 引擎無 CFG（cond_scale≡1 單 pass）vs 官方 CFG>1 慣用（H1）、D2 sigma ladder 端點差異、D3 tokenizer 實作不同、D5 256×256 rope 0.5 特權路徑。註：native 輸出字幕≈逐字 prompt → text token 內容路徑存活（H2 tokenizer 漂移降級）；結合首偏離=refiner，主嫌仍是 text conditioning 流在 refiner 後被破壞（CFG 為次）。
- P4E 已啟動（GPU）：native INT8 tiny 契約 × 3 語義 prompt（corgi 鋼琴 / 霓虹招牌 "OPEN 24 HOURS" 文字正向對照 / 中文茶座水墨）＋ rb* refiner subtaps。招牌正向對照：若招牌文字正確渲染而場景 prompt 偏離 → 文本路徑活、場景條件化斷。
- dump 政策修復（commit 見下）：rb subtaps 於 prep(dump_step<0) 會被 selective 政策抑制 → h3_dit.c:2494 rb 前綴 explicit allow ＋ infrastructure 旗標加 rb（bypass quota）；helpers_dump.c 為未編譯草稿。

## 進行中
- GPU 佇列：golden 完成前不啟動引擎重跑（串行規則）；chain_p4c.sh 待 refiner culprit 定案後改版再跑（原 block0 目標已被 refiner 首偏離取代）。
- commit 7d581c4 = tap 名稱修復（僅 dump 路徑，零生產風險）。PR #1 維持 OPEN。
