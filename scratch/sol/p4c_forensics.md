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

## 進行中
- 獨立 Golden：piH3 8390 golden_cat 512×288 39f 4-step seed1 同 prompt（第 2 次提交已收 got prompt；首提因 json 雙層 wrapper 被拒）。產出後 vision 判語義（貓 vs 字幕）→ 定案前端契約。
- 子代理：A requant 契約 / B 幀契約引文 / C prompt 條件化 diff / D refiner 重放（全部 CPU）。
- GPU 佇列：golden 完成前不啟動引擎重跑（串行規則）；chain_p4c.sh 待 refiner culprit 定案後改版再跑（原 block0 目標已被 refiner 首偏離取代）。
- commit 7d581c4 = tap 名稱修復（僅 dump 路徑，零生產風險）。PR #1 維持 OPEN。
