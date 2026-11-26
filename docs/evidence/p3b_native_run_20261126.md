# P3-B 首次 native A8W8 run（2026-11-26）— 功能 PASS／內容 FAIL（全黑成片）

設備：Mac Studio M5 Max 128G，Strata 常駐未動。配方：`-d model_native`（convrot 單檔符號連結 + 原版 TE/VAE 符號連結）、橘貓prompt、640×352×25f、steps 2、seed 1、`H3_NATIVE_INT8=1`、無儀改 env、`--profile`。

## 判讀（FACT，log: MV_projects/h3c_native/p3b_native_640x352x25f.log）
- exit 0，寫出 mp4（34,831 B，1.625 s）；swap=0（/usr/bin/time swaps=0，取樣器 swap free 全程 >500M）。
- native 載入成功：無 out_proj/fc2 報錯；dequant 日誌只見 qkv/fc1（#1/#51/#101/#151 取樣）⇒ out_proj/fc2 走 native int8（INFERENCE：日誌缺失=未觸發 dequant）。
- profile 各段 wall：TE 212.0s（alloc 46.86G）｜DiT load 256.3s（alloc 97.81G，peak 41.88G）｜**denoise 26.23s / 2 steps = 13.1 s/step**｜audio VAE 1.3s｜video VAE 36.3s｜端到端 534s。
- 基線對照（bf16-dequant 同權重同尺寸）：去雜 117s（steps 3 ⇒ ~39 s/step）⇒ native **每步 3.0× 快**（FACT 單次測量；P6 重測確認，注意 dump 與 fast-path 差異）。

## 內容判定（FAIL）
- 抽幀 t=0/0.5/0.8/1.0：PNG 皆 1470 B 純平黑色（vision 判定 flat black，無任何內容）；同法基線幀 219,567 B 有內容 ⇒ **成片全黑 = 數值輸出 FAIL**。
- 11:57 舊煙測（13f，20,920 B）抽幀同為 1470 B 黑幀 ⇒ 黑輸出並非本次新引入，native 路徑自始產出無效 velocity（FACT）。
- `Failed to import MPS module` 噪音在 m0/m1/forward_dump 健康 run 中同樣出現，排除為回歸源。

## 目錄教訓（已入 skill）
- 首次啟動誤用 `MiniMax-H3-comfy-int8/FL2VA/transformer`（該變體 out_proj/fc2 本檔 BF16）⇒ native 硬fail `lacks sidecar`（log 存檔 p3b_wrongdir_bf16shards_rc1.log）。convrot I8 承載只存在單檔 `minimax_h3_ref2va_int8_convrot.safetensors`；native run 必須用 model_native 組裝目錄。
- 驗收教訓：exit 0 + 檔案大小 ≠ 內容正確——生成後必須抽幀 vision QA（黑幀 PNG 特徵：全同 1470 B 級別）。

## 下一步（進行中）
- P3-C arm1：native 全 tap dump（`evidence/native_dump_20261126`，背景中）vs 基線 `evidence/fulldit_dump_baseline`（已從 scratch 遷入永續目錄，1932 taps/5.3G）逐 tap 定位首偏離節點（候選：b0_out_proj_out、b0_fc2_out、final head、velocity）。
