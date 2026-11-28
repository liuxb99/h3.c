# 全 DiT eager 重放對拍 PASS（50 blocks × 2 steps + final head + Euler 鏈）

日期：2026-11-26｜設備：Mac Studio M5 Max（torch 2.15 CPU eager，h3ref venv）
權重態：`/Volumes/SSD2T/models/MiniMax-H3-comfy-int8`（int8 convrot Ref2VA，LoRA 已折，bf16 計算）
DUMP：引擎 `H3_DUMP_DIR + H3_DUMP_ALL`（本 commit 儀改 `h3_dit.c`），25f×2 steps 最小合法 run，全 50 blocks × 2 steps 逐 tap（hidden_in/adaln/qkv_pre_norm/q,k post-qknorm+rope/v/attention_out/out_proj/post_attn/fc1/swiglu/fc2/block_out + final_layer 五 tap + 各步 velocity/latent）。

## 判定（PASS）
- **語意判定 = PASS（FACT）**：anchor 模式（每 block 直接重載引擎 `bNN_sS_hidden_in` ground-truth，無累積）逐 block 最大 rel 在 **全部 50 blocks × 2 steps 平坦於 5.7e-2–1.1e-1**（階梯為平台非爬升），與已定案的 bf16 requant 噪聲底（bank：qkv 2.3e-2 / out 2.7e-2 / fc1 3.4e-2 / fc2 3.5e-2）同量級 ⇒ 無語意偏差節點。
- **Final head = PASS**：`final_video_norm` rel **1.0e-2 (s0) / 1.4e-2 (s1)**；`velocity_bf16` rel **2.57e-2 (s0) / 3.21e-2 (s1)**（cos 0.9997/0.9995）——皆為 bf16 GEMM 噪聲底 ⇒ velocity 計算與引擎一致（INFERENCE： plateau 歸因於 requant/捨入，已由 bank requant 數與 calib 佐證）。
- **Euler 鏈**：以引擎 velocity tap 推進 latent，step0 rel 2.3e-8、final rel 4.6e-8（註：速度取自引擎 tap 故此項為同義反覆，真正的驗收是 velocity tap 本身的 rel，見上）。
- **Scheduler**：video sigma 公式（shift=12 time_shift）給 [1.0, 0.9230769231]，引擎存 [1.0, 0.923076928]＝同一數的 f32 表示 ⇒ 公式配對成立；引擎 audio 路徑 sigma1=0.75（非 video sigma1 再 time_shift），記錄為引擎既有行為。
- **chain 模式（drift 預算，非語意訊號）**：自餵 block_out 時 rel 隨深度線性爬升至 b45+ 約 0.8–1.0——50 層 residual 流的 bf16 累積（Metal 與 eager 捨入次序不同所致），與 anchor 平坦曲線互證：偏誤源為逐層噪聲累積而非結構錯誤。

## Final head 語意定案（源碼 h3_dit.c:2800–2865 + h3_shaders.metal:3095）
- LN gamma = `final_layer.norm.weight`（ckpt 本體 BF16）；dump tap `s0_final_video_w` 實為 `final_layer.video_out.weight` 的截斷 dump（命名陷阱）。
- AdaLN：FINAL_SLOTS=2，shift_slot=0 / scale_slot=1，per-token slot 由 `final_video_map_u32` gather（video rows 135–183；觀測 map＝0(s0)/1(s1)）。
- velocity = `final_video_norm @ final_layer.video_out.weight[96×5376]ᵀ + bias(96)`；GPU 端權重 dtype 由 tap 無法区間，重放雙候選取優 ⇒ **w_f32**（兩步同選）。
- 首 block calib（block0/step0 全 tap 2.0e-2–6.1e-2）與舊 b9c998a 單 block PASS 相接。

## 重產
```
H3_DUMP_DIR=<dir> H3_DUMP_ALL=1 ./h3 -d /Volumes/SSD2T/models/MiniMax-H3-comfy-int8 \
  -p "..." --width 640 --height 352 --frames 25 --steps 2 --seed 1 --skip-vae ...
H3_REPLAY_MODE=anchor python3 replay_full.py   # 語意判定
H3_REPLAY_MODE=chain  python3 replay_full.py   # drift 曲線
python3 headcheck.py                            # 單獨驗 final head
python3 crosscheck_v2.py                        # ComfyX/comfy_kitchen 第三方單點對拍
```
（replay 需 h3ref venv：torch 2.15 + numpy；tap 目錄路徑見腳本頂部常數。）

## 檔單
`replay_full.py`（anchor/chain 雙模式）、`headcheck.py`、`crosscheck_v2.py`、`metrics_{anchor,chain}.json`、`tap_table_{anchor,chain}.txt`、`anchor_run.log`、`chain_run.log`。
