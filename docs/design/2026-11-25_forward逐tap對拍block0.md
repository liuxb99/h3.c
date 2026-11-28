# Forward 逐 tap 對拍（block 0, step 0）— PASS

日期：2026-11-25。基準：feat/mac-pika-flex。方法：engine 加 `H3_DUMP_DIR`
env-gated 中間值 dump（預設路徑零行為變化，`h3_dump_dir()` 靜態快取 +
`if (h3_dump_dir() && !dit->dump_done)` 護欄），跑一次 tiny 生成落盤 29 個
tap（bf16/f32/f64 raw + manifest.tsv + row_map + rope 表 + latent/velocity）；
Python eager（torch CPU，h3ref venv）吃 **engine 實際權重 tap**（天然同
LoRA/int8 態）重放 block 0，kernel 級捨入規則復刻，ulp 步數對拍。
parent 獨立重放 `replay_block0.py` 與子代理報告逐數復現。

## Engine 跑法（EXIT 0，tiny.mp4 產出）

```
H3_DUMP_DIR=$D H3_DISABLE_NAX_LINEAR=1 H3_DISABLE_DIT_ACTIVATION_ALIAS=1 \
H3_DISABLE_INT8_ATTENTION_OUT=1 H3_DISABLE_FUSED_MLP=1 \
./h3 -d /Volumes/SSD2T/models/MiniMax-H3-comfy-int8 -p "a bridge over a stream" \
  --width 64 --height 64 --frames 25 --steps 2 --seed 1 \
  --use-slower-bf16-mlp --use-slower-bf16-qkv --use-slower-bf16-attention-output \
  -o $D/tiny.mp4
```

- DISABLE/slower flags：將 qkv/out/attn-out/MLP 從 int8-activation/NAX 快速
  路徑切到 plain bf16 kernel，eager 只需對拍 bf16 數學；int8 覆蓋 flag
  預設開啟，已逐檔 grep 核對預設語意。
- `--frames 2 --steps 1` 不可行：25 幀 chunk 與 steps∈[2,1000] 為訓練約束，
  最小可行為 25f/2 steps（FACT）。

## Tap 偏差表（gate：>0.5 值域步 = FAIL；ulp 列以值域最大元素歸一）

| tap | maxabs | engine \|max\| | 值域相對 | hits |
|---|---|---|---|---|
| b0_attn_adaln_out | 9.8e-4 | 9.19 | 1.1e-4 | 32/983808 |
| b0_qkv_pre_norm | 0.125 | 102 | 1.2e-3 | 3771/3935232 |
| b0_q_post_qknorm_rope | 0.031 | 11.31 | 2.8e-3 | 1206/1311744 |
| b0_k_post_qknorm_rope | 0.031 | 12.25 | 2.6e-3 | 1205/1311744 |
| b0_v_after_qkv | 0.125 | 49.75 | 2.5e-3 | 1330/1311744 |
| b0_attention_out | 0.125 | 43 | 2.9e-3 | 107464/1311744 |
| b0_out_proj_out | 4 | 776 | 5.2e-3 | 405574/983808 |
| b0_post_attn_hidden | 4 | 876 | 4.6e-3 | 152507/983808 |
| b0_mlp_adaln_in | 0.016 | 4.06 | 3.8e-3 | 108316/983808 |
| b0_fc1_out | 0.125 | 63.75 | 2.0e-3 | 2025177/5246976 |
| b0_swiglu_out | 4 | 3760 | 1.1e-3 | 1416642/2623488 |
| b0_fc2_out | 32 | 2.33e4 | 1.4e-3 | 575229/983808 |
| b0_block_out | 16 | 7.12e4 | 2.2e-4 | 222163/983808 |
| rope cos/sin（f32/bf16 表） | ≤1.2e-7 | 1 | 0 hits | 0/8784 |

判定：
- **第一個 >0.5 步 tap = `b0_attn_adaln_out`**（僅 32 元素翻轉，值域相對
  1.1e-4）。fp64 精確參考重算後 engine 仍剩 ≤2 個 ulp 翻轉 ⇒ 純 fp32
  捨入路徑差（Metal rsqrt 近似 + fma 鏈序 + double rounding），
  **非語意偏差**（INFERENCE，有 fp64 對照支撐）。
- 其後所有 tap 值域相對 ≤5.2e-3（≤0.25 步）：沿消除點累積的 ulp 翻轉，
  無 RoPE/布局/softmax/量化結構性錯誤。
- **無第一個語意偏差節點** ⇒ block-0 Linear+Attention+MLP 全鏈路數學與
  eager 一致。**PASS**。

## 順手修復（真實 bug）

sol-attn debug dump（block 2，舊 env dump）submit 後未呼叫 `h3_gpu_begin`，
任何開 dump 的 run 在 block 2 後必死於 "h3_gpu_begin() was not called"——
先前兩次 dump run 中途死亡之根因，非 MPS flake。本批 diff 已修。

## 邊界

- velocity 未做全模型 eager 對拍（需 28 塊全鏈重放，超出本級；已落盤
  `step0_velocity_bf16`、`step0_denoise_input_latents` 供下級用）。
- eager 復刻 engine kernel 捨入規則（bf16 落地點、fma 序）為近似復刻，
  故 gate 用值域相對步數而非逐位比對。
- int8-activation（A8W8）快速路徑本級刻意繞開；其語意驗收屬 metal8 任務。

證據檔（不入库大 raw）：`/Users/cyliu99/.hermes/cache/scratch/forward_dump/`
— 29 tap raw + manifest.tsv + tap_table.txt + replay_block0.py +
run_log.txt + replay_verify.log（parent 復現）+ tiny.mp4。
