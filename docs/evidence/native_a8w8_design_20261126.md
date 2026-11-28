# h3.c Metal 原生 A8W8 高效推理引擎 — 設計文件（Phase 2）

日期：2026-11-26　基線：PR #1 commit 7248ed5（分支開發，PR 保持 OPEN，不動 BF16 fallback）

## 0. 已固化前提（實測證據，非假設）

- convrot INT8 檔案存的是**旋轉後的權重** W_rot = W·H_g^T（normalized Hadamard, groupsize=256；
  `h3_shaders.metal:1343-1383` dequant kernel 與 finalized 語意）。檔案權重需做 per-head row
  permutation（grouped q|k|v → per-head interleaved，`h3_dit.c:390-410`）。
- 當前 sol-attn INT8 路徑是 **dequant→GPU requant（非旋轉空間）**，非原生；原生模式必須
  讓 activation 先乘 H（同一 256-group 布局）再量化，才能直接吃 W_rot 的 INT8 rows。
- 引擎 sol 量化發生在 `h3_dit.c:1222`（每 block、每 step 都 requant  activation）。
- QKV requant-vs-原始權重 noise = 2.3e-2 rel（bf16 e-mirror 校準實測）；block0 BF16
  eager-vs-engine tap 每層 rel 2.3e-2…6.1e-2，chain 發散由此複利——drift 根因已定位為
  **bf16 kernel 邊界捨入 + activation requant noise 逐層複利**，不是玄學。
- 四路徑對照工具：`docs/evidence/forward_dit_20261126/replay_full.py`（e 鏡像，已驗收）＋
  `scratch/metal8_p1/replay_prec.py`（PREC=r/g/e 階梯：FP32 裸参照／只捨入 matmul 出口的
  G 參照／引擎鏡像 e；anchor＋chain＋純 CPU e2e 去 engine-tap 循環）。e 檔與已驗收基線
  分數逐位一致。

## 1. Apple GPU INT8 調研結論（Phase 2.3 要求）

- Metal Shading Language SIMD-group matrix（`simdgroup_matrix<T>`）**只支援 half/bfloat/float
  8×8**，不存在 INT8 simdgroup matrix——不得假設 NVIDIA 式 INT8 Tensor Core。
- Metal 4 TensorOps `matmul2d`：M4 Max（無 NAX）上為 shader-core 模擬（Rigel,
  arXiv:2606.12765：fp8 matmul2d 僅 0.94× fp16 吞吐，無專用矩陣資料路徑，≥fp32 累加）。
  M5 Max 有 NAX（MPP Programming Guide：M5 GPU neural accelerators），本引擎既有 INT8
  matmul2d kernel 已在實機走 NAX——原生 INT8 GEMM 的唯一正路是 **NAX matmul2d（int8
  operands, fp32 accumulate）**。
- 候選路徑（全部要 A/B 實測，採最快且畫質合格者）：
  (a) NAX int8 matmul2d（A8W8，本設計主路）
  (b) 融合 dequant-GEMM：int8 權重 on-the-fly 解量化進 bf16 tile 再 matmul2d bf16
  (c) 純 BF16 matmul2d（現行 fallback，基準線）

## 2. 原生載入契約（Phase 2.1）

- DiT 線性層 QKV／attn-out／FC1／FC2 直接以 INT8 rows + per-row f32 scale 常駐 GPU，
  **禁止整份 dequant 成 BF16**。僅 TE／VAE／patchify／非量化層維持各自 dtype。
- 上傳時一次性把 convrot row permutation 套到 int8 rows 與 scales（150 MB 級 CPU 搬移，
  之後 GEMM 直接吃 row-major，輸出天然為 per-head interleaved，與現行 tap 布局一致）。
- `h3_weights.c` 需新增 native-int8 materialization 分支（現在 :372-445 一律 dequant）。

## 3. 新 kernel 計畫（Phase 2.2/2.6，融合優先）

1. **fused rot-quant kernel**（缺的就是這個）：bf16 activations → 每 256-group 在
   threadgroup memory 套 normalized H₈⊗H₃₂（FWLT 8 輪蝶形，複用 dequant 的 had 表）→
   per-row absmax reduce → 動態 INT8 量化輸出 rows＋scale。讀 x 一次、寫 int8 一次，
   零中間 bf16 buffer。旋轉額外算 ≈ K_rotation/N（QKV 4.7%、fc2 12.5%，相對 GEMM）。
2. **NAX int8 matmul2d GEMM**：A=int8 rows(動態), B=靜態 int8 權重 rows, C=f32 累加；
   既有 sol int8 GEMM kernel 已具雏形，參數化 K 至 5376 全尺寸。
3. **融合 epilogue**：C_out = acc × scale_row(a) × scale_row(w) + bias，直接在 GEMM
   kernel 尾段寫出（bf16 或 f32 按圖譜），砍掉獨立 scale kernel 與中間 buffer。
4. 非量化層（norm gamma、mod table、adaLN、rope、head）維持現行 bf16/f32 kernel 語意。
5. 量測後若 rotation 成本过高，備選：(b) 融合 dequant-GEMM 把 H⁻¹ 塞進 GEMM 前 tile
   變換（權重不落地 bf16，僅 per-tile 暫存）。

## 4. 精度閘（Phase 1 要求，生產驗收前結案）

- 每階段以 replay_prec 階梯對照：FP32 裸參照（r）→ G 參照 → engine bf16（e）→ A8W8。
- 驗收一律 replay **自算 velocity** 做 scheduler 演化（e2e 模式），禁止 engine velocity
  tap 餵自己；velocity/latent 誤差與 engine tap 對照僅作事後比較。
- drift 歸因表：捨入（e−g 差）、量化（A8W8−e 差）、attention/residual/實作差異逐層分離。
- 量化 noise 上限假設（待 chain 數據結案）：QKV 單層 2.3e-2 類比 noise 在 50 層×N step
  複利即 b45+ chain rel→O(1) 候選根因。

## 5. Benchmark 契約（Phase 3）

- 640×352×13f／544×288×124f（對照 Mac MPS 基線）／1344×768×124f（生產配方）。
- 記錄：載入時間、每 block/step、DiT/TE/VAE 分段、GPU util、peak memory/swap/memory
  pressure、冷啟動 vs 熱機、成片＋關鍵幀。instrumentation/dump run 一律不計入生產速度。
- 生產速度只許來自 `--benchmark`/clean run 路徑。

## 6. 生產配方整合（Phase 4）與畫質閘（Phase 5）

- 逐項移植 ComfyX 已固化 Pika H3 Flex 配方：SelfLift、兩段式 latent、sol-attn、VAE 加速、
  正確 scheduler（flow-matching euler，t 序列以 tap `s{step}_timestep_f32` 為準）。
- 禁止以減幀、降解析度、減步數、換低品質模型冒充加速。
- 畫質：同模型/seed/輸入/配方與 ComfyUI(ComfyX) 參考成片逐關鍵幀對照；無視覺能力時
  標 VISUAL_UNVERIFIED，不得冒充 PASS。

## 7. 分步落地順序

A. ladder 跑完（r/g/e anchor+chain、e2e×2）→ drift 歸因初報。
B. A8W8 chain dump（需 ssd-streaming＋sol-attn 開；與 e 同 seed/配方）補第四組對照。
C. fused rot-quant kernel ＋ native int8 upload（env gate：H3_NATIVE_INT8=1，不動 fallback）。
D. NAX GEMM 全尺寸＋融 scale/bias；單層對 e dump tap 驗收 rel 閘（目標 ≤ A8W8 量化 noise 预期）。
E. 全鏈開關對照（同 run 只切 native）→ benchmark → 配方移植 → 成片畫質閘。
F. 證據、工具、kernel、語意逐 commit 固化；skill 更新。
