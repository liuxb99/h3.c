# P3-C 原生 A8W8 refine 全 NaN 根因報告（2026-11-26）

## 症狀
native A8W8（model_native_v2）tiny/各尺寸 run：refine text cond GEMM 輸出 107520 元素
100% NaN → 全黑成片。eager（BF16 反量化對照）同 prompt 同 seed 輸出乾淨。

## 判定（FACT，皆有實測支撐）
**根因 = `newBufferWithBytesNoCopy` 包裝 `MAP_PRIVATE` 檔案 mmap 的零權重 buffer，
在 native pipeline 使用後，CPU 指標視圖與 GPU 映射視圖分歧。**
引擎預設（M5 + 路徑含 /transformer/）對 transformer 權重走此零拷貝路徑
（h3_gpu.m `h3_gpu_tensor_new_file*`，`H3_ZERO_COPY_WEIGHTS` 控制）。

### 證據鏈（目錄 /Volumes/SSD2T/h3c_evidence_20261126）
1. tap=檔案：e0k `e3_condition_w` tap SHA256 `7c548546…` == model_native(_v2)
   transformer 檔 condition_proj.weight 資料區（同 offset 16158599815..16213650055）。
   CPU readback 看到的永遠是原始檔案 bytes。
2. CPU 寫不可見 GPU：CONDDBG 在 dispatch 前用 CPU memcpy 預填 sentinel 0x3F00×107520
   → kernel 跑完 sentinel=0、nan=107520：kernel 有執行、輸出全部寫入，但讀不到 sentinel。
3. kernel 正確：eager 基線 GPU out vs CPU 用 tap bytes 重算 GEMM：
   max|Δ|=512（大數值 bf16 1ulp 內）、mean rel 2.82e-3。native e0l 同 kernel
   CPU 重算乾淨、GPU out 全 NaN ⇒ GPU 執行時看到的 bytes ≠ tap/CPU 視圖。
4. 機制隔離：`H3_ZERO_COPY_WEIGHTS=0`（e0o，全部 RAM copy，其余變量不動，
   native A8W8 保持）→ out 100% finite，mean rel 2.82e-3 == 基線、max|Δ| 512 == 基線。
5. 時序假設排除：單 queue 單 CB FIFO；`H3_REUSE_MPS_COMMAND=0`（e0l_reuse0）
   與 =1 同樣全 NaN（正交）；CBDBG completion 回調時間線（新 binary）序正確。
6. eager 为何乾淨：同零拷貝 buffer 但純唯讀——GPU 首讀即檔案 bytes，無分歧窗口。
   golden/eager 與 native 的差別正是「native pipeline 會動這些 tensor」。

## 修復（保 A8W8，不退回全量 BF16）
h3_gpu.m 載入閘門：native（H3_NATIVE_INT8 設定）時禁用 **隱式** M5 zero-copy 預設，
transformer 權重回退 RAM 拷貝（+31.7GB resident，128GB host 可承受，swap 峰值對照 e0o 正常）；
eager 唯讀路徑零拷貝利益保留；`H3_ZERO_COPY_WEIGHTS=1|transformer` 顯式 opt-in 不變。

## 永久回歸閘門
`scratch/sol/regress_p3.sh` + `regress_taps.py`：run 判 PASS 需同時滿足
mp4 產出、每個 tap dump 全有限（禁 kernel 未執行/未寫入/非有限判成功）、
LINEARDBG dispatch 證據、CONDDBG out nan=0、日誌無 Metal command 錯誤。

## 教訓
- NoCopy+mmap buffer 只可用於「建立後永不寫入（CPU 或 GPU）」的 tensor。
  任何 write_bf16/GPU 覆寫路徑碰到 borrowed buffer 必須先實體化（blit/copy）。
- CPU readback tap 驗證的是 CPU 視圖，不等於 GPU 執行時視圖；數值對帳要在
  「GPU 可見存储」上做（RAM copy 或 GPU 端 checksum kernel）。
