# R4 spec — sd.cpp IQ1_S Metal kernel 移植源規格（事實，2026-11-28 摘自 sd.cpp @e239e4b 源碼）

## 量化塊結構（ggml/src/ggml-common.h:419-430）
```c
// 1.5625 bpw
typedef struct {
    ggml_half d;        // super-block scale (fp16)
    uint8_t  qs[QK_K/8];   // 32 B: 8 sub-blocks × 4B (32 elems × 1 bit each)
    uint16_t qh[QK_K/32];  // 8 × uint16: 每 sub-block 3 bit 高位 + 1 bit sub-scale 符號
} block_iq1_s;  // = 34 B / 256 elems（static_assert: 2 + QK_K/8 + QK_K/16）
```
- QK_K=256；每 sub-block 32 elems。**無 sidecar**：sub-block scale 由 `qh[ib]>>12` 的 1 bit 編碼：`dl = d * (2*((qh[ib]>>12)&1) - 1) * (1 + 2*((qh[ib]>>13)...`（逐字見 ggml-quants.c:2686 `dequantize_row_iq1_s`，NEON 與 scalar 變體要一起讀）。
- 值域 = codebook `kgrid_iq1_s`（32 entries，ggml-quants.c 常數表）× 符號位（qs bit）×（qh 低 3 bit 加成，sd.cpp 版語義以 dequantize_row_iq1_s 源碼為準）。
- row_size = ggml_row_size(IQ1_S, k) = (k/256)*34；k 必須是 256 倍數（assert），h3 全部 208 tensor 行均 3072/1024 等 ✓。

## Metal kernel（ggml/src/ggml-metal/kernels/mul_mv.metal + ggml-metal-impl.h:65-67）
- `kernel_mul_mv_iq1_s_f32_impl<N_R0_IQ1_S, args_t>`：每 thread 處理 **N_R0=4 個 src0 row-blocks**，threadgroup = **N_SG_IQ1_S=2** simdgroups；`_disp` 依 `ne11`（=matmul 的 M/ncols）選 N_R0 4 或 8（split 路徑 `N_R0_IQ1_S_SPLIT=8`）。
- src0=IQ1_S compressed weights 常駐（**kernel 內查表解量化＋乘加，無 BF16 展開**）；src1=f32 activations（token-major 打包）→ 直接可作 DiT 多 token matmul 的移植基礎：h3.c 現況是「bf16 權重常駐 matmul」，R4 = 加 `src0_quant=iq1_s` 變體 kernel＋host dispatch（借 ggml-metal `kernel_mul_mv_id` 的分派思路：按 ne00/ne11 選 N_R0）。
- 反量化在 kernel 內用 `device const threadgroup float const * grid`（kgrid_iq1_s 以 constant buffer 上傳）＋ sub-block scale 逐段乘。
- 掛點（h3c 側）：`h3_dit.c:124-132` 現行 linear 走 `h3_dit_mm_bf16`（h3_mps_encode.c）；R4 新增 `h3_dit_mm_q8src`（iq1_s src0 × f16/f32 src1）三條路徑之一：mm_iq1s_f16。
- 參考分派參數表：ggml-metal-device.m 對 iq1_s 的 support/pipeline 選擇（grep `GGML_TYPE_IQ1_S` ggml-metal-device.m）＋ type_traits row 873。

## 驗收門檻（R4）
1. 數值：同一 layer input 下，Metal IQ1_S matmul vs CPU(ggml dequant×matmul) vs sd.cpp ggml-metal 三方 max rel diff ≤ 1e-3（bf16 累加路徑容差）並附逐 tensor JSON。
2. 速度：對 blocks.0.attn.qkv_proj.weight（[3072,1024]）實測 ms/layer 三方對比。
3. 記憶體：僅 IQ1_S 3.645 GiB 常駐 Metal buffer（mmap→newBufferWithBytesNoCopy 或分期上傳），RSS+Metal 峰值記錄；禁整檔 BF16 展開。
4. 未達門檻項目標 NOT_TESTED。
