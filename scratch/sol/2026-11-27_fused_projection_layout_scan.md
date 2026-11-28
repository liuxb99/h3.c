# 融合投影(QKV/KV/fused)ROW-ORDER 全庫掃描 — 2026-11-27

方法(FACT):safetensors header 直接開檔(8-byte LE length + JSON,u16 multiset 比對),bf16 解碼 uint16→uint32<<16→float32;int8 convrot 依 h3c/docs/design/2026-11-25_convrot量化語意定案.md 以 weight_scale group256 反量化;行序判別 = 行身分 idfrac(slab 對位命中率)+ slabmean + 全矩陣 cos,參考基準 = HF 原序(interleaved,逐 head [q,k,v] 三元組)與其 grouped_from_interleaved 變體([allQ][allK][allV])。腳本/原始量測:scratch/sol/results2_L1.jsonl(77 列)、results2_TE / AD2 / PIKA / SUL2。CPU-only、唯讀。

## 主表(asset × 張量 × 判定 × 實測 cos)

| Asset(檔) | 張量 | 量化 | 判定 | 實測(G=grouped 重建 / I=interleaved 重建) |
|---|---|---|---|---|
| HF FL2VA shards(MiniMaxAI/MiniMax-H3/FL2VA/transformer) | blocks.*.attn.qkv_proj.weight(b0/25/49)+ token_refiner.* | 無(bf16) | **INTERLEAVED(原生)** | vs_I=1.000,vs_G≈0.011–0.015 |
| OFFICIAL(minimax_h3_ref2va_int8_convrot.safetensors) | main qkv b0/25/49 | int8 convrot | **KIND_GROUPED**(vs_G cos 0.99964 idfrac 1.0)= 引擎 rowmap 輸入格式 ✓ | I:0.0107 / G:0.99964 |
| OFFICIAL | **token_refiner.blocks.0/1.attn.qkv_proj.weight** | **bf16 passthrough(u16 multiset==HF,ms=1)** | **INTERLEAVED(==HF 原序,未隨 main 重排)** → 對 Comfy-core 分組消費者是錯置 | vs_I **1.000** idI 1.0 / vs_G **0.0153** idG 0.012 |
| Comfy-Org bf16(minimax_h3_ref2va_bf16.safetensors) | main qkv | bf16 | KIND_GROUPED(vs_G 1.0 ms==HF…ms=1) | I 0.0107 / G 1.000 |
| Comfy-Org bf16 | token_refiner qkv | bf16 | **KIND_GROUPED**(vs_G 1.000,ms==HF 1)| I 0.0153 / G **1.000** |
| zy pruned FL2VA(pruned_int8_convrot) | main qkv | int8 convrot | KIND_GROUPED(G 0.99996)✓ core;refiner bf16 GROUPED(G 1.000 ms==HF 1)✓ core | 同上族 |
| zy pruned REF2VA / FUSED_FLEXLORA / REQUANT_RT / SINGULARITY_INT8 | main qkv(convrot)+ refiner qkv(bf16)+ fc1(passthrough 保持) | int8 convrot + bf16 | **全 FAMILY = KIND_GROUPED**(main G≥0.9994 idfrac 1.0;refiner bf16 G 0.9994–1.0 idG 1.0;fc1 idfrac 1.0 保持)| QUANT-NOISE-OK 級 |
| VPipe fp8_scaled | main qkv(fp8)/refiner(bf16) | fp8 scaled | KIND_GROUPED(b49 G 0.99965 idG 1.0;refiner G 1.0 ms==HF 1)| b0/b25 出現 −0.9997 = 我方反量化 scale 符號工件(非行序問題,INFERENCE)|
| Sulphur-3(RELIANT I8/e4m3 變體) | main qkv/fc1 | int8/e4m3 | qkv **GROUPED**(vs_G idfrac 1.0;cos 低係列向未去旋轉,非行序)/ fc1 PRESERVED | idG 1.0 idI 0.012 |
| TE NVFP4(qwen3vl_32b nvfp4_awq) | visual.blocks.{0,13,26}.attn.qkv.weight | **視覺塔未量化(存 bf16 原序,無旋轉)** | **PRESERVED(逐行==TE_ORIG,ms err 為命名前綴)** | vs TE_ORIG identity cos **1.000** idfrac 1.0 |
| TE NVFP4 | merger.linear_fc1 | bf16 | EQUIV(n/a: not fused)保持(上下半近似自同,idfrac 0.5 屬矩陣本身)| cos 1.000 |
| TE_INT8CONVROT(VPipe-comfy_h3/text_encoders) | visual qkv | 反量化路徑未解 | **UNKNOWN**(header 宣稱 BF16 但 payload 尺寸=shape×2,dequant 無法 reshape)| — |
| TE lm q_proj(nvfp4 語言塔) | q_proj | nvfp4 | EQUIV(n/a: not fused,單一種類)| 探測 KeyError 中止,不影響判定 |
| ADAPTER pi_h3_2step_dwd_adapter_int8_convrot.safetensors | dwd_pixel_refiner.blocks.{0,7}.attn.to_qkv.weight | 混合 | **逐 head 三元組(INTERLEAVED 向)**:within384 塊內 q/k/v norm 比 1.479/1.265(bias 2.045)而三段子段比僅 1.116/1.059 → kind 相鄰於每 head 內(語義判定 INFERENCE;數值 FACT);AD2↔AD4 同序(slabmean 0.995/0.998 idfrac 1.0)| — |
| PIKA loras(flex v1 / ref2va_comfy_v1,lora_B [out,rank1024]) | blocks.*.attn.qkv_proj.lora_B | 無 | **UNKNOWN**(rank=1024 > 128×7,slab 判別度量退化,identity 與 grouped 重建 slabmean 同值)| 0.4467/0.4467 等 |
| Pika fused-flexlora 輸出(實測=FUSED_FLEXLORA asset) | main qkv convrot | int8 | **GROUPED** = 折疊工具輸出與 Comfy 分組族一致(FACT via asset)| G 0.99959 |

## 消費者端 FACT(行序預期)
- **Comfy core(piH3 安裝)**:`comfy/ldm/minimax/model.py:183` `q,k,v = qkv.split(heads*head_dim, -1)` = **kind-分塊(grouped)消費**;TokenRefiner→RefinerBlock→同一 Attention(class,行 278/302 區)= refiner 亦 grouped。全 `comfy/` 樹 grep `row_permut|rowmap|qkv_perm` 於 quant_ops/minimax = **0 個 qkv 行置換** → 載入即照檔案行序(FACT)。
- **h3c engine**:`h3_shaders.metal:3799 h3_qkv_rope_bf16` 以 `args.grouped` 切換;**refiner** 走 `h3_gpu_grouped_qkv_rope_bf16` → `h3_gpu.m:4020-4023` 傳 grouped=**1** = 逐 head [q,k,v] 三元組 = **interleaved 消費**;main int8 路徑先經 `h3_dequant_convrot_rows`(metal:1355-1381,comment 1345-1351:「檔案為 q/k/v 分塊、native 為逐 head interleaved,rowmap 還原為精確原始 BF16」),rowmap 僅在建於 convrot-I8 且 name 尾綴 `*attn.qkv_proj.weight` 時產生(`h3_dit.c:406-427`);bf16 passthrough 走 `h3_dit.c:1040-1047 load_block` 直存 `h3_safetensors.c:540-557`(grep permute/swap/interleav 於 h3_safetensors.c = 0 命中)→ **refiner 於引擎載入端無任何行置換(FACT)**。
- 故:**OFFICIAL convrot 檔 = 引擎族自洽(main grouped+rowmap→interleaved ✓;refiner interleaved ✓),但對 core grouped 消費者的 refiner 錯置(兩層 b0/b1 皆然)**;zy 系 grouped refiner 檔 = core 自洽、對引擎 refiner 錯置。

## 本地 adapter 轉換器審查(convert_minimax_h3_adapter_int8_convrot.py,73 行)
- **不重排/不 reshape qkv 行**:全文無 view/permute/chunk/cat;僅 L62-63 `scale.reshape(-1,1)`、L69 lora_A reshape 與 L63 `.sum(axis=1)` 為 norm 校驗。key_map 的 `fn`(行 49 `target = key_map.get(name, name)`;行 54-55 套 used)來自 `lakonlab_loaders.minimax_h3_to_diffusers`:其中 **qkv 無任何行重排 fn**——qkv 僅作為「來源為拆分 to_q/to_k/to_v」時的 kind-分塊 **offset 目標**(行 136-138 `(0,0)/(0,inner)/(0,2*inner)`),且對「替換融合張量切片」的 weight 直接 `NotImplementedError`(行 52-53);fc1 的 `swap_scale_shift`(行 140)只作用於非 Comfy 命名來源的 swiglu 半swap(非 qkv)。
- **輸出序 = 來源序( FACT)**;來源 adapter(diffsynth/LakonLab 訓練格式)的 `dwd_pixel_refiner...to_qkv.weight` 經 AD2 實測為逐 head 三元組(INTERLEAVED 向)→ 與其消費者 sol_h3(interleaved)一致 ⇒ **本地 adapter 對本缺陷免疫(FACT:轉換器不動行序;INFERENCE:來源本身已是正確序)**。

## mismatches_found(重點)
1. **OFFICIAL `minimax_h3_ref2va_int8_convrot.safetensors` 的 token_refiner qkv(bf16 passthrough)行序 = HF 原序 interleaved,與同檔 main(convrot 前重排為 grouped)及 Comfy-Org 自家 bf16 檔(refiner grouped)不一致;載入 Comfy-core(grouped 拆分、零置換)時 refiner q/k/v 錯置 = latent defect 確認存在。** multiset==HF → 非置換損壞而是 passthrough 未重排(依判定規則屬「順序不符」類,但語意上屬「未按家族規格重排」)。
2. zy 系 grouped-refiner 檔(pruned/requant/flexlora/singularity/VPIPE)若交 h3c 引擎載入,refiner 端(引擎要 interleaved)會錯置——引擎僅對 main-I8-qkv 有 rowmap 閘,refiner-bf16 無閘(FACT of 機制;實際錯置後果 INFERENCE,取決於各檔案實際載入路徑)。
3. TE 族無錯置;TE_INT8CONVROT 變體 dequant 語意未解(UNKNOWN);PIKA 純 lora_B 度量退化(UNKNOWN)。

## 引擎側 load-time permutation gate(INFERENCE 可)
可行且模式已在:仿 `h3_dit.c:401-461`(is_convrot_int8 偵測 + rowmap 產生 + `h3_gpu_set_rowmap` 注入),對 `token_refiner*attn.qkv_proj.weight` 增一分支:若 asset 族宣告 grouped refiner(如 zy 系)則載入時套用 grouped→逐head 置换(或等價地將 refiner wrapper 的 grouped flag 依 asset 切換 0/1,`h3_gpu.m:4023`);另一等效修法為轉換器端一次性把 refiner 行按目標消費者重排。Engine 現行對 OFFICIAL 檔自洽,是否加閘取決於要同時支援兩族資產。

（判定規則覆核:quantized 張量 vs_G cos≥0.999 → OK ✓ 全部通過;passthrough multiset==HF 而序不符者僅 OFFICIAL-refiner(對 core)與 zy-refiner(對 engine)兩向,詳上。）
