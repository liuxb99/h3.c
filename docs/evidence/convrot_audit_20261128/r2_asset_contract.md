# R2 資產契約（2026-11-28，h3c-built 原生引擎實載）

## 三軌資產
| 資產 | 路徑 | 角色 | 引擎狀態 |
|---|---|---|---|
| HF golden | /Volumes/SSD2T/models/MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer | 數學 ground truth（bf16） | 僅供 golden/對拍腳本，非生產載入路徑 |
| 官方 ConvRot | /Volumes/SSD2T/models/MiniMax-H3-comfy-int8 | 官方 release（comfy quant line，含 FL2VA + nvfp4 TE） | 引擎既有驗證基線（loader 253 checks fails=0） |
| h3c-built | /Volumes/SSD2T/models/MiniMax-H3-h3c-built | 本倉庫 builder 重建資產（R1 gate PASS freeze） | R2 實載目標 |

## h3c-built 契約要點（FACT）
- 目錄鏡像 comfy-int8 佈局：convrot 單檔 I8（34.2G, sha256 dd098ba9…ffeeb, r-x 凍結）＋ FL2VA symlink（唯讀復用）＋ qwen3vl nvfp4 TE symlink（唯讀復用，引擎 single-file 模式硬性要求）＋ build_manifest.json v2 ＋ .ready。
- 引擎側語義（probe_sidecar 對拍證據 evidence/convrot_audit_20261128/probe_built_vs_official.txt）：built 與 official 逐 key 一致 — convrot=1、qkv/out/fc group=256、adaln group=64、payload=I8；token_refiner.* 兩邊同樣由 FL2VA/transformer bf16 副本優先解析（引擎既有行為，非 built 差異）。
- Loader 測試（h3_native_loader_tests <built dir>）：shards 發現、adaln/scheduler/vae/tap key 映射 253 checks fails=0，輸出與 official 逐行同構。
- `h3 --info -d <built>` 與 `--info -d official` 資產清單段 IDENTICAL（diff 空）。
- 語義鏈：built ≡ HF golden（gate rerun3：S250/Q200/L50/P285, failures=0，requant 多重集＋requant 對拍）；official ≡ HF golden（asset audit problems=0）⇒ built 對官方 release 語義等價；±1 requant rounding（exact≈0.926, relmax<4e-3, within1=1.0）屬設計內。
- 獨立 oracle（零 import 重寫線，2026-11-28 完成）：250/250 PASS、within1 最低 1.0、scale relmax 最壞 3.89e-3、zero-init 帶 0 違反、QKV grouped 實證、passthrough 285/285 byte-identical——已併入凍結 manifest v3。
- ComfyX 生產資產（Strata/ComfyX 所用的 comfy-int8 原件）未動；built 目錄僅新增符號連結與新檔，未覆寫任何生產檔。

## 已知邊界（NOT_VERIFIED→R3）
- 加權 load→DiT 前向數值對拍尚未跑（需 GPU 准入）；--info 僅 header-only。
- FL2VA bf16 token_refiner 副本優先解析：引擎對 ref2va 推論實際用哪份 token_refiner 待 R3 tap 對拍確認。
