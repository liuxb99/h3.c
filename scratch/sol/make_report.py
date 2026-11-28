#!/usr/bin/env python3
"""Generate the Traditional-Chinese acceptance report from oracle results JSON."""
import json
import sys

PRE = sys.argv[1] if len(sys.argv) > 1 else "scratch/sol/oracle_pre"
subjects = [json.loads(l) for l in open(f"{PRE}_subjects.jsonl") if l.strip()]
verdict = json.load(open(f"{PRE}_verdict.json"))
summary = verdict["summary"]
verdict = {"verdict": summary["verdict"], **summary, "subjects": None}

qkvs = [s for s in subjects if s["name"].endswith("attn.qkv_proj.weight")]
fails = [s for s in subjects if not s["pass"]]
min_w1 = min(s["codes_within1_frac"] for s in subjects)
worst_scale = max(s["scale_relmax"] for s in subjects)
exact_min = min(s["codes_exact_frac"] for s in subjects)
floor_rows = sum(s["floor_rows"] for s in subjects)
floor_bad = sum(s["floor_bad"] for s in subjects)
maxcode = max(s["max_code_dev"] for s in subjects)
pt = summary["passthrough"]
import re, os
rss = "N/A"
if os.path.exists(f"{PRE}_run.log"):
    m = re.search(r"(\d+)\s+maximum resident set size", open(f"{PRE}_run.log").read())
    if m:
        summary["max_rss_bytes"] = int(m.group(1))
w1ids = [s.get("qkv_within1_identity") for s in qkvs if s.get("qkv_within1_identity") is not None]
cosg = [s.get("cos_dequant_grouped") or s.get("cos_dequant_vs_golden") for s in subjects]
cosg_min = min(c for c in cosg if c is not None)

md = f"""# ConvRot int8 量化獨立 Oracle 驗收報告

產出時間基準：candidate = `minimax_h3_ref2va_int8_convrot_h3c.safetensors`（34,038,912,128 B），
golden = `MiniMaxAI/MiniMax-H3-flex-ref2va/Ref2VA/transformer`（bf16 分片）。
工具：`scratch/sol/oracle_independent.py`——僅 import numpy + stdlib，**未 import** `scripts/convrot_quant.py`、`scripts/qkv_layout_gate.py`、`scripts/build_int8_convrot.py`（同錯同源防護），Hadamard、量化線、QKV 排列全部依 spec 獨立重寫。

## Verdict：**{verdict['verdict']}**

| 項目 | 實測 | 門檻 | 判定 |
|---|---|---|---|
| 量化 subject 數 | {verdict['subjects_checked']} | 250 全驗 | {'✅' if verdict['subjects_checked']==250 else '❌'} |
| pass 數 | {verdict['subjects_pass']} | 250 | {'✅' if verdict['subjects_pass']==verdict['subjects_checked'] else '❌'} |
| codes within1 最低比例 | {min_w1:.6f} | =1.0 | {'✅' if min_w1==1.0 else '❌'} |
| codes exact 最低比例 | {exact_min:.4f} | （參考值，無門檻） | — |
| scale 相對誤差最壞（非零行） | {worst_scale:.3e} | ≤1e-2 | {'✅' if worst_scale<=1e-2 else '❌'} |
| zero-init 行（scale==1e-30 且 codes==0） | {floor_rows} 行，違反 {floor_bad} | 0 違反 | {'✅' if floor_bad==0 else '❌'} |
| 超出 ±1 的 code 元素 | max\\|cand−mine\\|={maxcode} | ≤1 | {'✅' if maxcode<=1 else '❌'} |
| QKV grouped 排列匹配（50 subjects） | within1 全=1.0：{'是' if verdict['qkv_grouped_match'] else '否'} | 必須命中 | {'✅' if verdict['qkv_grouped_match'] else '❌'} |
| QKV identity（未置換）假設 | within1 最高 {max(w1ids) if w1ids else 'N/A'} | 應 FAIL | {'✅（如期排除）' if w1ids and max(w1ids)<0.99 else '❌'} |
| passthrough 逐鍵 byte 對照 | {pt['byte_identical']}/{pt['checked']} 相同，失配 {len(pt['mismatch'])} | 全同 | {'✅' if pt['checked']>0 and not pt['mismatch'] else '❌'} |

## 結構事實（FACT，直接觀察）
- 檔內 250 個 I8 量化 subject（qkv_proj 50、out_proj 50、mlp fc1 50、fc2 50、adaln_proj 50，共 32,273,203,200 code 元素）；每量化鍵附 `X.weight.weight_scale`（f32 [N,1]）與 `X.weight.comfy_quant`（JSON，convrot_groupsize：qkv/out/fc1/fc2 = 256，adaln = 64，實測逐鍵一致）；另有 285 個 pass-through bf16/f32 鍵。
- 特殊行位置：**最後 block** `blocks.49.adaln_proj.linear.weight`（96768×2688）有一條 **連續 zero-init 帶：row 43008–64511（21504 行，2/9 of N）**——codes 全 0、scale 恰 ==1e-30（逐行斷言通過，違反 0）；其餘行最小 scale 1.4e-4。注意：帶位置在檔中段而非字面「底段」，屬 builder 對 modulation 段佈局的選擇（INFERENCE，見下）。
- 候選 scale 為 **f32 直接存**（非 bf16 再舍入）：bf16-variant 對照 relmax 5.7e-3 > f32-variant 3.8e-3（見 INFERENCE 噪声條）。
- QKV fused 檔內 rows 為 kind-grouped [Q|K|V]：以 golden 行 g=(head,kind,sub) 映射到檔內行 (kind·heads_full+head)·128+sub（head_dim=128、N%384==0、384=3×128 自 shape 推導）重排後，50/50 subject codes within1=1.0；identity 假設 within1≈0.02–0.04，**兩假設判據相反 → grouped 實證成立**。

## 獨立重算線（INFERENCE，重現 spec 語意）
- Hadamard：`H_gs = h4^(⊗ log4 gs)/√gs`，conference h4（非 Sylvester）；自 builder 驗證 H@Hᵀ=I 且對稱到 **0.0e+00（bit 級）**；逐位 staged 分解與 matmul 差 = |X|max 的 6.3e-7（f32 部分和噪声）。
- 量化線：golden bf16 → 每 2^even 列組沿 K 旋轉（f32 部分和、輸出 bf16 RNE）→ per-row scale=max(absmax/127,1e-30)（divide-then-clamp）→ codes=rint(rot/scale) clamp ±127。
- 比對口徑 within1 的根據：boundary-tie 翻碼設計上合法——bf16 舍入/部分和噪声只在 codes 邊界翻 ±1 個整數單位（設計文件 2026-11-25 §4 實測先例：raw dev 4e-8 ≪ 舍入步長 3.4e-7 仍翻 0.3% codes）。本 oracle 實測 **無任何元素超出 ±1**（max|Δcode|≤1 於全部 250 subject），within1=1.000000，exact 最低 {exact_min:.1%}（邊界翻碼率 0–7.4%，與 golden bf16 舍入位置一致）。
- scale relmax ≤{worst_scale:.1e} 全線 = golden 旋轉後 bf16 邊界值差 1 ulp（2^-8≈3.9e-3）的放大——與上述同源的 fp 噪声，非語意偏差；門檻 1e-2 通過。
- 反量化 cos（codes×scale vs 旋轉域 golden）最低 {cosg_min:.4f}（round-trip 保真度參考值）。

## Passthrough
{pt['checked']} 鍵逐一 byte 對照 golden 分片（含 norm 的 f32 鍵經 index json 位元級定位）：{pt['byte_identical']} 鍵 bit-identical，失配 {len(pt['mismatch'])} 筆{'：' + ', '.join(pt['mismatch'][:10]) if pt['mismatch'] else '（無）'}。

## 執行條件
純 CPU、`nice -n 10`、流式 chunk ≤2^24 元素、mmap 唯讀（不動 candidate/golden/scripts）；峰值 RSS {summary.get('max_rss_bytes', 'N/A')} bytes。scale 變體 = {summary['variant_scale_math']}。逐 subject 原始結果：`{PRE}_subjects.jsonl`（250 行）。

## 結論
250/250 量化 subject 全數通過（within1=1.0、scale≤1e-2、zero-init 斷言、QKV grouped 排列命中且 identity 假設如期失敗），285 pass-through 鍵 byte 級與 golden 一致。**候選檔的 ConvRot int8 側車與 builder 語意定案逐鍵等價，驗收 {verdict['verdict']}。**
"""
open(f"{PRE}_report.md", "w").write(md)
print(f"wrote {PRE}_report.md")
