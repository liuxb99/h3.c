
### COPY 時間線補記 (FACT, proc_9c7a5e60cc13)
- 引擎 copy 00:51 啟動、01:02:06 完成 (34,038,894,550B, 同盤 copy 10:29min), 輸出 sha aa0af27c… == manifest 記錄 ✓。
- chain p4f 於 01:11:25 啟動 (位複製完成之後) ⇒ 無半截載入污染; p4f_nativepost640 dump dir 01:15 建立, native 640 run 進行中。
- df FACT: SSD2T 僅 ~82GB free (92% -used) ⇒ 已調整: builder 改「單份建置+per-tensor sha manifest 雙跑比對」省 34GB; watchdog monitor 加 FREE5GB 分桶監控; 閘值: 低于 15GB 禁止新寫入。

### COPY 時間線補記 (FACT, proc_9c7a5e60cc13)
- 引擎 copy 00:51 啟動、01:02:06 完成 (34,038,894,550B, 同盤 copy 10:29min), 輸出 sha aa0af27c… == manifest 記錄 ✓。
- chain p4f 於 01:11:25 啟動 (位複製完成之後) ⇒ 無半截載入污染; p4f_nativepost640 dump dir 01:15 建立, native 640 run 進行中。
- df FACT: SSD2T 僅 ~82GB free (92% 已用) ⇒ builder 改「單份建置+per-tensor sha manifest 雙跑比對」省 34GB; watchdog monitor 加 FREE5GB 分桶監控; 閘值: 低于 15GB 禁止新寫入。

### 生產資產 SHA 凍結 (FACT, prod_sha.json → manifest_production)
- fl2va_pruned_int8_convrot e889202c41dafb67… 20,970,379,616B
- pi_h3_4step_dwd_adapter_int8_convrot a2fa1caae2cd4a12… 2,192,176,240B
- minimax_h3_video_vae_int8_convrot 52a2c8c73583c86e… 2,811,065,184B
- qwen3vl_32b_te_nvfp4_awq 35a88d51044231fe… 15,687,142,551B
- 四者自此受 asset_audit sha+mtime 監控; 任何後續偏移即 FAIL(CONTAMINATION)。

### 官方資產回滾失敗根因定位 (FACT, 2026-11-28 01:3x)
- revert exit1 原因 = REVERT_SHA_MISMATCH（非磁碟/權限/占用問題）：279 區全數還原成功且逐區 sha_before 斷言通過，但整檔 39789f8c ≠ 預期 9eef9340。
- 現場 live 官方檔 sha = aa0af27c（==patch 後內容==engine copy）；舊 log 宣稱的 e374130c 與實測矛盾 → 標記 UNRESOLVED（字面保留，來源不明）。
- HF API（Comfy-Org/MiniMax-H3, LFS oid）權威值：ref2va_int8_convrot = 9eef9340 全 sha, size 34,038,894,550 ✓ 與本地同名檔同尺寸。
- 判別實驗進行中：diff(pre-patch local tmp vs engine copy) → 若差異恰為 manifest 279 區 ⇒ 本地檔 14:31 前已與 HF 分岔（假設 A）；若有區外差異 ⇒ patch 範圍超記錄（假設 B）。
- 處置（依用戶指示）：官方檔標記 INVALID_REFERENCE、禁止作 golden；從 HF 官方源重新下載至 .hf_stage_official（背景, 完成須整檔 sha==9eef 全值比對）→ 原子替換 → chflags uchg + chmod 444 → 以稽核工具實測未授權修改會 FAIL。引擎資產(aa0af27c)不受影響、研究繼續。
- 附带修正：SSD2T 實際可用 150GB（先前 82GB 讀值為 tmp 建立前且計算口誤）；先前「<15GB 禁寫」門檻維持，builder 閘放至 free>40GB。
- 稽核工具新增：--manifest-dir / --single / header_contract(shape+dtype) 檢查；verify_ready.sh 載入閘門（.ready marker + audit --single, fail-closed exit 2）。負向測試 harness 已派子代理（S0-S7 含唯一誤報對照）。

### 2026-11-28 ~02:45 批量發現（自主研究 tick：tap 門檻比對 FAIL + 張量級探查 + 視覺黑屏）

**P4F tap 門檻比對（FACT，輸出 scratch/sol/p4f_tap_compare_verdict.txt，VERDICT=FAIL，門檻未放寬）**
- 共同 tap=21。輸入側逐位全同：e0 audio/video 輸入、e1_text_source、rope 表、b0 layout、e2 表、e3_condition_w/b、latents/indices/shift/scale/step。
- 首個偏離 tap = e1_after_cond（byte-offset 0 起偏離；native dump 107520/107520 = 100% NaN，golden 0% NaN）；其後 refiner 全鏈（after_refiner0/1、finalnorm、refined_text、hidden_packed）與 step0_velocity 全為 100% NaN。
- 判讀（INFERENCE）：污染點在 refiner 條件注入步；輸入張量逐位相同而輸出全 NaN ⇒ 計算該步所用的模型權重張量與 golden 運行時不同。

**張量級探查（FACT，convrot_engine_patched vs MiniMaxAI bf16 shards）**
- 修補區（279 manifest）抽樣 12 張量：與 HF 同名同 dtype/shape 同字节長，但 12/12 非逐位全同 ⇒ manifest 所稱「279 已替為 HF golden bytes」與 MiniMaxAI 原件不符（參考歧義待 9eef 發布檔消解）。
- 修補區外 dtype 相容張量僅 6 個可逐位比：只 rope.inv_freq 全同；condition_proj.weight、condition_proj.bias、final_layer.adaln_proj.linear.weight/bias、token_refiner.final_norm.weight 皆偏離。
- condition_proj 正是 e1_after_cond 步的輸入 ⇒ 黑屏/NaN 首要根因候選 = 這批基線殘留變更（發生於 pre-patch 之前、不在 279 恢復範圍 ⇒ 同時解釋 REVERT_SHA_MISMATCH 39789f8c≠9eef 與 patch 範圍無溢出；diff_prepatchlocal_vs_patched.json 跑中待 out=0 確認）。
- 建議下一步實驗（待用戶閘，本輪未動引擎 C 碼/資產）：從正確基準重恢 condition_proj.w/b + final_layer.adaln_proj + token_refiner.final_norm.w 後重跑 p4f tap dump（若 after_cond 轉 0% NaN 即閉環）。引擎內容變更須同步 expected_sha 重凍結。

**視覺驗收（VISION 行）**
- VISION tag=p4g_p1_cat verdict=FAIL evidence=first/mid/last 三幀皆近純黑細噪點；橘貓/草地/蝴蝶語義要素零命中
- VISION tag=p4g_p2_corgi_piano verdict=FAIL evidence=三幀全黑無柯基/鋼琴/音樂廳；且與 p1 out.mp4 逐字節相同(sha 56727b88...)，兩 run_log 各含不同 prompt 文字 ⇒ 產出對 prompt 不敏感，與 refiner-NaN 根因互證
- FACT tag=p4g_p3_ferris rc=1 無影片：run_log 末 = FL2VA shard 載入至 blocks.44/50 遭 Interrupted system call（infra 級中斷非引擎缺陷；鏈下一輪重跑）

**AUDIT 歸因（FACT/INFERENCE；ALERT 已在回報標示）**
- AUDIT_FAIL 組成：1× 官方 CONTAMINATION（=回滾進行中，屬預期進行態）＋ 12× engine passthrough FAIL（=上述既有偏差類，非引擎新增污染；engine sha aa0af27c 對 its manifest OK）。ALERT 內容：引擎 passthrough 張量與 MiniMaxAI 原件不一致 + 生成鏈 100% NaN；歸因歧義（引擎錯 vs 參考錯）由 compare_engine_vs_hf_release.py（已就位，含 A/B/MIXED 判類）在 9eef 下載完成後自動消解。

**REVERT/HF 進度（FACT）**
- hf_stage_official 下載進行：13.1GB/34GB（~02:45），無 token 未認證下載，ETA ~04:00；完成後流程：整檔 sha==9eef 驗證 → compare_engine_vs_hf_release.py → 官方檔替回+chmod444+uchg+freeze_pending=false → audit 復跑（屆時官方 CONTAMINATION+passthrough 線應清，殘 FAIL 即 ALERT）。
- P4H ladder 01:15 啟動，六輸出仍 absent（GPU 鏈排队中）；builder 目錄已建、report 未出。OPEN：p4b golden 運行時權重時代 provenance 待證。

### 2026-11-28 ~04:0x 批量發現（自主研究 tick：官方檔回滾閉環 + 歧義消解 MIXED + P4H 鏈死而復生）

**官方回滾完成（FACT，revert 閉環）**
- HF staged 下載完成（HF_DL_EXIT=0），整檔 sha256 實測 = 9eef9340…76a9 == LFS oid（RELEASE_SHA_OK）。
- 原子替換（mv staged→live，同 volume）+ rm .revert_tmp（淨 release ~68GB→SSD 可用 183GB）+ chmod a-w + chflags uchg（ls -lO 實測 uchg 標記在）。
- manifest_official：freeze_pending=false、verification_status=REVERT_DONE、golden_use_allowed=true、live_sha=9eef。
- REVERT_SHA_MISMATCH 根因閉環：pre-patch 本地位元組 39789f8c 本就 != release ⇒ 偏差發生在 patch 之前的基線（5 個 off-279 張量：condition_proj.w/b、final_layer.adaln_proj.w/b、token_refiner.final_norm.weight），非 patch 範圍溢出（off-279 偏離名單恰為該 5 項、patch 區 279 全在 manifest 內）。

**歧義消解（FACT，scratch/sol/release_disambig_verdict.txt）**
- engine vs release：1035 可比張量 — in_patch279 identical=0 deviated=279；off_patch identical=751 deviated=5（=上述基線殘留 5 項）。
- release vs MiniMaxAI：nonpatch 6/6 全同；patch12 identical=6 deviated=2 ⇒ CLASS=MIXED：release 的 pass-through 也非 bf16 原值（int8 roundtrip），engine 位元組契約 = release 檔only。
- 結論（INFERENCE）：先前「engine 偏離 MiniMaxAI」FAIL 屬參考錯配（bf16 vs int8 roundtrip）；真正的生成缺陷根因 = engine 在全部 279 patch 區與 release 契約不一致 ⇒ refiner NaN/黑屏與之一致。下一步：builder 須以 release 檔為唯一基準重建（或逐區對齊 release），再以 p4f tap 閉環複驗。

**audit 工具兩處錯判修正（FACT，scripts/asset_audit.py；門檻未降）**
1. uchg 語義：macOS UF_IMMUTABLE 下 mode 位元仍顯示可寫但已受保護，舊邏輯誤報 writable FAIL；現 uchg+mode-writable → INFO。負向保護不變（無 uchg 且可寫仍 FAIL）。
2. engine passthrough 參考：位元組契約改用 release_reference_dir（官方 9eef 檔），bf16 MiniMaxAI 目錄保留為 layout/name 索引用途；byte-identical 門檻原樣保留，僅換正確基準（實測引擎對 release 於抽樣 2 張 patch 樣本偏離，將如實 FAIL/記錄）。

**AUDIT 歸因更新（FACT/ALERT 延續）**
- 上一輪 AUDIT_FAIL problems=13 全數 = 12× engine-passthrough-vs-bf16（錯參考，本輪修正）＋ uchg 誤判（官方條目 sha 已 OK）。修正後預期官方+production 全清；殘 engine-vs-release 抽樣偏差為真實契約偏差，屬 builder 重建範圍，續列 ALERT。
- ALERT 內容不變：引擎 279 區與官方 release 契約零一致 + 生成鏈 100% NaN（p4g 黑屏互證）。

**P4H 鏈事故與重啟（FACT）**
- 01:15 啟動的 ladder 進程於 ~02:0x gateway 重啟中被殺（log 僅 start 行、進程消失、無 ARM_DONE；persist_on_release 未設）。03:45 以 persist_on_release=true 重啟。
- VISION tag=p4h_544_cat verdict=FAIL evidence=first/mid/last 三幀 sha 互同、純黑無噪點，橘貓/草地/蝴蝶零命中（rc=0 mp4 105KB，與 refiner 全 NaN 互證）
- P4G 名冊更正：實際 tag=p4g_p1_cat/p4g_p2_corgi_piano/p4g_p3_ferris（monitor 的 p4g_cat 等為舊名冊）；p3_ferris 因 FL2VA 載入中斷 rc=1 無影片，鏈未重跑該臂。

**三向取樣定讞（FACT，12 個 audit 抽樣張量：engine(E) vs release(R) vs MiniMaxAI(M)）**
- 10/10 非 qkv in-279 樣本（audio_patch_proj.w/b、adaln_proj.bias×2、q/k_norm×4、norm1/2×2）：R==M 而 E 與兩者皆不同 ⇒ engine 基線殘留污染（與 5 個 off-279 偏差同族），非參考錯配。
- 2/12 qkv 樣本：E/R/M 三方互異（R==M==grouped，E==宣稱 interleaved 但仍與 MiniMaxAI 逐位不同——疑為錯誤的 interleave 實作或另一輪量化）。
- 判讀（INFERENCE）：engine 宣告契約 passthrough_bf16=HF verbatim 自我矛盾且實證不成立 ⇒ engine 對任何正統基準皆不忠實 = 黑屏/NaN 根因實錘（p4h_544_cat 黑幀互證）。builder 重建須以 release(R) 為唯一位元組基準、qkv 轉換以「R→interleaved 的可驗證置换」单独定義並以 tap 閉環驗收。
- audit 第二次修正：uchg 在此 exFAT-class 盤的 st_flags 編碼為 2（非 APFS 0x200），改以 0x202 位元判保護 + 實測 EPERM 寫入封鎖為證；負向門檻不變。官方條目預期轉 OK+INFO；engine 12/12 FAIL 為真實契約偏差，保留。

### 2026-11-28 ~04:2x 批量發現（自主研究 tick：P4H 收官、audit 收斂、builder 停擺）

**P4H ladder 結果（FACT，CHAIN_P4H_DONE 04:13:57）**
- p4h_544_cat rc=0 105KB；p4h_544_corgi rc=0 105KB（與 cat out.mp4 逐字節同 sha 459c9566）；p4h_544_girl rc=1（載入至 core 15/50 再遭 Interrupted system call）；1280 三臂 rc=1 秒退。
- FACT p4h_1280_* 根因 = 參數幾何拒絕：'width and height must be multiples of 32' — 720%%32=16，使用者階梯 1280x720 违反引擎對齊契約（非引擎缺陷；修正選項 704/768，待閘）。
- FACT 第二次 EINTR（544_girl，與 p4g_p3_ferris 同型）：均為 SSD2T 大檔讀取中斷；idle 守衛（curl 失敗視為 idle）與 ComfyX 實際排程的競態為首要嫌疑（INFERENCE，待與 ComfyX queue 時間戳 correlations）。
- VISION tag=p4h_544_cat verdict=FAIL evidence=first/mid/last 三幀 sha 互同、純黑無噪點，橘貓/草地/蝴蝶零命中（rc=0 mp4 105KB，與 refiner 全 NaN 互證）
- VISION tag=p4h_544_corgi verdict=FAIL evidence=mid/last 幀純黑（與 cat 幀逐字節同），柯基/鋼琴/音樂廳零命中；對 prompt 不敏感再獲互證
- FACT tag=p4h_544_girl / p4h_1280_* 無影片，無從驗收。

**AUDIT 收斂（FACT，scratch/sol/audit_rerun2.log）**
- 官方/生產/HF 源全部 OK；官方條目 = sha match + immutable INFO（st_flags=2 經 EPERM 實測）。殘 problems=13 全數 = engine passthrough 12/12 + 匯總行 — 即已實錘的 engine 契約偏差（三向取樣 R=M 而 E 獨異），非工具錯配。ALERT 持續：engine 須以 release 為基準重建。
- revert_final_ok.txt 已建立（含 sha/freeze/EPERM/tmp_gone 證據）。

**BUILDER 停擺（FACT/ALERT）**
- /Volumes/SSD2T/models/MiniMax-H3-h3c-built/ 僅餘 1MB .writing 半成品（00:34 起無進展）；builder_report.md 不存在、無 builder 進程 ⇒ 子代理疑於 gateway 重啟中死亡。重建方向已定讞：基準=官方 release（9eef），qkv interleave 轉換須逐 tap 閉環驗收；p4f NaN 根因（基線殘留 + 不忠實 patch）已為其提供明確規格。
- 閘條件更新建議：milestone 完整 = builder 重建產出 + p4f tap 重跑轉 PASS；舊 ladder 產出全數黑屏/FAIL，不構成里程碑。

### 2026-11-28 ~08:50 tick（audit 門檻複核、EINTR 根因與補丁、builder 復蘇）

**AUDIT 語義變更複核（FACT/ALERT）**
- 08:04 一輪將 manifest_engine.json 新增 `passthrough_expected_deviation` 宣告、asset_audit.py 加 exp_dev 分支（偏差僅在 manifest 明確宣告時降 INFO，永不默認）、並建 audit_negative_tests.py S0–S9 負向保護；實測本輪 `asset_audit.py` → AUDIT_PASS problems=0（舊 audit_rerun2.log problems=13 為錯參考修正前基線）。
- ALERT（未閉環）：宣告文字稱「engine 279 區帶 HF-golden verbatim（interleaved qkv+bf16）」——與 04:2x 三向取樣 FACT（10/10 非 qkv：R==M 而 E 與兩者皆異）直接矛盾，engine 忠實性之「by design」敘事目前無證據支撐。門檻判定：工具層級未默認放寬（宣告制+負向測試在），但 ALERT 保留至 builder 重建檔案通過 repro+全區 byte-check+GPU tap 閉環；新 built 檔 passthrough 仍須對 release byte-identical、不得引用舊 engine 宣告。
- FACT 新量化路徑證據（scratch/sol/chain_parity.log 04:33）：butterfly vs matmul-ref 在 qkv 8.647074% codes 不同、adaln 9.034617%，max|scale diff| 1.5e-05/6.1e-05 ⇒ 快速路徑於捨入邊界翻碼（舍入邊界級別，非系統性縮放偏差）；此為 build1/build2 共用同一路徑，不削弱雙跑自洽性，但 contract 文字「verified f32 line」應註明以 butterfly 為準。

**EINTR 根因排除與程式修改（FACT/CODE）**
- build1（04:34）rc=1 根因 = `os.listdir(h3c-built)` InterruptedError（EINTR）；本輪同路徑 probe 立即成功（1 條目）⇒ 瞬斷、非資料損壞；同型事件累計 3 起（p4g_p3_ferris、p4h_544_girl 大檔讀取、build1 listdir），均 SSD2T exFAT-class 卷、均可重試通過。INFERENCE 源頭：卷層信號中斷，與 ComfyX IO 競態相關性未定讞。
- 本輪程式修改：scripts/build_int8_convrot.py 新增 `ei_retry()`（10 次遞增退避），包裹 listdir/os.remove/st_create/os.replace/FL2VA symlink——lint OK，語義不變、門檻未降。
- FACT：ComfyUI main.py 進程在（pid 71422），:8390 無 listener ⇒ 鏈之 idle() 探得空佇列屬預設放行路徑，記錄在案（未動進程/佇列）。

**_builder 復蘇（ACTION）**：以 subagent 重跑 scratch/sol/run_chain.py 雙跑重現協議（parity→build1→sha→delete→build2→sha→repro_compare→qkv_layout_gate ×3→compare_bytes），產出 /Volumes/SSD2T/models/MiniMax-H3-h3c-built/ 與 2026-11-28_h3c_convrot_builder_report.md；GPU 不碰（生產保留），nice 10。builder_report.md 不存在前 BUILDER≠report。

### 2026-11-28 11:3x tick（R1 結帳：gate rerun3 身分查證＋誤報定性＋負向拒收全過）

**rerun3 誤報定性（FACT，進程 proc_758efd80af98 原始命令實證）**：外層 exit 1 根因 = 收尾 `grep -c "^FAIL"` 匹配 0 筆時 grep 語義 rc=1；gate 本體 `GATE_EXIT 0 @11:17:06`。永久修復原則：完成判定只認 rc artifact＋JSON verdict，grep 收尾加 `|| true`；每次 run 用專屬 run_id 檔名（本次鏈內 gate 與 rerun3 共用 gate_built_vs_hf.json 路徑構成競態，現存 JSON mtime 11:17:06 = rerun3 所有，鏈內 rc=1 記錄屬舊跑、已被覆寫取代）。

**R1 語義閘全量結果（FACT，rerun3 run 10:49:39→11:17:06，GATE_EXIT 0，verdict PASS）**：
- S（scale 側車語義）250/250 PASS；Q（codes requant 多重集，非 qkv 200）200/200 PASS；L（50 組 QKV grouped 排列對拍）50/50 PASS（grouped_min 0.999953）；P（passthrough vs HF 可證等價）285/285 PASS；failures=0。
- fileorder cos below_min 50 = 恰為 50 組 fused QKV 的 kind-grouped 排列（設計預期，經 L 檢查閉環；inter 假設已被拒）。
- passthrough 傳遞鏈：built≡HF（gate P）；官方≡HF（08:0x audit problems=0）⇒ built≡官方 release 對等。

**負向拒收矩陣（FACT，scripts/gate_negative_tests.py，seed 20261128，門檻未放寬）**：CONTROL=PASS exit0；M1 錯 QKV 排列拒（QUANT_DRIFT inter=0.999965/grouped=0.332708）；M2 scale×2 拒（relrms 0.9986）；M3 雙重 Hadamard 拒（cos 0.0611）；M4 0.1% 碼破壞拒（cos 0.99896）；M5 側車缺失拒（[P] dtype/shape）。ALL EXPECTED，證據 evidence/convrot_audit_20261128/gate_negative_matrix.txt。

**R1 結帳（ACTION）**：.ready marker＋freeze（chmod a-w）＋manifest v2（含 gate rerun3、負向矩陣、oracle 待補、雙跑 repro、sha256 dd098ba9…ffeeb）寫入 /Volumes/SSD2T/models/MiniMax-H3-h3c-built/。獨立 oracle 子代理（不 import builder/gate 函數、流式全 250 重驗）仍在跑，結果作補充證據非阻斷項。

**下一步（R2）**：h3.c 原生引擎對 h3c-built 資產的實載契約與 smoke——重型 GPU 步驟一律先過 scripts/admit.py 准入（當前 swap 20.6G/compressor 16.5G 高於政策帶，heavy_gpu 現行會被 DENY，等子代理收尾＋壓力回落）；不動 ComfyX/Strata。

### 2026-11-28 12:0x tick（R2 實載契約建立：loader/probe/--info 三口徑全對等，GPU 待准入）

- **FACT**：`h3_native_loader_tests <built>` shards=2、253 checks fails=0，輸出與 official 同構；`probe_sidecar`（ argv 化＋補 qkv/adaln 鍵）built vs official 逐 key 一致（I8、convrot=1、gs256、adaln gs64；token_refiner 兩側皆由 FL2VA bf16 副本優先解析 = 引擎既有行為）；`h3 --info` 兩目錄資產清單 diff 空。TE symlink（qwen3vl nvfp4）復用官方原件補齊 single-file 佈局。契約檔 evidence/convrot_audit_20261128/r2_asset_contract.md。
- **修復**：probe_sidecar 連結失敗根因 = 隱式規則經 .d 檔把 SDKSettings.json 拖進 $^（clang 多輸出 -o 衝突）；Makefile 加 .o-only 顯式規則永久修復。admit.py watch 補 overspend 偵測（註冊 est_gb vs 實際 RSS >1.3x → ALERT）＋request 事件補 est_gb 欄；watch 常駐重啟（先前 nohup 因寫錯目錄靜默死亡，教訓：啟動後必驗 monitor.jsonl 有新行）。oracle 補登記（heavy_io est 28G，實測 26G；其 pid 91901 早於准入制度上線）。
- **FACT（oracle 進度）**：獨立重驗 146/250 全 pass、within1=1.000000、relmax<4e-3、layout 正確——與 gate 交叉印證至 block 35。
- **INFERENCE**：swap 20.5G 平台 + compressor 7.9→11G 震盪主因疑似 oracle 26G 常駐＋舊 panic 殘留；其收尾後預期回落。
- **下一步**：oracle 收尾＋壓力回落 → admit heavy_gpu → 引擎 192×192/22f/2-step smoke → R3 condition_proj NaN 回歸＋逐 tap golden 對拍。GPU 未准入前不強推。
