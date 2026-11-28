# 65139072/192MiB diff 數值出處缺失備忘（2026-11-28）

## 事實鏈（均有盤上檔案）
- scratch/sol/release_disambig_verdict.txt（DISAMBIG_DONE）：official release(9eef) vs MiniMaxAI 原始 = 279 in-patch tensor 偏離（roundtrip 改寫）＋ **5 個 off-patch tensor 偏離**（condition_proj.weight/bias、final_layer.adaln_proj.linear.weight/bias、token_refiner.final_norm.weight）；CLASS=MIXED。
- scratch/sol/revert_final_ok.txt：原地回滾失敗（REVERT_SHA_MISMATCH 39789f8c≠9eef）→ 放棄 byte-回滾路徑，改 HF 重下載，原子替換後 live sha==9eef ✓，freeze=chmod a-w+uchg（st_flags=2）。

## 缺失
- 「diff(pre-14:31 local, patched)=192MiB 全在 manifest 區外」的逐字節數字（201326592 等）**無任何檔案或會話紀錄支撐**（session_search FTS 零命中，已用 34038894550 驗證搜尋力）。
- 產生它的背景 job（01:34 啟動）在記憶體災難中被 SIGKILL（exit 137），輸出檔 diff_prepatchlocal_vs_patched.json 從未寫出；diff_run.log 已被重跑重定向截為 0 位元組。
- 唯一輸入 .revert_tmp.safetensors 已依 REVERT_FINAL_OK 刪除 → 精確重算需重建 34GB 輸入，評估後不做。

## 判定
- 定性結論「官方檔在 patch 前已偏離原始基線」= **FACT**（release_disambig_verdict.txt 獨立支撐，且機制上回滾不可行）。
- 精確字節拆分 = **NOT_VERIFIED／永久缺失**（輸入已刪，結論不依賴它）。
- 教訓：背景 job 的結論必須先落盤到不可截斷的 artifact 才可被引用為 FACT；唯一輸入檔在結論歸檔前不得刪除。
