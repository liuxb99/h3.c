# Stage-model 缺口記錄（2026-11-27）

〔FACT〕子代理 D 的 CPU 重放矩陣（weights×layout 全組合，含誠實基線）無法重現 pre-patch native 臂的 refiner tap（所有組合 cos 0.74–0.78，無法命中 0.693/0.676 觀測值）。

〔FACT〕判定不依賴此重放：根因建立在逐位權重表（pre-patch qkv = kind-grouped 排列、u16 multiset 與 oracle 相同、undoB(P) 逐位還原）＋ 279 張逐位對帳，與重放結果無關。

〔INFERENCE〕缺口最可能出自本人對 native dequant/loader 路徑的 stage 模型不完整（I8 convrot 反量化在 kernel 端（Metal shader gs=256 per-row scale）與 Python 端還原式未必逐位等價，且 refiner bf16 pass-through 在 native 臂的 kernel 選擇（convrot gate）與 eager 臂不同路）。

〔规则〕在 GPU post-patch tap 逐位驗收（p4f）通過之前，不得宣稱完整數學閉環已 PASS；「閉環」僅到：(1) 權重字節級 == golden、(2) eager 臂重放 == eager tap（cos 0.985+，重放舍入級）。

來源：/Users/cyliu99/.hermes/cache/scratch/refiner_audit/FINDINGS.md（caveat 節）、replay4/replay_decisive 系列腳本與結果檔。
