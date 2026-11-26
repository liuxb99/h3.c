# Mac Studio 原生 Metal H3 W4A8 影片生成引擎開發計畫（Charter, 2026-11-27）

## 一、任務目標
在 Mac Studio M5 Max 128GB 統一記憶體環境下，延續現有 Metal8、H3 native、MPS、A8W8 及 SelfLift 研究成果，新增 Sulphur 3 W4A8 支援，逐步建立通用 H3 低記憶體原生 Metal 影片生成引擎。模型來源：https://huggingface.co/SulphurAI/Sulphur-3

最終目標不是單純成功生成影片，而是：
- Strata LLM 持續常駐並正常提供 LAN 推理服務。
- 影片生成與 LLM 推理能夠同時進行。
- 大幅降低影片生成的統一記憶體占用。
- 避免大量 SSD swap。
- 保持人物外觀、動作、服裝及影片畫質。
- 支援原版 MiniMax H3、Pika H3 Flex、Sulphur 3 等衍生模型。
- 所有研究成果形成可重用的程式、配方、Skill 與自動驗收工具。

## 二、開發原則
1. 先完整閱讀現有 Git 倉庫、PR、研究報告、Metal kernel、權重 loader、SelfLift 與既有 Skill。
2. 不重新開發已經存在且驗收通過的功能。
3. 不破壞 Strata 及現有 H3 生產環境。
4. 使用獨立 Git 分支，建立 PR，保持 OPEN，不自行合併。
5. 先完成架構設計與技術調研文件，再開始實作。
6. 採用真機、真模型、真實影片驗收，不得以 mock、編譯成功或數值有限代替影片驗收。
7. 每輪實驗均保留配方、效能數據、視覺證據及 Git commit。
8. 不要為了追求速度犧牲畫質，也不要將所有模型永久反量化為 BF16。

## 三、Phase 0：完成現有 Metal8 問題修復（P3-C 接續）
重點檢查：MPSCommandBuffer reuse／MPSGraph 與直接 compute encoder 生命週期／command buffer commit、completion、fence 與同步順序／map-patch、refine-cond、condition projection 實際寫入／buffer ownership、未初始化記憶體及 GPU 資源競爭／native A8W8 與 BF16 golden 逐層偏差。
> 實測修正（2026-11-27）：MPSCommandBuffer 復用已真機 A/B+sentinel+CBDBG 證明**非根因**；根因=newBufferWithBytesNoCopy 包 MAP_PRIVATE mmap 的 CPU/GPU 視圖分歧（commit 1f9d773 修復，PR #1 OPEN）。內容驗收（s10 三路徑）進行中；非黑屏、無鬼影、無 NaN 影片驗收通過才進新模型。

## 四、Phase 1：Sulphur 3 模型調研 → `docs/research/s3_w4a8_format.md`（已完成頭部契约破解）
要點：優先下載正式 W4A8（不整倉 257.5G）；確認架構差異、packing、scale/zero-point/group/activation、rotation/per-channel/per-group、命名/shape/dtype/sidecar、loader 可重用性、正式版 vs raw、原生採樣步數/scheduler/CFG/VAE/TE 需求、Pika Flex LoRA 與 SelfLift 相容性。下載前查 SSD/SHA256 避免重複。

## 五、Phase 2：原生 Metal W4A8 推理
INT4 packed loader、INT8 activation 量化、fused unpack+dequant+GEMM、SIMD group matrix、tiled GEMM、scale 融合、FP32/INT32 累積依契約、attention/QKV/FFN/proj 低記憶體運算、scratch pool、command buffer 安全管理、權重常駐與 buffer reuse。
三路線：A=BF16 reference；B=W4A8 分塊反量化（正確性參考+fallback）；C=原生融合 W4A8 kernel（主要最佳化方向）。
數值驗收須覆蓋真實模型主要 tensor shape（零值、極端值、scale、padding、NaN、Inf、累積誤差）；不得用單一小矩陣宣稱整體成功。

## 六、Phase 3：端到端影片生成（Sulphur 3）
作者原生配方建基準 → 最小解析短影格 → 544×288×124f → 768×448×124f → 1280×720/現有產線解析度。
每階段記錄：載入時間、first/warm 生成、每 denoise step、VAE 解碼、峰值記憶體、Metal allocation、memory pressure、compressed memory、swap 增量、SSD 讀寫、GPU 使用率、實際畫質。
測試案例：南宮婉舞蹈參考圖為主 + 其他角色場景防單案例過度最佳化。視覺驗收：臉部一致性、髮絲服裝穩定、人體結構、舞蹈動作自然、時序連續、無黑屏/鬼影/閃爍/塗抹。

## 七、Phase 4：Pika H3 Flex 與 SelfLift 整合
A/B 矩陣：A 原版 H3／B Pika Flex／C Sulphur3 原生配方／D Sulphur3+Corrector／E Sulphur3+Flex LoRA／F Sulphur3+Flex LoRA+SelfLift。同參考圖/prompt/seed/解析度/影格數；採樣器依各模型契約＋等步數對比。研究 LoRA 疊加衝突/蒸餾失效/動作退化；不得假設 S3 天然支援 Pika 六步；融合失敗則保留獨立配方。同時研究 Pika Flex→W4A8 轉換可行性。

## 八、Phase 5：Strata 與影片引擎共存（最重要生產驗收）
初始預算（設計目標非實測）：Strata 64GB／影片引擎 36GB／macOS+服務 12GB／安全餘裕 16GB。先量測 Strata 現行模型、KV cache、長 context、實際峰值再調預算。
動態記憶體管理：依 memory pressure 決定是否接單、可釋放非必要 scratch、VAE 分塊解碼、attention 分塊、避免權重複製、GPU 工作分段與 LLM 優先權、記憶體不足安全延後、禁止無限制 swap。
四情境：A Strata 單獨／B 引擎單獨／C 常駐並生成／D 長 context 多請求並生成。記錄 TTFT、tokens/s、P95、影片時間、記憶體壓力。Strata 不得崩潰/卸載/重啟；資源不足優先保 Strata，影片安全降級或等待。

## 九、Phase 6：效能挑戰（基準 544×288×124f，對照現有 Mac H3 產線）
- 第一階段：峰值 ≤36GB、生成 ≤300s、無持續大量 swap、Strata 正常、畫質 PASS。
- 第二階段：≤180s、再降記憶體、連續 10 段無崩潰、Strata 持續正常。
- 第三階段：≤120s、連續 30 段、形成 ComfyX 可用 Mac 生產配方。
未達標必須如實標 PARTIAL，不得降解析度/影格/畫質後宣稱達標。

## 十、SSD 與證據管理
預設只存逐層統計/hash/誤差/必要樣本；只對首個偏離點完整 dump；每輪設 dump 容量上限；完成後清理可重建暫存；保留 golden reference/manifest/原始影片/關鍵證據/報告；不刪模型、使用者資料、Git 成果、Strata 資料；大規模清理先取得批准。

## 十一、成果交付
1 架構設計文件；2 S3 格式研究報告；3 通用 H3 W4A8 Metal loader+kernel；4 數值正確性自動驗證工具；5 端到端影片生成工具；6 Pika/Sulphur3/SelfLift 獨立配方；7 Strata 共存測試與記憶體管理工具；8 真實影片與視覺 QA 報告；9 效能比較 JSON/CSV/圖表；10 可重用 Hermes Skill；11 Git commit + PR 保持 OPEN。
閉環：調研→設計→實作→真機測試→視覺驗收→根因定位→修復→回歸→經驗萃取；每輪主動進下一輪，阻礙先定根因找替代，確需批准/硬體/外部資源再提具體問題。
驗收依據：實際影片品質、生成速度、統一記憶體占用、Strata 共存能力。
