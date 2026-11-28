# Mac Studio Kernel Panic 事故報告（案：mac-系統治理 / ai-0008）

Panic 事件：2026-11-28 05:02:32，`watchdog timeout: no checkins from watchdogd in 90 seconds`，
AppleARMWatchdogTimer，OS build 26B131。主機自 2026-11-24 開機（kernel uptime 93.8h，單一 panic 事件）。
證據來源：/Library/Logs/DiagnosticReports/panic-full-2026-11-28-080008.0002.panic（完整 stackshot，含 processByPid/threadById/kernelFrames）＋本目錄 unifiedlog_hang.txt / unifiedlog_0335_0420.txt / panic_procs.txt。

## 一、當機前時間線（FACT，來自 panic 快照 processUptime、unified log 流量、thread lastRunTime）

| 時間 (11-28) | 事件 |
|---|---|
| 11-27 12:29:26 | ComfyX 正式 ComfyUI server（pid 72635，Comfy Desktop venv python3.13）啟動（由 Hermes 派發，coalition 1169） |
| 11-27 ~12:36 | 該 server 全部 worker 執行緒進入 condvar 空閒（16h 無 GPU 工作），**常駐 48.8GB 模型記憶體重從未釋放** |
| 03:35–04:09 | 基座窗口相对安静（unifiedlog_0335_0420_stats.txt：memory-pressure 715、compressor 1,800、watchdog 45） |
| 04:10–04:19 | **壓力風暴開始**：該 10 分鐘桶日誌 17,469 行（其餘安靜桶僅 5–247 行）＝jetsam 登記風暴；同期 chain_p4h 梯隊之 1280 級引擎 arm（同 coalition，RSS ~11GB 級）於 04:0x–04:13 接連啟動即 abort——風暴早於 chain tensor-dump 啟動約 8 分鐘，觸發疊加不止 chain 一端 |
| 04:18:06 | chain tensor-dump 比對工作（pid 66564，uv→python3.11，同 coalition）啟動，resident 爬到 **60.2GB**，4.9M page faults，panic 瞬間仍在主動 faulting（讀safetensors）——風暴期間最重的一塊疊加，非起點 |
| 04:18–04:58 | unified log 流量維持**風暴級**而非正常：jetsam 登記 10 分鐘桶 22.4k(04:2x)→21.9k(04:3x)→17.7k(04:4x)→14.1k(04:5x)，緩降但未回歸基座；swap 與 compressor 吸收壓力；compressor 累計 664M 次壓縮事件；05:0x 桶驟降至 472 = 系統停擺、日誌輸出死亡 |
| 04:56 | 另一 Hermes python 工具程序短暫啟動（pid 6928） |
| ~04:59–05:00 | log 流量崩塌（1035→323→0 行/分）；watchdogd 最後活動 05:00:31 |
| 05:02:32 | watchdog panic。快照：wired 56.1GB(43.8%)、compressed 53.7GB(41.9%、segments 100%)、inactive 8.6GB、free 14MB、100 個 swapfiles 達系統上限、剩餘 swap LOW |
| ~05:03 前 | 重開機完成：kernel 05:02:53 `watchdog configured (source: inferred defaults)` 為災後重設線。风暴密度佐證（unifiedlog_hang.txt 04:20–05:05，136,220 行，統計見 unifiedlog_hang_stats.txt）：jetsam 登記行 79,695 hits（kills=0，見 FACT#2）、memory-pressure 4,530、compressor 3,933、watchdogd 5,173（多為 <private>，僅計數）；SMetal/AppleH15GFX 0 hits——窗口內無直接 GPU 事件行，Metal 常駐證據仍以 panic 快照 processByPid 為準 |

## 二、根因（分級）

**FACT（直接死因鏈）：**
1. 記憶體總承諾 > 物理上限：panic 時常駐佔用 ComfyX 48.8 + chain 60.2 + strata 27.9 + director 18.8 + alpha6 ≈12.6 ≈ **168GB vs 128GB 物理**；wired 56.1GB 不可分頁。
2. 無任何 relief：事故窗口 **jetsam kills = 0**；compressor 53.7GB 達 segments 100%（用戶證據吻合）；swapfiles 達 100 個上限且 LOW。
3. 死鎖點：panic 快照顯示 **strata 執行緒 44278512 處於 TH_WAIT+TH_UNINT（不可中斷）、TH_SFLAG_ABORT、已中止掛起狀態，持有 kernel mutex**；kernel_task 內 **1067 個執行緒 TH_UNINT** 全數阻塞等鎖；memoryPressureDetails：need 3086 pages / reclaimed 929 = 回收飢餓。
4. 鎖持有者不可中斷（等 swap I/O）→ 餵食路徑無法完成 check-in → AppleARMWatchdogTimer 90s 到期 → panic。
5. ComfyX GPU 閒置 16h 但佔 48.8GB——「GPU 空閒 ≠ 統一記憶體可用」由本事故直接證明。

**HYPOTHESIS（合理但未單因子證實）：**
- 直接觸發 = **疊加序列而非單點**：基座 ComfyX 48.8GB 常駐 + strata/director ~55GB 上，04:0x–04:13 先有 1280 級引擎 arm 接連 abort（jetsam 風暴自 04:10 起），04:18 再疊 chain 60GB 派發（單次最重）；系統撐約 50 分鐘（04:10→05:00）後在回收飢餓中停擺。教訓：啟動即 abort 的 arm 仍贡献分配抖動，失敗的 heavy 嘗試不等於零成本。
- strata 卡死模式 = swap 耗盡下其執行緒等 paging I/O 而持鎖（paging deadlock）；其卡入時機未獨立量測。
- 互為放大器：04:18–04:59 期間 compress/swap 抖動拖慢所有執行緒，最後無可回收頁可用。
- SMB／外接 SSD／Metal driver 獨立 stall：無獨立證據（log 流量至 04:59 正常、Metal 16h 空閒）；但 h3.c chain_p4h.log 佐證壓力窗口內外接碟 I/O 受干擾：04:0x–04:13 四個 arm 全 rc=1/mp4=0 快速失敗（其中 corgi arm 的 grep 對 /Volumes/SSD2T 報 "Interrupted system call"），數分鐘前同引擎同配方兩支 544 arm 才 rc=0（RSS ~11GB）→ SSD stall 是真實放大器、仍屬次生於記憶體壓力，非獨立根因。分頁子系統飽和本身就是最終 I/O 瓶頸。

**NOT_VERIFIED：** 24h/72h soak（cron 已排 11-29、12-01 09:00）；閘門門檻 A/B 校準；Strata `--ram-keep-gib` 效果；ComfyX 閒置卸載節律於正式產線。

## 三、責任歸屬（遵守「不缺證據不归咎」）

- 無單一程序可單獨歸咎：**派發政策失守**（兩個 40GB+ 重負載疊加在 ~90GB 常駐基座上、無申報、無閘門、無卸載）是系統性成因。
- ComfyX 閒置常駐 48.8GB 與 chain 60GB 是最大兩塊（coalition 1169 = Hermes 派發，合計 109GB）；strata 是死鎖持有人與最大 wired 常駐（57GB wired 現值同樣高）。
- 0 次 jetsam = 系統層無解體閥；macOS watchdog 只是最後炸點（政策：不動 watchdog、不動 swap 設定）。

## 四、 corrective 措施（已落地）

1. **Resource Governor daemon**：~/AI/workbench/governor/rgov.py，:8095，LaunchAgent com.workbench.rgov 常駐（gateway 內 launchctl 回 503 → 用 osascript do shell script 途徑）。端點：/status /admit /register /workers /queue /events /metrics /panics；事件流 state/events.jsonl 低負載持久化（重開機後可查）；指標 state/metrics/m-*.jsonl 每 5 分鐘一筆。
2. **門檻（可配置 rgov.json）**：free+inactive <20% 預警、<10% 禁重型；swap 10GB 預警/25GB 禁；SSD 保留 100GB；committed（service 申報 ∪ 常駐 footprint）102GB 預警/115GB 禁；單程序 footprint 55GB 預警/75GB 事件；swap 成長 >6GB/15min、程序成長 >8GB/30min 告警。
3. **派發契約**：每任務 /admit 申報 peak_ram/metal/disk、class、priority、checkpoint_support（RAM 與 Metal 不重複計：metal_gb 另列、合併計算）；已示範：當前狀態 40GB 實驗 → **DENY (committed 135>115)**。
4. **已登記 service**：strata 62GB、comfyx-8188 50GB、director 19GB、director-alpha6 13GB（priority high，優先於 h3.c 實驗）。
5. **ComfyX 生產恢復**：以 pre_ab20261121/launch_command.txt 原命令重啟（cwd=~/ComfyUI-Installs/ComfyUI），正式配方未改。
6. **工作臺整合**：daemon 每 5 分鐘心跳 PATCH workbench（board 狀態、gate 模式、禁止派發原因）；跨機登錄 ai-0008。
7. 24h／72h soak 驗收 cron（job 1292548726bd / dfbb9efc5784）。

## 五、殘餘風險與建議（待驗收）

- 現值：wired 57GB + 常駐 footprint ~92GB，基座依然高。建議（未批准不改正式配方）：
  - ComfyX：佇列閒置即 `POST /free {"unload_models":true,"free_memory":true}`（11-21 A/B 實測釋放 ~46GB、swap 72→25.5GB）——最大單項槓桿。
  - Strata：試 `--ram-keep-gib`（尚未實測）；研究負載與 chain 比對須過 /admit。
  - Hermes 子代理派發前一律 /admit（curl 契約見 skill macos-unified-memory-governor）。
