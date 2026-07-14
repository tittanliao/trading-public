# DEVELOPMENT.md — 開發規範

> 建立於 2026-07-11（網站拆頁重構同日）。
> 這份文件存在的原因：過去反覆發生「改了 A 忘了 B」的漏改事故（見文末事故記錄）。
> 所有規則都對應一個真實踩過的坑，不是形式主義。

---

## 1. 網站架構總覽（2026-07-11 拆頁後）

```
trading/
├── generate_site.py        ★ 唯一的網站生成器（舊 generate_index.py 已刪除）
├── assets/
│   ├── site.css            共用樣式（唯一 CSS 來源）
│   └── site.js             共用 JS（showMain/showTab/深連結）
├── content/                ★ 手寫內容 fragment（唯一合法手寫處）
│   ├── xauusd/
│   │   ├── opt.html            已確認策略區塊（整區手寫）
│   │   ├── weekly.html         週報分析區塊
│   │   ├── h2.html             2026 H1/H2 複盤區塊
│   │   ├── macro_indicator.html 宏觀指標解讀子分頁
│   │   └── macro_backtest.html  Macro 回測子分頁
│   ├── tx/
│   │   └── zheng2.html         正二回測區塊
│   └── sitemap.html            網站地圖整頁
├── data/
│   └── logs.json           ★ 對話記錄唯一來源（計數自動）
│
├── index.html              ⚠️ 生成物，禁止手改 ── Hub 首頁
├── xauusd.html             ⚠️ 生成物，禁止手改 ── XAUUSD 主頁
├── tx.html                 ⚠️ 生成物，禁止手改 ── TX 主頁
├── shared.html             ⚠️ 生成物，禁止手改 ── 跨商品分析
├── history.html            ⚠️ 生成物，禁止手改 ── 對話記錄
└── sitemap.html            ⚠️ 生成物，禁止手改 ── 網站地圖
```

生成指令（在 trading/ 根目錄）：
```bash
python3.12 generate_site.py              # 全部 6 頁
python3.12 generate_site.py --page xauusd  # 只生成單頁（降低風險）
```

## 2. 三類內容，各有唯一修改入口

| 內容類型 | 唯一修改入口 | 之後做什麼 |
|---------|------------|-----------|
| 動態數據（實驗排名、驗證表、熱力圖）| 對應 data 檔（`results.json`、`shared_results.json`、`doc/validation_results.json`…）| 重跑 `generate_site.py` |
| 手寫編輯內容（策略說明、H2 複盤、正二、sitemap）| `content/` 下對應 fragment | 重跑 `generate_site.py` |
| 對話記錄 | `data/logs.json` append 一筆 | 重跑 `generate_site.py`（「共 N 筆」自動算）|

**鐵律：6 個生成頁一律不手改。** 每頁開頭有 DO NOT EDIT banner；手改的內容會在下次生成時被覆蓋消失。想改頁面上的任何字，先問「它的來源是 fragment、data 檔、還是 generate_site.py 裡的函式？」，改來源。

## 3. Public evidence chain

| 資料 | Public evidence | Published copy |
|------|-----------------|----------------|
| 策略版本 / 參數 / 績效 | 可重跑程式、輸入資料版本／雜湊、產生報告與 Git commit | `xauusd/CLAUDE.md` 績效表、`content/xauusd/opt.html` |
| 私人倉位 / 風控規則 | 不在 Public repository 指定 Source of Truth | 無 |
| 對話記錄 | `data/logs.json` | 無，全部由生成器渲染 |
| 網站頁面結構 | `generate_site.py` | 無 |

規則：發現數字不一致時，先標記差異，再回溯產生程式、輸入版本、報告與 Git commit；不得從 Claude 或其他模型的私人記憶直接覆寫 Public 文件。

## 4. 常見任務 Checklist

### A. 新增一份分析報告（如 report_xxx.html）
- [ ] 報告檔放進對應策略資料夾（如 `xauusd/XAUUSD-Long-S1-AweWithBB/`）
- [ ] `data/logs.json` append 一筆記錄（含報告連結）
- [ ] `content/sitemap.html` 補上報告連結
- [ ] 重跑 `generate_site.py`
- [ ] commit（報告 + logs.json + sitemap fragment + 生成頁一起）

### B. 策略升版（如 V3.7 → V3.8）
- [ ] Pine 檔進策略資料夾，命名依 `VX.Y`（確認）/`VX.Y+1.1`（測試）規則
- [ ] 記錄可重跑程式、輸入資料版本／雜湊、產生報告與 Git commit
- [ ] 依同一 evidence chain 同步 `xauusd/CLAUDE.md` 績效表與 `content/xauusd/opt.html` 版本表格
- [ ] `data/logs.json` append 記錄
- [ ] 重跑 `generate_site.py`
- [ ] commit

### C. 網站新增頁面 / 區塊
- [ ] 手寫內容 → 新 fragment 進 `content/`；動態內容 → `generate_site.py` 加函式
- [ ] `generate_site.py` 的對應 build 函式掛上新區塊 + subnav 按鈕
- [ ] `content/sitemap.html` 登記
- [ ] `data/logs.json` append 記錄
- [ ] 重跑 `generate_site.py`，`git diff --stat` 檢查
- [ ] 本檔（DEVELOPMENT.md）第 1 節目錄圖若受影響，一併更新

### D. 日常「請分析」/ 交易紀錄（不動網站）
- 私人日常分析與交易紀錄不在 Public repository 維護，依 Private governance 執行

## 5. Commit 前強制檢查

1. `git status --short` — 確認沒有意料之外的檔案變動
2. `git diff --stat` — **出現大量非預期刪除（-100 行以上且你沒有刻意刪東西）→ 立刻停下**，先弄清原因再 commit（20260710 事故就是這樣造成的）
3. 生成頁的 diff 要人工掃一眼：新增的內容在、舊的內容沒消失
4. commit message 用中文描述「做了什麼」，格式沿用 git log 既有慣例（`feat:`/`fix:`/`docs:`/`refactor:`）
5. commit 後 push（GitHub Pages 自動部署整個 repo）

## 6. 檔案放置規範

| 類型 | 位置 |
|------|------|
| 一次性 / 半一次性分析腳本 | `xauusd/scripts/`（在 trading/ 根目錄執行）|
| 回測引擎模組 | `xauusd/experiments/`、`xauusd/analysis/` |
| 策略 Pine + 交易 CSV + 報告 | 各策略資料夾（`XAUUSD-Long-S1-AweWithBB/` 等）|
| CSV 資料 | `xauusd/csv/`、`tx/csv/`（TradingView 匯出直接覆蓋）|
| Claude 分析紀錄 | `xauusd/claude/`（daily/、trade_journal/、reports/）|
| 臨時檔 | 不要進 repo（用系統 scratchpad）|

## 7. 歷史事故記錄（規則的由來）

| 日期 | 事故 | 對應規則 |
|------|------|---------|
| 2026-07-10 | 跑舊 generate_index.py 整份重新生成，誤刪 index.html 裡 2000+ 行手寫內容（生成器與產出物早已脫節）| §2 鐵律、§5-2 diff 檢查；根治：拆頁重構，手寫內容全部進 content/ |
| 2026-07-10 | 發現 ANALYSIS_SKILL.md 記載「V3.4 TP2=2:1」與真實成交 CSV 不符（兩版出場結構其實相同）| §3 法定來源規則：寫數字前先查原始資料 |
| 2026-07-11 | 對話記錄手寫計數「共 18 筆」，實際有 22 筆 | §2：計數改由 logs.json 自動計算 |
| 2026-07-05 | 三套 .claude/memory 績效數字各自為政；context.md 輪替規則設計後從未執行 | §3 抄本規則；規則要附驗收機制 |
| 2026-06-13 W24 | 合併週報用了 W23 舊報告、CFTC 漏抓最新 | ANALYSIS_SKILL.md 週報 SOP（非本檔範圍，列此供對照）|
