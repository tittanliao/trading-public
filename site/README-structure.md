# 網站資訊架構

> 2026-08-30 起採單一 canonical 網站。不再用 `v1.0`、`v2.0` 或 `/zh/` 區分內容。

## 閱讀原則

- 首頁與各商品的 Sheet／研究索引維持英文，讓短標籤、欄位與跨工具名稱一致。
- 研究標題維持英文；進入研究後，問題、結論、證據說明與限制以繁體中文為主。
- 研究清單預設只有表格，不再同時產生 Cards／Table 兩套閱讀模式。
- Cards 只保留在首頁、每週展望或少量需要突出行動的入口，不拿來承載長篇研究。
- 每篇研究只有一個 canonical URL：`/research/studies/<study-id>/`。

## 目前結構

```text
/                    English homepage
/xauusd/             English XAUUSD sheet：signal context、weekly、study table
/tx/                 English TX sheet：study table
/research/           English cross-market study table
/lessons/            English negative-results／methodology sheet
/jargon/             中英對照術語
/research/studies/   English title + 中文研究內文
/v1/*                相容轉址，不是舊站
/zh/*                相容轉址，不是中文副站
```

Git 歷史就是舊版封存，因此不在 Public 再複製一套 `_retire` 網站。舊 URL 只留下輕量轉址，
避免書籤失效，也避免兩套內容繼續競爭。

## 新研究報告順序

1. 英文標題與研究識別
2. 中文研究問題
3. 一列表格交代樣本、主檢定與最重要的界限
4. 「判讀／結論／證據與限制」表格
5. 支撐結論的明細表格；圖只在無法用表格表達時使用
6. 實務影響、限制、方法與公開證據邊界
7. Results JSON、Python method、Study manifest

第一個完成遷移的範例是 `RS-XAUUSD-20260823-001`。其餘研究沿用原 canonical URL，
逐篇換版，不建立任何新的版本路徑。

## 加入新研究

1. 依 `docs/RESEARCH_PUBLICATION_SPEC.md` 完成審閱與三檔 Public package。
2. `study.json` 的 `title` 使用英文；新增或維護 `question_zh`、`findings[*].title_zh`、
   `findings[*].detail_zh`。
3. 優先使用既有表格 renderer；只有資料形狀真的不同才新增 generic renderer。
4. 執行 `python3 site/build.py`、`python3 site/build.py --check`、`python3 site/check.py`。
5. 確認 canonical 頁面沒有版本切換、語言切換或私人資料後再發布。
