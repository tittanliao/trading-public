# 網站資訊架構

> 2026-08-30 起採單一 canonical 網站。不再用 `v1.0`、`v2.0` 或 `/zh/` 區分內容。

## 閱讀原則

- 首頁與各商品的 Sheet／研究索引維持英文，讓短標籤、欄位與跨工具名稱一致。
- 研究標題維持英文；進入研究後，問題、結論、證據說明與限制以繁體中文為主。
- 研究清單預設只有表格，不再同時產生 Cards／Table 兩套閱讀模式。
- Cards 只保留在每週展望或少量需要突出行動的入口，不拿來承載研究目錄與研究報告。
- 每篇研究只有一個 canonical URL：`/research/studies/<study-id>/`。

## 目前結構

```text
/                    English directory table
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

## 研究報告順序

1. 英文標題與研究識別
2. 中文研究問題
3. 「判讀／結論／如何閱讀」表格
4. 核心數據表
5. `results.json` 已登錄的研究圖表；圖題、座標與圖內標註保留英文
6. 研究識別表
7. 實務影響與公開證據邊界
8. Results JSON、Python method、Study manifest

31 篇研究全部使用同一個 reader renderer，不再依 `results.json` 的十一種資料形狀建立
十一套頁面架構。已審閱完整中文 findings 的研究會顯示完整結論表；其餘研究顯示中文主結論
與關鍵限制，完整結構化數值與方法仍由頁尾連結提供。研究 Python 繼續負責可重跑計算與產圖，
不為網頁翻譯改寫；reader 會依 `results.json.charts` 自動載入圖表。任何已登錄圖表缺檔或未出現在
頁面，`site/check.py` 都必須失敗。

## 加入新研究

1. 依 `docs/RESEARCH_PUBLICATION_SPEC.md` 完成審閱與三檔 Public package。
2. `study.json` 的 `title` 使用英文；中文問題與摘要可由 `study.json` 既有欄位提供，
   或先登錄於 `site/study_copy_zh.json`。完整翻譯完成後再補
   `findings[*].title_zh`／`findings[*].detail_zh`。
3. 所有研究共用 reader renderer；新增結果資料形狀不需要新增頁面架構。
4. 執行 `python3 site/build.py`、`python3 site/build.py --check`、`python3 site/check.py`。
5. 確認 canonical 頁面沒有版本切換、語言切換或私人資料後再發布。
