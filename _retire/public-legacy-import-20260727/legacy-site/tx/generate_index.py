#!/usr/bin/env python3
"""
Generate root index.html.
Run after run_experiments.py and run_short_experiments.py.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent

# ─── Session log entries — append new entries here ────────────────
SESSION_LOG = [
    {
        "date": "2026-05-12",
        "title": "架構設計 — 從 XAUUSD 擴展到台指期",
        "items": [
            "決定標的：MTX 小台指期（每點 NT$50）",
            "確認資料來源：TradingView CSV（同 xauusd 格式）",
            "相關性指標：NQ 納斯達克期貨（同向，可切換 SOX/SPX/VIX）",
            "SL/TP 改為固定點數（30pts/60pts，R:R 1:2）",
            "時段：日盤（08:45–13:45）+ 夜盤（15:00–05:00）皆納入",
            "建立 CLAUDE.md、.claude/memory/（記憶 git 同步架構）",
        ],
    },
    {
        "date": "2026-05-13",
        "title": "第一版回測引擎完成 + Pine Script 生成",
        "items": [
            "完成 experiments/ 模組：loader、indicators、engine、runner、report",
            "20 多單策略 E01–E20（Oscillator / Trend / Breakout / Pattern / Session 五大類）",
            "20 空單策略 S01–S20（對應多單鏡像邏輯）",
            "真實資料回測（TAIFEX_DLY_MXF1!, 30.csv，2025-06 起 8585 bars）",
            "多單 E12 BB Squeeze Break 唯一獲利（PF=1.132, WR=36.1%）",
            "空單全部虧損 — 資料期間為大多頭，做空困難，為預期結果",
            "Pine Script v6 多空各一檔：下拉選單選策略、Enable 開關、R:R 可設定",
            "下一步：加入 NQ 相關性過濾 + MTF 過濾提升勝率",
        ],
    },
]


def _load_top3(results_path: Path, n: int = 3) -> list[dict]:
    if not results_path.exists():
        return []
    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)
    return [r for r in data["results"] if r["n_trades"] > 0][:n]


def _row(r: dict, direction: str) -> str:
    colour = "#1565c0" if direction == "long" else "#880e4f"
    pnl = r.get("net_pnl_ntd", 0)
    pnl_colour = "#2e7d32" if pnl > 0 else "#c62828"
    return (
        f"<tr>"
        f"<td style='color:{colour};font-weight:bold'>{r.get('rank','-')}. {r['code']}</td>"
        f"<td>{r['name']}</td>"
        f"<td>{r['win_rate']}%</td>"
        f"<td>{r['profit_factor']:.3f}</td>"
        f"<td style='color:{pnl_colour}'>NT${pnl:,.0f}</td>"
        f"<td>{r.get('score',0):.3f}</td>"
        f"</tr>"
    )


def _session_log_html() -> str:
    blocks = []
    for entry in SESSION_LOG:
        items_html = "".join(f"<li>{it}</li>" for it in entry["items"])
        blocks.append(
            f"<div class='log-entry'>"
            f"<div class='log-date'>{entry['date']}</div>"
            f"<div class='log-title'>{entry['title']}</div>"
            f"<ul class='log-items'>{items_html}</ul>"
            f"</div>"
        )
    return "\n".join(blocks)


def generate():
    long_top3  = _load_top3(ROOT / "TX-Long-Experiments"  / "results.json")
    short_top3 = _load_top3(ROOT / "TX-Short-Experiments" / "results.json")
    long_rows  = "\n".join(_row(r, "long")  for r in long_top3)
    short_rows = "\n".join(_row(r, "short") for r in short_top3)
    long_link  = "TX-Long-Experiments/report.html"
    short_link = "TX-Short-Experiments/report.html"
    long_pine  = "TX-Long-Experiments/pine/ALL_Long_Strategies.pine"
    short_pine = "TX-Short-Experiments/pine/ALL_Short_Strategies.pine"
    long_exists  = (ROOT / long_link).exists()
    short_exists = (ROOT / short_link).exists()

    html = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="utf-8">
<title>TX — 台指期策略分析</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:'Segoe UI',Arial,sans-serif;background:#f0f2f5;color:#212121}}
  .hero{{background:linear-gradient(135deg,#1a237e 0%,#4a148c 100%);color:white;padding:40px 32px 32px}}
  .hero h1{{font-size:2em;margin-bottom:6px}}
  .hero p{{opacity:.85;font-size:.95em}}
  .container{{max-width:1200px;margin:0 auto;padding:28px 16px}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px}}
  .card{{background:white;border-radius:10px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,.1)}}
  .card-header{{padding:16px 20px;font-weight:600;font-size:1.05em;color:white}}
  .card-long{{background:#1565c0}}.card-short{{background:#880e4f}}
  .card-body{{padding:16px 20px}}
  table{{width:100%;border-collapse:collapse;font-size:.86em}}
  th{{background:#f5f5f5;padding:7px 10px;text-align:left;color:#555;font-weight:600}}
  td{{padding:7px 10px;border-bottom:1px solid #eee}}
  .btn{{display:inline-block;margin-top:10px;padding:8px 16px;border-radius:6px;text-decoration:none;
        font-weight:600;font-size:.88em;color:white;margin-right:8px}}
  .btn-long{{background:#1565c0}}.btn-short{{background:#880e4f}}
  .btn-pine{{background:#2e7d32}}
  .info-card{{background:white;border-radius:10px;padding:20px;
              box-shadow:0 2px 8px rgba(0,0,0,.1);margin-bottom:20px}}
  .info-card h2{{font-size:1em;color:#333;margin-bottom:12px;
                 border-left:4px solid #1a237e;padding-left:10px}}
  .spec-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
  .spec{{background:#f8f9fa;border-radius:6px;padding:12px 14px}}
  .spec .label{{font-size:.78em;color:#777;margin-bottom:4px}}
  .spec .value{{font-size:1.1em;font-weight:600;color:#333}}
  .notice{{background:#fff8e1;border:1px solid #ffcc02;border-radius:6px;
           padding:12px 16px;font-size:.85em;color:#5d4037;margin-top:16px}}
  /* Session log */
  .log-entry{{display:flex;gap:20px;padding:16px 0;border-bottom:1px solid #eee}}
  .log-entry:last-child{{border-bottom:none}}
  .log-date{{font-size:.8em;color:#888;white-space:nowrap;min-width:90px;padding-top:2px}}
  .log-title{{font-weight:600;color:#1a237e;margin-bottom:6px}}
  .log-items{{padding-left:18px;font-size:.88em;color:#444;line-height:1.7}}
  @media(max-width:800px){{.grid,.spec-grid{{grid-template-columns:1fr}};.log-entry{{flex-direction:column;gap:4px}}}}
</style>
</head>
<body>

<div class="hero">
  <h1>TX — 台指期小台（MTX）策略分析</h1>
  <p>20 多單策略 × 20 空單策略 · SL 30pts / TP 60pts · R:R 可調 · 每點 NT$50</p>
</div>

<div class="container">

  <div class="info-card">
    <h2>回測規格</h2>
    <div class="spec-grid">
      <div class="spec"><div class="label">商品</div><div class="value">MTX 小台</div></div>
      <div class="spec"><div class="label">時間框架</div><div class="value">30 分鐘</div></div>
      <div class="spec"><div class="label">預設止損</div><div class="value">30 pts</div></div>
      <div class="spec"><div class="label">預設止盈</div><div class="value">60 pts</div></div>
      <div class="spec"><div class="label">時間止損</div><div class="value">48 bars (24h)</div></div>
      <div class="spec"><div class="label">每點價值</div><div class="value">NT$50</div></div>
      <div class="spec"><div class="label">交易時段</div><div class="value">日盤+夜盤</div></div>
      <div class="spec"><div class="label">R:R</div><div class="value">1:2 (可調)</div></div>
    </div>
    <div class="notice">
      📅 資料：<code>TAIFEX_DLY_MXF1!, 30.csv</code>（真實 MTX 行情，2025-06 起 8585 bars）。
      更新資料後重新執行 <code>python run_experiments.py &amp;&amp; python run_short_experiments.py &amp;&amp; python generate_index.py</code>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <div class="card-header card-long">多單實驗 Top 3 (E01–E20)</div>
      <div class="card-body">
        <table>
          <thead><tr><th>代碼</th><th>策略</th><th>勝率</th><th>PF</th><th>淨盈虧</th><th>分數</th></tr></thead>
          <tbody>{long_rows or "<tr><td colspan='6' style='color:#999'>尚無資料</td></tr>"}</tbody>
        </table>
        {"<a href='" + long_link + "' class='btn btn-long'>多單報告 →</a>" if long_exists else ""}
        <a href="{long_pine}" class="btn btn-pine" download>Pine Script ↓</a>
      </div>
    </div>

    <div class="card">
      <div class="card-header card-short">空單實驗 Top 3 (S01–S20)</div>
      <div class="card-body">
        <table>
          <thead><tr><th>代碼</th><th>策略</th><th>勝率</th><th>PF</th><th>淨盈虧</th><th>分數</th></tr></thead>
          <tbody>{short_rows or "<tr><td colspan='6' style='color:#999'>尚無資料</td></tr>"}</tbody>
        </table>
        {"<a href='" + short_link + "' class='btn btn-short'>空單報告 →</a>" if short_exists else ""}
        <a href="{short_pine}" class="btn btn-pine" download>Pine Script ↓</a>
      </div>
    </div>
  </div>

  <div class="info-card">
    <h2>Pine Script 使用說明</h2>
    <ol style="padding-left:20px;line-height:2;font-size:.9em">
      <li>下載 <code>ALL_Long_Strategies.pine</code> 或 <code>ALL_Short_Strategies.pine</code></li>
      <li>在 TradingView Pine Script Editor 貼上，套用到 <code>TAIFEX:MXF1!</code> 30m 圖表</li>
      <li><b>Strategy</b> 下拉選單選擇策略（E01–E20 / S01–S20）</li>
      <li><b>Enable Signals</b> 開關控制是否進場（關閉後仍顯示灰色參考箭頭）</li>
      <li><b>Stop Loss</b> 設定止損點數；<b>R:R Ratio</b> 自動計算止盈（TP = SL × R:R）</li>
      <li><b>Session</b> 可單獨開關日盤/夜盤</li>
    </ol>
  </div>

  <div class="info-card">
    <h2>對話記錄 / Session Log</h2>
    {_session_log_html()}
  </div>

</div>
</body>
</html>
"""

    out = ROOT / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"index.html → {out}")


if __name__ == "__main__":
    generate()
