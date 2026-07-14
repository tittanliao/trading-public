import pandas as pd
import pandas_ta as ta
import os
import sys
from datetime import datetime

INPUT_DIR = "market_data_csv"
DEFAULT_SYMBOL = "XAUUSD"

# 商品名稱映射
SYMBOLS = {
    "XAUUSD": "現貨黃金",
    "DXY": "美元指數",
    "MGC": "微型黃金期貨"
}

def load_and_calculate(symbol, interval):
    """讀取 CSV 並計算所有需要的技術指標"""
    filepath = f"{INPUT_DIR}/{symbol}_{interval}.csv"
    if not os.path.exists(filepath):
        print(f"找不到檔案: {filepath}")
        return None
    
    df = pd.read_csv(filepath, index_col="Datetime", parse_dates=True)
    df = df.sort_index(ascending=True)
    
    # 計算指標
    df.ta.ema(length=3, append=True)
    df.ta.sma(length=3, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.ao(append=True)
    df.ta.true_range(append=True)

    # 重新命名欄位
    rename_map = {
        'EMA_3': 'Fast_EMA_3',
        'SMA_3': 'MA_3',
        'BBL_20_2.0': 'BB_Lower',
        'BBM_20_2.0': 'BB_Basis',
        'BBU_20_2.0': 'BB_Upper',
        'ATRr_14': 'ATR_14',
        'AO_5_34': 'AO',
        'TRUERANGE_1': 'TR'
    }
    # pandas-ta a few versions ago changed its bbands column names. this handles both.
    if 'BBL_20_2.0_2.0' in df.columns:
        rename_map['BBL_20_2.0_2.0'] = 'BB_Lower'
        rename_map['BBM_20_2.0_2.0'] = 'BB_Basis'
        rename_map['BBU_20_2.0_2.0'] = 'BB_Upper'
        
    df.rename(columns=rename_map, inplace=True)
    
    return df

def generate_market_summary(symbol, output_filename):
    print(f"開始為 {symbol} 計算技術指標並生成摘要...")
    
    df_1wk = load_and_calculate(symbol, "1wk")
    df_1d = load_and_calculate(symbol, "1d")
    df_4h = load_and_calculate(symbol, "4h")
    df_1h = load_and_calculate(symbol, "1h")
    
    if any(df is None for df in [df_1wk, df_1d, df_4h, df_1h]):
        print(f"{symbol} 的資料不完整，無法生成摘要！")
        return

    summary = []
    summary.append(f"資料生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    summary.append(f"商品: {symbol} ({SYMBOLS.get(symbol, 'N/A')})\n")
    
    # 1. 四週大格局全覽
    summary.append("【四週大格局全覽】")
    last_4_weeks = df_1wk.tail(4)
    for i in range(4):
        row = last_4_weeks.iloc[3-i] 
        date_str = row.name.strftime('%Y-%m-%d')
        open_p, high_p, low_p, close_p = row['Open'], row['High'], row['Low'], row['Close']
        tr = row.get('TR', high_p - low_p)
        change = close_p - open_p
        summary.append(f"W-{i+1} ({date_str}): 開={open_p:.2f}, 高={high_p:.2f}, 低={low_p:.2f}, 收={close_p:.2f}, 漲跌={change:.2f}, 週波幅(TR)={tr:.2f}")
    summary.append("")

    # 2. 關鍵價位與指標
    summary.append("【關鍵價位與策略指標 (最新狀態)】")
    
    latest_1d = df_1d.iloc[-1]
    summary.append(f"[Daily 級別] 收盤價: {latest_1d['Close']:.2f}")
    summary.append(f"- ATR(14): {latest_1d.get('ATR_14', 0):.2f}")
    summary.append(f"- Fast EMA(3): {latest_1d.get('Fast_EMA_3', 0):.2f}")
    if 'BB_Upper' in latest_1d:
        summary.append(f"- Bollinger Bands: 上軌={latest_1d['BB_Upper']:.2f}, 中軌={latest_1d['BB_Basis']:.2f}, 下軌={latest_1d['BB_Lower']:.2f}")
    summary.append(f"- AO 指標: {latest_1d.get('AO', 0):.2f}")
    summary.append("- 前三日 ATR(14): " + ", ".join([f"{x:.2f}" for x in df_1d['ATR_14'].tail(4).head(3).tolist()]))
    summary.append("")

    latest_4h = df_4h.iloc[-1]
    summary.append(f"[4H 級別] 收盤價: {latest_4h['Close']:.2f}")
    summary.append(f"- ATR(14): {latest_4h.get('ATR_14', 0):.2f}")
    summary.append(f"- Fast EMA(3): {latest_4h.get('Fast_EMA_3', 0):.2f}")
    if 'BB_Upper' in latest_4h:
        summary.append(f"- Bollinger Bands: 上軌={latest_4h['BB_Upper']:.2f}, 中軌={latest_4h['BB_Basis']:.2f}, 下軌={latest_4h['BB_Lower']:.2f}")
    summary.append(f"- AO 指標: {latest_4h.get('AO', 0):.2f}")
    summary.append("")

    latest_1h = df_1h.iloc[-1]
    summary.append(f"[1H 濾網] 收盤價: {latest_1h['Close']:.2f}")
    summary.append(f"- MA(3): {latest_1h.get('MA_3', 0):.2f}")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(summary))
        
    print(f"✅ 成功為 {symbol} 生成摘要檔案: {output_filename}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        symbol = DEFAULT_SYMBOL
    
    output_file = f"{symbol}_Summary.txt"
    generate_market_summary(symbol, output_file)
