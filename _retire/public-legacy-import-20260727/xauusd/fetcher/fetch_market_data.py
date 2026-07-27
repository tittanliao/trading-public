import yfinance as yf
import requests
import pandas as pd
import os
import time
from datetime import datetime

# 🌟 填入你的 Twelve Data API Key
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")

output_dir = "/Users/tittan/googledrive/XAUUSD/weekly report/csv"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def fetch_twelvedata_xau():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 啟動引擎一：Twelve Data 獲取 XAUUSD...")
    ticker = "XAU/USD"
    safe_name = "XAUUSD"
    intervals = {"30min": "30m", "1h": "1h", "4h": "4h", "1day": "1d", "1week": "1wk"}
    
    for api_int, file_int in intervals.items():
        print(f"  正在下載 {safe_name} - {file_int}...")
        # 加上 &timezone=UTC 確保基準一致
        url = f"https://api.twelvedata.com/time_series?symbol={ticker}&interval={api_int}&outputsize=500&apikey={TWELVEDATA_API_KEY}&timezone=UTC"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            if "values" in data:
                df = pd.DataFrame(data["values"])
                df['Datetime'] = pd.to_datetime(df['datetime'])
                df = df.set_index('Datetime')
                
                # 🌟 時區轉換魔法：告訴 Pandas 這是 UTC，然後轉成台北時間，最後移除時區標籤保持乾淨
                df.index = df.index.tz_localize('UTC').tz_convert('Asia/Taipei').tz_localize(None)
                
                df = df[['open', 'high', 'low', 'close']]
                df.columns = ['Open', 'High', 'Low', 'Close']
                df['Volume'] = 0 
                df = df.sort_index(ascending=True) 
                
                filename = f"{output_dir}/{safe_name}_{file_int}.csv"
                df.to_csv(filename)
                print(f"    ✅ 已儲存: {filename} (已轉換為台灣時間)")
            else:
                print(f"    ⚠️ 無法取得資料: {data.get('message', '未知錯誤')}")
            time.sleep(2) 
            
        except Exception as e:
            print(f"    ❌ 發生錯誤: {e}")

def fetch_yfinance_mgc():
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 啟動引擎二：yfinance 獲取 MGC...")
    tickers = {
        "MGC": "MGC=F"
    }
    intervals = {
        "30m": "60d",
        "1h": "730d",
        "1d": "5y",
        "1wk": "5y"
    }
    
    for name, ticker in tickers.items():
        stock = yf.Ticker(ticker)
        
        for interval, period in intervals.items():
            print(f"  正在下載 {name} - {interval}...")
            try:
                df = stock.history(period=period, interval=interval)
                if df.empty:
                    print(f"    ⚠️ 無法取得 {name} {interval} 的資料")
                    continue
                
                # 🌟 時區轉換魔法：yfinance 通常帶有美東時區 (America/New_York)
                if df.index.tz is not None:
                    df.index = df.index.tz_convert('Asia/Taipei').tz_localize(None)
                else:
                    df.index = df.index.tz_localize('UTC').tz_convert('Asia/Taipei').tz_localize(None)

                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                df.index.name = 'Datetime'
                
                filename = f"{output_dir}/{name}_{interval}.csv"
                df.to_csv(filename)
                print(f"    ✅ 已儲存: {filename} (已轉換為台灣時間)")
                
                if interval == "1h":
                    print(f"  正在合成 {name} - 4h...")
                    df_4h = df.resample('4h').agg({
                        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                    }).dropna()
                    
                    filename_4h = f"{output_dir}/{name}_4h.csv"
                    df_4h.to_csv(filename_4h)
                    print(f"    ✅ 已儲存: {filename_4h} (已轉換為台灣時間)")
                    
            except Exception as e:
                print(f"    ❌ 下載 {name} {interval} 時發生錯誤: {e}")

if __name__ == "__main__":
    print("🚀 開始執行市場資料獲取任務...")
    #fetch_twelvedata_xau()
    fetch_yfinance_mgc()
    print("\n🎉 5 份 CSV 資料全數下載/合成完畢！所有時間皆已對齊台灣時間 (UTC+8)。")
