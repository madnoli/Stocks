import requests
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import urlencode
import os

BASE_URL = "http://localhost:3000/api/equity/historical"
SYMBOL = "VEDL"
DATE_START = (datetime.now().date() - timedelta(days=14)).strftime("%Y-%m-%d")
DATE_END = (datetime.now().date()).strftime("%Y-%m-%d")
OUTPUT_DIR = "output_intraday"
REQUESTED_MINUTES = [5, 10, 15, 30, 60]  # intraday targets

def fetch_eod(symbol: str, ds: str, de: str):
    url = f"{BASE_URL}/{symbol}?{urlencode({'dateStart': ds, 'dateEnd': de})}"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    # The sample response is a list containing one object with "data"
    if isinstance(data, list) and data and isinstance(data[0], dict) and "data" in data[0]:
        return data[0]["data"]
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    return []

def normalize_daily(rows: list) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["date","open","high","low","close","volume"])
    out = []
    for it in rows:
        # TIMESTAMP is ISO string at 18:30Z, representing the trading date’s session end
        ts = pd.to_datetime(it.get("TIMESTAMP"), utc=True, errors="coerce")
        # Fallback to CH_TIMESTAMP (YYYY-MM-DD) if needed
        if ts is pd.NaT:
            ts = pd.to_datetime(it.get("CH_TIMESTAMP"), errors="coerce")
        if ts is pd.NaT:
            continue
        out.append({
            "date": ts.tz_convert("Asia/Kolkata") if ts.tzinfo else ts,  # keep as IST if tz-aware
            "open": float(it.get("CH_OPENING_PRICE", 0) or 0),
            "high": float(it.get("CH_TRADE_HIGH_PRICE", 0) or 0),
            "low": float(it.get("CH_TRADE_LOW_PRICE", 0) or 0),
            "close": float(it.get("CH_CLOSING_PRICE", 0) or 0),
            "volume": float(it.get("CH_TOT_TRADED_QTY", 0) or 0),
            "vwap": float(it.get("VWAP", 0) or 0),
        })
    df = pd.DataFrame(out).dropna(subset=["date"]).sort_values("date").set_index("date")
    return df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = fetch_eod(SYMBOL, DATE_START, DATE_END)
    df_daily = normalize_daily(rows)

    if df_daily.empty:
        print("No daily rows returned from the API for the requested window.")
        return

    daily_path = os.path.join(OUTPUT_DIR, f"{SYMBOL}_daily_{DATE_START}_{DATE_END}.csv")
    df_daily.to_csv(daily_path, float_format="%.4f")
    print(f"Saved daily EOD data -> {daily_path}")

    # Intraday request cannot be fulfilled from daily candles
    print("The current endpoint provides daily OHLCV only; 5/10/15/30/60-minute bars require an intraday endpoint.")

    # Suggest intraday endpoint pattern and how to switch:
    print("\nTo produce intraday timeframes, switch BASE_URL to an intraday endpoint like:")
    print("http://localhost:3000/api/equity/intraday/{SYMBOL}?interval=1m&dateStart=YYYY-MM-DD&dateEnd=YYYY-MM-DD")
    print("Then resample the 1-minute DataFrame into 5/10/15/30/60-minute bars using pandas resample.")

if __name__ == '__main__':
    main()
