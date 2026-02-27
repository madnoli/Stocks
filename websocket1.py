# websocket1.py — TrueData parity scanner (1m stream -> 5m MTF)
# Requirements: pip install truedata pandas numpy pytz logzero requests
# Files needed: config.py with username/password; shares.txt with EQ symbols

import os
import time
import logging
import threading
from datetime import datetime, timedelta
from io import StringIO

import pandas as pd
import numpy as np
import pytz
import requests
from logzero import logger

from truedata import TD_live, TD_hist  # official Python client per v2.6 docs [file:108]
from config import username as TD_USERNAME, password as TD_PASSWORD

# ------------ Config ------------
IST = pytz.timezone("Asia/Kolkata")
MARKET_OPEN = (9, 15)
MARKET_CLOSE = (15, 30)

SHARES_FILE = "shares.txt"
HIST_FROM_DAYS = 5
BAR_CLOSE_BUFFER_SEC = 5  # post-close gating

INDICATOR_WEIGHTS = {
    "VolumeSurge": 3.5, "Momentum": 2.8, "ADX": 2.5, "ATR": 2.2, "ROC": 2.0,
    "RSI": 1.5, "MACD": 1.4, "EMA": 1.2, "VWAP": 1.2, "Bollinger": 2.5,
    "OBV": 1.0, "Stochastic": 0.8, "CCI": 0.8, "WWL": 0.7, "MA": 0.5,
}
TIMEFRAME_WEIGHTS = {"5m": 1.0, "10m": 1.1, "15m": 1.2, "30m": 1.3, "60m": 1.5}

# ------------ State ------------
bars_1m = {}        # symbol -> 1m DataFrame (tz-aware IST, right-edge)
bars_5m = {}        # symbol -> 5m DataFrame derived from 1m
previous_scores = {} 
lock = threading.Lock()

# ------------ Indicators ------------
def ema(series, length): return series.ewm(span=length, adjust=False).mean()

def vwap(df, period=None):
    price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = price * df["Volume"]
    if period:
        return (pv.rolling(period).sum() / df["Volume"].rolling(period).sum()).replace([np.inf, -np.inf], np.nan)
    return (pv.cumsum() / df["Volume"].cumsum()).replace([np.inf, -np.inf], np.nan)

def atr(df, period=14):
    hl = df["High"] - df["Low"]
    hc = (df["High"] - df["Close"].shift(1)).abs()
    lc = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def williams_r(df, period=14):
    highest = df["High"].rolling(period).max()
    lowest = df["Low"].rolling(period).min()
    return -100 * (highest - df["Close"]) / (highest - lowest)

def momentum(df, period=10): return df["Close"] / df["Close"].shift(period) - 1.0

def volume_surge(df, lookback=20):
    vol_ma = df["Volume"].rolling(lookback).mean()
    vol_std = df["Volume"].rolling(lookback).std()
    return ((df["Volume"] - vol_ma) / vol_std.replace(0, np.nan)).fillna(0)

def calculate_bollinger_bands(df, period=20, std_dev=2):
    if len(df) < period:
        return pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64')
    mid = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    up = mid + std_dev * std
    lo = mid - std_dev * std
    return mid, up, lo

def bollinger_band_width(df, period=20, std_dev=2):
    mid, up, lo = calculate_bollinger_bands(df, period, std_dev)
    return ((up - lo) / mid.replace(0, np.nan) * 100).fillna(0)

def calculate_rsi(df, period=14):
    if len(df) < period + 1: return pd.Series(dtype='float64')
    d = df['Close'].diff()
    gain = (d.where(d > 0, 0)).ewm(com=period-1, adjust=False).mean()
    loss = (-d.where(d < 0, 0)).ewm(com=period-1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rs.fillna(100, inplace=True)
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    if len(df) < slow + signal: return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    ef = df['Close'].ewm(span=fast, adjust=False).mean()
    es = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ef - es
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def calculate_stochastic(df, period=14, smooth_d=3):
    if len(df) < period + smooth_d: return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    rng = (high_max - low_min).replace(0, np.nan)
    k = 100 * ((df['Close'] - low_min) / rng)
    k.fillna(50, inplace=True)
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_moving_averages(df, short=50, long=200):
    if len(df) < long: return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    return df['Close'].rolling(window=short).mean(), df['Close'].rolling(window=long).mean()

def calculate_adx(df, period=14):
    if len(df) < period*2: return pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64')
    x = df.copy()
    x['H-L'] = x['High'] - x['Low']
    x['H-C'] = (x['High'] - x['Close'].shift(1)).abs()
    x['L-C'] = (x['Low'] - x['Close'].shift(1)).abs()
    x['TR'] = x[['H-L','H-C','L-C']].max(axis=1)
    x['+DM'] = np.where((x['High'] - x['High'].shift(1)) > (x['Low'].shift(1) - x['Low']), x['High'] - x['High'].shift(1), 0)
    x['-DM'] = np.where((x['Low'].shift(1) - x['Low']) > (x['High'] - x['High'].shift(1)), x['Low'].shift(1) - x['Low'], 0)
    atr_val = x['TR'].ewm(com=period-1, adjust=False).mean().replace(0, np.nan)
    pdi = (x['+DM'].ewm(com=period - 1, adjust=False).mean() / atr_val) * 100
    ndi = (x['-DM'].ewm(com=period - 1, adjust=False).mean() / atr_val) * 100
    p = (pdi + ndi).replace(0, np.nan)
    adx = (abs(pdi - ndi) / p).ewm(com=period - 1, adjust=False).mean() * 100
    return adx.fillna(20), pdi.fillna(20), ndi.fillna(20)

def calculate_roc(df, period=12):
    if len(df) < period + 1: return pd.Series(dtype='float64')
    base = df['Close'].shift(period).replace(0, np.nan)
    return ((df['Close'] - df['Close'].shift(period)) / base) * 100

def calculate_obv(df):
    if len(df) < 2: return pd.Series(dtype='float64')
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

def calculate_cci(df, period=20):
    if len(df) < period: return pd.Series(dtype='float64')
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).replace(0, np.nan)
    return (tp - sma_tp) / (0.015 * mad)

def get_indicator_scores(df):
    scores = {}
    rsi_series = calculate_rsi(df)
    if len(rsi_series) > 1 and pd.notna(rsi_series.iloc[-1]):
        rsi = rsi_series.iloc[-1]; prev_rsi = rsi_series.iloc[-2]
        if rsi > 60 and prev_rsi <= 60: scores['RSI'] = 2.0
        elif rsi > 50 and prev_rsi <= 50: scores['RSI'] = 1.0
        elif rsi < 40 and prev_rsi >= 40: scores['RSI'] = -2.0
        elif rsi < 50 and prev_rsi >= 50: scores['RSI'] = -1.0
        else: scores['RSI'] = 0.0
    else: scores['RSI'] = 0.0
    macd, signal = calculate_macd(df)
    if len(macd) and len(signal) and pd.notna(macd.iloc[-1]) and pd.notna(signal.iloc[-1]):
        scores['MACD'] = 1.0 if macd.iloc[-1] > signal.iloc[-1] else -1.0
    else: scores['MACD'] = 0.0
    k, d = calculate_stochastic(df)
    if len(k) and len(d) and pd.notna(k.iloc[-1]) and pd.notna(d.iloc[-1]):
        if k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80: scores['Stochastic'] = 1.0
        elif k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 20: scores['Stochastic'] = -1.0
        else: scores['Stochastic'] = 0.0
    else: scores['Stochastic'] = 0.0
    ma_s, ma_l = calculate_moving_averages(df)
    if len(ma_s) and len(ma_l) and pd.notna(ma_s.iloc[-1]) and pd.notna(ma_l.iloc[-1]):
        scores['MA'] = 1.0 if ma_s.iloc[-1] > ma_l.iloc[-1] else -1.0
    else: scores['MA'] = 0.0
    adx, pdi, ndi = calculate_adx(df)
    if len(adx) > 4 and pd.notna(adx.iloc[-1]):
        rising = adx.iloc[-1] > adx.iloc[-3]
        crossed = adx.iloc[-1] > 22 and adx.iloc[-2] <= 22
        if (adx.iloc[-1] > 22 and rising) or crossed:
            mult = 2.0 if crossed else 1.0
            scores['ADX'] = 1.5*mult if pdi.iloc[-1] > ndi.iloc[-1] else -1.5*mult
        else: scores['ADX'] = 0.0
    else: scores['ADX'] = 0.0
    bbw = bollinger_band_width(df)
    if len(bbw) > 50 and pd.notna(bbw.iloc[-1]):
        squeeze = bbw.iloc[-2] < bbw.rolling(50).min().iloc[-2]
        mid, up, lo = calculate_bollinger_bands(df)
        if not all(s.empty for s in [mid, up, lo]):
            close = df['Close'].iloc[-1]; z = volume_surge(df).iloc[-1]
            if squeeze and pd.notna(z) and z > 1.5:
                if close > up.iloc[-1]: scores['Bollinger'] = 2.0
                elif close < lo.iloc[-1]: scores['Bollinger'] = -2.0
                else: scores['Bollinger'] = 0.0
            elif pd.notna(close) and pd.notna(mid.iloc[-1]):
                scores['Bollinger'] = 0.5 if close > mid.iloc[-1] else -0.5
            else: scores['Bollinger'] = 0.0
        else: scores['Bollinger'] = 0.0
    else: scores['Bollinger'] = 0.0
    roc = calculate_roc(df).iloc[-1] if len(df) else np.nan
    scores['ROC'] = 1.0 if pd.notna(roc) and roc > 0 else (-1.0 if pd.notna(roc) else 0.0)
    obv = calculate_obv(df)
    if len(obv) >= 2 and pd.notna(obv.iloc[-1]) and pd.notna(obv.iloc[-2]):
        scores['OBV'] = 1.0 if obv.iloc[-1] > obv.iloc[-2] else -1.0
    else: scores['OBV'] = 0.0
    cci = calculate_cci(df).iloc[-1] if len(df) else np.nan
    if pd.notna(cci):
        if cci > 100: scores['CCI'] = 1.5
        elif cci > 0: scores['CCI'] = 1.0
        elif cci < -100: scores['CCI'] = -1.5
        elif cci < 0: scores['CCI'] = -1.0
        else: scores['CCI'] = 0.0
    else: scores['CCI'] = 0.0
    ef = ema(df["Close"], 20); es = ema(df["Close"], 50)
    if len(ef) and len(es) and pd.notna(ef.iloc[-1]) and pd.notna(es.iloc[-1]):
        scores["EMA"] = 1.0 if ef.iloc[-1] > es.iloc[-1] else -1.0
    else: scores["EMA"] = 0.0
    vwp = vwap(df, period=None)
    if len(vwp) and pd.notna(vwp.iloc[-1]) and pd.notna(df["Close"].iloc[-1]):
        scores["VWAP"] = 1.0 if df["Close"].iloc[-1] > vwp.iloc[-1] else -1.0
    else: scores["VWAP"] = 0.0
    a = atr(df, period=14)
    if len(a) >= 6 and all(pd.notna(val) for val in [a.iloc[-1], a.iloc[-5], df["Close"].iloc[-1], df["Close"].iloc[-5]]):
        rising = (a.iloc[-1] / a.iloc[-5]) > 1.1
        up = df["Close"].iloc[-1] > df["Close"].iloc[-5]
        scores["ATR"] = 1.5 if rising and up else (-1.5 if rising and not up else 0.0)
    else:
        scores["ATR"] = 0.0
    z = volume_surge(df, lookback=20)
    if len(z) and pd.notna(z.iloc[-1]) and len(df) >= 2:
        up_last = df["Close"].iloc[-1] > df["Close"].iloc[-2]
        if z.iloc[-1] >= 2.0: scores["VolumeSurge"] = 1.5 if up_last else 0.0
        elif z.iloc[-1] <= -2.0: scores["VolumeSurge"] = -1.5 if not up_last else 0.0
        else: scores["VolumeSurge"] = 0.0
    else:
        scores["VolumeSurge"] = 0.0
    wr = williams_r(df, period=14)
    if len(wr) and pd.notna(wr.iloc[-1]):
        scores["WWL"] = 1.0 if wr.iloc[-1] < -80 else (-1.0 if wr.iloc[-1] > -20 else 0.0)
    else:
        scores["WWL"] = 0.0
    for k in INDICATOR_WEIGHTS.keys():
        scores.setdefault(k, 0.0)
    return scores

def analyze_signals(tf_map):
    final_score, max_possible = 0.0, 0.0
    for tf_key, df in tf_map.items():
        if df is None or len(df) < 50: continue
        s = get_indicator_scores(df)
        tfw = TIMEFRAME_WEIGHTS.get(tf_key, 1.0)
        for ind, val in s.items():
            iw = INDICATOR_WEIGHTS.get(ind, 1.0)
            final_score += val * tfw * iw
            max_abs = max(abs(x) for x in s.values() if x != 0) if any(s.values()) else 1.0
            max_possible += max(abs(val), max_abs) * tfw * iw
    if max_possible == 0: return 'Neutral', 0.0
    norm = (final_score / max_possible) * 100.0
    if norm >= 65: t = 'Very Strong Buy'
    elif norm >= 25: t = 'Strong Buy'
    elif norm <= -65: t = 'Very Strong Sell'
    elif norm <= -25: t = 'Strong Sell'
    else: t = 'Neutral'
    return t, norm

# ------------ Helpers ------------
def between_market_hours(ts_ist):
    t = ts_ist.time()
    open_t = datetime.strptime(f"{MARKET_OPEN[0]:02d}:{MARKET_OPEN[1]:02d}", "%H:%M").time()
    close_t = datetime.strptime(f"{MARKET_CLOSE[0]:02d}:{MARKET_CLOSE[1]:02d}", "%H:%M").time()
    return open_t <= t <= close_t

def normalize_hist_df_csv(csv_text):
    df = pd.read_csv(StringIO(csv_text))
    if df.empty: return None
    df.rename(columns={"timestamp":"Date","open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume","oi":"OI"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)
    df["Date"] = df["Date"].dt.tz_localize(IST) if df["Date"].dt.tz is None else df["Date"].dt.tz_convert(IST)
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.dropna(subset=["Open","High","Low","Close"], inplace=True)
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    df = df.between_time(f"{MARKET_OPEN[0]:02d}:{MARKET_OPEN[1]:02d}", f"{MARKET_CLOSE[0]:02d}:{MARKET_CLOSE[1]:02d}")
    return df if len(df) >= 50 else None

def resample_1m_to_5m(df1):
    return df1.resample("5min", label="right", closed="right").agg(
        {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    ).dropna()

def build_mtf_from_5m(df5):
    agg = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}
    return {
        "5m": df5,
        "10m": df5.resample("10min", label="right", closed="right").agg(agg).dropna(),
        "15m": df5.resample("15min", label="right", closed="right").agg(agg).dropna(),
        "30m": df5.resample("30min", label="right", closed="right").agg(agg).dropna(),
        "60m": df5.resample("60min", label="right", closed="right").agg(agg).dropna(),
    }

def read_symbols(path):
    with open(path, "r") as f:
        return [line.strip().upper() for line in f if line.strip()]

# ------------ REST bootstrap (use TD_hist.get_history or authenticated session) ------------
def hist_get_bars_csv(td_hist_client, symbol, from_dt, to_dt, interval="1min"):
    # Preferred helper
    if hasattr(td_hist_client, "get_history"):
        return td_hist_client.get_history(symbol=symbol, from_dt=from_dt, to_dt=to_dt, interval=interval, response="csv")  # [file:108]
    # Fallback: use the client's authenticated session to call /getbars
    session = getattr(td_hist_client, "session", None)
    if session is None:
        raise RuntimeError("TD_hist missing get_history and session; update 'truedata' package. [docs v2.6]")
    params = {"symbol": symbol, "from": from_dt, "to": to_dt, "response": "csv", "interval": interval}
    resp = session.get("https://history.truedata.in/getbars", params=params, timeout=20)  # [file:108]
    resp.raise_for_status()
    return resp.text

def bootstrap_history(td_hist_client, symbols):
    now = datetime.now(IST)
    from_dt = (now - timedelta(days=HIST_FROM_DAYS)).strftime("%y%m%dT09:00:00")
    to_dt = now.strftime("%y%m%dT%H:%M:%S")
    ok = 0
    for s in symbols:
        try:
            csv_text = hist_get_bars_csv(td_hist_client, s, from_dt, to_dt, interval="1min")
            df1 = normalize_hist_df_csv(csv_text)
            if df1 is None:
                continue
            df5 = resample_1m_to_5m(df1)
            with lock:
                bars_1m[s] = df1.tail(5000)
                bars_5m[s] = df5.tail(1500)
            ok += 1
        except Exception as e:
            logger.error(f"History fetch failed {s}: {e}")
    logger.info(f"History bootstrap loaded {ok}/{len(symbols)} symbols.")

# ------------ Live WebSocket (1-minute bar stream) ------------
def bind_live_callbacks(td_live_client, symbols):
    @td_live_client.bar1min_callback
    def on_bar1(msg):
        try:
            sym = getattr(msg, "symbol", None) or getattr(msg, "symbolname", None) or str(getattr(msg, "symbolid", ""))
            ts = pd.to_datetime(getattr(msg, "timestamp"))
            ts = ts if ts.tzinfo else IST.localize(ts)
            ts = ts.astimezone(IST)
            o = float(getattr(msg, "open")); h = float(getattr(msg, "high")); l = float(getattr(msg, "low"))
            c = float(getattr(msg, "close")); v = float(getattr(msg, "volume", 0) or 0)
            now_ist = datetime.now(IST)
            delay = max(0.0, (ts - now_ist).total_seconds()) + BAR_CLOSE_BUFFER_SEC
            threading.Timer(delay, commit_1m_bar, args=(sym, ts, o, h, l, c, v)).start()
        except Exception as e:
            logger.error(f"bar1min callback error: {e}")

    td_live_client.add_symbols(symbols)

def commit_1m_bar(symbol, ts, o, h, l, c, v):
    with lock:
        df1 = bars_1m.get(symbol)
        bar = pd.DataFrame([[o, h, l, c, v]], columns=["Open","High","Low","Close","Volume"], index=[ts])
        if df1 is None or df1.empty:
            df1 = bar
        else:
            df1 = df1[~(df1.index == ts)]
            df1 = pd.concat([df1, bar]).sort_index()
        bars_1m[symbol] = df1.tail(5000)
        bars_5m[symbol] = resample_1m_to_5m(df1).tail(1500)
    # Evaluate on 5-min boundary only
    if ts.minute % 5 == 0:
        evaluate_if_due(ts)

def evaluate_if_due(bar_ts):
    if not between_market_hours(bar_ts):
        return
    threading.Thread(target=run_scan_at_boundary, args=(bar_ts,), daemon=True).start()

def run_scan_at_boundary(boundary_ts):
    time.sleep(1.0)
    signals = []
    current_scores = {}
    with lock:
        for s, df5 in bars_5m.items():
            if df5 is None or df5.empty: continue
            df5c = df5[df5.index <= boundary_ts]
            if df5c.empty or len(df5c) < 50: continue
            tf_map = build_mtf_from_5m(df5c)
            if not all(k in tf_map and len(tf_map[k]) >= 50 for k in ["5m","15m","60m"]):
                continue
            sig, score = analyze_signals(tf_map)
            current_scores[s] = score
            ma_s, ma_l = calculate_moving_averages(tf_map["60m"])
            trend = 'neutral'
            if len(ma_l) and pd.notna(ma_l.iloc[-1]):
                trend = 'bullish' if tf_map["60m"]["Close"].iloc[-1] > ma_l.iloc[-1] else 'bearish'
            if 'Strong' in sig:
                change = 'NA' if s not in previous_scores else score - previous_scores[s]
                if (trend == 'bullish' and 'Buy' in sig) or (trend == 'bearish' and 'Sell' in sig):
                    signals.append({"symbol": s, "signal": sig, "score": score, "trend": trend, "change": change})
        previous_scores.update(current_scores)

    signals.sort(key=lambda x: x["score"], reverse=True)
    top_bull = [r for r in signals if 'Buy' in r['signal']][:20]
    top_bear = sorted([r for r in signals if 'Sell' in r['signal']], key=lambda x: x['score'])[:20]

    print("\n" + "="*92)
    print(f"| LIVE 5-MIN SCANNER | SIGNALS AT {boundary_ts.strftime('%Y-%m-%d %H:%M')} IST".center(100) + " |")
    print("="*92)
    print(f"| {'Top 20 Bullish Breakouts':<88} |")
    print("-"*92)
    if not top_bull: print("| None".ljust(91) + " |")
    else:
        print(f"| {'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Action':<19} |")
        print("-"*92)
        for r in top_bull:
            ch = r['change']
            chs = "NA" if not isinstance(ch, (int,float,np.floating)) else f"{'+' if ch>0 else ''}{ch:>.2f}"
            print(f"| {r['symbol']:<15} | {r['signal']:<18} | {r['score']:>7.2f} | {chs:>19} | {r['trend']:<10} | {'Consider Long':<19} |")

    print("-"*92)
    print(f"| {'Top 20 Bearish Breakdowns':<88} |")
    print("-"*92)
    if not top_bear: print("| None".ljust(91) + " |")
    else:
        print(f"| {'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Action':<19} |")
        print("-"*92)
        for r in top_bear:
            ch = r['change']
            chs = "NA" if not isinstance(ch, (int,float,np.floating)) else f"{'+' if ch>0 else ''}{ch:>.2f}"
            print(f"| {r['symbol']:<15} | {r['signal']:<18} | {r['score']:>7.2f} | {chs:>19} | {r['trend']:<10} | {'Consider Short':<19} |")
    print("="*92)

# ------------ Entrypoint ------------
def main():
    if not TD_USERNAME or not TD_PASSWORD:
        raise SystemExit("config.py must define non-empty username/password. [TrueData v2.6]")
    symbols = read_symbols(SHARES_FILE)
    if not symbols:
        raise SystemExit("shares.txt is empty or missing.")

    td_hist_client = TD_hist(login_id=TD_USERNAME, password=TD_PASSWORD, log_level=logging.WARNING)
    td_live_client = TD_live(login_id=TD_USERNAME, password=TD_PASSWORD, log_level=logging.WARNING)

    bootstrap_history(td_hist_client, symbols)
    bind_live_callbacks(td_live_client, symbols)

    logger.info("Streaming started. Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("Exiting…")
    finally:
        try: td_live_client.remove_symbols(symbols)
        except Exception: pass
        try: td_live_client.disconnect()
        except Exception: pass

if __name__ == "__main__":
    for noisy in ("truedata", "urllib3", "websocket"):
        logging.getLogger(noisy).setLevel(logging.CRITICAL)
    main()
