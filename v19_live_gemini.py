# Final Trading Signal Script - v24 TEST (TrueData, 5/15/30/60/1D, custom indicators, multi-session 10 rps, tz-aware, single progress bar)
import os
import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from logzero import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from collections import defaultdict

from tqdm import tqdm  # pip install tqdm
from truedata.history import TD_hist  # pip install truedata
# NOTE: You need a config.py file with your TrueData username and password
# Example config.py:
# username = "YOUR_USERNAME"
# password = "YOUR_PASSWORD"
from config import username as TDUSERNAME, password as TDPASSWORD


# Colors for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    RESET = "\033[0m"

try:
    GREEN = Colors.GREEN; RED = Colors.RED; RESET = Colors.RESET
except Exception:
    GREEN = RED = RESET = ""

# Silence noisy third‑party loggers
for noisy in ("truedata", "truedata.history", "truedata_ws", "websocket", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

IST = pytz.timezone("Asia/Kolkata")

# ---------- Token-bucket limiter ----------
class TokenBucketLimiter:
    def __init__(self, rate_per_sec: float, bucket_size: int):
        self.rate = rate_per_sec
        self.capacity = bucket_size
        self.tokens = bucket_size
        self.lock = threading.Lock()
        self.last_refill = time.time()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_refill
                if elapsed > 0:
                    add = int(elapsed * self.rate)
                    if add > 0:
                        self.tokens = min(self.capacity, self.tokens + add)
                        self.last_refill = now
                if self.tokens > 0:
                    self.tokens -= 1
                    return
                sleep_for = max(0.0, 1.0 / self.rate)
            time.sleep(sleep_for)

# Global API call counter for rps
api_calls_done = 0
api_calls_lock = threading.Lock()

# Timeframe and indicator weights
TIMEFRAME_WEIGHTS = {5: 1.0, 15: 1.2, 30: 1.4, 60: 1.6, 1440: 2.0}
INDICATOR_WEIGHTS = {
    "RSI": 1.3,
    "MACD": 1.6,
    "Stochastic": 1.0,
    "MA": 1.8,
    "ADX": 1.5,
    "Bollinger": 1.4,
    "ROC": 1.2,
    "OBV": 1.6,
    "CCI": 1.1,
    "WWL": 1.0,
    "EMA": 1.7,
    "VWAP": 1.5,
    "ATR": 1.4,
    "VolumeSurge": 2.0,
    "Momentum": 1.9
}

BAR_SIZE_MAP = {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"}
DURATION_MAP = {5: "30 D", 15: "30 D", 30: "60 D", 60: "120 D", 1440: "365 D"}

# Live settings
SCAN_SECONDS = int(os.getenv("SCAN_SECONDS", "60"))   # scan cadence seconds
MARKET_START = "09:15"  # IST
MARKET_END = "15:30"    # IST
INCLUDE_PREOPEN = False  # True to start 09:00
MAX_BARS_KEEP = 1500     # cap per TF per symbol

# --- NEW SETTING ---
# Set to True to run one scan immediately and then exit (for testing).
# Set to False to run in live mode, waiting for market hours and scanning continuously.
RUN_ONCE_AND_EXIT = False


# ---------------- TrueData sessions ----------------
def authenticate_session():
    return TD_hist(TDUSERNAME, TDPASSWORD, log_level=logging.CRITICAL)

def build_sessions():
    sess_count = int(os.getenv("TD_HIST_SESSIONS", "3"))  # tune 2–5
    pool = []
    for i in range(sess_count):
        try:
            pool.append(authenticate_session())
        except Exception as e:
            logger.error(f"Session {i} init failed: {e}")
    if not pool:
        raise SystemExit("Failed to initialize TrueData sessions.")
    per_sess_rate = 10.0 / len(pool)
    limiters = [TokenBucketLimiter(rate_per_sec=per_sess_rate, bucket_size=10) for _ in pool]
    return pool, limiters

tdhist_pool, sess_limiters = build_sessions()
logger.info("TrueData login successful.")

# ---------------- Indicator helpers ----------------
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def vwap(df, period=None):
    price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = price * df["Volume"]
    if period:
        pv_sum = pv.rolling(period).sum()
        vol_sum = df["Volume"].rolling(period).sum()
    else:
        pv_sum = pv.cumsum()
        vol_sum = df["Volume"].cumsum()
    return pv_sum / vol_sum

def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def williams_r(df, period=14):
    highest = df["High"].rolling(period).max()
    lowest = df["Low"].rolling(period).min()
    return -100 * (highest - df["Close"]) / (highest - lowest)

def momentum(df, period=10):
    return df["Close"] / df["Close"].shift(period) - 1.0

def volume_surge(df, lookback=20):
    vol_ma = df["Volume"].rolling(lookback).mean()
    vol_std = df["Volume"].rolling(lookback).std()
    return (df["Volume"] - vol_ma) / vol_std

# ---------------- Existing base indicators ----------------
def calculate_rsi(df, period=14):
    if len(df) < period + 1: return pd.Series(dtype='float64')
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    if len(df) < slow + signal: return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_stochastic(df, period=14, smooth_d=3):
    if len(df) < period + smooth_d: return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_moving_averages(df, short=50, long=200):
    if len(df) < long: return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    return df['Close'].rolling(window=short).mean(), df['Close'].rolling(window=long).mean()

def calculate_adx(df, period=14):
    if len(df) < period * 2: return pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64')
    df_adx = df.copy()
    df_adx['H-L'] = df_adx['High'] - df_adx['Low']
    df_adx['H-C'] = abs(df_adx['High'] - df_adx['Close'].shift(1))
    df_adx['L-C'] = abs(df_adx['Low'] - df_adx['Close'].shift(1))
    df_adx['TR'] = df_adx[['H-L', 'H-C', 'L-C']].max(axis=1)
    df_adx['+DM'] = np.where(
        (df_adx['High'] - df_adx['High'].shift(1)) > (df_adx['Low'].shift(1) - df_adx['Low']),
        df_adx['High'] - df_adx['High'].shift(1),
        0
    )
    df_adx['-DM'] = np.where(
        (df_adx['Low'].shift(1) - df_adx['Low']) > (df_adx['High'] - df_adx['High'].shift(1)),
        df_adx['Low'].shift(1) - df_adx['Low'],
        0
    )
    atr_val = df_adx['TR'].ewm(com=period - 1, adjust=False).mean()
    pdi = (df_adx['+DM'].ewm(com=period - 1, adjust=False).mean() / atr_val) * 100
    ndi = (df_adx['-DM'].ewm(com=period - 1, adjust=False).mean() / atr_val) * 100
    adx = (abs(pdi - ndi) / (pdi + ndi)).ewm(com=period - 1, adjust=False).mean() * 100
    return adx, pdi, ndi

def calculate_bollinger_bands(df, period=20):
    if len(df) < period: return pd.Series(dtype='float64')
    return df['Close'].rolling(window=period).mean()

def calculate_roc(df, period=12):
    if len(df) < period + 1: return pd.Series(dtype='float64')
    return ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100

def calculate_obv(df):
    if len(df) < 2: return pd.Series(dtype='float64')
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

def calculate_cci(df, period=20):
    if len(df) < period: return pd.Series(dtype='float64')
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad)

# ---------------- Scoring ----------------
def get_indicator_scores(df):
    scores = {}

    # Base indicators
    rsi = calculate_rsi(df).iloc[-1] if len(df) else np.nan
    if pd.notna(rsi):
        if rsi > 70: scores['RSI'] = -1.5
        elif rsi > 55: scores['RSI'] = 1.0
        elif rsi < 30: scores['RSI'] = 1.5
        elif rsi < 45: scores['RSI'] = -1.0
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

    ma_short, ma_long = calculate_moving_averages(df)
    if len(ma_short) and len(ma_long) and pd.notna(ma_short.iloc[-1]) and pd.notna(ma_long.iloc[-1]):
        scores['MA'] = 1.0 if ma_short.iloc[-1] > ma_long.iloc[-1] else -1.0
    else: scores['MA'] = 0.0

    adx, pdi, ndi = calculate_adx(df)
    if len(adx) and pd.notna(adx.iloc[-1]) and adx.iloc[-1] > 25:
        scores['ADX'] = 1.5 if pdi.iloc[-1] > ndi.iloc[-1] else -1.5
    else: scores['ADX'] = 0.0

    middle = calculate_bollinger_bands(df)
    if len(middle) and pd.notna(middle.iloc[-1]) and pd.notna(df['Close'].iloc[-1]):
        scores['Bollinger'] = 1.0 if df['Close'].iloc[-1] > middle.iloc[-1] else -1.0
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

    # New indicators
    ema_fast = ema(df["Close"], 20)
    ema_slow = ema(df["Close"], 50)
    if len(ema_fast) and len(ema_slow) and pd.notna(ema_fast.iloc[-1]) and pd.notna(ema_slow.iloc[-1]):
        scores["EMA"] = 1.0 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1.0
    else:
        scores["EMA"] = 0.0

    vwap_line = vwap(df, period=None)
    if len(vwap_line) and pd.notna(vwap_line.iloc[-1]) and pd.notna(df["Close"].iloc[-1]):
        scores["VWAP"] = 1.0 if df["Close"].iloc[-1] > vwap_line.iloc[-1] else -1.0
    else:
        scores["VWAP"] = 0.0

    atr_val = atr(df, period=14)
    if len(atr_val) >= 6 and pd.notna(atr_val.iloc[-1]) and pd.notna(atr_val.iloc[-5]) and pd.notna(df["Close"].iloc[-1]) and pd.notna(df["Close"].iloc[-5]):
        atr_rising = atr_val.iloc[-1] > atr_val.iloc[-5]
        price_up = df["Close"].iloc[-1] > df["Close"].iloc[-5]
        if atr_rising and price_up: scores["ATR"] = 1.0
        elif atr_rising and not price_up: scores["ATR"] = -1.0
        else: scores["ATR"] = 0.0
    else:
        scores["ATR"] = 0.0

    zscore = volume_surge(df, lookback=20)
    if len(zscore) and pd.notna(zscore.iloc[-1]) and len(df) >= 2:
        price_up_last = df["Close"].iloc[-1] > df["Close"].iloc[-2]
        if zscore.iloc[-1] >= 2.0:
            scores["VolumeSurge"] = 1.0 if price_up_last else 0.0
        elif zscore.iloc[-1] <= -2.0:
            scores["VolumeSurge"] = -1.0 if not price_up_last else 0.0
        else:
            scores["VolumeSurge"] = 0.0
    else:
        scores["VolumeSurge"] = 0.0

    mom = momentum(df, period=10)
    if len(mom) and pd.notna(mom.iloc[-1]):
        scores["Momentum"] = 1.0 if mom.iloc[-1] > 0 else -1.0
    else:
        scores["Momentum"] = 0.0

    wr = williams_r(df, period=14)
    if len(wr) and pd.notna(wr.iloc[-1]):
        if wr.iloc[-1] < -80: scores["WWL"] = 1.0
        elif wr.iloc[-1] > -20: scores["WWL"] = -1.0
        else: scores["WWL"] = 0.0
    else:
        scores["WWL"] = 0.0

    for k in INDICATOR_WEIGHTS.keys():
        scores.setdefault(k, 0.0)

    return scores

def analyze_signals(timeframe_dataframes):
    final_score, max_possible = 0.0, 0.0
    for tf_min, df in timeframe_dataframes.items():
        if df is None or len(df) < 50: continue
        indicator_scores = get_indicator_scores(df)
        tf_weight = TIMEFRAME_WEIGHTS.get(tf_min, 1.0)
        for indicator, score in indicator_scores.items():
            ind_weight = INDICATOR_WEIGHTS.get(indicator, 1.0)
            final_score += score * tf_weight * ind_weight
            max_possible += max(abs(score), 1.0) * tf_weight * ind_weight
    if max_possible == 0: return 'Neutral', 0.0
    normalized = (final_score / max_possible) * 100.0
    if normalized >= 70: signal_text = 'Very Strong Buy'
    elif normalized >= 20: signal_text = 'Strong Buy'
    elif normalized <= -70: signal_text = 'Very Strong Sell'
    elif normalized <= -20: signal_text = 'Strong Sell'
    else: signal_text = 'Neutral'
    return signal_text, normalized

# ---------------- Fetch + normalize (tz-aware IST) ----------------
def normalize_hist_df(df, symbol):
    if df is None or len(df) == 0:
        return None
    try:
        out = df.copy()
        out.rename(columns={c: str(c).lower() for c in out.columns}, inplace=True)
        rename_map = {}
        for src, tgt in (
            ("timestamp", "Date"), ("time", "Date"), ("datetime", "Date"), ("date", "Date"),
            ("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"),
            ("volume", "Volume"), ("vol", "Volume")
        ):
            if src in out.columns: rename_map[src] = tgt
        out.rename(columns=rename_map, inplace=True)

        if "Date" not in out.columns:
            if isinstance(out.index, pd.DatetimeIndex):
                out["Date"] = out.index
            else:
                return None

        if "Volume" not in out.columns:
            out["Volume"] = 0

        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"])

        # tz-aware IST
        if pd.api.types.is_datetime64tz_dtype(out["Date"]):
            out["Date"] = out["Date"].dt.tz_convert(IST)
        else:
            out["Date"] = out["Date"].dt.tz_localize(IST)

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")

        out = out.dropna(subset=["Open", "High", "Low", "Close"])
        out = out.sort_values("Date").set_index("Date")
        return out if len(out) >= 50 else None
    except Exception as e:
        logger.error(f"Normalize error {symbol}: {e}")
        return None

def pick_session(symbol_orig, timeframe_minutes):
    return (hash(symbol_orig) ^ timeframe_minutes) % len(tdhist_pool)

def fetch_one(symbol_orig, timeframe_minutes, limiter, hist):
    td_symbol = symbol_orig.replace('-EQ', '')
    bar_size = BAR_SIZE_MAP.get(timeframe_minutes)
    duration = DURATION_MAP.get(timeframe_minutes)
    if not bar_size or not duration:
        return symbol_orig, timeframe_minutes, None
    try:
        t0 = time.time()
        limiter.acquire()
        df_raw = hist.get_historic_data(td_symbol, duration=duration, bar_size=bar_size)
        t1 = time.time()
        df = normalize_hist_df(df_raw, td_symbol)
        global api_calls_done
        with api_calls_lock:
            api_calls_done += 1
        # Reduce log noise by only logging every 50 calls
        if api_calls_done > 0 and api_calls_done % 50 == 0:
            logger.info(f"API calls: {api_calls_done}. Sample latency: {(t1 - t0):.2f}s")
        return symbol_orig, timeframe_minutes, df
    except Exception:
        return symbol_orig, timeframe_minutes, None

def rps_monitor(stop_event, bar):
    prev = 0
    while not stop_event.is_set():
        time.sleep(1.0)
        with api_calls_lock:
            curr = api_calls_done
        rps = curr - prev
        prev = curr
        bar.set_postfix_str(f"rps={rps}")

def prefetch_all(stocks, max_workers=64):
    tfs = [5, 15, 30, 60, 1440]
    total_calls = len(stocks) * len(tfs)
    stock_multi_data = defaultdict(dict)
    
    # Reset global counter for prefetch
    global api_calls_done
    with api_calls_lock:
        api_calls_done = 0

    stop_evt = threading.Event()
    with tqdm(total=total_calls, desc="Prefetching Data", ncols=100) as api_bar:
        mon = threading.Thread(target=rps_monitor, args=(stop_evt, api_bar), daemon=True)
        mon.start()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for s in stocks:
                for tf in tfs:
                    si = pick_session(s, tf)
                    futures.append(executor.submit(fetch_one, s, tf, sess_limiters[si], tdhist_pool[si]))
            for fut in as_completed(futures):
                symbol_orig, tf, df = fut.result()
                if df is not None:
                    stock_multi_data[symbol_orig][tf] = df
                api_bar.update(1)
        stop_evt.set()

    # Require at least 2 TFs present
    return {s: d for s, d in stock_multi_data.items() if len(d) >= 2}

# ---------------- Output helper (file append) ----------------
def append_table_output(file_path, scan_time_str, bullish_list, bearish_list):
    """
    Append a text table block with timestamp header and two tables (bullish/bearish).
    bullish_list / bearish_list are lists of dicts with keys: symbol, signal, score, trend, change.
    """
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"===== Scan Time: {scan_time_str} =====\n\n")
        # Bullish table
        f.write("Top 15 Bullish\n")
        f.write(f"{'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Action':<19}\n")
        f.write("-" * 92 + "\n")
        if not bullish_list:
            f.write("No strong bullish signals found.\n")
        for r in bullish_list:
            change_val = r['change']
            if isinstance(change_val, (int, float, np.floating)):
                sign = '+' if change_val > 0 else ''
                change_str = f"{sign}{change_val:>.2f}"
            else:
                change_str = "NA"
            action = "Consider Long" if 'Buy' in r['signal'] else "Consider Short"
            f.write(f"{r['symbol']:<15} | {r['signal']:<18} | {r['score']:>7.2f} | {change_str:>10} | {r['trend']:<10} | {action:<19}\n")
        f.write("\n")

        # Bearish table
        f.write("Top 15 Bearish\n")
        f.write(f"{'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Action':<19}\n")
        f.write("-" * 92 + "\n")
        if not bearish_list:
            f.write("No strong bearish signals found.\n")
        for r in bearish_list:
            change_val = r['change']
            if isinstance(change_val, (int, float, np.floating)):
                sign = '+' if change_val > 0 else ''
                change_str = f"{sign}{change_val:>.2f}"
            else:
                change_str = "NA"
            action = "Consider Long" if 'Buy' in r['signal'] else "Consider Short"
            f.write(f"{r['symbol']:<15} | {r['signal']:<18} | {r['score']:>7.2f} | {change_str:>10} | {r['trend']:<10} | {action:<19}\n")
        f.write("\n\n")

# ---------------- Live market loop ----------------
def in_market_hours(now_ist: datetime) -> bool:
    start_str = "09:00" if INCLUDE_PREOPEN else MARKET_START
    start_t = datetime.strptime(start_str, "%H:%M").time()
    end_t = datetime.strptime(MARKET_END, "%H:%M").time()
    return start_t <= now_ist.time() <= end_t

def trim_df(df: pd.DataFrame, max_len: int) -> pd.DataFrame:
    if len(df) <= max_len:
        return df
    return df.iloc[-max_len:]

def live_market_scanner():
    logger.info("Starting market scanner (IST)…")
    if RUN_ONCE_AND_EXIT:
        logger.info("Mode: Run Once and Exit (Test Mode).")
    else:
        logger.info("Mode: Live Continuous Scanning.")

    # Read stocks list
    try:
        with open('shares.txt', 'r') as f:
            stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(stocks)} stocks from shares.txt")
    except Exception as e:
        logger.error(f"Could not read shares.txt: {e}")
        return

    # Initial bulk prefetch to seed caches
    stock_multi_data = prefetch_all(stocks, max_workers=64)

    if not stock_multi_data:
        print("No stocks with sufficient data found on initial load. Exiting.")
        return

    # Output file per IST date
    output_filename = datetime.now(IST).strftime("%Y-%m-%d") + "_scan_results.txt"

    previous_scores = {}
    tfs = [5, 15, 30, 60, 1440]

    while True:
        now_ist = datetime.now(IST)
        # Rotate output file at midnight IST
        fname_today = now_ist.strftime("%Y-%m-%d") + "_scan_results.txt"
        if fname_today != output_filename:
            output_filename = fname_today
            previous_scores = {} # Reset scores for the new day

        # *** MODIFIED SECTION FOR TEST/LIVE MODE ***
        if not RUN_ONCE_AND_EXIT and not in_market_hours(now_ist):
            print(f"Waiting for market hours ({MARKET_START}-{MARKET_END} IST). Current time: {now_ist.strftime('%H:%M:%S')}", end="\r")
            time.sleep(30)
            continue

        # Refresh data (no tqdm here to avoid multiple bars in live mode)
        logger.info(f"[{now_ist.strftime('%H:%M:%S')}] Refreshing data for {len(stock_multi_data)} stocks...")
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = []
            # Use only the stocks that were successfully prefetched
            for s in stock_multi_data.keys():
                for tf in tfs:
                    si = pick_session(s, tf)
                    futures.append(executor.submit(fetch_one, s, tf, sess_limiters[si], tdhist_pool[si]))
            
            # Process results as they complete
            for fut in as_completed(futures):
                symbol_orig, tf, df = fut.result()
                if df is None or len(df) < 50:
                    continue
                # Merge into cache
                d = stock_multi_data.setdefault(symbol_orig, {})
                if tf in d and d[tf] is not None and not d[tf].empty:
                    merged = pd.concat([d[tf], df]).sort_index()
                    merged = merged[~merged.index.duplicated(keep='last')]
                    d[tf] = trim_df(merged, MAX_BARS_KEEP)
                else:
                    d[tf] = trim_df(df, MAX_BARS_KEEP)
        logger.info("Data refresh complete. Analyzing signals...")

        # Build signals for current time rounded to minute
        time_point_aware = now_ist.replace(second=0, microsecond=0)
        signals_this_scan = []
        current_scores = {}

        for symbol, timeframe_data in stock_multi_data.items():
            clean_symbol = symbol.replace('-EQ', '')

            filtered_timeframes = {}
            for tf, df in timeframe_data.items():
                if df is None or df.empty:
                    continue
                # Use data up to the current moment for analysis
                df_slice = df[df.index <= time_point_aware]
                if not df_slice.empty and len(df_slice) >= 50:
                    filtered_timeframes[tf] = df_slice

            if len(filtered_timeframes) < 2:
                continue

            signal, score = analyze_signals(filtered_timeframes)
            current_scores[clean_symbol] = score

            if 'Strong' in signal:
                change = 'NA' if clean_symbol not in previous_scores else score - previous_scores.get(clean_symbol, 0.0)
                direction = 'bullish' if 'Buy' in signal else 'bearish'
                signals_this_scan.append({
                    'symbol': clean_symbol, 'signal': signal, 'score': score,
                    'trend': direction, 'change': change
                })

        # Rank and select top 15 bullish/bearish
        signals_this_scan.sort(key=lambda x: x['score'], reverse=True)
        top_bullish = [r for r in signals_this_scan if 'Buy' in r['signal']][:15]
        bearish_sorted = sorted([r for r in signals_this_scan if 'Sell' in r['signal']], key=lambda x: x['score'])
        top_bearish = bearish_sorted[:15]

        # Console output
        print("\n" + "="*92)
        print(f"| LIVE SIGNALS AT {now_ist.strftime('%Y-%m-%d %H:%M')} IST".center(90) + " |")
        print("="*92)

        # Bullish table
        print(f"| {'Top 15 Bullish':<88} |")
        print("-"*92)
        if not top_bullish:
            print("| None".ljust(91) + " |")
        else:
            print(f"| {'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Action':<19} |")
            print("-"*92)
            for result in top_bullish:
                signal_text = result['signal']
                change_val = result['change']
                if isinstance(change_val, (int, float, np.floating)):
                    sign = '+' if change_val > 0 else ''
                    color = GREEN if change_val > 0 else RED
                    # Adjust padding for color codes
                    change_str = f"{color}{sign}{change_val:>.2f}{RESET}"
                    padding = 19
                else:
                    change_str = "NA"
                    padding = 10
                
                colored_signal = f"{GREEN}{signal_text:<18}{RESET}" if 'Buy' in signal_text else f"{RED}{signal_text:<18}{RESET}"
                action = f"{GREEN}Consider Long{RESET}" if 'Buy' in signal_text else f"{RED}Consider Short{RESET}"
                
                print(f"| {result['symbol']:<15} | {colored_signal} | {result['score']:>7.2f} | {change_str:>{padding}} | {result['trend']:<10} | {action:<29} |")

        # Bearish table
        print("-"*92)
        print(f"| {'Top 15 Bearish':<88} |")
        print("-"*92)
        if not top_bearish:
            print("| None".ljust(91) + " |")
        else:
            print(f"| {'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Action':<19} |")
            print("-"*92)
            for result in top_bearish:
                signal_text = result['signal']
                change_val = result['change']
                if isinstance(change_val, (int, float, np.floating)):
                    sign = '+' if change_val > 0 else ''
                    color = GREEN if change_val > 0 else RED
                    # Adjust padding for color codes
                    change_str = f"{color}{sign}{change_val:>.2f}{RESET}"
                    padding = 19
                else:
                    change_str = "NA"
                    padding = 10

                colored_signal = f"{GREEN}{signal_text:<18}{RESET}" if 'Buy' in signal_text else f"{RED}{signal_text:<18}{RESET}"
                action = f"{GREEN}Consider Long{RESET}" if 'Buy' in signal_text else f"{RED}Consider Short{RESET}"

                print(f"| {result['symbol']:<15} | {colored_signal} | {result['score']:>7.2f} | {change_str:>{padding}} | {result['trend']:<10} | {action:<29} |")
        print("="*92)

        # File append
        append_table_output(
            output_filename,
            now_ist.strftime('%Y-%m-%d %H:%M'),
            top_bullish,
            top_bearish
        )
        logger.info(f"Results appended to {output_filename}")

        # *** MODIFIED SECTION FOR TEST/LIVE MODE ***
        if RUN_ONCE_AND_EXIT:
            logger.info("Scan complete. Exiting as RUN_ONCE_AND_EXIT is True.")
            break # Exit the while loop

        previous_scores = current_scores.copy()
        logger.info(f"Waiting for {SCAN_SECONDS} seconds until the next scan...")
        time.sleep(SCAN_SECONDS)

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    try:
        live_market_scanner()
    except KeyboardInterrupt:
        print("\nScan interrupted by user. Shutting down.")
    finally:
        logger.info("Disconnecting TrueData sessions...")
        for sess in tdhist_pool:
            try:
                sess.disconnect()
            except Exception:
                pass
        logger.info("Shutdown complete.")