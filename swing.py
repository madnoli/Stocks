import os
import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
import pytz
from logzero import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from collections import defaultdict

from tqdm import tqdm
from truedata.history import TD_hist
from config import username as TDUSERNAME, password as TDPASSWORD

# ---------------- Terminal colors (safe) ----------------
class Colors:
    GREEN = "\033[92m"
    RED   = "\033[91m"
    YELLOW= "\033[93m"
    CYAN  = "\033[96m"
    BOLD  = "\033[1m"
    RESET = "\033[0m"

try:
    GREEN = Colors.GREEN; RED = Colors.RED; YELLOW = Colors.YELLOW; CYAN = Colors.CYAN; BOLD = Colors.BOLD; RESET = Colors.RESET
except Exception:
    GREEN = RED = YELLOW = CYAN = BOLD = RESET = ""

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

# --- Swing timeframes (derived from Daily EOD) ---
SWING_TFS = ["1D", "1W", "1M", "3M", "6M"]
DURATION_DAILY = "365 D"  # enough to build 6M indicators

# Emphasize weekly/monthly for swing conviction, daily for timing
TIMEFRAME_WEIGHTS = {
    "1D": 1.0,
    "1W": 2.5,
    "1M": 2.2,
    "3M": 1.2,
    "6M": 0.8,
}

INDICATOR_WEIGHTS = {
    "VolumeSurge": 3.5, "Momentum": 2.8, "ADX": 2.5, "ATR": 2.2, "ROC": 2.0,
    "RSI": 1.5, "MACD": 1.4, "EMA": 1.2, "VWAP": 1.2, "Bollinger": 2.5,
    "OBV": 1.0, "Stochastic": 0.8, "CCI": 0.8, "WWL": 0.7,
    "MA": 0.5,
}

# Settings: single-run only
MAX_BARS_KEEP = 1500

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
    z_score = (df["Volume"] - vol_ma) / vol_std.replace(0, np.nan)
    return z_score.fillna(0)

def calculate_bollinger_bands(df, period=20, std_dev=2):
    if len(df) < period:
        return pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64')
    middle = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return middle, upper, lower

def bollinger_band_width(df, period=20, std_dev=2):
    middle, upper, lower = calculate_bollinger_bands(df, period, std_dev)
    width = ((upper - lower) / middle.replace(0, np.nan)) * 100
    return width.fillna(0)

# ---------------- Base indicators ----------------
def calculate_rsi(df, period=14):
    if len(df) < period + 1: return pd.Series(dtype='float64')
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rs.fillna(100, inplace=True)
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
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan))
    k.fillna(50, inplace=True)
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
    df_adx['+DM'] = np.where((df_adx['High'] - df_adx['High'].shift(1)) > (df_adx['Low'].shift(1) - df_adx['Low']), df_adx['High'] - df_adx['High'].shift(1), 0)
    df_adx['-DM'] = np.where((df_adx['Low'].shift(1) - df_adx['Low']) > (df_adx['High'] - df_adx['High'].shift(1)), df_adx['Low'].shift(1) - df_adx['Low'], 0)
    atr_val = df_adx['TR'].ewm(com=period - 1, adjust=False).mean()
    atr_val_safe = atr_val.replace(0, np.nan)
    pdi = (df_adx['+DM'].ewm(com=period - 1, adjust=False).mean() / atr_val_safe) * 100
    ndi = (df_adx['-DM'].ewm(com=period - 1, adjust=False).mean() / atr_val_safe) * 100
    pdi_plus_ndi = (pdi + ndi).replace(0, np.nan)
    adx = (abs(pdi - ndi) / pdi_plus_ndi).ewm(com=period - 1, adjust=False).mean() * 100
    return adx.fillna(20), pdi.fillna(20), ndi.fillna(20)

def calculate_roc(df, period=12):
    if len(df) < period + 1: return pd.Series(dtype='float64')
    shifted_close = df['Close'].shift(period).replace(0, np.nan)
    return ((df['Close'] - df['Close'].shift(period)) / shifted_close) * 100

def calculate_obv(df):
    if len(df) < 2: return pd.Series(dtype='float64')
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

def calculate_cci(df, period=20):
    if len(df) < period: return pd.Series(dtype='float64')
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    mad_safe = mad.replace(0, np.nan)
    return (tp - sma_tp) / (0.015 * mad_safe)

# ---------------- Resampling helpers (swing) ----------------
def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    out = df.resample(rule, label="right", closed="right").apply(agg).dropna()
    return out

def build_swing_timeframes(df_daily: pd.DataFrame) -> dict:
    tf_map = {
        "1D": df_daily,
        "1W": resample_ohlcv(df_daily, "W"),
        "1M": resample_ohlcv(df_daily, "M"),
        "3M": resample_ohlcv(df_daily, "3M"),  # use "Q" for strict quarter alignment
        "6M": resample_ohlcv(df_daily, "6M"),
    }
    return {k: v for k, v in tf_map.items() if v is not None and len(v) >= 50}

# ---------------- Scoring ----------------
def get_indicator_scores(df):
    scores = {}

    rsi_series = calculate_rsi(df)
    if len(rsi_series) > 1 and pd.notna(rsi_series.iloc[-1]):
        rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2]
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

    ma_short, ma_long = calculate_moving_averages(df)
    if len(ma_short) and len(ma_long) and pd.notna(ma_short.iloc[-1]) and pd.notna(ma_long.iloc[-1]):
        scores['MA'] = 1.0 if ma_short.iloc[-1] > ma_long.iloc[-1] else -1.0
    else: scores['MA'] = 0.0

    adx, pdi, ndi = calculate_adx(df)
    if len(adx) > 4 and pd.notna(adx.iloc[-1]):
        is_rising = adx.iloc[-1] > adx.iloc[-3]
        just_crossed_threshold = adx.iloc[-1] > 22 and adx.iloc[-2] <= 22
        if (adx.iloc[-1] > 22 and is_rising) or just_crossed_threshold:
            score_multiplier = 2.0 if just_crossed_threshold else 1.0
            scores['ADX'] = (1.5 * score_multiplier) if pdi.iloc[-1] > ndi.iloc[-1] else (-1.5 * score_multiplier)
        else: scores['ADX'] = 0.0
    else: scores['ADX'] = 0.0

    bb_width = bollinger_band_width(df)
    if len(bb_width) > 50 and pd.notna(bb_width.iloc[-1]):
        is_in_squeeze = bb_width.iloc[-2] < bb_width.rolling(50).min().iloc[-2]
        middle, upper, lower = calculate_bollinger_bands(df)
        if not all(s.empty for s in [middle, upper, lower]):
            close = df['Close'].iloc[-1]
            zscore = volume_surge(df).iloc[-1]
            if is_in_squeeze and pd.notna(zscore) and zscore > 1.5:
                if close > upper.iloc[-1]: scores['Bollinger'] = 2.0
                elif close < lower.iloc[-1]: scores['Bollinger'] = -2.0
                else: scores['Bollinger'] = 0.0
            elif pd.notna(close) and pd.notna(middle.iloc[-1]):
                scores['Bollinger'] = 0.5 if close > middle.iloc[-1] else -0.5
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

    ema_fast = ema(df["Close"], 20)
    ema_slow = ema(df["Close"], 50)
    if len(ema_fast) and len(ema_slow) and pd.notna(ema_fast.iloc[-1]) and pd.notna(ema_slow.iloc[-1]):
        scores["EMA"] = 1.0 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1.0
    else: scores["EMA"] = 0.0

    vwap_line = vwap(df, period=None)
    if len(vwap_line) and pd.notna(vwap_line.iloc[-1]) and pd.notna(df["Close"].iloc[-1]):
        scores["VWAP"] = 1.0 if df["Close"].iloc[-1] > vwap_line.iloc[-1] else -1.0
    else: scores["VWAP"] = 0.0

    atr_val = atr(df, period=14)
    if len(atr_val) >= 6 and all(pd.notna(val) for val in [atr_val.iloc[-1], atr_val.iloc[-5], df["Close"].iloc[-1], df["Close"].iloc[-5]]):
        atr_rising_sharply = (atr_val.iloc[-1] / atr_val.iloc[-5]) > 1.1
        price_up = df["Close"].iloc[-1] > df["Close"].iloc[-5]
        if atr_rising_sharply and price_up: scores["ATR"] = 1.5
        elif atr_rising_sharply and not price_up: scores["ATR"] = -1.5
        else: scores["ATR"] = 0.0
    else: scores["ATR"] = 0.0

    zscore = volume_surge(df, lookback=20)
    if len(zscore) and pd.notna(zscore.iloc[-1]) and len(df) >= 2:
        price_up_last = df["Close"].iloc[-1] > df["Close"].iloc[-2]
        if zscore.iloc[-1] >= 2.0: scores["VolumeSurge"] = 1.5 if price_up_last else 0.0
        elif zscore.iloc[-1] <= -2.0: scores["VolumeSurge"] = -1.5 if not price_up_last else 0.0
        else: scores["VolumeSurge"] = 0.0
    else: scores["VolumeSurge"] = 0.0

    mom = momentum(df, period=10)
    if len(mom) and pd.notna(mom.iloc[-1]):
        scores["Momentum"] = 1.5 if mom.iloc[-1] > 0.01 else (-1.5 if mom.iloc[-1] < -0.01 else 0.0)
    else: scores["Momentum"] = 0.0

    wr = williams_r(df, period=14)
    if len(wr) and pd.notna(wr.iloc[-1]):
        if wr.iloc[-1] < -80: scores["WWL"] = 1.0
        elif wr.iloc[-1] > -20: scores["WWL"] = -1.0
        else: scores["WWL"] = 0.0
    else: scores["WWL"] = 0.0

    for k in INDICATOR_WEIGHTS.keys():
        scores.setdefault(k, 0.0)
    return scores

def analyze_signals(timeframe_dataframes):
    final_score, max_possible = 0.0, 0.0
    for tf_key, df in timeframe_dataframes.items():
        if df is None or len(df) < 50: continue
        indicator_scores = get_indicator_scores(df)
        tf_weight = TIMEFRAME_WEIGHTS.get(tf_key, 1.0)
        for indicator, score in indicator_scores.items():
            ind_weight = INDICATOR_WEIGHTS.get(indicator, 1.0)
            final_score += score * tf_weight * ind_weight
            max_abs_score = max(abs(s) for s in indicator_scores.values() if s != 0) if any(indicator_scores.values()) else 1.0
            max_possible += max(abs(score), max_abs_score) * tf_weight * ind_weight

    if max_possible == 0: return 'Neutral', 0.0
    normalized = (final_score / max_possible) * 100.0

    if normalized >= 65: signal_text = 'Very Strong Buy'
    elif normalized >= 25: signal_text = 'Strong Buy'
    elif normalized <= -65: signal_text = 'Very Strong Sell'
    elif normalized <= -25: signal_text = 'Strong Sell'
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

def pick_session(symbol_orig, seed_any) -> int:
    return (hash(symbol_orig) ^ int(seed_any)) % len(tdhist_pool)

def fetch_daily(symbol_orig, limiter, hist):
    td_symbol = symbol_orig.replace('-EQ', '')
    try:
        t0 = time.time()
        limiter.acquire()
        # Request EOD bars; duration with D/W/M/Y is supported
        df_raw = hist.get_historic_data(td_symbol, duration=DURATION_DAILY, bar_size="EOD")
        t1 = time.time()
        df = normalize_hist_df(df_raw, td_symbol)
        global api_calls_done
        with api_calls_lock:
            api_calls_done += 1
        if api_calls_done > 0 and api_calls_done % 50 == 0:
            logger.info(f"API calls: {api_calls_done}. Sample latency: {(t1 - t0):.2f}s")
        return symbol_orig, df
    except Exception:
        return symbol_orig, None

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
    total_calls = len(stocks)
    stock_multi_data = defaultdict(dict)

    global api_calls_done
    with api_calls_lock:
        api_calls_done = 0

    stop_evt = threading.Event()
    with tqdm(total=total_calls, desc="Prefetching Daily", ncols=100) as api_bar:
        mon = threading.Thread(target=rps_monitor, args=(stop_evt, api_bar), daemon=True)
        mon.start()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for s in stocks:
                si = pick_session(s, 1440)
                futures.append(executor.submit(fetch_daily, s, sess_limiters[si], tdhist_pool[si]))
            for fut in as_completed(futures):
                symbol_orig, df_daily = fut.result()
                if df_daily is not None:
                    swing_map = build_swing_timeframes(df_daily)
                    if len(swing_map) >= 2:
                        stock_multi_data[symbol_orig] = swing_map
                api_bar.update(1)
        stop_evt.set()

    return dict(stock_multi_data)

# ---------------- Output helper (file append) ----------------
def append_table_output(file_path, scan_time_str, bullish_list, bearish_list):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"===== Scan Time: {scan_time_str} =====\n\n")
        # Bullish table
        f.write("Top 20 Bullish (Momentum Breakouts)\n")
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
        f.write("Top 20 Bearish (Momentum Breakdowns)\n")
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

# ---------------- Single-run swing scan ----------------
def run_single_swing_scan():
    logger.info("Starting one-time swing scan (IST)…")

    # Read stocks list
    try:
        with open('shares.txt', 'r') as f:
            stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(stocks)} stocks from shares.txt. Ensure these are liquid F&O stocks.")
    except Exception as e:
        logger.error(f"Could not read shares.txt: {e}")
        return

    # Prefetch daily and build swing TFs
    stock_multi_data = prefetch_all(stocks, max_workers=64)
    if not stock_multi_data:
        print("No stocks with sufficient data found on initial load. Exiting.")
        return

    now_ist = datetime.now(IST)
    output_filename = now_ist.strftime("%Y-%m-%d") + "_options_scan_results.txt"

    signals_this_scan = []
    current_scores = {}

    # Completed periods only from resample(label='right', closed='right')
    time_point_aware = now_ist.replace(second=0, microsecond=0)

    for symbol, timeframe_data in stock_multi_data.items():
        clean_symbol = symbol.replace('-EQ', '')
        filtered_timeframes = {}
        for tf_key, df in timeframe_data.items():
            if df is None or df.empty:
                continue
            df_slice = df[df.index <= time_point_aware]
            if not df_slice.empty and len(df_slice) >= 50:
                filtered_timeframes[tf_key] = df_slice

        if len(filtered_timeframes) < 2:
            continue

        signal, score = analyze_signals(filtered_timeframes)
        current_scores[clean_symbol] = score

        if 'Strong' in signal:
            change = 'NA'
            direction = 'bullish' if 'Buy' in signal else 'bearish'
            signals_this_scan.append({
                'symbol': clean_symbol, 'signal': signal, 'score': score,
                'trend': direction, 'change': change
            })

    # Rank and select top 20 bullish/bearish
    signals_this_scan.sort(key=lambda x: x['score'], reverse=True)
    top_bullish = [r for r in signals_this_scan if 'Buy' in r['signal']][:20]
    bearish_sorted = sorted([r for r in signals_this_scan if 'Sell' in r['signal']], key=lambda x: x['score'])
    top_bearish = bearish_sorted[:20]

    # Console output
    print("\n" + "="*92)
    print(f"| OPTION BUYER SWING SCANNER | SIGNALS AT {now_ist.strftime('%Y-%m-%d %H:%M')} IST".center(100) + " |")
    print("="*92)

    # Bullish table
    print(f"| {'Top 20 Bullish Breakouts':<88} |")
    print("-"*92)
    if not top_bullish:
        print("| None".ljust(91) + " |")
    else:
        print(f"| {'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Action':<19} |")
        print("-"*92)
        for result in top_bullish:
            signal_text, change_val = result['signal'], result['change']
            if isinstance(change_val, (int, float, np.floating)):
                sign = '+' if change_val > 0 else ''
                color = GREEN if change_val > 0 else RED
                change_str = f"{color}{sign}{change_val:>.2f}{RESET}"
                padding = 19
            else:
                change_str = "NA"
                padding = 10
            colored_signal = f"{GREEN}{signal_text:<18}{RESET}" if 'Buy' in signal_text else f"{RED}{signal_text:<18}{RESET}"
            action = f"{GREEN}Consider Call{RESET}" if 'Buy' in signal_text else f"{RED}Consider Put{RESET}"
            print(f"| {result['symbol']:<15} | {colored_signal} | {result['score']:>7.2f} | {change_str:>{padding}} | {result['trend']:<10} | {action:<29} |")

    # Bearish table
    print("-"*92)
    print(f"| {'Top 20 Bearish Breakdowns':<88} |")
    print("-"*92)
    if not top_bearish:
        print("| None".ljust(91) + " |")
    else:
        print(f"| {'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Action':<19} |")
        print("-"*92)
        for result in top_bearish:
            signal_text, change_val = result['signal'], result['change']
            if isinstance(change_val, (int, float, np.floating)):
                sign = '+' if change_val > 0 else ''
                color = GREEN if change_val > 0 else RED
                change_str = f"{color}{sign}{change_val:>.2f}{RESET}"
                padding = 19
            else:
                change_str = "NA"
                padding = 10
            colored_signal = f"{GREEN}{signal_text:<18}{RESET}" if 'Buy' in signal_text else f"{RED}{signal_text:<18}{RESET}"
            action = f"{GREEN}Consider Call{RESET}" if 'Buy' in signal_text else f"{RED}Consider Put{RESET}"
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

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    try:
        run_single_swing_scan()
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
