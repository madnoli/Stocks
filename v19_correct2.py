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

# ---------- Feature toggles / env ----------
# Turn OFF options confirmation until fetch_option_chain is implemented to avoid dropping all candidates
OPTIONS_CONFIRM = os.getenv("OPTIONS_CONFIRM", "0") == "1"  # enable after implementing fetch_option_chain [web:7]
# Turn OFF index alignment influence while verifying index symbols; set to 0 later when correct symbols are used
INDEX_ALIGN_DISABLE = os.getenv("INDEX_ALIGN_DISABLE", "1") == "1"  # default 1 for first run [web:60]

OPTION_LTP_MIN = float(os.getenv("OPTION_LTP_MIN", "80"))
OPTION_LTP_MAX = float(os.getenv("OPTION_LTP_MAX", "300"))
OPTION_SPREAD_MAX_PCT = float(os.getenv("OPTION_SPREAD_MAX_PCT", "1.0"))  # %
OPTION_SPREAD_MAX_ABS = float(os.getenv("OPTION_SPREAD_MAX_ABS", "0.8"))  # INR
OPTION_VOL_Z_MIN = float(os.getenv("OPTION_VOL_Z_MIN", "2.0"))

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

# ---------- Timeframe and indicator weights ----------
TIMEFRAME_WEIGHTS = {1: 1.8, 5: 1.6, 15: 1.3, 30: 1.1, 60: 0.8, 1440: 0.5}  # entry emphasis on 1–5m [web:52]
INDICATOR_WEIGHTS = {
    "RSI": 1.2, "MACD": 1.8, "Stochastic": 0.8, "MA": 1.0, "ADX": 1.5,
    "Bollinger": 0.6, "ROC": 0.6, "OBV": 1.0, "CCI": 0.5, "WWL": 0.8,
    "EMA": 2.0, "VWAP": 1.8, "ATR": 1.2, "VolumeSurge": 2.2, "Momentum": 1.4,
    "VWAPDistance": 1.8, "EMACluster": 2.0, "IndexAlign": 0.0 if INDEX_ALIGN_DISABLE else 1.4, "ATRPercent": 1.2
}  # VWAP/EMA cluster + volume surge prioritized for option buying entries [web:72]

BAR_SIZE_MAP = {1: "1 min", 5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"}  # TF map [web:52]
DURATION_MAP = {1: "5 D", 5: "30 D", 15: "30 D", 30: "60 D", 60: "120 D", 1440: "365 D"}  # history windows [web:52]

# Live settings
SCAN_SECONDS = int(os.getenv("SCAN_SECONDS", "45"))  # 45s cadence for 1m timing [web:52]
MARKET_START = "09:15"
MARKET_END   = "15:30"
INCLUDE_PREOPEN = False
MAX_BARS_KEEP = 1500
RUN_ONCE_AND_EXIT = False

# Relaxed thresholds to surface names in midday conditions
MIN_ATR_PCT_5M = float(os.getenv("MIN_ATR_PCT_5M", "0.007"))  # 0.7% ATR% [web:52]
MIN_VOL_Z_PACE = float(os.getenv("MIN_VOL_Z_PACE", "1.05"))   # slightly above average pace [web:52]
MAX_VWAP_ATR_DIST = float(os.getenv("MAX_VWAP_ATR_DIST", "1.8"))  # allow farther trends [web:52]
MIN_ADX = float(os.getenv("MIN_ADX", "18.0"))  # earlier trend detection [web:52]
RSI_BUY_MIN = float(os.getenv("RSI_BUY_MIN", "50.0"))
RSI_BUY_MAX = float(os.getenv("RSI_BUY_MAX", "78.0"))
VOL_Z_ENTRY = float(os.getenv("VOL_Z_ENTRY", "1.2"))  # moderate volume surges [web:52]

# Diagnostics toggles
DEBUG_PREFILTER = os.getenv("DEBUG_PREFILTER", "1") == "1"  # on by default for debugging [web:52]
DEBUG_ENTRY = os.getenv("DEBUG_ENTRY", "1") == "1"  # on by default for debugging [web:52]

# ---------------- TrueData sessions ----------------
def authenticate_session():
    return TD_hist(TDUSERNAME, TDPASSWORD, log_level=logging.CRITICAL)  # auth via Market Data API [web:60]

def build_sessions():
    sess_count = int(os.getenv("TD_HIST_SESSIONS", "3"))  # 3–5 recommended [web:60]
    pool = []
    for i in range(sess_count):
        try:
            pool.append(authenticate_session())
        except Exception as e:
            logger.error(f"Session {i} init failed: {e}")
    if not pool:
        raise SystemExit("Failed to initialize TrueData sessions.")
    per_sess_rate = 10.0 / len(pool)  # global 10 rps split across sessions [web:69]
    limiters = [TokenBucketLimiter(rate_per_sec=per_sess_rate, bucket_size=10) for _ in pool]
    return pool, limiters

tdhist_pool, sess_limiters = build_sessions()
logger.info("TrueData login successful.")  # sessions ready [web:60]

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

# ---------------- Base indicators ----------------
def calculate_rsi(df, period=14):
    if len(df) < period + 1: return pd.Series(dtype='float64')
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    if len(df) < slow + signal: return pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64')
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return macd, signal_line, hist

def calculate_stochastic(df, period=14, smooth_d=3):
    if len(df) < period + smooth_d: return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_moving_averages(df, short=20, long=50):
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

# ---------------- Index cache ----------------
# If symbols don’t load on your account, disable index bias (INDEX_ALIGN_DISABLE=1) and later replace with exact names from TrueData symbol list [web:60]
INDEX_SYMBOLS = ["NIFTY 50", "NIFTY BANK"]  # verify these on your account [web:74]

def get_index_alignment(index_data):
    if INDEX_ALIGN_DISABLE:
        return 0
    align_score = 0
    for name, df in index_data.items():
        if df is None or len(df) < 50:
            continue
        v = vwap(df, period=None)
        ema20 = ema(df["Close"], 20)
        ema50 = ema(df["Close"], 50)
        if len(v)==0 or len(ema20)==0 or len(ema50)==0:
            continue
        if df["Close"].iloc[-1] > v.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]:
            align_score += 1
        elif df["Close"].iloc[-1] < v.iloc[-1] and ema20.iloc[-1] < ema50.iloc[-1]:
            align_score -= 1
    if align_score > 0: return 1
    if align_score < 0: return -1
    return 0

# ---------------- Scoring ----------------
def get_indicator_scores(df, index_align=0):
    scores = {}
    rsi_series = calculate_rsi(df)
    rsi = rsi_series.iloc[-1] if len(rsi_series) else np.nan
    if pd.notna(rsi):
        if RSI_BUY_MIN <= rsi <= RSI_BUY_MAX:
            scores['RSI'] = 1.0
        elif rsi > 80:
            scores['RSI'] = -0.8
        elif rsi < 45:
            scores['RSI'] = -0.5
        else:
            scores['RSI'] = 0.2
    else:
        scores['RSI'] = 0.0

    macd, signal, hist = calculate_macd(df)
    if len(hist) >= 2 and pd.notna(hist.iloc[-1]) and pd.notna(hist.iloc[-2]):
        rising = hist.iloc[-1] > hist.iloc[-2]
        above = pd.notna(macd.iloc[-1]) and pd.notna(signal.iloc[-1]) and macd.iloc[-1] > signal.iloc[-1]
        if above and rising:
            scores['MACD'] = 1.0
        elif above:
            scores['MACD'] = 0.5
        else:
            scores['MACD'] = -0.5
    else:
        scores['MACD'] = 0.0

    k, d = calculate_stochastic(df)
    if len(k) and len(d) and pd.notna(k.iloc[-1]) and pd.notna(d.iloc[-1]):
        if k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 85: scores['Stochastic'] = 0.4
        elif k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 15: scores['Stochastic'] = -0.4
        else: scores['Stochastic'] = 0.0
    else:
        scores['Stochastic'] = 0.0

    ma_short, ma_long = calculate_moving_averages(df, short=20, long=50)
    if len(ma_short) and len(ma_long) and pd.notna(ma_short.iloc[-1]) and pd.notna(ma_long.iloc[-1]):
        scores['MA'] = 0.3 if ma_short.iloc[-1] > ma_long.iloc[-1] else -0.3
    else:
        scores['MA'] = 0.0

    adx, pdi, ndi = calculate_adx(df)
    if len(adx) and pd.notna(adx.iloc[-1]) and adx.iloc[-1] >= MIN_ADX:
        scores['ADX'] = 1.0 if pdi.iloc[-1] > ndi.iloc[-1] else -1.0
    else:
        scores['ADX'] = 0.0

    middle = calculate_bollinger_bands(df)
    if len(middle) and pd.notna(middle.iloc[-1]) and pd.notna(df['Close'].iloc[-1]):
        scores['Bollinger'] = 0.3 if df['Close'].iloc[-1] > middle.iloc[-1] else -0.3
    else:
        scores['Bollinger'] = 0.0

    roc = calculate_roc(df).iloc[-1] if len(df) else np.nan
    scores['ROC'] = 0.3 if pd.notna(roc) and roc > 0 else (-0.3 if pd.notna(roc) else 0.0)

    obv = calculate_obv(df)
    if len(obv) >= 2 and pd.notna(obv.iloc[-1]) and pd.notna(obv.iloc[-2]):
        scores['OBV'] = 0.5 if obv.iloc[-1] > obv.iloc[-2] else -0.5
    else:
        scores['OBV'] = 0.0

    cci_val = calculate_cci(df).iloc[-1] if len(df) else np.nan
    if pd.notna(cci_val):
        if cci_val > 100: scores['CCI'] = 0.4
        elif cci_val < -100: scores['CCI'] = -0.4
        else: scores['CCI'] = 0.0
    else:
        scores['CCI'] = 0.0

    ema20 = ema(df["Close"], 20)
    ema50 = ema(df["Close"], 50)
    if len(ema20) and len(ema50) and pd.notna(ema20.iloc[-1]) and pd.notna(ema50.iloc[-1]):
        scores["EMA"] = 1.0 if ema20.iloc[-1] > ema50.iloc[-1] else -1.0
        if pd.notna(df["Close"].iloc[-1]):
            above_both = df["Close"].iloc[-1] > ema20.iloc[-1] and df["Close"].iloc[-1] > ema50.iloc[-1]
            scores["EMACluster"] = 1.0 if (above_both and ema20.iloc[-1] > ema50.iloc[-1]) else -0.8
    else:
        scores["EMA"] = 0.0
        scores["EMACluster"] = 0.0

    vwap_line = vwap(df, period=None)
    if len(vwap_line) and pd.notna(vwap_line.iloc[-1]) and pd.notna(df["Close"].iloc[-1]):
        scores["VWAP"] = 1.0 if df["Close"].iloc[-1] > vwap_line.iloc[-1] else -1.0
    else:
        scores["VWAP"] = 0.0

    atr_val = atr(df, period=14)
    if len(atr_val) >= 6 and pd.notna(atr_val.iloc[-1]) and pd.notna(atr_val.iloc[-5]) and pd.notna(df["Close"].iloc[-1]) and pd.notna(df["Close"].iloc[-5]):
        atr_rising = atr_val.iloc[-1] > atr_val.iloc[-5]
        price_up = df["Close"].iloc[-1] > df["Close"].iloc[-5]
        if atr_rising and price_up: scores["ATR"] = 0.6
        elif atr_rising and not price_up: scores["ATR"] = -0.6
        else: scores["ATR"] = 0.0
    else:
        scores["ATR"] = 0.0

    zscore = volume_surge(df, lookback=20)
    if len(zscore) and pd.notna(zscore.iloc[-1]) and len(df) >= 2:
        price_up_last = df["Close"].iloc[-1] > df["Close"].iloc[-2]
        if zscore.iloc[-1] >= VOL_Z_ENTRY:
            scores["VolumeSurge"] = 1.0 if price_up_last else 0.0
        elif zscore.iloc[-1] <= -VOL_Z_ENTRY:
            scores["VolumeSurge"] = -1.0 if not price_up_last else 0.0
        else:
            scores["VolumeSurge"] = 0.0
    else:
        scores["VolumeSurge"] = 0.0

    mom = momentum(df, period=10)
    if len(mom) and pd.notna(mom.iloc[-1]):
        scores["Momentum"] = 0.6 if mom.iloc[-1] > 0 else -0.6
    else:
        scores["Momentum"] = 0.0

    # VWAP distance anti-chase
    if len(atr_val) and len(vwap_line) and pd.notna(atr_val.iloc[-1]) and atr_val.iloc[-1] > 0 and pd.notna(vwap_line.iloc[-1]) and pd.notna(df["Close"].iloc[-1]):
        vdist = abs(df["Close"].iloc[-1] - vwap_line.iloc[-1]) / atr_val.iloc[-1]
        if vdist > MAX_VWAP_ATR_DIST:
            scores["VWAPDistance"] = -1.0
        else:
            scores["VWAPDistance"] = 0.5 if df["Close"].iloc[-1] >= vwap_line.iloc[-1] else -0.2
    else:
        scores["VWAPDistance"] = 0.0

    # ATR% range potential
    if len(atr_val) and pd.notna(atr_val.iloc[-1]) and pd.notna(df["Close"].iloc[-1]) and df["Close"].iloc[-1] != 0:
        atr_pct = atr_val.iloc[-1] / df["Close"].iloc[-1]
        scores["ATRPercent"] = 0.6 if atr_pct >= MIN_ATR_PCT_5M else -0.6
    else:
        scores["ATRPercent"] = 0.0

    if index_align == 1:
        scores["IndexAlign"] = 0.8
    elif index_align == -1:
        scores["IndexAlign"] = -0.8
    else:
        scores["IndexAlign"] = 0.0

    for k in INDICATOR_WEIGHTS.keys():
        scores.setdefault(k, 0.0)
    return scores

def analyze_signals(timeframe_dataframes, index_align=0):
    final_score, max_possible = 0.0, 0.0
    for tf_min, df in timeframe_dataframes.items():
        if df is None or len(df) < 50:
            continue
        indicator_scores = get_indicator_scores(df, index_align=index_align if tf_min in (1,5,15) else 0)
        tf_weight = TIMEFRAME_WEIGHTS.get(tf_min, 1.0)
        for indicator, score in indicator_scores.items():
            ind_weight = INDICATOR_WEIGHTS.get(indicator, 1.0)
            final_score += score * tf_weight * ind_weight
            max_possible += max(abs(score), 1.0) * tf_weight * ind_weight
    if max_possible == 0: return 'Neutral', 0.0
    normalized = (final_score / max_possible) * 100.0
    if normalized >= 60: signal_text = 'Very Strong Buy'
    elif normalized >= 35: signal_text = 'Strong Buy'
    elif normalized <= -60: signal_text = 'Very Strong Sell'
    elif normalized <= -35: signal_text = 'Strong Sell'
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
        if api_calls_done > 0 and api_calls_done % 50 == 0:
            logger.info(f"API calls: {api_calls_done}. Sample latency: {(t1 - t0):.2f}s")
        return symbol_orig, timeframe_minutes, df
    except Exception:
        return symbol_orig, timeframe_minutes, None

def rps_monitor(stop_event, bar):
    prev = 0
    while not stop_event.is_set():
        time.sleep(2.0)  # reduce overhead of frequent postfix updates [web:90]
        with api_calls_lock:
            curr = api_calls_done
        rps = curr - prev
        prev = curr
        try:
            bar.set_postfix_str(f"rps={rps}")
        except Exception:
            pass

def prefetch_all(stocks, max_workers=64):
    tfs = [1, 5, 15, 30, 60, 1440]
    total_calls = len(stocks) * len(tfs)
    stock_multi_data = defaultdict(dict)

    global api_calls_done
    with api_calls_lock:
        api_calls_done = 0

    stop_evt = threading.Event()
    with tqdm(total=total_calls + len(INDEX_SYMBOLS), desc="Prefetching Data", ncols=100) as api_bar:
        mon = threading.Thread(target=rps_monitor, args=(stop_evt, api_bar), daemon=True)
        mon.start()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for s in stocks:
                for tf in tfs:
                    si = pick_session(s, tf)
                    futures.append(executor.submit(fetch_one, s, tf, sess_limiters[si], tdhist_pool[si]))
            # index 5m
            for idx in INDEX_SYMBOLS:
                si = pick_session(idx, 5)
                futures.append(executor.submit(fetch_one, idx, 5, sess_limiters[si], tdhist_pool[si]))

            for fut in as_completed(futures):
                symbol_orig, tf, df = fut.result()
                if df is not None:
                    stock_multi_data[symbol_orig][tf] = df
                api_bar.update(1)
        stop_evt.set()

    keep = {}
    for s, d in stock_multi_data.items():
        if s in stocks and len(d) >= 2:
            keep[s] = d
        if s in INDEX_SYMBOLS:
            keep[s] = d
    return keep

def in_market_hours(now_ist: datetime) -> bool:
    start_str = "09:00" if INCLUDE_PREOPEN else MARKET_START
    start_t = datetime.strptime(start_str, "%H:%M").time()
    end_t = datetime.strptime(MARKET_END, "%H:%M").time()
    return start_t <= now_ist.time() <= end_t

def trim_df(df: pd.DataFrame, max_len: int) -> pd.DataFrame:
    if len(df) <= max_len:
        return df
    return df.iloc[-max_len:]

# ------------- Prefiltering helpers -------------
def compute_atr_pct(df):
    a = atr(df, 14)
    if len(a)==0 or pd.isna(a.iloc[-1]) or pd.isna(df["Close"].iloc[-1]) or df["Close"].iloc[-1]==0:
        return 0.0
    return float(a.iloc[-1] / df["Close"].iloc[-1])

def volume_pace_ratio(df, lookback=20):
    if len(df) < lookback + 1:
        return 1.0
    v = df["Volume"]
    mean = v.rolling(lookback).mean().iloc[-2]
    last = v.iloc[-1]
    if pd.isna(mean) or mean == 0:
        return 1.0
    return float(last / mean)

def universe_prefilter(symbol, slice_5m):
    if slice_5m is None or len(slice_5m) < 50:
        return False
    atrp = compute_atr_pct(slice_5m)
    pace = volume_pace_ratio(slice_5m, 20)
    keep = (atrp >= MIN_ATR_PCT_5M) and (pace >= MIN_VOL_Z_PACE)
    if DEBUG_PREFILTER:
        logger.info(f"[PREFILTER] {symbol}: atr%={atrp:.3%}, pace={pace:.2f}, keep={keep}")
    return keep

# ------------- Options chain confirmation (optional) -------------
def approximate_atm_strike(underlying_price, step=50.0):
    return int(round(underlying_price / step) * step)

def option_spread_ok(bid, ask, ltp):
    try:
        bid = float(bid); ask = float(ask); ltp = float(ltp)
    except Exception:
        return False
    if ltp <= 0: return False
    spread = ask - bid
    if spread < 0: return False
    if spread <= OPTION_SPREAD_MAX_ABS:
        return True
    return (spread / ltp * 100.0) <= OPTION_SPREAD_MAX_PCT

def pick_option_candidate(chain_df, is_call=True):
    side = "CE" if is_call else "PE"
    if chain_df is None or chain_df.empty:
        return None
    c = chain_df.copy()
    c["type"] = c["type"].str.upper()
    subset = c[c['type']==side]
    if subset.empty:
        return None
    for col in ["ltp","bid","ask","volume","strike"]:
        subset[col] = pd.to_numeric(subset[col], errors="coerce")
    subset = subset[subset["ltp"].between(OPTION_LTP_MIN, OPTION_LTP_MAX)]
    subset["spread_ok"] = subset.apply(lambda r: option_spread_ok(r.get("bid"), r.get("ask"), r.get("ltp")), axis=1)
    if "volume" in subset.columns and len(subset) >= 10:
        mv = subset["volume"].mean()
        sv = subset["volume"].std(ddof=0)
        subset["vol_z"] = (subset["volume"] - mv) / (sv if sv and sv>0 else 1.0)
    else:
        subset["vol_z"] = 0.0
    subset = subset[(subset["spread_ok"]==True) & (subset["vol_z"] >= OPTION_VOL_Z_MIN)]
    if subset.empty:
        return None
    subset = subset.sort_values(["vol_z","volume"], ascending=[False, False])
    return subset.iloc[0].to_dict()

def fetch_option_chain(hist, symbol, expiry=None, strikes_range=5):
    # Stub: implement with TrueData option-chain API; return DataFrame with:
    # ['type','strike','ltp','bid','ask','volume','oi','iv']
    # Keep OPTIONS_CONFIRM=0 until implemented to avoid suppressing candidates [web:7]
    try:
        return None
    except Exception:
        return None

# ---------------- Output helper (file append) ----------------
def append_table_output(file_path, scan_time_str, bullish_list, bearish_list):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"===== Scan Time: {scan_time_str} =====\n\n")
        # Bullish table
        f.write("Top 20 Bullish\n")
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
        f.write("Top 20 Bearish\n")
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

# ---------------- Live market loop (Top-20) ----------------
def live_market_scanner_top20():
    logger.info("Starting market scanner (IST)…")
    logger.info("Mode: Run Once and Exit (Test Mode)." if RUN_ONCE_AND_EXIT else "Mode: Live Continuous Scanning.")

    # Read stocks
    try:
        with open('shares.txt', 'r') as f:
            stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(stocks)} stocks from shares.txt")
    except Exception as e:
        logger.error(f"Could not read shares.txt: {e}")
        return

    # Prefetch (stocks + indices)
    stock_multi_data = prefetch_all(stocks, max_workers=64)
    if not stock_multi_data:
        print("No stocks with sufficient data found on initial load. Exiting.")
        return

    output_filename = datetime.now(IST).strftime("%Y-%m-%d") + "_scan_results.txt"
    previous_scores = {}
    tfs = [1, 5, 15, 30, 60, 1440]
    entry_state = {}  # symbol -> {"last_qual_bar": ts, "window_bars_left": int}

    while True:
        now_ist = datetime.now(IST)
        fname_today = now_ist.strftime("%Y-%m-%d") + "_scan_results.txt"
        if fname_today != output_filename:
            output_filename = fname_today
            previous_scores = {}
            entry_state.clear()

        if not RUN_ONCE_AND_EXIT and not in_market_hours(now_ist):
            print(f"Waiting for market hours ({MARKET_START}-{MARKET_END} IST). Current time: {now_ist.strftime('%H:%M:%S')}", end="\r")
            time.sleep(30)
            continue

        logger.info(f"[{now_ist.strftime('%H:%M:%S')}] Refreshing data for {len(stock_multi_data)} symbols...")
        with ThreadPoolExecutor(max_workers=64) as executor:
            futures = []
            # update stocks
            for s in list(stock_multi_data.keys()):
                if s in INDEX_SYMBOLS:
                    continue
                for tf in tfs:
                    si = pick_session(s, tf)
                    futures.append(executor.submit(fetch_one, s, tf, sess_limiters[si], tdhist_pool[si]))
            # update indices 5m
            for idx in INDEX_SYMBOLS:
                si = pick_session(idx, 5)
                futures.append(executor.submit(fetch_one, idx, 5, sess_limiters[si], tdhist_pool[si]))

            for fut in as_completed(futures):
                symbol_orig, tf, df = fut.result()
                if df is None or len(df) < 50:
                    continue
                d = stock_multi_data.setdefault(symbol_orig, {})
                if tf in d and d[tf] is not None and not d[tf].empty:
                    merged = pd.concat([d[tf], df]).sort_index()
                    merged = merged[~merged.index.duplicated(keep='last')]
                    d[tf] = trim_df(merged, MAX_BARS_KEEP)
                else:
                    d[tf] = trim_df(df, MAX_BARS_KEEP)
        logger.info("Data refresh complete. Analyzing signals...")

        # Index alignment (5m)
        index_data_5m = {idx: stock_multi_data.get(idx, {}).get(5) for idx in INDEX_SYMBOLS}
        index_align = get_index_alignment(index_data_5m)

        time_point_aware = now_ist.replace(second=0, microsecond=0)
        signals_this_scan = []
        current_scores = {}

        for symbol, timeframe_data in stock_multi_data.items():
            if symbol in INDEX_SYMBOLS:
                continue
            clean_symbol = symbol.replace('-EQ', '')

            # slice TFs up to now
            filtered_timeframes = {}
            for tf, df in timeframe_data.items():
                if df is None or df.empty:
                    continue
                df_slice = df[df.index <= time_point_aware]
                if not df_slice.empty and len(df_slice) >= 50:
                    filtered_timeframes[tf] = df_slice
            if len(filtered_timeframes) < 2:
                continue

            # Prefilter on 5m for range/liquidity
            df5 = filtered_timeframes.get(5)
            if not universe_prefilter(clean_symbol, df5):
                continue

            # Primary signal scoring with index bias
            signal, score = analyze_signals(filtered_timeframes, index_align=index_align)
            current_scores[clean_symbol] = score

            # Entry window logic (1m confirmation + micro pullback)
            df1 = filtered_timeframes.get(1)
            qualified = False
            if df1 is not None and len(df1) >= 3:
                v = vwap(df1, period=None)
                e20 = ema(df1["Close"], 20)
                e50 = ema(df1["Close"], 50)
                adx1, pdi1, ndi1 = calculate_adx(df1)
                rsi1 = calculate_rsi(df1)
                z = volume_surge(df1)
                atr1 = atr(df1, 14)
                if len(v) and len(e20) and len(e50) and len(adx1) and len(rsi1) and len(z) >= 2:
                    last = df1.index[-1]
                    close = df1["Close"].iloc[-1]
                    prev_close = df1["Close"].iloc[-2]
                    low = df1["Low"].iloc[-1]
                    vdist = 0.0
                    if len(atr1) and pd.notna(atr1.iloc[-1]) and atr1.iloc[-1] > 0:
                        vdist = abs(close - v.iloc[-1]) / atr1.iloc[-1]
                    # Qualify bar for call-buyers
                    conds = {
                        "above_vwap": close > v.iloc[-1],
                        "ema20_gt_ema50": e20.iloc[-1] > e50.iloc[-1],
                        "close_above_ema20": close > e20.iloc[-1],
                        "close_above_ema50": close > e50.iloc[-1],
                        "adx_min": pd.notna(adx1.iloc[-1]) and adx1.iloc[-1] >= MIN_ADX,
                        "pdi_gt_ndi": pdi1.iloc[-1] > ndi1.iloc[-1],
                        "rsi_band": pd.notna(rsi1.iloc[-1]) and RSI_BUY_MIN <= rsi1.iloc[-1] <= RSI_BUY_MAX,
                        "vol_z": pd.notna(z.iloc[-1]) and z.iloc[-1] >= VOL_Z_ENTRY,
                        "vwap_dist_ok": vdist <= MAX_VWAP_ATR_DIST
                    }
                    if all(conds.values()):
                        qualified = True
                        entry_state[clean_symbol] = {"last_qual_bar": last, "window_bars_left": 4}
                        if DEBUG_ENTRY:
                            logger.info(f"[ENTRY QUAL] {clean_symbol} 1m ok: "
                                        f"RSI={rsi1.iloc[-1]:.1f} ADX={adx1.iloc[-1]:.1f} "
                                        f"z={z.iloc[-1]:.2f} vdist={vdist:.2f}")
                    else:
                        st = entry_state.get(clean_symbol)
                        if st and st["window_bars_left"] > 0:
                            # Micro pullback: wick touch or close near EMA20, close > VWAP, lower vol
                            vol_ok = df1["Volume"].iloc[-1] <= df1["Volume"].iloc[-2] if len(df1) >= 2 else True
                            if (close >= e20.iloc[-1] or low <= e20.iloc[-1]) and vol_ok and close > v.iloc[-1] and conds["ema20_gt_ema50"]:
                                qualified = True
                                if DEBUG_ENTRY:
                                    logger.info(f"[ENTRY PULL] {clean_symbol} pullback ok: "
                                                f"close={close:.2f}, ema20={e20.iloc[-1]:.2f}, low={low:.2f}")
                            st["window_bars_left"] -= 1
                        else:
                            entry_state.pop(clean_symbol, None)
                            if DEBUG_ENTRY and 'Strong Buy' in signal:
                                # Print first failing gate for diagnostics
                                for k, v_ok in conds.items():
                                    if not v_ok:
                                        logger.info(f"[ENTRY FAIL] {clean_symbol} gate failed: {k}")
                                        break

            # Option-chain confirmation (optional)
            if OPTIONS_CONFIRM and qualified and ('Strong Buy' in signal or 'Very Strong Buy' in signal):
                chain = fetch_option_chain(tdhist_pool[0], clean_symbol, expiry=None, strikes_range=5)
                pick = pick_option_candidate(chain, is_call=True) if chain is not None else None
                if pick is None:
                    qualified = False
                    if DEBUG_ENTRY:
                        logger.info(f"[OPTIONS DROP] {clean_symbol}: no tradable CE found in chain")

            if ('Strong' in signal) and qualified:
                change = 'NA' if clean_symbol not in previous_scores else score - previous_scores.get(clean_symbol, 0.0)
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
