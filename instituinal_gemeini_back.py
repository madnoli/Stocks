import os
import logging
import warnings
import argparse
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from logzero import logger, setup_default_logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from collections import defaultdict

from tqdm import tqdm
from truedata.history import TD_hist

# ======== Config ========
TDUSERNAME = os.getenv("TRUEDATA_USER", "tdwsp751")
TDPASSWORD = os.getenv("TRUEDATA_PASS", "raj@751")

MARKET_START = "09:15"   # IST
FIRST_RUN_AT = "09:20"   # IST; first 5-min close
MARKET_END   = "15:30"   # IST
SETTLE_DELAY_SECONDS = 5  # wait after bar close
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "48"))
TD_HIST_SESSIONS = int(os.getenv("TD_HIST_SESSIONS", "3"))
LOG_LEVEL = logging.INFO # Change to logging.DEBUG for detailed filter logs

# Universe file
SHARES_FILE = os.getenv("SHARES_FILE", "shares.txt")

# --- New Filters for Option Buyers ---
MIN_STOCK_PRICE = 100.0              # Minimum stock price to consider
MIN_AVG_DAILY_TURNOVER = 5_00_00_000 # Minimum 20-day avg turnover (5 Crore INR)

IST = pytz.timezone("Asia/Kolkata")

# Setup logging level
setup_default_logger(level=LOG_LEVEL)

# Silence noisy third‑party loggers
for noisy in ("truedata", "truedata.history", "truedata_ws", "websocket", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

# ---------- 5-minute boundary helpers ----------
def next_5min_boundary_ist(now_ist: datetime) -> datetime:
    minute = (now_ist.minute // 5) * 5
    boundary = now_ist.replace(minute=minute, second=0, microsecond=0)
    if boundary <= now_ist:
        boundary = boundary + timedelta(minutes=5)
    return boundary

def parse_hhmm(s: str):
    h, m = map(int, s.split(":"))
    return h, m

def today_ist_dt(hhmm: str) -> datetime:
    now = datetime.now(IST)
    h, m = parse_hhmm(hhmm)
    return now.replace(hour=h, minute=m, second=0, microsecond=0)

def sleep_until(ts: datetime):
    while True:
        now = datetime.now(IST)
        delta = (ts - now).total_seconds()
        if delta <= 0:
            break
        time.sleep(min(0.5, delta))

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

api_calls_done = 0
api_calls_lock = threading.Lock()

# ---------- Weights (Updated for Option Buyers) ----------
ENHANCED_INDICATOR_WEIGHTS = {
    "VolumeSurge": 2.2, "Momentum": 2.1, "ATR": 1.8, "ROC": 1.5,
    "ADX": 1.8, "VWAP": 1.7, "EMA": 1.7,
    "MACD": 1.5, "OBV": 1.5,
    "Bollinger": 1.3, "RSI": 1.2,
    "Stochastic": 1.0, "CCI": 1.0, "MA": 1.0, "WWL": 1.0,
}
INDICATOR_WEIGHTS = ENHANCED_INDICATOR_WEIGHTS | {
    "CMF": 1.6, "ADL": 1.4, "RelVol": 1.3, "VWAPRegime": 1.6, "OBVConfirm": 1.2
}
TIMEFRAME_WEIGHTS = {15: 3.0, 5: 2.5, 30: 2.0, 60: 1.5, "daily": 1.0}

BAR_SIZE_MAP = {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"}
DURATION_MAP = {5: "30 D", 15: "30 D", 30: "60 D", 60: "120 D", 1440: "365 D"}

# ---------- TrueData sessions ----------
def authenticate_session():
    return TD_hist(TDUSERNAME, TDPASSWORD, log_level=logging.CRITICAL)

def build_sessions():
    sess_count = TD_HIST_SESSIONS
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

# ---------- Indicators & Helpers (omitted for brevity, no changes from previous version) ----------
def ema(series, length): return series.ewm(span=length, adjust=False).mean()
def vwap(df, period=None):
    price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = price * df["Volume"]
    pv_sum = pv.rolling(period).sum() if period else pv.cumsum()
    vol_sum = df["Volume"].rolling(period).sum() if period else df["Volume"].cumsum()
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
    return -100 * (highest - df["Close"]) / (highest - lowest).replace(0, np.nan)
def momentum(df, period=10): return df["Close"] / df["Close"].shift(period) - 1.0
def volume_surge(df, lookback=20):
    vol_ma = df["Volume"].rolling(lookback).mean()
    vol_std = df["Volume"].rolling(lookback).std()
    z_score = (df["Volume"] - vol_ma) / vol_std.replace(0, np.nan)
    return z_score.fillna(0)
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
    atr_val = df_adx['TR'].ewm(com=period - 1, adjust=False).mean().replace(0, np.nan)
    pdi = (df_adx['+DM'].ewm(com=period - 1, adjust=False).mean() / atr_val) * 100
    ndi = (df_adx['-DM'].ewm(com=period - 1, adjust=False).mean() / atr_val) * 100
    adx = (abs(pdi - ndi) / (pdi + ndi).replace(0, np.nan)).ewm(com=period - 1, adjust=False).mean() * 100
    return adx.fillna(20), pdi.fillna(20), ndi.fillna(20)
def calculate_bollinger_bands(df, period=20, std_dev=2):
    if len(df) < period: return pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64')
    middle = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return middle, upper, lower
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
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).replace(0, np.nan)
    return (tp - sma_tp) / (0.015 * mad)
def adl(df):
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = mfm.fillna(0)
    mfv = mfm * df["Volume"]
    return mfv.cumsum()
def cmf(df, period=20):
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = mfm.fillna(0)
    mfv = mfm * df["Volume"]
    mfv_sum = mfv.rolling(period).sum()
    vol_sum = df["Volume"].rolling(period).sum().replace(0, np.nan)
    return (mfv_sum / vol_sum).fillna(0)
def relative_volume(df, lookback=50):
    vol_ma = df["Volume"].rolling(lookback).mean()
    return (df["Volume"] / vol_ma.replace(0, np.nan)).fillna(0)
def vwap_distance(df, period=None):
    v = vwap(df, period=period)
    return ((df["Close"] - v) / v.replace(0, np.nan)).fillna(0)
def get_volatility_metric(df, period=14):
    if df is None or len(df) < period + 1: return 0.0
    try:
        atr_series = atr(df, period=period)
        last_close = df['Close'].iloc[-1]
        last_atr = atr_series.iloc[-1]
        if pd.isna(last_close) or pd.isna(last_atr) or last_close == 0: return 0.0
        return (last_atr / last_close) * 100.0
    except (IndexError, KeyError): return 0.0
def get_indicator_scores(df):
    scores = {}
    # This function is long and unchanged, so it's omitted here for brevity
    # The actual code block below will contain the full function.
    rsi_series = calculate_rsi(df)
    if len(rsi_series) > 1 and pd.notna(rsi_series.iloc[-1]):
        rsi = rsi_series.iloc[-1]; prev_rsi = rsi_series.iloc[-2]
        if rsi > 60 and prev_rsi <= 60: scores['RSI'] = 2.0
        elif rsi > 50 and prev_rsi <= 50: scores['RSI'] = 1.0
        elif rsi < 40 and prev_rsi >= 40: scores['RSI'] = -2.0
        elif rsi < 50 and prev_rsi >= 50: scores['RSI'] = -1.0
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
    ma_short, ma_long = calculate_moving_averages(df)
    if len(ma_short) and len(ma_long) and pd.notna(ma_short.iloc[-1]) and pd.notna(ma_long.iloc[-1]):
        scores['MA'] = 1.0 if ma_short.iloc[-1] > ma_long.iloc[-1] else -1.0
    else: scores['MA'] = 0.0
    adx, pdi, ndi = calculate_adx(df)
    if len(adx) > 4 and pd.notna(adx.iloc[-1]):
        is_rising = adx.iloc[-1] > adx.iloc[-3]
        just_crossed = adx.iloc[-1] > 22 and adx.iloc[-2] <= 22
        if (adx.iloc[-1] > 22 and is_rising) or just_crossed:
            mul = 2.0 if just_crossed else 1.0
            scores['ADX'] = (1.5 * mul) if pdi.iloc[-1] > ndi.iloc[-1] else (-1.5 * mul)
    else: scores['ADX'] = 0.0
    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(df)
    if len(bb_middle) and pd.notna(bb_middle.iloc[-1]):
        scores['Bollinger'] = 0.5 if df['Close'].iloc[-1] > bb_middle.iloc[-1] else -0.5
    else: scores['Bollinger'] = 0.0
    roc = calculate_roc(df).iloc[-1] if len(df) > 13 else np.nan
    scores['ROC'] = 1.0 if pd.notna(roc) and roc > 0 else (-1.0 if pd.notna(roc) and roc < 0 else 0.0)
    obv_line = calculate_obv(df)
    if len(obv_line) >= 2: scores['OBV'] = 1.0 if obv_line.iloc[-1] > obv_line.iloc[-2] else -1.0
    else: scores['OBV'] = 0.0
    cci_val = calculate_cci(df).iloc[-1] if len(df) > 20 else np.nan
    if pd.notna(cci_val):
        if cci_val > 100: scores['CCI'] = 1.5
        elif cci_val > 0: scores['CCI'] = 1.0
        elif cci_val < -100: scores['CCI'] = -1.5
        else: scores['CCI'] = -1.0
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
    if len(atr_val) >= 6:
        atr_rising_sharply = (atr_val.iloc[-1] / atr_val.iloc[-5]) > 1.1
        price_up = df["Close"].iloc[-1] > df["Close"].iloc[-5]
        if atr_rising_sharply: scores["ATR"] = 1.5 if price_up else -1.5
    else: scores["ATR"] = 0.0
    zscore = volume_surge(df, lookback=20)
    if len(zscore) and pd.notna(zscore.iloc[-1]) and len(df) >= 2:
        price_up_last = df["Close"].iloc[-1] > df["Close"].iloc[-2]
        if zscore.iloc[-1] >= 2.0: scores["VolumeSurge"] = 1.5 if price_up_last else 0.0
        elif zscore.iloc[-1] <= -2.0: scores["VolumeSurge"] = -1.5 if not price_up_last else 0.0
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
    cmf20 = cmf(df, period=20)
    if len(cmf20) and pd.notna(cmf20.iloc[-1]):
        val = cmf20.iloc[-1]
        if val > 0.1: scores["CMF"] = 1.5
        elif val < -0.1: scores["CMF"] = -1.5
    else: scores["CMF"] = 0.0
    adl_line = adl(df)
    if len(adl_line) >= 6: scores["ADL"] = 1.2 if adl_line.iloc[-1] > adl_line.iloc[-5] else -1.2
    else: scores["ADL"] = 0.0
    rv = relative_volume(df, lookback=50)
    if len(rv) and pd.notna(rv.iloc[-1]):
        if rv.iloc[-1] >= 2.0: scores["RelVol"] = 1.0
        elif rv.iloc[-1] <= 0.5: scores["RelVol"] = -0.5
    else: scores["RelVol"] = 0.0
    vd = vwap_distance(df, period=None)
    if len(vd) and pd.notna(vd.iloc[-1]):
        d = vd.iloc[-1]
        if d > 0.002: scores["VWAPRegime"] = 1.3
        elif d < -0.002: scores["VWAPRegime"] = -1.3
    else: scores["VWAPRegime"] = 0.0
    obv_line2 = calculate_obv(df)
    if len(obv_line2) >= 6:
        obv_up = obv_line2.iloc[-1] > obv_line2.iloc[-5]
        price_up = df["Close"].iloc[-1] > df["Close"].iloc[-5]
        if obv_up and price_up: scores["OBVConfirm"] = 1.0
        elif (not obv_up) and (not price_up): scores["OBVConfirm"] = -1.0
    else: scores["OBVConfirm"] = 0.0
    for k in INDICATOR_WEIGHTS.keys(): scores.setdefault(k, 0.0)
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
            max_abs_score = max(abs(s) for s in indicator_scores.values() if s != 0) if any(indicator_scores.values()) else 1.0
            max_possible += max(abs(score), max_abs_score) * tf_weight * ind_weight
    if max_possible == 0: return 'Neutral', 0.0
    normalized = (final_score / max_possible) * 100.0
    if normalized >= 65: return 'Very Strong Buy', normalized
    elif normalized >= 25: return 'Strong Buy', normalized
    elif normalized <= -65: return 'Very Strong Sell', normalized
    elif normalized <= -25: return 'Strong Sell', normalized
    else: return 'Neutral', normalized
def infer_institutional_flow(tf_data):
    dfs = [tf_data.get(t) for t in (5, 15, 30) if tf_data.get(t) is not None and len(tf_data.get(t)) >= 50]
    if not dfs: return "Unknown"
    votes = 0
    for df in dfs:
        cmf20 = cmf(df, 20); vd = vwap_distance(df, None); rv = relative_volume(df, 50)
        adx_val, pdi, ndi = calculate_adx(df); obv_line = calculate_obv(df)
        ok = lambda s: len(s) and pd.notna(s.iloc[-1])
        c_cmf = ok(cmf20) and cmf20.iloc[-1]; c_vd = ok(vd) and vd.iloc[-1]; c_rv = ok(rv) and rv.iloc[-1]
        c_adx = ok(adx_val) and adx_val.iloc[-1]
        c_obv = len(obv_line) >= 2 and (obv_line.iloc[-1] > obv_line.iloc[-2])
        buy_cond = c_cmf > 0.1 and c_vd > 0.0 and c_rv >= 1.5 and c_adx > 20 and c_obv
        sell_cond = c_cmf < -0.1 and c_vd < 0.0 and c_rv >= 1.5 and c_adx > 20 and not c_obv
        if buy_cond: votes += 1
        if sell_cond: votes -= 1
    if votes >= 2: return "Institutional Accumulation"
    if votes <= -2: return "Institutional Distribution"
    return "Mixed/Unclear"

# ---------- Fetch + normalize (Updated for backtesting) ----------
def normalize_hist_df(df, symbol):
    if df is None or df.empty:
        return None
    try:
        out = df.copy()
        # Create a mapping from various possible column names (lowercase) to a standard format
        rename_map = {
            'timestamp': 'Date', 'time': 'Date', 'datetime': 'Date', 'date': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume', 'vol': 'Volume'
        }
        # Rename columns by looking up their lowercase version in the map
        out.rename(columns=lambda c: rename_map.get(str(c).lower(), str(c).capitalize()), inplace=True)

        if "Date" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
            out["Date"] = out.index

        if "Date" not in out.columns:
            return None # Cannot proceed without a date column

        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"])

        if out["Date"].dt.tz is None:
            out["Date"] = out["Date"].dt.tz_localize(IST, ambiguous='infer')
        else:
            out["Date"] = out["Date"].dt.tz_convert(IST)

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
            else:
                out[c] = 0 # Assign default value if column is missing

        out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_values("Date").set_index("Date")
        return out if len(out) >= 50 else None
    except Exception as e:
        logger.error(f"Normalize error for {symbol}: {e}")
        return None
        
def pick_session(symbol_orig, timeframe_minutes):
    return (hash(symbol_orig) ^ timeframe_minutes) % len(tdhist_pool)

def fetch_one(symbol_orig, timeframe_minutes, limiter, hist, end_ts=None):
    td_symbol = symbol_orig.replace('-EQ', '')
    bar_size = BAR_SIZE_MAP.get(timeframe_minutes)
    duration = DURATION_MAP.get(timeframe_minutes)
    if not bar_size or not duration:
        return symbol_orig, timeframe_minutes, None
    try:
        limiter.acquire()
        # Add end_date parameter for backtesting
        kwargs = {'duration': duration, 'bar_size': bar_size}
        if end_ts:
            kwargs['end_date'] = end_ts
        df_raw = hist.get_historic_data(td_symbol, **kwargs)
        df = normalize_hist_df(df_raw, td_symbol)
        global api_calls_done
        with api_calls_lock: api_calls_done += 1
        return symbol_orig, timeframe_minutes, df
    except Exception:
        return symbol_orig, timeframe_minutes, None

def prefetch_all(stocks, max_workers=MAX_WORKERS, end_ts=None):
    tfs = [5, 15, 30, 60, 1440]
    total_calls = len(stocks) * len(tfs)
    stock_multi_data = defaultdict(dict)
    global api_calls_done
    with api_calls_lock: api_calls_done = 0

    # Hide progress bar during live run to avoid clutter
    pbar_desc = f"Prefetching data as of {end_ts.strftime('%H:%M')}" if end_ts else "Prefetching Data"
    disable_pbar = not bool(end_ts)

    with tqdm(total=total_calls, desc=pbar_desc, ncols=100, disable=disable_pbar) as api_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_one, s, tf, sess_limiters[(hash(s) ^ tf) % len(tdhist_pool)], tdhist_pool[(hash(s) ^ tf) % len(tdhist_pool)], end_ts) for s in stocks for tf in tfs}
            for fut in as_completed(futures):
                symbol_orig, tf, df = fut.result()
                if df is not None: stock_multi_data[symbol_orig][tf] = df
                api_bar.update(1)
    return {s: d for s, d in stock_multi_data.items() if len(d) >= 2}

# ---------- Core Scanning Logic (Refactored with Debugging) ----------
def process_scan_at_timestamp(now_ist, stocks, previous_scores, output_filename, is_first_run):
    logger.info(f"[{now_ist.strftime('%H:%M:%S')}] Starting scan...")
    stock_multi_data = prefetch_all(stocks, max_workers=MAX_WORKERS, end_ts=now_ist)
    
    logger.info("Filtering for liquid, option-friendly stocks...")
    eligible_stocks = {}
    
    # --- Start Debug Block ---
    if is_first_run: logger.info("--- STARTING FILTER DEBUG ---")
    total_checked = 0
    passed_price_filter = 0
    passed_turnover_filter = 0
    # --- End Debug Block ---

    for symbol, timeframe_data in stock_multi_data.items():
        daily_df = timeframe_data.get(1440)
        if daily_df is None or len(daily_df) < 21:
            continue
        
        total_checked += 1
        try:
            last_close = daily_df['Close'].iloc[-1]
            if last_close < MIN_STOCK_PRICE:
                if is_first_run: logger.debug(f"FILTERED (PRICE): {symbol:<15} | Price: {last_close:<8.2f} < {MIN_STOCK_PRICE}")
                continue
            
            passed_price_filter += 1
            
            daily_df['Turnover'] = daily_df['Close'] * daily_df['Volume']
            avg_turnover = daily_df['Turnover'].rolling(20).mean().iloc[-1]
            
            if avg_turnover < MIN_AVG_DAILY_TURNOVER:
                if is_first_run: logger.debug(f"FILTERED (TURNOVER): {symbol:<15} | Avg Turnover: {avg_turnover/1_00_00_000:<8.2f} Cr < {MIN_AVG_DAILY_TURNOVER/1_00_00_000:.2f} Cr")
                continue

            passed_turnover_filter += 1
            eligible_stocks[symbol] = timeframe_data
        except (IndexError, KeyError):
            continue

    # --- Start Debug Block ---
    if is_first_run:
        logger.info("--- FILTER DEBUG SUMMARY ---")
        logger.info(f"Total stocks with daily data: {total_checked}")
        logger.info(f"Passed price filter (> {MIN_STOCK_PRICE}): {passed_price_filter}")
        logger.info(f"Passed turnover filter (> {MIN_AVG_DAILY_TURNOVER/1_00_00_000:.2f} Cr): {passed_turnover_filter}")
        logger.info(f"Final eligible stocks: {len(eligible_stocks)}")
        logger.info("--- ENDING FILTER DEBUG ---")
    # --- End Debug Block ---

    logger.info(f"Found {len(eligible_stocks)} eligible stocks. Analyzing signals...")

    signals_this_scan = []
    current_scores = {}
    for symbol, timeframe_data in eligible_stocks.items():
        clean_symbol = symbol.replace('-EQ', '')
        filtered_timeframes = {tf: df[df.index <= now_ist] for tf, df in timeframe_data.items() if df is not None and not df.empty and len(df[df.index <= now_ist]) >= 50}
        if len(filtered_timeframes) < 2: continue

        signal, score = analyze_signals(filtered_timeframes)
        current_scores[clean_symbol] = score
        if 'Strong' in signal:
            change = 'NA' if clean_symbol not in previous_scores else score - previous_scores.get(clean_symbol, 0.0)
            signals_this_scan.append({
                'symbol': clean_symbol, 'signal': signal, 'score': score,
                'trend': 'bullish' if 'Buy' in signal else 'bearish', 'change': change,
                'flow': infer_institutional_flow(filtered_timeframes),
                'volatility_pct': get_volatility_metric(filtered_timeframes.get(60, filtered_timeframes.get(30)))
            })

    top_bullish = sorted([r for r in signals_this_scan if 'Buy' in r['signal']], key=lambda x: x['score'], reverse=True)[:20]
    top_bearish = sorted([r for r in signals_this_scan if 'Sell' in r['signal']], key=lambda x: x['score'])[:20]

    width = 130
    header = f"| OPTION BUYER SCANNER | SIGNALS AT {now_ist.strftime('%Y-%m-%d %H:%M')} IST"
    bullish_header = f"| {'Top 20 Bullish Breakouts':<{width-4}} |"
    bearish_header = f"| {'Top 20 Bearish Breakdowns':<{width-4}} |"
    col_header = f"| {'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'ATR %':>7} | {'Change':>10} | {'Trend':<10} | {'Flow':<26} | {'Action':<19} |"
    
    output_str = "\n" + "="*width + "\n" + header.center(width+8) + " |\n" + "="*width + "\n"
    output_str += bullish_header + "\n" + "-"*width + "\n"
    if not top_bullish: output_str += "| None".ljust(width-1) + " |\n"
    else:
        output_str += col_header + "\n" + "-"*width + "\n"
        for r in top_bullish:
            change_str = "NA" if not isinstance(r['change'], (int, float, np.floating)) else f"{'+' if r['change'] > 0 else ''}{r['change']:.2f}"
            action = "Consider Call"
            output_str += f"| {r['symbol']:<15} | {r['signal']:<18} | {r['score']:>7.2f} | {r['volatility_pct']:>6.2f}% | {change_str:>10} | {r['trend']:<10} | {r.get('flow','Unknown'):<26} | {action:<19} |\n"
    
    output_str += "-"*width + "\n" + bearish_header + "\n" + "-"*width + "\n"
    if not top_bearish: output_str += "| None".ljust(width-1) + " |\n"
    else:
        output_str += col_header + "\n" + "-"*width + "\n"
        for r in top_bearish:
            change_str = "NA" if not isinstance(r['change'], (int, float, np.floating)) else f"{'+' if r['change'] > 0 else ''}{r['change']:.2f}"
            action = "Consider Put"
            output_str += f"| {r['symbol']:<15} | {r['signal']:<18} | {r['score']:>7.2f} | {r['volatility_pct']:>6.2f}% | {change_str:>10} | {r['trend']:<10} | {r.get('flow','Unknown'):<26} | {action:<19} |\n"
    output_str += "="*width + "\n"
    print(output_str)

    with open(output_filename, "a", encoding="utf-8") as f: f.write(output_str.replace("|", " ").replace("-", "─").replace("=", "═"))

    return {**previous_scores, **current_scores}

# ---------- Main Execution Loops ----------
def run_live_5min(stocks):
    first_run_dt = today_ist_dt(FIRST_RUN_AT)
    now = datetime.now(IST)
    if now < first_run_dt:
        logger.info(f"Waiting until {FIRST_RUN_AT}:00 IST for first 5-min close...")
        sleep_until(first_run_dt + timedelta(seconds=SETTLE_DELAY_SECONDS))

    previous_scores = {}
    output_filename = datetime.now(IST).strftime("%Y-%m-%d") + "_options_scan_results.txt"
    logger.info(f"Live mode started. Logging results to {output_filename}")
    is_first = True
    while True:
        now_ist = datetime.now(IST)
        if now_ist.time() > datetime.strptime(MARKET_END, "%H:%M").time():
            logger.info("Market closed. Exiting.")
            break
        
        previous_scores = process_scan_at_timestamp(now_ist, stocks, previous_scores, output_filename, is_first)
        is_first = False
        
        nxt = next_5min_boundary_ist(datetime.now(IST))
        logger.info(f"Scan complete. Sleeping until next cycle at {nxt.strftime('%H:%M:%S')}")
        sleep_until(nxt + timedelta(seconds=SETTLE_DELAY_SECONDS))

def run_backtest(stocks, asof_date_str):
    logger.info(f"Starting backtest for date: {asof_date_str}")
    output_filename = f"{asof_date_str}_options_scan_results.txt"
    if os.path.exists(output_filename): os.remove(output_filename) # Clear old results
    logger.info(f"Backtest mode started. Logging results to {output_filename}")

    start_ts = pd.Timestamp(f'{asof_date_str} {FIRST_RUN_AT}:00', tz=IST)
    end_ts = pd.Timestamp(f'{asof_date_str} {MARKET_END}:00', tz=IST)
    scan_times = pd.date_range(start=start_ts, end=end_ts, freq='5min')
    
    previous_scores = {}
    is_first = True
    for timestamp in tqdm(scan_times, desc=f"Backtesting {asof_date_str}"):
        previous_scores = process_scan_at_timestamp(timestamp, stocks, previous_scores, output_filename, is_first)
        is_first = False
    logger.info(f"Backtest for {asof_date_str} complete.")

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Option Buyer Stock Scanner with Live and Backtest modes.")
    parser.add_argument('--asof', type=str, help='Run a backtest for a specific date (YYYY-MM-DD format).')
    args = parser.parse_args()

    try:
        with open(SHARES_FILE, 'r') as f:
            stock_universe = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(stock_universe)} symbols from {SHARES_FILE}")

        if args.asof:
            try:
                datetime.strptime(args.asof, '%Y-%m-%d')
                run_backtest(stock_universe, args.asof)
            except ValueError:
                raise SystemExit("Invalid date format for --asof. Please use YYYY-MM-DD.")
        else:
            run_live_5min(stock_universe)

    except (KeyboardInterrupt, SystemExit) as e:
        if str(e): print(f"\nERROR: {e}")
        else: print("\nScan interrupted by user. Shutting down.")
    finally:
        logger.info("Disconnecting TrueData sessions...")
        for sess in tdhist_pool:
            try: sess.disconnect()
            except Exception: pass
        logger.info("Shutdown complete.")