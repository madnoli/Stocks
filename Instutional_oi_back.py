# live_scanner.py
# Live 5-minute close scanner for option buyers (TrueData REST)
# - Starts at 09:20 IST (first 5m close), then at every 5-minute boundary
# - Waits +5s after each boundary for bar settlement
# - Fetches 5/15/30/60/1D bars each run, computes signals with institutional flow tags
# - Prints Top 20 bullish/bearish and appends to daily file

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

from tqdm import tqdm
from truedata.history import TD_hist  # TrueData historical REST module [web:139]

# ================= Config =================
TDUSERNAME = os.getenv("TRUEDATA_USER", "")
TDPASSWORD = os.getenv("TRUEDATA_PASS", "")

MARKET_START = "09:15"         # IST
FIRST_RUN_AT = "09:20"         # IST (first 5-min bar close)
MARKET_END   = "15:30"         # IST
SETTLE_DELAY_SECONDS = 5       # wait after each 5-min boundary
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "48"))
TD_HIST_SESSIONS = int(os.getenv("TD_HIST_SESSIONS", "3"))
SHARES_FILE = os.getenv("SHARES_FILE", "shares.txt")

IST = pytz.timezone("Asia/Kolkata")

# Silence noisy third‑party loggers
for noisy in ("truedata", "truedata.history", "truedata_ws", "websocket", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

# =============== 5-minute boundary helpers ===============
def parse_hhmm(s: str):
    h, m = map(int, s.split(":"))
    return h, m

def today_ist_dt(hhmm: str) -> datetime:
    now = datetime.now(IST)
    h, m = parse_hhmm(hhmm)
    return now.replace(hour=h, minute=m, second=0, microsecond=0)

def next_5min_boundary_ist(now_ist: datetime) -> datetime:
    # Next HH:MM where MM % 5 == 0 and seconds=0
    minute = (now_ist.minute // 5) * 5
    boundary = now_ist.replace(minute=minute, second=0, microsecond=0)
    if boundary <= now_ist:
        boundary = boundary + timedelta(minutes=5)
    return boundary

def sleep_until(ts: datetime):
    # Sleep precisely until given IST-aware timestamp
    while True:
        now = datetime.now(IST)
        delta = (ts - now).total_seconds()
        if delta <= 0:
            break
        time.sleep(min(0.5, delta))  # small increments to reduce drift [web:148]

# =============== Token-bucket limiter ===============
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

# =============== Weights and TFs ===============
ENHANCED_INDICATOR_WEIGHTS = {
    "VolumeSurge": 2.0, "Momentum": 1.9, "ADX": 1.8, "VWAP": 1.7, "EMA": 1.7,
    "MACD": 1.5, "OBV": 1.5, "ATR": 1.4,
    "Bollinger": 1.3, "RSI": 1.2, "ROC": 1.1,
    "Stochastic": 1.0, "CCI": 1.0, "MA": 1.0, "WWL": 1.0,
}
INDICATOR_WEIGHTS = ENHANCED_INDICATOR_WEIGHTS | {
    "CMF": 1.6, "ADL": 1.4, "RelVol": 1.3, "VWAPRegime": 1.6, "OBVConfirm": 1.2
}
TIMEFRAME_WEIGHTS = {15: 3.0, 5: 2.5, 30: 2.0, 60: 1.5, "daily": 1.0}

BAR_SIZE_MAP = {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"}
DURATION_MAP = {5: "30 D", 15: "30 D", 30: "60 D", 60: "120 D", 1440: "365 D"}

# =============== TrueData sessions ===============
def authenticate_session():
    return TD_hist(TDUSERNAME, TDPASSWORD, log_level=logging.CRITICAL)  # [web:139]

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
    per_sess_rate = 10.0 / len(pool)  # ~10 rps shared
    limiters = [TokenBucketLimiter(rate_per_sec=per_sess_rate, bucket_size=10) for _ in pool]
    return pool, limiters

tdhist_pool, sess_limiters = build_sessions()
logger.info("TrueData login successful.")

# =============== Indicators ===============
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def vwap(df, period=None):
    price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = price * df["Volume"]
    if period:
        pv_sum = pv.rolling(period).sum(); vol_sum = df["Volume"].rolling(period).sum()
    else:
        pv_sum = pv.cumsum(); vol_sum = df["Volume"].cumsum()
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
    if len(df) < period:
        return pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64')
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

# Institutional utilities
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

# =============== Scoring and labels ===============
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
    else: scores['ADX'] = 0.0

    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(df)
    if len(bb_middle) and pd.notna(bb_middle.iloc[-1]):
        close = df['Close'].iloc[-1]
        scores['Bollinger'] = 0.5 if close > bb_middle.iloc[-1] else -0.5
    else: scores['Bollinger'] = 0.0

    roc = calculate_roc(df).iloc[-1] if len(df) else np.nan
    scores['ROC'] = 1.0 if pd.notna(roc) and roc > 0 else (-1.0 if pd.notna(roc) else 0.0)

    obv_line = calculate_obv(df)
    if len(obv_line) >= 2 and pd.notna(obv_line.iloc[-1]) and pd.notna(obv_line.iloc[-2]):
        scores['OBV'] = 1.0 if obv_line.iloc[-1] > obv_line.iloc[-2] else -1.0
    else: scores['OBV'] = 0.0

    cci_val = calculate_cci(df).iloc[-1] if len(df) else np.nan
    if pd.notna(cci_val):
        if cci_val > 100: scores['CCI'] = 1.5
        elif cci_val > 0: scores['CCI'] = 1.0
        elif cci_val < -100: scores['CCI'] = -1.5
        elif cci_val < 0: scores['CCI'] = -1.0
        else: scores['CCI'] = 0.0
    else: scores['CCI'] = 0.0

    ema_fast = ema(df["Close"], 20); ema_slow = ema(df["Close"], 50)
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

    # Institutional proxies
    cmf20 = cmf(df, period=20)
    if len(cmf20) and pd.notna(cmf20.iloc[-1]):
        val = cmf20.iloc[-1]
        if val > 0.1: scores["CMF"] = 1.5
        elif val < -0.1: scores["CMF"] = -1.5
        else: scores["CMF"] = 0.0
    else: scores["CMF"] = 0.0

    adl_line = adl(df)
    if len(adl_line) >= 6 and pd.notna(adl_line.iloc[-1]) and pd.notna(adl_line.iloc[-5]):
        slope = adl_line.iloc[-1] - adl_line.iloc[-5]
        scores["ADL"] = 1.2 if slope > 0 else -1.2
    else: scores["ADL"] = 0.0

    rv = relative_volume(df, lookback=50)
    if len(rv) and pd.notna(rv.iloc[-1]):
        if rv.iloc[-1] >= 2.0: scores["RelVol"] = 1.0
        elif rv.iloc[-1] <= 0.5: scores["RelVol"] = -0.5
        else: scores["RelVol"] = 0.0
    else: scores["RelVol"] = 0.0

    vd = vwap_distance(df, period=None)
    if len(vd) and pd.notna(vd.iloc[-1]):
        d = vd.iloc[-1]
        if d > 0.002: scores["VWAPRegime"] = 1.3
        elif d < -0.002: scores["VWAPRegime"] = -1.3
        else: scores["VWAPRegime"] = 0.0
    else: scores["VWAPRegime"] = 0.0

    obv_line2 = calculate_obv(df)
    if len(obv_line2) >= 6 and pd.notna(obv_line2.iloc[-1]) and pd.notna(obv_line2.iloc[-5]):
        obv_up = obv_line2.iloc[-1] > obv_line2.iloc[-5]
        price_up = df["Close"].iloc[-1] > df["Close"].iloc[-5]
        if obv_up and price_up: scores["OBVConfirm"] = 1.0
        elif (not obv_up) and (not price_up): scores["OBVConfirm"] = -1.0
        else: scores["OBVConfirm"] = 0.0
    else: scores["OBVConfirm"] = 0.0

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

def infer_institutional_flow(tf_data):
    dfs = [tf_data.get(t) for t in (5, 15, 30) if tf_data.get(t) is not None]
    if not dfs:
        return "Unknown"
    votes = 0
    for df in dfs:
        if len(df) < 50:
            continue
        cmf20 = cmf(df, 20)
        vd = vwap_distance(df, None)
        rv = relative_volume(df, 50)
        adx_val, pdi, ndi = calculate_adx(df)
        obv_line = calculate_obv(df)
        ok = lambda s: (len(s) and pd.notna(s.iloc[-1]))
        c_cmf = ok(cmf20) and cmf20.iloc[-1]
        c_vd  = ok(vd) and vd.iloc[-1]
        c_rv  = ok(rv) and rv.iloc[-1]
        c_adx = (len(adx_val) and pd.notna(adx_val.iloc[-1])) and adx_val.iloc[-1]
        c_obv = (len(obv_line)>=2 and pd.notna(obv_line.iloc[-1]) and pd.notna(obv_line.iloc[-2]) and (obv_line.iloc[-1] > obv_line.iloc[-2]))
        buy_cond  = (c_cmf is not False and c_cmf > 0.1) and (c_vd is not False and c_vd > 0.0) and (c_rv is not False and c_rv >= 1.5) and (c_adx is not False and c_adx > 20) and c_obv
        sell_cond = (c_cmf is not False and c_cmf < -0.1) and (c_vd is not False and c_vd < 0.0) and (c_rv is not False and c_rv >= 1.5) and (c_adx is not False and c_adx > 20) and (not c_obv)
        if buy_cond: votes += 1
        if sell_cond: votes -= 1
    if votes >= 2: return "Institutional Accumulation"
    if votes <= -2: return "Institutional Distribution"
    return "Mixed/Unclear"

# =============== Fetch + normalize ===============
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
        df_raw = hist.get_historic_data(td_symbol, duration=duration, bar_size=bar_size)  # [web:139][web:152]
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

def prefetch_all(stocks, max_workers=MAX_WORKERS):
    tfs = [5, 15, 30, 60, 1440]
    total_calls = len(stocks) * len(tfs)
    stock_multi_data = defaultdict(dict)

    global api_calls_done
    with api_calls_lock:
        api_calls_done = 0

    with tqdm(total=total_calls, desc="Prefetching Data", ncols=100) as api_bar:
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
    return {s: d for s, d in stock_multi_data.items() if len(d) >= 2}

# =============== Live 5-minute loop ===============
def run_live_5min():
    # Load universe
    try:
        with open(SHARES_FILE, 'r') as f:
            stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(stocks)} symbols from {SHARES_FILE}")
    except Exception as e:
        raise SystemExit(f"Could not read {SHARES_FILE}: {e}")

    # Wait until first 5-min close: 09:20 IST, then +5s settle
    first_run = today_ist_dt(FIRST_RUN_AT)
    now = datetime.now(IST)
    if now < first_run:
        logger.info(f"Waiting until {FIRST_RUN_AT}:00 IST for first 5-min close…")
        sleep_until(first_run)
    sleep_until(first_run + timedelta(seconds=SETTLE_DELAY_SECONDS))

    previous_scores = {}
    output_filename = datetime.now(IST).strftime("%Y-%m-%d") + "_options_scan_results.txt"

    while True:
        now_ist = datetime.now(IST)

        # Stop after session end (grace 1 minute), then sleep until next day 09:20:05
        end_h, end_m = parse_hhmm(MARKET_END)
        session_end = now_ist.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
        if now_ist > session_end + timedelta(minutes=1):
            logger.info("Market closed. Sleeping until next session.")
            tomorrow = (now_ist + timedelta(days=1)).astimezone(IST)
            next_first = tomorrow.replace(hour=int(FIRST_RUN_AT.split(':')[0]),
                                          minute=int(FIRST_RUN_AT.split(':')[1]),
                                          second=0, microsecond=0)
            sleep_until(next_first + timedelta(seconds=SETTLE_DELAY_SECONDS))
            continue

        # Refresh data windows (historical REST on each bar close) [web:108]
        logger.info(f"[{now_ist.strftime('%H:%M:%S')}] Refreshing data for {len(stocks)} stocks…")
        stock_multi_data = prefetch_all(stocks, max_workers=MAX_WORKERS)
        logger.info("Data refresh complete. Analyzing signals…")

        time_point_aware = now_ist.replace(second=0, microsecond=0)
        signals_this_scan = []
        current_scores = {}

        for symbol, timeframe_data in stock_multi_data.items():
            clean_symbol = symbol.replace('-EQ', '')
            filtered_timeframes = {}
            for tf, df in timeframe_data.items():
                if df is None or df.empty:
                    continue
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
                flow_tag = infer_institutional_flow(filtered_timeframes)
                signals_this_scan.append({
                    'symbol': clean_symbol, 'signal': signal, 'score': score,
                    'trend': direction, 'change': change, 'flow': flow_tag
                })

        # Rank and select top 20 bullish/bearish
        signals_this_scan.sort(key=lambda x: x['score'], reverse=True)
        top_bullish = [r for r in signals_this_scan if 'Buy' in r['signal']][:20]
        bearish_sorted = sorted([r for r in signals_this_scan if 'Sell' in r['signal']], key=lambda x: x['score'])
        top_bearish = bearish_sorted[:20]

        # Console output
        width = 120
        print("\n" + "="*width)
        hdr = f"| OPTION BUYER SCANNER | SIGNALS AT {now_ist.strftime('%Y-%m-%d %H:%M')} IST"
        print(hdr.center(width+8) + " |")
        print("="*width)

        print(f"| {'Top 20 Bullish Breakouts':<{width-4}} |")
        print("-"*width)
        if not top_bullish:
            print("| None".ljust(width-1) + " |")
        else:
            print(f"| {'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Flow':<26} | {'Action':<19} |")
            print("-"*width)
            for result in top_bullish:
                signal_text, change_val = result['signal'], result['change']
                if isinstance(change_val, (int, float, np.floating)):
                    sign = '+' if change_val > 0 else ''
                    change_str = f"{sign}{change_val:>.2f}"
                else:
                    change_str = "NA"
                action = "Consider Call" if 'Buy' in signal_text else "Consider Put"
                print(f"| {result['symbol']:<15} | {signal_text:<18} | {result['score']:>7.2f} | {change_str:>10} | {result['trend']:<10} | {result.get('flow','Unknown'):<26} | {action:<19} |")

        print("-"*width)
        print(f"| {'Top 20 Bearish Breakdowns':<{width-4}} |")
        print("-"*width)
        if not top_bearish:
            print("| None".ljust(width-1) + " |")
        else:
            print(f"| {'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Flow':<26} | {'Action':<19} |")
            print("-"*width)
            for result in top_bearish:
                signal_text, change_val = result['signal'], result['change']
                if isinstance(change_val, (int, float, np.floating)):
                    sign = '+' if change_val > 0 else ''
                    change_str = f"{sign}{change_val:>.2f}"
                else:
                    change_str = "NA"
                action = "Consider Call" if 'Buy' in signal_text else "Consider Put"
                print(f"| {result['symbol']:<15} | {signal_text:<18} | {result['score']:>7.2f} | {change_str:>10} | {result['trend']:<10} | {result.get('flow','Unknown'):<26} | {action:<19} |")
        print("="*width)

        # Append to file
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(f"===== Scan Time: {now_ist.strftime('%Y-%m-%d %H:%M')} =====\n\n")
            f.write("Top 20 Bullish (Momentum Breakouts)\n")
            f.write(f"{'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Flow':<26} | {'Action':<19}\n")
            f.write("-"*120 + "\n")
            if not top_bullish:
                f.write("No strong bullish signals found.\n")
            for r in top_bullish:
                change_val = r['change']
                if isinstance(change_val, (int, float, np.floating)):
                    sign = '+' if change_val > 0 else ''
                    change_str = f"{sign}{change_val:>.2f}"
                else:
                    change_str = "NA"
                action = "Consider Long" if 'Buy' in r['signal'] else "Consider Short"
                f.write(f"{r['symbol']:<15} | {r['signal']:<18} | {r['score']:>7.2f} | {change_str:>10} | {r['trend']:<10} | {r.get('flow','Unknown'):<26} | {action:<19}\n")
            f.write("\n")
            f.write("Top 20 Bearish (Momentum Breakdowns)\n")
            f.write(f"{'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Flow':<26} | {'Action':<19}\n")
            f.write("-"*120 + "\n")
            if not top_bearish:
                f.write("No strong bearish signals found.\n")
            for r in top_bearish:
                change_val = r['change']
                if isinstance(change_val, (int, float, np.floating)):
                    sign = '+' if change_val > 0 else ''
                    change_str = f"{sign}{change_val:>.2f}"
                else:
                    change_str = "NA"
                action = "Consider Long" if 'Buy' in r['signal'] else "Consider Short"
                f.write(f"{r['symbol']:<15} | {r['signal']:<18} | {r['score']:>7.2f} | {change_str:>10} | {r['trend']:<10} | {r.get('flow','Unknown'):<26} | {action:<19}\n")
            f.write("\n\n")

        previous_scores = {**previous_scores, **current_scores}

        # Sleep until the next 5-min boundary + settle delay
        nxt = next_5min_boundary_ist(datetime.now(IST))
        sleep_until(nxt + timedelta(seconds=SETTLE_DELAY_SECONDS))

# ================= Entrypoint =================
if __name__ == "__main__":
    try:
        run_live_5min()
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
