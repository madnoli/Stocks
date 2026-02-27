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
import argparse

from tqdm import tqdm
from truedata.history import TD_hist

# Rich for colored tables
from rich.console import Console
from rich.table import Table
from rich import box

# ======== Config ========
TDUSERNAME = os.getenv("TRUEDATA_USER", "tdwsp751")
TDPASSWORD = os.getenv("TRUEDATA_PASS", "raj@751")

MARKET_START = "09:15"   # IST
FIRST_RUN_AT = "09:20"   # IST; first 5-min close
MARKET_END   = "15:30"   # IST
SETTLE_DELAY_SECONDS = 5  # wait after bar close
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "48"))
TD_HIST_SESSIONS = int(os.getenv("TD_HIST_SESSIONS", "3"))

# Universe file
SHARES_FILE = os.getenv("SHARES_FILE", "shares.txt")

IST = pytz.timezone("Asia/Kolkata")

# Silence noisy third‑party loggers
for noisy in ("truedata", "truedata.history", "truedata_ws", "websocket", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

console = Console()

# For new-entrant highlighting across scans
last_bull_symbols = set()
last_bear_symbols = set()

# To compute Change across checkpoints (all symbols, not only top lists)
previous_scores = {}

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

# ---------- Weights ----------
ENHANCED_INDICATOR_WEIGHTS = {
    "VolumeSurge": 2.0, "Momentum": 1.9, "ADX": 1.8, "VWAP": 1.7, "EMA": 1.7,
    "MACD": 1.5, "OBV": 1.5, "ATR": 1.4,
    "Bollinger": 1.3, "RSI": 1.2, "ROC": 1.1,
    "Stochastic": 1.0, "CCI": 1.0, "MA": 1.0, "WWL": 1.0,
}
INDICATOR_WEIGHTS = ENHANCED_INDICATOR_WEIGHTS | {
    "CMF": 1.8, "ADL": 1.6, "RelVol": 1.5, "VWAPRegime": 1.7, "OBVConfirm": 1.2
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
    per_sess_rate = 10.0 / len(pool)  # target ~10 rps across pool
    limiters = [TokenBucketLimiter(rate_per_sec=per_sess_rate, bucket_size=10) for _ in pool]
    return pool, limiters

tdhist_pool, sess_limiters = build_sessions()
logger.info("TrueData login successful.")

# ---------- Indicators ----------
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

# ---- New helpers for institutional inference ----
def slope(series, lookback=10):
    if len(series) < lookback: return np.nan
    y = series.tail(lookback).values.astype(float)
    x = np.arange(len(y))
    x = (x - x.mean()) / (x.std() + 1e-9)
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return m

def vwap_full_session(df):
    # Robust session-anchored VWAP aligned to df index; avoids length mismatch on assignment
    if df is None or df.empty:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')

    dfx = df.sort_index()
    dfx = dfx[~dfx.index.duplicated(keep='last')]

    day = dfx.index[-1].date()
    dfd = dfx[dfx.index.date == day]
    if dfd.empty:
        vw = vwap(dfx, period=None)
        return vw.reindex_like(dfx).ffill().reindex(df.index, method='ffill')

    price = (dfd['High'] + dfd['Low'] + dfd['Close']) / 3.0
    pv = price * dfd['Volume'].clip(lower=0)
    vol_cum = dfd['Volume'].clip(lower=0).cumsum().replace(0, np.nan)
    sess_vwap = (pv.cumsum() / vol_cum).astype(float)

    out = pd.Series(index=dfx.index, dtype='float64')
    out.update(sess_vwap)  # label-aligned safe assignment
    out = out.ffill()
    out = out.reindex(df.index, method='ffill')
    return out

def adl_series(df):
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = mfm.fillna(0)
    mfv = mfm * df["Volume"]
    return mfv.cumsum()

# ---------- Scoring ----------
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

    cmf20 = cmf(df, period=20)
    if len(cmf20) and pd.notna(cmf20.iloc[-1]):
        val = cmf20.iloc[-1]
        if val > 0.1: scores["CMF"] = 1.5
        elif val < -0.1: scores["CMF"] = -1.5
        else: scores["CMF"] = 0.0
    else: scores["CMF"] = 0.0

    adl_line = adl(df)
    if len(adl_line) >= 6 and pd.notna(adl_line.iloc[-1]) and pd.notna(adl_line.iloc[-5]):
        slope5 = adl_line.iloc[-1] - adl_line.iloc[-5]
        scores["ADL"] = 1.2 if slope5 > 0 else -1.2
    else: scores["ADL"] = 0.0

    rv = relative_volume(df, lookback=50)
    if len(rv) and pd.notna(rv.iloc[-1]):
        if rv.iloc[-1] >= 1.5: scores["RelVol"] = 1.0
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

# ---- Institutional inference (rewritten) ----
def infer_institutional_flow(tf_data):
    frames = [tf_data.get(t) for t in (5, 15, 30) if tf_data.get(t) is not None and len(tf_data.get(t)) >= 60]
    if not frames:
        return "Unknown"
    votes = 0
    for df in frames:
        cmf20 = cmf(df, 20)
        adl_line = adl_series(df)
        adx_val, pdi, ndi = calculate_adx(df)
        rv50 = relative_volume(df, 50)
        vwap_day = vwap_full_session(df)
        vdist = ((df["Close"] - vwap_day) / vwap_day.replace(0, np.nan)).fillna(0)

        def last_ok(s): return (len(s) and pd.notna(s.iloc[-1]))
        cmf_last = cmf20.iloc[-1] if last_ok(cmf20) else np.nan
        cmf_slope_10 = slope(cmf20, 10)
        adl_slope_10 = slope(adl_line, 10)
        adx_last = adx_val.iloc[-1] if last_ok(adx_val) else np.nan
        p_over_n = (pdi.iloc[-1] > ndi.iloc[-1]) if (len(pdi) and len(ndi) and pd.notna(pdi.iloc[-1]) and pd.notna(ndi.iloc[-1])) else False
        rv_last = rv50.iloc[-1] if last_ok(rv50) else np.nan
        vdist_last = vdist.iloc[-1] if last_ok(vdist) else np.nan

        near_vwap_ok = pd.notna(vdist_last) and (abs(vdist_last) <= 0.01 or (abs(vdist_last) <= 0.02 and pd.notna(rv_last) and rv_last >= 2.0))
        strong_rvol = pd.notna(rv_last) and rv_last >= 1.5

        buy_cond = (
            pd.notna(vdist_last) and vdist_last > 0 and near_vwap_ok and
            pd.notna(cmf_last) and cmf_last > 0.1 and
            (not np.isnan(cmf_slope_10) and cmf_slope_10 > 0) and
            (not np.isnan(adl_slope_10) and adl_slope_10 > 0) and
            pd.notna(adx_last) and adx_last > 20 and p_over_n and
            strong_rvol
        )
        sell_cond = (
            pd.notna(vdist_last) and vdist_last < 0 and near_vwap_ok and
            pd.notna(cmf_last) and cmf_last < -0.1 and
            (not np.isnan(cmf_slope_10) and cmf_slope_10 < 0) and
            (not np.isnan(adl_slope_10) and adl_slope_10 < 0) and
            pd.notna(adx_last) and adx_last > 20 and (not p_over_n) and
            strong_rvol
        )

        if buy_cond: votes += 1
        if sell_cond: votes -= 1

    if votes >= 2: return "Institutional Accumulation"
    if votes <= -2: return "Institutional Distribution"
    return "Mixed/Unclear"

def institutional_flow_votes(tf_data):
    votes = []
    for t in (5, 15, 30):
        df = tf_data.get(t)
        if df is None or len(df) < 60:
            continue
        cmf20 = cmf(df, 20); adl_line = adl_series(df)
        adx_val, pdi, ndi = calculate_adx(df); rv50 = relative_volume(df, 50)
        vwap_day = vwap_full_session(df)
        vdist = ((df["Close"] - vwap_day) / vwap_day.replace(0, np.nan)).fillna(0)

        def ok_last(s): return (len(s) and pd.notna(s.iloc[-1]))
        cmf_last = cmf20.iloc[-1] if ok_last(cmf20) else np.nan
        adl_slope_10 = slope(adl_line, 10); cmf_slope_10 = slope(cmf20, 10)
        adx_last = adx_val.iloc[-1] if ok_last(adx_val) else np.nan
        p_over_n = (pdi.iloc[-1] > ndi.iloc[-1]) if (len(pdi) and len(ndi) and pd.notna(pdi.iloc[-1]) and pd.notna(ndi.iloc[-1])) else False
        rv_last = rv50.iloc[-1] if ok_last(rv50) else np.nan
        vdist_last = vdist.iloc[-1] if ok_last(vdist) else np.nan

        near_vwap_ok = pd.notna(vdist_last) and (abs(vdist_last) <= 0.01 or (abs(vdist_last) <= 0.02 and pd.notna(rv_last) and rv_last >= 2.0))
        strong_rvol = pd.notna(rv_last) and rv_last >= 1.5

        buy = (pd.notna(vdist_last) and vdist_last > 0 and near_vwap_ok and
               pd.notna(cmf_last) and cmf_last > 0.1 and (not np.isnan(cmf_slope_10) and cmf_slope_10 > 0) and
               (not np.isnan(adl_slope_10) and adl_slope_10 > 0) and
               pd.notna(adx_last) and adx_last > 20 and p_over_n and strong_rvol)
        sell = (pd.notna(vdist_last) and vdist_last < 0 and near_vwap_ok and
                pd.notna(cmf_last) and cmf_last < -0.1 and (not np.isnan(cmf_slope_10) and cmf_slope_10 < 0) and
                (not np.isnan(adl_slope_10) and adl_slope_10 < 0) and
                pd.notna(adx_last) and adx_last > 20 and (not p_over_n) and strong_rvol)
        votes.append((t, 1 if buy else (-1 if sell else 0)))
    return votes

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

    # Institutional additive score (explicit votes)
    votes = institutional_flow_votes(timeframe_dataframes)
    inst_score = 0.0
    for tf, v in votes:
        inst_score += v * TIMEFRAME_WEIGHTS.get(tf, 1.0) * 2.0
        max_possible += 2.0 * TIMEFRAME_WEIGHTS.get(tf, 1.0)
    final_score += inst_score

    if max_possible == 0: return 'Neutral', 0.0
    normalized = (final_score / max_possible) * 100.0
    if normalized >= 65: signal_text = 'Very Strong Buy'
    elif normalized >= 25: signal_text = 'Strong Buy'
    elif normalized <= -65: signal_text = 'Very Strong Sell'
    elif normalized <= -25: signal_text = 'Strong Sell'
    else: signal_text = 'Neutral'
    return signal_text, normalized

# ---------- Fetch + normalize ----------
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
        out = out[~out.index.duplicated(keep='last')]
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

# ---------- Backtest helpers ----------
def parse_asof(s: str):
    # Accept YYYY-MM-DD or YYYY-MM-DDTHH:MM
    if 'T' in s:
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M")
    else:
        dt = datetime.strptime(s, "%Y-%m-%d")
        h, m = parse_hhmm(MARKET_END)
        dt = dt.replace(hour=h, minute=m)
    return IST.localize(dt)

def day_checkpoints_ist(day_date: datetime):
    # 5-min checkpoints from first close (09:20) to 15:30 IST
    d = day_date.date()
    start_h, start_m = parse_hhmm("09:20")
    end_h, end_m = parse_hhmm(MARKET_END)
    start_dt = IST.localize(datetime(d.year, d.month, d.day, start_h, start_m))
    end_dt   = IST.localize(datetime(d.year, d.month, d.day, end_h, end_m))
    rng = pd.date_range(start=start_dt, end=end_dt, freq="5T", tz=IST, inclusive="both")
    return list(rng.to_pydatetime())

# ---------- Rendering with Rich ----------
def render_top_lists(now_ts, top_bullish, top_bearish):
    global last_bull_symbols, last_bear_symbols

    title = f"| OPTION BUYER SCANNER | SNAPSHOT AT {now_ts.strftime('%Y-%m-%d %H:%M')} IST"
    console.rule(title)

    bull_table = Table(title="Top 20 Bullish Breakouts", box=box.SIMPLE_HEAVY, header_style="white on dark_green", style="white on black")
    for col, style, justify in [
        ("Stock","cyan","left"), ("Signal","bright_white","left"), ("Score","yellow","right"),
        ("Change","magenta","right"), ("Trend","bright_white","left"),
        ("Flow","bright_white","left"), ("Action","bright_white","left")
    ]:
        bull_table.add_column(col, style=style, justify=justify)

    for r in top_bullish:
        sym = r['symbol']
        is_new = sym not in last_bull_symbols
        row_style = "black on green" if is_new else None
        ch = r['change']
        if isinstance(ch, (int, float, np.floating)):
            change_str = f"{ch:+.2f}"
        else:
            change_str = "NA"
        bull_table.add_row(sym, r['signal'], f"{r['score']:.2f}", change_str, r['trend'], r.get('flow','Unknown'), ("Consider Call" if 'Buy' in r['signal'] else "Consider Put"), style=row_style)

    console.print(bull_table)

    bear_table = Table(title="Top 20 Bearish Breakdowns", box=box.SIMPLE_HEAVY, header_style="white on dark_red", style="white on black")
    for col, style, justify in [
        ("Stock","cyan","left"), ("Signal","bright_white","left"), ("Score","yellow","right"),
        ("Change","magenta","right"), ("Trend","bright_white","left"),
        ("Flow","bright_white","left"), ("Action","bright_white","left")
    ]:
        bear_table.add_column(col, style=style, justify=justify)

    for r in top_bearish:
        sym = r['symbol']
        is_new = sym not in last_bear_symbols
        row_style = "white on red" if is_new else None
        ch = r['change']
        if isinstance(ch, (int, float, np.floating)):
            change_str = f"{ch:+.2f}"
        else:
            change_str = "NA"
        bear_table.add_row(sym, r['signal'], f"{r['score']:.2f}", change_str, r['trend'], r.get('flow','Unknown'), ("Consider Put" if 'Sell' in r['signal'] else "Consider Call"), style=row_style)

    console.print(bear_table)
    console.rule()

    # Update last sets
    last_bull_symbols = {r['symbol'] for r in top_bullish}
    last_bear_symbols = {r['symbol'] for r in top_bearish}

# ---------- Full-day backtest ----------
def run_backtest_day(day_str: str, stocks):
    day_date = datetime.strptime(day_str, "%Y-%m-%d")
    logger.info(f"[{day_str}] Backtest (full day) prefetch for {len(stocks)} symbols...")
    stock_multi_data = prefetch_all(stocks, max_workers=MAX_WORKERS)
    logger.info("Prefetch complete. Running 5-min checkpoints...")

    checkpoints = day_checkpoints_ist(day_date)
    output_filename = day_date.strftime("%Y-%m-%d") + "_options_scan_results.txt"
    try:
        if os.path.exists(output_filename):
            os.remove(output_filename)
    except Exception:
        pass

    global previous_scores, last_bull_symbols, last_bear_symbols
    previous_scores = {}
    last_bull_symbols = set()
    last_bear_symbols = set()

    for asof_ts in checkpoints:
        time_point_aware = asof_ts.replace(second=0, microsecond=0)
        signals_this_scan = []
        current_scores = {}

        for symbol, timeframe_data in stock_multi_data.items():
            clean_symbol = symbol.replace('-EQ', '')
            filtered_timeframes = {}
            for tf, df in timeframe_data.items():
                if df is None or df.empty:
                    continue
                df_clean = df.sort_index()
                df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
                df_slice = df_clean[df_clean.index <= time_point_aware]
                if not df_slice.empty and len(df_slice) >= 50:
                    filtered_timeframes[tf] = df_slice
            if len(filtered_timeframes) < 2:
                continue

            signal, score = analyze_signals(filtered_timeframes)
            current_scores[clean_symbol] = score  # track all symbols

            if 'Strong' in signal:
                prev = previous_scores.get(clean_symbol, 'NA')
                change_val = 'NA' if isinstance(prev, str) else (score - prev)
                direction = 'bullish' if 'Buy' in signal else 'bearish'
                flow_tag = infer_institutional_flow(filtered_timeframes)
                signals_this_scan.append({
                    'symbol': clean_symbol, 'signal': signal, 'score': score,
                    'trend': direction, 'change': change_val, 'flow': flow_tag
                })

        # After processing universe, roll previous_scores to current
        previous_scores = current_scores.copy()

        # Rank/format like live
        signals_this_scan.sort(key=lambda x: x['score'], reverse=True)
        top_bullish = [r for r in signals_this_scan if 'Buy' in r['signal']][:20]
        bearish_sorted = sorted([r for r in signals_this_scan if 'Sell' in r['signal']], key=lambda x: x['score'])
        top_bearish = bearish_sorted[:20]

        # Render colored tables
        render_top_lists(asof_ts, top_bullish, top_bearish)

        # Append to file each checkpoint (plain text)
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(f"===== Snapshot Time: {asof_ts.strftime('%Y-%m-%d %H:%M')} =====\n\n")
            f.write("Top 20 Bullish (Momentum Breakouts)\n")
            f.write(f"{'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Flow':<26} | {'Action':<19}\n")
            f.write("-"*120 + "\n")
            if not top_bullish:
                f.write("No strong bullish signals found.\n")
            for r in top_bullish:
                ch = r['change']
                change_str = ch if isinstance(ch, str) else f"{ch:+.2f}"
                action = "Consider Long" if 'Buy' in r['signal'] else "Consider Short"
                f.write(f"{r['symbol']:<15} | {r['signal']:<18} | {r['score']:>7.2f} | {change_str:>10} | {r['trend']:<10} | {r.get('flow','Unknown'):<26} | {action:<19}\n")
            f.write("\n")
            f.write("Top 20 Bearish (Momentum Breakdowns)\n")
            f.write(f"{'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Flow':<26} | {'Action':<19}\n")
            f.write("-"*120 + "\n")
            if not top_bearish:
                f.write("No strong bearish signals found.\n")
            for r in top_bearish:
                ch = r['change']
                change_str = ch if isinstance(ch, str) else f"{ch:+.2f}"
                action = "Consider Long" if 'Buy' in r['signal'] else "Consider Short"
                f.write(f"{r['symbol']:<15} | {r['signal']:<18} | {r['score']:>7.2f} | {change_str:>10} | {r['trend']:<10} | {r.get('flow','Unknown'):<26} | {action:<19}\n")
            f.write("\n\n")

# ---------- Single snapshot backtest ----------
def run_once_asof(asof_ts, stocks):
    logger.info(f"[{asof_ts.strftime('%Y-%m-%d %H:%M')}] Backtest snapshot: fetching data for {len(stocks)} symbols...")
    stock_multi_data = prefetch_all(stocks, max_workers=MAX_WORKERS)
    logger.info("Data fetch complete. Analyzing snapshot...")

    time_point_aware = asof_ts.replace(second=0, microsecond=0)
    signals_this_scan = []

    for symbol, timeframe_data in stock_multi_data.items():
        clean_symbol = symbol.replace('-EQ', '')
        filtered_timeframes = {}
        for tf, df in timeframe_data.items():
            if df is None or df.empty:
                continue
            df_clean = df.sort_index()
            df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
            df_slice = df_clean[df_clean.index <= time_point_aware]
            if not df_slice.empty and len(df_slice) >= 50:
                filtered_timeframes[tf] = df_slice
        if len(filtered_timeframes) < 2:
            continue

        signal, score = analyze_signals(filtered_timeframes)
        if 'Strong' in signal:
            direction = 'bullish' if 'Buy' in signal else 'bearish'
            flow_tag = infer_institutional_flow(filtered_timeframes)
            signals_this_scan.append({
                'symbol': clean_symbol, 'signal': signal, 'score': score,
                'trend': direction, 'change': 'NA', 'flow': flow_tag
            })

    signals_this_scan.sort(key=lambda x: x['score'], reverse=True)
    top_bullish = [r for r in signals_this_scan if 'Buy' in r['signal']][:20]
    bearish_sorted = sorted([r for r in signals_this_scan if 'Sell' in r['signal']], key=lambda x: x['score'])
    top_bearish = bearish_sorted[:20]

    render_top_lists(asof_ts, top_bullish, top_bearish)

    # Also write to dated file
    output_filename = asof_ts.strftime("%Y-%m-%d") + "_options_scan_results.txt"
    with open(output_filename, "a", encoding="utf-8") as f:
        f.write(f"===== Snapshot Time: {asof_ts.strftime('%Y-%m-%d %H:%M')} =====\n\n")
        f.write("Top 20 Bullish (Momentum Breakouts)\n")
        f.write(f"{'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Flow':<26} | {'Action':<19}\n")
        f.write("-"*120 + "\n")
        if not top_bullish:
            f.write("No strong bullish signals found.\n")
        for r in top_bullish:
            action = "Consider Long" if 'Buy' in r['signal'] else "Consider Short"
            f.write(f"{r['symbol']:<15} | {r['signal']:<18} | {r['score']:>7.2f} | {'NA':>10} | {r['trend']:<10} | {r.get('flow','Unknown'):<26} | {action:<19}\n")
        f.write("\n")
        f.write("Top 20 Bearish (Momentum Breakdowns)\n")
        f.write(f"{'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Flow':<26} | {'Action':<19}\n")
        f.write("-"*120 + "\n")
        if not top_bearish:
            f.write("No strong bearish signals found.\n")
        for r in top_bearish:
            action = "Consider Long" if 'Buy' in r['signal'] else "Consider Short"
            f.write(f"{r['symbol']:<15} | {r['signal']:<18} | {r['score']:>7.2f} | {'NA':>10} | {r['trend']:<10} | {r.get('flow','Unknown'):<26} | {action:<19}\n")
        f.write("\n\n")

# ---------- Live 5-min loop ----------
def run_live_5min():
    # Load universe
    try:
        with open(SHARES_FILE, 'r') as f:
            stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(stocks)} symbols from {SHARES_FILE}")
    except Exception as e:
        raise SystemExit(f"Could not read {SHARES_FILE}: {e}")

    # Initial wait until 09:20:05 IST
    first_run = today_ist_dt(FIRST_RUN_AT)
    now = datetime.now(IST)
    if now < first_run:
        logger.info(f"Waiting until {FIRST_RUN_AT}:00 IST for first 5-min close...")
        sleep_until(first_run)
    settle_ts = first_run + timedelta(seconds=SETTLE_DELAY_SECONDS)
    sleep_until(settle_ts)

    global previous_scores, last_bull_symbols, last_bear_symbols
    previous_scores = {}
    last_bull_symbols = set()
    last_bear_symbols = set()

    output_filename = datetime.now(IST).strftime("%Y-%m-%d") + "_options_scan_results.txt"

    while True:
        now_ist = datetime.now(IST)
        end_h, end_m = parse_hhmm(MARKET_END)
        session_end = now_ist.replace(hour=end_h, minute=end_m, second=0, microsecond=0)
        if now_ist > session_end + timedelta(minutes=1):
            logger.info("Market closed. Sleeping until next session.")
            tomorrow = (now_ist + timedelta(days=1)).astimezone(IST)
            next_first = tomorrow.replace(hour=int(FIRST_RUN_AT.split(":")[0]),
                                          minute=int(FIRST_RUN_AT.split(":")[1]),
                                          second=0, microsecond=0)
            sleep_until(next_first + timedelta(seconds=SETTLE_DELAY_SECONDS))
            continue

        # Refresh data windows
        logger.info(f"[{now_ist.strftime('%H:%M:%S')}] Refreshing data ...")
        stock_multi_data = prefetch_all(stocks, max_workers=MAX_WORKERS)
        logger.info("Data refresh complete. Analyzing signals...")

        time_point_aware = now_ist.replace(second=0, microsecond=0)
        signals_this_scan = []
        current_scores = {}

        for symbol, timeframe_data in stock_multi_data.items():
            clean_symbol = symbol.replace('-EQ', '')
            filtered_timeframes = {}
            for tf, df in timeframe_data.items():
                if df is None or df.empty:
                    continue
                df_clean = df.sort_index()
                df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
                df_slice = df_clean[df_clean.index <= time_point_aware]
                if not df_slice.empty and len(df_slice) >= 50:
                    filtered_timeframes[tf] = df_slice
            if len(filtered_timeframes) < 2:
                continue

            signal, score = analyze_signals(filtered_timeframes)
            current_scores[clean_symbol] = score  # track all symbols for Change

            if 'Strong' in signal:
                prev = previous_scores.get(clean_symbol, 'NA')
                change_val = 'NA' if isinstance(prev, str) else (score - prev)
                direction = 'bullish' if 'Buy' in signal else 'bearish'
                flow_tag = infer_institutional_flow(filtered_timeframes)
                signals_this_scan.append({
                    'symbol': clean_symbol, 'signal': signal, 'score': score,
                    'trend': direction, 'change': change_val, 'flow': flow_tag
                })

        # Roll previous_scores to current
        previous_scores = current_scores.copy()

        # Rank and select top 20 bullish/bearish
        signals_this_scan.sort(key=lambda x: x['score'], reverse=True)
        top_bullish = [r for r in signals_this_scan if 'Buy' in r['signal']][:20]
        bearish_sorted = sorted([r for r in signals_this_scan if 'Sell' in r['signal']], key=lambda x: x['score'])
        top_bearish = bearish_sorted[:20]

        # Render colored tables
        render_top_lists(now_ist, top_bullish, top_bearish)

        # Append to file
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(f"===== Scan Time: {now_ist.strftime('%Y-%m-%d %H:%M')} =====\n\n")
            f.write("Top 20 Bullish (Momentum Breakouts)\n")
            f.write(f"{'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Flow':<26} | {'Action':<19}\n")
            f.write("-"*120 + "\n")
            if not top_bullish:
                f.write("No strong bullish signals found.\n")
            for r in top_bullish:
                ch = r['change']
                change_str = ch if isinstance(ch, str) else f"{ch:+.2f}"
                action = "Consider Long" if 'Buy' in r['signal'] else "Consider Short"
                f.write(f"{r['symbol']:<15} | {r['signal']:<18} | {r['score']:>7.2f} | {change_str:>10} | {r['trend']:<10} | {r.get('flow','Unknown'):<26} | {action:<19}\n")
            f.write("\n")
            f.write("Top 20 Bearish (Momentum Breakdowns)\n")
            f.write(f"{'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Flow':<26} | {'Action':<19}\n")
            f.write("-"*120 + "\n")
            if not top_bearish:
                f.write("No strong bearish signals found.\n")
            for r in top_bearish:
                ch = r['change']
                change_str = ch if isinstance(ch, str) else f"{ch:+.2f}"
                action = "Consider Long" if 'Buy' in r['signal'] else "Consider Short"
                f.write(f"{r['symbol']:<15} | {r['signal']:<18} | {r['score']:>7.2f} | {change_str:>10} | {r['trend']:<10} | {r.get('flow','Unknown'):<26} | {action:<19}\n")
            f.write("\n\n")

        # Sleep until the next 5-min boundary + settle delay
        nxt = next_5min_boundary_ist(datetime.now(IST))
        sleep_until(nxt + timedelta(seconds=SETTLE_DELAY_SECONDS))

# ---------------- Main (with backtest modes) ----------------
def main():
    parser = argparse.ArgumentParser(description="Options buyer scanner with institutional flow, colored tables, and backtest modes")
    parser.add_argument("--asof", type=str, default=None, help="Snapshot as-of time, e.g. 2025-09-26 or 2025-09-26T09:50")
    parser.add_argument("--backtest-date", type=str, default=None, help="Full-day backtest at 5-min intervals, e.g. 2025-09-26")
    args = parser.parse_args()

    # Load universe
    try:
        with open(SHARES_FILE, 'r') as f:
            stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(stocks)} symbols from {SHARES_FILE}")
    except Exception as e:
        raise SystemExit(f"Could not read {SHARES_FILE}: {e}")

    if args.backtest_date:
        run_backtest_day(args.backtest_date, stocks)
    elif args.asof:
        asof_ts = parse_asof(args.asof)
        run_once_asof(asof_ts, stocks)
    else:
        run_live_5min()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScan interrupted by user. Shutting down.")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        raise
    finally:
        logger.info("Disconnecting TrueData sessions...")
        try:
            for sess in tdhist_pool:
                try:
                    sess.disconnect()
                except Exception:
                    pass
        except Exception:
            pass
        logger.info("Shutdown complete.")
