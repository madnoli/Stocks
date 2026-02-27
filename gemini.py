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
import csv
from retrying import retry

from tqdm import tqdm
from truedata.history import TD_hist

# Rich for colored tables
from rich.console import Console
from rich.table import Table
from rich import box

# ======== Config ========
CONFIG = {
    "TDUSERNAME": os.getenv("TRUEDATA_USER", "tdwsp751"),
    "TDPASSWORD": os.getenv("TRUEDATA_PASS", "raj@751"),

    # Market times (IST)
    "MARKET_START": "09:15",
    "FIRST_RUN_AT": "09:20",
    "MARKET_END": "15:30",
    "SETTLE_DELAY_SECONDS": 5,

    # Concurrency and rate
    "MAX_WORKERS": int(os.getenv("MAX_WORKERS", "64")),
    "TD_HIST_SESSIONS": int(os.getenv("TD_HIST_SESSIONS", "8")),
    "RATE_PER_SECOND_TOTAL": float(os.getenv("RATE_PER_SECOND_TOTAL", "20.0")),
    "BUCKET_SIZE": int(os.getenv("BUCKET_SIZE", "40")),
    "RETRY_ATTEMPTS": int(os.getenv("RETRY_ATTEMPTS", "3")),
    "RETRY_DELAY_MS": int(os.getenv("RETRY_DELAY_MS", "1000")),

    # Output and logging
    "SHARES_FILE": os.getenv("SHARES_FILE", "shares.txt"),
    "SHOW_PROGRESS": os.getenv("SHOW_PROGRESS", "false").lower() == "true",
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "ERROR"),
    "DATA_DIR": "backtest_data",

    # Market Context Filter
    "MARKET_INDEX": "NIFTY 50", # Change to "BANKNIFTY" or other index if needed

    # Data choices
    "SKIP_DAILY": os.getenv("SKIP_DAILY", "true").lower() == "true",

    # Indicator settings
    "INDICATOR_PERIODS": {
        "RSI": 14, "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
        "STOCHASTIC_K": 14, "STOCHASTIC_D": 3, "MA_SHORT": 50, "MA_LONG": 200,
        "ADX": 14, "BB_PERIOD": 20, "BB_STD_DEV": 2, "ROC": 12, "CCI": 20,
        "EMA_FAST": 20, "EMA_SLOW": 50, "ATR": 14, "VOLUME_SURGE": 20,
        "MOMENTUM": 10, "WILLIAMS_R": 14, "CMF": 20, "ADL_LOOKBACK": 10,
        "REL_VOL": 50, "VWAP_REGIME": None, "OBV_CONFIRM": 5, "OI_SURGE": 20,
        "OI_MOMENTUM": 10, "BBW_SQUEEZE_THRESHOLD": 0.025
    },
    "INDICATOR_WEIGHTS": {
        "VolumeSurge": 2.0, "Momentum": 1.9, "ADX": 1.8, "VWAP": 1.7, "EMA": 1.7,
        "MACD": 1.5, "OBV": 1.5, "ATR": 1.4, "Bollinger": 1.3, "RSI": 1.2,
        "ROC": 1.1, "Stochastic": 1.0, "CCI": 1.0, "MA": 1.0, "WWL": 1.0,
        "CMF": 1.8, "ADL": 1.6, "RelVol": 1.5, "VWAPRegime": 1.7, "OBVConfirm": 1.2,
        "Candlestick": 2.5, "VolatilitySqueeze": 1.8,
        "OISurge": 2.2, "OIMomentum": 2.0, "CallBias": 2.5, "PutBias": 2.5
    },
    "TIMEFRAME_WEIGHTS": {15: 3.2, 5: 2.8, 30: 2.0, 60: 1.2, "daily": 1.0},
    "BAR_SIZE_MAP": {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"},
    "DURATION_MAP": {5: "30 D", 15: "30 D", 30: "60 D", 60: "120 D", 1440: "365 D"},
}

# Root logger threshold
level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
logging.getLogger().setLevel(level_map.get(CONFIG["LOG_LEVEL"], logging.ERROR))

IST = pytz.timezone("Asia/Kolkata")

# Silence noisy third-party loggers
for noisy in ("truedata", "truedata.history", "truedata_ws", "websocket", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

console = Console()

# Global state
last_run_results = {}
api_calls_done = 0
api_calls_lock = threading.Lock()
performance_metrics = defaultdict(int)
failed_symbols = set()

# ---------- 5-minute boundary helpers ----------
def next_5min_boundary_ist(now_ist: datetime) -> datetime:
    minute = (now_ist.minute // 5) * 5
    boundary = now_ist.replace(minute=minute, second=0, microsecond=0)
    if boundary <= now_ist:
        boundary += timedelta(minutes=5)
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
                sleep_for = max(0.0, 1.0 / max(self.rate, 0.001))
            time.sleep(sleep_for)

# ---------- TrueData sessions ----------
def authenticate_session():
    return TD_hist(CONFIG["TDUSERNAME"], CONFIG["TDPASSWORD"], log_level=logging.CRITICAL)

def build_sessions():
    sess_count = CONFIG["TD_HIST_SESSIONS"]
    pool, limiters = [], []
    for i in range(sess_count):
        try:
            pool.append(authenticate_session())
        except Exception as e:
            logger.error(f"Session {i} init failed: {e}")
    if not pool:
        raise SystemExit("Failed to initialize TrueData sessions.")
    per_sess_rate = CONFIG["RATE_PER_SECOND_TOTAL"] / len(pool)
    for _ in pool:
        limiters.append(TokenBucketLimiter(rate_per_sec=per_sess_rate, bucket_size=CONFIG["BUCKET_SIZE"]))
    return pool, limiters

tdhist_pool, sess_limiters = build_sessions()

# ---------- Indicators ----------
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
    return pv_sum / vol_sum.replace(0, np.nan)

def atr(df, period=CONFIG["INDICATOR_PERIODS"]["ATR"]):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def williams_r(df, period=CONFIG["INDICATOR_PERIODS"]["WILLIAMS_R"]):
    highest = df["High"].rolling(period).max()
    lowest = df["Low"].rolling(period).min()
    denom = (highest - lowest).replace(0, np.nan)
    return -100 * (highest - df["Close"]) / denom

def momentum(df, period=CONFIG["INDICATOR_PERIODS"]["MOMENTUM"]):
    return df["Close"] / df["Close"].shift(period).replace(0, np.nan) - 1.0

def volume_surge(df, lookback=CONFIG["INDICATOR_PERIODS"]["VOLUME_SURGE"]):
    vol_ma = df["Volume"].rolling(lookback).mean()
    vol_std = df["Volume"].rolling(lookback).std()
    z_score = (df["Volume"] - vol_ma) / vol_std.replace(0, np.nan)
    return z_score

def calculate_rsi(df, period=CONFIG["INDICATOR_PERIODS"]["RSI"]):
    if len(df) < period + 1: return pd.Series(index=df.index, dtype='float64')
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rs.fillna(100, inplace=True)
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=CONFIG["INDICATOR_PERIODS"]["MACD_FAST"], slow=CONFIG["INDICATOR_PERIODS"]["MACD_SLOW"], signal=CONFIG["INDICATOR_PERIODS"]["MACD_SIGNAL"]):
    if len(df) < slow + signal: return pd.Series(index=df.index, dtype='float64'), pd.Series(index=df.index, dtype='float64')
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_stochastic(df, period=CONFIG["INDICATOR_PERIODS"]["STOCHASTIC_K"], smooth_d=CONFIG["INDICATOR_PERIODS"]["STOCHASTIC_D"]):
    if len(df) < period + smooth_d: return pd.Series(index=df.index, dtype='float64'), pd.Series(index=df.index, dtype='float64')
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    k = 100 * ((df['Close'] - low_min) / denom)
    k.fillna(50, inplace=True)
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_moving_averages(df, short=CONFIG["INDICATOR_PERIODS"]["MA_SHORT"], long=CONFIG["INDICATOR_PERIODS"]["MA_LONG"]):
    if len(df) < long: return pd.Series(index=df.index, dtype='float64'), pd.Series(index=df.index, dtype='float64')
    return df['Close'].rolling(window=short).mean(), df['Close'].rolling(window=long).mean()

def calculate_adx(df, period=CONFIG["INDICATOR_PERIODS"]["ADX"]):
    if len(df) < period * 2: return pd.Series(index=df.index, dtype='float64'), pd.Series(index=df.index, dtype='float64'), pd.Series(index=df.index, dtype='float64')
    x = df[['High', 'Low', 'Close']].copy()
    x['H-L'] = x['High'] - x['Low']
    x['H-C'] = abs(x['High'] - x['Close'].shift(1))
    x['L-C'] = abs(x['Low'] - x['Close'].shift(1))
    x['TR'] = x[['H-L', 'H-C', 'L-C']].max(axis=1)
    x['+DM'] = np.where((x['High'] - x['High'].shift(1)) > (x['Low'].shift(1) - x['Low']), x['High'] - x['High'].shift(1), 0)
    x['-DM'] = np.where((x['Low'].shift(1) - x['Low']) > (x['High'] - x['High'].shift(1)), x['Low'].shift(1) - x['Low'], 0)
    atr_val = x['TR'].ewm(com=period - 1, adjust=False).mean().replace(0, np.nan)
    pdi = (x['+DM'].ewm(com=period - 1, adjust=False).mean() / atr_val) * 100
    ndi = (x['-DM'].ewm(com=period - 1, adjust=False).mean() / atr_val) * 100
    adx = (abs(pdi - ndi) / (pdi + ndi).replace(0, np.nan)).ewm(com=period - 1, adjust=False).mean() * 100
    return adx.fillna(20), pdi.fillna(20), ndi.fillna(20)

def calculate_bollinger_bands(df, period=CONFIG["INDICATOR_PERIODS"]["BB_PERIOD"], std_dev=CONFIG["INDICATOR_PERIODS"]["BB_STD_DEV"]):
    if len(df) < period: return pd.Series(index=df.index, dtype='float64'), pd.Series(index=df.index, dtype='float64'), pd.Series(index=df.index, dtype='float64')
    middle = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return middle, upper, lower

def calculate_roc(df, period=CONFIG["INDICATOR_PERIODS"]["ROC"]):
    if len(df) < period + 1: return pd.Series(index=df.index, dtype='float64')
    shifted_close = df['Close'].shift(period).replace(0, np.nan)
    return ((df['Close'] - shifted_close) / shifted_close) * 100

def calculate_obv(df):
    if len(df) < 2: return pd.Series(index=df.index, dtype='float64')
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

def calculate_cci(df, period=CONFIG["INDICATOR_PERIODS"]["CCI"]):
    if len(df) < period: return pd.Series(index=df.index, dtype='float64')
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).replace(0, np.nan)
    return (tp - sma_tp) / (0.015 * mad)

def adl(df):
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = mfm.fillna(0)
    mfv = mfm * df["Volume"]
    return mfv.cumsum()

def cmf(df, period=CONFIG["INDICATOR_PERIODS"]["CMF"]):
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = mfm.fillna(0)
    mfv = mfm * df["Volume"]
    mfv_sum = mfv.rolling(period).sum()
    vol_sum = df["Volume"].rolling(period).sum().replace(0, np.nan)
    return (mfv_sum / vol_sum)

def relative_volume(df, lookback=CONFIG["INDICATOR_PERIODS"]["REL_VOL"]):
    vol_ma = df["Volume"].rolling(lookback).mean()
    return (df["Volume"] / vol_ma.replace(0, np.nan))

def vwap_distance(df, period=None):
    v = vwap(df, period=period)
    return ((df["Close"] - v) / v.replace(0, np.nan))

def slope(series, lookback=CONFIG["INDICATOR_PERIODS"]["ADL_LOOKBACK"]):
    if len(series) < lookback: return np.nan
    y = series.tail(lookback).values.astype(float)
    x = np.arange(len(y))
    x = (x - x.mean()) / (x.std() + 1e-9)
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return m

def vwap_full_session(df):
    if df is None or df.empty: return pd.Series(index=df.index if df is not None else [], dtype='float64')
    dfx = df.sort_index()[['High', 'Low', 'Close', 'Volume']].copy()
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
    out.update(sess_vwap)
    out = out.ffill()
    return out.reindex(df.index, method='ffill')

def adl_series(df):
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = mfm.fillna(0)
    mfv = mfm * df["Volume"]
    return mfv.cumsum()

def oi_surge(df, lookback=CONFIG["INDICATOR_PERIODS"]["OI_SURGE"]):
    oi_ma = df["OI"].rolling(lookback).mean()
    oi_std = df["OI"].rolling(lookback).std()
    z_score = (df["OI"] - oi_ma) / oi_std.replace(0, np.nan)
    return z_score

def oi_momentum(df, period=CONFIG["INDICATOR_PERIODS"]["OI_MOMENTUM"]):
    return df["OI"] / df["OI"].shift(period).replace(0, np.nan) - 1.0

def bollinger_bandwidth(df):
    _, upper, lower = calculate_bollinger_bands(df)
    middle = df['Close'].rolling(window=CONFIG["INDICATOR_PERIODS"]["BB_PERIOD"]).mean()
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    return bandwidth

def get_volatility_status(df):
    if len(df) < CONFIG["INDICATOR_PERIODS"]["BB_PERIOD"]: return "Unknown"
    bbw = bollinger_bandwidth(df)
    if bbw.empty or pd.isna(bbw.iloc[-1]): return "Unknown"
    lookback_period = 120
    lowest_bbw = bbw.rolling(window=min(len(bbw), lookback_period)).min()
    if pd.notna(lowest_bbw.iloc[-1]) and bbw.iloc[-1] <= lowest_bbw.iloc[-1] * 1.1:
         return "Squeeze"
    return "Expanding"

def detect_engulfing_pattern(df):
    if len(df) < 2: return 0
    last = df.iloc[-1]; prev = df.iloc[-2]
    is_bullish_engulfing = (last['Close'] > last['Open'] and prev['Close'] < prev['Open'] and last['Close'] >= prev['Open'] and last['Open'] <= prev['Close'])
    if is_bullish_engulfing: return 1.0
    is_bearish_engulfing = (last['Close'] < last['Open'] and prev['Close'] > prev['Open'] and last['Open'] >= prev['Close'] and last['Close'] <= prev['Open'])
    if is_bearish_engulfing: return -1.0
    return 0.0

def get_market_regime(df):
    adx, _, _ = calculate_adx(df)
    if adx.empty or pd.isna(adx.iloc[-1]): return "Unknown"
    adx_val = adx.iloc[-1]; is_rising = len(adx) >= 4 and adx.iloc[-1] > adx.iloc[-4]
    if adx_val > 25 and is_rising: return "Trending"
    elif adx_val < 20: return "Ranging"
    else: return "Transition"

def get_indicator_scores(df):
    scores = {}
    periods = CONFIG["INDICATOR_PERIODS"].values()
    valid_periods = [p for p in periods if (p is not None and isinstance(p, (int, float)))]
    min_bars = (max(valid_periods) if valid_periods else 50) + 10
    if len(df) < min_bars:
        return {k: 0.0 for k in CONFIG["INDICATOR_WEIGHTS"]}
    rsi_series = calculate_rsi(df)
    if len(rsi_series) and pd.notna(rsi_series.iloc[-1]):
        rsi = rsi_series.iloc[-1]; prev_rsi = rsi_series.iloc[-2] if len(rsi_series) > 1 else rsi
        scores['RSI'] = (2.0 if rsi > 60 and prev_rsi <= 60 else 1.0 if rsi > 50 and prev_rsi <= 50 else -2.0 if rsi < 40 and prev_rsi >= 40 else -1.0 if rsi < 50 and prev_rsi >= 50 else 0.0)
    else: scores['RSI'] = 0.0
    macd, signal = calculate_macd(df)
    if len(macd) and len(signal) and pd.notna(macd.iloc[-1]) and pd.notna(signal.iloc[-1]): scores['MACD'] = 1.0 if macd.iloc[-1] > signal.iloc[-1] else -1.0
    else: scores['MACD'] = 0.0
    k, d = calculate_stochastic(df)
    if len(k) and len(d) and pd.notna(k.iloc[-1]) and pd.notna(d.iloc[-1]): scores['Stochastic'] = (1.0 if k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80 else -1.0 if k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 20 else 0.0)
    else: scores['Stochastic'] = 0.0
    ma_short, ma_long = calculate_moving_averages(df)
    if len(ma_short) and len(ma_long) and pd.notna(ma_short.iloc[-1]) and pd.notna(ma_long.iloc[-1]): scores['MA'] = 1.0 if ma_short.iloc[-1] > ma_long.iloc[-1] else -1.0
    else: scores['MA'] = 0.0
    adx, pdi, ndi = calculate_adx(df)
    if len(adx) and pd.notna(adx.iloc[-1]) and len(adx) > 4:
        is_rising = adx.iloc[-1] > adx.iloc[-3]; just_crossed = adx.iloc[-1] > 22 and adx.iloc[-2] <= 22
        scores['ADX'] = (1.5 * (2.0 if just_crossed else 1.0) if (adx.iloc[-1] > 22 and is_rising) and pdi.iloc[-1] > ndi.iloc[-1] else -1.5 * (2.0 if just_crossed else 1.0) if (adx.iloc[-1] > 22 and is_rising) and pdi.iloc[-1] <= ndi.iloc[-1] else 0.0)
    else: scores['ADX'] = 0.0
    bb_middle, _, _ = calculate_bollinger_bands(df)
    if len(bb_middle) and pd.notna(bb_middle.iloc[-1]): scores['Bollinger'] = 0.5 if df['Close'].iloc[-1] > bb_middle.iloc[-1] else -0.5
    else: scores['Bollinger'] = 0.0
    roc = calculate_roc(df)
    scores['ROC'] = 1.0 if len(roc) and pd.notna(roc.iloc[-1]) and roc.iloc[-1] > 0 else (-1.0 if len(roc) and pd.notna(roc.iloc[-1]) else 0.0)
    obv_line = calculate_obv(df)
    if len(obv_line) >= 2 and pd.notna(obv_line.iloc[-1]) and pd.notna(obv_line.iloc[-2]): scores['OBV'] = 1.0 if obv_line.iloc[-1] > obv_line.iloc[-2] else -1.0
    else: scores['OBV'] = 0.0
    cci_val = calculate_cci(df)
    if len(cci_val) and pd.notna(cci_val.iloc[-1]):
        cci = cci_val.iloc[-1]
        scores['CCI'] = (1.5 if cci > 100 else 1.0 if cci > 0 else -1.5 if cci < -100 else -1.0 if cci < 0 else 0.0)
    else: scores['CCI'] = 0.0
    ema_fast = ema(df["Close"], CONFIG["INDICATOR_PERIODS"]["EMA_FAST"])
    ema_slow = ema(df["Close"], CONFIG["INDICATOR_PERIODS"]["EMA_SLOW"])
    if len(ema_fast) and len(ema_slow) and pd.notna(ema_fast.iloc[-1]) and pd.notna(ema_slow.iloc[-1]): scores["EMA"] = 1.0 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1.0
    else: scores["EMA"] = 0.0
    vwap_line = vwap(df, period=None)
    if len(vwap_line) and pd.notna(vwap_line.iloc[-1]) and pd.notna(df["Close"].iloc[-1]): scores["VWAP"] = 1.0 if df["Close"].iloc[-1] > vwap_line.iloc[-1] else -1.0
    else: scores["VWAP"] = 0.0
    atr_val = atr(df)
    if len(atr_val) >= 6 and all(pd.notna(val) for val in [atr_val.iloc[-1], atr_val.iloc[-5], df["Close"].iloc[-1], df["Close"].iloc[-5]]):
        atr_rising = (atr_val.iloc[-1] / atr_val.iloc[-5]) > 1.1; price_up5 = df["Close"].iloc[-1] > df["Close"].iloc[-5]
        scores["ATR"] = 1.5 if atr_rising and price_up5 else (-1.5 if atr_rising and not price_up5 else 0.0)
    else: scores["ATR"] = 0.0
    zscore = volume_surge(df)
    if len(zscore) >= 2 and pd.notna(zscore.iloc[-1]):
        price_up_last = df["Close"].iloc[-1] > df["Close"].iloc[-2]
        scores["VolumeSurge"] = (1.5 if zscore.iloc[-1] >= 2.0 and price_up_last else -1.5 if zscore.iloc[-1] <= -2.0 and not price_up_last else 0.0)
    else: scores["VolumeSurge"] = 0.0
    mom = momentum(df)
    if len(mom) and pd.notna(mom.iloc[-1]): scores["Momentum"] = 1.5 if mom.iloc[-1] > 0.01 else (-1.5 if mom.iloc[-1] < -0.01 else 0.0)
    else: scores["Momentum"] = 0.0
    wr = williams_r(df)
    if len(wr) and pd.notna(wr.iloc[-1]): scores["WWL"] = 1.0 if wr.iloc[-1] < -80 else (-1.0 if wr.iloc[-1] > -20 else 0.0)
    else: scores["WWL"] = 0.0
    cmf20 = cmf(df)
    if len(cmf20) and pd.notna(cmf20.iloc[-1]):
        val = cmf20.iloc[-1]; scores["CMF"] = 1.5 if val > 0.1 else (-1.5 if val < -0.1 else 0.0)
    else: scores["CMF"] = 0.0
    adl_line = adl(df)
    if len(adl_line) >= 6 and pd.notna(adl_line.iloc[-1]) and pd.notna(adl_line.iloc[-5]): scores["ADL"] = 1.2 if (adl_line.iloc[-1] - adl_line.iloc[-5]) > 0 else -1.2
    else: scores["ADL"] = 0.0
    rv = relative_volume(df)
    if len(rv) and pd.notna(rv.iloc[-1]): scores["RelVol"] = 1.0 if rv.iloc[-1] >= 1.5 else (-0.5 if rv.iloc[-1] <= 0.5 else 0.0)
    else: scores["RelVol"] = 0.0
    vd = vwap_distance(df, period=None)
    if len(vd) and pd.notna(vd.iloc[-1]):
        d = vd.iloc[-1]; scores["VWAPRegime"] = 1.3 if d > 0.002 else (-1.3 if d < -0.002 else 0.0)
    else: scores["VWAPRegime"] = 0.0
    obv_line2 = calculate_obv(df)
    if len(obv_line2) >= 6 and pd.notna(obv_line2.iloc[-1]) and pd.notna(obv_line2.iloc[-5]):
        obv_up = obv_line2.iloc[-1] > obv_line2.iloc[-5]; price_up5 = df["Close"].iloc[-1] > df["Close"].iloc[-5]
        scores["OBVConfirm"] = 1.0 if obv_up and price_up5 else (-1.0 if not obv_up and not price_up5 else 0.0)
    else: scores["OBVConfirm"] = 0.0
    oi_col = 'OI' if 'OI' in df.columns else ('Oi' if 'Oi' in df.columns else None)
    if oi_col is not None:
        df_oi = df.copy()
        if oi_col != 'OI': df_oi['OI'] = pd.to_numeric(df_oi[oi_col], errors='coerce').fillna(0)
        oi_z = oi_surge(df_oi, lookback=CONFIG["INDICATOR_PERIODS"]["OI_SURGE"]); oi_mom = oi_momentum(df_oi, period=CONFIG["INDICATOR_PERIODS"]["OI_MOMENTUM"]); rel_vol = relative_volume(df_oi, lookback=CONFIG["INDICATOR_PERIODS"]["REL_VOL"]); vwap_day = vwap_full_session(df_oi); vdist = ((df_oi["Close"] - vwap_day) / vwap_day.replace(0, np.nan)); adx_val, pdi, ndi = calculate_adx(df_oi); conds_ok = all(len(x) for x in [oi_z, oi_mom, rel_vol, vdist, adx_val])
        if conds_ok and all(pd.notna(s.iloc[-1]) for s in [oi_z, oi_mom, rel_vol, vdist, adx_val]):
            price_up = df_oi["Close"].iloc[-1] > df_oi["Close"].iloc[-2] if len(df_oi) >= 2 else False; price_down = df_oi["Close"].iloc[-1] < df_oi["Close"].iloc[-2] if len(df_oi) >= 2 else False; above_vwap = vdist.iloc[-1] > 0; below_vwap = vdist.iloc[-1] < 0; strong_rvol = rel_vol.iloc[-1] >= 1.5; strong_adx = adx_val.iloc[-1] > 20; p_over_n = pdi.iloc[-1] > ndi.iloc[-1] if len(pdi) and len(ndi) and pd.notna(pdi.iloc[-1]) and pd.notna(ndi.iloc[-1]) else False; call_bias = price_up and above_vwap and strong_rvol and strong_adx and p_over_n and oi_z.iloc[-1] >= 1.0 and oi_mom.iloc[-1] > 0; put_bias = price_down and below_vwap and strong_rvol and strong_adx and (not p_over_n) and oi_z.iloc[-1] >= 1.0 and oi_mom.iloc[-1] > 0; call_trap = price_up and rel_vol.iloc[-1] < 1.0 and oi_mom.iloc[-1] < 0; put_trap = price_down and rel_vol.iloc[-1] < 1.0 and oi_mom.iloc[-1] < 0; scores["OISurge"] = 1.0 if oi_z.iloc[-1] >= 1.0 else (-0.5 if oi_z.iloc[-1] <= -1.0 else 0.0); scores["OIMomentum"] = 1.0 if oi_mom.iloc[-1] > 0 else (-0.5 if oi_mom.iloc[-1] < 0 else 0.0); scores["CallBias"] = 1.5 if call_bias else (-1.0 if call_trap else 0.0); scores["PutBias"] = -1.5 if put_bias else (1.0 if put_trap else 0.0)
    scores['Candlestick'] = detect_engulfing_pattern(df) * 2.0; vol_status = get_volatility_status(df)
    scores['VolatilitySqueeze'] = 0.0
    if vol_status == "Squeeze":
        if scores.get("VolumeSurge", 0.0) > 1.0: scores['VolatilitySqueeze'] = 1.5
        elif scores.get("VolumeSurge", 0.0) < -1.0: scores['VolatilitySqueeze'] = -1.5
    max_abs_score = max(abs(s) for s in scores.values()) if any(scores.values()) else 1.0
    for k in scores: scores[k] /= max_abs_score
    return scores

def trend_quality(df, lookback=10):
    if len(df) < lookback: return 0
    recent_df = df.tail(lookback); hh = (recent_df['High'].diff().dropna() >= 0).sum(); ll = (recent_df['Low'].diff().dropna() >= 0).sum(); uptrend_consistency = (hh + ll) / (2 * (lookback - 1)); lh = (recent_df['High'].diff().dropna() <= 0).sum(); ll_down = (recent_df['Low'].diff().dropna() <= 0).sum(); downtrend_consistency = (lh + ll_down) / (2 * (lookback - 1));
    if uptrend_consistency > 0.7: return 1
    if downtrend_consistency > 0.7: return -1
    return 0

def institutional_flow_votes(tf_data):
    votes = []
    for t in (5, 15, 30):
        df = tf_data.get(t)
        if df is None or len(df) < 60: continue
        cmf20 = cmf(df); adl_line = adl_series(df); adx_val, pdi, ndi = calculate_adx(df); rv50 = relative_volume(df); vwap_day = vwap_full_session(df); vdist = ((df["Close"] - vwap_day) / vwap_day.replace(0, np.nan)); tq = trend_quality(df); cmf_last = cmf20.iloc[-1] if len(cmf20) and pd.notna(cmf20.iloc[-1]) else np.nan; cmf_slope = slope(cmf20); adl_slope = slope(adl_line); adx_last = adx_val.iloc[-1] if len(adx_val) and pd.notna(adx_val.iloc[-1]) else np.nan; p_over_n = pdi.iloc[-1] > ndi.iloc[-1] if len(pdi) and len(ndi) and pd.notna(pdi.iloc[-1]) and pd.notna(ndi.iloc[-1]) else False; rv_last = rv50.iloc[-1] if len(rv50) and pd.notna(rv50.iloc[-1]) else np.nan; vdist_last = vdist.iloc[-1] if len(vdist) and pd.notna(vdist.iloc[-1]) else np.nan; strong_rvol = pd.notna(rv_last) and rv_last >= 1.5;
        buy = (pd.notna(vdist_last) and vdist_last > -0.005 and pd.notna(cmf_last) and cmf_last > 0.05 and not np.isnan(cmf_slope) and cmf_slope >= 0 and not np.isnan(adl_slope) and adl_slope > 0 and pd.notna(adx_last) and adx_last > 20 and p_over_n and strong_rvol and tq == 1)
        sell = (pd.notna(vdist_last) and vdist_last < 0.005 and pd.notna(cmf_last) and cmf_last < -0.05 and not np.isnan(cmf_slope) and cmf_slope <= 0 and not np.isnan(adl_slope) and adl_slope < 0 and pd.notna(adx_last) and adx_last > 20 and not p_over_n and strong_rvol and tq == -1)
        votes.append((t, 1 if buy else (-1 if sell else 0)))
    return votes

def infer_institutional_flow(tf_data):
    votes = institutional_flow_votes(tf_data)
    vote_sum = sum(v for _, v in votes)
    if vote_sum >= 2: return "Institutional Accumulation"
    if vote_sum <= -2: return "Institutional Distribution"
    return "Mixed/Unclear"

def analyze_signals(timeframe_dataframes):
    final_score, max_possible = 0.0, 0.0; vol_statuses = [get_volatility_status(timeframe_dataframes.get(tf)) for tf in [5, 15] if timeframe_dataframes.get(tf) is not None]; overall_vol_status = "Squeeze" if "Squeeze" in vol_statuses else "Expanding"
    for tf_min, df in timeframe_dataframes.items():
        if df is None or len(df) < 50: continue
        indicator_scores = get_indicator_scores(df); tf_weight = CONFIG["TIMEFRAME_WEIGHTS"].get(tf_min, 1.0); regime = get_market_regime(df); trending_indicators = {"MA", "MACD", "ADX", "EMA", "Momentum"}; ranging_indicators = {"RSI", "Stochastic", "CCI", "WWL"};
        for indicator, score in indicator_scores.items():
            ind_weight = CONFIG["INDICATOR_WEIGHTS"].get(indicator, 1.0)
            if regime == "Trending" and indicator in trending_indicators: ind_weight *= 1.25
            elif regime == "Ranging" and indicator in ranging_indicators: ind_weight *= 1.25
            elif regime == "Transition": ind_weight *= 0.8
            final_score += score * tf_weight * ind_weight; max_possible += 1.0 * tf_weight * ind_weight
    inst_votes = institutional_flow_votes(timeframe_dataframes); inst_score = sum(v * CONFIG["TIMEFRAME_WEIGHTS"].get(tf, 1.0) * 2.0 for tf, v in inst_votes); max_possible += sum(2.0 * CONFIG["TIMEFRAME_WEIGHTS"].get(tf, 1.0) for tf, _ in inst_votes); final_score += inst_score
    if max_possible == 0: return 'Neutral', 0.0, "Unknown"
    normalized = (final_score / max_possible) * 100.0
    if normalized >= 65: signal_text = 'Very Strong Buy'
    elif normalized >= 25: signal_text = 'Strong Buy'
    elif normalized <= -65: signal_text = 'Very Strong Sell'
    elif normalized <= -25: signal_text = 'Strong Sell'
    else: signal_text = 'Neutral'
    return signal_text, normalized, overall_vol_status

def normalize_hist_df(df, symbol, timeframe_minutes):
    if df is None or df.empty: return None
    try:
        out = df.copy(); out.columns = out.columns.str.lower()
        rename_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume', 'oi': 'Oi', 'open_interest': 'Oi', 'timestamp': 'Timestamp', 'time': 'Timestamp', 'date': 'Timestamp', 'datetime': 'Timestamp'}
        out.rename(columns={c: rename_map.get(c, c.capitalize()) for c in out.columns}, inplace=True)
        for req_col in ['Open', 'High', 'Low', 'Close', 'Timestamp']:
            if req_col not in out.columns: return None
        if 'Volume' not in out.columns: out['Volume'] = 0
        if 'Oi' not in out.columns: out['Oi'] = 0
        out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce").dt.tz_localize(IST)
        out = out.dropna(subset=["Timestamp"])
        for col in ["Open", "High", "Low", "Close", "Volume", "Oi"]: out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_values("Timestamp").set_index("Timestamp")
        out = out[~out.index.duplicated(keep='last')]
        if 'OI' not in out.columns and 'Oi' in out.columns: out['OI'] = out['Oi']
        return out if len(out) >= 50 else None
    except Exception: return None

def pick_session(symbol_orig, timeframe_minutes):
    return (hash((symbol_orig, timeframe_minutes)) & 0x7fffffff) % len(tdhist_pool)

@retry(stop_max_attempt_number=CONFIG["RETRY_ATTEMPTS"], wait_exponential_multiplier=max(1, int(CONFIG["RETRY_DELAY_MS"] / 2)), wait_exponential_max=8000, retry_on_exception=lambda e: True)
def fetch_one(symbol_orig, timeframe_minutes, limiter, hist):
    td_symbol = symbol_orig.replace('-EQ', '')
    if td_symbol in failed_symbols: return symbol_orig, timeframe_minutes, None
    if CONFIG.get("SKIP_DAILY", True) and timeframe_minutes == 1440: return symbol_orig, timeframe_minutes, None
    bar_size = CONFIG["BAR_SIZE_MAP"].get(timeframe_minutes); duration = CONFIG["DURATION_MAP"].get(timeframe_minutes)
    if not bar_size or not duration: return symbol_orig, timeframe_minutes, None
    try:
        limiter.acquire(); df_raw = hist.get_historic_data(td_symbol, duration=duration, bar_size=bar_size)
        if df_raw is None or df_raw.empty: failed_symbols.add(td_symbol); return symbol_orig, timeframe_minutes, None
        df = normalize_hist_df(df_raw, td_symbol, timeframe_minutes)
        global api_calls_done;
        with api_calls_lock: api_calls_done += 1
        if df is None: failed_symbols.add(td_symbol)
        return symbol_orig, timeframe_minutes, df
    except Exception: failed_symbols.add(td_symbol); return symbol_orig, timeframe_minutes, None

def prefetch_all(stocks, max_workers=CONFIG["MAX_WORKERS"]):
    tfs = [5, 15, 30, 60] if CONFIG.get("SKIP_DAILY", True) else [5, 15, 30, 60, 1440]
    stock_multi_data = defaultdict(dict); global api_calls_done; api_calls_done = 0
    valid_stocks = [s for s in stocks if s]
    progress_kwargs = dict(total=len(valid_stocks) * len(tfs), desc="Fetching data from API", ncols=100, disable=not CONFIG["SHOW_PROGRESS"])
    with tqdm(**progress_kwargs) as api_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(fetch_one, s, tf, sess_limiters[pick_session(s, tf)], tdhist_pool[pick_session(s, tf)]) for s in valid_stocks for tf in tfs]
            for fut in as_completed(futures):
                try:
                    symbol_orig, tf, df = fut.result()
                    if df is not None: stock_multi_data[symbol_orig][tf] = df
                except Exception: pass
                api_bar.update(1)
    return {s: d for s, d in stock_multi_data.items() if len(d) >= 2}

def save_data_locally(stocks):
    console.rule("[bold cyan]Starting data download and save process[/bold cyan]"); logger.info(f"Fetching data for {len(stocks)} symbols to save locally...")
    stock_multi_data = prefetch_all(stocks)
    if not stock_multi_data: logger.error("No data fetched. Nothing to save."); return
    save_dir = CONFIG['DATA_DIR']; os.makedirs(save_dir, exist_ok=True); saved_count = 0
    for symbol, tf_data in tqdm(stock_multi_data.items(), desc="Saving data locally"):
        symbol_dir = os.path.join(save_dir, symbol); os.makedirs(symbol_dir, exist_ok=True)
        for tf, df in tf_data.items():
            file_path = os.path.join(symbol_dir, f"{tf}.csv"); df.to_csv(file_path); saved_count += 1
    console.print(f"[bold green]✅ Success![/bold green] Saved {saved_count} data files for {len(stock_multi_data)} symbols in '{save_dir}' directory.")

def load_local_data(stocks):
    stock_multi_data = defaultdict(dict); load_dir = CONFIG['DATA_DIR']
    if not os.path.isdir(load_dir):
        logger.error(f"Data directory '{load_dir}' not found. Please run with --save-data first."); return {}
    for symbol in tqdm(stocks, desc="Loading local data"):
        symbol_dir = os.path.join(load_dir, symbol)
        if os.path.isdir(symbol_dir):
            for file_name in os.listdir(symbol_dir):
                if file_name.endswith('.csv'):
                    try:
                        tf = int(file_name.replace('.csv', '')); file_path = os.path.join(symbol_dir, file_name); df = pd.read_csv(file_path, index_col='Timestamp', parse_dates=True)
                        df.index = df.index.tz_convert(IST); stock_multi_data[symbol][tf] = df
                    except Exception: pass
    return {s: d for s, d in stock_multi_data.items() if len(d) >= 2}

def parse_asof(s: str):
    dt = datetime.strptime(s, "%Y-%m-%dT%H:%M") if 'T' in s else datetime.strptime(s, "%Y-%m-%d")
    if 'T' not in s: h, m = parse_hhmm(CONFIG["MARKET_END"]); dt = dt.replace(hour=h, minute=m)
    return IST.localize(dt)

def day_checkpoints_ist(day_date: datetime):
    d = day_date.date(); start_h, start_m = parse_hhmm(CONFIG["FIRST_RUN_AT"]); end_h, end_m = parse_hhmm(CONFIG["MARKET_END"]); start_dt = IST.localize(datetime(d.year, d.month, d.day, start_h, start_m)); end_dt = IST.localize(datetime(d.year, d.month, d.day, end_h, end_m)); return list(pd.date_range(start=start_dt, end=end_dt, freq="5T", tz=IST, inclusive="both").to_pydatetime())

def render_top_lists(now_ts, top_bullish, top_bearish, market_regime=""):
    title = f"| OPTION BUYER SCANNER | SNAPSHOT AT {now_ts.strftime('%Y-%m-%d %H:%M')} IST | MARKET REGIME: {market_regime}"
    console.rule(title)
    bull_table = Table(title="Top 20 Bullish Breakouts", box=box.SIMPLE_HEAVY, header_style="white on dark_green", style="white on black"); headers = [("Stock", "cyan", "left"), ("Signal", "bright_white", "left"), ("Score", "yellow", "right"), ("State Change", "magenta", "left"), ("Volatility", "bright_white", "left"), ("Flow", "bright_white", "left"), ("Action", "bright_white", "left")]
    for col, style, justify in headers: bull_table.add_column(col, style=style, justify=justify)
    for r in top_bullish:
        sym = r['symbol']; prev_result = last_run_results.get(sym, {}); prev_signal = prev_result.get('signal', 'Neutral'); is_new_breakout = 'Strong' in r['signal'] and 'Strong' not in prev_signal; row_style = "black on green" if is_new_breakout else None; state_change_str = f"{prev_signal} -> {r['signal']}" if prev_signal != r['signal'] else r['signal']; bull_table.add_row(sym, r['signal'], f"{r['score']:.2f}", state_change_str, r.get('volatility', 'N/A'), r.get('flow', 'Unknown'), "Consider Call", style=row_style)
    console.print(bull_table)
    bear_table = Table(title="Top 20 Bearish Breakdowns", box=box.SIMPLE_HEAVY, header_style="white on dark_red", style="white on black")
    for col, style, justify in headers: bear_table.add_column(col, style=style, justify=justify)
    for r in top_bearish:
        sym = r['symbol']; prev_result = last_run_results.get(sym, {}); prev_signal = prev_result.get('signal', 'Neutral'); is_new_breakdown = 'Strong' in r['signal'] and 'Strong' not in prev_signal; row_style = "white on red" if is_new_breakdown else None; state_change_str = f"{prev_signal} -> {r['signal']}" if prev_signal != r['signal'] else r['signal']; bear_table.add_row(sym, r['signal'], f"{r['score']:.2f}", state_change_str, r.get('volatility', 'N/A'), r.get('flow', 'Unknown'), "Consider Put", style=row_style)
    console.print(bear_table); console.rule()

def export_to_csv(now_ts, top_bullish, top_bearish, filename, market_regime=""):
    with open(filename, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow([f"Snapshot Time: {now_ts.strftime('%Y-%m-%d %H:%M')}, Market Regime: {market_regime}"]); headers = ["Stock", "Signal", "Score", "State Change", "Volatility", "Flow", "Action"]
        writer.writerow(["Top 20 Bullish (Momentum Breakouts)"]); writer.writerow(headers)
        if not top_bullish: writer.writerow(["No strong bullish signals found."])
        for r in top_bullish:
            prev_signal = last_run_results.get(r['symbol'], {}).get('signal', 'Neutral'); state_change_str = f"{prev_signal} -> {r['signal']}" if prev_signal != r['signal'] else r['signal']; writer.writerow([r['symbol'], r['signal'], f"{r['score']:.2f}", state_change_str, r.get('volatility', 'N/A'), r.get('flow', 'Unknown'), "Consider Call"])
        writer.writerow([]); writer.writerow(["Top 20 Bearish (Momentum Breakdowns)"]); writer.writerow(headers)
        if not top_bearish: writer.writerow(["No strong bearish signals found."])
        for r in top_bearish:
            prev_signal = last_run_results.get(r['symbol'], {}).get('signal', 'Neutral'); state_change_str = f"{prev_signal} -> {r['signal']}" if prev_signal != r['signal'] else r['signal']; writer.writerow([r['symbol'], r['signal'], f"{r['score']:.2f}", state_change_str, r.get('volatility', 'N/A'), r.get('flow', 'Unknown'), "Consider Put"])
        writer.writerow([])

def process_scan(stock_multi_data, time_point_aware):
    global last_run_results, performance_metrics; signals_this_scan = []; current_results = {}
    for symbol, timeframe_data in stock_multi_data.items():
        clean_symbol = symbol.replace('-EQ', ''); filtered_timeframes = {tf: df[df.index <= time_point_aware] for tf, df in timeframe_data.items() if df is not None and not df.empty and len(df[df.index <= time_point_aware]) >= 50}
        if len(filtered_timeframes) < 2: continue
        signal, score, vol_status = analyze_signals(filtered_timeframes); flow_tag = infer_institutional_flow(filtered_timeframes); result_dict = {'symbol': clean_symbol, 'signal': signal, 'score': score, 'flow': flow_tag, 'volatility': vol_status}; current_results[clean_symbol] = result_dict
        if 'Strong' in signal:
            signals_this_scan.append(result_dict); direction = 'bullish' if 'Buy' in signal else 'bearish'; performance_metrics[f"{direction}_signals"] += 1
    signals_this_scan.sort(key=lambda x: x['score'], reverse=True); top_bullish = [r for r in signals_this_scan if 'Buy' in r['signal']][:20]; top_bearish = sorted([r for r in signals_this_scan if 'Sell' in r['signal']], key=lambda x: x['score'])[:20]; last_run_results = current_results.copy(); return top_bullish, top_bearish

def get_market_regime_live(hist_session: TD_hist) -> str:
    index_symbol = CONFIG["MARKET_INDEX"]
    try:
        df_raw = hist_session.get_historic_data(index_symbol, duration=CONFIG["DURATION_MAP"][15], bar_size=CONFIG["BAR_SIZE_MAP"][15])
        df_index = normalize_hist_df(df_raw, index_symbol, 15)
        if df_index is None or len(df_index) < 50: return "Sideways/Choppy"
        ema_fast = ema(df_index['Close'], 8); ema_slow = ema(df_index['Close'], 20)
        adx, pdi, ndi = calculate_adx(df_index, period=14)
        if adx.empty or ema_fast.empty or ema_slow.empty: return "Sideways/Choppy"
        last_adx = adx.iloc[-1]; is_trending = last_adx > 22
        if is_trending and ema_fast.iloc[-1] > ema_slow.iloc[-1] and pdi.iloc[-1] > ndi.iloc[-1]: return "Bullish"
        elif is_trending and ema_fast.iloc[-1] < ema_slow.iloc[-1] and ndi.iloc[-1] > pdi.iloc[-1]: return "Bearish"
        else: return "Sideways/Choppy"
    except Exception as e:
        logger.error(f"Could not determine market regime for {index_symbol}: {e}")
        return "Sideways/Choppy"

def run_backtest_day(day_str: str, stocks):
    day_date = datetime.strptime(day_str, "%Y-%m-%d"); global last_run_results, performance_metrics, failed_symbols; last_run_results.clear(); performance_metrics.clear(); failed_symbols.clear()
    if os.path.isdir(CONFIG['DATA_DIR']):
        console.print(f"[bold yellow]Found local data directory. Loading for consistent backtest.[/bold yellow]"); stock_multi_data = load_local_data(stocks)
    else:
        console.print("[bold red]Local data not found. Falling back to live API (results may be inconsistent).[/bold red]"); console.print(f"Run with the '--save-data' flag first for reproducible backtests."); stock_multi_data = prefetch_all(stocks)
    if not stock_multi_data: console.print("[bold red]Error: No data available for backtest. Exiting.[/bold red]"); return
    checkpoints = day_checkpoints_ist(day_date); output_filename = f"{day_date.strftime('%Y-%m-%d')}_options_scan_results.txt"; csv_filename = f"{day_date.strftime('%Y-%m-%d')}_options_scan_results.csv"
    if os.path.exists(output_filename): os.remove(output_filename)
    if os.path.exists(csv_filename): os.remove(csv_filename)
    for asof_ts in checkpoints:
        top_bullish, top_bearish = process_scan(stock_multi_data, asof_ts)
        render_top_lists(asof_ts, top_bullish, top_bearish)
        export_to_csv(asof_ts, top_bullish, top_bearish, csv_filename)

def run_once_asof(asof_ts, stocks):
    global last_run_results, performance_metrics, failed_symbols; last_run_results.clear(); performance_metrics.clear(); failed_symbols.clear()
    stock_multi_data = prefetch_all(stocks)
    top_bullish, top_bearish = process_scan(stock_multi_data, asof_ts)
    render_top_lists(asof_ts, top_bullish, top_bearish)

def run_live_5min():
    try:
        with open(CONFIG["SHARES_FILE"], 'r') as f:
            stocks = [line.strip().upper() for line in f if line.strip()]
    except Exception as e:
        raise SystemExit(f"Could not read {CONFIG['SHARES_FILE']}: {e}")
    now = datetime.now(IST); first_run_dt = today_ist_dt(CONFIG["FIRST_RUN_AT"])
    if now < first_run_dt:
        logger.info(f"Waiting until {CONFIG['FIRST_RUN_AT']} IST for first scan...")
        sleep_until(first_run_dt + timedelta(seconds=CONFIG["SETTLE_DELAY_SECONDS"]))
    global last_run_results, performance_metrics, failed_symbols; last_run_results.clear(); performance_metrics.clear(); failed_symbols.clear()
    output_filename = datetime.now(IST).strftime("%Y-%m-%d") + "_options_scan_results.txt"
    csv_filename = datetime.now(IST).strftime("%Y-%m-%d") + "_options_scan_results.csv"
    index_check_session = tdhist_pool[0]
    while True:
        now_ist = datetime.now(IST); session_end = today_ist_dt(CONFIG["MARKET_END"])
        if now_ist > session_end + timedelta(minutes=1):
            logger.info("Market closed. Sleeping until next session.")
            tomorrow = now_ist.date() + timedelta(days=1); next_start_h, next_start_m = parse_hhmm(CONFIG["FIRST_RUN_AT"]); next_first_run = IST.localize(datetime(tomorrow.year, tomorrow.month, tomorrow.day, next_start_h, next_start_m)); sleep_until(next_first_run + timedelta(seconds=CONFIG["SETTLE_DELAY_SECONDS"]))
            failed_symbols.clear(); continue
        market_regime = get_market_regime_live(index_check_session)
        stock_multi_data = prefetch_all(stocks)
        top_bullish, top_bearish = process_scan(stock_multi_data, now_ist)
        filtered_bullish, filtered_bearish = top_bullish, top_bearish
        if market_regime == "Bearish":
            if top_bullish: console.print(f"[bold yellow]MARKET REGIME is BEARISH. Ignoring {len(top_bullish)} bullish signal(s).[/bold yellow]")
            filtered_bullish = []
        if market_regime == "Bullish":
            if top_bearish: console.print(f"[bold yellow]MARKET REGIME is BULLISH. Ignoring {len(top_bearish)} bearish signal(s).[/bold yellow]")
            filtered_bearish = []
        render_top_lists(now_ist, filtered_bullish, filtered_bearish, market_regime)
        export_to_csv(now_ist, filtered_bullish, filtered_bearish, csv_filename, market_regime)
        nxt = next_5min_boundary_ist(datetime.now(IST))
        sleep_until(nxt + timedelta(seconds=CONFIG["SETTLE_DELAY_SECONDS"]))

def main():
    parser = argparse.ArgumentParser(description="Advanced Options Buyer Scanner with local data support for backtesting.", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--asof", type=str, default=None, help="Snapshot as-of time, e.g. 2025-09-26T09:50")
    parser.add_argument("--backtest-date", type=str, default=None, help="Full-day backtest using local data, e.g. 2025-09-26")
    parser.add_argument("--save-data", action="store_true", help="Run in data acquisition mode. Fetches all data from API and saves it locally for future backtests.")
    args = parser.parse_args()
    try:
        with open(CONFIG["SHARES_FILE"], 'r') as f:
            stocks = [line.strip().upper() for line in f if line.strip()]
    except Exception as e:
        raise SystemExit(f"Could not read {CONFIG['SHARES_FILE']}: {e}")
    if args.save_data:
        save_data_locally(stocks)
    elif args.backtest_date:
        run_backtest_day(args.backtest_date, stocks)
    elif args.asof:
        run_once_asof(parse_asof(args.asof), stocks)
    else:
        run_live_5min()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScan interrupted by user. Shutting down.")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
    finally:
        logger.info("Disconnecting TrueData sessions...")
        for sess in tdhist_pool:
            try: sess.disconnect()
            except Exception: pass
        logger.info(f"Shutdown complete. Final Metrics: {dict(performance_metrics)}")