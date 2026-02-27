
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

# ---------- Enhanced Weights for Option Buyer Perspective ----------
OPTION_BUYER_WEIGHTS = {
    "VolumeSurge": 3.0,          # High weight for volume spikes
    "VolumeProfile": 2.8,        # Volume profile analysis  
    "OIAnalysis": 2.5,           # Open Interest changes
    "Momentum": 2.3,             # Price momentum
    "VWAPBreakout": 2.2,         # VWAP breakouts
    "InstitutionalFlow": 2.0,    # Smart money flow
    "ADX": 1.9,                  # Trend strength
    "EMA": 1.8,                  # Moving average alignment
    "MACD": 1.7,                 # MACD signals
    "RSI": 1.5,                  # RSI momentum
    "ATR": 1.4,                  # Volatility expansion
    "Bollinger": 1.3,            # Bollinger band position
    "Stochastic": 1.2,           # Stochastic signals
    "ROC": 1.1,                  # Rate of change
    "OBV": 1.0,                  # On-balance volume
}

TIMEFRAME_WEIGHTS = {5: 3.0, 15: 2.5, 30: 2.0, 60: 1.5, "daily": 1.0}

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

# ---------- Enhanced Indicators for Option Buyers ----------
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

def volume_surge(df, lookback=20):
    vol_ma = df["Volume"].rolling(lookback).mean()
    vol_std = df["Volume"].rolling(lookback).std()
    z_score = (df["Volume"] - vol_ma) / vol_std.replace(0, np.nan)
    return z_score.fillna(0)

# Enhanced Volume Analysis for Option Buyers
def volume_profile_analysis(df, lookback=50):
    """Analyze volume distribution and buying/selling pressure"""
    if len(df) < lookback:
        return {"score": 0.0, "signal": "Neutral"}

    recent_vol = df["Volume"].tail(lookback)
    vol_mean = recent_vol.mean()
    vol_std = recent_vol.std()

    # Current volume vs average
    current_vol = df["Volume"].iloc[-1]
    vol_zscore = (current_vol - vol_mean) / (vol_std + 1e-9)

    # Price-volume relationship
    price_change = (df["Close"].iloc[-1] - df["Close"].iloc[-2]) / df["Close"].iloc[-2]

    # Volume breakout detection
    vol_breakout = vol_zscore > 2.0
    price_momentum = abs(price_change) > 0.01

    if vol_breakout and price_change > 0 and price_momentum:
        return {"score": 2.5, "signal": "Strong Bullish Volume"}
    elif vol_breakout and price_change < 0 and price_momentum:
        return {"score": -2.5, "signal": "Strong Bearish Volume"}
    elif vol_zscore > 1.0:
        return {"score": 1.0, "signal": "Moderate Volume"}
    else:
        return {"score": 0.0, "signal": "Normal Volume"}

# Open Interest Analysis (simulated based on volume patterns)
def oi_analysis(df, lookback=20):
    """Simulate OI analysis using volume and price patterns"""
    if len(df) < lookback:
        return {"score": 0.0, "signal": "Insufficient Data"}

    # Use volume as proxy for OI changes
    vol_trend = df["Volume"].rolling(5).mean().diff().iloc[-1]
    price_trend = df["Close"].diff().iloc[-1]

    # Volume-Price divergence analysis
    vol_increasing = vol_trend > 0
    price_increasing = price_trend > 0

    # Strong directional moves with volume support
    if vol_increasing and price_increasing:
        return {"score": 2.0, "signal": "Call Buying Interest"}
    elif vol_increasing and not price_increasing:
        return {"score": -2.0, "signal": "Put Buying Interest"}
    else:
        return {"score": 0.0, "signal": "Mixed Interest"}

# VWAP Breakout Analysis
def vwap_breakout_analysis(df):
    """Enhanced VWAP analysis for option buyers"""
    if len(df) < 50:
        return {"score": 0.0, "signal": "Insufficient Data"}

    vwap_line = vwap(df, period=None)
    current_price = df["Close"].iloc[-1]
    vwap_current = vwap_line.iloc[-1]

    # VWAP distance
    vwap_dist = (current_price - vwap_current) / vwap_current

    # Volume confirmation
    vol_surge_val = volume_surge(df, 20).iloc[-1]

    # Strong breakout above VWAP with volume
    if vwap_dist > 0.005 and vol_surge_val > 1.5:
        return {"score": 2.2, "signal": "VWAP Bullish Breakout"}
    elif vwap_dist < -0.005 and vol_surge_val > 1.5:
        return {"score": -2.2, "signal": "VWAP Bearish Breakout"}
    elif abs(vwap_dist) > 0.002:
        return {"score": 1.0 if vwap_dist > 0 else -1.0, "signal": "VWAP Deviation"}
    else:
        return {"score": 0.0, "signal": "Near VWAP"}

# Institutional Flow Detection (Enhanced)
def institutional_flow_detection(df):
    """Detect institutional buying/selling activity"""
    if len(df) < 30:
        return {"score": 0.0, "signal": "Insufficient Data"}

    # Large volume bars with sustained price movement
    vol_ma = df["Volume"].rolling(20).mean()
    vol_ratio = df["Volume"] / vol_ma

    # Price momentum
    momentum = (df["Close"] / df["Close"].shift(10) - 1) * 100

    # ATR for volatility context
    atr_val = atr(df, 14)
    price_change_normalized = abs(df["Close"].diff()) / atr_val

    # Look for institutional patterns
    large_volume = vol_ratio.iloc[-1] > 2.0
    sustained_move = momentum.iloc[-1] > 1.5
    efficient_move = price_change_normalized.iloc[-1] > 0.5

    if large_volume and sustained_move and efficient_move:
        direction = 1 if momentum.iloc[-1] > 0 else -1
        return {"score": 2.0 * direction, "signal": "Institutional " + ("Buying" if direction > 0 else "Selling")}
    elif large_volume:
        return {"score": 0.5, "signal": "Large Volume Activity"}
    else:
        return {"score": 0.0, "signal": "Regular Activity"}

# Standard indicators
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

def calculate_adx(df, period=14):
    if len(df) < period * 2: return pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64')
    df_adx = df.copy()
    df_adx['H-L'] = df_adx['High'] - df_adx['Low']
    df_adx['H-C'] = abs(df_adx['High'] - df_adx['Close'].shift(1))
    df_adx['L-C'] = abs(df_adx['Low'] - df_adx['Close'].shift(1))
    df_adx['TR'] = df_adx[['H-L', 'H-C', 'L-C']].max(axis=1)
    df_adx['+DM'] = np.where((df_adx['High'] - df_adx['High'].shift(1)) > (df_adx['Low'].shift(1) - df_adx['Low']), 
                             df_adx['High'] - df_adx['High'].shift(1), 0)
    df_adx['-DM'] = np.where((df_adx['Low'].shift(1) - df_adx['Low']) > (df_adx['High'] - df_adx['High'].shift(1)), 
                             df_adx['Low'].shift(1) - df_adx['Low'], 0)
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

def calculate_stochastic(df, period=14, smooth_d=3):
    if len(df) < period + smooth_d: return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan))
    k.fillna(50, inplace=True)
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_roc(df, period=12):
    if len(df) < period + 1: return pd.Series(dtype='float64')
    shifted_close = df['Close'].shift(period).replace(0, np.nan)
    return ((df['Close'] - df['Close'].shift(period)) / shifted_close) * 100

def calculate_obv(df):
    if len(df) < 2: return pd.Series(dtype='float64')
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

# ---------- Enhanced Scoring for Option Buyers ----------
def get_option_buyer_scores(df):
    """Calculate scores optimized for option buyer strategies"""
    scores = {}

    # Volume Analysis (High Weight)
    vol_analysis = volume_profile_analysis(df)
    scores["VolumeProfile"] = vol_analysis["score"]

    # Volume Surge
    zscore = volume_surge(df, lookback=20)
    if len(zscore) and pd.notna(zscore.iloc[-1]) and len(df) >= 2:
        price_up_last = df["Close"].iloc[-1] > df["Close"].iloc[-2]
        if zscore.iloc[-1] >= 2.0:
            scores["VolumeSurge"] = 2.5 if price_up_last else 1.0
        elif zscore.iloc[-1] >= 1.0:
            scores["VolumeSurge"] = 1.5 if price_up_last else 0.5
        else:
            scores["VolumeSurge"] = 0.0
    else:
        scores["VolumeSurge"] = 0.0

    # OI Analysis (simulated)
    oi_result = oi_analysis(df)
    scores["OIAnalysis"] = oi_result["score"]

    # VWAP Breakout
    vwap_result = vwap_breakout_analysis(df)
    scores["VWAPBreakout"] = vwap_result["score"]

    # Institutional Flow
    inst_result = institutional_flow_detection(df)
    scores["InstitutionalFlow"] = inst_result["score"]

    # Momentum (Enhanced for Option Buyers)
    if len(df) >= 10:
        momentum_5 = (df["Close"].iloc[-1] / df["Close"].iloc[-6] - 1) * 100  # 5-day momentum
        momentum_3 = (df["Close"].iloc[-1] / df["Close"].iloc[-4] - 1) * 100  # 3-day momentum

        if momentum_5 > 2.0 and momentum_3 > 1.0:
            scores["Momentum"] = 2.5
        elif momentum_5 > 1.0:
            scores["Momentum"] = 1.5
        elif momentum_5 < -2.0 and momentum_3 < -1.0:
            scores["Momentum"] = -2.5
        elif momentum_5 < -1.0:
            scores["Momentum"] = -1.5
        else:
            scores["Momentum"] = 0.0
    else:
        scores["Momentum"] = 0.0

    # RSI with Option Buyer Context
    rsi_series = calculate_rsi(df)
    if len(rsi_series) > 1 and pd.notna(rsi_series.iloc[-1]):
        rsi = rsi_series.iloc[-1]
        rsi_prev = rsi_series.iloc[-2]
        # Look for momentum breakouts
        if rsi > 55 and rsi_prev <= 50 and rsi < 75:  # Breaking above 50, not overbought
            scores['RSI'] = 2.0
        elif rsi < 45 and rsi_prev >= 50 and rsi > 25:  # Breaking below 50, not oversold
            scores['RSI'] = -2.0
        elif rsi > 60:
            scores['RSI'] = 1.0
        elif rsi < 40:
            scores['RSI'] = -1.0
        else:
            scores['RSI'] = 0.0
    else:
        scores['RSI'] = 0.0

    # MACD
    macd, signal = calculate_macd(df)
    if len(macd) and len(signal) and pd.notna(macd.iloc[-1]) and pd.notna(signal.iloc[-1]):
        macd_val = macd.iloc[-1]
        signal_val = signal.iloc[-1]
        if len(macd) > 1:
            macd_prev = macd.iloc[-2]
            signal_prev = signal.iloc[-2]
            # Look for crossovers
            if macd_val > signal_val and macd_prev <= signal_prev:
                scores['MACD'] = 2.0  # Bullish crossover
            elif macd_val < signal_val and macd_prev >= signal_prev:
                scores['MACD'] = -2.0  # Bearish crossover
            else:
                scores['MACD'] = 1.0 if macd_val > signal_val else -1.0
        else:
            scores['MACD'] = 1.0 if macd_val > signal_val else -1.0
    else:
        scores['MACD'] = 0.0

    # ADX - Trend Strength
    adx, pdi, ndi = calculate_adx(df)
    if len(adx) > 4 and pd.notna(adx.iloc[-1]):
        adx_val = adx.iloc[-1]
        pdi_val = pdi.iloc[-1] if pd.notna(pdi.iloc[-1]) else 20
        ndi_val = ndi.iloc[-1] if pd.notna(ndi.iloc[-1]) else 20

        if adx_val > 25:  # Strong trend
            if pdi_val > ndi_val:
                scores['ADX'] = 2.0  # Strong uptrend
            else:
                scores['ADX'] = -2.0  # Strong downtrend
        elif adx_val > 20:
            if pdi_val > ndi_val:
                scores['ADX'] = 1.0
            else:
                scores['ADX'] = -1.0
        else:
            scores['ADX'] = 0.0
    else:
        scores['ADX'] = 0.0

    # EMA Alignment
    ema_fast = ema(df["Close"], 12)
    ema_slow = ema(df["Close"], 26)
    if len(ema_fast) and len(ema_slow) and pd.notna(ema_fast.iloc[-1]) and pd.notna(ema_slow.iloc[-1]):
        if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            # Check if price is above both EMAs
            if df["Close"].iloc[-1] > ema_fast.iloc[-1]:
                scores["EMA"] = 2.0
            else:
                scores["EMA"] = 1.0
        else:
            if df["Close"].iloc[-1] < ema_fast.iloc[-1]:
                scores["EMA"] = -2.0
            else:
                scores["EMA"] = -1.0
    else:
        scores["EMA"] = 0.0

    # ATR - Volatility Expansion (Good for Option Buyers)
    atr_val = atr(df, period=14)
    if len(atr_val) >= 6 and all(pd.notna(val) for val in [atr_val.iloc[-1], atr_val.iloc[-6]]):
        atr_change = (atr_val.iloc[-1] / atr_val.iloc[-6] - 1) * 100
        if atr_change > 10:  # ATR expanding
            scores["ATR"] = 1.5
        elif atr_change < -10:  # ATR contracting
            scores["ATR"] = -0.5
        else:
            scores["ATR"] = 0.0
    else:
        scores["ATR"] = 0.0

    # Bollinger Bands
    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(df)
    if len(bb_middle) and pd.notna(bb_middle.iloc[-1]):
        close = df['Close'].iloc[-1]
        upper = bb_upper.iloc[-1] if pd.notna(bb_upper.iloc[-1]) else close
        lower = bb_lower.iloc[-1] if pd.notna(bb_lower.iloc[-1]) else close

        if close > upper:
            scores['Bollinger'] = 1.5  # Breakout above upper band
        elif close < lower:
            scores['Bollinger'] = -1.5  # Breakout below lower band
        elif close > bb_middle.iloc[-1]:
            scores['Bollinger'] = 0.5
        else:
            scores['Bollinger'] = -0.5
    else:
        scores['Bollinger'] = 0.0

    # Stochastic
    k, d = calculate_stochastic(df)
    if len(k) and len(d) and pd.notna(k.iloc[-1]) and pd.notna(d.iloc[-1]):
        if k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80:
            scores['Stochastic'] = 1.0
        elif k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 20:
            scores['Stochastic'] = -1.0
        else:
            scores['Stochastic'] = 0.0
    else:
        scores['Stochastic'] = 0.0

    # ROC
    roc = calculate_roc(df).iloc[-1] if len(df) else np.nan
    scores['ROC'] = 1.0 if pd.notna(roc) and roc > 0 else (-1.0 if pd.notna(roc) else 0.0)

    # OBV
    obv_line = calculate_obv(df)
    if len(obv_line) >= 2 and pd.notna(obv_line.iloc[-1]) and pd.notna(obv_line.iloc[-2]):
        scores['OBV'] = 1.0 if obv_line.iloc[-1] > obv_line.iloc[-2] else -1.0
    else:
        scores['OBV'] = 0.0

    # Set default scores for missing indicators
    for indicator in OPTION_BUYER_WEIGHTS.keys():
        if indicator not in scores:
            scores[indicator] = 0.0

    return scores

def analyze_option_signals(timeframe_dataframes):
    """Analyze signals specifically for option buyers"""
    final_score, max_possible = 0.0, 0.0
    signal_details = {}

    for tf_min, df in timeframe_dataframes.items():
        if df is None or len(df) < 50: 
            continue

        indicator_scores = get_option_buyer_scores(df)
        tf_weight = TIMEFRAME_WEIGHTS.get(tf_min, 1.0)

        for indicator, score in indicator_scores.items():
            ind_weight = OPTION_BUYER_WEIGHTS.get(indicator, 1.0)
            weighted_score = score * tf_weight * ind_weight
            final_score += weighted_score
            max_possible += 2.5 * tf_weight * ind_weight  # Max possible score per indicator

            # Store signal details for the primary timeframe (5min)
            if tf_min == 5:
                signal_details[indicator] = score

    if max_possible == 0: 
        return 'Neutral', 0.0, {}

    normalized = (final_score / max_possible) * 100.0

    # Option buyer specific thresholds
    if normalized >= 60:
        signal_text = 'Very Strong Call Signal'
    elif normalized >= 30:
        signal_text = 'Strong Call Signal'
    elif normalized <= -60:
        signal_text = 'Very Strong Put Signal'
    elif normalized <= -30:
        signal_text = 'Strong Put Signal'
    else:
        signal_text = 'Neutral'

    return signal_text, normalized, signal_details

def get_option_recommendation(signal, score, symbol):
    """Generate specific option trading recommendations"""
    if 'Call Signal' in signal:
        if score >= 60:
            return f"BUY {symbol} ATM/OTM CALLS (High Confidence)"
        else:
            return f"BUY {symbol} CALLS (Moderate Confidence)"
    elif 'Put Signal' in signal:
        if score <= -60:
            return f"BUY {symbol} ATM/OTM PUTS (High Confidence)"
        else:
            return f"BUY {symbol} PUTS (Moderate Confidence)"
    else:
        return f"HOLD/WATCH {symbol} (No Clear Direction)"

def get_key_signals(signal_details):
    """Extract top contributing signals"""
    sorted_signals = sorted(signal_details.items(), key=lambda x: abs(x[1]), reverse=True)
    return sorted_signals[:3]

# ---------- Data fetch and normalize ----------
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

    with tqdm(total=total_calls, desc="Fetching Option Trading Data", ncols=100) as api_bar:
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

# ---------- Main Execution Functions ----------
def run_live_option_scanner():
    """Run live 5-minute option buyer scanner"""
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

    previous_scores = {}
    output_filename = datetime.now(IST).strftime("%Y-%m-%d") + "_option_buyer_signals.txt"

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

        # Refresh data
        logger.info(f"[{now_ist.strftime('%H:%M:%S')}] Scanning {len(stocks)} stocks for option opportunities...")
        stock_multi_data = prefetch_all(stocks, max_workers=MAX_WORKERS)
        logger.info("Data refresh complete. Analyzing option signals...")

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

            signal, score, signal_details = analyze_option_signals(filtered_timeframes)
            current_scores[clean_symbol] = score

            if 'Signal' in signal and signal != 'Neutral':
                change = 'NA' if clean_symbol not in previous_scores else score - previous_scores.get(clean_symbol, 0.0)
                direction = 'Call' if 'Call' in signal else 'Put'
                recommendation = get_option_recommendation(signal, score, clean_symbol)
                key_signals = get_key_signals(signal_details)

                signals_this_scan.append({
                    'symbol': clean_symbol,
                    'signal': signal,
                    'score': score,
                    'direction': direction,
                    'change': change,
                    'recommendation': recommendation,
                    'key_signals': key_signals
                })

        # Sort by score
        signals_this_scan.sort(key=lambda x: abs(x['score']), reverse=True)
        top_call_signals = [r for r in signals_this_scan if 'Call' in r['direction']][:15]
        top_put_signals = [r for r in signals_this_scan if 'Put' in r['direction']][:15]

        # Console output
        width = 130
        print("\n" + "="*width)
        hdr = f"| LIVE OPTION BUYER SCANNER | {now_ist.strftime('%Y-%m-%d %H:%M')} IST"
        print(hdr.center(width+8) + " |")
        print("="*width)

        print(f"| {'Top 15 CALL Option Opportunities':<{width-4}} |")
        print("-"*width)
        if not top_call_signals:
            print("| No strong call signals found.".ljust(width-1) + " |")
        else:
            print(f"| {'Stock':<12} | {'Signal':<22} | {'Score':>7} | {'Change':>8} | {'Recommendation':<40} | {'Key Signals':<30} |")
            print("-"*width)
            for result in top_call_signals:
                key_sigs = ", ".join([f"{k}({v:.1f})" for k, v in result['key_signals']])[:28]
                change_val = result['change']
                if isinstance(change_val, (int, float, np.floating)):
                    change_str = f"{change_val:+.1f}"
                else:
                    change_str = "NA"
                print(f"| {result['symbol']:<12} | {result['signal']:<22} | {result['score']:>7.1f} | {change_str:>8} | {result['recommendation']:<40} | {key_sigs:<30} |")

        print("-"*width)
        print(f"| {'Top 15 PUT Option Opportunities':<{width-4}} |")
        print("-"*width)
        if not top_put_signals:
            print("| No strong put signals found.".ljust(width-1) + " |")
        else:
            print(f"| {'Stock':<12} | {'Signal':<22} | {'Score':>7} | {'Change':>8} | {'Recommendation':<40} | {'Key Signals':<30} |")
            print("-"*width)
            for result in top_put_signals:
                key_sigs = ", ".join([f"{k}({v:.1f})" for k, v in result['key_signals']])[:28]
                change_val = result['change']
                if isinstance(change_val, (int, float, np.floating)):
                    change_str = f"{change_val:+.1f}"
                else:
                    change_str = "NA"
                print(f"| {result['symbol']:<12} | {result['signal']:<22} | {result['score']:>7.1f} | {change_str:>8} | {result['recommendation']:<40} | {key_sigs:<30} |")
        print("="*width)

        # Save to file
        with open(output_filename, "a", encoding="utf-8") as f:
            f.write(f"===== Option Signals at {now_ist.strftime('%Y-%m-%d %H:%M')} =====\n\n")
            f.write("CALL OPTION OPPORTUNITIES:\n")
            for r in top_call_signals:
                key_sigs = ", ".join([f"{k}({v:.1f})" for k, v in r['key_signals']])
                change_str = f"{r['change']:+.1f}" if isinstance(r['change'], (int, float, np.floating)) else "NA"
                f.write(f"{r['symbol']:<12} | {r['signal']:<22} | {r['score']:>7.1f} | {change_str:>8} | {r['recommendation']:<40} | {key_sigs}\n")

            f.write("\nPUT OPTION OPPORTUNITIES:\n")
            for r in top_put_signals:
                key_sigs = ", ".join([f"{k}({v:.1f})" for k, v in r['key_signals']])
                change_str = f"{r['change']:+.1f}" if isinstance(r['change'], (int, float, np.floating)) else "NA"
                f.write(f"{r['symbol']:<12} | {r['signal']:<22} | {r['score']:>7.1f} | {change_str:>8} | {r['recommendation']:<40} | {key_sigs}\n")
            f.write("\n\n")

        previous_scores = {**previous_scores, **current_scores}

        # Sleep until next 5-min boundary
        nxt = next_5min_boundary_ist(datetime.now(IST))
        sleep_until(nxt + timedelta(seconds=SETTLE_DELAY_SECONDS))

def parse_asof(s: str):
    if 'T' in s:
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M")
    else:
        dt = datetime.strptime(s, "%Y-%m-%d")
        h, m = parse_hhmm(MARKET_END)
        dt = dt.replace(hour=h, minute=m)
    return IST.localize(dt)

def run_snapshot_analysis(asof_ts, stocks):
    """Run snapshot analysis at specific time"""
    logger.info(f"Running snapshot analysis at {asof_ts.strftime('%Y-%m-%d %H:%M')} IST...")
    stock_multi_data = prefetch_all(stocks, max_workers=MAX_WORKERS)

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

        signal, score, signal_details = analyze_option_signals(filtered_timeframes)

        if 'Signal' in signal and signal != 'Neutral':
            direction = 'Call' if 'Call' in signal else 'Put'
            recommendation = get_option_recommendation(signal, score, clean_symbol)
            key_signals = get_key_signals(signal_details)

            signals_this_scan.append({
                'symbol': clean_symbol,
                'signal': signal,
                'score': score,
                'direction': direction,
                'recommendation': recommendation,
                'key_signals': key_signals
            })

    # Sort and display results
    signals_this_scan.sort(key=lambda x: abs(x['score']), reverse=True)
    top_call_signals = [r for r in signals_this_scan if 'Call' in r['direction']][:20]
    top_put_signals = [r for r in signals_this_scan if 'Put' in r['direction']][:20]

    width = 130
    print("\n" + "="*width)
    hdr = f"| OPTION BUYER SNAPSHOT | {asof_ts.strftime('%Y-%m-%d %H:%M')} IST"
    print(hdr.center(width+8) + " |")
    print("="*width)

    print(f"| {'Top 20 CALL Opportunities':<{width-4}} |")
    print("-"*width)
    if not top_call_signals:
        print("| No call signals found.".ljust(width-1) + " |")
    else:
        print(f"| {'Stock':<12} | {'Signal':<22} | {'Score':>7} | {'Recommendation':<40} | {'Key Signals':<38} |")
        print("-"*width)
        for result in top_call_signals:
            key_sigs = ", ".join([f"{k}({v:.1f})" for k, v in result['key_signals']])[:36]
            print(f"| {result['symbol']:<12} | {result['signal']:<22} | {result['score']:>7.1f} | {result['recommendation']:<40} | {key_sigs:<38} |")

    print("-"*width)
    print(f"| {'Top 20 PUT Opportunities':<{width-4}} |")
    print("-"*width)
    if not top_put_signals:
        print("| No put signals found.".ljust(width-1) + " |")
    else:
        print(f"| {'Stock':<12} | {'Signal':<22} | {'Score':>7} | {'Recommendation':<40} | {'Key Signals':<38} |")
        print("-"*width)
        for result in top_put_signals:
            key_sigs = ", ".join([f"{k}({v:.1f})" for k, v in result['key_signals']])[:36]
            print(f"| {result['symbol']:<12} | {result['signal']:<22} | {result['score']:>7.1f} | {result['recommendation']:<40} | {key_sigs:<38} |")
    print("="*width)

# ---------- Main Function ----------
def main():
    parser = argparse.ArgumentParser(description="Enhanced Option Buyer Scanner with OI/Volume Analysis")
    parser.add_argument("--asof", type=str, default=None, help="Snapshot analysis at specific time (YYYY-MM-DD or YYYY-MM-DDTHH:MM)")
    args = parser.parse_args()

    # Load stock universe
    try:
        with open(SHARES_FILE, 'r') as f:
            stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(stocks)} symbols from {SHARES_FILE}")
    except Exception as e:
        raise SystemExit(f"Could not read {SHARES_FILE}: {e}")

    if args.asof:
        asof_ts = parse_asof(args.asof)
        run_snapshot_analysis(asof_ts, stocks)
    else:
        run_live_option_scanner()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOption scanner interrupted by user. Shutting down.")
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
