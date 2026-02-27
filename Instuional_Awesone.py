
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

# Rich table imports
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich.align import Align
    from rich import box
    from rich.live import Live
    from rich.layout import Layout
    RICH_AVAILABLE = True
except ImportError:
    print("Installing rich library for beautiful tables...")
    os.system("pip install rich")
    try:
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.columns import Columns
        from rich.text import Text
        from rich.align import Align
        from rich import box
        from rich.live import Live
        from rich.layout import Layout
        RICH_AVAILABLE = True
    except ImportError:
        RICH_AVAILABLE = False
        print("Rich library not available. Using plain tables.")

console = Console() if RICH_AVAILABLE else None

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

# ---------- IMPROVED WEIGHTS FOR BETTER OPTION SELECTION ----------
ENHANCED_INDICATOR_WEIGHTS = {
    # Volume and momentum prioritized for options
    "VolumeSurge": 4.0, "Breakout_Volume": 3.8, "Momentum": 3.5, 
    "OI_Momentum": 3.5, "Volume_OI_Ratio": 3.2, "Institutional_Flow": 3.0,

    # Price action indicators
    "VWAP": 2.8, "EMA": 2.5, "MACD": 2.3, "ADX": 2.2, "RSI": 2.0,

    # Support indicators  
    "OBV": 1.8, "ATR": 1.5, "Bollinger": 1.5, "ROC": 1.2,
    "Stochastic": 1.0, "CCI": 1.0, "MA": 1.0, "WWL": 1.0,

    # Enhanced option-specific
    "Option_Flow": 2.5, "Price_OI_Divergence": 2.2
}
INDICATOR_WEIGHTS = ENHANCED_INDICATOR_WEIGHTS | {
    "CMF": 2.5, "ADL": 2.0, "RelVol": 2.2, "VWAPRegime": 2.0, "OBVConfirm": 1.8
}
TIMEFRAME_WEIGHTS = {15: 4.0, 5: 3.5, 30: 3.0, 60: 2.0, "daily": 1.0}

BAR_SIZE_MAP = {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"}
DURATION_MAP = {5: "30 D", 15: "30 D", 30: "60 D", 60: "120 D", 1440: "365 D"}

# ---------- IMPROVED OPTION SELECTION CRITERIA ----------
OPTION_QUALITY_FILTERS = {
    'min_volume_ratio': 1.2,      # Minimum relative volume
    'min_price_momentum': 0.015,   # Minimum price movement %
    'min_score_strength': 15.0,    # Minimum score for consideration
    'liquidity_preference': True,  # Prefer liquid stocks
    'volatility_range': (0.5, 5.0), # ATR/Price range for good option premiums
}

# Liquid FNO stocks for better option trading
PREFERRED_OPTION_STOCKS = {
    'NIFTY_50': ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'HDFC', 'SBIN', 'BHARTIARTL', 'ITC', 'ASIANPAINT'],
    'BANK_NIFTY': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'INDUSINDBK', 'PNB', 'BANKBARODA'],
    'HIGH_VOLUME': ['TATASTEEL', 'TATAMOTORS', 'BAJFINANCE', 'LT', 'HCLTECH', 'WIPRO', 'MARUTI', 'POWERGRID'],
    'MID_CAP': ['ADANIPORTS', 'NTPC', 'ONGC', 'IOC', 'BPCL', 'HINDPETRO', 'COALINDIA', 'SAIL']
}

# 5-minute boundary helpers
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

def parse_asof(s: str):
    if 'T' in s:
        dt = datetime.strptime(s, "%Y-%m-%dT%H:%M")
    else:
        dt = datetime.strptime(s, "%Y-%m-%d")
        h, m = parse_hhmm(MARKET_END)
        dt = dt.replace(hour=h, minute=m)
    return IST.localize(dt)

def day_checkpoints_ist(day_date: datetime):
    d = day_date.date()
    start_h, start_m = parse_hhmm("09:20")
    end_h, end_m = parse_hhmm(MARKET_END)

    start_dt = IST.localize(datetime(d.year, d.month, d.day, start_h, start_m))
    end_dt = IST.localize(datetime(d.year, d.month, d.day, end_h, end_m))

    rng = pd.date_range(start=start_dt, end=end_dt, freq='5T', tz=IST, inclusive='both')
    return list(rng.to_pydatetime())

# Token-bucket limiter
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

# TrueData sessions
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

# ---------- IMPROVED Technical Indicators ----------
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def vwap(df, period=None):
    price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = price * df["Volume"]
    if period:
        pv_sum = pv.rolling(period).sum(); vol_sum = df["Volume"].rolling(period).sum()
    else:
        pv_sum = pv.cumsum(); vol_sum = df["Volume"].cumsum()
    return pv_sum / vol_sum.replace(0, np.nan)

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

def momentum(df, period=10):
    return df["Close"] / df["Close"].shift(period) - 1.0

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

def calculate_obv(df):
    if len(df) < 2: return pd.Series(dtype='float64')
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

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

def slope(series, lookback=10):
    if len(series) < lookback: return np.nan
    y = series.tail(lookback).values.astype(float)
    x = np.arange(len(y))
    x = (x - x.mean()) / (x.std() + 1e-9)
    A = np.vstack([x, np.ones_like(x)]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return m

# ---------- IMPROVED OPTION-SPECIFIC INDICATORS ----------
def calculate_option_quality_score(df, symbol):
    """Calculate option quality score based on liquidity, volatility, and momentum"""
    quality_score = 0.0

    try:
        # Volume quality (30% of score)
        rel_vol = relative_volume(df, 20)
        if len(rel_vol) and pd.notna(rel_vol.iloc[-1]):
            vol_score = min(30, rel_vol.iloc[-1] * 15)  # Cap at 30
            quality_score += vol_score

        # Volatility quality (25% of score) 
        atr_val = atr(df, 14)
        if len(atr_val) and pd.notna(atr_val.iloc[-1]):
            volatility = atr_val.iloc[-1] / df["Close"].iloc[-1]
            if OPTION_QUALITY_FILTERS['volatility_range'][0] <= volatility <= OPTION_QUALITY_FILTERS['volatility_range'][1]:
                quality_score += 25
            else:
                quality_score += 10  # Partial credit

        # Momentum quality (25% of score)
        momentum_val = momentum(df, 5)  # Shorter period for options
        if len(momentum_val) and pd.notna(momentum_val.iloc[-1]):
            mom_abs = abs(momentum_val.iloc[-1])
            if mom_abs >= OPTION_QUALITY_FILTERS['min_price_momentum']:
                quality_score += 25
            elif mom_abs >= OPTION_QUALITY_FILTERS['min_price_momentum'] * 0.5:
                quality_score += 15

        # Liquidity bonus (20% of score)
        symbol_clean = symbol.replace('-I', '').replace('-EQ', '')
        for category, stocks in PREFERRED_OPTION_STOCKS.items():
            if symbol_clean in stocks:
                quality_score += 20
                break
        else:
            quality_score += 5  # Small bonus for other stocks

    except Exception as e:
        logger.debug(f"Option quality calculation error for {symbol}: {e}")

    return min(100, quality_score)

def detect_breakout_volume_enhanced(df, lookback=20, volume_threshold=1.8):
    """Enhanced breakout detection with lower thresholds for more signals"""
    if len(df) < lookback + 5:
        return pd.Series(dtype='float64')

    vol_ma = df["Volume"].rolling(lookback).mean()
    vol_ratio = df["Volume"] / vol_ma.replace(0, np.nan)

    high_ma = df["High"].rolling(lookback).max()
    low_ma = df["Low"].rolling(lookback).min()

    # More sensitive breakout detection
    strong_breakout_up = df["Close"] > high_ma.shift(1) * 1.015    # 1.5% breakout
    medium_breakout_up = df["Close"] > high_ma.shift(1) * 1.008    # 0.8% breakout

    strong_breakout_down = df["Close"] < low_ma.shift(1) * 0.985   # -1.5% breakdown
    medium_breakout_down = df["Close"] < low_ma.shift(1) * 0.992   # -0.8% breakdown

    high_volume_surge = vol_ratio >= volume_threshold
    medium_volume_surge = vol_ratio >= (volume_threshold * 0.7)

    signal = pd.Series(0.0, index=df.index)

    signal[strong_breakout_up & high_volume_surge] = 3.5
    signal[medium_breakout_up & high_volume_surge] = 2.8
    signal[strong_breakout_up & medium_volume_surge] = 2.5
    signal[medium_breakout_up & medium_volume_surge] = 2.0

    signal[strong_breakout_down & high_volume_surge] = -3.5
    signal[medium_breakout_down & high_volume_surge] = -2.8
    signal[strong_breakout_down & medium_volume_surge] = -2.5
    signal[medium_breakout_down & medium_volume_surge] = -2.0

    return signal

def analyze_institutional_flow_enhanced(df, short_period=10, long_period=40):
    """Enhanced institutional flow with better sensitivity"""
    if len(df) < long_period:
        return pd.Series(dtype='float64')

    vwap_short = vwap(df, period=short_period)
    vwap_long = vwap(df, period=long_period)

    volume_delta = df["Volume"].rolling(short_period).sum() - df["Volume"].rolling(long_period).mean() * short_period

    price_above_vwap = df["Close"] > vwap_short
    price_strength = (df["Close"] - vwap_short) / vwap_short.replace(0, np.nan)

    # More sensitive thresholds
    strong_institutional_buying = (
        price_above_vwap & 
        (volume_delta > volume_delta.rolling(15).mean()) & 
        (vwap_short > vwap_long) &
        (price_strength > 0.003)  # Lowered from 0.005
    )

    medium_institutional_buying = (
        price_above_vwap & 
        (volume_delta > 0) & 
        (vwap_short > vwap_long) &
        (price_strength > 0.001)  # Lowered from 0.002
    )

    strong_institutional_selling = (
        ~price_above_vwap & 
        (volume_delta > volume_delta.rolling(15).mean()) & 
        (vwap_short < vwap_long) &
        (price_strength < -0.003)  # Adjusted
    )

    medium_institutional_selling = (
        ~price_above_vwap & 
        (volume_delta > 0) & 
        (vwap_short < vwap_long) &
        (price_strength < -0.001)  # Adjusted
    )

    flow_signal = pd.Series(0.0, index=df.index)
    flow_signal[strong_institutional_buying] = 3.0   # Increased
    flow_signal[medium_institutional_buying] = 2.0   # Increased
    flow_signal[strong_institutional_selling] = -3.0 # Increased
    flow_signal[medium_institutional_selling] = -2.0 # Increased

    return flow_signal

def calculate_option_momentum_score_enhanced(df, fast=8, slow=21):
    """Enhanced momentum specifically for options with shorter periods"""
    if len(df) < slow + 10:
        return pd.Series(dtype='float64')

    ema_fast = ema(df["Close"], fast)
    ema_slow = ema(df["Close"], slow)

    vol_momentum = volume_surge(df, lookback=15)  # Shorter lookback
    vol_ma = df["Volume"].rolling(15).mean()
    current_vol_ratio = df["Volume"] / vol_ma.replace(0, np.nan)

    price_momentum = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)

    momentum_score = price_momentum * (1 + vol_momentum.abs() * 0.3) * current_vol_ratio
    momentum_score = momentum_score * 3.0  # Increased multiplier

    return momentum_score.fillna(0)

# ---------- ENHANCED SCORING SYSTEM ----------
def get_enhanced_scores_improved(df, symbol):
    """Improved scoring system with option quality consideration"""
    scores = {}

    # Calculate option quality score
    option_quality = calculate_option_quality_score(df, symbol)
    quality_multiplier = 1.0 + (option_quality / 100.0)  # 1.0 to 2.0 multiplier

    # Enhanced RSI with lower thresholds
    try:
        rsi_series = calculate_rsi(df)
        if len(rsi_series) > 1 and pd.notna(rsi_series.iloc[-1]):
            rsi = rsi_series.iloc[-1]; prev_rsi = rsi_series.iloc[-2]
            if rsi > 60 and prev_rsi <= 60: scores['RSI'] = 2.5 * quality_multiplier
            elif rsi > 52: scores['RSI'] = 1.5 * quality_multiplier
            elif rsi < 40 and prev_rsi >= 40: scores['RSI'] = -2.5 * quality_multiplier
            elif rsi < 48: scores['RSI'] = -1.5 * quality_multiplier
            else: scores['RSI'] = 0.0
        else: scores['RSI'] = 0.0
    except: scores['RSI'] = 0.0

    # Enhanced breakout detection
    try:
        breakout_signal = detect_breakout_volume_enhanced(df)
        if len(breakout_signal) and pd.notna(breakout_signal.iloc[-1]):
            scores['Breakout_Volume'] = breakout_signal.iloc[-1] * quality_multiplier
        else: scores['Breakout_Volume'] = 0.0
    except: scores['Breakout_Volume'] = 0.0

    # Enhanced institutional flow
    try:
        inst_flow = analyze_institutional_flow_enhanced(df)
        if len(inst_flow) and pd.notna(inst_flow.iloc[-1]):
            scores['Institutional_Flow'] = inst_flow.iloc[-1] * quality_multiplier
        else: scores['Institutional_Flow'] = 0.0
    except: scores['Institutional_Flow'] = 0.0

    # Enhanced option momentum
    try:
        opt_momentum = calculate_option_momentum_score_enhanced(df)
        if len(opt_momentum) and pd.notna(opt_momentum.iloc[-1]):
            momentum_val = opt_momentum.iloc[-1]
            if momentum_val > 0.015: scores['OI_Momentum'] = 2.8 * quality_multiplier
            elif momentum_val > 0.008: scores['OI_Momentum'] = 1.8 * quality_multiplier
            elif momentum_val < -0.015: scores['OI_Momentum'] = -2.8 * quality_multiplier
            elif momentum_val < -0.008: scores['OI_Momentum'] = -1.8 * quality_multiplier
            else: scores['OI_Momentum'] = 0.0
        else: scores['OI_Momentum'] = 0.0
    except: scores['OI_Momentum'] = 0.0

    # Enhanced MACD
    try:
        macd, signal_line = calculate_macd(df)
        if len(macd) and len(signal_line) and pd.notna(macd.iloc[-1]) and pd.notna(signal_line.iloc[-1]):
            macd_diff = macd.iloc[-1] - signal_line.iloc[-1]
            if macd_diff > 0 and len(macd) > 1 and macd.iloc[-1] > macd.iloc[-2]: 
                scores['MACD'] = 2.2 * quality_multiplier
            elif macd_diff > 0: 
                scores['MACD'] = 1.5 * quality_multiplier
            elif macd_diff < 0 and len(macd) > 1 and macd.iloc[-1] < macd.iloc[-2]: 
                scores['MACD'] = -2.2 * quality_multiplier
            else: 
                scores['MACD'] = -1.5 * quality_multiplier
        else: scores['MACD'] = 0.0
    except: scores['MACD'] = 0.0

    # Enhanced volume surge
    try:
        vol_surge_val = volume_surge(df, lookback=15)
        if len(vol_surge_val) and pd.notna(vol_surge_val.iloc[-1]):
            if vol_surge_val.iloc[-1] > 1.2: scores['VolumeSurge'] = 2.5 * quality_multiplier
            elif vol_surge_val.iloc[-1] > 0.6: scores['VolumeSurge'] = 1.5 * quality_multiplier
            else: scores['VolumeSurge'] = 0.0
        else: scores['VolumeSurge'] = 0.0
    except: scores['VolumeSurge'] = 0.0

    # Enhanced momentum
    try:
        momentum_val = momentum(df, period=8)
        if len(momentum_val) and pd.notna(momentum_val.iloc[-1]):
            if momentum_val.iloc[-1] > 0.02: scores['Momentum'] = 2.0 * quality_multiplier
            elif momentum_val.iloc[-1] < -0.02: scores['Momentum'] = -2.0 * quality_multiplier
            else: scores['Momentum'] = 0.0
        else: scores['Momentum'] = 0.0
    except: scores['Momentum'] = 0.0

    # VWAP scoring
    try:
        vwap_val = vwap(df, period=20)
        if len(vwap_val) and pd.notna(vwap_val.iloc[-1]):
            current_price = df["Close"].iloc[-1]
            vwap_price = vwap_val.iloc[-1]
            if current_price > vwap_price * 1.002: scores['VWAP'] = 1.5 * quality_multiplier
            elif current_price < vwap_price * 0.998: scores['VWAP'] = -1.5 * quality_multiplier
            else: scores['VWAP'] = 0.0
        else: scores['VWAP'] = 0.0
    except: scores['VWAP'] = 0.0

    # EMA crossover
    try:
        ema_fast = ema(df["Close"], 12)
        ema_slow = ema(df["Close"], 26)
        if len(ema_fast) and len(ema_slow) and pd.notna(ema_fast.iloc[-1]) and pd.notna(ema_slow.iloc[-1]):
            scores["EMA"] = (1.2 * quality_multiplier) if ema_fast.iloc[-1] > ema_slow.iloc[-1] else (-1.2 * quality_multiplier)
        else: scores["EMA"] = 0.0
    except: scores["EMA"] = 0.0

    # Add remaining indicators with basic scoring
    for indicator in ['ADX', 'Bollinger', 'ROC', 'OBV', 'CCI', 'Stochastic', 'MA', 'WWL', 'ATR', 
                      'Volume_OI_Ratio', 'Option_Flow', 'Price_OI_Divergence', 'CMF', 'ADL', 
                      'RelVol', 'VWAPRegime', 'OBVConfirm']:
        if indicator not in scores:
            scores[indicator] = 0.0

    return scores, option_quality

# ---------- RICH TABLE DISPLAY FUNCTIONS ----------
def create_rich_option_table(signals, title, signal_type="call"):
    """Create beautiful rich table for option signals"""
    if not RICH_AVAILABLE:
        return None

    # Color scheme
    if signal_type == "call":
        title_color = "bright_green"
        border_color = "green"
        signal_color = "bright_green"
        score_color = "bright_cyan"
    else:
        title_color = "bright_red" 
        border_color = "red"
        signal_color = "bright_red"
        score_color = "bright_magenta"

    table = Table(
        title=f"[bold {title_color}]{title}[/bold {title_color}]",
        box=box.ROUNDED,
        border_style=border_color,
        header_style="bold white on blue",
        show_header=True,
        show_lines=True,
        expand=True
    )

    # Add columns
    table.add_column("Stock", style="bold white", width=12, justify="center")
    table.add_column("Signal", style=f"bold {signal_color}", width=16, justify="center")
    table.add_column("Score", style=f"bold {score_color}", width=8, justify="right")
    table.add_column("Option Action", style="bold yellow", width=20, justify="center")
    table.add_column("Entry Strategy", style="cyan", width=35)
    table.add_column("Confidence", style="bold magenta", width=12, justify="center")
    table.add_column("Quality", style="bold green", width=8, justify="center")

    if not signals:
        table.add_row(
            "[dim]No signals[/dim]", 
            "[dim]detected[/dim]", 
            "[dim]0.0[/dim]", 
            "[dim]WAIT[/dim]", 
            "[dim]Wait for clearer signals[/dim]", 
            "[dim]None[/dim]",
            "[dim]N/A[/dim]"
        )
    else:
        for r in signals:
            # Color coding for scores
            score_val = r['score']
            if abs(score_val) >= 35:
                score_style = "bold bright_white on green" if score_val > 0 else "bold bright_white on red"
            elif abs(score_val) >= 25:
                score_style = "bold green" if score_val > 0 else "bold red"
            else:
                score_style = "yellow" if score_val > 0 else "orange3"

            # Color coding for confidence
            conf_color = {
                'Very High': 'bold bright_green',
                'High': 'bold green', 
                'Medium': 'bold yellow',
                'Low': 'bold orange3'
            }.get(r['confidence'], 'white')

            # Quality score coloring
            quality = r.get('quality', 0)
            if quality >= 80:
                quality_style = "bold bright_green"
            elif quality >= 60:
                quality_style = "bold green"
            elif quality >= 40:
                quality_style = "bold yellow"
            else:
                quality_style = "bold orange3"

            table.add_row(
                f"[bold white]{r['symbol'].replace('-I', '')}[/bold white]",
                f"[{signal_color}]{r['signal']}[/{signal_color}]",
                f"[{score_style}]{r['score']:>6.1f}[/{score_style}]",
                f"[bold yellow]{r['option_action']}[/bold yellow]",
                f"[cyan]{r['entry_strategy'][:35]}[/cyan]",
                f"[{conf_color}]{r['confidence']}[/{conf_color}]",
                f"[{quality_style}]{quality:.0f}[/{quality_style}]"
            )

    return table

def display_rich_results(signals_this_scan, timestamp):
    """Display results using rich tables"""
    if not RICH_AVAILABLE:
        print_plain_results(signals_this_scan, timestamp)
        return

    # Separate signals
    top_bullish = [r for r in signals_this_scan if 'Buy' in r['signal']][:12]
    top_bearish = [r for r in signals_this_scan if 'Sell' in r['signal']][:12]

    # Create header panel
    header = Panel(
        f"[bold bright_blue]🚀 ENHANCED OPTIONS SCANNER - CALL/PUT SIGNALS 🚀[/bold bright_blue]\n"
        f"[bold white]{timestamp} IST[/bold white] | "
        f"[green]📈 {len(top_bullish)} Call Signals[/green] | "
        f"[red]📉 {len(top_bearish)} Put Signals[/red]",
        border_style="bright_blue",
        padding=(1, 2)
    )

    # Create tables
    call_table = create_rich_option_table(top_bullish, "🔥 CALL BUYING OPPORTUNITIES 🔥", "call")
    put_table = create_rich_option_table(top_bearish, "💥 PUT BUYING OPPORTUNITIES 💥", "put")

    # Clear screen and display
    console.clear()
    console.print(header)
    console.print()
    console.print(call_table)
    console.print()
    console.print(put_table)
    console.print()

    # Add footer with stats
    total_signals = len(top_bullish) + len(top_bearish)
    footer = Panel(
        f"[bold cyan]📊 Scan Summary: {total_signals} Total Signals | "
        f"💹 Ready for Option Trading | "
        f"⚡ Next scan in 5 minutes[/bold cyan]",
        border_style="cyan",
        padding=(0, 2)
    )
    console.print(footer)

def print_plain_results(signals_this_scan, timestamp):
    """Fallback plain text display"""
    width = 140
    print("\n" + "="*width)
    hdr = f"| ENHANCED OPTIONS SCANNER - CALL/PUT SIGNALS | {timestamp} IST"
    print(hdr.center(width+8) + " |")
    print("="*width)

    top_bullish = [r for r in signals_this_scan if 'Buy' in r['signal']][:12]
    top_bearish = [r for r in signals_this_scan if 'Sell' in r['signal']][:12]

    print(f"| {'CALL BUYING OPPORTUNITIES':<{width-4}} |")
    print("-"*width)
    if not top_bullish:
        print("| No strong call signals detected".ljust(width-1) + " |")
    else:
        print(f"| {'Stock':<12} | {'Signal':<16} | {'Score':>6} | {'Option Action':<20} | {'Entry Strategy':<35} | {'Conf':<8} | {'Qual':>4} |")
        print("-"*width)
        for r in top_bullish:
            quality = r.get('quality', 0)
            entry_short = r['entry_strategy'][:35]
            print(f"| {r['symbol']:<12} | {r['signal']:<16} | {r['score']:>6.1f} | {r['option_action']:<20} | {entry_short:<35} | {r['confidence']:<8} | {quality:>4.0f} |")

    print("\n" + "-"*width)
    print(f"| {'PUT BUYING OPPORTUNITIES':<{width-4}} |")
    print("-"*width)
    if not top_bearish:
        print("| No strong put signals detected".ljust(width-1) + " |")
    else:
        print(f"| {'Stock':<12} | {'Signal':<16} | {'Score':>6} | {'Option Action':<20} | {'Entry Strategy':<35} | {'Conf':<8} | {'Qual':>4} |")
        print("-"*width)
        for r in top_bearish:
            quality = r.get('quality', 0)
            entry_short = r['entry_strategy'][:35]
            print(f"| {r['symbol']:<12} | {r['signal']:<16} | {r['score']:>6.1f} | {r['option_action']:<20} | {entry_short:<35} | {r['confidence']:<8} | {quality:>4.0f} |")

    print("="*width)

# ---------- IMPROVED OPTION RECOMMENDATION SYSTEM ----------
def generate_enhanced_option_recommendation(signal, score, symbol, flow_analysis="Unknown", quality_score=0):
    """Enhanced option recommendations with quality consideration"""
    recommendations = {
        'symbol': symbol,
        'signal': signal,
        'score': score,
        'flow': flow_analysis,
        'quality': quality_score,
        'recommendation': '',
        'confidence': 'Low',
        'entry_strategy': '',
        'risk_level': 'Medium'
    }

    # Enhanced confidence thresholds with quality adjustment
    quality_bonus = quality_score / 100.0  # 0.0 to 1.0
    adjusted_score = abs(score) * (1 + quality_bonus * 0.3)  # Up to 30% bonus

    if adjusted_score >= 30:  # Was 35
        recommendations['confidence'] = 'Very High'
        recommendations['risk_level'] = 'High Reward'
    elif adjusted_score >= 20:  # Was 25
        recommendations['confidence'] = 'High'
        recommendations['risk_level'] = 'Medium-High'
    elif adjusted_score >= 12:  # Was 15
        recommendations['confidence'] = 'Medium'
    else:
        recommendations['confidence'] = 'Low'
        recommendations['risk_level'] = 'Conservative'

    # Enhanced strike selection based on score and quality
    if 'Buy' in signal:
        if score >= 30 and quality_score >= 70:
            recommendations['recommendation'] = 'BUY CALLS (ATM/ITM)'
            recommendations['entry_strategy'] = 'Strong signal: Buy ATM calls aggressively'
        elif score >= 20 and quality_score >= 50:
            recommendations['recommendation'] = 'BUY CALLS (Slightly OTM)'
            recommendations['entry_strategy'] = 'Good signal: Buy 1-2 strikes OTM calls'
        elif score >= 12:
            recommendations['recommendation'] = 'BUY CALLS (OTM)'
            recommendations['entry_strategy'] = 'Moderate signal: Buy 2-3 strikes OTM calls'
        else:
            recommendations['recommendation'] = 'BUY CALLS (Conservative)'
            recommendations['entry_strategy'] = 'Weak signal: Small position, tight stops'
    elif 'Sell' in signal:
        if score <= -30 and quality_score >= 70:
            recommendations['recommendation'] = 'BUY PUTS (ATM/ITM)'
            recommendations['entry_strategy'] = 'Strong signal: Buy ATM puts aggressively'
        elif score <= -20 and quality_score >= 50:
            recommendations['recommendation'] = 'BUY PUTS (Slightly OTM)'
            recommendations['entry_strategy'] = 'Good signal: Buy 1-2 strikes OTM puts'
        elif score <= -12:
            recommendations['recommendation'] = 'BUY PUTS (OTM)'
            recommendations['entry_strategy'] = 'Moderate signal: Buy 2-3 strikes OTM puts'
        else:
            recommendations['recommendation'] = 'BUY PUTS (Conservative)'
            recommendations['entry_strategy'] = 'Weak signal: Small position, tight stops'
    else:
        recommendations['recommendation'] = 'HOLD/NO ACTION'
        recommendations['entry_strategy'] = 'Wait for clearer signals'
        recommendations['risk_level'] = 'No Risk'

    return recommendations

# ---------- IMPROVED SIGNAL ANALYSIS ----------
def analyze_signals_improved(timeframe_dataframes, symbol):
    """Improved signal analysis with quality consideration"""
    final_score, max_possible = 0.0, 0.0

    for tf_min, df in timeframe_dataframes.items():
        if df is None or len(df) < 40:  # Reduced from 50
            continue

        indicator_scores, quality = get_enhanced_scores_improved(df, symbol)
        tf_weight = TIMEFRAME_WEIGHTS.get(tf_min, 1.0)

        for indicator, score in indicator_scores.items():
            ind_weight = INDICATOR_WEIGHTS.get(indicator, 1.0)
            weighted_score = score * tf_weight * ind_weight
            final_score += weighted_score
            max_possible += abs(weighted_score) if weighted_score != 0 else 1.0

    if max_possible == 0: 
        return 'Neutral', 0.0, 0

    normalized = (final_score / max_possible) * 100.0

    # Calculate average quality across timeframes
    avg_quality = 0
    valid_tfs = 0
    for tf_min, df in timeframe_dataframes.items():
        if df is not None and len(df) >= 40:
            avg_quality += calculate_option_quality_score(df, symbol)
            valid_tfs += 1

    avg_quality = avg_quality / valid_tfs if valid_tfs > 0 else 0

    # More aggressive thresholds for options
    if normalized >= 25: signal_text = 'Very Strong Buy'     # Was 40
    elif normalized >= 12: signal_text = 'Strong Buy'        # Was 20
    elif normalized <= -25: signal_text = 'Very Strong Sell' # Was -40
    elif normalized <= -12: signal_text = 'Strong Sell'      # Was -20
    else: signal_text = 'Neutral'

    return signal_text, normalized, avg_quality

# ---------- DATA PROCESSING (SIMPLIFIED FOR BREVITY) ----------
def normalize_hist_df(df, symbol):
    if df is None or len(df) == 0: return None
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
            if isinstance(out.index, pd.DatetimeIndex): out["Date"] = out.index
            else: return None

        if "Volume" not in out.columns: out["Volume"] = 0
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
        return out if len(out) >= 40 else None
    except Exception as e:
        logger.error(f"Normalize error {symbol}: {e}")
        return None

def pick_session(symbol_orig, timeframe_minutes):
    return (hash(symbol_orig) ^ timeframe_minutes) % len(tdhist_pool)

def fetch_one(symbol_orig, timeframe_minutes, limiter, hist):
    td_symbol = symbol_orig.replace('-EQ', '')
    bar_size = BAR_SIZE_MAP.get(timeframe_minutes)
    duration = DURATION_MAP.get(timeframe_minutes)
    if not bar_size or not duration: return symbol_orig, timeframe_minutes, None
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
    with api_calls_lock: api_calls_done = 0

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

# ---------- ENHANCED BACKTEST FUNCTION ----------
def run_backtest_day_enhanced(day_str: str, stocks):
    """Enhanced backtest with rich tables and improved selection"""
    day_date = datetime.strptime(day_str, "%Y-%m-%d")
    logger.info(f"{day_str}: Enhanced Rich Options backtest for {len(stocks)} symbols...")

    stock_multi_data = prefetch_all(stocks, max_workers=MAX_WORKERS)
    logger.info("Prefetch complete. Running enhanced checkpoints with Rich tables...")

    checkpoints = day_checkpoints_ist(day_date)

    for checkpoint_idx, asof_ts in enumerate(checkpoints):
        time_point_aware = asof_ts.replace(second=0, microsecond=0)
        signals_this_scan = []

        for symbol, timeframe_data in stock_multi_data.items():
            clean_symbol = symbol.replace('-EQ', '')
            filtered_timeframes = {}

            for tf, df in timeframe_data.items():
                if df is None or df.empty: continue
                df_clean = df.sort_index()
                df_clean = df_clean[~df_clean.index.duplicated(keep='last')]
                df_slice = df_clean[df_clean.index <= time_point_aware]
                if not df_slice.empty and len(df_slice) >= 40:
                    filtered_timeframes[tf] = df_slice

            if len(filtered_timeframes) < 2: continue

            signal, score, quality = analyze_signals_improved(filtered_timeframes, clean_symbol)

            if signal != 'Neutral':
                direction = 'bullish' if 'Buy' in signal else 'bearish'

                rec = generate_enhanced_option_recommendation(
                    signal, score, clean_symbol, "Enhanced", quality
                )

                signals_this_scan.append({
                    'symbol': clean_symbol, 'signal': signal, 'score': score,
                    'trend': direction, 'flow': 'Enhanced', 'quality': quality,
                    'option_action': rec['recommendation'],
                    'entry_strategy': rec['entry_strategy'],
                    'confidence': rec['confidence'],
                    'risk_level': rec['risk_level']
                })

        # Sort by quality-adjusted score
        signals_this_scan.sort(key=lambda x: abs(x['score']) * (1 + x['quality']/100), reverse=True)

        # Display using rich tables
        timestamp = asof_ts.strftime('%Y-%m-%d %H:%M')
        display_rich_results(signals_this_scan, timestamp)

        # Small delay to see the results
        time.sleep(0.5)

def main():
    """Enhanced main function"""
    parser = argparse.ArgumentParser(description="Enhanced Options Scanner with Rich Tables")
    parser.add_argument("--mode", choices=["live", "backtest", "snapshot"], default="backtest")
    parser.add_argument("--date", type=str, help="Date for backtest mode (YYYY-MM-DD)")
    parser.add_argument("--universe-file", type=str, help="Path to universe file")

    args = parser.parse_args()

    if args.universe_file:
        global SHARES_FILE
        SHARES_FILE = args.universe_file

    if RICH_AVAILABLE:
        console.print(Panel(
            "[bold bright_blue]🚀 ENHANCED OPTIONS SCANNER WITH RICH TABLES 🚀[/bold bright_blue]\n"
            "[bold green]Features: Beautiful colored tables, improved option selection, quality scoring[/bold green]",
            border_style="bright_blue"
        ))

    logger.info("Enhanced Options Scanner with Rich Tables starting...")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Rich tables: {'Available' if RICH_AVAILABLE else 'Not available'}")

    try:
        with open(SHARES_FILE, 'r') as f:
            stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(stocks)} symbols from {SHARES_FILE}")
    except Exception as e:
        raise SystemExit(f"Could not read {SHARES_FILE}: {e}")

    try:
        if args.mode == "backtest":
            if not args.date:
                logger.error("--date required for backtest mode")
                return
            run_backtest_day_enhanced(args.date, stocks)
    except KeyboardInterrupt:
        logger.info("Enhanced scanner terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
    finally:
        logger.info("Enhanced scanner shutdown complete")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if RICH_AVAILABLE:
            console.print("[bold red]Scanner interrupted by user[/bold red]")
        else:
            print("Scanner interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
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
