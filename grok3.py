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
from truedata import TD_hist

# Rich for colored tables
from rich.console import Console
from rich.table import Table
from rich import box

# ======== COMPLETE CONFIG WITH LIVE MARKET SUPPORT ========
CONFIG = {
    "TDUSERNAME": os.getenv("TRUEDATA_USER", "tdwsp751"),
    "TDPASSWORD": os.getenv("TRUEDATA_PASS", "raj@751"),

    # Market times (IST)
    "MARKET_START": "09:15",
    "FIRST_RUN_AT": "09:20",
    "MARKET_END": "15:30",
    "SETTLE_DELAY_SECONDS": 5,

    # IMPROVED Concurrency and rate - optimized for stability
    "MAX_WORKERS": int(os.getenv("MAX_WORKERS", "32")),
    "TD_HIST_SESSIONS": int(os.getenv("TD_HIST_SESSIONS", "4")),
    "RATE_PER_SECOND_TOTAL": float(os.getenv("RATE_PER_SECOND_TOTAL", "15.0")),
    "BUCKET_SIZE": int(os.getenv("BUCKET_SIZE", "20")),
    "RETRY_ATTEMPTS": int(os.getenv("RETRY_ATTEMPTS", "5")),
    "RETRY_DELAY_MS": int(os.getenv("RETRY_DELAY_MS", "2000")),

    # Output and logging
    "SHARES_FILE": os.getenv("SHARES_FILE", "shares.txt"),
    "SHOW_PROGRESS": os.getenv("SHOW_PROGRESS", "true").lower() == "true",
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),

    # Data choices
    "SKIP_DAILY": os.getenv("SKIP_DAILY", "true").lower() == "true",

    # Data quality thresholds for consistency
    "MIN_BARS_REQUIRED": 50,
    "MAX_MISSING_DATA_PCT": 10,
    "SIGNAL_CONFIRMATION_BARS": 3,

    # IMPROVED Indicator settings with consistent periods
    "INDICATOR_PERIODS": {
        "RSI": 14,
        "MACD_FAST": 12,
        "MACD_SLOW": 26,
        "MACD_SIGNAL": 9,
        "STOCHASTIC_K": 14,
        "STOCHASTIC_D": 3,
        "MA_SHORT": 20,
        "MA_LONG": 50,
        "ADX": 14,
        "BB_PERIOD": 20,
        "BB_STD_DEV": 2,
        "ROC": 12,
        "CCI": 20,
        "EMA_FAST": 9,
        "EMA_SLOW": 21,
        "ATR": 14,
        "VOLUME_SURGE": 20,
        "MOMENTUM": 10,
        "WILLIAMS_R": 14,
        "CMF": 20,
        "ADL_LOOKBACK": 10,
        "REL_VOL": 20,
        "VWAP_REGIME": 20,
        "OBV_CONFIRM": 5,
        "OI_SURGE": 20,
        "OI_MOMENTUM": 10,
    },
    
    # ENHANCED OI-focused weights for option trading
    "INDICATOR_WEIGHTS": {
        "VolumeSurge": 2.5,
        "Momentum": 2.2,
        "ADX": 2.0,
        "VWAP": 1.8,
        "EMA": 1.9,
        "MACD": 1.7,
        "OBV": 1.6,
        "ATR": 1.5,
        "Bollinger": 1.4,
        "RSI": 1.3,
        "ROC": 1.2,
        "Stochastic": 1.1,
        "CCI": 1.0,
        "MA": 1.2,
        "WWL": 1.0,
        "CMF": 2.0,
        "ADL": 1.8,
        "RelVol": 1.7,
        "VWAPRegime": 1.9,
        "OBVConfirm": 1.4,
        # OI indicators get HIGHEST weights for option buyers
        "OISurge": 3.5,
        "OIMomentum": 3.2,
        "CallBias": 4.0,
        "PutBias": 4.0,
        "OIVolConfirm": 3.0,
    },
    
    # IMPROVED Timeframe weights - more balanced for consistent signals
    "TIMEFRAME_WEIGHTS": {15: 2.5, 5: 2.2, 30: 1.8, 60: 1.2, "daily": 1.0},

    # Broker API mapping
    "BAR_SIZE_MAP": {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"},
    "DURATION_MAP": {5: "45 D", 15: "45 D", 30: "90 D", 60: "180 D", 1440: "365 D"},
}

# Root logger threshold
level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
logging.getLogger().setLevel(level_map.get(CONFIG["LOG_LEVEL"], logging.INFO))

IST = pytz.timezone("Asia/Kolkata")

# Silence noisy third-party loggers
for noisy in ("truedata", "truedata.history", "truedata_ws", "websocket", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

console = Console()

# Global state
last_bull_symbols = set()
last_bear_symbols = set()
previous_scores = {}
api_calls_done = 0
api_calls_lock = threading.Lock()
performance_metrics = defaultdict(int)
failed_symbols = set()
oi_symbols_found = set()

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

def day_checkpoints_ist(day_date: datetime):
    d = day_date.date()
    start_h, start_m = parse_hhmm(CONFIG["FIRST_RUN_AT"])
    end_h, end_m = parse_hhmm(CONFIG["MARKET_END"])
    start_dt = IST.localize(datetime(d.year, d.month, d.day, start_h, start_m))
    end_dt = IST.localize(datetime(d.year, d.month, d.day, end_h, end_m))
    rng = pd.date_range(start=start_dt, end=end_dt, freq="5T", tz=IST, inclusive="both")
    return list(rng.to_pydatetime())

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
    # Distribute aggregate rate across sessions
    per_sess_rate = CONFIG["RATE_PER_SECOND_TOTAL"] / len(pool)
    for _ in pool:
        limiters.append(TokenBucketLimiter(rate_per_sec=per_sess_rate, bucket_size=CONFIG["BUCKET_SIZE"]))
    return pool, limiters

tdhist_pool, sess_limiters = build_sessions()
logger.info("TrueData login successful.")

# ======== IMPROVED Indicator Functions with Better Error Handling ========
def safe_calculate_indicator(func, *args, **kwargs):
    """Wrapper function to safely calculate indicators with proper error handling"""
    try:
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            return tuple(r.fillna(method='ffill').fillna(0) if isinstance(r, pd.Series) else r for r in result)
        elif isinstance(result, pd.Series):
            return result.fillna(method='ffill').fillna(0)
        return result
    except Exception as e:
        logger.warning(f"Error calculating {func.__name__}: {e}")
        if hasattr(args[0], 'index'):
            default_series = pd.Series(0, index=args[0].index)
            return default_series
        return 0

def ema_improved(series, length):
    """Improved EMA calculation with better handling of edge cases"""
    if len(series) < length:
        return pd.Series(index=series.index, dtype='float64')
    return series.ewm(span=length, adjust=True, min_periods=length//2).mean()

def vwap_improved(df, period=None):
    """Improved VWAP calculation with better error handling"""
    if df is None or df.empty or len(df) < 10:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    
    try:
        required_cols = ['High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                return pd.Series(index=df.index, dtype='float64')
        
        high_vals = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low_vals = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')
        close_vals = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        volume_vals = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        
        typical_price = (high_vals + low_vals + close_vals) / 3.0
        pv = typical_price * volume_vals
        
        if period:
            pv_sum = pv.rolling(period, min_periods=period//2).sum()
            vol_sum = volume_vals.rolling(period, min_periods=period//2).sum()
        else:
            pv_sum = pv.cumsum()
            vol_sum = volume_vals.cumsum()
        
        vol_sum = vol_sum.replace(0, np.nan)
        vwap_result = pv_sum / vol_sum
        
        return vwap_result.fillna(method='ffill').fillna(typical_price.mean())
    
    except Exception as e:
        logger.warning(f"Error in VWAP calculation: {e}")
        return pd.Series(index=df.index, dtype='float64')

def calculate_rsi_improved(df, period=14):
    """Improved RSI calculation with better stability"""
    if df is None or len(df) < period + 10:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    
    try:
        close_prices = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        delta = close_prices.diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()
        
        for i in range(period, len(gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    except Exception as e:
        logger.warning(f"Error calculating RSI: {e}")
        return pd.Series(50, index=df.index)

def calculate_macd_improved(df, fast=12, slow=26, signal=9):
    """Improved MACD calculation with better error handling"""
    if df is None or len(df) < slow + signal + 10:
        empty_series = pd.Series(index=df.index if df is not None else [], dtype='float64')
        return empty_series, empty_series
    
    try:
        close_prices = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        
        ema_fast = close_prices.ewm(span=fast, min_periods=fast//2).mean()
        ema_slow = close_prices.ewm(span=slow, min_periods=slow//2).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, min_periods=signal//2).mean()
        
        return macd.fillna(0), signal_line.fillna(0)
    
    except Exception as e:
        logger.warning(f"Error calculating MACD: {e}")
        empty_series = pd.Series(0, index=df.index)
        return empty_series, empty_series

def calculate_adx_improved(df, period=14):
    """Improved ADX calculation with better stability and error handling"""
    if df is None or len(df) < period * 3:
        empty_series = pd.Series(index=df.index if df is not None else [], dtype='float64')
        return empty_series, empty_series, empty_series
    
    try:
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        
        hl = high - low
        hc = (high - close.shift(1)).abs()
        lc = (low - close.shift(1)).abs()
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        
        plus_dm = (high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0)
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        
        minus_dm = (low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        
        atr = tr.ewm(span=period, min_periods=period//2).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, min_periods=period//2).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, min_periods=period//2).mean() / atr)
        
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.inf)
        adx = dx.ewm(span=period, min_periods=period//2).mean()
        
        return adx.fillna(20), plus_di.fillna(20), minus_di.fillna(20)
    
    except Exception as e:
        logger.warning(f"Error calculating ADX: {e}")
        default_series = pd.Series(20, index=df.index)
        return default_series, default_series, default_series

def volume_surge_improved(df, lookback=20):
    """Improved volume surge calculation with outlier handling"""
    if df is None or len(df) < lookback + 10:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    
    try:
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        
        vol_ma = volume.rolling(lookback, min_periods=lookback//2).mean()
        vol_std = volume.rolling(lookback, min_periods=lookback//2).std()
        
        vol_std = vol_std.where(vol_std > vol_ma * 0.01, vol_ma * 0.1)
        z_score = (volume - vol_ma) / vol_std
        z_score = z_score.clip(-5, 5)
        
        return z_score.fillna(0)
    
    except Exception as e:
        logger.warning(f"Error calculating volume surge: {e}")
        return pd.Series(0, index=df.index)

def momentum_improved(df, period=10):
    """Improved momentum calculation with better error handling"""
    if df is None or len(df) < period + 5:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    
    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        shifted_close = close.shift(period)
        shifted_close = shifted_close.replace(0, np.nan)
        momentum_val = (close / shifted_close) - 1.0
        
        return momentum_val.fillna(0)
    
    except Exception as e:
        logger.warning(f"Error calculating momentum: {e}")
        return pd.Series(0, index=df.index)

def oi_surge_improved(df, lookback=20):
    """CRITICAL: OI surge calculation for option trading"""
    if df is None or len(df) < lookback + 5:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    
    try:
        # Look for OI column in different formats
        oi_col = None
        for col in df.columns:
            if col.upper() == 'OI' or 'openinterest' in col.lower():
                oi_col = col
                break
        
        if oi_col is None:
            # Use volume as proxy for OI
            return volume_surge_improved(df, lookback)
        
        oi = pd.to_numeric(df[oi_col], errors='coerce').fillna(0)
        
        # Check if we have actual OI data (non-zero values)
        if oi.sum() == 0:
            return volume_surge_improved(df, lookback)
        
        oi_ma = oi.rolling(lookback, min_periods=lookback//2).mean()
        oi_std = oi.rolling(lookback, min_periods=lookback//2).std()
        
        # Handle cases where std is 0 or very small
        oi_std = oi_std.where(oi_std > oi_ma * 0.01, oi_ma * 0.1)
        z_score = (oi - oi_ma) / oi_std
        z_score = z_score.clip(-5, 5)
        
        return z_score.fillna(0)
    
    except Exception as e:
        logger.warning(f"Error calculating OI surge: {e}")
        return pd.Series(0, index=df.index)

def oi_momentum_improved(df, period=10):
    """CRITICAL: OI momentum for detecting option activity"""
    if df is None or len(df) < period + 5:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    
    try:
        # Look for OI column
        oi_col = None
        for col in df.columns:
            if col.upper() == 'OI' or 'openinterest' in col.lower():
                oi_col = col
                break
        
        if oi_col is None:
            return momentum_improved(df, period)  # Fallback to volume momentum
        
        oi = pd.to_numeric(df[oi_col], errors='coerce').fillna(0)
        
        if oi.sum() == 0:
            return momentum_improved(df, period)
        
        shifted_oi = oi.shift(period).replace(0, np.nan)
        
        return ((oi / shifted_oi) - 1.0).fillna(0)
    
    except Exception as e:
        logger.warning(f"Error calculating OI momentum: {e}")
        return pd.Series(0, index=df.index)

def atr_improved(df, period=14):
    """Improved ATR calculation"""
    if df is None or len(df) < period + 5:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    
    try:
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        
        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        return tr.ewm(span=period, min_periods=period//2).mean().fillna(0)
    
    except Exception as e:
        logger.warning(f"Error calculating ATR: {e}")
        return pd.Series(0, index=df.index)

def cmf_improved(df, period=20):
    """Improved Chaikin Money Flow calculation"""
    if df is None or len(df) < period + 5:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    
    try:
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        mfm = mfm.fillna(0)
        mfv = mfm * volume
        
        mfv_sum = mfv.rolling(period, min_periods=period//2).sum()
        vol_sum = volume.rolling(period, min_periods=period//2).sum().replace(0, np.nan)
        
        return (mfv_sum / vol_sum).fillna(0)
    
    except Exception as e:
        logger.warning(f"Error calculating CMF: {e}")
        return pd.Series(0, index=df.index)

def adl_improved(df):
    """Improved Accumulation/Distribution Line calculation"""
    if df is None or len(df) < 2:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    
    try:
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        mfm = mfm.fillna(0)
        mfv = mfm * volume
        
        return mfv.cumsum()
    
    except Exception as e:
        logger.warning(f"Error calculating ADL: {e}")
        return pd.Series(0, index=df.index)

def slope_improved(series, lookback=10):
    """Improved slope calculation with better error handling"""
    try:
        if len(series) < lookback or series.isna().all():
            return 0.0
        
        y = series.tail(lookback).dropna().values.astype(float)
        if len(y) < 3:
            return 0.0
        
        x = np.arange(len(y))
        if x.std() == 0:
            return 0.0
        
        x = (x - x.mean()) / x.std()
        A = np.vstack([x, np.ones_like(x)]).T
        
        try:
            m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
            return float(m)
        except:
            return 0.0
    
    except Exception as e:
        logger.warning(f"Error calculating slope: {e}")
        return 0.0

# ======== ENHANCED SCORING SYSTEM ========
def validate_data_quality(df, min_bars=50):
    """Validate data quality before analysis"""
    if df is None or df.empty:
        return False, "No data"
    
    if len(df) < min_bars:
        return False, f"Insufficient data: {len(df)} < {min_bars}"
    
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            return False, f"Missing column: {col}"
        
        missing_pct = (df[col].isna().sum() / len(df)) * 100
        if missing_pct > CONFIG['MAX_MISSING_DATA_PCT']:
            return False, f"Too much missing data in {col}: {missing_pct:.1f}%"
    
    if (df['High'] < df['Low']).any():
        return False, "Invalid OHLC data: High < Low"
    
    if (df['High'] < df['Close']).any() or (df['Low'] > df['Close']).any():
        return False, "Invalid OHLC data: Close outside High-Low range"
    
    return True, "Data quality OK"

def analyze_signals_with_oi(timeframe_dataframes, symbol):
    """ENHANCED: OI-focused signal analysis for option buyers"""
    if not timeframe_dataframes:
        return 'Neutral', 0.0, 'Normal'

    final_score, max_possible = 0.0, 0.0
    valid_timeframes = 0
    has_oi_data = False
    oi_status = 'Normal'

    for tf_min, df in timeframe_dataframes.items():
        if df is None or len(df) < CONFIG['MIN_BARS_REQUIRED']:
            continue

        is_valid, _ = validate_data_quality(df, CONFIG['MIN_BARS_REQUIRED'])
        if not is_valid:
            continue

        valid_timeframes += 1
        tf_weight = CONFIG["TIMEFRAME_WEIGHTS"].get(tf_min, 1.0)
        
        # Check for OI data
        if 'OI' in df.columns and df['OI'].sum() > 0:
            has_oi_data = True
            global oi_symbols_found
            oi_symbols_found.add(symbol)
        
        # Calculate all indicators with improved error handling
        scores = {}
        
        # RSI Analysis with multi-level scoring
        rsi_series = calculate_rsi_improved(df)
        if len(rsi_series) >= CONFIG['SIGNAL_CONFIRMATION_BARS']:
            rsi_current = rsi_series.iloc[-1]
            rsi_prev = rsi_series.iloc[-CONFIG['SIGNAL_CONFIRMATION_BARS']]
            
            if rsi_current > 70:
                scores['RSI'] = 2.0 if rsi_prev <= 70 else 1.5
            elif rsi_current > 60:
                scores['RSI'] = 1.5 if rsi_prev <= 60 else 1.0
            elif rsi_current > 50:
                scores['RSI'] = 1.0 if rsi_prev <= 50 else 0.5
            elif rsi_current < 30:
                scores['RSI'] = -2.0 if rsi_prev >= 30 else -1.5
            elif rsi_current < 40:
                scores['RSI'] = -1.5 if rsi_prev >= 40 else -1.0
            elif rsi_current < 50:
                scores['RSI'] = -1.0 if rsi_prev >= 50 else -0.5
            else:
                scores['RSI'] = 0.0
        else:
            scores['RSI'] = 0.0

        # MACD with trend confirmation
        macd, signal_line = calculate_macd_improved(df)
        if len(macd) >= CONFIG['SIGNAL_CONFIRMATION_BARS'] and len(signal_line) >= CONFIG['SIGNAL_CONFIRMATION_BARS']:
            macd_diff = macd.iloc[-1] - signal_line.iloc[-1]
            prev_diff = macd.iloc[-CONFIG['SIGNAL_CONFIRMATION_BARS']] - signal_line.iloc[-CONFIG['SIGNAL_CONFIRMATION_BARS']]
            
            if macd_diff > 0 and prev_diff <= 0:
                scores['MACD'] = 2.0  # Bullish crossover
            elif macd_diff > 0:
                scores['MACD'] = 1.0  # Continued bullish
            elif macd_diff < 0 and prev_diff >= 0:
                scores['MACD'] = -2.0  # Bearish crossover
            elif macd_diff < 0:
                scores['MACD'] = -1.0  # Continued bearish
            else:
                scores['MACD'] = 0.0
        else:
            scores['MACD'] = 0.0

        # ADX with directional bias
        adx, pdi, ndi = calculate_adx_improved(df)
        if len(adx) >= CONFIG['SIGNAL_CONFIRMATION_BARS']:
            adx_current = adx.iloc[-1]
            adx_trend = adx.iloc[-1] > adx.iloc[-CONFIG['SIGNAL_CONFIRMATION_BARS']]
            pdi_current = pdi.iloc[-1]
            ndi_current = ndi.iloc[-1]
            
            if adx_current > 25 and adx_trend:
                if pdi_current > ndi_current:
                    scores['ADX'] = 2.0  # Strong bullish trend
                else:
                    scores['ADX'] = -2.0  # Strong bearish trend
            elif adx_current > 20:
                if pdi_current > ndi_current:
                    scores['ADX'] = 1.0  # Moderate bullish trend
                else:
                    scores['ADX'] = -1.0  # Moderate bearish trend
            else:
                scores['ADX'] = 0.0  # Weak trend
        else:
            scores['ADX'] = 0.0

        # Volume surge with price confirmation
        vol_surge = volume_surge_improved(df)
        if len(vol_surge) >= CONFIG['SIGNAL_CONFIRMATION_BARS']:
            current_surge = vol_surge.iloc[-1]
            price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-CONFIG['SIGNAL_CONFIRMATION_BARS']]) - 1
            
            if current_surge >= 2.0:
                if price_change > 0.01:  # 1% price increase with high volume
                    scores['VolumeSurge'] = 2.5
                elif price_change < -0.01:  # 1% price decrease with high volume
                    scores['VolumeSurge'] = -2.5
                else:
                    scores['VolumeSurge'] = 1.0  # High volume but no clear direction
            elif current_surge >= 1.5:
                scores['VolumeSurge'] = 1.5 if price_change > 0 else -1.5
            elif current_surge <= -1.5:
                scores['VolumeSurge'] = -1.0  # Unusual low volume
            else:
                scores['VolumeSurge'] = 0.0
        else:
            scores['VolumeSurge'] = 0.0

        # Momentum with consistency check
        mom = momentum_improved(df)
        if len(mom) >= CONFIG['SIGNAL_CONFIRMATION_BARS']:
            current_mom = mom.iloc[-1]
            consistent_direction = all(mom.iloc[-CONFIG['SIGNAL_CONFIRMATION_BARS']:] > 0) or all(mom.iloc[-CONFIG['SIGNAL_CONFIRMATION_BARS']:] < 0)
            
            if current_mom > 0.02:  # 2% momentum
                scores['Momentum'] = 2.0 if consistent_direction else 1.5
            elif current_mom > 0.005:  # 0.5% momentum
                scores['Momentum'] = 1.5 if consistent_direction else 1.0
            elif current_mom < -0.02:
                scores['Momentum'] = -2.0 if consistent_direction else -1.5
            elif current_mom < -0.005:
                scores['Momentum'] = -1.5 if consistent_direction else -1.0
            else:
                scores['Momentum'] = 0.0
        else:
            scores['Momentum'] = 0.0

        # VWAP analysis with improved calculation
        vwap_line = vwap_improved(df)
        if len(vwap_line) >= CONFIG['SIGNAL_CONFIRMATION_BARS']:
            price_vs_vwap = (df['Close'].iloc[-1] / vwap_line.iloc[-1]) - 1
            consistent_above = all(df['Close'].iloc[-CONFIG['SIGNAL_CONFIRMATION_BARS']:] > vwap_line.iloc[-CONFIG['SIGNAL_CONFIRMATION_BARS']:])
            consistent_below = all(df['Close'].iloc[-CONFIG['SIGNAL_CONFIRMATION_BARS']:] < vwap_line.iloc[-CONFIG['SIGNAL_CONFIRMATION_BARS']:])
            
            if price_vs_vwap > 0.005:  # 0.5% above VWAP
                scores['VWAP'] = 2.0 if consistent_above else 1.0
            elif price_vs_vwap < -0.005:  # 0.5% below VWAP
                scores['VWAP'] = -2.0 if consistent_below else -1.0
            else:
                scores['VWAP'] = 0.0
        else:
            scores['VWAP'] = 0.0

        # EMA crossover system
        ema_fast = ema_improved(df['Close'], CONFIG["INDICATOR_PERIODS"]["EMA_FAST"])
        ema_slow = ema_improved(df['Close'], CONFIG["INDICATOR_PERIODS"]["EMA_SLOW"])
        if len(ema_fast) >= CONFIG['SIGNAL_CONFIRMATION_BARS'] and len(ema_slow) >= CONFIG['SIGNAL_CONFIRMATION_BARS']:
            current_diff = ema_fast.iloc[-1] - ema_slow.iloc[-1]
            prev_diff = ema_fast.iloc[-CONFIG['SIGNAL_CONFIRMATION_BARS']] - ema_slow.iloc[-CONFIG['SIGNAL_CONFIRMATION_BARS']]
            
            if current_diff > 0 and prev_diff <= 0:
                scores['EMA'] = 2.0  # Golden cross
            elif current_diff > 0:
                scores['EMA'] = 1.0  # Bullish alignment
            elif current_diff < 0 and prev_diff >= 0:
                scores['EMA'] = -2.0  # Death cross
            elif current_diff < 0:
                scores['EMA'] = -1.0  # Bearish alignment
            else:
                scores['EMA'] = 0.0
        else:
            scores['EMA'] = 0.0

        # ========= CRITICAL OI ANALYSIS FOR OPTION BUYERS =========
        
        # OI Surge Analysis - KEY for option activity detection
        oi_z = oi_surge_improved(df)
        if len(oi_z) >= CONFIG['SIGNAL_CONFIRMATION_BARS']:
            oi_surge_current = oi_z.iloc[-1]
            price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-CONFIG['SIGNAL_CONFIRMATION_BARS']]) - 1
            
            if oi_surge_current >= 2.0:  # Strong OI surge
                scores['OISurge'] = 3.0 if price_change > 0 else -3.0
                oi_status = 'High OI Activity'
            elif oi_surge_current >= 1.0:  # Moderate OI surge
                scores['OISurge'] = 2.0 if price_change > 0 else -2.0
                oi_status = 'Moderate OI Activity'
            elif oi_surge_current <= -1.0:  # OI decline
                scores['OISurge'] = -1.0
            else:
                scores['OISurge'] = 0.0
        else:
            scores['OISurge'] = 0.0

        # OI Momentum Analysis - Trend in option interest
        oi_mom = oi_momentum_improved(df)
        if len(oi_mom) >= CONFIG['SIGNAL_CONFIRMATION_BARS']:
            oi_mom_current = oi_mom.iloc[-1]
            if oi_mom_current > 0.1:  # 10% OI increase
                scores['OIMomentum'] = 3.0
            elif oi_mom_current > 0.05:  # 5% OI increase
                scores['OIMomentum'] = 2.0
            elif oi_mom_current < -0.05:  # 5% OI decrease
                scores['OIMomentum'] = -1.0
            else:
                scores['OIMomentum'] = 0.0
        else:
            scores['OIMomentum'] = 0.0

        # Call/Put Bias Analysis - MOST IMPORTANT for option buyers
        price_up = df['Close'].iloc[-1] > df['Close'].iloc[-CONFIG['SIGNAL_CONFIRMATION_BARS']]
        vol_high = vol_surge.iloc[-1] >= 1.5 if len(vol_surge) >= CONFIG['SIGNAL_CONFIRMATION_BARS'] else False
        oi_active = oi_z.iloc[-1] >= 1.0 if len(oi_z) >= CONFIG['SIGNAL_CONFIRMATION_BARS'] else False
        
        # Strong Call Bias Conditions
        strong_call_conditions = (
            price_up and
            scores.get('OISurge', 0) >= 2.0 and
            scores.get('OIMomentum', 0) >= 2.0 and
            vol_high and
            scores.get('VolumeSurge', 0) > 0
        )
        
        # Strong Put Bias Conditions
        strong_put_conditions = (
            not price_up and
            scores.get('OISurge', 0) <= -2.0 and
            vol_high and
            scores.get('VolumeSurge', 0) < 0
        )
        
        # Moderate conditions
        moderate_call_conditions = (
            price_up and
            oi_active and
            vol_high
        )
        
        moderate_put_conditions = (
            not price_up and
            oi_active and
            vol_high
        )

        if strong_call_conditions:
            scores['CallBias'] = 4.0  # Maximum score for strong call setup
            scores['PutBias'] = 0.0
            oi_status = 'Strong Call Setup'
        elif moderate_call_conditions:
            scores['CallBias'] = 2.0
            scores['PutBias'] = 0.0
        elif strong_put_conditions:
            scores['PutBias'] = -4.0  # Maximum score for strong put setup
            scores['CallBias'] = 0.0
            oi_status = 'Strong Put Setup'
        elif moderate_put_conditions:
            scores['PutBias'] = -2.0
            scores['CallBias'] = 0.0
        else:
            scores['CallBias'] = 0.0
            scores['PutBias'] = 0.0

        # OI-Volume Confirmation
        if oi_active and vol_high:
            if price_up:
                scores['OIVolConfirm'] = 2.0
            else:
                scores['OIVolConfirm'] = -2.0
        else:
            scores['OIVolConfirm'] = 0.0

        # Additional indicators for comprehensive analysis
        # CMF analysis
        cmf_vals = cmf_improved(df)
        if len(cmf_vals) >= CONFIG['SIGNAL_CONFIRMATION_BARS']:
            cmf_current = cmf_vals.iloc[-1]
            cmf_slope = slope_improved(cmf_vals)
            
            if cmf_current > 0.1 and cmf_slope > 0:
                scores['CMF'] = 2.0
            elif cmf_current > 0.05:
                scores['CMF'] = 1.0
            elif cmf_current < -0.1 and cmf_slope < 0:
                scores['CMF'] = -2.0
            elif cmf_current < -0.05:
                scores['CMF'] = -1.0
            else:
                scores['CMF'] = 0.0
        else:
            scores['CMF'] = 0.0

        # ADL analysis
        adl_vals = adl_improved(df)
        if len(adl_vals) >= CONFIG['SIGNAL_CONFIRMATION_BARS']:
            adl_slope = slope_improved(adl_vals)
            
            if adl_slope > 0:
                scores['ADL'] = 1.5
            elif adl_slope < 0:
                scores['ADL'] = -1.5
            else:
                scores['ADL'] = 0.0
        else:
            scores['ADL'] = 0.0

        # Fill remaining indicators with default values
        remaining_indicators = set(CONFIG["INDICATOR_WEIGHTS"].keys()) - set(scores.keys())
        for indicator in remaining_indicators:
            scores[indicator] = 0.0

        # Calculate weighted scores for this timeframe
        for indicator, score in scores.items():
            ind_weight = CONFIG["INDICATOR_WEIGHTS"].get(indicator, 1.0)
            weighted_score = score * tf_weight * ind_weight
            final_score += weighted_score
            max_possible += 4.0 * tf_weight * ind_weight  # Maximum possible score per indicator

    if valid_timeframes < 1 or max_possible == 0:
        return 'Neutral', 0.0, oi_status

    normalized = (final_score / max_possible) * 100.0

    # Normalize scores to ensure consistency
    if abs(normalized) > 100:
        normalized = np.sign(normalized) * 100

    # Enhanced signal classification with OI focus
    if normalized >= 75:
        signal_text = 'Very Strong Buy (Call Focus)'
    elif normalized >= 40:
        signal_text = 'Strong Buy (Call Focus)'
    elif normalized <= -75:
        signal_text = 'Very Strong Sell (Put Focus)'
    elif normalized <= -40:
        signal_text = 'Strong Sell (Put Focus)'
    elif normalized >= 15:
        signal_text = 'Buy (Call Potential)'
    elif normalized <= -15:
        signal_text = 'Sell (Put Potential)'
    else:
        signal_text = 'Neutral'

    return signal_text, normalized, oi_status

# ======== ENHANCED Data Normalization with OI Support ========
def normalize_hist_df_with_oi(df, symbol, timeframe_minutes):
    """Enhanced normalization that preserves OI data"""
    if df is None or df.empty:
        return None
    
    try:
        if CONFIG["LOG_LEVEL"] == "DEBUG":
            logger.debug(f"Raw columns for {symbol} timeframe {timeframe_minutes} min: {list(df.columns)}")
        
        out = df.copy()
        out.columns = out.columns.str.lower()
        
        # Enhanced column mapping including OI
        rename_map = {}
        for col in out.columns:
            col_lower = col.lower()
            if any(x in col_lower for x in ['time', 'date', 'timestamp']):
                rename_map[col] = 'Timestamp'
            elif 'open' in col_lower and 'interest' not in col_lower:
                rename_map[col] = 'Open'
            elif 'high' in col_lower:
                rename_map[col] = 'High'
            elif 'low' in col_lower:
                rename_map[col] = 'Low'
            elif 'close' in col_lower:
                rename_map[col] = 'Close'
            elif 'volume' in col_lower or col_lower == 'vol':
                rename_map[col] = 'Volume'
            elif any(x in col_lower for x in ['oi', 'open_interest', 'openinterest', 'open_int']):
                rename_map[col] = 'OI'  # PRESERVE OI DATA
        
        out.rename(columns=rename_map, inplace=True)
        
        # Check for required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Timestamp']
        missing_cols = [col for col in required_cols if col not in out.columns]
        
        if missing_cols:
            # Handle missing Volume
            if 'Volume' in missing_cols:
                out['Volume'] = 1000  # Default volume
                missing_cols.remove('Volume')
            
            # Handle missing Timestamp - FIXED: Better handling
            if 'Timestamp' in missing_cols:
                now = datetime.now(IST)
                out['Timestamp'] = pd.date_range(
                    start=now - timedelta(minutes=timeframe_minutes * len(out)),
                    periods=len(out),
                    freq=f'{timeframe_minutes}T',
                    tz=IST
                )
                missing_cols.remove('Timestamp')
            
            if missing_cols:
                if CONFIG["LOG_LEVEL"] == "DEBUG":
                    logger.error(f"Missing required columns for {symbol} tf {timeframe_minutes} min: {missing_cols}")
                return None
        
        # Handle OI column - CRITICAL for option analysis
        oi_found = False
        if 'OI' not in out.columns:
            if CONFIG["LOG_LEVEL"] == "DEBUG":
                logger.debug(f"No OI data for {symbol} - will use volume-based proxies")
            out['OI'] = out['Volume'] * 0.1  # Rough proxy: 10% of volume as OI estimate
        else:
            out['OI'] = pd.to_numeric(out['OI'], errors='coerce').fillna(0)
        
        # CRITICAL: Remove rows with null timestamps BEFORE any datetime operations
        initial_len = len(out)
        out = out.dropna(subset=['Timestamp'])
        if len(out) < initial_len:
            logger.warning(f"Removed {initial_len - len(out)} rows with null timestamps for {symbol}")
        
        if out.empty:
            logger.warning(f"No valid timestamp data for {symbol}")
            return None
        
        # Convert numeric columns with proper error handling
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors='coerce')
        
        # FIXED: Ensure timezone awareness with proper None checking
        try:
            # Convert Timestamp to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(out['Timestamp']):
                out['Timestamp'] = pd.to_datetime(out['Timestamp'], errors='coerce')
            
            # Remove any rows where timestamp conversion failed
            out = out.dropna(subset=['Timestamp'])
            if out.empty:
                logger.warning(f"No valid timestamps after conversion for {symbol}")
                return None
            
            # Handle timezone conversion safely
            if out['Timestamp'].dt.tz is None:
                out['Timestamp'] = out['Timestamp'].dt.tz_localize(IST, ambiguous='infer', nonexistent='shift_forward')
            else:
                out['Timestamp'] = out['Timestamp'].dt.tz_convert(IST)
        except Exception as e:
            logger.error(f"Timestamp processing error for {symbol}: {e}")
            return None
        
        # Check if we have actual OI data (non-zero)
        if out['OI'].sum() > 0:
            oi_found = True
            if CONFIG["LOG_LEVEL"] == "DEBUG":
                logger.debug(f"Found OI data for {symbol}")
        
        # Data quality checks
        if (out['High'] < out['Low']).any():
            logger.warning(f"Invalid OHLC data for {symbol}: High < Low")
            return None
        
        if (out['High'] < out['Close']).any() or (out['Low'] > out['Close']).any():
            logger.warning(f"Invalid OHLC data for {symbol}: Close outside High-Low range")
            return None
        
        return out
        
    except Exception as e:
        logger.error(f"Normalize error for {symbol} timeframe {timeframe_minutes} min: {e}")
        return None


def pick_session(symbol_orig, timeframe_minutes):
    """Stable spread across sessions"""
    return hash((symbol_orig, timeframe_minutes)) & 0x7fffffff % len(tdhist_pool)

# ======== FIXED Data Fetching Function ========
@retry(
    stop_max_attempt_number=CONFIG["RETRY_ATTEMPTS"],
    wait_exponential_multiplier=max(1, int(CONFIG["RETRY_DELAY_MS"] / 2)),
    wait_exponential_max=8000,
    retry_on_exception=lambda e: True
)
@retry(stop_max_attempt_number=CONFIG["RETRY_ATTEMPTS"], 
       wait_exponential_multiplier=max(1, int(CONFIG["RETRY_DELAY_MS"] / 2)), 
       wait_exponential_max=8000,
       retry_on_exception=lambda e: True)
def fetch_one_with_oi(symbol_orig, timeframe_minutes, limiter, hist):
    """Enhanced fetch function with OI data support"""
    td_symbol = symbol_orig.replace('-EQ', '')
    
    if td_symbol in failed_symbols:
        if CONFIG["LOG_LEVEL"] == "DEBUG":
            logger.debug(f"Skipping {td_symbol} timeframe {timeframe_minutes} min due to previous failures")
        return symbol_orig, timeframe_minutes, None
    
    if CONFIG.get("SKIP_DAILY", True) and timeframe_minutes == 1440:
        return symbol_orig, timeframe_minutes, None
    
    bar_size = CONFIG["BAR_SIZE_MAP"].get(timeframe_minutes)
    duration = CONFIG["DURATION_MAP"].get(timeframe_minutes)
    
    if not bar_size or not duration:
        if CONFIG["LOG_LEVEL"] == "DEBUG":
            logger.error(f"Invalid timeframe {timeframe_minutes} for {td_symbol}")
        return symbol_orig, timeframe_minutes, None
    
    try:
        limiter.acquire()
        
        # Update API call counter
        global api_calls_done
        with api_calls_lock:
            api_calls_done += 1
        
        if CONFIG["LOG_LEVEL"] == "DEBUG" and api_calls_done % 100 == 0:
            logger.debug(f"API calls: {api_calls_done}")
        
        try:
            # FIXED: Using correct method name
            df_raw = hist.get_historical_data(td_symbol, duration=duration, bar_size=bar_size)
        except Exception as api_error:
            logger.warning(f"API call failed for {td_symbol}: {api_error}")
            raise api_error
        
        if df_raw is None or df_raw.empty:
            if CONFIG["LOG_LEVEL"] == "DEBUG":
                logger.warning(f"No data by API for {td_symbol} (tf: {timeframe_minutes}m, dur: {duration}, bar: {bar_size})")
            failed_symbols.add(td_symbol)
            return symbol_orig, timeframe_minutes, None
        
        # Use the enhanced normalization function that preserves OI
        df = normalize_hist_df_with_oi(df_raw, td_symbol, timeframe_minutes)
        
        if df is None:
            failed_symbols.add(td_symbol)
            return symbol_orig, timeframe_minutes, df
        
    except Exception as e:
        if CONFIG["LOG_LEVEL"] in ["DEBUG", "INFO"]:
            logger.error(f"Fetch failed for {td_symbol} (tf: {timeframe_minutes}m, dur: {duration}, bar: {bar_size}): {e}")
        failed_symbols.add(td_symbol)
        return symbol_orig, timeframe_minutes, None
    
    return symbol_orig, timeframe_minutes, df


def prefetch_with_oi(stocks, max_workers=CONFIG["MAX_WORKERS"]):
    """Enhanced prefetch with OI data tracking"""
    tfs = [5, 15, 30, 60] if CONFIG.get("SKIP_DAILY", True) else [5, 15, 30, 60, 1440]
    total_calls = len(stocks) * len(tfs)
    stock_multi_data = defaultdict(dict)
    
    global api_calls_done, oi_symbols_found
    with api_calls_lock:
        api_calls_done = 0
    
    # Reset OI tracking for each prefetch
    oi_symbols_found = set()
    
    valid_stocks = [s for s in stocks if s]
    if CONFIG["LOG_LEVEL"] in ["DEBUG", "INFO"]:
        logger.info(f"Valid symbols to fetch: {len(valid_stocks)}")
    
    progress_kwargs = dict(
        total=total_calls,
        desc="Fetching OI Data",
        ncols=100,
        disable=not CONFIG["SHOW_PROGRESS"]
    )
    
    with tqdm(**progress_kwargs) as api_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for s in valid_stocks:
                for tf in tfs:
                    si = pick_session(s, tf)
                    futures.append(executor.submit(fetch_one_with_oi, s, tf, sess_limiters[si], tdhist_pool[si]))
            
            for fut in as_completed(futures):
                try:
                    symbol_orig, tf, df = fut.result()
                    if df is not None:
                        stock_multi_data[symbol_orig][tf] = df
                except Exception as e:
                    if CONFIG["LOG_LEVEL"] == "DEBUG":
                        logger.error(f"Future failed: {e}")
                api_bar.update(1)
    
    valid_data = {s: d for s, d in stock_multi_data.items() if len(d) >= 1}  # At least one timeframe
    
    if CONFIG["LOG_LEVEL"] in ["DEBUG", "INFO"]:
        logger.info(f"Prefetch complete. {len(valid_data)} total symbols, {len(oi_symbols_found)} with OI data.")
        if len(oi_symbols_found) > 0:
            logger.info(f"OI symbols found: {sorted(list(oi_symbols_found))}")
        if failed_symbols:
            logger.info(f"Failed symbols: {sorted(failed_symbols)}")
    
    return valid_data

# ======== ENHANCED Data Filtering Functions ========
def filter_timeframe_data(symbol, timeframe_data, timepoint_aware):
    """Enhanced filtering with proper None handling"""
    filtered_timeframes = {}
    
    for tf, df in timeframe_data.items():
        if df is None or df.empty:
            continue
        
        try:
            # FIXED: Check for Timestamp column existence and valid data
            if 'Timestamp' not in df.columns:
                logger.warning(f"No Timestamp column for {symbol} tf {tf}")
                continue
            
            # FIXED: Handle None values in Timestamp column
            valid_timestamps = df['Timestamp'].notna()
            if not valid_timestamps.any():
                logger.warning(f"No valid timestamps for {symbol} tf {tf}")
                continue
            
            # Filter out rows with None timestamps
            df_clean = df[valid_timestamps].copy()
            
            if df_clean.empty:
                continue
            
            # FIXED: Safe datetime comparison
            try:
                # Ensure timepoint_aware is timezone-aware
                if timepoint_aware.tzinfo is None:
                    timepoint_aware = timepoint_aware.replace(tzinfo=IST)
                
                # Ensure DataFrame timestamps are timezone-aware
                if df_clean['Timestamp'].dt.tz is None:
                    df_clean['Timestamp'] = df_clean['Timestamp'].dt.tz_localize(IST, ambiguous='infer', nonexistent='shift_forward')
                else:
                    df_clean['Timestamp'] = df_clean['Timestamp'].dt.tz_convert(IST)
                
                # Now safe to compare
                mask = df_clean['Timestamp'] <= timepoint_aware
                if mask.any():
                    filtered_df = df_clean[mask].copy()
                    if len(filtered_df) >= CONFIG["MIN_BARS_REQUIRED"]:
                        filtered_timeframes[tf] = filtered_df
                        
            except Exception as ts_error:
                logger.error(f"Timestamp filtering error for {symbol} tf {tf}: {ts_error}")
                continue
                
        except Exception as e:
            logger.error(f"Filter error for {symbol} tf {tf}: {e}")
            continue
    
    return filtered_timeframes


# Institutional inference function
def infer_institutional_flow(tf_data):
    """Enhanced institutional flow detection using multiple indicators"""
    frames = [tf_data.get(t) for t in (5, 15, 30) if tf_data.get(t) is not None and len(tf_data.get(t)) >= 60]
    if not frames:
        return "Unknown"
    
    votes = 0
    for df in frames:
        try:
            cmf_vals = cmf_improved(df)
            adl_vals = adl_improved(df)
            adx_val, pdi, ndi = calculate_adx_improved(df)
            vwap_day = vwap_improved(df)
            atr_val = atr_improved(df)
            
            if len(cmf_vals) and len(adl_vals) and len(adx_val) and len(vwap_day) and len(atr_val):
                vwap_threshold = (atr_val.iloc[-1] / df["Close"].iloc[-1]) * 2.0 if pd.notna(atr_val.iloc[-1]) else 0.01
                vdist = ((df["Close"] - vwap_day) / vwap_day.replace(0, np.nan))
                
                cmf_last = cmf_vals.iloc[-1] if pd.notna(cmf_vals.iloc[-1]) else np.nan
                cmf_slope = slope_improved(cmf_vals)
                adl_slope = slope_improved(adl_vals)
                adx_last = adx_val.iloc[-1] if pd.notna(adx_val.iloc[-1]) else np.nan
                p_over_n = pdi.iloc[-1] > ndi.iloc[-1] if len(pdi) and len(ndi) and pd.notna(pdi.iloc[-1]) and pd.notna(ndi.iloc[-1]) else False
                vdist_last = vdist.iloc[-1] if len(vdist) and pd.notna(vdist.iloc[-1]) else np.nan
                
                near_vwap_ok = pd.notna(vdist_last) and abs(vdist_last) <= vwap_threshold
                
                buy_cond = (
                    pd.notna(vdist_last) and vdist_last > 0 and near_vwap_ok and
                    pd.notna(cmf_last) and cmf_last > 0.1 and
                    not np.isnan(cmf_slope) and cmf_slope > 0 and
                    not np.isnan(adl_slope) and adl_slope > 0 and
                    pd.notna(adx_last) and adx_last > 20 and p_over_n
                )
                sell_cond = (
                    pd.notna(vdist_last) and vdist_last < 0 and near_vwap_ok and
                    pd.notna(cmf_last) and cmf_last < -0.1 and
                    not np.isnan(cmf_slope) and cmf_slope < 0 and
                    not np.isnan(adl_slope) and adl_slope < 0 and
                    pd.notna(adx_last) and adx_last > 20 and not p_over_n
                )
                
                if buy_cond:
                    votes += 1
                if sell_cond:
                    votes -= 1
        except Exception as e:
            logger.debug(f"Error in institutional flow analysis: {e}")
            continue
    
    if votes >= 2:
        return "Institutional Accumulation"
    if votes <= -2:
        return "Institutional Distribution"
    return "Mixed/Unclear"

# ======== ENHANCED RENDERING WITH OI INFORMATION ========
def render_top_lists_with_oi(now_ts, top_bullish, top_bearish):
    """Enhanced rendering to show OI information"""
    global last_bull_symbols, last_bear_symbols
    title = f"| OI-FOCUSED OPTION SCANNER | SNAPSHOT AT {now_ts.strftime('%Y-%m-%d %H:%M')} IST"
    console.rule(title)

    bull_table = Table(title="🚀 Top Call Option Opportunities", box=box.SIMPLE_HEAVY, header_style="white on dark_green", style="white on black")
    for col, style, justify in [
        ("Stock", "cyan", "left"), ("Signal", "bright_white", "left"), ("Score", "yellow", "right"),
        ("Change", "magenta", "right"), ("Trend", "bright_white", "left"),
        ("Flow", "bright_white", "left"), ("OI Status", "green", "left"), ("Action", "bright_white", "left")
    ]:
        bull_table.add_column(col, style=style, justify=justify)

    for r in top_bullish:
        sym = r['symbol']
        is_new = sym not in last_bull_symbols
        row_style = "black on green" if is_new else None
        ch = r['change']
        change_str = f"{ch:+.2f}" if isinstance(ch, (int, float, np.floating)) else "NA"
        bull_table.add_row(sym, r['signal'], f"{r['score']:.2f}", change_str, r['trend'], 
                          r.get('flow', 'Unknown'), r.get('oi_status', 'Normal'), "🔥 CALL BUY", style=row_style)

    console.print(bull_table)

    bear_table = Table(title="📉 Top Put Option Opportunities", box=box.SIMPLE_HEAVY, header_style="white on dark_red", style="white on black")
    for col, style, justify in [
        ("Stock", "cyan", "left"), ("Signal", "bright_white", "left"), ("Score", "yellow", "right"),
        ("Change", "magenta", "right"), ("Trend", "bright_white", "left"),
        ("Flow", "bright_white", "left"), ("OI Status", "red", "left"), ("Action", "bright_white", "left")
    ]:
        bear_table.add_column(col, style=style, justify=justify)

    for r in top_bearish:
        sym = r['symbol']
        is_new = sym not in last_bear_symbols
        row_style = "white on red" if is_new else None
        ch = r['change']
        change_str = f"{ch:+.2f}" if isinstance(ch, (int, float, np.floating)) else "NA"
        bear_table.add_row(sym, r['signal'], f"{r['score']:.2f}", change_str, r['trend'], 
                          r.get('flow', 'Unknown'), r.get('oi_status', 'Normal'), "🔥 PUT BUY", style=row_style)

    console.print(bear_table)
    console.rule()

    last_bull_symbols = {r['symbol'] for r in top_bullish}
    last_bear_symbols = {r['symbol'] for r in top_bearish}

# ======== LIVE MARKET FUNCTIONS ========
def render_live_results(scan_time, top_bullish, top_bearish, scan_number):
    """Enhanced live results display"""
    global last_bull_symbols, last_bear_symbols
    
    title = f"🔴 LIVE SCAN #{scan_number} | {scan_time.strftime('%H:%M')} IST | OI-FOCUSED SIGNALS"
    console.rule(title, style="bold red")
    
    # Bullish table with live styling
    if top_bullish:
        bull_table = Table(title="🚀 LIVE Call Opportunities", box=box.DOUBLE_EDGE, header_style="bold white on green")
        for col in ["Rank", "Stock", "Signal", "Score", "Price", "Volume", "OI Status", "Alert"]:
            bull_table.add_column(col)
        
        for i, r in enumerate(top_bullish, 1):
            sym = r['symbol']
            is_new = sym not in last_bull_symbols
            
            # Color coding based on urgency
            if "URGENT" in r['alert_level']:
                row_style = "bold white on red"
            elif "HIGH" in r['alert_level']:
                row_style = "bold black on yellow"  
            elif is_new:
                row_style = "black on green"
            else:
                row_style = None
            
            price_str = f"₹{r['price']:.2f}" if r['price'] > 0 else "NA"
            volume_str = f"{int(r['volume']/1000)}K" if r['volume'] > 1000 else f"{int(r['volume'])}"
            
            bull_table.add_row(
                f"#{i}", sym, r['signal'], f"{r['score']:.1f}",
                price_str, volume_str, r['oi_status'][:15], r['alert_level'],
                style=row_style
            )
        
        console.print(bull_table)
    
    # Bearish table
    if top_bearish:
        bear_table = Table(title="📉 LIVE Put Opportunities", box=box.DOUBLE_EDGE, header_style="bold white on red")
        for col in ["Rank", "Stock", "Signal", "Score", "Price", "Volume", "OI Status", "Alert"]:
            bear_table.add_column(col)
        
        for i, r in enumerate(top_bearish, 1):
            sym = r['symbol']
            is_new = sym not in last_bear_symbols
            
            if "URGENT" in r['alert_level']:
                row_style = "bold white on red"
            elif "HIGH" in r['alert_level']:
                row_style = "bold black on yellow"
            elif is_new:
                row_style = "white on red"
            else:
                row_style = None
            
            price_str = f"₹{r['price']:.2f}" if r['price'] > 0 else "NA"
            volume_str = f"{int(r['volume']/1000)}K" if r['volume'] > 1000 else f"{int(r['volume'])}"
            
            bear_table.add_row(
                f"#{i}", sym, r['signal'], f"{r['score']:.1f}",
                price_str, volume_str, r['oi_status'][:15], r['alert_level'],
                style=row_style
            )
        
        console.print(bear_table)
    
    # Update tracking
    last_bull_symbols = {r['symbol'] for r in top_bullish}
    last_bear_symbols = {r['symbol'] for r in top_bearish}
    
    # Show summary
    total_signals = len(top_bullish) + len(top_bearish)
    urgent_count = sum(1 for r in top_bullish + top_bearish if "URGENT" in r['alert_level'])
    
    summary = f"📊 Signals: {total_signals} | 🚨 Urgent: {urgent_count} | ⏰ Next scan: +5min"
    console.print(f"[bold blue]{summary}[/bold blue]")
    console.rule()

def handle_urgent_alerts(alerts, scan_time, alerts_file):
    """Handle high-priority trading alerts"""
    with open(alerts_file, "a", encoding="utf-8") as f:
        f.write(f"\n=== URGENT ALERTS - {scan_time.strftime('%H:%M:%S')} ===\n")
        
        for alert in alerts:
            alert_msg = (
                f"🚨 {alert['symbol']}: {alert['signal']} "
                f"(Score: {alert['score']:.1f}) - {alert['oi_status']} "
                f"Price: ₹{alert['price']:.2f} Vol: {int(alert['volume'])}\n"
            )
            f.write(alert_msg)
            
            # Also print to console for immediate attention
            action = "🔥 BUY CALLS" if alert['score'] > 0 else "🔥 BUY PUTS"
            console.print(f"[bold red]{alert_msg.strip()} -> {action}[/bold red]")

def run_live_scanner(stocks):
    """Enhanced live market scanner with real-time OI analysis"""
    logger.info("🚀 Starting LIVE OI-focused option scanner...")
    
    # Check market hours
    now_ist = datetime.now(IST)
    market_start = today_ist_dt(CONFIG["MARKET_START"])
    market_end = today_ist_dt(CONFIG["MARKET_END"])
    
    if now_ist < market_start:
        wait_seconds = (market_start - now_ist).total_seconds()
        logger.info(f"⏰ Market opens in {wait_seconds/60:.1f} minutes. Waiting...")
        sleep_until(market_start)
    elif now_ist > market_end:
        logger.info("📈 Market is closed for today.")
        return
    
    # Initial data prefetch
    logger.info(f"📊 Prefetching initial data for {len(stocks)} symbols...")
    stock_multi_data = prefetch_with_oi(stocks)
    logger.info(f"✅ Initial prefetch complete. {len(stock_multi_data)} symbols ready.")
    
    # Live scanning variables
    next_boundary = next_5min_boundary_ist(datetime.now(IST))
    scan_count = 0
    
    # Create live output files
    live_csv = f"live_options_scan_{datetime.now(IST).strftime('%Y%m%d')}.csv"
    alerts_log = f"trading_alerts_{datetime.now(IST).strftime('%Y%m%d')}.txt"
    
    # Initialize CSV with headers
    with open(live_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Symbol", "Signal", "Score", "OI_Status", "Action", "Price", "Volume", "Alert_Level"])
    
    global previous_scores, performance_metrics
    previous_scores = {}
    performance_metrics = defaultdict(int)
    
    logger.info("🎯 LIVE SCANNER ACTIVE - Press Ctrl+C to stop")
    console.rule("🔴 LIVE MARKET SCANNING STARTED", style="bold red")
    
    try:
        while datetime.now(IST) <= market_end:
            current_time = datetime.now(IST)
            
            # Wait for next boundary
            if current_time < next_boundary:
                sleep_seconds = min(30, (next_boundary - current_time).total_seconds())
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
                continue
            
            scan_count += 1
            scan_time = current_time.replace(second=0, microsecond=0)
            
            logger.info(f"🔍 Scan #{scan_count} at {scan_time.strftime('%H:%M')} IST")
            
            # Real-time analysis
            signals_this_scan = []
            current_scores = {}
            high_priority_alerts = []
            
            # Analyze each symbol
            for symbol, timeframe_data in stock_multi_data.items():
                clean_symbol = symbol.replace('-EQ', '')
                filtered_timeframes = filter_timeframe_data(clean_symbol, timeframe_data, scan_time)
                
                if len(filtered_timeframes) < 1:
                    continue
                
                signal, score, oi_status = analyze_signals_with_oi(filtered_timeframes, clean_symbol)
                current_scores[clean_symbol] = score
                
                # Get current price and volume for alerts
                latest_data = None
                for tf in [5, 15, 30]:  # Check shortest timeframes first
                    if tf in filtered_timeframes and not filtered_timeframes[tf].empty:
                        latest_data = filtered_timeframes[tf].iloc[-1]
                        break
                
                current_price = latest_data['Close'] if latest_data is not None else 0
                current_volume = latest_data['Volume'] if latest_data is not None else 0
                
                # Determine alert level
                alert_level = "NORMAL"
                if "Very Strong" in signal:
                    alert_level = "🚨 URGENT"
                elif "Strong" in signal:
                    alert_level = "⚠️ HIGH"
                elif abs(score) >= 15:
                    alert_level = "📢 MEDIUM"
                
                # Include signals worth monitoring (lowered threshold for live)
                if abs(score) >= 10 or "Strong" in signal:
                    prev = previous_scores.get(clean_symbol, 'NA')
                    change_val = 'NA' if isinstance(prev, str) else (score - prev)
                    direction = 'bullish' if score > 0 else 'bearish'
                    flow_tag = infer_institutional_flow(filtered_timeframes)
                    
                    signal_data = {
                        'symbol': clean_symbol, 'signal': signal, 'score': score,
                        'trend': direction, 'change': change_val, 'oi_status': oi_status,
                        'flow': flow_tag, 'price': current_price, 'volume': current_volume, 
                        'alert_level': alert_level
                    }
                    
                    signals_this_scan.append(signal_data)
                    performance_metrics[f"{direction}_signals"] += 1
                    
                    # High priority alerts for immediate action
                    if alert_level in ["🚨 URGENT", "⚠️ HIGH"]:
                        high_priority_alerts.append(signal_data)
            
            previous_scores = current_scores.copy()
            
            # Sort and display results
            signals_this_scan.sort(key=lambda x: abs(x['score']), reverse=True)
            top_bullish = [r for r in signals_this_scan if r['score'] > 0][:15]  # Top 15 for live
            top_bearish = [r for r in signals_this_scan if r['score'] < 0][:15]
            
            # Display results
            if top_bullish or top_bearish:
                render_live_results(scan_time, top_bullish, top_bearish, scan_count)
                
                # Save to CSV
                with open(live_csv, "a", newline='', encoding="utf-8") as f:
                    writer = csv.writer(f)
                    for r in top_bullish + top_bearish:
                        writer.writerow([
                            scan_time.strftime('%H:%M'),
                            r['symbol'], r['signal'], f"{r['score']:.2f}",
                            r['oi_status'], 
                            "CALL BUY" if r['score'] > 0 else "PUT BUY",
                            f"{r['price']:.2f}" if r['price'] > 0 else "NA",
                            int(r['volume']) if r['volume'] > 0 else "NA",
                            r['alert_level']
                        ])
            
            # Handle high priority alerts
            if high_priority_alerts:
                handle_urgent_alerts(high_priority_alerts, scan_time, alerts_log)
            
            # Update next boundary
            next_boundary = next_5min_boundary_ist(datetime.now(IST))
            
            # Periodic data refresh (every 30 minutes)
            if scan_count % 6 == 0:  # Every 30 minutes
                logger.info("🔄 Refreshing data...")
                try:
                    # Refresh data for top performers and high-volume stocks
                    priority_symbols = list(set([r['symbol'] for r in signals_this_scan[:20]]))
                    if priority_symbols:
                        updated_data = prefetch_with_oi(priority_symbols)
                        stock_multi_data.update(updated_data)
                        logger.info(f"✅ Refreshed data for {len(priority_symbols)} priority symbols")
                except Exception as e:
                    logger.warning(f"Data refresh failed: {e}")
            
            # Memory cleanup
            if scan_count % 12 == 0:  # Every hour
                import gc
                gc.collect()
                logger.debug("🧹 Memory cleanup completed")
    
    except KeyboardInterrupt:
        logger.info("👤 Live scanner stopped by user")
    except Exception as e:
        logger.error(f"💥 Live scanner error: {e}")
    finally:
        console.rule("🔴 LIVE SCANNING STOPPED", style="bold red")
        logger.info(f"📊 Final Statistics: {dict(performance_metrics)}")
        logger.info(f"💾 Results saved to: {live_csv}")
        logger.info(f"🚨 Alerts logged to: {alerts_log}")

# CSV Export
def export_to_csv(now_ts, top_bullish, top_bearish, filename):
    with open(filename, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Snapshot Time: {now_ts.strftime('%Y-%m-%d %H:%M')}"])
        writer.writerow(["Top 20 Bullish (Call Opportunities)"])
        writer.writerow(["Stock", "Signal", "Score", "Change", "Trend", "Flow", "OI Status", "Action"])
        if not top_bullish:
            writer.writerow(["No strong bullish signals found."])
        for r in top_bullish:
            ch = r['change']
            change_str = f"{ch:+.2f}" if isinstance(ch, (int, float, np.floating)) else "NA"
            writer.writerow([r['symbol'], r['signal'], f"{r['score']:.2f}", change_str, r['trend'], 
                           r.get('flow', 'Unknown'), r.get('oi_status', 'Normal'), "🔥 CALL BUY"])
        writer.writerow([])
        writer.writerow(["Top 20 Bearish (Put Opportunities)"])
        writer.writerow(["Stock", "Signal", "Score", "Change", "Trend", "Flow", "OI Status", "Action"])
        if not top_bearish:
            writer.writerow(["No strong bearish signals found."])
        for r in top_bearish:
            ch = r['change']
            change_str = f"{ch:+.2f}" if isinstance(ch, (int, float, np.floating)) else "NA"
            writer.writerow([r['symbol'], r['signal'], f"{r['score']:.2f}", change_str, r['trend'], 
                           r.get('flow', 'Unknown'), r.get('oi_status', 'Normal'), "🔥 PUT BUY"])
        writer.writerow([])

# Full-day backtest
def run_oi_backtest_day(day_str: str, stocks):
    """FIXED: Enhanced backtest execution with OI analysis"""
    day_date = datetime.strptime(day_str, "%Y-%m-%d")
    if CONFIG["LOG_LEVEL"] in ("DEBUG", "INFO"):
        logger.info(f"[{day_str}] OI-focused backtest for {len(stocks)} symbols...")
    
    stock_multi_data = prefetch_with_oi(stocks)
    if CONFIG["LOG_LEVEL"] in ("DEBUG", "INFO"):
        logger.info(f"Prefetch complete. {len(stock_multi_data)} symbols with valid data.")
    
    # Ensure we continue with analysis even if no specific OI symbols found
    if len(stock_multi_data) == 0:
        logger.error("No valid data found. Exiting.")
        return

    checkpoints = day_checkpoints_ist(day_date)
    output_filename = day_date.strftime("%Y-%m-%d") + "_oi_options_scan_results.csv"
    
    try:
        if os.path.exists(output_filename):
            os.remove(output_filename)
    except Exception:
        pass

    global previous_scores, last_bull_symbols, last_bear_symbols, performance_metrics
    previous_scores = {}
    last_bull_symbols = set()
    last_bear_symbols = set()
    performance_metrics = defaultdict(int)

    logger.info(f"Running OI analysis for {len(checkpoints)} time checkpoints...")

    for i, asof_ts in enumerate(checkpoints):
        if i % 20 == 0:  # Log progress every 20 checkpoints
            logger.info(f"Processing checkpoint {i+1}/{len(checkpoints)}: {asof_ts.strftime('%H:%M')}")
            
        time_point_aware = asof_ts.replace(second=0, microsecond=0)
        signals_this_scan = []
        current_scores = {}

        for symbol, timeframe_data in stock_multi_data.items():
            clean_symbol = symbol.replace('-EQ', '')
            filtered_timeframes = filter_timeframe_data(clean_symbol, timeframe_data, time_point_aware)
            
            if len(filtered_timeframes) < 1:
                continue

            signal, score, oi_status = analyze_signals_with_oi(filtered_timeframes, clean_symbol)
            current_scores[clean_symbol] = score
            
            # Include strong signals and decent scores for analysis
            if 'Strong' in signal or abs(score) >= 15:  # Include more signals for comprehensive analysis
                prev = previous_scores.get(clean_symbol, 'NA')
                change_val = 'NA' if isinstance(prev, str) else (score - prev)
                direction = 'bullish' if score > 0 else 'bearish'
                flow_tag = infer_institutional_flow(filtered_timeframes)
                
                signals_this_scan.append({
                    'symbol': clean_symbol, 'signal': signal, 'score': score,
                    'trend': direction, 'change': change_val, 'oi_status': oi_status, 'flow': flow_tag
                })
                performance_metrics[f"{direction}_signals"] += 1

        previous_scores = current_scores.copy()
        signals_this_scan.sort(key=lambda x: abs(x['score']), reverse=True)
        top_bullish = [r for r in signals_this_scan if r['score'] > 0][:20]
        top_bearish = [r for r in signals_this_scan if r['score'] < 0][:20]

        # Show results only when we have significant signals
        if top_bullish or top_bearish:
            render_top_lists_with_oi(asof_ts, top_bullish, top_bearish)
            export_to_csv(asof_ts, top_bullish, top_bearish, output_filename)

    logger.info(f"OI-focused backtest complete. Final Metrics: {dict(performance_metrics)}")
    logger.info(f"Results saved to: {output_filename}")

def load_stock_list(file_name):
    """Load stock symbols from file with better error handling"""
    if not os.path.exists(file_name):
        logger.error(f"Stock file {file_name} not found!")
        # Create a sample file with common FNO stocks that have good OI activity
        sample_stocks = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK", "SBIN", "BHARTIARTL",
            "ITC", "KOTAKBANK", "LT", "ASIANPAINT", "AXISBANK", "MARUTI", "SUNPHARMA", "ULTRACEMCO",
            "WIPRO", "NESTLEIND", "HCLTECH", "BAJFINANCE", "TITAN", "POWERGRID", "NTPC", "ONGC",
            "TECHM", "DRREDDY", "BAJAJFINSV", "INDUSINDBK", "CIPLA", "COALINDIA", "GRASIM", "BPCL",
            "TATASTEEL", "HINDALCO", "ADANIPORTS", "BRITANNIA", "DIVISLAB", "TATAMOTORS", "HEROMOTOCO",
            "JSWSTEEL", "SHREECEM", "UPL", "APOLLOHOSP", "BAJAJ_AUTO", "EICHERMOT", "SBILIFE",
            "HDFCLIFE", "PIDILITIND", "ADANIENT", "TATACONSUM", "DABUR", "GODREJCP", "MARICO",
            "COLPAL", "NESTLEIND", "BRITANNIA", "UBL", "BERGEPAINT", "PAGEIND", "SHREECEM",
            "GRASIM", "ULTRAcemco", "JSWSTEEL", "TATASTEEL", "HINDALCO", "VEDL", "NATIONALUM",
            "HINDZINC", "SAIL", "NMDC", "MOIL", "GMRINFRA", "ADANIGREEN", "NTPC", "POWERGRID",
            "RECLTD", "PFC", "IRCTC", "RAILTEL", "CONCOR", "INDIGO", "SPICEJET", "JUBLFOOD",
            "ZOMATO", "NYKAA", "PAYTM", "POLICYBZR", "LICI", "SBICARD", "HDFCAMC", "MFSL",
            "CDSL", "CAMS", "BSE", "MCX", "DMART", "FRETAIL", "TRENT", "ABFRL", "MCDOWELL-N",
            "RADICO", "UBL", "JUBILANT", "PFIZER", "DRREDDY", "SUNPHARMA", "CIPLA", "LUPIN",
            "TORNTPHARM", "BIOCON", "CADILAHC", "GLENMARK", "ALKEM", "AUROPHARMA", "ZYDUSLIFE"
        ]
        
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                for stock in sample_stocks:
                    f.write(f"{stock}\n")
            logger.info(f"Created sample {file_name} with {len(sample_stocks)} FNO stocks (OI-focused)")
            return sample_stocks
        except Exception as e:
            logger.error(f"Could not create sample file: {e}")
            return []
    
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        # Remove any comments or empty lines
        stocks = []
        for line in lines:
            if line and not line.startswith('#'):
                # Handle different formats (symbol only or symbol with description)
                symbol = line.split(',')[0].split('\t')[0].strip().upper()
                if symbol:
                    stocks.append(symbol)
        
        logger.info(f"Loaded {len(stocks)} symbols from {file_name} for OI analysis")
        return stocks
        
    except Exception as e:
        logger.error(f"Error loading stock list from {file_name}: {e}")
        return []

# ======== MAIN EXECUTION WITH LIVE AND BACKTEST MODES ========
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Complete OI-Focused Option Buyer Scanner")
        parser.add_argument("--backtest-date", help="Backtest date in YYYY-MM-DD format")
        parser.add_argument("--stocks-file", default=CONFIG["SHARES_FILE"], help="File containing stock symbols")
        parser.add_argument("--live", action="store_true", help="Run in live market mode")
        parser.add_argument("--test-oi", action="store_true", help="Test OI data availability")
        
        args = parser.parse_args()
        
        # Load stock symbols
        stocks = load_stock_list(args.stocks_file)
        if not stocks:
            logger.error("No valid stock symbols loaded. Please check your shares.txt file.")
            exit(1)
        
        if args.test_oi:
            # TEST MODE: Check OI data availability
            logger.info("🧪 Testing OI data availability...")
            test_stocks = stocks[:20]  # Test first 20 stocks
            logger.info(f"Testing OI data for {len(test_stocks)} symbols...")
            
            stock_multi_data = prefetch_with_oi(test_stocks)
            
            oi_available = []
            no_oi = []
            
            for symbol, timeframe_data in stock_multi_data.items():
                has_oi = False
                for tf, df in timeframe_data.items():
                    if df is not None and 'OI' in df.columns and df['OI'].sum() > 0:
                        has_oi = True
                        break
                
                if has_oi:
                    oi_available.append(symbol)
                else:
                    no_oi.append(symbol)
            
            logger.info(f"✅ Symbols with OI data: {len(oi_available)}")
            if oi_available:
                logger.info(f"OI symbols: {sorted(oi_available)}")
            
            logger.info(f"❌ Symbols without OI data: {len(no_oi)}")
            if no_oi and len(no_oi) <= 10:
                logger.info(f"No OI symbols: {sorted(no_oi)}")
            
        elif args.live:
            # LIVE MARKET MODE
            logger.info("🔴 Starting LIVE market scanner...")
            run_live_scanner(stocks)
            
        elif args.backtest_date:
            # BACKTEST MODE
            try:
                # Validate date format
                datetime.strptime(args.backtest_date, "%Y-%m-%d")
                logger.info(f"📊 Starting OI-focused backtest for {args.backtest_date}")
                run_oi_backtest_day(args.backtest_date, stocks)
            except ValueError:
                logger.error("❌ Invalid date format. Use YYYY-MM-DD format.")
                exit(1)
            except Exception as e:
                logger.error(f"❌ Backtest failed: {e}")
                exit(1)
        else:
            # Show usage information
            print("\n🎯 OI-Focused Option Buyer Scanner")
            print("=" * 50)
            print("Usage:")
            print("  python grok3.py --live                    # Live market scanning")
            print("  python grok3.py --backtest-date 2025-09-24 # Historical backtest")
            print("  python grok3.py --test-oi                 # Test OI data availability")
            print("\nOptions:")
            print("  --stocks-file FILENAME                     # Custom stock list file")
            print("\nFeatures:")
            print("  ✅ Real-time OI surge detection")
            print("  ✅ Call/Put bias analysis")
            print("  ✅ Volume-OI confirmation")
            print("  ✅ Institutional flow detection")
            print("  ✅ 5-minute boundary scanning")
            print("  ✅ Live alerts and CSV export")
            print("  ✅ Market hours validation")
            print("\nFor help: python grok3.py --help")
    
    except KeyboardInterrupt:
        print("\n👤 Interrupted by user. Shutting down gracefully...")
        console.print("[bold yellow]Scanner stopped by user.[/bold yellow]")
    except Exception as e:
        logger.exception(f"💥 Fatal error: {e}")
        console.print(f"[bold red]Fatal error: {e}[/bold red]")
        raise
    finally:
        # Graceful shutdown
        logger.info("🔌 Disconnecting TrueData sessions...")
        for sess in tdhist_pool:
            try:
                if hasattr(sess, 'disconnect'):
                    sess.disconnect()
                elif hasattr(sess, 'close'):
                    sess.close()
            except Exception as e:
                logger.debug(f"Session cleanup error: {e}")
        
        # Final statistics
        if performance_metrics:
            logger.info(f"📊 Final Statistics: {dict(performance_metrics)}")
            
            # Summary report
            total_signals = sum(performance_metrics.values())
            if total_signals > 0:
                console.print("\n📈 [bold green]SESSION SUMMARY[/bold green]")
                console.print(f"Total Signals Generated: [bold cyan]{total_signals}[/bold cyan]")
                
                if 'bullish_signals' in performance_metrics:
                    bull_pct = (performance_metrics['bullish_signals'] / total_signals) * 100
                    console.print(f"Bullish Signals: [bold green]{performance_metrics['bullish_signals']} ({bull_pct:.1f}%)[/bold green]")
                
                if 'bearish_signals' in performance_metrics:
                    bear_pct = (performance_metrics['bearish_signals'] / total_signals) * 100
                    console.print(f"Bearish Signals: [bold red]{performance_metrics['bearish_signals']} ({bear_pct:.1f}%)[/bold red]")
                
                if len(oi_symbols_found) > 0:
                    console.print(f"Symbols with OI Data: [bold yellow]{len(oi_symbols_found)}[/bold yellow]")
        
        logger.info("✅ Shutdown complete")
        print("\n🎯 Thank you for using OI-Focused Option Scanner!")

# ======== ADDITIONAL UTILITY FUNCTIONS ========
def test_api_connection():
    """Test TrueData API connection"""
    try:
        test_session = authenticate_session()
        logger.info("✅ TrueData API connection successful")
        
        # Test a simple data fetch
        test_data = test_session.get_historic_data("RELIANCE", duration="5 D", bar_size="5 min")
        if test_data is not None and not test_data.empty:
            logger.info(f"✅ Test data fetch successful: {len(test_data)} records")
            return True
        else:
            logger.error("❌ Test data fetch failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ API connection test failed: {e}")
        return False

def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check required credentials
    if not CONFIG["TDUSERNAME"] or not CONFIG["TDPASSWORD"]:
        errors.append("TrueData credentials not configured")
    
    # Check market times
    try:
        parse_hhmm(CONFIG["MARKET_START"])
        parse_hhmm(CONFIG["MARKET_END"])
        parse_hhmm(CONFIG["FIRST_RUN_AT"])
    except ValueError:
        errors.append("Invalid market time format")
    
    # Check numeric settings
    if CONFIG["MAX_WORKERS"] <= 0:
        errors.append("MAX_WORKERS must be positive")
    
    if CONFIG["RATE_PER_SECOND_TOTAL"] <= 0:
        errors.append("RATE_PER_SECOND_TOTAL must be positive")
    
    # Check indicator weights
    if not CONFIG["INDICATOR_WEIGHTS"]:
        errors.append("No indicator weights configured")
    
    if errors:
        logger.error("❌ Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    logger.info("✅ Configuration validation passed")
    return True

def create_status_report():
    """Create a status report of the system"""
    report = {
        "timestamp": datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "config": {
            "max_workers": CONFIG["MAX_WORKERS"],
            "td_sessions": CONFIG["TD_HIST_SESSIONS"],
            "rate_limit": CONFIG["RATE_PER_SECOND_TOTAL"],
            "min_bars": CONFIG["MIN_BARS_REQUIRED"]
        },
        "indicators": len(CONFIG["INDICATOR_WEIGHTS"]),
        "timeframes": len(CONFIG["TIMEFRAME_WEIGHTS"]),
        "market_hours": f"{CONFIG['MARKET_START']} - {CONFIG['MARKET_END']}",
        "oi_symbols_found": len(oi_symbols_found) if oi_symbols_found else 0,
        "failed_symbols": len(failed_symbols) if failed_symbols else 0,
        "api_calls_made": api_calls_done
    }
    return report

def print_startup_banner():
    """Print startup banner with system information"""
    console.print("\n" + "="*70, style="bold blue")
    console.print("🎯 [bold cyan]OI-FOCUSED OPTION BUYER SCANNER[/bold cyan] 🎯", justify="center")
    console.print("="*70, style="bold blue")
    console.print(f"Version: [bold green]2.0[/bold green] | Market Focus: [bold yellow]NSE F&O[/bold yellow] | Mode: [bold red]OI Analysis[/bold red]")
    console.print(f"Time: [bold white]{datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S IST')}[/bold white]")
    console.print(f"Sessions: [cyan]{CONFIG['TD_HIST_SESSIONS']}[/cyan] | Workers: [cyan]{CONFIG['MAX_WORKERS']}[/cyan] | Rate: [cyan]{CONFIG['RATE_PER_SECOND_TOTAL']}/s[/cyan]")
    console.print("="*70, style="bold blue")
    
    # Show key features
    console.print("🚀 [bold green]KEY FEATURES:[/bold green]")
    console.print("  ✅ Real-time OI surge detection")
    console.print("  ✅ Call/Put bias analysis with institutional flow")
    console.print("  ✅ Multi-timeframe momentum confirmation")
    console.print("  ✅ Volume-OI cross-validation")
    console.print("  ✅ Live market scanning with alerts")
    console.print("  ✅ Historical backtesting capabilities")
    console.print("="*70, style="bold blue")

# Print startup banner when module is imported
if __name__ == "__main__":
    print_startup_banner()

# ======== END OF COMPLETE CODE ========
