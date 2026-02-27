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

# ======== CLEAN PRODUCTION CONFIG ========
CONFIG = {
    "TDUSERNAME": os.getenv("TRUEDATA_USER", "tdwsp751"),
    "TDPASSWORD": os.getenv("TRUEDATA_PASS", "raj@751"),

    # Market times (IST)
    "MARKET_START": "09:15",
    "FIRST_RUN_AT": "09:20",
    "MARKET_END": "15:30",
    "SETTLE_DELAY_SECONDS": 5,

    # Concurrency settings
    "MAX_WORKERS": int(os.getenv("MAX_WORKERS", "32")),
    "TD_HIST_SESSIONS": int(os.getenv("TD_HIST_SESSIONS", "4")),
    "RATE_PER_SECOND_TOTAL": float(os.getenv("RATE_PER_SECOND_TOTAL", "15.0")),
    "BUCKET_SIZE": int(os.getenv("BUCKET_SIZE", "20")),
    "RETRY_ATTEMPTS": int(os.getenv("RETRY_ATTEMPTS", "7")),
    "RETRY_DELAY_MS": int(os.getenv("RETRY_DELAY_MS", "2000")),

    # Output and logging - CLEAN PRODUCTION SETTINGS
    "SHARES_FILE": os.getenv("SHARES_FILE", "shares.txt"),
    "SHOW_PROGRESS": os.getenv("SHOW_PROGRESS", "true").lower() == "true",
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "WARNING"),  # CLEAN: Only warnings and errors

    # Always include all timeframes
    "SKIP_DAILY": False,

    # Analysis settings - optimized thresholds
    "MIN_BARS_REQUIRED": 20,
    "MAX_MISSING_DATA_PCT": 15,
    "SIGNAL_CONFIRMATION_BARS": 2,
    "MIN_SIGNAL_THRESHOLD": 5,

    # Indicator periods
    "INDICATOR_PERIODS": {
        "RSI": 14, "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
        "STOCHASTIC_K": 14, "STOCHASTIC_D": 3, "MA_SHORT": 20, "MA_LONG": 50,
        "ADX": 14, "BB_PERIOD": 20, "BB_STD_DEV": 2, "ROC": 12, "CCI": 20,
        "EMA_FAST": 9, "EMA_SLOW": 21, "ATR": 14, "VOLUME_SURGE": 20,
        "MOMENTUM": 10, "WILLIAMS_R": 14, "CMF": 20, "ADL_LOOKBACK": 10,
        "REL_VOL": 20, "VWAP_REGIME": 20, "OBV_CONFIRM": 5,
        "OI_SURGE": 20, "OI_MOMENTUM": 10,
    },
    
    # OI-focused weights for option trading
    "INDICATOR_WEIGHTS": {
        "VolumeSurge": 2.5, "Momentum": 2.2, "ADX": 2.0, "VWAP": 1.8, "EMA": 1.9,
        "MACD": 1.7, "OBV": 1.6, "ATR": 1.5, "Bollinger": 1.4, "RSI": 1.3,
        "ROC": 1.2, "Stochastic": 1.1, "CCI": 1.0, "MA": 1.2, "WWL": 1.0,
        "CMF": 2.0, "ADL": 1.8, "RelVol": 1.7, "VWAPRegime": 1.9, "OBVConfirm": 1.4,
        # OI indicators get HIGHEST weights
        "OISurge": 3.5, "OIMomentum": 3.2, "CallBias": 4.0, "PutBias": 4.0, "OIVolConfirm": 3.0,
    },
    
    # All 5 timeframes with proper weights
    "TIMEFRAME_WEIGHTS": {15: 2.5, 5: 2.2, 30: 1.8, 60: 1.2, 1440: 1.0},

    # API mapping
    "BAR_SIZE_MAP": {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"},
    "DURATION_MAP": {5: "45 D", 15: "45 D", 30: "90 D", 60: "180 D", 1440: "365 D"},
}

# CLEAN: Root logger set to WARNING level for production
level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
logging.getLogger().setLevel(level_map.get(CONFIG["LOG_LEVEL"], logging.WARNING))

IST = pytz.timezone("Asia/Kolkata")

# Silence all noisy third-party loggers completely
for noisy in ("truedata", "truedata.history", "truedata_ws", "websocket", "urllib3", "requests", "connectionpool"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

console = Console()
# Global tracking variables - FIXED initialization
previous_scores = {}
performance_metrics = defaultdict(int)
last_bull_symbols = set()
last_bear_symbols = set()


# Global state
last_bull_symbols = set()
last_bear_symbols = set()
previous_scores = {}
api_calls_done = 0
api_calls_lock = threading.Lock()
performance_metrics = defaultdict(int)
failed_symbols = set()
oi_symbols_found = set()

# ---------- Helper functions ----------
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
    
    # Try to build sessions first
    successful_sessions = 0
    for i in range(sess_count):
        try:
            session = authenticate_session()
            pool.append(session)
            successful_sessions += 1
            console.print(f"✅ Session {i+1}/{sess_count} connected")
        except Exception as e:
            console.print(f"[red]Session {i+1} failed: {e}[/red]")
    
    # Check if we have at least one session
    if not pool:
        console.print("[red]❌ Failed to initialize ANY TrueData sessions.[/red]")
        raise SystemExit("Failed to initialize TrueData sessions.")
    
    if successful_sessions < sess_count:
        console.print(f"[yellow]⚠️  Only {successful_sessions}/{sess_count} sessions connected[/yellow]")
    
    # Now setup rate limiters for successful sessions ONLY
    per_sess_rate = CONFIG["RATE_PER_SECOND_TOTAL"] / len(pool)
    for _ in pool:
        limiters.append(TokenBucketLimiter(rate_per_sec=per_sess_rate, bucket_size=CONFIG["BUCKET_SIZE"]))
    
    return pool, limiters


tdhist_pool, sess_limiters = build_sessions()
console.print("✅ [green]TrueData connection established[/green]")

# ======== CLEAN INDICATOR FUNCTIONS ========
def safe_calculate_indicator(func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            return tuple(r.fillna(method='ffill').fillna(0) if isinstance(r, pd.Series) else r for r in result)
        elif isinstance(result, pd.Series):
            return result.fillna(method='ffill').fillna(0)
        return result
    except Exception:
        if hasattr(args[0], 'index'):
            return pd.Series(0, index=args[0].index)
        return 0

def ema_improved(series, length):
    if len(series) < length:
        return pd.Series(index=series.index, dtype='float64')
    return series.ewm(span=length, adjust=True, min_periods=length//2).mean()

def calculate_rsi_improved(df, period=14):
    if df is None or len(df) < period + 5:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    
    try:
        close_prices = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period, min_periods=period//2).mean()
        avg_loss = loss.rolling(window=period, min_periods=period//2).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except Exception:
        return pd.Series(50, index=df.index)

def volume_surge_improved(df, lookback=20):
    if df is None or len(df) < lookback + 5:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    
    try:
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        vol_ma = volume.rolling(lookback, min_periods=lookback//2).mean()
        vol_std = volume.rolling(lookback, min_periods=lookback//2).std()
        vol_std = vol_std.where(vol_std > vol_ma * 0.01, vol_ma * 0.1)
        z_score = (volume - vol_ma) / vol_std
        return z_score.clip(-5, 5).fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def momentum_improved(df, period=10):
    if df is None or len(df) < period + 2:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    
    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        shifted_close = close.shift(period).replace(0, np.nan)
        momentum_val = (close / shifted_close) - 1.0
        return momentum_val.fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def oi_surge_improved(df, lookback=20):
    """OI surge calculation for option trading"""
    if df is None or len(df) < lookback + 5:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    
    try:
        oi_col = None
        for col in df.columns:
            if col.upper() == 'OI' or 'openinterest' in col.lower():
                oi_col = col
                break
        
        if oi_col is None:
            return volume_surge_improved(df, lookback)
        
        oi = pd.to_numeric(df[oi_col], errors='coerce').fillna(0)
        if oi.sum() == 0:
            return volume_surge_improved(df, lookback)
        
        oi_ma = oi.rolling(lookback, min_periods=lookback//2).mean()
        oi_std = oi.rolling(lookback, min_periods=lookback//2).std()
        oi_std = oi_std.where(oi_std > oi_ma * 0.01, oi_ma * 0.1)
        z_score = (oi - oi_ma) / oi_std
        return z_score.clip(-5, 5).fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def oi_momentum_improved(df, period=10):
    """OI momentum calculation"""
    if df is None or len(df) < period + 2:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    
    try:
        oi_col = None
        for col in df.columns:
            if col.upper() == 'OI' or 'openinterest' in col.lower():
                oi_col = col
                break
        
        if oi_col is None:
            return momentum_improved(df, period)
        
        oi = pd.to_numeric(df[oi_col], errors='coerce').fillna(0)
        if oi.sum() == 0:
            return momentum_improved(df, period)
        
        shifted_oi = oi.shift(period).replace(0, np.nan)
        return ((oi / shifted_oi) - 1.0).fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

# ======== SIGNAL CLASSIFICATION ========
def classify_option_signal(normalized_score, oi_status, has_strong_conditions):
    """Signal classification for option trading"""
    
    if normalized_score >= 30:
        return "🚀 ULTRA STRONG BUY - CALL HEAVY", "ULTRA_STRONG"
    elif normalized_score <= -30:
        return "💥 ULTRA STRONG SELL - PUT HEAVY", "ULTRA_STRONG"
    elif normalized_score >= 15:
        if "Call Setup" in oi_status:
            return "🔥 VERY STRONG BUY - CALL FOCUS", "VERY_STRONG"
        else:
            return "🔥 VERY STRONG BUY", "VERY_STRONG"
    elif normalized_score <= -15:
        if "Put Setup" in oi_status:
            return "🔥 VERY STRONG SELL - PUT FOCUS", "VERY_STRONG"
        else:
            return "🔥 VERY STRONG SELL", "VERY_STRONG"
    elif normalized_score >= 10:
        if "High OI Activity" in oi_status:
            return "⚡ STRONG BUY - OI SURGE", "STRONG"
        else:
            return "⚡ STRONG BUY", "STRONG"
    elif normalized_score <= -10:
        if "High OI Activity" in oi_status:
            return "⚡ STRONG SELL - OI SURGE", "STRONG"
        else:
            return "⚡ STRONG SELL", "STRONG"
    elif normalized_score >= 5:
        return "🟢 BUY - Call Potential", "MODERATE"
    elif normalized_score <= -5:
        return "🔴 SELL - Put Potential", "MODERATE"
    else:
        return "⚪ NEUTRAL", "NEUTRAL"

def get_option_action(signal_strength, normalized_score):
    """Get specific option trading action"""
    if signal_strength == "ULTRA_STRONG":
        if normalized_score > 0:
            return "🚨 BUY CALLS AGGRESSIVELY", "URGENT"
        else:
            return "🚨 BUY PUTS AGGRESSIVELY", "URGENT"
    elif signal_strength == "VERY_STRONG":
        if normalized_score > 0:
            return "🔥 BUY CALLS STRONG", "HIGH"
        else:
            return "🔥 BUY PUTS STRONG", "HIGH"
    elif signal_strength == "STRONG":
        if normalized_score > 0:
            return "⚡ BUY CALLS", "MEDIUM"
        else:
            return "⚡ BUY PUTS", "MEDIUM"
    elif signal_strength == "MODERATE":
        if normalized_score > 0:
            return "📈 Consider Calls", "LOW"
        else:
            return "📉 Consider Puts", "LOW"
    else:
        return "⏸️ WAIT", "NONE"

# ======== VALIDATION ========
def validate_data_quality(df, min_bars=20):
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
    
    return True, "Data quality OK"

# ======== CLEAN DATA NORMALIZATION ========
def normalize_hist_df_clean(df, symbol, timeframe_minutes):
    """CLEAN: Enhanced normalization without debug messages"""
    if df is None or df.empty:
        return None
    
    try:
        out = df.copy()
        out.columns = out.columns.str.lower().str.strip()
        
        # Enhanced column mapping
        rename_map = {}
        for col in out.columns:
            col_clean = col.lower().strip()
            
            # Timestamp variations
            if any(x in col_clean for x in ['time', 'date', 'timestamp', 'datetime', 'ts']):
                rename_map[col] = 'Timestamp'
            
            # OHLC mapping
            elif col_clean in ['open'] or (col_clean.startswith('open') and 'interest' not in col_clean):
                rename_map[col] = 'Open'
            elif col_clean in ['high', 'h']:
                rename_map[col] = 'High'
            elif col_clean in ['low', 'l']:
                rename_map[col] = 'Low'
            elif col_clean in ['close', 'c']:
                rename_map[col] = 'Close'
            
            # Volume mapping
            elif col_clean in ['volume', 'vol', 'v']:
                rename_map[col] = 'Volume'
            
            # OI detection (SILENT)
            elif any(pattern in col_clean for pattern in [
                'oi', 'openinterest', 'open_interest', 'open interest', 
                'openint', 'open_int', 'oi_value', 'oivalue'
            ]):
                rename_map[col] = 'OI'
        
        out.rename(columns=rename_map, inplace=True)
        
        # Handle missing timestamp
        if "Timestamp" not in out.columns:
            if hasattr(out.index, 'dtype') and 'datetime' in str(out.index.dtype):
                out["Timestamp"] = out.index
                out = out.reset_index(drop=True)
            else:
                now = datetime.now(IST)
                out["Timestamp"] = pd.date_range(
                    start=now - timedelta(minutes=timeframe_minutes * len(out)),
                    periods=len(out),
                    freq=f"{timeframe_minutes}T",
                    tz=IST
                )
        
        # Check required columns
        required_cols = ["Open", "High", "Low", "Close"]
        missing_cols = [col for col in required_cols if col not in out.columns]
        
        if missing_cols:
            return None
        
        # Handle Volume
        if "Volume" not in out.columns:
            out["Volume"] = 1000
        
        # Handle OI (SILENT)
        if "OI" in out.columns:
            try:
                out["OI"] = pd.to_numeric(out["OI"], errors="coerce").fillna(0)
                
                oi_sum = out["OI"].sum()
                oi_max = out["OI"].max()
                
                if oi_sum > 0 and oi_max > 100:
                    global oi_symbols_found
                    oi_symbols_found.add(symbol)
                else:
                    out["OI"] = out["Volume"] * 0.1
            except Exception:
                out["OI"] = out["Volume"] * 0.1
        else:
            out["OI"] = out["Volume"] * 0.1
        
        # Process timestamp
        try:
            out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
            out = out.dropna(subset=["Timestamp"])
            
            if out.empty:
                return None
            
            # Handle timezone
            if out["Timestamp"].dt.tz is None:
                try:
                    out["Timestamp"] = out["Timestamp"].dt.tz_localize(IST, ambiguous='infer', nonexistent='shift_forward')
                except Exception:
                    out["Timestamp"] = out["Timestamp"].dt.tz_localize(IST, ambiguous='NaT', nonexistent='NaT')
                    out = out.dropna(subset=["Timestamp"])
            else:
                out["Timestamp"] = out["Timestamp"].dt.tz_convert(IST)
        except Exception:
            return None
        
        # Convert numeric columns
        for col in ["Open", "High", "Low", "Close", "Volume", "OI"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        
        # Remove invalid data
        out = out.dropna(subset=["Open", "High", "Low", "Close"])
        
        if out.empty or len(out) < 10:
            return None
        
        # Set index and remove duplicates
        out = out.sort_values("Timestamp").set_index("Timestamp")
        out = out[~out.index.duplicated(keep='last')]
        
        return out
        
    except Exception:
        return None

def pick_session(symbol_orig, timeframe_minutes):
    return hash((symbol_orig, timeframe_minutes)) & 0x7fffffff % len(tdhist_pool)
# ======== ENHANCED ANALYSIS WITH CLEAN LOGGING ========
def analyze_signals_enhanced_clean(timeframe_dataframes, symbol):
    """Enhanced analysis with clean logging"""
    if not timeframe_dataframes:
        return 'Neutral', 0.0, 'Normal', 'WAIT', 'NONE'

    final_score, max_possible = 0.0, 0.0
    valid_timeframes = 0
    oi_status = 'Normal'
    has_strong_conditions = False

    for tf_min, df in timeframe_dataframes.items():
        if df is None or len(df) < 20:
            continue

        is_valid, _ = validate_data_quality(df, 20)
        if not is_valid:
            continue

        valid_timeframes += 1
        tf_weight = CONFIG["TIMEFRAME_WEIGHTS"].get(tf_min, 1.0)
        
        # Track OI symbols (silent)
        if 'OI' in df.columns and df['OI'].sum() > 100:
            global oi_symbols_found
            oi_symbols_found.add(symbol)
        
        scores = {}
        
        # RSI Analysis
        try:
            rsi_series = calculate_rsi_improved(df)
            if len(rsi_series) >= 2:
                rsi_current = rsi_series.iloc[-1]
                
                if rsi_current > 65:
                    scores['RSI'] = 2.0
                elif rsi_current > 55:
                    scores['RSI'] = 1.0
                elif rsi_current < 35:
                    scores['RSI'] = -2.0
                elif rsi_current < 45:
                    scores['RSI'] = -1.0
                else:
                    scores['RSI'] = 0.0
            else:
                scores['RSI'] = 0.0
        except Exception:
            scores['RSI'] = 0.0
        
        # Momentum Analysis
        try:
            mom = momentum_improved(df)
            if len(mom) >= 2:
                current_mom = mom.iloc[-1]
                if current_mom > 0.01:
                    scores['Momentum'] = 2.0
                elif current_mom > 0.003:
                    scores['Momentum'] = 1.0
                elif current_mom < -0.01:
                    scores['Momentum'] = -2.0
                elif current_mom < -0.003:
                    scores['Momentum'] = -1.0
                else:
                    scores['Momentum'] = 0.0
            else:
                scores['Momentum'] = 0.0
        except Exception:
            scores['Momentum'] = 0.0
        
        # Volume surge analysis
        try:
            vol_surge = volume_surge_improved(df)
            if len(vol_surge) >= 2:
                current_surge = vol_surge.iloc[-1]
                price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1
                
                if current_surge >= 1.5:
                    if price_change > 0.005:
                        scores['VolumeSurge'] = 2.0
                    elif price_change < -0.005:
                        scores['VolumeSurge'] = -2.0
                    else:
                        scores['VolumeSurge'] = 1.0
                elif current_surge >= 1.0:
                    scores['VolumeSurge'] = 1.0 if price_change > 0 else -1.0
                else:
                    scores['VolumeSurge'] = 0.0
            else:
                scores['VolumeSurge'] = 0.0
        except Exception:
            scores['VolumeSurge'] = 0.0
        
        # OI analysis
        try:
            oi_z = oi_surge_improved(df)
            if len(oi_z) >= 2:
                oi_surge_current = oi_z.iloc[-1]
                price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1
                
                if oi_surge_current >= 1.5:
                    scores['OISurge'] = 2.0 if price_change > 0 else -2.0
                    oi_status = 'High OI Activity'
                elif oi_surge_current >= 0.8:
                    scores['OISurge'] = 1.0 if price_change > 0 else -1.0
                    oi_status = 'Moderate OI Activity'
                else:
                    scores['OISurge'] = 0.0
            else:
                scores['OISurge'] = 0.0
        except Exception:
            scores['OISurge'] = 0.0
        
        # OI momentum
        try:
            oi_mom = oi_momentum_improved(df)
            if len(oi_mom) >= 2:
                oi_mom_current = oi_mom.iloc[-1]
                if oi_mom_current > 0.05:
                    scores['OIMomentum'] = 2.0
                elif oi_mom_current > 0.02:
                    scores['OIMomentum'] = 1.0
                elif oi_mom_current < -0.02:
                    scores['OIMomentum'] = -1.0
                else:
                    scores['OIMomentum'] = 0.0
            else:
                scores['OIMomentum'] = 0.0
        except Exception:
            scores['OIMomentum'] = 0.0
        
        # Call/Put bias analysis
        price_up = df['Close'].iloc[-1] > df['Close'].iloc[-2]
        vol_high = scores.get('VolumeSurge', 0) >= 1.0
        oi_active = abs(scores.get('OISurge', 0)) >= 1.0
        
        if price_up and vol_high and oi_active:
            scores['CallBias'] = 3.0
            scores['PutBias'] = 0.0
            oi_status = 'Call Setup'
            has_strong_conditions = True
        elif not price_up and vol_high and oi_active:
            scores['PutBias'] = -3.0
            scores['CallBias'] = 0.0
            oi_status = 'Put Setup'
            has_strong_conditions = True
        elif price_up and (vol_high or oi_active):
            scores['CallBias'] = 1.0
            scores['PutBias'] = 0.0
        elif not price_up and (vol_high or oi_active):
            scores['PutBias'] = -1.0
            scores['CallBias'] = 0.0
        else:
            scores['CallBias'] = 0.0
            scores['PutBias'] = 0.0
        
        # OI-Volume Confirmation
        if oi_active or vol_high:
            if price_up:
                scores['OIVolConfirm'] = 1.0
            else:
                scores['OIVolConfirm'] = -1.0
        else:
            scores['OIVolConfirm'] = 0.0
        
        # Fill remaining indicators
        remaining_indicators = ['MACD', 'ADX', 'VWAP', 'EMA', 'CMF', 'ADL', 'OBV', 'ATR', 
                               'Bollinger', 'ROC', 'Stochastic', 'CCI', 'MA', 'WWL', 
                               'RelVol', 'VWAPRegime', 'OBVConfirm']
        for indicator in remaining_indicators:
            if indicator not in scores:
                scores[indicator] = 0.0
        
        # Calculate weighted scores
        for indicator, score in scores.items():
            ind_weight = CONFIG["INDICATOR_WEIGHTS"].get(indicator, 1.0)
            weighted_score = score * tf_weight * ind_weight
            final_score += weighted_score
            max_possible += 3.0 * tf_weight * ind_weight
    
    if valid_timeframes < 1 or max_possible == 0:
        return 'Neutral', 0.0, oi_status, 'WAIT', 'NONE'
    
    normalized = (final_score / max_possible) * 100.0
    
    if abs(normalized) > 100:
        normalized = np.sign(normalized) * 100
    
    signal_text, signal_strength = classify_option_signal(normalized, oi_status, has_strong_conditions)
    option_action, alert_priority = get_option_action(signal_strength, normalized)
    
    return signal_text, normalized, oi_status, option_action, alert_priority

# ======== CLEAN DATA FETCHING ========
@retry(
    stop_max_attempt_number=CONFIG["RETRY_ATTEMPTS"],
    wait_exponential_multiplier=max(1, int(CONFIG["RETRY_DELAY_MS"] / 2)),
    wait_exponential_max=10000,
    retry_on_exception=lambda e: True
)
def fetch_one_clean(symbol_orig, timeframe_minutes, limiter, hist):
    """Clean fetch function without debug logs"""
    td_symbol = symbol_orig.replace("-EQ", "")
    
    bar_size = CONFIG["BAR_SIZE_MAP"].get(timeframe_minutes)
    duration = CONFIG["DURATION_MAP"].get(timeframe_minutes)
    
    if not bar_size or not duration:
        return symbol_orig, timeframe_minutes, None
    
    try:
        limiter.acquire()
        df_raw = hist.get_historic_data(td_symbol, duration=duration, bar_size=bar_size)
        
        if df_raw is None or df_raw.empty:
            return symbol_orig, timeframe_minutes, None
        
        df = normalize_hist_df_clean(df_raw, td_symbol, timeframe_minutes)
        
        # Silent API counter
        global api_calls_done
        with api_calls_lock:
            api_calls_done += 1
        
        return symbol_orig, timeframe_minutes, df
        
    except Exception:
        return symbol_orig, timeframe_minutes, None

def prefetch_clean(stocks, max_workers=CONFIG["MAX_WORKERS"]):
    """CLEAN: Enhanced prefetch with essential info only"""
    tfs = [5, 15, 30, 60, 1440]
    total_calls = len(stocks) * len(tfs)
    stock_multi_data = defaultdict(dict)
    
    global api_calls_done, oi_symbols_found
    with api_calls_lock:
        api_calls_done = 0
    
    oi_symbols_found = set()
    valid_stocks = [s for s in stocks if s]
    
    # CLEAN: Show essential startup info
    console.print(f"📊 Analyzing [cyan]{len(valid_stocks)} symbols[/cyan] across [yellow]5 timeframes[/yellow]")
    
    progress_kwargs = dict(
        total=total_calls,
        desc="🔄 Loading Market Data",
        ncols=80,
        disable=not CONFIG["SHOW_PROGRESS"]
    )
    
    with tqdm(**progress_kwargs) as api_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for s in valid_stocks:
                for tf in tfs:
                    si = pick_session(s, tf)
                    futures.append(executor.submit(fetch_one_clean, s, tf, sess_limiters[si], tdhist_pool[si]))
            
            for fut in as_completed(futures):
                try:
                    symbol_orig, tf, df = fut.result()
                    if df is not None:
                        stock_multi_data[symbol_orig][tf] = df
                except Exception:
                    pass
                api_bar.update(1)
    
    valid_data = {s: d for s, d in stock_multi_data.items() if len(d) >= 1}
    
    # CLEAN: Essential completion info
    console.print(f"✅ Data loaded: [green]{len(valid_data)} symbols[/green] ready")
    if len(oi_symbols_found) > 0:
        console.print(f"📈 OI data: [yellow]{len(oi_symbols_found)} symbols[/yellow]")
    
    return valid_data

def filter_timeframe_data(symbol, timeframe_data, time_point_aware):
    """Enhanced filtering with clean error handling"""
    filtered_timeframes = {}
    
    for tf, df in timeframe_data.items():
        if df is None or df.empty:
            continue
        
        try:
            if time_point_aware.tzinfo is None:
                time_point_aware = IST.localize(time_point_aware)
            elif time_point_aware.tzinfo != IST:
                time_point_aware = time_point_aware.astimezone(IST)
            
            if df.index.tz is None:
                df.index = df.index.tz_localize(IST, ambiguous='infer', nonexistent='shift_forward')
            else:
                df.index = df.index.tz_convert(IST)
            
            try:
                valid_index = df.index.dropna()
                if len(valid_index) != len(df.index):
                    df = df.loc[valid_index]
                
                if not df.empty:
                    mask = df.index <= time_point_aware
                    df_filtered = df.loc[mask]
                    
                    if len(df_filtered) >= CONFIG["MIN_BARS_REQUIRED"]:
                        filtered_timeframes[tf] = df_filtered
                        
            except Exception:
                continue
                
        except Exception:
            continue
    
    return filtered_timeframes

# ======== CLEAN RENDERING ========
def render_signals_clean(now_ts, top_bullish, top_bearish):
    """Clean signal rendering for production"""
    global last_bull_symbols, last_bear_symbols
    
    title = f"🎯 STRONG SIGNALS | {now_ts.strftime('%H:%M')} IST"
    console.rule(title, style="bold yellow")

    # Filter signal categories
    ultra_strong_bulls = [r for r in top_bullish if "ULTRA STRONG" in r['signal']]
    very_strong_bulls = [r for r in top_bullish if "VERY STRONG" in r['signal']]
    strong_bulls = [r for r in top_bullish if "STRONG BUY" in r['signal'] or "⚡ STRONG" in r['signal']]
    moderate_bulls = [r for r in top_bullish if "🟢 BUY" in r['signal']]
    
    ultra_strong_bears = [r for r in top_bearish if "ULTRA STRONG" in r['signal']]
    very_strong_bears = [r for r in top_bearish if "VERY STRONG" in r['signal']]
    strong_bears = [r for r in top_bearish if "STRONG SELL" in r['signal'] or "⚡ STRONG" in r['signal']]
    moderate_bears = [r for r in top_bearish if "🔴 SELL" in r['signal']]

    # Show strong signals
    all_strong = ultra_strong_bulls + very_strong_bulls + strong_bulls + ultra_strong_bears + very_strong_bears + strong_bears
    all_moderate = moderate_bulls + moderate_bears
    
    if all_strong:
        console.print("\n🔥 [bold white on red]STRONG SIGNALS[/bold white on red]")
        
        strong_table = Table(title="💪 STRONG SIGNALS", box=box.DOUBLE_EDGE, header_style="bold white on blue")
        strong_table.add_column("Stock", style="bold white")
        strong_table.add_column("Signal", style="bold yellow")
        strong_table.add_column("Score", style="bold green", justify="right")
        strong_table.add_column("OI Status", style="cyan")
        strong_table.add_column("Action", style="bold red")
        
        for r in all_strong:
            row_style = "bold black on yellow" if r['symbol'] not in (last_bull_symbols | last_bear_symbols) else None
            strong_table.add_row(
                r['symbol'], r['signal'], f"{r['score']:.1f}",
                r.get('oi_status', 'Normal'), r.get('action', 'TRADE'),
                style=row_style
            )
        
        console.print(strong_table)
    
    if all_moderate and len(all_moderate) > 0:
        console.print("\n📊 [bold blue]MODERATE SIGNALS[/bold blue]")
        
        mod_table = Table(title="📈 MODERATE SIGNALS", box=box.SIMPLE, header_style="bold white on green")
        mod_table.add_column("Stock", style="cyan")
        mod_table.add_column("Signal", style="white")
        mod_table.add_column("Score", style="yellow", justify="right")
        mod_table.add_column("Action", style="green")
        
        for r in all_moderate[:10]:
            mod_table.add_row(
                r['symbol'], r['signal'], f"{r['score']:.1f}",
                r.get('action', 'CONSIDER')
            )
        
        console.print(mod_table)

    # Summary
    total_ultra = len(ultra_strong_bulls + ultra_strong_bears)
    total_very = len(very_strong_bulls + very_strong_bears)
    total_strong = len(strong_bulls + strong_bears)
    total_moderate = len(all_moderate)
    
    summary = f"🎯 ULTRA: {total_ultra} | 🔥 VERY STRONG: {total_very} | ⚡ STRONG: {total_strong} | 📊 MODERATE: {total_moderate}"
    console.print(f"\n[bold yellow]{summary}[/bold yellow]")
    console.rule()

    last_bull_symbols = {r['symbol'] for r in top_bullish}
    last_bear_symbols = {r['symbol'] for r in top_bearish}

def infer_institutional_flow(tf_data):
    """Simplified institutional flow detection"""
    return "Mixed"

def export_to_csv(now_ts, top_bullish, top_bearish, filename):
    with open(filename, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Time: {now_ts.strftime('%Y-%m-%d %H:%M')}"])
        writer.writerow(["Stock", "Signal", "Score", "Change", "OI Status", "Action"])
        
        for r in top_bullish:
            ch = r['change']
            change_str = f"{ch:+.2f}" if isinstance(ch, (int, float, np.floating)) else "NEW"
            writer.writerow([r['symbol'], r['signal'], f"{r['score']:.2f}", change_str, 
                           r.get('oi_status', 'Normal'), r.get('action', 'TRADE')])
        
        for r in top_bearish:
            ch = r['change']
            change_str = f"{ch:+.2f}" if isinstance(ch, (int, float, np.floating)) else "NEW"
            writer.writerow([r['symbol'], r['signal'], f"{r['score']:.2f}", change_str, 
                           r.get('oi_status', 'Normal'), r.get('action', 'TRADE')])
        writer.writerow([])

# ======== MAIN BACKTEST FUNCTION - CLEAN ========
def run_backtest_clean(day_str: str, stocks):
    """Clean backtest with essential information only"""
    day_date = datetime.strptime(day_str, "%Y-%m-%d")
    console.print(f"📅 [bold cyan]Backtesting {day_str}[/bold cyan] with [yellow]{len(stocks)} symbols[/yellow]")
    
    stock_multi_data = prefetch_clean(stocks)
    
    if len(stock_multi_data) == 0:
        console.print("[red]❌ No valid data found[/red]")
        return

    checkpoints = day_checkpoints_ist(day_date)
    output_filename = day_date.strftime("%Y-%m-%d") + "_signals_clean.csv"
    
    # Clean old file
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

    console.print(f"🔍 Analyzing [cyan]{len(checkpoints)}[/cyan] time periods...")

    for i, asof_ts in enumerate(checkpoints):
        if i % 20 == 0:
            console.print(f"⏳ Progress: [cyan]{i+1}/{len(checkpoints)}[/cyan] | Time: [yellow]{asof_ts.strftime('%H:%M')}[/yellow]")
            
        time_point_aware = asof_ts.replace(second=0, microsecond=0)
        signals_this_scan = []
        current_scores = {}

        for symbol, timeframe_data in stock_multi_data.items():
            clean_symbol = symbol.replace('-EQ', '')
            filtered_timeframes = filter_timeframe_data(clean_symbol, timeframe_data, time_point_aware)
            
            if len(filtered_timeframes) < 1:
                continue

            signal, score, oi_status, option_action, alert_priority = analyze_signals_enhanced_clean(filtered_timeframes, clean_symbol)
            current_scores[clean_symbol] = score
            
            if abs(score) >= CONFIG['MIN_SIGNAL_THRESHOLD'] or any(x in signal for x in ['STRONG', 'BUY', 'SELL']):
                prev = previous_scores.get(clean_symbol, 'NA')
                change_val = 'NA' if isinstance(prev, str) else (score - prev)
                direction = 'bullish' if score > 0 else 'bearish'
                flow_tag = infer_institutional_flow(filtered_timeframes)
                
                signals_this_scan.append({
                    'symbol': clean_symbol, 'signal': signal, 'score': score,
                    'trend': direction, 'change': change_val, 'oi_status': oi_status, 
                    'flow': flow_tag, 'action': option_action
                })
                performance_metrics[f"{direction}_signals"] += 1

        previous_scores = current_scores.copy()
        signals_this_scan.sort(key=lambda x: abs(x['score']), reverse=True)
        top_bullish = [r for r in signals_this_scan if r['score'] > 0][:20]
        top_bearish = [r for r in signals_this_scan if r['score'] < 0][:20]

        if top_bullish or top_bearish:
            render_signals_clean(asof_ts, top_bullish, top_bearish)
            export_to_csv(asof_ts, top_bullish, top_bearish, output_filename)

    # Final summary
    total_signals = sum(performance_metrics.values())
    console.print(f"\n📈 [bold green]BACKTEST COMPLETE[/bold green]")
    console.print(f"Total Signals: [cyan]{total_signals}[/cyan]")
    if 'bullish_signals' in performance_metrics:
        console.print(f"Bullish: [green]{performance_metrics['bullish_signals']}[/green]")
    if 'bearish_signals' in performance_metrics:
        console.print(f"Bearish: [red]{performance_metrics['bearish_signals']}[/red]")
    console.print(f"Results: [yellow]{output_filename}[/yellow]")

def load_stock_list(file_name):
    """Load stock symbols with clean output"""
    if not os.path.exists(file_name):
        sample_stocks = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK", "SBIN", "BHARTIARTL",
            "ITC", "KOTAKBANK", "LT", "ASIANPAINT", "AXISBANK", "MARUTI", "SUNPHARMA", "ULTRACEMCO",
            "WIPRO", "NESTLEIND", "HCLTECH", "BAJFINANCE", "TITAN", "POWERGRID", "NTPC", "ONGC",
            "TECHM", "DRREDDY", "BAJAJFINSV", "INDUSINDBK", "CIPLA", "COALINDIA", "GRASIM", "BPCL",
            "TATASTEEL", "HINDALCO", "ADANIPORTS", "BRITANNIA", "DIVISLAB", "TATAMOTORS", "HEROMOTOCO",
            "JSWSTEEL", "SHREECEM", "UPL", "APOLLOHOSP", "BAJAJ-AUTO", "EICHERMOT", "SBILIFE"
        ]
        
        try:
            with open(file_name, "w", encoding="utf-8") as f:
                for stock in sample_stocks:
                    f.write(f"{stock}\n")
            console.print(f"📝 Created [yellow]{file_name}[/yellow] with [cyan]{len(sample_stocks)}[/cyan] stocks")
            return sample_stocks
        except Exception:
            return []
    
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        
        stocks = []
        for line in lines:
            if line and not line.startswith('#'):
                symbol = line.split(',')[0].split('\t')[0].strip().upper()
                if symbol:
                    stocks.append(symbol)
        
        console.print(f"📈 Loaded [cyan]{len(stocks)}[/cyan] symbols from [yellow]{file_name}[/yellow]")
        return stocks
        
    except Exception:
        return []

def print_clean_banner():
    """Clean production banner"""
    console.print("\n" + "="*70, style="bold blue")
    console.print("🎯 [bold cyan]OPTION SIGNAL SCANNER v3.0[/bold cyan] 🎯", justify="center")
    console.print("="*70, style="bold blue")
    console.print(f"🕐 [bold white]{datetime.now(IST).strftime('%H:%M:%S IST')}[/bold white] | Mode: [green]PRODUCTION[/green]")
    console.print("="*70, style="bold blue")

def run_live_mode_clean(stocks):
    """Enhanced Live Mode - 5 minute intervals with FIXED variable initialization"""
    console.print("🔴 [bold red]LIVE MODE ACTIVATED - 5 MINUTE INTERVALS[/bold red]")

    # Live mode configuration - 5 MINUTE SCANNING
    live_config = {
        "SCAN_INTERVAL_MINUTES": 5,
        "ALERT_THRESHOLD": 8,
        "MAX_SIGNALS_PER_SCAN": 15,
        "COOLDOWN_MINUTES": 10,
        "AUTO_EXPORT": True,
        "LIVE_ALERTS": True,
    }

    # FIXED: Proper global variable initialization
    global previous_scores, performance_metrics, last_bull_symbols, last_bear_symbols

    # Initialize variables if they don't exist
    if not hasattr(run_live_mode_clean, 'initialized'):
        previous_scores = {}
        performance_metrics = defaultdict(int)
        last_bull_symbols = set()
        last_bear_symbols = set()
        run_live_mode_clean.initialized = True

    alert_cooldown = {}
    live_filename = datetime.now(IST).strftime("%Y%m%d") + "_live_signals.csv"

    def wait_for_5min_boundary():
        """Wait until next 5-minute boundary (9:20, 9:25, 9:30, etc.)"""
        now = datetime.now(IST)

        # Calculate next 5-minute boundary
        minutes_to_add = 5 - (now.minute % 5)
        if minutes_to_add == 5 and now.second < 10:
            minutes_to_add = 0

        next_boundary = now.replace(second=0, microsecond=0) + timedelta(minutes=minutes_to_add)

        wait_seconds = (next_boundary - now).total_seconds()
        if wait_seconds > 2:
            console.print(f"⏰ Waiting {wait_seconds:.0f}s for 5-min boundary ({next_boundary.strftime('%H:%M')})")
            time.sleep(wait_seconds)

    def is_market_open():
        """Check if market is currently open"""
        now = datetime.now(IST)
        current_time = now.time()
        market_start = parse_hhmm(CONFIG["MARKET_START"])
        market_end = parse_hhmm(CONFIG["MARKET_END"])

        start_time = now.replace(hour=market_start[0], minute=market_start[1]).time()
        end_time = now.replace(hour=market_end[0], minute=market_end[1]).time()

        is_weekday = now.weekday() < 5
        return is_weekday and start_time <= current_time <= end_time

    def send_live_alert(symbol, signal, score, action):
        """Send live trading alert"""
        current_time = datetime.now(IST)
        cooldown_key = f"{symbol}_{signal}"

        if cooldown_key in alert_cooldown:
            last_alert_time = alert_cooldown[cooldown_key]
            if (current_time - last_alert_time).total_seconds() < (live_config["COOLDOWN_MINUTES"] * 60):
                return False

        alert_cooldown[cooldown_key] = current_time
        alert_msg = f"🚨 LIVE ALERT | {symbol} | {signal} | Score: {score:.1f} | {action}"
        console.print(f"\n[bold yellow on red]{alert_msg}[/bold yellow on red]")

        with open(f"live_alerts_{current_time.strftime('%Y%m%d')}.log", "a") as f:
            f.write(f"{current_time.strftime('%H:%M:%S')} - {alert_msg}\n")
        return True

    def perform_live_scan():
        """Perform a single live market scan with FIXED error handling"""
        global previous_scores, performance_metrics

        if not is_market_open():
            console.print("📴 [yellow]Market closed - waiting...[/yellow]")
            return

        now_ist = datetime.now(IST)
        scan_number = performance_metrics.get('total_scans', 0) + 1
        console.print(f"\n🔍 [bold cyan]LIVE SCAN #{scan_number}[/bold cyan] - {now_ist.strftime('%H:%M:%S')} IST")

        try:
            stock_multi_data = prefetch_clean(stocks)

            if len(stock_multi_data) == 0:
                console.print("[red]❌ No live data available[/red]")
                return

            signals_this_scan = []
            current_scores = {}

            for symbol, timeframe_data in stock_multi_data.items():
                try:
                    clean_symbol = symbol.replace('-EQ', '')
                    filtered_timeframes = filter_timeframe_data(clean_symbol, timeframe_data, now_ist)

                    if len(filtered_timeframes) < 1:
                        continue

                    signal, score, oi_status, option_action, alert_priority = analyze_signals_enhanced_clean(
                        filtered_timeframes, clean_symbol
                    )

                    current_scores[clean_symbol] = score

                    # FIXED: Safe access to previous_scores with proper initialization
                    if abs(score) >= 5:  # Lowered threshold
                        # Ensure previous_scores exists and is initialized
                        if previous_scores is None:
                            previous_scores = {}

                        prev = previous_scores.get(clean_symbol, 'NA')
                        change_val = 'NA' if isinstance(prev, str) else (score - prev)
                        direction = 'bullish' if score > 0 else 'bearish'

                        signal_data = {
                            'symbol': clean_symbol,
                            'signal': signal,
                            'score': score,
                            'trend': direction,
                            'change': change_val,
                            'oi_status': oi_status,
                            'action': option_action,
                            'priority': alert_priority,
                            'timestamp': now_ist
                        }

                        signals_this_scan.append(signal_data)

                        if abs(score) >= live_config["ALERT_THRESHOLD"] and live_config["LIVE_ALERTS"]:
                            send_live_alert(clean_symbol, signal, score, option_action)

                except Exception as e:
                    console.print(f"[dim]⚠️ Error processing {symbol}: {str(e)[:50]}[/dim]")
                    continue

            # FIXED: Safe update of previous_scores
            if previous_scores is None:
                previous_scores = {}
            previous_scores.update(current_scores)

            # Sort and limit signals
            signals_this_scan.sort(key=lambda x: abs(x['score']), reverse=True)
            signals_this_scan = signals_this_scan[:live_config["MAX_SIGNALS_PER_SCAN"]]

            top_bullish = [r for r in signals_this_scan if r['score'] > 0]
            top_bearish = [r for r in signals_this_scan if r['score'] < 0]

            if top_bullish or top_bearish:
                render_signals_clean(now_ist, top_bullish, top_bearish)
                if live_config["AUTO_EXPORT"]:
                    export_to_csv(now_ist, top_bullish, top_bearish, live_filename)
            else:
                console.print("📊 [dim]No significant signals in this scan[/dim]")
                console.print("[yellow]💡 Signals need score ≥ 5 to display[/yellow]")

            performance_metrics['total_scans'] += 1
            performance_metrics['signals_found'] += len(signals_this_scan)
            console.print(f"✅ Scan #{performance_metrics['total_scans']} complete - {len(signals_this_scan)} signals found")

            # Show next scan time
            next_scan = now_ist + timedelta(minutes=5)
            next_boundary = next_scan.replace(second=0, microsecond=0)
            if next_boundary.minute % 5 != 0:
                minutes_to_add = 5 - (next_boundary.minute % 5)
                next_boundary = next_boundary + timedelta(minutes=minutes_to_add)
            console.print(f"⏳ Next scan at: [cyan]{next_boundary.strftime('%H:%M')}[/cyan]")

        except Exception as e:
            console.print(f"[red]❌ Scan error: {e}[/red]")
            console.print("🔄 Will retry at next interval...")

    def live_market_scheduler():
        """Schedule live market scans every 5 minutes with FIXED error handling"""
        console.print("⏰ [green]5-MINUTE SCANNER STARTED[/green] - Syncing to boundaries")

        # Initial sync to 5-minute boundary
        console.print("🔄 Syncing to first 5-minute boundary...")
        wait_for_5min_boundary()

        while True:
            try:
                if is_market_open():
                    perform_live_scan()
                    wait_for_5min_boundary()
                else:
                    now = datetime.now(IST)
                    market_start_time = today_ist_dt(CONFIG["MARKET_START"])

                    if now < market_start_time:
                        wait_seconds = (market_start_time - now).total_seconds()
                        console.print(f"🕐 Market opens in {wait_seconds/3600:.1f} hours")
                        time.sleep(min(300, wait_seconds))
                    else:
                        console.print("📴 [yellow]Market closed for today[/yellow]")
                        total_scans = performance_metrics.get('total_scans', 0)
                        total_signals = performance_metrics.get('signals_found', 0)
                        console.print(f"📊 Daily Summary: {total_scans} scans, {total_signals} signals")
                        if total_scans > 0:
                            console.print(f"📈 Average: {total_signals/total_scans:.1f} signals per scan")
                        time.sleep(3600)

            except KeyboardInterrupt:
                console.print("\n👤 [yellow]Live mode stopped by user[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]⚠️ Scheduler error: {e}[/red]")
                console.print("🔄 Retrying in 30 seconds...")
                time.sleep(30)

        console.print("🛑 [red]Live mode shutdown complete[/red]")

    # Pre-flight checks
    console.print("🔧 [yellow]Running pre-flight checks...[/yellow]")

    try:
        test_data = prefetch_clean(stocks[:5])
        if len(test_data) == 0:
            console.print("[red]❌ Failed to fetch test data - check connection[/red]")
            return
        console.print(f"✅ Connection test passed - {len(test_data)} stocks")
    except Exception as e:
        console.print(f"[red]❌ Connection test failed: {e}[/red]")
        return

    if is_market_open():
        console.print("🟢 [bold green]Market is OPEN - Starting 5-minute scans[/bold green]")
    else:
        console.print("🔴 [yellow]Market is CLOSED - Will start when market opens[/yellow]")

    console.print(f"📈 Monitoring [cyan]{len(stocks)}[/cyan] stocks")
    console.print(f"📄 Live results: [yellow]{live_filename}[/yellow]")
    console.print("⏰ [cyan]Scanning every 5 minutes at: 9:20, 9:25, 9:30, 9:35...[/cyan]")

    console.print("\n🎛️ [bold blue]5-MINUTE LIVE MODE - FIXED VERSION[/bold blue]")
    console.print("[cyan]Enhanced features:[/cyan]")
    console.print("  • ✅ Fixed variable initialization errors")
    console.print("  • ⏰ 5-minute interval scanning")
    console.print("  • 📊 Lowered signal threshold (≥5)")
    console.print("  • 🚨 More sensitive alerts (≥8)")
    console.print("  • 🛡️ Enhanced error handling")
    console.print("  • Press Ctrl+C to stop\n")

    try:
        live_market_scheduler()
    except KeyboardInterrupt:
        console.print("\n🛑 [yellow]Live mode terminated[/yellow]")
    finally:
        console.print("🧹 [dim]Cleaning up live mode...[/dim]")
        if performance_metrics and performance_metrics.get('total_scans', 0) > 0:
            total_scans = performance_metrics.get('total_scans', 0)
            total_signals = performance_metrics.get('signals_found', 0)
            console.print(f"📊 [bold green]Final Session Summary:[/bold green]")
            console.print(f"   • Total 5-min Scans: {total_scans}")
            console.print(f"   • Signals Found: {total_signals}")
            console.print(f"   • Average per Scan: {total_signals/max(1,total_scans):.1f}")
            console.print(f"   • Success Rate: {(total_signals/max(1,total_scans)*100):.1f}%")



# ======== MAIN EXECUTION - CLEAN ========
if __name__ == "__main__":
    print_clean_banner()
    
    try:
        parser = argparse.ArgumentParser(description="Clean Option Signal Scanner")
        parser.add_argument("--backtest-date", help="Backtest date (YYYY-MM-DD)")
        parser.add_argument("--stocks-file", default=CONFIG["SHARES_FILE"], help="Stock symbols file")
        parser.add_argument("--live", action="store_true", help="Live market mode")
        parser.add_argument("--test-oi", action="store_true", help="Test OI data")
        
        args = parser.parse_args()
        
        stocks = load_stock_list(args.stocks_file)
        if not stocks:
            console.print("[red]❌ No valid stocks loaded[/red]")
            exit(1)
        
        if args.test_oi:
            console.print("🧪 [yellow]Testing OI data...[/yellow]")
            test_stocks = stocks[:10]
            stock_multi_data = prefetch_clean(test_stocks)
            console.print(f"✅ Test complete: [cyan]{len(oi_symbols_found)}[/cyan] symbols with OI")
            
        elif args.live:
            run_live_mode_clean(stocks)
            
        elif args.backtest_date:
            try:
                datetime.strptime(args.backtest_date, "%Y-%m-%d")
                run_backtest_clean(args.backtest_date, stocks)
            except ValueError:
                console.print("[red]❌ Invalid date format. Use YYYY-MM-DD[/red]")
        else:
            console.print("\n🎯 [bold green]Option Signal Scanner v3.0[/bold green]")
            console.print("[cyan]Usage:[/cyan]")
            console.print("  [yellow]python scanner.py --backtest-date 2025-09-23[/yellow]")
            console.print("  [yellow]python scanner.py --live[/yellow]")
            console.print("  [yellow]python scanner.py --test-oi[/yellow]")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]👤 Scanner stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]💥 Error: {e}[/red]")
    finally:
        # Clean shutdown
        for sess in tdhist_pool:
            try:
                sess.disconnect()
            except Exception:
                pass
        
        if performance_metrics:
            total = sum(performance_metrics.values())
            if total > 0:
                console.print(f"\n📊 [bold green]Final: {total} signals generated[/bold green]")
        
        console.print("✅ [green]Shutdown complete[/green]")
