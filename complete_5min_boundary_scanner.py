
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

# ======== ENHANCED CONFIG FOR EXPERIENCED OPTION BUYER ========
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
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "WARNING"),
    "SKIP_DAILY": False,

    # Analysis settings - optimized thresholds for option trading
    "MIN_BARS_REQUIRED": 20,
    "MAX_MISSING_DATA_PCT": 15,
    "SIGNAL_CONFIRMATION_BARS": 2,
    "MIN_SIGNAL_THRESHOLD": 8,

    # Enhanced indicator periods for option trading
    "INDICATOR_PERIODS": {
        "RSI": 14, "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
        "STOCHASTIC_K": 14, "STOCHASTIC_D": 3, "MA_SHORT": 20, "MA_LONG": 50,
        "ADX": 14, "BB_PERIOD": 20, "BB_STD_DEV": 2, "ROC": 12, "CCI": 20,
        "EMA_FAST": 9, "EMA_SLOW": 21, "ATR": 14, "VOLUME_SURGE": 20,
        "MOMENTUM": 10, "WILLIAMS_R": 14, "CMF": 20, "ADL_LOOKBACK": 10,
        "REL_VOL": 20, "VWAP_REGIME": 20, "OBV_CONFIRM": 5,
        "OI_SURGE": 20, "OI_MOMENTUM": 10, "PRICE_VELOCITY": 5,
        "GAMMA_SQUEEZE": 3, "IV_CRUSH": 10, "OPTION_FLOW": 15
    },

    # ======== EXPERIENCED OPTION BUYER WEIGHTS ========
    "INDICATOR_WEIGHTS": {
        # PRIMARY: OI & Volume Analysis (Highest weights)
        "OISurge": 5.0, "OIMomentum": 4.5, "VolumeSurge": 4.2, "OIVolConfirm": 4.0,
        "CallBias": 4.8, "PutBias": 4.8, "OptionFlow": 4.5,

        # SECONDARY: Price & Momentum
        "Momentum": 3.8, "PriceVelocity": 3.5, "ADX": 3.2, "GammaSqueezeRisk": 3.0, "VWAP": 2.8,

        # TERTIARY: Technical Confirmation
        "EMA": 2.5, "MACD": 2.3, "ATR": 2.2, "RSI": 2.0, "Bollinger": 1.8, "Stochastic": 1.5,

        # QUATERNARY: Volume & Flow Analysis
        "CMF": 2.8, "OBV": 2.5, "RelVol": 2.3, "ADL": 2.2, "VWAPRegime": 2.0, "OBVConfirm": 1.8,

        # QUINARY: Support indicators
        "ROC": 1.8, "CCI": 1.5, "MA": 1.5, "WWL": 1.3, "IVCrushRisk": 2.5,
    },

    # Enhanced timeframe weights for option trading
    "TIMEFRAME_WEIGHTS": {5: 3.0, 15: 2.8, 30: 2.2, 60: 1.8, 1440: 1.5},
    "BAR_SIZE_MAP": {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"},
    "DURATION_MAP": {5: "45 D", 15: "45 D", 30: "90 D", 60: "180 D", 1440: "365 D"},
}

# Setup logging and timezone
level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
logging.getLogger().setLevel(level_map.get(CONFIG["LOG_LEVEL"], logging.WARNING))
IST = pytz.timezone("Asia/Kolkata")

# Silence noisy loggers
for noisy in ("truedata", "truedata.history", "truedata_ws", "websocket", "urllib3", "requests", "connectionpool"):
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

# ======== CLEAN 5-MINUTE BOUNDARY FUNCTIONS (NO DEBUG LOGS) ========
def get_completed_5min_boundary(current_time_ist: datetime) -> datetime:
    """Get the most recent COMPLETED 5-minute boundary"""
    minute = (current_time_ist.minute // 5) * 5
    current_boundary = current_time_ist.replace(minute=minute, second=0, microsecond=0)

    if current_time_ist.minute % 5 == 0 and current_time_ist.second < 30:
        completed_boundary = current_boundary - timedelta(minutes=5)
    else:
        completed_boundary = current_boundary

    return completed_boundary

def wait_for_next_completed_5min_candle():
    """Wait for the next 5-minute candle to complete"""
    now_ist = datetime.now(IST)

    current_minute = now_ist.minute
    next_5min_mark = ((current_minute // 5) + 1) * 5

    if next_5min_mark >= 60:
        next_boundary = now_ist.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        next_boundary = now_ist.replace(minute=next_5min_mark, second=0, microsecond=0)

    next_analysis_time = next_boundary + timedelta(seconds=30)

    current_time = datetime.now(IST)
    if next_analysis_time > current_time:
        wait_seconds = int((next_analysis_time - current_time).total_seconds())
        console.print(f"⏳ Next analysis at [cyan]{next_analysis_time.strftime('%H:%M:%S')}[/cyan] IST ({wait_seconds}s)")

        while datetime.now(IST) < next_analysis_time:
            time.sleep(1)

def get_analysis_cutoff_time():
    """Get the cutoff time for data analysis (last completed 5-min boundary)"""
    now_ist = datetime.now(IST)
    completed_boundary = get_completed_5min_boundary(now_ist)
    return completed_boundary

# Helper functions
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

# Token bucket limiter
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

# TrueData session management
def authenticate_session():
    return TD_hist(CONFIG["TDUSERNAME"], CONFIG["TDPASSWORD"], log_level=logging.CRITICAL)

def build_sessions():
    sess_count = CONFIG["TD_HIST_SESSIONS"]
    pool, limiters = [], []

    successful_sessions = 0
    for i in range(sess_count):
        try:
            session = authenticate_session()
            pool.append(session)
            successful_sessions += 1
            console.print(f"✅ Session {i+1}/{sess_count} connected")
        except Exception as e:
            console.print(f"[red]Session {i+1} failed: {e}[/red]")

    if not pool:
        console.print("[red]❌ Failed to initialize ANY TrueData sessions.[/red]")
        raise SystemExit("Failed to initialize TrueData sessions.")

    if successful_sessions < sess_count:
        console.print(f"[yellow]⚠️  Only {successful_sessions}/{sess_count} sessions connected[/yellow]")

    per_sess_rate = CONFIG["RATE_PER_SECOND_TOTAL"] / len(pool)
    for _ in pool:
        limiters.append(TokenBucketLimiter(rate_per_sec=per_sess_rate, bucket_size=CONFIG["BUCKET_SIZE"]))

    return pool, limiters

# Initialize TrueData sessions
tdhist_pool, sess_limiters = build_sessions()
console.print("✅ [green]TrueData connection established[/green]")

# Enhanced indicator functions for option trading
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
        z_score = z_score.where(volume > vol_ma * 0.5, z_score * 0.3)
        return z_score.clip(-5, 5).fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def momentum_improved(df, period=10):
    if df is None or len(df) < period + 2:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')

    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')

        hl2 = (high + low) / 2
        shifted_close = close.shift(period).replace(0, np.nan)
        momentum_val = (hl2 / shifted_close) - 1.0

        return momentum_val.fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def oi_surge_improved(df, lookback=20):
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
        oi_trend = oi.rolling(3).mean() / oi.rolling(10).mean()
        trend_boost = np.where(oi_trend > 1.1, 1.3, 1.0)
        enhanced_score = z_score * trend_boost

        return enhanced_score.clip(-5, 5).fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def oi_momentum_improved(df, period=10):
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

        oi_mom_short = (oi / oi.shift(period//2)).replace([np.inf, -np.inf], 1.0) - 1.0
        oi_mom_long = (oi / oi.shift(period)).replace([np.inf, -np.inf], 1.0) - 1.0
        combined_mom = (oi_mom_short * 0.7) + (oi_mom_long * 0.3)

        return combined_mom.fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

# Signal classification functions
def classify_option_signal(normalized_score, oi_status, has_strong_conditions):
    if normalized_score >= 35:
        return "🚀 ULTRA STRONG BUY - AGGRESSIVE CALLS", "ULTRA_STRONG"
    elif normalized_score <= -35:
        return "💥 ULTRA STRONG SELL - AGGRESSIVE PUTS", "ULTRA_STRONG"
    elif normalized_score >= 20:
        if "Call Setup" in oi_status or "High OI Activity" in oi_status:
            return "🔥 VERY STRONG BUY - CALL FOCUS", "VERY_STRONG"
        else:
            return "🔥 VERY STRONG BUY", "VERY_STRONG"
    elif normalized_score <= -20:
        if "Put Setup" in oi_status or "High OI Activity" in oi_status:
            return "🔥 VERY STRONG SELL - PUT FOCUS", "VERY_STRONG"
        else:
            return "🔥 VERY STRONG SELL", "VERY_STRONG"
    elif normalized_score >= 12:
        if "High OI Activity" in oi_status:
            return "⚡ STRONG BUY - OI SURGE", "STRONG"
        else:
            return "⚡ STRONG BUY", "STRONG"
    elif normalized_score <= -12:
        if "High OI Activity" in oi_status:
            return "⚡ STRONG SELL - OI SURGE", "STRONG"
        else:
            return "⚡ STRONG SELL", "STRONG"
    elif normalized_score >= 8:
        return "🟢 BUY - Call Potential", "MODERATE"
    elif normalized_score <= -8:
        return "🔴 SELL - Put Potential", "MODERATE"
    else:
        return "⚪ NEUTRAL", "NEUTRAL"

def get_option_action(signal_strength, normalized_score, oi_status=""):
    if signal_strength == "ULTRA_STRONG":
        if normalized_score > 0:
            return "🚨 BUY CALLS AGGRESSIVELY - ATM/ITM", "URGENT"
        else:
            return "🚨 BUY PUTS AGGRESSIVELY - ATM/ITM", "URGENT"
    elif signal_strength == "VERY_STRONG":
        if normalized_score > 0:
            if "OI" in oi_status:
                return "🔥 BUY CALLS STRONG - ATM PREFERRED", "HIGH"
            else:
                return "🔥 BUY CALLS STRONG", "HIGH"
        else:
            if "OI" in oi_status:
                return "🔥 BUY PUTS STRONG - ATM PREFERRED", "HIGH"
            else:
                return "🔥 BUY PUTS STRONG", "HIGH"
    elif signal_strength == "STRONG":
        if normalized_score > 0:
            return "⚡ BUY CALLS - ATM/OTM", "MEDIUM"
        else:
            return "⚡ BUY PUTS - ATM/OTM", "MEDIUM"
    elif signal_strength == "MODERATE":
        if normalized_score > 0:
            return "📈 Consider Calls - OTM Safe", "LOW"
        else:
            return "📉 Consider Puts - OTM Safe", "LOW"
    else:
        return "⏸️ WAIT - No Clear Direction", "NONE"

# Validation function
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

# Data normalization function
def normalize_hist_df_clean(df, symbol, timeframe_minutes):
    if df is None or df.empty:
        return None

    try:
        out = df.copy()
        out.columns = out.columns.str.lower().str.strip()

        # Enhanced column mapping
        rename_map = {}
        for col in out.columns:
            col_clean = col.lower().strip()

            if any(x in col_clean for x in ['time', 'date', 'timestamp', 'datetime', 'ts']):
                rename_map[col] = 'Timestamp'
            elif col_clean in ['open'] or (col_clean.startswith('open') and 'interest' not in col_clean):
                rename_map[col] = 'Open'
            elif col_clean in ['high', 'h']:
                rename_map[col] = 'High'
            elif col_clean in ['low', 'l']:
                rename_map[col] = 'Low'
            elif col_clean in ['close', 'c']:
                rename_map[col] = 'Close'
            elif col_clean in ['volume', 'vol', 'v']:
                rename_map[col] = 'Volume'
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

        # Handle OI (silently)
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

# Enhanced analysis function
def analyze_signals_enhanced_clean(timeframe_dataframes, symbol):
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

        # Track OI symbols (silently)
        if 'OI' in df.columns and df['OI'].sum() > 100:
            global oi_symbols_found
            oi_symbols_found.add(symbol)

        scores = {}

        # OI Surge Analysis
        try:
            oi_z = oi_surge_improved(df)
            if len(oi_z) >= 2:
                oi_surge_current = oi_z.iloc[-1]
                price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1

                if oi_surge_current >= 2.0:
                    scores['OISurge'] = 3.0 if price_change > 0 else -3.0
                    oi_status = 'Extreme OI Activity'
                    has_strong_conditions = True
                elif oi_surge_current >= 1.5:
                    scores['OISurge'] = 2.5 if price_change > 0 else -2.5
                    oi_status = 'High OI Activity'
                elif oi_surge_current >= 1.0:
                    scores['OISurge'] = 1.5 if price_change > 0 else -1.5
                    oi_status = 'Moderate OI Activity'
                else:
                    scores['OISurge'] = 0.0
            else:
                scores['OISurge'] = 0.0
        except Exception:
            scores['OISurge'] = 0.0

        # OI Momentum
        try:
            oi_mom = oi_momentum_improved(df)
            if len(oi_mom) >= 2:
                oi_mom_current = oi_mom.iloc[-1]
                if oi_mom_current > 0.08:
                    scores['OIMomentum'] = 3.0
                elif oi_mom_current > 0.04:
                    scores['OIMomentum'] = 2.0
                elif oi_mom_current > 0.02:
                    scores['OIMomentum'] = 1.0
                elif oi_mom_current < -0.04:
                    scores['OIMomentum'] = -2.0
                elif oi_mom_current < -0.02:
                    scores['OIMomentum'] = -1.0
                else:
                    scores['OIMomentum'] = 0.0
            else:
                scores['OIMomentum'] = 0.0
        except Exception:
            scores['OIMomentum'] = 0.0

        # Volume Surge Analysis
        try:
            vol_surge = volume_surge_improved(df)
            if len(vol_surge) >= 2:
                current_surge = vol_surge.iloc[-1]
                price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1

                if current_surge >= 2.5:
                    scores['VolumeSurge'] = 3.0 if price_change > 0.008 else -3.0
                    has_strong_conditions = True
                elif current_surge >= 1.8:
                    scores['VolumeSurge'] = 2.5 if price_change > 0.005 else -2.5
                elif current_surge >= 1.2:
                    scores['VolumeSurge'] = 1.5 if price_change > 0.003 else -1.5
                else:
                    scores['VolumeSurge'] = 0.0
            else:
                scores['VolumeSurge'] = 0.0
        except Exception:
            scores['VolumeSurge'] = 0.0

        # Enhanced Momentum Analysis
        try:
            mom = momentum_improved(df)
            if len(mom) >= 2:
                current_mom = mom.iloc[-1]
                if current_mom > 0.02:
                    scores['Momentum'] = 3.0
                elif current_mom > 0.01:
                    scores['Momentum'] = 2.0
                elif current_mom > 0.005:
                    scores['Momentum'] = 1.0
                elif current_mom < -0.02:
                    scores['Momentum'] = -3.0
                elif current_mom < -0.01:
                    scores['Momentum'] = -2.0
                elif current_mom < -0.005:
                    scores['Momentum'] = -1.0
                else:
                    scores['Momentum'] = 0.0
            else:
                scores['Momentum'] = 0.0
        except Exception:
            scores['Momentum'] = 0.0

        # Call/Put Bias Analysis
        price_up = df['Close'].iloc[-1] > df['Close'].iloc[-2]
        vol_high = scores.get('VolumeSurge', 0) >= 1.5
        oi_active = abs(scores.get('OISurge', 0)) >= 1.5
        momentum_strong = abs(scores.get('Momentum', 0)) >= 1.0

        if price_up and vol_high and oi_active and momentum_strong:
            scores['CallBias'] = 4.0
            scores['PutBias'] = 0.0
            oi_status = 'Strong Call Setup'
            has_strong_conditions = True
        elif not price_up and vol_high and oi_active and momentum_strong:
            scores['PutBias'] = -4.0
            scores['CallBias'] = 0.0
            oi_status = 'Strong Put Setup'
            has_strong_conditions = True
        elif price_up and (vol_high or oi_active):
            scores['CallBias'] = 2.0
            scores['PutBias'] = 0.0
            if oi_active:
                oi_status = 'Call Setup'
        elif not price_up and (vol_high or oi_active):
            scores['PutBias'] = -2.0
            scores['CallBias'] = 0.0
            if oi_active:
                oi_status = 'Put Setup'
        else:
            scores['CallBias'] = 0.0
            scores['PutBias'] = 0.0

        # OI-Volume Confirmation
        if (vol_high or oi_active) and momentum_strong:
            if price_up:
                scores['OIVolConfirm'] = 2.0
            else:
                scores['OIVolConfirm'] = -2.0
        elif vol_high or oi_active:
            if price_up:
                scores['OIVolConfirm'] = 1.0
            else:
                scores['OIVolConfirm'] = -1.0
        else:
            scores['OIVolConfirm'] = 0.0

        # RSI Analysis
        try:
            rsi_series = calculate_rsi_improved(df)
            if len(rsi_series) >= 2:
                rsi_current = rsi_series.iloc[-1]
                rsi_prev = rsi_series.iloc[-2]

                if rsi_current > 70 and rsi_prev <= 70:
                    scores['RSI'] = 2.0
                elif rsi_current < 30 and rsi_prev >= 30:
                    scores['RSI'] = -2.0
                elif rsi_current > 60:
                    scores['RSI'] = 1.0
                elif rsi_current < 40:
                    scores['RSI'] = -1.0
                else:
                    scores['RSI'] = 0.0
            else:
                scores['RSI'] = 0.0
        except Exception:
            scores['RSI'] = 0.0

        # Fill remaining indicators with simple calculations
        remaining_indicators = ['ADX', 'VWAP', 'MACD', 'EMA', 'CMF', 'ADL', 'OBV', 'ATR', 
                               'Bollinger', 'ROC', 'Stochastic', 'CCI', 'MA', 'WWL', 'RelVol', 
                               'VWAPRegime', 'OBVConfirm', 'PriceVelocity', 'GammaSqueezeRisk', 
                               'IVCrushRisk', 'OptionFlow']

        for indicator in remaining_indicators:
            if indicator not in scores:
                try:
                    price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-5]) - 1
                    if abs(price_change) > 0.02:
                        scores[indicator] = 1.0 * np.sign(price_change)
                    elif abs(price_change) > 0.01:
                        scores[indicator] = 0.5 * np.sign(price_change)
                    else:
                        scores[indicator] = 0.0
                except Exception:
                    scores[indicator] = 0.0

        # Calculate weighted scores for this timeframe
        for indicator, score in scores.items():
            ind_weight = CONFIG["INDICATOR_WEIGHTS"].get(indicator, 1.0)
            weighted_score = score * tf_weight * ind_weight
            final_score += weighted_score
            max_possible += 4.0 * tf_weight * ind_weight

    if valid_timeframes < 1 or max_possible == 0:
        return 'Neutral', 0.0, oi_status, 'WAIT', 'NONE'

    # Enhanced normalization for option trading
    normalized = (final_score / max_possible) * 100.0

    # Cap the normalized score
    if abs(normalized) > 100:
        normalized = np.sign(normalized) * 100

    signal_text, signal_strength = classify_option_signal(normalized, oi_status, has_strong_conditions)
    option_action, alert_priority = get_option_action(signal_strength, normalized, oi_status)

    return signal_text, normalized, oi_status, option_action, alert_priority

# Data fetching function
@retry(
    stop_max_attempt_number=CONFIG["RETRY_ATTEMPTS"],
    wait_exponential_multiplier=max(1, int(CONFIG["RETRY_DELAY_MS"] / 2)),
    wait_exponential_max=10000,
    retry_on_exception=lambda e: True
)
def fetch_one_clean(symbol_orig, timeframe_minutes, limiter, hist):
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

        global api_calls_done
        with api_calls_lock:
            api_calls_done += 1

        return symbol_orig, timeframe_minutes, df

    except Exception:
        return symbol_orig, timeframe_minutes, None

def prefetch_clean(stocks, max_workers=CONFIG["MAX_WORKERS"]):
    tfs = [5, 15, 30, 60, 1440]
    total_calls = len(stocks) * len(tfs)
    stock_multi_data = defaultdict(dict)

    global api_calls_done, oi_symbols_found
    with api_calls_lock:
        api_calls_done = 0

    oi_symbols_found = set()
    valid_stocks = [s for s in stocks if s]

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

    console.print(f"✅ Data loaded: [green]{len(valid_data)} symbols[/green] ready")
    if len(oi_symbols_found) > 0:
        console.print(f"📈 OI data: [yellow]{len(oi_symbols_found)} symbols[/yellow]")

    return valid_data

# ======== CLEAN FILTER WITH NO DEBUG LOGS ========
def filter_timeframe_data_to_boundary(symbol, timeframe_data, cutoff_time_aware):
    """Filter timeframe data to only include COMPLETED candles (NO DEBUG LOGS)"""
    filtered_timeframes = {}

    for tf, df in timeframe_data.items():
        if df is None or df.empty:
            continue

        try:
            # Ensure cutoff time has timezone
            if cutoff_time_aware.tzinfo is None:
                cutoff_time_aware = IST.localize(cutoff_time_aware)
            elif cutoff_time_aware.tzinfo != IST:
                cutoff_time_aware = cutoff_time_aware.astimezone(IST)

            # Ensure dataframe index has timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize(IST, ambiguous='infer', nonexistent='shift_forward')
            else:
                df.index = df.index.tz_convert(IST)

            # Clean invalid timestamps
            valid_index = df.index.dropna()
            if len(valid_index) != len(df.index):
                df = df.loc[valid_index]

            if not df.empty:
                # Only include candles that are COMPLETELY BEFORE the cutoff
                mask = df.index <= cutoff_time_aware
                df_filtered = df.loc[mask]

                if len(df_filtered) >= CONFIG["MIN_BARS_REQUIRED"]:
                    filtered_timeframes[tf] = df_filtered

        except Exception:
            continue

    return filtered_timeframes

def infer_institutional_flow(tf_data):
    return "Mixed"

def render_signals_clean(now_ts, top_bullish, top_bearish):
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

def print_clean_banner():
    console.print("\n" + "="*70, style="bold blue")
    console.print("🎯 [bold cyan]OPTION SIGNAL SCANNER v3.0 - CLEAN PRODUCTION[/bold cyan] 🎯", justify="center")
    console.print("="*70, style="bold blue")
    console.print(f"🕐 [bold white]{datetime.now(IST).strftime('%H:%M:%S IST')}[/bold white] | Mode: [green]5-MIN BOUNDARY ANALYSIS[/green]")
    console.print("="*70, style="bold blue")

def load_stock_list(file_name):
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

# ======== CLEAN BACKTEST (NO DEBUG LOGS) ========
def run_backtest_clean(day_str: str, stocks):
    """Clean backtest with 5-minute boundary logic (no debug logs)"""
    day_date = datetime.strptime(day_str, "%Y-%m-%d")

    console.print(f"📊 [bold cyan]Backtesting {day_str}[/bold cyan] - [yellow]{len(stocks)} symbols[/yellow]")

    # Fetch data for all stocks
    stock_multi_data = prefetch_clean(stocks)

    if len(stock_multi_data) == 0:
        console.print("[red]❌ No valid data found[/red]")
        return

    # Get time checkpoints for the day (every 5 minutes)
    checkpoints = day_checkpoints_ist(day_date)

    # Create output filename
    output_filename = day_date.strftime("%Y-%m-%d") + "_signals_clean.csv"

    # Clean old file
    try:
        if os.path.exists(output_filename):
            os.remove(output_filename)
    except Exception:
        pass

    # Initialize globals
    global previous_scores, last_bull_symbols, last_bear_symbols, performance_metrics
    previous_scores = {}
    last_bull_symbols = set()
    last_bear_symbols = set()
    performance_metrics = defaultdict(int)

    console.print(f"🔍 Analyzing [cyan]{len(checkpoints)}[/cyan] 5-minute boundaries...")

    # Process each checkpoint (representing completed 5-minute candles)
    for i, as_of_ts in enumerate(checkpoints):
        if (i + 1) % 20 == 0:
            console.print(f"📈 Progress: [cyan]{i+1}/{len(checkpoints)}[/cyan] | Boundary: [yellow]{as_of_ts.strftime('%H:%M')}[/yellow]")

        # Use the checkpoint as the cutoff time (this represents a completed 5-minute candle)
        analysis_cutoff = as_of_ts.replace(second=0, microsecond=0)

        signals_this_scan = []
        current_scores = {}

        # Analyze each symbol
        for symbol, timeframe_data in stock_multi_data.items():
            clean_symbol = symbol.replace('-EQ', '')

            # Filter timeframe data to only include completed candles up to cutoff
            filtered_timeframes = filter_timeframe_data_to_boundary(clean_symbol, timeframe_data, analysis_cutoff)

            if len(filtered_timeframes) < 1:
                continue

            # Analyze signals
            signal, score, oi_status, option_action, alert_priority = analyze_signals_enhanced_clean(filtered_timeframes, clean_symbol)

            current_scores[clean_symbol] = score

            # Only include significant signals
            if abs(score) >= CONFIG['MIN_SIGNAL_THRESHOLD'] or any(x in signal for x in ['STRONG', 'BUY', 'SELL']):
                prev = previous_scores.get(clean_symbol, 'NA')
                change_val = 'NA' if isinstance(prev, str) else (score - prev)
                direction = 'bullish' if score > 0 else 'bearish'
                flow_tag = infer_institutional_flow(filtered_timeframes)

                signals_this_scan.append({
                    'symbol': clean_symbol,
                    'signal': signal,
                    'score': score,
                    'trend': direction,
                    'change': change_val,
                    'oi_status': oi_status,
                    'flow': flow_tag,
                    'action': option_action
                })

                performance_metrics[f"{direction}_signals"] += 1

        # Update previous scores
        previous_scores = current_scores.copy()

        # Sort and filter signals
        signals_this_scan.sort(key=lambda x: abs(x['score']), reverse=True)
        top_bullish = [r for r in signals_this_scan if r['score'] > 0][:20]
        top_bearish = [r for r in signals_this_scan if r['score'] < 0][:20]

        # Display and save results
        if top_bullish or top_bearish:
            render_signals_clean(as_of_ts, top_bullish, top_bearish)
            export_to_csv(as_of_ts, top_bullish, top_bearish, output_filename)

    # Final summary
    total_signals = sum(performance_metrics.values())
    console.print(f"\n📈 [bold green]BACKTEST COMPLETE[/bold green]")
    console.print(f"Total Signals: [cyan]{total_signals}[/cyan]")
    if 'bullish_signals' in performance_metrics:
        console.print(f"Bullish: [green]{performance_metrics['bullish_signals']}[/green]")
    if 'bearish_signals' in performance_metrics:
        console.print(f"Bearish: [red]{performance_metrics['bearish_signals']}[/red]")
    console.print(f"Results: [yellow]{output_filename}[/yellow]")

# ======== CLEAN LIVE MODE (NO DEBUG LOGS) ========
def run_live_mode_clean(stocks):
    """Clean live mode with 5-minute boundary logic (no debug logs)"""
    console.print("🚀 [bold red]LIVE MODE - CLEAN 5-MINUTE BOUNDARIES[/bold red]")
    console.print(f"📊 Monitoring [cyan]{len(stocks)}[/cyan] symbols")

    # Check if market is open
    now = datetime.now(IST)
    market_start = today_ist_dt(CONFIG["MARKET_START"])
    market_end = today_ist_dt(CONFIG["MARKET_END"])

    if not (market_start <= now <= market_end):
        console.print(f"⏰ Market closed. Next open: {CONFIG['MARKET_START']} IST")
        sleep_until(market_start)
        console.print("🔔 [bold green]Market OPEN![/bold green]")

    global previous_scores, last_bull_symbols, last_bear_symbols, performance_metrics
    previous_scores = {}
    last_bull_symbols = set()
    last_bear_symbols = set()
    performance_metrics = defaultdict(int)

    live_signals_file = f"live_signals_clean_{datetime.now(IST).strftime('%Y%m%d')}.csv"
    try:
        if os.path.exists(live_signals_file):
            os.remove(live_signals_file)
    except Exception:
        pass

    console.print(f"💾 Clean signals: [yellow]{live_signals_file}[/yellow]")

    scan_count = 0

    try:
        while (market_start <= datetime.now(IST) <= market_end):
            scan_count += 1

            # Wait for the next 5-minute candle to complete
            wait_for_next_completed_5min_candle()

            # Get the analysis cutoff time (last completed 5-minute boundary)
            analysis_cutoff = get_analysis_cutoff_time()

            console.print(f"\n🔄 [bold yellow]SCAN #{scan_count}[/bold yellow] | Boundary: [cyan]{analysis_cutoff.strftime('%H:%M')}[/cyan] IST")

            try:
                # Fetch fresh data
                stock_multi_data = prefetch_clean(stocks)
                if not stock_multi_data:
                    console.print("[red]⚠️ No data received, waiting...[/red]")
                    continue
            except Exception as e:
                console.print(f"[red]❌ Data error: {e}[/red]")
                continue

            signals_this_scan = []
            current_scores = {}

            # Analyze each symbol with boundary cutoff
            for symbol, timeframe_data in stock_multi_data.items():
                clean_symbol = symbol.replace('-EQ', '')

                # Filter to only include completed candles
                filtered_timeframes = filter_timeframe_data_to_boundary(clean_symbol, timeframe_data, analysis_cutoff)

                if len(filtered_timeframes) < 1:
                    continue

                signal, score, oi_status, option_action, alert_priority = analyze_signals_enhanced_clean(filtered_timeframes, clean_symbol)
                current_scores[clean_symbol] = score

                if abs(score) >= CONFIG['MIN_SIGNAL_THRESHOLD'] or any(x in signal for x in ['STRONG', 'BUY', 'SELL']):
                    prev = previous_scores.get(clean_symbol, 'NA')
                    change_val = 'NA' if isinstance(prev, str) else (score - prev)
                    direction = 'bullish' if score > 0 else 'bearish'

                    signals_this_scan.append({
                        'symbol': clean_symbol, 'signal': signal, 'score': score,
                        'trend': direction, 'change': change_val, 'oi_status': oi_status,
                        'action': option_action, 'alert_priority': alert_priority
                    })
                    performance_metrics[f"{direction}_signals"] += 1

            previous_scores = current_scores.copy()
            signals_this_scan.sort(key=lambda x: abs(x['score']), reverse=True)
            top_bullish = [r for r in signals_this_scan if r['score'] > 0][:20]
            top_bearish = [r for r in signals_this_scan if r['score'] < 0][:20]

            if top_bullish or top_bearish:
                render_signals_clean(analysis_cutoff, top_bullish, top_bearish)
                export_to_csv(analysis_cutoff, top_bullish, top_bearish, live_signals_file)

                # Alert for urgent signals (no debug)
                urgent_signals = [r for r in signals_this_scan if r.get('alert_priority') == 'URGENT']
                if urgent_signals:
                    console.print("\n🚨 [bold red]URGENT SIGNALS![/bold red]")
                    for signal in urgent_signals:
                        console.print(f"🚨 {signal['symbol']}: {signal['action']}")
            else:
                console.print("⚪ No significant signals")

            # Performance summary every 10 scans
            if scan_count % 10 == 0:
                total_signals = sum(performance_metrics.values())
                console.print(f"\n📊 [bold blue]Session Summary (#{scan_count})[/bold blue]")
                console.print(f"Signals: [cyan]{total_signals}[/cyan] | Boundaries: [yellow]{scan_count}[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[yellow]👤 Live mode stopped[/yellow]")
    finally:
        total_signals = sum(performance_metrics.values())
        console.print(f"\n📈 [bold green]SESSION COMPLETE[/bold green]")
        console.print(f"Boundaries: [cyan]{scan_count}[/cyan] | Signals: [cyan]{total_signals}[/cyan]")
        if 'bullish_signals' in performance_metrics:
            console.print(f"Bullish: [green]{performance_metrics['bullish_signals']}[/green]")
        if 'bearish_signals' in performance_metrics:
            console.print(f"Bearish: [red]{performance_metrics['bearish_signals']}[/red]")
        console.print(f"Results: [yellow]{live_signals_file}[/yellow]")

def main():
    print_clean_banner()

    try:
        parser = argparse.ArgumentParser(description="Option Scanner - Clean Production")
        parser.add_argument("--backtest-date", help="Backtest date (YYYY-MM-DD)")
        parser.add_argument("--stocks-file", default=CONFIG["SHARES_FILE"], help="Stock symbols file")
        parser.add_argument("--live", action="store_true", help="Live market mode")
        parser.add_argument("--test-oi", action="store_true", help="Test OI data")

        args = parser.parse_args()

        stocks = load_stock_list(args.stocks_file)
        if not stocks:
            console.print("[red]❌ No valid stocks loaded[/red]")
            return

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
            console.print("\n🎯 [bold green]Option Scanner - Clean Production[/bold green]")
            console.print("[cyan]Features:[/cyan]")
            console.print("  ✅ 5-minute boundary analysis")
            console.print("  ✅ Enhanced option buyer logic")
            console.print("  ✅ Clean output (no debug logs)")
            console.print("  ✅ OI detection and institutional flow")
            console.print("\n[cyan]Usage:[/cyan]")
            console.print("  [yellow]python scanner.py --backtest-date 2025-09-30[/yellow]")
            console.print("  [yellow]python scanner.py --live[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[yellow]👤 Scanner stopped[/yellow]")
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
                console.print(f"\n📊 [bold green]Final: {total} signals[/bold green]")

        console.print("✅ [green]Shutdown complete[/green]")

if __name__ == "__main__":
    main()
