
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

    # Always include all timeframes
    "SKIP_DAILY": False,

    # Analysis settings - optimized thresholds for option trading
    "MIN_BARS_REQUIRED": 20,
    "MAX_MISSING_DATA_PCT": 15,
    "SIGNAL_CONFIRMATION_BARS": 2,
    "MIN_SIGNAL_THRESHOLD": 8,  # Lower threshold for option signals

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
    # Based on 10 years of option trading experience, these weights prioritize:
    # 1. OI & Volume (most important for option flow)
    # 2. Momentum & Price velocity (for quick moves)
    # 3. Volatility indicators (for option premium)
    # 4. Traditional indicators (supporting confirmation)
    "INDICATOR_WEIGHTS": {
        # PRIMARY: OI & Volume Analysis (Highest weights)
        "OISurge": 5.0,           # Most important - shows big money moves
        "OIMomentum": 4.5,        # OI direction change
        "VolumeSurge": 4.2,       # Volume spikes indicate institutional interest
        "OIVolConfirm": 4.0,      # Combined OI+Volume confirmation
        "CallBias": 4.8,          # Call flow analysis
        "PutBias": 4.8,           # Put flow analysis
        "OptionFlow": 4.5,        # Overall option flow direction

        # SECONDARY: Price & Momentum (Critical for option timing)
        "Momentum": 3.8,          # Price momentum for direction
        "PriceVelocity": 3.5,     # Speed of price movement
        "ADX": 3.2,              # Trend strength
        "GammaSqueezeRisk": 3.0,  # Potential for explosive moves
        "VWAP": 2.8,             # Institution price levels

        # TERTIARY: Technical Confirmation (Supporting indicators)
        "EMA": 2.5,              # Trend direction
        "MACD": 2.3,             # Momentum confirmation
        "ATR": 2.2,              # Volatility for option selection
        "RSI": 2.0,              # Overbought/oversold
        "Bollinger": 1.8,        # Volatility bands
        "Stochastic": 1.5,       # Momentum oscillator

        # QUATERNARY: Volume & Flow Analysis
        "CMF": 2.8,              # Money flow
        "OBV": 2.5,              # Volume trend
        "RelVol": 2.3,           # Relative volume
        "ADL": 2.2,              # Accumulation/Distribution
        "VWAPRegime": 2.0,       # VWAP trend
        "OBVConfirm": 1.8,       # Volume confirmation

        # QUINARY: Support indicators
        "ROC": 1.8,              # Rate of change
        "CCI": 1.5,              # Commodity Channel Index
        "MA": 1.5,               # Moving averages
        "WWL": 1.3,              # Williams %R
        "IVCrushRisk": 2.5,      # IV crush probability
    },

    # Enhanced timeframe weights for option trading
    # 5 & 15 min for entries, 30-60 min for trend, daily for overall direction
    "TIMEFRAME_WEIGHTS": {
        5: 3.0,      # Highest - precise entry timing
        15: 2.8,     # High - confirmation and momentum
        30: 2.2,     # Medium-High - short-term trend
        60: 1.8,     # Medium - intermediate trend  
        1440: 1.5    # Lower - overall direction only
    },

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

# Initialize TrueData sessions
tdhist_pool, sess_limiters = build_sessions()
console.print("✅ [green]TrueData connection established[/green]")

# ======== ENHANCED INDICATOR FUNCTIONS FOR OPTION TRADING ========
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
    """Enhanced volume surge for option flow detection"""
    if df is None or len(df) < lookback + 5:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')

    try:
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        vol_ma = volume.rolling(lookback, min_periods=lookback//2).mean()
        vol_std = volume.rolling(lookback, min_periods=lookback//2).std()
        vol_std = vol_std.where(vol_std > vol_ma * 0.01, vol_ma * 0.1)
        z_score = (volume - vol_ma) / vol_std

        # Enhanced for option trading - penalize low volume
        z_score = z_score.where(volume > vol_ma * 0.5, z_score * 0.3)

        return z_score.clip(-5, 5).fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def momentum_improved(df, period=10):
    """Enhanced momentum for option timing"""
    if df is None or len(df) < period + 2:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')

    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')

        # Use HL2 for better momentum calculation
        hl2 = (high + low) / 2
        shifted_close = close.shift(period).replace(0, np.nan)
        momentum_val = (hl2 / shifted_close) - 1.0

        return momentum_val.fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def price_velocity(df, period=5):
    """Price velocity for option entry timing"""
    if df is None or len(df) < period + 2:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')

    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')

        # Calculate rate of change over multiple periods
        roc_1 = close.pct_change(1)
        roc_3 = close.pct_change(3)
        roc_5 = close.pct_change(5)

        # Weighted velocity (recent periods have more weight)
        velocity = (roc_1 * 0.5) + (roc_3 * 0.3) + (roc_5 * 0.2)

        return velocity.fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def gamma_squeeze_risk(df):
    """Detect potential gamma squeeze conditions"""
    if df is None or len(df) < 10:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')

    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')

        # Calculate conditions for gamma squeeze
        price_range = (high - low) / close
        vol_surge = volume / volume.rolling(20, min_periods=5).mean()

        # Rapid price movement with high volume
        rapid_move = abs(close.pct_change(3)) > 0.02  # 2% move in 3 periods
        high_vol = vol_surge > 1.5
        tight_range_before = price_range.rolling(5).mean() < 0.015  # Low volatility before

        gamma_risk = (rapid_move & high_vol & tight_range_before).astype(float)

        return gamma_risk.fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def iv_crush_risk(df, period=10):
    """Estimate IV crush risk based on price action"""
    if df is None or len(df) < period + 5:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')

    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')

        # Calculate implied volatility proxy using price ranges
        true_range = np.maximum(high - low, 
                               np.maximum(abs(high - close.shift(1)), 
                                        abs(low - close.shift(1))))

        atr = true_range.rolling(period, min_periods=period//2).mean()
        current_range = high - low

        # High ATR followed by low range suggests IV crush risk
        high_iv_period = atr > atr.rolling(20, min_periods=10).quantile(0.8)
        current_low_range = current_range < atr * 0.5

        iv_crush_risk_val = (high_iv_period.shift(1) & current_low_range).astype(float)

        return iv_crush_risk_val.fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def option_flow_direction(df):
    """Determine overall option flow direction"""
    if df is None or len(df) < 10:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')

    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)

        # Price-volume relationship for option flow
        price_change = close.pct_change()
        vol_weighted_price = price_change * (volume / volume.rolling(20, min_periods=5).mean())

        # Smooth the flow direction
        flow_direction = vol_weighted_price.rolling(5, min_periods=3).mean()

        return flow_direction.fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def oi_surge_improved(df, lookback=20):
    """Enhanced OI surge calculation for option trading"""
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

        # Enhanced OI analysis
        oi_ma = oi.rolling(lookback, min_periods=lookback//2).mean()
        oi_std = oi.rolling(lookback, min_periods=lookback//2).std()
        oi_std = oi_std.where(oi_std > oi_ma * 0.01, oi_ma * 0.1)

        # Z-score with additional filtering
        z_score = (oi - oi_ma) / oi_std

        # Boost score for consistent OI increases
        oi_trend = oi.rolling(3).mean() / oi.rolling(10).mean()
        trend_boost = np.where(oi_trend > 1.1, 1.3, 1.0)

        enhanced_score = z_score * trend_boost

        return enhanced_score.clip(-5, 5).fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def oi_momentum_improved(df, period=10):
    """Enhanced OI momentum calculation"""
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

        # Multi-period OI momentum
        oi_mom_short = (oi / oi.shift(period//2)).replace([np.inf, -np.inf], 1.0) - 1.0
        oi_mom_long = (oi / oi.shift(period)).replace([np.inf, -np.inf], 1.0) - 1.0

        # Combined momentum with more weight on recent
        combined_mom = (oi_mom_short * 0.7) + (oi_mom_long * 0.3)

        return combined_mom.fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

# ======== ENHANCED SIGNAL CLASSIFICATION FOR OPTION TRADING ========
def classify_option_signal(normalized_score, oi_status, has_strong_conditions):
    """Enhanced signal classification optimized for option buying"""

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
    """Enhanced option trading action for experienced buyers"""
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

# ======== ENHANCED ANALYSIS WITH EXPERIENCED OPTION BUYER LOGIC ========
def analyze_signals_enhanced_clean(timeframe_dataframes, symbol):
    """Enhanced analysis optimized for experienced option buyers"""
    if not timeframe_dataframes:
        return 'Neutral', 0.0, 'Normal', 'WAIT', 'NONE'

    final_score, max_possible = 0.0, 0.0
    valid_timeframes = 0
    oi_status = 'Normal'
    has_strong_conditions = False

    # Track option-specific conditions
    option_conditions = {
        'high_oi_activity': False,
        'volume_surge': False,
        'momentum_strong': False,
        'gamma_risk': False,
        'iv_crush_risk': False
    }

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

        # ======== PRIMARY INDICATORS (Highest Weight) ========

        # 1. OI Surge Analysis (Most Important)
        try:
            oi_z = oi_surge_improved(df)
            if len(oi_z) >= 2:
                oi_surge_current = oi_z.iloc[-1]
                price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1

                if oi_surge_current >= 2.0:
                    scores['OISurge'] = 3.0 if price_change > 0 else -3.0
                    oi_status = 'Extreme OI Activity'
                    option_conditions['high_oi_activity'] = True
                    has_strong_conditions = True
                elif oi_surge_current >= 1.5:
                    scores['OISurge'] = 2.5 if price_change > 0 else -2.5
                    oi_status = 'High OI Activity'
                    option_conditions['high_oi_activity'] = True
                elif oi_surge_current >= 1.0:
                    scores['OISurge'] = 1.5 if price_change > 0 else -1.5
                    oi_status = 'Moderate OI Activity'
                else:
                    scores['OISurge'] = 0.0
            else:
                scores['OISurge'] = 0.0
        except Exception:
            scores['OISurge'] = 0.0

        # 2. OI Momentum (Critical for Direction)
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

        # 3. Volume Surge Analysis (Option Flow Detection)
        try:
            vol_surge = volume_surge_improved(df)
            if len(vol_surge) >= 2:
                current_surge = vol_surge.iloc[-1]
                price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1

                if current_surge >= 2.5:
                    scores['VolumeSurge'] = 3.0 if price_change > 0.008 else -3.0
                    option_conditions['volume_surge'] = True
                    has_strong_conditions = True
                elif current_surge >= 1.8:
                    scores['VolumeSurge'] = 2.5 if price_change > 0.005 else -2.5
                    option_conditions['volume_surge'] = True
                elif current_surge >= 1.2:
                    scores['VolumeSurge'] = 1.5 if price_change > 0.003 else -1.5
                else:
                    scores['VolumeSurge'] = 0.0
            else:
                scores['VolumeSurge'] = 0.0
        except Exception:
            scores['VolumeSurge'] = 0.0

        # 4. Enhanced Momentum Analysis
        try:
            mom = momentum_improved(df)
            if len(mom) >= 2:
                current_mom = mom.iloc[-1]
                if current_mom > 0.02:
                    scores['Momentum'] = 3.0
                    option_conditions['momentum_strong'] = True
                elif current_mom > 0.01:
                    scores['Momentum'] = 2.0
                    option_conditions['momentum_strong'] = True
                elif current_mom > 0.005:
                    scores['Momentum'] = 1.0
                elif current_mom < -0.02:
                    scores['Momentum'] = -3.0
                    option_conditions['momentum_strong'] = True
                elif current_mom < -0.01:
                    scores['Momentum'] = -2.0
                    option_conditions['momentum_strong'] = True
                elif current_mom < -0.005:
                    scores['Momentum'] = -1.0
                else:
                    scores['Momentum'] = 0.0
            else:
                scores['Momentum'] = 0.0
        except Exception:
            scores['Momentum'] = 0.0

        # 5. Price Velocity (Entry Timing)
        try:
            velocity = price_velocity(df)
            if len(velocity) >= 2:
                current_velocity = velocity.iloc[-1]
                if abs(current_velocity) > 0.015:
                    scores['PriceVelocity'] = 2.5 * np.sign(current_velocity)
                elif abs(current_velocity) > 0.008:
                    scores['PriceVelocity'] = 1.5 * np.sign(current_velocity)
                elif abs(current_velocity) > 0.003:
                    scores['PriceVelocity'] = 1.0 * np.sign(current_velocity)
                else:
                    scores['PriceVelocity'] = 0.0
            else:
                scores['PriceVelocity'] = 0.0
        except Exception:
            scores['PriceVelocity'] = 0.0

        # 6. Gamma Squeeze Risk Detection
        try:
            gamma_risk = gamma_squeeze_risk(df)
            if len(gamma_risk) >= 1:
                if gamma_risk.iloc[-1] > 0:
                    scores['GammaSqueezeRisk'] = 2.0
                    option_conditions['gamma_risk'] = True
                else:
                    scores['GammaSqueezeRisk'] = 0.0
            else:
                scores['GammaSqueezeRisk'] = 0.0
        except Exception:
            scores['GammaSqueezeRisk'] = 0.0

        # 7. IV Crush Risk Assessment
        try:
            iv_risk = iv_crush_risk(df)
            if len(iv_risk) >= 1:
                if iv_risk.iloc[-1] > 0:
                    scores['IVCrushRisk'] = -1.5  # Negative because it's a risk
                    option_conditions['iv_crush_risk'] = True
                else:
                    scores['IVCrushRisk'] = 0.0
            else:
                scores['IVCrushRisk'] = 0.0
        except Exception:
            scores['IVCrushRisk'] = 0.0

        # 8. Option Flow Direction
        try:
            flow_dir = option_flow_direction(df)
            if len(flow_dir) >= 2:
                current_flow = flow_dir.iloc[-1]
                if current_flow > 0.01:
                    scores['OptionFlow'] = 2.0
                elif current_flow > 0.005:
                    scores['OptionFlow'] = 1.0
                elif current_flow < -0.01:
                    scores['OptionFlow'] = -2.0
                elif current_flow < -0.005:
                    scores['OptionFlow'] = -1.0
                else:
                    scores['OptionFlow'] = 0.0
            else:
                scores['OptionFlow'] = 0.0
        except Exception:
            scores['OptionFlow'] = 0.0

        # ======== CALL/PUT BIAS ANALYSIS (Enhanced) ========
        price_up = df['Close'].iloc[-1] > df['Close'].iloc[-2]
        vol_high = scores.get('VolumeSurge', 0) >= 1.5
        oi_active = abs(scores.get('OISurge', 0)) >= 1.5
        momentum_strong = abs(scores.get('Momentum', 0)) >= 1.0

        # Strong Call Bias Conditions
        if price_up and vol_high and oi_active and momentum_strong:
            scores['CallBias'] = 4.0
            scores['PutBias'] = 0.0
            oi_status = 'Strong Call Setup'
            has_strong_conditions = True
        # Strong Put Bias Conditions
        elif not price_up and vol_high and oi_active and momentum_strong:
            scores['PutBias'] = -4.0
            scores['CallBias'] = 0.0
            oi_status = 'Strong Put Setup'
            has_strong_conditions = True
        # Moderate Call Bias
        elif price_up and (vol_high or oi_active):
            scores['CallBias'] = 2.0
            scores['PutBias'] = 0.0
            if oi_active:
                oi_status = 'Call Setup'
        # Moderate Put Bias
        elif not price_up and (vol_high or oi_active):
            scores['PutBias'] = -2.0
            scores['CallBias'] = 0.0
            if oi_active:
                oi_status = 'Put Setup'
        else:
            scores['CallBias'] = 0.0
            scores['PutBias'] = 0.0

        # OI-Volume Confirmation (Enhanced)
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

        # ======== SECONDARY INDICATORS ========

        # RSI Analysis (Overbought/Oversold for Options)
        try:
            rsi_series = calculate_rsi_improved(df)
            if len(rsi_series) >= 2:
                rsi_current = rsi_series.iloc[-1]
                rsi_prev = rsi_series.iloc[-2]

                # Enhanced RSI for options - look for divergences and extremes
                if rsi_current > 70 and rsi_prev <= 70:
                    scores['RSI'] = 2.0  # Fresh overbought - calls risky
                elif rsi_current < 30 and rsi_prev >= 30:
                    scores['RSI'] = -2.0  # Fresh oversold - puts risky
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

        # ADX for Trend Strength
        try:
            close = df['Close']
            high = df['High']
            low = df['Low']

            # Simplified ADX calculation
            tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
            atr = tr.rolling(14).mean()

            plus_dm = np.where((high - high.shift(1)) > (low.shift(1) - low), np.maximum(high - high.shift(1), 0), 0)
            minus_dm = np.where((low.shift(1) - low) > (high - high.shift(1)), np.maximum(low.shift(1) - low, 0), 0)

            plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr)

            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean()

            if len(adx) >= 2 and not pd.isna(adx.iloc[-1]):
                adx_current = adx.iloc[-1]
                if adx_current > 25:
                    # Strong trend - good for options
                    trend_direction = 1 if plus_di.iloc[-1] > minus_di.iloc[-1] else -1
                    scores['ADX'] = 2.0 * trend_direction if adx_current > 40 else 1.5 * trend_direction
                else:
                    scores['ADX'] = 0.0
            else:
                scores['ADX'] = 0.0
        except Exception:
            scores['ADX'] = 0.0

        # VWAP Analysis
        try:
            close = df['Close']
            volume = df['Volume']
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            vwap = (typical_price * volume).cumsum() / volume.cumsum()

            if len(vwap) >= 2:
                if close.iloc[-1] > vwap.iloc[-1] * 1.005:
                    scores['VWAP'] = 1.5
                elif close.iloc[-1] < vwap.iloc[-1] * 0.995:
                    scores['VWAP'] = -1.5
                else:
                    scores['VWAP'] = 0.0
            else:
                scores['VWAP'] = 0.0
        except Exception:
            scores['VWAP'] = 0.0

        # Fill remaining indicators with simplified calculations
        remaining_indicators = ['MACD', 'EMA', 'CMF', 'ADL', 'OBV', 'ATR', 
                               'Bollinger', 'ROC', 'Stochastic', 'CCI', 'MA', 'WWL', 
                               'RelVol', 'VWAPRegime', 'OBVConfirm']

        for indicator in remaining_indicators:
            if indicator not in scores:
                # Simple momentum-based scoring for remaining indicators
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
            max_possible += 4.0 * tf_weight * ind_weight  # Increased max for enhanced scoring

    if valid_timeframes < 1 or max_possible == 0:
        return 'Neutral', 0.0, oi_status, 'WAIT', 'NONE'

    # Enhanced normalization for option trading
    normalized = (final_score / max_possible) * 100.0

    # Apply option-specific adjustments
    if option_conditions['high_oi_activity'] and option_conditions['volume_surge']:
        normalized *= 1.2  # Boost for strong option activity

    if option_conditions['gamma_risk']:
        normalized *= 1.15  # Boost for gamma squeeze potential

    if option_conditions['iv_crush_risk']:
        normalized *= 0.8  # Reduce for IV crush risk

    # Cap the normalized score
    if abs(normalized) > 100:
        normalized = np.sign(normalized) * 100

    signal_text, signal_strength = classify_option_signal(normalized, oi_status, has_strong_conditions)
    option_action, alert_priority = get_option_action(signal_strength, normalized, oi_status)

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
        console.print("🔥 [bold white on red]STRONG SIGNALS[/bold white on red]")

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
        console.print("📊 [bold blue]MODERATE SIGNALS[/bold blue]")

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
    console.print(f"[bold yellow]{summary}[/bold yellow]")
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
    """Clean production banner"""
    console.print("" + "="*70, style="bold blue")
    console.print("🎯 [bold cyan]OPTION SIGNAL SCANNER v3.0 - ENHANCED OPTION BUYER LOGIC[/bold cyan] 🎯", justify="center")
    console.print("="*70, style="bold blue")
    console.print(f"🕐 [bold white]{datetime.now(IST).strftime('%H:%M:%S IST')}[/bold white] | Mode: [green]LIVE/PRODUCTION[/green]")
    console.print("="*70, style="bold blue")

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
                    f.write(f"{stock}")
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
                symbol = line.split(',')[0].split('	')[0].strip().upper()
                if symbol:
                    stocks.append(symbol)

        console.print(f"📈 Loaded [cyan]{len(stocks)}[/cyan] symbols from [yellow]{file_name}[/yellow]")
        return stocks

    except Exception:
        return []

def run_live_mode_clean(stocks):
    """Option live trading mode with enhanced indicators for buyers"""
    console.print("🚀 [bold red]LIVE MODE STARTING - OPTION BUYER LOGIC[/bold red]")
    console.print(f"📊 Monitoring [cyan]{len(stocks)}[/cyan] symbols")

    # Wait for market
    now = datetime.now(IST)
    market_start = today_ist_dt(CONFIG["MARKET_START"])
    market_end = today_ist_dt(CONFIG["MARKET_END"])
    if not (market_start <= now <= market_end):
        sleep_until(market_start)
        console.print("🔔 [bold green]Market is now OPEN![/bold green]")

    global previous_scores, last_bull_symbols, last_bear_symbols, performance_metrics
    previous_scores = {}
    last_bull_symbols = set()
    last_bear_symbols = set()
    performance_metrics = defaultdict(int)

    live_signals_file = f"live_signals_{datetime.now(IST).strftime('%Y%m%d')}.csv"
    try:
        if os.path.exists(live_signals_file):
            os.remove(live_signals_file)
    except Exception:
        pass
    console.print(f"💾 Live signals will be saved to: [yellow]{live_signals_file}[/yellow]")
    scan_count = 0
    try:
        while (market_start <= datetime.now(IST) <= market_end):
            scan_count += 1
            now_ist = datetime.now(IST)
            next_boundary = next_5min_boundary_ist(now_ist)
            wait_seconds = int((next_boundary - now_ist).total_seconds())
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            current_time = datetime.now(IST)
            console.print(f"🔄 [bold yellow]LIVE SCAN {scan_count}[/bold yellow] | {current_time.strftime('%H:%M:%S')} IST")
            try:
                stock_multi_data = prefetch_clean(stocks)
                if not stock_multi_data:
                    console.print("[red]⚠️ No data received, retrying in 30 seconds...[/red]")
                    time.sleep(30)
                    continue
            except Exception as e:
                console.print(f"[red]❌ Data fetch error: {e}[/red]")
                time.sleep(60)
                continue
            signals_this_scan = []
            current_scores = {}
            for symbol, timeframe_data in stock_multi_data.items():
                clean_symbol = symbol.replace('-EQ', '')
                filtered_timeframes = filter_timeframe_data(clean_symbol, timeframe_data, current_time)
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
                render_signals_clean(current_time, top_bullish, top_bearish)
                export_to_csv(current_time, top_bullish, top_bearish, live_signals_file)
            else:
                console.print("⚪ No significant signals at this time")
            if scan_count % 10 == 0:
                total_signals = sum(performance_metrics.values())
                console.print(f"📊 [bold blue]Session Summary (Scan {scan_count})[/bold blue]")
                console.print(f"Total Signals: [cyan]{total_signals}[/cyan]")
    except KeyboardInterrupt:
        console.print("[yellow]👤 Live mode stopped by user[/yellow]")
    finally:
        total_signals = sum(performance_metrics.values())
        console.print(f"📈 [bold green]LIVE SESSION COMPLETE[/bold green]")
        console.print(f"Total Scans: [cyan]{scan_count}[/cyan]")
        console.print(f"Total Signals: [cyan]{total_signals}[/cyan]")
        if 'bullish_signals' in performance_metrics:
            console.print(f"Bullish Signals: [green]{performance_metrics['bullish_signals']}[/green]")
        if 'bearish_signals' in performance_metrics:
            console.print(f"Bearish Signals: [red]{performance_metrics['bearish_signals']}[/red]")
        console.print(f"Results saved: [yellow]{live_signals_file}[/yellow]")

def main():
    print_clean_banner()
    try:
        parser = argparse.ArgumentParser(description="Option Signal Scanner - Enhanced Option Buyer Logic")
        parser.add_argument("--backtest-date", help="Backtest date (YYYY-MM-DD)")
        parser.add_argument("--stocks-file", default=CONFIG["SHARES_FILE"], help="Stock symbols file")
        parser.add_argument("--live", action="store_true", help="Live market mode")
        args = parser.parse_args()
        stocks = load_stock_list(args.stocks_file)
        if not stocks:
            console.print("[red]❌ No valid stocks loaded[/red]")
            return
        if args.live:
            run_live_mode_clean(stocks)
        elif args.backtest_date:
            try:
                datetime.strptime(args.backtest_date, "%Y-%m-%d")
                run_backtest_clean(args.backtest_date, stocks)
            except ValueError:
                console.print("[red]❌ Invalid date format. Use YYYY-MM-DD[/red]")
        else:
            console.print("🎯 [bold green]Option Signal Scanner - Option Buyer Enhanced[/bold green]")
            console.print("[cyan]Usage:[/cyan]")
            console.print("  [yellow]python scanner.py --backtest-date 2025-09-30[/yellow]")
            console.print("  [yellow]python scanner.py --live[/yellow]")
    except KeyboardInterrupt:
        console.print("[yellow]👤 Scanner stopped[/yellow]")
    finally:
        for sess in tdhist_pool:
            try:
                sess.disconnect()
            except Exception:
                pass
        if performance_metrics:
            total = sum(performance_metrics.values())
            if total > 0:
                console.print(f"📊 [bold green]Final: {total} signals generated[/bold green]")
        console.print("✅ [green]Shutdown complete[/green]")

if __name__ == "__main__":
    main()
