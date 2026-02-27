# ==============================================================================
# ENHANCED OPTION BUYER SCANNER v3.3 - BOLLINGER BAND SQUEEZE BREAKOUT EDITION
# Added: Bollinger Band Squeeze Detection + Volume & OI Spike Confirmation
# Enhanced: Scalping-focused signals for Option Buyers
# ==============================================================================

import os
import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
import threading
from collections import defaultdict
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from tqdm import tqdm
from truedata.history import TD_hist

# Enhanced table formatting libraries
try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Installing rich: pip install rich")

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    print("Installing colorama: pip install colorama")

try:
    from great_tables import GT, md, html, style, loc
    from great_tables.data import sp500
    GREAT_TABLES_AVAILABLE = True
except ImportError:
    GREAT_TABLES_AVAILABLE = False
    print("Installing great-tables: pip install great-tables")

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# Initialize console for rich output
if RICH_AVAILABLE:
    console = Console()

# Create a simple logger replacement
class Logger:
    def info(self, msg): print(f"[INFO] {msg}")
    def error(self, msg): print(f"[ERROR] {msg}")
    def warning(self, msg): print(f"[WARNING] {msg}")
    def exception(self, msg): print(f"[EXCEPTION] {msg}")

logger = Logger()

# ======== Enhanced Configuration with Bollinger Squeeze ========
class Config:
    TDUSERNAME = os.getenv("TRUEDATA_USER", "tdwsp751")
    TDPASSWORD = os.getenv("TRUEDATA_PASS", "raj@751")

    MARKET_START = "09:15"  # IST
    FIRST_RUN_AT = "09:20"  # IST; First scan after 09:15-09:20 candle
    FIRST_SCAN_DELAY = 15   # Wait 15 seconds after 09:20 for settlement
    MARKET_END   = "15:30"  # IST
    SETTLE_DELAY_SECONDS = 15  # wait after bar close for data settlement
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "64"))
    TD_HIST_SESSIONS = int(os.getenv("TD_HIST_SESSIONS", "5"))
    SHARES_FILE = os.getenv("SHARES_FILE", "shares.txt")
    BENCHMARK_INDEX = "NIFTY 50"

    # --- Backtesting Configuration ---
    BACKTEST_INTERVAL_MINUTES = 5
    BACKTEST_START_DELAY = 5
    BACKTEST_TOP_DISPLAY = 10

    # --- Enhanced Indicator Group Weights (Bollinger Squeeze Focus) ---
    GROUP_WEIGHTS = {
        "Trend": 2.5, "Momentum": 2.0, "Volume": 2.8, "Volatility": 3.0, "OI": 2.5,
    }

    # --- Enhanced Individual Indicator Weights (Bollinger Squeeze Priority) ---
    INDICATOR_WEIGHTS = {
        "MA_Slope": 2.0, "ADX": 1.8, "VWAP": 1.7, "EMA": 1.7, "MACD_Trend": 1.5,
        "RSI": 2.0, "Stochastic": 1.2, "CCI": 1.2, "ROC": 1.1, "WilliamsR": 1.0,
        "VolumeSurge": 2.8, "OBV": 1.8, "CMF": 1.8, "RelVol": 2.0,
        "VolatilityExpansion": 2.5, "Bollinger": 2.8, "BollingerSqueeze": 3.5,  # NEW
        "OptionBuyerMomentum": 2.8, "OIChange": 2.5, "VolumeOISync": 2.2,
    }

    # --- Enhanced Scoring & Signal Thresholds (Bollinger Squeeze Optimized) ---
    SCORE_THRESHOLD_MIN = 12.0  # Increased for quality
    SIGNAL_THRESHOLDS = {
        'Very Strong Buy': 60.0, 'Strong Buy': 35.0, 'Buy Signal': 18.0,
        'Very Strong Sell': -60.0, 'Strong Sell': -35.0, 'Sell Signal': -18.0,
    }

    # --- Bollinger Squeeze Specific Settings ---
    BB_PERIOD = 20
    BB_STD_DEV = 2.0
    SQUEEZE_THRESHOLD = 0.1  # BandWidth threshold for squeeze detection
    SQUEEZE_MIN_PERIODS = 5  # Minimum periods of squeeze before breakout
    BREAKOUT_VOLUME_MULTIPLIER = 1.8  # Volume must be 180% of average for valid breakout
    OI_SPIKE_THRESHOLD = 15.0  # OI must increase by 15% for confirmation

    # --- Market Regime Multipliers ---
    REGIME_MULTIPLIERS = {
        'bullish_in_bull_market': 1.15, 'bearish_in_bear_market': 1.15,
        'bullish_in_bear_market': 0.8, 'bearish_in_bull_market': 0.8,
    }

# Constants
IST = pytz.timezone("Asia/Kolkata")
BAR_SIZE_MAP = {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"}
DURATION_MAP = {5: "30 D", 15: "30 D", 30: "60 D", 60: "120 D", 1440: "365 D"}
TIMEFRAME_WEIGHTS = {5: 3.0, 15: 2.8, 30: 2.0, 60: 1.5, 1440: 1.0}  # Higher weight for 5min

# Silence noisy loggers
for noisy in ("truedata", "truedata.history", "truedata_ws", "websocket", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

# Enhanced state management for proper Volume/OI tracking
previous_scan_results = {}
previous_oi_data = {}
previous_volume_data = {}
intraday_volume_data = {}  # Track 5-minute volume changes
intraday_oi_data = {}      # Track 5-minute OI changes
scan_count = 0
backtest_stock_history = {}
current_scan_data = {}
bollinger_squeeze_history = {}  # Track squeeze states

# Color definitions
class Colors:
    HEADER = '\033[95m'; BLUE = '\033[94m'; CYAN = '\033[96m'
    GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'
    BOLD = '\033[1m'; UNDERLINE = '\033[4m'; END = '\033[0m'
    MAGENTA = '\033[35m'; ORANGE = '\033[33m'

def print_colored(text, color):
    if COLORAMA_AVAILABLE:
        color_map = {
            Colors.HEADER: Fore.MAGENTA + Style.BRIGHT,
            Colors.BLUE: Fore.BLUE + Style.BRIGHT,
            Colors.CYAN: Fore.CYAN + Style.BRIGHT,
            Colors.GREEN: Fore.GREEN + Style.BRIGHT,
            Colors.YELLOW: Fore.YELLOW + Style.BRIGHT,
            Colors.RED: Fore.RED + Style.BRIGHT,
            Colors.BOLD: Style.BRIGHT,
            Colors.MAGENTA: Fore.MAGENTA + Style.BRIGHT,
            Colors.ORANGE: Fore.YELLOW + Style.BRIGHT,
        }
        print(color_map.get(color, '') + text)
    else:
        print(f"{color}{text}{Colors.END}")

# ========== BOLLINGER BAND SQUEEZE FUNCTIONS ==========
def calculate_bollinger_bands_enhanced(df, period=20, std_dev=2):
    """Enhanced Bollinger Bands calculation with squeeze detection."""
    if len(df) < period:
        return pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index)

    middle = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    # Calculate BandWidth (volatility measure)
    bandwidth = (upper - lower) / middle

    return middle, upper, lower, bandwidth

def detect_bollinger_squeeze(df, period=20, std_dev=2, squeeze_threshold=0.1):
    """Detect Bollinger Band Squeeze conditions."""
    try:
        middle, upper, lower, bandwidth = calculate_bollinger_bands_enhanced(df, period, std_dev)

        if bandwidth.empty or len(bandwidth) < period + 10:
            return False, 0, 0, "insufficient_data"

        # Calculate BandWidth moving average for comparison
        bandwidth_ma = bandwidth.rolling(window=20).mean()

        # Current BandWidth
        current_bandwidth = bandwidth.iloc[-1]
        avg_bandwidth = bandwidth_ma.iloc[-1]

        # Squeeze detection: BandWidth below threshold and below average
        is_squeeze = (current_bandwidth < squeeze_threshold) and (current_bandwidth < avg_bandwidth * 0.8)

        # Count consecutive squeeze periods
        squeeze_periods = 0
        if not bandwidth.empty:
            for i in range(len(bandwidth) - 1, max(-1, len(bandwidth) - 20), -1):
                if bandwidth.iloc[i] < squeeze_threshold:
                    squeeze_periods += 1
                else:
                    break

        # Breakout detection
        close_price = df['Close'].iloc[-1]
        upper_band = upper.iloc[-1] if not upper.empty else 0
        lower_band = lower.iloc[-1] if not lower.empty else 0

        breakout_direction = "none"
        if close_price > upper_band:
            breakout_direction = "bullish"
        elif close_price < lower_band:
            breakout_direction = "bearish"
        elif not is_squeeze and squeeze_periods >= Config.SQUEEZE_MIN_PERIODS:
            # Price approaching bands after squeeze
            middle_price = middle.iloc[-1] if not middle.empty else close_price
            if close_price > middle_price * 1.005:
                breakout_direction = "bullish_pending"
            elif close_price < middle_price * 0.995:
                breakout_direction = "bearish_pending"

        return is_squeeze, squeeze_periods, current_bandwidth, breakout_direction

    except Exception as e:
        logger.error(f"Error detecting Bollinger squeeze: {e}")
        return False, 0, 0, "error"

def bollinger_squeeze_score(df, symbol):
    """Calculate Bollinger Squeeze score for signal generation."""
    try:
        is_squeeze, squeeze_periods, bandwidth, breakout_direction = detect_bollinger_squeeze(df)

        # Base score calculation
        base_score = 0

        if breakout_direction == "bullish":
            base_score = 3.0  # Strong bullish breakout
        elif breakout_direction == "bearish":
            base_score = -3.0  # Strong bearish breakout
        elif breakout_direction == "bullish_pending":
            base_score = 2.0  # Potential bullish breakout
        elif breakout_direction == "bearish_pending":
            base_score = -2.0  # Potential bearish breakout
        elif is_squeeze and squeeze_periods >= Config.SQUEEZE_MIN_PERIODS:
            # Long squeeze building pressure
            pressure_multiplier = min(squeeze_periods / 10, 2.0)
            base_score = 1.5 * pressure_multiplier  # Neutral but building pressure

        # Update squeeze history for this symbol
        global bollinger_squeeze_history
        bollinger_squeeze_history[symbol] = {
            'is_squeeze': is_squeeze,
            'squeeze_periods': squeeze_periods,
            'bandwidth': bandwidth,
            'breakout_direction': breakout_direction,
            'score': base_score
        }

        return base_score

    except Exception as e:
        logger.error(f"Error calculating Bollinger squeeze score for {symbol}: {e}")
        return 0

def validate_breakout_with_volume_oi(df, symbol, breakout_direction):
    """Validate Bollinger breakout with volume and OI confirmation."""
    try:
        if breakout_direction not in ["bullish", "bearish"]:
            return False, "no_breakout"

        # Volume validation
        current_volume = df['Volume'].iloc[-1] if len(df) > 0 else 0
        avg_volume = df['Volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else current_volume
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

        volume_confirmed = volume_ratio >= Config.BREAKOUT_VOLUME_MULTIPLIER

        # OI validation (using synthetic OI if not available)
        if 'OpenInterest' in df.columns:
            current_oi = df['OpenInterest'].iloc[-1]
            prev_oi = df['OpenInterest'].iloc[-2] if len(df) > 1 else current_oi
        else:
            # Synthetic OI based on volume patterns
            current_oi = current_volume * 0.3
            prev_oi = df['Volume'].iloc[-2] * 0.3 if len(df) > 1 else current_oi

        oi_change_pct = ((current_oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0
        oi_confirmed = oi_change_pct >= Config.OI_SPIKE_THRESHOLD

        # Both confirmations needed for high-quality signal
        if volume_confirmed and oi_confirmed:
            return True, "fully_confirmed"
        elif volume_confirmed:
            return True, "volume_confirmed"
        elif oi_confirmed:
            return True, "oi_confirmed"
        else:
            return False, "weak_breakout"

    except Exception as e:
        logger.error(f"Error validating breakout for {symbol}: {e}")
        return False, "error"

# ========== ENHANCED TECHNICAL INDICATORS ==========
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

def momentum(df, period=10):
    return df["Close"] / df["Close"].shift(period) - 1.0

def volume_surge_enhanced(df, lookback=20, breakout_threshold=1.8):
    """Enhanced volume surge detection for breakouts."""
    vol_ma = df["Volume"].rolling(lookback).mean()
    vol_std = df["Volume"].rolling(lookback).std()
    z_score = (df["Volume"] - vol_ma) / vol_std.replace(0, np.nan)

    # Additional check for breakout volume
    current_vol_ratio = df["Volume"] / vol_ma
    is_breakout_volume = current_vol_ratio >= breakout_threshold

    # Combine Z-score with breakout volume flag
    enhanced_score = z_score.copy()
    if not enhanced_score.empty and not current_vol_ratio.empty:
        enhanced_score.iloc[-1] = enhanced_score.iloc[-1] * (2.0 if is_breakout_volume.iloc[-1] else 1.0)

    return enhanced_score.fillna(0)

def calculate_rsi(df, period=14):
    if len(df) < period + 1: 
        return pd.Series(dtype='float64', index=df.index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rs.fillna(100, inplace=True)
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    if len(df) < slow + signal: 
        return pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index)
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_stochastic(df, period=14, smooth_d=3):
    if len(df) < period + smooth_d: 
        return pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index)
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan))
    k.fillna(50, inplace=True)
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_adx(df, period=14):
    if len(df) < period * 2: 
        return pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index)
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

def calculate_roc(df, period=12):
    if len(df) < period + 1: 
        return pd.Series(dtype='float64', index=df.index)
    shifted_close = df['Close'].shift(period).replace(0, np.nan)
    return ((df['Close'] - df['Close'].shift(period)) / shifted_close) * 100

def calculate_obv(df):
    if len(df) < 2: 
        return pd.Series(dtype='float64', index=df.index)
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

def calculate_cci(df, period=20):
    if len(df) < period: 
        return pd.Series(dtype='float64', index=df.index)
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).replace(0, np.nan)
    return (tp - sma_tp) / (0.015 * mad)

def cmf(df, period=20):
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = mfm.fillna(0)
    mfv = mfm * df["Volume"]
    mfv_sum = mfv.rolling(period).sum()
    vol_sum = df["Volume"].rolling(period).sum().replace(0, np.nan)
    return (mfv_sum / vol_sum).fillna(0)

def relative_volume(df, lookback=50):
    vol_ma = df["Volume"].rolling(lookback).mean()
    return (df["Volume"] / vol_ma.replace(0, np.nan)).fillna(1.0)

def calculate_oi_volume_ratio(df):
    if 'OpenInterest' not in df.columns:
        df['OpenInterest'] = df['Volume'].rolling(20).mean() * 0.3
    ratio = df['OpenInterest'] / df['Volume'].replace(0, np.nan)
    return ratio.fillna(0)

def detect_oi_buildup_enhanced(df, lookback=20):
    """Enhanced OI buildup detection for squeeze breakouts."""
    if 'OpenInterest' not in df.columns:
        df['OpenInterest'] = df['Volume'].rolling(20).mean() * 0.3

    oi_ma = df['OpenInterest'].rolling(lookback).mean()
    current_oi = df['OpenInterest'].iloc[-1] if len(df) > 0 else 0
    avg_oi = oi_ma.iloc[-1] if len(oi_ma) > 0 else 0

    if avg_oi > 0:
        oi_strength = (current_oi - avg_oi) / avg_oi

        # Enhanced scoring for breakout scenarios
        base_score = max(min(oi_strength * 100, 100), -100)

        # Boost score if volume is also elevated (breakout confirmation)
        vol_ratio = relative_volume(df).iloc[-1] if len(df) > 0 else 1
        if vol_ratio > Config.BREAKOUT_VOLUME_MULTIPLIER:
            base_score *= 1.5

        return base_score
    return 0

def volume_oi_sync_analysis_enhanced(df):
    """Enhanced volume-OI sync analysis for squeeze breakouts."""
    if len(df) < 10: return 0
    if 'OpenInterest' not in df.columns:
        df['OpenInterest'] = df['Volume'].rolling(20).mean() * 0.3

    vol_change = df['Volume'].pct_change(5).fillna(0)
    oi_change = df['OpenInterest'].pct_change(5).fillna(0)

    # Current values
    vol_chg_current = vol_change.iloc[-1]
    oi_chg_current = oi_change.iloc[-1]

    # Base sync score
    sync_score = vol_chg_current + oi_chg_current

    # Enhanced scoring for breakout patterns
    if vol_chg_current > 0.5 and oi_chg_current > 0.3:  # Both surging (bullish breakout)
        sync_score *= 2.0
    elif vol_chg_current > 0.5 and oi_chg_current < -0.2:  # Volume up, OI down (bearish signal)
        sync_score = -abs(sync_score) * 1.5

    return min(max(sync_score * 50, -100), 100)

def option_buyer_momentum_enhanced(df, symbol):
    """Enhanced option buyer momentum with Bollinger squeeze consideration."""
    if len(df) < 20: return 0

    # Base momentum calculation
    price_mom = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
    vol_surge_val = volume_surge_enhanced(df, lookback=20).iloc[-1] if len(df) > 20 else 0
    oi_buildup = detect_oi_buildup_enhanced(df, lookback=20)

    # Bollinger squeeze component
    squeeze_score = bollinger_squeeze_score(df, symbol)

    # Enhanced weighting with squeeze factor
    combined_score = (price_mom * 0.3) + (vol_surge_val * 0.25) + (oi_buildup * 0.25) + (squeeze_score * 0.2)

    return min(max(combined_score, -100), 100)

def slope(series, lookback=10):
    if len(series) < lookback: return 0.0
    y = series.tail(lookback).values
    x = np.arange(len(y))
    if len(y) < 2: return 0.0
    try:
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]
    except:
        return 0.0

# ========== ENHANCED SCORING ENGINE WITH BOLLINGER SQUEEZE ==========
def normalize_score(value, bullish_range, bearish_range, score_range=(-2.0, 2.0)):
    low_score, high_score = score_range
    bull_min, bull_max = bullish_range
    if value >= bull_max: return high_score
    if value > bull_min:
        return high_score * ((value - bull_min) / (bull_max - bull_min))
    bear_max, bear_min = bearish_range
    if value <= bear_min: return low_score
    if value < bear_max:
        return low_score * ((bear_max - value) / (bear_max - bear_min))
    return 0.0

def calculate_indicator_scores_enhanced(df, symbol):
    """Enhanced indicator scoring with Bollinger Squeeze priority."""
    scores = defaultdict(float)
    if df is None or len(df) < 50: return scores

    try:
        # --- Trend Group ---
        adx, pdi, ndi = calculate_adx(df)
        if not adx.empty and len(adx) > 3 and adx.iloc[-1] > 20 and adx.iloc[-1] > adx.iloc[-3]:
            scores['ADX'] = 2.0 if pdi.iloc[-1] > ndi.iloc[-1] else -2.0

        ema20, ema50 = ema(df['Close'], 20), ema(df['Close'], 50)
        if not ema20.empty and not ema50.empty and len(ema20) > 0 and len(ema50) > 0:
            ema_ratio = ema20.iloc[-1] / ema50.iloc[-1] if ema50.iloc[-1] != 0 else 1
            scores['EMA'] = normalize_score(ema_ratio, (1.001, 1.02), (0.999, 0.98))

        vwap_line = vwap(df, period=None)
        if not vwap_line.empty and len(vwap_line) > 0:
            vwap_ratio = df['Close'].iloc[-1] / vwap_line.iloc[-1] if vwap_line.iloc[-1] != 0 else 1
            scores['VWAP'] = normalize_score(vwap_ratio, (1.002, 1.025), (0.998, 0.975))

        macd, signal = calculate_macd(df)
        if not macd.empty and not signal.empty and len(macd) > 0:
            if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-1] > 0:
                scores['MACD_Trend'] = 2.0
            elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-1] < 0:
                scores['MACD_Trend'] = -2.0
            else:
                scores['MACD_Trend'] = 0

        if not ema20.empty and len(ema20) >= 5:
            ma20_slope = slope(ema20, 5)
            price_norm_slope = ma20_slope / df['Close'].iloc[-1] * 1000 if df['Close'].iloc[-1] != 0 else 0
            scores['MA_Slope'] = normalize_score(price_norm_slope, (0.1, 0.5), (-0.1, -0.5), (-2.5, 2.5))

        # --- Momentum Group ---
        rsi = calculate_rsi(df)
        if not rsi.empty and len(rsi) > 0:
            scores['RSI'] = normalize_score(rsi.iloc[-1], (60, 85), (40, 15))

        k, d = calculate_stochastic(df)
        if not k.empty and not d.empty and len(k) > 0 and len(d) > 0:
            if k.iloc[-1] > d.iloc[-1]:
                scores['Stochastic'] = normalize_score(k.iloc[-1], (20, 80), (100, 100))
            elif k.iloc[-1] < d.iloc[-1]:
                scores['Stochastic'] = normalize_score(k.iloc[-1], (0,0), (80, 20))

        cci = calculate_cci(df)
        if not cci.empty and len(cci) > 0:
            scores['CCI'] = normalize_score(cci.iloc[-1], (100, 200), (-100, -200))

        roc = calculate_roc(df)
        if not roc.empty and len(roc) > 0:
            scores['ROC'] = normalize_score(roc.iloc[-1], (0.5, 2.0), (-0.5, -2.0))

        wr = williams_r(df)
        if not wr.empty and len(wr) > 0:
            scores['WilliamsR'] = normalize_score(wr.iloc[-1], (-100, -80), (-20, 0))

        # --- Enhanced Volume Group ---
        zscore = volume_surge_enhanced(df, lookback=20)
        if not zscore.empty and len(zscore) > 1:
            price_up = df['Close'].iloc[-1] > df['Close'].iloc[-2]
            if price_up:
                scores['VolumeSurge'] = normalize_score(zscore.iloc[-1], (1.5, 3.5), (0,0))  # Higher threshold
            else:
                scores['VolumeSurge'] = normalize_score(zscore.iloc[-1], (0,0), (-1.5, -3.5))

        obv_line = calculate_obv(df)
        if len(obv_line) > 5:
            obv_slope = slope(obv_line, 5)
            scores['OBV'] = normalize_score(obv_slope, (1, 1e9), (-1, -1e9))

        cmf20 = cmf(df, period=20)
        if not cmf20.empty and len(cmf20) > 0:
            scores['CMF'] = normalize_score(cmf20.iloc[-1], (0.1, 0.25), (-0.1, -0.25))

        rv = relative_volume(df, lookback=50)
        if not rv.empty and len(rv) > 0:
            scores['RelVol'] = normalize_score(rv.iloc[-1], (1.8, 3.5), (0.5, 0.5))  # Higher threshold

        # --- Enhanced Volatility Group with Bollinger Squeeze ---
        atr_val = atr(df, period=14)
        if len(atr_val) > 20:
            atr_ma = atr_val.rolling(20).mean()
            if len(atr_ma) > 0 and atr_ma.iloc[-1] != 0:
                atr_ratio = atr_val.iloc[-1] / atr_ma.iloc[-1]
                atr_slope_ratio = (atr_val.iloc[-1] / atr_val.iloc[-5]) if len(atr_val) >= 5 and atr_val.iloc[-5] > 0 else 1
                if atr_ratio > 1.1 and atr_slope_ratio > 1.1:
                    price_direction = 1 if df['Close'].iloc[-1] > df['Close'].iloc[-5] else -1
                    scores['VolatilityExpansion'] = 2.5 * price_direction

        # Standard Bollinger Bands
        _, bb_upper, bb_lower = calculate_bollinger_bands_enhanced(df)
        if not bb_upper.empty and not bb_lower.empty and len(bb_upper) > 0 and len(bb_lower) > 0:
            if df['Close'].iloc[-1] > bb_upper.iloc[-1]: 
                scores['Bollinger'] = 2.8  # Increased weight
            elif df['Close'].iloc[-1] < bb_lower.iloc[-1]: 
                scores['Bollinger'] = -2.8

        # NEW: Enhanced Bollinger Squeeze Detection
        squeeze_score = bollinger_squeeze_score(df, symbol)
        scores['BollingerSqueeze'] = squeeze_score

        # Validate breakout if detected
        squeeze_info = bollinger_squeeze_history.get(symbol, {})
        breakout_direction = squeeze_info.get('breakout_direction', 'none')

        if breakout_direction in ['bullish', 'bearish']:
            is_confirmed, confirmation_type = validate_breakout_with_volume_oi(df, symbol, breakout_direction)
            if is_confirmed:
                # Boost Bollinger squeeze score for confirmed breakouts
                if confirmation_type == "fully_confirmed":
                    scores['BollingerSqueeze'] *= 1.8
                elif confirmation_type in ["volume_confirmed", "oi_confirmed"]:
                    scores['BollingerSqueeze'] *= 1.4

        # --- Enhanced OI Group ---
        scores['OIChange'] = normalize_score(detect_oi_buildup_enhanced(df, 20), (15, 35), (-15, -35))  # Higher thresholds
        scores['VolumeOISync'] = normalize_score(volume_oi_sync_analysis_enhanced(df), (20, 45), (-20, -45))
        scores['OptionBuyerMomentum'] = normalize_score(option_buyer_momentum_enhanced(df, symbol), (25, 55), (-25, -55), (-3.0, 3.0))

    except Exception as e:
        logger.error(f"Error calculating enhanced indicator scores for {symbol}: {e}")

    return scores

def analyze_signals_enhanced(timeframe_data, market_regime='neutral', symbol=''):
    """Enhanced signal analysis with Bollinger Squeeze priority."""
    total_score, total_weight = 0.0, 0.0
    group_scores = defaultdict(float)
    group_weights = defaultdict(float)

    for tf_min, df in timeframe_data.items():
        if df is None or len(df) < 50: continue

        indicator_scores = calculate_indicator_scores_enhanced(df, symbol)
        tf_weight = TIMEFRAME_WEIGHTS.get(tf_min, 1.0)

        for group, weight in Config.GROUP_WEIGHTS.items():
            grp_score, grp_weight = 0.0, 0.0
            for indicator, ind_weight in Config.INDICATOR_WEIGHTS.items():
                if indicator in indicator_scores:
                    belongs_to_group = (
                        (group == 'Trend' and any(term in indicator for term in ['MA', 'ADX', 'VWAP', 'EMA', 'MACD'])) or
                        (group == 'Momentum' and any(term in indicator for term in ['RSI', 'Stochastic', 'CCI', 'ROC', 'Williams'])) or
                        (group == 'Volume' and any(term in indicator for term in ['Vol', 'OBV', 'CMF'])) or
                        (group == 'Volatility' and any(term in indicator for term in ['Volatility', 'Bollinger'])) or
                        (group == 'OI' and any(term in indicator for term in ['OI', 'Option']))
                    )

                    if belongs_to_group:
                        grp_score += indicator_scores[indicator] * ind_weight
                        grp_weight += abs(indicator_scores[indicator]) * ind_weight

            if grp_weight > 0:
                norm_grp_score = (grp_score / grp_weight) * weight * tf_weight
                group_scores[group] += norm_grp_score
                group_weights[group] += weight * tf_weight

    final_score = 0
    max_possible_score = 0
    for group, score in group_scores.items():
        final_score += score
        max_possible_score += group_weights[group]

    if max_possible_score == 0: return 'Neutral', 0.0, {}

    normalized_score = (final_score / max_possible_score) * 100

    # Apply regime multipliers
    if normalized_score > 0 and market_regime == 'bullish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bullish_in_bull_market']
    elif normalized_score > 0 and market_regime == 'bearish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bullish_in_bear_market']
    elif normalized_score < 0 and market_regime == 'bearish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bearish_in_bear_market']
    elif normalized_score < 0 and market_regime == 'bullish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bearish_in_bull_market']

    # Enhanced signal classification
    if normalized_score >= Config.SIGNAL_THRESHOLDS['Very Strong Buy']: 
        signal = 'Very Strong Buy'
    elif normalized_score >= Config.SIGNAL_THRESHOLDS['Strong Buy']: 
        signal = 'Strong Buy'
    elif normalized_score >= Config.SIGNAL_THRESHOLDS['Buy Signal']: 
        signal = 'Buy Signal'
    elif normalized_score <= Config.SIGNAL_THRESHOLDS['Very Strong Sell']: 
        signal = 'Very Strong Sell'
    elif normalized_score <= Config.SIGNAL_THRESHOLDS['Strong Sell']: 
        signal = 'Strong Sell'
    elif normalized_score <= Config.SIGNAL_THRESHOLDS['Sell Signal']: 
        signal = 'Sell Signal'
    else: 
        signal = 'Neutral'

    final_sub_scores = {}
    for group in group_scores:
        if group_weights[group] > 0:
            final_sub_scores[group] = group_scores[group] / group_weights[group] * 10

    return signal, normalized_score, final_sub_scores

# ========== TABLE DISPLAY FUNCTIONS ==========
def create_enhanced_bollinger_squeeze_table(data, title, new_stocks=None, show_time=None):
    """Enhanced table showing Bollinger Squeeze specific information."""
    if not data:
        print_colored(f"\n{title}", Colors.HEADER)
        print_colored("No Bollinger Squeeze opportunities found.", Colors.YELLOW)
        return

    if RICH_AVAILABLE:
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold blue")

        # Enhanced columns for Bollinger Squeeze
        table.add_column("Stock", style="bold white", width=12, justify="left")
        table.add_column("Signal", style="bold", width=16, justify="center")
        table.add_column("Score", style="bold", width=8, justify="right")
        table.add_column("Squeeze", style="cyan", width=8, justify="center")
        table.add_column("Periods", style="yellow", width=7, justify="right")
        table.add_column("Breakout", style="magenta", width=12, justify="center")
        table.add_column("Vol Δ%", style="bright_green", width=8, justify="right")
        table.add_column("OI Δ%", style="bright_cyan", width=8, justify="right")
        table.add_column("Confirm", style="bold", width=10, justify="center")
        table.add_column("Action", style="bold", width=16, justify="center")

        for item in data:
            symbol = item['symbol']
            is_new = new_stocks and symbol in new_stocks

            # Get Bollinger squeeze info
            squeeze_info = bollinger_squeeze_history.get(symbol, {})
            is_squeeze = squeeze_info.get('is_squeeze', False)
            squeeze_periods = squeeze_info.get('squeeze_periods', 0)
            breakout_direction = squeeze_info.get('breakout_direction', 'none')

            # Format squeeze status
            squeeze_status = "🔥 YES" if is_squeeze else "No"
            squeeze_color = "bright_red" if is_squeeze else "dim"

            # Format breakout direction
            breakout_display = {
                'bullish': '🚀 BULL',
                'bearish': '📉 BEAR', 
                'bullish_pending': '⬆️ Pending',
                'bearish_pending': '⬇️ Pending',
                'none': 'None'
            }.get(breakout_direction, 'Unknown')

            # Determine confirmation status
            confirmation = "Strong" if abs(item['score']) > 35 and item.get('volume_change_pct', 0) > 20 else "Weak"
            confirmation_color = "bright_green" if confirmation == "Strong" else "yellow"

            # Enhanced action based on squeeze and breakout
            if breakout_direction == 'bullish' and confirmation == "Strong":
                action = "🔥 Strong Call Buy"
                action_color = "bright_green"
            elif breakout_direction == 'bearish' and confirmation == "Strong":
                action = "🔥 Strong Put Buy"
                action_color = "bright_red"
            elif is_squeeze and squeeze_periods >= 5:
                action = "⏳ Watch Breakout"
                action_color = "yellow"
            else:
                action = item.get('action', 'Consider')
                action_color = "white"

            # Style stock symbol
            if is_new:
                stock_style = f"[bold bright_magenta]{symbol} ✨[/bold bright_magenta]"
            else:
                stock_style = symbol

            # Format volume and OI changes
            vol_chg = item.get('volume_change_pct', 0)
            oi_chg = item.get('oi_change_pct', 0)

            vol_display = f"[bright_green]{vol_chg:+.1f}%[/bright_green]" if vol_chg > 20 else f"[green]{vol_chg:+.1f}%[/green]" if vol_chg > 0 else f"[red]{vol_chg:+.1f}%[/red]" if vol_chg < 0 else "[dim]0.0%[/dim]"
            oi_display = f"[bright_cyan]{oi_chg:+.1f}%[/bright_cyan]" if oi_chg > 15 else f"[cyan]{oi_chg:+.1f}%[/cyan]" if oi_chg > 0 else f"[red]{oi_chg:+.1f}%[/red]" if oi_chg < 0 else "[dim]0.0%[/dim]"

            table.add_row(
                stock_style,
                f"[bold]{item['signal']}[/bold]",
                f"[bold]{item['score']:.1f}[/bold]",
                f"[{squeeze_color}]{squeeze_status}[/{squeeze_color}]",
                f"[yellow]{squeeze_periods}[/yellow]" if squeeze_periods > 0 else "[dim]0[/dim]",
                f"[magenta]{breakout_display}[/magenta]",
                vol_display,
                oi_display,
                f"[{confirmation_color}]{confirmation}[/{confirmation_color}]",
                f"[{action_color}]{action}[/{action_color}]"
            )

        # Display with enhanced title
        if show_time:
            console.print(f"\n[bold magenta]{title} - {show_time}[/bold magenta]")
        else:
            console.print(f"\n[bold magenta]{title}[/bold magenta]")

        console.print("[bold cyan]🎯 Bollinger Band Squeeze Breakout Analysis for Option Buyers[/bold cyan]")
        console.print(table)

        # Add legend
        console.print("\n[dim]Legend: 🔥=Active Squeeze, 🚀=Bullish Breakout, 📉=Bearish Breakout, ⏳=Watch Setup[/dim]")

    else:
        # ASCII fallback
        print_colored(f"\n{title} - Bollinger Squeeze Analysis", Colors.HEADER)
        print_colored("="*150, Colors.BLUE)

        header = f"{'Stock':<12} | {'Signal':<16} | {'Score':>8} | {'Squeeze':>8} | {'Periods':>7} | {'Breakout':<12} | {'Vol%':>7} | {'OI%':>7} | {'Action':<16}"
        print_colored(header, Colors.BOLD)
        print_colored("-"*150, Colors.BLUE)

        for item in data:
            symbol = item['symbol']
            squeeze_info = bollinger_squeeze_history.get(symbol, {})
            is_squeeze = squeeze_info.get('is_squeeze', False)
            squeeze_periods = squeeze_info.get('squeeze_periods', 0)
            breakout_direction = squeeze_info.get('breakout_direction', 'none')

            squeeze_status = "YES" if is_squeeze else "No"
            breakout_display = breakout_direction.replace('_', ' ').title()

            vol_chg = item.get('volume_change_pct', 0)
            oi_chg = item.get('oi_change_pct', 0)

            action = "Strong Call" if breakout_direction == 'bullish' and abs(item['score']) > 35 else "Strong Put" if breakout_direction == 'bearish' and abs(item['score']) > 35 else "Watch" if is_squeeze else item.get('action', 'Consider')

            row = f"{symbol:<12} | {item['signal']:<16} | {item['score']:>8.1f} | {squeeze_status:>8} | {squeeze_periods:>7} | {breakout_display:<12} | {vol_chg:>6.1f}% | {oi_chg:>6.1f}% | {action:<16}"

            if is_new and symbol in (new_stocks or []):
                print_colored(row + " ← ✨ NEW!", Colors.MAGENTA)
            else:
                print(row)

        print_colored("="*150, Colors.BLUE)

# ========== Continue with the rest of the original script functions ==========
# (Including all the data fetching, timing, main scanner logic functions)

# [Note: Due to length constraints, I'm showing the key enhanced parts. The complete script would include all the original functions like:]
# - Authentication and session management
# - Data fetching functions (prefetch_all_timeaware, etc.)
# - Volume/OI tracking functions
# - Market regime detection
# - Live scanner main loop
# - Backtest functionality
# - All utility functions

def main_bollinger_squeeze_scanner():
    """Main function for the enhanced Bollinger Squeeze scanner."""
    parser = argparse.ArgumentParser(description="Enhanced Bollinger Squeeze Options Scanner v3.3")
    parser.add_argument("--asof", type=str, help="Backtest snapshot: 2025-09-30T14:50")
    parser.add_argument("--backtest", type=str, help="Full day backtest: 2025-09-30")
    parser.add_argument("--squeeze-only", action='store_true', help="Show only Bollinger squeeze opportunities")
    args = parser.parse_args()

    try:
        with open(Config.SHARES_FILE, 'r') as f:
            stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(stocks)} symbols for Bollinger Squeeze analysis")
    except Exception:
        stocks = ["RELIANCE-EQ", "TCS-EQ", "HDFCBANK-EQ", "INFY-EQ", "HINDUNILVR-EQ"]
        logger.warning("Using sample stocks for Bollinger Squeeze analysis")

    print_colored("\n🎯 BOLLINGER BAND SQUEEZE BREAKOUT SCANNER v3.3", Colors.HEADER)
    print_colored("✅ Enhanced: Squeeze Detection + Volume/OI Confirmation", Colors.GREEN)
    print_colored("🎯 Focus: High-probability scalping setups for option buyers", Colors.CYAN)

    # Display Bollinger Squeeze specific settings
    print_colored("\n📊 SQUEEZE DETECTION SETTINGS:", Colors.CYAN)
    print(f"  BandWidth Threshold: {Config.SQUEEZE_THRESHOLD}")
    print(f"  Minimum Squeeze Periods: {Config.SQUEEZE_MIN_PERIODS}")
    print(f"  Breakout Volume Multiplier: {Config.BREAKOUT_VOLUME_MULTIPLIER}x")
    print(f"  OI Spike Threshold: {Config.OI_SPIKE_THRESHOLD}%")

    # Run the appropriate mode based on arguments
    # (Implementation would continue with the scanning logic using the enhanced functions)

if __name__ == "__main__":
    main_bollinger_squeeze_scanner()
