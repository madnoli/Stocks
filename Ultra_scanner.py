# ==============================================================================
# ULTRA-ENHANCED OPTION BUYER SCANNER v4.0 - PART 1: IMPORTS & CONFIGURATION
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
import traceback

# Enhanced imports with fallbacks
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.total = kwargs.get('total', 100)
            self.count = 0
        def update(self, n=1):
            self.count += n
            if self.count % 10 == 0:
                print(f"Progress: {self.count}/{self.total}")
        def __enter__(self): return self
        def __exit__(self, *args): pass

try:
    from truedata.history import TD_hist
    TRUEDATA_AVAILABLE = True
except ImportError:
    TRUEDATA_AVAILABLE = False
    print("❌ TrueData not available. Install: pip install truedata-ws")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.text import Text
    from rich import box
    from rich.panel import Panel
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

try:
    from colorama import init, Fore, Back, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False

try:
    from great_tables import GT, md, html, style, loc
    GREAT_TABLES_AVAILABLE = True
except ImportError:
    GREAT_TABLES_AVAILABLE = False

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

# Silence specific loggers
for logger_name in ["truedata", "truedata.history", "truedata_ws", "websocket", "urllib3", "requests"]:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)

# Enhanced logger
class UltraLogger:
    def __init__(self):
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(__name__)

    def info(self, msg): self.logger.info(msg)
    def error(self, msg): self.logger.error(msg)
    def warning(self, msg): self.logger.warning(msg)
    def exception(self, msg): self.logger.exception(msg)

logger = UltraLogger()

# =============================================================================
# ULTRA-ENHANCED CONFIGURATION
# =============================================================================

class UltraConfig:
    """Ultra-Enhanced Configuration for Maximum Accuracy"""

    # TrueData Configuration
    TDUSERNAME = os.getenv("TRUEDATA_USER", "tdwsp751")
    TDPASSWORD = os.getenv("TRUEDATA_PASS", "raj@751")

    # Market Timing (IST)
    MARKET_START = "09:15"
    FIRST_RUN_AT = "09:20"
    FIRST_SCAN_DELAY = 15
    MARKET_END = "15:30"
    SETTLE_DELAY_SECONDS = 15

    # Performance Settings
    MAX_WORKERS = min(64, (os.cpu_count() or 1) * 4)
    TD_HIST_SESSIONS = max(3, min(8, os.cpu_count() or 1))
    SHARES_FILE = os.getenv("SHARES_FILE", "shares.txt")
    BENCHMARK_INDEX = "NIFTY 50"

    # Backtesting Configuration
    BACKTEST_INTERVAL_MINUTES = 5
    BACKTEST_START_DELAY = 5
    BACKTEST_TOP_DISPLAY = 10

    # Ultra-Enhanced Indicator Group Weights
    GROUP_WEIGHTS = {
        "Trend": 3.2,
        "Momentum": 2.8,
        "Volume": 3.5,
        "Volatility": 2.2,
        "OI": 4.0,
    }

    # Ultra-Refined Individual Indicator Weights
    INDICATOR_WEIGHTS = {
        "MA_Slope": 2.5, "ADX": 2.3, "VWAP": 2.2, "EMA": 2.0, "MACD_Trend": 2.0,
        "RSI": 2.2, "Stochastic": 1.8, "CCI": 1.5, "ROC": 1.3, "WilliamsR": 1.2,
        "MFI": 2.0, "VolumeSurge": 3.0, "OBV": 2.3, "CMF": 2.5, "RelVol": 2.2,
        "VolatilityExpansion": 3.2, "Bollinger": 2.0,
        "OptionBuyerMomentum": 4.0, "OIChange": 3.8, "VolumeOISync": 3.5,
    }

    # Ultra-Precise Signal Thresholds
    SIGNAL_THRESHOLDS = {
        'Ultra Strong Buy': 70.0, 'Very Strong Buy': 55.0, 'Strong Buy': 30.0,
        'Buy Signal': 15.0, 'Weak Buy': 8.0,
        'Ultra Strong Sell': -70.0, 'Very Strong Sell': -55.0, 'Strong Sell': -30.0,
        'Sell Signal': -15.0, 'Weak Sell': -8.0,
    }

    # Market Regime Multipliers
    REGIME_MULTIPLIERS = {
        'bullish_in_bull_market': 1.25,
        'bearish_in_bear_market': 1.25,
        'bullish_in_bear_market': 0.75,
        'bearish_in_bull_market': 0.75,
        'trending_market': 1.15,
        'sideways_market': 0.85,
    }

    # Minimum thresholds
    SCORE_THRESHOLD_MIN = 10.0
    MIN_VOLUME_SURGE = 1.2
    MIN_OI_CHANGE = 2.0
    MIN_BARS_REQUIRED = 100

# Constants and Globals
IST = pytz.timezone("Asia/Kolkata")
BAR_SIZE_MAP = {1: "1 min", 2: "2 min", 3: "3 min", 5: "5 min", 15: "15 min", 
                30: "30 min", 60: "60 min", 1440: "1 day"}
DURATION_MAP = {1: "5 D", 2: "10 D", 3: "15 D", 5: "30 D", 15: "30 D", 
                30: "60 D", 60: "120 D", 1440: "365 D"}
TIMEFRAME_WEIGHTS = {1: 1.8, 2: 2.0, 3: 2.2, 5: 2.8, 15: 3.2, 30: 2.5, 60: 1.8, 1440: 1.2}

# Global state management
previous_scan_results = {}
previous_oi_data = {}
previous_volume_data = {}
intraday_volume_data = {}
intraday_oi_data = {}
scan_count = 0
backtest_stock_history = {}
current_scan_data = {}
api_calls_done = 0
api_calls_lock = threading.Lock()

# Color definitions
class Colors:
    HEADER = '\033[95m'; BLUE = '\033[94m'; CYAN = '\033[96m'
    GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'
    BOLD = '\033[1m'; UNDERLINE = '\033[4m'; END = '\033[0m'
    MAGENTA = '\033[35m'; ORANGE = '\033[33m'
    BRIGHT_GREEN = '\033[92m\033[1m'; BRIGHT_RED = '\033[91m\033[1m'

def print_colored(text: str, color: str):
    """Enhanced colored printing with fallbacks"""
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
# ==============================================================================
# ULTRA-ENHANCED OPTION BUYER SCANNER v4.0 - PART 2: TECHNICAL INDICATORS
# ==============================================================================

def ultra_ema(series: pd.Series, length: int) -> pd.Series:
    """Ultra-precise Exponential Moving Average"""
    if len(series) < length:
        return pd.Series(index=series.index, dtype='float64')
    return series.ewm(span=length, adjust=False).mean()

def ultra_vwap(df: pd.DataFrame, period=None) -> pd.Series:
    """Ultra-precise Volume Weighted Average Price"""
    if df.empty or len(df) < 2:
        return pd.Series(index=df.index, dtype='float64')

    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3.0
    pv = typical_price * df["Volume"]

    if period:
        pv_sum = pv.rolling(window=period, min_periods=1).sum()
        vol_sum = df["Volume"].rolling(window=period, min_periods=1).sum()
    else:
        pv_sum = pv.cumsum()
        vol_sum = df["Volume"].cumsum()

    return (pv_sum / vol_sum.replace(0, np.nan)).fillna(0)

def ultra_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Ultra-precise Average True Range"""
    if len(df) < 2:
        return pd.Series(index=df.index, dtype='float64')

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1/period, adjust=False).mean()

def ultra_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Ultra-precise Williams %R"""
    if len(df) < period:
        return pd.Series(index=df.index, dtype='float64')

    highest = df["High"].rolling(window=period, min_periods=period).max()
    lowest = df["Low"].rolling(window=period, min_periods=period).min()

    wr = -100 * ((highest - df["Close"]) / (highest - lowest).replace(0, np.nan))
    return wr.fillna(-50)

def ultra_volume_surge(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Ultra-precise Volume Surge Detection"""
    if len(df) < lookback + 1:
        return pd.Series(index=df.index, dtype='float64')

    vol_ma = df["Volume"].rolling(window=lookback, min_periods=lookback).mean()
    vol_std = df["Volume"].rolling(window=lookback, min_periods=lookback).std()

    z_score = (df["Volume"] - vol_ma) / vol_std.replace(0, np.nan)
    return z_score.fillna(0)

def calculate_ultra_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Ultra-precise RSI calculation with Wilder's smoothing"""
    if len(df) < period + 1: 
        return pd.Series(dtype='float64', index=df.index)

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()

    rs = gain / loss.replace(0, np.nan)
    rs.fillna(100, inplace=True)
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)

def calculate_ultra_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Ultra-precise MACD calculation"""
    if len(df) < slow + signal: 
        empty_series = pd.Series(dtype='float64', index=df.index)
        return empty_series, empty_series, empty_series

    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()

    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line

    return macd, signal_line, histogram

def calculate_ultra_stochastic(df: pd.DataFrame, period: int = 14, smooth_d: int = 3) -> tuple:
    """Ultra-precise Stochastic Oscillator"""
    if len(df) < period + smooth_d: 
        empty_series = pd.Series(dtype='float64', index=df.index)
        return empty_series, empty_series

    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()

    k = 100 * ((df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan))
    k.fillna(50, inplace=True)
    d = k.rolling(window=smooth_d).mean()

    return k, d

def calculate_ultra_adx(df: pd.DataFrame, period: int = 14) -> tuple:
    """Ultra-precise ADX calculation"""
    if len(df) < period * 2: 
        empty_series = pd.Series(dtype='float64', index=df.index)
        return empty_series, empty_series, empty_series

    df_adx = df.copy()
    df_adx['H-L'] = df_adx['High'] - df_adx['Low']
    df_adx['H-C'] = abs(df_adx['High'] - df_adx['Close'].shift(1))
    df_adx['L-C'] = abs(df_adx['Low'] - df_adx['Close'].shift(1))
    df_adx['TR'] = df_adx[['H-L', 'H-C', 'L-C']].max(axis=1)

    df_adx['+DM'] = np.where(
        (df_adx['High'] - df_adx['High'].shift(1)) > (df_adx['Low'].shift(1) - df_adx['Low']), 
        df_adx['High'] - df_adx['High'].shift(1), 0
    )
    df_adx['+DM'] = np.where(df_adx['+DM'] < 0, 0, df_adx['+DM'])

    df_adx['-DM'] = np.where(
        (df_adx['Low'].shift(1) - df_adx['Low']) > (df_adx['High'] - df_adx['High'].shift(1)), 
        df_adx['Low'].shift(1) - df_adx['Low'], 0
    )
    df_adx['-DM'] = np.where(df_adx['-DM'] < 0, 0, df_adx['-DM'])

    atr_val = df_adx['TR'].ewm(com=period - 1, adjust=False).mean().replace(0, np.nan)
    pdi = (df_adx['+DM'].ewm(com=period - 1, adjust=False).mean() / atr_val) * 100
    ndi = (df_adx['-DM'].ewm(com=period - 1, adjust=False).mean() / atr_val) * 100

    dx = (abs(pdi - ndi) / (pdi + ndi).replace(0, np.nan)) * 100
    adx = dx.ewm(com=period - 1, adjust=False).mean()

    return adx.fillna(20), pdi.fillna(20), ndi.fillna(20)

def calculate_ultra_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> tuple:
    """Ultra-precise Bollinger Bands"""
    if len(df) < period:
        empty_series = pd.Series(dtype='float64', index=df.index)
        return empty_series, empty_series, empty_series

    middle = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return middle, upper, lower

def calculate_ultra_roc(df: pd.DataFrame, period: int = 12) -> pd.Series:
    """Ultra-precise Rate of Change"""
    if len(df) < period + 1: 
        return pd.Series(dtype='float64', index=df.index)

    shifted_close = df['Close'].shift(period).replace(0, np.nan)
    roc = ((df['Close'] - df['Close'].shift(period)) / shifted_close) * 100

    return roc.fillna(0)

def calculate_ultra_obv(df: pd.DataFrame) -> pd.Series:
    """Ultra-precise On Balance Volume"""
    if len(df) < 2: 
        return pd.Series(dtype='float64', index=df.index)

    obv_values = []
    obv = 0

    price_change = df['Close'].diff()

    for i, (price_diff, volume) in enumerate(zip(price_change, df['Volume'])):
        if i == 0:
            obv_values.append(0)
        elif price_diff > 0:
            obv += volume
            obv_values.append(obv)
        elif price_diff < 0:
            obv -= volume
            obv_values.append(obv)
        else:
            obv_values.append(obv)

    return pd.Series(obv_values, index=df.index)

def calculate_ultra_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Ultra-precise Commodity Channel Index"""
    if len(df) < period: 
        return pd.Series(dtype='float64', index=df.index)

    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()

    mad = tp.rolling(window=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    ).replace(0, np.nan)

    cci = (tp - sma_tp) / (0.015 * mad)
    return cci.fillna(0)

def ultra_cmf(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Ultra-precise Chaikin Money Flow"""
    if len(df) < period:
        return pd.Series(index=df.index, dtype='float64')

    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = mfm.fillna(0)

    mfv = mfm * df["Volume"]

    mfv_sum = mfv.rolling(window=period, min_periods=period).sum()
    volume_sum = df["Volume"].rolling(window=period, min_periods=period).sum()

    cmf = mfv_sum / volume_sum.replace(0, np.nan)
    return cmf.fillna(0)

def ultra_relative_volume(df: pd.DataFrame, lookback: int = 50) -> pd.Series:
    """Ultra-precise Relative Volume"""
    if len(df) < lookback:
        return pd.Series(index=df.index, dtype='float64')

    vol_ma = df["Volume"].rolling(window=lookback, min_periods=lookback).mean()
    relative_vol = df["Volume"] / vol_ma.replace(0, np.nan)

    return relative_vol.fillna(1.0)

def ultra_slope(series: pd.Series, lookback: int = 10) -> float:
    """Calculate slope of series using linear regression"""
    if len(series) < lookback: 
        return 0.0

    y = series.tail(lookback).values
    x = np.arange(len(y))
    if len(y) < 2: 
        return 0.0

    try:
        coeffs = np.polyfit(x, y, 1)
        return coeffs[0]
    except:
        return 0.0

def ultra_money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Ultra-precise Money Flow Index"""
    if len(df) < period + 1:
        return pd.Series(index=df.index, dtype='float64')

    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']

    positive_flow = pd.Series(0.0, index=df.index)
    negative_flow = pd.Series(0.0, index=df.index)

    for i in range(1, len(df)):
        if typical_price.iloc[i] > typical_price.iloc[i-1]:
            positive_flow.iloc[i] = raw_money_flow.iloc[i]
        elif typical_price.iloc[i] < typical_price.iloc[i-1]:
            negative_flow.iloc[i] = raw_money_flow.iloc[i]

    positive_mf = positive_flow.rolling(window=period, min_periods=period).sum()
    negative_mf = negative_flow.rolling(window=period, min_periods=period).sum()

    money_flow_ratio = positive_mf / negative_mf.replace(0, np.nan)
    mfi = 100 - (100 / (1 + money_flow_ratio))

    return mfi.fillna(50)
# ==============================================================================
# ULTRA-ENHANCED OPTION BUYER SCANNER v4.0 - PART 3: OI ANALYSIS & PATTERNS
# ==============================================================================

def has_real_oi(df: pd.DataFrame) -> bool:
    """Check if DataFrame has real OpenInterest data"""
    if 'OpenInterest' not in df.columns:
        return False

    oi_series = df['OpenInterest']
    non_zero_count = (oi_series > 0).sum()
    total_count = len(oi_series)

    return non_zero_count >= (total_count * 0.5)

def calculate_ultra_oi_volume_ratio(df: pd.DataFrame) -> pd.Series:
    """Ultra-precise OI/Volume Ratio"""
    if not has_real_oi(df):
        return pd.Series(index=df.index, dtype='float64')

    ratio = df['OpenInterest'] / df['Volume'].replace(0, np.nan)
    return ratio.fillna(0)

def detect_ultra_oi_buildup(df: pd.DataFrame, lookback: int = 20):
    """Ultra-precise OI Buildup Detection"""
    if not has_real_oi(df) or len(df) < lookback:
        return None

    oi_ma = df['OpenInterest'].rolling(lookback).mean()
    if len(oi_ma) == 0 or pd.isna(oi_ma.iloc[-1]):
        return None

    current_oi = df['OpenInterest'].iloc[-1]
    avg_oi = oi_ma.iloc[-1]

    if avg_oi > 0 and pd.notna(current_oi):
        oi_strength = (current_oi - avg_oi) / avg_oi
        return max(min(oi_strength * 100, 100), -100)

    return None

def ultra_oi_change(df: pd.DataFrame, periods: list = [1, 5, 10]) -> dict:
    """Ultra-precise OI Change Analysis"""
    if not has_real_oi(df):
        return {p: pd.Series(index=df.index, dtype='float64') for p in periods}

    oi_changes = {}
    oi_series = df['OpenInterest']

    for period in periods:
        if len(df) > period:
            pct_change = oi_series.pct_change(periods=period) * 100
            oi_changes[period] = pct_change.fillna(0)
        else:
            oi_changes[period] = pd.Series(index=df.index, dtype='float64')

    return oi_changes

def ultra_oi_volume_divergence(df: pd.DataFrame) -> pd.Series:
    """Ultra-precise OI-Volume Divergence Detection"""
    if not has_real_oi(df) or len(df) < 10:
        return pd.Series(index=df.index, dtype='float64')

    oi_norm = (df['OpenInterest'] - df['OpenInterest'].mean()) / df['OpenInterest'].std()
    vol_norm = (df['Volume'] - df['Volume'].mean()) / df['Volume'].std()

    divergence = oi_norm.rolling(window=10, min_periods=5).corr(vol_norm.rolling(window=10, min_periods=5))

    divergence_signal = 1 - (2 * np.abs(divergence - 1))

    return divergence_signal.fillna(0)

def ultra_volume_oi_sync_analysis(df: pd.DataFrame):
    """Ultra-precise Volume-OI Synchronization"""
    if len(df) < 10 or not has_real_oi(df):
        return None

    vol_change = df['Volume'].pct_change(5).fillna(0)
    oi_change = df['OpenInterest'].pct_change(5).fillna(0)
    sync_score = vol_change.iloc[-1] + oi_change.iloc[-1]
    return min(max(sync_score * 50, -100), 100)

def ultra_option_buyer_momentum(df: pd.DataFrame):
    """Ultra-precise Option Buyer Momentum Detection"""
    if len(df) < 20:
        return None

    price_mom = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
    vol_surge_val = ultra_volume_surge(df, lookback=20).iloc[-1] if len(df) > 20 else 0
    oi_buildup = detect_ultra_oi_buildup(df, lookback=20)

    if oi_buildup is None:
        return None

    combined_score = (price_mom * 0.4) + (vol_surge_val * 0.3) + (oi_buildup * 0.3)
    return min(max(combined_score, -100), 100)

def detect_breakout_patterns(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    """Ultra-precise Breakout Pattern Detection"""
    if len(df) < lookback * 2:
        return pd.Series(index=df.index, dtype='float64')

    high_resistance = df['High'].rolling(window=lookback, min_periods=lookback).max()
    low_support = df['Low'].rolling(window=lookback, min_periods=lookback).min()

    current_close = df['Close']

    breakout_signals = []

    for i in range(len(df)):
        if i < lookback:
            breakout_signals.append(0)
            continue

        # Resistance breakout
        if current_close.iloc[i] > high_resistance.iloc[i-1]:
            volume_confirmation = df['Volume'].iloc[i] > df['Volume'].iloc[i-5:i].mean() * 1.5
            breakout_signals.append(2.0 if volume_confirmation else 1.0)

        # Support breakdown
        elif current_close.iloc[i] < low_support.iloc[i-1]:
            volume_confirmation = df['Volume'].iloc[i] > df['Volume'].iloc[i-5:i].mean() * 1.5
            breakout_signals.append(-2.0 if volume_confirmation else -1.0)

        else:
            breakout_signals.append(0)

    return pd.Series(breakout_signals, index=df.index)

def detect_candlestick_patterns(df: pd.DataFrame) -> pd.Series:
    """Ultra-precise Candlestick Pattern Detection"""
    if len(df) < 5:
        return pd.Series(index=df.index, dtype='float64')

    pattern_signals = []

    for i in range(len(df)):
        if i < 2:
            pattern_signals.append(0)
            continue

        curr = df.iloc[i]
        prev = df.iloc[i-1]
        prev2 = df.iloc[i-2] if i >= 2 else None

        signal = 0

        # Bullish patterns
        body_size = abs(curr['Close'] - curr['Open'])
        lower_shadow = min(curr['Open'], curr['Close']) - curr['Low']
        upper_shadow = curr['High'] - max(curr['Open'], curr['Close'])

        # Hammer
        if lower_shadow > 2 * body_size and upper_shadow < body_size:
            signal += 1.5

        # Bullish Engulfing
        if (curr['Close'] > curr['Open'] and prev['Close'] < prev['Open'] and
            curr['Open'] < prev['Close'] and curr['Close'] > prev['Open']):
            signal += 2.0

        # Morning Star (3-candle pattern)
        if (prev2 is not None and 
            prev2['Close'] < prev2['Open'] and
            abs(prev['Close'] - prev['Open']) < abs(prev2['Close'] - prev2['Open']) * 0.3 and
            curr['Close'] > curr['Open'] and curr['Close'] > prev2['Close']):
            signal += 2.5

        # Bearish patterns
        # Shooting Star
        if upper_shadow > 2 * body_size and lower_shadow < body_size:
            signal -= 1.5

        # Bearish Engulfing
        if (curr['Close'] < curr['Open'] and prev['Close'] > prev['Open'] and
            curr['Open'] > prev['Close'] and curr['Close'] < prev['Open']):
            signal -= 2.0

        # Evening Star (3-candle pattern)
        if (prev2 is not None and
            prev2['Close'] > prev2['Open'] and
            abs(prev['Close'] - prev['Open']) < abs(prev2['Open'] - prev2['Close']) * 0.3 and
            curr['Close'] < curr['Open'] and curr['Close'] < prev2['Close']):
            signal -= 2.5

        pattern_signals.append(signal)

    return pd.Series(pattern_signals, index=df.index)

def detect_support_resistance_levels(df: pd.DataFrame, lookback: int = 50) -> tuple:
    """Ultra-precise Support/Resistance Level Detection"""
    if len(df) < lookback:
        empty_series = pd.Series(index=df.index, dtype='float64')
        return empty_series, empty_series

    resistance_levels = []
    support_levels = []

    for i in range(len(df)):
        if i < lookback:
            resistance_levels.append(df['High'].iloc[:i+1].max() if i > 0 else df['High'].iloc[i])
            support_levels.append(df['Low'].iloc[:i+1].min() if i > 0 else df['Low'].iloc[i])
        else:
            window_data = df.iloc[max(0, i-lookback):i+1]
            resistance_levels.append(window_data['High'].quantile(0.95))
            support_levels.append(window_data['Low'].quantile(0.05))

    return pd.Series(resistance_levels, index=df.index), pd.Series(support_levels, index=df.index)

def calculate_price_strength(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate price strength relative to recent range"""
    if len(df) < period:
        return pd.Series(index=df.index, dtype='float64')

    high_max = df['High'].rolling(window=period, min_periods=period).max()
    low_min = df['Low'].rolling(window=period, min_periods=period).min()

    price_strength = ((df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)) * 100

    return price_strength.fillna(50)

def enhanced_institutional_flow_analysis(tf_data: dict) -> str:
    """Enhanced institutional flow analysis"""
    frames = [tf_data.get(t) for t in (5, 15, 30) if tf_data.get(t) is not None and len(tf_data.get(t)) >= 60]
    if not frames: 
        return "Unknown"

    votes = 0
    for df in frames:
        cmf_series = ultra_cmf(df, 20)
        rv_series = ultra_relative_volume(df, 50)

        if cmf_series.empty or rv_series.empty: 
            continue

        cmf_last = cmf_series.iloc[-1]
        rv_last = rv_series.iloc[-1]

        if cmf_last > 0.05 and rv_last > 1.2: 
            votes += 1
        elif cmf_last < -0.05 and rv_last > 1.2: 
            votes -= 1

    if votes >= 2: 
        return "Institutional Accumulation"
    elif votes <= -2: 
        return "Institutional Distribution"
    else: 
        return "Mixed/Neutral"
# ==============================================================================
# ULTRA-ENHANCED OPTION BUYER SCANNER v4.0 - PART 4: SCORING ENGINE
# ==============================================================================

def ultra_normalize_score(value: float, bullish_range: tuple, bearish_range: tuple, 
                         score_range: tuple = (-2.0, 2.0)) -> float:
    """Ultra-precise score normalization with enhanced sensitivity"""
    if pd.isna(value):
        return 0.0

    low_score, high_score = score_range
    bull_min, bull_max = bullish_range
    bear_max, bear_min = bearish_range

    # Bullish territory
    if value >= bull_max:
        return high_score
    elif value > bull_min:
        ratio = (value - bull_min) / (bull_max - bull_min)
        return high_score * ratio

    # Bearish territory  
    elif value <= bear_min:
        return low_score
    elif value < bear_max:
        ratio = (bear_max - value) / (bear_max - bear_min)
        return low_score * ratio

    # Neutral zone - apply small gradient
    neutral_range = bull_min - bear_max
    if neutral_range > 0:
        neutral_position = (value - bear_max) / neutral_range
        return (neutral_position - 0.5) * 0.2

    return 0.0

def calculate_ultra_indicator_scores(df: pd.DataFrame) -> dict:
    """Ultra-Enhanced Indicator Score Calculation"""
    scores = defaultdict(float)

    if df is None or len(df) < UltraConfig.MIN_BARS_REQUIRED:
        logger.warning(f"Insufficient data: {len(df) if df is not None else 0} bars")
        return scores

    try:
        # =============================================================================
        # TREND INDICATORS
        # =============================================================================

        # ADX with Directional Indicators
        adx, plus_di, minus_di = calculate_ultra_adx(df)
        if not adx.empty and len(adx) > 5:
            current_adx = adx.iloc[-1]
            current_plus_di = plus_di.iloc[-1]
            current_minus_di = minus_di.iloc[-1]

            if current_adx > 25:
                if current_plus_di > current_minus_di:
                    scores['ADX'] = ultra_normalize_score(current_adx, (25, 50), (0, 0))
                else:
                    scores['ADX'] = ultra_normalize_score(current_adx, (0, 0), (25, 50))

        # Enhanced EMA Analysis
        ema_fast = ultra_ema(df['Close'], 12)
        ema_slow = ultra_ema(df['Close'], 26)
        if not ema_fast.empty and not ema_slow.empty:
            ema_ratio = ema_fast.iloc[-1] / ema_slow.iloc[-1] if ema_slow.iloc[-1] != 0 else 1
            scores['EMA'] = ultra_normalize_score(ema_ratio, (1.005, 1.03), (0.995, 0.97))

        # Enhanced VWAP Analysis
        vwap_line = ultra_vwap(df)
        if not vwap_line.empty:
            vwap_ratio = df['Close'].iloc[-1] / vwap_line.iloc[-1] if vwap_line.iloc[-1] != 0 else 1
            scores['VWAP'] = ultra_normalize_score(vwap_ratio, (1.003, 1.02), (0.997, 0.98))

        # MACD Enhanced
        macd_line, signal_line, histogram = calculate_ultra_macd(df['Close'])
        if not macd_line.empty and not signal_line.empty:
            macd_current = macd_line.iloc[-1]
            signal_current = signal_line.iloc[-1]
            histogram_current = histogram.iloc[-1]

            macd_cross = 2.0 if macd_current > signal_current else -2.0
            macd_momentum = ultra_normalize_score(histogram_current, (0.1, 1.0), (-0.1, -1.0))
            scores['MACD_Trend'] = (macd_cross + macd_momentum) / 2

        # Moving Average Slope (Enhanced)
        if len(ema_fast) >= 10:
            recent_slope = (ema_fast.iloc[-1] - ema_fast.iloc[-5]) / ema_fast.iloc[-5] if ema_fast.iloc[-5] != 0 else 0
            slope_normalized = recent_slope * 1000
            scores['MA_Slope'] = ultra_normalize_score(slope_normalized, (0.5, 2.0), (-0.5, -2.0), (-3.0, 3.0))

        # =============================================================================
        # MOMENTUM INDICATORS  
        # =============================================================================

        # Enhanced RSI
        rsi = calculate_ultra_rsi(df)
        if not rsi.empty:
            rsi_current = rsi.iloc[-1]
            scores['RSI'] = ultra_normalize_score(rsi_current, (55, 80), (45, 20))

        # Enhanced Stochastic
        stoch_k, stoch_d = calculate_ultra_stochastic(df)
        if not stoch_k.empty and not stoch_d.empty:
            k_current = stoch_k.iloc[-1]
            d_current = stoch_d.iloc[-1]

            if k_current > d_current:
                scores['Stochastic'] = ultra_normalize_score(k_current, (25, 75), (100, 100))
            else:
                scores['Stochastic'] = ultra_normalize_score(k_current, (0, 0), (75, 25))

        # Enhanced CCI
        cci = calculate_ultra_cci(df)
        if not cci.empty:
            cci_current = cci.iloc[-1]
            scores['CCI'] = ultra_normalize_score(cci_current, (50, 150), (-50, -150))

        # Enhanced ROC
        roc = calculate_ultra_roc(df)
        if not roc.empty:
            roc_current = roc.iloc[-1]
            scores['ROC'] = ultra_normalize_score(roc_current, (1.0, 3.0), (-1.0, -3.0))

        # Williams %R
        williams_r = ultra_williams_r(df)
        if not williams_r.empty:
            wr_current = williams_r.iloc[-1]
            scores['WilliamsR'] = ultra_normalize_score(wr_current, (-80, -20), (-20, -80))

        # Money Flow Index
        mfi = ultra_money_flow_index(df)
        if not mfi.empty:
            mfi_current = mfi.iloc[-1]
            scores['MFI'] = ultra_normalize_score(mfi_current, (55, 80), (45, 20))

        # =============================================================================
        # VOLUME INDICATORS
        # =============================================================================

        # Enhanced Volume Surge
        vol_surge = ultra_volume_surge(df)
        if not vol_surge.empty and len(vol_surge) > 1:
            surge_current = vol_surge.iloc[-1]
            price_direction = 1 if df['Close'].iloc[-1] > df['Close'].iloc[-2] else -1

            if price_direction > 0:
                scores['VolumeSurge'] = ultra_normalize_score(surge_current, (1.0, 3.0), (0, 0))
            else:
                scores['VolumeSurge'] = ultra_normalize_score(surge_current, (0, 0), (-1.0, -3.0))

        # Enhanced OBV
        obv = calculate_ultra_obv(df)
        if len(obv) >= 10:
            obv_slope = (obv.iloc[-1] - obv.iloc[-5]) / abs(obv.iloc[-5]) if obv.iloc[-5] != 0 else 0
            scores['OBV'] = ultra_normalize_score(obv_slope, (0.05, 0.2), (-0.05, -0.2))

        # Enhanced CMF
        cmf = ultra_cmf(df)
        if not cmf.empty:
            cmf_current = cmf.iloc[-1]
            scores['CMF'] = ultra_normalize_score(cmf_current, (0.05, 0.2), (-0.05, -0.2))

        # Relative Volume
        rel_vol = ultra_relative_volume(df)
        if not rel_vol.empty:
            rv_current = rel_vol.iloc[-1]
            scores['RelVol'] = ultra_normalize_score(rv_current, (1.2, 2.5), (0.8, 0.5))

        # =============================================================================
        # VOLATILITY INDICATORS
        # =============================================================================

        # Enhanced ATR Analysis
        atr = ultra_atr(df)
        if len(atr) >= 25:
            atr_current = atr.iloc[-1]
            atr_ma = atr.rolling(20).mean().iloc[-1]

            if atr_ma > 0:
                atr_expansion = (atr_current - atr_ma) / atr_ma
                price_direction = 1 if df['Close'].iloc[-1] > df['Close'].iloc[-5] else -1

                if atr_expansion > 0.1:
                    scores['VolatilityExpansion'] = 2.5 * price_direction
                elif atr_expansion < -0.1:
                    scores['VolatilityExpansion'] = -1.5 * price_direction

        # Enhanced Bollinger Bands
        bb_middle, bb_upper, bb_lower = calculate_ultra_bollinger_bands(df['Close'])
        if not bb_upper.empty and not bb_lower.empty:
            close_current = df['Close'].iloc[-1]
            upper_current = bb_upper.iloc[-1]
            lower_current = bb_lower.iloc[-1]
            middle_current = bb_middle.iloc[-1]

            if close_current > upper_current:
                scores['Bollinger'] = 2.5
            elif close_current < lower_current:
                scores['Bollinger'] = -2.5
            else:
                band_position = (close_current - middle_current) / (upper_current - middle_current)
                scores['Bollinger'] = ultra_normalize_score(band_position, (0.5, 1.0), (-0.5, -1.0))

        # =============================================================================
        # OI INDICATORS (REAL DATA ONLY)
        # =============================================================================

        if has_real_oi(df):
            # OI Change Analysis
            oi_changes = ultra_oi_change(df, [1, 5, 10])
            if 5 in oi_changes and not oi_changes[5].empty:
                oi_change_5 = oi_changes[5].iloc[-1]
                scores['OIChange'] = ultra_normalize_score(oi_change_5, (5, 20), (-5, -20))

            # OI-Volume Divergence
            oi_vol_divergence = ultra_oi_volume_divergence(df)
            if not oi_vol_divergence.empty:
                divergence_current = oi_vol_divergence.iloc[-1]
                scores['VolumeOISync'] = ultra_normalize_score(divergence_current, (0.5, 1.0), (-0.5, -1.0))

            # OI Buildup Strength
            oi_buildup = detect_ultra_oi_buildup(df)
            if oi_buildup is not None:
                scores['OI_Buildup'] = ultra_normalize_score(oi_buildup, (10, 40), (-10, -40))

            # Option Buyer Momentum (Enhanced)
            option_momentum = ultra_option_buyer_momentum(df)
            if option_momentum is not None:
                scores['OptionBuyerMomentum'] = ultra_normalize_score(option_momentum, (15, 50), (-15, -50), (-4.0, 4.0))

        # =============================================================================
        # PRICE ACTION INDICATORS
        # =============================================================================

        # Breakout Pattern Detection
        breakout_signals = detect_breakout_patterns(df)
        if not breakout_signals.empty:
            breakout_current = breakout_signals.iloc[-1]
            scores['Breakout_Strength'] = breakout_current

        # Candlestick Patterns
        candlestick_signals = detect_candlestick_patterns(df)
        if not candlestick_signals.empty:
            candlestick_current = candlestick_signals.iloc[-1]
            scores['Candlestick_Patterns'] = candlestick_current

        # Support/Resistance Analysis
        if len(df) >= 50:
            resistance_levels, support_levels = detect_support_resistance_levels(df)
            current_price = df['Close'].iloc[-1]

            if not resistance_levels.empty and not support_levels.empty:
                recent_resistance = resistance_levels.iloc[-1]
                recent_support = support_levels.iloc[-1]

                resistance_distance = (current_price - recent_resistance) / recent_resistance
                support_distance = (current_price - recent_support) / recent_support

                if resistance_distance > -0.02:
                    scores['Support_Resistance'] = ultra_normalize_score(resistance_distance, (0, 0.05), (-0.05, -0.1))
                elif support_distance < 0.02:
                    scores['Support_Resistance'] = ultra_normalize_score(support_distance, (-0.05, 0.05), (0, -0.05))

    except Exception as e:
        logger.error(f"Error in indicator calculation: {e}")
        logger.error(traceback.format_exc())

    return scores

def ultra_analyze_signals(timeframe_data: dict, market_regime: str = 'neutral') -> tuple:
    """Ultra-Enhanced Signal Analysis Engine"""

    total_weighted_score = 0.0
    total_weight = 0.0
    group_scores = defaultdict(float)
    group_weights = defaultdict(float)

    # Process each timeframe
    for tf_minutes, df in timeframe_data.items():
        if df is None or len(df) < UltraConfig.MIN_BARS_REQUIRED:
            continue

        # Calculate indicator scores for this timeframe
        indicator_scores = calculate_ultra_indicator_scores(df)

        # Get timeframe weight
        tf_weight = TIMEFRAME_WEIGHTS.get(tf_minutes, 1.0)

        # Process each indicator group
        for group_name, group_weight in UltraConfig.GROUP_WEIGHTS.items():
            group_score = 0.0
            group_indicator_count = 0

            # Map indicators to groups
            group_indicators = {
                'Trend': ['ADX', 'EMA', 'VWAP', 'MACD_Trend', 'MA_Slope'],
                'Momentum': ['RSI', 'Stochastic', 'CCI', 'ROC', 'WilliamsR', 'MFI'],
                'Volume': ['VolumeSurge', 'OBV', 'CMF', 'RelVol'],
                'Volatility': ['VolatilityExpansion', 'Bollinger'],
                'OI': ['OptionBuyerMomentum', 'OIChange', 'VolumeOISync', 'OI_Buildup'],
            }

            # Calculate group score
            for indicator_name in group_indicators.get(group_name, []):
                if indicator_name in indicator_scores:
                    indicator_weight = UltraConfig.INDICATOR_WEIGHTS.get(indicator_name, 1.0)
                    weighted_score = indicator_scores[indicator_name] * indicator_weight
                    group_score += weighted_score
                    group_indicator_count += 1

            # Apply group score if we have indicators
            if group_indicator_count > 0:
                normalized_group_score = (group_score / group_indicator_count) * group_weight * tf_weight
                group_scores[group_name] += normalized_group_score
                group_weights[group_name] += group_weight * tf_weight

    # Calculate final score
    final_score = 0.0
    max_possible_score = 0.0

    for group_name, score in group_scores.items():
        final_score += score
        max_possible_score += group_weights[group_name]

    if max_possible_score == 0:
        return 'Neutral', 0.0, {}

    # Normalize to percentage
    normalized_score = (final_score / max_possible_score) * 100

    # Apply market regime multipliers
    regime_multiplier = 1.0
    if normalized_score > 0:
        if market_regime == 'bullish':
            regime_multiplier = UltraConfig.REGIME_MULTIPLIERS['bullish_in_bull_market']
        elif market_regime == 'bearish':
            regime_multiplier = UltraConfig.REGIME_MULTIPLIERS['bullish_in_bear_market']
    else:
        if market_regime == 'bearish':
            regime_multiplier = UltraConfig.REGIME_MULTIPLIERS['bearish_in_bear_market']
        elif market_regime == 'bullish':
            regime_multiplier = UltraConfig.REGIME_MULTIPLIERS['bearish_in_bull_market']

    normalized_score *= regime_multiplier

    # Determine signal strength
    signal = 'Neutral'
    abs_score = abs(normalized_score)

    if normalized_score > 0:
        if abs_score >= UltraConfig.SIGNAL_THRESHOLDS['Ultra Strong Buy']:
            signal = 'Ultra Strong Buy'
        elif abs_score >= UltraConfig.SIGNAL_THRESHOLDS['Very Strong Buy']:
            signal = 'Very Strong Buy'
        elif abs_score >= UltraConfig.SIGNAL_THRESHOLDS['Strong Buy']:
            signal = 'Strong Buy'
        elif abs_score >= UltraConfig.SIGNAL_THRESHOLDS['Buy Signal']:
            signal = 'Buy Signal'
        elif abs_score >= UltraConfig.SIGNAL_THRESHOLDS['Weak Buy']:
            signal = 'Weak Buy'
    else:
        if abs_score >= UltraConfig.SIGNAL_THRESHOLDS['Ultra Strong Buy']:
            signal = 'Ultra Strong Sell'
        elif abs_score >= UltraConfig.SIGNAL_THRESHOLDS['Very Strong Buy']:
            signal = 'Very Strong Sell'
        elif abs_score >= UltraConfig.SIGNAL_THRESHOLDS['Strong Buy']:
            signal = 'Strong Sell'
        elif abs_score >= UltraConfig.SIGNAL_THRESHOLDS['Buy Signal']:
            signal = 'Sell Signal'
        elif abs_score >= UltraConfig.SIGNAL_THRESHOLDS['Weak Buy']:
            signal = 'Weak Sell'

    # Calculate sub-scores for display
    final_sub_scores = {}
    for group_name in group_scores:
        if group_weights[group_name] > 0:
            final_sub_scores[group_name] = (group_scores[group_name] / group_weights[group_name]) * 15

    return signal, normalized_score, final_sub_scores
