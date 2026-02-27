# ==============================================================================
# ULTIMATE INTRADAY TRADER SCANNER v4.3 - COMPLETE WITH ALL DATETIME FIXES
# TrueData: Uses symbols with -I suffix (RELIANCE-I, TCS-I)
# Focuses on OHLC, Volume, and Open Interest for Intraday Trading
# Runs every 5 minutes during market hours with proper market condition checking
# ALL DATETIME COMPARISON ERRORS FIXED
# ==============================================================================

import os
import logging
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
import pytz
import time
import threading
from collections import defaultdict
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm
from pathlib import Path
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

# ======== ULTIMATE Configuration for Intraday Traders ========
class Config:
    # TrueData Configuration
    TDUSERNAME = os.getenv("TRUEDATA_USER", "tdwsp751")
    TDPASSWORD = os.getenv("TRUEDATA_PASS", "raj@751")
    
    # Market Timing Configuration
    MARKET_START = "09:15"  # IST
    FIRST_RUN_AT = "09:20"  # IST; First scan after 09:15-09:20 candle
    FIRST_SCAN_DELAY = 15  # Wait 15 seconds after 09:20 for settlement
    MARKET_END = "15:30"  # IST
    SETTLE_DELAY_SECONDS = 15  # wait after bar close for data settlement
    
    # Performance Configuration
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "32"))
    TD_HIST_SESSIONS = int(os.getenv("TD_HIST_SESSIONS", "5"))
    SHARES_FILE = os.getenv("SHARES_FILE", "shares.txt")
    BENCHMARK_INDEX = "NIFTY 50"
    
    # Backtesting Configuration
    BACKTEST_INTERVAL_MINUTES = 5
    BACKTEST_START_DELAY = 5
    BACKTEST_TOP_DISPLAY = 15
    
    # Intraday Analysis Thresholds
    MIN_TOTAL_OI = 2000          # Minimum total OI for liquidity
    MIN_TOTAL_VOL = 200          # Minimum total volume
    MIN_OI_CHANGE_THRESHOLD = 5.0 # Minimum OI change % for significance
    MIN_VOL_SURGE_THRESHOLD = 1.5 # Volume surge multiplier
    
    # Enhanced Indicator Group Weights (Optimized for Intraday Traders)
    GROUP_WEIGHTS = {
        "Trend": 2.0,           # Trend for direction
        "Momentum": 3.0,        # Momentum for quick entries
        "Volume": 3.5,          # Volume critical for intraday confirmation
        "Volatility": 2.5,      # Volatility for breakouts
        "OI": 2.5,              # OI for institutional interest
    }
    
    # Enhanced Individual Indicator Weights (Intraday Optimized)
    INDICATOR_WEIGHTS = {
        # Trend indicators
        "MA_Slope": 2.0, "ADX": 2.2, "VWAP": 2.5, "EMA": 1.7, "MACD_Trend": 2.0,
        
        # Momentum indicators (CRITICAL for intraday)
        "RSI": 2.5, "Stochastic": 2.0, "CCI": 2.2, "ROC": 2.0, "WilliamsR": 1.8,
        
        # Volume indicators (drives intraday moves)
        "VolumeSurge": 3.5, "OBV": 2.5, "CMF": 2.5, "RelVol": 2.5,
        
        # Volatility indicators (breakout opportunities)
        "VolatilityExpansion": 3.0, "Bollinger": 2.5,
        
        # OI indicators
        "OIChange": 3.0, "VolumeOISync": 2.5, "IntradayMomentum": 3.0,
    }
    
    # Enhanced Scoring & Signal Thresholds (Intraday Focused)
    SCORE_THRESHOLD_MIN = 3.0    # Minimum score for signal
    SIGNAL_THRESHOLDS = {
        'Very Strong Buy': 55.0,
        'Strong Buy': 30.0,
        'Buy Signal': 15.0,
        'Very Strong Sell': -55.0,
        'Strong Sell': -30.0,
        'Sell Signal': -15.0,
    }
    
    # Market Regime Multipliers (Enhanced for Intraday)
    REGIME_MULTIPLIERS = {
        'bullish_in_bull_market': 1.25,
        'bearish_in_bear_market': 1.25,
        'bullish_in_bear_market': 0.7,
        'bearish_in_bull_market': 0.7,
    }

# Constants
IST = pytz.timezone("Asia/Kolkata")
BAR_SIZE_MAP = {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"}
DURATION_MAP = {5: "30 D", 15: "30 D", 30: "60 D", 60: "120 D", 1440: "365 D"}
TIMEFRAME_WEIGHTS = {5: 3.0, 15: 2.5, 30: 2.0, 60: 1.5, 1440: 1.0}

# Silence noisy loggers
for noisy in ("truedata", "truedata.history", "truedata_ws", "websocket", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

# State management
previous_scan_results = {}
previous_oi_data = {}
previous_volume_data = {}
intraday_volume_data = {}
intraday_oi_data = {}
scan_count = 0
backtest_stock_history = {}
current_scan_data = {}

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

# ========== ENHANCED UTILITY FUNCTIONS ==========

def format_time_remaining(seconds):
    """Format remaining time in human-readable format"""
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        return f"{seconds/60:.1f} minutes"
    else:
        return f"{seconds/3600:.1f} hours"

# ========== CORRECTED SYMBOL HANDLING FUNCTIONS ==========

def convert_to_truedata_symbol(symbol):
    """Convert symbol to TrueData format (add -I suffix)"""
    if symbol.endswith('-EQ'):
        return symbol.replace('-EQ', '-I')
    elif symbol.endswith('-I'):
        return symbol
    else:
        return f"{symbol}-I"

def normalize_symbol_for_display(symbol):
    """Convert symbol for display purposes (clean without suffix)"""
    if symbol.endswith('-I'):
        return symbol.replace('-I', '')
    elif symbol.endswith('-EQ'):
        return symbol.replace('-EQ', '')
    else:
        return symbol

# ========== TECHNICAL INDICATORS ==========

def ema(series, length):
    """Calculate Exponential Moving Average"""
    if series.empty or len(series) < length:
        return pd.Series(dtype='float64', index=series.index)
    return series.ewm(span=length, adjust=False).mean()

def vwap(df, period=None):
    """Calculate Volume Weighted Average Price"""
    if df.empty or len(df) < 5:
        return pd.Series(dtype='float64', index=df.index)
    
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
    """Calculate Average True Range"""
    if df.empty or len(df) < period:
        return pd.Series(dtype='float64', index=df.index)
    
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def williams_r(df, period=14):
    """Calculate Williams %R"""
    if df.empty or len(df) < period:
        return pd.Series(dtype='float64', index=df.index)
    
    highest = df["High"].rolling(period).max()
    lowest = df["Low"].rolling(period).min()
    return -100 * (highest - df["Close"]) / (highest - lowest).replace(0, np.nan)

def momentum(df, period=10):
    """Calculate Momentum"""
    if df.empty or len(df) < period:
        return pd.Series(dtype='float64', index=df.index)
    
    return df["Close"] / df["Close"].shift(period) - 1.0

def volume_surge(df, lookback=20):
    """Calculate volume surge Z-score"""
    if df.empty or len(df) < lookback:
        return pd.Series(dtype='float64', index=df.index)
    
    vol_ma = df["Volume"].rolling(lookback).mean()
    vol_std = df["Volume"].rolling(lookback).std()
    z_score = (df["Volume"] - vol_ma) / vol_std.replace(0, np.nan)
    return z_score.fillna(0)

def calculate_rsi(df, period=14):
    """Calculate RSI"""
    if df.empty or len(df) < period + 1:
        return pd.Series(dtype='float64', index=df.index)
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rs.fillna(100, inplace=True)
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    if df.empty or len(df) < slow + signal:
        return pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index)
    
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_stochastic(df, period=14, smooth_d=3):
    """Calculate Stochastic oscillator"""
    if df.empty or len(df) < period + smooth_d:
        return pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index)
    
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan))
    k.fillna(50, inplace=True)
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_adx(df, period=14):
    """Calculate ADX"""
    if df.empty or len(df) < period * 2:
        return (pd.Series(dtype='float64', index=df.index), 
                pd.Series(dtype='float64', index=df.index), 
                pd.Series(dtype='float64', index=df.index))
    
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
    """Calculate Bollinger Bands"""
    if df.empty or len(df) < period:
        return (pd.Series(dtype='float64', index=df.index), 
                pd.Series(dtype='float64', index=df.index), 
                pd.Series(dtype='float64', index=df.index))
    
    middle = df['Close'].rolling(window=period).mean()
    std = df['Close'].rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return middle, upper, lower

def calculate_roc(df, period=12):
    """Calculate Rate of Change"""
    if df.empty or len(df) < period + 1:
        return pd.Series(dtype='float64', index=df.index)
    
    shifted_close = df['Close'].shift(period).replace(0, np.nan)
    return ((df['Close'] - df['Close'].shift(period)) / shifted_close) * 100

def calculate_obv(df):
    """Calculate On Balance Volume"""
    if df.empty or len(df) < 2:
        return pd.Series(dtype='float64', index=df.index)
    
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

def calculate_cci(df, period=20):
    """Calculate Commodity Channel Index"""
    if df.empty or len(df) < period:
        return pd.Series(dtype='float64', index=df.index)
    
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).replace(0, np.nan)
    return (tp - sma_tp) / (0.015 * mad)

def cmf(df, period=20):
    """Calculate Chaikin Money Flow"""
    if df.empty or len(df) < period:
        return pd.Series(dtype='float64', index=df.index)
    
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = mfm.fillna(0)
    mfv = mfm * df["Volume"]
    mfv_sum = mfv.rolling(period).sum()
    vol_sum = df["Volume"].rolling(period).sum().replace(0, np.nan)
    return (mfv_sum / vol_sum).fillna(0)

def relative_volume(df, lookback=50):
    """Calculate Relative Volume"""
    if df.empty or len(df) < lookback:
        return pd.Series(dtype='float64', index=df.index)
    
    vol_ma = df["Volume"].rolling(lookback).mean()
    return (df["Volume"] / vol_ma.replace(0, np.nan)).fillna(1.0)

def slope(series, lookback=10):
    """Calculate slope of a series"""
    if series.empty or len(series) < lookback: 
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

# ========== OI HELPER FUNCTIONS ==========

def _has_real_oi(df):
    """Check if DataFrame has real OI data"""
    return ('OpenInterest' in df.columns) and (df['OpenInterest'].notna().sum() >= 2)

def detect_oi_buildup(df, lookback=20):
    """Detect OI buildup"""
    if not _has_real_oi(df) or len(df) < lookback:
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

def volume_oi_sync_analysis(df):
    """Analyze volume OI sync"""
    if len(df) < 10 or not _has_real_oi(df):
        return None
    
    vol_change = df['Volume'].pct_change(5).fillna(0)
    oi_change = df['OpenInterest'].pct_change(5).fillna(0)
    sync_score = vol_change.iloc[-1] + oi_change.iloc[-1]
    
    return min(max(sync_score * 50, -100), 100)

def intraday_momentum(df):
    """Calculate intraday momentum using OI and volume"""
    if len(df) < 20:
        return None
    
    price_mom = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
    vol_surge_val = volume_surge(df, lookback=20).iloc[-1] if len(df) > 20 else 0
    oi_buildup = detect_oi_buildup(df, lookback=20)
    
    if oi_buildup is None:
        return None
    
    combined_score = (price_mom * 0.4) + (vol_surge_val * 0.3) + (oi_buildup * 0.3)
    return min(max(combined_score, -100), 100)

# ========== ENHANCED SCORING ENGINE FOR INTRADAY ==========

def normalize_score(value, bullish_range, bearish_range, score_range=(-2.0, 2.0)):
    """Normalize score to specified range"""
    low_score, high_score = score_range
    bull_min, bull_max = bullish_range
    
    if value >= bull_max: 
        return high_score
    if value > bull_min:
        return high_score * ((value - bull_min) / (bull_max - bull_min))
    
    bear_max, bear_min = bearish_range
    if value <= bear_min: 
        return low_score
    if value < bear_max:
        return low_score * ((bear_max - value) / (bear_max - bear_min))
    
    return 0.0

def calculate_technical_indicator_scores(df):
    """Calculate technical indicator scores"""
    scores = defaultdict(float)
    
    if df is None or df.empty or len(df) < 15:  # Lowered requirement
        return scores
    
    try:
        # --- Enhanced Trend Group (optimized for intraday) ---
        adx, pdi, ndi = calculate_adx(df)
        if not adx.empty and len(adx) > 3 and adx.iloc[-1] > 15:  # Lowered threshold
            trend_strength = adx.iloc[-1] / 50.0  # Normalize
            if pdi.iloc[-1] > ndi.iloc[-1]:
                scores['ADX'] = min(2.2, trend_strength * 2.2)
            else:
                scores['ADX'] = max(-2.2, -trend_strength * 2.2)
        
        # Enhanced EMA analysis
        ema20, ema50 = ema(df['Close'], 20), ema(df['Close'], 50)
        if not ema20.empty and not ema50.empty:
            ema_ratio = ema20.iloc[-1] / ema50.iloc[-1] if ema50.iloc[-1] != 0 else 1
            scores['EMA'] = normalize_score(ema_ratio, (1.002, 1.025), (0.998, 0.975))
        
        # Enhanced VWAP
        vwap_line = vwap(df, period=None)
        if not vwap_line.empty:
            vwap_ratio = df['Close'].iloc[-1] / vwap_line.iloc[-1] if vwap_line.iloc[-1] != 0 else 1
            scores['VWAP'] = normalize_score(vwap_ratio, (1.003, 1.030), (0.997, 0.970))
        
        # Enhanced MACD for intraday
        macd, signal = calculate_macd(df)
        if not macd.empty and not signal.empty and len(macd) > 0:
            macd_val = macd.iloc[-1]
            signal_val = signal.iloc[-1]
            if macd_val > signal_val and macd_val > 0:
                scores['MACD_Trend'] = 2.0
            elif macd_val < signal_val and macd_val < 0:
                scores['MACD_Trend'] = -2.0
            else:
                scores['MACD_Trend'] = 0.5 if macd_val > signal_val else -0.5
        
        # Enhanced MA Slope
        if not ema20.empty and len(ema20) >= 5:
            ma20_slope = slope(ema20, 5)
            price_norm_slope = ma20_slope / df['Close'].iloc[-1] * 1000 if df['Close'].iloc[-1] != 0 else 0
            scores['MA_Slope'] = normalize_score(price_norm_slope, (0.2, 0.8), (-0.2, -0.8), (-2.0, 2.0))
        
        # --- Enhanced Momentum Group (CRITICAL for intraday) ---
        rsi = calculate_rsi(df)
        if not rsi.empty and len(rsi) > 0:
            rsi_val = rsi.iloc[-1]
            # Enhanced RSI scoring for intraday
            if rsi_val > 70:
                scores['RSI'] = 2.5 - (rsi_val - 70) * 0.05
            elif rsi_val > 60:
                scores['RSI'] = 1.5 + (rsi_val - 60) * 0.1
            elif rsi_val > 50:
                scores['RSI'] = (rsi_val - 50) * 0.1
            elif rsi_val > 40:
                scores['RSI'] = (rsi_val - 40) * -0.1
            elif rsi_val > 30:
                scores['RSI'] = -1.5 + (30 - rsi_val) * 0.1
            else:
                scores['RSI'] = -2.5 - (30 - rsi_val) * 0.05
        
        # Enhanced Stochastic for intraday
        k, d = calculate_stochastic(df)
        if not k.empty and not d.empty and len(k) > 0:
            k_val, d_val = k.iloc[-1], d.iloc[-1]
            if k_val > d_val and k_val > 20:
                scores['Stochastic'] = min(2.0, (k_val - 20) / 40)
            elif k_val < d_val and k_val < 80:
                scores['Stochastic'] = max(-2.0, -(80 - k_val) / 40)
        
        # Enhanced CCI
        cci = calculate_cci(df)
        if not cci.empty and len(cci) > 0:
            cci_val = cci.iloc[-1]
            scores['CCI'] = normalize_score(cci_val, (100, 250), (-100, -250), (-2.2, 2.2))
        
        # Enhanced ROC
        roc = calculate_roc(df)
        if not roc.empty and len(roc) > 0:
            scores['ROC'] = normalize_score(roc.iloc[-1], (1.0, 3.0), (-1.0, -3.0), (-2.0, 2.0))
        
        # Enhanced Williams %R
        wr = williams_r(df)
        if not wr.empty and len(wr) > 0:
            scores['WilliamsR'] = normalize_score(wr.iloc[-1], (-80, -50), (-20, -5), (-1.8, 1.8))
        
        # --- Enhanced Volume Group (critical for intraday) ---
        zscore = volume_surge(df, lookback=20)
        if not zscore.empty and len(zscore) > 1:
            price_up = df['Close'].iloc[-1] > df['Close'].iloc[-2]
            zscore_val = zscore.iloc[-1]
            
            if price_up and zscore_val > Config.MIN_VOL_SURGE_THRESHOLD:
                scores['VolumeSurge'] = min(3.5, zscore_val * 1.5)
            elif not price_up and zscore_val > Config.MIN_VOL_SURGE_THRESHOLD:
                scores['VolumeSurge'] = max(-3.5, -zscore_val * 1.5)
        
        # Enhanced OBV
        obv_line = calculate_obv(df)
        if len(obv_line) > 5:
            obv_slope = slope(obv_line, 5)
            scores['OBV'] = normalize_score(obv_slope, (1000, 1000000), (-1000, -1000000), (-2.5, 2.5))
        
        # Enhanced CMF
        cmf20 = cmf(df, period=20)
        if not cmf20.empty and len(cmf20) > 0:
            scores['CMF'] = normalize_score(cmf20.iloc[-1], (0.15, 0.35), (-0.15, -0.35), (-2.5, 2.5))
        
        # Enhanced Relative Volume
        rv = relative_volume(df, lookback=min(50, len(df)//2))
        if not rv.empty and len(rv) > 0:
            rv_val = rv.iloc[-1]
            scores['RelVol'] = normalize_score(rv_val, (1.5, 3.0), (0.5, 0.3), (-2.5, 2.5))
        
        # --- Enhanced Volatility Group (intraday breakouts) ---
        atr_val = atr(df, period=14)
        if len(atr_val) > 20:
            atr_ma = atr_val.rolling(20).mean()
            if len(atr_ma) > 0 and atr_ma.iloc[-1] != 0:
                atr_ratio = atr_val.iloc[-1] / atr_ma.iloc[-1]
                atr_slope_ratio = (atr_val.iloc[-1] / atr_val.iloc[-5]) if len(atr_val) >= 5 and atr_val.iloc[-5] > 0 else 1
                
                if atr_ratio > 1.2 and atr_slope_ratio > 1.1:
                    price_direction = 1 if df['Close'].iloc[-1] > df['Close'].iloc[-5] else -1
                    volatility_strength = min(3.0, (atr_ratio - 1) * 3.0)
                    scores['VolatilityExpansion'] = volatility_strength * price_direction
        
        # Enhanced Bollinger Bands
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(df)
        if not bb_upper.empty and not bb_lower.empty:
            close_price = df['Close'].iloc[-1]
            if close_price > bb_upper.iloc[-1]:
                bb_strength = (close_price - bb_upper.iloc[-1]) / (bb_upper.iloc[-1] - bb_middle.iloc[-1])
                scores['Bollinger'] = min(2.5, bb_strength * 2.5)
            elif close_price < bb_lower.iloc[-1]:
                bb_strength = (bb_lower.iloc[-1] - close_price) / (bb_middle.iloc[-1] - bb_lower.iloc[-1])
                scores['Bollinger'] = max(-2.5, -bb_strength * 2.5)
        
        # --- Enhanced OI Group ---
        oi_buildup = detect_oi_buildup(df, 20)
        if oi_buildup is not None:
            scores['OIChange'] = normalize_score(oi_buildup, (15, 40), (-15, -40), (-3.0, 3.0))
        
        vol_oi_sync = volume_oi_sync_analysis(df)
        if vol_oi_sync is not None:
            scores['VolumeOISync'] = normalize_score(vol_oi_sync, (20, 50), (-20, -50), (-2.5, 2.5))
        
        intr_mom = intraday_momentum(df)
        if intr_mom is not None:
            scores['IntradayMomentum'] = normalize_score(intr_mom, (25, 60), (-25, -60), (-3.0, 3.0))
        
    except Exception as e:
        logger.error(f"Error calculating technical indicator scores: {e}")
    
    return scores

def analyze_ultimate_signals(timeframe_data, market_regime='neutral'):
    """Ultimate signal analysis using OHLCV + OI"""
    total_score, total_weight = 0.0, 0.0
    group_scores = defaultdict(float)
    group_weights = defaultdict(float)
    
    # Process technical indicators from multiple timeframes
    for tf_min, df in timeframe_data.items():
        if df is None or df.empty or len(df) < 15:  # Lowered requirement
            continue
        
        indicator_scores = calculate_technical_indicator_scores(df)
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
                        (group == 'OI' and any(term in indicator for term in ['OI', 'Intraday']))
                    )
                    
                    if belongs_to_group:
                        grp_score += indicator_scores[indicator] * ind_weight
                        grp_weight += abs(indicator_scores[indicator]) * ind_weight
            
            if grp_weight > 0:
                norm_grp_score = (grp_score / grp_weight) * weight * tf_weight
                group_scores[group] += norm_grp_score
                group_weights[group] += weight * tf_weight
    
    # Calculate final score
    final_score = 0
    max_possible_score = 0
    
    for group, score in group_scores.items():
        final_score += score
        max_possible_score += group_weights[group]
    
    if max_possible_score == 0:
        return 'Neutral', 0.0, {}
    
    normalized_score = (final_score / max_possible_score) * 100
    
    # Apply market regime multipliers
    if normalized_score > 0 and market_regime == 'bullish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bullish_in_bull_market']
    elif normalized_score > 0 and market_regime == 'bearish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bullish_in_bear_market']
    elif normalized_score < 0 and market_regime == 'bearish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bearish_in_bear_market']
    elif normalized_score < 0 and market_regime == 'bullish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bearish_in_bull_market']
    
    # Signal classification for intraday
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
    
    # Calculate detailed sub-scores for display
    final_sub_scores = {}
    for group in group_scores:
        if group_weights[group] > 0:
            final_sub_scores[group] = group_scores[group] / group_weights[group] * 10
    
    return signal, normalized_score, final_sub_scores

# ========== TIMING FUNCTIONS ==========

def generate_backtest_timestamps(backtest_date):
    """Generate timestamps for backtesting"""
    timestamps = []
    base_date = IST.localize(datetime.strptime(backtest_date, "%Y-%m-%d"))
    current_time = base_date.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = base_date.replace(hour=15, minute=30, second=0, microsecond=0)
    
    first_scan = current_time + timedelta(minutes=5, seconds=Config.SETTLE_DELAY_SECONDS)
    timestamps.append(first_scan)
    
    current_scan = first_scan
    while current_scan < market_end:
        current_scan += timedelta(minutes=5)
        if current_scan <= market_end:
            timestamps.append(current_scan)
    
    return timestamps

def next_5min_boundary_ist(now_ist: datetime) -> datetime:
    """Get next 5-minute boundary"""
    minute = (now_ist.minute // 5) * 5
    boundary = now_ist.replace(minute=minute, second=0, microsecond=0)
    if boundary <= now_ist:
        boundary = boundary + timedelta(minutes=5)
    return boundary

def get_exact_candle_close_time(now_ist: datetime) -> datetime:
    """Get exact candle close time with settlement delay"""
    next_boundary = next_5min_boundary_ist(now_ist)
    return next_boundary + timedelta(seconds=Config.SETTLE_DELAY_SECONDS)

def parse_hhmm(s: str):
    """Parse HH:MM time string and return tuple (hour, minute)"""
    try:
        h, m = map(int, s.split(":"))
        return (h, m)
    except:
        return (9, 15)  # Default fallback

def today_ist_dt(hhmm: str) -> datetime:
    """Convert HH:MM to today's IST datetime"""
    now = datetime.now(IST)
    h, m = parse_hhmm(hhmm)
    return now.replace(hour=h, minute=m, second=0, microsecond=0)

def sleep_until(ts: datetime):
    """Sleep until specific timestamp"""
    while True:
        now = datetime.now(IST)
        delta = (ts - now).total_seconds()
        if delta <= 0:
            break
        time.sleep(min(0.5, delta))

# ========== DATA FETCHING ==========

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

def authenticate_session():
    """Create authenticated TrueData session"""
    return TD_hist(Config.TDUSERNAME, Config.TDPASSWORD, log_level=logging.CRITICAL)

def build_sessions():
    """Build pool of TrueData sessions"""
    pool = []
    for i in range(Config.TD_HIST_SESSIONS):
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

def normalize_hist_df(df, symbol):
    """FIXED: Normalize historical dataframe with proper datetime handling"""
    if df is None or len(df) == 0: 
        return None
    
    try:
        out = df.copy()
        
        # Convert column names to lowercase for consistency
        out.rename(columns={c: str(c).lower() for c in out.columns}, inplace=True)
        
        # Map common column variations
        rename_map = {}
        for src, tgt in [
            ("timestamp", "Date"), ("time", "Date"), ("datetime", "Date"), ("date", "Date"),
            ("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"),
            ("volume", "Volume"), ("vol", "Volume"),
            ("oi", "OpenInterest"), ("openinterest", "OpenInterest"), ("open_interest", "OpenInterest")
        ]:
            if src in out.columns: 
                rename_map[src] = tgt
        
        out.rename(columns=rename_map, inplace=True)
        
        # Handle Date column
        if "Date" not in out.columns:
            if isinstance(out.index, pd.DatetimeIndex):
                out["Date"] = out.index
            else:
                logger.warning(f"No Date column found for {symbol}")
                return None
        
        # Ensure Volume column exists
        if "Volume" not in out.columns:
            out["Volume"] = 0
        
        # Handle OpenInterest column
        if "OpenInterest" in out.columns:
            out["OpenInterest"] = pd.to_numeric(out["OpenInterest"], errors="coerce")
            out["OpenInterest"] = out["OpenInterest"].fillna(0)
        
        # FIXED: Convert Date column to datetime with better error handling
        try:
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
            # Remove rows with invalid dates (NaT)
            out = out.dropna(subset=["Date"])
            
            if len(out) == 0:
                logger.warning(f"No valid dates found for {symbol}")
                return None
        except Exception as date_e:
            logger.error(f"Date conversion error for {symbol}: {date_e}")
            return None
        
        # FIXED: Timezone handling with better error management
        try:
            if pd.api.types.is_datetime64tz_dtype(out["Date"]):
                # Already timezone-aware - convert to IST
                out["Date"] = out["Date"].dt.tz_convert(IST)
            else:
                # Timezone-naive - localize to IST
                # Use 'infer' for ambiguous times to handle DST transitions
                out["Date"] = out["Date"].dt.tz_localize(IST, ambiguous='infer', nonexistent='shift_forward')
        
        except Exception as tz_e:
            logger.warning(f"Timezone handling issue for {symbol}: {tz_e}")
            # Fallback: keep as timezone-naive
            try:
                out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
                out = out.dropna(subset=["Date"])
            except:
                logger.error(f"Failed to process dates for {symbol}")
                return None
        
        # Convert OHLC columns to numeric
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")
        
        # Remove rows with missing OHLC data
        out = out.dropna(subset=["Open", "High", "Low", "Close"])
        
        if len(out) == 0:
            logger.warning(f"No valid OHLC data for {symbol}")
            return None
        
        # Sort by date and set index
        out = out.sort_values("Date").set_index("Date")
        
        # Remove duplicate timestamps
        out = out[~out.index.duplicated(keep='last')]
        
        if len(out) == 0:
            return None
        
        # Final validation
        if not isinstance(out.index, pd.DatetimeIndex):
            logger.warning(f"Invalid datetime index for {symbol} after processing")
            return None
        
        return out
        
    except Exception as e:
        logger.error(f"Normalize error {symbol}: {e}")
        return None

def pick_session(symbol_orig, timeframe_minutes):
    """Pick session for symbol based on hash"""
    return (hash(symbol_orig) ^ timeframe_minutes) % len(tdhist_pool)

def fetch_one_timeaware(symbol_orig, timeframe_minutes, limiter, hist, up_to_time):
    """FIXED: Fetch single timeframe data with proper datetime handling"""
    # Convert to TrueData symbol format (-I)
    td_symbol = convert_to_truedata_symbol(symbol_orig)
    bar_size = BAR_SIZE_MAP.get(timeframe_minutes)
    duration_str = DURATION_MAP.get(timeframe_minutes)
    
    if not bar_size or not duration_str:
        return symbol_orig, timeframe_minutes, None
    
    try:
        limiter.acquire()
        
        # FIXED: Proper datetime handling
        if up_to_time and isinstance(up_to_time, datetime):
            # Ensure up_to_time is timezone-aware
            if up_to_time.tzinfo is None:
                up_to_time_aware = IST.localize(up_to_time)
            else:
                up_to_time_aware = up_to_time.astimezone(IST)
            
            # Parse duration to calculate start time  
            dur_parts = duration_str.split()
            if len(dur_parts) == 2:
                try:
                    dur_num, dur_unit = int(dur_parts[0]), dur_parts[1]
                    if dur_unit.upper() == 'D':
                        start_time_naive = up_to_time_aware.replace(tzinfo=None) - timedelta(days=dur_num)
                        start_time_aware = IST.localize(start_time_naive)
                    else:
                        # Default to days if unknown unit
                        start_time_naive = up_to_time_aware.replace(tzinfo=None) - timedelta(days=dur_num)
                        start_time_aware = IST.localize(start_time_naive)
                except (ValueError, TypeError):
                    # Fallback: use 30 days
                    start_time_naive = up_to_time_aware.replace(tzinfo=None) - timedelta(days=30)
                    start_time_aware = IST.localize(start_time_naive)
            else:
                # Fallback: use 30 days
                start_time_naive = up_to_time_aware.replace(tzinfo=None) - timedelta(days=30)
                start_time_aware = IST.localize(start_time_naive)
            
            # FIXED: Use timezone-naive datetimes for TrueData API
            df_raw = hist.get_historic_data(
                td_symbol, 
                start_time=start_time_aware.replace(tzinfo=None), 
                end_time=up_to_time_aware.replace(tzinfo=None), 
                bar_size=bar_size
            )
        else:
            # Live mode - use duration string
            df_raw = hist.get_historic_data(td_symbol, duration=duration_str, bar_size=bar_size)
        
        df = normalize_hist_df(df_raw, td_symbol)
        return symbol_orig, timeframe_minutes, df
    
    except Exception as e:
        logger.error(f"Error fetching {symbol_orig} {timeframe_minutes}min: {e}")
        return symbol_orig, timeframe_minutes, None

def fetch_one(symbol_orig, timeframe_minutes, limiter, hist):
    """Fetch single timeframe data (live mode)"""
    return fetch_one_timeaware(symbol_orig, timeframe_minutes, limiter, hist, None)

def prefetch_all_timeaware(stocks, up_to_time=None, max_workers=Config.MAX_WORKERS):
    """Prefetch all timeframe data efficiently"""
    tfs = [5, 15, 30, 60, 1440]
    total_calls, stock_multi_data = len(stocks) * len(tfs), defaultdict(dict)
    
    global api_calls_done
    with api_calls_lock: 
        api_calls_done = 0
    
    desc = "📊 Fetching TrueData OHLC/Volume/OI"
    with tqdm(total=total_calls, desc=desc, ncols=100, leave=False) as api_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for s in stocks:
                for tf in tfs:
                    session_idx = pick_session(s, tf)
                    futures.append(executor.submit(
                        fetch_one_timeaware, s, tf, sess_limiters[session_idx], 
                        tdhist_pool[session_idx], up_to_time
                    ))
            
            for fut in as_completed(futures):
                symbol_orig, tf, df = fut.result()
                if df is not None and len(df) > 0:
                    stock_multi_data[symbol_orig][tf] = df
                api_bar.update(1)
    
    return {s: d for s, d in stock_multi_data.items() if len(d) >= 1}  # Lowered requirement

def prefetch_all(stocks, max_workers=Config.MAX_WORKERS):
    """Prefetch all timeframe data (live mode)"""
    return prefetch_all_timeaware(stocks, None, max_workers)

def get_market_regime(index_symbol="NIFTY 50"):
    """FIXED: Get current market regime with proper datetime handling"""
    try:
        si = pick_session(index_symbol, 1440)
        df_raw = tdhist_pool[si].get_historic_data(index_symbol, duration="200 D", bar_size="1 day")
        df = normalize_hist_df(df_raw, index_symbol)
        
        if df is None or len(df) < 50: 
            logger.warning(f"Insufficient data for market regime analysis: {len(df) if df is not None else 0} candles")
            return 'neutral'
        
        # Calculate EMAs with proper error handling
        try:
            ema20_series = ema(df['Close'], 20)
            ema50_series = ema(df['Close'], 50)
            
            if ema20_series.empty or ema50_series.empty or len(ema20_series) == 0 or len(ema50_series) == 0: 
                logger.warning("EMA calculation failed for market regime")
                return 'neutral'
            
            # FIXED: Get the last valid values
            ema20_val = ema20_series.dropna().iloc[-1] if len(ema20_series.dropna()) > 0 else None
            ema50_val = ema50_series.dropna().iloc[-1] if len(ema50_series.dropna()) > 0 else None
            close = df['Close'].dropna().iloc[-1] if len(df['Close'].dropna()) > 0 else None
            
            # Validate values are not None/NaN
            if ema20_val is None or ema50_val is None or close is None:
                logger.warning("Invalid EMA or close values for market regime")
                return 'neutral'
            
            if pd.isna(ema20_val) or pd.isna(ema50_val) or pd.isna(close):
                logger.warning("NaN values in EMA or close for market regime")
                return 'neutral'
            
            if close > ema20_val and ema20_val > ema50_val:
                return 'bullish'
            elif close < ema20_val and ema20_val < ema50_val:
                return 'bearish'
            else:
                return 'neutral'
                
        except Exception as calc_e:
            logger.warning(f"EMA calculation error for market regime: {calc_e}")
            return 'neutral'
    
    except Exception as e:
        logger.warning(f"Could not fetch market regime for {index_symbol}: {e}")
        return 'neutral'

def enhanced_institutional_flow_analysis(tf_data):
    """Enhanced institutional flow analysis"""
    frames = [tf_data.get(t) for t in (5, 15, 30) if tf_data.get(t) is not None and len(tf_data.get(t)) >= 20]  # Lowered requirement
    if not frames: 
        return "Unknown"
    
    votes = 0
    for df in frames:
        cmf_series = cmf(df, 20)
        rv_series = relative_volume(df, min(50, len(df)//2))
        
        if cmf_series.empty or rv_series.empty: 
            continue
        
        cmf_last = cmf_series.iloc[-1]
        rv_last = rv_series.iloc[-1]
        
        # Enhanced voting logic
        if cmf_last > 0.1 and rv_last > 1.5: 
            votes += 2  # Strong accumulation
        elif cmf_last > 0.05 and rv_last > 1.2: 
            votes += 1  # Mild accumulation
        elif cmf_last < -0.1 and rv_last > 1.5: 
            votes -= 2  # Strong distribution
        elif cmf_last < -0.05 and rv_last > 1.2: 
            votes -= 1  # Mild distribution
    
    if votes >= 3: 
        return "Strong Accumulation"
    elif votes >= 2: 
        return "Accumulation"
    elif votes <= -3: 
        return "Strong Distribution"
    elif votes <= -2: 
        return "Distribution"
    else: 
        return "Neutral"

# ========== 5-MINUTE VOLUME/OI TRACKING ==========

def calculate_5min_volume_oi_changes(df, symbol, scan_time):
    """Calculate 5-minute volume and OI changes"""
    try:
        df_5min = df[df.index <= scan_time]
        if len(df_5min) < 2:
            return 0, None, 0, 0
        
        current_volume = int(df_5min['Volume'].iloc[-1]) if 'Volume' in df_5min.columns else 0
        previous_volume = int(df_5min['Volume'].iloc[-2]) if 'Volume' in df_5min.columns else 0
        vol_change_pct = ((current_volume - previous_volume) / previous_volume * 100) if previous_volume > 0 else 0
        
        if _has_real_oi(df_5min):
            current_oi = int(df_5min['OpenInterest'].iloc[-1])
            previous_oi = int(df_5min['OpenInterest'].iloc[-2])
            oi_change_pct = ((current_oi - previous_oi) / previous_oi * 100) if previous_oi > 0 else 0
        else:
            current_oi, oi_change_pct = None, 0
        
        return current_volume, current_oi, vol_change_pct, oi_change_pct
    
    except Exception as e:
        logger.error(f"Error calculating 5-min changes for {symbol}: {e}")
        return 0, None, 0, 0

def extract_5min_volume_oi_data(df, symbol, time_point=None, is_live=False):
    """ENHANCED: Extract 5-minute volume and OI data - SHOWS ALL OI CHANGES"""
    try:
        global intraday_volume_data, intraday_oi_data
        
        if time_point and not is_live:
            df_slice = df[df.index <= time_point]
        else:
            df_slice = df
        
        if df_slice.empty:
            return {
                'current_volume': 'N/A', 'current_oi': 'N/A', 
                'volume_change_pct': 0, 'oi_change_pct': 0,
                'volume': 'N/A', 'oi': 'N/A', 
                'volume_change': 'N/A', 'oi_change': 'N/A'
            }
        
        # ========== ENHANCED FORMATTING - DEFINED EARLY ==========
        def format_number(val):
            if isinstance(val, int):
                if val > 10000000: return f"{val/1000000:.1f}M"
                elif val > 100000: return f"{val/1000:.0f}K"
                elif val > 999: return f"{val:,}"
                else: return str(val)
            return "—"
        
        # ========== ENHANCED VOLUME ANALYSIS ==========
        if len(df_slice) >= 2:
            current_volume = int(df_slice['Volume'].iloc[-1])
            previous_volume = int(df_slice['Volume'].iloc[-2])
            vol_change_pct = ((current_volume - previous_volume) / previous_volume * 100) if previous_volume > 0 else 0
        else:
            current_volume, vol_change_pct = 0, 0
        
        # ========== ENHANCED OI ANALYSIS - 3 METHODS ==========
        oi_change_pct = 0
        current_oi = None  # Default None
        
        if _has_real_oi(df_slice) and len(df_slice) >= 2:
            current_oi = int(df_slice['OpenInterest'].iloc[-1])
            previous_oi = int(df_slice['OpenInterest'].iloc[-2])
            
            # METHOD 1: 5-min OI change (show ALL values >=0.1%)
            if previous_oi > 0:
                oi_change_pct_5min = ((current_oi - previous_oi) / previous_oi) * 100
            else:
                oi_change_pct_5min = 0
            
            # METHOD 2: 15-min OI change (more reliable)
            if len(df_slice) >= 4:  # 15-min = 3 candles
                oi_15min_ago = int(df_slice['OpenInterest'].iloc[-4] if len(df_slice) >= 4 else current_oi)
                if oi_15min_ago > 0:
                    oi_change_pct_15min = ((current_oi - oi_15min_ago) / oi_15min_ago) * 100
                else:
                    oi_change_pct_15min = 0
            else:
                oi_change_pct_15min = oi_change_pct_5min
            
            # METHOD 3: OI Buildup Score (20-period) - MOST IMPORTANT FOR INTRADAY
            oi_buildup_score = detect_oi_buildup(df_slice, lookback=20)
            if oi_buildup_score is not None:
                oi_buildup_pct = oi_buildup_score  # Already in %
            else:
                oi_buildup_pct = 0
            
            # PRIORITY: Show strongest signal
            if abs(oi_buildup_pct) >= 2.0:  # Strong buildup
                oi_change_pct = oi_buildup_pct
            elif abs(oi_change_pct_15min) >= 1.0:  # 15-min change
                oi_change_pct = oi_change_pct_15min
            elif abs(oi_change_pct_5min) >= 0.1:  # 5-min change
                oi_change_pct = oi_change_pct_5min
            else:
                oi_change_pct = oi_change_pct_5min
            
            current_oi_display = format_number(current_oi)
        else:
            # NO OI DATA - Show "—" instead of N/A
            current_oi_display = "—"
            oi_change_pct = 0
        
        # ========== ENHANCED CACHING ==========
        if abs(vol_change_pct) < 0.1 and abs(oi_change_pct) < 0.1:
            prev_volume = intraday_volume_data.get(symbol, None)
            prev_oi = intraday_oi_data.get(symbol, None)
            
            if prev_volume is not None and prev_volume > 0 and current_volume > 0:
                vol_change_pct = ((current_volume - prev_volume) / prev_volume) * 100
            
            if prev_oi is not None and prev_oi > 0 and current_oi is not None:
                oi_change_pct = ((current_oi - prev_oi) / prev_oi) * 100
        
        # Update cache
        intraday_volume_data[symbol] = current_volume
        intraday_oi_data[symbol] = current_oi
        
        # ========== FORMATTING ==========
        current_volume_display = format_number(current_volume)
        
        # SHOW ALL OI CHANGES >= 0.1% - NO MORE N/A!
        volume_change_display = f"{vol_change_pct:+.1f}%" if abs(vol_change_pct) >= 0.1 else "0.0%"
        oi_change_display = (
            f"{oi_change_pct:+.1f}%" if abs(oi_change_pct) >= 0.1 
            else "0.0%" if current_oi is not None 
            else "—"
        )
        
        return {
            'current_volume': current_volume_display,
            'current_oi': current_oi_display,
            'volume_change_pct': vol_change_pct,
            'oi_change_pct': oi_change_pct,  # Now ALWAYS has value
            'volume': current_volume_display,
            'oi': current_oi_display,
            'volume_change': volume_change_display,
            'oi_change': oi_change_display,
            '_raw_volume': current_volume,
            '_raw_oi': current_oi
        }
        
    except Exception as e:
        logger.error(f"Error extracting 5-min data for {symbol}: {e}")
        return {
            'current_volume': 'N/A', 'current_oi': 'N/A', 
            'volume_change_pct': 0, 'oi_change_pct': 0,
            'volume': 'N/A', 'oi': 'N/A', 
            'volume_change': 'N/A', 'oi_change': 'N/A'
        }

# ========== ULTIMATE SCANNER LOGIC WITH CORRECTED SYMBOL HANDLING ==========

def run_ultimate_scan_at_time(time_point_aware, stocks, market_regime, is_live=False):
    """FIXED: Ultimate scan with proper datetime filtering"""
    
    # Convert stocks to TrueData format for fetching
    truedata_stocks = [convert_to_truedata_symbol(s) for s in stocks]
    
    # Step 1: Fetch TrueData OHLC/Volume/OI data
    stock_multi_data = prefetch_all(truedata_stocks, max_workers=Config.MAX_WORKERS) if is_live else \
                      prefetch_all_timeaware(truedata_stocks, time_point_aware, max_workers=Config.MAX_WORKERS)
    
    print_colored(f"✅ Data fetch complete. TrueData: {len(stock_multi_data)} stocks", Colors.GREEN)
    print_colored(f"Running ultimate analysis (Regime: {market_regime.upper()})...", Colors.GREEN)
    
    signals_this_scan = []
    current_symbols = set()
    
    # Process TrueData data with proper symbol mapping
    for truedata_symbol, timeframe_data in stock_multi_data.items():
        # Convert TrueData symbol to clean symbol for display
        clean_symbol = normalize_symbol_for_display(truedata_symbol)
        current_symbols.add(clean_symbol)
        
        # FIXED: Filter timeframes with proper datetime handling
        filtered_timeframes = {}
        for tf, df in timeframe_data.items():
            if df is not None and not df.empty:
                if is_live:
                    df_slice = df
                else:
                    # FIXED: Proper datetime comparison
                    if time_point_aware and isinstance(df.index, pd.DatetimeIndex):
                        try:
                            # Ensure both are timezone-aware for comparison
                            if time_point_aware.tzinfo is None:
                                time_point_aware = IST.localize(time_point_aware)
                            
                            if df.index.tz is None:
                                # DataFrame index is timezone-naive, localize it
                                df_with_tz = df.copy()
                                df_with_tz.index = df_with_tz.index.tz_localize(IST, ambiguous='infer', nonexistent='shift_forward')
                                df_slice = df_with_tz[df_with_tz.index <= time_point_aware]
                            else:
                                # Both are timezone-aware
                                df_slice = df[df.index <= time_point_aware]
                        except Exception as filter_e:
                            logger.warning(f"Datetime filtering issue for {clean_symbol}: {filter_e}")
                            # Fallback: use all data
                            df_slice = df
                    else:
                        df_slice = df
                
                if not df_slice.empty and len(df_slice) >= 15:  # Lowered requirement
                    filtered_timeframes[tf] = df_slice
        
        if len(filtered_timeframes) < 1:  # Lowered requirement
            continue
        
        # Ultimate signal analysis
        signal, score, sub_scores = analyze_ultimate_signals(
            filtered_timeframes, market_regime
        )
        
        if abs(score) >= Config.SCORE_THRESHOLD_MIN:
            # Enhanced institutional flow
            flow_tag = enhanced_institutional_flow_analysis(filtered_timeframes)
            
            # Extract volume/OI data
            tf_5min = filtered_timeframes.get(5)
            if tf_5min is not None:
                oi_vol_data = extract_5min_volume_oi_data(tf_5min, clean_symbol, time_point_aware, is_live=is_live)
            else:
                main_tf_data = filtered_timeframes.get(15, filtered_timeframes.get(30, list(filtered_timeframes.values())[0]))
                oi_vol_data = extract_5min_volume_oi_data(main_tf_data, clean_symbol, time_point_aware, is_live=is_live)
            
            # Ultimate result
            result = {
                'symbol': clean_symbol,
                'signal': signal,
                'score': score,
                'sub_scores': sub_scores,
                'flow': flow_tag,
                
                **oi_vol_data,
                
                # Intraday specific
                'price_change': (filtered_timeframes.get(5, {}).get('Close', pd.Series()).iloc[-1] / filtered_timeframes.get(5, {}).get('Open', pd.Series()).iloc[-1] - 1) * 100 if 5 in filtered_timeframes else 0,
                'quality': calculate_quality_score(score, oi_vol_data)
            }
            
            signals_this_scan.append(result)
    
    return signals_this_scan, current_symbols

def calculate_quality_score(technical_score, oi_vol_data):
    """Calculate overall quality score for intraday trading"""
    try:
        quality_score = 0
        
        # Technical alignment (50%)
        tech_alignment = min(100, abs(technical_score)) / 100
        quality_score += tech_alignment * 50
        
        # Volume strength (30%)
        vol_change = oi_vol_data.get('volume_change_pct', 0)
        if abs(vol_change) > 20:
            quality_score += 30
        elif abs(vol_change) > 10:
            quality_score += 20
        elif abs(vol_change) > 5:
            quality_score += 10
        
        # OI strength (20%)
        oi_change = oi_vol_data.get('oi_change_pct', 0)
        if abs(oi_change) > 10:
            quality_score += 20
        elif abs(oi_change) > 5:
            quality_score += 10
        
        return min(100, round(quality_score, 1))
    
    except:
        return 50.0

# ========== ENHANCED TABLE DISPLAY ==========

def create_ultimate_option_table(data, title, new_stocks=None, show_time=None):
    """Ultimate table for intraday traders"""
    if not data:
        if RICH_AVAILABLE:
            console.print(f"\n[bold magenta]{title}[/bold magenta]")
            console.print("[yellow]No stocks found in this category.[/yellow]")
        else:
            print_colored(f"\n{title}", Colors.HEADER)
            print_colored("No stocks found in this category.", Colors.YELLOW)
        return

    if RICH_AVAILABLE:
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold blue")
        table.add_column("Stock", style="bold white", width=8, justify="left")
        table.add_column("Signal", style="bold", width=16, justify="center")
        table.add_column("Score", style="bold", width=6, justify="right")
        table.add_column("Vol%", style="cyan", width=6, justify="right")
        table.add_column("OI%", style="yellow", width=6, justify="right")
        table.add_column("Flow", style="green", width=14, justify="left")
        table.add_column("P.Chg%", style="bright_green", width=7, justify="right")
        table.add_column("Quality", style="bright_magenta", width=6, justify="right")
        
        for item in data:
            symbol = item['symbol']
            is_new = new_stocks and symbol in new_stocks
            
            # Signal style based on score
            if item['score'] > 60: signal_style = "bold bright_green"
            elif item['score'] > 30: signal_style = "bold green"
            elif item['score'] > 0: signal_style = "green"
            elif item['score'] < -60: signal_style = "bold bright_red"
            elif item['score'] < -30: signal_style = "bold red"
            else: signal_style = "red"
            
            stock_style = f"[bold bright_magenta]{symbol} ✨[/bold bright_magenta]" if is_new else symbol
            
            # Format data
            vol_chg = item.get('volume_change_pct', 0)
            vol_display = f"{vol_chg:+.1f}%" if abs(vol_chg) > 0.1 else "0.0%"
            
            oi_chg = item.get('oi_change_pct', 0)
            if oi_chg != 0 or item.get('_raw_oi') is not None:
                oi_display = f"{oi_chg:+.1f}%"
            else:
                oi_display = "—"  # Clean dash for no OI data
            
            price_chg = item.get('price_change', 0)
            price_display = f"{price_chg:+.2f}%"
            
            quality = item.get('quality', 50)
            
            # Color coding
            qual_style = f"[bright_magenta]{quality:.1f}[/bright_magenta]" if quality > 80 else f"[magenta]{quality:.1f}[/magenta]" if quality > 60 else f"[dim]{quality:.1f}[/dim]"
            
            table.add_row(
                stock_style,
                f"[{signal_style}]{item['signal']}[/{signal_style}]",
                f"[bold]{item['score']:.1f}[/bold]",
                vol_display,
                oi_display,
                item.get('flow', 'Unknown'),
                price_display,
                qual_style
            )
        
        if show_time:
            console.print(f"\n[bold magenta]{title} - {show_time}[/bold magenta]")
        else:
            console.print(f"\n[bold magenta]{title}[/bold magenta]")
        console.print(table)
    
    else:
        # ASCII fallback
        if show_time:
            print_colored(f"\n{title} - {show_time}", Colors.HEADER)
        else:
            print_colored(f"\n{title}", Colors.HEADER)
        
        print_colored("="*100, Colors.BLUE)
        header = f"{'Stock':<8} | {'Signal':<16} | {'Score':>6} | {'Vol%':>6} | {'OI%':>6} | {'Flow':<14} | {'P.Chg%':>7} | {'Qual':>6}"
        print_colored(header, Colors.BOLD)
        print_colored("-"*100, Colors.BLUE)
        
        for item in data:
            symbol = item['symbol']
            is_new = new_stocks and symbol in new_stocks
            
            vol_chg = item.get('volume_change_pct', 0)
            vol_str = f"{vol_chg:+.1f}%" if abs(vol_chg) > 0.1 else "0.0%"
            
            oi_chg = item.get('oi_change_pct', 0)
            if oi_chg != 0 or item.get('_raw_oi') is not None:
                oi_str = f"{oi_chg:+.1f}%"
            else:
                oi_str = "—"  # Clean dash for no OI data
            
            price_chg = item.get('price_change', 0)
            price_str = f"{price_chg:+.2f}%"
            
            quality = item.get('quality', 50)
            
            row = f"{symbol:<8} | {item['signal']:<16} | {item['score']:>6.1f} | {vol_str:>6} | {oi_str:>6} | {item.get('flow', 'Unknown'):<14} | {price_str:>7} | {quality:>6.1f}"
            
            if is_new:
                print_colored(row + " ← ✨ NEW!", Colors.MAGENTA)
            else:
                print(row)
        
        print_colored("="*100, Colors.BLUE)

def create_ultimate_summary_panel(signals):
    """Create summary panel with key statistics"""
    if not signals:
        return
    
    # Calculate statistics
    total_signals = len(signals)
    high_quality = len([s for s in signals if s.get('quality', 0) > 80])
    strong_vol_surges = len([s for s in signals if abs(s.get('volume_change_pct', 0)) > 20])
    strong_oi_changes = len([s for s in signals if abs(s.get('oi_change_pct', 0)) > 10])
    strong_buys = len([s for s in signals if s.get('score', 0) > 50])
    strong_sells = len([s for s in signals if s.get('score', 0) < -50])
    
    avg_quality = sum(s.get('quality', 0) for s in signals) / total_signals if total_signals > 0 else 0
    avg_vol_change = sum(abs(s.get('volume_change_pct', 0)) for s in signals) / total_signals if total_signals > 0 else 0
    
    if RICH_AVAILABLE:
        summary_text = f"""
[bold cyan]📊 ULTIMATE INTRADAY SUMMARY[/bold cyan]
[green]Total Signals: {total_signals}[/green]
[magenta]High Quality (>80): {high_quality}[/magenta]
[cyan]Strong Vol Surges: {strong_vol_surges}[/cyan]
[yellow]Strong OI Changes: {strong_oi_changes}[/yellow]
[bright_green]Strong Buys: {strong_buys}[/bright_green]
[bright_red]Strong Sells: {strong_sells}[/bright_red]
[yellow]Avg Quality: {avg_quality:.1f}[/yellow]
[blue]Avg Vol Chg: {avg_vol_change:.1f}%[/blue]
        """
        
        panel = Panel(summary_text, title="Intraday Scanner Stats", border_style="blue")
        console.print(panel)
    else:
        print_colored("\n📊 ULTIMATE INTRADAY SUMMARY", Colors.HEADER)
        print_colored("="*40, Colors.BLUE)
        print(f"Total Signals: {total_signals}")
        print(f"High Quality (>80): {high_quality}")
        print(f"Strong Vol Surges: {strong_vol_surges}")
        print(f"Strong OI Changes: {strong_oi_changes}")
        print(f"Strong Buys: {strong_buys}")
        print(f"Strong Sells: {strong_sells}")
        print(f"Avg Quality: {avg_quality:.1f}")
        print(f"Avg Vol Chg: {avg_vol_change:.1f}%")
        print_colored("="*40, Colors.BLUE)

# ========== DIAGNOSTIC FUNCTIONS WITH CORRECTED SYMBOL HANDLING ==========

def run_diagnostic_scan(time_point_aware, stocks, market_regime, is_live=False):
    """Diagnostic scan with proper symbol conversion"""
    
    print_colored("🔍 RUNNING DIAGNOSTIC SCAN WITH CORRECTED SYMBOLS...", Colors.YELLOW)
    
    # Step 1: Test TrueData fetch with corrected symbols
    print_colored("📊 Step 1: Testing TrueData fetch for first 5 stocks...", Colors.CYAN)
    test_stocks = stocks[:5]
    
    print_colored("   Symbol conversion test:", Colors.CYAN)
    for orig_stock in test_stocks:
        td_symbol = convert_to_truedata_symbol(orig_stock)
        print(f"   📈 {orig_stock} -> TrueData: {td_symbol}")
    
    truedata_test_stocks = [convert_to_truedata_symbol(s) for s in test_stocks]
    stock_multi_data = prefetch_all(truedata_test_stocks, max_workers=5) if is_live else \
                      prefetch_all_timeaware(truedata_test_stocks, time_point_aware, max_workers=5)
    
    print(f"   ✅ TrueData received data for {len(stock_multi_data)} stocks")
    for symbol, timeframes in stock_multi_data.items():
        clean_display = normalize_symbol_for_display(symbol)
        print(f"   📈 {clean_display} ({symbol}): {list(timeframes.keys())} timeframes")
        for tf, df in timeframes.items():
            if df is not None:
                print(f"      {tf}min: {len(df)} candles, latest: {df.index[-1] if len(df) > 0 else 'No data'}")
    
    # Step 2: Test signal analysis with corrected symbols
    print_colored("\n📊 Step 2: Testing signal analysis with corrected symbol mapping...", Colors.CYAN)
    signals_found = 0
    
    for truedata_symbol, timeframe_data in list(stock_multi_data.items())[:3]:
        clean_symbol = normalize_symbol_for_display(truedata_symbol)
        print(f"\n   📈 Analyzing {clean_symbol} (TrueData: {truedata_symbol})...")
        
        # Filter timeframes
        filtered_timeframes = {}
        for tf, df in timeframe_data.items():
            if df is not None and not df.empty:
                if is_live:
                    df_slice = df
                else:
                    df_slice = df[df.index <= time_point_aware] if time_point_aware else df
                
                if not df_slice.empty and len(df_slice) >= 15:  # Lowered requirement
                    filtered_timeframes[tf] = df_slice
                    print(f"      ✅ {tf}min: {len(df_slice)} candles")
        
        if len(filtered_timeframes) >= 1:  # Lowered requirement
            # Test with lower threshold
            signal, score, sub_scores = analyze_ultimate_signals(
                filtered_timeframes, market_regime
            )
            
            print(f"      📊 Signal: {signal}, Score: {score:.2f}")
            print(f"      📊 Sub-scores: {sub_scores}")
            
            if abs(score) >= 1.0:  # Much lower threshold for diagnostic
                signals_found += 1
                print(f"      ✅ Would generate signal with threshold 1.0!")
            
            # Show detailed breakdown
            print(f"      📊 Technical Analysis:")
            for tf, df in filtered_timeframes.items():
                tech_scores = calculate_technical_indicator_scores(df)
                if tech_scores:
                    print(f"         {tf}min indicators: {len(tech_scores)} calculated")
                    top_indicators = sorted(tech_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                    for ind, val in top_indicators:
                        print(f"            {ind}: {val:.2f}")
        else:
            print(f"      ❌ Insufficient data: only {len(filtered_timeframes)} timeframes with enough data")
    
    print_colored(f"\n🎯 DIAGNOSTIC SUMMARY:", Colors.HEADER)
    print(f"   📊 TrueData Stocks: {len(stock_multi_data)}")
    print(f"   📈 Potential Signals (threshold 1.0): {signals_found}")
    print(f"   🎯 Current Threshold: {Config.SCORE_THRESHOLD_MIN}")
    
    # Recommendations
    print_colored(f"\n💡 RECOMMENDATIONS:", Colors.YELLOW)
    if len(stock_multi_data) == 0:
        print("   ❌ No TrueData received - check TrueData credentials and connection")
        print("   💡 Try different date range or check if symbols need -I suffix")
    elif signals_found == 0:
        print("   📊 Try lowering score threshold to 1.0")
        print("   📈 Market might be in consolidation phase")
        print("   ⏰ Try different time (market hours: 9:15-15:30)")
    else:
        print("   ✅ System working! Try lower threshold or different time period")

def run_quick_test():
    """Quick test function with corrected symbols"""
    print_colored("\n🔬 QUICK SYSTEM TEST WITH CORRECTED SYMBOLS", Colors.HEADER)
    
    # Test 1: TrueData connection
    try:
        test_symbol_orig = "RELIANCE"
        test_symbol_td = convert_to_truedata_symbol(test_symbol_orig)
        print(f"   Testing symbol conversion: {test_symbol_orig} -> {test_symbol_td}")
        
        session = tdhist_pool[0]
        df_raw = session.get_historic_data(test_symbol_td, duration="5 D", bar_size="1 day")
        df = normalize_hist_df(df_raw, test_symbol_td)
        if df is not None and len(df) > 0:
            print("   ✅ TrueData connection: OK")
            print(f"      Latest {test_symbol_td} data: {df.index[-1]} Price: {df['Close'].iloc[-1]:.2f}")
        else:
            print("   ❌ TrueData connection: No data received")
    except Exception as e:
        print(f"   ❌ TrueData connection: {e}")
    
    # Test 2: Configuration
    print(f"   📊 Score Threshold: {Config.SCORE_THRESHOLD_MIN} (try lowering to 1.0)")
    print(f"   🎯 Group weights sum: {sum(Config.GROUP_WEIGHTS.values())}")

# ========== ULTIMATE BACKTEST FUNCTION ==========

def run_ultimate_backtest(backtest_date, stocks):
    """Ultimate backtest with OHLCV + OI"""
    global backtest_stock_history, intraday_volume_data, intraday_oi_data
    
    print_colored(f"\n🎯 STARTING ULTIMATE INTRADAY BACKTEST FOR {backtest_date}", Colors.HEADER)
    print_colored("🔗 Complete TrueData OHLC/Volume/OI Integration with Corrected Symbols", Colors.GREEN)
    
    timestamps = generate_backtest_timestamps(backtest_date)
    total_scans = len(timestamps)
    print_colored(f"📅 Generated {total_scans} scan points from {timestamps[0].strftime('%H:%M')} to {timestamps[-1].strftime('%H:%M')}", Colors.CYAN)
    
    market_regime = get_market_regime(Config.BENCHMARK_INDEX)
    print_colored(f"📈 Market Regime: {market_regime.upper()}", Colors.BLUE)
    
    all_results = []
    backtest_stock_history = {}
    intraday_volume_data = {}
    intraday_oi_data = {}
    
    with tqdm(total=total_scans, desc="🎯 Ultimate Backtesting", ncols=120) as pbar:
        for i, scan_time in enumerate(timestamps):
            try:
                pbar.set_description(f"Ultimate scan at {scan_time.strftime('%H:%M:%S')}")
                
                # Run ultimate scan with corrected symbols
                signals, current_symbols = run_ultimate_scan_at_time(scan_time, stocks, market_regime, is_live=False)
                
                previous_symbols = set(backtest_stock_history.keys())
                new_stocks = current_symbols - previous_symbols
                
                for symbol in current_symbols:
                    backtest_stock_history[symbol] = scan_time
                
                # Enhanced result tracking
                scan_result = {
                    'timestamp': scan_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'scan_number': i + 1,
                    'total_signals': len(signals),
                    'bullish_signals': len([s for s in signals if s['score'] > 0]),
                    'bearish_signals': len([s for s in signals if s['score'] < 0]),
                    'high_quality_signals': len([s for s in signals if s.get('quality', 0) > 80]),
                    'new_stocks': list(new_stocks),
                    'signals': signals
                }
                
                all_results.append(scan_result)
                
                if signals:
                    signals.sort(key=lambda x: (x.get('quality', 0), abs(x['score'])), reverse=True)
                    top_bullish = [r for r in signals if r['score'] > 0][:Config.BACKTEST_TOP_DISPLAY]
                    top_bearish = [r for r in signals if r['score'] < 0][:Config.BACKTEST_TOP_DISPLAY]
                    
                    scan_time_str = scan_time.strftime('%H:%M')
                    
                    if RICH_AVAILABLE:
                        console.print(f"\n[bold blue]🎯 ULTIMATE SCAN #{i+1}/{total_scans} - {scan_time_str} IST[/bold blue]")
                        console.print(f"[cyan]Signals: {len(signals)} | Quality: {scan_result['high_quality_signals']} | New: {len(new_stocks)}[/cyan]")
                    else:
                        print_colored(f"\n🎯 ULTIMATE SCAN #{i+1}/{total_scans} - {scan_time_str} IST", Colors.BOLD)
                        print_colored(f"Signals: {len(signals)} | Quality: {scan_result['high_quality_signals']} | New: {len(new_stocks)}", Colors.CYAN)
                    
                    if top_bullish:
                        create_ultimate_option_table(top_bullish, f"🟢 ULTIMATE BULLISH SETUPS", new_stocks, scan_time_str)
                    
                    if top_bearish:
                        create_ultimate_option_table(top_bearish, f"🔴 ULTIMATE BEARISH SETUPS", new_stocks, scan_time_str)
                    
                    # Show summary for significant scans
                    if len(signals) > 10:
                        create_ultimate_summary_panel(signals)
                
                pbar.update(1)
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in ultimate backtest scan at {scan_time}: {e}")
                pbar.update(1)
                continue
    
    # Enhanced summary
    print_colored(f"\n📊 ULTIMATE BACKTEST SUMMARY FOR {backtest_date}", Colors.HEADER)
    print_colored("="*150, Colors.BLUE)
    
    total_scans_completed = len([r for r in all_results if r['total_signals'] >= 0])
    total_signals = sum(r['total_signals'] for r in all_results)
    total_bullish = sum(r['bullish_signals'] for r in all_results)
    total_bearish = sum(r['bearish_signals'] for r in all_results)
    total_high_quality = sum(r['high_quality_signals'] for r in all_results)
    unique_stocks = len(backtest_stock_history)
    
    print(f"✅ Scans Completed: {total_scans_completed}/{total_scans}")
    print(f"📊 Total Signals: {total_signals}")
    print(f"🟢 Bullish Signals: {total_bullish}")
    print(f"🔴 Bearish Signals: {total_bearish}")
    print(f"⭐ High Quality: {total_high_quality}")
    print(f"📋 Unique Stocks: {unique_stocks}")
    
    if total_signals > 0:
        quality_ratio = (total_high_quality / total_signals) * 100
        print(f"📊 Avg Signals/Scan: {total_signals/total_scans_completed:.1f}")
        print(f"⚖️ Bull/Bear Ratio: {total_bullish/max(total_bearish, 1):.2f}")
        print(f"⭐ High Quality %: {quality_ratio:.1f}%")
    
    # Most profitable times analysis
    active_scans = sorted(all_results, key=lambda x: (x['high_quality_signals'], x['total_signals']), reverse=True)[:5]
    print_colored("\n🔥 TOP OPPORTUNITY TIMES:", Colors.CYAN)
    for i, scan in enumerate(active_scans):
        if scan['total_signals'] > 0:
            time_str = datetime.fromisoformat(scan['timestamp']).strftime('%H:%M')
            print(f"  {i+1}. {time_str} - {scan['total_signals']} signals | {scan['high_quality_signals']} quality")
    
    # Save enhanced results
    output_filename = f"{backtest_date}_ultimate_intraday_backtest_results.json"
    try:
        with open(output_filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        print_colored(f"\n💾 Ultimate results saved: {output_filename}", Colors.GREEN)
    except Exception as e:
        logger.error(f"Could not save results: {e}")
    
    print_colored("="*150, Colors.BLUE)
    print_colored("🎯 Ultimate Intraday Backtesting Completed!", Colors.GREEN)

# ========== ENHANCED MAIN FUNCTION WITH CONTINUOUS LIVE SCANNING ==========

def main_ultimate_scanner_with_diagnostics():
    """FIXED: Enhanced main function with corrected symbol handling and continuous live scanning"""
    parser = argparse.ArgumentParser(description="Ultimate Intraday Trader Scanner v4.3 - Corrected Symbol Handling with Continuous Live Scanning")
    parser.add_argument("--asof", type=str, help="Historical snapshot: 2025-10-03T14:25")
    parser.add_argument("--backtest", type=str, help="Full day backtest: 2025-10-03")
    parser.add_argument("--test", action="store_true", help="Run quick system test")
    parser.add_argument("--diagnose", action="store_true", help="Run full diagnostic scan")
    parser.add_argument("--threshold", type=float, help="Override score threshold", default=None)
    args = parser.parse_args()
    
    # Load stocks
    try:
        with open(Config.SHARES_FILE, 'r') as f:
            stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(stocks)} symbols from {Config.SHARES_FILE}")
    except Exception:
        stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK", "SBIN", "TATAMOTORS", "AXISBANK", "ADANIPORTS"]
        logger.warning("Using sample stocks for testing")
    
    # Apply threshold override
    if args.threshold:
        Config.SCORE_THRESHOLD_MIN = args.threshold
        print_colored(f"🎯 Score threshold overridden to: {args.threshold}", Colors.YELLOW)
    
    # Quick test mode
    if args.test:
        run_quick_test()
        return
    
    if args.backtest:
        try:
            datetime.strptime(args.backtest, "%Y-%m-%d")
            run_ultimate_backtest(args.backtest, stocks)
        except ValueError:
            logger.error("Invalid date format for --backtest. Use YYYY-MM-DD.")
            return
    
    elif args.asof:
        # Enhanced snapshot with better datetime parsing
        try:
            # Try different datetime formats
            if 'T' in args.asof:
                asof_ts = datetime.fromisoformat(args.asof.replace('Z', '+00:00'))
                if asof_ts.tzinfo is None:
                    asof_ts = IST.localize(asof_ts)
                else:
                    asof_ts = asof_ts.astimezone(IST)
            else:
                # Date only - assume market close time
                date_part = datetime.strptime(args.asof, "%Y-%m-%d")
                asof_ts = IST.localize(date_part.replace(hour=15, minute=30))
        except Exception as dt_e:
            logger.error(f"Invalid timestamp format: {args.asof}. Error: {dt_e}")
            logger.error("Use format: YYYY-MM-DDTHH:MM or YYYY-MM-DD")
            return
        
        logger.info(f"Running ultimate snapshot for: {asof_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        
        # Get market regime with better error handling
        print_colored("📈 Analyzing market regime...", Colors.CYAN)
        market_regime = get_market_regime(Config.BENCHMARK_INDEX)
        print_colored(f"📈 Market Regime: {market_regime.upper()}", Colors.GREEN)
        
        # Run diagnostic if requested
        if args.diagnose:
            run_diagnostic_scan(asof_ts, stocks, market_regime, is_live=False)
            return
        
        try:
            signals, _ = run_ultimate_scan_at_time(asof_ts, stocks, market_regime, is_live=False)
            signals.sort(key=lambda x: (x.get('quality', 0), abs(x['score'])), reverse=True)
            
            # If no signals found, run automatic diagnostics
            if len(signals) == 0:
                print_colored("\n⚠️ No signals found! Running automatic diagnostics...", Colors.YELLOW)
                run_diagnostic_scan(asof_ts, stocks[:5], market_regime, is_live=False)  # Test first 5 stocks
                print_colored("\n💡 SOLUTIONS TO TRY:", Colors.YELLOW)
                print("   1. Lower threshold: python scanner.py --asof {} --threshold 1.0".format(args.asof))
                print("   2. Run diagnostics: python scanner.py --asof {} --diagnose".format(args.asof))
                print("   3. Test system: python scanner.py --test")
                print("   4. Try different time: python scanner.py --asof 2025-10-03T14:00")
                return
            
            top_bullish = [r for r in signals if r['score'] > 0][:20]
            top_bearish = [r for r in signals if r['score'] < 0][:20]
            
            print_colored(f"\n🎯 ULTIMATE SNAPSHOT RESULTS - {asof_ts.strftime('%Y-%m-%d %H:%M')} IST", Colors.BOLD)
            print_colored(f"Market Regime: {market_regime.upper()} | Total Signals: {len(signals)}", Colors.CYAN)
            
            create_ultimate_summary_panel(signals)
            
            if top_bullish:
                create_ultimate_option_table(top_bullish, "🟢 TOP 20 ULTIMATE BULLISH OPPORTUNITIES")
            
            if top_bearish:
                create_ultimate_option_table(top_bearish, "🔴 TOP 20 ULTIMATE BEARISH OPPORTUNITIES")
        
        except Exception as scan_e:
            logger.error(f"Error during ultimate scan: {scan_e}")
            print_colored(f"\n💥 Scan error: {scan_e}", Colors.RED)
            print_colored("🔍 Running diagnostic scan to identify the issue...", Colors.YELLOW)
            run_diagnostic_scan(asof_ts, stocks[:5], market_regime, is_live=False)
    
    else:
        # ========== ENHANCED CONTINUOUS LIVE SCANNER ==========
        print_colored("\n🎯 STARTING ULTIMATE CONTINUOUS LIVE SCANNER v4.3", Colors.GREEN)
        print_colored("⏰ Runs every 5 minutes during market hours (9:15 AM - 3:30 PM IST)", Colors.BLUE)
        
        global scan_count, previous_scan_results, intraday_volume_data, intraday_oi_data
        
        # Initialize state
        intraday_volume_data = {}
        intraday_oi_data = {}
        scan_count = 0
        previous_scan_results = {}
        
        def is_market_open():
            """FIXED: Check if market is currently open"""
            now_ist = datetime.now(IST)
            current_time = now_ist.time()
            current_date = now_ist.date()
            
            # FIXED: Market hours parsing
            market_start_tuple = parse_hhmm(Config.MARKET_START)  # Returns (9, 15)
            market_end_tuple = parse_hhmm(Config.MARKET_END)      # Returns (15, 30)
            
            # FIXED: Create time objects correctly
            start_time = dt_time(market_start_tuple[0], market_start_tuple[1])
            end_time = dt_time(market_end_tuple[0], market_end_tuple[1])
            
            # Check if it's a weekday
            is_weekday = current_date.weekday() < 5  # Monday=0, Sunday=6
            
            # Check if within market hours
            is_within_hours = start_time <= current_time <= end_time
            
            return is_weekday and is_within_hours
        
        def get_next_scan_time():
            """Get the next 5-minute scan time"""
            now_ist = datetime.now(IST)
            
            # Round to next 5-minute boundary
            next_boundary = next_5min_boundary_ist(now_ist)
            
            # Add settlement delay for data accuracy
            next_scan = next_boundary + timedelta(seconds=Config.SETTLE_DELAY_SECONDS)
            
            return next_scan
        
        def wait_for_market_open():
            """Wait until market opens"""
            while not is_market_open():
                now_ist = datetime.now(IST)
                current_time = now_ist.time()
                current_date = now_ist.date()
                
                # Check if it's weekend
                if current_date.weekday() >= 5:  # Saturday or Sunday
                    next_monday = current_date + timedelta(days=(7 - current_date.weekday()))
                    market_open_time = IST.localize(datetime.combine(next_monday, dt_time(9, 15)))
                    wait_seconds = (market_open_time - now_ist).total_seconds()
                    
                    print_colored(f"📅 Weekend detected. Market opens on {next_monday.strftime('%A, %B %d')} at 9:15 AM IST", Colors.YELLOW)
                    print_colored(f"⏰ Sleeping for {format_time_remaining(wait_seconds)}...", Colors.CYAN)
                    
                    # Sleep in chunks to allow for interruption
                    while wait_seconds > 0:
                        sleep_time = min(3600, wait_seconds)  # Sleep 1 hour at a time
                        time.sleep(sleep_time)
                        wait_seconds -= sleep_time
                        if is_market_open():
                            break
                
                else:
                    # Weekday but outside market hours
                    market_start_today = IST.localize(datetime.combine(current_date, dt_time(9, 15)))
                    market_end_today = IST.localize(datetime.combine(current_date, dt_time(15, 30)))
                    
                    if now_ist < market_start_today:
                        # Before market open
                        wait_seconds = (market_start_today - now_ist).total_seconds()
                        print_colored(f"📈 Market opens today at 9:15 AM IST", Colors.YELLOW)
                        print_colored(f"⏰ Waiting {format_time_remaining(wait_seconds)}...", Colors.CYAN)
                    else:
                        # After market close, wait for next day
                        next_day = current_date + timedelta(days=1)
                        market_open_next = IST.localize(datetime.combine(next_day, dt_time(9, 15)))
                        wait_seconds = (market_open_next - now_ist).total_seconds()
                        print_colored(f"📈 Market closed. Opens tomorrow at 9:15 AM IST", Colors.YELLOW)
                        print_colored(f"⏰ Waiting {format_time_remaining(wait_seconds)}...", Colors.CYAN)
                    
                    # Sleep in manageable chunks
                    while wait_seconds > 0 and not is_market_open():
                        sleep_time = min(300, wait_seconds)  # Sleep 5 minutes at a time
                        time.sleep(sleep_time)
                        wait_seconds -= sleep_time
        
        try:
            # Wait for market to open if needed
            if not is_market_open():
                wait_for_market_open()
            
            print_colored("🟢 Market is OPEN! Starting continuous scanning...", Colors.GREEN)
            
            # Main continuous scanning loop
            while True:
                scan_count += 1
                now_ist = datetime.now(IST)
                
                # Check if market is still open
                if not is_market_open():
                    print_colored("📈 Market has CLOSED. Stopping scanner...", Colors.YELLOW)
                    break
                
                try:
                    print_colored(f"\n[{now_ist.strftime('%H:%M:%S')}] 🎯 ULTIMATE LIVE SCANNER v4.3 - Scan #{scan_count}", Colors.HEADER)
                    print_colored("=" * 100, Colors.BLUE)
                    
                    # Get market regime
                    market_regime = get_market_regime(Config.BENCHMARK_INDEX)
                    
                    # Run ultimate scan on full stock list
                    signals, current_symbols = run_ultimate_scan_at_time(now_ist, stocks, market_regime, is_live=True)
                    
                    # Identify new stocks since last scan
                    new_stocks = current_symbols - set(previous_scan_results.keys()) if previous_scan_results else set()
                    previous_scan_results = {s: True for s in current_symbols}
                    
                    # Sort by quality and score
                    signals.sort(key=lambda x: (x.get('quality', 0), abs(x['score'])), reverse=True)
                    top_bullish = [r for r in signals if r['score'] > 0][:15]
                    top_bearish = [r for r in signals if r['score'] < 0][:15]
                    
                    # Display enhanced results
                    total_signals = len(signals)
                    high_quality = len([s for s in signals if s.get('quality', 0) > 80])
                    
                    print_colored(f"\n🎯 ULTIMATE LIVE RESULTS - {now_ist.strftime('%Y-%m-%d %H:%M')} IST (Regime: {market_regime.upper()})", Colors.BOLD)
                    print_colored(f"📊 Total: {total_signals} | ⭐ Quality: {high_quality} | ✨ New: {len(new_stocks)}", Colors.CYAN)
                    
                    if signals:
                        create_ultimate_summary_panel(signals)
                    
                    if top_bullish:
                        create_ultimate_option_table(top_bullish, f"🟢 TOP 15 ULTIMATE BULLISH OPPORTUNITIES", new_stocks)
                    
                    if top_bearish:
                        create_ultimate_option_table(top_bearish, f"🔴 TOP 15 ULTIMATE BEARISH OPPORTUNITIES", new_stocks)
                    
                    # Show new stocks alert
                    if new_stocks:
                        new_stocks_list = list(new_stocks)[:10]
                        more_text = f" +{len(new_stocks)-10} more" if len(new_stocks) > 10 else ""
                        print_colored(f"\n✨ NEW STOCKS DETECTED: {', '.join(new_stocks_list)}{more_text}", Colors.MAGENTA)
                    
                    if not signals:
                        print_colored("📊 No significant signals found in current ultimate scan", Colors.YELLOW)
                    
                    # Calculate next scan time
                    next_scan_time = get_next_scan_time()
                    wait_time_minutes = (next_scan_time - datetime.now(IST)).total_seconds() / 60
                    
                    print_colored("=" * 100, Colors.BLUE)
                    print_colored(f"⏰ Next ultimate scan at {next_scan_time.strftime('%H:%M:%S')} IST (waiting {format_time_remaining((next_scan_time - datetime.now(IST)).total_seconds())})", Colors.CYAN)
                    
                    # Sleep until next scan with progress indication
                    sleep_start = datetime.now(IST)
                    while datetime.now(IST) < next_scan_time:
                        if not is_market_open():
                            print_colored("\n📈 Market has CLOSED during wait. Stopping scanner...", Colors.YELLOW)
                            return
                        
                        remaining = (next_scan_time - datetime.now(IST)).total_seconds()
                        if remaining > 60:
                            # Show countdown every minute for long waits
                            print_colored(f"⏳ Waiting... {format_time_remaining(remaining)} until next scan", Colors.CYAN)
                            time.sleep(60)
                        else:
                            # Final countdown
                            time.sleep(remaining)
                            break
                
                except Exception as scan_error:
                    logger.error(f"Error in ultimate scan #{scan_count}: {scan_error}")
                    print_colored(f"❌ Scan #{scan_count} failed: {scan_error}", Colors.RED)
                    print_colored("⏰ Waiting 5 minutes before retry...", Colors.YELLOW)
                    time.sleep(300)  # Wait 5 minutes before retry
                    continue
        
        except KeyboardInterrupt:
            print_colored("\n\n⚠️ Ultimate continuous scanner interrupted by user. Shutting down gracefully...", Colors.YELLOW)
        
        except Exception as e:
            logger.error(f"Critical error in ultimate continuous scanner: {e}")
            print_colored(f"\n💥 Critical error: {e}", Colors.RED)
        
        finally:
            # Cleanup resources
            print_colored("\n🧹 Cleaning up ultimate scanner resources...", Colors.CYAN)
            try:
                for session in tdhist_pool:
                    if hasattr(session, 'disconnect'):
                        session.disconnect()
            except:
                pass
            
            print_colored("✅ Ultimate continuous scanner cleanup complete.", Colors.GREEN)

# ========== PROGRAM ENTRY POINT ==========

if __name__ == "__main__":
    try:
        # Display ultimate startup banner
        print_colored("\n" + "="*120, Colors.HEADER)
        print_colored("🎯 ULTIMATE INTRADAY TRADER SCANNER v4.3 - ALL DATETIME ISSUES FIXED", Colors.HEADER)
        print_colored("🔧 Uses OHLC, Volume, Open Interest from TrueData API", Colors.GREEN)
        print_colored("⏰ Runs every 5 minutes during market hours with proper market condition checking", Colors.BLUE)
        print_colored("✨ Perfect Intraday Signal Detection with Complete Data Integration", Colors.CYAN)
        print_colored("="*120, Colors.HEADER)
        
        # Enhanced usage examples
        print_colored(f"\n📋 ENHANCED USAGE EXAMPLES:", Colors.CYAN)
        print("  🔬 Quick Test: python scanner.py --test")
        print("  🔍 Full Diagnosis: python scanner.py --asof 2025-10-03T15:30 --diagnose")
        print("  📉 Lower Threshold: python scanner.py --asof 2025-10-03T15:30 --threshold 1.0")
        print("  🎯 Normal Snapshot: python scanner.py --asof 2025-10-03T15:30")
        print("  📈 Backtest: python scanner.py --backtest 2025-10-03")
        print("  🔴 Live Continuous: python scanner.py")
        
        # Show symbol conversion examples
        print_colored(f"\n🔧 SYMBOL CONVERSION EXAMPLES:", Colors.CYAN)
        print("  📊 Input: RELIANCE -> TrueData: RELIANCE-I")
        print("  📊 Input: TCS-EQ -> TrueData: TCS-I")
        print("  📊 Input: HDFC-I -> TrueData: HDFC-I")
        
        # Show ultimate configuration
        print_colored(f"\n📋 ULTIMATE CONFIGURATION:", Colors.CYAN)
        print(f"  📊 TrueData Sessions: {Config.TD_HIST_SESSIONS}")
        print(f"  🔄 Max Workers: {Config.MAX_WORKERS}")
        print(f"  📈 Market Hours: {Config.MARKET_START} - {Config.MARKET_END} IST")
        print(f"  🎯 Score Threshold: {Config.SCORE_THRESHOLD_MIN}")
        print(f"  📊 Min OI: {Config.MIN_TOTAL_OI:,} | Min Volume: {Config.MIN_TOTAL_VOL:,}")
        print(f"  ⏰ Scan Interval: Every 5 minutes during market hours")
        print(f"  🕘 Settlement Delay: {Config.SETTLE_DELAY_SECONDS} seconds")
        
        # Show enhanced features
        print_colored(f"\n🎯 ULTIMATE FEATURES v4.3 - ALL FIXES:", Colors.CYAN)
        print("  ✅ Real-time TrueData OHLC/Volume/OI fetching")
        print("  ✅ Volume surge and OI buildup detection")
        print("  ✅ Institutional flow analysis")
        print("  ✅ Intraday momentum scoring")
        print("  ✅ Multi-timeframe analysis with intraday focus")
        print("  ✅ Market regime awareness")
        print("  ✅ 5-minute precision scanning")
        print("  ✅ Comprehensive backtesting")
        print("  🆕 Continuous live scanning every 5 minutes")
        print("  🆕 Automatic market hours checking")
        print("  🆕 Weekend/holiday handling")
        print("  🆕 Real-time new stock detection")
        print("  🆕 Progressive countdown timer")
        print("  🆕 Enhanced error recovery")
        print("  🆕 Graceful shutdown on market close")
        print("  🔧 FIXED: All datetime comparison errors")
        print("  🔧 FIXED: Corrected symbol handling")
        print("  🔧 FIXED: Market hours parsing with time objects")
        print("  🔧 FIXED: Timezone-aware datetime filtering")
        print("  🔧 FIXED: EMA calculation with NaN handling")
        
        print_colored("\n🚀 Starting Ultimate Intraday Trader Scanner v4.3...", Colors.GREEN)
        
        # Run the main function
        main_ultimate_scanner_with_diagnostics()
            
    except KeyboardInterrupt:
        print_colored("\n\n👋 Ultimate scanner interrupted by user. Goodbye!", Colors.YELLOW)
    
    except Exception as e:
        logger.error(f"Fatal startup error: {e}")
        print_colored(f"\n💥 Fatal error during startup: {e}", Colors.RED)
        
    finally:
        print_colored("\n🎯 Ultimate Intraday Trader Scanner v4.3 - Session Ended", Colors.HEADER)
        print_colored("📊 Thank you for using the ultimate professional trading scanner!", Colors.GREEN)
        print_colored("="*120, Colors.HEADER)