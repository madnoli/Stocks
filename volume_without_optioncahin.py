# ==============================================================================
# ULTIMATE TECHNICAL SCANNER v4.3 - (MODIFIED: OPTION CHAIN REMOVED)
# TrueData: Uses symbols with -I suffix (RELIANCE-I, TCS-I)
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
import requests
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
# ======== ULTIMATE Configuration for Technical Scanner ========
class Config:
    # TrueData Configuration
    TDUSERNAME = os.getenv("TRUEDATA_USER", "tdwsp751")
    TDPASSWORD = os.getenv("TRUEDATA_PASS", "raj@751")
    
    # Market Timing Configuration
    MARKET_START = "09:15" # IST
    FIRST_RUN_AT = "09:20" # IST; First scan after 09:15-09:20 candle
    FIRST_SCAN_DELAY = 15 # Wait 15 seconds after 09:20 for settlement
    MARKET_END = "15:30" # IST
    SETTLE_DELAY_SECONDS = 15 # wait after bar close for data settlement
    
    # Performance Configuration
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "32"))
    TD_HIST_SESSIONS = int(os.getenv("TD_HIST_SESSIONS", "5"))
    SHARES_FILE = os.getenv("SHARES_FILE", "shares.txt")
    BENCHMARK_INDEX = "NIFTY 50"
    
    # Backtesting Configuration
    BACKTEST_INTERVAL_MINUTES = 5
    BACKTEST_START_DELAY = 5
    BACKTEST_TOP_DISPLAY = 15
    
    MIN_VOL_SURGE_THRESHOLD = 1.5 # Volume surge multiplier
    
    # Enhanced Indicator Group Weights
    GROUP_WEIGHTS = {
        "Trend": 2.5,
        "Momentum": 3.0,
        "Volume": 2.5,
        "Volatility": 2.2,
        "OI": 2.0,
    }
    
    # Enhanced Individual Indicator Weights
    INDICATOR_WEIGHTS = {
        # Trend indicators
        "MA_Slope": 2.0, "ADX": 2.2, "VWAP": 1.8, "EMA": 1.7, "MACD_Trend": 2.0,
        
        # Momentum indicators
        "RSI": 2.5, "Stochastic": 2.0, "CCI": 1.8, "ROC": 2.0, "WilliamsR": 1.5,
        
        # Volume indicators
        "VolumeSurge": 3.0, "OBV": 2.0, "CMF": 2.2, "RelVol": 2.0,
        
        # Volatility indicators
        "VolatilityExpansion": 2.8, "Bollinger": 2.0,
        
        # OI indicators
        "OptionBuyerMomentum": 3.0, "OIChange": 2.5, "VolumeOISync": 2.2,
    }
    
    # Enhanced Scoring & Signal Thresholds
    SCORE_THRESHOLD_MIN = 3.0
    SIGNAL_THRESHOLDS = {
        'Perfect Buy': 70.0,
        'Perfect Sell': -70.0,
        'Very Strong Buy': 55.0,
        'Strong Buy': 30.0,
        'Buy Signal': 15.0,
        'Very Strong Sell': -55.0,
        'Strong Sell': -30.0,
        'Sell Signal': -15.0,
    }
    
    # Market Regime Multipliers
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
TIMEFRAME_WEIGHTS = {15: 3.0, 5: 2.5, 30: 2.0, 60: 1.5, 1440: 1.0}
# Silence noisy loggers
for noisy in ("truedata", "truedata.history", "truedata_ws", "websocket", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)
# State management
previous_scan_results = {}
previous_oi_data = {}
previous_volume_data = {}
intraday_volume_data = {}
intraday_oi_data = {}
option_chain_cache = {}
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
        # Remove -EQ and add -I
        return symbol.replace('-EQ', '-I')
    elif symbol.endswith('-I'):
        # Already in correct format
        return symbol
    else:
        # Add -I suffix
        return f"{symbol}-I"
def convert_to_localhost_symbol(symbol):
    """Convert symbol to clean format (remove suffix)"""
    if symbol.endswith('-I'):
        return symbol.replace('-I', '')
    elif symbol.endswith('-EQ'):
        return symbol.replace('-EQ', '')
    else:
        return symbol
def normalize_symbol_for_display(symbol):
    """Convert symbol for display purposes (clean without suffix)"""
    return convert_to_localhost_symbol(symbol)

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
def option_buyer_momentum(df):
    """Calculate option buyer momentum"""
    if len(df) < 20:
        return None
    
    price_mom = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
    vol_surge_val = volume_surge(df, lookback=20).iloc[-1] if len(df) > 20 else 0
    oi_buildup = detect_oi_buildup(df, lookback=20)
    
    if oi_buildup is None:
        return None
    
    combined_score = (price_mom * 0.4) + (vol_surge_val * 0.3) + (oi_buildup * 0.3)
    return min(max(combined_score, -100), 100)
# ========== ENHANCED SCORING ENGINE ==========
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
    
    if df is None or df.empty or len(df) < 15:
        return scores
    
    try:
        # --- Enhanced Trend Group ---
        adx, pdi, ndi = calculate_adx(df)
        if not adx.empty and len(adx) > 3 and adx.iloc[-1] > 15:
            trend_strength = adx.iloc[-1] / 50.0 # Normalize
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
        
        # Enhanced MACD
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
        
        # --- Enhanced Momentum Group ---
        rsi = calculate_rsi(df)
        if not rsi.empty and len(rsi) > 0:
            rsi_val = rsi.iloc[-1]
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
        
        # Enhanced Stochastic
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
            scores['CCI'] = normalize_score(cci_val, (100, 250), (-100, -250), (-1.8, 1.8))
        
        # Enhanced ROC
        roc = calculate_roc(df)
        if not roc.empty and len(roc) > 0:
            scores['ROC'] = normalize_score(roc.iloc[-1], (1.0, 3.0), (-1.0, -3.0), (-2.0, 2.0))
        
        # Enhanced Williams %R
        wr = williams_r(df)
        if not wr.empty and len(wr) > 0:
            scores['WilliamsR'] = normalize_score(wr.iloc[-1], (-80, -50), (-20, -5), (-1.5, 1.5))
        
        # --- Enhanced Volume Group ---
        zscore = volume_surge(df, lookback=20)
        if not zscore.empty and len(zscore) > 1:
            price_up = df['Close'].iloc[-1] > df['Close'].iloc[-2]
            zscore_val = zscore.iloc[-1]
            
            if price_up and zscore_val > Config.MIN_VOL_SURGE_THRESHOLD:
                scores['VolumeSurge'] = min(3.0, zscore_val * 1.5)
            elif not price_up and zscore_val > Config.MIN_VOL_SURGE_THRESHOLD:
                scores['VolumeSurge'] = max(-3.0, -zscore_val * 1.5)
        
        # Enhanced OBV
        obv_line = calculate_obv(df)
        if len(obv_line) > 5:
            obv_slope = slope(obv_line, 5)
            scores['OBV'] = normalize_score(obv_slope, (1000, 1000000), (-1000, -1000000), (-2.0, 2.0))
        
        # Enhanced CMF
        cmf20 = cmf(df, period=20)
        if not cmf20.empty and len(cmf20) > 0:
            scores['CMF'] = normalize_score(cmf20.iloc[-1], (0.15, 0.35), (-0.15, -0.35), (-2.2, 2.2))
        
        # Enhanced Relative Volume
        rv = relative_volume(df, lookback=min(50, len(df)//2))
        if not rv.empty and len(rv) > 0:
            rv_val = rv.iloc[-1]
            scores['RelVol'] = normalize_score(rv_val, (1.5, 3.0), (0.5, 0.3), (-2.0, 2.0))
        
        # --- Enhanced Volatility Group ---
        atr_val = atr(df, period=14)
        if len(atr_val) > 20:
            atr_ma = atr_val.rolling(20).mean()
            if len(atr_ma) > 0 and atr_ma.iloc[-1] != 0:
                atr_ratio = atr_val.iloc[-1] / atr_ma.iloc[-1]
                atr_slope_ratio = (atr_val.iloc[-1] / atr_val.iloc[-5]) if len(atr_val) >= 5 and atr_val.iloc[-5] > 0 else 1
                
                if atr_ratio > 1.2 and atr_slope_ratio > 1.1:
                    price_direction = 1 if df['Close'].iloc[-1] > df['Close'].iloc[-5] else -1
                    volatility_strength = min(2.8, (atr_ratio - 1) * 2.8)
                    scores['VolatilityExpansion'] = volatility_strength * price_direction
        
        # Enhanced Bollinger Bands
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(df)
        if not bb_upper.empty and not bb_lower.empty:
            close_price = df['Close'].iloc[-1]
            if close_price > bb_upper.iloc[-1]:
                bb_strength = (close_price - bb_upper.iloc[-1]) / (bb_upper.iloc[-1] - bb_middle.iloc[-1])
                scores['Bollinger'] = min(2.0, bb_strength * 2.0)
            elif close_price < bb_lower.iloc[-1]:
                bb_strength = (bb_lower.iloc[-1] - close_price) / (bb_middle.iloc[-1] - bb_lower.iloc[-1])
                scores['Bollinger'] = max(-2.0, -bb_strength * 2.0)
        
        # --- Enhanced OI Group (only if real OI exists) ---
        oi_buildup = detect_oi_buildup(df, 20)
        if oi_buildup is not None:
            scores['OIChange'] = normalize_score(oi_buildup, (15, 40), (-15, -40), (-2.5, 2.5))
        
        vol_oi_sync = volume_oi_sync_analysis(df)
        if vol_oi_sync is not None:
            scores['VolumeOISync'] = normalize_score(vol_oi_sync, (20, 50), (-20, -50), (-2.2, 2.2))
        
        opt_buyer_mom = option_buyer_momentum(df)
        if opt_buyer_mom is not None:
            scores['OptionBuyerMomentum'] = normalize_score(opt_buyer_mom, (25, 60), (-25, -60), (-3.0, 3.0))
        
    except Exception as e:
        logger.error(f"Error calculating technical indicator scores: {e}")
    
    return scores
def analyze_ultimate_signals(timeframe_data, market_regime='neutral'):
    """Ultimate signal analysis combining technical data"""
    total_score, total_weight = 0.0, 0.0
    group_scores = defaultdict(float)
    group_weights = defaultdict(float)
    
    # Process technical indicators from multiple timeframes
    for tf_min, df in timeframe_data.items():
        if df is None or df.empty or len(df) < 15:
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
                        (group == 'OI' and any(term in indicator for term in ['OI', 'Option']))
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
    
    # Signal classification
    if normalized_score >= Config.SIGNAL_THRESHOLDS['Perfect Buy']:
        signal = 'Perfect Buy'
    elif normalized_score >= Config.SIGNAL_THRESHOLDS['Very Strong Buy']:
        signal = 'Very Strong Buy'
    elif normalized_score >= Config.SIGNAL_THRESHOLDS['Strong Buy']:
        signal = 'Strong Buy'
    elif normalized_score >= Config.SIGNAL_THRESHOLDS['Buy Signal']:
        signal = 'Buy Signal'
    elif normalized_score <= Config.SIGNAL_THRESHOLDS['Perfect Sell']:
        signal = 'Perfect Sell'
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
        return (9, 15) # Default fallback
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
            
            sleep_for = max(0.0, 0.0 / self.rate)
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
        
        out.rename(columns={c: str(c).lower() for c in out.columns}, inplace=True)
        
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
        
        if "Date" not in out.columns:
            if isinstance(out.index, pd.DatetimeIndex):
                out["Date"] = out.index
            else:
                logger.warning(f"No Date column found for {symbol}")
                return None
        
        if "Volume" not in out.columns:
            out["Volume"] = 0
        
        if "OpenInterest" in out.columns:
            out["OpenInterest"] = pd.to_numeric(out["OpenInterest"], errors="coerce")
            out["OpenInterest"] = out["OpenInterest"].fillna(0)
        
        try:
            out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
            out = out.dropna(subset=["Date"])
            
            if len(out) == 0:
                logger.warning(f"No valid dates found for {symbol}")
                return None
        except Exception as date_e:
            logger.error(f"Date conversion error for {symbol}: {date_e}")
            return None
        
        try:
            if pd.api.types.is_datetime64tz_dtype(out["Date"]):
                out["Date"] = out["Date"].dt.tz_convert(IST)
            else:
                out["Date"] = out["Date"].dt.tz_localize(IST, ambiguous='infer', nonexistent='shift_forward')
        
        except Exception as tz_e:
            logger.warning(f"Timezone handling issue for {symbol}: {tz_e}")
            try:
                out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
                out = out.dropna(subset=["Date"])
            except:
                logger.error(f"Failed to process dates for {symbol}")
                return None
        
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")
        
        out = out.dropna(subset=["Open", "High", "Low", "Close"])
        
        if len(out) == 0:
            logger.warning(f"No valid OHLC data for {symbol}")
            return None
        
        out = out.sort_values("Date").set_index("Date")
        
        out = out[~out.index.duplicated(keep='last')]
        
        if len(out) == 0:
            return None
        
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
    td_symbol = convert_to_truedata_symbol(symbol_orig)
    bar_size = BAR_SIZE_MAP.get(timeframe_minutes)
    duration_str = DURATION_MAP.get(timeframe_minutes)
    
    if not bar_size or not duration_str:
        return symbol_orig, timeframe_minutes, None
    
    try:
        limiter.acquire()
        
        if up_to_time and isinstance(up_to_time, datetime):
            if up_to_time.tzinfo is None:
                up_to_time_aware = IST.localize(up_to_time)
            else:
                up_to_time_aware = up_to_time.astimezone(IST)
            
            dur_parts = duration_str.split()
            if len(dur_parts) == 2:
                try:
                    dur_num, dur_unit = int(dur_parts[0]), dur_parts[1]
                    if dur_unit.upper() == 'D':
                        start_time_naive = up_to_time_aware.replace(tzinfo=None) - timedelta(days=dur_num)
                        start_time_aware = IST.localize(start_time_naive)
                    else:
                        start_time_naive = up_to_time_aware.replace(tzinfo=None) - timedelta(days=dur_num)
                        start_time_aware = IST.localize(start_time_naive)
                except (ValueError, TypeError):
                    start_time_naive = up_to_time_aware.replace(tzinfo=None) - timedelta(days=30)
                    start_time_aware = IST.localize(start_time_naive)
            else:
                start_time_naive = up_to_time_aware.replace(tzinfo=None) - timedelta(days=30)
                start_time_aware = IST.localize(start_time_naive)
            
            df_raw = hist.get_historic_data(
                td_symbol,
                start_time=start_time_aware.replace(tzinfo=None),
                end_time=up_to_time_aware.replace(tzinfo=None),
                bar_size=bar_size
            )
        else:
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
    
    return {s: d for s, d in stock_multi_data.items() if len(d) >= 1}
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
        
        try:
            ema20_series = ema(df['Close'], 20)
            ema50_series = ema(df['Close'], 50)
            
            if ema20_series.empty or ema50_series.empty or len(ema20_series) == 0 or len(ema50_series) == 0:
                logger.warning("EMA calculation failed for market regime")
                return 'neutral'
            
            ema20_val = ema20_series.dropna().iloc[-1] if len(ema20_series.dropna()) > 0 else None
            ema50_val = ema50_series.dropna().iloc[-1] if len(ema50_series.dropna()) > 0 else None
            close = df['Close'].dropna().iloc[-1] if len(df['Close'].dropna()) > 0 else None
            
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
    frames = [tf_data.get(t) for t in (5, 15, 30) if tf_data.get(t) is not None and len(tf_data.get(t)) >= 20]
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
        
        if cmf_last > 0.1 and rv_last > 1.5:
            votes += 2
        elif cmf_last > 0.05 and rv_last > 1.2:
            votes += 1
        elif cmf_last < -0.1 and rv_last > 1.5:
            votes -= 2
        elif cmf_last < -0.05 and rv_last > 1.2:
            votes -= 1
    
    if votes >= 3:
        return "Strong Institutional Accumulation"
    elif votes >= 2:
        return "Institutional Accumulation"
    elif votes <= -3:
        return "Strong Institutional Distribution"
    elif votes <= -2:
        return "Institutional Distribution"
    else:
        return "Mixed/Neutral"
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
    """Extract enhanced 5-minute volume and OI data"""
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
        
        current_volume, current_oi, vol_change_pct, oi_change_pct = calculate_5min_volume_oi_changes(
            df_slice, symbol, df_slice.index[-1]
        )
        
        if abs(vol_change_pct) < 0.1 and abs(oi_change_pct) < 0.1:
            prev_volume = intraday_volume_data.get(symbol, None)
            prev_oi = intraday_oi_data.get(symbol, None)
            
            if prev_volume is not None and prev_volume > 0 and current_volume and current_volume > 0:
                vol_change_pct = ((current_volume - prev_volume) / prev_volume) * 100
            
            if prev_oi is not None and prev_oi > 0 and current_oi and current_oi > 0:
                oi_change_pct = ((current_oi - prev_oi) / prev_oi) * 100
        
        intraday_volume_data[symbol] = current_volume if isinstance(current_volume, int) else 0
        intraday_oi_data[symbol] = current_oi if isinstance(current_oi, int) else 0
        
        def format_number(val):
            if isinstance(val, int):
                if val > 10000000:
                    return f"{val/1000000:.1f}M"
                elif val > 100000:
                    return f"{val/1000:.0f}K"
                elif val > 999:
                    return f"{val:,}"
                else:
                    return str(val)
            return "N/A"
        
        current_volume_display = format_number(current_volume)
        current_oi_display = format_number(current_oi)
        
        volume_change_legacy = f"{vol_change_pct:+.1f}%" if isinstance(vol_change_pct, (int, float)) and abs(vol_change_pct) > 0.1 else "N/A"
        oi_change_legacy = f"{oi_change_pct:+.1f}%" if isinstance(oi_change_pct, (int, float)) and abs(oi_change_pct) > 0.1 else "N/A"
        
        return {
            'current_volume': current_volume_display,
            'current_oi': current_oi_display,
            'volume_change_pct': vol_change_pct if isinstance(vol_change_pct, (int, float)) and abs(vol_change_pct) > 0.1 else 0,
            'oi_change_pct': oi_change_pct if isinstance(oi_change_pct, (int, float)) and abs(oi_change_pct) > 0.1 else 0,
            'volume': current_volume_display,
            'oi': current_oi_display,
            'volume_change': volume_change_legacy,
            'oi_change': oi_change_legacy,
            '_raw_volume': current_volume if isinstance(current_volume, int) else 0,
            '_raw_oi': current_oi if isinstance(current_oi, int) else 0
        }
        
    except Exception as e:
        logger.error(f"Error extracting 5-min data for {symbol}: {e}")
        return {
            'current_volume': 'N/A', 'current_oi': 'N/A',
            'volume_change_pct': 0, 'oi_change_pct': 0,
            'volume': 'N/A', 'oi': 'N/A',
            'volume_change': 'N/A', 'oi_change': 'N/A'
        }
# ========== ULTIMATE SCANNER LOGIC ==========
def run_ultimate_scan_at_time(time_point_aware, stocks, market_regime, is_live=False):
    """Ultimate scan with proper datetime filtering"""
    
    truedata_stocks = [convert_to_truedata_symbol(s) for s in stocks]
    
    stock_multi_data = prefetch_all(truedata_stocks, max_workers=Config.MAX_WORKERS) if is_live else \
                        prefetch_all_timeaware(truedata_stocks, time_point_aware, max_workers=Config.MAX_WORKERS)
    
    print_colored(f"✅ Data fetch complete. TrueData: {len(stock_multi_data)} stocks", Colors.GREEN)
    print_colored(f"Running ultimate analysis (Regime: {market_regime.upper()})...", Colors.GREEN)
    
    signals_this_scan = []
    current_symbols = set()
    
    for truedata_symbol, timeframe_data in stock_multi_data.items():
        clean_symbol = convert_to_localhost_symbol(truedata_symbol)
        current_symbols.add(clean_symbol)
        
        filtered_timeframes = {}
        for tf, df in timeframe_data.items():
            if df is not None and not df.empty:
                if is_live:
                    df_slice = df
                else:
                    if time_point_aware and isinstance(df.index, pd.DatetimeIndex):
                        try:
                            if time_point_aware.tzinfo is None:
                                time_point_aware = IST.localize(time_point_aware)
                            
                            if df.index.tz is None:
                                df_with_tz = df.copy()
                                df_with_tz.index = df_with_tz.index.tz_localize(IST, ambiguous='infer', nonexistent='shift_forward')
                                df_slice = df_with_tz[df_with_tz.index <= time_point_aware]
                            else:
                                df_slice = df[df.index <= time_point_aware]
                        except Exception as filter_e:
                            logger.warning(f"Datetime filtering issue for {clean_symbol}: {filter_e}")
                            df_slice = df
                    else:
                        df_slice = df
                
                if not df_slice.empty and len(df_slice) >= 15:
                    filtered_timeframes[tf] = df_slice
        
        if len(filtered_timeframes) < 1:
            continue
        
        tf_5min = filtered_timeframes.get(5)
        if tf_5min is not None and len(tf_5min) >= 20:
            vol_ma = tf_5min["Volume"].rolling(20).mean().iloc[-1]
            current_vol = tf_5min["Volume"].iloc[-1]
            if current_vol < 3 * vol_ma:
                continue
        else:
            logger.warning(f"Skipping {clean_symbol}: Insufficient 5-min data for volume filter")
            continue
        
        signal, score, sub_scores = analyze_ultimate_signals(
            filtered_timeframes, market_regime
        )
        
        if abs(score) >= Config.SCORE_THRESHOLD_MIN:
            flow_tag = enhanced_institutional_flow_analysis(filtered_timeframes)
            
            tf_5min = filtered_timeframes.get(5)
            if tf_5min is not None:
                oi_vol_data = extract_5min_volume_oi_data(tf_5min, clean_symbol, time_point_aware, is_live=is_live)
            else:
                main_tf_data = filtered_timeframes.get(15, filtered_timeframes.get(30, list(filtered_timeframes.values())[0]))
                oi_vol_data = extract_5min_volume_oi_data(main_tf_data, clean_symbol, time_point_aware, is_live=is_live)
            
            action = signal.replace(" Signal", "").replace(" Buy", " Trade").replace(" Sell", " Trade")
            
            result = {
                'symbol': clean_symbol,
                'signal': signal,
                'score': score,
                'sub_scores': sub_scores,
                'flow': flow_tag,
                'action': action,
                **oi_vol_data,
            }
            
            signals_this_scan.append(result)
    
    return signals_this_scan, current_symbols

def determine_ultimate_action(signal, score):
    """Determine a simplified action based on the technical signal"""
    if 'Perfect' in signal:
        return f"🎯 {signal}"
    elif 'Very Strong' in signal:
        return f"🚀 {signal}"
    elif 'Strong' in signal:
        return f"📈 {signal}"
    else:
        return f"🤔 {signal}"

# ========== ENHANCED TABLE DISPLAY ==========
def create_ultimate_option_table(data, title, new_stocks=None, show_time=None):
    """Simplified table for technical-only signals"""
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
        table.add_column("Stock", style="bold white", width=12, justify="left")
        table.add_column("Signal", style="bold", width=18, justify="center")
        table.add_column("Score", style="bold", width=8, justify="right")
        table.add_column("Volume", style="cyan", width=10, justify="right")
        table.add_column("Vol %", style="yellow", width=8, justify="right")
        table.add_column("Flow", style="green", width=25, justify="left")
        
        for item in data:
            symbol = item['symbol']
            is_new = new_stocks and symbol in new_stocks
            
            if item['score'] > 60: signal_style = "bold bright_green"
            elif item['score'] > 30: signal_style = "bold green"
            elif item['score'] > 0: signal_style = "green"
            elif item['score'] < -60: signal_style = "bold bright_red"
            elif item['score'] < -30: signal_style = "bold red"
            else: signal_style = "red"
            
            stock_style = f"[bold bright_magenta]{symbol} ✨[/bold bright_magenta]" if is_new else symbol
            
            vol_chg = item.get('volume_change_pct', 0)
            vol_chg_display = f"{vol_chg:+.1f}%" if vol_chg != 0 else "N/A"
            
            table.add_row(
                stock_style,
                f"[{signal_style}]{item['signal']}[/{signal_style}]",
                f"[bold]{item['score']:.1f}[/bold]",
                item.get('current_volume', 'N/A'),
                vol_chg_display,
                item.get('flow', 'Unknown'),
            )
        
        caption_text = f"Scan Time: {show_time}" if show_time else ""
        if show_time:
            console.print(f"\n[bold magenta]{title} - {show_time}[/bold magenta]")
        else:
            console.print(f"\n[bold magenta]{title}[/bold magenta]")
        console.print(table)
    
    else: # Fallback for non-rich console
        if show_time:
            print_colored(f"\n{title} - {show_time}", Colors.HEADER)
        else:
            print_colored(f"\n{title}", Colors.HEADER)
        
        print_colored("="*100, Colors.BLUE)
        header = f"{'Stock':<12} | {'Signal':<18} | {'Score':>8} | {'Volume':>10} | {'Vol %':>8} | {'Flow':<25}"
        print_colored(header, Colors.BOLD)
        print_colored("-"*100, Colors.BLUE)
        
        for item in data:
            symbol = item['symbol']
            is_new = new_stocks and symbol in new_stocks
            
            vol_chg = item.get('volume_change_pct', 0)
            vol_chg_str = f"{vol_chg:+.1f}%" if vol_chg != 0 else "N/A"
            
            row = f"{symbol:<12} | {item['signal']:<18} | {item['score']:>8.1f} | {str(item.get('current_volume', 'N/A')):>10} | {vol_chg_str:>8} | {item.get('flow', 'Unknown'):<25}"
            
            if is_new:
                print_colored(row + " ✨ NEW!", Colors.MAGENTA)
            else:
                print(row)
        
        print_colored("="*100, Colors.BLUE)

def create_ultimate_summary_panel(signals):
    """Create summary panel with key statistics"""
    if not signals:
        return
    
    total_signals = len(signals)
    perfect_setups = len([s for s in signals if 'Perfect' in s.get('signal', '')])
    strong_buys = len([s for s in signals if 'Strong Buy' in s.get('signal', '')])
    strong_sells = len([s for s in signals if 'Strong Sell' in s.get('signal', '')])

    if RICH_AVAILABLE:
        summary_text = f"""
[bold cyan]📊 TECHNICAL SCAN SUMMARY[/bold cyan]
[green]Total Signals: {total_signals}[/green]
[bright_green]Perfect Setups: {perfect_setups}[/bright_green]
[green]Strong Buys: {strong_buys}[/green]
[red]Strong Sells: {strong_sells}[/red]
        """
        panel = Panel(summary_text, title="Technical Scanner Stats", border_style="blue")
        console.print(panel)
    else:
        print_colored("\n📊 TECHNICAL SCAN SUMMARY", Colors.HEADER)
        print_colored("="*40, Colors.BLUE)
        print(f"Total Signals: {total_signals}")
        print(f"Perfect Setups: {perfect_setups}")
        print(f"Strong Buys: {strong_buys}")
        print(f"Strong Sells: {strong_sells}")
        print_colored("="*40, Colors.BLUE)
# ========== DIAGNOSTIC FUNCTIONS ==========
def run_diagnostic_scan(time_point_aware, stocks, market_regime, is_live=False):
    """Diagnostic scan for the technical-only scanner"""
    
    print_colored("🔍 RUNNING DIAGNOSTIC SCAN...", Colors.YELLOW)
    
    print_colored("📊 Step 1: Testing TrueData fetch for first 5 stocks...", Colors.CYAN)
    test_stocks = stocks[:5]
    
    truedata_test_stocks = [convert_to_truedata_symbol(s) for s in test_stocks]
    stock_multi_data = prefetch_all(truedata_test_stocks, max_workers=5) if is_live else \
                        prefetch_all_timeaware(truedata_test_stocks, time_point_aware, max_workers=5)
    
    print(f" ✅ TrueData received data for {len(stock_multi_data)} stocks")
    for symbol, timeframes in stock_multi_data.items():
        clean_display = convert_to_localhost_symbol(symbol)
        print(f" 📈 {clean_display} ({symbol}): {list(timeframes.keys())} timeframes")
        for tf, df in timeframes.items():
            if df is not None:
                print(f" {tf}min: {len(df)} candles, latest: {df.index[-1] if len(df) > 0 else 'No data'}")
    
    print_colored("\n📊 Step 2: Testing signal analysis...", Colors.CYAN)
    signals_found = 0
    
    for truedata_symbol, timeframe_data in list(stock_multi_data.items())[:3]:
        clean_symbol = convert_to_localhost_symbol(truedata_symbol)
        print(f"\n 📈 Analyzing {clean_symbol}...")
        
        filtered_timeframes = {}
        for tf, df in timeframe_data.items():
            if df is not None and not df.empty:
                df_slice = df[df.index <= time_point_aware] if time_point_aware and not is_live else df
                if not df_slice.empty and len(df_slice) >= 15:
                    filtered_timeframes[tf] = df_slice
                    print(f" ✅ {tf}min: {len(df_slice)} candles")
        
        if len(filtered_timeframes) >= 1:
            signal, score, sub_scores = analyze_ultimate_signals(
                filtered_timeframes, market_regime
            )
            
            print(f" 📊 Signal: {signal}, Score: {score:.2f}")
            print(f" 📊 Sub-scores: {sub_scores}")
            
            if abs(score) >= 1.0:
                signals_found += 1
                print(f" ✅ Would generate signal with threshold 1.0!")
        else:
            print(f" ❌ Insufficient data for {clean_symbol}")
    
    print_colored(f"\n🎯 DIAGNOSTIC SUMMARY:", Colors.HEADER)
    print(f" 📊 TrueData Stocks Fetched: {len(stock_multi_data)}")
    print(f" 📈 Potential Signals (threshold 1.0): {signals_found}")
    print(f" 🎯 Current Threshold: {Config.SCORE_THRESHOLD_MIN}")
    
    print_colored(f"\n💡 RECOMMENDATIONS:", Colors.YELLOW)
    if len(stock_multi_data) == 0:
        print(" ❌ No TrueData received - check credentials and connection.")
    elif signals_found == 0:
        print(" 📊 Try lowering score threshold: --threshold 1.0")
        print(" 📈 Market might be quiet. Try a different time or date.")
    else:
        print(" ✅ System appears to be working!")

def run_quick_test():
    """Quick test function"""
    print_colored("\n🔬 QUICK SYSTEM TEST", Colors.HEADER)
    
    try:
        test_symbol_orig = "RELIANCE"
        test_symbol_td = convert_to_truedata_symbol(test_symbol_orig)
        
        session = tdhist_pool[0]
        df_raw = session.get_historic_data(test_symbol_td, duration="5 D", bar_size="1 day")
        df = normalize_hist_df(df_raw, test_symbol_td)
        if df is not None and len(df) > 0:
            print(" ✅ TrueData connection: OK")
            print(f" Latest {test_symbol_td} data: {df.index[-1]} Price: {df['Close'].iloc[-1]:.2f}")
        else:
            print(" ❌ TrueData connection: No data received")
    except Exception as e:
        print(f" ❌ TrueData connection error: {e}")

# ========== ULTIMATE BACKTEST FUNCTION ==========
def run_ultimate_backtest(backtest_date, stocks):
    """Ultimate backtest with technical-only analysis"""
    global backtest_stock_history, intraday_volume_data, intraday_oi_data
    
    print_colored(f"\n🎯 STARTING TECHNICAL BACKTEST FOR {backtest_date}", Colors.HEADER)
    
    timestamps = generate_backtest_timestamps(backtest_date)
    total_scans = len(timestamps)
    print_colored(f"📅 Generated {total_scans} scan points from {timestamps[0].strftime('%H:%M')} to {timestamps[-1].strftime('%H:%M')}", Colors.CYAN)
    
    market_regime = get_market_regime(Config.BENCHMARK_INDEX)
    print_colored(f"📈 Market Regime: {market_regime.upper()}", Colors.BLUE)
    
    all_results = []
    backtest_stock_history = {}
    intraday_volume_data = {}
    intraday_oi_data = {}
    
    with tqdm(total=total_scans, desc="🎯 Technical Backtesting", ncols=120) as pbar:
        for i, scan_time in enumerate(timestamps):
            try:
                pbar.set_description(f"Scanning at {scan_time.strftime('%H:%M:%S')}")
                
                signals, current_symbols = run_ultimate_scan_at_time(scan_time, stocks, market_regime, is_live=False)
                
                previous_symbols = set(backtest_stock_history.keys())
                new_stocks = current_symbols - previous_symbols
                
                for symbol in current_symbols:
                    backtest_stock_history[symbol] = scan_time
                
                scan_result = {
                    'timestamp': scan_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'scan_number': i + 1,
                    'total_signals': len(signals),
                    'bullish_signals': len([s for s in signals if s['score'] > 0]),
                    'bearish_signals': len([s for s in signals if s['score'] < 0]),
                    'perfect_setups': len([s for s in signals if 'Perfect' in s.get('signal', '')]),
                    'new_stocks': list(new_stocks),
                    'signals': signals
                }
                
                all_results.append(scan_result)
                
                if signals:
                    signals.sort(key=lambda x: abs(x['score']), reverse=True)
                    top_bullish = [r for r in signals if r['score'] > 0][:Config.BACKTEST_TOP_DISPLAY]
                    top_bearish = [r for r in signals if r['score'] < 0][:Config.BACKTEST_TOP_DISPLAY]
                    
                    scan_time_str = scan_time.strftime('%H:%M')
                    
                    if top_bullish:
                        create_ultimate_option_table(top_bullish, f"🟢 TECHNICAL BULLISH SIGNALS", new_stocks, scan_time_str)
                    
                    if top_bearish:
                        create_ultimate_option_table(top_bearish, f"🔴 TECHNICAL BEARISH SIGNALS", new_stocks, scan_time_str)
                
                pbar.update(1)
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in backtest scan at {scan_time}: {e}")
                pbar.update(1)
                continue
    
    print_colored(f"\n📊 TECHNICAL BACKTEST SUMMARY FOR {backtest_date}", Colors.HEADER)
    print_colored("="*150, Colors.BLUE)
    
    total_signals = sum(r['total_signals'] for r in all_results)
    print(f"✅ Scans Completed: {len(all_results)}/{total_scans}")
    print(f"📊 Total Signals Found: {total_signals}")
    
    output_filename = f"{backtest_date}_technical_backtest_results.json"
    try:
        with open(output_filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        print_colored(f"\n💾 Results saved to: {output_filename}", Colors.GREEN)
    except Exception as e:
        logger.error(f"Could not save results: {e}")
    
    print_colored("🎯 Technical Backtesting Completed!", Colors.GREEN)
# ========== MAIN FUNCTION ==========
def main_ultimate_scanner_with_diagnostics():
    """Main function for the technical scanner"""
    global scan_count, previous_scan_results, intraday_volume_data, intraday_oi_data

    parser = argparse.ArgumentParser(description="Ultimate Technical Scanner v4.3")
    parser.add_argument("--asof", type=str, help="Historical snapshot: 2025-10-03T14:25")
    parser.add_argument("--backtest", type=str, help="Full day backtest: 2025-10-03")
    parser.add_argument("--test", action="store_true", help="Run quick system test")
    parser.add_argument("--diagnose", action="store_true", help="Run full diagnostic scan")
    parser.add_argument("--threshold", type=float, help="Override score threshold", default=None)
    args = parser.parse_args()
    
    try:
        with open(Config.SHARES_FILE, 'r') as f:
            stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(stocks)} symbols from {Config.SHARES_FILE}")
    except Exception:
        stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", "ICICIBANK", "SBIN", "TATAMOTORS", "AXISBANK", "ADANIPORTS"]
        logger.warning("Using sample stocks for testing")
    
    if args.threshold:
        Config.SCORE_THRESHOLD_MIN = args.threshold
        print_colored(f"🎯 Score threshold overridden to: {args.threshold}", Colors.YELLOW)
    
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
    
    if args.asof:
        try:
            if 'T' in args.asof:
                asof_ts = datetime.fromisoformat(args.asof.replace('Z', '+00:00'))
            else:
                date_part = datetime.strptime(args.asof, "%Y-%m-%d")
                asof_ts = date_part.replace(hour=15, minute=30)
            
            if asof_ts.tzinfo is None:
                asof_ts = IST.localize(asof_ts)
            else:
                asof_ts = asof_ts.astimezone(IST)
        except Exception as dt_e:
            logger.error(f"Invalid timestamp format: {args.asof}. Use YYYY-MM-DDTHH:MM or YYYY-MM-DD. Error: {dt_e}")
            return
        
        logger.info(f"Running snapshot for: {asof_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        market_regime = get_market_regime(Config.BENCHMARK_INDEX)
        print_colored(f"📈 Market Regime: {market_regime.upper()}", Colors.GREEN)
        
        if args.diagnose:
            run_diagnostic_scan(asof_ts, stocks, market_regime, is_live=False)
            return
        
        signals, _ = run_ultimate_scan_at_time(asof_ts, stocks, market_regime, is_live=False)
        signals.sort(key=lambda x: abs(x['score']), reverse=True)
        
        if len(signals) == 0:
            print_colored("\n⚠️ No signals found! Running automatic diagnostics...", Colors.YELLOW)
            run_diagnostic_scan(asof_ts, stocks[:5], market_regime, is_live=False)
            return

        top_bullish = [r for r in signals if r['score'] > 0][:20]
        top_bearish = [r for r in signals if r['score'] < 0][:20]
        
        if top_bullish:
            create_ultimate_option_table(top_bullish, "🟢 TOP 20 TECHNICAL BULLISH SIGNALS")
        if top_bearish:
            create_ultimate_option_table(top_bearish, "🔴 TOP 20 TECHNICAL BEARISH SIGNALS")
        
    else:
        # Continuous Live Scanner
        print_colored("\n🎯 STARTING CONTINUOUS LIVE TECHNICAL SCANNER", Colors.GREEN)
        
        intraday_volume_data = {}
        intraday_oi_data = {}
        scan_count = 0
        previous_scan_results = {}
        
        def is_market_open():
            now_ist = datetime.now(IST)
            start_time = dt_time(*parse_hhmm(Config.MARKET_START))
            end_time = dt_time(*parse_hhmm(Config.MARKET_END))
            return now_ist.weekday() < 5 and start_time <= now_ist.time() <= end_time
        
        def wait_for_market_open():
            while not is_market_open():
                print_colored("📈 Market is closed. Waiting...", Colors.YELLOW, end='\r')
                time.sleep(60)
            print_colored("🟢 Market is OPEN! Starting continuous scanning...", Colors.GREEN)

        wait_for_market_open()

        while True:
            if not is_market_open():
                print_colored("📈 Market has CLOSED. Stopping scanner...", Colors.YELLOW)
                break

            scan_count += 1
            now_ist = datetime.now(IST)
            print_colored(f"\n[{now_ist.strftime('%H:%M:%S')}] 🎯 LIVE SCAN #{scan_count}", Colors.HEADER)
            
            market_regime = get_market_regime(Config.BENCHMARK_INDEX)
            signals, current_symbols = run_ultimate_scan_at_time(now_ist, stocks, market_regime, is_live=True)
            
            new_stocks = current_symbols - set(previous_scan_results.keys()) if previous_scan_results else set()
            previous_scan_results = {s: True for s in current_symbols}
            
            signals.sort(key=lambda x: abs(x['score']), reverse=True)
            top_bullish = [r for r in signals if r['score'] > 0][:15]
            top_bearish = [r for r in signals if r['score'] < 0][:15]

            if signals:
                 create_ultimate_summary_panel(signals)
            if top_bullish:
                create_ultimate_option_table(top_bullish, "🟢 TOP 15 BULLISH SIGNALS", new_stocks)
            if top_bearish:
                create_ultimate_option_table(top_bearish, "🔴 TOP 15 BEARISH SIGNALS", new_stocks)
            if not signals:
                print_colored("📊 No significant signals in this scan.", Colors.YELLOW)

            next_scan_time = next_5min_boundary_ist(datetime.now(IST)) + timedelta(seconds=Config.SETTLE_DELAY_SECONDS)
            wait_seconds = (next_scan_time - datetime.now(IST)).total_seconds()
            print_colored(f"⏰ Next scan at {next_scan_time.strftime('%H:%M:%S')}. Waiting {format_time_remaining(wait_seconds)}...", Colors.CYAN)
            time.sleep(max(0, wait_seconds))

# ========== PROGRAM ENTRY POINT ==========
if __name__ == "__main__":
    try:
        main_ultimate_scanner_with_diagnostics()
    except KeyboardInterrupt:
        print_colored("\n\n👋 Scanner interrupted by user. Goodbye!", Colors.YELLOW)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print_colored(f"\n💥 Fatal error: {e}", Colors.RED)
    finally:
        print_colored("\n🎯 Technical Scanner Session Ended", Colors.HEADER)