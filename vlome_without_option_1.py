# ==============================================================================
# ULTIMATE TECHNICAL SCANNER v4.4 - BACKTEST-LIVE CONSISTENCY FIX
# TrueData: Uses symbols with -I suffix (RELIANCE-I, TCS-I)
# Runs every 5 minutes during market hours with proper market condition checking
# FIXED: Backtest now produces same results as live mode
# ==============================================================================
import os
import logging
logging.getLogger().setLevel(logging.CRITICAL)  # Suppress all library logs
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

import lz4.block  # Added to handle LZ4BlockError specifically

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
    MARKET_START = "09:15"  # IST
    FIRST_RUN_AT = "09:20"  # IST; First scan after 09:15-09:20 candle
    FIRST_SCAN_DELAY = 15  # Wait 15 seconds after 09:20 for settlement
    MARKET_END = "15:30"  # IST
    SETTLE_DELAY_SECONDS = 15  # wait after bar close for data settlement
    
    # Performance Configuration
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "32"))
    TD_HIST_SESSIONS = int(os.getenv("TD_HIST_SESSIONS", "5"))
    SHARES_FILE = os.getenv("SHARES_FILE", "shares.txt")
    BENCHMARK_INDEX = "NIFTY 50"  # Correct symbol for Nifty 50 index
    
    # Backtesting Configuration
    BACKTEST_INTERVAL_MINUTES = 5
    BACKTEST_START_DELAY = 5
    BACKTEST_TOP_DISPLAY = 15
    
    MIN_VOL_SURGE_THRESHOLD = 1.5  # Volume surge multiplier
    
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

# ========== FIX #1: SEPARATE STATE FOR BACKTEST VS LIVE ==========
class ScannerState:
    """Encapsulated state management to prevent cross-contamination between modes"""
    def __init__(self, mode='live'):
        self.mode = mode  # 'live' or 'backtest'
        self.previous_scan_results = {}
        self.previous_oi_data = {}
        self.previous_volume_data = {}
        self.intraday_volume_data = {}
        self.intraday_oi_data = {}
        self.scan_count = 0
        self.stock_history = {}
        self.current_scan_data = {}
    
    def reset(self):
        """Reset all state"""
        self.previous_scan_results = {}
        self.previous_oi_data = {}
        self.previous_volume_data = {}
        self.intraday_volume_data = {}
        self.intraday_oi_data = {}
        self.scan_count = 0
        self.stock_history = {}
        self.current_scan_data = {}

# Global scanner state (will be replaced with mode-specific states)
scanner_state = ScannerState(mode='live')

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

# ========== TD_HIST SESSION POOL ==========
tdhist_pool = {}
pool_lock = threading.Lock()

def pick_session(symbol, timeframe_min):
    """Pick TD_hist session from pool"""
    key = f"{symbol}_{timeframe_min}"
    with pool_lock:
        if key not in tdhist_pool:
            idx = len(tdhist_pool) % Config.TD_HIST_SESSIONS
            sess_name = f"session_{idx}"
            if sess_name not in tdhist_pool:
                tdhist_pool[sess_name] = TD_hist(Config.TDUSERNAME, Config.TDPASSWORD)
            return sess_name
        return tdhist_pool.get(key, "session_0")

def init_tdhist_pool():
    """Initialize TD_hist connection pool"""
    for i in range(Config.TD_HIST_SESSIONS):
        sess = TD_hist(Config.TDUSERNAME, Config.TDPASSWORD)
        tdhist_pool[f"session_{i}"] = sess
    logger.info(f"✅ Initialized {Config.TD_HIST_SESSIONS} TD_hist sessions")

# ========== NORMALIZE HIST DF ==========
def normalize_hist_df(df_raw, symbol_name=""):
    """Normalize TrueData historical data"""
    if df_raw is None or df_raw.empty:
        return None
    
    df = df_raw.copy()
    
    try:
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
            df.set_index('datetime', inplace=True)
        
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notnull()]
        
        if df.index.tz is None:
            df.index = df.index.tz_localize(IST, ambiguous='infer', nonexistent='shift_forward')
        else:
            df.index = df.index.tz_convert(IST)
        
        col_map = {}
        for col in df.columns:
            lower_col = col.lower()
            if 'open' in lower_col and 'openinterest' not in lower_col:
                col_map[col] = 'Open'
            elif 'high' in lower_col:
                col_map[col] = 'High'
            elif 'low' in lower_col:
                col_map[col] = 'Low'
            elif 'close' in lower_col:
                col_map[col] = 'Close'
            elif 'volume' in lower_col:
                col_map[col] = 'Volume'
            elif 'openinterest' in lower_col or 'oi' == lower_col:
                col_map[col] = 'OpenInterest'
        
        df.rename(columns=col_map, inplace=True)
        
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0).astype(int)
        else:
            df['Volume'] = 0
        
        if 'OpenInterest' in df.columns:
            df['OpenInterest'] = pd.to_numeric(df['OpenInterest'], errors='coerce').fillna(0).astype(int)
        else:
            df['OpenInterest'] = 0
        
        df.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    except Exception as e:
        logger.error(f"Error normalizing data for {symbol_name}: {e}")
        return None

# ========== FIX #2: HISTORICAL MARKET REGIME ==========
def get_market_regime(index_symbol=Config.BENCHMARK_INDEX, up_to_time=None):
    """Get market regime with optional historical timestamp"""
    try:
        si = pick_session(index_symbol, 1440)
        
        # FIXED: Fetch historical regime if up_to_time provided
        if up_to_time:
            # Ensure timezone aware
            if up_to_time.tzinfo is None:
                up_to_time_aware = IST.localize(up_to_time)
            else:
                up_to_time_aware = up_to_time.astimezone(IST)
            
            # Calculate start time (200 days back)
            start_time_naive = up_to_time_aware.replace(tzinfo=None) - timedelta(days=200)
            start_time_aware = IST.localize(start_time_naive)
            
            # Fetch historical data
            df_raw = tdhist_pool[si].get_historic_data(
                index_symbol,
                start_time=start_time_aware.replace(tzinfo=None),
                end_time=up_to_time_aware.replace(tzinfo=None),
                bar_size="1 day"
            )
        else:
            # Live mode: get current regime
            df_raw = tdhist_pool[si].get_historic_data(
                index_symbol, 
                duration="200 D", 
                bar_size="1 day"
            )
        
        df = normalize_hist_df(df_raw, index_symbol)
        
        if df is None or len(df) < 50:
            return 'neutral'
        
        ema20_series = ema(df['Close'], 20)
        ema50_series = ema(df['Close'], 50)
        
        if ema20_series.empty or ema50_series.empty:
            return 'neutral'
        
        ema20_val = ema20_series.dropna().iloc[-1]
        ema50_val = ema50_series.dropna().iloc[-1]
        close = df['Close'].dropna().iloc[-1]
        
        if close > ema20_val and ema20_val > ema50_val:
            return 'bullish'
        elif close < ema20_val and ema20_val < ema50_val:
            return 'bearish'
        else:
            return 'neutral'
    
    except Exception as e:
        logger.warning(f"Market regime error: {e}")
        return 'neutral'

# ========== LOAD SHARES ==========
def load_shares(filepath=Config.SHARES_FILE):
    """Load shares from file"""
    if not os.path.exists(filepath):
        logger.error(f"❌ {filepath} not found")
        return []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    shares = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        symbol = line.split()[0].upper()
        shares.append(symbol)
    
    logger.info(f"✅ Loaded {len(shares)} symbols from {filepath}")
    return shares

# ========== TECHNICAL INDICATOR FUNCTIONS ==========
def sma(series, period):
    """Simple Moving Average"""
    return series.rolling(window=period).mean()

def ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    """MACD indicator"""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def bollinger_bands(series, period=20, std_dev=2):
    """Bollinger Bands"""
    middle = sma(series, period)
    std = series.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return upper, middle, lower

def atr(df, period=14):
    """Average True Range"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def adx(df, period=14):
    """Average Directional Index"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    tr_val = atr(df, 1)
    plus_di = 100 * (plus_dm.ewm(alpha=1/period).mean() / tr_val)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/period).mean() / tr_val))
    
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx_val = dx.ewm(alpha=1/period).mean()
    
    return adx_val

def stochastic(df, period=14, smooth_k=3, smooth_d=3):
    """Stochastic Oscillator"""
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    k = k.rolling(window=smooth_k).mean()
    d = k.rolling(window=smooth_d).mean()
    
    return k, d

def cci(df, period=20):
    """Commodity Channel Index"""
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = sma(tp, period)
    mad = tp.rolling(window=period).apply(lambda x: abs(x - x.mean()).mean())
    
    return (tp - sma_tp) / (0.015 * mad)

def williams_r(df, period=14):
    """Williams %R"""
    highest_high = df['High'].rolling(window=period).max()
    lowest_low = df['Low'].rolling(window=period).min()
    
    return -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))

def obv(df):
    """On Balance Volume"""
    obv_val = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv_val

def vwap(df):
    """Volume Weighted Average Price"""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    return (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()

def cmf(df, period=20):
    """Chaikin Money Flow"""
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfm = mfm.fillna(0)
    mfv = mfm * df['Volume']
    return mfv.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()

def roc(series, period=12):
    """Rate of Change"""
    return ((series - series.shift(period)) / series.shift(period)) * 100

# ========== DATA FETCHING FUNCTIONS ==========
def fetch_single_timeframe(symbol, timeframe_min, max_retries=3):
    """Fetch single timeframe data for a symbol"""
    bar_size = BAR_SIZE_MAP.get(timeframe_min, "5 min")
    duration = DURATION_MAP.get(timeframe_min, "30 D")
    
    si = pick_session(symbol, timeframe_min)
    
    for attempt in range(max_retries):
        try:
            df_raw = tdhist_pool[si].get_historic_data(symbol, duration=duration, bar_size=bar_size)
            df = normalize_hist_df(df_raw, symbol)
            
            if df is not None and not df.empty:
                return df
            
        except lz4.block.LZ4BlockError as lz4_e:
            if attempt == max_retries - 1:
                logger.warning(f"Decompression error for {symbol} @ {timeframe_min}min: {lz4_e}. Possibly invalid response or no data.")
            time.sleep(0.5)
            
        except Exception as e:
            if attempt == max_retries - 1:
                logger.warning(f"Failed to fetch {symbol} @ {timeframe_min}min: {e}")
            time.sleep(0.5)
    
    return None

def fetch_single_timeframe_timeaware(symbol, timeframe_min, up_to_time, max_retries=3):
    """Fetch single timeframe data up to specific time (for backtesting)"""
    bar_size = BAR_SIZE_MAP.get(timeframe_min, "5 min")
    duration = DURATION_MAP.get(timeframe_min, "30 D")
    
    si = pick_session(symbol, timeframe_min)
    
    # Ensure timezone aware
    if up_to_time.tzinfo is None:
        up_to_time_aware = IST.localize(up_to_time)
    else:
        up_to_time_aware = up_to_time.astimezone(IST)
    
    # Calculate start time based on duration
    duration_days = int(duration.split()[0])
    start_time_naive = up_to_time_aware.replace(tzinfo=None) - timedelta(days=duration_days)
    start_time_aware = IST.localize(start_time_naive)
    
    for attempt in range(max_retries):
        try:
            df_raw = tdhist_pool[si].get_historic_data(
                symbol,
                start_time=start_time_aware.replace(tzinfo=None),
                end_time=up_to_time_aware.replace(tzinfo=None),
                bar_size=bar_size
            )
            df = normalize_hist_df(df_raw, symbol)
            
            if df is not None and not df.empty:
                return df
            
        except lz4.block.LZ4BlockError as lz4_e:
            if attempt == max_retries - 1:
                logger.warning(f"Decompression error for {symbol} @ {timeframe_min}min up to {up_to_time}: {lz4_e}. Possibly invalid symbol or no data.")
            time.sleep(0.5)
            
        except Exception as e:
            if attempt == max_retries - 1:
                logger.warning(f"Failed to fetch {symbol} @ {timeframe_min}min up to {up_to_time}: {e}")
            time.sleep(0.5)
    
    return None

def prefetch_all(symbols, timeframes=[5, 15, 30, 60, 1440], max_workers=32):
    """Prefetch all timeframes for all symbols (LIVE MODE)"""
    results = defaultdict(dict)
    
    tasks = []
    for sym in symbols:
        for tf in timeframes:
            tasks.append((sym, tf))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(fetch_single_timeframe, sym, tf): (sym, tf) 
                      for sym, tf in tasks}
        
        for future in as_completed(future_map):
            sym, tf = future_map[future]
            try:
                df = future.result()
                if df is not None:
                    results[sym][tf] = df
            except Exception as e:
                logger.error(f"Error fetching {sym} @ {tf}min: {e}")
    
    return results

def prefetch_all_timeaware(symbols, up_to_time, timeframes=[5, 15, 30, 60, 1440], max_workers=32):
    """Prefetch all timeframes for all symbols up to specific time (BACKTEST MODE)"""
    results = defaultdict(dict)
    
    tasks = []
    for sym in symbols:
        for tf in timeframes:
            tasks.append((sym, tf))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(fetch_single_timeframe_timeaware, sym, tf, up_to_time): (sym, tf) 
                      for sym, tf in tasks}
        
        for future in as_completed(future_map):
            sym, tf = future_map[future]
            try:
                df = future.result()
                if df is not None:
                    results[sym][tf] = df
            except Exception as e:
                logger.error(f"Error fetching {sym} @ {tf}min up to {up_to_time}: {e}")
    
    return results

# ========== FIX #3: CONTEXT-AWARE VOLUME/OI TRACKING ==========
def _has_real_oi(df):
    """Check if dataframe has real OI data"""
    if 'OpenInterest' not in df.columns:
        return False
    oi_series = df['OpenInterest']
    if oi_series.sum() == 0:
        return False
    non_zero_count = (oi_series > 0).sum()
    return non_zero_count >= 3

def calculate_5min_volume_oi_changes(df, symbol, scan_time, state):
    """Calculate 5-minute changes with state context"""
    try:
        # Filter data up to scan time
        df_5min = df[df.index <= scan_time]
        if len(df_5min) < 2:
            return 0, None, 0, 0
        
        current_volume = int(df_5min['Volume'].iloc[-1])
        previous_volume = int(df_5min['Volume'].iloc[-2])
        
        # Use state-specific tracking to handle gaps
        if symbol in state.intraday_volume_data:
            historical_volume = state.intraday_volume_data[symbol]
            if previous_volume == 0 and historical_volume > 0:
                previous_volume = historical_volume
        
        vol_change_pct = ((current_volume - previous_volume) / previous_volume * 100) if previous_volume > 0 else 0
        
        # OI calculation
        if _has_real_oi(df_5min):
            current_oi = int(df_5min['OpenInterest'].iloc[-1])
            previous_oi = int(df_5min['OpenInterest'].iloc[-2])
            
            if symbol in state.intraday_oi_data:
                historical_oi = state.intraday_oi_data[symbol]
                if previous_oi == 0 and historical_oi > 0:
                    previous_oi = historical_oi
            
            oi_change_pct = ((current_oi - previous_oi) / previous_oi * 100) if previous_oi > 0 else 0
        else:
            current_oi, oi_change_pct = None, 0
        
        # Update state
        state.intraday_volume_data[symbol] = current_volume
        if current_oi is not None:
            state.intraday_oi_data[symbol] = current_oi
        
        return current_volume, current_oi, vol_change_pct, oi_change_pct
    
    except Exception as e:
        logger.error(f"Error calculating 5-min changes for {symbol}: {e}")
        return 0, None, 0, 0

def extract_5min_volume_oi_data(df, symbol, scan_time, state, is_live=False):
    """Extract 5-minute volume and OI data with state awareness"""
    try:
        if scan_time and not is_live:
            if scan_time.tzinfo is None:
                scan_time = IST.localize(scan_time)
        
        current_volume, current_oi, vol_change_pct, oi_change_pct = calculate_5min_volume_oi_changes(
            df, symbol, scan_time, state
        )
        
        if len(df) >= 20:
            recent_vol = df['Volume'].iloc[-20:]
            avg_vol = recent_vol.mean()
            rel_vol = (current_volume / avg_vol) if avg_vol > 0 else 1.0
        else:
            rel_vol = 1.0
        
        return {
            '5m_volume': f"{current_volume:,}",
            '5m_vol_change': f"{vol_change_pct:+.1f}%",
            '5m_oi': f"{current_oi:,}" if current_oi is not None else "N/A",
            '5m_oi_change': f"{oi_change_pct:+.1f}%" if current_oi is not None else "N/A",
            'rel_vol': f"{rel_vol:.2f}x",
        }
    
    except Exception as e:
        logger.error(f"Error extracting 5-min data for {symbol}: {e}")
        return {
            '5m_volume': "N/A",
            '5m_vol_change': "N/A",
            '5m_oi': "N/A",
            '5m_oi_change': "N/A",
            'rel_vol': "N/A",
        }

# ========== SIGNAL ANALYSIS FUNCTIONS ==========
def analyze_single_timeframe(df, timeframe_min):
    """Analyze signals for a single timeframe"""
    if df is None or len(df) < 50:
        return 0, {}
    
    signals = {}
    score = 0
    
    try:
        # Trend Indicators
        close = df['Close']
        
        # MA Slope
        ma20 = sma(close, 20)
        if not ma20.empty and len(ma20) >= 5:
            ma_slope = (ma20.iloc[-1] - ma20.iloc[-5]) / ma20.iloc[-5] * 100
            signals['MA_Slope'] = 1 if ma_slope > 0.5 else (-1 if ma_slope < -0.5 else 0)
            score += signals['MA_Slope'] * Config.INDICATOR_WEIGHTS.get('MA_Slope', 1.0)
        
        # ADX
        adx_val = adx(df, 14)
        if not adx_val.empty:
            adx_current = adx_val.iloc[-1]
            signals['ADX'] = 1 if adx_current > 25 else 0
            score += signals['ADX'] * Config.INDICATOR_WEIGHTS.get('ADX', 1.0) * (1 if close.iloc[-1] > close.iloc[-2] else -1)
        
        # VWAP
        vwap_val = vwap(df)
        if not vwap_val.empty:
            signals['VWAP'] = 1 if close.iloc[-1] > vwap_val.iloc[-1] else -1
            score += signals['VWAP'] * Config.INDICATOR_WEIGHTS.get('VWAP', 1.0)
        
        # EMA Alignment
        ema9 = ema(close, 9)
        ema21 = ema(close, 21)
        if not ema9.empty and not ema21.empty:
            signals['EMA'] = 1 if ema9.iloc[-1] > ema21.iloc[-1] else -1
            score += signals['EMA'] * Config.INDICATOR_WEIGHTS.get('EMA', 1.0)
        
        # MACD
        macd_line, signal_line, histogram = macd(close)
        if not macd_line.empty and not signal_line.empty:
            signals['MACD_Trend'] = 1 if macd_line.iloc[-1] > signal_line.iloc[-1] else -1
            score += signals['MACD_Trend'] * Config.INDICATOR_WEIGHTS.get('MACD_Trend', 1.0)
        
        # Momentum Indicators
        # RSI
        rsi_val = rsi(close, 14)
        if not rsi_val.empty:
            rsi_current = rsi_val.iloc[-1]
            if rsi_current > 60:
                signals['RSI'] = 1
            elif rsi_current < 40:
                signals['RSI'] = -1
            else:
                signals['RSI'] = 0
            score += signals['RSI'] * Config.INDICATOR_WEIGHTS.get('RSI', 1.0)
        
        # Stochastic
        k, d = stochastic(df)
        if not k.empty and not d.empty:
            k_current = k.iloc[-1]
            if k_current > 70:
                signals['Stochastic'] = 1
            elif k_current < 30:
                signals['Stochastic'] = -1
            else:
                signals['Stochastic'] = 0
            score += signals['Stochastic'] * Config.INDICATOR_WEIGHTS.get('Stochastic', 1.0)
        
        # CCI
        cci_val = cci(df, 20)
        if not cci_val.empty:
            cci_current = cci_val.iloc[-1]
            if cci_current > 100:
                signals['CCI'] = 1
            elif cci_current < -100:
                signals['CCI'] = -1
            else:
                signals['CCI'] = 0
            score += signals['CCI'] * Config.INDICATOR_WEIGHTS.get('CCI', 1.0)
        
        # ROC
        roc_val = roc(close, 12)
        if not roc_val.empty:
            roc_current = roc_val.iloc[-1]
            signals['ROC'] = 1 if roc_current > 0 else -1
            score += signals['ROC'] * Config.INDICATOR_WEIGHTS.get('ROC', 1.0)
        
        # Williams %R
        wr = williams_r(df, 14)
        if not wr.empty:
            wr_current = wr.iloc[-1]
            if wr_current > -20:
                signals['WilliamsR'] = 1
            elif wr_current < -80:
                signals['WilliamsR'] = -1
            else:
                signals['WilliamsR'] = 0
            score += signals['WilliamsR'] * Config.INDICATOR_WEIGHTS.get('WilliamsR', 1.0)
        
        # Volume Indicators
        volume = df['Volume']
        
        # Volume Surge
        if len(volume) >= 20:
            vol_ma = volume.rolling(20).mean().iloc[-1]
            current_vol = volume.iloc[-1]
            if current_vol > 2 * vol_ma:
                signals['VolumeSurge'] = 1 if close.iloc[-1] > close.iloc[-2] else -1
                score += signals['VolumeSurge'] * Config.INDICATOR_WEIGHTS.get('VolumeSurge', 1.0)
        
        # OBV
        obv_val = obv(df)
        if not obv_val.empty and len(obv_val) >= 5:
            obv_slope = (obv_val.iloc[-1] - obv_val.iloc[-5])
            signals['OBV'] = 1 if obv_slope > 0 else -1
            score += signals['OBV'] * Config.INDICATOR_WEIGHTS.get('OBV', 1.0)
        
        # CMF
        cmf_val = cmf(df, 20)
        if not cmf_val.empty:
            cmf_current = cmf_val.iloc[-1]
            signals['CMF'] = 1 if cmf_current > 0.05 else (-1 if cmf_current < -0.05 else 0)
            score += signals['CMF'] * Config.INDICATOR_WEIGHTS.get('CMF', 1.0)
        
        # Volatility Indicators
        # Bollinger Bands
        upper, middle, lower = bollinger_bands(close, 20, 2)
        if not upper.empty and not lower.empty:
            close_current = close.iloc[-1]
            bb_position = (close_current - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])
            
            if bb_position > 0.8:
                signals['Bollinger'] = 1
            elif bb_position < 0.2:
                signals['Bollinger'] = -1
            else:
                signals['Bollinger'] = 0
            score += signals['Bollinger'] * Config.INDICATOR_WEIGHTS.get('Bollinger', 1.0)
        
        # Volatility Expansion
        atr_val = atr(df, 14)
        if not atr_val.empty and len(atr_val) >= 14:
            atr_current = atr_val.iloc[-1]
            atr_avg = atr_val.iloc[-14:].mean()
            if atr_current > 1.2 * atr_avg:
                signals['VolatilityExpansion'] = 1 if close.iloc[-1] > close.iloc[-2] else -1
                score += signals['VolatilityExpansion'] * Config.INDICATOR_WEIGHTS.get('VolatilityExpansion', 1.0)
        
        # OI Indicators (if available)
        if _has_real_oi(df):
            oi = df['OpenInterest']
            
            # OI Change
            if len(oi) >= 5:
                oi_change = (oi.iloc[-1] - oi.iloc[-5]) / oi.iloc[-5] * 100 if oi.iloc[-5] > 0 else 0
                if abs(oi_change) > 5:
                    signals['OIChange'] = 1 if oi_change > 0 and close.iloc[-1] > close.iloc[-2] else -1
                    score += signals['OIChange'] * Config.INDICATOR_WEIGHTS.get('OIChange', 1.0)
            
            # Volume-OI Sync
            if len(volume) >= 5 and len(oi) >= 5:
                vol_trend = 1 if volume.iloc[-1] > volume.iloc[-5] else -1
                oi_trend = 1 if oi.iloc[-1] > oi.iloc[-5] else -1
                price_trend = 1 if close.iloc[-1] > close.iloc[-5] else -1
                
                if vol_trend == oi_trend == price_trend:
                    signals['VolumeOISync'] = price_trend
                    score += signals['VolumeOISync'] * Config.INDICATOR_WEIGHTS.get('VolumeOISync', 1.0)
        
    except Exception as e:
        logger.error(f"Error analyzing timeframe {timeframe_min}: {e}")
    
    return score, signals

def analyze_ultimate_signals(timeframe_data, market_regime='neutral'):
    """Analyze signals across multiple timeframes with regime awareness"""
    total_score = 0
    all_sub_scores = {}
    
    for tf, df in timeframe_data.items():
        if df is not None and not df.empty:
            tf_score, tf_signals = analyze_single_timeframe(df, tf)
            
            # Apply timeframe weight
            tf_weight = TIMEFRAME_WEIGHTS.get(tf, 1.0)
            weighted_score = tf_score * tf_weight
            
            total_score += weighted_score
            all_sub_scores[f"{tf}min"] = {
                'score': round(weighted_score, 2),
                'signals': tf_signals
            }
    
    # Apply market regime multiplier
    regime_key = None
    if total_score > 0 and market_regime == 'bullish':
        regime_key = 'bullish_in_bull_market'
    elif total_score < 0 and market_regime == 'bearish':
        regime_key = 'bearish_in_bear_market'
    elif total_score > 0 and market_regime == 'bearish':
        regime_key = 'bullish_in_bear_market'
    elif total_score < 0 and market_regime == 'bullish':
        regime_key = 'bearish_in_bull_market'
    
    if regime_key:
        regime_multiplier = Config.REGIME_MULTIPLIERS.get(regime_key, 1.0)
        total_score *= regime_multiplier
    
    # Determine signal category
    signal = "Neutral"
    for threshold_name, threshold_value in sorted(Config.SIGNAL_THRESHOLDS.items(), 
                                                   key=lambda x: abs(x[1]), reverse=True):
        if 'Buy' in threshold_name and total_score >= threshold_value:
            signal = threshold_name
            break
        elif 'Sell' in threshold_name and total_score <= threshold_value:
            signal = threshold_name
            break
    
    return signal, round(total_score, 2), all_sub_scores

def enhanced_institutional_flow_analysis(timeframe_data):
    """Enhanced institutional flow detection"""
    flow_signals = []
    
    for tf, df in timeframe_data.items():
        if df is None or len(df) < 20:
            continue
        
        try:
            volume = df['Volume']
            close = df['Close']
            
            # High volume with price strength
            vol_ma = volume.rolling(20).mean().iloc[-1]
            current_vol = volume.iloc[-1]
            
            if current_vol > 2 * vol_ma:
                price_change = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
                
                if price_change > 1:
                    flow_signals.append("🟢 INST BUY")
                elif price_change < -1:
                    flow_signals.append("🔴 INST SELL")
            
            # OI analysis
            if _has_real_oi(df):
                oi = df['OpenInterest']
                if len(oi) >= 5:
                    oi_change = (oi.iloc[-1] - oi.iloc[-5]) / oi.iloc[-5] * 100 if oi.iloc[-5] > 0 else 0
                    
                    if oi_change > 10 and close.iloc[-1] > close.iloc[-5]:
                        flow_signals.append("📈 OI+Price↑")
                    elif oi_change > 10 and close.iloc[-1] < close.iloc[-5]:
                        flow_signals.append("📉 OI+Price↓")
        
        except Exception as e:
            logger.warning(f"Flow analysis error for {tf}min: {e}")
    
    return " | ".join(flow_signals) if flow_signals else "Neutral"

# ========== FIX #4: UPDATED SCANNER FUNCTION WITH STATE AWARENESS ==========
def run_ultimate_scan_at_time(time_point_aware, stocks, market_regime, state, is_live=False):
    """Ultimate scan with state-aware processing"""
    
    # Convert to TrueData symbols
    truedata_stocks = [convert_to_truedata_symbol(s) for s in stocks]
    
    # Fetch data based on mode
    if is_live:
        stock_multi_data = prefetch_all(truedata_stocks, max_workers=Config.MAX_WORKERS)
    else:
        stock_multi_data = prefetch_all_timeaware(truedata_stocks, time_point_aware, max_workers=Config.MAX_WORKERS)
    
    print_colored(f"✅ Data fetch complete: {len(stock_multi_data)} stocks", Colors.GREEN)
    
    signals_this_scan = []
    current_symbols = set()
    
    for truedata_symbol, timeframe_data in stock_multi_data.items():
        clean_symbol = convert_to_localhost_symbol(truedata_symbol)
        current_symbols.add(clean_symbol)
        
        # Skip if all timeframes failed
        if all(df is None for df in timeframe_data.values()):
            logger.warning(f"Skipping {clean_symbol}: No data available for any timeframe")
            continue
        
        # Filter dataframes up to scan time (for backtest mode)
        filtered_timeframes = {}
        for tf, df in timeframe_data.items():
            if df is not None and not df.empty:
                if is_live:
                    df_slice = df
                else:
                    # Apply time filtering for backtest
                    if time_point_aware and isinstance(df.index, pd.DatetimeIndex):
                        try:
                            # Ensure timezone aware comparison
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
        
        # Volume filter - require significant volume
        tf_5min = filtered_timeframes.get(5)
        if tf_5min is not None and len(tf_5min) >= 20:
            vol_ma = tf_5min["Volume"].rolling(20).mean().iloc[-1]
            current_vol = tf_5min["Volume"].iloc[-1]
            if current_vol < 3 * vol_ma:
                continue
        else:
            continue
        
        # Analyze signals
        signal, score, sub_scores = analyze_ultimate_signals(
            filtered_timeframes, market_regime
        )
        
        # Filter by minimum score threshold
        if abs(score) >= Config.SCORE_THRESHOLD_MIN:
            # Institutional flow analysis
            flow_tag = enhanced_institutional_flow_analysis(filtered_timeframes)
            
            # Extract 5-minute volume/OI data with state
            tf_5min = filtered_timeframes.get(5)
            if tf_5min is not None:
                oi_vol_data = extract_5min_volume_oi_data(tf_5min, clean_symbol, time_point_aware, state, is_live=is_live)
            else:
                main_tf_data = filtered_timeframes.get(15, filtered_timeframes.get(30, list(filtered_timeframes.values())[0]))
                oi_vol_data = extract_5min_volume_oi_data(main_tf_data, clean_symbol, time_point_aware, state, is_live=is_live)
            
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

# ========== DISPLAY FUNCTIONS ==========
def create_ultimate_option_table(signals, title="TECHNICAL SIGNALS", new_stocks=None, scan_time_str=None):
    """Create enhanced display table for signals"""
    if not signals:
        return
    
    new_stocks_set = set(new_stocks) if new_stocks else set()
    
    if RICH_AVAILABLE:
        table = Table(title=f"\n{title}", box=box.ROUNDED, show_header=True, header_style="bold magenta")
        
        table.add_column("Rank", justify="center", style="cyan", width=5)
        table.add_column("Symbol", justify="left", style="bold yellow", width=12)
        table.add_column("Signal", justify="center", style="bold", width=18)
        table.add_column("Score", justify="right", style="bold", width=8)
        table.add_column("5m Vol", justify="right", width=12)
        table.add_column("Vol Δ", justify="right", width=10)
        table.add_column("5m OI", justify="right", width=12)
        table.add_column("OI Δ", justify="right", width=10)
        table.add_column("RelVol", justify="right", width=8)
        table.add_column("Flow", justify="center", width=20)
        table.add_column("Action", justify="center", style="bold green", width=15)
        
        for idx, r in enumerate(signals, 1):
            symbol = r['symbol']
            is_new = symbol in new_stocks_set
            symbol_display = f"🆕 {symbol}" if is_new else symbol
            
            signal_style = "bold green" if "Buy" in r['signal'] else "bold red"
            score_style = "green" if r['score'] > 0 else "red"
            
            table.add_row(
                str(idx),
                symbol_display,
                r['signal'],
                f"{r['score']:+.1f}",
                r.get('5m_volume', 'N/A'),
                r.get('5m_vol_change', 'N/A'),
                r.get('5m_oi', 'N/A'),
                r.get('5m_oi_change', 'N/A'),
                r.get('rel_vol', 'N/A'),
                r.get('flow', 'N/A'),
                r.get('action', 'N/A'),
            )
        
        if scan_time_str:
            table.caption = f"Scan Time: {scan_time_str}"
        
        console.print(table)
    
    elif TABULATE_AVAILABLE:
        headers = ["Rank", "Symbol", "Signal", "Score", "5m Vol", "Vol Δ", "5m OI", "OI Δ", "RelVol", "Flow", "Action"]
        rows = []
        
        for idx, r in enumerate(signals, 1):
            symbol = r['symbol']
            is_new = symbol in new_stocks_set
            symbol_display = f"🆕 {symbol}" if is_new else symbol
            
            rows.append([
                idx,
                symbol_display,
                r['signal'],
                f"{r['score']:+.1f}",
                r.get('5m_volume', 'N/A'),
                r.get('5m_vol_change', 'N/A'),
                r.get('5m_oi', 'N/A'),
                r.get('5m_oi_change', 'N/A'),
                r.get('rel_vol', 'N/A'),
                r.get('flow', 'N/A'),
                r.get('action', 'N/A'),
            ])
        
        print(f"\n{title}")
        if scan_time_str:
            print(f"Scan Time: {scan_time_str}")
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    
    else:
        print(f"\n{title}")
        if scan_time_str:
            print(f"Scan Time: {scan_time_str}")
        print("-" * 120)
        for idx, r in enumerate(signals, 1):
            symbol = r['symbol']
            is_new = symbol in new_stocks_set
            symbol_display = f"🆕 {symbol}" if is_new else symbol
            print(f"{idx}. {symbol_display} | {r['signal']} | Score: {r['score']:+.1f} | {r.get('action', 'N/A')}")

# ========== TIMESTAMP GENERATION FOR BACKTEST ==========
def generate_backtest_timestamps(date_str):
    """Generate scan timestamps for a given date"""
    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        logger.error(f"Invalid date format: {date_str}. Use YYYY-MM-DD")
        return []
    
    market_start_time = datetime.strptime(Config.FIRST_RUN_AT, "%H:%M").time()
    market_end_time = datetime.strptime(Config.MARKET_END, "%H:%M").time()
    
    current_time = datetime.combine(date_obj, market_start_time)
    current_time = IST.localize(current_time)
    
    end_time = datetime.combine(date_obj, market_end_time)
    end_time = IST.localize(end_time)
    
    # Add initial delay
    current_time += timedelta(seconds=Config.BACKTEST_START_DELAY)
    
    timestamps = []
    while current_time <= end_time:
        timestamps.append(current_time)
        current_time += timedelta(minutes=Config.BACKTEST_INTERVAL_MINUTES)
    
    return timestamps

# ========== FIX #5: UPDATED BACKTEST FUNCTION ==========
def run_ultimate_backtest(backtest_date, stocks):
    """Ultimate backtest with isolated state"""
    # Check if backtest date is in the future
    now = datetime.now(IST)
    try:
        backtest_dt = datetime.strptime(backtest_date + " 00:00", "%Y-%m-%d %H:%M")
        backtest_dt = IST.localize(backtest_dt)
    except ValueError:
        logger.error(f"Invalid backtest date format: {backtest_date}")
        return
    
    if backtest_dt.date() > now.date():
        logger.error("Backtest date is in the future. No data available.")
        return
    
    # Create separate backtest state
    backtest_state = ScannerState(mode='backtest')
    
    print_colored(f"\n🎯 STARTING TECHNICAL BACKTEST FOR {backtest_date}", Colors.HEADER)
    
    # Generate timestamps
    timestamps = generate_backtest_timestamps(backtest_date)
    total_scans = len(timestamps)
    print_colored(f"📅 Generated {total_scans} scan points", Colors.CYAN)
    
    if total_scans == 0:
        logger.error("No timestamps generated. Check date and market hours.")
        return
    
    # If current day, limit timestamps to current time
    if backtest_dt.date() == now.date():
        timestamps = [ts for ts in timestamps if ts <= now]
        if not timestamps:
            logger.error("No valid timestamps for current day backtest.")
            return
    
    # FIXED: Get historical market regime
    market_regime = get_market_regime(Config.BENCHMARK_INDEX, up_to_time=timestamps[-1])
    print_colored(f"📈 Market Regime (as of {backtest_date}): {market_regime.upper()}", Colors.BLUE)
    
    all_results = []
    
    with tqdm(total=len(timestamps), desc="🎯 Technical Backtesting", ncols=120) as pbar:
        for i, scan_time in enumerate(timestamps):
            try:
                pbar.set_description(f"Scanning at {scan_time.strftime('%H:%M:%S')}")
                
                # Pass backtest state to scan function
                signals, current_symbols = run_ultimate_scan_at_time(
                    scan_time, stocks, market_regime, backtest_state, is_live=False
                )
                
                # Track new stocks
                previous_symbols = set(backtest_state.stock_history.keys())
                new_stocks = current_symbols - previous_symbols
                
                # Update stock history
                for symbol in current_symbols:
                    backtest_state.stock_history[symbol] = scan_time
                
                # Compile scan results
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
                
                # Display results
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
    
    # Summary
    print_colored(f"\n📊 TECHNICAL BACKTEST SUMMARY FOR {backtest_date}", Colors.HEADER)
    
    total_signals = sum(r['total_signals'] for r in all_results)
    total_bullish = sum(r['bullish_signals'] for r in all_results)
    total_bearish = sum(r['bearish_signals'] for r in all_results)
    total_perfect = sum(r['perfect_setups'] for r in all_results)
    
    print(f"✅ Scans Completed: {len(all_results)}/{len(timestamps)}")
    print(f"📊 Total Signals Found: {total_signals}")
    print(f"🟢 Bullish Signals: {total_bullish}")
    print(f"🔴 Bearish Signals: {total_bearish}")
    print(f"⭐ Perfect Setups: {total_perfect}")
    
    # Save results
    output_filename = f"{backtest_date}_technical_backtest_results.json"
    try:
        with open(output_filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        print_colored(f"\n💾 Results saved to: {output_filename}", Colors.GREEN)
    except Exception as e:
        logger.error(f"Could not save results: {e}")
    
    print_colored("🎯 Technical Backtesting Completed!", Colors.GREEN)

# ========== LIVE TRADING MODE ==========
def run_live_scanner(stocks):
    """Run live scanner continuously during market hours"""
    global scanner_state
    
    # Use live mode state
    scanner_state = ScannerState(mode='live')
    
    print_colored("\n🚀 STARTING LIVE TECHNICAL SCANNER", Colors.HEADER)
    print_colored(f"📊 Monitoring {len(stocks)} stocks", Colors.CYAN)
    
    # Get current market regime
    market_regime = get_market_regime(Config.BENCHMARK_INDEX)
    print_colored(f"📈 Current Market Regime: {market_regime.upper()}", Colors.BLUE)
    
    first_scan = True
    
    while True:
        try:
            now_ist = datetime.now(IST)
            current_time = now_ist.time()
            
            # Check if market is open
            market_start = datetime.strptime(Config.MARKET_START, "%H:%M").time()
            market_end = datetime.strptime(Config.MARKET_END, "%H:%M").time()
            
            if current_time < market_start:
                wait_seconds = (datetime.combine(now_ist.date(), market_start) - 
                               datetime.combine(now_ist.date(), current_time)).total_seconds()
                print_colored(f"⏳ Market opens in {format_time_remaining(wait_seconds)}", Colors.YELLOW)
                time.sleep(60)
                continue
            
            if current_time > market_end:
                print_colored("🔔 Market closed for today", Colors.YELLOW)
                break
            
            # First scan at FIRST_RUN_AT
            if first_scan:
                first_run_time = datetime.strptime(Config.FIRST_RUN_AT, "%H:%M").time()
                if current_time < first_run_time:
                    wait_seconds = (datetime.combine(now_ist.date(), first_run_time) - 
                                   datetime.combine(now_ist.date(), current_time)).total_seconds()
                    print_colored(f"⏳ First scan in {format_time_remaining(wait_seconds)}", Colors.YELLOW)
                    time.sleep(30)
                    continue
                
                # Wait for settlement delay
                print_colored(f"⏳ Waiting {Config.FIRST_SCAN_DELAY}s for data settlement...", Colors.YELLOW)
                time.sleep(Config.FIRST_SCAN_DELAY)
                first_scan = False
            
            # Check if it's time for next scan (every 5 minutes)
            if now_ist.minute % 5 == 0 and now_ist.second < 30:
                # Wait for bar to close + settlement
                time.sleep(Config.SETTLE_DELAY_SECONDS)
                
                scan_time = datetime.now(IST)
                print_colored(f"\n🔍 SCANNING AT {scan_time.strftime('%H:%M:%S')}", Colors.HEADER)
                
                # Refresh market regime periodically
                if scanner_state.scan_count % 12 == 0:  # Every hour
                    market_regime = get_market_regime(Config.BENCHMARK_INDEX)
                    print_colored(f"📈 Market Regime: {market_regime.upper()}", Colors.BLUE)
                
                # Run scan with live state
                signals, current_symbols = run_ultimate_scan_at_time(
                    scan_time, stocks, market_regime, scanner_state, is_live=True
                )
                
                scanner_state.scan_count += 1
                
                # Track new stocks
                previous_symbols = set(scanner_state.stock_history.keys())
                new_stocks = current_symbols - previous_symbols
                
                for symbol in current_symbols:
                    scanner_state.stock_history[symbol] = scan_time
                
                # Display results
                if signals:
                    signals.sort(key=lambda x: abs(x['score']), reverse=True)
                    top_bullish = [r for r in signals if r['score'] > 0][:15]
                    top_bearish = [r for r in signals if r['score'] < 0][:15]
                    
                    scan_time_str = scan_time.strftime('%H:%M')
                    
                    if top_bullish:
                        create_ultimate_option_table(top_bullish, f"🟢 LIVE BULLISH SIGNALS", new_stocks, scan_time_str)
                    
                    if top_bearish:
                        create_ultimate_option_table(top_bearish, f"🔴 LIVE BEARISH SIGNALS", new_stocks, scan_time_str)
                    
                    print_colored(f"\n📊 Scan #{scanner_state.scan_count} | Total: {len(signals)} | Bullish: {len(top_bullish)} | Bearish: {len(top_bearish)}", Colors.CYAN)
                else:
                    print_colored(f"📊 Scan #{scanner_state.scan_count} | No signals above threshold", Colors.YELLOW)
                
                # Wait until next 5-minute mark
                time.sleep(240)  # 4 minutes
            else:
                time.sleep(30)
        
        except KeyboardInterrupt:
            print_colored("\n⚠️ Scanner stopped by user", Colors.YELLOW)
            break
        except Exception as e:
            logger.error(f"Error in live scanner: {e}")
            time.sleep(60)

# ========== MAIN FUNCTION ==========
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Ultimate Technical Scanner v4.4 - Backtest-Live Consistency")
    parser.add_argument('--mode', choices=['live', 'backtest'], default='live', help="Run mode")
    parser.add_argument('--date', type=str, help="Backtest date (YYYY-MM-DD)")
    parser.add_argument('--shares', type=str, default=Config.SHARES_FILE, help="Path to shares file")
    
    args = parser.parse_args()
    
    print_colored("=" * 80, Colors.HEADER)
    print_colored("🎯 ULTIMATE TECHNICAL SCANNER v4.4", Colors.HEADER)
    print_colored("✅ BACKTEST-LIVE CONSISTENCY FIXED", Colors.GREEN)
    print_colored("=" * 80, Colors.HEADER)
    
    # Initialize TD_hist pool
    print_colored("\n🔌 Initializing TrueData connections...", Colors.CYAN)
    init_tdhist_pool()
    
    # Load stocks
    Config.SHARES_FILE = args.shares
    stocks = load_shares(Config.SHARES_FILE)
    
    if not stocks:
        logger.error("No stocks loaded. Exiting.")
        return
    
    print_colored(f"✅ Loaded {len(stocks)} stocks", Colors.GREEN)
    
    # Run based on mode
    if args.mode == 'backtest':
        if not args.date:
            logger.error("❌ Please provide --date for backtest mode (YYYY-MM-DD)")
            return
        
        run_ultimate_backtest(args.date, stocks)
    
    else:
        run_live_scanner(stocks)
    
    print_colored("\n✅ Scanner completed successfully!", Colors.GREEN)

if __name__ == "__main__":
    main()