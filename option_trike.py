# ==============================================================================
# ENHANCED OPTION BUYER SCANNER v4.1 - COMPLETE PRODUCTION VERSION
# Part 1: Core Infrastructure, Configuration & Basic Functions
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
import math

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
    GREAT_TABLES_AVAILABLE = True
except ImportError:
    GREAT_TABLES_AVAILABLE = False
    print("Installing great-tables: pip install great-tables")

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False

try:
    import requests
    import yfinance as yf
    MARKET_DATA_AVAILABLE = True
except ImportError:
    MARKET_DATA_AVAILABLE = False
    print("Installing market data: pip install requests yfinance")

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

# ======== Enhanced Configuration ========
class Config:
    TDUSERNAME = os.getenv("TRUEDATA_USER", "tdwsp751")
    TDPASSWORD = os.getenv("TRUEDATA_PASS", "raj@751")

    MARKET_START = "09:15"
    FIRST_RUN_AT = "09:20"
    FIRST_SCAN_DELAY = 15
    MARKET_END   = "15:30"
    SETTLE_DELAY_SECONDS = 15
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "64"))
    TD_HIST_SESSIONS = int(os.getenv("TD_HIST_SESSIONS", "5"))
    SHARES_FILE = os.getenv("SHARES_FILE", "shares.txt")
    BENCHMARK_INDEX = "NIFTY 50"

    # Option-Specific Configuration
    OPTION_EXPIRY_DAYS = [0, 7, 14]
    MIN_OPTION_VOLUME = 100
    MIN_OPTION_OI = 500
    MAX_OPTION_SPREAD = 0.05
    ATM_RANGE = 0.02
    OTM_RANGE = 0.10
    MIN_DELTA = 0.15
    MAX_THETA_DECAY = -10
    HIGH_IV_PERCENTILE = 75
    LOW_IV_PERCENTILE = 25
    VOLATILITY_EXPANSION_THRESHOLD = 1.5
    
    BACKTEST_INTERVAL_MINUTES = 5
    BACKTEST_START_DELAY = 5
    BACKTEST_TOP_DISPLAY = 15

    # Enhanced Group Weights
    GROUP_WEIGHTS = {
        "Trend": 2.5, "Momentum": 2.8, "Volume": 2.5, "Volatility": 3.0, "OI": 2.8,
        "OptionFlow": 3.2, "Greeks": 2.5, "MarketRegime": 2.0,
    }

    # Enhanced Individual Indicator Weights
    INDICATOR_WEIGHTS = {
        "MA_Slope": 2.0, "ADX": 1.8, "VWAP": 1.7, "EMA": 1.7, "MACD_Trend": 1.5,
        "RSI": 2.0, "Stochastic": 1.2, "CCI": 1.2, "ROC": 1.1, "WilliamsR": 1.0,
        "VolumeSurge": 2.5, "OBV": 1.8, "CMF": 1.8, "RelVol": 1.5,
        "VolatilityExpansion": 3.0, "Bollinger": 1.3,
        "OptionBuyerMomentum": 2.8, "OIChange": 2.5, "VolumeOISync": 2.2,
        "OptionFlowDirection": 3.2, "UnusualOptionActivity": 3.0, "StrikeMomentum": 2.8,
        "IVPercentile": 2.5, "IVExpansion": 3.0, "GammaExpansion": 2.2,
        "VIXSignal": 2.0, "MarketRegimeFilter": 1.8, "InstitutionalFlow": 2.5,
    }

    SCORE_THRESHOLD_MIN = 12.0
    SIGNAL_THRESHOLDS = {
        'Very Strong Buy': 65.0, 'Strong Buy': 40.0, 'Buy Signal': 18.0,
        'Very Strong Sell': -65.0, 'Strong Sell': -40.0, 'Sell Signal': -18.0,
    }
    
    REGIME_MULTIPLIERS = {
        'bullish_in_bull_market': 1.2, 'bearish_in_bear_market': 1.2,
        'bullish_in_bear_market': 0.75, 'bearish_in_bull_market': 0.75,
        'high_vix_expansion': 1.3, 'low_vix_compression': 0.9,
    }

# Constants
IST = pytz.timezone("Asia/Kolkata")
BAR_SIZE_MAP = {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"}
DURATION_MAP = {5: "30 D", 15: "30 D", 30: "60 D", 60: "120 D", 1440: "365 D"}
TIMEFRAME_WEIGHTS = {15: 3.2, 5: 2.8, 30: 2.2, 60: 1.8, 1440: 1.0}

# Silence noisy loggers
for noisy in ("truedata", "truedata.history", "truedata_ws", "websocket", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

# Enhanced state management
previous_scan_results = {}
previous_oi_data = {}
previous_volume_data = {}
intraday_volume_data = {}
intraday_oi_data = {}
option_chain_cache = {}
vix_data_cache = {}
iv_history_cache = {}
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

# ========== VIX DATA FETCHING ==========
def fetch_vix_data():
    """Fetch current VIX data for market regime analysis."""
    global vix_data_cache
    
    try:
        if MARKET_DATA_AVAILABLE:
            vix_symbol = "^INDIAVIX"
            vix_data = yf.download(vix_symbol, period="30d", interval="1d")
            
            if not vix_data.empty:
                current_vix = float(vix_data['Close'].iloc[-1])
                vix_20d_avg = float(vix_data['Close'].rolling(20).mean().iloc[-1])
                vix_min = float(vix_data['Close'].min())
                vix_max = float(vix_data['Close'].max())
                
                if vix_max > vix_min:
                    vix_percentile = (current_vix - vix_min) / (vix_max - vix_min) * 100
                else:
                    vix_percentile = 50.0
                
                vix_data_cache = {
                    'current_vix': current_vix,
                    'vix_20d_avg': vix_20d_avg,
                    'vix_percentile': vix_percentile,
                    'timestamp': datetime.now(IST)
                }
                return vix_data_cache
    except Exception as e:
        logger.error(f"Error fetching VIX data: {e}")
    
    return {
        'current_vix': 20.0, 'vix_20d_avg': 18.5, 'vix_percentile': 50.0,
        'timestamp': datetime.now(IST)
    }

def get_vix_signal():
    """Generate VIX-based market signal for option buying."""
    vix_info = fetch_vix_data()
    vix_percentile = vix_info['vix_percentile']
    
    if vix_percentile > 80:
        return 1.5
    elif vix_percentile > 60:
        return 1.0
    elif vix_percentile < 20:
        return -0.5
    else:
        return 0.0

# ========== OPTION CHAIN DATA SIMULATION ==========
def simulate_option_chain_data(symbol, current_price, days_to_expiry=7):
    """Simulate option chain data for demonstration."""
    try:
        if current_price <= 0:
            return []
            
        strikes = []
        base_strike = int(current_price / 50) * 50
        
        for i in range(-8, 9):
            strike = base_strike + (i * 50)
            if strike > 0:
                strikes.append(strike)
        
        option_data = []
        time_to_expiry = max(days_to_expiry / 365.0, 0.01)
        
        for strike in strikes:
            moneyness = strike / current_price if current_price > 0 else 1.0
            
            if moneyness < 0.95:
                delta = min(0.99, max(0.01, 0.8 + (moneyness - 0.9) * 2))
            elif moneyness > 1.05:
                delta = min(0.99, max(0.01, 0.2 - (moneyness - 1.05) * 2))
            else:
                delta = 0.5
            
            gamma = 0.05 * math.exp(-((moneyness - 1) * 5) ** 2)
            theta = -current_price * 0.0002 * delta * (days_to_expiry / 7)
            vega = current_price * 0.01 * math.sqrt(time_to_expiry) * gamma * 10
            
            if 0.98 <= moneyness <= 1.02:
                volume = np.random.randint(500, 2000)
                oi = np.random.randint(1000, 5000)
            elif 0.95 <= moneyness <= 1.05:
                volume = np.random.randint(200, 1000)
                oi = np.random.randint(500, 3000)
            else:
                volume = np.random.randint(50, 500)
                oi = np.random.randint(100, 1500)
            
            base_iv = 0.25 + abs(moneyness - 1) * 0.5
            iv = max(0.08, min(0.8, base_iv + np.random.normal(0, 0.02)))
            
            option_data.append({
                'strike': strike,
                'type': 'CALL',
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'volume': volume,
                'oi': oi,
                'iv': iv,
                'moneyness': moneyness,
                'days_to_expiry': days_to_expiry,
            })
        
        return option_data
    
    except Exception as e:
        logger.error(f"Error simulating option chain for {symbol}: {e}")
        return []

def analyze_option_chain(symbol, current_price, option_data):
    """Analyze option chain for optimal strike selection and flow."""
    try:
        if not option_data or current_price <= 0:
            return {'recommended_strikes': [], 'flow_analysis': 'Insufficient data'}
        
        df = pd.DataFrame(option_data)
        
        liquid_options = df[
            (df['volume'] >= Config.MIN_OPTION_VOLUME) &
            (df['oi'] >= Config.MIN_OPTION_OI) &
            (df['delta'] >= Config.MIN_DELTA)
        ].copy()
        
        if liquid_options.empty:
            return {'recommended_strikes': [], 'flow_analysis': 'Insufficient liquidity'}
        
        atm_options = liquid_options[
            (liquid_options['moneyness'] >= (1 - Config.ATM_RANGE)) &
            (liquid_options['moneyness'] <= (1 + Config.ATM_RANGE))
        ].copy()
        
        otm_options = liquid_options[
            (liquid_options['moneyness'] > (1 + Config.ATM_RANGE)) &
            (liquid_options['moneyness'] <= (1 + Config.OTM_RANGE))
        ].copy()
        
        total_call_volume = int(liquid_options['volume'].sum())
        total_call_oi = int(liquid_options['oi'].sum())
        
        if total_call_volume > 0:
            avg_iv = float((liquid_options['iv'] * liquid_options['volume']).sum() / total_call_volume)
        else:
            avg_iv = float(liquid_options['iv'].mean()) if not liquid_options['iv'].empty else 0.25
        
        liquid_options['volume_oi_ratio'] = liquid_options['volume'] / liquid_options['oi'].replace(0, 1)
        unusual_activity = liquid_options[liquid_options['volume_oi_ratio'] > 2.0]
        
        recommended_strikes = []
        
        if not atm_options.empty:
            best_atm_idx = atm_options['gamma'].idxmax()
            best_atm = atm_options.loc[best_atm_idx]
            recommended_strikes.append({
                'strike': int(best_atm['strike']),
                'type': 'ATM_MOMENTUM',
                'delta': float(best_atm['delta']),
                'gamma': float(best_atm['gamma']),
                'iv': float(best_atm['iv']),
                'volume': int(best_atm['volume']),
                'oi': int(best_atm['oi']),
                'reason': 'High gamma for momentum'
            })
        
        if not otm_options.empty:
            otm_options = otm_options.copy()
            otm_options['score'] = (
                otm_options['volume'] * 0.3 +
                otm_options['oi'] * 0.2 +
                otm_options['delta'] * 1000 * 0.5
            )
            best_otm_idx = otm_options['score'].idxmax()
            best_otm = otm_options.loc[best_otm_idx]
            recommended_strikes.append({
                'strike': int(best_otm['strike']),
                'type': 'OTM_SWING',
                'delta': float(best_otm['delta']),
                'gamma': float(best_otm['gamma']),
                'iv': float(best_otm['iv']),
                'volume': int(best_otm['volume']),
                'oi': int(best_otm['oi']),
                'reason': 'Best OTM risk-reward'
            })
        
        flow_signals = []
        if len(unusual_activity) > 0:
            flow_signals.append("Unusual Call Activity")
        
        if avg_iv > 0.35:
            flow_signals.append("High IV Environment")
        elif avg_iv < 0.15:
            flow_signals.append("Low IV - Volatility Expansion Expected")
        
        if total_call_volume > total_call_oi * 0.5:
            flow_signals.append("Strong Call Buying")
        
        flow_analysis = '; '.join(flow_signals) if flow_signals else 'Normal flow'
        
        return {
            'recommended_strikes': recommended_strikes,
            'flow_analysis': flow_analysis,
            'total_call_volume': total_call_volume,
            'total_call_oi': total_call_oi,
            'avg_iv': round(avg_iv * 100, 1),
            'unusual_activity_strikes': unusual_activity['strike'].tolist() if not unusual_activity.empty else [],
            'liquidity_score': len(liquid_options),
        }
        
    except Exception as e:
        logger.error(f"Error analyzing option chain for {symbol}: {e}")
        return {'recommended_strikes': [], 'flow_analysis': 'Analysis error'}

# ========== IV PERCENTILE CALCULATION ==========
def calculate_iv_percentile(symbol, current_iv, lookback_days=252):
    """Calculate IV percentile for the symbol."""
    global iv_history_cache
    
    try:
        cache_key = f"{symbol}_{lookback_days}"
        
        if cache_key not in iv_history_cache or \
           (datetime.now(IST) - iv_history_cache[cache_key]['timestamp']).seconds > 3600:
            
            # Simulate historical IV data
            base_iv = current_iv
            iv_history = []
            
            for i in range(lookback_days):
                daily_change = np.random.normal(0, 0.01)
                if base_iv > 0.4:
                    daily_change -= 0.001
                elif base_iv < 0.15:
                    daily_change += 0.001
                
                base_iv = max(0.08, min(0.8, base_iv + daily_change))
                iv_history.append(base_iv)
            
            iv_history_cache[cache_key] = {
                'iv_history': iv_history,
                'timestamp': datetime.now(IST)
            }
        
        iv_history = iv_history_cache[cache_key]['iv_history']
        
        iv_array = np.array(iv_history)
        percentile = (np.sum(iv_array <= current_iv) / len(iv_array)) * 100
        
        return percentile
        
    except Exception as e:
        logger.error(f"Error calculating IV percentile for {symbol}: {e}")
        return 50.0
def safe_dataframe_check(df, min_length=1):
    """Enhanced safe DataFrame/Series checking."""
    try:
        return (df is not None and 
                hasattr(df, '__len__') and 
                len(df) >= min_length and 
                hasattr(df, 'empty') and 
                not df.empty)
    except Exception:
        return False
# ========== SAFE SERIES CHECKING ==========
def safe_series_check(series, min_length=1):
    """Safely check if series has data before operations."""
    return (series is not None and 
            hasattr(series, '__len__') and 
            len(series) >= min_length and 
            not series.empty if hasattr(series, 'empty') else True)

# ========== ENHANCED TECHNICAL INDICATORS ==========
def ema(series, length):
    """Safe EMA calculation."""
    try:
        if not safe_series_check(series, length):
            return pd.Series(dtype='float64')
        return series.ewm(span=length, adjust=False).mean()
    except Exception:
        return pd.Series(dtype='float64')

def vwap(df, period=None):
    """Safe VWAP calculation."""
    try:
        if df is None or len(df) < 1:
            return pd.Series(dtype='float64')
        
        price = (df["High"] + df["Low"] + df["Close"]) / 3.0
        pv = price * df["Volume"]
        
        if period:
            pv_sum = pv.rolling(period).sum()
            vol_sum = df["Volume"].rolling(period).sum()
        else:
            pv_sum = pv.cumsum()
            vol_sum = df["Volume"].cumsum()
        
        vol_sum = vol_sum.replace(0, np.nan)
        return pv_sum / vol_sum
    except Exception:
        return pd.Series(dtype='float64')

def atr(df, period=14):
    """Safe ATR calculation."""
    try:
        if df is None or len(df) < period:
            return pd.Series(dtype='float64')
        
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift(1)).abs()
        low_close = (df["Low"] - df["Close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()
    except Exception:
        return pd.Series(dtype='float64')

def williams_r(df, period=14):
    """Safe Williams %R calculation."""
    try:
        if df is None or len(df) < period:
            return pd.Series(dtype='float64')
        
        highest = df["High"].rolling(period).max()
        lowest = df["Low"].rolling(period).min()
        denom = (highest - lowest).replace(0, np.nan)
        return -100 * (highest - df["Close"]) / denom
    except Exception:
        return pd.Series(dtype='float64')

def volume_surge(df, lookback=20):
    """Safe volume surge calculation."""
    try:
        if df is None or len(df) < lookback:
            return pd.Series(dtype='float64')
        
        vol_ma = df["Volume"].rolling(lookback).mean()
        vol_std = df["Volume"].rolling(lookback).std()
        vol_std = vol_std.replace(0, np.nan)
        z_score = (df["Volume"] - vol_ma) / vol_std
        return z_score.fillna(0)
    except Exception:
        return pd.Series(dtype='float64')

def calculate_rsi(df, period=14):
    """Safe RSI calculation."""
    try:
        if df is None or len(df) < period + 1:
            return pd.Series(dtype='float64')
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        rs.fillna(100, inplace=True)
        return 100 - (100 / (1 + rs))
    except Exception:
        return pd.Series(dtype='float64')

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Safe MACD calculation."""
    try:
        if df is None or len(df) < slow + signal:
            return pd.Series(dtype='float64'), pd.Series(dtype='float64')
        
        ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    except Exception:
        return pd.Series(dtype='float64'), pd.Series(dtype='float64')

def calculate_stochastic(df, period=14, smooth_d=3):
    """FIXED: Safe Stochastic calculation."""
    try:
        if df is None or len(df) < period + smooth_d:
            return pd.Series(dtype='float64'), pd.Series(dtype='float64')
        
        low_min = df['Low'].rolling(window=period).min()
        high_max = df['High'].rolling(window=period).max()
        denom = (high_max - low_min).replace(0, np.nan)
        k = 100 * ((df['Close'] - low_min) / denom)
        k.fillna(50, inplace=True)
        d = k.rolling(window=smooth_d).mean()
        return k, d
    except Exception:
        return pd.Series(dtype='float64'), pd.Series(dtype='float64')

def calculate_adx(df, period=14):
    """Safe ADX calculation."""
    try:
        if df is None or len(df) < period * 2:
            return pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64')
        
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
        
        denom = (pdi + ndi).replace(0, np.nan)
        adx = (abs(pdi - ndi) / denom).ewm(com=period - 1, adjust=False).mean() * 100
        
        return adx.fillna(20), pdi.fillna(20), ndi.fillna(20)
    except Exception:
        empty_series = pd.Series(dtype='float64')
        return empty_series, empty_series, empty_series

def calculate_bollinger_bands(df, period=20, std_dev=2):
    """FIXED: Safe Bollinger Bands calculation."""
    try:
        if df is None or len(df) < period:
            empty_series = pd.Series(dtype='float64')
            return empty_series, empty_series, empty_series
        
        middle = df['Close'].rolling(window=period).mean()
        std = df['Close'].rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return middle, upper, lower
    except Exception:
        empty_series = pd.Series(dtype='float64')
        return empty_series, empty_series, empty_series

def calculate_roc(df, period=12):
    """Safe ROC calculation."""
    try:
        if df is None or len(df) < period + 1:
            return pd.Series(dtype='float64')
        
        shifted_close = df['Close'].shift(period).replace(0, np.nan)
        return ((df['Close'] - df['Close'].shift(period)) / shifted_close) * 100
    except Exception:
        return pd.Series(dtype='float64')

def calculate_obv(df):
    """Safe OBV calculation."""
    try:
        if df is None or len(df) < 2:
            return pd.Series(dtype='float64')
        
        return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    except Exception:
        return pd.Series(dtype='float64')

def calculate_cci(df, period=20):
    """Safe CCI calculation."""
    try:
        if df is None or len(df) < period:
            return pd.Series(dtype='float64')
        
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).replace(0, np.nan)
        return (tp - sma_tp) / (0.015 * mad)
    except Exception:
        return pd.Series(dtype='float64')

def cmf(df, period=20):
    """Safe CMF calculation."""
    try:
        if df is None or len(df) < period:
            return pd.Series(dtype='float64')
        
        denom = (df["High"] - df["Low"]).replace(0, np.nan)
        mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / denom
        mfm = mfm.fillna(0)
        mfv = mfm * df["Volume"]
        mfv_sum = mfv.rolling(period).sum()
        vol_sum = df["Volume"].rolling(period).sum().replace(0, np.nan)
        return (mfv_sum / vol_sum).fillna(0)
    except Exception:
        return pd.Series(dtype='float64')

def relative_volume(df, lookback=50):
    """Safe relative volume calculation."""
    try:
        if df is None or len(df) < lookback:
            return pd.Series(dtype='float64')
        
        vol_ma = df["Volume"].rolling(lookback).mean().replace(0, np.nan)
        return (df["Volume"] / vol_ma).fillna(1.0)
    except Exception:
        return pd.Series(dtype='float64')

def calculate_oi_volume_ratio(df):
    """Safe OI/Volume ratio calculation."""
    try:
        if df is None or len(df) < 1:
            return pd.Series(dtype='float64')
        
        if 'OpenInterest' not in df.columns:
            df = df.copy()
            df['OpenInterest'] = df['Volume'].rolling(20).mean() * 0.3
        
        volume_safe = df['Volume'].replace(0, np.nan)
        ratio = df['OpenInterest'] / volume_safe
        return ratio.fillna(0)
    except Exception:
        return pd.Series(dtype='float64')

def detect_oi_buildup(df, lookback=20):
    """Safe OI buildup detection."""
    try:
        if df is None or len(df) < lookback:
            return 0
        
        df = df.copy()
        if 'OpenInterest' not in df.columns:
            df['OpenInterest'] = df['Volume'].rolling(20).mean() * 0.3
        
        oi_ma = df['OpenInterest'].rolling(lookback).mean()
        
        if safe_series_check(oi_ma) and safe_series_check(df['OpenInterest']):
            current_oi = float(df['OpenInterest'].iloc[-1])
            avg_oi = float(oi_ma.iloc[-1])
            
            if avg_oi > 0:
                oi_strength = (current_oi - avg_oi) / avg_oi
                return max(min(oi_strength * 100, 100), -100)
        
        return 0
    except Exception:
        return 0

def volume_oi_sync_analysis(df):
    """Safe Volume/OI sync analysis."""
    try:
        if df is None or len(df) < 10:
            return 0
        
        df = df.copy()
        if 'OpenInterest' not in df.columns:
            df['OpenInterest'] = df['Volume'].rolling(20).mean() * 0.3
        
        vol_change = df['Volume'].pct_change(5).fillna(0)
        oi_change = df['OpenInterest'].pct_change(5).fillna(0)
        
        if safe_series_check(vol_change) and safe_series_check(oi_change):
            sync_score = float(vol_change.iloc[-1]) + float(oi_change.iloc[-1])
            return min(max(sync_score * 50, -100), 100)
        
        return 0
    except Exception:
        return 0

def option_buyer_momentum(df):
    """Safe option buyer momentum calculation."""
    try:
        if df is None or len(df) < 20:
            return 0
        
        if len(df) >= 5:
            price_mom = (float(df['Close'].iloc[-1]) / float(df['Close'].iloc[-5]) - 1) * 100
        else:
            price_mom = 0
        
        vol_surge_series = volume_surge(df, lookback=20)
        vol_surge_val = float(vol_surge_series.iloc[-1]) if safe_series_check(vol_surge_series) else 0
        
        oi_buildup = detect_oi_buildup(df, lookback=20)
        
        combined_score = (price_mom * 0.4) + (vol_surge_val * 0.3) + (oi_buildup * 0.3)
        return min(max(combined_score, -100), 100)
    except Exception:
        return 0
# ==============================================================================
# ENHANCED OPTION BUYER SCANNER v4.1 - Part 2
# Enhanced Option-Specific Indicators & Advanced Scoring Engine
# ==============================================================================

# ========== ENHANCED OPTION-SPECIFIC INDICATORS ==========
def option_flow_direction_analysis(df):
    """FIXED: Safe option flow direction analysis."""
    try:
        if df is None or len(df) < 10:
            return 0
        
        price_changes = df['Close'].pct_change(periods=5).fillna(0)
        volume_changes = df['Volume'].pct_change(periods=5).fillna(0)
        
        if not safe_series_check(price_changes) or not safe_series_check(volume_changes):
            return 0
        
        recent_price_change = float(price_changes.iloc[-1])
        recent_volume_change = float(volume_changes.iloc[-1])
        
        if recent_price_change > 0.02 and recent_volume_change > 0.5:
            return 75
        elif recent_price_change > 0.01 and recent_volume_change > 0.2:
            return 50
        elif recent_price_change < -0.02 and recent_volume_change > 0.5:
            return -75
        elif recent_price_change < -0.01 and recent_volume_change > 0.2:
            return -50
        else:
            return 0
            
    except Exception as e:
        logger.error(f"Error in option flow direction analysis: {e}")
        return 0

def unusual_option_activity_detector(df):
    """FIXED: Safe unusual option activity detector."""
    try:
        if df is None or len(df) < 20:
            return 0
        
        vol_zscore = volume_surge(df, lookback=20)
        if not safe_series_check(vol_zscore):
            return 0
        
        recent_zscore = float(vol_zscore.iloc[-1])
        
        if len(df) >= 2:
            price_change_5min = (float(df['Close'].iloc[-1]) / float(df['Close'].iloc[-2]) - 1) * 100
        else:
            price_change_5min = 0
        
        if recent_zscore > 2.5 and abs(price_change_5min) > 1:
            return 85
        elif recent_zscore > 2.0:
            return 65
        elif recent_zscore > 1.5:
            return 45
        else:
            return 0
            
    except Exception as e:
        logger.error(f"Error in unusual activity detection: {e}")
        return 0

def strike_momentum_analysis(df):
    """FIXED: Safe strike momentum analysis."""
    try:
        if df is None or len(df) < 15:
            return 0
        
        close_prices = df['Close']
        if not safe_series_check(close_prices, 2):
            return 0
        
        if len(close_prices) >= 2:
            mom_1min = (float(close_prices.iloc[-1]) / float(close_prices.iloc[-2]) - 1) * 100
        else:
            mom_1min = 0
        
        if len(close_prices) >= 6:
            mom_5min = (float(close_prices.iloc[-1]) / float(close_prices.iloc[-6]) - 1) * 100
        else:
            mom_5min = 0
        
        if len(close_prices) >= 16:
            mom_15min = (float(close_prices.iloc[-1]) / float(close_prices.iloc[-16]) - 1) * 100
        else:
            mom_15min = 0
        
        atr_series = atr(df)
        volatility_expansion = 0
        
        if safe_series_check(atr_series, 24):
            atr_current = float(atr_series.iloc[-1])
            atr_avg_series = atr_series.rolling(10).mean()
            if safe_series_check(atr_avg_series):
                atr_avg = float(atr_avg_series.iloc[-1])
                if atr_avg > 0:
                    volatility_expansion = (atr_current / atr_avg - 1) * 100
        
        momentum_score = (mom_1min * 0.2 + mom_5min * 0.5 + mom_15min * 0.3) * 2
        volatility_score = min(volatility_expansion * 0.5, 25)
        
        combined_score = momentum_score + volatility_score
        return max(min(combined_score, 100), -100)
        
    except Exception as e:
        logger.error(f"Error in strike momentum analysis: {e}")
        return 0

def iv_expansion_detector(df):
    """FIXED: Safe IV expansion detector."""
    try:
        if df is None or len(df) < 20:
            return 0
        
        returns = df['Close'].pct_change().dropna()
        if not safe_series_check(returns, 5):
            return 0
        
        current_vol_series = returns.rolling(5).std()
        avg_vol_series = returns.rolling(20).std()
        
        if not safe_series_check(current_vol_series) or not safe_series_check(avg_vol_series):
            return 0
        
        current_vol = float(current_vol_series.iloc[-1]) * np.sqrt(252)
        avg_vol = float(avg_vol_series.mean()) * np.sqrt(252)
        
        if avg_vol > 0:
            vol_expansion_ratio = current_vol / avg_vol
        else:
            vol_expansion_ratio = 1
        
        atr_expansion = 0
        atr_series = atr(df, 14)
        
        if safe_series_check(atr_series, 20):
            current_atr = float(atr_series.iloc[-1])
            if len(atr_series) >= 11:
                atr_ma_series = atr_series.rolling(10).mean()
                if safe_series_check(atr_ma_series):
                    avg_atr = float(atr_ma_series.iloc[-11])
                    if avg_atr > 0:
                        atr_expansion = (current_atr / avg_atr - 1) * 100
        
        vol_score = (vol_expansion_ratio - 1) * 100
        combined_score = (vol_score * 0.7 + atr_expansion * 0.3)
        
        return max(min(combined_score, 100), -50)
        
    except Exception as e:
        logger.error(f"Error in IV expansion detection: {e}")
        return 0

def gamma_expansion_analysis(df):
    """FIXED: Safe gamma expansion analysis."""
    try:
        if df is None or len(df) < 10:
            return 0
        
        close_prices = df['Close']
        if not safe_series_check(close_prices):
            return 0
        
        current_price = float(close_prices.iloc[-1])
        
        price_std_series = close_prices.rolling(20).std()
        if safe_series_check(price_std_series):
            price_std = float(price_std_series.iloc[-1])
        else:
            price_std = current_price * 0.02
        
        round_levels = [int(current_price / 50) * 50, (int(current_price / 50) + 1) * 50]
        min_distance = min([abs(current_price - level) for level in round_levels])
        
        recent_volume_series = df['Volume'].iloc[-5:]
        if safe_series_check(recent_volume_series):
            recent_volume = float(recent_volume_series.mean())
        else:
            recent_volume = 0
        
        avg_volume_series = df['Volume'].rolling(20).mean()
        if safe_series_check(avg_volume_series):
            avg_volume = float(avg_volume_series.iloc[-1])
        else:
            avg_volume = recent_volume
        
        if avg_volume > 0:
            volume_ratio = recent_volume / avg_volume
        else:
            volume_ratio = 1
        
        if price_std > 0:
            distance_score = max(0, (1 - min_distance / (price_std * 2))) * 50
        else:
            distance_score = 0
        
        volume_score = min((volume_ratio - 1) * 30, 30)
        
        combined_score = distance_score + volume_score
        return max(min(combined_score, 100), 0)
        
    except Exception as e:
        logger.error(f"Error in gamma expansion analysis: {e}")
        return 0

def market_regime_filter(market_regime, vix_signal):
    """FIXED: Safe market regime filter."""
    try:
        base_score = 0
        
        if isinstance(vix_signal, (int, float)):
            if vix_signal > 1:
                base_score += 20
            elif vix_signal < -0.5:
                base_score -= 10
        
        if market_regime == 'bullish':
            base_score += 15
        elif market_regime == 'bearish':
            base_score += 10
        
        return max(min(base_score, 50), -25)
        
    except Exception as e:
        logger.error(f"Error in market regime filter: {e}")
        return 0

def institutional_flow_detector(df):
    """FIXED: Safe institutional flow detector."""
    try:
        if df is None or len(df) < 30:
            return 0
        
        volume_series = df['Volume']
        if not safe_series_check(volume_series, 20):
            return 0
        
        volume_ma = volume_series.rolling(20).mean()
        volume_std = volume_series.rolling(20).std()
        
        if not safe_series_check(volume_ma) or not safe_series_check(volume_std):
            return 0
        
        threshold = volume_ma.iloc[-1] + 2 * volume_std.iloc[-1]
        recent_volumes = volume_series.iloc[-5:]
        
        large_blocks = recent_volumes[recent_volumes > threshold]
        
        price_changes = df['Close'].pct_change()
        if safe_series_check(price_changes, 5):
            recent_price_impact = float(price_changes.iloc[-5:].mean()) * 100
        else:
            recent_price_impact = 0
        
        block_score = len(large_blocks) * 15
        impact_score = abs(recent_price_impact) * 5 if abs(recent_price_impact) > 0.5 else 0
        
        combined_score = min(block_score + impact_score, 75)
        
        if recent_price_impact > 0:
            return combined_score
        else:
            return -combined_score
            
    except Exception as e:
        logger.error(f"Error in institutional flow detection: {e}")
        return 0

def slope(series, lookback=10):
    """FIXED: Safe slope calculation."""
    try:
        if not safe_series_check(series, lookback):
            return 0.0
        
        y = series.tail(lookback).values
        x = np.arange(len(y))
        
        if len(y) < 2:
            return 0.0
        
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])
    except Exception:
        return 0.0

# ========== ENHANCED SCORING ENGINE ==========
def normalize_score(value, bullish_range, bearish_range, score_range=(-2.0, 2.0)):
    """Safe score normalization."""
    try:
        if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
            return 0.0
        
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
    except Exception:
        return 0.0

def calculate_enhanced_indicator_scores(df, symbol, current_price):
    """FIXED: Enhanced indicator calculation with proper error handling."""
    scores = defaultdict(float)
    
    try:
        if df is None or len(df) < 50:
            return scores

        # FIXED: ADX calculation
        adx, pdi, ndi = calculate_adx(df)
        if (safe_series_check(adx, 4) and safe_series_check(pdi) and safe_series_check(ndi) and
            len(adx) > 3 and float(adx.iloc[-1]) > 20 and float(adx.iloc[-1]) > float(adx.iloc[-3])):
            if float(pdi.iloc[-1]) > float(ndi.iloc[-1]):
                scores['ADX'] = 2.0
            else:
                scores['ADX'] = -2.0
        
        # FIXED: EMA calculation
        ema20 = ema(df['Close'], 20)
        ema50 = ema(df['Close'], 50)
        if safe_series_check(ema20) and safe_series_check(ema50):
            ema20_val = float(ema20.iloc[-1])
            ema50_val = float(ema50.iloc[-1])
            if ema50_val != 0:
                ema_ratio = ema20_val / ema50_val
                scores['EMA'] = normalize_score(ema_ratio, (1.001, 1.02), (0.999, 0.98))

        # FIXED: VWAP calculation
        vwap_line = vwap(df, period=None)
        if safe_series_check(vwap_line) and current_price > 0:
            vwap_val = float(vwap_line.iloc[-1])
            if vwap_val != 0:
                vwap_ratio = current_price / vwap_val
                scores['VWAP'] = normalize_score(vwap_ratio, (1.002, 1.025), (0.998, 0.975))

        # FIXED: MACD calculation
        macd, signal = calculate_macd(df)
        if safe_series_check(macd) and safe_series_check(signal):
            macd_val = float(macd.iloc[-1])
            signal_val = float(signal.iloc[-1])
            
            if macd_val > signal_val and macd_val > 0:
                scores['MACD_Trend'] = 2.0
            elif macd_val < signal_val and macd_val < 0:
                scores['MACD_Trend'] = -2.0
            else:
                scores['MACD_Trend'] = 0

        # FIXED: MA Slope calculation
        if safe_series_check(ema20, 5) and current_price > 0:
            ma20_slope = slope(ema20, 5)
            price_norm_slope = ma20_slope / current_price * 1000
            scores['MA_Slope'] = normalize_score(price_norm_slope, (0.1, 0.5), (-0.1, -0.5), (-2.5, 2.5))
        
        # FIXED: RSI calculation
        rsi = calculate_rsi(df)
        if safe_series_check(rsi):
            rsi_val = float(rsi.iloc[-1])
            scores['RSI'] = normalize_score(rsi_val, (60, 85), (40, 15))

        # FIXED: Stochastic calculation
        k, d = calculate_stochastic(df)
        if safe_series_check(k) and safe_series_check(d):
            k_val = float(k.iloc[-1])
            d_val = float(d.iloc[-1])
            
            if k_val > d_val:
                scores['Stochastic'] = normalize_score(k_val, (20, 80), (100, 100))
            else:
                scores['Stochastic'] = normalize_score(k_val, (0, 0), (80, 20))

        # FIXED: CCI calculation
        cci = calculate_cci(df)
        if safe_series_check(cci):
            cci_val = float(cci.iloc[-1])
            scores['CCI'] = normalize_score(cci_val, (100, 200), (-100, -200))

        # FIXED: ROC calculation
        roc = calculate_roc(df)
        if safe_series_check(roc):
            roc_val = float(roc.iloc[-1])
            scores['ROC'] = normalize_score(roc_val, (0.5, 2.0), (-0.5, -2.0))

        # FIXED: Williams %R calculation
        wr = williams_r(df)
        if safe_series_check(wr):
            wr_val = float(wr.iloc[-1])
            scores['WilliamsR'] = normalize_score(wr_val, (-100, -80), (-20, 0))

        # FIXED: Volume indicators
        zscore = volume_surge(df, lookback=20)
        if safe_series_check(zscore, 2) and len(df) >= 2:
            zscore_val = float(zscore.iloc[-1])
            price_up = float(df['Close'].iloc[-1]) > float(df['Close'].iloc[-2])
            
            if price_up:
                scores['VolumeSurge'] = normalize_score(zscore_val, (1.5, 3.0), (0, 0))
            else:
                scores['VolumeSurge'] = normalize_score(zscore_val, (0, 0), (-1.5, -3.0))

        # FIXED: OBV calculation
        obv_line = calculate_obv(df)
        if safe_series_check(obv_line, 5):
            obv_slope = slope(obv_line, 5)
            scores['OBV'] = normalize_score(obv_slope, (1, 1e9), (-1, -1e9))

        # FIXED: CMF calculation
        cmf20 = cmf(df, period=20)
        if safe_series_check(cmf20):
            cmf_val = float(cmf20.iloc[-1])
            scores['CMF'] = normalize_score(cmf_val, (0.1, 0.25), (-0.1, -0.25))

        # FIXED: Relative Volume calculation
        rv = relative_volume(df, lookback=50)
        if safe_series_check(rv):
            rv_val = float(rv.iloc[-1])
            scores['RelVol'] = normalize_score(rv_val, (1.5, 3.0), (0.5, 0.5))

        # FIXED: Volatility expansion calculation
        atr_val = atr(df, period=14)
        if safe_series_check(atr_val, 20):
            atr_ma = atr_val.rolling(20).mean()
            if safe_series_check(atr_ma):
                atr_current = float(atr_val.iloc[-1])
                atr_avg = float(atr_ma.iloc[-1])
                
                if atr_avg > 0:
                    atr_ratio = atr_current / atr_avg
                    
                    if len(atr_val) >= 5:
                        atr_slope_ratio = float(atr_val.iloc[-1]) / float(atr_val.iloc[-5]) if float(atr_val.iloc[-5]) > 0 else 1
                    else:
                        atr_slope_ratio = 1
                    
                    if atr_ratio > 1.1 and atr_slope_ratio > 1.1 and len(df) >= 5:
                        price_direction = 1 if float(df['Close'].iloc[-1]) > float(df['Close'].iloc[-5]) else -1
                        scores['VolatilityExpansion'] = 3.0 * price_direction

        # FIXED: Bollinger Bands calculation
        _, bb_upper, bb_lower = calculate_bollinger_bands(df)
        if safe_series_check(bb_upper) and safe_series_check(bb_lower) and current_price > 0:
            bb_upper_val = float(bb_upper.iloc[-1])
            bb_lower_val = float(bb_lower.iloc[-1])
            
            if current_price > bb_upper_val:
                scores['Bollinger'] = 2.0
            elif current_price < bb_lower_val:
                scores['Bollinger'] = -2.0

        # Original OI indicators (already safe)
        scores['OIChange'] = normalize_score(detect_oi_buildup(df, 20), (10, 30), (-10, -30))
        scores['VolumeOISync'] = normalize_score(volume_oi_sync_analysis(df), (15, 40), (-15, -40))
        scores['OptionBuyerMomentum'] = normalize_score(option_buyer_momentum(df), (20, 50), (-20, -50), (-3.0, 3.0))

        # FIXED: Enhanced option-specific indicators
        scores['OptionFlowDirection'] = normalize_score(option_flow_direction_analysis(df), (40, 75), (-40, -75), (-3.2, 3.2))
        scores['UnusualOptionActivity'] = normalize_score(unusual_option_activity_detector(df), (45, 85), (0, 0), (0, 3.0))
        scores['StrikeMomentum'] = normalize_score(strike_momentum_analysis(df), (30, 70), (-30, -70), (-2.8, 2.8))
        scores['IVExpansion'] = normalize_score(iv_expansion_detector(df), (20, 60), (-20, 0), (-1.0, 3.0))
        scores['GammaExpansion'] = normalize_score(gamma_expansion_analysis(df), (30, 70), (0, 0), (0, 2.2))
        scores['InstitutionalFlow'] = normalize_score(institutional_flow_detector(df), (25, 75), (-25, -75), (-2.5, 2.5))
        
        # VIX signal
        vix_signal = get_vix_signal()
        scores['VIXSignal'] = normalize_score(vix_signal, (0.5, 1.5), (-1.0, -0.3), (-2.0, 2.0))

    except Exception as e:
        logger.error(f"Error calculating enhanced indicator scores: {e}")

    return scores

def analyze_enhanced_signals_pro(timeframe_data, market_regime='neutral', symbol=None):
    """FIXED: Enhanced signal analysis with proper error handling."""
    try:
        total_score, total_weight = 0.0, 0.0
        group_scores = defaultdict(float)
        group_weights = defaultdict(float)
        
        current_price = 0
        for tf_min, df in timeframe_data.items():
            if df is not None and len(df) > 0:
                current_price = float(df['Close'].iloc[-1])
                break

        for tf_min, df in timeframe_data.items():
            if df is None or len(df) < 50:
                continue
            
            indicator_scores = calculate_enhanced_indicator_scores(df, symbol, current_price)
            tf_weight = TIMEFRAME_WEIGHTS.get(tf_min, 1.0)

            vix_signal = get_vix_signal()

            for group, weight in Config.GROUP_WEIGHTS.items():
                grp_score, grp_weight = 0.0, 0.0
                
                for indicator, ind_weight in Config.INDICATOR_WEIGHTS.items():
                    if indicator in indicator_scores:
                        belongs_to_group = (
                            (group == 'Trend' and any(term in indicator for term in ['MA', 'ADX', 'VWAP', 'EMA', 'MACD'])) or
                            (group == 'Momentum' and any(term in indicator for term in ['RSI', 'Stochastic', 'CCI', 'ROC', 'Williams'])) or
                            (group == 'Volume' and any(term in indicator for term in ['Vol', 'OBV', 'CMF']) and 'Option' not in indicator) or
                            (group == 'Volatility' and any(term in indicator for term in ['Volatility', 'Bollinger', 'IVExpansion'])) or
                            (group == 'OI' and any(term in indicator for term in ['OI', 'Option']) and 'Flow' not in indicator) or
                            (group == 'OptionFlow' and any(term in indicator for term in ['OptionFlow', 'Unusual', 'Strike', 'Institutional'])) or
                            (group == 'Greeks' and any(term in indicator for term in ['Gamma', 'IV'])) or
                            (group == 'MarketRegime' and any(term in indicator for term in ['VIX', 'MarketRegime']))
                        )
                        
                        if belongs_to_group:
                            score = float(indicator_scores[indicator]) * float(ind_weight)
                            grp_score += score
                            grp_weight += abs(float(indicator_scores[indicator])) * float(ind_weight)

                if grp_weight > 0:
                    norm_grp_score = (grp_score / grp_weight) * weight * tf_weight
                    group_scores[group] += norm_grp_score
                    group_weights[group] += weight * tf_weight
        
        final_score = sum(group_scores.values())
        max_possible_score = sum(group_weights.values())

        if max_possible_score == 0:
            return 'Neutral', 0.0, {}
        
        normalized_score = (final_score / max_possible_score) * 100

        # Market regime adjustments
        vix_signal = get_vix_signal()
        
        if normalized_score > 0:
            if market_regime == 'bullish':
                normalized_score *= Config.REGIME_MULTIPLIERS['bullish_in_bull_market']
            elif market_regime == 'bearish':
                normalized_score *= Config.REGIME_MULTIPLIERS['bullish_in_bear_market']
            
            if vix_signal > 1.0:
                normalized_score *= Config.REGIME_MULTIPLIERS['high_vix_expansion']
            elif vix_signal < -0.5:
                normalized_score *= Config.REGIME_MULTIPLIERS['low_vix_compression']
        else:
            if market_regime == 'bearish':
                normalized_score *= Config.REGIME_MULTIPLIERS['bearish_in_bear_market']
            elif market_regime == 'bullish':
                normalized_score *= Config.REGIME_MULTIPLIERS['bearish_in_bull_market']

        # Signal classification
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
                final_sub_scores[group] = (group_scores[group] / group_weights[group]) * 10
        
        return signal, normalized_score, final_sub_scores
        
    except Exception as e:
        logger.error(f"Error in enhanced signal analysis: {e}")
        return 'Neutral', 0.0, {}

# ========== DATA FETCHING INFRASTRUCTURE ==========
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
    return TD_hist(Config.TDUSERNAME, Config.TDPASSWORD, log_level=logging.CRITICAL)

def build_sessions():
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
    if df is None or len(df) == 0: return None
    try:
        out = df.copy()
        out.rename(columns={c: str(c).lower() for c in out.columns}, inplace=True)
        rename_map = {}
        for src, tgt in (("timestamp", "Date"), ("time", "Date"), ("datetime", "Date"),("date", "Date"),
                         ("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"),
                         ("volume", "Volume"), ("vol", "Volume")):
            if src in out.columns: rename_map[src] = tgt
        out.rename(columns=rename_map, inplace=True)
        if "Date" not in out.columns and isinstance(out.index, pd.DatetimeIndex): 
            out["Date"] = out.index
        elif "Date" not in out.columns: 
            return None
        if "Volume" not in out.columns: 
            out["Volume"] = 0
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"])
        if pd.api.types.is_datetime64tz_dtype(out["Date"]): 
            out["Date"] = out["Date"].dt.tz_convert(IST)
        else: 
            out["Date"] = out["Date"].dt.tz_localize(IST)
        for c in ["Open", "High", "Low", "Close", "Volume"]: 
            out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")
        out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_values("Date").set_index("Date")
        return out[~out.index.duplicated(keep='last')]
    except Exception as e:
        logger.error(f"Normalize error {symbol}: {e}")
        return None

def pick_session(symbol_orig, timeframe_minutes):
    return (hash(symbol_orig) ^ timeframe_minutes) % len(tdhist_pool)

def fetch_one_timeaware(symbol_orig, timeframe_minutes, limiter, hist, up_to_time):
    td_symbol = symbol_orig.replace('-EQ', '')
    bar_size, duration = BAR_SIZE_MAP.get(timeframe_minutes), DURATION_MAP.get(timeframe_minutes)
    if not bar_size or not duration: return symbol_orig, timeframe_minutes, None
    
    try:
        limiter.acquire()
        df_raw = hist.get_historic_data(td_symbol, duration=duration, bar_size=bar_size)
        df = normalize_hist_df(df_raw, td_symbol)
        
        if df is not None and up_to_time:
            df = df[df.index <= up_to_time]
            
        global api_calls_done
        with api_calls_lock: api_calls_done += 1
        return symbol_orig, timeframe_minutes, df
    except Exception as e:
        logger.error(f"Error fetching {symbol_orig} {timeframe_minutes}min: {e}")
        return symbol_orig, timeframe_minutes, None

def fetch_one(symbol_orig, timeframe_minutes, limiter, hist):
    return fetch_one_timeaware(symbol_orig, timeframe_minutes, limiter, hist, None)

def prefetch_all_timeaware(stocks, up_to_time=None, max_workers=Config.MAX_WORKERS):
    tfs = [5, 15, 30, 60, 1440]
    total_calls, stock_multi_data = len(stocks) * len(tfs), defaultdict(dict)
    
    global api_calls_done
    with api_calls_lock: api_calls_done = 0
    
    desc = f"Fetching data up to {up_to_time.strftime('%H:%M')}" if up_to_time else "Prefetching Data"
    with tqdm(total=total_calls, desc=desc, ncols=100, leave=False) as api_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for s in stocks:
                for tf in tfs:
                    session_idx = pick_session(s, tf)
                    if up_to_time:
                        futures.append(executor.submit(
                            fetch_one_timeaware, s, tf, sess_limiters[session_idx], 
                            tdhist_pool[session_idx], up_to_time
                        ))
                    else:
                        futures.append(executor.submit(
                            fetch_one, s, tf, sess_limiters[session_idx], tdhist_pool[session_idx]
                        ))
            
            for fut in as_completed(futures):
                symbol_orig, tf, df = fut.result()
                if df is not None and len(df) > 0:
                    stock_multi_data[symbol_orig][tf] = df
                api_bar.update(1)
    
    return {s: d for s, d in stock_multi_data.items() if len(d) >= 2}

def prefetch_all(stocks, max_workers=Config.MAX_WORKERS):
    return prefetch_all_timeaware(stocks, None, max_workers)

def get_market_regime(index_symbol="NIFTY 50"):
    try:
        si = pick_session(index_symbol, 1440)
        df_raw = tdhist_pool[si].get_historic_data(index_symbol, duration="200 D", bar_size="1 day")
        df = normalize_hist_df(df_raw, index_symbol)
        
        if df is None or len(df) < 50:
            return 'neutral'
        
        ema20_series = ema(df['Close'], 20)
        ema50_series = ema(df['Close'], 50)
        
        if not safe_series_check(ema20_series) or not safe_series_check(ema50_series):
            return 'neutral'
        
        ema20_val = float(ema20_series.iloc[-1])
        ema50_val = float(ema50_series.iloc[-1])
        close = float(df['Close'].iloc[-1])
        
        if close > ema20_val and ema20_val > ema50_val:
            return 'bullish'
        elif close < ema20_val and ema20_val < ema50_val:
            return 'bearish'
        else:
            return 'neutral'
    except Exception as e:
        logger.warning(f"Could not fetch market regime for {index_symbol}: {e}")
        return 'neutral'

def enhanced_institutional_flow_analysis(tf_data):
    """FIXED: Safe institutional flow analysis with proper DataFrame checks."""
    try:
        # FIXED: Safer DataFrame filtering
        frames = []
        for t in [5, 15, 30]:
            df = tf_data.get(t)
            if (df is not None and 
                hasattr(df, 'empty') and 
                not df.empty and 
                len(df) > 60):
                frames.append(df)
        
        if not frames:
            return 'Unknown'
        
        votes = 0
        for df in frames:
            try:
                cmf_series = cmf(df, 20)
                rv_series = relative_volume(df, 50)
                
                if not safe_series_check(cmf_series) or not safe_series_check(rv_series):
                    continue
                
                cmf_last = float(cmf_series.iloc[-1])
                rv_last = float(rv_series.iloc[-1])
                
                if cmf_last > 0.05 and rv_last > 1.2:
                    votes += 1
                elif cmf_last < -0.05 and rv_last > 1.2:
                    votes -= 1
                    
            except Exception as frame_error:
                logger.error(f"Error in flow analysis frame: {frame_error}")
                continue
        
        if votes >= 2:
            return 'Institutional Accumulation'
        elif votes <= -2:
            return 'Institutional Distribution'
        else:
            return 'Mixed/Neutral'
            
    except Exception as e:
        logger.error(f"Error in enhanced institutional flow analysis: {e}")
        return 'Unknown'

# ==============================================================================
# ENHANCED OPTION BUYER SCANNER v4.1 - Part 3 (FINAL)
# Scanner Functions, Enhanced Display Tables & Main Application
# ==============================================================================

# ========== VOLUME/OI TRACKING FUNCTIONS ==========
def calculate_5min_volume_oi_changes(df, symbol, scan_time):
    """Calculate volume/OI changes from 5-minute timeframe data."""
    try:
        df_5min = df[df.index <= scan_time] if scan_time else df
        
        if len(df_5min) < 2:
            return 0, 0, 0, 0
        
        current_volume = int(df_5min['Volume'].iloc[-1])
        current_oi = int(df_5min.get('OpenInterest', df_5min['Volume'] * 0.3).iloc[-1])
        
        previous_volume = int(df_5min['Volume'].iloc[-2])
        previous_oi = int(df_5min.get('OpenInterest', df_5min['Volume'] * 0.3).iloc[-2])
        
        vol_change_pct = ((current_volume - previous_volume) / previous_volume * 100) if previous_volume > 0 else 0
        oi_change_pct = ((current_oi - previous_oi) / previous_oi * 100) if previous_oi > 0 else 0
        
        return current_volume, current_oi, vol_change_pct, oi_change_pct
        
    except Exception as e:
        logger.error(f"Error calculating 5-min changes for {symbol}: {e}")
        return 0, 0, 0, 0

def extract_5min_volume_oi_data(df, symbol, time_point=None, is_live=False):
    """FINAL FIXED: Safe volume/OI data extraction with DataFrame validation."""
    try:
        global intraday_volume_data, intraday_oi_data
        
        # FIXED: Safe DataFrame validation before operations
        if not safe_dataframe_check(df):
            return {
                'current_volume': 'N/A', 'current_oi': 'N/A',
                'volume_change_pct': 0, 'oi_change_pct': 0,
                'volume': 'N/A', 'oi': 'N/A', 'volume_change': 'N/A', 'oi_change': 'N/A',
                'current_price': 0
            }
        
        # FIXED: Safe DataFrame slicing
        df_slice = df
        if time_point and not is_live:
            try:
                df_slice = df[df.index <= time_point]
                if not safe_dataframe_check(df_slice):
                    df_slice = df  # Fallback to original DataFrame
            except Exception as slice_error:
                logger.error(f"DataFrame slicing error for {symbol}: {slice_error}")
                df_slice = df  # Use original DataFrame
        
        # FIXED: Safe price extraction
        try:
            current_price = float(df_slice['Close'].iloc[-1])
        except Exception as price_error:
            logger.error(f"Price extraction error for {symbol}: {price_error}")
            current_price = 0
        
        # Calculate volume/OI changes safely
        current_volume, current_oi, vol_change_pct, oi_change_pct = calculate_5min_volume_oi_changes(
            df_slice, symbol, time_point
        )
        
        # Rest of the function remains the same...
        if abs(vol_change_pct) < 0.1 and abs(oi_change_pct) < 0.1:
            prev_volume = intraday_volume_data.get(symbol, None)
            prev_oi = intraday_oi_data.get(symbol, None)
            
            if prev_volume is not None and prev_volume > 0 and current_volume > 0:
                vol_change_pct = ((current_volume - prev_volume) / prev_volume * 100)
            if prev_oi is not None and prev_oi > 0 and current_oi > 0:
                oi_change_pct = ((current_oi - prev_oi) / prev_oi * 100)
        
        intraday_volume_data[symbol] = current_volume
        intraday_oi_data[symbol] = current_oi
        
        current_volume_display = f"{current_volume:,}" if current_volume > 999 else str(current_volume)
        current_oi_display = f"{current_oi:,}" if current_oi > 999 else str(current_oi)
        
        volume_change_legacy = f"{vol_change_pct:+.1f}%" if abs(vol_change_pct) > 0.1 else "N/A"
        oi_change_legacy = f"{oi_change_pct:+.1f}%" if abs(oi_change_pct) > 0.1 else "N/A"
        
        return {
            'current_volume': current_volume_display,
            'current_oi': current_oi_display,
            'volume_change_pct': vol_change_pct if abs(vol_change_pct) > 0.1 else 0,
            'oi_change_pct': oi_change_pct if abs(oi_change_pct) > 0.1 else 0,
            'volume': current_volume_display,
            'oi': current_oi_display,
            'volume_change': volume_change_legacy,
            'oi_change': oi_change_legacy,
            'current_price': current_price,
            '_raw_volume': current_volume,
            '_raw_oi': current_oi
        }
        
    except Exception as e:
        logger.error(f"Error extracting 5-min data for {symbol}: {e}")
        return {
            'current_volume': 'N/A', 'current_oi': 'N/A',
            'volume_change_pct': 0, 'oi_change_pct': 0,
            'volume': 'N/A', 'oi': 'N/A', 'volume_change': 'N/A', 'oi_change': 'N/A',
            'current_price': 0
        }

# ========== TIMING FUNCTIONS ==========
def generate_backtest_timestamps(backtest_date):
    """Generate all 5-minute scan timestamps starting from 09:20:15."""
    timestamps = []
    base_date = IST.localize(datetime.strptime(backtest_date, "%Y-%m-%d"))
    
    current_time = base_date.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = base_date.replace(hour=15, minute=30, second=0, microsecond=0)
    
    first_scan = current_time + timedelta(minutes=5, seconds=15)
    timestamps.append(first_scan)
    
    current_scan = first_scan
    while current_scan < market_end:
        current_scan += timedelta(minutes=5)
        if current_scan <= market_end:
            timestamps.append(current_scan)
    
    return timestamps

def next_5min_boundary_ist(now_ist: datetime) -> datetime:
    minute = (now_ist.minute // 5) * 5
    boundary = now_ist.replace(minute=minute, second=0, microsecond=0)
    if boundary <= now_ist:
        boundary = boundary + timedelta(minutes=5)
    return boundary

def get_exact_candle_close_time(now_ist: datetime) -> datetime:
    next_boundary = next_5min_boundary_ist(now_ist)
    return next_boundary + timedelta(seconds=Config.SETTLE_DELAY_SECONDS)

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

# ========== ENHANCED OPTION ACTION RECOMMENDATIONS ==========
def get_enhanced_action_recommendation(item, best_strike):
    """Get enhanced action recommendation for option trading."""
    score = item['score']
    signal = item['signal']
    
    if not best_strike:
        return "No Liquid Options"
    
    strike_type = best_strike.get('type', 'ATM')
    
    if 'Very Strong' in signal:
        if score > 0:
            if strike_type == 'ATM_MOMENTUM':
                return f"🚀 ATM Call {best_strike['strike']}"
            else:
                return f"🎯 OTM Call {best_strike['strike']}"
        else:
            return f"🔻 Put {best_strike['strike']}"
    elif 'Strong' in signal:
        if score > 0:
            return f"📈 Call {best_strike['strike']}"
        else:
            return f"📉 Put {best_strike['strike']}"
    elif 'Buy' in signal:
        if score > 0:
            return f"💡 Consider Call {best_strike['strike']}"
        else:
            return f"💡 Consider Put {best_strike['strike']}"
    else:
        return "⏸️ Hold/Monitor"

# ========== ENHANCED TABLE CREATION FUNCTIONS ==========
def create_great_table_fixed(data, title, new_stocks=None, show_time=None):
    """FIXED: Create beautiful tables using great-tables with option enhancements."""
    if not data or not GREAT_TABLES_AVAILABLE:
        create_rich_enhanced_option_table(data, title, new_stocks, show_time)
        return

    try:
        df_data = []
        for item in data:
            # Get option analysis
            current_price = item.get('current_price', 0)
            option_chain = simulate_option_chain_data(item['symbol'], current_price)
            option_analysis = analyze_option_chain(item['symbol'], current_price, option_chain)
            recommended_strikes = option_analysis.get('recommended_strikes', [])
            best_strike = recommended_strikes[0] if recommended_strikes else None
            
            row = {
                'Stock': item['symbol'],
                'Signal': item['signal'],
                'Score': round(item['score'], 2),
                'Trend': round(item['sub_scores'].get('Trend', 0), 2),
                'Momentum': round(item['sub_scores'].get('Momentum', 0), 2),
                'Volume': round(item['sub_scores'].get('Volume', 0), 2),
                'OI': round(item['sub_scores'].get('OI', 0), 2),
                'OptFlow': round(item['sub_scores'].get('OptionFlow', 0), 2),
                'Greeks': round(item['sub_scores'].get('Greeks', 0), 2),
                'CurrVol': item.get('current_volume', 'N/A'),
                'CurrOI': item.get('current_oi', 'N/A'),
                'VolChange': item.get('volume_change_pct', 0),
                'OIChange': item.get('oi_change_pct', 0),
                'Strike': f"{best_strike['strike']}" if best_strike else 'N/A',
                'Delta': f"{best_strike['delta']:.2f}" if best_strike else 'N/A',
                'Flow': item.get('flow', 'Unknown'),
                'Action': get_enhanced_action_recommendation(item, best_strike),
                'Is_New': 1 if (new_stocks and item['symbol'] in new_stocks) else 0
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        
        # Enhanced console display with option focus
        print("\n" + "="*180)
        if show_time:
            print(f"📊 {title} - {show_time}")
        else:
            print(f"📊 {title}")
        print("="*180)
        
        print("✨ Enhanced Option Scanner Display:")
        for i, row in df.iterrows():
            marker = "🆕 " if row['Is_New'] == 1 else "   "
            
            # Format volume and OI changes
            vol_chg = row['VolChange']
            oi_chg = row['OIChange']
            
            if isinstance(vol_chg, (int, float)) and abs(vol_chg) > 0.1:
                vol_chg_str = f"{vol_chg:+.1f}%"
            else:
                vol_chg_str = "N/A"
                
            if isinstance(oi_chg, (int, float)) and abs(oi_chg) > 0.1:
                oi_chg_str = f"{oi_chg:+.1f}%"
            else:
                oi_chg_str = "N/A"
            
            color = Colors.MAGENTA if row['Is_New'] == 1 else Colors.END
            
            display_text = (f"{marker}{row['Stock']:<12} | {row['Signal']:<16} | {row['Score']:>7.2f} | "
                          f"{row['Strike']:<8} | {row['Delta']:<6} | {vol_chg_str:<8} | {oi_chg_str:<8} | "
                          f"{row['Action']:<25}")
            print_colored(display_text, color)
        
        print("="*180)
        
    except Exception as e:
        logger.error(f"Error creating great table: {e}")
        create_rich_enhanced_option_table(data, title, new_stocks, show_time)

def create_rich_enhanced_option_table(data, title, new_stocks=None, show_time=None):
    """Enhanced Rich table with option-specific information."""
    if not data:
        if RICH_AVAILABLE:
            console.print(f"\n[bold magenta]{title}[/bold magenta]")
            console.print("[yellow]No options opportunities found.[/yellow]")
        else:
            print_colored(f"\n{title}", Colors.HEADER)
            print_colored("No options opportunities found.", Colors.YELLOW)
        return

    if RICH_AVAILABLE:
        table = Table(box=box.ROUNDED, show_header=True, header_style="bold blue")
        
        # Enhanced columns for option trading
        table.add_column("Stock", style="bold white", width=12, justify="left")
        table.add_column("Signal", style="bold", width=16, justify="center")
        table.add_column("Score", style="bold", width=8, justify="right")
        table.add_column("OptFlow", style="bright_green", width=8, justify="right")
        table.add_column("Greeks", style="bright_yellow", width=8, justify="right")
        table.add_column("Strike", style="bright_cyan", width=8, justify="right")
        table.add_column("Delta", style="bright_magenta", width=6, justify="right")
        table.add_column("Vol Δ%", style="green", width=8, justify="right")
        table.add_column("OI Δ%", style="yellow", width=8, justify="right")
        table.add_column("Action", style="bold", width=25, justify="center")

        for item in data:
            symbol = item['symbol']
            is_new = new_stocks and symbol in new_stocks
            
            # Get option analysis
            current_price = item.get('current_price', 0)
            option_chain = simulate_option_chain_data(symbol, current_price)
            option_analysis = analyze_option_chain(symbol, current_price, option_chain)
            recommended_strikes = option_analysis.get('recommended_strikes', [])
            best_strike = recommended_strikes[0] if recommended_strikes else None
            
            # Signal color
            if item['score'] > 50:
                signal_style = "bold bright_green"
            elif item['score'] > 25:
                signal_style = "bold green"
            elif item['score'] > 0:
                signal_style = "green"
            elif item['score'] < -50:
                signal_style = "bold bright_red"
            elif item['score'] < -25:
                signal_style = "bold red"
            else:
                signal_style = "red"

            stock_style = f"[bold bright_magenta]{symbol} ✨[/bold bright_magenta]" if is_new else symbol
            
            # Format option-specific data
            strike_display = str(best_strike['strike']) if best_strike else "N/A"
            delta_display = f"{best_strike['delta']:.2f}" if best_strike else "N/A"
            
            # Volume/OI changes
            vol_chg = item.get('volume_change_pct', 0)
            oi_chg = item.get('oi_change_pct', 0)
            
            vol_display = f"{vol_chg:+.1f}%" if abs(vol_chg) > 0.1 else "N/A"
            oi_display = f"{oi_chg:+.1f}%" if abs(oi_chg) > 0.1 else "N/A"
            
            action_text = get_enhanced_action_recommendation(item, best_strike)
            
            table.add_row(
                stock_style,
                f"[{signal_style}]{item['signal']}[/{signal_style}]",
                f"[bold]{item['score']:.2f}[/bold]",
                f"{item['sub_scores'].get('OptionFlow', 0):.2f}",
                f"{item['sub_scores'].get('Greeks', 0):.2f}",
                f"[bright_cyan]{strike_display}[/bright_cyan]",
                f"[bright_magenta]{delta_display}[/bright_magenta]",
                f"[green]{vol_display}[/green]",
                f"[yellow]{oi_display}[/yellow]",
                f"[bold]{action_text}[/bold]"
            )

        if show_time:
            console.print(f"\n[bold magenta]{title} - {show_time}[/bold magenta]")
        else:
            console.print(f"\n[bold magenta]{title}[/bold magenta]")
        
        console.print(table)
    else:
        create_enhanced_ascii_option_table(data, title, new_stocks, show_time)

def create_enhanced_ascii_option_table(data, title, new_stocks=None, show_time=None):
    """ASCII table with option-specific columns."""
    if not data:
        print_colored(f"\n{title}", Colors.HEADER)
        print_colored("No options opportunities found.", Colors.YELLOW)
        return

    if show_time:
        print_colored(f"\n{title} - {show_time}", Colors.HEADER)
    else:
        print_colored(f"\n{title}", Colors.HEADER)
    
    print_colored("="*170, Colors.BLUE)
    header = (f"{'Stock':<12} | {'Signal':<16} | {'Score':>8} | {'OptFlow':>8} | "
              f"{'Strike':<8} | {'Delta':<6} | {'Vol Δ%':<8} | {'OI Δ%':<8} | {'Action':<25}")
    print_colored(header, Colors.BOLD)
    print_colored("-"*170, Colors.BLUE)

    for item in data:
        symbol = item['symbol']
        is_new = new_stocks and symbol in new_stocks
        
        # Get option data
        current_price = item.get('current_price', 0)
        option_chain = simulate_option_chain_data(symbol, current_price)
        option_analysis = analyze_option_chain(symbol, current_price, option_chain)
        recommended_strikes = option_analysis.get('recommended_strikes', [])
        best_strike = recommended_strikes[0] if recommended_strikes else None
        
        # Format data
        strike_display = str(best_strike['strike']) if best_strike else "N/A"
        delta_display = f"{best_strike['delta']:.2f}" if best_strike else "N/A"
        
        vol_chg = item.get('volume_change_pct', 0)
        oi_chg = item.get('oi_change_pct', 0)
        vol_display = f"{vol_chg:+.1f}%" if abs(vol_chg) > 0.1 else "N/A"
        oi_display = f"{oi_chg:+.1f}%" if abs(oi_chg) > 0.1 else "N/A"
        
        action_text = get_enhanced_action_recommendation(item, best_strike)
        
        row = (f"{symbol:<12} | {item['signal']:<16} | {item['score']:>8.2f} | "
               f"{item['sub_scores'].get('OptionFlow', 0):>8.2f} | {strike_display:<8} | "
               f"{delta_display:<6} | {vol_display:<8} | {oi_display:<8} | {action_text:<25}")

        if is_new:
            print_colored(row + " ← ✨ NEW!", Colors.MAGENTA)
        else:
            print(row)
    
    print_colored("="*170, Colors.BLUE)

def create_compact_backtest_table(data, title, new_stocks=None, show_time=None):
    """Compact table for backtesting with new columns."""
    if not data:
        return

    if GREAT_TABLES_AVAILABLE:
        create_great_table_fixed(data[:5], f"Compact {title}", new_stocks, show_time)
    elif RICH_AVAILABLE:
        table = Table(box=box.SIMPLE, show_header=True, header_style="bold blue")
        
        table.add_column("#", width=3, justify="right")
        table.add_column("Stock", style="bold white", width=12)
        table.add_column("Signal", style="bold", width=14)
        table.add_column("Score", style="bold", width=8, justify="right")
        table.add_column("CurrVol", style="bright_green", width=8, justify="right")
        table.add_column("CurrOI", style="bright_magenta", width=8, justify="right")
        table.add_column("Vol Δ%", style="bright_yellow", width=7, justify="right")
        table.add_column("OI Δ%", style="bright_cyan", width=7, justify="right")
        table.add_column("Action", style="bold", width=15)

        for i, item in enumerate(data, 1):
            symbol = item['symbol']
            is_new = new_stocks and symbol in new_stocks
            
            stock_display = f"[bright_magenta]{symbol}[/bright_magenta]" if is_new else symbol
            
            vol_chg = item.get('volume_change_pct', 0)
            if isinstance(vol_chg, str) or abs(vol_chg) < 0.1:
                vol_display = "[dim]N/A[/dim]"
            elif vol_chg > 0:
                vol_display = f"[bright_green]{vol_chg:+.1f}%[/bright_green]"
            else:
                vol_display = f"[red]{vol_chg:+.1f}%[/red]"
            
            oi_chg = item.get('oi_change_pct', 0)
            if isinstance(oi_chg, str) or abs(oi_chg) < 0.1:
                oi_display = "[dim]N/A[/dim]"
            elif oi_chg > 0:
                oi_display = f"[bright_cyan]{oi_chg:+.1f}%[/bright_cyan]"
            else:
                oi_display = f"[red]{oi_chg:+.1f}%[/red]"

            table.add_row(
                str(i),
                stock_display,
                item['signal'],
                f"{item['score']:.1f}",
                str(item.get('current_volume', 'N/A')),
                str(item.get('current_oi', 'N/A')),
                vol_display,
                oi_display,
                item.get('action', 'Consider')[:15]
            )

        if show_time:
            console.print(f"[bold blue]{title} - {show_time}[/bold blue]")
        else:
            console.print(f"[bold blue]{title}[/bold blue]")
        console.print(table)
    else:
        create_enhanced_ascii_option_table(data[:5], title, new_stocks, show_time)

# ========== MAIN SCAN FUNCTION ==========
# ========== FINAL FIX FOR DATAFRAME AMBIGUITY ERROR ==========
# Replace the run_scan_at_time_5min_fixed function with this corrected version:

def run_scan_at_time_5min_fixed(scan_time, stocks, market_regime='neutral', is_live=True):
    """FINAL FIXED: Enhanced 5-minute option scanner with proper DataFrame handling."""
    global scan_count, previous_scan_results, current_scan_data
    scan_count += 1
    
    if is_live:
        time_point_aware = scan_time - timedelta(seconds=Config.SETTLE_DELAY_SECONDS)
        data = prefetch_all_timeaware(stocks, up_to_time=time_point_aware, max_workers=Config.MAX_WORKERS)
    else:
        data = prefetch_all_timeaware(stocks, up_to_time=scan_time, max_workers=Config.MAX_WORKERS)
    
    print_colored(f"Data fetch complete. Analyzing signals... Market Regime: {market_regime.upper()}...", Colors.CYAN)
    
    signals_this_scan, current_symbols = [], set()
    
    for symbol, timeframe_data in data.items():
        try:
            clean_symbol = symbol.replace('-EQ', '')
            current_symbols.add(clean_symbol)
            
            # FIXED: Filter valid timeframes with proper DataFrame checks
            filtered_timeframes = {}
            for tf, df in timeframe_data.items():
                # CRITICAL FIX: Proper DataFrame validation
                if df is not None and hasattr(df, 'empty') and not df.empty and len(df) > 0:
                    if not is_live and scan_time is not None:
                        # FIXED: Safe DataFrame slicing
                        try:
                            df_slice = df[df.index <= scan_time]
                        except Exception as slice_error:
                            logger.error(f"Error slicing DataFrame for {symbol} {tf}min: {slice_error}")
                            continue
                    else:
                        df_slice = df
                    
                    # FIXED: Check DataFrame validity after slicing
                    if (df_slice is not None and 
                        hasattr(df_slice, 'empty') and 
                        not df_slice.empty and 
                        len(df_slice) >= 50):
                        filtered_timeframes[tf] = df_slice
            
            # Skip if insufficient valid timeframes
            if len(filtered_timeframes) < 2:
                continue
            
            # Enhanced signal analysis with option-specific indicators
            signal, score, sub_scores = analyze_enhanced_signals_pro(filtered_timeframes, market_regime, clean_symbol)
            
            if abs(score) < Config.SCORE_THRESHOLD_MIN:
                continue
            
            # Enhanced flow analysis
            flow_tag = enhanced_institutional_flow_analysis(filtered_timeframes)
            
            # Main timeframe data (prefer 15-min, fallback to others)
            main_tf_data = None
            for preferred_tf in [15, 5, 30, 60]:
                if preferred_tf in filtered_timeframes:
                    main_tf_data = filtered_timeframes[preferred_tf]
                    break
            
            if main_tf_data is None:
                main_tf_data = list(filtered_timeframes.values())[0]
            
            # Extract enhanced volume/OI data
            oi_vol_data = extract_5min_volume_oi_data(
                main_tf_data, clean_symbol, 
                time_point_aware if is_live else scan_time, 
                is_live=is_live
            )
            
            # Option-specific action recommendation
            current_price = oi_vol_data.get('current_price', 0)
            
            # Simulate option chain analysis for action
            option_chain = simulate_option_chain_data(clean_symbol, current_price)
            option_analysis = analyze_option_chain(clean_symbol, current_price, option_chain)
            recommended_strikes = option_analysis.get('recommended_strikes', [])
            best_strike = recommended_strikes[0] if recommended_strikes else None
            
            action = get_enhanced_action_recommendation({'score': score, 'signal': signal}, best_strike)
            
            signals_this_scan.append({
                'symbol': clean_symbol,
                'signal': signal,
                'score': score,
                'sub_scores': sub_scores,
                'flow': flow_tag,
                'action': action,
                **oi_vol_data
            })
            
        except Exception as symbol_error:
            logger.error(f"Error processing symbol {symbol}: {symbol_error}")
            continue
    
    return signals_this_scan, current_symbols

# ========== BACKTEST FUNCTION ==========
def run_full_day_backtest_5min_fixed(backtest_date, stocks):
    """FIXED: Complete day backtesting with proper 5-minute volume/OI tracking."""
    print_colored(f"\n🔄 ENHANCED OPTION BACKTESTING: {backtest_date}", Colors.HEADER)
    print_colored("="*120, Colors.BLUE)
    
    all_results = []
    timestamps = generate_backtest_timestamps(backtest_date)
    market_regime = get_market_regime(Config.BENCHMARK_INDEX)
    total_scans = len(timestamps)
    
    print_colored(f"📊 Simulating {total_scans} scans from 09:20:15 to 15:30:00", Colors.CYAN)
    print_colored(f"📈 Market Regime: {market_regime.upper()}", Colors.YELLOW)
    
    with tqdm(total=total_scans, desc="Enhanced Option Backtest", ncols=100) as pbar:
        for i, scan_time in enumerate(timestamps):
            try:
                scan_time_str = scan_time.strftime('%H:%M')
                
                signals, current_symbols = run_scan_at_time_5min_fixed(
                    scan_time, stocks, market_regime, is_live=False
                )
                
                new_stocks = current_symbols - set(previous_scan_results.keys()) if previous_scan_results else set()
                
                # Store results
                all_results.append({
                    'timestamp': scan_time.isoformat(),
                    'signals': signals,
                    'total_signals': len(signals),
                    'bullish_signals': len([s for s in signals if s['score'] > 0]),
                    'bearish_signals': len([s for s in signals if s['score'] < 0]),
                    'new_stocks': list(new_stocks),
                    'market_regime': market_regime
                })
                
                # Display top results
                signals.sort(key=lambda x: abs(x['score']), reverse=True)
                top_bullish = [r for r in signals if r['score'] > 0][:Config.BACKTEST_TOP_DISPLAY]
                top_bearish = [r for r in signals if r['score'] < 0][:Config.BACKTEST_TOP_DISPLAY]
                
                # Count stocks with meaningful volume/OI changes
                vol_with_changes = sum(1 for s in signals if isinstance(s.get('volume_change_pct', 0), (int, float)) and abs(s.get('volume_change_pct', 0)) > 0.1)
                oi_with_changes = sum(1 for s in signals if isinstance(s.get('oi_change_pct', 0), (int, float)) and abs(s.get('oi_change_pct', 0)) > 0.1)
                
                if RICH_AVAILABLE:
                    console.print(f"[bold blue]📍 SCAN {i+1}/{total_scans} - {scan_time_str} IST[/bold blue]")
                    console.print(f"[cyan]Signals: {len(signals)} | Bullish: {len([s for s in signals if s['score'] > 0])} | Bearish: {len([s for s in signals if s['score'] < 0])} | New: {len(new_stocks)}[/cyan]")
                    console.print(f"[yellow]Volume Changes: {vol_with_changes} stocks | OI Changes: {oi_with_changes} stocks[/yellow]")
                else:
                    print_colored(f"📍 SCAN {i+1}/{total_scans} - {scan_time_str} IST", Colors.BOLD)
                    print_colored(f"Signals: {len(signals)} | Bullish: {len([s for s in signals if s['score'] > 0])} | Bearish: {len([s for s in signals if s['score'] < 0])} | New: {len(new_stocks)}", Colors.CYAN)
                    print_colored(f"Volume Changes: {vol_with_changes} stocks | OI Changes: {oi_with_changes} stocks", Colors.YELLOW)
                
                if top_bullish:
                    if GREAT_TABLES_AVAILABLE:
                        create_great_table_fixed(top_bullish, "TOP BULLISH OPTIONS", new_stocks, scan_time_str)
                    else:
                        create_compact_backtest_table(top_bullish, "TOP BULLISH OPTIONS", new_stocks, scan_time_str)
                
                if top_bearish:
                    if GREAT_TABLES_AVAILABLE:
                        create_great_table_fixed(top_bearish, "TOP BEARISH OPTIONS", new_stocks, scan_time_str)
                    else:
                        create_compact_backtest_table(top_bearish, "TOP BEARISH OPTIONS", new_stocks, scan_time_str)
                
                if new_stocks and len(new_stocks) > 0:
                    new_stocks_display = list(new_stocks)[:10]
                    more_text = f"... +{len(new_stocks)-10}" if len(new_stocks) > 10 else ""
                    print_colored(f"🆕 NEW OPTION OPPORTUNITIES: {', '.join(new_stocks_display)}{more_text}", Colors.MAGENTA)
                
                pbar.update(1)
                time.sleep(0.1)
            
            except Exception as e:
                logger.error(f"Error in backtest scan at {scan_time}: {e}")
                pbar.update(1)
                continue
    
    # Enhanced Volume/OI change statistics
    print_colored("\n📊 OPTION TRADING STATISTICS", Colors.CYAN)
    vol_changes = []
    oi_changes = []
    
    for result in all_results[1:]:  # Skip first scan
        for signal in result['signals']:
            vol_chg = signal.get('volume_change_pct', 0)
            oi_chg = signal.get('oi_change_pct', 0)
            
            if isinstance(vol_chg, (int, float)) and abs(vol_chg) > 0.1:
                vol_changes.append(vol_chg)
            if isinstance(oi_chg, (int, float)) and abs(oi_chg) > 0.1:
                oi_changes.append(oi_chg)
    
    if vol_changes:
        avg_vol_chg = sum(vol_changes) / len(vol_changes)
        max_vol_chg = max(vol_changes)
        min_vol_chg = min(vol_changes)
        print(f"📈 Volume Changes: {len(vol_changes)} stocks with meaningful changes")
        print(f"   Average: {avg_vol_chg:.1f}% | Max: {max_vol_chg:.1f}% | Min: {min_vol_chg:.1f}%")
    else:
        print(f"📈 Volume Changes: No meaningful changes detected (threshold: 0.1%)")
    
    if oi_changes:
        avg_oi_chg = sum(oi_changes) / len(oi_changes)
        max_oi_chg = max(oi_changes)
        min_oi_chg = min(oi_changes)
        print(f"📊 OI Changes: {len(oi_changes)} stocks with meaningful changes")
        print(f"   Average: {avg_oi_chg:.1f}% | Max: {max_oi_chg:.1f}% | Min: {min_oi_chg:.1f}%")
    else:
        print(f"📊 OI Changes: No meaningful changes detected (threshold: 0.1%)")
    
    # Most active times for option trading
    active_scans = sorted(all_results, key=lambda x: x['total_signals'], reverse=True)[:5]
    print_colored("\n🔥 MOST ACTIVE OPTION TRADING TIMES", Colors.CYAN)
    for i, scan in enumerate(active_scans):
        if scan['total_signals'] > 0:
            time_str = datetime.fromisoformat(scan['timestamp']).strftime('%H:%M')
            vol_active = sum(1 for s in scan['signals'] if isinstance(s.get('volume_change_pct', 0), (int, float)) and abs(s.get('volume_change_pct', 0)) > 0.1)
            print(f"{i+1}. {time_str} - {scan['total_signals']} option signals ({scan['bullish_signals']}B/{scan['bearish_signals']}S) | {vol_active} vol changes")
    
    # Save results
    output_filename = f"{backtest_date}_enhanced_option_backtest_results.json"
    try:
        with open(output_filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        print_colored(f"💾 Results saved: {output_filename}", Colors.GREEN)
    except Exception as e:
        logger.error(f"Could not save results: {e}")
    
    print_colored("="*120, Colors.BLUE)
    print_colored("✅ Enhanced Option Backtesting completed!", Colors.GREEN)

# ========== MAIN FUNCTION ==========
def main_final_fixed():
    """MAIN FUNCTION - Enhanced Option Scanner with all improvements."""
    parser = argparse.ArgumentParser(description="Enhanced Options Buyer Scanner v4.1 - Complete Production Version")
    parser.add_argument('--as-of', type=str, help="Backtest snapshot (2025-09-30T14:50)")
    parser.add_argument('--backtest', type=str, help="Full day backtest (2025-09-30)")
    args = parser.parse_args()
    
    # Display startup banner
    print_colored("="*120, Colors.HEADER)
    print_colored("🚀 ENHANCED OPTION BUYER SCANNER v4.1 - PRODUCTION READY", Colors.HEADER)
    print_colored("   🎯 Real-time Option Chain Analysis | Strike Recommendations | VIX Integration", Colors.CYAN)
    print_colored("   📊 Enhanced Flow Detection | Greeks Analysis | Market Regime Adaptation", Colors.CYAN)
    print_colored("="*120, Colors.HEADER)
    
    # Load stocks
    try:
        with open(Config.SHARES_FILE, 'r') as f:
            stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(stocks)} symbols from {Config.SHARES_FILE}")
    except Exception:
        stocks = ['RELIANCE-EQ', 'TCS-EQ', 'HDFCBANK-EQ', 'INFY-EQ', 'HINDUNILVR-EQ', 
                 'ICICIBANK-EQ', 'SBIN-EQ', 'BHARTIARTL-EQ', 'LICI-EQ', 'ITC-EQ']
        logger.warning(f"Could not load {Config.SHARES_FILE}. Using sample stocks.")
    
    if args.backtest:
        try:
            datetime.strptime(args.backtest, "%Y-%m-%d")
            run_full_day_backtest_5min_fixed(args.backtest, stocks)
        except ValueError:
            logger.error("Invalid date format for --backtest. Use YYYY-MM-DD.")
            return
    elif args.as_of:
        try:
            as_of_ts = IST.localize(datetime.fromisoformat(args.as_of))
        except ValueError:
            try:
                as_of_ts = IST.localize(datetime.strptime(args.as_of, "%Y-%m-%d"))
                as_of_ts = as_of_ts.replace(hour=15, minute=30)
            except ValueError:
                logger.error(f"Invalid timestamp format: {args.as_of}")
                return
        
        logger.info(f"Running enhanced option snapshot for {as_of_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        market_regime = get_market_regime(Config.BENCHMARK_INDEX)
        
        signals, _ = run_scan_at_time_5min_fixed(as_of_ts, stocks, market_regime, is_live=False)
        signals.sort(key=lambda x: abs(x['score']), reverse=True)
        
        top_bullish = [r for r in signals if r['score'] > 0][:25]
        top_bearish = [r for r in signals if r['score'] < 0][:25]
        
        print_colored(f"📊 ENHANCED OPTION SNAPSHOT - {as_of_ts.strftime('%Y-%m-%d %H:%M')} IST", Colors.BOLD)
        
        if GREAT_TABLES_AVAILABLE:
            create_great_table_fixed(top_bullish, "TOP 25 BULLISH OPTION OPPORTUNITIES")
            create_great_table_fixed(top_bearish, "TOP 25 BEARISH OPTION OPPORTUNITIES")
        else:
            create_rich_enhanced_option_table(top_bullish, "TOP 25 BULLISH OPTION OPPORTUNITIES")
            create_rich_enhanced_option_table(top_bearish, "TOP 25 BEARISH OPTION OPPORTUNITIES")
    else:
        # Live enhanced option scanner
        global intraday_volume_data, intraday_oi_data, scan_count, previous_scan_results
        
        now_ist = datetime.now(IST)
        first_run_time = today_ist_dt(Config.FIRST_RUN_AT)
        first_scan_time = first_run_time + timedelta(seconds=Config.FIRST_SCAN_DELAY)
        
        if now_ist < first_scan_time:
            logger.info(f"⏰ Waiting until {first_scan_time.strftime('%H:%M:%S')} IST for first option scan...")
            sleep_until(first_scan_time)
        
        print_colored("🔄 STARTING LIVE ENHANCED OPTION SCANNER...", Colors.GREEN)
        
        while True:
            scan_count += 1
            now_ist = datetime.now(IST)
            
            if now_ist.time() > datetime.strptime(Config.MARKET_END, "%H:%M").time():
                logger.info("🔔 Market closed. Shutting down Enhanced Option Scanner.")
                break
            
            print_colored(f"\n{now_ist.strftime('%H:%M:%S')} | 🚀 ENHANCED OPTION SCANNER v4.1 - Scan #{scan_count}", Colors.HEADER)
            
            market_regime = get_market_regime(Config.BENCHMARK_INDEX)
            signals, current_symbols = run_scan_at_time_5min_fixed(now_ist, stocks, market_regime, is_live=True)
            
            new_stocks = current_symbols - set(previous_scan_results.keys()) if previous_scan_results else set()
            previous_scan_results = {s: True for s in current_symbols}
            
            signals.sort(key=lambda x: abs(x['score']), reverse=True)
            top_bullish = [r for r in signals if r['score'] > 0][:25]
            top_bearish = [r for r in signals if r['score'] < 0][:25]
            
            print_colored(f"📊 LIVE OPTION SCANNER RESULTS - {now_ist.strftime('%Y-%m-%d %H:%M')} IST | Regime: {market_regime.upper()}", Colors.BOLD)
            
            if GREAT_TABLES_AVAILABLE:
                create_great_table_fixed(top_bullish, "TOP 25 BULLISH OPTION OPPORTUNITIES", new_stocks)
                create_great_table_fixed(top_bearish, "TOP 25 BEARISH OPTION OPPORTUNITIES", new_stocks)
            else:
                create_rich_enhanced_option_table(top_bullish, "TOP 25 BULLISH OPTION OPPORTUNITIES", new_stocks)
                create_rich_enhanced_option_table(top_bearish, "TOP 25 BEARISH OPTION OPPORTUNITIES", new_stocks)
            
            # Wait for next exact candle close time
            next_scan_time = get_exact_candle_close_time(now_ist)
            logger.info(f"⏰ Next option scan at {next_scan_time.strftime('%H:%M:%S')} IST")
            sleep_until(next_scan_time)

if __name__ == "__main__":
    main_final_fixed()
