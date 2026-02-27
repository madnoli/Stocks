# ==============================================================================
# MERGED OPTION BUYER SCANNER v4.0 - TrueData + Localhost API Integration
# Combines OHLC/Volume/OI from TrueData with Option Chain data from localhost
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
import requests
from pathlib import Path

# TrueData API
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

# ======== Enhanced Configuration ========
class Config:
    # TrueData Configuration
    TDUSERNAME = os.getenv("TRUEDATA_USER", "tdwsp751")
    TDPASSWORD = os.getenv("TRUEDATA_PASS", "raj@751")
    
    # Localhost API Configuration
    LOCALHOST_API_TMPL = "http://localhost:3000/api/equity/options/{symbol}"
    LOCALHOST_TIMEOUT = 20
    
    # Market Timings
    MARKET_START = "09:15"  # IST
    FIRST_RUN_AT = "09:20"  # IST; First scan after 09:15-09:20 candle
    FIRST_SCAN_DELAY = 15   # Wait 15 seconds after 09:20 for settlement
    MARKET_END = "15:30"    # IST
    SETTLE_DELAY_SECONDS = 15  # wait after bar close for data settlement
    
    # Threading and Performance
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "64"))
    TD_HIST_SESSIONS = int(os.getenv("TD_HIST_SESSIONS", "5"))
    SHARES_FILE = os.getenv("SHARES_FILE", "shares.txt")
    BENCHMARK_INDEX = "NIFTY 50"
    
    # Backtesting Configuration
    BACKTEST_INTERVAL_MINUTES = 5
    BACKTEST_START_DELAY = 5
    BACKTEST_TOP_DISPLAY = 15  # Show top 15 as requested
    
    # Localhost API Option Chain Filters
    MIN_TOTAL_OI = 2000
    MIN_TOTAL_VOL = 200
    PCR_TOL = 0.03
    EPS = 1e-6
    
    # Indicator Group Weights (Enhanced for option chain data)
    GROUP_WEIGHTS = {
        "Trend": 2.5, "Momentum": 2.0, "Volume": 2.2, "Volatility": 1.8, 
        "OI": 2.5, "OptionChain": 3.0  # New group for option chain metrics
    }
    
    # Individual Indicator Weights within Groups (Enhanced)
    INDICATOR_WEIGHTS = {
        "MA_Slope": 2.0, "ADX": 1.8, "VWAP": 1.7, "EMA": 1.7, "MACD_Trend": 1.5,
        "RSI": 2.0, "Stochastic": 1.2, "CCI": 1.2, "ROC": 1.1, "WilliamsR": 1.0,
        "VolumeSurge": 2.5, "OBV": 1.8, "CMF": 1.8, "RelVol": 1.5,
        "VolatilityExpansion": 2.5, "Bollinger": 1.3,
        "OptionBuyerMomentum": 2.8, "OIChange": 2.5, "VolumeOISync": 2.2,
        # New Option Chain Indicators
        "PCR_Signal": 3.0, "OI_Change_Pct": 2.8, "ATM_Dominance": 2.5,
        "IV_Skew": 2.2, "Volume_OI_Ratio": 2.0
    }
    
    # Scoring & Signal Thresholds
    SCORE_THRESHOLD_MIN = 10.0
    SIGNAL_THRESHOLDS = {
        'Very Strong Buy': 55.0, 'Strong Buy': 30.0, 'Buy Signal': 15.0,
        'Very Strong Sell': -55.0, 'Strong Sell': -30.0, 'Sell Signal': -15.0,
    }
    
    # Market Regime Multipliers
    REGIME_MULTIPLIERS = {
        'bullish_in_bull_market': 1.15, 'bearish_in_bear_market': 1.15,
        'bullish_in_bear_market': 0.8, 'bearish_in_bull_market': 0.8,
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
intraday_volume_data = {}  # Track 5-minute volume changes
intraday_oi_data = {}  # Track 5-minute OI changes
option_chain_cache = {}  # Cache for option chain data
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

# ========== OPTION CHAIN API FUNCTIONS ==========

def safe_div(a, b):
    """Safe division to avoid division by zero"""
    if b is None or abs(b) < Config.EPS:
        return float('inf') if a > 0 else 0.0
    return a / b

def pct_change(now, prev):
    """Calculate percentage change"""
    if now is None or prev is None:
        return None
    denom = prev if abs(prev) > Config.EPS else Config.EPS
    return ((now - prev) / denom) * 100.0

def parse_expiry(s):
    """Parse expiry date string"""
    try:
        return datetime.strptime(s, "%d-%b-%Y")
    except (ValueError, TypeError):
        return None

def choose_current_expiry(records):
    """Choose the current/nearest expiry date"""
    exps = records.get("expiryDates") or []
    exps_parsed = [(e, parse_expiry(e)) for e in exps]
    now = datetime.now()
    
    # Get future expiries first
    future = [e for e in exps_parsed if e[1] and e[1] >= now]
    if future:
        return min(future, key=lambda x: x[1])[0]
    
    # If no future expiry, get the latest past expiry
    past = [e for e in exps_parsed if e[1]]
    if past:
        return max(past, key=lambda x: x[1])[0]
    
    return None

def convert_truedata_to_localhost_symbol(td_symbol):
    """Convert TrueData symbol format to localhost format"""
    # TrueData: "RELIANCE-EQ" -> Localhost: "RELIANCE"
    if td_symbol.endswith("-EQ"):
        return td_symbol.replace("-EQ", "")
    elif td_symbol.endswith("-I"):
        return td_symbol.replace("-I", "")
    return td_symbol

def fetch_option_chain_metrics(symbol):
    """Fetch option chain metrics from localhost API"""
    try:
        # Convert symbol format for localhost API
        localhost_symbol = convert_truedata_to_localhost_symbol(symbol)
        url = Config.LOCALHOST_API_TMPL.format(symbol=localhost_symbol)
        
        r = requests.get(url, timeout=Config.LOCALHOST_TIMEOUT)
        r.raise_for_status()
        obj = r.json()
        
        recs = obj.get("records", {})
        curr_exp = choose_current_expiry(recs)
        
        if not curr_exp:
            raise ValueError("No valid expiry found")
        
        rows = [row for row in recs.get("data", []) if row.get("expiryDate") == curr_exp]
        
        if not rows:
            raise ValueError("No rows for current expiry")
        
        # Get underlying price
        underlying = None
        for row in rows:
            for val in [row.get("CE", {}).get("underlyingValue"), row.get("PE", {}).get("underlyingValue")]:
                if isinstance(val, (int, float)):
                    underlying = val
                    break
            if underlying:
                break
        
        if underlying is None:
            raise ValueError("Underlying price not found")
        
        # Calculate aggregated metrics
        ce_oi_sum, pe_oi_sum = 0, 0
        ce_vol_sum, pe_vol_sum = 0, 0
        ce_oi_wsum, pe_oi_wsum = 0.0, 0.0
        ce_oi_w, pe_oi_w = 0.0, 0.0
        ce_iv_wsum, pe_iv_wsum = 0.0, 0.0
        ce_iv_w, pe_iv_w = 0.0, 0.0
        
        for row in rows:
            ce = row.get("CE") or {}
            pe = row.get("PE") or {}
            
            ce_oi = ce.get("openInterest") or 0
            pe_oi = pe.get("openInterest") or 0
            ce_vol = ce.get("totalTradedVolume") or 0
            pe_vol = pe.get("totalTradedVolume") or 0
            
            ce_oi_sum += ce_oi
            pe_oi_sum += pe_oi
            ce_vol_sum += ce_vol
            pe_vol_sum += pe_vol
            
            # Weighted OI change calculation
            if isinstance(ce.get("pchangeinOpenInterest"), (int, float)) and ce_oi > 0:
                ce_oi_wsum += ce.get("pchangeinOpenInterest") * ce_oi
                ce_oi_w += ce_oi
            
            if isinstance(pe.get("pchangeinOpenInterest"), (int, float)) and pe_oi > 0:
                pe_oi_wsum += pe.get("pchangeinOpenInterest") * pe_oi
                pe_oi_w += pe_oi
            
            # IV calculation
            ce_iv = ce.get("impliedVolatility") or 0
            pe_iv = pe.get("impliedVolatility") or 0
            
            if ce_iv > 0 and ce_oi > 0:
                ce_iv_wsum += ce_iv * ce_oi
                ce_iv_w += ce_oi
            
            if pe_iv > 0 and pe_oi > 0:
                pe_iv_wsum += pe_iv * pe_oi
                pe_iv_w += pe_oi
        
        # Calculate final metrics
        total_oi = ce_oi_sum + pe_oi_sum
        total_vol = ce_vol_sum + pe_vol_sum
        pcr = safe_div(pe_oi_sum, ce_oi_sum)
        
        ce_oi_chg_pct = safe_div(ce_oi_wsum, ce_oi_w)
        pe_oi_chg_pct = safe_div(pe_oi_wsum, pe_oi_w)
        blended_oi_chg = safe_div((ce_oi_chg_pct * ce_oi_sum) + (pe_oi_chg_pct * pe_oi_sum), total_oi)
        
        avg_ce_iv = safe_div(ce_iv_wsum, ce_iv_w)
        avg_pe_iv = safe_div(pe_iv_wsum, pe_iv_w)
        avg_iv = safe_div((avg_ce_iv * ce_oi_sum) + (avg_pe_iv * pe_oi_sum), total_oi) * 100
        
        vol_oi_ratio = safe_div(total_vol, total_oi)
        
        # ATM analysis
        atm_strike_row = min(rows, key=lambda r: abs(r.get("strikePrice", float('inf')) - underlying))
        atm_ce = atm_strike_row.get("CE", {})
        atm_pe = atm_strike_row.get("PE", {})
        atm_pcr = safe_div(atm_pe.get("openInterest", 0), atm_ce.get("openInterest", 0))
        
        atm_ce_vol = atm_ce.get("totalTradedVolume", 0)
        atm_pe_vol = atm_pe.get("totalTradedVolume", 0)
        atm_vol_dom = "CALLS" if atm_ce_vol > atm_pe_vol else ("PUTS" if atm_pe_vol > atm_ce_vol else "NEUTRAL")
        atm_signal = f"PCR:{atm_pcr:.2f}|VOL:{atm_vol_dom}"
        
        # Classification logic
        def classify_sentiment(pcr, ce_oi, pe_oi, ce_vol, pe_vol):
            is_low_liq = (ce_oi + pe_oi < Config.MIN_TOTAL_OI) or (ce_vol + pe_vol < Config.MIN_TOTAL_VOL)
            
            if abs(pcr - 1.0) <= Config.PCR_TOL:
                return "Neutral"
            
            ce_oi_dom = ce_oi > pe_oi
            pe_oi_dom = pe_oi > ce_oi
            ce_vol_dom = ce_vol >= pe_vol
            pe_vol_dom = pe_vol >= ce_vol
            
            if pcr < 0.8 and ce_oi_dom and ce_vol_dom:
                return "Strong Bullish" if not is_low_liq else "Mild Bullish"
            if pcr > 1.2 and pe_oi_dom and pe_vol_dom:
                return "Strong Bearish" if not is_low_liq else "Mild Bearish"
            if pcr < 1.0 and (ce_oi_dom or ce_vol_dom):
                return "Mild Bullish"
            if pcr > 1.0 and (pe_oi_dom or pe_vol_dom):
                return "Mild Bearish"
            
            return "Neutral"
        
        sentiment = classify_sentiment(pcr, ce_oi_sum, pe_oi_sum, ce_vol_sum, pe_vol_sum)
        
        return {
            "Stock": localhost_symbol,
            "Price": underlying,
            "CE_Volume": ce_vol_sum,
            "PE_Volume": pe_vol_sum,
            "CE_OI": ce_oi_sum,
            "PE_OI": pe_oi_sum,
            "Total_Volume": total_vol,
            "Total_OI": total_oi,
            "OI_Change_Pct": blended_oi_chg,
            "PCR": pcr,
            "Avg_IV": avg_iv,
            "Vol_OI_Ratio": vol_oi_ratio,
            "ATM_Signal": atm_signal,
            "ATM_PCR": atm_pcr,
            "Sentiment": sentiment,
            "Expiry": curr_exp,
            "CE_OI_Change": ce_oi_chg_pct,
            "PE_OI_Change": pe_oi_chg_pct
        }
        
    except Exception as e:
        logger.error(f"Error fetching option chain for {symbol}: {e}")
        return {
            "Stock": convert_truedata_to_localhost_symbol(symbol),
            "Error": str(e),
            "PCR": None,
            "OI_Change_Pct": None,
            "Sentiment": "Error"
        }

def fetch_multiple_option_chains(symbols, max_workers=10):
    """Fetch option chain data for multiple symbols in parallel"""
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(fetch_option_chain_metrics, symbol): symbol 
            for symbol in symbols
        }
        
        # Collect results with progress bar
        with tqdm(total=len(symbols), desc="Fetching Option Chains", ncols=100, leave=False) as pbar:
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results[symbol] = result
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Failed to fetch option chain for {symbol}: {e}")
                    results[symbol] = {"Stock": symbol, "Error": str(e), "Sentiment": "Error"}
                    pbar.update(1)
    
    return results

# ========== UTILITY FUNCTIONS ==========

def next_5min_boundary(now_ist):
    """Get next 5-minute boundary"""
    minute = now_ist.minute - (now_ist.minute % 5)
    boundary = now_ist.replace(minute=minute, second=0, microsecond=0)
    if boundary <= now_ist:
        boundary = boundary + timedelta(minutes=5)
    return boundary

def get_exact_candle_close_time(now_ist):
    """Get exact candle close time"""
    next_boundary = next_5min_boundary(now_ist)
    return next_boundary + timedelta(seconds=Config.SETTLE_DELAY_SECONDS)

def parse_hhmm(s):
    """Parse HH:MM time string"""
    h, m = map(int, s.split(":"))
    return h, m

def today_ist_dt(hhmm):
    """Convert HH:MM to today's IST datetime"""
    now = datetime.now(IST)
    h, m = parse_hhmm(hhmm)
    return now.replace(hour=h, minute=m, second=0, microsecond=0)

def sleep_until(ts):
    """Sleep until specific timestamp"""
    while True:
        now = datetime.now(IST)
        delta = (ts - now).total_seconds()
        if delta <= 0:
            break
        time.sleep(min(0.5, delta))

# ========== TECHNICAL INDICATORS ==========

def ema(series, length):
    """Calculate Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()

def vwap(df, period=None):
    """Calculate Volume Weighted Average Price"""
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
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calculate_rsi(df, period=14):
    """Calculate RSI"""
    if len(df) < period + 1:
        return pd.Series(dtype='float64', index=df.index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rs.fillna(100, inplace=True)
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    if len(df) < slow + signal:
        return pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index)
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_adx(df, period=14):
    """Calculate ADX"""
    if len(df) < period * 2:
        return pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index)
    
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

def volume_surge(df, lookback=20):
    """Calculate volume surge Z-score"""
    vol_ma = df["Volume"].rolling(lookback).mean()
    vol_std = df["Volume"].rolling(lookback).std()
    z_score = (df["Volume"] - vol_ma) / vol_std.replace(0, np.nan)
    return z_score.fillna(0)

def calculate_obv(df):
    """Calculate On Balance Volume"""
    if len(df) < 2:
        return pd.Series(dtype='float64', index=df.index)
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

def cmf(df, period=20):
    """Calculate Chaikin Money Flow"""
    mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"]).replace(0, np.nan)
    mfm = mfm.fillna(0)
    mfv = mfm * df["Volume"]
    mfv_sum = mfv.rolling(period).sum()
    vol_sum = df["Volume"].rolling(period).sum().replace(0, np.nan)
    return (mfv_sum / vol_sum).fillna(0)

def slope(series, lookback=10):
    """Calculate slope of a series"""
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
        # If OI is unavailable, do not emit this indicator to avoid bias
        return None
    
    combined_score = (price_mom * 0.4) + (vol_surge_val * 0.3) + (oi_buildup * 0.3)
    return min(max(combined_score, -100), 100)
# ========== PANDAS VERSION COMPATIBILITY FIX ==========

def is_datetime64_tz_aware(series_or_index):
    """Check if datetime series/index is timezone aware - compatible with all pandas versions"""
    try:
        # Try the modern pandas method first
        return pd.api.types.is_datetime64_tz_dtype(series_or_index)
    except AttributeError:
        # Fallback for older/newer pandas versions
        try:
            return hasattr(series_or_index, 'dt') and series_or_index.dt.tz is not None
        except:
            # Final fallback - check dtype string
            return 'datetime64[ns,' in str(series_or_index.dtype)

# ========== TRUEDATA API FUNCTIONS ==========

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

td_hist_pool, sess_limiters = build_sessions()
logger.info("TrueData login successful.")

def normalize_hist_df(df, symbol):
    """Normalize historical dataframe - PANDAS VERSION COMPATIBLE"""
    if df is None or len(df) == 0:
        return None
    
    try:
        out = df.copy()
        out.rename(columns={str(c): str(c).lower() for c in out.columns}, inplace=True)
        
        rename_map = {}
        for src, tgt in [("timestamp", "Date"), ("time", "Date"), ("datetime", "Date"), ("date", "Date"),
                         ("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"),
                         ("volume", "Volume"), ("vol", "Volume"),
                         ("oi", "OpenInterest"), ("openinterest", "OpenInterest"), ("open_interest", "OpenInterest")]:
            if src in out.columns:
                rename_map[src] = tgt
        
        out.rename(columns=rename_map, inplace=True)
        
        if "Date" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
            out["Date"] = out.index
        elif "Date" not in out.columns:
            return None
        
        if "Volume" not in out.columns:
            out["Volume"] = 0
        
        if "OpenInterest" in out.columns:
            out["OpenInterest"] = pd.to_numeric(out["OpenInterest"], errors='coerce')
        
        out["Date"] = pd.to_datetime(out["Date"], errors='coerce')
        out = out.dropna(subset=["Date"])
        
        # FIXED: Compatible timezone checking
        if is_datetime64_tz_aware(out["Date"]):
            out["Date"] = out["Date"].dt.tz_convert(IST)
        else:
            out["Date"] = out["Date"].dt.tz_localize(IST)
        
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            out[c] = pd.to_numeric(out.get(c, np.nan), errors='coerce')
        
        out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_values("Date").set_index("Date")
        out = out[~out.index.duplicated(keep='last')]
        
        if len(out) == 0:
            return None
        
        if not isinstance(out.index, pd.DatetimeIndex) or out.index.hasnans:
            logger.warning(f"Invalid or incomplete datetime index for {symbol}, skipping")
            return None
        
        return out
    except Exception as e:
        logger.error(f"Normalize error {symbol}: {e}")
        return None

def pick_session(symbol_orig, timeframe_minutes):
    """Pick session for symbol based on hash"""
    return hash((symbol_orig, timeframe_minutes)) % len(td_hist_pool)

def fetch_one_real(symbol_orig, timeframe_minutes, limiter, hist, up_to_time):
    """EXACT FROM WORKING CODE - Fetch single timeframe data"""
    td_symbol = symbol_orig.replace('-EQ', '')
    bar_size = BAR_SIZE_MAP.get(timeframe_minutes)
    duration_str = DURATION_MAP.get(timeframe_minutes)
    
    if not bar_size or not duration_str:
        return symbol_orig, timeframe_minutes, None
    
    try:
        limiter.acquire()
        
        if up_to_time:
            # Parse duration to delta
            dur_num, dur_unit = duration_str.split()
            dur_num = int(dur_num)
            if dur_unit == 'D':
                delta = timedelta(days=dur_num)
            else:
                # Add other units if needed, but all are D
                delta = timedelta(days=dur_num)
            start_time = up_to_time - delta
            df_raw = hist.get_historic_data(td_symbol, start_time=start_time, end_time=up_to_time, bar_size=bar_size)
        else:
            df_raw = hist.get_historic_data(td_symbol, duration=duration_str, bar_size=bar_size)
        
        df = normalize_hist_df(df_raw, td_symbol)
        return symbol_orig, timeframe_minutes, df
    
    except Exception as e:
        logger.error(f"Error fetching {symbol_orig} {timeframe_minutes}min: {e}")
        return symbol_orig, timeframe_minutes, None

def prefetch_all_real(stocks, up_to_time=None, max_workers=Config.MAX_WORKERS):
    """EXACT FROM WORKING CODE - Prefetch all data"""
    tfs = [5, 15, 30, 60, 1440]
    total_calls, stock_multi_data = len(stocks) * len(tfs), defaultdict(dict)
    
    global api_calls_done
    with api_calls_lock:
        api_calls_done = 0
    
    desc = f"Fetching data up to {up_to_time.strftime('%H:%M')}" if up_to_time else "Prefetching Data"
    with tqdm(total=total_calls, desc=desc, ncols=100, leave=False) as api_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for s in stocks:
                for tf in tfs:
                    session_idx = pick_session(s, tf)
                    futures.append(executor.submit(
                        fetch_one_real, s, tf, sess_limiters[session_idx], 
                        td_hist_pool[session_idx], up_to_time
                    ))
            
            for fut in as_completed(futures):
                symbol_orig, tf, df = fut.result()
                if df is not None and len(df) > 0:
                    stock_multi_data[symbol_orig][tf] = df
                api_bar.update(1)
    
    return {s: d for s, d in stock_multi_data.items() if len(d) >= 2}

def get_market_regime(index_symbol="NIFTY 50"):
    """Get current market regime - FIXED METHOD CALL"""
    try:
        si = pick_session(index_symbol, 1440)
        # FIXED: Use the correct method name
        df_raw = td_hist_pool[si].get_historic_data(index_symbol, duration="200 D", bar_size="1 day")
        df = normalize_hist_df(df_raw, index_symbol)
        
        if df is None or len(df) < 50:
            return "neutral"
        
        ema20_series = ema(df['Close'], 20)
        ema50_series = ema(df['Close'], 50)
        
        if ema20_series.empty or ema50_series.empty:
            return "neutral"
        
        ema20_val = ema20_series.iloc[-1]
        ema50_val = ema50_series.iloc[-1]
        close = df['Close'].iloc[-1]
        
        if close > ema20_val and ema20_val > ema50_val:
            return "bullish"
        elif close < ema20_val and ema20_val < ema50_val:
            return "bearish"
        else:
            return "neutral"
    
    except Exception as e:
        logger.warning(f"Could not fetch market regime for {index_symbol}: {e}")
        return "neutral"

# ========== ENHANCED SCORING ENGINE WITH OPTION CHAIN INTEGRATION ==========

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

def calculate_option_chain_scores(option_chain_data):
    """Calculate scores from option chain data"""
    scores = defaultdict(float)
    
    if not option_chain_data or "Error" in option_chain_data:
        return scores
    
    try:
        # PCR Signal Score (Key indicator for option buyers)
        pcr = option_chain_data.get("PCR", 1.0)
        if pcr is not None and pcr != float('inf'):
            # For option buyers: Low PCR = Bullish, High PCR = Bearish
            scores['PCR_Signal'] = normalize_score(pcr, (0.5, 0.8), (1.2, 1.8), (-3.0, 3.0))
        
        # OI Change Percentage Score
        oi_change_pct = option_chain_data.get("OI_Change_Pct", 0)
        if oi_change_pct is not None and oi_change_pct != float('inf'):
            scores['OI_Change_Pct'] = normalize_score(oi_change_pct, (5, 20), (-5, -20), (-2.8, 2.8))
        
        # ATM Dominance Score
        atm_pcr = option_chain_data.get("ATM_PCR", 1.0)
        if atm_pcr is not None and atm_pcr != float('inf'):
            scores['ATM_Dominance'] = normalize_score(atm_pcr, (0.6, 0.9), (1.1, 1.5), (-2.5, 2.5))
        
        # Volume to OI Ratio Score
        vol_oi_ratio = option_chain_data.get("Vol_OI_Ratio", 0)
        if vol_oi_ratio is not None and vol_oi_ratio != float('inf'):
            # Higher ratio indicates fresh positions (good for option buyers)
            scores['Volume_OI_Ratio'] = normalize_score(vol_oi_ratio, (0.3, 0.8), (0.05, 0.1), (-2.0, 2.0))
        
        # IV Skew Score (difference between CE and PE IV)
        ce_oi_change = option_chain_data.get("CE_OI_Change", 0)
        pe_oi_change = option_chain_data.get("PE_OI_Change", 0)
        
        if ce_oi_change is not None and pe_oi_change is not None:
            if ce_oi_change != float('inf') and pe_oi_change != float('inf'):
                iv_skew = ce_oi_change - pe_oi_change  # Positive = Call buying, Negative = Put buying
                scores['IV_Skew'] = normalize_score(iv_skew, (5, 15), (-5, -15), (-2.2, 2.2))
        
    except Exception as e:
        logger.error(f"Error calculating option chain scores: {e}")
    
    return scores

def calculate_indicator_scores(df):
    """Calculate technical indicator scores"""
    scores = defaultdict(float)
    
    if df is None or len(df) < 50:
        return scores
    
    try:
        # --- Trend Group ---
        adx, pdi, ndi = calculate_adx(df)
        if not adx.empty and len(adx) > 3 and adx.iloc[-1] > 20 and adx.iloc[-1] > adx.iloc[-3]:
            scores['ADX'] = 2.0 if pdi.iloc[-1] > ndi.iloc[-1] else -2.0
        
        ema20, ema50 = ema(df['Close'], 20), ema(df['Close'], 50)
        if not ema20.empty and not ema50.empty:
            ema_ratio = ema20.iloc[-1] / ema50.iloc[-1] if ema50.iloc[-1] != 0 else 1
            scores['EMA'] = normalize_score(ema_ratio, (1.001, 1.02), (0.999, 0.98))
        
        vwap_line = vwap(df, period=None)
        if not vwap_line.empty:
            vwap_ratio = df['Close'].iloc[-1] / vwap_line.iloc[-1] if vwap_line.iloc[-1] != 0 else 1
            scores['VWAP'] = normalize_score(vwap_ratio, (1.002, 1.025), (0.998, 0.975))
        
        macd, signal = calculate_macd(df)
        if not macd.empty and not signal.empty and len(macd) > 0:
            if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-1] > 0:
                scores['MACD_Trend'] = 2.0
            elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-1] < 0:
                scores['MACD_Trend'] = -2.0
        
        if not ema20.empty and len(ema20) >= 5:
            ma20_slope = slope(ema20, 5)
            price_norm_slope = ma20_slope / df['Close'].iloc[-1] * 1000 if df['Close'].iloc[-1] != 0 else 0
            scores['MA_Slope'] = normalize_score(price_norm_slope, (0.1, 0.5), (-0.1, -0.5), (-2.5, 2.5))
        
        # --- Momentum Group ---
        rsi = calculate_rsi(df)
        if not rsi.empty and len(rsi) > 0:
            scores['RSI'] = normalize_score(rsi.iloc[-1], (60, 85), (40, 15))
        
        # --- Volume Group ---
        zscore = volume_surge(df, lookback=20)
        if not zscore.empty and len(zscore) > 1:
            price_up = df['Close'].iloc[-1] > df['Close'].iloc[-2]
            if price_up:
                scores['VolumeSurge'] = normalize_score(zscore.iloc[-1], (1.5, 3.0), (0, 0))
            else:
                scores['VolumeSurge'] = normalize_score(zscore.iloc[-1], (0, 0), (-1.5, -3.0))
        
        obv_line = calculate_obv(df)
        if len(obv_line) > 5:
            obv_slope = slope(obv_line, 5)
            scores['OBV'] = normalize_score(obv_slope, (1, 1e9), (-1, -1e9))
        
        cmf20 = cmf(df, period=20)
        if not cmf20.empty and len(cmf20) > 0:
            scores['CMF'] = normalize_score(cmf20.iloc[-1], (0.1, 0.25), (-0.1, -0.25))
        
        # --- OI Group (if available) ---
        if 'OpenInterest' in df.columns and df['OpenInterest'].notna().sum() >= 2:
            current_oi = df['OpenInterest'].iloc[-1]
            prev_oi = df['OpenInterest'].iloc[-2] if len(df) > 1 else current_oi
            
            if prev_oi > 0:
                oi_change = (current_oi - prev_oi) / prev_oi * 100
                scores['OIChange'] = normalize_score(oi_change, (5, 20), (-5, -20))
        
        # Enhanced OI-based indicators
        oi_buildup = detect_oi_buildup(df, lookback=20)
        if oi_buildup is not None:
            scores['OIChange'] = normalize_score(oi_buildup, (10, 50), (-10, -50))
        
        vol_oi_sync = volume_oi_sync_analysis(df)
        if vol_oi_sync is not None:
            scores['VolumeOISync'] = normalize_score(vol_oi_sync, (20, 80), (-20, -80))
        
        opt_buyer_mom = option_buyer_momentum(df)
        if opt_buyer_mom is not None:
            scores['OptionBuyerMomentum'] = normalize_score(opt_buyer_mom, (30, 80), (-30, -80))
        
    except Exception as e:
        logger.error(f"Error calculating indicator scores: {e}")
    
    return scores

def analyze_signals_enhanced(timeframe_data, option_chain_data=None, market_regime='neutral'):
    """Enhanced signal analysis with option chain integration"""
    total_score, total_weight = 0.0, 0.0
    group_scores = defaultdict(float)
    group_weights = defaultdict(float)
    
    # Process timeframe data (existing logic)
    for tf_min, df in timeframe_data.items():
        if df is None or len(df) < 50:
            continue
        
        indicator_scores = calculate_indicator_scores(df)
        tf_weight = TIMEFRAME_WEIGHTS.get(tf_min, 1.0)
        
        for group, weight in Config.GROUP_WEIGHTS.items():
            if group == "OptionChain":  # Skip option chain group for now
                continue
                
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
    
    # Process option chain data (new enhancement)
    if option_chain_data and "Error" not in option_chain_data:
        option_scores = calculate_option_chain_scores(option_chain_data)
        option_weight = Config.GROUP_WEIGHTS.get("OptionChain", 3.0)
        
        oc_grp_score, oc_grp_weight = 0.0, 0.0
        
        for indicator, ind_weight in Config.INDICATOR_WEIGHTS.items():
            if indicator in option_scores:
                belongs_to_option_chain = any(term in indicator for term in ['PCR', 'OI_Change', 'ATM', 'IV', 'Volume_OI'])
                
                if belongs_to_option_chain:
                    oc_grp_score += option_scores[indicator] * ind_weight
                    oc_grp_weight += abs(option_scores[indicator]) * ind_weight
        
        if oc_grp_weight > 0:
            norm_oc_score = (oc_grp_score / oc_grp_weight) * option_weight
            group_scores["OptionChain"] += norm_oc_score
            group_weights["OptionChain"] += option_weight
    
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
    
    # Determine signal strength
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
    
    # Calculate final sub-scores for display
    final_sub_scores = {}
    for group in group_scores:
        if group_weights[group] > 0:
            final_sub_scores[group] = group_scores[group] / group_weights[group] * 10
    
    return signal, normalized_score, final_sub_scores

# ========== MAIN SCANNER LOGIC WITH PARALLEL API CALLS ==========

def run_merged_scan_at_time(timepoint_aware, stocks, market_regime, is_live=False):
    """Run merged scan combining TrueData and Localhost API data"""
    
    # Step 1: Fetch TrueData OHLC/Volume/OI data in parallel
    print_colored(f"📊 Fetching TrueData OHLC/Volume/OI data...", Colors.CYAN)
    if is_live:
        stock_multi_data = prefetch_all_real(stocks, None, max_workers=Config.MAX_WORKERS)
    else:
        stock_multi_data = prefetch_all_real(stocks, timepoint_aware, max_workers=Config.MAX_WORKERS)
    
    # Step 2: Fetch Option Chain data from localhost API in parallel
    print_colored(f"🔗 Fetching Option Chain data from localhost API...", Colors.CYAN)
    option_chain_data = fetch_multiple_option_chains(stocks, max_workers=20)
    
    print_colored(f"✅ Data fetch complete. Analyzing signals... (Market Regime: {market_regime.upper()})", Colors.GREEN)
    
    signals_this_scan = []
    current_symbols = set()
    
    for symbol, timeframe_data in stock_multi_data.items():
        clean_symbol = symbol.replace("-EQ", "").replace("-I", "")
        current_symbols.add(clean_symbol)
        
        # Filter timeframes with sufficient data
        filtered_timeframes = {}
        for tf, df in timeframe_data.items():
            if df is not None and not df.empty:
                if is_live:
                    df_slice = df
                else:
                    df_slice = df[df.index <= timepoint_aware] if timepoint_aware else df
                
                if not df_slice.empty and len(df_slice) >= 50:
                    filtered_timeframes[tf] = df_slice
        
        if len(filtered_timeframes) < 2:
            continue
        
        # Get corresponding option chain data
        symbol_option_data = option_chain_data.get(symbol, {})
        
        # Analyze signals with both TrueData and Option Chain data
        signal, score, sub_scores = analyze_signals_enhanced(
            filtered_timeframes, 
            symbol_option_data, 
            market_regime
        )
        
        if abs(score) >= Config.SCORE_THRESHOLD_MIN:
            # Enhanced volume/OI data extraction
            tf_5min = filtered_timeframes.get(5)
            if tf_5min is not None:
                volume_oi_data = extract_5min_volume_oi_data(tf_5min, clean_symbol, timepoint_aware, is_live=is_live)
            else:
                main_tf_data = list(filtered_timeframes.values())[0]
                volume_oi_data = extract_5min_volume_oi_data(main_tf_data, clean_symbol, timepoint_aware, is_live=is_live)
            
            # Determine action based on signal and option chain sentiment
            action = "Consider Call" if score > 0 else "Consider Put"
            if "Strong" in signal:
                if symbol_option_data.get("Sentiment") in ["Strong Bullish", "Mild Bullish"] and score > 0:
                    action = "Strong Call Buy"
                elif symbol_option_data.get("Sentiment") in ["Strong Bearish", "Mild Bearish"] and score < 0:
                    action = "Strong Put Buy"
            
            # Enhanced result with option chain data
            result = {
                'symbol': clean_symbol,
                'signal': signal,
                'score': score,
                'sub_scores': sub_scores,
                'flow': determine_institutional_flow(filtered_timeframes),
                'action': action,
                **volume_oi_data,
                # Add option chain specific data
                'pcr': symbol_option_data.get('PCR', 'N/A'),
                'option_oi_change': symbol_option_data.get('OI_Change_Pct', 'N/A'),
                'option_sentiment': symbol_option_data.get('Sentiment', 'Unknown'),
                'atm_signal': symbol_option_data.get('ATM_Signal', 'N/A'),
                'option_iv': symbol_option_data.get('Avg_IV', 'N/A'),
                'option_vol_oi_ratio': symbol_option_data.get('Vol_OI_Ratio', 'N/A')
            }
            
            signals_this_scan.append(result)
    
    return signals_this_scan, current_symbols

def extract_5min_volume_oi_data(df, symbol, timepoint=None, is_live=False):
    """Extract 5-minute volume and OI data with change calculations"""
    try:
        global intraday_volume_data, intraday_oi_data
        
        df_slice = df[df.index <= timepoint] if timepoint and not is_live else df
        
        if df_slice.empty:
            return {
                'current_volume': 'N/A', 'current_oi': 'N/A', 
                'volume_change_pct': 0, 'oi_change_pct': 0,
                'volume': 'N/A', 'oi': 'N/A', 
                'volume_change': 'N/A', 'oi_change': 'N/A'
            }
        
        # Get current values
        current_volume = int(df_slice['Volume'].iloc[-1]) if 'Volume' in df_slice.columns else 0
        current_oi = int(df_slice['OpenInterest'].iloc[-1]) if 'OpenInterest' in df_slice.columns and df_slice['OpenInterest'].notna().sum() > 0 else None
        
        # Calculate percentage changes
        vol_change_pct = 0
        oi_change_pct = 0
        
        if len(df_slice) >= 2:
            previous_volume = int(df_slice['Volume'].iloc[-2])
            if previous_volume > 0:
                vol_change_pct = (current_volume - previous_volume) / previous_volume * 100
            
            if current_oi is not None and 'OpenInterest' in df_slice.columns:
                previous_oi = int(df_slice['OpenInterest'].iloc[-2])
                if previous_oi > 0:
                    oi_change_pct = (current_oi - previous_oi) / previous_oi * 100
        
        # Use cached data for better change tracking
        if abs(vol_change_pct) < 0.1 and abs(oi_change_pct) < 0.1:
            prev_volume = intraday_volume_data.get(symbol, None)
            prev_oi = intraday_oi_data.get(symbol, None)
            
            if prev_volume is not None and prev_volume > 0 and current_volume > 0:
                vol_change_pct = (current_volume - prev_volume) / prev_volume * 100
            
            if prev_oi is not None and prev_oi > 0 and current_oi and current_oi > 0:
                oi_change_pct = (current_oi - prev_oi) / prev_oi * 100
        
        # Update cache
        intraday_volume_data[symbol] = current_volume if isinstance(current_volume, int) else 0
        intraday_oi_data[symbol] = current_oi if isinstance(current_oi, int) else 0
        
        # Format display values
        current_volume_display = f"{current_volume:,}" if isinstance(current_volume, int) and current_volume > 999 else str(current_volume)
        current_oi_display = f"{current_oi:,}" if isinstance(current_oi, int) and current_oi and current_oi > 999 else str(current_oi) if current_oi is not None else "N/A"
        
        volume_change_legacy = f"{vol_change_pct:.1f}" if isinstance(vol_change_pct, (int, float)) and abs(vol_change_pct) > 0.1 else "N/A"
        oi_change_legacy = f"{oi_change_pct:.1f}" if isinstance(oi_change_pct, (int, float)) and abs(oi_change_pct) > 0.1 else "N/A"
        
        return {
            'current_volume': current_volume_display,
            'current_oi': current_oi_display,
            'volume_change_pct': vol_change_pct if isinstance(vol_change_pct, (int, float)) and abs(vol_change_pct) > 0.1 else 0,
            'oi_change_pct': oi_change_pct if isinstance(oi_change_pct, (int, float)) and abs(oi_change_pct) > 0.1 else 0,
            'volume': current_volume_display,
            'oi': current_oi_display,
            'volume_change': volume_change_legacy,
            'oi_change': oi_change_legacy,
            'raw_volume': current_volume if isinstance(current_volume, int) else 0,
            'raw_oi': current_oi if isinstance(current_oi, int) else 0
        }
        
    except Exception as e:
        logger.error(f"Error extracting 5-min data for {symbol}: {e}")
        return {
            'current_volume': 'N/A', 'current_oi': 'N/A', 
            'volume_change_pct': 0, 'oi_change_pct': 0,
            'volume': 'N/A', 'oi': 'N/A', 
            'volume_change': 'N/A', 'oi_change': 'N/A'
        }

def determine_institutional_flow(tf_data):
    """Determine institutional flow from timeframe data"""
    frames = [tf_data.get(t) for t in [5, 15, 30] if tf_data.get(t) is not None and len(tf_data.get(t)) >= 60]
    
    if not frames:
        return "Unknown"
    
    votes = 0
    
    for df in frames:
        cmf_series = cmf(df, 20)
        if cmf_series.empty:
            continue
        
        cmf_last = cmf_series.iloc[-1]
        
        if cmf_last > 0.05:
            votes += 1
        elif cmf_last < -0.05:
            votes -= 1
    
    if votes >= 2:
        return "Institutional Accumulation"
    elif votes <= -2:
        return "Institutional Distribution"
    else:
        return "Mixed/Neutral"

# ========== ENHANCED TABLE DISPLAY FUNCTIONS ==========

def create_enhanced_results_table(data, title, new_stocks=None, show_time=None):
    """Create enhanced results table with option chain data - FIXED SENTIMENT DISPLAY"""
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
        table.add_column("Stock", style="bold white", width=10, justify="left")
        table.add_column("Signal", style="bold", width=14, justify="center")
        table.add_column("Score", style="bold", width=6, justify="right")
        table.add_column("PCR", style="cyan", width=5, justify="right")
        table.add_column("OI_Chg%", style="yellow", width=7, justify="right")  # Changed header
        table.add_column("Opt_Bias", style="green", width=10, justify="left")  # Clearer header
        table.add_column("Vol", style="bright_green", width=8, justify="right")
        table.add_column("OI", style="bright_magenta", width=8, justify="right")
        table.add_column("Vol Δ%", style="bright_yellow", width=6, justify="right")
        table.add_column("OI Δ%", style="bright_cyan", width=6, justify="right")
        table.add_column("Action", style="bold", width=12, justify="center")
        
        for item in data:
            symbol = item['symbol']
            is_new = new_stocks and symbol in new_stocks
            
            # Signal style based on score
            if item['score'] > 50: signal_style = "bold bright_green"
            elif item['score'] > 25: signal_style = "bold green"
            elif item['score'] > 0: signal_style = "green"
            elif item['score'] < -50: signal_style = "bold bright_red"
            elif item['score'] < -25: signal_style = "bold red"
            else: signal_style = "red"
            
            stock_style = f"[bold bright_magenta]{symbol} ✨[/bold bright_magenta]" if is_new else symbol
            
            # Format PCR
            pcr_val = item.get('pcr', 'N/A')
            pcr_display = f"{pcr_val:.2f}" if isinstance(pcr_val, (int, float)) else "N/A"
            
            # Format Option OI Change
            opt_oi_chg = item.get('option_oi_change', 'N/A')
            opt_oi_display = f"{opt_oi_chg:.1f}%" if isinstance(opt_oi_chg, (int, float)) else "N/A"
            
            # FIXED: Get Technical Signal Direction instead of Option Chain sentiment
            technical_signal = item['signal']
            if 'Buy' in technical_signal:
                opt_bias = "📈 Tech Bull"  # Technical signal is bullish
                opt_bias_style = "[green]📈 Tech Bull[/green]"
            elif 'Sell' in technical_signal:
                opt_bias = "📉 Tech Bear"  # Technical signal is bearish  
                opt_bias_style = "[red]📉 Tech Bear[/red]"
            else:
                opt_bias = "➡️  Neutral"
                opt_bias_style = "[yellow]➡️  Neutral[/yellow]"
            
            # Add option chain bias as additional info
            option_sentiment = item.get('option_sentiment', 'Unknown')
            if option_sentiment in ["Strong Bullish", "Mild Bullish"]:
                opt_bias_style += "[dim] |📞Call[/dim]"
            elif option_sentiment in ["Strong Bearish", "Mild Bearish"]:
                opt_bias_style += "[dim] |📞Put[/dim]"
            else:
                opt_bias_style += "[dim] |❓[/dim]"
            
            # Format volume and OI changes
            vol_change_raw = item.get('volume_change_pct', 0)
            vol_change_style = f"[green]{vol_change_raw:+.1f}%[/green]" if vol_change_raw > 0.1 else f"[red]{vol_change_raw:+.1f}%[/red]" if vol_change_raw < -0.1 else "[dim]N/A[/dim]"
            
            oi_change_raw = item.get('oi_change_pct', 0)
            oi_change_style = f"[cyan]{oi_change_raw:+.1f}%[/cyan]" if oi_change_raw > 0.1 else f"[red]{oi_change_raw:+.1f}%[/red]" if oi_change_raw < -0.1 else "[dim]N/A[/dim]"
            
            table.add_row(
                stock_style,
                f"[{signal_style}]{item['signal']}[/{signal_style}]",
                f"[bold]{item['score']:.1f}[/bold]",
                pcr_display,
                opt_oi_display,
                opt_bias_style,  # Now shows technical bias + option bias
                f"[bright_green]{item.get('current_volume', 'N/A')}[/bright_green]",
                f"[bright_magenta]{item.get('current_oi', 'N/A')}[/bright_magenta]",
                vol_change_style,
                oi_change_style,
                f"[bold]{item.get('action', 'Consider')}[/bold]"
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
        
        print_colored("="*150, Colors.BLUE)
        header = f"{'Stock':<10} | {'Signal':<14} | {'Score':>6} | {'PCR':>5} | {'OI_Ch%':>6} | {'TechBias':<10} | {'Vol':>8} | {'OI':>8} | {'VolΔ%':>6} | {'OIΔ%':>6} | {'Action':<12}"
        print_colored(header, Colors.BOLD)
        print_colored("-"*150, Colors.BLUE)
        
        for item in data:
            symbol = item['symbol']
            is_new = new_stocks and symbol in new_stocks
            
            pcr_val = item.get('pcr', 'N/A')
            pcr_str = f"{pcr_val:.2f}" if isinstance(pcr_val, (int, float)) else "N/A"
            
            opt_oi_chg = item.get('option_oi_change', 'N/A')
            opt_oi_str = f"{opt_oi_chg:.1f}%" if isinstance(opt_oi_chg, (int, float)) else "N/A"
            
            # FIXED: Show technical bias instead of option sentiment
            technical_signal = item['signal']
            if 'Buy' in technical_signal:
                tech_bias = "📈 Bull"
            elif 'Sell' in technical_signal:
                tech_bias = "📉 Bear"
            else:
                tech_bias = "➡️  Neut"
            
            vol_chg = item.get('volume_change_pct', 0)
            vol_chg_str = f"{vol_chg:+.1f}" if isinstance(vol_chg, (int, float)) and abs(vol_chg) > 0.1 else "N/A"
            
            oi_chg = item.get('oi_change_pct', 0)
            oi_chg_str = f"{oi_chg:+.1f}" if isinstance(oi_chg, (int, float)) and abs(oi_chg) > 0.1 else "N/A"
            
            row = f"{symbol:<10} | {item['signal']:<14} | {item['score']:>6.1f} | {pcr_str:>5} | {opt_oi_str:>6} | {tech_bias:<10} | {item.get('current_volume', 'N/A'):>8} | {item.get('current_oi', 'N/A'):>8} | {vol_chg_str:>5}% | {oi_chg_str:>5}% | {item.get('action', 'Consider'):<12}"
            
            if is_new:
                print_colored(row + " ← ✨ NEW!", Colors.MAGENTA)
            else:
                print(row)
        
        print_colored("="*150, Colors.BLUE)


# ========== MAIN EXECUTION AND LIVE SCANNER ==========

def run_full_day_backtest_merged(backtest_date, stocks):
    """Run full day backtest with merged TrueData + Localhost data"""
    global backtest_stock_history, intraday_volume_data, intraday_oi_data
    
    print_colored(f"\n🚀 STARTING MERGED BACKTEST FOR {backtest_date}", Colors.HEADER)
    print_colored("📊 Using TrueData OHLC/Volume/OI + Localhost Option Chain data", Colors.CYAN)
    
    # Generate timestamps for 5-minute intervals
    base_date = IST.localize(datetime.strptime(backtest_date, "%Y-%m-%d"))
    timestamps = []
    current_time = base_date.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = base_date.replace(hour=15, minute=30, second=0, microsecond=0)
    
    first_scan = current_time + timedelta(minutes=5, seconds=Config.SETTLE_DELAY_SECONDS)
    timestamps.append(first_scan)
    
    current_scan = first_scan
    while current_scan < market_end:
        current_scan += timedelta(minutes=5)
        if current_scan <= market_end:
            timestamps.append(current_scan)
    
    total_scans = len(timestamps)
    print_colored(f"📅 Generated {total_scans} scan points from {timestamps[0].strftime('%H:%M')} to {timestamps[-1].strftime('%H:%M')}", Colors.CYAN)
    
    market_regime = get_market_regime(Config.BENCHMARK_INDEX)
    print_colored(f"📈 Market Regime: {market_regime.upper()}", Colors.BLUE)
    
    all_results = []
    backtest_stock_history = {}
    intraday_volume_data = {}
    intraday_oi_data = {}
    
    with tqdm(total=total_scans, desc="Merged Backtesting", ncols=120) as pbar:
        for i, scan_time in enumerate(timestamps):
            try:
                pbar.set_description(f"Scanning at {scan_time.strftime('%H:%M:%S')}")
                
                signals, current_symbols = run_merged_scan_at_time(scan_time, stocks, market_regime, is_live=False)
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
                    'new_stocks': list(new_stocks),
                    'signals': signals
                }
                all_results.append(scan_result)
                
                if signals:
                    signals.sort(key=lambda x: abs(x['score']), reverse=True)
                    top_bullish = [r for r in signals if r['score'] > 0][:Config.BACKTEST_TOP_DISPLAY]
                    top_bearish = [r for r in signals if r['score'] < 0][:Config.BACKTEST_TOP_DISPLAY]
                    
                    scan_time_str = scan_time.strftime('%H:%M')
                    
                    vol_with_changes = sum(1 for s in signals if isinstance(s.get('volume_change_pct', 0), (int, float)) and abs(s.get('volume_change_pct', 0)) > 0.1)
                    oi_with_changes = sum(1 for s in signals if isinstance(s.get('oi_change_pct', 0), (int, float)) and abs(s.get('oi_change_pct', 0)) > 0.1)
                    
                    if RICH_AVAILABLE:
                        console.print(f"\n[bold blue]📊 SCAN {i+1}/{total_scans} - {scan_time_str} IST[/bold blue]")
                        console.print(f"[cyan]Signals: {len(signals)} | Bullish: {len([s for s in signals if s['score'] > 0])} | Bearish: {len([s for s in signals if s['score'] < 0])} | New: {len(new_stocks)}[/cyan]")
                        console.print(f"[yellow]Volume Changes: {vol_with_changes} stocks | OI Changes: {oi_with_changes} stocks[/yellow]")
                    else:
                        print_colored(f"\n📊 SCAN {i+1}/{total_scans} - {scan_time_str} IST", Colors.BOLD)
                        print_colored(f"Signals: {len(signals)} | Bullish: {len([s for s in signals if s['score'] > 0])} | Bearish: {len([s for s in signals if s['score'] < 0])} | New: {len(new_stocks)}", Colors.CYAN)
                        print_colored(f"Volume Changes: {vol_with_changes} stocks | OI Changes: {oi_with_changes} stocks", Colors.YELLOW)
                    
                    if top_bullish:
                        create_enhanced_results_table(top_bullish, "🟢 TOP BULLISH", new_stocks, scan_time_str)
                    
                    if top_bearish:
                        create_enhanced_results_table(top_bearish, "🔴 TOP BEARISH", new_stocks, scan_time_str)
                
                pbar.update(1)
                time.sleep(0.1)  # Brief pause between scans
                
            except Exception as e:
                logger.error(f"Error in backtest scan at {scan_time}: {e}")
                pbar.update(1)
                continue
    
    # Save results
    output_filename = f"{backtest_date}_merged_backtest_results.json"
    try:
        with open(output_filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        print_colored(f"\n💾 Results saved: {output_filename}", Colors.GREEN)
    except Exception as e:
        logger.error(f"Could not save results: {e}")
    
    # Show enhanced summary
    print_colored("="*120, Colors.BLUE)
    print_colored(f"📊 MERGED BACKTEST SUMMARY FOR {backtest_date}", Colors.HEADER)
    print_colored("="*120, Colors.BLUE)
    
    total_scans_completed = len([r for r in all_results if r['total_signals'] > 0])
    total_signals = sum(r['total_signals'] for r in all_results)
    total_bullish = sum(r['bullish_signals'] for r in all_results)
    total_bearish = sum(r['bearish_signals'] for r in all_results)
    unique_stocks = len(backtest_stock_history)
    
    print(f"✅ Scans: {total_scans_completed}/{total_scans}")
    print(f"📊 Total Signals: {total_signals}")
    print(f"🟢 Bullish: {total_bullish}")
    print(f"🔴 Bearish: {total_bearish}")
    print(f"📈 Unique Stocks: {unique_stocks}")
    
    if total_signals > 0:
        print(f"📊 Avg Signals/Scan: {total_signals/total_scans_completed:.1f}")
        print(f"⚖️  Bull/Bear Ratio: {total_bullish/max(total_bearish, 1):.2f}")
    
    print_colored("="*120, Colors.BLUE)
    print_colored("✅ Merged backtesting completed!", Colors.GREEN)

def main_merged_scanner():
    """Main function for merged scanner"""
    parser = argparse.ArgumentParser(description="Merged Option Buyer Scanner v4.0 - TrueData + Localhost API")
    parser.add_argument("--asof", type=str, help="Backtest snapshot: 2025-09-30T14:50")
    parser.add_argument("--backtest", type=str, help="Full day backtest: 2025-09-30")
    args = parser.parse_args()
    
    try:
        with open(Config.SHARES_FILE, 'r') as f:
            stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(stocks)} symbols from {Config.SHARES_FILE}")
    except Exception:
        stocks = ["RELIANCE-EQ", "TCS-EQ", "HDFCBANK-EQ", "INFY-EQ", "HINDUNILVR-EQ", "ICICIBANK-EQ", "SBIN-EQ"]
        logger.warning(f"Could not load {Config.SHARES_FILE}. Using sample stocks.")
    
    if args.backtest:
        try:
            datetime.strptime(args.backtest, "%Y-%m-%d")
            run_full_day_backtest_merged(args.backtest, stocks)
        except ValueError:
            logger.error("Invalid date format for --backtest. Use YYYY-MM-DD.")
            return
    
    elif args.asof:
        try:
            asof_ts = IST.localize(datetime.fromisoformat(args.asof))
        except ValueError:
            try:
                asof_ts = IST.localize(datetime.strptime(args.asof, "%Y-%m-%d"))
                asof_ts = asof_ts.replace(hour=15, minute=30)
            except ValueError:
                logger.error(f"Invalid timestamp format: {args.asof}")
                return
        
        logger.info(f"Running merged snapshot for: {asof_ts.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        market_regime = get_market_regime(Config.BENCHMARK_INDEX)
        
        signals, _ = run_merged_scan_at_time(asof_ts, stocks, market_regime, is_live=False)
        signals.sort(key=lambda x: abs(x['score']), reverse=True)
        
        top_bullish = [r for r in signals if r['score'] > 0][:15]
        top_bearish = [r for r in signals if r['score'] < 0][:15]
        
        print_colored(f"\n🎯 MERGED SNAPSHOT RESULTS - {asof_ts.strftime('%Y-%m-%d %H:%M')} IST", Colors.BOLD)
        create_enhanced_results_table(top_bullish, "🟢 TOP 15 BULLISH OPPORTUNITIES")
        create_enhanced_results_table(top_bearish, "🔴 TOP 15 BEARISH OPPORTUNITIES")
    
    else:
        # Live scanner
        print_colored("\n🚀 STARTING MERGED LIVE SCANNER v4.0", Colors.GREEN)
        print_colored("📊 TrueData OHLC/Volume/OI + Localhost Option Chain Integration", Colors.CYAN)
        
        global scan_count, previous_scan_results, intraday_volume_data, intraday_oi_data
        
        intraday_volume_data = {}
        intraday_oi_data = {}
        scan_count = 0
        previous_scan_results = {}
        
        # Wait for first scan at 09:20:15
        now_ist = datetime.now(IST)
        first_run_time = today_ist_dt(Config.FIRST_RUN_AT)
        first_scan_time = first_run_time + timedelta(seconds=Config.FIRST_SCAN_DELAY)
        
        if now_ist < first_scan_time:
            logger.info(f"Waiting until {first_scan_time.strftime('%H:%M:%S')} IST for first scan...")
            sleep_until(first_scan_time)
        
        while True:
            scan_count += 1
            now_ist = datetime.now(IST)
            
            if now_ist.time() > datetime.strptime(Config.MARKET_END, "%H:%M").time():
                logger.info("Market closed. Shutting down.")
                break
            
            print_colored(f"\n[{now_ist.strftime('%H:%M:%S')}] MERGED SCANNER v4.0 - Scan #{scan_count}", Colors.HEADER)
            
            market_regime = get_market_regime(Config.BENCHMARK_INDEX)
            signals, current_symbols = run_merged_scan_at_time(now_ist, stocks, market_regime, is_live=True)
            
            new_stocks = current_symbols - set(previous_scan_results.keys()) if previous_scan_results else set()
            previous_scan_results = {s: True for s in current_symbols}
            
            signals.sort(key=lambda x: abs(x['score']), reverse=True)
            top_bullish = [r for r in signals if r['score'] > 0][:15]
            top_bearish = [r for r in signals if r['score'] < 0][:15]
            
            print_colored(f"\n🎯 MERGED SCANNER RESULTS - {now_ist.strftime('%Y-%m-%d %H:%M')} IST (Regime: {market_regime.upper()})", Colors.BOLD)
            
            create_enhanced_results_table(top_bullish, "🟢 TOP 15 BULLISH OPPORTUNITIES", new_stocks)
            create_enhanced_results_table(top_bearish, "🔴 TOP 15 BEARISH OPPORTUNITIES", new_stocks)
            
            next_scan_time = get_exact_candle_close_time(datetime.now(IST))
            logger.info(f"Next scan at {next_scan_time.strftime('%H:%M:%S')}")
            sleep_until(next_scan_time)

# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    try:
        print_colored("\n🎯 MERGED OPTION BUYER SCANNER v4.0", Colors.HEADER)
        print_colored("🔗 TrueData OHLC/Volume/OI + Localhost Option Chain Integration", Colors.GREEN)
        print_colored("✨ Parallel API calls with enhanced scoring for option buyers", Colors.CYAN)
        
        if RICH_AVAILABLE:
            print_colored("✨ Rich: Available for enhanced visualization", Colors.GREEN)
        else:
            print_colored("ℹ️ ASCII: Using fallback table formatting", Colors.YELLOW)
        
        print_colored("\n🔧 MERGED SCANNER FEATURES:", Colors.CYAN)
        print(" ✅ TrueData: OHLC, Volume, OI data with multiple timeframes")
        print(" ✅ Localhost: Option chain, PCR, OI%, ATM analysis")
        print(" ✅ Parallel: Both APIs called simultaneously for speed")
        print(" ✅ Enhanced: Option buyer specific scoring weights")
        print(" ✅ Real-time: 5-minute candle tracking with volume/OI changes")
        
        print_colored("\n📋 USAGE EXAMPLES:", Colors.YELLOW)
        print(" 🔴 Live Trading: python merged_scanner.py")
        print(" 🔍 Single Snapshot: python merged_scanner.py --asof 2025-09-30T14:25")
        print(" 📈 Full Day Backtest: python merged_scanner.py --backtest 2025-09-30")
        
        print_colored("="*100, Colors.HEADER)
        print_colored("\n🎯 Starting Merged Scanner...", Colors.GREEN)
        
        main_merged_scanner()
        
    except KeyboardInterrupt:
        print_colored("\n\n⚠️ Scanner interrupted by user. Shutting down gracefully...", Colors.YELLOW)
    except Exception as e:
        logger.error(f"Critical error: {e}")
        print_colored(f"\n💥 Critical error occurred: {e}", Colors.RED)
