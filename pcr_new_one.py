# ==============================================================================
# ULTIMATE OPTION BUYER SCANNER v4.3 - COMPLETE WITH ALL DATETIME FIXES
# TrueData: Uses symbols with -I suffix (RELIANCE-I, TCS-I)
# Localhost API: Uses clean symbols without -I (RELIANCE, TCS)
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

# ======== ULTIMATE Configuration for Option Buyers ========
class Config:
    # TrueData Configuration
    TDUSERNAME = os.getenv("TRUEDATA_USER", "tdwsp751")
    TDPASSWORD = os.getenv("TRUEDATA_PASS", "raj@751")
    
    # Localhost Option Chain API Configuration
    LOCALHOST_API_TMPL = "http://localhost:3000/api/equity/options/{symbol}"
    LOCALHOST_TIMEOUT = 20
    
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
    
    # Option Chain Analysis Thresholds (for perfect option buying)
    MIN_TOTAL_OI = 2000          # Minimum total OI for liquidity
    MIN_TOTAL_VOL = 200          # Minimum total volume
    PCR_BULLISH_THRESHOLD = 0.8  # PCR below this = bullish for calls
    PCR_BEARISH_THRESHOLD = 1.2  # PCR above this = bearish for puts
    PCR_TOL = 0.03               # PCR tolerance for neutral
    MIN_OI_CHANGE_THRESHOLD = 5.0 # Minimum OI change % for significance
    MIN_VOL_SURGE_THRESHOLD = 1.5 # Volume surge multiplier
    
    # Enhanced Indicator Group Weights (Optimized for Option Buyers)
    GROUP_WEIGHTS = {
        "Trend": 2.0,           # Reduced - option buyers care more about momentum
        "Momentum": 3.0,        # Increased - critical for option timing
        "Volume": 2.5,          # Increased - volume drives option premiums
        "Volatility": 2.2,      # Increased - volatility = option opportunity
        "OI": 2.0,              # Standard - open interest tracking
        "OptionChain": 3.5,     # NEW - highest weight for option-specific signals
    }
    
    # Enhanced Individual Indicator Weights (Option-Buyer Optimized)
    INDICATOR_WEIGHTS = {
        # Trend indicators
        "MA_Slope": 2.0, "ADX": 2.2, "VWAP": 1.8, "EMA": 1.7, "MACD_Trend": 2.0,
        
        # Momentum indicators (CRITICAL for options)
        "RSI": 2.5, "Stochastic": 2.0, "CCI": 1.8, "ROC": 2.0, "WilliamsR": 1.5,
        
        # Volume indicators (drives option premiums)
        "VolumeSurge": 3.0, "OBV": 2.0, "CMF": 2.2, "RelVol": 2.0,
        
        # Volatility indicators (option opportunity)
        "VolatilityExpansion": 2.8, "Bollinger": 2.0,
        
        # OI indicators
        "OptionBuyerMomentum": 3.0, "OIChange": 2.5, "VolumeOISync": 2.2,
        
        # NEW: Option Chain specific indicators (HIGHEST WEIGHTS)
        "PCR_Signal": 3.5,           # PCR analysis for option direction
        "OI_Change_Signal": 3.2,     # OI percentage changes
        "Option_Volume_Surge": 3.0,  # CE/PE volume analysis
        "ATM_Dominance": 2.8,        # At-the-money activity
        "IV_Skew_Signal": 2.5,       # Implied volatility skew
        "Liquidity_Score": 2.2,      # Option liquidity assessment
    }
    
    # Enhanced Scoring & Signal Thresholds (Option-Buyer Focused)
    SCORE_THRESHOLD_MIN = 3.0    # Lowered for better signal detection
    SIGNAL_THRESHOLDS = {
        'Perfect Call Buy': 70.0,    # NEW: Perfect call setup
        'Perfect Put Buy': -70.0,    # NEW: Perfect put setup
        'Very Strong Buy': 55.0,
        'Strong Buy': 30.0,          # Lowered for more signals
        'Buy Signal': 15.0,          # Lowered for more signals
        'Very Strong Sell': -55.0,
        'Strong Sell': -30.0,        # Lowered for more signals
        'Sell Signal': -15.0,        # Lowered for more signals
    }
    
    # Market Regime Multipliers (Enhanced for Options)
    REGIME_MULTIPLIERS = {
        'bullish_in_bull_market': 1.25,   # Increased
        'bearish_in_bear_market': 1.25,   # Increased
        'bullish_in_bear_market': 0.7,    # Decreased penalty
        'bearish_in_bull_market': 0.7,    # Decreased penalty
    }

    # ========== ENHANCED VOLUME FILTER (20 YEARS EXPERIENCE) ==========
    VOLUME_FILTER_MODE = 'relaxed'  # 'strict' | 'relaxed' | 'off'
    MIN_VOLUME_SMA_RATIO = 5.0
    VOLUME_SMA_PERIOD = 20
    EXTREME_VOLUME_THRESHOLD = 10.0
    HIGH_VOLUME_THRESHOLD = 5.0
    MODERATE_VOLUME_THRESHOLD = 3.0
    VOLUME_10X_MULTIPLIER = 1.5
    VOLUME_5X_MULTIPLIER = 1.3
    VOLUME_3X_MULTIPLIER = 1.1
    

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
    """Convert symbol to localhost API format (clean symbol without suffix)"""
    if symbol.endswith('-I'):
        return symbol.replace('-I', '')
    elif symbol.endswith('-EQ'):
        return symbol.replace('-EQ', '')
    else:
        return symbol

def normalize_symbol_for_display(symbol):
    """Convert symbol for display purposes (clean without suffix)"""
    return convert_to_localhost_symbol(symbol)

# ========== LOCALHOST OPTION CHAIN API INTEGRATION ==========

class LocalhostOptionChainAPI:
    def __init__(self):
        self.API_TMPL = Config.LOCALHOST_API_TMPL
        self.TIMEOUT = Config.LOCALHOST_TIMEOUT
        self.MIN_TOTAL_OI = Config.MIN_TOTAL_OI
        self.MIN_TOTAL_VOL = Config.MIN_TOTAL_VOL
        self.PCR_TOL = Config.PCR_TOL
        self.EPS = 1e-6

    def safe_div(self, a, b):
        """Safe division to avoid division by zero"""
        if b is None or abs(b) < self.EPS:
            return float('inf') if a > 0 else 0.0
        return a / b

    def pct_change(self, now, prev):
        """Calculate percentage change"""
        if now is None or prev is None:
            return None
        denom = prev if abs(prev) > self.EPS else self.EPS
        return ((now - prev) / denom) * 100.0

    def parse_expiry(self, s):
        """Parse expiry date string"""
        try:
            return datetime.strptime(s, "%d-%b-%Y")
        except (ValueError, TypeError):
            return None

    def choose_current_expiry(self, records):
        """Choose the current/nearest expiry date"""
        exps = records.get("expiryDates") or []
        exps_parsed = [(e, self.parse_expiry(e)) for e in exps]
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

    def fetch_symbol_option_data(self, symbol):
        """Fetch comprehensive option chain data from localhost API"""
        try:
            # Use clean symbol for localhost API
            clean_symbol = convert_to_localhost_symbol(symbol)
            url = self.API_TMPL.format(symbol=clean_symbol)
            r = requests.get(url, timeout=self.TIMEOUT)
            r.raise_for_status()
            obj = r.json()
            
            recs = obj.get("records", {})
            curr_exp = self.choose_current_expiry(recs)
            
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
            
            # Comprehensive option chain analysis
            ce_oi_sum, pe_oi_sum = 0, 0
            ce_vol_sum, pe_vol_sum = 0, 0
            ce_oi_wsum, pe_oi_wsum = 0.0, 0.0
            ce_oi_w, pe_oi_w = 0.0, 0.0
            ce_iv_wsum, pe_iv_wsum = 0.0, 0.0
            ce_iv_w, pe_iv_w = 0.0, 0.0
            
            # Strike-wise analysis for ATM detection
            strikes_analysis = []
            
            for row in rows:
                strike_price = row.get("strikePrice", 0)
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
                ce_oi_change = ce.get("pchangeinOpenInterest")
                pe_oi_change = pe.get("pchangeinOpenInterest")
                
                if isinstance(ce_oi_change, (int, float)) and ce_oi > 0:
                    ce_oi_wsum += ce_oi_change * ce_oi
                    ce_oi_w += ce_oi
                
                if isinstance(pe_oi_change, (int, float)) and pe_oi > 0:
                    pe_oi_wsum += pe_oi_change * pe_oi
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
                
                # Store strike analysis
                strikes_analysis.append({
                    'strike': strike_price,
                    'distance_from_spot': abs(strike_price - underlying),
                    'ce_oi': ce_oi,
                    'pe_oi': pe_oi,
                    'ce_vol': ce_vol,
                    'pe_vol': pe_vol,
                    'ce_oi_change': ce_oi_change,
                    'pe_oi_change': pe_oi_change
                })
            
            # Calculate final aggregated metrics
            total_oi = ce_oi_sum + pe_oi_sum
            total_vol = ce_vol_sum + pe_vol_sum
            pcr = self.safe_div(pe_oi_sum, ce_oi_sum)
            
            ce_oi_chg_pct = self.safe_div(ce_oi_wsum, ce_oi_w)
            pe_oi_chg_pct = self.safe_div(pe_oi_wsum, pe_oi_w)
            blended_oi_chg = self.safe_div((ce_oi_chg_pct * ce_oi_sum) + (pe_oi_chg_pct * pe_oi_sum), total_oi)
            
            avg_ce_iv = self.safe_div(ce_iv_wsum, ce_iv_w)
            avg_pe_iv = self.safe_div(pe_iv_wsum, pe_iv_w)
            avg_iv = self.safe_div((avg_ce_iv * ce_oi_sum) + (avg_pe_iv * pe_oi_sum), total_oi) * 100
            
            vol_oi_ratio = self.safe_div(total_vol, total_oi)
            
            # Enhanced ATM analysis
            if strikes_analysis:
                atm_strike_data = min(strikes_analysis, key=lambda x: x['distance_from_spot'])
                atm_pcr = self.safe_div(atm_strike_data['pe_oi'], atm_strike_data['ce_oi'])
                atm_vol_ratio = self.safe_div(atm_strike_data['ce_vol'] + atm_strike_data['pe_vol'], total_vol) * 100
                
                # ATM dominance calculation
                atm_ce_vol = atm_strike_data['ce_vol']
                atm_pe_vol = atm_strike_data['pe_vol']
                atm_vol_dom = "CALLS" if atm_ce_vol > atm_pe_vol else ("PUTS" if atm_pe_vol > atm_ce_vol else "NEUTRAL")
                
                # Enhanced ATM signal
                atm_signal = f"Strike:{atm_strike_data['strike']:.0f}|PCR:{atm_pcr:.2f}|{atm_vol_dom}|Vol:{atm_vol_ratio:.1f}%"
            else:
                atm_strike_data = {'strike': underlying}
                atm_pcr = pcr
                atm_vol_ratio = 0
                atm_vol_dom = "NEUTRAL"
                atm_signal = f"Strike:{underlying:.0f}|PCR:{pcr:.2f}|NEUTRAL|Vol:0%"
            
            # Liquidity assessment
            liquidity_score = self.calculate_liquidity_score(total_oi, total_vol, strikes_analysis)
            
            # Enhanced sentiment classification for option buyers
            sentiment = self.classify_option_buyer_sentiment(
                pcr, ce_oi_sum, pe_oi_sum, ce_vol_sum, pe_vol_sum, 
                blended_oi_chg, atm_vol_dom, liquidity_score
            )
            
            # Option buyer specific signals
            call_strength = self.calculate_call_strength(
                pcr, ce_vol_sum, pe_vol_sum, ce_oi_chg_pct, atm_vol_dom, liquidity_score
            )
            
            put_strength = self.calculate_put_strength(
                pcr, ce_vol_sum, pe_vol_sum, pe_oi_chg_pct, atm_vol_dom, liquidity_score
            )
            
            return {
                # Basic data
                "Stock": clean_symbol,
                "Price": underlying,
                "Expiry": curr_exp,
                
                # Volume data
                "CE_Volume": ce_vol_sum,
                "PE_Volume": pe_vol_sum,
                "Total_Volume": total_vol,
                
                # OI data
                "CE_OI": ce_oi_sum,
                "PE_OI": pe_oi_sum,
                "Total_OI": total_oi,
                
                # Key metrics
                "PCR": pcr,
                "OI_Change_Pct": blended_oi_chg,
                "CE_OI_Change": ce_oi_chg_pct,
                "PE_OI_Change": pe_oi_chg_pct,
                
                # Advanced metrics
                "Avg_IV": avg_iv,
                "Vol_OI_Ratio": vol_oi_ratio,
                "Liquidity_Score": liquidity_score,
                
                # ATM analysis
                "ATM_Strike": atm_strike_data['strike'],
                "ATM_PCR": atm_pcr,
                "ATM_Signal": atm_signal,
                "ATM_Vol_Dominance": atm_vol_dom,
                "ATM_Vol_Ratio": atm_vol_ratio,
                
                # Sentiment and signals
                "Sentiment": sentiment,
                "Call_Strength": call_strength,
                "Put_Strength": put_strength,
                
                # Raw data for further analysis
                "Strikes_Data": strikes_analysis[:10]  # Top 10 strikes near ATM
            }
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {e}")
            return {
                "Stock": convert_to_localhost_symbol(symbol),
                "Error": str(e),
                "PCR": None,
                "OI_Change_Pct": None,
                "Sentiment": "Error",
                "Call_Strength": 0,
                "Put_Strength": 0
            }

    def calculate_liquidity_score(self, total_oi, total_vol, strikes_data):
        """Calculate liquidity score for option trading"""
        try:
            # Base liquidity on total OI and volume
            oi_score = min(100, (total_oi / self.MIN_TOTAL_OI) * 50)
            vol_score = min(100, (total_vol / self.MIN_TOTAL_VOL) * 50)
            
            # Strike distribution score (more strikes with activity = better)
            active_strikes = sum(1 for s in strikes_data if (s['ce_oi'] + s['pe_oi']) > 100)
            strike_score = min(100, active_strikes * 10)
            
            # Weighted average
            liquidity_score = (oi_score * 0.4) + (vol_score * 0.4) + (strike_score * 0.2)
            
            return round(liquidity_score, 1)
        
        except:
            return 50.0  # Default neutral score

    def classify_option_buyer_sentiment(self, pcr, ce_oi, pe_oi, ce_vol, pe_vol, oi_change, atm_dom, liquidity):
        """Enhanced sentiment classification for option buyers"""
        
        # Liquidity check
        total_oi = ce_oi + pe_oi
        total_vol = ce_vol + pe_vol
        is_liquid = (total_oi >= self.MIN_TOTAL_OI) and (total_vol >= self.MIN_TOTAL_VOL) and (liquidity > 60)
        
        # PCR-based signals
        strong_bullish_pcr = pcr < Config.PCR_BULLISH_THRESHOLD
        strong_bearish_pcr = pcr > Config.PCR_BEARISH_THRESHOLD
        neutral_pcr = abs(pcr - 1.0) <= self.PCR_TOL
        
        # Volume dominance
        ce_vol_dom = ce_vol > pe_vol
        pe_vol_dom = pe_vol > ce_vol
        
        # OI change signals
        significant_oi_change = abs(oi_change) > Config.MIN_OI_CHANGE_THRESHOLD
        oi_increasing = oi_change > 0
        
        # ATM activity
        atm_call_dom = atm_dom == "CALLS"
        atm_put_dom = atm_dom == "PUTS"
        
        # Classification logic
        if strong_bullish_pcr and ce_vol_dom and atm_call_dom and is_liquid:
            if significant_oi_change and oi_increasing:
                return "Perfect Call Setup"
            else:
                return "Strong Bullish"
        
        elif strong_bearish_pcr and pe_vol_dom and atm_put_dom and is_liquid:
            if significant_oi_change and oi_increasing:
                return "Perfect Put Setup"
            else:
                return "Strong Bearish"
        
        elif strong_bullish_pcr and (ce_vol_dom or atm_call_dom):
            return "Mild Bullish" if is_liquid else "Weak Bullish"
        
        elif strong_bearish_pcr and (pe_vol_dom or atm_put_dom):
            return "Mild Bearish" if is_liquid else "Weak Bearish"
        
        elif neutral_pcr:
            if ce_vol_dom and atm_call_dom:
                return "Neutral Bullish"
            elif pe_vol_dom and atm_put_dom:
                return "Neutral Bearish"
            else:
                return "Neutral"
        
        else:
            return "Mixed Signal" if is_liquid else "Low Liquidity"

    def calculate_call_strength(self, pcr, ce_vol, pe_vol, ce_oi_change, atm_dom, liquidity):
        """Calculate call option strength score (0-100)"""
        score = 0
        
        # PCR contribution (40 points max)
        if pcr < 0.6:
            score += 40
        elif pcr < 0.8:
            score += 30
        elif pcr < 1.0:
            score += 20
        elif pcr < 1.2:
            score += 10
        
        # Volume dominance (30 points max)
        vol_ratio = ce_vol / max(pe_vol, 1)
        if vol_ratio > 3:
            score += 30
        elif vol_ratio > 2:
            score += 25
        elif vol_ratio > 1.5:
            score += 20
        elif vol_ratio > 1:
            score += 15
        
        # OI change (20 points max)
        if ce_oi_change > 15:
            score += 20
        elif ce_oi_change > 10:
            score += 15
        elif ce_oi_change > 5:
            score += 10
        elif ce_oi_change > 0:
            score += 5
        
        # ATM dominance (10 points max)
        if atm_dom == "CALLS":
            score += 10
        elif atm_dom == "NEUTRAL":
            score += 5
        
        # Liquidity penalty
        if liquidity < 50:
            score *= 0.5
        elif liquidity < 70:
            score *= 0.8
        
        return min(100, max(0, round(score)))

    def calculate_put_strength(self, pcr, ce_vol, pe_vol, pe_oi_change, atm_dom, liquidity):
        """Calculate put option strength score (0-100)"""
        score = 0
        
        # PCR contribution (40 points max)
        if pcr > 1.5:
            score += 40
        elif pcr > 1.2:
            score += 30
        elif pcr > 1.0:
            score += 20
        elif pcr > 0.8:
            score += 10
        
        # Volume dominance (30 points max)
        vol_ratio = pe_vol / max(ce_vol, 1)
        if vol_ratio > 3:
            score += 30
        elif vol_ratio > 2:
            score += 25
        elif vol_ratio > 1.5:
            score += 20
        elif vol_ratio > 1:
            score += 15
        
        # OI change (20 points max)
        if pe_oi_change > 15:
            score += 20
        elif pe_oi_change > 10:
            score += 15
        elif pe_oi_change > 5:
            score += 10
        elif pe_oi_change > 0:
            score += 5
        
        # ATM dominance (10 points max)
        if atm_dom == "PUTS":
            score += 10
        elif atm_dom == "NEUTRAL":
            score += 5
        
        # Liquidity penalty
        if liquidity < 50:
            score *= 0.5
        elif liquidity < 70:
            score *= 0.8
        
        return min(100, max(0, round(score)))

    def fetch_multiple_symbols(self, symbols, max_workers=20):
        """Fetch option chain data for multiple symbols in parallel"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self.fetch_symbol_option_data, symbol): symbol 
                for symbol in symbols
            }
            
            # Collect results with progress bar
            with tqdm(total=len(symbols), desc="🔗 Fetching Option Chains", ncols=100, leave=False) as pbar:
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        # Store with clean symbol as key
                        clean_symbol = convert_to_localhost_symbol(symbol)
                        results[clean_symbol] = result
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Failed to fetch option chain for {symbol}: {e}")
                        clean_symbol = convert_to_localhost_symbol(symbol)
                        results[clean_symbol] = {"Stock": clean_symbol, "Error": str(e), "Sentiment": "Error"}
                        pbar.update(1)
        
        return results

# Initialize localhost API client
localhost_api = LocalhostOptionChainAPI()

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


# ========== ENHANCED VOLUME ANALYSIS (20 YEARS OPTION BUYER EXPERIENCE) ==========

def calculate_volume_sma_ratio(df, sma_period=20):
    """Calculate volume vs 20-period SMA - PRIMARY FILTER"""
    if df.empty or len(df) < sma_period + 1:
        return 0.0
    try:
        current_volume = df['Volume'].iloc[-1]
        sma_volume = df['Volume'].rolling(window=sma_period).mean().iloc[-1]
        return current_volume / sma_volume if sma_volume > 0 and not pd.isna(sma_volume) else 0.0
    except:
        return 0.0


def enhanced_volume_surge_analysis(df, sma_period=20):
    """CRITICAL: 5x-10x SMA detection"""
    if df.empty or len(df) < sma_period + 1:
        return {'volume_sma_ratio': 0.0, 'is_5x_surge': False, 'is_10x_surge': False,
                'surge_strength': 'None', 'volume_quality_score': 0, 'volume_percentile': 0,
                'current_volume': 0, 'sma_volume': 0}
    try:
        current_volume = df['Volume'].iloc[-1]
        sma_volume = df['Volume'].rolling(window=sma_period).mean().iloc[-1]
        volume_percentile = (df['Volume'].iloc[-50:] < current_volume).sum() / 50 * 100 if len(df) >= 50 else 0

        if sma_volume <= 0 or pd.isna(sma_volume):
            return {'volume_sma_ratio': 0.0, 'is_5x_surge': False, 'is_10x_surge': False,
                    'surge_strength': 'None', 'volume_quality_score': 0, 'volume_percentile': volume_percentile,
                    'current_volume': int(current_volume) if current_volume > 0 else 0, 'sma_volume': 0}

        volume_ratio = current_volume / sma_volume
        is_10x_surge = volume_ratio >= 10.0
        is_5x_surge = volume_ratio >= 5.0

        if is_10x_surge:
            surge_strength, quality_score = 'Extreme (10x+)', 100
        elif is_5x_surge:
            surge_strength, quality_score = 'Very High (5-10x)', 85
        elif volume_ratio >= 3.0:
            surge_strength, quality_score = 'High (3-5x)', 70
        elif volume_ratio >= 2.0:
            surge_strength, quality_score = 'Moderate (2-3x)', 50
        else:
            surge_strength, quality_score = 'Normal', 10

        return {'volume_sma_ratio': round(volume_ratio, 2), 'is_5x_surge': is_5x_surge,
                'is_10x_surge': is_10x_surge, 'surge_strength': surge_strength,
                'volume_quality_score': quality_score, 'volume_percentile': round(volume_percentile, 1),
                'current_volume': int(current_volume), 'sma_volume': int(sma_volume)}
    except:
        return {'volume_sma_ratio': 0.0, 'is_5x_surge': False, 'is_10x_surge': False,
                'surge_strength': 'Error', 'volume_quality_score': 0, 'volume_percentile': 0,
                'current_volume': 0, 'sma_volume': 0}


def analyze_strike_volume_surge(option_chain_data):
    """Analyze strike volumes for 5x-10x surges"""
    if not option_chain_data or 'Strikes_Data' not in option_chain_data:
        return {'max_ce_volume_surge': 0, 'max_pe_volume_surge': 0, 'atm_volume_surge': 0,
                'high_volume_strikes': [], 'ce_pe_volume_ratio': 1.0}
    try:
        strikes_data = option_chain_data.get('Strikes_Data', [])
        if not strikes_data:
            return {'max_ce_volume_surge': 0, 'max_pe_volume_surge': 0, 'atm_volume_surge': 0,
                    'high_volume_strikes': [], 'ce_pe_volume_ratio': 1.0}

        ce_volumes = [s['ce_vol'] for s in strikes_data if s['ce_vol'] > 0]
        pe_volumes = [s['pe_vol'] for s in strikes_data if s['pe_vol'] > 0]
        avg_ce_vol = np.mean(ce_volumes) if ce_volumes else 1
        avg_pe_vol = np.mean(pe_volumes) if pe_volumes else 1

        high_volume_strikes = []
        max_ce_surge = max_pe_surge = atm_volume_surge = 0
        atm_strike = min(strikes_data, key=lambda x: x['distance_from_spot'])

        for strike in strikes_data:
            ce_surge = strike['ce_vol'] / avg_ce_vol if avg_ce_vol > 0 else 0
            pe_surge = strike['pe_vol'] / avg_pe_vol if avg_pe_vol > 0 else 0
            max_ce_surge = max(max_ce_surge, ce_surge)
            max_pe_surge = max(max_pe_surge, pe_surge)

            if ce_surge >= 5.0 or pe_surge >= 5.0:
                high_volume_strikes.append({'strike': strike['strike'], 'ce_surge': round(ce_surge, 2),
                    'pe_surge': round(pe_surge, 2), 'type': 'CE' if ce_surge > pe_surge else 'PE'})

            if strike['strike'] == atm_strike['strike']:
                atm_volume_surge = (strike['ce_vol'] + strike['pe_vol']) / (avg_ce_vol + avg_pe_vol) if (avg_ce_vol + avg_pe_vol) > 0 else 0

        return {'max_ce_volume_surge': round(max_ce_surge, 2), 'max_pe_volume_surge': round(max_pe_surge, 2),
                'atm_volume_surge': round(atm_volume_surge, 2), 'high_volume_strikes': high_volume_strikes[:5],
                'ce_pe_volume_ratio': sum(ce_volumes) / sum(pe_volumes) if sum(pe_volumes) > 0 else 1.0}
    except:
        return {'max_ce_volume_surge': 0, 'max_pe_volume_surge': 0, 'atm_volume_surge': 0,
                'high_volume_strikes': [], 'ce_pe_volume_ratio': 1.0}


def format_volume_display(volume_data):
    """Format volume display"""
    ratio = volume_data.get('volume_sma_ratio', 0)
    strength = volume_data.get('surge_strength', 'None')

    if ratio >= 10:
        color, emoji = Colors.GREEN + Colors.BOLD, "🔥🔥🔥"
    elif ratio >= 5:
        color, emoji = Colors.GREEN, "🔥🔥"
    elif ratio >= 3:
        color, emoji = Colors.YELLOW, "🔥"
    else:
        color, emoji = Colors.END, ""

    return f"{color}{ratio:.1f}x {emoji} ({strength}){Colors.END}"


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

def calculate_option_chain_scores(option_chain_data, stock_volume_data=None):
    """Calculate enhanced scores from option chain data"""
    scores = defaultdict(float)
    
    if not option_chain_data or "Error" in option_chain_data:
        return scores
    
    try:
        # Enhanced PCR Signal (most important for option buyers)
        pcr = option_chain_data.get("PCR", 1.0)
        if pcr is not None and pcr != float('inf'):
            # Enhanced PCR scoring for option buyers
            if pcr < 0.6:
                scores['PCR_Signal'] = 3.5  # Perfect call setup
            elif pcr < 0.8:
                scores['PCR_Signal'] = 2.5  # Strong call setup
            elif pcr < 1.0:
                scores['PCR_Signal'] = 1.5  # Mild call bias
            elif pcr < 1.2:
                scores['PCR_Signal'] = -1.5  # Mild put bias
            elif pcr < 1.4:
                scores['PCR_Signal'] = -2.5  # Strong put setup
            else:
                scores['PCR_Signal'] = -3.5  # Perfect put setup
        
        # Enhanced OI Change Signal
        oi_change_pct = option_chain_data.get("OI_Change_Pct", 0)
        if oi_change_pct is not None and oi_change_pct != float('inf'):
            scores['OI_Change_Signal'] = normalize_score(
                oi_change_pct, (10, 25), (-10, -25), (-3.2, 3.2)
            )
        
        # Option Volume Surge Analysis
        ce_vol = option_chain_data.get("CE_Volume", 0) or 0
        pe_vol = option_chain_data.get("PE_Volume", 0) or 0
        total_vol = ce_vol + pe_vol
        
        if total_vol > Config.MIN_TOTAL_VOL:
            vol_ratio = ce_vol / max(pe_vol, 1) if pe_vol > 0 else float('inf')
            if vol_ratio > 3:
                scores['Option_Volume_Surge'] = 3.0  # Strong call volume
            elif vol_ratio > 2:
                scores['Option_Volume_Surge'] = 2.0
            elif vol_ratio > 1.5:
                scores['Option_Volume_Surge'] = 1.0
            elif vol_ratio < 0.33:
                scores['Option_Volume_Surge'] = -3.0  # Strong put volume
            elif vol_ratio < 0.5:
                scores['Option_Volume_Surge'] = -2.0
            elif vol_ratio < 0.67:
                scores['Option_Volume_Surge'] = -1.0
        
        # ATM Dominance Score
        atm_vol_dom = option_chain_data.get("ATM_Vol_Dominance", "NEUTRAL")
        atm_vol_ratio = option_chain_data.get("ATM_Vol_Ratio", 0)
        
        if atm_vol_dom == "CALLS" and atm_vol_ratio > 10:
            scores['ATM_Dominance'] = 2.8
        elif atm_vol_dom == "CALLS":
            scores['ATM_Dominance'] = 1.5
        elif atm_vol_dom == "PUTS" and atm_vol_ratio > 10:
            scores['ATM_Dominance'] = -2.8
        elif atm_vol_dom == "PUTS":
            scores['ATM_Dominance'] = -1.5
        
        # Liquidity Score (critical for option trading)
        liquidity_score = option_chain_data.get("Liquidity_Score", 50)
        if liquidity_score > 80:
            scores['Liquidity_Score'] = 2.2
        elif liquidity_score > 60:
            scores['Liquidity_Score'] = 1.0
        elif liquidity_score < 40:
            scores['Liquidity_Score'] = -1.0
        elif liquidity_score < 20:
            scores['Liquidity_Score'] = -2.2
        
        # IV Skew Signal
        ce_oi_change = option_chain_data.get("CE_OI_Change", 0) or 0
        pe_oi_change = option_chain_data.get("PE_OI_Change", 0) or 0
        
        if isinstance(ce_oi_change, (int, float)) and isinstance(pe_oi_change, (int, float)):
            iv_skew = ce_oi_change - pe_oi_change
            scores['IV_Skew_Signal'] = normalize_score(
                iv_skew, (10, 20), (-10, -20), (-2.5, 2.5)
            )
        
    except Exception as e:
        logger.error(f"Error calculating option chain scores: {e}")
    
    return scores

def calculate_technical_indicator_scores(df):
    """Calculate technical indicator scores"""
    scores = defaultdict(float)
    
    if df is None or df.empty or len(df) < 15:  # Lowered requirement
        return scores
    
    try:
        # --- Enhanced Trend Group (optimized for options) ---
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
        
        # Enhanced MACD for options
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
        
        # --- Enhanced Momentum Group (CRITICAL for options) ---
        rsi = calculate_rsi(df)
        if not rsi.empty and len(rsi) > 0:
            rsi_val = rsi.iloc[-1]
            # Enhanced RSI scoring for options
            if rsi_val > 70:
                scores['RSI'] = 2.5 - (rsi_val - 70) * 0.05  # Diminishing returns above 70
            elif rsi_val > 60:
                scores['RSI'] = 1.5 + (rsi_val - 60) * 0.1
            elif rsi_val > 50:
                scores['RSI'] = (rsi_val - 50) * 0.1
            elif rsi_val > 40:
                scores['RSI'] = (rsi_val - 40) * -0.1
            elif rsi_val > 30:
                scores['RSI'] = -1.5 + (30 - rsi_val) * 0.1
            else:
                scores['RSI'] = -2.5 - (30 - rsi_val) * 0.05  # Diminishing returns below 30
        
        # Enhanced Stochastic for options
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
        
        # --- Enhanced Volume Group (drives option premiums) ---
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
        
        # --- Enhanced Volatility Group (option opportunity) ---
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

def analyze_ultimate_signals(timeframe_data, option_chain_data=None, market_regime='neutral'):
    """Ultimate signal analysis combining technical + option chain data"""
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
            if group == "OptionChain":  # Skip for now, handle separately
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
    
    # Process option chain data (CRITICAL for option buyers)
    if option_chain_data and "Error" not in option_chain_data:
        option_scores = calculate_option_chain_scores(option_chain_data)
        option_weight = Config.GROUP_WEIGHTS.get("OptionChain", 3.5)
        
        oc_grp_score, oc_grp_weight = 0.0, 0.0
        
        for indicator, ind_weight in Config.INDICATOR_WEIGHTS.items():
            if indicator in option_scores:
                belongs_to_option_chain = any(term in indicator for term in 
                    ['PCR', 'OI_Change', 'Volume_Surge', 'ATM', 'IV_Skew', 'Liquidity'])
                
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
    
    # Apply market regime multipliers (enhanced)
    if normalized_score > 0 and market_regime == 'bullish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bullish_in_bull_market']
    elif normalized_score > 0 and market_regime == 'bearish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bullish_in_bear_market']
    elif normalized_score < 0 and market_regime == 'bearish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bearish_in_bear_market']
    elif normalized_score < 0 and market_regime == 'bullish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bearish_in_bull_market']
    
    # Enhanced signal classification for option buyers
    if normalized_score >= Config.SIGNAL_THRESHOLDS['Perfect Call Buy']:
        signal = 'Perfect Call Buy'
    elif normalized_score >= Config.SIGNAL_THRESHOLDS['Very Strong Buy']:
        signal = 'Very Strong Buy'
    elif normalized_score >= Config.SIGNAL_THRESHOLDS['Strong Buy']:
        signal = 'Strong Buy'
    elif normalized_score >= Config.SIGNAL_THRESHOLDS['Buy Signal']:
        signal = 'Buy Signal'
    elif normalized_score <= Config.SIGNAL_THRESHOLDS['Perfect Put Buy']:
        signal = 'Perfect Put Buy'
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
        
        # Enhanced caching with better change detection
        if abs(vol_change_pct) < 0.1 and abs(oi_change_pct) < 0.1:
            prev_volume = intraday_volume_data.get(symbol, None)
            prev_oi = intraday_oi_data.get(symbol, None)
            
            if prev_volume is not None and prev_volume > 0 and current_volume and current_volume > 0:
                vol_change_pct = ((current_volume - prev_volume) / prev_volume) * 100
            
            if prev_oi is not None and prev_oi > 0 and current_oi and current_oi > 0:
                oi_change_pct = ((current_oi - prev_oi) / prev_oi) * 100
        
        # Update cache
        intraday_volume_data[symbol] = current_volume if isinstance(current_volume, int) else 0
        intraday_oi_data[symbol] = current_oi if isinstance(current_oi, int) else 0
        
        # Enhanced formatting
        def format_number(val):
            if isinstance(val, int):
                if val > 10000000:  # 10M+
                    return f"{val/1000000:.1f}M"
                elif val > 100000:  # 100K+
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

# ========== ULTIMATE SCANNER LOGIC WITH CORRECTED SYMBOL HANDLING ==========

def run_ultimate_scan_at_time(time_point_aware, stocks, market_regime, is_live=False):
    """FIXED: Ultimate scan with proper datetime filtering"""
    
    # Convert stocks to TrueData format for fetching
    truedata_stocks = [convert_to_truedata_symbol(s) for s in stocks]
    
    # Step 1: Fetch TrueData OHLC/Volume/OI data
    stock_multi_data = prefetch_all(truedata_stocks, max_workers=Config.MAX_WORKERS) if is_live else \
                      prefetch_all_timeaware(truedata_stocks, time_point_aware, max_workers=Config.MAX_WORKERS)
    
    # Step 2: Fetch Option Chain data from localhost API
    # Convert stocks to clean format for localhost API
    clean_stock_symbols = [convert_to_localhost_symbol(s) for s in stocks]
    option_chain_data = localhost_api.fetch_multiple_symbols(clean_stock_symbols, max_workers=20)
    
    print_colored(f"✅ Data fetch complete. TrueData: {len(stock_multi_data)} stocks | Option Chain: {len(option_chain_data)} stocks", Colors.GREEN)
    print_colored(f"Running ultimate analysis (Regime: {market_regime.upper()})...", Colors.GREEN)
    
    signals_this_scan = []
    current_symbols = set()
    
    # Process both TrueData and Option Chain data with proper symbol mapping
    for truedata_symbol, timeframe_data in stock_multi_data.items():
        # Convert TrueData symbol to clean symbol for display and option chain matching
        clean_symbol = convert_to_localhost_symbol(truedata_symbol)
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
        
        # Get corresponding option chain data
        option_data = option_chain_data.get(clean_symbol, {})
        
        # Ultimate signal analysis combining both data sources
        signal, score, sub_scores = analyze_ultimate_signals(
            filtered_timeframes, option_data, market_regime
        )
        
        if abs(score) >= Config.SCORE_THRESHOLD_MIN:
            # Enhanced institutional flow analysis
            flow_tag = enhanced_institutional_flow_analysis(filtered_timeframes)
            
            # Extract volume/OI data from TrueData
            tf_5min = filtered_timeframes.get(5)
            if tf_5min is not None:
                oi_vol_data = extract_5min_volume_oi_data(tf_5min, clean_symbol, time_point_aware, is_live=is_live)
            else:
                main_tf_data = filtered_timeframes.get(15, filtered_timeframes.get(30, list(filtered_timeframes.values())[0]))
                oi_vol_data = extract_5min_volume_oi_data(main_tf_data, clean_symbol, time_point_aware, is_live=is_live)
            
            # Ultimate action determination using all available data
            action = determine_ultimate_action(signal, score, option_data, market_regime)
            
            # Ultimate result with complete integration
            result = {
                # Basic info
                'symbol': clean_symbol,
                'signal': signal,
                'score': score,
                'sub_scores': sub_scores,
                'flow': flow_tag,
                'action': action,
                
                # TrueData volume/OI data
                **oi_vol_data,
                
                # Option chain data from localhost API
                'pcr': option_data.get('PCR', 'N/A'),
                'option_oi_change': option_data.get('OI_Change_Pct', 'N/A'),
                'option_sentiment': option_data.get('Sentiment', 'Unknown'),
                'atm_signal': option_data.get('ATM_Signal', 'N/A'),
                'option_iv': option_data.get('Avg_IV', 'N/A'),
                'option_vol_oi_ratio': option_data.get('Vol_OI_Ratio', 'N/A'),
                'ce_volume': option_data.get('CE_Volume', 'N/A'),
                'pe_volume': option_data.get('PE_Volume', 'N/A'),
                'ce_oi': option_data.get('CE_OI', 'N/A'),
                'pe_oi': option_data.get('PE_OI', 'N/A'),
                'expiry': option_data.get('Expiry', 'N/A'),
                'liquidity_score': option_data.get('Liquidity_Score', 50),
                'call_strength': option_data.get('Call_Strength', 0),
                'put_strength': option_data.get('Put_Strength', 0),
                'atm_strike': option_data.get('ATM_Strike', 'N/A'),
                'atm_pcr': option_data.get('ATM_PCR', 'N/A'),
                'atm_vol_dominance': option_data.get('ATM_Vol_Dominance', 'N/A'),
                'underlying_price': option_data.get('Price', 'N/A'),
                
                # Quality score for option buyers
                'option_quality': calculate_option_quality_score(option_data, score)
            }
            
            signals_this_scan.append(result)
    
    return signals_this_scan, current_symbols

def determine_ultimate_action(signal, score, option_data, market_regime):
    """Determine ultimate action for option buyers using all data"""
    option_sentiment = option_data.get('Sentiment', 'Unknown')
    call_strength = option_data.get('Call_Strength', 0)
    put_strength = option_data.get('Put_Strength', 0)
    liquidity_score = option_data.get('Liquidity_Score', 50)
    pcr = option_data.get('PCR', 1.0)
    
    # Perfect setups (highest confidence)
    if signal == 'Perfect Call Buy':
        if option_sentiment in ['Perfect Call Setup', 'Strong Bullish'] and call_strength > 80:
            return "🎯 PERFECT CALL BUY"
        else:
            return "🚀 Very Strong Call"
    
    elif signal == 'Perfect Put Buy':
        if option_sentiment in ['Perfect Put Setup', 'Strong Bearish'] and put_strength > 80:
            return "🎯 PERFECT PUT BUY"
        else:
            return "📉 Very Strong Put"
    
    # Strong signals with option confirmation
    elif 'Very Strong Buy' in signal:
        if call_strength > 70 and liquidity_score > 60:
            return "🚀 Strong Call Buy"
        elif call_strength > 50:
            return "📈 Call Buy"
        else:
            return "🤔 Consider Call"
    
    elif 'Very Strong Sell' in signal:
        if put_strength > 70 and liquidity_score > 60:
            return "📉 Strong Put Buy"
        elif put_strength > 50:
            return "📉 Put Buy"
        else:
            return "🤔 Consider Put"
    
    # Standard signals
    elif 'Strong Buy' in signal:
        if call_strength > 60:
            return "📈 Call Buy"
        else:
            return "🤔 Consider Call"
    
    elif 'Strong Sell' in signal:
        if put_strength > 60:
            return "📉 Put Buy"
        else:
            return "🤔 Consider Put"
    
    # Basic signals
    elif 'Buy' in signal:
        if call_strength > 50:
            return "📈 Mild Call"
        else:
            return "⚠️ Weak Call"
    
    elif 'Sell' in signal:
        if put_strength > 50:
            return "📉 Mild Put"
        else:
            return "⚠️ Weak Put"
    
    else:
        return "➡️ Hold/Neutral"

def calculate_option_quality_score(option_data, technical_score):
    """Calculate overall quality score for option trading"""
    try:
        quality_score = 0
        
        # Liquidity component (30%)
        liquidity = option_data.get('Liquidity_Score', 50)
        quality_score += (liquidity / 100) * 30
        
        # Technical score alignment (25%)
        tech_alignment = min(100, abs(technical_score)) / 100
        quality_score += tech_alignment * 25
        
        # Option strength (25%)
        call_strength = option_data.get('Call_Strength', 0)
        put_strength = option_data.get('Put_Strength', 0)
        max_strength = max(call_strength, put_strength)
        quality_score += (max_strength / 100) * 25
        
        # PCR quality (20%)
        pcr = option_data.get('PCR', 1.0)
        if isinstance(pcr, (int, float)) and pcr != float('inf'):
            # Good PCR ranges: 0.6-0.8 (calls) or 1.2-1.5 (puts)
            if 0.6 <= pcr <= 0.8 or 1.2 <= pcr <= 1.5:
                quality_score += 20
            elif 0.8 < pcr < 1.2:
                quality_score += 10
        
        return min(100, round(quality_score, 1))
    
    except:
        return 50.0

# ========== ENHANCED TABLE DISPLAY WITH COMPLETE OPTION CHAIN DATA ==========

def create_ultimate_option_table(data, title, new_stocks=None, show_time=None):
    """Ultimate table with comprehensive option chain data"""
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
        table.add_column("PCR", style="cyan", width=5, justify="right")
        table.add_column("OI%", style="yellow", width=6, justify="right")
        table.add_column("Sentiment", style="green", width=14, justify="left")
        table.add_column("C.Str", style="bright_green", width=5, justify="right")
        table.add_column("P.Str", style="bright_red", width=5, justify="right")
        table.add_column("Liq", style="bright_cyan", width=4, justify="right")
        table.add_column("Quality", style="bright_magenta", width=6, justify="right")
        table.add_column("Action", style="bold", width=20, justify="center")
        
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
            
            # Format option chain data
            pcr_val = item.get('pcr', 'N/A')
            pcr_display = f"{pcr_val:.2f}" if isinstance(pcr_val, (int, float)) and pcr_val != float('inf') else "N/A"
            
            oi_chg = item.get('option_oi_change', 'N/A')
            oi_chg_display = f"{oi_chg:+.1f}%" if isinstance(oi_chg, (int, float)) and oi_chg != float('inf') else "N/A"
            
            call_strength = item.get('call_strength', 0)
            put_strength = item.get('put_strength', 0)
            liquidity = item.get('liquidity_score', 50)
            quality = item.get('option_quality', 50)
            
            # Color coding for strengths
            call_str_style = f"[bright_green]{call_strength}[/bright_green]" if call_strength > 70 else f"[green]{call_strength}[/green]" if call_strength > 50 else f"[dim]{call_strength}[/dim]"
            put_str_style = f"[bright_red]{put_strength}[/bright_red]" if put_strength > 70 else f"[red]{put_strength}[/red]" if put_strength > 50 else f"[dim]{put_strength}[/dim]"
            liq_style = f"[bright_cyan]{liquidity:.0f}[/bright_cyan]" if liquidity > 70 else f"[cyan]{liquidity:.0f}[/cyan]" if liquidity > 50 else f"[dim]{liquidity:.0f}[/dim]"
            qual_style = f"[bright_magenta]{quality:.1f}[/bright_magenta]" if quality > 80 else f"[magenta]{quality:.1f}[/magenta]" if quality > 60 else f"[dim]{quality:.1f}[/dim]"
            
            table.add_row(
                stock_style,
                f"[{signal_style}]{item['signal']}[/{signal_style}]",
                f"[bold]{item['score']:.1f}[/bold]",
                pcr_display,
                oi_chg_display,
                item.get('option_sentiment', 'Unknown'),
                call_str_style,
                put_str_style,
                liq_style,
                qual_style,
                f"[bold]{item.get('action', 'Consider')}[/bold]"
            )
        
        if show_time:
            console.print(f"\n[bold magenta]{title} - {show_time}[/bold magenta]")
        else:
            console.print(f"\n[bold magenta]{title}[/bold magenta]")
        console.print(table)
    
    else:
        # ASCII fallback with complete option data
        if show_time:
            print_colored(f"\n{title} - {show_time}", Colors.HEADER)
        else:
            print_colored(f"\n{title}", Colors.HEADER)
        
        print_colored("="*200, Colors.BLUE)
        header = f"{'Stock':<8} | {'Signal':<16} | {'Score':>6} | {'PCR':>5} | {'OI%':>6} | {'Sentiment':<14} | {'C.Str':>5} | {'P.Str':>5} | {'Liq':>4} | {'Qual':>6} | {'Action':<20}"
        print_colored(header, Colors.BOLD)
        print_colored("-"*200, Colors.BLUE)
        
        for item in data:
            symbol = item['symbol']
            is_new = new_stocks and symbol in new_stocks
            
            pcr_val = item.get('pcr', 'N/A')
            pcr_str = f"{pcr_val:.2f}" if isinstance(pcr_val, (int, float)) and pcr_val != float('inf') else "N/A"
            
            oi_chg = item.get('option_oi_change', 'N/A')
            oi_chg_str = f"{oi_chg:+.1f}%" if isinstance(oi_chg, (int, float)) and oi_chg != float('inf') else "N/A"
            
            call_strength = item.get('call_strength', 0)
            put_strength = item.get('put_strength', 0)
            liquidity = item.get('liquidity_score', 50)
            quality = item.get('option_quality', 50)
            
            row = f"{symbol:<8} | {item['signal']:<16} | {item['score']:>6.1f} | {pcr_str:>5} | {oi_chg_str:>6} | {item.get('option_sentiment', 'Unknown'):<14} | {call_strength:>5} | {put_strength:>5} | {liquidity:>4.0f} | {quality:>6.1f} | {item.get('action', 'Consider'):<20}"
            
            if is_new:
                print_colored(row + " ← ✨ NEW!", Colors.MAGENTA)
            else:
                print(row)
        
        print_colored("="*200, Colors.BLUE)

def create_ultimate_summary_panel(signals):
    """Create summary panel with key statistics"""
    if not signals:
        return
    
    # Calculate statistics
    total_signals = len(signals)
    perfect_setups = len([s for s in signals if 'Perfect' in s.get('signal', '')])
    high_quality = len([s for s in signals if s.get('option_quality', 0) > 80])
    high_liquidity = len([s for s in signals if s.get('liquidity_score', 0) > 70])
    strong_calls = len([s for s in signals if s.get('call_strength', 0) > 70])
    strong_puts = len([s for s in signals if s.get('put_strength', 0) > 70])
    
    avg_quality = sum(s.get('option_quality', 0) for s in signals) / total_signals if total_signals > 0 else 0
    avg_liquidity = sum(s.get('liquidity_score', 0) for s in signals) / total_signals if total_signals > 0 else 0
    
    if RICH_AVAILABLE:
        summary_text = f"""
[bold cyan]📊 ULTIMATE SCAN SUMMARY[/bold cyan]
[green]Total Signals: {total_signals}[/green]
[bright_green]Perfect Setups: {perfect_setups}[/bright_green]
[magenta]High Quality (>80): {high_quality}[/magenta]
[cyan]High Liquidity (>70): {high_liquidity}[/cyan]
[bright_green]Strong Calls: {strong_calls}[/bright_green]
[bright_red]Strong Puts: {strong_puts}[/bright_red]
[yellow]Avg Quality: {avg_quality:.1f}[/yellow]
[blue]Avg Liquidity: {avg_liquidity:.1f}[/blue]
        """
        
        panel = Panel(summary_text, title="Ultimate Scanner Stats", border_style="blue")
        console.print(panel)
    else:
        print_colored("\n📊 ULTIMATE SCAN SUMMARY", Colors.HEADER)
        print_colored("="*40, Colors.BLUE)
        print(f"Total Signals: {total_signals}")
        print(f"Perfect Setups: {perfect_setups}")
        print(f"High Quality (>80): {high_quality}")
        print(f"High Liquidity (>70): {high_liquidity}")
        print(f"Strong Calls: {strong_calls}")
        print(f"Strong Puts: {strong_puts}")
        print(f"Avg Quality: {avg_quality:.1f}")
        print(f"Avg Liquidity: {avg_liquidity:.1f}")
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
        clean_symbol = convert_to_localhost_symbol(orig_stock)
        print(f"   📈 {orig_stock} -> TrueData: {td_symbol} | Localhost: {clean_symbol}")
    
    truedata_test_stocks = [convert_to_truedata_symbol(s) for s in test_stocks]
    stock_multi_data = prefetch_all(truedata_test_stocks, max_workers=5) if is_live else \
                      prefetch_all_timeaware(truedata_test_stocks, time_point_aware, max_workers=5)
    
    print(f"   ✅ TrueData received data for {len(stock_multi_data)} stocks")
    for symbol, timeframes in stock_multi_data.items():
        clean_display = convert_to_localhost_symbol(symbol)
        print(f"   📈 {clean_display} ({symbol}): {list(timeframes.keys())} timeframes")
        for tf, df in timeframes.items():
            if df is not None:
                print(f"      {tf}min: {len(df)} candles, latest: {df.index[-1] if len(df) > 0 else 'No data'}")
    
    # Step 2: Test localhost API
    print_colored("\n🔗 Step 2: Testing localhost option chain API for first 3 stocks...", Colors.CYAN)
    clean_symbols = [convert_to_localhost_symbol(s) for s in test_stocks[:3]]
    try:
        option_chain_data = localhost_api.fetch_multiple_symbols(clean_symbols, max_workers=3)
        print(f"   ✅ Received option data for {len(option_chain_data)} symbols")
        for symbol, data in option_chain_data.items():
            if 'Error' in data:
                print(f"   ❌ {symbol}: {data.get('Error', 'Unknown error')}")
            else:
                print(f"   ✅ {symbol}: PCR={data.get('PCR', 'N/A'):.3f}, Sentiment={data.get('Sentiment', 'N/A')}")
    except Exception as e:
        print(f"   ❌ Localhost API Error: {e}")
        print("   💡 Make sure localhost:3000 server is running!")
        option_chain_data = {}
    
    # Step 3: Test signal analysis with corrected symbols
    print_colored("\n📊 Step 3: Testing signal analysis with corrected symbol mapping...", Colors.CYAN)
    signals_found = 0
    
    for truedata_symbol, timeframe_data in list(stock_multi_data.items())[:3]:
        clean_symbol = convert_to_localhost_symbol(truedata_symbol)
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
            option_data = option_chain_data.get(clean_symbol, {})
            signal, score, sub_scores = analyze_ultimate_signals(
                filtered_timeframes, option_data, market_regime
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
            
            if option_data and 'Error' not in option_data:
                option_scores = calculate_option_chain_scores(option_data)
                if option_scores:
                    print(f"      📊 Option Chain Scores: {len(option_scores)} calculated")
                    for ind, val in option_scores.items():
                        print(f"         {ind}: {val:.2f}")
        else:
            print(f"      ❌ Insufficient data: only {len(filtered_timeframes)} timeframes with enough data")
    
    print_colored(f"\n🎯 DIAGNOSTIC SUMMARY:", Colors.HEADER)
    print(f"   📊 TrueData Stocks: {len(stock_multi_data)}")
    print(f"   🔗 Option Chain: {len([d for d in option_chain_data.values() if 'Error' not in d])}")
    print(f"   📈 Potential Signals (threshold 1.0): {signals_found}")
    print(f"   🎯 Current Threshold: {Config.SCORE_THRESHOLD_MIN}")
    
    # Recommendations
    print_colored(f"\n💡 RECOMMENDATIONS:", Colors.YELLOW)
    if len(stock_multi_data) == 0:
        print("   ❌ No TrueData received - check TrueData credentials and connection")
        print("   💡 Try different date range or check if symbols need -I suffix")
    elif len([d for d in option_chain_data.values() if 'Error' not in d]) == 0:
        print("   ❌ No option chain data - check if localhost:3000 server is running")
        print("   💡 Try: cd your_option_server && npm start")
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
    
    # Test 2: Localhost API
    try:
        test_clean_symbol = convert_to_localhost_symbol("RELIANCE")
        test_url = Config.LOCALHOST_API_TMPL.format(symbol=test_clean_symbol)
        print(f"   Testing localhost API URL: {test_url}")
        
        response = requests.get(test_url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("   ✅ Localhost API: OK")
            records = data.get('records', {})
            if records:
                print(f"      Sample data received for {test_clean_symbol}")
            else:
                print("      ⚠️ No records in response")
        else:
            print(f"   ❌ Localhost API: HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Localhost API: {e}")
        print("   💡 Make sure to start: cd option_server && npm start")
    
    # Test 3: Configuration
    print(f"   📊 Score Threshold: {Config.SCORE_THRESHOLD_MIN} (try lowering to 1.0)")
    print(f"   🎯 Group weights sum: {sum(Config.GROUP_WEIGHTS.values())}")

def run_option_chain_only_test():
    """Test with option chain data only (no TrueData requirement) - CORRECTED symbols"""
    print_colored("\n🔗 OPTION CHAIN ONLY TEST - CORRECTED SYMBOLS", Colors.HEADER)
    
    # Test with sample stocks that showed good signals
    test_stocks = ["ABB", "360ONE", "ABCAPITAL", "RELIANCE", "TCS", "HDFCBANK"]
    
    try:
        # Convert to clean symbols for localhost API
        clean_symbols = [convert_to_localhost_symbol(s) for s in test_stocks]
        print_colored(f"📊 Testing with symbols: {clean_symbols}", Colors.CYAN)
        
        option_chain_data = localhost_api.fetch_multiple_symbols(clean_symbols, max_workers=6)
        
        print_colored(f"✅ Received option data for {len(option_chain_data)} symbols", Colors.GREEN)
        
        signals = []
        for symbol, data in option_chain_data.items():
            if 'Error' not in data:
                print(f"   📊 {symbol}: PCR={data.get('PCR', 'N/A'):.3f}, Sentiment={data.get('Sentiment', 'N/A')}")
                
                # Create signal from option chain only
                call_strength = data.get('Call_Strength', 0)
                put_strength = data.get('Put_Strength', 0)
                sentiment = data.get('Sentiment', 'Unknown')
                pcr = data.get('PCR', 1.0)
                liquidity = data.get('Liquidity_Score', 50)
                
                # Simple scoring based on option data only
                if sentiment == 'Perfect Call Setup' and call_strength > 80:
                    score = 85
                    signal = 'Perfect Call Buy'
                elif sentiment == 'Perfect Put Setup' and put_strength > 80:
                    score = -85
                    signal = 'Perfect Put Buy'
                elif call_strength > 70:
                    score = call_strength
                    signal = 'Very Strong Buy'
                elif put_strength > 70:
                    score = -put_strength
                    signal = 'Very Strong Sell'
                elif call_strength > 50:
                    score = call_strength * 0.8
                    signal = 'Strong Buy'
                elif put_strength > 50:
                    score = -put_strength * 0.8
                    signal = 'Strong Sell'
                elif call_strength > 30:
                    score = call_strength * 0.6
                    signal = 'Buy Signal'
                elif put_strength > 30:
                    score = -put_strength * 0.6
                    signal = 'Sell Signal'
                else:
                    continue
                
                result = {
                    'symbol': symbol,
                    'signal': signal,
                    'score': score,
                    'sub_scores': {'OptionChain': score/10},
                    'flow': 'Option Based',
                    'action': determine_ultimate_action(signal, score, data, 'neutral'),
                    'pcr': pcr,
                    'option_oi_change': data.get('OI_Change_Pct', 'N/A'),
                    'option_sentiment': sentiment,
                    'atm_signal': data.get('ATM_Signal', 'N/A'),
                    'option_iv': data.get('Avg_IV', 'N/A'),
                    'call_strength': call_strength,
                    'put_strength': put_strength,
                    'liquidity_score': liquidity,
                    'option_quality': calculate_option_quality_score(data, score),
                    'current_volume': 'N/A',
                    'current_oi': 'N/A'
                }
                signals.append(result)
        
        if signals:
            signals.sort(key=lambda x: abs(x['score']), reverse=True)
            print_colored(f"\n🎯 OPTION CHAIN SIGNALS FOUND: {len(signals)}", Colors.GREEN)
            create_ultimate_option_table(signals, "🔗 PURE OPTION CHAIN SIGNALS (No TrueData Required)")
            create_ultimate_summary_panel(signals)
        else:
            print_colored("❌ No option chain signals generated", Colors.RED)
            
    except Exception as e:
        print_colored(f"❌ Option chain test failed: {e}", Colors.RED)

# ========== ULTIMATE BACKTEST FUNCTION ==========

def run_ultimate_backtest(backtest_date, stocks):
    """Ultimate backtest with complete option chain integration and corrected symbols"""
    global backtest_stock_history, intraday_volume_data, intraday_oi_data
    
    print_colored(f"\n🎯 STARTING ULTIMATE OPTION BUYER BACKTEST FOR {backtest_date}", Colors.HEADER)
    print_colored("🔗 Complete TrueData + Localhost Option Chain Integration with Corrected Symbols", Colors.GREEN)
    
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
                    'perfect_setups': len([s for s in signals if 'Perfect' in s.get('signal', '')]),
                    'high_quality_signals': len([s for s in signals if s.get('option_quality', 0) > 80]),
                    'new_stocks': list(new_stocks),
                    'signals': signals
                }
                
                all_results.append(scan_result)
                
                if signals:
                    signals.sort(key=lambda x: (x.get('option_quality', 0), abs(x['score'])), reverse=True)
                    top_bullish = [r for r in signals if r['score'] > 0][:Config.BACKTEST_TOP_DISPLAY]
                    top_bearish = [r for r in signals if r['score'] < 0][:Config.BACKTEST_TOP_DISPLAY]
                    
                    scan_time_str = scan_time.strftime('%H:%M')
                    
                    if RICH_AVAILABLE:
                        console.print(f"\n[bold blue]🎯 ULTIMATE SCAN #{i+1}/{total_scans} - {scan_time_str} IST[/bold blue]")
                        console.print(f"[cyan]Signals: {len(signals)} | Perfect: {scan_result['perfect_setups']} | Quality: {scan_result['high_quality_signals']} | New: {len(new_stocks)}[/cyan]")
                    else:
                        print_colored(f"\n🎯 ULTIMATE SCAN #{i+1}/{total_scans} - {scan_time_str} IST", Colors.BOLD)
                        print_colored(f"Signals: {len(signals)} | Perfect: {scan_result['perfect_setups']} | Quality: {scan_result['high_quality_signals']} | New: {len(new_stocks)}", Colors.CYAN)
                    
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
    
    # Enhanced summary with option-specific metrics
    print_colored(f"\n📊 ULTIMATE BACKTEST SUMMARY FOR {backtest_date}", Colors.HEADER)
    print_colored("="*150, Colors.BLUE)
    
    total_scans_completed = len([r for r in all_results if r['total_signals'] >= 0])
    total_signals = sum(r['total_signals'] for r in all_results)
    total_bullish = sum(r['bullish_signals'] for r in all_results)
    total_bearish = sum(r['bearish_signals'] for r in all_results)
    total_perfect = sum(r['perfect_setups'] for r in all_results)
    total_high_quality = sum(r['high_quality_signals'] for r in all_results)
    unique_stocks = len(backtest_stock_history)
    
    print(f"✅ Scans Completed: {total_scans_completed}/{total_scans}")
    print(f"📊 Total Signals: {total_signals}")
    print(f"🟢 Bullish Signals: {total_bullish}")
    print(f"🔴 Bearish Signals: {total_bearish}")
    print(f"🎯 Perfect Setups: {total_perfect}")
    print(f"⭐ High Quality: {total_high_quality}")
    print(f"📋 Unique Stocks: {unique_stocks}")
    
    if total_signals > 0:
        perfect_ratio = (total_perfect / total_signals) * 100
        quality_ratio = (total_high_quality / total_signals) * 100
        print(f"📊 Avg Signals/Scan: {total_signals/total_scans_completed:.1f}")
        print(f"⚖️ Bull/Bear Ratio: {total_bullish/max(total_bearish, 1):.2f}")
        print(f"🎯 Perfect Setup %: {perfect_ratio:.1f}%")
        print(f"⭐ High Quality %: {quality_ratio:.1f}%")
    
    # Most profitable times analysis
    active_scans = sorted(all_results, key=lambda x: (x['perfect_setups'], x['high_quality_signals'], x['total_signals']), reverse=True)[:5]
    print_colored("\n🔥 TOP OPPORTUNITY TIMES:", Colors.CYAN)
    for i, scan in enumerate(active_scans):
        if scan['total_signals'] > 0:
            time_str = datetime.fromisoformat(scan['timestamp']).strftime('%H:%M')
            print(f"  {i+1}. {time_str} - {scan['total_signals']} signals | {scan['perfect_setups']} perfect | {scan['high_quality_signals']} quality")
    
    # Save enhanced results
    output_filename = f"{backtest_date}_ultimate_option_backtest_results.json"
    try:
        with open(output_filename, 'w') as f:
            json.dump(all_results, f, indent=2)
        print_colored(f"\n💾 Ultimate results saved: {output_filename}", Colors.GREEN)
    except Exception as e:
        logger.error(f"Could not save results: {e}")
    
    print_colored("="*150, Colors.BLUE)
    print_colored("🎯 Ultimate Option Buyer Backtesting Completed!", Colors.GREEN)

# ========== ENHANCED MAIN FUNCTION WITH CONTINUOUS LIVE SCANNING ==========

def main_ultimate_scanner_with_diagnostics():
    """FIXED: Enhanced main function with corrected symbol handling and continuous live scanning"""
    parser = argparse.ArgumentParser(description="Ultimate Option Buyer Scanner v4.3 - Corrected Symbol Handling with Continuous Live Scanning")
    parser.add_argument("--asof", type=str, help="Historical snapshot: 2025-10-03T14:25")
    parser.add_argument("--backtest", type=str, help="Full day backtest: 2025-10-03")
    parser.add_argument("--test", action="store_true", help="Run quick system test")
    parser.add_argument("--diagnose", action="store_true", help="Run full diagnostic scan")
    parser.add_argument("--option-only", action="store_true", help="Test option chain only")
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
    
    # Option chain only test
    if args.option_only:
        run_option_chain_only_test()
        return
    
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
            signals.sort(key=lambda x: (x.get('option_quality', 0), abs(x['score'])), reverse=True)
            
            # If no signals found, run automatic diagnostics
            if len(signals) == 0:
                print_colored("\n⚠️ No signals found! Running automatic diagnostics...", Colors.YELLOW)
                run_diagnostic_scan(asof_ts, stocks[:5], market_regime, is_live=False)  # Test first 5 stocks
                print_colored("\n💡 SOLUTIONS TO TRY:", Colors.YELLOW)
                print("   1. Lower threshold: python scanner.py --asof {} --threshold 1.0".format(args.asof))
                print("   2. Run diagnostics: python scanner.py --asof {} --diagnose".format(args.asof))
                print("   3. Test system: python scanner.py --test")
                print("   4. Option chain only: python scanner.py --option-only")
                print("   5. Try different time: python scanner.py --asof 2025-10-03T14:00")
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
        print_colored("🔗 Complete TrueData + Localhost Option Chain Integration with Corrected Symbols", Colors.CYAN)
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
                    signals.sort(key=lambda x: (x.get('option_quality', 0), abs(x['score'])), reverse=True)
                    top_bullish = [r for r in signals if r['score'] > 0][:15]
                    top_bearish = [r for r in signals if r['score'] < 0][:15]
                    
                    # Display enhanced results
                    total_signals = len(signals)
                    perfect_setups = len([s for s in signals if 'Perfect' in s.get('signal', '')])
                    high_quality = len([s for s in signals if s.get('option_quality', 0) > 80])
                    
                    print_colored(f"\n🎯 ULTIMATE LIVE RESULTS - {now_ist.strftime('%Y-%m-%d %H:%M')} IST (Regime: {market_regime.upper()})", Colors.BOLD)
                    print_colored(f"📊 Total: {total_signals} | 🎯 Perfect: {perfect_setups} | ⭐ Quality: {high_quality} | ✨ New: {len(new_stocks)}", Colors.CYAN)
                    
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
        print_colored("🎯 ULTIMATE OPTION BUYER SCANNER v4.3 - ALL DATETIME ISSUES FIXED", Colors.HEADER)
        print_colored("🔧 TrueData uses symbols with -I suffix | Localhost API uses clean symbols", Colors.GREEN)
        print_colored("⏰ Runs every 5 minutes during market hours with proper market condition checking", Colors.BLUE)
        print_colored("✨ Perfect Stock Detection for Option Buyers with Complete Data Integration", Colors.CYAN)
        print_colored("="*120, Colors.HEADER)
        
        # Enhanced usage examples
        print_colored(f"\n📋 ENHANCED USAGE EXAMPLES:", Colors.CYAN)
        print("  🔬 Quick Test: python scanner.py --test")
        print("  🔗 Option Chain Only: python scanner.py --option-only")
        print("  🔍 Full Diagnosis: python scanner.py --asof 2025-10-03T15:30 --diagnose")
        print("  📉 Lower Threshold: python scanner.py --asof 2025-10-03T15:30 --threshold 1.0")
        print("  🎯 Normal Snapshot: python scanner.py --asof 2025-10-03T15:30")
        print("  📈 Backtest: python scanner.py --backtest 2025-10-03")
        print("  🔴 Live Continuous: python scanner.py")
        
        # Show symbol conversion examples
        print_colored(f"\n🔧 SYMBOL CONVERSION EXAMPLES:", Colors.CYAN)
        print("  📊 Input: RELIANCE -> TrueData: RELIANCE-I | Localhost: RELIANCE")
        print("  📊 Input: TCS-EQ -> TrueData: TCS-I | Localhost: TCS")
        print("  📊 Input: HDFC-I -> TrueData: HDFC-I | Localhost: HDFC")
        
        # Show ultimate configuration
        print_colored(f"\n📋 ULTIMATE CONFIGURATION:", Colors.CYAN)
        print(f"  📊 TrueData Sessions: {Config.TD_HIST_SESSIONS}")
        print(f"  🔄 Max Workers: {Config.MAX_WORKERS}")
        print(f"  📈 Market Hours: {Config.MARKET_START} - {Config.MARKET_END} IST")
        print(f"  🎯 Score Threshold: {Config.SCORE_THRESHOLD_MIN}")
        print(f"  🔗 Localhost API: {Config.LOCALHOST_API_TMPL}")
        print(f"  📊 Min OI: {Config.MIN_TOTAL_OI:,} | Min Volume: {Config.MIN_TOTAL_VOL:,}")
        print(f"  🎯 PCR Thresholds: Bullish<{Config.PCR_BULLISH_THRESHOLD} | Bearish>{Config.PCR_BEARISH_THRESHOLD}")
        print(f"  ⏰ Scan Interval: Every 5 minutes during market hours")
        print(f"  🕘 Settlement Delay: {Config.SETTLE_DELAY_SECONDS} seconds")
        
        # Show enhanced features
        print_colored(f"\n🎯 ULTIMATE FEATURES v4.3 - ALL FIXES:", Colors.CYAN)
        print("  ✅ Real-time TrueData OHLC/Volume/OI fetching")
        print("  ✅ Parallel localhost option chain API integration")
        print("  ✅ PCR analysis with call/put strength calculation")
        print("  ✅ Liquidity scoring and quality assessment")
        print("  ✅ Perfect setup detection for option buyers")
        print("  ✅ Enhanced technical indicators with option focus")
        print("  ✅ Market regime awareness with multipliers")
        print("  ✅ 5-minute precision with proper candle boundaries")
        print("  ✅ Comprehensive backtesting with option metrics")
        print("  🆕 Continuous live scanning every 5 minutes")
        print("  🆕 Automatic market hours checking")
        print("  🆕 Weekend/holiday handling")
        print("  🆕 Real-time new stock detection")
        print("  🆕 Progressive countdown timer")
        print("  🆕 Enhanced error recovery")
        print("  🆕 Graceful shutdown on market close")
        print("  🔧 FIXED: All datetime comparison errors")
        print("  🔧 FIXED: Corrected symbol handling for both APIs")
        print("  🔧 FIXED: Market hours parsing with time objects")
        print("  🔧 FIXED: Timezone-aware datetime filtering")
        print("  🔧 FIXED: EMA calculation with NaN handling")
        
        print_colored("\n🚀 Starting Ultimate Option Buyer Scanner v4.3...", Colors.GREEN)
        
        # Run the main function
        main_ultimate_scanner_with_diagnostics()
            
    except KeyboardInterrupt:
        print_colored("\n\n👋 Ultimate scanner interrupted by user. Goodbye!", Colors.YELLOW)
    
    except Exception as e:
        logger.error(f"Fatal startup error: {e}")
        print_colored(f"\n💥 Fatal error during startup: {e}", Colors.RED)
        
    finally:
        print_colored("\n🎯 Ultimate Option Buyer Scanner v4.3 - Session Ended", Colors.HEADER)
        print_colored("📊 Thank you for using the ultimate professional trading scanner!", Colors.GREEN)
        print_colored("="*120, Colors.HEADER)

# ========== END OF ULTIMATE COMPLETE SCANNER CODE v4.3 WITH ALL FIXES ==========

"""
🎯 ULTIMATE ENHANCED OPTION BUYER SCANNER v4.3 - ALL DATETIME ISSUES FIXED:

✅ FIXED: All datetime comparison errors resolved
✅ FIXED: Market hours parsing with proper time objects  
✅ FIXED: Timezone-aware datetime filtering for historical data
✅ FIXED: EMA calculation with NaN value handling
✅ FIXED: TrueData API timezone-naive datetime requirements
✅ FIXED: Live mode continuous scanning without crashes
✅ CONTINUOUS 5-MINUTE SCANNING - Runs automatically every 5 minutes during market hours
✅ CORRECTED SYMBOL HANDLING - TrueData uses -I suffix, Localhost uses clean symbols
✅ MARKET HOURS AWARENESS - Automatically starts/stops based on market schedule (9:15-15:30 IST)
✅ WEEKEND/HOLIDAY HANDLING - Waits until market opens with progress indicators
✅ REAL-TIME REGIME DETECTION - Updates market regime each scan
✅ NEW STOCK ALERTS - Highlights newly detected opportunities
✅ PROGRESSIVE COUNTDOWN - Shows time until next scan with human-readable format
✅ ERROR RECOVERY - Continues scanning even if one scan fails
✅ GRACEFUL SHUTDOWN - Proper cleanup on market close or interruption
✅ COMPLETE INTEGRATION - TrueData OHLC/Volume/OI + Localhost Option Chain API  
✅ PERFECT OPTION DETECTION - PCR, CE/PE volumes, OI changes, liquidity scoring
✅ ULTIMATE QUALITY SCORING - Combined technical + option chain quality assessment
✅ ENHANCED SIGNAL CLASSIFICATION - Perfect Call/Put setups with confidence levels

MAJOR DATETIME FIXES:
  🔧 Fixed: "descriptor 'time' for 'datetime.datetime' objects doesn't apply to a 'int' object"
  🔧 Fixed: "'<' not supported between instances of 'NoneType' and 'datetime.datetime'"
  🔧 Fixed: Market hours checking with proper time.time() vs datetime.time() usage
  🔧 Fixed: Timezone-aware vs naive datetime comparisons
  🔧 Fixed: EMA calculation with proper NaN/None value handling

SYMBOL CONVERSION LOGIC:
  📊 Input stocks can be: RELIANCE, TCS-EQ, HDFC-I
  🔗 TrueData API uses: RELIANCE-I, TCS-I, HDFC-I
  🔗 Localhost API uses: RELIANCE, TCS, HDFC
  
ULTIMATE USAGE:
  🎯 Continuous Live: python ultimate_scanner.py (runs every 5 minutes!)
  🔍 Ultimate Snapshot: python ultimate_scanner.py --asof 2025-10-03T14:25  
  📈 Complete Backtest: python ultimate_scanner.py --backtest 2025-10-03
  🔬 Quick Test: python ultimate_scanner.py --test
  🔗 Option Chain Only: python ultimate_scanner.py --option-only
  🔍 Diagnostics: python ultimate_scanner.py --asof 2025-10-03T15:30 --diagnose
  📉 Lower Threshold: python ultimate_scanner.py --asof 2025-10-03T15:30 --threshold 1.0

ALL ISSUES FIXED ✅ PROFESSIONAL TRADING READY ✅ CONTINUOUS SCANNING ✅
"""

