import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import requests
import json
import time as timemodule
import pytz
import logging
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from truedata.history import TD_hist
import warnings
warnings.filterwarnings("ignore")
from rich.console import Console
from rich.table import Table
import argparse

logger = logging.getLogger(__name__)

# =========================
# --- TRUE DATA CONFIG ---
# =========================
TDUSERNAME = os.getenv("TD_USERNAME", "tdwsp751")
TDPASSWORD = os.getenv("TD_PASSWORD", "raj@751")
tdhist = TD_hist(TDUSERNAME, TDPASSWORD, log_level=logging.WARNING)

# =========================
# --- COLOR CODES ---
# =========================
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

# =========================
# --- ENHANCED INDICATOR WEIGHTS FOR OPTION BUYERS ---
# =========================
ENHANCED_INDICATOR_WEIGHTS = {
    # Tier 1: Critical for Option Buyers (Highest Priority)
    "VolumeOIFlow": 2.5,      # NEW: Volume-OI divergence analysis
    "InstitutionalFlow": 2.3,  # NEW: Large volume + OI changes
    "VolumeSurge": 2.2,       # Enhanced volume surge detection
    "OIChangeRate": 2.1,      # NEW: Open Interest momentum
    "VolumeBreakout": 2.0,    # NEW: Volume confirmation on breakouts
    
    # Tier 2: Strong momentum indicators
    "Momentum": 1.9,          # Enhanced with volume-price momentum
    "ADX": 1.8,               # Trend strength
    "VWAP": 1.7,              # Volume-weighted levels
    "EMA": 1.7,               # Price momentum
    
    # Tier 3: Supporting confirmation
    "MACD": 1.5,
    "OBV": 1.5,               # Enhanced with OI correlation
    "ATR": 1.4,               # Volatility for option premium
    "VolumeProfile": 1.3,     # NEW: Volume distribution analysis
    
    # Tier 4: Traditional indicators
    "Bollinger": 1.2,
    "RSI": 1.1,
    "ROC": 1.0,
    "Stochastic": 1.0,
    "CCI": 1.0,
    "MA": 1.0,
    "WWL": 1.0,
}

# =========================
# --- TIMEFRAME WEIGHTS ---
# =========================
TIMEFRAME_WEIGHTS = {
    15: 3.0,    # Primary for option trading
    5: 2.8,     # Entry/exit precision
    30: 2.2,    # Trend confirmation
    60: 1.8,    # Medium-term view
    "daily": 1.5,  # Overall context
}

# =========================
# --- NSE INDEX TO SECTOR MAPPING ---
# =========================
NSE_INDEX_TO_SECTOR = {
    "NIFTY IT": "Technology",
    "NIFTY PHARMA": "Pharma",
    "NIFTY FMCG": "Consumer",
    "NIFTY BANK": "Banking",
    "NIFTY AUTO": "Auto",
    "NIFTY METAL": "Metal",
    "NIFTY ENERGY": "Energy",
    "NIFTY REALTY": "Realty",
    "NIFTY INFRA": "Infrastructure",
    "NIFTY PSU BANK": "PSU Bank",
    "NIFTY PSE": "PSE",
    "NIFTY COMMODITIES": "Commodities",
    "NIFTY MNC": "Finance",
    "NIFTY FINANCIAL SERVICES": "Finance",
    "NIFTY INFRASTRUCTURE": "Infrastructure",
    "BANKNIFTY": "Banking",
    "NIFTYAUTO": "Auto",
    "NIFTYIT": "Technology",
    "NIFTYPHARMA": "Pharma",
    "NIFTY CONSUMER DURABLES": "Consumer Durables",
    "NIFTY HEALTHCARE INDEX": "Healthcare",
    "NIFTY CAPITAL MARKETS": "Capital Market",
    "NIFTY PRIVATE BANK": "Private Bank",
    "NIFTY OIL & GAS": "Oil and Gas",
    "NIFTY INDIA DEFENCE": "Defence",
    "NIFTY CORE HOUSING": "Core Housing",
    "NIFTY SERVICES SECTOR": "Services Sector",
    "NIFTY FINANCIAL SERVICES 25/50": "Financial Services 2550",
    "NIFTY INDIA TOURISM": "Tourism",
}

# Sector to stocks mapping (same as before)
SECTOR_STOCKS = {
    "Technology": ["TCS-I", "INFY-I", "HCLTECH-I", "WIPRO-I", "TECHM-I", "LTIM-I", "MPHASIS-I", "COFORGE-I", "PERSISTENT-I", "CYIENT-I", "KPITTECH-I", "TATAELXSI-I", "SONACOMS-I", "KAYNES-I", "OFSS-I"],
    "Auto": ["MARUTI-I", "TATAMOTORS-I", "M&M-I", "BAJAJ-AUTO-I", "HEROMOTOCO-I", "TVSMOTOR-I", "BHARATFORG-I", "EICHERMOT-I", "ASHOKLEY-I", "BOSCHLTD-I", "TIINDIA-I", "MOTHERSON-I"],
    "Banking": ["HDFCBANK-I", "ICICIBANK-I", "SBIN-I", "KOTAKBANK-I", "AXISBANK-I", "PNB-I", "BANKBARODA-I", "CANBK-I", "IDFCFIRSTB-I", "INDUSINDBK-I", "AUBANK-I", "FEDERALBNK-I"],
    "Pharma": ["SUNPHARMA-I", "DRREDDY-I", "CIPLA-I", "LUPIN-I", "AUROPHARMA-I", "TORNTPHARM-I", "GLENMARK-I", "ALKEM-I", "LAURUSLABS-I", "BIOCON-I", "ZYDUSLIFE-I", "MANKIND-I", "SYNGENE-I", "PPLPHARMA-I"],
    "Energy": ["RELIANCE-I", "NTPC-I", "BPCL-I", "IOC-I", "ONGC-I", "GAIL-I", "HINDPETRO-I", "ADANIGREEN-I", "ADANIENSOL-I", "JSWENERGY-I", "COALINDIA-I", "TATAPOWER-I", "SUZLON-I", "PETRONET-I", "OIL-I", "POWERGRID-I", "NHPC-I", "ADANIPORTS-I", "ABB-I", "SIEMENS-I", "CGPOWER-I", "INOXWIND-I"],
    "Metal": ["TATASTEEL-I", "JSWSTEEL-I", "SAIL-I", "JINDALSTEL-I", "HINDALCO-I", "NMDC-I"],
    "Consumer": ["HINDUNILVR-I", "ITC-I", "NESTLEIND-I", "BRITANNIA-I", "TATACONSUM-I", "DABUR-I", "AMBER-I", "UNITDSPR-I", "GODREJCP-I", "MARICO-I", "COLPAL-I", "UPL-I", "VBL-I"],
    "PSU Bank": ["SBIN-I", "PNB-I", "BANKBARODA-I", "CANBK-I", "UNIONBANK-I", "BANKINDIA-I"],
    "Finance": ["BAJFINANCE-I", "SHRIRAMFIN-I", "CHOLAFIN-I", "HDFCLIFE-I", "ICICIPRULI-I", "ETERNAL-I"],
    "Realty": ["DLF-I", "LODHA-I", "PRESTIGE-I", "GODREJPROP-I", "OBEROIRLTY-I", "PHOENIXLTD-I", "NCC-I", "NBCC-I"],
    "PSE": ["BEL-I", "BHEL-I", "NHPC-I", "GAIL-I", "IOC-I", "NTPC-I", "POWERGRID-I", "HINDPETRO-I", "OIL-I", "RECLTD-I", "ONGC-I", "NMDC-I", "BPCL-I", "HAL-I", "RVNL-I", "PFC-I", "COALINDIA-I", "IRCTC-I", "IRFC-I"],
    "Commodities": ["AMBUJACEM-I", "APLAPOLLO-I", "ULTRACEMCO-I", "SHREECEM-I", "JSWSTEEL-I", "HINDALCO-I", "NHPC-I", "IOC-I", "NTPC-I", "HINDPETRO-I", "ADANIGREEN-I", "OIL-I", "VEDL-I", "PIIND-I", "ONGC-I", "NMDC-I", "UPL-I", "BPCL-I", "JSWENERGY-I", "GRASIM-I", "RELIANCE-I", "TORNTPOWER-I", "TATAPOWER-I", "COALINDIA-I", "PIDILITIND-I", "SRF-I", "ADANIENSOL-I", "JINDALSTEL-I", "TATASTEEL-I", "HINDALCO-I"],
    "Consumer Durables": ["TITAN-I", "DIXON-I", "HAVELLS-I", "CROMPTON-I", "POLYCAB-I", "EXIDEIND-I", "AMBER-I", "KAYNES-I", "VOLTAS-I", "PGEL-I", "BLUESTARCO-I"],
    "Healthcare": ["SUNPHARMA-I", "DIVISLAB-I", "CIPLA-I", "TORNTPHARM-I", "MAXHEALTH-I", "APOLLOHOSP-I", "DRREDDY-I", "MANKIND-I", "ZYDUSLIFE-I", "LUPIN-I", "FORTIS-I", "ALKEM-I", "AUROPHARMA-I", "GLENMARK-I", "BIOCON-I", "LAURUSLABS-I", "SYNGENE-I", "GRANULES-I"],
    "Capital Market": ["HDFCAMC-I", "BSE-I", "360ONE-I", "MCX-I", "CDSL-I", "NUVAMA-I", "ANGELONE-I", "KFINTECH-I", "CAMS-I", "IEX-I"],
    "Private Bank": ["HDFCBANK-I", "ICICIBANK-I", "KOTAKBANK-I", "AXISBANK-I", "YESBANK-I", "IDFCFIRSTB-I", "INDUSINDBK-I", "FEDERALBNK-I", "BANDHANBNK-I", "RBLBANK-I"],
    "Oil and Gas": ["RELIANCE-I", "ONGC-I", "IOC-I", "BPCL-I", "GAIL-I", "HINDPETRO-I", "OIL-I", "PETRONET-I", "IGL-I"],
    "Defence": ["HAL-I", "BEL-I", "SOLARINDS-I", "MAZDOCK-I", "BDL-I"],
    "Core Housing": ["ULTRACEMCO-I", "ASIANPAINT-I", "GRASIM-I", "DLF-I", "AMBUJACEM-I", "LODHA-I", "DIXON-I", "POLYCAB-I", "SHREECEM-I", "HAVELLS-I", "PRESTIGE-I", "GODREJPROP-I", "OBEROIRLTY-I", "PHOENIXLTD-I", "VOLTAS-I", "DALBHARAT-I", "KEI-I", "BLUESTARCO-I", "LICHSGFIN-I", "PNBHOUSING-I", "CROMPTON-I"],
    "Services Sector": ["HDFCBANK-I", "BHARTIARTL-I", "TCS-I", "ICICIBANK-I", "SBIN-I", "INFY-I", "BAJFINANCE-I", "HCLTECH-I", "KOTAKBANK-I", "AXISBANK-I", "BAJAJFINSV-I", "NTPC-I", "ZOMATO-I", "ADANIPORTS-I", "DMART-I", "POWERGRID-I", "WIPRO-I", "INDIGO-I", "JIOFINSERV-I", "SBILIFE-I", "HDFCLIFE-I", "LTIM-I", "TECHM-I", "TATAPOWER-I", "SHRIRAMFIN-I", "GAIL-I", "MAXHEALTH-I", "APOLLOHOSP-I", "NAUKRI-I", "INDUSINDBK-I"],
    "Financial Services 2550": ["HDFCBANK-I", "ICICIBANK-I", "SBIN-I", "BAJFINANCE-I", "KOTAKBANK-I", "AXISBANK-I", "BAJAJFINSV-I", "JIOFIN-I", "SBILIFE-I", "HDFCLIFE-I", "PFC-I", "CHOLAFIN-I", "HDFCAMC-I", "SHRIRAMFIN-I", "MUTHOOTFIN-I", "RECLTD-I", "ICICIGI-I", "ICICIPRULI-I", "SBICARD-I", "LICHSGFIN-I"],
    "Tourism": ["INDIGO-I", "INDHOTEL-I", "IRCTC-I", "JUBLFOOD-I"]
}

# =========================
# --- ENHANCED TECHNICAL INDICATORS FOR OPTION BUYERS ---
# =========================
class EnhancedOptionBuyerIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        """Calculate all indicators including new volume and open interest based ones"""
        indicators = {}
        if df is None or len(df) < 20:
            return indicators
        
        try:
            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            vol = df["Volume"]
            
            # Get Open Interest if available
            oi = df.get("OpenInterest", pd.Series([0] * len(df), index=df.index))
            
            # ===== EXISTING INDICATORS (Enhanced) =====
            
            # 1. RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            indicators["RSI"] = 100 - (100 / (1 + rs))

            # 2. MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            indicators["MACD"] = macd_line - signal_line

            # 3. Stochastic
            low14 = low.rolling(window=14).min()
            high14 = high.rolling(window=14).max()
            indicators["Stochastic"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)

            # 4. MA 20
            indicators["MA"] = close.rolling(window=20).mean()

            # 5. EMA 21
            indicators["EMA"] = close.ewm(span=21).mean()

            # 6. ADX
            high_diff = high.diff()
            low_diff = low.diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
            minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff > 0), 0.0)
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr.replace(0, np.nan))
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr.replace(0, np.nan))
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
            indicators["ADX"] = dx.rolling(window=14).mean()

            # 7. Bollinger position
            ma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20
            indicators["Bollinger"] = (close - ma20) / (upper - lower).replace(0, np.nan) * 100

            # 8. ROC
            indicators["ROC"] = close.pct_change(periods=12) * 100

            # 9. Enhanced OBV with OI correlation
            obv = np.sign(close.diff().fillna(0)) * vol.fillna(0)
            obv = obv.cumsum()
            indicators["OBV"] = obv.pct_change(periods=10) * 100

            # 10. CCI
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(window=20).mean()
            mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
            indicators["CCI"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

            # 11. Williams %R
            hh = high.rolling(window=14).max()
            ll = low.rolling(window=14).min()
            indicators["WWL"] = (hh - close) / (hh - ll).replace(0, np.nan) * -100

            # 12. VWAP
            if len(df) >= 20:
                tpv = (high + low + close) / 3
                vwap_num = (tpv * vol).rolling(window=20).sum()
                vwap_den = vol.rolling(window=20).sum().replace(0, np.nan)
                indicators["VWAP"] = vwap_num / vwap_den

            # 13. ATR
            indicators["ATR"] = atr

            # ===== NEW INDICATORS FOR OPTION BUYERS =====
            
            # 14. Enhanced Volume Surge (considers historical volatility)
            if len(df) >= 20:
                avg_vol_20 = vol.rolling(window=20).mean()
                vol_std = vol.rolling(window=20).std()
                current_vol = vol
                
                # Z-score based volume surge
                vol_zscore = (current_vol - avg_vol_20) / vol_std.replace(0, np.nan)
                
                # Convert to 0-100 scale with enhanced sensitivity
                indicators["VolumeSurge"] = np.clip(50 + vol_zscore * 15, 0, 100)

            # 15. Open Interest Change Rate
            if oi.sum() > 0:  # Only if OI data is available
                oi_change = oi.pct_change(periods=1) * 100
                oi_momentum = oi.pct_change(periods=5) * 100
                
                # Combine short-term and medium-term OI changes
                indicators["OIChangeRate"] = np.clip(50 + (oi_change * 0.3 + oi_momentum * 0.7) * 2, 0, 100)
            else:
                # Create a pandas Series with default values
                indicators["OIChangeRate"] = pd.Series([50] * len(df), index=df.index)

            # 16. Volume-OI Flow Analysis
            if oi.sum() > 0:
                # Volume to OI ratio trend
                vol_oi_ratio = vol / (oi + 1)  # +1 to avoid division by zero
                vol_oi_trend = vol_oi_ratio.rolling(window=10).mean()
                vol_oi_current = vol_oi_ratio
                
                # Divergence score
                vol_trend = vol.rolling(window=10).mean()
                oi_trend = oi.rolling(window=10).mean()
                
                vol_direction = np.where(vol > vol_trend, 1, -1)
                oi_direction = np.where(oi > oi_trend, 1, -1)
                
                # Flow convergence/divergence
                flow_score = (vol_direction + oi_direction) / 2 * 50 + 50
                indicators["VolumeOIFlow"] = pd.Series(flow_score, index=df.index)
            else:
                indicators["VolumeOIFlow"] = pd.Series([50] * len(df), index=df.index)

            # 17. Institutional Activity Score
            if len(df) >= 20:
                # Large volume + price movement correlation
                price_change = close.pct_change() * 100
                vol_percentile = vol.rolling(window=20).rank(pct=True) * 100
                
                # High volume with significant price moves indicates institutional activity
                institutional_score = np.where(
                    (vol_percentile > 80) & (abs(price_change) > 1.5),
                    75 + (vol_percentile - 80) * 1.25,  # Boost for high volume + price moves
                    50 + (vol_percentile - 50) * 0.3    # Normal volume activity
                )
                indicators["InstitutionalFlow"] = pd.Series(np.clip(institutional_score, 0, 100), index=df.index)

            # 18. Volume Profile Analysis
            if len(df) >= 20:
                # Volume at different price levels
                current_price_level = close.iloc[-1]
                recent_high = high.rolling(window=10).max().iloc[-1]
                recent_low = low.rolling(window=10).min().iloc[-1]
                
                # Position in recent range
                if recent_high > recent_low:
                    price_position = (current_price_level - recent_low) / (recent_high - recent_low)
                    volume_profile_score = 50 + (price_position - 0.5) * 100
                else:
                    volume_profile_score = 50
                    
                indicators["VolumeProfile"] = pd.Series([np.clip(volume_profile_score, 0, 100)] * len(df), 
                                                      index=df.index)

            # 19. Volume Breakout Confirmation
            if len(df) >= 20:
                # Price breakout with volume confirmation
                price_ma = close.rolling(window=20).mean()
                vol_ma = vol.rolling(window=20).mean()
                
                price_breakout = (close - price_ma) / price_ma * 100
                volume_confirmation = vol / vol_ma
                
                # Breakout score: price movement confirmed by volume
                breakout_score = np.where(
                    abs(price_breakout) > 2,  # Significant price move
                    50 + price_breakout * volume_confirmation * 5,
                    50 + price_breakout * 10
                )
                indicators["VolumeBreakout"] = pd.Series(np.clip(breakout_score, 0, 100), index=df.index)

            # 20. Enhanced Momentum (Price + Volume + OI)
            if len(df) >= 10:
                price_mom = close.pct_change(periods=10) * 100
                vol_mom = (vol / vol.rolling(window=10).mean() - 1) * 100
                
                if oi.sum() > 0:
                    oi_mom = (oi / oi.rolling(window=10).mean() - 1) * 100
                    # Triple momentum: price, volume, and OI
                    combined_momentum = price_mom * 0.5 + vol_mom * 0.3 + oi_mom * 0.2
                else:
                    # Dual momentum: price and volume
                    combined_momentum = price_mom * 0.7 + vol_mom * 0.3
                
                indicators["Momentum"] = pd.Series(np.clip(50 + combined_momentum * 1.2, 0, 100), index=df.index)

            return indicators
        
        except Exception as e:
            logger.error(f"Error calculating enhanced option buyer indicators: {e}")
            return indicators

# =========================
# --- NORMALIZATION HELPERS ---
# =========================
def normalize_indicator_value(indicator_name, value):
    """Enhanced normalization for new indicators"""
    try:
        if indicator_name == "RSI":
            return max(0, min(100, value))
        elif indicator_name == "MACD":
            return 50 + max(-25, min(25, value / 10))
        elif indicator_name == "Stochastic":
            return max(0, min(100, value))
        elif indicator_name in ("MA", "EMA", "VWAP"):
            return 50
        elif indicator_name == "ADX":
            return max(0, min(100, value))
        elif indicator_name == "Bollinger":
            return max(0, min(100, (value + 100) / 2))
        elif indicator_name == "ROC":
            return 50 + max(-25, min(25, value / 2))
        elif indicator_name == "OBV":
            return 50 + max(-25, min(25, value))
        elif indicator_name == "CCI":
            return max(0, min(100, (value + 200) / 4))
        elif indicator_name == "WWL":
            return max(0, min(100, (value + 100)))
        elif indicator_name == "ATR":
            return 50
        elif indicator_name in ("VolumeSurge", "OIChangeRate", "VolumeOIFlow", 
                               "InstitutionalFlow", "VolumeProfile", "VolumeBreakout", "Momentum"):
            return max(0, min(100, value))
        else:
            return 50
    except Exception:
        return 50

# =========================
# --- ENHANCED SCANNER CLASS ---
# =========================
class EnhancedOptionBuyerScanner:
    def __init__(self, mode='live', backtest_date=None):
        self.is_running = False
        self.current_signals = []
        self.best_sectors = ["Pharma", "Healthcare", "Technology", "Financial Services 2550"]
        self.worst_sectors = ["Defence", "Energy", "PSU Bank", "Realty"]
        self.sectoral_history = []
        self.last_sectoral_update = None
        self.api_errors = []
        self.sector_update_attempts = 0
        self.successful_updates = 0
        self.gapdown_filtered_count = 0
        self.mode = mode
        self.backtest_date = backtest_date

        # Score tracking for deltas
        self.last_cycle_scores = {}
        self.current_cycle_scores = {}

        # Market hours
        self.market_start = time(9, 15)
        self.market_end = time(15, 30)
        self.scan_interval = 300  # 5 minutes

        logger.info("Enhanced Option Buyer Scanner with Volume+OI Analysis initialized")

    def show_initialization_status(self):
        print(f"{Colors.CYAN}{Colors.BOLD}ENHANCED OPTION BUYER SCANNER WITH VOLUME & OPEN INTEREST{Colors.RESET}")
        print(f"{Colors.CYAN}{'-'*80}{Colors.RESET}")
        print(f"Mode: {Colors.YELLOW}{self.mode.upper()}{Colors.RESET}")
        if self.mode == 'backtest' and self.backtest_date:
            print(f"Backtest Date: {Colors.YELLOW}{self.backtest_date.strftime('%Y-%m-%d')}{Colors.RESET}")
        print(f"Timeframes: {Colors.YELLOW}5min, 15min, 30min, 60min, Daily{Colors.RESET}")
        print(f"Strategy: {Colors.GREEN}Volume+OI Flow Analysis{Colors.RESET} for {Colors.BLUE}Option Buyers{Colors.RESET}")
        print(f"New Indicators: {Colors.MAGENTA}Volume-OI Flow, Institutional Activity, OI Change Rate{Colors.RESET}")
        print(f"{Colors.YELLOW}OPTION BUYER FOCUSED WEIGHTS{Colors.RESET}")
        
        key_indicators = ["VolumeOIFlow", "InstitutionalFlow", "VolumeSurge", "OIChangeRate", "VolumeBreakout"]
        for indicator in key_indicators:
            if indicator in ENHANCED_INDICATOR_WEIGHTS:
                weight = ENHANCED_INDICATOR_WEIGHTS[indicator]
                print(f" - {Colors.GREEN}{indicator}: {weight}{Colors.RESET}")
        
        self.show_sector_status()
        if self.mode == 'live':
            self.test_api_connection()
        else:
            print(f"{Colors.YELLOW}Backtest mode: Skipping API connection test{Colors.RESET}")
        print(f"{Colors.YELLOW}Running initial sector update...{Colors.RESET}")
        self.force_sector_update()
        print(f"{Colors.CYAN}{'-'*80}{Colors.RESET}")

    def test_api_connection(self):
        print(f"{Colors.BLUE}API CONNECTION TEST{Colors.RESET}")
        try:
            response = requests.get("http://localhost:3000/api/allIndices", timeout=10)
            if response.status_code == 200:
                print(f"API Connection {Colors.GREEN}SUCCESS{Colors.RESET}")
                data = response.json()
                if isinstance(data, list):
                    print(f"Items Count: {len(data)}")
                elif isinstance(data, dict):
                    print(f"Dict Keys: {list(data.keys())}")
            else:
                print(f"API Connection {Colors.RED}FAILED{Colors.RESET} - Status {response.status_code}")
        except Exception as e:
            print(f"API Connection {Colors.RED}ERROR{Colors.RESET} - {str(e)}")

    def show_sector_status(self):
        print(f"{Colors.MAGENTA}CURRENT SECTOR STATUS{Colors.RESET}")
        print(f"Top 4 Best Sectors: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"Top 4 Worst Sectors: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
        print(f"Last Update: {self.last_sectoral_update or 'Never'}")

    def fetch_live_sectoral_performance(self):
        try:
            current_time = self.backtest_date if self.mode == 'backtest' and self.backtest_date else datetime.now()
            if self.mode == 'backtest':
                logger.info("Generating random sector performance for backtest...")
                sectoral_performance = []
                for index_name, sector in NSE_INDEX_TO_SECTOR.items():
                    change_percent = np.random.uniform(-5, 5)
                    sectoral_performance.append({
                        "index": index_name,
                        "sector": sector,
                        "changepercent": change_percent,
                        "timestamp": current_time,
                    })
            else:
                logger.info("Fetching live sector performance from API...")
                response = requests.get("http://localhost:3000/api/allIndices", timeout=10)
                
                if response.status_code != 200:
                    return False

                indices_data = response.json()
                if isinstance(indices_data, str):
                    indices_data = json.loads(indices_data)

                # Handle the dict with 'data' key structure you showed
                if isinstance(indices_data, dict):
                    if "data" in indices_data:
                        indices_data = indices_data["data"]
                    elif "indices" in indices_data:
                        indices_data = indices_data["indices"]
                    elif "results" in indices_data:
                        indices_data = indices_data["results"]

                if not isinstance(indices_data, list):
                    return False

                sectoral_performance = []
                
                for index in indices_data:
                    if not isinstance(index, dict):
                        continue
                        
                    index_name = next((str(index[field]).strip().upper()
                                     for field in ("name", "symbol", "index", "indexName")
                                     if field in index and index[field]), None)
                    
                    if index_name and index_name in NSE_INDEX_TO_SECTOR:
                        change_percent = 0.0
                        for field in ("changepercent", "changePercent", "pChange", "percentChange", "change", "pchg"):
                            if field in index and index[field] is not None:
                                try:
                                    change_percent = float(index[field])
                                    break
                                except (ValueError, TypeError):
                                    continue
                        
                        sectoral_performance.append({
                            "index": index_name,
                            "sector": NSE_INDEX_TO_SECTOR[index_name],
                            "changepercent": change_percent,
                            "timestamp": current_time,
                        })

            if not sectoral_performance:
                return False

            sectoral_performance.sort(key=lambda x: x["changepercent"], reverse=True)
            
            n = len(sectoral_performance)
            best_count = min(4, n)
            worst_count = min(4, n)
            
            self.best_sectors = [sectoral_performance[i]["sector"] for i in range(best_count)]
            self.worst_sectors = [sectoral_performance[-i]["sector"] for i in range(1, worst_count + 1)]
            
            self.last_sectoral_update = current_time
            self.sectoral_history.append({
                "timestamp": current_time,
                "best": self.best_sectors[:],
                "worst": self.worst_sectors[:],
                "fulldata": sectoral_performance[:],
            })
            
            if len(self.sectoral_history) > 20:
                self.sectoral_history = self.sectoral_history[-20:]

            return True
        except Exception as e:
            logger.error(f"Error fetching API sectoral data: {e}")
            self.api_errors.append((datetime.now(), str(e)))
            return False

    def force_sector_update(self):
        print(f"{Colors.YELLOW}FORCING SECTOR UPDATE WITH API...{Colors.RESET}")
        self.sector_update_attempts += 1
        success = self.fetch_live_sectoral_performance()
        if success:
            self.successful_updates += 1
            print("API sectoral update successful!")
        else:
            print("API sectoral update failed - using defaults")
        return success

    def is_market_open(self):
        if self.mode == 'backtest':
            return True
        now = datetime.now()
        ct = now.time()
        if now.weekday() >= 5:
            return False
        return self.market_start <= ct <= self.market_end

    def normalize_live_data(self, df, symbol):
        try:
            if df is None or len(df) == 0:
                return None
            
            dfc = df.copy()
            dfc.rename(columns={c: c.lower() for c in dfc.columns}, inplace=True)

            # Map common fields
            col_map = {}
            for src, tgt in (
                ("time", "Date"), ("timestamp", "Date"), ("date", "Date"),
                ("open", "Open"), ("high", "High"), ("low", "Low"),
                ("close", "Close"), ("vol", "Volume"), ("volume", "Volume"),
                ("oi", "OpenInterest"), ("openinterest", "OpenInterest"),
                ("open_interest", "OpenInterest")
            ):
                if src in dfc.columns:
                    col_map[src] = tgt
            dfc.rename(columns=col_map, inplace=True)

            # Recover Date from index if needed
            if "Date" not in dfc.columns:
                if isinstance(dfc.index, pd.DatetimeIndex):
                    dfc["Date"] = dfc.index
                else:
                    for cand in ("datetime", "barstarttime", "bartime", "time"):
                        if cand in dfc.columns:
                            dfc.rename(columns={cand: "Date"}, inplace=True)
                            break

            # Require OHLC
            required = ["Open", "High", "Low", "Close"]
            if not all(col in dfc.columns for col in required):
                return None

            if "Volume" not in dfc.columns:
                dfc["Volume"] = 0
                
            if "OpenInterest" not in dfc.columns:
                dfc["OpenInterest"] = 0

            # Parse Date
            if "Date" in dfc.columns:
                dfc["Date"] = pd.to_datetime(dfc["Date"], errors="coerce", utc=False)
            else:
                idx = pd.to_datetime(dfc.index, errors="coerce", utc=False)
                dfc["Date"] = idx

            # Drop bad rows
            dfc = dfc.dropna(subset=["Date", "Open", "High", "Low", "Close"])

            # Ensure numeric
            for col in ["Open", "High", "Low", "Close", "Volume", "OpenInterest"]:
                if col in dfc.columns:
                    dfc[col] = pd.to_numeric(dfc[col], errors="coerce")
            dfc = dfc.dropna(subset=["Open", "High", "Low", "Close"])

            # Remove timezone
            if pd.api.types.is_datetime64tz_dtype(dfc["Date"]):
                dfc["Date"] = dfc["Date"].dt.tz_convert(None)

            # Index and sort
            dfc.set_index("Date", inplace=True, drop=True)
            if not isinstance(dfc.index, pd.DatetimeIndex):
                new_idx = pd.to_datetime(dfc.index, errors="coerce", utc=False)
                dfc = dfc[~new_idx.isna()]
                dfc.index = pd.to_datetime(dfc.index, errors="coerce", utc=False)
            dfc = dfc.sort_index()

            return dfc if len(dfc) >= 20 else None
        except Exception as e:
            logger.error(f"Normalize error {symbol}: {e}")
            return None

    def check_gapdown(self, df):
        try:
            if df is None or len(df) < 2:
                return False
            current_open = df["Open"].iloc[-1]
            previous_close = df["Close"].iloc[-2]
            if pd.isna(current_open) or pd.isna(previous_close) or previous_close == 0:
                return False
            gap_percentage = (current_open - previous_close) / previous_close * 100
            return gap_percentage <= -1.0
        except Exception as e:
            return False

    def fetch_live_data(self, symbol, timeframe):
        try:
            tfmap = {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 mins", "daily": "EOD"}
            bar_size = tfmap.get(timeframe)
            if not bar_size:
                return None, False

            tf_days_map = {5: 10, 15: 10, 30: 20, 60: 60, "daily": 365}
            days = tf_days_map.get(timeframe, 10)

            if self.mode == 'backtest':
                if not self.backtest_date:
                    raise ValueError("Backtest date not provided")
                start_time = self.backtest_date - timedelta(days=days)
                end_time = self.backtest_date
                rawdf = tdhist.get_historic_data(symbol, start_time=start_time, end_time=end_time, bar_size=bar_size)
            else:
                if timeframe in (5, 15):
                    duration = "10 D"
                elif timeframe == 30:
                    duration = "20 D"
                elif timeframe == 60:
                    duration = "60 D"
                elif timeframe == "daily":
                    duration = "365 D"
                else:
                    duration = "10 D"
                rawdf = tdhist.get_historic_data(symbol, duration=duration, bar_size=bar_size)

            if rawdf is None or len(rawdf) == 0:
                return None, False

            normalized_df = self.normalize_live_data(rawdf, symbol)
            if normalized_df is None or len(normalized_df) < 20:
                return None, False

            is_gapdown = False
            if timeframe in (5, 15, 30):
                is_gapdown = self.check_gapdown(normalized_df)

            if timeframe == "daily":
                return normalized_df.tail(250), is_gapdown
            elif timeframe == 60:
                return normalized_df.tail(200), is_gapdown
            else:
                return normalized_df.tail(100), is_gapdown
        except Exception as e:
            logger.error(f"Live data fetch error {symbol}@{timeframe}: {e}")
            return None, False

    def calculate_option_buyer_signals(self, symbol, timeframes_data):
        """Enhanced signal calculation for option buyers using volume and OI - FIXED VERSION"""
        try:
            if not timeframes_data:
                return "Neutral", 0

            sector = next((s for s, stocks in SECTOR_STOCKS.items() if symbol in stocks), None)
            if not sector:
                return "Neutral", 0

            total_weighted_score = 0.0
            total_weight = 0.0
            timeframe_scores = {}

            for tf, df in timeframes_data.items():
                if df is None or len(df) < 20:
                    continue

                indicators = EnhancedOptionBuyerIndicators.calculate_all_indicators(df)
                if not indicators:
                    continue

                tf_score = 0.0
                tf_weight = 0.0
                current_price = df["Close"].iloc[-1]

                for name, weight in ENHANCED_INDICATOR_WEIGHTS.items():
                    if name in indicators and indicators[name] is not None:
                        # FIXED: Handle both pandas Series and numpy arrays
                        indicator_data = indicators[name]
                        
                        # Check if data exists and is not empty
                        has_data = False
                        if hasattr(indicator_data, 'empty'):  # pandas Series
                            has_data = not indicator_data.empty
                        elif hasattr(indicator_data, 'size'):  # numpy array
                            has_data = indicator_data.size > 0
                        else:
                            has_data = indicator_data is not None
                            
                        if has_data:
                            # Get latest value safely
                            try:
                                if hasattr(indicator_data, 'iloc'):  # pandas Series
                                    latest_val = indicator_data.iloc[-1]
                                elif hasattr(indicator_data, '__getitem__'):  # numpy array or list
                                    latest_val = indicator_data[-1]
                                else:
                                    latest_val = float(indicator_data)
                            except (IndexError, TypeError, ValueError):
                                continue
                                
                            if pd.isna(latest_val):
                                continue

                            if name in ("MA", "EMA", "VWAP"):
                                base = latest_val
                                if pd.isna(base) or base == 0:
                                    norm_score = 50
                                else:
                                    price_vs = (current_price - base) / base * 100
                                    if price_vs >= 2:
                                        norm_score = 75
                                    elif price_vs >= 0:
                                        norm_score = 60
                                    elif price_vs >= -2:
                                        norm_score = 50
                                    elif price_vs >= -5:
                                        norm_score = 40
                                    else:
                                        norm_score = 25
                            else:
                                norm_score = normalize_indicator_value(name, latest_val)

                            tf_score += norm_score * weight
                            tf_weight += weight

                if tf_weight <= 0:
                    continue

                tf_final_score = tf_score / tf_weight
                tf_multiplier = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                timeframe_scores[tf] = tf_final_score
                total_weighted_score += tf_final_score * tf_multiplier
                total_weight += tf_multiplier

            if total_weight <= 0:
                return "Neutral", 0

            base_score = total_weighted_score / total_weight

            # Multi-timeframe confirmation bonus (enhanced for option trading)
            num_timeframes = len(timeframe_scores)
            if num_timeframes >= 4:
                bullish_count = sum(1 for v in timeframe_scores.values() if v >= 55)
                bearish_count = sum(1 for v in timeframe_scores.values() if v <= 45)
                if bullish_count >= 3:
                    base_score += 12  # Increased bonus for option buyers
                elif bearish_count >= 3:
                    base_score -= 12

            # Enhanced sector boost for option buyers
            sector_boost = 0
            has_longer_tf = ("daily" in timeframes_data) or (60 in timeframes_data)

            if sector in self.best_sectors:
                rank = self.best_sectors.index(sector) + 1
                if has_longer_tf:
                    boost_map = {1: 30, 2: 25, 3: 20, 4: 15}  # Higher boost for options
                else:
                    boost_map = {1: 25, 2: 20, 3: 15, 4: 10}
                sector_boost = boost_map.get(rank, 0)
            elif sector in self.worst_sectors:
                rank = self.worst_sectors.index(sector) + 1
                if has_longer_tf:
                    boost_map = {1: -30, 2: -25, 3: -20, 4: -15}
                else:
                    boost_map = {1: -25, 2: -20, 3: -15, 4: -10}
                sector_boost = boost_map.get(rank, 0)

            base_score += sector_boost

            # Option buyer specific classification (more sensitive)
            if base_score >= 85:
                return "Strong Call Buy", base_score
            elif base_score >= 75:
                return "Call Buy", base_score
            elif base_score >= 60:
                return "Moderate Call", base_score
            elif base_score <= 15:
                return "Strong Put Buy", base_score
            elif base_score <= 25:
                return "Put Buy", base_score
            elif base_score <= 40:
                return "Moderate Put", base_score
            else:
                return "Neutral", base_score

        except Exception as e:
            logger.error(f"Option buyer signal calculation error for {symbol}: {e}")
            return "Neutral", 0

    def enhanced_scan_cycle(self):
        if not self.is_market_open():
            logger.info("Market closed. Next scan in 5 minutes...")
            return

        start_time = timemodule.time()
        current_time = self.backtest_date if self.mode == 'backtest' and self.backtest_date else datetime.now()
        print(f"{Colors.CYAN}Starting ENHANCED OPTION BUYER scan at {current_time.strftime('%H:%M:%S')}{Colors.RESET}")
        print("Analyzing: 5min 15min 30min 60min Daily with Volume & Open Interest")
        print(f"Focus: {Colors.GREEN}Volume-OI Flow{Colors.RESET}, {Colors.BLUE}Institutional Activity{Colors.RESET}, {Colors.MAGENTA}Options Momentum{Colors.RESET}")

        # Update sectors
        if not self.fetch_live_sectoral_performance():
            print("API sectoral update failed, continuing with previous sectors")

        # Build target stocks
        target_stocks_set = set()

        # Best sectors: more stocks for call opportunities
        for i, sector in enumerate(self.best_sectors):
            if sector in SECTOR_STOCKS:
                if i == 0:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:15])  # More stocks for top sector
                elif i == 1:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:12])
                elif i == 2:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:10])
                elif i == 3:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:8])

        # Worst sectors: fewer stocks but important for put opportunities
        for i, sector in enumerate(self.worst_sectors):
            if sector in SECTOR_STOCKS:
                if i == 0:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:10])
                elif i == 1:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:8])
                elif i == 2:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:6])
                elif i == 3:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:5])

        target_stocks = list(target_stocks_set)
        if not target_stocks:
            print("No target stocks found.")
            return

        print(f"Enhanced option buyer scanning {len(target_stocks)} stocks with Volume+OI analysis")
        live_signals = []
        gapdown_filtered = 0

        def process_stock(symbol):
            try:
                timeframes_data = {}
                timeframes_to_fetch = [5, 15, 30, 60, "daily"]
                
                for tf in timeframes_to_fetch:
                    df, is_gapdown = self.fetch_live_data(symbol, tf)
                    if df is not None:
                        timeframes_data[tf] = df
                    timemodule.sleep(0.8)  # Slightly faster for option scanning

                if len(timeframes_data) >= 3:
                    signal, score = self.calculate_option_buyer_signals(symbol, timeframes_data)
                    
                    # More sensitive threshold for option opportunities
                    if abs(score - 50) >= 12:
                        sector = next((s for s, st in SECTOR_STOCKS.items() if symbol in st), "NA")

                        # Extract volume and OI from smallest timeframe
                        small_df = None
                        if 5 in timeframes_data:
                            small_df = timeframes_data[5]
                        elif 15 in timeframes_data:
                            small_df = timeframes_data[15]
                        else:
                            small_df = list(timeframes_data.values())[0]

                        current_vol = small_df["Volume"].iloc[-1] if "Volume" in small_df.columns else 0
                        prev_vol = small_df["Volume"].iloc[-2] if len(small_df) > 1 and "Volume" in small_df.columns else 0
                        vol_change = ((current_vol - prev_vol) / prev_vol * 100) if prev_vol > 0 else 0

                        current_oi = small_df["OpenInterest"].iloc[-1] if "OpenInterest" in small_df.columns else 0
                        prev_oi = small_df["OpenInterest"].iloc[-2] if len(small_df) > 1 and "OpenInterest" in small_df.columns else 0
                        oi_change = ((current_oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0

                        result = {
                            "symbol": symbol,
                            "signal": signal,
                            "score": score,
                            "sector": sector,
                            "timeframes": len(timeframes_data),
                            "timestamp": current_time,
                            "tfdetails": list(timeframes_data.keys()),
                            "current_vol": current_vol,
                            "vol_change": vol_change,
                            "current_oi": current_oi,
                            "oi_change": oi_change,
                        }
                        self.current_cycle_scores[symbol] = score
                        return result, False
                return None, False
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                return None, False

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:  # Slightly more workers
                futures = [executor.submit(process_stock, symbol) for symbol in target_stocks]
                for future in as_completed(futures):
                    result, is_gap = future.result()
                    if is_gap:
                        gapdown_filtered += 1
                    elif result:
                        live_signals.append(result)
                        
            self.gapdown_filtered_count += gapdown_filtered
            scan_time = timemodule.time() - start_time
            logger.info(f"Option buyer scan completed in {scan_time:.2f}s - {len(live_signals)} signals")
            self.display_option_buyer_signals(live_signals, scan_time, gapdown_filtered, current_time)
        except Exception as e:
            logger.error(f"Error in option buyer scan: {e}")

    def display_option_buyer_signals(self, signals, scan_time, gapdown_filtered, current_time):
        console = Console()
        console.print(f"[cyan bold]{'-'*150}[/]")
        console.print(f"ENHANCED OPTION BUYER SCANNER - VOLUME & OPEN INTEREST ANALYSIS - {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
        console.print(f"{'-'*150}")
        console.print(f"Analysis: [yellow]5m[/] [yellow]15m[/] [yellow]30m[/] [cyan]60m[/] [magenta]Daily[/] | [green]Volume+OI Flow[/]")
        
        best_str = ", ".join(self.best_sectors)
        worst_str = ", ".join(self.worst_sectors)
        console.print(f"Call Focus: [green bold]{best_str}[/]")
        console.print(f"Put Focus: [red bold]{worst_str}[/]")
        console.print(f"Scan Time: {scan_time:.2f}s | Filtered: [magenta]{gapdown_filtered}[/]")

        # Separate table for Indices
        if self.sectoral_history:
            current_data = self.sectoral_history[-1]["fulldata"]
            prev_map = {}
            if len(self.sectoral_history) >= 2:
                prev_data = self.sectoral_history[-2]["fulldata"]
                prev_map = {d["index"]: d["changepercent"] for d in prev_data}

            # Top 5 Best
            best_table = Table(title="Top 5 Performing Sectoral Indices", title_style="bold green")
            best_table.add_column("Rank", style="cyan")
            best_table.add_column("Index", style="magenta")
            best_table.add_column("Sector", style="blue")
            best_table.add_column("% Change", justify="right", style="green")
            best_table.add_column("Δ Prev", justify="right", style="yellow")

            current_data_sorted = sorted(current_data, key=lambda x: x["changepercent"], reverse=True)
            for i, d in enumerate(current_data_sorted[:5], 1):
                prev = prev_map.get(d["index"], 0)
                delta = d["changepercent"] - prev
                delta_str = f"{delta:+.2f}"
                if abs(delta) > 0.5:
                    delta_str = f"[bold red]{delta_str}[/]" if delta > 0 else f"[bold green]{delta_str}[/]"
                best_table.add_row(str(i), d["index"], d["sector"], f"{d['changepercent']:.2f}", delta_str)
            console.print(best_table)

            # Top 5 Worst
            worst_table = Table(title="Bottom 5 Performing Sectoral Indices", title_style="bold red")
            worst_table.add_column("Rank", style="cyan")
            worst_table.add_column("Index", style="magenta")
            worst_table.add_column("Sector", style="blue")
            worst_table.add_column("% Change", justify="right", style="red")
            worst_table.add_column("Δ Prev", justify="right", style="yellow")

            current_data_sorted_asc = sorted(current_data, key=lambda x: x["changepercent"])
            for i, d in enumerate(current_data_sorted_asc[:5], 1):
                prev = prev_map.get(d["index"], 0)
                delta = d["changepercent"] - prev
                delta_str = f"{delta:+.2f}"
                if abs(delta) > 0.5:
                    delta_str = f"[bold red]{delta_str}[/]" if delta > 0 else f"[bold green]{delta_str}[/]"
                worst_table.add_row(str(i), d["index"], d["sector"], f"{d['changepercent']:.2f}", delta_str)
            console.print(worst_table)

        if not signals:
            console.print(f"[yellow]No significant option buying opportunities found in this cycle.[/]")
        else:
            # Separate call and put opportunities
            call_signals = [s for s in signals if "Call" in s["signal"]]
            put_signals = [s for s in signals if "Put" in s["signal"]]
            
            call_signals.sort(key=lambda x: x["score"], reverse=True)
            put_signals.sort(key=lambda x: x["score"])

            # Display call opportunities
            call_table = Table(title="🔥 TOP CALL BUYING OPPORTUNITIES (Volume+OI Analysis)", title_style="bold green")
            call_table.add_column("Stock", style="white")
            call_table.add_column("Sector", style="yellow")
            call_table.add_column("Signal", style="green")
            call_table.add_column("Score", justify="right", style="white")
            call_table.add_column("Score Δ", justify="right", style="cyan")
            call_table.add_column("TFs", justify="right", style="cyan")
            call_table.add_column("Strength", style="green")
            call_table.add_column("Curr Vol", justify="right", style="blue")
            call_table.add_column("Vol %Δ", justify="right", style="blue")
            call_table.add_column("Curr OI", justify="right", style="magenta")
            call_table.add_column("OI %Δ", justify="right", style="magenta")

            for s in call_signals[:15]:
                sector_name = s["sector"]
                sector_color = "yellow"
                sector_display = sector_name
                if sector_name in self.best_sectors:
                    rank = self.best_sectors.index(sector_name) + 1
                    stars = "🚀" * rank
                    sector_color = "green"
                    sector_display = f"{stars}{sector_name}"

                prev = self.last_cycle_scores.get(s["symbol"])
                delta_display = "n/a" if prev is None else f"{s['score'] - prev:+.1f}"

                strength = "🔥Strong" if s["score"] >= 80 else "📈Moderate" if s["score"] >= 70 else "⚡Light"
                signal_style = "green bold" if "Strong" in s["signal"] else "green"

                vol_str = f"{int(s['current_vol']):,}" if 'current_vol' in s else "0"
                vol_change = s.get('vol_change', 0)
                vol_change_str = f"{vol_change:+.1f}%"
                vol_change_style = "green" if vol_change > 0 else "red" if vol_change < 0 else "white"
                vol_change_str = f"[{vol_change_style}]{vol_change_str}[/]"

                oi_str = f"{int(s['current_oi']):,}" if 'current_oi' in s else "0"
                oi_change = s.get('oi_change', 0)
                oi_change_str = f"{oi_change:+.1f}%"
                oi_change_style = "green" if oi_change > 0 else "red" if oi_change < 0 else "white"
                oi_change_str = f"[{oi_change_style}]{oi_change_str}[/]"

                call_table.add_row(
                    s['symbol'],
                    f"[{sector_color}]{sector_display}[/]",
                    f"[{signal_style}]{s['signal']}[/]",
                    f"{s['score']:.1f}",
                    delta_display,
                    str(s['timeframes']),
                    strength,
                    vol_str,
                    vol_change_str,
                    oi_str,
                    oi_change_str
                )

            console.print(call_table)

            # Display put opportunities
            put_table = Table(title="🔻 TOP PUT BUYING OPPORTUNITIES (Volume+OI Analysis)", title_style="bold red")
            put_table.add_column("Stock", style="white")
            put_table.add_column("Sector", style="yellow")
            put_table.add_column("Signal", style="red")
            put_table.add_column("Score", justify="right", style="white")
            put_table.add_column("Score Δ", justify="right", style="cyan")
            put_table.add_column("TFs", justify="right", style="cyan")
            put_table.add_column("Strength", style="red")
            put_table.add_column("Curr Vol", justify="right", style="blue")
            put_table.add_column("Vol %Δ", justify="right", style="blue")
            put_table.add_column("Curr OI", justify="right", style="magenta")
            put_table.add_column("OI %Δ", justify="right", style="magenta")

            for s in put_signals[:15]:
                sector_name = s["sector"]
                sector_color = "yellow"
                sector_display = sector_name
                if sector_name in self.worst_sectors:
                    rank = self.worst_sectors.index(sector_name) + 1
                    stars = "📉" * rank
                    sector_color = "red"
                    sector_display = f"{stars}{sector_name}"

                prev = self.last_cycle_scores.get(s["symbol"])
                delta_display = "n/a" if prev is None else f"{s['score'] - prev:+.1f}"

                strength = "🔥Strong" if s["score"] <= 20 else "📉Moderate" if s["score"] <= 30 else "⚡Light"
                signal_style = "red bold" if "Strong" in s["signal"] else "red"

                vol_str = f"{int(s['current_vol']):,}" if 'current_vol' in s else "0"
                vol_change = s.get('vol_change', 0)
                vol_change_str = f"{vol_change:+.1f}%"
                vol_change_style = "green" if vol_change > 0 else "red" if vol_change < 0 else "white"
                vol_change_str = f"[{vol_change_style}]{vol_change_str}[/]"

                oi_str = f"{int(s['current_oi']):,}" if 'current_oi' in s else "0"
                oi_change = s.get('oi_change', 0)
                oi_change_str = f"{oi_change:+.1f}%"
                oi_change_style = "green" if oi_change > 0 else "red" if oi_change < 0 else "white"
                oi_change_str = f"[{oi_change_style}]{oi_change_str}[/]"

                put_table.add_row(
                    s['symbol'],
                    f"[{sector_color}]{sector_display}[/]",
                    f"[{signal_style}]{s['signal']}[/]",
                    f"{s['score']:.1f}",
                    delta_display,
                    str(s['timeframes']),
                    strength,
                    vol_str,
                    vol_change_str,
                    oi_str,
                    oi_change_str
                )

            console.print(put_table)

        next_scan_time = (current_time + timedelta(minutes=5)).strftime("%H:%M:%S")
        console.print(f"[cyan bold]Next option scan at {next_scan_time}[/]")
        console.print(f"[blue]🎯 Enhanced for Option Buyers: Volume-OI Flow + Institutional Activity + Options Momentum[/]")

        # Rotate score maps
        self.last_cycle_scores = self.current_cycle_scores
        self.current_cycle_scores = {}

    def run_enhanced_scanner(self):
        self.is_running = True
        logger.info("Starting Enhanced Option Buyer Scanner...")
        self.show_initialization_status()
        try:
            while self.is_running:
                self.enhanced_scan_cycle()
                if self.is_running:
                    logger.info("Waiting 5 minutes for next option scan...")
                    timemodule.sleep(self.scan_interval)
        except KeyboardInterrupt:
            logger.info("Option buyer scanner stopped by user")
        finally:
            self.stop()

    def stop(self):
        self.is_running = False
        print(f"{Colors.YELLOW}Enhanced option buyer scanner stopped{Colors.RESET}")

# =========================
# --- MAIN EXECUTION ---
# =========================
def main():
    parser = argparse.ArgumentParser(description="Enhanced Option Buyer Scanner")
    parser.add_argument('--backtest', type=str, help='Run in backtest mode with date YYYY-MM-DD')
    parser.add_argument('--live', action='store_true', help='Run in live mode with real API data')
    args = parser.parse_args()

    if args.backtest and args.live:
        print("Cannot use both --backtest and --live")
        exit(1)

    backtest_date = None
    if args.backtest:
        mode = 'backtest'
        try:
            backtest_date = datetime.strptime(args.backtest, '%Y-%m-%d')
        except ValueError:
            print("Invalid date format. Use YYYY-MM-DD")
            exit(1)
    else:
        mode = 'live'

    print(f"{Colors.CYAN}{Colors.BOLD}ENHANCED OPTION BUYER SCANNER WITH VOLUME & OPEN INTEREST{Colors.RESET}")
    print(f"{Colors.YELLOW}🎯 Specialized for NSE Option Trading{Colors.RESET}")
    print(f"{Colors.GREEN}📊 Volume-OI Flow Analysis | 🏛️ Institutional Activity Detection{Colors.RESET}")
    print(f"{Colors.MAGENTA}🚀 Enhanced Call/Put Opportunity Detection{Colors.RESET}")
    print(f"{Colors.BLUE}📈 Real-time Volume & Open Interest Integration{Colors.RESET}")
    
    scanner = EnhancedOptionBuyerScanner(mode=mode, backtest_date=backtest_date)
    try:
        scanner.run_enhanced_scanner()
    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Shutting down option buyer scanner...{Colors.RESET}")
        scanner.stop()

if __name__ == "__main__":
    main()