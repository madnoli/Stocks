import pandas as pd
import numpy as np
from datetime import datetime, time
import requests
import json
import time as timemodule
import pytz
from logzero import logger
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from truedata.history import TD_hist
import logging
import warnings
warnings.filterwarnings("ignore")

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
    "daily": 1.5,   # Overall context
}

# =========================
# --- NSE INDEX TO SECTOR MAPPING ---
# =========================
NSE_INDEX_TO_SECTOR = {
    "NIFTY 50": "Nifty", # Added for completeness
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

# =========================
# --- SECTOR TO STOCKS MAPPING ---
# =========================
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
        # ... (This class is unchanged as it was correct) ...
        """Calculate all indicators including new volume and open interest based ones"""
        indicators = {}
        if df is None or len(df) < 20:
            return indicators
        
        try:
            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            vol = df["Volume"]
            
            oi = df.get("OpenInterest", pd.Series([0] * len(df), index=df.index))
            
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            indicators["RSI"] = 100 - (100 / (1 + rs))

            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            indicators["MACD"] = macd_line - signal_line

            low14 = low.rolling(window=14).min()
            high14 = high.rolling(window=14).max()
            indicators["Stochastic"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)

            indicators["MA"] = close.rolling(window=20).mean()
            indicators["EMA"] = close.ewm(span=21).mean()

            high_diff, low_diff = high.diff(), low.diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
            minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff > 0), 0.0)
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr.replace(0, np.nan))
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr.replace(0, np.nan))
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
            indicators["ADX"] = dx.rolling(window=14).mean()

            ma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            upper, lower = ma20 + 2 * std20, ma20 - 2 * std20
            indicators["Bollinger"] = (close - ma20) / (upper - lower).replace(0, np.nan) * 100

            indicators["ROC"] = close.pct_change(periods=12) * 100

            obv = (np.sign(close.diff().fillna(0)) * vol.fillna(0)).cumsum()
            indicators["OBV"] = obv.pct_change(periods=10) * 100

            tp = (high + low + close) / 3
            sma_tp = tp.rolling(window=20).mean()
            mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
            indicators["CCI"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

            hh, ll = high.rolling(window=14).max(), low.rolling(window=14).min()
            indicators["WWL"] = (hh - close) / (hh - ll).replace(0, np.nan) * -100

            if len(df) >= 20:
                tpv = (high + low + close) / 3
                vwap_num = (tpv * vol).rolling(window=20).sum()
                vwap_den = vol.rolling(window=20).sum().replace(0, np.nan)
                indicators["VWAP"] = vwap_num / vwap_den

            indicators["ATR"] = atr

            if len(df) >= 20:
                avg_vol_20 = vol.rolling(window=20).mean()
                vol_std = vol.rolling(window=20).std()
                vol_zscore = (vol - avg_vol_20) / vol_std.replace(0, np.nan)
                indicators["VolumeSurge"] = np.clip(50 + vol_zscore * 15, 0, 100)

            if oi.sum() > 0:
                oi_change = oi.pct_change(periods=1) * 100
                oi_momentum = oi.pct_change(periods=5) * 100
                indicators["OIChangeRate"] = np.clip(50 + (oi_change * 0.3 + oi_momentum * 0.7) * 2, 0, 100)
            else:
                indicators["OIChangeRate"] = pd.Series([50] * len(df), index=df.index)

            if oi.sum() > 0:
                vol_trend, oi_trend = vol.rolling(window=10).mean(), oi.rolling(window=10).mean()
                vol_direction, oi_direction = np.where(vol > vol_trend, 1, -1), np.where(oi > oi_trend, 1, -1)
                flow_score = (vol_direction + oi_direction) / 2 * 50 + 50
                indicators["VolumeOIFlow"] = pd.Series(flow_score, index=df.index)
            else:
                indicators["VolumeOIFlow"] = pd.Series([50] * len(df), index=df.index)

            if len(df) >= 20:
                price_change = close.pct_change() * 100
                vol_percentile = vol.rolling(window=20).rank(pct=True) * 100
                institutional_score = np.where((vol_percentile > 80) & (abs(price_change) > 1.5), 75 + (vol_percentile - 80) * 1.25, 50 + (vol_percentile - 50) * 0.3)
                indicators["InstitutionalFlow"] = pd.Series(np.clip(institutional_score, 0, 100), index=df.index)

            if len(df) >= 20:
                recent_high, recent_low = high.rolling(window=10).max().iloc[-1], low.rolling(window=10).min().iloc[-1]
                if recent_high > recent_low:
                    price_position = (close.iloc[-1] - recent_low) / (recent_high - recent_low)
                    volume_profile_score = 50 + (price_position - 0.5) * 100
                else:
                    volume_profile_score = 50
                indicators["VolumeProfile"] = pd.Series([np.clip(volume_profile_score, 0, 100)] * len(df), index=df.index)

            if len(df) >= 20:
                price_ma, vol_ma = close.rolling(window=20).mean(), vol.rolling(window=20).mean()
                price_breakout = (close - price_ma) / price_ma * 100
                volume_confirmation = vol / vol_ma.replace(0, np.nan)
                breakout_score = np.where(abs(price_breakout) > 2, 50 + price_breakout * volume_confirmation * 5, 50 + price_breakout * 10)
                indicators["VolumeBreakout"] = pd.Series(np.clip(breakout_score, 0, 100), index=df.index)

            if len(df) >= 10:
                price_mom = close.pct_change(periods=10) * 100
                vol_mom = (vol / vol.rolling(window=10).mean().replace(0, np.nan) - 1) * 100
                if oi.sum() > 0:
                    oi_mom = (oi / oi.rolling(window=10).mean().replace(0, np.nan) - 1) * 100
                    combined_momentum = price_mom * 0.5 + vol_mom * 0.3 + oi_mom * 0.2
                else:
                    combined_momentum = price_mom * 0.7 + vol_mom * 0.3
                indicators["Momentum"] = pd.Series(np.clip(50 + combined_momentum * 1.2, 0, 100), index=df.index)

            return indicators
        except Exception as e:
            logger.error(f"Error calculating indicators for {df.index.name if df.index.name else 'a stock'}: {e}")
            return indicators


# =========================
# --- NORMALIZATION HELPERS ---
# =========================
def normalize_indicator_value(indicator_name, value):
    # ... (This function is unchanged as it was correct) ...
    try:
        if indicator_name == "RSI": return max(0, min(100, value))
        if indicator_name == "MACD": return 50 + max(-25, min(25, value / 10))
        if indicator_name == "Stochastic": return max(0, min(100, value))
        if indicator_name in ("MA", "EMA", "VWAP"): return 50
        if indicator_name == "ADX": return max(0, min(100, value))
        if indicator_name == "Bollinger": return max(0, min(100, (value + 100) / 2))
        if indicator_name == "ROC": return 50 + max(-25, min(25, value / 2))
        if indicator_name == "OBV": return 50 + max(-25, min(25, value))
        if indicator_name == "CCI": return max(0, min(100, (value + 200) / 4))
        if indicator_name == "WWL": return max(0, min(100, (value + 100)))
        if indicator_name == "ATR": return 50
        if indicator_name in ("VolumeSurge", "OIChangeRate", "VolumeOIFlow", "InstitutionalFlow", "VolumeProfile", "VolumeBreakout", "Momentum"):
            return max(0, min(100, value))
        return 50
    except Exception:
        return 50

# =========================
# --- ENHANCED SCANNER CLASS ---
# =========================
class EnhancedOptionBuyerScanner:
    def __init__(self):
        self.is_running = False
        self.best_sectors = ["Pharma", "Healthcare", "Technology", "Financial Services 2550"]
        self.worst_sectors = ["Defence", "Energy", "PSU Bank", "Realty"]
        self.sectoral_history = []
        self.last_sectoral_update = None
        self.api_errors = []
        self.sector_update_attempts = 0
        self.successful_updates = 0
        self.last_cycle_scores = {}
        self.market_start = time(9, 15)
        self.market_end = time(15, 30)
        self.scan_interval = 300
        logger.info("Enhanced Option Buyer Scanner with Volume+OI Analysis initialized")

    def show_initialization_status(self):
        print(f"{Colors.CYAN}{Colors.BOLD}ENHANCED OPTION BUYER SCANNER WITH VOLUME & OPEN INTEREST{Colors.RESET}")
        print(f"{Colors.CYAN}{'-'*80}{Colors.RESET}")
        print(f"Timeframes: {Colors.YELLOW}5min, 15min, 30min, 60min, Daily{Colors.RESET}")
        print(f"Strategy: {Colors.GREEN}Volume+OI Flow Analysis{Colors.RESET} for {Colors.BLUE}Option Buyers{Colors.RESET}")
        key_indicators = ["VolumeOIFlow", "InstitutionalFlow", "VolumeSurge", "OIChangeRate", "VolumeBreakout"]
        print(f"Key Indicators: {Colors.MAGENTA}{', '.join(key_indicators)}{Colors.RESET}")
        self.show_sector_status()
        self.test_api_connection()
        print(f"{Colors.YELLOW}Running initial sector update...{Colors.RESET}")
        self.force_sector_update()
        print(f"{Colors.CYAN}{'-'*80}{Colors.RESET}")

    def test_api_connection(self):
        print(f"{Colors.BLUE}API CONNECTION TEST{Colors.RESET}")
        try:
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            if response.status_code == 200:
                print(f"API Connection {Colors.GREEN}SUCCESS{Colors.RESET}")
            else:
                print(f"API Connection {Colors.RED}FAILED{Colors.RESET} - Status {response.status_code}")
        except Exception as e:
            print(f"API Connection {Colors.RED}ERROR{Colors.RESET} - {str(e)}")

    def show_sector_status(self):
        print(f"{Colors.MAGENTA}CURRENT SECTOR STATUS{Colors.RESET}")
        print(f"Top 4 Best Sectors: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"Top 4 Worst Sectors: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
        print(f"Last Update: {self.last_sectoral_update or 'Never'}")

    # MODIFIED: Made the parsing logic more robust to match the user's API response.
    def fetch_live_sectoral_performance(self):
        try:
            logger.info("Fetching live sector performance from API...")
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            if response.status_code != 200:
                return False

            indices_data = response.json()
            if isinstance(indices_data, str): indices_data = json.loads(indices_data)
            if isinstance(indices_data, dict) and "data" in indices_data: indices_data = indices_data["data"]
            if not isinstance(indices_data, list): return False

            sectoral_performance = []
            for index in indices_data:
                # FIX: Added "index" and "indexSymbol" to the list of keys to check for the name.
                name_keys = ("index", "indexSymbol", "name", "symbol", "indexName")
                change_keys = ("percentChange", "pChange", "change", "pchg")

                index_name = next((str(index[field]).strip().upper() for field in name_keys if field in index), None)
                
                if index_name and index_name in NSE_INDEX_TO_SECTOR:
                    change_percent = 0.0
                    for field in change_keys:
                        if field in index and index[field] is not None:
                            try:
                                change_percent = float(index[field])
                                break
                            except (ValueError, TypeError): continue
                    
                    sectoral_performance.append({"sector": NSE_INDEX_TO_SECTOR[index_name], "changepercent": change_percent})

            if not sectoral_performance:
                logger.warning("API call successful, but no valid sectors were parsed from the response.")
                return False

            sectoral_performance.sort(key=lambda x: x["changepercent"], reverse=True)
            self.best_sectors = [s["sector"] for s in sectoral_performance[:4]]
            self.worst_sectors = [s["sector"] for s in sectoral_performance[-4:]][::-1]
            self.last_sectoral_update = datetime.now()
            self.sectoral_history.append({"timestamp": self.last_sectoral_update, "fulldata": sectoral_performance})
            if len(self.sectoral_history) > 20: self.sectoral_history.pop(0)
            return True
        except Exception as e:
            logger.error(f"Error fetching/parsing API sectoral data: {e}")
            self.api_errors.append((datetime.now(), str(e)))
            return False


    def force_sector_update(self):
        print(f"{Colors.YELLOW}FORCING SECTOR UPDATE WITH API...{Colors.RESET}")
        self.sector_update_attempts += 1
        if self.fetch_live_sectoral_performance():
            self.successful_updates += 1
            print("API sectoral update successful!")
        else:
            print("API sectoral update failed - using defaults")

    def is_market_open(self):
        now_ist = datetime.now(pytz.timezone('Asia/Kolkata'))
        if now_ist.weekday() >= 5: return False
        return self.market_start <= now_ist.time() <= self.market_end

    def normalize_live_data(self, df, symbol):
        try:
            if df is None or df.empty: return None
            dfc = df.copy()
            dfc.rename(columns={c: c.lower() for c in dfc.columns}, inplace=True)
            col_map = {"time": "Date", "timestamp": "Date", "date": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "vol": "Volume", "volume": "Volume", "oi": "OpenInterest", "openinterest": "OpenInterest"}
            dfc.rename(columns={col: tgt for col, tgt in col_map.items() if col in dfc.columns}, inplace=True)
            if "Date" not in dfc.columns and isinstance(dfc.index, pd.DatetimeIndex): dfc["Date"] = dfc.index
            required = ["Date", "Open", "High", "Low", "Close"]
            if not all(col in dfc.columns for col in required): return None
            for col in ["Volume", "OpenInterest"]:
                if col not in dfc.columns: dfc[col] = 0
            dfc["Date"] = pd.to_datetime(dfc["Date"], errors='coerce')
            dfc.dropna(subset=required, inplace=True)
            for col in required[1:] + ["Volume", "OpenInterest"]:
                dfc[col] = pd.to_numeric(dfc[col], errors='coerce')
            dfc.dropna(subset=required[1:], inplace=True)
            if pd.api.types.is_datetime64tz_dtype(dfc["Date"]):
                dfc["Date"] = dfc["Date"].dt.tz_localize(None)
            dfc.set_index("Date", inplace=True)
            return dfc.sort_index() if len(dfc) >= 20 else None
        except Exception as e:
            logger.error(f"Normalize error for {symbol}: {e}")
            return None

    def fetch_live_data(self, symbol, timeframe):
        try:
            tfmap = {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 mins", "daily": "EOD"}
            bar_size = tfmap.get(timeframe)
            if not bar_size: return None
            duration_map = {5: "10 D", 15: "10 D", 30: "20 D", 60: "60 D", "daily": "365 D"}
            duration = duration_map.get(timeframe, "10 D")
            rawdf = tdhist.get_historic_data(symbol, duration=duration, bar_size=bar_size)
            if rawdf is None or rawdf.empty: return None
            normalized_df = self.normalize_live_data(rawdf, symbol)
            if normalized_df is None: return None
            tail_map = {"daily": 250, 60: 200}
            return normalized_df.tail(tail_map.get(timeframe, 100))
        except Exception as e:
            logger.error(f"Live data fetch error {symbol}@{timeframe}: {e}")
            return None

    def calculate_option_buyer_signals(self, symbol, timeframes_data):
        # ... (This function is unchanged as it was correct) ...
        try:
            if not timeframes_data: return "Neutral", 0
            sector = next((s for s, stocks in SECTOR_STOCKS.items() if symbol in stocks), "Unknown")
            total_weighted_score, total_weight, timeframe_scores = 0.0, 0.0, {}

            for tf, df in timeframes_data.items():
                if df is None or len(df) < 20: continue
                indicators = EnhancedOptionBuyerIndicators.calculate_all_indicators(df)
                if not indicators: continue
                tf_score, tf_weight, current_price = 0.0, 0.0, df["Close"].iloc[-1]

                for name, weight in ENHANCED_INDICATOR_WEIGHTS.items():
                    if name in indicators and not pd.isna(indicators[name].iloc[-1]):
                        latest_val = indicators[name].iloc[-1]
                        if name in ("MA", "EMA", "VWAP"):
                            if latest_val == 0: continue
                            price_vs = (current_price - latest_val) / latest_val * 100
                            norm_score = 75 if price_vs >= 2 else 60 if price_vs >= 0 else 40 if price_vs >= -5 else 25
                        else: norm_score = normalize_indicator_value(name, latest_val)
                        tf_score += norm_score * weight; tf_weight += weight
                
                if tf_weight > 0:
                    tf_final_score = tf_score / tf_weight
                    tf_multiplier = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                    timeframe_scores[tf] = tf_final_score
                    total_weighted_score += tf_final_score * tf_multiplier; total_weight += tf_multiplier
            
            if total_weight == 0: return "Neutral", 0
            base_score = total_weighted_score / total_weight

            bullish_count = sum(1 for v in timeframe_scores.values() if v >= 55)
            bearish_count = sum(1 for v in timeframe_scores.values() if v <= 45)
            if bullish_count >= 3: base_score += 12
            elif bearish_count >= 3: base_score -= 12

            if sector in self.best_sectors: base_score += (30 - self.best_sectors.index(sector) * 5)
            elif sector in self.worst_sectors: base_score -= (30 - self.worst_sectors.index(sector) * 5)

            if base_score >= 85: return "Strong Call Buy", base_score
            if base_score >= 75: return "Call Buy", base_score
            if base_score >= 60: return "Moderate Call", base_score
            if base_score <= 15: return "Strong Put Buy", base_score
            if base_score <= 25: return "Put Buy", base_score
            if base_score <= 40: return "Moderate Put", base_score
            return "Neutral", base_score
        except Exception as e:
            logger.error(f"Signal calculation error for {symbol}: {e}")
            return "Neutral", 0

    def display_scan_summary(self, signals, scan_time):
        # ... (This function is unchanged from last correct version) ...
        os.system("clear" if os.name == "posix" else "cls")
        current_time = datetime.now()
        print(f"{Colors.CYAN}{Colors.BOLD}{'-'*110}{Colors.RESET}")
        print(f"ENHANCED OPTION BUYER SCANNER - {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
        print(f"Scan Time: {scan_time:.2f}s | Call Focus: {Colors.GREEN}{', '.join(self.best_sectors)}{Colors.RESET} | Put Focus: {Colors.RED}{', '.join(self.worst_sectors)}{Colors.RESET}")
        print(f"{Colors.CYAN}{'-'*110}{Colors.RESET}")

        if len(self.sectoral_history) >= 1:
            sector_table_data = []
            latest_sectors = {item['sector']: item['changepercent'] for item in self.sectoral_history[-1]['fulldata']}
            previous_sectors = {}
            if len(self.sectoral_history) >= 2:
                previous_sectors = {item['sector']: item['changepercent'] for item in self.sectoral_history[-2]['fulldata']}
            
            for sector, current_pct in latest_sectors.items():
                prev_pct = previous_sectors.get(sector, current_pct)
                delta = current_pct - prev_pct
                sector_table_data.append({"Sector Name": sector, "Current %": current_pct, "Change %": delta})
            
            df_sectors = pd.DataFrame(sector_table_data).sort_values("Current %", ascending=False)
            try:
                from great_tables import GT
                print("--- Sectoral Performance Snapshot ---")
                sector_table = GT(df_sectors.head(10))
                print(sector_table.fmt_number(columns=["Current %", "Change %"], decimals=2).render_console())
            except ImportError:
                logger.warning("`great_tables` is not installed. Skipping sectoral table.")
            except Exception as e:
                logger.error(f"Sector table render error: {e}")

        if not signals:
            print(f"\n{Colors.YELLOW}No significant option buying opportunities found in this cycle.{Colors.RESET}")
            return

        call_signals = sorted([s for s in signals if "Call" in s["signal"]], key=lambda x: x["score"], reverse=True)
        put_signals = sorted([s for s in signals if "Put" in s["signal"]], key=lambda x: x["score"])

        if call_signals:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🔥 TOP CALL OPPORTUNITIES{Colors.RESET}")
            print(f"{'Stock':<10} {'Sector':<20} {'Signal':<20} {'Score':>8} {'Score Δ':>8}")
            print(f"{Colors.GREEN}{'-'*70}{Colors.RESET}")
            for s in call_signals[:7]:
                delta = s['score'] - self.last_cycle_scores.get(s["symbol"], s['score'])
                print(f"{s['symbol']:<10} {s['sector']:<20} {s['signal']:<20} {s['score']:>8.1f} {f'{delta:+.1f}':>8}")
        
        if put_signals:
            print(f"\n{Colors.RED}{Colors.BOLD}🔻 TOP PUT OPPORTUNITIES{Colors.RESET}")
            print(f"{'Stock':<10} {'Sector':<20} {'Signal':<20} {'Score':>8} {'Score Δ':>8}")
            print(f"{Colors.RED}{'-'*70}{Colors.RESET}")
            for s in put_signals[:7]:
                delta = s['score'] - self.last_cycle_scores.get(s["symbol"], s['score'])
                print(f"{s['symbol']:<10} {s['sector']:<20} {s['signal']:<20} {s['score']:>8.1f} {f'{delta:+.1f}':>8}")

    def enhanced_scan_cycle(self):
        if not self.is_market_open():
            logger.info("Market is closed. Waiting...")
            return

        start_time = timemodule.time()
        print(f"\n{Colors.CYAN}Starting new scan at {datetime.now().strftime('%H:%M:%S')}...{Colors.RESET}")
        
        self.fetch_live_sectoral_performance()
        target_stocks = list(set(stock for sector in self.best_sectors + self.worst_sectors for stock in SECTOR_STOCKS.get(sector, [])))
        if not target_stocks:
            print("No target stocks to scan."); return

        print(f"Scanning {len(target_stocks)} stocks...")
        per_symbol_tfs = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_stock = {executor.submit(self.fetch_live_data, stock, tf): (stock, tf) for stock in target_stocks for tf in [5, 15, 30, 60, "daily"]}
            for future in as_completed(future_to_stock):
                stock, tf = future_to_stock[future]
                # FIX: Added robust error handling for each thread.
                try:
                    df = future.result()
                    if df is not None:
                        if stock not in per_symbol_tfs: per_symbol_tfs[stock] = {}
                        per_symbol_tfs[stock][tf] = df
                except Exception as e:
                    logger.error(f"A thread failed while processing {stock}@{tf}. Error: {e}")

        live_signals = [
            {"symbol": sym, "signal": sig, "score": sco, "sector": next((s for s, stocks in SECTOR_STOCKS.items() if sym in stocks), "Unknown")}
            for sym, tfs in per_symbol_tfs.items()
            for sig, sco in [self.calculate_option_buyer_signals(sym, tfs)] if sig != "Neutral"
        ]
        
        scan_time = timemodule.time() - start_time
        self.display_scan_summary(live_signals, scan_time)
        
        table_results = []
        for s in live_signals:
            sym, tfs = s['symbol'], per_symbol_tfs.get(s['symbol'], {})
            vol, vol_chg, oi, oi_chg, sec_chg = None, None, None, None, 0.0
            
            if 5 in tfs and len(tfs[5]) > 1:
                rdf = tfs[5]
                vol, vol_prev = rdf["Volume"].iloc[-1], rdf["Volume"].iloc[-2]
                if vol_prev > 0: vol_chg = (vol - vol_prev) / vol_prev * 100.0
                oi, oi_prev = rdf["OpenInterest"].iloc[-1], rdf["OpenInterest"].iloc[-2]
                if oi_prev > 0: oi_chg = (oi - oi_prev) / oi_prev * 100.0
            
            if self.sectoral_history:
                sector_data = next((item for item in self.sectoral_history[-1]["fulldata"] if item["sector"] == s['sector']), None)
                if sector_data: sec_chg = sector_data.get("changepercent", 0.0)

            table_results.append({
                "Symbol": sym, "Sector": s['sector'], "Signal": s['signal'], "Score": s['score'],
                "Volume": int(vol) if vol is not None else None,
                "Vol %": f"{vol_chg:.2f}" if vol_chg is not None else None,
                "OI": int(oi) if oi is not None else None,
                "OI %": f"{oi_chg:.2f}" if oi_chg is not None else None,
                "Sector %": f"{sec_chg:.2f}",
            })

        if table_results:
            df_results = pd.DataFrame(table_results).sort_values("Score", ascending=False)
            try:
                from great_tables import GT
                print("\n--- Detailed Signal Analysis Table ---")
                stock_table = GT(df_results)
                print(stock_table.fmt_number(columns="Score", decimals=1).render_console())
            except ImportError:
                logger.warning("`great_tables` is not installed. Skipping detailed stock table.")
            except Exception as e:
                logger.error(f"Stock table render error: {e}")
        
        self.last_cycle_scores = {s["symbol"]: s["score"] for s in live_signals}

    def run_scanner(self):
        self.is_running = True
        self.show_initialization_status()
        try:
            while self.is_running:
                self.enhanced_scan_cycle()
                print(f"\n--- Next scan in {self.scan_interval} seconds ---")
                timemodule.sleep(self.scan_interval)
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logger.error(f"Unhandled error in run loop: {e}")
            self.stop()
            
    def stop(self):
        self.is_running = False
        print("\nScanner stopped by user.")

def main():
    scanner = EnhancedOptionBuyerScanner()
    scanner.run_scanner()

if __name__ == "__main__":
    main()