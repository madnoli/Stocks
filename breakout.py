import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import requests
import json
import time as time_module
import pytz
from logzero import logger
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from truedata.history import TD_hist
import logging
import warnings
warnings.filterwarnings('ignore')

# --- TRUEDATA CONFIG ---
TD_USERNAME = "tdwsp751"
TD_PASSWORD = "raj@751"
try:
    td_hist = TD_hist(TD_USERNAME, TD_PASSWORD, log_level=logging.WARNING)
except Exception as e:
    print(f"Failed to initialize Truedata history client: {e}")
    td_hist = None

# --- MASTER STOCK LIST FOR SCANNING ---
ALL_NSE_STOCKS = [
    "CHOLAFIN", "GMRAIRPORT", "CYIENT", "HFCL", "AMBER", "KOTAKBANK", "PERSISTENT", "NHPC",
    "LT", "PAGEIND", "M&M", "RVNL", "SUPREMEIND", "BHARATFORG", "TATAPOWER", "KEI",
    "MARUTI", "POLYCAB", "PRESTIGE", "MOTHERSON", "OFSS", "NCC", "EICHERMOT", "BLUESTARCO",
    "BHARTIARTL", "PHOENIXLTD", "NBCC", "MUTHOOTFIN", "LTF", "MANAPPURAM", "TATASTEEL",
    "IIFL", "SUZLON", "AXISBANK", "VEDL", "UNOMINDA", "JSWENERGY", "TIINDIA", "CUMMINSIND",
    "CONCOR", "GRASIM", "COFORGE", "DLF", "UPL", "JSWSTEEL", "GAIL", "ASTRAL", "ETERNAL",
    "HAVELLS", "ONGC", "BOSCHLTD", "GODREJPROP", "NTPC", "ULTRACEMCO", "NYKAA", "HCLTECH",
    "UNITDSPR", "360ONE", "BEL", "BHEL", "TCS", "LODHA", "WIPRO", "SHREECEM", "DELHIVERY",
    "OIL", "DMART", "CAMS", "PPLPHARMA", "HAL", "ADANIPORTS", "SOLARINDS", "AMBUJACEM",
    "POLICYBZR", "SBIN", "TECHM", "KALYANKJIL", "KAYNES", "DRREDDY", "POWERGRID",
    "MAZDOCK", "DIXON", "DIVISLAB", "CIPLA", "IOC", "ADANIENT", "JINDALSTEL",
    "CROMPTON", "TVSMOTOR", "ICICIGI", "TITAN", "CANBK", "HDFCAMC", "SIEMENS",
    "EXIDEIND", "IRFC", "PETRONET", "HINDPETRO", "RECLTD", "BIOCON", "BAJAJ-AUTO",
    "LTIM", "DALBHARAT", "SUNPHARMA", "HEROMOTOCO", "HUDCO",  "APOLLOHOSP",
    "HINDZINC", "ASHOKLEY", "RELIANCE", "IGL", "TATAELXSI", "MPHASIS", "IREDA", "LUPIN",
    "INDUSINDBK", "HINDALCO", "PFC", "TRENT", "PAYTM", "IRCTC", "COALINDIA",
    "SAMMAANCAP", "PATANJALI", "ABB", "INFY", "OBEROIRLTY", "JUBLFOOD", "ICICIBANK", "BPCL",
    "ADANIGREEN", "IEX", "SRF", "CGPOWER", "ITC", "SAIL", "FEDERALBNK", "KFINTECH", "ALKEM",
    "TATAMOTORS", "JIOFIN", "BDL", "BAJAJFINSV", "HINDUNILVR","INOXWIND", "INDIGO", "HDFCBANK", "LAURUSLABS", "TORNTPHARM", "TATATECH", "PNB",
    "ADANIENSOL", "VOLTAS", "NMDC", "IDFCFIRSTB", "LICI", "NATIONALUM", "BRITANNIA",
    "APLAPOLLO", "SBILIFE", "ZYDUSLIFE", "ICICIPRULI", "ABCAPITAL",
    "CDSL", "KPITTECH", "PIIND", "LICHSGFIN", "AUBANK", "SONACOMS", "TORNTPOWER", "HDFCLIFE",
    "SBICARD", "BANKINDIA", "COLPAL", "INDUSTOWER", "NUVAMA", "MARICO", "PNBHOUSING", "PGEL",
    "MANKIND", "BAJFINANCE", "NESTLEIND", "NAUKRI", "AUROPHARMA", "ASIANPAINT", "SHRIRAMFIN",
    "TATACONSUM", "ANGELONE", "MFSL", "DABUR", "TITAGARH", "GLENMARK", "FORTIS", "BSE",
    "MAXHEALTH", "MCX", "INDHOTEL", "VBL", "SYNGENE", "GODREJCP"
]

# --- COLOR CODES ---
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

# --- NSE INDEX TO SECTOR MAPPING ---
NSE_INDEX_TO_SECTOR = {
    "NIFTY IT": "Technology", "NIFTY PHARMA": "Pharma", "NIFTY FMCG": "Consumer",
    "NIFTY BANK": "Banking", "NIFTY AUTO": "Auto", "NIFTY METAL": "Metal",
    "NIFTY ENERGY": "Energy", "NIFTY REALTY": "Realty", "NIFTY INFRA": "Infrastructure",
    "NIFTY PSU BANK": "PSU Bank", "NIFTY PSE": "PSE", "NIFTY COMMODITIES": "Commodities",
    "NIFTY MNC": "Finance", "NIFTY FINANCIAL SERVICES": "Finance",
    "NIFTY INFRASTRUCTURE": "Infrastructure", "BANKNIFTY": "Banking",
    "NIFTYAUTO": "Auto", "NIFTYIT": "Technology", "NIFTYPHARMA": "Pharma",
    "NIFTY CONSUMER DURABLES": "Consumer Durables", "NIFTY HEALTHCARE INDEX": "Healthcare",
    "NIFTY CAPITAL MARKETS": "Capital Market", "NIFTY PRIVATE BANK": "Private Bank",
    "NIFTY OIL & GAS": "Oil and Gas", "NIFTY INDIA DEFENCE": "Defence",
    "NIFTY CORE HOUSING": "Core Housing", "NIFTY SERVICES SECTOR": "Services Sector",
    "NIFTY FINANCIAL SERVICES 25/50": "Financial Services 25/50", "NIFTY INDIA TOURISM": "Tourism",
}

# --- SECTOR TO STOCKS MAPPING ---
SECTOR_STOCKS =  {
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
    "Financial Services 25/50": ["HDFCBANK-I", "ICICIBANK-I", "SBIN-I", "BAJFINANCE-I", "KOTAKBANK-I", "AXISBANK-I", "BAJAJFINSV-I", "JIOFIN-I", "SBILIFE-I", "HDFCLIFE-I", "PFC-I", "CHOLAFIN-I", "HDFCAMC-I", "SHRIRAMFIN-I", "MUTHOOTFIN-I", "RECLTD-I", "ICICIGI-I", "ICICIPRULI-I", "SBICARD-I", "LICHSGFIN-I"],
    "Tourism": ["INDIGO-I", "INDHOTEL-I", "IRCTC-I", "JUBLFOOD-I"]
}

# --- Token Bucket Rate Limiter (thread-safe) ---
class TokenBucket:
    def __init__(self, rate_per_sec=9.0, bucket_size=12, per_min_ceiling=300):
        self.rate = rate_per_sec
        self.bucket_size = bucket_size
        self.tokens = bucket_size
        self.last_refill = time_module.time()
        self.lock = threading.Lock()
        self.per_min_ceiling = per_min_ceiling
        self.min_window = deque()

    def acquire(self):
        while True:
            with self.lock:
                now = time_module.time()
                elapsed = now - self.last_refill
                refill = elapsed * self.rate
                if refill > 0:
                    self.tokens = min(self.bucket_size, self.tokens + refill)
                    self.last_refill = now

                while self.min_window and self.min_window[0] <= now - 60.0:
                    self.min_window.popleft()

                if self.tokens >= 1.0 and len(self.min_window) < self.per_min_ceiling:
                    self.tokens -= 1.0
                    self.min_window.append(now)
                    return
            time_module.sleep(0.03)

api_limiter = TokenBucket(rate_per_sec=10.0, bucket_size=12, per_min_ceiling=450) # Increased ceiling for multi-tf calls

# === Indicator Engines (No changes needed here) ===
class EnhancedTechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        indicators = {}
        if df is None or len(df) < 30:
            return indicators
        try:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            indicators['RSI'] = 100 - (100 / (1 + rs))

            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            indicators['MACD'] = macd_line - signal_line

            tr1 = df['High'] - df['Low']
            tr2 = (df['High'] - df['Close'].shift()).abs()
            tr3 = (df['Low'] - df['Close'].shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            ATR = true_range.rolling(window=14).mean()
            indicators['ATR'] = ATR

            avg_volume_20 = df['Volume'].rolling(window=20).mean().replace(0, 1)
            volume_ratio = df['Volume'] / avg_volume_20
            indicators['Volume_Surge'] = np.clip((volume_ratio - 0.5) * 40, 0, 100)

            price_momentum = df['Close'].pct_change(periods=10) * 100
            avg_volume_10 = df['Volume'].rolling(window=10).mean().replace(0, 1)
            volume_momentum = (df['Volume'] / avg_volume_10 - 1) * 100
            momentum_score = (price_momentum * 0.7 + volume_momentum * 0.3)
            indicators['Momentum'] = 50 + np.clip(momentum_score * 1.5, -50, 50)
        except Exception as e:
            logger.error(f"Error in base indicators: {e}")
        return indicators

class OptionsReadyIndicators(EnhancedTechnicalIndicators):
    @staticmethod
    def calculate_all_indicators(
        df,
        squeeze_window=120,
        squeeze_quantile=0.1,
        require_two_bar_atr_accel=True
    ):
        indicators = super(OptionsReadyIndicators, OptionsReadyIndicators).calculate_all_indicators(df)
        if not indicators:
            return indicators
        try:
            ma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            upper_band = ma20 + (std20 * 2.0)
            lower_band = ma20 - (std20 * 2.0)

            bbw = (upper_band - lower_band) / ma20.replace(0, np.nan)
            indicators['BBW'] = bbw

            bbw_quantile = bbw.rolling(window=squeeze_window, min_periods=20).quantile(
                squeeze_quantile, interpolation='linear'
            )
            indicators['BBW_q'] = bbw_quantile

            in_squeeze = bbw <= bbw_quantile
            indicators['in_squeeze'] = in_squeeze

            breaks_upper = df['Close'] > upper_band.shift(1)
            breaks_lower = df['Close'] < lower_band.shift(1)
            vol_surge_now = indicators.get('Volume_Surge', pd.Series(index=df.index, data=0))
            vol_confirm = vol_surge_now > 60.0
            squeeze_on = in_squeeze.shift(1).fillna(False)

            indicators['squeeze_fire_up'] = squeeze_on & breaks_upper & vol_confirm
            indicators['squeeze_fire_down'] = squeeze_on & breaks_lower & vol_confirm

            ATR = indicators.get('ATR', pd.Series(index=df.index, data=np.nan))
            ATR_EMA10 = ATR.ewm(span=10, adjust=False).mean()
            atr_gt = ATR > ATR_EMA10
            atr_rising = ATR > ATR.shift(1)
            if require_two_bar_atr_accel:
                atr_accel = atr_gt & atr_gt.shift(1).fillna(False) & atr_rising
            else:
                atr_accel = atr_gt & atr_rising
            indicators['ATR_EMA10'] = ATR_EMA10
            indicators['ATR_accel'] = atr_accel.astype(float)
        except Exception as e:
            logger.error(f"Error in options indicators: {e}")
        return indicators

# === Scanner with Direct API Calls & Caching ===
class OptionsBreakoutScanner:
    def __init__(self):
        self.is_running = False
        self.best_sectors = []
        self.worst_sectors = []
        self.best_sector_details = []
        self.worst_sector_details = []
        self.last_sectoral_update = None
        self.market_start = time(9, 15)
        self.market_end = time(15, 30)
        self.scan_interval = 300
        self.min_avg_vol_5m = 50000
        self.min_avg_vol_15m = 30000
        self.use_mtf_confirmation = True

        # --- CHANGE: Data cache for multi-timeframe data ---
        self.data_cache = {}

        # --- CHANGE: Configuration for multi-timeframe API calls ---
        self.timeframe_params = {
            5:  {'bar_size': '5 min', 'initial_duration': '3 D', 'update_duration': '1 D'},
            15: {'bar_size': '15 min', 'initial_duration': '5 D', 'update_duration': '1 D'},
            30: {'bar_size': '30 min', 'initial_duration': '10 D', 'update_duration': '2 D'},
            60: {'bar_size': '60 min', 'initial_duration': '15 D', 'update_duration': '2 D'}
        }
        self.signal_timeframes = [5, 15, 30] # TFs to check for squeeze signals
        self.trend_timeframe = 60 # TF for higher-level trend bias

        logger.info("🚀 Options Breakout Scanner Initialized (Direct API Mode)")

    def show_initialization_status(self):
        print(f"\n{Colors.CYAN}{Colors.BOLD}🚀 OPTIONS BREAKOUT SCANNER (DIRECT API & CACHING){Colors.RESET}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"⏰ Timeframes: {Colors.YELLOW}5m, 15m, 30m (Signals) & 60m (Trend){Colors.RESET}")
        print(f"🎯 Strategy: {Colors.MAGENTA}Find Bollinger Band Squeeze Breakouts for Option Buyers{Colors.RESET}")
        print(f"⚡ Stocks Scanned: {Colors.BOLD}{len(ALL_NSE_STOCKS)} stocks from Master List{Colors.RESET}")
        print(f"☁️ Data Mode: {Colors.BOLD}Direct API calls per timeframe (no resampling){Colors.RESET}")

    def fetch_live_sectoral_performance(self):
        try:
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            if response.status_code == 200:
                indices_data = response.json()
                if isinstance(indices_data, str):
                    indices_data = json.loads(indices_data)
                if isinstance(indices_data, dict):
                    indices_data = indices_data.get('data') or indices_data.get('indices') or indices_data.get('results', [])

                sectoral_performance = []
                for index in indices_data:
                    if not isinstance(index, dict):
                        continue
                    index_name = next((str(index[field]).strip().upper() for field in ['name', 'symbol', 'indexName'] if field in index and index[field]), None)
                    if index_name and index_name in NSE_INDEX_TO_SECTOR:
                        change_percent_str = next((str(index.get(field)) for field in ['pChange', 'percentChange', 'change', 'change_percent'] if index.get(field) is not None), None)
                        if change_percent_str:
                            try:
                                change_percent = float(change_percent_str)
                                sectoral_performance.append({'index': index_name, 'sector': NSE_INDEX_TO_SECTOR[index_name], 'change_percent': change_percent})
                            except (ValueError, TypeError):
                                continue

                if sectoral_performance:
                    sectoral_performance.sort(key=lambda x: x['change_percent'], reverse=True)
                    self.best_sector_details = sectoral_performance[:3]
                    self.worst_sector_details = list(reversed(sectoral_performance[-3:]))
                    self.best_sectors = [s['sector'] for s in self.best_sector_details]
                    self.worst_sectors = [s['sector'] for s in self.worst_sector_details]
                    self.last_sectoral_update = datetime.now()
                    return True
        except requests.exceptions.ConnectionError:
            logger.warning("Could not connect to sectoral data API. Continuing without it.")
        except Exception as e:
            logger.error(f"API sectoral data error: {e}")
        return False

    def force_sector_update(self):
        print(f"\n{Colors.YELLOW}🔄 Fetching live sectoral performance...{Colors.RESET}")
        if self.fetch_live_sectoral_performance():
            print(f"✅ API sectoral update successful!")
        else:
            print(f"{Colors.RED}❌ API sectoral update failed. Scanner will run without sectoral bias.{Colors.RESET}")

    def normalize_live_data(self, df, symbol):
        try:
            if df is None or df.empty:
                return None
            df_clean = df.copy()
            col_lookup = {col.lower(): col for col in df_clean.columns}

            date_col = col_lookup.get('timestamp') or col_lookup.get('time')
            open_col = col_lookup.get('open')
            high_col = col_lookup.get('high')
            low_col = col_lookup.get('low')
            close_col = col_lookup.get('close')
            vol_col = col_lookup.get('volume') or col_lookup.get('vol')

            if not all([date_col, open_col, high_col, low_col, close_col, vol_col]):
                logger.error(f"Missing required columns for {symbol}. Available: {list(df.columns)}")
                return None

            final_df = pd.DataFrame({
                'Date': pd.to_datetime(df_clean[date_col]),
                'Open': pd.to_numeric(df_clean[open_col], errors='coerce'),
                'High': pd.to_numeric(df_clean[high_col], errors='coerce'),
                'Low': pd.to_numeric(df_clean[low_col], errors='coerce'),
                'Close': pd.to_numeric(df_clean[close_col], errors='coerce'),
                'Volume': pd.to_numeric(df_clean[vol_col], errors='coerce')
            })
            final_df.set_index('Date', inplace=True)
            return final_df.dropna().sort_index()
        except Exception as e:
            logger.error(f"Normalize error for {symbol}: {e}")
            return None

    # --- NEW: Function to fetch data for a specific timeframe ---
    def fetch_data_for_timeframe(self, symbol, bar_size, duration):
        if not td_hist: return None
        try:
            api_limiter.acquire()
            raw_df = td_hist.get_historic_data(symbol, duration=duration, bar_size=bar_size)
            return self.normalize_live_data(raw_df, symbol)
        except Exception as e:
            logger.error(f"Error fetching {bar_size} data for {symbol}: {e}")
            return None

    # --- NEW: Initial data load for all stocks and timeframes ---
    def initial_data_load(self):
        print(f"\n{Colors.YELLOW}⏳ Performing initial bulk data load for {len(ALL_NSE_STOCKS)} stocks... This may take a few minutes.{Colors.RESET}")
        
        with ThreadPoolExecutor(max_workers=12) as executor:
            future_to_stock_tf = {}
            for symbol in ALL_NSE_STOCKS:
                self.data_cache[symbol] = {}
                for tf_int, params in self.timeframe_params.items():
                    future = executor.submit(self.fetch_data_for_timeframe, symbol, params['bar_size'], params['initial_duration'])
                    future_to_stock_tf[future] = (symbol, tf_int)
            
            for i, future in enumerate(as_completed(future_to_stock_tf)):
                symbol, tf_int = future_to_stock_tf[future]
                try:
                    df = future.result()
                    if df is not None and not df.empty:
                        self.data_cache[symbol][tf_int] = df
                except Exception as e:
                    logger.error(f"Error in initial load for {symbol} {tf_int}m: {e}")
                print(f"Loading... {i+1}/{len(ALL_NSE_STOCKS) * len(self.timeframe_params)}", end="\r")

        print(f"\n{Colors.GREEN}✅ Initial data load complete.{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")
        
    def _passes_liquidity(self, timeframes_data):
        df5 = timeframes_data.get(5)
        if df5 is not None and not df5.empty:
            avg5 = df5['Volume'].tail(20).mean()
            if not np.isnan(avg5) and avg5 < self.min_avg_vol_5m:
                return False
        return True

    # --- UPDATED: Use 60min data for a more robust trend bias ---
    def _mtf_trend_bias(self, timeframes_data):
        df_trend = timeframes_data.get(self.trend_timeframe)
        if df_trend is None or len(df_trend) < 30:
            return 0
        
        inds = EnhancedTechnicalIndicators.calculate_all_indicators(df_trend)
        if not inds: return 0
        
        macd = inds.get('MACD')
        ma20 = df_trend['Close'].rolling(20).mean()
        bias = 0
        if macd is not None and not macd.empty and pd.notna(macd.iloc[-1]):
            bias += 1 if macd.iloc[-1] > 0 else -1
        if not ma20.empty and pd.notna(ma20.iloc[-1]):
            bias += 1 if df_trend['Close'].iloc[-1] > ma20.iloc[-1] else -1
        return bias

    def calculate_options_signals(self, symbol, timeframes_data):
        final_score = 50.0
        strongest_signal = "Neutral"
        squeeze_status = "No Squeeze"
        try:
            if not self._passes_liquidity(timeframes_data):
                return 'Neutral', 50.0, "Liquidity Fail"

            mtf_bias = self._mtf_trend_bias(timeframes_data) if self.use_mtf_confirmation else 0

            for tf in self.signal_timeframes:
                df = timeframes_data.get(tf)
                if df is None or df.empty: continue
                
                indicators = OptionsReadyIndicators.calculate_all_indicators(df)
                if not indicators: continue
                
                latest = {name: ind.iloc[-1] for name, ind in indicators.items() if hasattr(ind, 'iloc') and len(ind) > 0 and pd.notna(ind.iloc[-1])}

                is_firing_up = bool(latest.get('squeeze_fire_up', False))
                is_firing_down = bool(latest.get('squeeze_fire_down', False))
                vol_surge_score = float(latest.get('Volume_Surge', 0))
                momentum_score = float(latest.get('Momentum', 50))
                atr_accel = float(latest.get('ATR_accel', 0)) > 0
                
                sector = next((s for s, st in SECTOR_STOCKS.items() if symbol in st), 'N/A')
                sector_boost = 5 if sector in self.best_sectors else -5 if sector in self.worst_sectors else 0

                if is_firing_up:
                    score = 80 + sector_boost
                    if momentum_score > 65: score += 10
                    if vol_surge_score > 60: score += 15
                    if atr_accel: score += 10
                    if self.use_mtf_confirmation and mtf_bias < 0: score -= 15 # Stronger penalty
                    
                    if score > final_score:
                        final_score, strongest_signal, squeeze_status = score, "Explosive Buy", f"{tf}m Squeeze FIRE UP"
                
                elif is_firing_down:
                    score = 20 + sector_boost
                    if momentum_score < 35: score -= 10
                    if vol_surge_score > 60: score -= 15
                    if atr_accel: score -= 10
                    if self.use_mtf_confirmation and mtf_bias > 0: score += 15 # Stronger penalty
                    
                    if score < final_score:
                        final_score, strongest_signal, squeeze_status = score, "Explosive Sell", f"{tf}m Squeeze FIRE DOWN"

                elif bool(latest.get('in_squeeze', False)) and squeeze_status == "No Squeeze":
                    squeeze_status = f"{tf}m Squeeze Coiling"

            return strongest_signal, float(np.clip(final_score, 0, 100)), squeeze_status
        except Exception as e:
            logger.error(f"Signal calc error for {symbol}: {e}")
            return 'Neutral', 50.0, "Error"

    # --- NEW: Function to update and process a single stock using the cache ---
    def update_and_process_stock(self, symbol):
        try:
            # 1. Update data from API
            for tf_int, params in self.timeframe_params.items():
                new_data = self.fetch_data_for_timeframe(symbol, params['bar_size'], params['update_duration'])
                if new_data is not None and not new_data.empty:
                    cached_data = self.data_cache.get(symbol, {}).get(tf_int)
                    if cached_data is not None:
                        combined = pd.concat([cached_data, new_data])
                        combined = combined[~combined.index.duplicated(keep='last')]
                        self.data_cache[symbol][tf_int] = combined.sort_index().tail(500) # Keep cache size reasonable
                    else: # If cache was empty for some reason
                        self.data_cache[symbol][tf_int] = new_data.tail(500)
            
            # 2. Process the updated data
            timeframes_data = self.data_cache.get(symbol)
            if timeframes_data and len(timeframes_data) >= len(self.timeframe_params):
                signal, score, squeeze = self.calculate_options_signals(symbol, timeframes_data)
                if "Explosive" in signal:
                    sector = next((s for s, st in SECTOR_STOCKS.items() if symbol in st), 'N/A')
                    return {'symbol': symbol, 'signal': signal, 'score': score, 'sector': sector, 'squeeze': squeeze}
        except Exception as e:
            logger.error(f"Update/Process error for {symbol}: {e}")
        return None

    def display_signals(self, signals, scan_time):
        os.system('clear' if os.name == 'posix' else 'cls')
        print(f"\n{Colors.CYAN}{Colors.BOLD}🚀 OPTIONS BREAKOUT SCANNER - {datetime.now().strftime('%H:%M:%S')} IST{Colors.RESET}")
        print(f"⚡ Scan Time: {scan_time:.2f}s | Mode: Direct API (Cached)")

        if self.last_sectoral_update:
            best_str = ', '.join([f"{item['sector']} ({item['change_percent']:.2f}%)" for item in self.best_sector_details])
            print(f"🏆 Best Sectors: {Colors.GREEN}{best_str}{Colors.RESET}")

        if not signals:
            print(f"\n{Colors.YELLOW}📭 No high-conviction breakout signals found.{Colors.RESET}")
            return

        bullish = sorted([s for s in signals if 'Buy' in s['signal']], key=lambda x: x['score'], reverse=True)
        bearish = sorted([s for s in signals if 'Sell' in s['signal']], key=lambda x: x['score'])

        if bullish:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🚀 TOP BULLISH BREAKOUTS (CALL OPTIONS):{Colors.RESET}")
            print(f"{'Stock':<10} {'Sector':<18} {'Signal':<15} {'Score':>8} {'Squeeze Status'}")
            print(f"{Colors.GREEN}{'-' * 90}{Colors.RESET}")
            for s in bullish[:10]:
                sector_color = Colors.GREEN if s['sector'] in self.best_sectors else Colors.YELLOW
                print(f"{Colors.WHITE}{s['symbol']:<10}{Colors.RESET} {sector_color}{s['sector']:<18}{Colors.RESET} {Colors.GREEN}{Colors.BOLD}{s['signal']:<15}{Colors.RESET} {Colors.WHITE}{s['score']:>8.1f}{Colors.RESET} {Colors.MAGENTA}{s['squeeze']}{Colors.RESET}")

        if bearish:
            print(f"\n{Colors.RED}{Colors.BOLD}📉 TOP BEARISH BREAKOUTS (PUT OPTIONS):{Colors.RESET}")
            print(f"{'Stock':<10} {'Sector':<18} {'Signal':<15} {'Score':>8} {'Squeeze Status'}")
            print(f"{Colors.RED}{'-' * 90}{Colors.RESET}")
            for s in bearish[:10]:
                sector_color = Colors.RED if s['sector'] in self.worst_sectors else Colors.YELLOW
                print(f"{Colors.WHITE}{s['symbol']:<10}{Colors.RESET} {sector_color}{s['sector']:<18}{Colors.RESET} {Colors.RED}{Colors.BOLD}{s['signal']:<15}{Colors.RESET} {Colors.WHITE}{s['score']:>8.1f}{Colors.RESET} {Colors.MAGENTA}{s['squeeze']}{Colors.RESET}")

    def is_market_open(self):
        now = datetime.now()
        return self.market_start <= now.time() <= self.market_end and now.weekday() < 5

    def run_scanner(self):
        self.is_running = True
        try:
            self.show_initialization_status()
            
            # --- CHANGE: Perform initial data load before starting the loop ---
            self.initial_data_load()

            while self.is_running:
                if not self.is_market_open():
                    print("Market is closed. Waiting for the next open session...", end="\r")
                    time_module.sleep(60)
                    continue

                start_time = time_module.time()
                self.force_sector_update()
                live_signals = []

                with ThreadPoolExecutor(max_workers=12) as executor:
                    futures = {executor.submit(self.update_and_process_stock, symbol): symbol for symbol in ALL_NSE_STOCKS}
                    for i, future in enumerate(as_completed(futures)):
                        print(f"Scanning... {i+1}/{len(ALL_NSE_STOCKS)}", end="\r")
                        try:
                            result = future.result()
                            if result:
                                live_signals.append(result)
                        except Exception as e:
                            logger.error(f"Error getting result for a stock: {e}")

                self.display_signals(live_signals, time_module.time() - start_time)

                print(f"\n{Colors.CYAN}Next scan in {self.scan_interval/60:.0f} minutes... Press Ctrl+C to stop.{Colors.RESET}")
                time_module.sleep(self.scan_interval)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.is_running = False
        print(f"\n{Colors.YELLOW}🛑 Scanner stopped by user.{Colors.RESET}")

if __name__ == "__main__":
    if td_hist is None:
        print(f"{Colors.RED}Could not start scanner because Truedata client failed to initialize.{Colors.RESET}")
        print("Please check your credentials and network connection.")
    else:
        scanner = OptionsBreakoutScanner()
        scanner.run_scanner()