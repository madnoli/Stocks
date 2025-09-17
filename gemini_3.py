# ENHANCED REAL-TIME SCANNER FOR OPTION BUYERS
# Features: Bollinger Band Squeeze, ATR Acceleration, Conviction Scoring
# SPEED-OPTIMIZED WITH LOCAL CACHING
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
SECTOR_STOCKS = {
    "Technology": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "CYIENT", "KPITTECH", "TATAELXSI","SONACOMS","KAYNES","OFSS"],
    "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR","BHARATFORG", "EICHERMOT", "ASHOKLEY", "BOSCHLTD","TIINDIA","MOTHERSON"],
    "Banking": ["HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","PNB","BANKBARODA","CANBK","IDFCFIRSTB","INDUSINDBK","AUBANK","FEDERALBNK"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA", "TORNTPHARM","GLENMARK","ALKEM","LAURUSLABS","BIOCON","ZYDUSLIFE","MANKIND","SYNGENE","PPLPHARMA"],
    "Energy": ["RELIANCE","NTPC","BPCL","IOC","ONGC","GAIL","HINDPETRO","ADANIGREEN","ADANIENSOL","JSWENERGY","COALINDIA","TATAPOWER","SUZLON","PETRONET","OIL","POWERGRID","NHPC","ADANIPORTS","ABB","SIEMENS","CGPOWER","INOXWIND"],
    "Metal": ["TATASTEEL", "JSWSTEEL", "SAIL", "JINDALSTEL", "HINDALCO", "NMDC"],
    "Consumer": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR","AMBER","UNITDSPR","GODREJCP","MARICO","COLPAL","UPL","VBL"],
    "PSU Bank": ["SBIN", "PNB", "BANKBARODA", "CANBK", "UNIONBANK", "BANKINDIA"],
    "Finance": ["BAJFINANCE", "SHRIRAMFIN", "CHOLAFIN", "HDFCLIFE", "ICICIPRULI","ETERNAL"],
    "Realty": ["DLF","LODHA","PRESTIGE","GODREJPROP","OBEROIRLTY","PHOENIXLTD","NCC","NBCC"],
    "PSE": ["BEL","BHEL","NHPC","GAIL","IOC","NTPC","POWERGRID","HINDPETRO","OIL","RECLTD","ONGC","NMDC","BPCL","HAL","RVNL","PFC","COALINDIA","IRCTC","IRFC"],
    "Commodities": ["AMBUJACEM","APLAPOLLO","ULTRACEMCO","SHREECEM","JSWSTEEL","HINDALCO","NHPC","IOC","NTPC","HINDPETRO","ADANIGREEN","OIL","VEDL","PIIND","ONGC","NMDC","UPL","BPCL","JSWENERGY","GRASIM","RELIANCE","TORNTPOWER","TATAPOWER","COALINDIA","PIDILITIND","SRF","ADANIENSOL","JINDALSTEL","TATASTEEL","HINDALCO"],
    "Consumer Durables": ["TITAN","DIXON","HAVELLS","CROMPTON","POLYCAB","EXIDEIND","AMBER","KAYNES","VOLTAS","PGEL","BLUESTARCO"],
    "Healthcare": ["SUNPHARMA","DIVISLAB","CIPLA","TORNTPHARM","MAXHEALTH","APOLLOHOSP","DRREDDY","MANKIND","ZYDUSLIFE","LUPIN","FORTIS","ALKEM","AUROPHARMA","GLENMARK","BIOCON","LAURUSLABS","SYNGENE","GRANULES"],
    "Capital Market": ["HDFCAMC","BSE","360ONE","MCX","CDSL","NUVAMA","ANGELONE","KFINTECH","CAMS","IEX"],
    "Private Bank": ["HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","YESBANK","IDFCFIRSTB","INDUSINDBK","FEDERALBNK","BANDHANBNK","RBLBANK"],
    "Oil and Gas": ["RELIANCE","ONGC","IOC","BPCL","GAIL","HINDPETRO","OIL","PETRONET","IGL"],
    "Defence": ["HAL","BEL","SOLARINDS","MAZDOCK","BDL"],
    "Core Housing": ["ULTRACEMCO","ASIANPAINT","GRASIM","DLF","AMBUJACEM","LODHA","DIXON","POLYCAB","SHREECEM","HAVELLS","PRESTIGE","GODREJPROP","OBEROIRLTY","PHOENIXLTD","VOLTAS","DALBHARAT","KEI","BLUESTARCO","LICHSGFIN","PNBHOUSING","CROMPTON"],
    "Services Sector": ["HDFCBANK", "BHARTIARTL", "TCS", "ICICIBANK", "SBIN", "INFY", "BAJFINANCE", "HCLTECH", "KOTAKBANK", "AXISBANK", "BAJAJFINSV", "NTPC", "ZOMATO", "ADANIPORTS", "DMART", "POWERGRID", "WIPRO", "INDIGO", "JIOFINSERV", "SBILIFE", "HDFCLIFE", "LTIM", "TECHM", "TATAPOWER", "SHRIRAMFIN", "GAIL", "MAXHEALTH", "APOLLOHOSP", "NAUKRI", "INDUSINDBK"],
    "Financial Services 25/50": ["HDFCBANK", "ICICIBANK", "SBIN", "BAJFINANCE", "KOTAKBANK", "AXISBANK", "BAJAJFINSV", "JIOFIN", "SBILIFE", "HDFCLIFE", "PFC", "CHOLAFIN", "HDFCAMC", "SHRIRAMFIN", "MUTHOOTFIN", "RECLTD", "ICICIGI", "ICICIPRULI", "SBICARD", "LICHSGFIN"],
    "Tourism": ["INDIGO","INDHOTEL","IRCTC","JUBLFOOD"]
}

class EnhancedTechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        indicators = {}
        if df is None or len(df) < 20: return indicators
        try:
            close = df['Close']
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs))
            
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            indicators['MACD'] = macd_line - signal_line
            
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - close.shift())
            tr3 = abs(df['Low'] - close.shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            indicators['ATR'] = true_range.rolling(window=14).mean()
            
            volume = df['Volume']
            avg_volume_20 = volume.rolling(window=20).mean().replace(0, 1)
            volume_ratio = volume / avg_volume_20
            indicators['Volume_Surge'] = np.clip((volume_ratio - 0.5) * 40, 0, 100)
            
            price_momentum = close.pct_change(periods=10) * 100
            avg_volume_10 = volume.rolling(window=10).mean().replace(0, 1)
            volume_momentum = (volume / avg_volume_10 - 1) * 100
            momentum_score = (price_momentum * 0.7 + volume_momentum * 0.3)
            indicators['Momentum'] = 50 + np.clip(momentum_score * 1.5, -50, 50)
            
        except Exception as e: logger.error(f"Error in base indicators: {e}")
        return indicators

class OptionsReadyIndicators(EnhancedTechnicalIndicators):
    @staticmethod
    def calculate_all_indicators(df, squeeze_period=120):
        indicators = super(OptionsReadyIndicators, OptionsReadyIndicators).calculate_all_indicators(df)
        if not indicators: return indicators
        try:
            close = df['Close']
            ma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            upper_band = ma20 + (std20 * 2)
            lower_band = ma20 - (std20 * 2)
            
            bandwidth = (upper_band - lower_band) / ma20
            lowest_bandwidth = bandwidth.rolling(window=squeeze_period).min()
            indicators['in_squeeze'] = bandwidth <= lowest_bandwidth
            
            squeeze_on = indicators['in_squeeze'].shift(1)
            breaks_upper = close > upper_band.shift(1)
            breaks_lower = close < lower_band.shift(1)
            indicators['squeeze_fire_up'] = squeeze_on & breaks_upper
            indicators['squeeze_fire_down'] = squeeze_on & breaks_lower
            
            if 'ATR' in indicators and not indicators['ATR'].empty:
                indicators['ATR_accel'] = (indicators['ATR'] > indicators['ATR'].rolling(window=10).mean()) * 1.0
        except Exception as e: logger.error(f"Error in options indicators: {e}")
        return indicators

class OptionsBreakoutScanner:
    def __init__(self):
        self.is_running = False
        self.best_sectors, self.worst_sectors = [], []
        self.best_sector_details, self.worst_sector_details = [], []
        self.last_sectoral_update = None
        self.gap_down_filtered_count = 0
        self.market_start = time(9, 15); self.market_end = time(15, 30)
        self.scan_interval = 300
        self.api_call_timestamps = deque()
        self.api_lock = threading.Lock()
        self.RATE_LIMIT_COUNT = 9; self.RATE_LIMIT_PERIOD = 1
        
        # <<< --- NEW: CACHE SETUP --- >>>
        self.cache_dir = "truedata_cache"
        self.cache_expiry_days = 1
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            
        logger.info("🚀 Options Breakout Scanner Initialized")
        self.show_initialization_status()

    def show_initialization_status(self):
        print(f"\n{Colors.CYAN}{Colors.BOLD}🚀 OPTIONS BREAKOUT SCANNER (CACHING ENABLED){Colors.RESET}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"⏰ Timeframes: {Colors.YELLOW}5min, 15min, 30min, 60min{Colors.RESET}")
        print(f"🎯 Strategy: {Colors.MAGENTA}Find Bollinger Band Squeeze Breakouts for Option Buyers{Colors.RESET}")
        print(f"⚡ Stocks Scanned: {Colors.BOLD}{len(ALL_NSE_STOCKS)} stocks from Master List{Colors.RESET}")
        print(f"📁 Cache Directory: {self.cache_dir}")
        print(f"\n{Colors.YELLOW}🔄 Running initial setup... (First scan of the day may be slow){Colors.RESET}")
        self.force_sector_update()
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")
    
    def fetch_live_sectoral_performance(self):
        try:
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            if response.status_code == 200:
                indices_data = response.json()
                if isinstance(indices_data, str): indices_data = json.loads(indices_data)
                if isinstance(indices_data, dict):
                    indices_data = indices_data.get('data') or indices_data.get('indices') or indices_data.get('results', [])
                
                sectoral_performance = []
                for index in indices_data:
                    if not isinstance(index, dict): continue
                    index_name = next((str(index[field]).strip().upper() for field in ['name', 'symbol', 'indexName'] if field in index and index[field]), None)
                    if index_name and index_name in NSE_INDEX_TO_SECTOR:
                        change_percent_str = next((str(index.get(field)) for field in ['pChange', 'percentChange', 'change', 'change_percent'] if index.get(field) is not None), None)
                        if change_percent_str:
                            try:
                                sectoral_performance.append({'index': index_name, 'sector': NSE_INDEX_TO_SECTOR[index_name], 'change_percent': float(change_percent_str)})
                            except (ValueError, TypeError): continue

                if sectoral_performance:
                    sectoral_performance.sort(key=lambda x: x['change_percent'], reverse=True)
                    self.best_sector_details = sectoral_performance[:3]
                    self.worst_sector_details = list(reversed(sectoral_performance[-3:]))
                    self.best_sectors = [s['sector'] for s in self.best_sector_details]
                    self.worst_sectors = [s['sector'] for s in self.worst_sector_details]
                    self.last_sectoral_update = datetime.now()
                    return True
        except requests.exceptions.ConnectionError:
            logger.warning("Could not connect to sectoral data API.")
        except Exception as e:
            logger.error(f"API sectoral data error: {e}")
        return False

    def force_sector_update(self):
        print(f"{Colors.YELLOW}🔄 Fetching live sectoral performance...{Colors.RESET}")
        if self.fetch_live_sectoral_performance():
            print(f"✅ API sectoral update successful!")
        else:
            print(f"{Colors.RED}❌ API sectoral update failed. Scanner will run without sectoral bias.{Colors.RESET}")

    def normalize_live_data(self, df, symbol):
        try:
            if df is None or df.empty: return None
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

    def check_gap_down(self, df):
        if df is None or len(df) < 2: return False
        try:
            return ((df['Open'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) <= -1.0
        except (IndexError, KeyError): return False

    # <<< --- NEW: CACHING LOGIC TO SPEED UP DATA FETCHING --- >>>
    def get_updated_data(self, symbol):
        if not td_hist: return None
        cache_path = os.path.join(self.cache_dir, f"{symbol}_1m.pkl")
        
        try:
            # Check if a valid cache file exists
            if os.path.exists(cache_path) and (time_module.time() - os.path.getmtime(cache_path)) < (self.cache_expiry_days * 86400):
                cached_df = pd.read_pickle(cache_path)
                last_timestamp = cached_df.index[-1]
                
                # Fetch only new data since the last timestamp
                # Adding a small buffer to ensure we don't miss the last bar
                start_date = last_timestamp - timedelta(minutes=5)
                new_data_df = td_hist.get_historic_data(symbol, start=start_date, bar_size='1 min')
                
                if new_data_df is not None and not new_data_df.empty:
                    normalized_new_df = self.normalize_live_data(new_data_df, symbol)
                    # Combine old and new data, remove duplicates
                    combined_df = pd.concat([cached_df, normalized_new_df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                    combined_df.to_pickle(cache_path)
                    return combined_df
                else:
                    return cached_df # Return old data if fetch fails
            else:
                # Fetch full history if cache is old or doesn't exist
                full_df_raw = td_hist.get_historic_data(symbol, duration='10 D', bar_size='1 min')
                if full_df_raw is not None and not full_df_raw.empty:
                    full_df = self.normalize_live_data(full_df_raw, symbol)
                    if full_df is not None:
                        full_df.to_pickle(cache_path)
                        return full_df
        except Exception as e:
            logger.error(f"Caching/Data fetch error for {symbol}: {e}")
        return None


    def process_stock(self, symbol):
        try:
            with self.api_lock: # Rate limit even cached requests to be safe
                current_time = time_module.time()
                while self.api_call_timestamps and self.api_call_timestamps[0] <= current_time - self.RATE_LIMIT_PERIOD:
                    self.api_call_timestamps.popleft()
                if len(self.api_call_timestamps) >= self.RATE_LIMIT_COUNT:
                    time_to_wait = (self.api_call_timestamps[0] + self.RATE_LIMIT_PERIOD) - current_time
                    if time_to_wait > 0: time_module.sleep(time_to_wait)
                self.api_call_timestamps.append(time_module.time())

            df_1m = self.get_updated_data(symbol) # Use the new caching function
            if df_1m is None or df_1m.empty: return None, False

            is_gap_down = self.check_gap_down(df_1m)
            if is_gap_down: return None, True

            ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
            timeframes_data = {}
            for tf in [5, 15, 30, 60]:
                df_resampled = df_1m.resample(f'{tf}T', label='right', closed='right').agg(ohlc_dict).dropna()
                if len(df_resampled) >= 20: timeframes_data[tf] = df_resampled.tail(150)
            
            if len(timeframes_data) >= 2:
                signal, score, squeeze = self.calculate_options_signals(symbol, timeframes_data)
                if "Explosive" in signal:
                    sector = next((s for s, st in SECTOR_STOCKS.items() if symbol in st), 'N/A')
                    return {'symbol': symbol, 'signal': signal, 'score': score, 'sector': sector, 'squeeze': squeeze}, False
        except Exception as e: logger.error(f"Processing error for {symbol}: {e}")
        return None, False

    def calculate_options_signals(self, symbol, timeframes_data):
        final_score, strongest_signal, squeeze_status = 50, "Neutral", "No Squeeze"
        try:
            for tf in [15, 30, 60]:
                if tf not in timeframes_data: continue
                indicators = OptionsReadyIndicators.calculate_all_indicators(timeframes_data[tf])
                if not indicators: continue
                latest = {name: ind.iloc[-1] for name, ind in indicators.items() if not ind.empty and pd.notna(ind.iloc[-1])}
                
                is_firing_up, is_firing_down = latest.get('squeeze_fire_up', False), latest.get('squeeze_fire_down', False)
                vol_surge_score, momentum_score, atr_accel = latest.get('Volume_Surge', 0), latest.get('Momentum', 50), latest.get('ATR_accel', 0) > 0

                sector = next((s for s, st in SECTOR_STOCKS.items() if symbol in st), 'N/A')
                sector_boost = 5 if sector in self.best_sectors else -5 if sector in self.worst_sectors else 0

                if is_firing_up:
                    score = 80 + sector_boost + (10 if momentum_score > 65 else 0) + (15 if vol_surge_score > 60 else 0) + (10 if atr_accel else 0)
                    if score > final_score: final_score, strongest_signal, squeeze_status = score, "Explosive Buy", f"{tf}m Squeeze FIRE UP"
                elif is_firing_down:
                    score = 20 + sector_boost - (10 if momentum_score < 35 else 0) - (15 if vol_surge_score > 60 else 0) - (10 if atr_accel else 0)
                    if score < final_score: final_score, strongest_signal, squeeze_status = score, "Explosive Sell", f"{tf}m Squeeze FIRE DOWN"
                elif latest.get('in_squeeze', False) and squeeze_status == "No Squeeze":
                    squeeze_status = f"{tf}m Squeeze Coiling"
            return strongest_signal, np.clip(final_score, 0, 100), squeeze_status
        except Exception as e:
            logger.error(f"Options signal calc error for {symbol}: {e}")
            return 'Neutral', 50, "Error"
        
    def display_signals(self, signals, scan_time):
        os.system('clear' if os.name == 'posix' else 'cls')
        print(f"\n{Colors.CYAN}{Colors.BOLD}🚀 OPTIONS BREAKOUT SCANNER - {datetime.now().strftime('%H:%M:%S')} IST{Colors.RESET}")
        print(f"⚡ Scan Time: {scan_time:.2f}s | 🚫 Gap-down Filtered: {self.gap_down_filtered_count}")
        
        if self.last_sectoral_update:
            best_str = ', '.join([f"{item['sector']} ({item['change_percent']:.2f}%)" for item in self.best_sector_details])
            print(f"🏆 Best Sectors: {Colors.GREEN}{best_str}{Colors.RESET}")
        
        if not signals: print(f"\n{Colors.YELLOW}📭 No high-conviction breakout signals found.{Colors.RESET}"); return

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
            while self.is_running:
                if not self.is_market_open():
                    print("Market is closed. Waiting for the next open session...", end="\r")
                    time_module.sleep(60)
                    continue
                
                start_time = time_module.time()
                self.force_sector_update()
                live_signals, gap_down_filtered = [], 0
                
                with ThreadPoolExecutor(max_workers=15) as executor:
                    futures = {executor.submit(self.process_stock, symbol): symbol for symbol in ALL_NSE_STOCKS}
                    for i, future in enumerate(as_completed(futures)):
                        print(f"Scanning... {i+1}/{len(ALL_NSE_STOCKS)}", end="\r")
                        try:
                            result, is_gap_down = future.result()
                            if is_gap_down: gap_down_filtered += 1
                            elif result: live_signals.append(result)
                        except Exception as e:
                            logger.error(f"Error getting result for a stock: {e}")

                self.gap_down_filtered_count = gap_down_filtered
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