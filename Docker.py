# Enhanced Multi-Indicator Scanner with 5-Minute Sectoral Detection & Debug Features
# Features: Automatic best/worst sector detection every 5 minutes with comprehensive debugging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import requests
import json
import time as time_module
import pytz
from logzero import logger
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from truedata.history import TD_hist
import logging
import warnings
warnings.filterwarnings('ignore')

# --- TRUEDATA CONFIG ---
TD_USERNAME = "Trial106"
TD_PASSWORD = "raj106"
td_hist = TD_hist(TD_USERNAME, TD_PASSWORD, log_level=logging.WARNING)

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

# --- NSE INDEX MAPPING ---
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
    "NIFTY CPSE": "CPSE",
    "NIFTY FINANCIAL SERVICES": "Finance",
    "NIFTY INFRASTRUCTURE": "Infrastructure",
    "BANKNIFTY": "Banking",
    "NIFTYFIN": "Finance",
    "NIFTYAUTO": "Auto",
    "NIFTYIT": "Technology",
    "NIFTYPHARMA": "Pharma",
    "NIFTY CONSUMER DURABLES": "CONSUMER DURABLES",
    "NIFTY HEALTHCARE INDEX": "Healthcare",
    "NIFTY CAPITAL MARKETS": "Capital Market",
    "NIFTY PRIVATE BANK": "Private Bank",
    "NIFTY OIL & GAS": "OIL and GAS",
    "NIFTY INDIA CONSUMPTION": "INDIA CONSUMPTION",
}

SECTOR_STOCKS = {
    "Technology": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "CYIENT", "KPITTECH", "TATAELXSI","SONACOMS","KAYNES","OFSS"],
    "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR","BHARATFORG", "EICHERMOT", "ASHOKLEY", "BOSCHLTD","TIINDIA","MOTHERSON"],
    "Banking": ["HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","PNB","BANKBARODA","CANBK","IDFCFIRSTB","INDUSINDBK","AUBANK","FEDERALBNK"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA", "TORNTPHARM","GLENMARK","ALKEM","LAURUSLABS","BIOCON","ZYDUSLIFE","MANKIND","SYNGENE","PPLPHARMA"],
    "Energy": ["RELIANCE","NTPC","BPCL","IOC","ONGC","GAIL","HINDPETRO","ADANIGREEN","ADANIENSOL","JSWENERGY","COALINDIA","TATAPOWER","SUZLON","PETRONET","OIL","POWERGRID","NHPC","ADANIPORTS","ABB","SIEMENS","CGPOWER","INOXWIND"],
    "Metal": ["TATASTEEL", "JSWSTEEL", "SAIL", "JINDALSTEL", "HINDALCO", "NMDC"],
    "Consumer": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR","AMBER","UNITDSPR","GODREJCP","MARICO","COLPAL","UPL","VBL"],
    "PSU Bank": ["SBIN", "PNB", "BANKBARODA", "CANBK", "UNIONBANK", "BANKINDIA"],
    "Finance": ["BAJFINANCE", "SHRIRAMFIN", "CHOLAFIN", "HDFCLIFE", "ICICIPRULI","ETERNAL",],
    "Realty": ["DLF","LODHA","PRESTIGE","GODREJPROP","OBEROIRLTY","PHOENIXLTD","BRIGADE","NCC","NBCC"],
    "CPSE": ["NHPC","NBCC","NTPC","POWERGRID","OIL","ONGC","COALINDIA","BEL"],
    "PSE": ["BEL","BHEL","NHPC","GAIL","IOC","NTPC","POWERGRID","NTPC","HINDPETRO","OIL","RECLTD","ONGC","NMDC","BPCL","HAL","RVNL","PFC","COALINDIA","IRCTC","IRFC",],
    "Commodities": ["AMBUJACEM","APLAPOLLO","ULTRACEMCO","SHREECEM","JSWSTEEL","HINDALCO","NHPC","IOC","NTPC","HINDPETRO","ADANIGREEN","OIL","VEDL","PIIND","ONGC","NMDC","UPL","BPCL","JSWENERGY","GRASIM","RELIANCE","TORNTPOWER","TATAPOWER","COALINDIA","PIDILITIND","SRF","ADANIENSOL","JINDALSTEL","TATASTEEL","HINDALCO"],
    "CONSUMER DURABLES": ["TITAN","DIXON","HAVELLS","CROMPTON","POLYCAB","EXIDEIND","AMBER","KAYNES","VOLTAS","PGEL","BLUESTARCO"],
    "Healthcare": ["SUNPHARMA","DIVISLAB","CIPLA","TORNTPHARM","MAXHEALTH","APOLLOHOSP","DRREDDY","MANKIND","ZYDUSLIFE","LUPIN","FORTIS","ALKEM","AUROPHARMA","GLENMARK","BIOCON","LAURUSLABS","SYNGENE","GRANULES"],
    "Capital Market": ["HDFCAMC","BSE","360ONE","MCX","CDSL","NUVAMA","ANGELONE","KFINTECH","CAMS","IEX"],
    "Private Bank": ["HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","YESBANK","IDFCFIRSTB","INDUSINDBK","FEDERALBNK","BANDHANBNK","RBLBANK"],
    "OIL and GAS": ["RELIANCE","ONGC","IOC","BPCL","GAIL","HINDPETRO","OIL","PETRONET","IGL"],
    "INDIA CONSUMPTION": ["BHARTIARTL","HINDUNILVR","ITC","MARUTI","M&M","TITAN","ETERNAL","DMART","BAJAJ-AUTO","ASIANPAINT","ADANIPOWER","NESTLEIND","INDIGO","DLF","EICHERMOT","TRENT","TVSMOTOR","VBL","BRITANNIA","GODREJCP","TATAPOWER","MAXHEALTH","APOLLOHOSP","INDHOTEL","TATACONSUM","HEROMOTOCO","HAVELLS","UNITDSPR","NAUKRI","COLPAL"]
}

# --- TIMEFRAME & INDICATOR WEIGHTS ---
TIMEFRAME_WEIGHTS = {5: 1.0, 15: 1.5, 30: 2.0}
INDICATOR_WEIGHTS = {
    'RSI': 1.0, 'MACD': 1.2, 'Stochastic': 0.8, 'MA': 1.5,
    'ADX': 1.2, 'Bollinger': 1.0, 'ROC': 0.7, 'OBV': 1.3, 'CCI': 0.9, 'WWL': 0.9
}

# --- TECHNICAL INDICATORS CLASS ---
class TechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        """Calculate all technical indicators"""
        indicators = {}
        if df is None or len(df) < 20:
            return indicators

        try:
            # 1. RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs))

            # 2. MACD
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            indicators['MACD'] = macd_line - signal_line

            # 3. Stochastic
            low14 = df['Low'].rolling(window=14).min()
            high14 = df['High'].rolling(window=14).max()
            indicators['Stochastic'] = 100 * (df['Close'] - low14) / (high14 - low14)

            # 4. Moving Average
            indicators['MA'] = df['Close'].rolling(window=20).mean()

            # 5. ADX
            high_diff = df['High'].diff()
            low_diff = df['Low'].diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = (-low_diff).where((low_diff < high_diff) & (low_diff < 0), 0)
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift())
            tr3 = abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            indicators['ADX'] = dx.rolling(window=14).mean()

            # 6. Bollinger Bands Position
            ma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            upper_band = ma20 + (std20 * 2)
            lower_band = ma20 - (std20 * 2)
            indicators['Bollinger'] = (df['Close'] - ma20) / (upper_band - lower_band) * 100

            # 7. ROC
            indicators['ROC'] = df['Close'].pct_change(periods=12) * 100

            # 8. OBV
            obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            indicators['OBV'] = obv.pct_change(periods=10) * 100

            # 9. CCI
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            indicators['CCI'] = (typical_price - sma_tp) / (0.015 * mad)

            # 10. Williams %R
            highest_high = df['High'].rolling(window=14).max()
            lowest_low = df['Low'].rolling(window=14).min()
            indicators['WWL'] = (highest_high - df['Close']) / (highest_high - lowest_low) * -100

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

        return indicators

# --- ENHANCED LIVE SCANNER WITH COMPLETE DEBUG FEATURES ---
class Enhanced5MinLiveScanner:
    def __init__(self):
        self.is_running = False
        self.current_signals = {}
        self.best_sector = "Technology"
        self.worst_sector = "Auto"
        self.sectoral_history = []
        self.last_sectoral_update = None
        self.api_errors = []
        self.sector_update_attempts = 0
        self.successful_updates = 0
        
        # Market hours
        self.market_start = time(9, 15)
        self.market_end = time(15, 30)
        
        # 5-minute intervals
        self.scan_interval = 300  # 5 minutes
        self.sectoral_update_interval = 300  # 5 minutes
        
        logger.info("🚀 Enhanced 5-Min Live Scanner with Debug Features initialized")
        
        # Initialize with debug checks
        self.show_initialization_status()

    def show_initialization_status(self):
        """Show initialization status and run initial checks"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}🚀 ENHANCED SCANNER INITIALIZATION{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*60}{Colors.RESET}")
        
        # Show current sector status
        self.show_sector_status()
        
        # Test API connection
        self.test_api_connection()
        
        # Force initial sector update
        print(f"\n{Colors.YELLOW}🔄 Running initial sector update...{Colors.RESET}")
        success = self.force_sector_update()
        
        print(f"{Colors.CYAN}{'='*60}{Colors.RESET}")

    def test_api_connection(self):
        """Test API connection and show response structure"""
        print(f"\n{Colors.BLUE}🔍 API CONNECTION TEST:{Colors.RESET}")
        try:
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            
            if response.status_code == 200:
                print(f"✅ API Connection: {Colors.GREEN}SUCCESS{Colors.RESET}")
                data = response.json()
                print(f"📊 Response Type: {type(data)}")
                
                if isinstance(data, list) and data:
                    print(f"📋 Items Count: {len(data)}")
                    print(f"🔍 First Item Keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
                    if isinstance(data[0], dict):
                        sample = data[0]
                        print(f"📝 Sample Item: {sample}")
                        
                elif isinstance(data, dict):
                    print(f"🗂️ Dict Keys: {list(data.keys())}")
                    
            else:
                print(f"❌ API Connection: {Colors.RED}FAILED{Colors.RESET} (Status: {response.status_code})")
                
        except Exception as e:
            print(f"❌ API Connection: {Colors.RED}ERROR{Colors.RESET} - {str(e)}")

    def show_sector_status(self):
        """Show current sector selection status"""
        print(f"\n{Colors.MAGENTA}📊 CURRENT SECTOR STATUS:{Colors.RESET}")
        print(f"🏆 Best Sector: {Colors.GREEN}{Colors.BOLD}{self.best_sector}{Colors.RESET}")
        print(f"📉 Worst Sector: {Colors.RED}{Colors.BOLD}{self.worst_sector}{Colors.RESET}")
        print(f"🕐 Last Update: {self.last_sectoral_update or 'Never'}")
        print(f"📈 Total Updates: {len(self.sectoral_history)}")
        print(f"🔄 Update Attempts: {self.sector_update_attempts}")
        print(f"✅ Successful Updates: {self.successful_updates}")
        
        if self.sectoral_history:
            print(f"\n📋 Recent Sector Changes (Last 3):")
            for i, entry in enumerate(self.sectoral_history[-3:]):
                time_str = entry['timestamp'].strftime('%H:%M:%S')
                print(f"  {i+1}. {time_str}: {Colors.GREEN}{entry['best']}{Colors.RESET} (+{entry['best_change']:.1f}%) | {Colors.RED}{entry['worst']}{Colors.RESET} ({entry['worst_change']:.1f}%)")

    def force_sector_update(self):
        """Force a sector update with detailed logging"""
        print(f"\n{Colors.YELLOW}🔄 FORCING SECTOR UPDATE...{Colors.RESET}")
        self.sector_update_attempts += 1
        
        success = self.fetch_live_sectoral_performance_5min_debug()
        
        if success:
            self.successful_updates += 1
            print(f"✅ Update successful!")
            print(f"🏆 Best: {Colors.GREEN}{Colors.BOLD}{self.best_sector}{Colors.RESET}")
            print(f"📉 Worst: {Colors.RED}{Colors.BOLD}{self.worst_sector}{Colors.RESET}")
        else:
            print(f"❌ Update failed - using defaults")
            print(f"🏆 Default Best: {Colors.YELLOW}{self.best_sector}{Colors.RESET}")
            print(f"📉 Default Worst: {Colors.YELLOW}{self.worst_sector}{Colors.RESET}")
        
        return success

    def is_market_open(self):
        """Check if market is open"""
        now = datetime.now()
        current_time = now.time()
        
        if now.weekday() > 4:  # Weekend
            return False
            
        return self.market_start <= current_time <= self.market_end

    def fetch_live_sectoral_performance_5min_debug(self):
        """Enhanced sectoral performance fetching with comprehensive debugging"""
        try:
            logger.info("🔍 Fetching live sectoral performance (5-min update with debug)...")
            
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            print(f"\n{Colors.BLUE}📡 API RESPONSE DEBUG:{Colors.RESET}")
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                indices_data = response.json()
                print(f"Response Type: {type(indices_data)}")
                
                # Handle different response formats with detailed logging
                original_type = type(indices_data).__name__
                
                if isinstance(indices_data, str):
                    indices_data = json.loads(indices_data)
                    print(f"✓ Parsed string to JSON")
                    
                if isinstance(indices_data, dict):
                    available_keys = list(indices_data.keys())
                    print(f"Available keys: {available_keys}")
                    
                    if 'data' in indices_data:
                        indices_data = indices_data['data']
                        print(f"✓ Using 'data' key")
                    elif 'indices' in indices_data:
                        indices_data = indices_data['indices']
                        print(f"✓ Using 'indices' key")
                    elif 'results' in indices_data:
                        indices_data = indices_data['results']
                        print(f"✓ Using 'results' key")
                    else:
                        print(f"⚠️ No standard key found, using raw dict")
                
                print(f"Final data type: {type(indices_data)}")
                
                if isinstance(indices_data, list):
                    print(f"Data length: {len(indices_data)}")
                    if indices_data:
                        print(f"First item type: {type(indices_data[0])}")
                        if isinstance(indices_data[0], dict):
                            print(f"First item keys: {list(indices_data[0].keys())}")
                            print(f"Sample item: {indices_data[0]}")
                
                sectoral_performance = []
                current_time = datetime.now()
                
                matched_sectors = 0
                unmatched_indices = []
                
                print(f"\n{Colors.BLUE}🔍 INDEX MATCHING DEBUG:{Colors.RESET}")
                
                for i, index in enumerate(indices_data):
                    if not isinstance(index, dict):
                        print(f"⚠️ Item {i} is not a dict: {type(index)}")
                        continue
                    
                    # Try different possible field names for index name
                    possible_name_fields = ['name', 'symbol', 'index', 'indexName', 'Symbol', 'Name']
                    index_name = None
                    
                    for field in possible_name_fields:
                        if field in index and index[field]:
                            index_name = str(index[field]).strip().upper()
                            break
                    
                    if not index_name:
                        print(f"⚠️ No name field found in item {i}: {index}")
                        continue
                    
                    print(f"Checking: '{index_name}'", end="")
                    
                    if index_name in NSE_INDEX_TO_SECTOR:
                        # Try different possible field names for change percentage
                        possible_change_fields = ['change_percent', 'changePercent', 'pChange', 'percentChange', 'change', 'pchg']
                        change_percent = 0
                        
                        for field in possible_change_fields:
                            if field in index and index[field] is not None:
                                try:
                                    change_percent = float(index[field])
                                    break
                                except (ValueError, TypeError):
                                    continue
                        
                        sector_info = {
                            'index': index_name,
                            'sector': NSE_INDEX_TO_SECTOR[index_name],
                            'change_percent': change_percent,
                            'timestamp': current_time
                        }
                        sectoral_performance.append(sector_info)
                        matched_sectors += 1
                        print(f" → ✅ {Colors.GREEN}{NSE_INDEX_TO_SECTOR[index_name]}{Colors.RESET} ({change_percent:+.2f}%)")
                    else:
                        unmatched_indices.append(index_name)
                        print(f" → ❌ No match")
                
                print(f"\n📊 MATCHING SUMMARY:")
                print(f"✅ Matched sectors: {Colors.GREEN}{matched_sectors}{Colors.RESET}")
                print(f"❌ Unmatched indices: {len(unmatched_indices)}")
                
                if unmatched_indices:
                    print(f"Unmatched: {unmatched_indices[:5]}{'...' if len(unmatched_indices) > 5 else ''}")
                
                if sectoral_performance:
                    # Sort by performance
                    sectoral_performance.sort(key=lambda x: x['change_percent'], reverse=True)
                    
                    # Update best and worst sectors
                    old_best = self.best_sector
                    old_worst = self.worst_sector
                    
                    self.best_sector = sectoral_performance[0]['sector']
                    self.worst_sector = sectoral_performance[-1]['sector']
                    self.last_sectoral_update = current_time
                    
                    print(f"\n🏆 NEW SECTOR LEADERS:")
                    print(f"Best: {Colors.GREEN}{Colors.BOLD}{self.best_sector}{Colors.RESET} ({sectoral_performance[0]['change_percent']:+.2f}%)")
                    print(f"Worst: {Colors.RED}{Colors.BOLD}{self.worst_sector}{Colors.RESET} ({sectoral_performance[-1]['change_percent']:+.2f}%)")
                    
                    # Store in history
                    self.sectoral_history.append({
                        'timestamp': current_time,
                        'best': self.best_sector,
                        'worst': self.worst_sector,
                        'best_change': sectoral_performance[0]['change_percent'],
                        'worst_change': sectoral_performance[-1]['change_percent'],
                        'full_data': sectoral_performance
                    })
                    
                    # Keep only last 20 updates
                    if len(self.sectoral_history) > 20:
                        self.sectoral_history = self.sectoral_history[-20:]
                    
                    # Display sectoral update
                    self.display_sectoral_update(sectoral_performance, old_best, old_worst)
                    return True
                else:
                    print(f"❌ No sectoral data matched - check index names and mapping")
                    return False
                    
        except Exception as e:
            error_msg = f"Error fetching sectoral data: {e}"
            logger.error(f"❌ {error_msg}")
            self.api_errors.append(f"{datetime.now()}: {error_msg}")
            print(f"❌ Exception: {str(e)}")
            return False

    def display_sectoral_update(self, sectoral_performance, old_best, old_worst):
        """Display enhanced sectoral performance update"""
        current_time = datetime.now()
        
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'='*100}")
        print(f"📊 SECTORAL PERFORMANCE UPDATE - {current_time.strftime('%H:%M:%S')} IST")
        print(f"{'='*100}{Colors.RESET}")
        
        # Show sector changes with enhanced formatting
        if old_best != self.best_sector:
            print(f"🔄 Best Sector Changed: {Colors.YELLOW}{old_best}{Colors.RESET} → {Colors.GREEN}{Colors.BOLD}{self.best_sector}{Colors.RESET}")
        else:
            print(f"🏆 Best Sector Remains: {Colors.GREEN}{Colors.BOLD}{self.best_sector}{Colors.RESET}")
            
        if old_worst != self.worst_sector:
            print(f"🔄 Worst Sector Changed: {Colors.YELLOW}{old_worst}{Colors.RESET} → {Colors.RED}{Colors.BOLD}{self.worst_sector}{Colors.RESET}")
        else:
            print(f"📉 Worst Sector Remains: {Colors.RED}{Colors.BOLD}{self.worst_sector}{Colors.RESET}")
        
        # Show top 5 and bottom 5 sectors with enhanced display
        print(f"\n🔝 Top 5 Performing Sectors:")
        for i, sector_data in enumerate(sectoral_performance[:5]):
            if i == 0:
                color = f"{Colors.GREEN}{Colors.BOLD}"
            elif i == 1:
                color = Colors.GREEN
            else:
                color = Colors.YELLOW
            
            print(f"   {i+1}. {color}{sector_data['sector']:<15}{Colors.RESET} {sector_data['change_percent']:+6.2f}% ({sector_data['index']})")
        
        print(f"\n🔻 Bottom 5 Performing Sectors:")
        bottom_sectors = sectoral_performance[-5:]
        for i, sector_data in enumerate(bottom_sectors):
            if i == len(bottom_sectors) - 1:  # Worst sector
                color = f"{Colors.RED}{Colors.BOLD}"
            elif i == len(bottom_sectors) - 2:  # Second worst
                color = Colors.RED
            else:
                color = Colors.YELLOW
            
            pos = len(sectoral_performance) - len(bottom_sectors) + i + 1
            print(f"   {pos}. {color}{sector_data['sector']:<15}{Colors.RESET} {sector_data['change_percent']:+6.2f}% ({sector_data['index']})")
        
        print(f"{Colors.MAGENTA}{'='*100}{Colors.RESET}")

    def normalize_live_data(self, df, symbol):
        """Fast data normalization"""
        try:
            if df is None or df.empty:
                return None

            df_clean = df.copy()
            cols = [c.lower() for c in df_clean.columns]
            column_mapping = {}

            for i, col in enumerate(cols):
                original_col = df_clean.columns[i]
                if any(x in col for x in ['timestamp', 'date', 'time']):
                    column_mapping[original_col] = 'Date'
                elif 'open' in col:
                    column_mapping[original_col] = 'Open'
                elif 'high' in col:
                    column_mapping[original_col] = 'High'
                elif 'low' in col:
                    column_mapping[original_col] = 'Low'
                elif 'close' in col:
                    column_mapping[original_col] = 'Close'
                elif 'volume' in col:
                    column_mapping[original_col] = 'Volume'

            df_clean = df_clean.rename(columns=column_mapping)

            required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            if not all(col in df_clean.columns for col in required_cols):
                return None

            if 'Volume' not in df_clean.columns:
                df_clean['Volume'] = 1000

            df_clean = df_clean[required_cols + ['Volume']].copy()
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')

            if df_clean['Date'].dt.tz is not None:
                df_clean['Date'] = df_clean['Date'].dt.tz_localize(None)

            df_clean.set_index('Date', inplace=True)

            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

            df_clean = df_clean.dropna().sort_index()
            return df_clean if len(df_clean) >= 20 else None

        except Exception as e:
            logger.error(f"Normalize error {symbol}: {e}")
            return None

    def fetch_live_data(self, symbol, timeframe):
        """Fetch live data for symbol and timeframe"""
        try:
            tf_map = {5: '5 min', 15: '15 min', 30: '30 min'}
            bar_size = tf_map.get(timeframe)
            if not bar_size:
                return None

            duration = '10 D' if timeframe <= 15 else '20 D'
            raw_df = td_hist.get_historic_data(symbol, duration=duration, bar_size=bar_size)

            if raw_df is not None and len(raw_df) > 0:
                normalized_df = self.normalize_live_data(raw_df, symbol)
                if normalized_df is not None and len(normalized_df) >= 20:
                    return normalized_df.tail(100)

            return None

        except Exception as e:
            logger.error(f"Live data fetch error {symbol}_{timeframe}min: {e}")
            return None

    def normalize_indicator_value(self, indicator_name, value):
        """Normalize indicator values to 0-100 scale"""
        try:
            if indicator_name == 'RSI':
                return max(0, min(100, value))
            elif indicator_name == 'MACD':
                return 50 + min(25, max(-25, value * 10))
            elif indicator_name == 'Stochastic':
                return max(0, min(100, value))
            elif indicator_name == 'MA':
                return 50
            elif indicator_name == 'ADX':
                return max(0, min(100, value))
            elif indicator_name == 'Bollinger':
                return max(0, min(100, (value + 100) / 2))
            elif indicator_name == 'ROC':
                return 50 + min(25, max(-25, value * 2))
            elif indicator_name == 'OBV':
                return 50 + min(25, max(-25, value))
            elif indicator_name == 'CCI':
                return max(0, min(100, (value + 200) / 4))
            elif indicator_name == 'WWL':
                return max(0, min(100, value + 100))
            else:
                return 50
        except Exception:
            return 50

    def calculate_multi_indicator_signals(self, symbol, timeframes_data):
        """Calculate signals using all technical indicators"""
        try:
            if not timeframes_data or len(timeframes_data) < 2:
                return 'Neutral', 0

            # Determine sector
            sector = None
            for sect_name, stock_list in SECTOR_STOCKS.items():
                if symbol in stock_list:
                    sector = sect_name
                    break

            if not sector:
                return 'Neutral', 0

            total_weighted_score = 0
            total_weight = 0

            for tf, df in timeframes_data.items():
                if df is None or len(df) < 20:
                    continue

                tf_weight = TIMEFRAME_WEIGHTS.get(tf, 1.0)

                # Calculate all indicators
                indicators = TechnicalIndicators.calculate_all_indicators(df)
                if not indicators:
                    continue

                tf_indicator_score = 0
                tf_indicator_weight = 0

                for indicator_name, indicator_weight in INDICATOR_WEIGHTS.items():
                    if indicator_name in indicators:
                        indicator_series = indicators[indicator_name]
                        if indicator_series is not None and len(indicator_series) > 0:
                            latest_value = indicator_series.iloc[-1]
                            if pd.notna(latest_value):
                                normalized_score = self.normalize_indicator_value(indicator_name, latest_value)
                                tf_indicator_score += normalized_score * indicator_weight
                                tf_indicator_weight += indicator_weight

                if tf_indicator_weight > 0:
                    tf_avg_score = tf_indicator_score / tf_indicator_weight
                    total_weighted_score += tf_avg_score * tf_weight
                    total_weight += tf_weight

            if total_weight == 0:
                return 'Neutral', 0

            base_score = total_weighted_score / total_weight

            # ENHANCED SECTOR ADJUSTMENT WITH 5-MIN UPDATES
            sector_boost = 0
            if sector == self.best_sector:
                sector_boost = 15
                base_score += sector_boost
            elif sector == self.worst_sector:
                sector_boost = -15
                base_score += sector_boost

            # Signal classification
            if base_score >= 70:
                return 'Very Strong Buy', base_score
            elif base_score >= 60:
                return 'Strong Buy', base_score
            elif base_score >= 55:
                return 'Buy', base_score
            elif base_score <= 30:
                return 'Very Strong Sell', base_score
            elif base_score <= 40:
                return 'Strong Sell', base_score
            elif base_score <= 45:
                return 'Sell', base_score
            else:
                return 'Neutral', base_score

        except Exception as e:
            logger.error(f"Signal calculation error for {symbol}: {e}")
            return 'Neutral', 0

    def enhanced_5min_scan_cycle(self):
        """Enhanced 5-minute scan with sectoral updates"""
        if not self.is_market_open():
            logger.info("🕐 Market closed. Next scan in 5 minutes...")
            return

        start_time = time_module.time()
        current_time = datetime.now()

        # ALWAYS UPDATE SECTORAL PERFORMANCE EVERY 5 MINUTES
        print(f"\n{Colors.CYAN}🔄 Starting enhanced scan cycle at {current_time.strftime('%H:%M:%S')}{Colors.RESET}")
        
        sectoral_success = self.fetch_live_sectoral_performance_5min_debug()
        
        if not sectoral_success:
            print(f"⚠️ Sectoral update failed, continuing with previous sectors")

        logger.info(f"🔄 Enhanced scan with sectoral update at {current_time.strftime('%H:%M:%S')}")

        try:
            # Get target stocks from updated sectors
            target_stocks = []
            if self.best_sector in SECTOR_STOCKS:
                target_stocks.extend(SECTOR_STOCKS[self.best_sector][:10])
            if self.worst_sector in SECTOR_STOCKS and self.worst_sector != self.best_sector:
                target_stocks.extend(SECTOR_STOCKS[self.worst_sector][:8])

            if not target_stocks:
                print(f"⚠️ No target stocks found for sectors: {self.best_sector}, {self.worst_sector}")
                return

            print(f"🎯 Scanning {len(target_stocks)} stocks from {self.best_sector} and {self.worst_sector} sectors")
            
            live_signals = []

            # Multi-threaded processing
            with ThreadPoolExecutor(max_workers=6) as executor:
                def process_stock(symbol):
                    try:
                        timeframes_data = {}
                        for tf in [5, 15, 30]:
                            df = self.fetch_live_data(symbol, tf)
                            if df is not None:
                                timeframes_data[tf] = df
                            time_module.sleep(0.05)

                        if len(timeframes_data) >= 2:
                            signal, score = self.calculate_multi_indicator_signals(symbol, timeframes_data)
                            if abs(score - 50) > 10:  # Only significant signals
                                sector = None
                                for sect_name, stock_list in SECTOR_STOCKS.items():
                                    if symbol in stock_list:
                                        sector = sect_name
                                        break

                                return {
                                    'symbol': symbol,
                                    'signal': signal,
                                    'score': score,
                                    'sector': sector,
                                    'timeframes': len(timeframes_data),
                                    'timestamp': datetime.now()
                                }

                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                    return None

                futures = [executor.submit(process_stock, symbol) for symbol in target_stocks]
                
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        live_signals.append(result)

            scan_time = time_module.time() - start_time
            logger.info(f"⚡ Enhanced scan completed in {scan_time:.2f}s - {len(live_signals)} signals")

            self.display_enhanced_signals(live_signals, scan_time)

        except Exception as e:
            logger.error(f"Error in enhanced scan: {e}")

    def display_enhanced_signals(self, signals, scan_time):
        """Display enhanced signals with sectoral context"""
        current_time = datetime.now()

        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')

        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*110}")
        print(f"🎯 ENHANCED MULTI-INDICATOR LIVE SCANNER - {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
        print(f"{'='*110}{Colors.RESET}")

        # Show current sectoral leaders with enhanced formatting
        if self.last_sectoral_update:
            update_time = self.last_sectoral_update.strftime('%H:%M:%S')
            print(f"{Colors.MAGENTA}📊 Sectoral Update:{Colors.RESET} {Colors.YELLOW}{update_time}{Colors.RESET} | "
                  f"🏆 Best: {Colors.GREEN}{Colors.BOLD}{self.best_sector}{Colors.RESET} | "
                  f"📉 Worst: {Colors.RED}{Colors.BOLD}{self.worst_sector}{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}⚠️ No sectoral update yet - using defaults{Colors.RESET}")

        # Show update statistics
        print(f"{Colors.BLUE}📈 Updates:{Colors.RESET} {self.successful_updates}/{self.sector_update_attempts} successful | "
              f"⚡ Scan Time: {scan_time:.2f}s | "
              f"🔄 Timeframes: 5min(1.0x), 15min(1.5x), 30min(2.0x)")
        
        print(f"{Colors.BLUE}🔧 Indicators:{Colors.RESET} RSI(1.0), MACD(1.2), Stoch(0.8), MA(1.5), ADX(1.2), BB(1.0), ROC(0.7), OBV(1.3), CCI(0.9), WWL(0.9)")

        if not signals:
            print(f"\n{Colors.YELLOW}📭 No significant signals (Score deviation < 10 from neutral){Colors.RESET}")
        else:
            print(f"\n{Colors.WHITE}{Colors.BOLD}🎯 {len(signals)} MULTI-INDICATOR SIGNALS (Updated every 5 minutes):{Colors.RESET}")
            print(f"\n{'Stock':<10} {'Sector':<12} {'Signal':<18} {'Score':>8} {'TFs':>4} {'Strength':<12}")
            print(f"{Colors.CYAN}{'-' * 95}{Colors.RESET}")

            signals.sort(key=lambda x: abs(x['score'] - 50), reverse=True)

            for i, s in enumerate(signals[:15]):
                # Enhanced sector colors with indicators
                if s['sector'] == self.best_sector:
                    sector_color = f"{Colors.GREEN}★{Colors.RESET}{Colors.GREEN}"
                elif s['sector'] == self.worst_sector:
                    sector_color = f"{Colors.RED}★{Colors.RESET}{Colors.RED}"
                else:
                    sector_color = Colors.YELLOW

                # Signal colors
                if 'Very Strong Buy' in s['signal']:
                    signal_color = f"{Colors.GREEN}{Colors.BOLD}"
                elif 'Buy' in s['signal']:
                    signal_color = Colors.GREEN
                elif 'Very Strong Sell' in s['signal']:
                    signal_color = f"{Colors.RED}{Colors.BOLD}"
                elif 'Sell' in s['signal']:
                    signal_color = Colors.RED
                else:
                    signal_color = Colors.YELLOW

                # Enhanced strength indicators
                deviation = abs(s['score'] - 50)
                if deviation >= 25:
                    strength = f"{Colors.GREEN}{Colors.BOLD}Very Strong{Colors.RESET}"
                elif deviation >= 20:
                    strength = f"{Colors.GREEN}Strong{Colors.RESET}"
                elif deviation >= 15:
                    strength = f"{Colors.YELLOW}Strong{Colors.RESET}"
                else:
                    strength = f"{Colors.BLUE}Moderate{Colors.RESET}"

                print(f"{Colors.WHITE}{s['symbol']:<10}{Colors.RESET} "
                      f"{sector_color}{s['sector']:<12}{Colors.RESET} "
                      f"{signal_color}{s['signal']:<18}{Colors.RESET} "
                      f"{Colors.WHITE}{s['score']:>8.1f}{Colors.RESET} "
                      f"{Colors.CYAN}{s['timeframes']:>4}{Colors.RESET} "
                      f"{strength}")

        next_scan_time = (current_time + timedelta(minutes=5)).strftime('%H:%M:%S')
        print(f"\n{Colors.CYAN}{Colors.BOLD}⏰ Next scan + sectoral update at {next_scan_time} | "
              f"📊 Sectoral detection frequency: Every 5 minutes{Colors.RESET}")

        # Show any recent API errors
        if self.api_errors:
            print(f"\n{Colors.RED}⚠️ Recent API Errors: {len(self.api_errors)}{Colors.RESET}")

    def run_enhanced_5min_scanner(self):
        """Main enhanced scanner with 5-minute sectoral detection"""
        self.is_running = True
        logger.info("🚀 Starting Enhanced 5-Minute Scanner with Sectoral Detection...")

        # Initial sectoral update
        print(f"\n{Colors.YELLOW}🔄 Initial sectoral performance update...{Colors.RESET}")
        self.fetch_live_sectoral_performance_5min_debug()

        try:
            while self.is_running:
                # BOTH SCAN AND SECTORAL UPDATE EVERY 5 MINUTES
                self.enhanced_5min_scan_cycle()

                if self.is_running:
                    # Wait exactly 5 minutes
                    logger.info(f"💤 Waiting 5 minutes for next scan + sectoral update...")
                    time_module.sleep(self.scan_interval)

        except KeyboardInterrupt:
            logger.info("🛑 Enhanced scanner stopped by user")
        except Exception as e:
            logger.error(f"❌ Enhanced scanner error: {e}")
        finally:
            self.is_running = False

    def stop(self):
        """Stop the scanner"""
        self.is_running = False
        print(f"{Colors.YELLOW}🛑 Scanner stopped{Colors.RESET}")

# --- MAIN EXECUTION ---
def main():
    """Start enhanced scanner with 5-minute sectoral detection"""
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("🎯 ENHANCED MULTI-INDICATOR SCANNER WITH 5-MIN SECTORAL DETECTION & DEBUG")
    print("=" * 90)
    print("Key Features:")
    print("✓ Automatic sectoral performance detection every 5 minutes")
    print("✓ Best/worst sector identification with live updates")
    print("✓ Complete technical indicator suite (10 indicators)")
    print("✓ Multi-timeframe analysis (5min, 15min, 30min)")
    print("✓ Real-time NSE API integration with debugging")
    print("✓ Enhanced sector-aware signal generation")
    print("✓ Sectoral change notifications with visual indicators")
    print("✓ Performance history tracking")
    print("✓ Comprehensive error tracking and debugging")
    print("✓ API response structure analysis")
    print("=" * 90)
    print("Update Frequency:")
    print("• Stock signals: Every 5 minutes")
    print("• Sectoral performance: Every 5 minutes (synchronized)")
    print("• Market hours: 9:15 AM - 3:30 PM IST")
    print("• Debug information: Real-time")
    print("=" * 90)
    print(f"{Colors.RESET}")

    scanner = Enhanced5MinLiveScanner()
    
    try:
        scanner.run_enhanced_5min_scanner()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Shutting down enhanced scanner...{Colors.RESET}")
        scanner.stop()

if __name__ == "__main__":
    main()
