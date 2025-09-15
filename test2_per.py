# Enhanced Multi-Indicator Scanner with 5-Minute Sectoral Detection & Debug Features
# Features: Automatic best/worst sector detection every 5 minutes with comprehensive debugging
# Added 60-minute and Daily timeframes for comprehensive analysis
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
    "NIFTY FINANCIAL SERVICES": "Finance",
    "NIFTY INFRASTRUCTURE": "Infrastructure",
    "BANKNIFTY": "Banking",
    "NIFTYAUTO": "Auto",
    "NIFTYIT": "Technology",
    "NIFTYPHARMA": "Pharma",
    "NIFTY CONSUMER DURABLES": "CONSUMER DURABLES",
    "NIFTY HEALTHCARE INDEX": "Healthcare",
    "NIFTY CAPITAL MARKETS": "Capital Market",
    "NIFTY PRIVATE BANK": "Private Bank",
    "NIFTY OIL & GAS": "OIL and GAS",
    "NIFTY INDIA DEFENCE": "DEFENCE",
    "NIFTY CORE HOUSING": "CORE HOUSING",
    "NIFTY SERVICES SECTOR": "SERVICES SECTOR",
    "NIFTY FINANCIAL SERVICES 25/50": "FINANCIAL SERVICES 25/50",
    "NIFTY INDIA TOURISM": "TOURISM",
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
    "Realty": ["DLF","LODHA","PRESTIGE","GODREJPROP","OBEROIRLTY","PHOENIXLTD","NCC","NBCC"],
    "PSE": ["BEL","BHEL","NHPC","GAIL","IOC","NTPC","POWERGRID","HINDPETRO","OIL","RECLTD","ONGC","NMDC","BPCL","HAL","RVNL","PFC","COALINDIA","IRCTC","IRFC",],
    "Commodities": ["AMBUJACEM","APLAPOLLO","ULTRACEMCO","SHREECEM","JSWSTEEL","HINDALCO","NHPC","IOC","NTPC","HINDPETRO","ADANIGREEN","OIL","VEDL","PIIND","ONGC","NMDC","UPL","BPCL","JSWENERGY","GRASIM","RELIANCE","TORNTPOWER","TATAPOWER","COALINDIA","PIDILITIND","SRF","ADANIENSOL","JINDALSTEL","TATASTEEL","HINDALCO"],
    "CONSUMER DURABLES": ["TITAN","DIXON","HAVELLS","CROMPTON","POLYCAB","EXIDEIND","AMBER","KAYNES","VOLTAS","PGEL","BLUESTARCO"],
    "Healthcare": ["SUNPHARMA","DIVISLAB","CIPLA","TORNTPHARM","MAXHEALTH","APOLLOHOSP","DRREDDY","MANKIND","ZYDUSLIFE","LUPIN","FORTIS","ALKEM","AUROPHARMA","GLENMARK","BIOCON","LAURUSLABS","SYNGENE","GRANULES"],
    "Capital Market": ["HDFCAMC","BSE","360ONE","MCX","CDSL","NUVAMA","ANGELONE","KFINTECH","CAMS","IEX"],
    "Private Bank": ["HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","YESBANK","IDFCFIRSTB","INDUSINDBK","FEDERALBNK","BANDHANBNK","RBLBANK"],
    "OIL and GAS": ["RELIANCE","ONGC","IOC","BPCL","GAIL","HINDPETRO","OIL","PETRONET","IGL"],
    "DEFENCE": ["HAL","BEL","SOLARINDS","MAZDOCK","BDL"],
    "CORE HOUSING": ["ULTRACEMCO","ASIANPAINT","GRASIM","DLF","AMBUJACEM","LODHA","DIXON","POLYCAB","SHREECEM","HAVELLS","PRESTIGE","GODREJPROP","OBEROIRLTY","PHOENIXLTD","VOLTAS","DALBHARAT","KEI","BLUESTARCO","LICHSGFIN","PNBHOUSING","CROMPTON"],
    "SERVICES SECTOR": ["HDFCBANK", "BHARTIARTL", "TCS", "ICICIBANK", "SBIN", "INFY", "BAJFINANCE", "HCLTECH", "KOTAKBANK", "AXISBANK", "BAJAJFINSV", "NTPC", "ZOMATO", "ADANIPORTS", "DMART", "POWERGRID", "WIPRO", "INDIGO", "JIOFINSERV", "SBILIFE", "HDFCLIFE", "LTIM", "TECHM", "TATAPOWER", "SHRIRAMFIN", "GAIL", "MAXHEALTH", "APOLLOHOSP", "NAUKRI", "INDUSINDBK"],
    "FINANCIAL SERVICES 25/50": ["HDFCBANK", "ICICIBANK", "SBIN", "BAJFINANCE", "KOTAKBANK", "AXISBANK", "BAJAJFINSV", "JIOFIN", "SBILIFE", "HDFCLIFE", "PFC", "CHOLAFIN", "HDFCAMC", "SHRIRAMFIN", "MUTHOOTFIN", "RECLTD", "ICICIGI", "ICICIPRULI", "SBICARD", "LICHSGFIN"],
    "TOURISM": ["INDIGO","INDHOTEL","IRCTC","JUBLFOOD"]
}

# --- ENHANCED TIMEFRAME & INDICATOR WEIGHTS ---
# Added 60min and Daily with higher weights for longer timeframes
TIMEFRAME_WEIGHTS = {5: 1.0, 15: 1.5, 30: 2.0, 60: 2.5, 'daily': 3.0}

# Enhanced indicator weights with consideration for longer timeframes
INDICATOR_WEIGHTS = {
    'RSI': 1.2, 'MACD': 1.4, 'Stochastic': 0.9, 'MA': 1.6,
    'ADX': 1.3, 'Bollinger': 1.1, 'ROC': 0.8, 'OBV': 1.4, 
    'CCI': 1.0, 'WWL': 0.9, 'EMA': 1.5, 'VWAP': 1.3
}


# --- ENHANCED TECHNICAL INDICATORS CLASS ---
class TechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        """Calculate all technical indicators including enhanced ones for longer timeframes"""
        indicators = {}
        if df is None or len(df) < 20:
            return indicators

        try:
            # 1. RSI (14-period)
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs))

            # 2. MACD (12,26,9)
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            indicators['MACD'] = macd_line - signal_line

            # 3. Stochastic (14-period)
            low14 = df['Low'].rolling(window=14).min()
            high14 = df['High'].rolling(window=14).max()
            indicators['Stochastic'] = 100 * (df['Close'] - low14) / (high14 - low14)

            # 4. Simple Moving Average (20-period)
            indicators['MA'] = df['Close'].rolling(window=20).mean()
            
            # 5. Exponential Moving Average (21-period) - Enhanced for longer timeframes
            indicators['EMA'] = df['Close'].ewm(span=21).mean()

            # 6. ADX (Average Directional Index) - 14 period
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

            # 7. Bollinger Bands Position (20,2)
            ma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            upper_band = ma20 + (std20 * 2)
            lower_band = ma20 - (std20 * 2)
            indicators['Bollinger'] = (df['Close'] - ma20) / (upper_band - lower_band) * 100

            # 8. Rate of Change (ROC) - 12 period
            indicators['ROC'] = df['Close'].pct_change(periods=12) * 100

            # 9. On-Balance Volume (OBV)
            obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            indicators['OBV'] = obv.pct_change(periods=10) * 100

            # 10. Commodity Channel Index (CCI) - 20 period
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            indicators['CCI'] = (typical_price - sma_tp) / (0.015 * mad)

            # 11. Williams %R (14-period)
            highest_high = df['High'].rolling(window=14).max()
            lowest_low = df['Low'].rolling(window=14).min()
            indicators['WWL'] = (highest_high - df['Close']) / (highest_high - lowest_low) * -100
            
            # 12. VWAP (Volume Weighted Average Price) - 20 period rolling
            if len(df) >= 20:
                typical_price_vwap = (df['High'] + df['Low'] + df['Close']) / 3
                vwap_numerator = (typical_price_vwap * df['Volume']).rolling(window=20).sum()
                vwap_denominator = df['Volume'].rolling(window=20).sum()
                indicators['VWAP'] = vwap_numerator / vwap_denominator

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

        return indicators


# --- ENHANCED LIVE SCANNER WITH 60MIN & DAILY TIMEFRAMES ---
class Enhanced5MinLiveScanner:
    def __init__(self):
        self.is_running = False
        self.current_signals = {}
        # Store top 2 best and worst sectors
        self.best_sectors = ["Technology", "Pharma"]
        self.worst_sectors = ["Auto", "Metal"]
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
        
        logger.info("🚀 Enhanced 5-Min Live Scanner with 60min & Daily timeframes initialized")
        
        # Initialize with debug checks
        self.show_initialization_status()

    def show_initialization_status(self):
        """Show initialization status and run initial checks"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}🚀 ENHANCED MULTI-TIMEFRAME SCANNER INITIALIZATION{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
        print(f"⏰ Timeframes: {Colors.YELLOW}5min, 15min, 30min, 60min, Daily{Colors.RESET}")
        
        # Show current sector status
        self.show_sector_status()
        
        # Test API connection
        self.test_api_connection()
        
        # Force initial sector update
        print(f"\n{Colors.YELLOW}🔄 Running initial sector update...{Colors.RESET}")
        self.force_sector_update()
        
        print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")

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
        print(f"🏆 Top 2 Best Sectors: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"📉 Top 2 Worst Sectors: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
        print(f"🕐 Last Update: {self.last_sectoral_update or 'Never'}")
        print(f"📈 Total Updates: {len(self.sectoral_history)}")
        print(f"🔄 Update Attempts: {self.sector_update_attempts}")
        print(f"✅ Successful Updates: {self.successful_updates}")
        
        if self.sectoral_history:
            print(f"\n📋 Recent Sector Changes (Last 3):")
            for i, entry in enumerate(self.sectoral_history[-3:]):
                time_str = entry['timestamp'].strftime('%H:%M:%S')
                best_display = ', '.join(entry['best'][:2])  # Show top 2
                worst_display = ', '.join(entry['worst'][:2])  # Show top 2
                print(f"  {i+1}. {time_str}: Best: {Colors.GREEN}{best_display}{Colors.RESET} | Worst: {Colors.RED}{worst_display}{Colors.RESET}")

    def force_sector_update(self):
        """Force a sector update with detailed logging"""
        print(f"\n{Colors.YELLOW}🔄 FORCING SECTOR UPDATE...{Colors.RESET}")
        self.sector_update_attempts += 1
        
        success = self.fetch_live_sectoral_performance_5min_debug()
        
        if success:
            self.successful_updates += 1
            print(f"✅ Update successful!")
            print(f"🏆 Top 2 Best: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
            print(f"📉 Top 2 Worst: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
        else:
            print(f"❌ Update failed - using defaults")
            print(f"🏆 Default Best: {Colors.YELLOW}{', '.join(self.best_sectors)}{Colors.RESET}")
            print(f"📉 Default Worst: {Colors.YELLOW}{', '.join(self.worst_sectors)}{Colors.RESET}")
        
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
                
                if isinstance(indices_data, str):
                    indices_data = json.loads(indices_data)
                    print(f"✓ Parsed string to JSON")
                    
                if isinstance(indices_data, dict):
                    if 'data' in indices_data: indices_data = indices_data['data']
                    elif 'indices' in indices_data: indices_data = indices_data['indices']
                    elif 'results' in indices_data: indices_data = indices_data['results']

                if not isinstance(indices_data, list):
                    logger.error("❌ Processed API data is not a list.")
                    return False
                
                sectoral_performance = []
                current_time = datetime.now()
                
                for index in indices_data:
                    if not isinstance(index, dict): continue
                    
                    index_name = next((str(index[field]).strip().upper() for field in ['name', 'symbol', 'index', 'indexName'] if field in index and index[field]), None)
                    
                    if index_name and index_name in NSE_INDEX_TO_SECTOR:
                        change_percent = 0.0
                        for field in ['change_percent', 'changePercent', 'pChange', 'percentChange', 'change', 'pchg']:
                            if field in index and index[field] is not None:
                                try:
                                    change_percent = float(index[field])
                                    break
                                except (ValueError, TypeError): continue
                        
                        sectoral_performance.append({
                            'index': index_name, 'sector': NSE_INDEX_TO_SECTOR[index_name],
                            'change_percent': change_percent, 'timestamp': current_time
                        })
                
                if sectoral_performance:
                    sectoral_performance.sort(key=lambda x: x['change_percent'], reverse=True)
                    
                    old_best = self.best_sectors[:]
                    old_worst = self.worst_sectors[:]
                    
                    # Update top 2 best and worst sectors
                    if len(sectoral_performance) >= 4:
                        self.best_sectors = [sectoral_performance[0]['sector'], sectoral_performance[1]['sector']]
                        self.worst_sectors = [sectoral_performance[-1]['sector'], sectoral_performance[-2]['sector']]
                    elif len(sectoral_performance) >= 2:
                        self.best_sectors = [sectoral_performance[0]['sector']]
                        self.worst_sectors = [sectoral_performance[-1]['sector']]
                    
                    self.last_sectoral_update = current_time
                    
                    self.sectoral_history.append({
                        'timestamp': current_time,
                        'best': self.best_sectors, 'worst': self.worst_sectors,
                        'full_data': sectoral_performance
                    })
                    
                    if len(self.sectoral_history) > 20:
                        self.sectoral_history = self.sectoral_history[-20:]
                    
                    self.display_sectoral_update(sectoral_performance, old_best, old_worst)
                    return True
                else:
                    print(f"❌ No sectoral data matched.")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error fetching sectoral data: {e}")
            self.api_errors.append(f"{datetime.now()}: {e}")
            return False

    def display_sectoral_update(self, sectoral_performance, old_best, old_worst):
        """Display enhanced sectoral performance update"""
        current_time = datetime.now()
        
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'='*100}")
        print(f"📊 SECTORAL PERFORMANCE UPDATE - {current_time.strftime('%H:%M:%S')} IST")
        print(f"{'='*100}{Colors.RESET}")
        
        print(f"🏆 Top 2 Best: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"📉 Top 2 Worst: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
        
        print(f"\n🔝 Top 5 Performing Sectors:")
        for i, sector_data in enumerate(sectoral_performance[:5]):
            color = Colors.YELLOW
            if sector_data['sector'] in self.best_sectors:
                color = Colors.GREEN + (Colors.BOLD if sector_data['sector'] == self.best_sectors[0] else "")
            print(f"  {i+1}. {color}{sector_data['sector']:<20}{Colors.RESET} {sector_data['change_percent']:+6.2f}% ({sector_data['index']})")
        
        print(f"\n🔻 Bottom 5 Performing Sectors:")
        for i, sector_data in enumerate(sectoral_performance[-5:]):
            color = Colors.YELLOW
            if sector_data['sector'] in self.worst_sectors:
                 color = Colors.RED + (Colors.BOLD if sector_data['sector'] == self.worst_sectors[0] else "")
            pos = len(sectoral_performance) - 5 + i + 1
            print(f"  {pos}. {color}{sector_data['sector']:<20}{Colors.RESET} {sector_data['change_percent']:+6.2f}% ({sector_data['index']})")
        
        print(f"{Colors.MAGENTA}{'='*100}{Colors.RESET}")

    def normalize_live_data(self, df, symbol):
        """Fast data normalization for all timeframes"""
        try:
            if df is None or df.empty: return None

            df_clean = df.copy()
            cols = [c.lower() for c in df_clean.columns]
            column_mapping = {df_clean.columns[i]: new_name for i, col in enumerate(cols)
                              for key, new_name in [('time', 'Date'), ('open', 'Open'), ('high', 'High'),
                                                    ('low', 'Low'), ('close', 'Close'), ('vol', 'Volume')]
                              if key in col}
            df_clean = df_clean.rename(columns=column_mapping)
            
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            if not all(col in df_clean.columns for col in required_cols): return None
            
            if 'Volume' not in df_clean.columns: df_clean['Volume'] = 1000
            
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            if df_clean['Date'].dt.tz is not None:
                df_clean['Date'] = df_clean['Date'].dt.tz_localize(None)
            df_clean.set_index('Date', inplace=True)
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
            return df_clean.dropna().sort_index() if len(df_clean) >= 20 else None
        except Exception as e:
            logger.error(f"Normalize error {symbol}: {e}")
            return None

    def fetch_live_data(self, symbol, timeframe):
        """Enhanced fetch live data for symbol and timeframe including 60min and daily"""
        try:
            # Enhanced timeframe mapping including 60min and EOD (daily)
            tf_map = {
                5: '5 min', 
                15: '15 min', 
                30: '30 min',
                60: '60 mins',  # 60-minute timeframe
                'daily': 'EOD'  # Daily/End of Day
            }
            bar_size = tf_map.get(timeframe)
            if not bar_size: return None

            # Enhanced duration mapping for different timeframes
            if timeframe == 5 or timeframe == 15:
                duration = '10 D'
            elif timeframe == 30:
                duration = '20 D' 
            elif timeframe == 60:
                duration = '60 D'  # 60 days for 60-minute data
            elif timeframe == 'daily':
                duration = '365 D'  # 1 year for daily data
            else:
                duration = '10 D'
                
            raw_df = td_hist.get_historic_data(symbol, duration=duration, bar_size=bar_size)

            if raw_df is not None and len(raw_df) > 0:
                normalized_df = self.normalize_live_data(raw_df, symbol)
                if normalized_df is not None and len(normalized_df) >= 20:
                    # Return appropriate data length for each timeframe
                    if timeframe == 'daily':
                        return normalized_df.tail(250)  # 250 days for daily
                    elif timeframe == 60:
                        return normalized_df.tail(200)  # 200 bars for 60min
                    else:
                        return normalized_df.tail(100)  # 100 bars for shorter timeframes
            return None
        except Exception as e:
            logger.error(f"Live data fetch error {symbol}_{timeframe}: {e}")
            return None

    def normalize_indicator_value(self, indicator_name, value):
        """Enhanced normalize indicator values to 0-100 scale"""
        try:
            if indicator_name == 'RSI': 
                return max(0, min(100, value))
            elif indicator_name == 'MACD': 
                return 50 + min(25, max(-25, value * 10))
            elif indicator_name == 'Stochastic': 
                return max(0, min(100, value))
            elif indicator_name == 'MA': 
                return 50
            elif indicator_name == 'EMA': 
                return 50  # Baseline for EMA
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
            elif indicator_name == 'VWAP': 
                return 50  # Baseline for VWAP
            else:
                return 50
        except: 
            return 50

    def calculate_multi_indicator_signals(self, symbol, timeframes_data):
        """Enhanced calculate signals using all technical indicators across multiple timeframes"""
        try:
            if not timeframes_data: return 'Neutral', 0

            sector = next((s for s, stocks in SECTOR_STOCKS.items() if symbol in stocks), None)
            if not sector: return 'Neutral', 0

            total_weighted_score, total_weight = 0, 0
            timeframe_scores = {}  # Track individual timeframe scores
            timeframe_details = []
            
            for tf, df in timeframes_data.items():
                if df is None or len(df) < 20: continue
                
                indicators = TechnicalIndicators.calculate_all_indicators(df)
                if not indicators: continue

                tf_score, tf_weight = 0, 0
                current_price = df['Close'].iloc[-1]
                
                # Calculate score for each indicator
                for name, weight in INDICATOR_WEIGHTS.items():
                    if name in indicators and indicators[name] is not None and not indicators[name].empty:
                        latest_val = indicators[name].iloc[-1]
                        if pd.notna(latest_val):
                            # Enhanced normalization for different indicators
                            if name in ['MA', 'EMA', 'VWAP']:
                                # For MA, EMA, VWAP compare with current price
                                if latest_val > 0:
                                    price_vs_ma = (current_price - latest_val) / latest_val * 100
                                    if price_vs_ma > 2:
                                        norm_score = 75  # Strong bullish
                                    elif price_vs_ma > 0:
                                        norm_score = 60  # Bullish
                                    elif price_vs_ma > -2:
                                        norm_score = 50  # Neutral
                                    elif price_vs_ma > -5:
                                        norm_score = 40  # Bearish
                                    else:
                                        norm_score = 25  # Strong bearish
                                else:
                                    norm_score = 50
                            else:
                                norm_score = self.normalize_indicator_value(name, latest_val)
                            
                            tf_score += norm_score * weight
                            tf_weight += weight
                
                if tf_weight > 0:
                    tf_final_score = tf_score / tf_weight
                    timeframe_scores[tf] = tf_final_score
                    timeframe_details.append(tf)
                    
                    # Apply timeframe weights
                    tf_multiplier = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                    total_weighted_score += tf_final_score * tf_multiplier
                    total_weight += tf_multiplier

            if total_weight == 0: return 'Neutral', 0
            base_score = total_weighted_score / total_weight

            # Enhanced sector boost for multi-timeframe analysis
            sector_boost = 0
            has_longer_tf = 'daily' in timeframes_data or 60 in timeframes_data
            
            if sector in self.best_sectors:
                if has_longer_tf:
                    sector_boost = 22 if sector == self.best_sectors[0] else 16
                else:
                    sector_boost = 15 if sector == self.best_sectors[0] else 10
            elif sector in self.worst_sectors:
                if has_longer_tf:
                    sector_boost = -22 if sector == self.worst_sectors[0] else -16
                else:
                    sector_boost = -15 if sector == self.worst_sectors[0] else -10
            
            base_score += sector_boost

            # Multi-timeframe confirmation bonus
            num_timeframes = len(timeframes_data)
            if num_timeframes >= 4:
                # Check for alignment across timeframes
                bullish_count = sum(1 for score in timeframe_scores.values() if score > 55)
                bearish_count = sum(1 for score in timeframe_scores.values() if score < 45)
                
                if bullish_count >= 3:
                    base_score += 5  # Multi-timeframe bullish alignment
                elif bearish_count >= 3:
                    base_score -= 5  # Multi-timeframe bearish alignment

            # Enhanced signal classification with multi-timeframe consideration
            if base_score >= 78: return 'Very Strong Buy', base_score
            elif base_score >= 68: return 'Strong Buy', base_score
            elif base_score >= 57: return 'Buy', base_score
            elif base_score <= 22: return 'Very Strong Sell', base_score
            elif base_score <= 32: return 'Strong Sell', base_score
            elif base_score <= 43: return 'Sell', base_score
            else: return 'Neutral', base_score
            
        except Exception as e:
            logger.error(f"Signal calculation error for {symbol}: {e}")
            return 'Neutral', 0

    def enhanced_5min_scan_cycle(self):
        """Enhanced 5-minute scan with 60min and daily timeframes"""
        if not self.is_market_open():
            logger.info("🕐 Market closed. Next scan in 5 minutes...")
            return

        start_time = time_module.time()
        current_time = datetime.now()
        
        print(f"\n{Colors.CYAN}🔄 Starting enhanced multi-timeframe scan at {current_time.strftime('%H:%M:%S')}{Colors.RESET}")
        print(f"⏰ Analyzing: {Colors.YELLOW}5min → 15min → 30min → 60min → Daily{Colors.RESET}")
        
        if not self.fetch_live_sectoral_performance_5min_debug():
            print(f"⚠️ Sectoral update failed, continuing with previous sectors")

        try:
            # Get target stocks from the top 2 best and top 2 worst sectors
            target_stocks_set = set()
            if len(self.best_sectors) > 0 and self.best_sectors[0] in SECTOR_STOCKS:
                target_stocks_set.update(SECTOR_STOCKS[self.best_sectors[0]][:10]) # 10 from best
            if len(self.best_sectors) > 1 and self.best_sectors[1] in SECTOR_STOCKS:
                target_stocks_set.update(SECTOR_STOCKS[self.best_sectors[1]][:8]) # 8 from 2nd best
            if len(self.worst_sectors) > 0 and self.worst_sectors[0] in SECTOR_STOCKS:
                target_stocks_set.update(SECTOR_STOCKS[self.worst_sectors[0]][:10]) # 10 from worst
            if len(self.worst_sectors) > 1 and self.worst_sectors[1] in SECTOR_STOCKS:
                target_stocks_set.update(SECTOR_STOCKS[self.worst_sectors[1]][:8]) # 8 from 2nd worst

            target_stocks = list(target_stocks_set)
            
            if not target_stocks:
                print(f"⚠️ No target stocks found.")
                return

            print(f"🎯 Multi-timeframe scanning {len(target_stocks)} stocks across 5 timeframes")
            
            live_signals = []
            with ThreadPoolExecutor(max_workers=3) as executor:  # Reduced workers for more API calls per stock
                def process_stock(symbol):
                    try:
                        timeframes_data = {}
                        
                        # Enhanced timeframe fetching including 60min and daily
                        timeframes_to_fetch = [5, 15, 30, 60, 'daily']
                        
                        for tf in timeframes_to_fetch:
                            df = self.fetch_live_data(symbol, tf)
                            if df is not None: 
                                timeframes_data[tf] = df
                            # Increased sleep to handle more API calls respectfully
                            time_module.sleep(1.0)

                        # Require at least 3 timeframes for enhanced analysis
                        if len(timeframes_data) >= 3:
                            signal, score = self.calculate_multi_indicator_signals(symbol, timeframes_data)
                            # Higher threshold for multi-timeframe signals
                            if abs(score - 50) > 15:
                                sector = next((s for s, st in SECTOR_STOCKS.items() if symbol in st), 'N/A')
                                return {
                                    'symbol': symbol, 'signal': signal, 'score': score, 'sector': sector,
                                    'timeframes': len(timeframes_data), 'timestamp': datetime.now(),
                                    'tf_details': list(timeframes_data.keys())
                                }
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                    return None

                futures = [executor.submit(process_stock, symbol) for symbol in target_stocks]
                live_signals = [future.result() for future in as_completed(futures) if future.result()]

            scan_time = time_module.time() - start_time
            logger.info(f"⚡ Enhanced multi-timeframe scan completed in {scan_time:.2f}s - {len(live_signals)} signals")
            self.display_enhanced_signals(live_signals, scan_time)

        except Exception as e:
            logger.error(f"Error in enhanced scan: {e}")

    def display_enhanced_signals(self, signals, scan_time):
        """Display enhanced signals with multi-timeframe context"""
        os.system('clear' if os.name == 'posix' else 'cls')
        current_time = datetime.now()
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*130}")
        print(f"🎯 ENHANCED MULTI-TIMEFRAME LIVE SCANNER - {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
        print(f"{'='*130}{Colors.RESET}")
        
        # Display timeframe analysis summary
        print(f"{Colors.BLUE}📊 Analysis: {Colors.YELLOW}5m{Colors.RESET} + {Colors.YELLOW}15m{Colors.RESET} + "
              f"{Colors.YELLOW}30m{Colors.RESET} + {Colors.CYAN}60m{Colors.RESET} + {Colors.MAGENTA}Daily{Colors.RESET}")
        
        # Display top 2 best/worst sectors
        if self.last_sectoral_update:
            best_str = ', '.join(self.best_sectors)
            worst_str = ', '.join(self.worst_sectors)
            print(f"{Colors.MAGENTA}📈 Sectoral Update:{Colors.RESET} {Colors.YELLOW}{self.last_sectoral_update.strftime('%H:%M:%S')}{Colors.RESET} | "
                  f"🏆 Top 2 Best: {Colors.GREEN}{Colors.BOLD}{best_str}{Colors.RESET} | "
                  f"📉 Top 2 Worst: {Colors.RED}{Colors.BOLD}{worst_str}{Colors.RESET}")
        
        print(f"{Colors.BLUE}📈 Updates:{Colors.RESET} {self.successful_updates}/{self.sector_update_attempts} successful | "
              f"⚡ Scan Time: {scan_time:.2f}s | 🎯 Multi-TF Stocks: {len(signals)}")

        if not signals:
            print(f"\n{Colors.YELLOW}📭 No significant multi-timeframe signals found in this cycle.{Colors.RESET}")
            print(f"{Colors.CYAN}💡 Signals require confirmation across at least 3 timeframes with 15+ score deviation{Colors.RESET}")
        else:
            print(f"\n{Colors.WHITE}{Colors.BOLD}🎯 {len(signals)} MULTI-TIMEFRAME CONFIRMED SIGNALS:{Colors.RESET}")
            print(f"\n{'Stock':<10} {'Sector':<18} {'Signal':<18} {'Score':>8} {'TFs':>4} {'TF Coverage':<20} {'Strength':<12}")
            print(f"{Colors.CYAN}{'-' * 130}{Colors.RESET}")

            # Sort by score deviation from neutral (50)
            signals.sort(key=lambda x: abs(x['score'] - 50), reverse=True)

            for s in signals[:30]:  # Show top 30 signals
                # Enhanced sector color coding for top 2
                sector_color, sector_name = Colors.YELLOW, s['sector']
                if s['sector'] in self.best_sectors:
                    sector_icon = "★★" if s['sector'] == self.best_sectors[0] else "★ "
                    sector_color, sector_name = Colors.GREEN, f"{sector_icon}{s['sector']}"
                elif s['sector'] in self.worst_sectors:
                    sector_icon = "★★" if s['sector'] == self.worst_sectors[0] else "★ "
                    sector_color, sector_name = Colors.RED, f"{sector_icon}{s['sector']}"

                # Signal color coding
                signal_color = Colors.YELLOW
                if 'Buy' in s['signal']: 
                    signal_color = Colors.GREEN + (Colors.BOLD if 'Very' in s['signal'] else "")
                elif 'Sell' in s['signal']: 
                    signal_color = Colors.RED + (Colors.BOLD if 'Very' in s['signal'] else "")
                
                # Strength calculation
                deviation = abs(s['score'] - 50)
                if deviation >= 35:
                    strength = f"{Colors.GREEN}{Colors.BOLD}Exceptional{Colors.RESET}"
                elif deviation >= 25:
                    strength = f"{Colors.GREEN}{Colors.BOLD}Very Strong{Colors.RESET}"
                elif deviation >= 20:
                    strength = f"{Colors.GREEN}Strong{Colors.RESET}"
                else:
                    strength = f"{Colors.YELLOW}Moderate{Colors.RESET}"
                
                # Timeframe coverage display
                tf_details = s.get('tf_details', [])
                tf_display = []
                for tf in tf_details:
                    if isinstance(tf, int):
                        tf_display.append(f"{tf}m")
                    else:
                        tf_display.append("D")  # Daily
                tf_coverage = ','.join(tf_display[:5])  # Show up to 5 timeframes

                print(f"{Colors.WHITE}{s['symbol']:<10}{Colors.RESET} "
                      f"{sector_color}{sector_name:<18}{Colors.RESET} "
                      f"{signal_color}{s['signal']:<18}{Colors.RESET} "
                      f"{Colors.WHITE}{s['score']:>8.1f}{Colors.RESET} "
                      f"{Colors.CYAN}{s['timeframes']:>4}{Colors.RESET} "
                      f"{Colors.MAGENTA}{tf_coverage:<20}{Colors.RESET} "
                      f"{strength}")
        
        next_scan_time = (current_time + timedelta(minutes=5)).strftime('%H:%M:%S')
        print(f"\n{Colors.CYAN}{Colors.BOLD}⏰ Next comprehensive multi-timeframe scan at {next_scan_time}{Colors.RESET}")
        print(f"{Colors.BLUE}💡 Multi-timeframe analysis provides higher accuracy with trend confirmation{Colors.RESET}")

    def run_enhanced_5min_scanner(self):
        """Main enhanced scanner with multi-timeframe analysis"""
        self.is_running = True
        logger.info("🚀 Starting Enhanced Multi-Timeframe Scanner...")
        
        self.force_sector_update() # Initial run
        
        try:
            while self.is_running:
                self.enhanced_5min_scan_cycle()
                if self.is_running:
                    logger.info(f"💤 Waiting 5 minutes for next comprehensive multi-timeframe cycle...")
                    time_module.sleep(self.scan_interval)
        except KeyboardInterrupt:
            logger.info("🛑 Enhanced multi-timeframe scanner stopped by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the scanner"""
        self.is_running = False
        print(f"{Colors.YELLOW}🛑 Multi-timeframe scanner stopped{Colors.RESET}")


# --- MAIN EXECUTION ---
def main():
    print(f"{Colors.CYAN}{Colors.BOLD}🎯 ENHANCED MULTI-TIMEFRAME SCANNER WITH COMPREHENSIVE ANALYSIS{Colors.RESET}")
    print(f"{Colors.YELLOW}📊 Timeframes: 5min, 15min, 30min, 60min, Daily (EOD){Colors.RESET}")
    print(f"{Colors.CYAN}🔧 Enhanced Features: Multi-TF confirmation, Sector boost, 12 indicators{Colors.RESET}")
    print(f"{Colors.MAGENTA}⚡ Updates every 5 minutes with sectoral performance tracking{Colors.RESET}")
    
    scanner = Enhanced5MinLiveScanner()
    try:
        scanner.run_enhanced_5min_scanner()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Shutting down multi-timeframe scanner...{Colors.RESET}")
        scanner.stop()


if __name__ == "__main__":
    main()
