# ENHANCED REAL-TIME 3-SECTOR SCANNER WITH SMART RATE LIMITING
# Features: ATR, Volume Surge, Momentum + API Sectoral Data + Thread-Safe Rate Limiter
# MODIFIED: Scans a fixed master list, uses 5/15/30 min timeframes, displays sectoral index names, and respects API rate limits.
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
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from truedata.history import TD_hist
import logging
import warnings
warnings.filterwarnings('ignore')

# --- TRUEDATA CONFIG ---
TD_USERNAME = "tdwsp751"
TD_PASSWORD = "raj@751"
td_hist = TD_hist(TD_USERNAME, TD_PASSWORD, log_level=logging.WARNING)

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

# --- ENHANCED INDICATOR WEIGHTS (YOUR EXACT SPECIFICATIONS) ---
ENHANCED_INDICATOR_WEIGHTS = {
    'RSI': 1.3, 'MACD': 1.6, 'Stochastic': 1.0, 'MA': 1.8,
    'ADX': 1.5, 'Bollinger': 1.4, 'ROC': 1.2, 'OBV': 1.6,
    'CCI': 1.1, 'WWL': 1.0, 'EMA': 1.7, 'VWAP': 1.5,
    'ATR': 1.4, 'Volume_Surge': 2.0, 'Momentum': 1.9
}

# --- TIMEFRAME WEIGHTS (MODIFIED) ---
TIMEFRAME_WEIGHTS = {5: 1.0, 15: 1.5, 30: 2.0}

# --- NSE INDEX TO SECTOR MAPPING (UPDATED FOR API) ---
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
    "NIFTY FINANCIAL SERVICES 25/50": "Financial Services 25/50",
    "NIFTY INDIA TOURISM": "Tourism",
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

# --- ENHANCED TECHNICAL INDICATORS CLASS ---
class EnhancedTechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        """Calculate all technical indicators including ATR, Volume Surge, and Momentum"""
        indicators = {}
        if df is None or len(df) < 20:
            return indicators

        try:
            # 1. RSI (14-period) - Weight: 1.3
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs))

            # 2. MACD (12,26,9) - Weight: 1.6
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            indicators['MACD'] = macd_line - signal_line

            # 3. Stochastic (14-period) - Weight: 1.0
            low14 = df['Low'].rolling(window=14).min()
            high14 = df['High'].rolling(window=14).max()
            indicators['Stochastic'] = 100 * (df['Close'] - low14) / (high14 - low14)

            # 4. Simple Moving Average (20-period) - Weight: 1.8
            indicators['MA'] = df['Close'].rolling(window=20).mean()

            # 5. Exponential Moving Average (21-period) - Weight: 1.7
            indicators['EMA'] = df['Close'].ewm(span=21).mean()

            # 6. ADX (Average Directional Index) - Weight: 1.5
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

            # 7. Bollinger Bands Position (20,2) - Weight: 1.4
            ma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            upper_band = ma20 + (std20 * 2)
            lower_band = ma20 - (std20 * 2)
            indicators['Bollinger'] = (df['Close'] - ma20) / (upper_band - lower_band) * 100

            # 8. Rate of Change (ROC) - Weight: 1.2
            indicators['ROC'] = df['Close'].pct_change(periods=12) * 100

            # 9. On-Balance Volume (OBV) - Weight: 1.6
            obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            indicators['OBV'] = obv.pct_change(periods=10) * 100

            # 10. Commodity Channel Index (CCI) - Weight: 1.1
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            indicators['CCI'] = (typical_price - sma_tp) / (0.015 * mad)

            # 11. Williams %R (14-period) - Weight: 1.0
            highest_high = df['High'].rolling(window=14).max()
            lowest_low = df['Low'].rolling(window=14).min()
            indicators['WWL'] = (highest_high - df['Close']) / (highest_high - lowest_low) * -100

            # 12. VWAP (Volume Weighted Average Price) - Weight: 1.5
            if len(df) >= 20:
                typical_price_vwap = (df['High'] + df['Low'] + df['Close']) / 3
                vwap_numerator = (typical_price_vwap * df['Volume']).rolling(window=20).sum()
                vwap_denominator = df['Volume'].rolling(window=20).sum()
                indicators['VWAP'] = vwap_numerator / vwap_denominator

            # 13. ATR (Average True Range) - Weight: 1.4 - NEW
            indicators['ATR'] = atr

            # 14. Volume Surge - Weight: 2.0 - NEW HIGHEST PRIORITY
            if len(df) >= 20:
                avg_volume_20 = df['Volume'].rolling(window=20).mean()
                current_volume = df['Volume']
                volume_ratio = current_volume / avg_volume_20
                # Scale: >3x avg = 100, 2x avg = 66.67, 1x avg = 33.33, <0.5x avg = 0
                indicators['Volume_Surge'] = np.clip((volume_ratio - 0.5) * 40, 0, 100)

            # 15. Momentum - Weight: 1.9 - NEW SECOND HIGHEST PRIORITY
            if len(df) >= 10:
                # Combined price and volume momentum
                price_momentum = df['Close'].pct_change(periods=10) * 100

                # Volume momentum
                avg_volume_10 = df['Volume'].rolling(window=10).mean()
                volume_momentum = (df['Volume'] / avg_volume_10 - 1) * 100

                # Weighted combination (70% price, 30% volume)
                momentum_score = (price_momentum * 0.7 + volume_momentum * 0.3)

                # Normalize to 0-100 scale
                indicators['Momentum'] = 50 + np.clip(momentum_score * 1.5, -50, 50)

        except Exception as e:
            logger.error(f"Error calculating enhanced indicators: {e}")

        return indicators

# --- ENHANCED 3-SECTOR SCANNER WITH FIXED API ---
class Enhanced3SectorScanner:
    def __init__(self):
        self.is_running = False
        self.current_signals = {}

        # Store top 3 best and worst sectors
        self.best_sectors = ["Technology", "Pharma", "Banking"]
        self.worst_sectors = ["Auto", "Metal", "Energy"]
        self.best_sector_details = []  # To store {'sector': name, 'index': index_name}
        self.worst_sector_details = [] # To store {'sector': name, 'index': index_name}
        self.sectoral_history = []
        self.last_sectoral_update = None
        self.api_errors = []
        self.sector_update_attempts = 0
        self.successful_updates = 0
        self.gap_down_filtered_count = 0

        # Market hours
        self.market_start = time(9, 15)
        self.market_end = time(15, 30)

        # 5-minute intervals
        self.scan_interval = 300  # 5 minutes
        
        # --- NEW: Rate Limiting ---
        self.api_call_timestamps = deque()
        self.api_lock = threading.Lock()
        self.RATE_LIMIT_COUNT = 9  # Safely below the 10 calls/sec limit
        self.RATE_LIMIT_PERIOD = 1  # 1 second

        logger.info("🚀 Enhanced 3-Sector Scanner with API Sectoral Data initialized")
        self.show_initialization_status()

    def show_initialization_status(self):
        """Show enhanced initialization status"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}🚀 ENHANCED 3-SECTOR SCANNER WITH API SECTORAL DATA{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"⏰ Timeframes: {Colors.YELLOW}5min, 15min, 30min{Colors.RESET}")
        print(f"🎯 Strategy: {Colors.GREEN}Top 3 Best{Colors.RESET} + {Colors.RED}Top 3 Worst{Colors.RESET} sectors (Display Only)")
        print(f"⚡ Stocks Scanned: {Colors.BOLD}{len(ALL_NSE_STOCKS)} stocks from Master List{Colors.RESET}")
        print(f"🚫 Filter: {Colors.MAGENTA}Gap-down exclusion{Colors.RESET}")
        print(f"🌐 Sectoral Data: {Colors.GREEN}API http://localhost:3001/api/allIndices{Colors.RESET}")
        print(f"⏱️ Rate Limit: {Colors.CYAN}Enforced at {self.RATE_LIMIT_COUNT} calls / {self.RATE_LIMIT_PERIOD} second{Colors.RESET}")
        print(f"\n{Colors.YELLOW}📊 NEW INDICATORS:{Colors.RESET}")
        print(f"  • ATR (Weight: 1.4) - Volatility measurement")
        print(f"  • Volume Surge (Weight: 2.0) - HIGH PRIORITY volume analysis")
        print(f"  • Momentum (Weight: 1.9) - HIGH PRIORITY price+volume momentum")
        print(f"\n{Colors.BLUE}📈 ENHANCED WEIGHTS:{Colors.RESET}")
        for indicator, weight in ENHANCED_INDICATOR_WEIGHTS.items():
            color = Colors.GREEN if weight >= 1.5 else Colors.YELLOW if weight >= 1.2 else Colors.WHITE
            print(f"  • {indicator}: {color}{weight}{Colors.RESET}")

        self.show_sector_status()
        self.test_api_connection()

        print(f"\n{Colors.YELLOW}🔄 Running initial sector update...{Colors.RESET}")
        self.force_sector_update()
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")

    def test_api_connection(self):
        """Test API connection"""
        print(f"\n{Colors.BLUE}🔍 API CONNECTION TEST:{Colors.RESET}")
        try:
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            if response.status_code == 200:
                print(f"✅ API Connection: {Colors.GREEN}SUCCESS{Colors.RESET}")
                data = response.json()
                print(f"📊 Response Type: {type(data)}")
                if isinstance(data, list) and data:
                    print(f"📋 Items Count: {len(data)}")
                elif isinstance(data, dict):
                    print(f"🗂️ Dict Keys: {list(data.keys())}")
            else:
                print(f"❌ API Connection: {Colors.RED}FAILED{Colors.RESET} (Status: {response.status_code})")
        except Exception as e:
            print(f"❌ API Connection: {Colors.RED}ERROR{Colors.RESET} - {str(e)}")

    def show_sector_status(self):
        """Show current sector status"""
        print(f"\n{Colors.MAGENTA}📊 CURRENT 3-SECTOR STATUS:{Colors.RESET}")
        best_str = ', '.join([f"{item.get('sector', '')} ({item.get('index', '')})" for item in self.best_sector_details]) or ", ".join(self.best_sectors)
        worst_str = ', '.join([f"{item.get('sector', '')} ({item.get('index', '')})" for item in self.worst_sector_details]) or ", ".join(self.worst_sectors)
        print(f"🏆 Top 3 Best Sectors: {Colors.GREEN}{Colors.BOLD}{best_str}{Colors.RESET}")
        print(f"📉 Top 3 Worst Sectors: {Colors.RED}{Colors.BOLD}{worst_str}{Colors.RESET}")
        print(f"🕐 Last Update: {self.last_sectoral_update or 'Never'}")
        print(f"🚫 Gap-down Filtered: {self.gap_down_filtered_count}")

    def fetch_live_sectoral_performance_3sector_debug(self):
        """FIXED: Enhanced sectoral performance fetching using API endpoint"""
        try:
            logger.info("🔍 Fetching live 3-sector performance from API...")

            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            # This is a different API and likely has different rate limits; we are not throttling this one.
            # print(f"\n{Colors.BLUE}📡 API RESPONSE DEBUG:{Colors.RESET}")
            # print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                indices_data = response.json()
                # print(f"Response Type: {type(indices_data)}")

                if isinstance(indices_data, str):
                    indices_data = json.loads(indices_data)
                    # print(f"✓ Parsed string to JSON")

                if isinstance(indices_data, dict):
                    if 'data' in indices_data:
                        indices_data = indices_data['data']
                    elif 'indices' in indices_data:
                        indices_data = indices_data['indices']
                    elif 'results' in indices_data:
                        indices_data = indices_data['results']

                if not isinstance(indices_data, list):
                    logger.error("❌ Processed API data is not a list.")
                    return False

                sectoral_performance = []
                current_time = datetime.now()

                for index in indices_data:
                    if not isinstance(index, dict):
                        continue

                    index_name = next((str(index[field]).strip().upper() for field in ['name', 'symbol', 'index', 'indexName'] if field in index and index[field]), None)

                    if index_name and index_name in NSE_INDEX_TO_SECTOR:
                        change_percent = 0.0
                        for field in ['change_percent', 'changePercent', 'pChange', 'percentChange', 'change', 'pchg']:
                            if field in index and index[field] is not None:
                                try:
                                    change_percent = float(index[field])
                                    break
                                except (ValueError, TypeError):
                                    continue

                        sectoral_performance.append({
                            'index': index_name,
                            'sector': NSE_INDEX_TO_SECTOR[index_name],
                            'change_percent': change_percent,
                            'timestamp': current_time
                        })
                        # logger.info(f"✅ {index_name} ({NSE_INDEX_TO_SECTOR[index_name]}): {change_percent:+.2f}%")

                if sectoral_performance:
                    sectoral_performance.sort(key=lambda x: x['change_percent'], reverse=True)

                    old_best = self.best_sectors[:]
                    old_worst = self.worst_sectors[:]

                    # Update top 3 best and worst sectors
                    if len(sectoral_performance) >= 6:
                        self.best_sectors = [item['sector'] for item in sectoral_performance[:3]]
                        self.worst_sectors = [item['sector'] for item in sectoral_performance[-3:]]
                        self.best_sector_details = sectoral_performance[:3]
                        self.worst_sector_details = list(reversed(sectoral_performance[-3:]))
                    elif len(sectoral_performance) >= 3:
                        self.best_sectors = [item['sector'] for item in sectoral_performance[:2]]
                        self.worst_sectors = [item['sector'] for item in sectoral_performance[-2:]]
                        self.best_sector_details = sectoral_performance[:2]
                        self.worst_sector_details = list(reversed(sectoral_performance[-2:]))


                    self.last_sectoral_update = current_time

                    self.sectoral_history.append({
                        'timestamp': current_time,
                        'best': self.best_sectors,
                        'worst': self.worst_sectors,
                        'full_data': sectoral_performance
                    })

                    if len(self.sectoral_history) > 20:
                        self.sectoral_history = self.sectoral_history[-20:]

                    self.display_3sector_update(sectoral_performance, old_best, old_worst)
                    return True
                else:
                    print(f"❌ No sectoral data matched from API response.")
                    return False

            else:
                print(f"❌ API request failed with status code: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"❌ Error fetching API sectoral data: {e}")
            self.api_errors.append(f"{datetime.now()}: {e}")
            return False

    def display_3sector_update(self, sectoral_performance, old_best, old_worst):
        """Display enhanced 3-sector performance update"""
        current_time = datetime.now()

        print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'='*100}")
        print(f"📊 3-SECTOR PERFORMANCE UPDATE - {current_time.strftime('%H:%M:%S')} IST")
        print(f"{'='*100}{Colors.RESET}")

        best_str = ', '.join([f"{item['sector']} ({item['index']})" for item in self.best_sector_details])
        worst_str = ', '.join([f"{item['sector']} ({item['index']})" for item in self.worst_sector_details])

        print(f"🏆 Top 3 Best: {Colors.GREEN}{Colors.BOLD}{best_str}{Colors.RESET}")
        print(f"📉 Top 3 Worst: {Colors.RED}{Colors.BOLD}{worst_str}{Colors.RESET}")

        print(f"\n🔝 Top 5 Performing Sectors:")
        for i, sector_data in enumerate(sectoral_performance[:5]):
            color = Colors.YELLOW
            if sector_data['sector'] in self.best_sectors:
                rank = self.best_sectors.index(sector_data['sector']) + 1
                color = Colors.GREEN + (Colors.BOLD if rank == 1 else "")
            print(f"  {i+1}. {color}{sector_data['sector']:<20}{Colors.RESET} {sector_data['change_percent']:+6.2f}% ({sector_data['index']})")

        print(f"\n🔻 Bottom 5 Performing Sectors:")
        for i, sector_data in enumerate(sectoral_performance[-5:]):
            color = Colors.YELLOW
            if sector_data['sector'] in self.worst_sectors:
                rank = self.worst_sectors.index(sector_data['sector']) + 1
                color = Colors.RED + (Colors.BOLD if rank == 1 else "")
            pos = len(sectoral_performance) - 5 + i + 1
            print(f"  {pos}. {color}{sector_data['sector']:<20}{Colors.RESET} {sector_data['change_percent']:+6.2f}% ({sector_data['index']})")

        print(f"{Colors.MAGENTA}{'='*100}{Colors.RESET}")

    def force_sector_update(self):
        """FIXED: Force a sector update using API endpoint"""
        print(f"\n{Colors.YELLOW}🔄 FORCING REAL SECTOR UPDATE WITH API...{Colors.RESET}")
        self.sector_update_attempts += 1

        # Use API endpoint for sectoral data
        success = self.fetch_live_sectoral_performance_3sector_debug()

        if success:
            self.successful_updates += 1
            print(f"✅ API sectoral update successful!")
        else:
            print(f"❌ API sectoral update failed - using defaults")
        
        best_str = ', '.join([f"{item['sector']} ({item['index']})" for item in self.best_sector_details]) or ", ".join(self.best_sectors)
        worst_str = ', '.join([f"{item['sector']} ({item['index']})" for item in self.worst_sector_details]) or ", ".join(self.worst_sectors)

        print(f"🏆 Top 3 Best: {Colors.GREEN}{Colors.BOLD}{best_str}{Colors.RESET}")
        print(f"📉 Top 3 Worst: {Colors.RED}{Colors.BOLD}{worst_str}{Colors.RESET}")
        return success

    def is_market_open(self):
        """Check if market is open"""
        now = datetime.now()
        current_time = now.time()

        if now.weekday() > 4:  # Weekend
            return False

        return self.market_start <= current_time <= self.market_end

    def normalize_live_data(self, df, symbol):
        """Normalize data for indicators"""
        try:
            if df is None or df.empty:
                return None

            df_clean = df.copy()
            cols = [c.lower() for c in df_clean.columns]
            column_mapping = {df_clean.columns[i]: new_name for i, col in enumerate(cols)
                              for key, new_name in [('time', 'Date'), ('open', 'Open'), ('high', 'High'),
                                                    ('low', 'Low'), ('close', 'Close'), ('vol', 'Volume')]
                              if key in col}
            df_clean = df_clean.rename(columns=column_mapping)

            required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            if not all(col in df_clean.columns for col in required_cols):
                return None

            if 'Volume' not in df_clean.columns:
                df_clean['Volume'] = 1000

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

    def check_gap_down(self, df):
        """Check for gap down"""
        try:
            if df is None or len(df) < 2:
                return False

            current_open = df['Open'].iloc[-1]
            previous_close = df['Close'].iloc[-2]

            if pd.isna(current_open) or pd.isna(previous_close) or previous_close <= 0:
                return False

            gap_percentage = ((current_open - previous_close) / previous_close) * 100
            return gap_percentage <= -1.0

        except Exception as e:
            logger.error(f"Error checking gap down: {e}")
            return False

    def fetch_live_data(self, symbol, timeframe):
        """Fetch live data with gap-down detection and rate limiting."""
        try:
            tf_map = {
                5: '5 min',
                15: '15 min',
                30: '30 min',
            }
            bar_size = tf_map.get(timeframe)
            if not bar_size:
                return None, False

            if timeframe == 5 or timeframe == 15:
                duration = '10 D'
            elif timeframe == 30:
                duration = '20 D'
            else:
                duration = '10 D'

            # --- NEW: Smart Rate Limiting Logic ---
            with self.api_lock:
                current_time = time_module.time()

                # Remove timestamps older than the rate limit period
                while self.api_call_timestamps and self.api_call_timestamps[0] <= current_time - self.RATE_LIMIT_PERIOD:
                    self.api_call_timestamps.popleft()

                # If we have too many recent calls, wait
                if len(self.api_call_timestamps) >= self.RATE_LIMIT_COUNT:
                    oldest_call_time = self.api_call_timestamps[0]
                    time_to_wait = (oldest_call_time + self.RATE_LIMIT_PERIOD) - current_time
                    if time_to_wait > 0:
                        time_module.sleep(time_to_wait)
                
                # Record the timestamp of this call
                self.api_call_timestamps.append(time_module.time())
            # --- End of Rate Limiting Logic ---
            
            # Make the actual API call
            raw_df = td_hist.get_historic_data(symbol, duration=duration, bar_size=bar_size)

            if raw_df is not None and len(raw_df) > 0:
                normalized_df = self.normalize_live_data(raw_df, symbol)
                if normalized_df is not None and len(normalized_df) >= 20:
                    is_gap_down = self.check_gap_down(normalized_df)
                    return normalized_df.tail(100), is_gap_down
            return None, False
        except Exception as e:
            logger.error(f"Live data fetch error {symbol}_{timeframe}: {e}")
            return None, False


    def normalize_indicator_value(self, indicator_name, value):
        """Enhanced normalize with new indicators"""
        try:
            if indicator_name == 'RSI':
                return max(0, min(100, value))
            elif indicator_name == 'MACD':
                return 50 + min(25, max(-25, value * 10))
            elif indicator_name == 'Stochastic':
                return max(0, min(100, value))
            elif indicator_name in ['MA', 'EMA', 'VWAP']:
                return 50  # Handled separately
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
            elif indicator_name == 'ATR':
                # Higher ATR = more volatile = potentially higher scores
                return min(100, max(0, value * 20))
            elif indicator_name == 'Volume_Surge':
                return max(0, min(100, value))  # Already normalized
            elif indicator_name == 'Momentum':
                return max(0, min(100, value))  # Already normalized
            else:
                return 50
        except:
            return 50

    def calculate_enhanced_signals(self, symbol, timeframes_data):
        """Calculate signals using enhanced indicators and weights"""
        try:
            if not timeframes_data:
                return 'Neutral', 0

            sector = next((s for s, stocks in SECTOR_STOCKS.items() if symbol in stocks), None)
            if not sector:
                return 'Neutral', 0

            total_weighted_score, total_weight = 0, 0
            timeframe_scores = {}

            for tf, df in timeframes_data.items():
                if df is None or len(df) < 20:
                    continue

                indicators = EnhancedTechnicalIndicators.calculate_all_indicators(df)
                if not indicators:
                    continue

                tf_score, tf_weight = 0, 0
                current_price = df['Close'].iloc[-1]

                for name, weight in ENHANCED_INDICATOR_WEIGHTS.items():
                    if name in indicators and indicators[name] is not None and not indicators[name].empty:
                        latest_val = indicators[name].iloc[-1]
                        if pd.notna(latest_val):
                            if name in ['MA', 'EMA', 'VWAP']:
                                if latest_val > 0:
                                    price_vs_ma = (current_price - latest_val) / latest_val * 100
                                    if price_vs_ma > 2:
                                        norm_score = 75
                                    elif price_vs_ma > 0:
                                        norm_score = 60
                                    elif price_vs_ma > -2:
                                        norm_score = 50
                                    elif price_vs_ma > -5:
                                        norm_score = 40
                                    else:
                                        norm_score = 25
                                else:
                                    norm_score = 50
                            else:
                                norm_score = self.normalize_indicator_value(name, latest_val)

                            tf_score += norm_score * weight
                            tf_weight += weight

                if tf_weight > 0:
                    tf_final_score = tf_score / tf_weight
                    timeframe_scores[tf] = tf_final_score

                    tf_multiplier = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                    total_weighted_score += tf_final_score * tf_multiplier
                    total_weight += tf_multiplier

            if total_weight == 0:
                return 'Neutral', 0

            base_score = total_weighted_score / total_weight

            # Enhanced 3-sector boost system (Simplified for intraday)
            sector_boost = 0
            if sector in self.best_sectors:
                rank = self.best_sectors.index(sector) + 1
                sector_boost = [20, 15, 10][rank-1] if rank <= 3 else 5
            elif sector in self.worst_sectors:
                rank = self.worst_sectors.index(sector) + 1
                sector_boost = [-20, -15, -10][rank-1] if rank <= 3 else -5

            base_score += sector_boost

            # Multi-timeframe confirmation bonus
            num_timeframes = len(timeframes_data)
            if num_timeframes >= 2: # Check for at least 2 timeframes
                bullish_count = sum(1 for score in timeframe_scores.values() if score > 55)
                bearish_count = sum(1 for score in timeframe_scores.values() if score < 45)

                if bullish_count >= 2:
                    base_score += 8  # Enhanced bonus
                elif bearish_count >= 2:
                    base_score -= 8

            # Enhanced signal classification with new thresholds
            if base_score >= 82: return 'Very Strong Buy', base_score
            elif base_score >= 72: return 'Strong Buy', base_score
            elif base_score >= 60: return 'Buy', base_score
            elif base_score <= 18: return 'Very Strong Sell', base_score
            elif base_score <= 28: return 'Strong Sell', base_score
            elif base_score <= 40: return 'Sell', base_score
            else: return 'Neutral', base_score

        except Exception as e:
            logger.error(f"Enhanced signal calculation error for {symbol}: {e}")
            return 'Neutral', 0

    def enhanced_scan_cycle(self):
        """Enhanced scan cycle with new indicators and API sectoral data"""
        if not self.is_market_open():
            logger.info("🕐 Market closed. Next scan in 5 minutes...")
            return

        start_time = time_module.time()
        current_time = datetime.now()

        print(f"\n{Colors.CYAN}🔄 Starting ENHANCED scan at {current_time.strftime('%H:%M:%S')}{Colors.RESET}")
        print(f"⏰ Analyzing: {Colors.YELLOW}5min → 15min → 30min{Colors.RESET}")
        print(f"⚡ Stocks Scanned: {Colors.BOLD}{len(ALL_NSE_STOCKS)} from Master List{Colors.RESET}")
        print(f"📊 New Indicators: {Colors.MAGENTA}ATR(1.4), Volume_Surge(2.0), Momentum(1.9){Colors.RESET}")
        print(f"🌐 Sectoral Source: {Colors.GREEN}API localhost:3001/api/allIndices{Colors.RESET}")

        # Update sectors with API data
        if not self.fetch_live_sectoral_performance_3sector_debug():
            print(f"⚠️ API sectoral update failed, continuing with previous sectors")

        try:
            # Use the master list of stocks
            target_stocks = ALL_NSE_STOCKS

            if not target_stocks:
                print(f"⚠️ No target stocks found in the master list.")
                return

            print(f"🎯 Enhanced scanning {len(target_stocks)} stocks from the master list")

            live_signals = []
            gap_down_filtered = 0

            with ThreadPoolExecutor(max_workers=3) as executor:
                def process_stock(symbol):
                    try:
                        timeframes_data = {}
                        has_gap_down = False

                        timeframes_to_fetch = [5, 15, 30]

                        for tf in timeframes_to_fetch:
                            df, is_gap_down = self.fetch_live_data(symbol, tf)
                            if df is not None:
                                timeframes_data[tf] = df
                                if is_gap_down:
                                    has_gap_down = True
                            # REMOVED: time_module.sleep(1.0) - Replaced by smart rate limiter

                        # Exclude gap-down stocks
                        if has_gap_down:
                            logger.info(f"🚫 {symbol} filtered out due to gap-down")
                            return None, True

                        if len(timeframes_data) >= 2: # Require at least 2 TFs for a signal
                            signal, score = self.calculate_enhanced_signals(symbol, timeframes_data)
                            if abs(score - 50) > 15:  # Only significant signals
                                sector = next((s for s, st in SECTOR_STOCKS.items() if symbol in st), 'N/A')
                                return {
                                    'symbol': symbol, 'signal': signal, 'score': score, 'sector': sector,
                                    'timeframes': len(timeframes_data), 'timestamp': datetime.now(),
                                    'tf_details': list(timeframes_data.keys())
                                }, False
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                    return None, False

                futures = [executor.submit(process_stock, symbol) for symbol in target_stocks]

                for future in as_completed(futures):
                    result, is_gap_down_filtered = future.result()
                    if is_gap_down_filtered:
                        gap_down_filtered += 1
                    elif result:
                        live_signals.append(result)

            self.gap_down_filtered_count = gap_down_filtered
            scan_time = time_module.time() - start_time
            logger.info(f"⚡ Enhanced scan completed in {scan_time:.2f}s - {len(live_signals)} signals, {gap_down_filtered} gap-down filtered")
            self.display_enhanced_signals(live_signals, scan_time, gap_down_filtered)

        except Exception as e:
            logger.error(f"Error in enhanced scan: {e}")

    def display_enhanced_signals(self, signals, scan_time, gap_down_filtered):
        """Display enhanced signals with new indicator context"""
        os.system('clear' if os.name == 'posix' else 'cls')
        current_time = datetime.now()

        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*150}")
        print(f"🚀 ENHANCED 3-SECTOR SCANNER WITH API SECTORAL DATA - {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
        print(f"{'='*150}{Colors.RESET}")

        print(f"{Colors.BLUE}📊 Analysis: {Colors.YELLOW}5m{Colors.RESET} + {Colors.YELLOW}15m{Colors.RESET} + "
              f"{Colors.YELLOW}30m{Colors.RESET}")
        print(f"⚡ Stocks Scanned: {Colors.BOLD}{len(ALL_NSE_STOCKS)} stocks from Master List{Colors.RESET}")
        print(f"📊 Enhanced Indicators: {Colors.MAGENTA}15 indicators{Colors.RESET} including {Colors.GREEN}{Colors.BOLD}Volume_Surge(2.0){Colors.RESET} & {Colors.GREEN}{Colors.BOLD}Momentum(1.9){Colors.RESET}")
        print(f"🌐 Sectoral Data: {Colors.GREEN}LIVE API{Colors.RESET}")

        if self.last_sectoral_update:
            best_str = ', '.join([f"{item['sector']} ({item['index']})" for item in self.best_sector_details])
            worst_str = ', '.join([f"{item['sector']} ({item['index']})" for item in self.worst_sector_details])
            print(f"{Colors.MAGENTA}📈 API Sectoral Update:{Colors.RESET} {Colors.YELLOW}{self.last_sectoral_update.strftime('%H:%M:%S')}{Colors.RESET}")
            print(f"🏆 Top 3 Best: {Colors.GREEN}{Colors.BOLD}{best_str}{Colors.RESET}")
            print(f"📉 Top 3 Worst: {Colors.RED}{Colors.BOLD}{worst_str}{Colors.RESET}")

        print(f"{Colors.BLUE}📈 Updates:{Colors.RESET} {self.successful_updates}/{self.sector_update_attempts} | "
              f"⚡ Scan Time: {scan_time:.2f}s | 🚫 Gap-down Filtered: {Colors.MAGENTA}{gap_down_filtered}{Colors.RESET}")

        if not signals:
            print(f"\n{Colors.YELLOW}📭 No significant enhanced signals found in this cycle.{Colors.RESET}")
            print(f"{Colors.CYAN}💡 {gap_down_filtered} stocks filtered due to gap-down opening{Colors.RESET}")
        else:
            # Separate bullish and bearish signals for top 10 each
            bullish_signals = [s for s in signals if 'Buy' in s['signal']]
            bearish_signals = [s for s in signals if 'Sell' in s['signal']]

            bullish_signals.sort(key=lambda x: x['score'], reverse=True)
            bearish_signals.sort(key=lambda x: x['score'])

            print(f"\n{Colors.GREEN}{Colors.BOLD}🚀 TOP 10 BULLISH SIGNALS (ENHANCED WITH API SECTORAL):{Colors.RESET}")
            print(f"{'Stock':<10} {'Sector':<18} {'Signal':<20} {'Score':>8} {'TFs':>4} {'TF Coverage':<20} {'Strength':<15}")
            print(f"{Colors.GREEN}{'-' * 150}{Colors.RESET}")

            for s in bullish_signals[:10]:
                sector_color, sector_name = Colors.YELLOW, s['sector']
                if s['sector'] in self.best_sectors:
                    rank = self.best_sectors.index(s['sector']) + 1
                    stars = "★" * rank
                    sector_color, sector_name = Colors.GREEN, f"{stars}{s['sector']}"

                signal_color = Colors.GREEN + (Colors.BOLD if 'Very' in s['signal'] else "")

                # Enhanced strength calculation
                deviation = abs(s['score'] - 50)
                if deviation >= 40:
                    strength = f"{Colors.GREEN}{Colors.BOLD}Exceptional+{Colors.RESET}"
                elif deviation >= 30:
                    strength = f"{Colors.GREEN}{Colors.BOLD}Very Strong{Colors.RESET}"
                elif deviation >= 20:
                    strength = f"{Colors.GREEN}Strong{Colors.RESET}"
                else:
                    strength = f"{Colors.YELLOW}Moderate{Colors.RESET}"

                tf_details = s.get('tf_details', [])
                tf_display = [f"{tf}m" if isinstance(tf, int) else "D" for tf in tf_details]
                tf_coverage = ','.join(tf_display[:5])

                print(f"{Colors.WHITE}{s['symbol']:<10}{Colors.RESET} "
                      f"{sector_color}{sector_name:<18}{Colors.RESET} "
                      f"{signal_color}{s['signal']:<20}{Colors.RESET} "
                      f"{Colors.WHITE}{s['score']:>8.1f}{Colors.RESET} "
                      f"{Colors.CYAN}{s['timeframes']:>4}{Colors.RESET} "
                      f"{Colors.MAGENTA}{tf_coverage:<20}{Colors.RESET} "
                      f"{strength}")

            if bearish_signals:
                print(f"\n{Colors.RED}{Colors.BOLD}📉 TOP 10 BEARISH SIGNALS (ENHANCED WITH API SECTORAL):{Colors.RESET}")
                print(f"{'Stock':<10} {'Sector':<18} {'Signal':<20} {'Score':>8} {'TFs':>4} {'TF Coverage':<20} {'Strength':<15}")
                print(f"{Colors.RED}{'-' * 150}{Colors.RESET}")

                for s in bearish_signals[:10]:
                    sector_color, sector_name = Colors.YELLOW, s['sector']
                    if s['sector'] in self.worst_sectors:
                        rank = self.worst_sectors.index(s['sector']) + 1
                        stars = "★" * rank
                        sector_color, sector_name = Colors.RED, f"{stars}{s['sector']}"

                    signal_color = Colors.RED + (Colors.BOLD if 'Very' in s['signal'] else "")

                    deviation = abs(s['score'] - 50)
                    if deviation >= 40:
                        strength = f"{Colors.RED}{Colors.BOLD}Exceptional+{Colors.RESET}"
                    elif deviation >= 30:
                        strength = f"{Colors.RED}{Colors.BOLD}Very Strong{Colors.RESET}"
                    elif deviation >= 20:
                        strength = f"{Colors.RED}Strong{Colors.RESET}"
                    else:
                        strength = f"{Colors.YELLOW}Moderate{Colors.RESET}"

                    tf_details = s.get('tf_details', [])
                    tf_display = [f"{tf}m" if isinstance(tf, int) else "D" for tf in tf_details]
                    tf_coverage = ','.join(tf_display[:5])

                    print(f"{Colors.WHITE}{s['symbol']:<10}{Colors.RESET} "
                          f"{sector_color}{sector_name:<18}{Colors.RESET} "
                          f"{signal_color}{s['signal']:<20}{Colors.RESET} "
                          f"{Colors.WHITE}{s['score']:>8.1f}{Colors.RESET} "
                          f"{Colors.CYAN}{s['timeframes']:>4}{Colors.RESET} "
                          f"{Colors.MAGENTA}{tf_coverage:<20}{Colors.RESET} "
                          f"{strength}")

        next_scan_time = (current_time + timedelta(minutes=5)).strftime('%H:%M:%S')
        print(f"\n{Colors.CYAN}{Colors.BOLD}⏰ Next enhanced scan at {next_scan_time}{Colors.RESET}")
        print(f"{Colors.BLUE}💡 Enhanced strategy with Volume Surge & Momentum + LIVE API sectoral data{Colors.RESET}")
        if gap_down_filtered > 0:
            print(f"{Colors.MAGENTA}🚫 Gap-down filter excluded {gap_down_filtered} stocks for risk management{Colors.RESET}")

    def run_enhanced_scanner(self):
        """Main enhanced scanner execution"""
        self.is_running = True
        logger.info("🚀 Starting Enhanced 3-Sector Scanner with API Sectoral Data...")

        self.force_sector_update()

        try:
            while self.is_running:
                self.enhanced_scan_cycle()
                if self.is_running:
                    logger.info(f"💤 Waiting 5 minutes for next enhanced cycle...")
                    time_module.sleep(self.scan_interval)
        except KeyboardInterrupt:
            logger.info("🛑 Enhanced scanner stopped by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the enhanced scanner"""
        self.is_running = False
        print(f"{Colors.YELLOW}🛑 Enhanced 3-sector scanner stopped{Colors.RESET}")


# --- MAIN EXECUTION ---
def main():
    print(f"{Colors.CYAN}{Colors.BOLD}🚀 ENHANCED SCANNER WITH API SECTORAL DATA & MASTER LIST{Colors.RESET}")
    print(f"{Colors.YELLOW}📊 Timeframes: 5min, 15min, 30min{Colors.RESET}")
    print(f"🔧 Features: Master stock list, Enhanced indicators, API sectoral data{Colors.RESET}")
    print(f"{Colors.MAGENTA}⚡ NEW: ATR, Volume Surge (2.0 weight), Momentum (1.9 weight) indicators{Colors.RESET}")
    print(f"🌐 LIVE API sectoral updates from http://localhost:3001/api/allIndices{Colors.RESET}")
    print(f"🎯 Updates every 5 minutes with REAL sectoral performance{Colors.RESET}")

    scanner = Enhanced3SectorScanner()
    try:
        scanner.run_enhanced_scanner()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Shutting down enhanced scanner...{Colors.RESET}")
        scanner.stop()


if __name__ == "__main__":
    main()