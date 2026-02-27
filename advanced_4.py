# ENHANCED REAL-TIME 4-SECTOR SCANNER WITH FIXED API SECTORAL DATA
# - Top 4 best + top 4 worst sectors
# - Full sector stock universe (no 50-stock restriction)
# - Weekly timeframe via daily->weekly resample (more stable)
# - Stabilized fetching: serial, retries, shorter durations, throttling

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

# --- INDICATOR WEIGHTS ---
ENHANCED_INDICATOR_WEIGHTS = {
    'RSI': 1.3, 'MACD': 1.6, 'Stochastic': 1.0, 'MA': 1.8,
    'ADX': 1.5, 'Bollinger': 1.4, 'ROC': 1.2, 'OBV': 1.6,
    'CCI': 1.1, 'WWL': 1.0, 'EMA': 1.7, 'VWAP': 1.5,
    'ATR': 1.4, 'Volume_Surge': 2.0, 'Momentum': 1.9
}

# --- TIMEFRAME WEIGHTS (weekly added) ---
TIMEFRAME_WEIGHTS = {5: 1.0, 15: 1.5, 30: 2.0, 60: 2.5, 'daily': 3.0, 'weekly': 4.0}

# --- NSE INDEX TO SECTOR MAP (must match API fields) ---
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

# --- SECTOR -> STOCKS (full lists; no 50-stock filter) ---
SECTOR_STOCKS = {
    "Technology": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "CYIENT", "KPITTECH", "TATAELXSI","SONACOMS","KAYNES","OFSS"],
    "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR","BHARATFORG", "EICHERMOT", "ASHOKLEY", "BOSCHLTD","TIINDIA","MOTHERSON"],
    "Banking": ["HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","PNB","BANKBARODA","CANBK","IDFCFIRSTB","INDUSINDBK","AUBANK","FEDERALBNK"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA", "TORNTPHARM","GLENMARK","ALKEM","LAURUSLABS","BIOCON","ZYDUSLIFE","MANKIND","SYNGENE","PPLPHARMA"],
    "Energy": ["RELIANCE","NTPC","BPCL","IOC","ONGC","GAIL","HINDPETRO","ADANIGREEN","ADANIENSOL","JSWENERGY","COALINDIA","TATAPOWER","SUZLON","PETRONET","OIL","POWERGRID","NHPC","ADANIPORTS","ABB","SIEMENS","CGPOWER","INOXWIND"],
    "Metal": ["TATASTEEL", "JSWSTEEL", "SAIL", "JINDALSTEL", "HINDALCO", "NMDC"],
    "Consumer": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR","AMBER","UNITDSPR","GODREJCP","MARICO","COLPAL","UPL","VBL"],
    "PSU Bank": ["SBIN", "PNB", "BANKBARODA", "CANBK", "UNIONBANK", "BANKINDIA"],
    "Finance": ["BAJFINANCE", "SHRIRAMFIN", "CHOLAFIN", "HDFCLIFE", "ICICIPRULI"],
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

# --- INDICATORS ---
class EnhancedTechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        indicators = {}
        if df is None or len(df) < 20:
            return indicators
        try:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs))

            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            indicators['MACD'] = macd_line - signal_line

            low14 = df['Low'].rolling(window=14).min()
            high14 = df['High'].rolling(window=14).max()
            indicators['Stochastic'] = 100 * (df['Close'] - low14) / (high14 - low14)

            indicators['MA'] = df['Close'].rolling(window=20).mean()
            indicators['EMA'] = df['Close'].ewm(span=21).mean()

            high_diff = df['High'].diff()
            low_diff = df['Low'].diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = (-low_diff).where((low_diff < high_diff) & (low_diff < 0), 0)
            tr1 = df['High'] - df['Low']
            tr2 = (df['High'] - df['Close'].shift()).abs()
            tr3 = (df['Low'] - df['Close'].shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            indicators['ADX'] = dx.rolling(window=14).mean()

            ma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            upper_band = ma20 + (std20 * 2)
            lower_band = ma20 - (std20 * 2)
            indicators['Bollinger'] = (df['Close'] - ma20) / (upper_band - lower_band) * 100

            indicators['ROC'] = df['Close'].pct_change(periods=12) * 100

            obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            indicators['OBV'] = obv.pct_change(periods=10) * 100

            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            indicators['CCI'] = (typical_price - sma_tp) / (0.015 * mad)

            highest_high = df['High'].rolling(window=14).max()
            lowest_low = df['Low'].rolling(window=14).min()
            indicators['WWL'] = (highest_high - df['Close']) / (highest_high - lowest_low) * -100

            if len(df) >= 20:
                typical_price_vwap = (df['High'] + df['Low'] + df['Close']) / 3
                vwap_num = (typical_price_vwap * df['Volume']).rolling(window=20).sum()
                vwap_den = df['Volume'].rolling(window=20).sum()
                indicators['VWAP'] = vwap_num / vwap_den

            indicators['ATR'] = atr

            if len(df) >= 20:
                avg_volume_20 = df['Volume'].rolling(window=20).mean()
                volume_ratio = df['Volume'] / avg_volume_20
                indicators['Volume_Surge'] = np.clip((volume_ratio - 0.5) * 40, 0, 100)

            if len(df) >= 10:
                price_mom = df['Close'].pct_change(periods=10) * 100
                avg_vol_10 = df['Volume'].rolling(window=10).mean()
                vol_mom = (df['Volume'] / avg_vol_10 - 1) * 100
                indicators['Momentum'] = 50 + np.clip(price_mom * 0.7 + vol_mom * 0.3, -50, 50)
        except Exception as e:
            logger.error(f"Indicator calc error: {e}")
        return indicators

# --- SCANNER ---
class Enhanced4SectorScanner:
    def __init__(self):
        self.is_running = False
        self.current_signals = {}

        self.best_sectors = ["Technology", "Pharma", "Banking", "Energy"]
        self.worst_sectors = ["Auto", "Metal", "Realty", "PSU Bank"]
        self.sectoral_history = []
        self.last_sectoral_update = None
        self.api_errors = []
        self.sector_update_attempts = 0
        self.successful_updates = 0
        self.gap_down_filtered_count = 0

        self.market_start = time(9, 15)
        self.market_end = time(15, 30)
        self.scan_interval = 300  # seconds

        logger.info("Enhanced 4-Sector Scanner initialized")
        self.show_initialization_status()

    def show_initialization_status(self):
        print(f"\n{Colors.CYAN}{Colors.BOLD}ENHANCED 4-SECTOR SCANNER WITH API SECTORAL DATA{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"Timeframes: {Colors.YELLOW}5m, 15m, 30m, 60m, Daily, Weekly{Colors.RESET}")
        print(f"Strategy: {Colors.GREEN}Top 4 Best{Colors.RESET} + {Colors.RED}Top 4 Worst{Colors.RESET} sectors")
        print(f"Filter: {Colors.MAGENTA}Gap-down exclusion (intraday){Colors.RESET}")
        print(f"Sectoral API: {Colors.GREEN}http://localhost:3001/api/allIndices{Colors.RESET}")
        self.show_sector_status()
        self.test_api_connection()
        print(f"\n{Colors.YELLOW}Running initial sector update...{Colors.RESET}")
        self.force_sector_update()
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")

    def test_api_connection(self):
        print(f"\n{Colors.BLUE}API CONNECTION TEST:{Colors.RESET}")
        try:
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            if response.status_code == 200:
                print(f"API Connection: {Colors.GREEN}SUCCESS{Colors.RESET}")
            else:
                print(f"API Connection: {Colors.RED}FAILED{Colors.RESET} (Status: {response.status_code})")
        except Exception as e:
            print(f"API Connection: {Colors.RED}ERROR{Colors.RESET} - {str(e)}")

    def show_sector_status(self):
        print(f"\n{Colors.MAGENTA}CURRENT 4-SECTOR STATUS:{Colors.RESET}")
        print(f"Top 4 Best: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"Top 4 Worst: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
        print(f"Last Update: {self.last_sectoral_update or 'Never'}")
        print(f"Gap-down Filtered: {self.gap_down_filtered_count}")

    def fetch_live_sectoral_performance_debug(self):
        try:
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            if response.status_code != 200:
                return False

            indices_data = response.json()
            if isinstance(indices_data, str):
                indices_data = json.loads(indices_data)
            if isinstance(indices_data, dict):
                if 'data' in indices_data: indices_data = indices_data['data']
                elif 'indices' in indices_data: indices_data = indices_data['indices']
                elif 'results' in indices_data: indices_data = indices_data['results']

            if not isinstance(indices_data, list):
                return False

            sectoral_performance = []
            now = datetime.now()
            for index in indices_data:
                if not isinstance(index, dict):
                    continue
                index_name = None
                for field in ['name', 'symbol', 'index', 'indexName']:
                    if field in index and index[field]:
                        index_name = str(index[field]).strip().upper()
                        break
                if not index_name or index_name not in NSE_INDEX_TO_SECTOR:
                    continue

                change_percent = 0.0
                for field in ['change_percent', 'changePercent', 'pChange', 'percentChange', 'change', 'pchg']:
                    if field in index and index[field] is not None:
                        try:
                            change_percent = float(index[field])
                            break
                        except (ValueError, TypeError):
                            pass

                sectoral_performance.append({
                    'index': index_name,
                    'sector': NSE_INDEX_TO_SECTOR[index_name],
                    'change_percent': change_percent,
                    'timestamp': now
                })

            if not sectoral_performance:
                return False

            sectoral_performance.sort(key=lambda x: x['change_percent'], reverse=True)

            if len(sectoral_performance) >= 8:
                self.best_sectors = [s['sector'] for s in sectoral_performance[:4]]
                self.worst_sectors = [s['sector'] for s in sectoral_performance[-4:]]
            elif len(sectoral_performance) >= 4:
                self.best_sectors = [s['sector'] for s in sectoral_performance[:2]]
                self.worst_sectors = [s['sector'] for s in sectoral_performance[-2:]]

            self.last_sectoral_update = now
            self.sectoral_history.append({
                'timestamp': now,
                'best': self.best_sectors,
                'worst': self.worst_sectors,
                'full_data': sectoral_performance
            })
            if len(self.sectoral_history) > 20:
                self.sectoral_history = self.sectoral_history[-20:]

            self.display_sector_update(sectoral_performance)
            return True
        except Exception as e:
            logger.error(f"Sectoral API error: {e}")
            self.api_errors.append(f"{datetime.now()}: {e}")
            return False

    def display_sector_update(self, sectoral_performance):
        now = datetime.now()
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'='*100}")
        print(f"4-SECTOR PERFORMANCE UPDATE - {now.strftime('%H:%M:%S')} IST")
        print(f"{'='*100}{Colors.RESET}")
        print(f"Top 4 Best: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"Top 4 Worst: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")

        print(f"\nTop 5 Performing Sectors:")
        for i, s in enumerate(sectoral_performance[:5]):
            color = Colors.GREEN if s['sector'] in self.best_sectors else Colors.YELLOW
            print(f"  {i+1}. {color}{s['sector']:<20}{Colors.RESET} {s['change_percent']:+6.2f}% ({s['index']})")

        print(f"\nBottom 5 Performing Sectors:")
        for i, s in enumerate(sectoral_performance[-5:]):
            color = Colors.RED if s['sector'] in self.worst_sectors else Colors.YELLOW
            pos = len(sectoral_performance) - 5 + i + 1
            print(f"  {pos}. {color}{s['sector']:<20}{Colors.RESET} {s['change_percent']:+6.2f}% ({s['index']})")

        print(f"{Colors.MAGENTA}{'='*100}{Colors.RESET}")

    def force_sector_update(self):
        ok = self.fetch_live_sectoral_performance_debug()
        print("API sectoral update successful!" if ok else "API sectoral update failed - using defaults")
        print(f"Top 4 Best: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"Top 4 Worst: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
        return ok

    def is_market_open(self):
        now = datetime.now()
        if now.weekday() > 4:
            return False
        return self.market_start <= now.time() <= self.market_end

    # -------- Stable fetching helpers --------
    def _fetch_with_retry(self, symbol, duration, bar_size, retries=3, base_sleep=0.9):
        for attempt in range(1, retries+1):
            try:
                return td_hist.get_historic_data(symbol, duration=duration, bar_size=bar_size)
            except Exception as e:
                logger.error(f"Fetch error {symbol} {bar_size} {duration} (try {attempt}/{retries}): {e}")
                time_module.sleep(base_sleep * (1 + 0.3*np.random.rand()))
        return None

    def normalize_live_data(self, df, symbol):
        try:
            if df is None or len(df) == 0:
                return None
            dfc = df.copy()
            cols = [c.lower() for c in dfc.columns]
            mapping = {}
            for i, c in enumerate(cols):
                if 'time' in c: mapping[dfc.columns[i]] = 'Date'
                if 'open' in c: mapping[dfc.columns[i]] = 'Open'
                if 'high' in c: mapping[dfc.columns[i]] = 'High'
                if 'low' in c: mapping[dfc.columns[i]] = 'Low'
                if 'close' in c: mapping[dfc.columns[i]] = 'Close'
                if 'vol' in c: mapping[dfc.columns[i]] = 'Volume'
            dfc = dfc.rename(columns=mapping)

            req = ['Date','Open','High','Low','Close']
            if not all(r in dfc.columns for r in req):
                return None
            if 'Volume' not in dfc.columns:
                dfc['Volume'] = 1000

            dfc['Date'] = pd.to_datetime(dfc['Date'], errors='coerce')
            if dfc['Date'].dt.tz is not None:
                dfc['Date'] = dfc['Date'].dt.tz_localize(None)
            dfc = dfc.dropna(subset=['Date'])
            dfc = dfc.set_index('Date')

            for c in ['Open','High','Low','Close','Volume']:
                dfc[c] = pd.to_numeric(dfc[c], errors='coerce')
            dfc = dfc.dropna().sort_index()
            return dfc if len(dfc) >= 20 else None
        except Exception as e:
            logger.error(f"Normalize error {symbol}: {e}")
            return None

    def check_gap_down(self, df):
        try:
            if df is None or len(df) < 2:
                return False
            o = df['Open'].iloc[-1]
            pc = df['Close'].iloc[-2]
            if pd.isna(o) or pd.isna(pc) or pc <= 0:
                return False
            gap_pct = (o - pc) / pc * 100
            return gap_pct <= -1.0
        except Exception as e:
            logger.error(f"Gap-down check error: {e}")
            return False

    def _resample_weekly(self, df):
        try:
            if df is None or df.empty or not isinstance(df.index, pd.DatetimeIndex):
                return None
            w = df.resample('W-FRI').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
            return w if len(w) >= 20 else None
        except Exception as e:
            logger.error(f"Weekly resample error: {e}")
            return None

    def fetch_live_data(self, symbol, timeframe):
        try:
            # Weekly: always resample from EOD daily (more robust against LZ4 issues)
            if timeframe == 'weekly':
                raw_daily = self._fetch_with_retry(symbol, '1000 D', 'EOD')
                daily = self.normalize_live_data(raw_daily, symbol)
                weekly = self._resample_weekly(daily)
                return (weekly.tail(250), False) if weekly is not None else (None, False)

            tf_map = {5:'5 min', 15:'15 min', 30:'30 min', 60:'60 mins', 'daily':'EOD'}
            bar_size = tf_map.get(timeframe)
            if not bar_size:
                return None, False

            if timeframe in [5,15]:
                duration = '5 D'
            elif timeframe == 30:
                duration = '10 D'
            elif timeframe == 60:
                duration = '20 D'
            elif timeframe == 'daily':
                duration = '365 D'
            else:
                duration = '5 D'

            raw_df = self._fetch_with_retry(symbol, duration, bar_size)
            if raw_df is None:
                return None, False

            norm = self.normalize_live_data(raw_df, symbol)
            if norm is None or len(norm) < 20:
                return None, False

            is_gap_down = self.check_gap_down(norm) if timeframe in [5, 15, 30] else False
            tail_n = 250 if timeframe == 'daily' else (200 if timeframe == 60 else 100)
            return norm.tail(tail_n), is_gap_down
        except Exception as e:
            logger.error(f"Fetch error {symbol}_{timeframe}: {e}")
            return None, False

    def normalize_indicator_value(self, name, value):
        try:
            if name == 'RSI': return max(0, min(100, value))
            if name == 'MACD': return 50 + min(25, max(-25, value * 10))
            if name == 'Stochastic': return max(0, min(100, value))
            if name in ['MA','EMA','VWAP']: return 50
            if name == 'ADX': return max(0, min(100, value))
            if name == 'Bollinger': return max(0, min(100, (value + 100) / 2))
            if name == 'ROC': return 50 + min(25, max(-25, value * 2))
            if name == 'OBV': return 50 + min(25, max(-25, value))
            if name == 'CCI': return max(0, min(100, (value + 200) / 4))
            if name == 'WWL': return max(0, min(100, value + 100))
            if name == 'ATR': return min(100, max(0, value * 20))
            if name == 'Volume_Surge': return max(0, min(100, value))
            if name == 'Momentum': return max(0, min(100, value))
            return 50
        except:
            return 50

    def calculate_enhanced_signals(self, symbol, timeframes_data):
        try:
            if not timeframes_data:
                return 'Neutral', 0

            sector = next((s for s, stocks in SECTOR_STOCKS.items() if symbol in stocks), None)
            if not sector:
                return 'Neutral', 0

            total_weighted_score, total_weight = 0.0, 0.0
            timeframe_scores = {}

            for tf, df in timeframes_data.items():
                if df is None or len(df) < 20:
                    continue
                indicators = EnhancedTechnicalIndicators.calculate_all_indicators(df)
                if not indicators:
                    continue

                tf_score, tf_weight = 0.0, 0.0
                current_price = df['Close'].iloc[-1]

                for name, weight in ENHANCED_INDICATOR_WEIGHTS.items():
                    series = indicators.get(name)
                    if series is None or len(series) == 0 or pd.isna(series.iloc[-1]):
                        continue
                    latest_val = series.iloc[-1]
                    if name in ['MA','EMA','VWAP']:
                        if latest_val > 0:
                            price_vs_ma = (current_price - latest_val) / latest_val * 100
                            if price_vs_ma > 2: norm_score = 75
                            elif price_vs_ma > 0: norm_score = 60
                            elif price_vs_ma > -2: norm_score = 50
                            elif price_vs_ma > -5: norm_score = 40
                            else: norm_score = 25
                        else:
                            norm_score = 50
                    else:
                        norm_score = self.normalize_indicator_value(name, latest_val)

                    tf_score += norm_score * weight
                    tf_weight += weight

                if tf_weight > 0:
                    tf_final = tf_score / tf_weight
                    timeframe_scores[tf] = tf_final
                    tf_mult = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                    total_weighted_score += tf_final * tf_mult
                    total_weight += tf_mult

            if total_weight == 0:
                return 'Neutral', 0

            base_score = total_weighted_score / total_weight

            sector_boost = 0
            has_longer_tf = any(t in timeframes_data for t in ['daily', 'weekly', 60])
            if sector in self.best_sectors:
                rank = self.best_sectors.index(sector) + 1
                boosts = [25, 20, 15, 10] if has_longer_tf else [20, 15, 10, 5]
                sector_boost = boosts[rank-1] if rank <= 4 else 0
            elif sector in self.worst_sectors:
                rank = self.worst_sectors.index(sector) + 1
                drags = [-25, -20, -15, -10] if has_longer_tf else [-20, -15, -10, -5]
                sector_boost = drags[rank-1] if rank <= 4 else 0

            base_score += sector_boost

            if len(timeframes_data) >= 5:
                bullish = sum(1 for v in timeframe_scores.values() if v > 55)
                bearish = sum(1 for v in timeframe_scores.values() if v < 45)
                if bullish >= 3: base_score += 8
                elif bearish >= 3: base_score -= 8

            if base_score >= 82: return 'Very Strong Buy', base_score
            if base_score >= 72: return 'Strong Buy', base_score
            if base_score >= 60: return 'Buy', base_score
            if base_score <= 18: return 'Very Strong Sell', base_score
            if base_score <= 28: return 'Strong Sell', base_score
            if base_score <= 40: return 'Sell', base_score
            return 'Neutral', base_score
        except Exception as e:
            logger.error(f"Signal calc error {symbol}: {e}")
            return 'Neutral', 0

    def enhanced_scan_cycle(self):
        start_time = time_module.time()
        now = datetime.now()
        print(f"\n{Colors.CYAN}Starting ENHANCED 4-sector scan at {now.strftime('%H:%M:%S')}{Colors.RESET}")
        print(f"Analyzing: {Colors.YELLOW}5m, 15m, 30m, 60m, Daily, Weekly{Colors.RESET}")
        print(f"Strategy: {Colors.GREEN}Top 4 Best{Colors.RESET} + {Colors.RED}Top 4 Worst{Colors.RESET}")

        self.fetch_live_sectoral_performance_debug()

        target = set()
        for i, sector in enumerate(self.best_sectors[:4]):
            if sector in SECTOR_STOCKS:
                take = [12, 10, 8, 6][i]
                target.update(SECTOR_STOCKS[sector][:take])
        for i, sector in enumerate(self.worst_sectors[:4]):
            if sector in SECTOR_STOCKS:
                take = [12, 10, 8, 6][i]
                target.update(SECTOR_STOCKS[sector][:take])

        symbols = list(target)
        if not symbols:
            print("No target stocks derived from sectors.")
            return

        print(f"Scanning {len(symbols)} symbols across selected sectors...")

        signals = []
        gap_filtered = 0

        # SERIAL FETCHING TO AVOID LZ4 ERRORS
        with ThreadPoolExecutor(max_workers=1) as executor:
            def process_symbol(sym):
                try:
                    timeframes = {}
                    gap_flag = False
                    for tf in [5, 15, 30, 60, 'daily', 'weekly']:
                        df, is_gap = self.fetch_live_data(sym, tf)
                        if df is not None:
                            timeframes[tf] = df
                            if is_gap and tf in [5, 15, 30]:
                                gap_flag = True
                        time_module.sleep(1.5)  # throttle to reduce server-side truncation risks
                    if gap_flag:
                        return None, True
                    if len(timeframes) >= 3:
                        sig, score = self.calculate_enhanced_signals(sym, timeframes)
                        if abs(score - 50) > 15:
                            sec = next((s for s, st in SECTOR_STOCKS.items() if sym in st), 'N/A')
                            return {
                                'symbol': sym, 'signal': sig, 'score': score, 'sector': sec,
                                'timeframes': list(timeframes.keys()), 'time': datetime.now()
                            }, False
                except Exception as e:
                    logger.error(f"Process error {sym}: {e}")
                return None, False

            futures = [executor.submit(process_symbol, s) for s in symbols]
            for fut in as_completed(futures):
                res, gf = fut.result()
                if gf:
                    gap_filtered += 1
                elif res:
                    signals.append(res)

        elapsed = time_module.time() - start_time
        self.gap_down_filtered_count = gap_filtered
        self.display_signals(signals, elapsed, gap_filtered)

    def display_signals(self, signals, elapsed, gap_filtered):
        if os.name == 'posix':
            os.system('clear')
        else:
            os.system('cls')

        now = datetime.now()
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*140}")
        print(f"ENHANCED 4-SECTOR SCANNER - {now.strftime('%Y-%m-%d %H:%M:%S')} IST")
        print(f"{'='*140}{Colors.RESET}")
        if self.last_sectoral_update:
            print(f"Sectors updated: {self.last_sectoral_update.strftime('%H:%M:%S')} | "
                  f"Best: {', '.join(self.best_sectors)} | Worst: {', '.join(self.worst_sectors)}")
        print(f"Scan time: {elapsed:.2f}s | Gap-down filtered: {gap_filtered}")
        if not signals:
            print(f"{Colors.YELLOW}No significant signals this cycle.{Colors.RESET}")
            return

        signals.sort(key=lambda x: x['score'], reverse=True)
        print(f"\nTop signals:")
        for s in signals[:30]:
            print(f"  {s['symbol']:<12} | {s['sector']:<18} | {s['signal']:<18} | Score: {s['score']:.1f} | TF: {','.join(map(str,s['timeframes']))}")

    def run(self):
        print(f"\n{Colors.GREEN}Starting scanner loop...{Colors.RESET}")
        self.is_running = True
        while self.is_running:
            if self.is_market_open():
                self.enhanced_scan_cycle()
                time_module.sleep(self.scan_interval)
            else:
                print(f"{Colors.YELLOW}Market closed. Running a single preview scan in 20s...{Colors.RESET}")
                self.force_sector_update()
                self.enhanced_scan_cycle()
                print(f"{Colors.YELLOW}Sleeping 20 minutes (off-hours).{Colors.RESET}")
                time_module.sleep(1200)

if __name__ == "__main__":
    scanner = Enhanced4SectorScanner()
    scanner.run()
