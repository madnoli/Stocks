# Top 10 Bullish & Bearish - LIVE SCANNER
# Features: Scans a master list in real-time and ranks the top 10 buy and sell signals every 5 minutes during market hours.
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import time as time_module
import pytz
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from truedata.history import TD_hist
import logging
import warnings
from collections import deque

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# --- TRUEDATA CONFIG ---
TD_USERNAME = "Trial106"
TD_PASSWORD = "raj106"
try:
    td_hist = TD_hist(TD_USERNAME, TD_PASSWORD, log_level=logging.WARNING)
except Exception as e:
    logger.error(f"Failed to initialize TrueData history client: {e}")
    exit()

# --- COLOR CODES ---
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

# --- MASTER STOCK LIST ---
MASTER_STOCK_LIST = """
NSE:IDEA, NSE:BSE, NSE:INDUSTOWER, NSE:RBLBANK, NSE:GLENMARK, NSE:KFINTECH, NSE:NATIONALUM, NSE:EICHERMOT, NSE:ASTRAL, NSE:M&M, NSE:VEDL,
NSE:ASHOKLEY, NSE:VOLTAS, NSE:CHOLAFIN, NSE:BIOCON, NSE:CAMS, NSE:ANGELONE, NSE:EXIDEIND, NSE:MARUTI, NSE:UNOMINDA, NSE:IRFC, NSE:NMDC,
NSE:SAIL, NSE:NYKAA, NSE:ABCAPITAL, NSE:TVSMOTOR, NSE:POWERGRID, NSE:AMBER, NSE:DRREDDY, NSE:LTF, NSE:RELIANCE, NSE:PNBHOUSING, NSE:NAUKRI,
NSE:SHRIRAMFIN, NSE:PHOENIXLTD, NSE:PFC, NSE:PAYTM, NSE:KAYNES, NSE:INOXWIND, NSE:IREDA, NSE:CANBK, NSE:CDSL, NSE:NUVAMA, NSE:ETERNAL,
NSE:MAXHEALTH, NSE:TATAPOWER, NSE:PPLPHARMA, NSE:BDL, NSE:BHARTIARTL, NSE:SBILIFE, NSE:AUROPHARMA, NSE:SUZLON, NSE:LAURUSLABS, NSE:RVNL,
NSE:YESBANK, NSE:MFSL, NSE:SONACOMS, NSE:SUNPHARMA, NSE:OIL, NSE:HDFCLIFE, NSE:SAMMAANCAP, NSE:KPITTECH, NSE:HINDALCO, NSE:IIFL, NSE:BAJAJFINSV,
NSE:TATAMOTORS, NSE:ALKEM, NSE:BHEL, NSE:HINDZINC, NSE:HUDCO, NSE:BANDHANBNK, NSE:AXISBANK, NSE:TATASTEEL, NSE:RECLTD, NSE:IDFCFIRSTB,
NSE:NBCC, NSE:BHARATFORG, NSE:360ONE, NSE:ASIANPAINT, NSE:BOSCHLTD, NSE:TATAELXSI, NSE:MUTHOOTFIN, NSE:IRCTC, NSE:UNIONBANK, NSE:BANKINDIA,
NSE:FEDERALBNK, NSE:SHREECEM, NSE:TITAGARH, NSE:JSWENERGY, NSE:PNB, NSE:COALINDIA, NSE:BAJFINANCE, NSE:MOTHERSON, NSE:JINDALSTEL,
NSE:INDUSINDBK, NSE:JUBLFOOD, NSE:LUPIN, NSE:HEROMOTOCO, NSE:HDFCBANK, NSE:CNXMIDCAP, NSE:ZYDUSLIFE, NSE:BAJAJ-AUTO, NSE:MANAPPURAM,
NSE:BANKBARODA, NSE:TATACONSUM, NSE:CONCOR, NSE:ADANIENT, NSE:BANKNIFTY, NSE:DALBHARAT, NSE:JSWSTEEL, NSE:HDFCAMC, NSE:NIFTY, NSE:CUMMINSIND,
NSE:DIXON, NSE:ADANIGREEN, NSE:INDIANB, NSE:KALYANKJIL, NSE:INDHOTEL, NSE:TRENT, NSE:LICHSGFIN, NSE:JIOFIN, NSE:IOC, NSE:BLUESTARCO,
NSE:CROMPTON, NSE:LICI, NSE:BRITANNIA, NSE:BPCL, NSE:HAVELLS, NSE:PGEL, NSE:OFSS, NSE:AMBUJACEM, NSE:ICICIBANK, NSE:TIINDIA, NSE:GRASIM,
NSE:FORTIS, NSE:SBICARD, NSE:HFCL, NSE:KOTAKBANK, NSE:HINDPETRO, NSE:SUPREMEIND, NSE:LTIM, NSE:AUBANK, NSE:ADANIENSOL, NSE:NESTLEIND, NSE:DLF,
NSE:SBIN, NSE:NHPC, NSE:MAZDOCK, NSE:NCC, NSE:ULTRACEMCO, NSE:POLYCAB, NSE:DELHIVERY, NSE:GAIL, NSE:NTPC, NSE:INDIGO, NSE:PETRONET, NSE:BEL,
NSE:ADANIPORTS, NSE:APLAPOLLO, NSE:IEX, NSE:MCX, NSE:ICICIPRULI, NSE:CGPOWER, NSE:WIPRO, NSE:TORNTPHARM, NSE:TATACHEM, NSE:TATATECH, NSE:ONGC,
NSE:GMRAIRPORT, NSE:TITAN, NSE:MANKIND, NSE:UNITDSPR, NSE:HAL, NSE:DMART, NSE:PIDILITIND, NSE:PAGEIND, NSE:ABB, NSE:MARICO, NSE:UPL,
NSE:SOLARINDS, NSE:LT, NSE:DABUR, NSE:GODREJCP, NSE:PATANJALI, NSE:APOLLOHOSP, NSE:HINDUNILVR, NSE:INFY, NSE:SYNGENE, NSE:SRF, NSE:LODHA,
NSE:CYIENT, NSE:TECHM, NSE:TCS, NSE:CIPLA, NSE:ICICIGI, NSE:COLPAL, NSE:HCLTECH, NSE:IGL, NSE:OBEROIRLTY, NSE:COFORG, NSE:DIVISLAB,
NSE:GODREJPROP, NSE:PIIND, NSE:ITC, NSE:SIEMENS, NSE:KEI, NSE:MPHASIS, NSE:POLICYBZR, NSE:TORNTPOWER, NSE:PRESTIGE, NSE:PERSISTENT, NSE:VBL
"""

# --- RATE LIMITER ---
class RateLimiter:
    def __init__(self, calls_per_second):
        self.rate_limit = calls_per_second
        self.requests = deque()
        self.lock = threading.Lock()
        logger.info(f"Rate Limiter initialized to {calls_per_second} calls/sec.")

    def acquire(self):
        with self.lock:
            while self.requests and time_module.time() - self.requests[0] > 1.0:
                self.requests.popleft()
            if len(self.requests) >= self.rate_limit:
                wait_time = (self.requests[0] + 1.0) - time_module.time()
                if wait_time > 0:
                    time_module.sleep(wait_time)
                self.requests.popleft()
            self.requests.append(time_module.time())

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
        indicators = {}
        if df is None or len(df) < 26: return indicators
        try:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], 0)))
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            indicators['MACD'] = macd_line - signal_line
            low14 = df['Low'].rolling(window=14).min()
            high14 = df['High'].rolling(window=14).max()
            indicators['Stochastic'] = 100 * (df['Close'] - low14) / (high14 - low14)
            indicators['MA'] = df['Close'].rolling(window=20).mean()
            tr = pd.DataFrame(index=df.index); tr['h-l'] = df['High'] - df['Low']; tr['h-pc'] = abs(df['High'] - df['Close'].shift(1)); tr['l-pc'] = abs(df['Low'] - df['Close'].shift(1)); true_range = tr.max(axis=1); atr = true_range.ewm(alpha=1/14, adjust=False).mean(); plus_dm = df['High'].diff().apply(lambda x: x if x > 0 else 0); minus_dm = df['Low'].diff().apply(lambda x: -x if x < 0 else 0); plus_di = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr; minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr; dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di); indicators['ADX'] = dx.ewm(alpha=1/14, adjust=False).mean()
            ma20 = df['Close'].rolling(window=20).mean(); std20 = df['Close'].rolling(window=20).std(); upper_band = ma20 + (std20 * 2); lower_band = ma20 - (std20 * 2); indicators['Bollinger'] = (df['Close'] - lower_band) / (upper_band - lower_band) * 100
            indicators['ROC'] = df['Close'].pct_change(periods=12) * 100
            obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum(); indicators['OBV'] = obv.pct_change(periods=10) * 100
            tp = (df['High'] + df['Low'] + df['Close']) / 3; ma_tp = tp.rolling(window=20).mean(); mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True); indicators['CCI'] = (tp - ma_tp) / (0.015 * mad)
            highest_high = df['High'].rolling(window=14).max(); lowest_low = df['Low'].rolling(window=14).min(); indicators['WWL'] = -100 * (highest_high - df['Close']) / (highest_high - lowest_low)
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        return indicators

# --- LIVE SCANNER CLASS ---
class LiveScanner:
    def __init__(self):
        self.rate_limiter = RateLimiter(calls_per_second=9)
        self.is_running = False
        self.scan_interval = 300 # 5 minutes
        self.market_start = time(9, 15)
        self.market_end = time(15, 30)
        logger.info("🚀 Live Scanner Initialized")

    def get_all_scannable_stocks(self):
        indices_to_exclude = ['NIFTY', 'BANKNIFTY', 'CNXMIDCAP']
        symbols = [s.strip().replace('NSE:', '') for s in MASTER_STOCK_LIST.split(',') if s.strip()]
        filtered_symbols = [s for s in symbols if s not in indices_to_exclude]
        unique_symbols = sorted(list(set(filtered_symbols)))
        logger.info(f"Loaded {len(unique_symbols)} unique stocks for scanning (excluded {len(indices_to_exclude)} indices).")
        return unique_symbols

    def is_market_open(self):
        """Checks if the current time is within Indian market hours."""
        kolkata_time = datetime.now(pytz.timezone('Asia/Kolkata'))
        return self.market_start <= kolkata_time.time() <= self.market_end and kolkata_time.weekday() < 5
        
    def normalize_data(self, df):
        if df is None or df.empty: return None
        df_copy = df.copy()
        df_copy.columns = df_copy.columns.str.lower()
        time_col_name = None
        if 'time' in df_copy.columns: time_col_name = 'time'
        elif 'timestamp' in df_copy.columns: time_col_name = 'timestamp'
        elif 'date' in df_copy.columns: time_col_name = 'date'
        if time_col_name is None: return None
        rename_map = {time_col_name: 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'vol': 'Volume', 'volume': 'Volume'}
        required_cols = [col for col in rename_map if col in df_copy.columns]
        df_copy = df_copy[required_cols].rename(columns=rename_map)
        df_copy['Date'] = pd.to_datetime(df_copy['Date'])
        df_copy.set_index('Date', inplace=True)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        return df_copy.dropna().sort_index()

    def fetch_live_data(self, symbol, timeframe):
        try:
            tf_map = {5: '5 min', 15: '15 min', 30: '30 min'}
            self.rate_limiter.acquire()
            duration = '10 D' if timeframe <= 15 else '20 D'
            raw_df = td_hist.get_historic_data(symbol, duration=duration, bar_size=tf_map.get(timeframe))
            return self.normalize_data(raw_df)
        except Exception as e:
            logger.error(f"Data fetch error for {symbol} ({timeframe}min): {e}")
            return None

    def normalize_indicator_value(self, name, value):
        if pd.isna(value): return 50
        try:
            if name == 'RSI': return max(0, min(100, value))
            if name == 'MACD': return 50 + min(25, max(-25, value * 10))
            if name == 'Stochastic': return max(0, min(100, value))
            if name == 'ADX': return max(0, min(100, value))
            if name == 'Bollinger': return max(0, min(100, value))
            if name == 'ROC': return 50 + min(25, max(-25, value * 2))
            if name == 'OBV': return 50 + min(25, max(-25, value))
            if name == 'CCI': return max(0, min(100, (value + 200) / 4))
            if name == 'WWL': return max(0, min(100, value + 100))
        except: pass
        return 50

    def calculate_signal_score(self, symbol, timeframes_data):
        total_weighted_score, total_weight = 0, 0
        for tf, df in timeframes_data.items():
            if df is None or len(df) < 26: continue
            indicators = TechnicalIndicators.calculate_all_indicators(df)
            if not indicators: continue
            tf_score, tf_weight = 0, 0
            for name, weight in INDICATOR_WEIGHTS.items():
                if name in indicators and not indicators[name].empty:
                    latest_val = indicators[name].iloc[-1]
                    norm_score = self.normalize_indicator_value(name, latest_val)
                    tf_score += norm_score * weight; tf_weight += weight
            if tf_weight > 0:
                total_weighted_score += (tf_score / tf_weight) * TIMEFRAME_WEIGHTS.get(tf, 1.0)
                total_weight += TIMEFRAME_WEIGHTS.get(tf, 1.0)
        if total_weight == 0: return 'Neutral', 50
        final_score = total_weighted_score / total_weight
        if final_score >= 70: signal = 'Very Strong Buy'
        elif final_score >= 60: signal = 'Strong Buy'
        elif final_score >= 55: signal = 'Buy'
        elif final_score <= 30: signal = 'Very Strong Sell'
        elif final_score <= 40: signal = 'Strong Sell'
        elif final_score <= 45: signal = 'Sell'
        else: signal = 'Neutral'
        return signal, final_score

    def scan_cycle(self):
        logger.info("Starting new scan cycle...")
        start_time = time_module.time()
        target_stocks = self.get_all_scannable_stocks()
        all_signals = []
        with ThreadPoolExecutor(max_workers=15) as executor:
            def process_stock(symbol):
                try:
                    timeframes_data = {tf: self.fetch_live_data(symbol, tf) for tf in [5, 15, 30]}
                    timeframes_data = {tf: df for tf, df in timeframes_data.items() if df is not None}
                    if len(timeframes_data) >= 2:
                        signal, score = self.calculate_signal_score(symbol, timeframes_data)
                        if signal != 'Neutral':
                            return {'symbol': symbol, 'signal': signal, 'score': score, 'tfs': len(timeframes_data)}
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                return None
            futures = [executor.submit(process_stock, symbol) for symbol in target_stocks]
            all_signals = [future.result() for future in as_completed(futures) if future.result()]
        scan_time = time_module.time() - start_time
        self.display_scan_results(all_signals, scan_time, len(target_stocks))

    def display_scan_results(self, signals, scan_time, total_scanned):
        os.system('clear' if os.name == 'posix' else 'cls')
        now_str = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{Colors.CYAN}{Colors.BOLD}{'='*80}")
        print(f"🎯 LIVE SCANNER RESULTS - {now_str} IST")
        print(f"{'='*80}{Colors.RESET}")
        print(f"⚡ Scan completed for {total_scanned} stocks in {scan_time:.2f} seconds.")
        print("-" * 80)
        bullish = sorted([s for s in signals if 'Buy' in s['signal']], key=lambda x: x['score'], reverse=True)
        bearish = sorted([s for s in signals if 'Sell' in s['signal']], key=lambda x: x['score'])
        print(f"\n{Colors.GREEN}{Colors.BOLD}📈 TOP 10 BULLISH SIGNALS{Colors.RESET}")
        if not bullish: print(f"{Colors.YELLOW}No significant bullish signals found.{Colors.RESET}")
        else:
            print(f"{'Rank':<5} {'Stock':<12} {'Signal':<18} {'Score':>7} {'TFs':>4}")
            print(f"{Colors.CYAN}{'-'*50}{Colors.RESET}")
            for i, s in enumerate(bullish[:10]):
                color = Colors.GREEN + (Colors.BOLD if 'Very' in s['signal'] else "")
                print(f"{i+1:<5} {Colors.WHITE}{s['symbol']:<12}{Colors.RESET} {color}{s['signal']:<18}{Colors.RESET} {s['score']:>7.1f} {Colors.BLUE}{s['tfs']:>4}{Colors.RESET}")
        print(f"\n{Colors.RED}{Colors.BOLD}📉 TOP 10 BEARISH SIGNALS{Colors.RESET}")
        if not bearish: print(f"{Colors.YELLOW}No significant bearish signals found.{Colors.RESET}")
        else:
            print(f"{'Rank':<5} {'Stock':<12} {'Signal':<18} {'Score':>7} {'TFs':>4}")
            print(f"{Colors.CYAN}{'-'*50}{Colors.RESET}")
            for i, s in enumerate(bearish[:10]):
                color = Colors.RED + (Colors.BOLD if 'Very' in s['signal'] else "")
                print(f"{i+1:<5} {Colors.WHITE}{s['symbol']:<12}{Colors.RESET} {color}{s['signal']:<18}{Colors.RESET} {s['score']:>7.1f} {Colors.BLUE}{s['tfs']:>4}{Colors.RESET}")
        print(f"\n{Colors.CYAN}{'='*80}{Colors.RESET}")

    def run(self):
        self.is_running = True
        logger.info("Scanner is starting. Press Ctrl+C to stop.")
        try:
            while self.is_running:
                if self.is_market_open():
                    self.scan_cycle()
                else:
                    logger.info("Market is currently closed. Checking again in 5 minutes.")
                
                # Wait until the next 5-minute interval
                wait_time = self.scan_interval - (time_module.time() % self.scan_interval)
                next_scan_time = (datetime.now() + timedelta(seconds=wait_time)).strftime('%H:%M:%S')
                print(f"{Colors.YELLOW}⏰ Next check at approximately {next_scan_time}{Colors.RESET}")
                time_module.sleep(wait_time)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.is_running = False
        print(f"\n{Colors.YELLOW}🛑 Scanner has been stopped by the user.{Colors.RESET}")
        logger.info("Scanner stopped.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    scanner = LiveScanner()
    scanner.run()