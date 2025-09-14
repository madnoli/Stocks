# Enhanced Multi-Indicator Scanner with Top 10 Bullish/Bearish Signals
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time as time_module
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

# --- ALL NSE STOCKS ---
ALL_NSE_STOCKS = [
    "IDEA", "BSE", "INDUSTOWER", "RBLBANK", "GLENMARK", "KFINTECH", "NATIONALUM", "EICHERMOT",
    "ASTRAL", "M&M", "VEDL", "ASHOKLEY", "VOLTAS", "CHOLAFIN", "BIOCON", "CAMS", "ANGELONE",
    "EXIDEIND", "MARUTI", "UNOMINDA", "IRFC", "NMDC", "SAIL", "NYKAA", "ABCAPITAL", "TVSMOTOR",
    "POWERGRID", "AMBER", "DRREDDY", "LTF", "RELIANCE", "PNBHOUSING", "NAUKRI", "SHRIRAMFIN",
    "PHOENIXLTD", "PFC", "PAYTM", "KAYNES", "INOXWIND", "IREDA", "CANBK", "CDSL", "NUVAMA",
    "ETERNAL", "MAXHEALTH", "TATAPOWER", "PPLPHARMA", "BDL", "BHARTIARTL", "SBILIFE",
    "AUROPHARMA"]

# --- RATE LIMITER ---
class RateLimitedApiManager:
    def __init__(self, requests_per_second=9, burst_capacity=12):
        self.rate = requests_per_second
        self.capacity = burst_capacity
        self.tokens = burst_capacity
        self.last_refill = time_module.time()
        self.lock = threading.Lock()
    
    def acquire(self):
        with self.lock:
            now = time_module.time()
            tokens_to_add = (now - self.last_refill) * self.rate
            self.tokens = min(self.capacity, self.tokens + tokens_to_add)
            self.last_refill = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return (1 - self.tokens) / self.rate

# --- TECHNICAL INDICATORS ---
class TechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        indicators = {}
        if df is None or len(df) < 20:
            return indicators
        try:
            # RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            indicators['MACD'] = macd_line - signal_line
            
            # Stochastic
            low14 = df['Low'].rolling(window=14).min()
            high14 = df['High'].rolling(window=14).max()
            indicators['Stochastic'] = 100 * (df['Close'] - low14) / (high14 - low14)
            
            # Moving Average
            indicators['MA'] = df['Close'].rolling(window=20).mean()
            
            # ADX
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
            
            # Bollinger Bands
            ma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            upper_band = ma20 + (std20 * 2)
            lower_band = ma20 - (std20 * 2)
            indicators['Bollinger'] = (df['Close'] - ma20) / (upper_band - lower_band) * 100
            
            # ROC
            indicators['ROC'] = df['Close'].pct_change(periods=12) * 100
            
            # OBV
            obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            indicators['OBV'] = obv.pct_change(periods=10) * 100
            
            # CCI
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            indicators['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
            
            # Williams %R
            highest_high = df['High'].rolling(window=14).max()
            lowest_low = df['Low'].rolling(window=14).min()
            indicators['WWL'] = (highest_high - df['Close']) / (highest_high - lowest_low) * -100
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
        return indicators

# --- SIGNAL CALCULATION ---
INDICATOR_WEIGHTS = {
    'RSI': 1.0, 'MACD': 1.2, 'Stochastic': 0.8, 'MA': 1.5,
    'ADX': 1.2, 'Bollinger': 1.0, 'ROC': 0.7, 'OBV': 1.3,
    'CCI': 0.9, 'WWL': 0.9
}
TIMEFRAME_WEIGHTS = {5: 1.0, 15: 1.5, 30: 2.0}

def normalize_indicator_value(indicator_name, value):
    try:
        if indicator_name == 'RSI': return max(0, min(100, value))
        if indicator_name == 'MACD': return 50 + min(25, max(-25, value * 10))
        if indicator_name == 'Stochastic': return max(0, min(100, value))
        if indicator_name == 'MA': return 50
        if indicator_name == 'ADX': return max(0, min(100, value))
        if indicator_name == 'Bollinger': return max(0, min(100, (value + 100) / 2))
        if indicator_name == 'ROC': return 50 + min(25, max(-25, value * 2))
        if indicator_name == 'OBV': return 50 + min(25, max(-25, value))
        if indicator_name == 'CCI': return max(0, min(100, (value + 200) / 4))
        if indicator_name == 'WWL': return max(0, min(100, value + 100))
        return 50
    except:
        return 50

def calculate_multi_indicator_signals(timeframes_data):
    try:
        total_weighted_score, total_weight = 0, 0
        for tf, df in timeframes_data.items():
            if df is None or len(df) < 20: continue
            indicators = TechnicalIndicators.calculate_all_indicators(df)
            if not indicators: continue
            
            tf_score, tf_weight = 0, 0
            for name, weight in INDICATOR_WEIGHTS.items():
                if name in indicators and indicators[name] is not None and not indicators[name].empty:
                    latest_val = indicators[name].iloc[-1]
                    if pd.notna(latest_val):
                        norm_score = normalize_indicator_value(name, latest_val)
                        tf_score += norm_score * weight
                        tf_weight += weight
            
            if tf_weight > 0:
                total_weighted_score += (tf_score / tf_weight) * TIMEFRAME_WEIGHTS.get(tf, 1.0)
                total_weight += TIMEFRAME_WEIGHTS.get(tf, 1.0)
        
        if total_weight == 0: return 'Neutral', 0
        base_score = total_weighted_score / total_weight
        
        if base_score >= 70: return 'Very Strong Buy', base_score
        if base_score >= 60: return 'Strong Buy', base_score
        if base_score >= 55: return 'Buy', base_score
        if base_score <= 30: return 'Very Strong Sell', base_score
        if base_score <= 40: return 'Strong Sell', base_score
        if base_score <= 45: return 'Sell', base_score
        return 'Neutral', base_score
    except Exception as e:
        print(f"Signal calculation error: {e}")
        return 'Neutral', 0

# --- DATA NORMALIZATION ---
def normalize_live_data(df):
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
        print(f"Normalize error: {e}")
        return None

# --- LIVE DATA FETCH ---
def fetch_live_data(symbol, timeframe):
    try:
        tf_map = {5: '5 min', 15: '15 min', 30: '30 min'}
        bar_size = tf_map.get(timeframe)
        if not bar_size: return None
        duration = '10 D' if timeframe <= 15 else '20 D'
        raw_df = td_hist.get_historic_data(symbol, duration=duration, bar_size=bar_size)
        if raw_df is not None and len(raw_df) > 0:
            normalized_df = normalize_live_data(raw_df)
            if normalized_df is not None and len(normalized_df) >= 20:
                return normalized_df.tail(100)
        return None
    except Exception as e:
        print(f"Live data fetch error {symbol}_{timeframe}min: {e}")
        return None

# --- MAIN SCANNER CLASS ---
class Top10BullishBearishScanner:
    def __init__(self):
        self.rate_limiter = RateLimitedApiManager(requests_per_second=9, burst_capacity=12)
        self.is_running = True
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'scan_start_time': None,
            'last_scan_duration': 0
        }
    
    def rate_limited_fetch(self, symbol, timeframe):
        result = self.rate_limiter.acquire()
        if result is not True:
            time_module.sleep(result)
        
        self.stats['total_requests'] += 1
        try:
            data = fetch_live_data(symbol, timeframe)
            if data is not None:
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1
            return data
        except Exception as e:
            self.stats['failed_requests'] += 1
            return None
    
    def scan_all_stocks(self):
        print(f"\n{Colors.CYAN}{Colors.BOLD}🚀 Starting scan of {len(ALL_NSE_STOCKS)} stocks...{Colors.RESET}")
        scan_start = time_module.time()
        self.stats['scan_start_time'] = scan_start
        
        batch_size = 50
        all_signals = []
        
        for batch_start in range(0, len(ALL_NSE_STOCKS), batch_size):
            batch = ALL_NSE_STOCKS[batch_start:batch_start+batch_size]
            batch_num = (batch_start // batch_size) + 1
            total_batches = (len(ALL_NSE_STOCKS) + batch_size - 1) // batch_size
            
            print(f"{Colors.YELLOW}📦 Processing batch {batch_num}/{total_batches} ({len(batch)} stocks)...{Colors.RESET}")
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(self.process_stock, stock) for stock in batch]
                results = [f.result() for f in as_completed(futures) if f.result() is not None]
                all_signals.extend(results)
            
            time_module.sleep(2)
        
        scan_duration = time_module.time() - scan_start
        self.stats['last_scan_duration'] = scan_duration
        
        print(f"{Colors.GREEN}✅ Scan complete: {len(all_signals)} signals, time: {scan_duration:.1f}s{Colors.RESET}")
        self.display_top_signals(all_signals)
    
    def process_stock(self, symbol):
        try:
            timeframes_data = {}
            for tf in [5, 15, 30]:
                df = self.rate_limited_fetch(symbol, tf)
                if df is not None: 
                    timeframes_data[tf] = df
            
            if len(timeframes_data) >= 2:
                signal, score = calculate_multi_indicator_signals(timeframes_data)
                if abs(score - 50) > 10:  # Only significant signals
                    return {
                        'symbol': symbol,
                        'signal': signal,
                        'score': score,
                        'timeframes': len(timeframes_data),
                        'timestamp': datetime.now()
                    }
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
        return None
    
    def display_top_signals(self, signals):
        """Display top 10 bullish and top 10 bearish signals"""
        if not signals:
            print(f"\n{Colors.RED}❌ No significant signals found.{Colors.RESET}")
            return
        
        # Separate bullish and bearish signals
        bullish_signals = [s for s in signals if 'Buy' in s['signal']]
        bearish_signals = [s for s in signals if 'Sell' in s['signal']]
        
        # Sort by signal strength
        bullish_signals.sort(key=lambda x: x['score'], reverse=True)  # Highest scores first
        bearish_signals.sort(key=lambda x: x['score'])  # Lowest scores first
        
        current_time = datetime.now().strftime('%H:%M:%S')
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*100}")
        print(f"🎯 TOP BULLISH & BEARISH SIGNALS - {current_time} IST")
        print(f"{'='*100}{Colors.RESET}")
        
        # API Statistics
        success_rate = (self.stats['successful_requests'] / max(1, self.stats['total_requests'])) * 100
        print(f"{Colors.BLUE}📊 API Stats: {self.stats['successful_requests']}/{self.stats['total_requests']} "
              f"({success_rate:.1f}% success) | ⚡ Duration: {self.stats['last_scan_duration']:.1f}s{Colors.RESET}")
        
        # Display Top 10 Bullish
        print(f"\n{Colors.GREEN}{Colors.BOLD}🟢 TOP 10 BULLISH SIGNALS:{Colors.RESET}")
        print(f"{'Rank':<5} {'Stock':<12} {'Signal':<18} {'Score':>8} {'TFs':>4} {'Strength':<15}")
        print(f"{Colors.GREEN}{'='*75}{Colors.RESET}")
        
        for i, signal in enumerate(bullish_signals[:10]):
            rank = i + 1
            deviation = abs(signal['score'] - 50)
            
            if deviation >= 25:
                strength = f"{Colors.RED}🔥 Very Strong{Colors.RESET}"
            elif deviation >= 20:
                strength = f"{Colors.YELLOW}💪 Strong{Colors.RESET}"
            elif deviation >= 15:
                strength = f"{Colors.CYAN}📈 Moderate{Colors.RESET}"
            else:
                strength = f"{Colors.WHITE}⚡ Weak{Colors.RESET}"
            
            color = Colors.GREEN if deviation >= 20 else Colors.YELLOW
            print(f"{rank:<5} {color}{signal['symbol']:<12}{Colors.RESET} "
                  f"{color}{signal['signal']:<18}{Colors.RESET} "
                  f"{Colors.WHITE}{signal['score']:>8.1f}{Colors.RESET} "
                  f"{Colors.CYAN}{signal['timeframes']:>4}{Colors.RESET} {strength}")
        
        # Display Top 10 Bearish
        print(f"\n{Colors.RED}{Colors.BOLD}🔴 TOP 10 BEARISH SIGNALS:{Colors.RESET}")
        print(f"{'Rank':<5} {'Stock':<12} {'Signal':<18} {'Score':>8} {'TFs':>4} {'Strength':<15}")
        print(f"{Colors.RED}{'='*75}{Colors.RESET}")
        
        for i, signal in enumerate(bearish_signals[:10]):
            rank = i + 1
            deviation = abs(signal['score'] - 50)
            
            if deviation >= 25:
                strength = f"{Colors.RED}🔥 Very Strong{Colors.RESET}"
            elif deviation >= 20:
                strength = f"{Colors.YELLOW}💪 Strong{Colors.RESET}"
            elif deviation >= 15:
                strength = f"{Colors.CYAN}📈 Moderate{Colors.RESET}"
            else:
                strength = f"{Colors.WHITE}⚡ Weak{Colors.RESET}"
            
            color = Colors.RED if deviation >= 20 else Colors.YELLOW
            print(f"{rank:<5} {color}{signal['symbol']:<12}{Colors.RESET} "
                  f"{color}{signal['signal']:<18}{Colors.RESET} "
                  f"{Colors.WHITE}{signal['score']:>8.1f}{Colors.RESET} "
                  f"{Colors.CYAN}{signal['timeframes']:>4}{Colors.RESET} {strength}")
        
        # Summary statistics
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}📊 SUMMARY:{Colors.RESET}")
        print(f"{Colors.GREEN}🟢 Total Bullish: {len(bullish_signals)}{Colors.RESET}")
        print(f"{Colors.RED}🔴 Total Bearish: {len(bearish_signals)}{Colors.RESET}")
        print(f"{Colors.CYAN}📈 Total Signals: {len(signals)}{Colors.RESET}")
        
        if bullish_signals:
            avg_bullish_score = sum(s['score'] for s in bullish_signals) / len(bullish_signals)
            print(f"{Colors.GREEN}🟢 Avg Bullish Score: {avg_bullish_score:.1f}{Colors.RESET}")
        
        if bearish_signals:
            avg_bearish_score = sum(s['score'] for s in bearish_signals) / len(bearish_signals)
            print(f"{Colors.RED}🔴 Avg Bearish Score: {avg_bearish_score:.1f}{Colors.RESET}")
        
        print(f"{Colors.CYAN}{Colors.BOLD}{'='*100}{Colors.RESET}")
    
    def run_continuous_scanner(self):
        """Run scanner every 5 minutes"""
        try:
            while self.is_running:
                self.scan_all_stocks()
                if self.is_running:
                    print(f"\n{Colors.YELLOW}💤 Waiting 5 minutes for next scan...{Colors.RESET}")
                    time_module.sleep(300)  # 5 minutes
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}🛑 Scanner stopped by user{Colors.RESET}")
            self.is_running = False

# --- ENTRY POINT ---
if __name__ == "__main__":
    print(f"\n{Colors.CYAN}{Colors.BOLD}🎯 TOP 10 BULLISH/BEARISH STOCK SCANNER{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*60}{Colors.RESET}")
    
    scanner = Top10BullishBearishScanner()
    
    try:
        # Run single scan
        scanner.scan_all_stocks()
        
        # Uncomment below for continuous scanning every 5 minutes
        # scanner.run_continuous_scanner()
        
    except KeyboardInterrupt:
        scanner.is_running = False
        print(f"\n{Colors.YELLOW}🛑 Scanner stopped by user{Colors.RESET}")
