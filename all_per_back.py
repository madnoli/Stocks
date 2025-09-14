# Enhanced Multi-Indicator Scanner with Historical Simulation for September 12, 2025
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
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
    "AUROPHARMA"
]

# --- RATE LIMITER ---
class RateLimitedApiManager:
    def __init__(self, requests_per_second=8, burst_capacity=10):
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

# --- HISTORICAL SIMULATION SCANNER ---
class HistoricalSimulationScanner:
    """Scanner that simulates real-time behavior using historical data from Sept 12, 2025"""
    
    def __init__(self, target_date="2025-09-12"):
        self.target_date = target_date
        self.simulation_start_time = dt_time(9, 15)  # Market opening
        self.simulation_end_time = dt_time(15, 30)   # Market closing
        self.current_simulation_time = None
        self.is_running = False
        
        # Rate limiting for historical data (TrueData historical limits: 10/sec, 600/min)
        self.rate_limiter = RateLimitedApiManager(requests_per_second=8, burst_capacity=10)
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'scan_cycles': 0,
            'signals_generated': []
        }
        
        print(f"🏛️ Historical Simulation Scanner initialized for {target_date}")
        
    def normalize_historical_data(self, df):
        """Normalize historical data similar to live data"""
        try:
            if df is None or df.empty:
                return None
                
            df_clean = df.copy()
            
            # Handle column mapping
            cols = [c.lower() for c in df_clean.columns]
            column_mapping = {}
            
            for i, col in enumerate(cols):
                original_col = df_clean.columns[i]
                if 'time' in col or 'date' in col:
                    column_mapping[original_col] = 'Date'
                elif 'open' in col:
                    column_mapping[original_col] = 'Open'
                elif 'high' in col:
                    column_mapping[original_col] = 'High'
                elif 'low' in col:
                    column_mapping[original_col] = 'Low'
                elif 'close' in col:
                    column_mapping[original_col] = 'Close'
                elif 'vol' in col:
                    column_mapping[original_col] = 'Volume'
            
            df_clean = df_clean.rename(columns=column_mapping)
            
            # Ensure required columns exist
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            if not all(col in df_clean.columns for col in required_cols):
                return None
                
            if 'Volume' not in df_clean.columns:
                df_clean['Volume'] = 1000
                
            # Process datetime
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            if df_clean['Date'].dt.tz is not None:
                df_clean['Date'] = df_clean['Date'].dt.tz_localize(None)
                
            df_clean.set_index('Date', inplace=True)
            
            # Convert price columns to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
            return df_clean.dropna().sort_index() if len(df_clean) >= 20 else None
            
        except Exception as e:
            print(f"Normalize error: {e}")
            return None

    def get_historical_data_at_time(self, symbol, timeframe, target_datetime):
        """Fetch historical data as if it was 'current' at target_datetime"""
        try:
            # Map timeframes to TrueData format
            tf_map = {5: '5 min', 15: '15 min', 30: '30 min'}
            bar_size = tf_map.get(timeframe)
            if not bar_size:
                return None
            
            # Calculate duration needed - get enough historical data
            duration = '15 D'  # 15 days should be sufficient
            
            # Fetch historical data ending at our target date
            raw_df = td_hist.get_historic_data(
                symbol, 
                duration=duration, 
                bar_size=bar_size
            )
            
            if raw_df is not None and len(raw_df) > 0:
                normalized_df = self.normalize_historical_data(raw_df)
                if normalized_df is not None and len(normalized_df) >= 20:
                    # Filter data up to our target time
                    target_date_only = target_datetime.date()
                    target_time_filter = normalized_df.index.date <= target_date_only
                    filtered_df = normalized_df[target_time_filter]
                    
                    # Further filter to specific time if same day
                    if len(filtered_df) > 0:
                        same_day_data = filtered_df[filtered_df.index.date == target_date_only]
                        if len(same_day_data) > 0:
                            # Get data up to the target time on the same day
                            time_filtered = same_day_data[same_day_data.index.time <= target_datetime.time()]
                            if len(time_filtered) >= 20:
                                return time_filtered.tail(100)
                            
                    # If not enough same-day data, return recent historical data
                    if len(filtered_df) >= 20:
                        return filtered_df.tail(100)
                        
            return None
            
        except Exception as e:
            print(f"Historical data fetch error {symbol}_{timeframe}min: {e}")
            return None

    def rate_limited_historical_fetch(self, symbol, timeframe, target_datetime):
        """Rate-limited historical data fetch"""
        result = self.rate_limiter.acquire()
        if result is not True:
            time_module.sleep(result)
        
        self.stats['total_requests'] += 1
        try:
            data = self.get_historical_data_at_time(symbol, timeframe, target_datetime)
            if data is not None:
                self.stats['successful_requests'] += 1
            else:
                self.stats['failed_requests'] += 1
            return data
        except Exception as e:
            self.stats['failed_requests'] += 1
            return None

    def simulate_5min_intervals(self):
        """Generate 5-minute intervals for the trading day"""
        target_date = datetime.strptime(self.target_date, "%Y-%m-%d").date()
        
        # Create 5-minute intervals for the trading day
        current_time = datetime.combine(target_date, self.simulation_start_time)
        end_time = datetime.combine(target_date, self.simulation_end_time)
        
        interval_times = []
        while current_time <= end_time:
            interval_times.append(current_time)
            current_time += timedelta(minutes=5)
        
        return interval_times

    def process_stock_historical(self, symbol, target_datetime):
        """Process a single stock using historical data up to target_datetime"""
        try:
            timeframes_data = {}
            
            # Fetch historical data for all timeframes up to target_datetime
            for tf in [5, 15, 30]:
                df = self.rate_limited_historical_fetch(symbol, tf, target_datetime)
                if df is not None:
                    timeframes_data[tf] = df
            
            # Need at least 2 timeframes for reliable signals
            if len(timeframes_data) >= 2:
                signal, score = calculate_multi_indicator_signals(timeframes_data)
                
                if abs(score - 50) > 10:  # Only significant signals
                    return {
                        'symbol': symbol,
                        'signal': signal,
                        'score': score,
                        'timeframes': len(timeframes_data),
                        'timestamp': target_datetime
                    }
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
        
        return None

    def scan_stocks_at_time(self, target_datetime):
        """Scan all stocks as if it was the target datetime"""
        print(f"{Colors.YELLOW}📊 Scanning {len(ALL_NSE_STOCKS)} stocks at {target_datetime.strftime('%H:%M:%S')}...{Colors.RESET}")
        
        scan_start = time_module.time()
        all_signals = []
        
        batch_size = 25  # Smaller batches for historical data to respect rate limits
        
        for batch_start in range(0, len(ALL_NSE_STOCKS), batch_size):
            batch = ALL_NSE_STOCKS[batch_start:batch_start+batch_size]
            
            with ThreadPoolExecutor(max_workers=5) as executor:  # Reduced workers for historical
                futures = [executor.submit(self.process_stock_historical, stock, target_datetime) for stock in batch]
                results = [f.result() for f in as_completed(futures) if f.result() is not None]
                all_signals.extend(results)
            
            time_module.sleep(1)  # Small delay between batches
        
        scan_duration = time_module.time() - scan_start
        print(f"{Colors.GREEN}✅ Historical scan complete: {len(all_signals)} signals in {scan_duration:.1f}s{Colors.RESET}")
        
        return all_signals

    def display_simulation_results(self, signals, sim_time, cycle_num):
        """Display results for this simulation cycle"""
        if not signals:
            print(f"{Colors.RED}❌ No significant signals at {sim_time.strftime('%H:%M:%S')}{Colors.RESET}")
            return
        
        # Separate bullish and bearish
        bullish_signals = [s for s in signals if 'Buy' in s['signal']]
        bearish_signals = [s for s in signals if 'Sell' in s['signal']]
        
        bullish_signals.sort(key=lambda x: x['score'], reverse=True)
        bearish_signals.sort(key=lambda x: x['score'])
        
        print(f"\n{Colors.MAGENTA}📊 SIGNALS AT {sim_time.strftime('%H:%M:%S')} - Cycle {cycle_num}{Colors.RESET}")
        print(f"{Colors.BLUE}🟢 Bullish: {len(bullish_signals)} | 🔴 Bearish: {len(bearish_signals)} | 📈 Total: {len(signals)}{Colors.RESET}")
        
        # Show top 10 bullish
        if bullish_signals:
            print(f"\n{Colors.GREEN}🟢 TOP 10 BULLISH:{Colors.RESET}")
            print(f"{'Rank':<5} {'Stock':<12} {'Signal':<18} {'Score':>8} {'Strength'}")
            print(f"{Colors.GREEN}{'='*60}{Colors.RESET}")
            for i, signal in enumerate(bullish_signals[:10]):
                rank = i + 1
                deviation = abs(signal['score'] - 50)
                if deviation >= 25:
                    strength = "🔥 Very Strong"
                elif deviation >= 20:
                    strength = "💪 Strong"
                elif deviation >= 15:
                    strength = "📈 Moderate"
                else:
                    strength = "⚡ Weak"
                
                print(f"{rank:<5} {Colors.GREEN}{signal['symbol']:<12}{Colors.RESET} "
                      f"{signal['signal']:<18} {signal['score']:>8.1f} {strength}")
        
        # Show top 10 bearish
        if bearish_signals:
            print(f"\n{Colors.RED}🔴 TOP 10 BEARISH:{Colors.RESET}")
            print(f"{'Rank':<5} {'Stock':<12} {'Signal':<18} {'Score':>8} {'Strength'}")
            print(f"{Colors.RED}{'='*60}{Colors.RESET}")
            for i, signal in enumerate(bearish_signals[:10]):
                rank = i + 1
                deviation = abs(signal['score'] - 50)
                if deviation >= 25:
                    strength = "🔥 Very Strong"
                elif deviation >= 20:
                    strength = "💪 Strong"
                elif deviation >= 15:
                    strength = "📈 Moderate"
                else:
                    strength = "⚡ Weak"
                
                print(f"{rank:<5} {Colors.RED}{signal['symbol']:<12}{Colors.RESET} "
                      f"{signal['signal']:<18} {signal['score']:>8.1f} {strength}")

    def run_historical_simulation(self):
        """Run the complete historical simulation"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}🏛️ HISTORICAL SIMULATION FOR SEPTEMBER 12, 2025{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")
        
        # Get all 5-minute intervals for the day
        simulation_times = self.simulate_5min_intervals()
        
        print(f"📅 Simulation Date: {Colors.YELLOW}{self.target_date}{Colors.RESET}")
        print(f"⏰ Trading Hours: {Colors.YELLOW}{self.simulation_start_time} - {self.simulation_end_time}{Colors.RESET}")
        print(f"🔄 Total Intervals: {Colors.YELLOW}{len(simulation_times)} (every 5 minutes){Colors.RESET}")
        print(f"🎯 Stocks to Scan: {Colors.YELLOW}{len(ALL_NSE_STOCKS)}{Colors.RESET}")
        
        self.is_running = True
        
        try:
            for i, sim_time in enumerate(simulation_times):
                if not self.is_running:
                    break
                    
                self.current_simulation_time = sim_time
                cycle_num = i + 1
                
                print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*80}")
                print(f"📊 SCAN CYCLE {cycle_num}/{len(simulation_times)} - {sim_time.strftime('%H:%M:%S')} IST")
                print(f"{'='*80}{Colors.RESET}")
                
                # Run scan for this time interval
                signals = self.scan_stocks_at_time(sim_time)
                
                # Display results
                self.display_simulation_results(signals, sim_time, cycle_num)
                
                # Store results
                self.stats['signals_generated'].append({
                    'time': sim_time,
                    'cycle': cycle_num,
                    'signals': signals,
                    'total_signals': len(signals)
                })
                
                self.stats['scan_cycles'] += 1
                
                # Brief pause for readability (remove this for faster simulation)
                print(f"{Colors.YELLOW}⏳ Next scan in 5 minutes (simulated)...{Colors.RESET}")
                time_module.sleep(2)  # 2 second pause between cycles
                
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}🛑 Historical simulation stopped by user{Colors.RESET}")
        finally:
            self.is_running = False
            self.show_simulation_summary()

    def show_simulation_summary(self):
        """Show complete simulation summary"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*80}")
        print(f"📋 HISTORICAL SIMULATION SUMMARY - {self.target_date}")
        print(f"{'='*80}{Colors.RESET}")
        
        total_signals = sum(cycle['total_signals'] for cycle in self.stats['signals_generated'])
        avg_signals = total_signals / max(1, len(self.stats['signals_generated']))
        
        success_rate = (self.stats['successful_requests'] / max(1, self.stats['total_requests'])) * 100
        
        print(f"{Colors.BLUE}📊 API Performance:{Colors.RESET}")
        print(f"  • Total Requests: {self.stats['total_requests']}")
        print(f"  • Success Rate: {success_rate:.1f}%")
        print(f"  • Scan Cycles: {self.stats['scan_cycles']}")
        
        print(f"\n{Colors.GREEN}📈 Signal Performance:{Colors.RESET}")
        print(f"  • Total Signals Generated: {total_signals}")
        print(f"  • Average Signals per Cycle: {avg_signals:.1f}")
        print(f"  • Most Active Time: {self.get_most_active_time()}")
        
        # Show signal timeline
        print(f"\n{Colors.MAGENTA}📅 Signal Activity Timeline:{Colors.RESET}")
        for cycle in self.stats['signals_generated']:
            time_str = cycle['time'].strftime('%H:%M')
            signal_count = cycle['total_signals']
            bar = '█' * min(20, signal_count)  # Visual bar
            print(f"  {time_str}: {signal_count:>3} signals {Colors.CYAN}{bar}{Colors.RESET}")
    
    def get_most_active_time(self):
        """Find the time with most signals"""
        if not self.stats['signals_generated']:
            return "N/A"
        
        max_signals = max(self.stats['signals_generated'], key=lambda x: x['total_signals'])
        return max_signals['time'].strftime('%H:%M:%S')

# --- ENTRY POINT ---
if __name__ == "__main__":
    print(f"\n{Colors.CYAN}{Colors.BOLD}🏛️ HISTORICAL SIMULATION SCANNER{Colors.RESET}")
    print(f"{Colors.CYAN}Simulating real-time scanning using September 12, 2025 data{Colors.RESET}")
    print(f"{Colors.CYAN}This will show you top 10 bullish/bearish signals every 5 minutes{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*70}{Colors.RESET}")
    
    # Initialize and run the historical simulation
    historical_scanner = HistoricalSimulationScanner(target_date="2025-09-12")
    
    try:
        historical_scanner.run_historical_simulation()
    except KeyboardInterrupt:
        historical_scanner.is_running = False
        print(f"\n{Colors.YELLOW}🛑 Simulation stopped by user{Colors.RESET}")
