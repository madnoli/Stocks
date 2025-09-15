# Enhanced Multi-Indicator Scanner with Backtesting & 3 Best/Worst Sectors & Gap-Down Filter
# Features: Live trading, Historical backtesting, Real-time simulation, Performance tracking
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
import csv
from collections import defaultdict
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

# --- BACKTESTING CONFIG ---
class BacktestConfig:
    def __init__(self):
        self.mode = "LIVE"  # LIVE, BACKTEST, REPLAY
        self.backtest_date = None
        self.replay_speed = 1.0  # 1.0 = normal speed, 2.0 = 2x speed
        self.start_time = time(9, 15)
        self.end_time = time(15, 30)
        self.interval_minutes = 5
        self.save_results = True
        self.results_file = None

    def set_backtest_mode(self, date_str, replay_speed=1.0):
        """Set backtesting mode with specific date"""
        self.mode = "BACKTEST"
        self.backtest_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        self.replay_speed = replay_speed
        self.results_file = f"backtest_results_{date_str}_{datetime.now().strftime('%H%M%S')}.csv"
        print(f"🎬 Backtest mode set for {date_str} at {replay_speed}x speed")

    def set_replay_mode(self, date_str, replay_speed=2.0):
        """Set replay mode for faster simulation"""
        self.mode = "REPLAY"
        self.backtest_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        self.replay_speed = replay_speed
        self.results_file = f"replay_results_{date_str}_{datetime.now().strftime('%H%M%S')}.csv"
        print(f"⚡ Replay mode set for {date_str} at {replay_speed}x speed")

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
    "Finance": ["BAJFINANCE", "SHRIRAMFIN", "CHOLAFIN", "HDFCLIFE", "ICICIPRULI","ETERNAL"],
    "Realty": ["DLF","LODHA","PRESTIGE","GODREJPROP","OBEROIRLTY","PHOENIXLTD","NCC","NBCC"],
    "PSE": ["BEL","BHEL","NHPC","GAIL","IOC","NTPC","POWERGRID","HINDPETRO","OIL","RECLTD","ONGC","NMDC","BPCL","HAL","RVNL","PFC","COALINDIA","IRCTC","IRFC"],
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
TIMEFRAME_WEIGHTS = {5: 1.0, 15: 1.5, 30: 2.0, 60: 2.5, 'daily': 3.0}

INDICATOR_WEIGHTS = {
    'RSI': 1.2, 'MACD': 1.4, 'Stochastic': 0.9, 'MA': 1.6,
    'ADX': 1.3, 'Bollinger': 1.1, 'ROC': 0.8, 'OBV': 1.4, 
    'CCI': 1.0, 'WWL': 0.9, 'EMA': 1.5, 'VWAP': 1.3
}

# --- BACKTESTING PERFORMANCE TRACKER ---
class BacktestTracker:
    def __init__(self):
        self.trades = []
        self.signals_history = []
        self.sector_performance = []
        self.hourly_stats = defaultdict(list)
        self.total_signals = 0
        self.profitable_signals = 0
        self.start_time = None
        self.end_time = None
        self.cycle_count = 0

    def add_signal(self, signal_data, current_time):
        """Add signal to tracking"""
        self.signals_history.append({
            'time': current_time,
            'symbol': signal_data['symbol'],
            'signal': signal_data['signal'],
            'score': signal_data['score'],
            'sector': signal_data['sector'],
            'timeframes': signal_data['timeframes']
        })

        hour = current_time.hour
        self.hourly_stats[hour].append(signal_data)
        self.total_signals += 1

    def add_sector_update(self, best_sectors, worst_sectors, current_time):
        """Track sector performance changes"""
        self.sector_performance.append({
            'time': current_time,
            'best': best_sectors[:],
            'worst': worst_sectors[:]
        })

    def save_to_csv(self, filename):
        """Save backtest results to CSV"""
        if not self.signals_history:
            print("No signals to save")
            return

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['time', 'symbol', 'signal', 'score', 'sector', 'timeframes']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for signal in self.signals_history:
                writer.writerow(signal)
        print(f"✅ Backtest results saved to {filename}")

    def generate_report(self):
        """Generate comprehensive backtest report"""
        if not self.signals_history:
            return "No signals generated during backtest"

        report = []
        report.append("="*80)
        report.append(f"BACKTEST PERFORMANCE REPORT")
        report.append("="*80)

        # Time range
        if self.signals_history:
            start = min(s['time'] for s in self.signals_history)
            end = max(s['time'] for s in self.signals_history)
            report.append(f"Time Range: {start.strftime('%H:%M:%S')} - {end.strftime('%H:%M:%S')}")
            report.append(f"Total Cycles: {self.cycle_count}")

        # Signal statistics
        report.append(f"Total Signals: {self.total_signals}")

        # Signal breakdown by type
        signal_counts = defaultdict(int)
        for signal in self.signals_history:
            signal_counts[signal['signal']] += 1

        report.append("\nSignal Distribution:")
        for signal_type, count in sorted(signal_counts.items()):
            percentage = (count / self.total_signals) * 100 if self.total_signals > 0 else 0
            report.append(f"  {signal_type}: {count} ({percentage:.1f}%)")

        # Sector distribution
        sector_counts = defaultdict(int)
        for signal in self.signals_history:
            sector_counts[signal['sector']] += 1

        report.append("\nTop Sectors by Signal Count:")
        for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / self.total_signals) * 100 if self.total_signals > 0 else 0
            report.append(f"  {sector}: {count} ({percentage:.1f}%)")

        # Hourly distribution
        report.append("\nHourly Signal Distribution:")
        for hour in sorted(self.hourly_stats.keys()):
            count = len(self.hourly_stats[hour])
            report.append(f"  {hour:02d}:00-{hour:02d}:59: {count} signals")

        # Sector rotation analysis
        if self.sector_performance:
            report.append("\nSector Rotation Summary:")
            report.append(f"Total sector updates: {len(self.sector_performance)}")

            # Most frequent best sectors
            best_sector_counts = defaultdict(int)
            for update in self.sector_performance:
                for sector in update['best']:
                    best_sector_counts[sector] += 1

            report.append("\nMost Frequently Best Sectors:")
            for sector, count in sorted(best_sector_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                report.append(f"  {sector}: {count} times")

        return "\n".join(report)

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

            # 5. Exponential Moving Average (21-period)
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

# --- ENHANCED 3-SECTOR SCANNER WITH BACKTESTING ---
class Enhanced3SectorBacktestScanner:
    def __init__(self, config=None):
        self.config = config or BacktestConfig()
        self.is_running = False
        self.current_signals = {}

        # CHANGE: Store top 3 best and worst sectors
        self.best_sectors = ["Technology", "Pharma", "Banking"]
        self.worst_sectors = ["Auto", "Metal", "Energy"]
        self.sectoral_history = []
        self.last_sectoral_update = None
        self.api_errors = []
        self.sector_update_attempts = 0
        self.successful_updates = 0
        self.gap_down_filtered_count = 0

        # Backtesting specific
        self.backtest_tracker = BacktestTracker()
        self.historical_data_cache = {}
        self.current_simulation_time = None
        self.simulation_data = {}
        self.trading_time_slots = []
        self.sector_simulation = {}

        # Market hours
        self.market_start = time(9, 15)
        self.market_end = time(15, 30)

        # Intervals based on mode
        if self.config.mode in ["BACKTEST", "REPLAY"]:
            base_interval = 60  # 1 minute base for simulation timing
            self.scan_interval = base_interval / self.config.replay_speed
            self.sectoral_update_interval = base_interval / self.config.replay_speed
        else:
            # 5-minute intervals for live mode
            self.scan_interval = 300  # 5 minutes
            self.sectoral_update_interval = 300  # 5 minutes

        logger.info(f"🚀 Enhanced 3-Sector Scanner initialized in {self.config.mode} mode")

        # Initialize with debug checks
        self.show_initialization_status()

    def show_initialization_status(self):
        """Show initialization status and run initial checks"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}🚀 ENHANCED 3-SECTOR SCANNER WITH BACKTESTING{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*75}{Colors.RESET}")

        if self.config.mode == "LIVE":
            print(f"🔴 Mode: {Colors.GREEN}LIVE TRADING{Colors.RESET}")
        elif self.config.mode == "BACKTEST":
            print(f"📊 Mode: {Colors.BLUE}BACKTESTING{Colors.RESET}")
            print(f"📅 Date: {Colors.YELLOW}{self.config.backtest_date}{Colors.RESET}")
            print(f"⚡ Speed: {Colors.MAGENTA}{self.config.replay_speed}x{Colors.RESET}")
        elif self.config.mode == "REPLAY":
            print(f"🎬 Mode: {Colors.MAGENTA}REPLAY SIMULATION{Colors.RESET}")
            print(f"📅 Date: {Colors.YELLOW}{self.config.backtest_date}{Colors.RESET}")
            print(f"⚡ Speed: {Colors.MAGENTA}{self.config.replay_speed}x{Colors.RESET}")

        print(f"⏰ Timeframes: {Colors.YELLOW}5min, 15min, 30min, 60min, Daily{Colors.RESET}")
        print(f"🎯 Strategy: {Colors.GREEN}Top 3 Best{Colors.RESET} + {Colors.RED}Top 3 Worst{Colors.RESET} sectors")
        print(f"🚫 Filter: {Colors.MAGENTA}Gap-down exclusion{Colors.RESET}")

        # Show current sector status
        self.show_sector_status()

        if self.config.mode == "LIVE":
            # Test API connection for live mode
            self.test_api_connection()
            # Force initial sector update
            print(f"\n{Colors.YELLOW}🔄 Running initial sector update...{Colors.RESET}")
            self.force_sector_update()
        else:
            print(f"\n{Colors.BLUE}📊 Preparing historical data for backtesting...{Colors.RESET}")
            self.prepare_backtest_data()

        print(f"{Colors.CYAN}{'='*75}{Colors.RESET}")

    def prepare_backtest_data(self):
        """Prepare historical data for backtesting"""
        if self.config.mode == "LIVE":
            return

        print(f"\n{Colors.BLUE}🔍 PREPARING BACKTEST DATA FOR {self.config.backtest_date}{Colors.RESET}")

        # Generate time slots for the trading day
        self.generate_trading_time_slots()

        # Pre-load sector data for simulation
        self.simulate_sector_performance()

        print(f"✅ Backtest data prepared for {len(self.trading_time_slots)} time slots")

    def generate_trading_time_slots(self):
        """Generate 5-minute time slots from 9:15 AM to 3:30 PM"""
        self.trading_time_slots = []

        current_time = datetime.combine(self.config.backtest_date, self.market_start)
        end_time = datetime.combine(self.config.backtest_date, self.market_end)

        while current_time <= end_time:
            self.trading_time_slots.append(current_time)
            current_time += timedelta(minutes=self.config.interval_minutes)

        print(f"📅 Generated {len(self.trading_time_slots)} time slots from {self.market_start} to {self.market_end}")

    def simulate_sector_performance(self):
        """Simulate realistic sector performance throughout the day"""
        print("🔄 Simulating sector performance changes...")

        # Create realistic sector performance simulation
        base_sectors = list(NSE_INDEX_TO_SECTOR.values())
        unique_sectors = list(set(base_sectors))

        self.sector_simulation = {}

        for i, time_slot in enumerate(self.trading_time_slots):
            # Simulate random but realistic sector movements
            sector_changes = {}
            for sector in unique_sectors:
                # Create some trend with random variation
                base_change = np.random.normal(0, 1.5)  # Random normal distribution
                trend_factor = np.sin(i * 0.1) * 0.5  # Gradual trend
                sector_changes[sector] = base_change + trend_factor

            # Sort sectors by performance for this time slot
            sorted_sectors = sorted(sector_changes.items(), key=lambda x: x[1], reverse=True)

            best_3 = [sector for sector, _ in sorted_sectors[:3]]
            worst_3 = [sector for sector, _ in sorted_sectors[-3:]]

            self.sector_simulation[time_slot] = {
                'best': best_3,
                'worst': worst_3,
                'all_changes': sector_changes
            }

    def get_simulated_sectors_for_time(self, simulation_time):
        """Get sector data for specific simulation time"""
        if simulation_time in self.sector_simulation:
            return self.sector_simulation[simulation_time]

        # Find closest time if exact match not found
        closest_time = min(self.sector_simulation.keys(), 
                         key=lambda x: abs((x - simulation_time).total_seconds()))
        return self.sector_simulation[closest_time]

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

                elif isinstance(data, dict):
                    print(f"🗂️ Dict Keys: {list(data.keys())}")

            else:
                print(f"❌ API Connection: {Colors.RED}FAILED{Colors.RESET} (Status: {response.status_code})")

        except Exception as e:
            print(f"❌ API Connection: {Colors.RED}ERROR{Colors.RESET} - {str(e)}")

    def show_sector_status(self):
        """Show current sector selection status for 3 best/worst"""
        print(f"\n{Colors.MAGENTA}📊 CURRENT 3-SECTOR STATUS:{Colors.RESET}")
        print(f"🏆 Top 3 Best Sectors: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"📉 Top 3 Worst Sectors: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")

        if self.config.mode == "LIVE":
            print(f"🕐 Last Update: {self.last_sectoral_update or 'Never'}")
            print(f"📈 Total Updates: {len(self.sectoral_history)}")
            print(f"🔄 Update Attempts: {self.sector_update_attempts}")
            print(f"✅ Successful Updates: {self.successful_updates}")
        else:
            print(f"🎬 Simulation Mode: Sectors will change throughout the day")

        print(f"🚫 Gap-down Filtered: {self.gap_down_filtered_count}")

        if self.sectoral_history:
            print(f"\n📋 Recent Sector Changes (Last 3):")
            for i, entry in enumerate(self.sectoral_history[-3:]):
                time_str = entry['timestamp'].strftime('%H:%M:%S')
                best_display = ', '.join(entry['best'][:3])  # Show top 3
                worst_display = ', '.join(entry['worst'][:3])  # Show top 3
                print(f"  {i+1}. {time_str}: Best: {Colors.GREEN}{best_display}{Colors.RESET} | Worst: {Colors.RED}{worst_display}{Colors.RESET}")

    def force_sector_update(self):
        """Force a sector update (live mode) or simulate (backtest mode)"""
        if self.config.mode == "LIVE":
            print(f"\n{Colors.YELLOW}🔄 FORCING 3-SECTOR UPDATE...{Colors.RESET}")
            self.sector_update_attempts += 1

            success = self.fetch_live_sectoral_performance_3sector_debug()

            if success:
                self.successful_updates += 1
                print(f"✅ Update successful!")
                print(f"🏆 Top 3 Best: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
                print(f"📉 Top 3 Worst: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
            else:
                print(f"❌ Update failed - using defaults")
                print(f"🏆 Default Best: {Colors.YELLOW}{', '.join(self.best_sectors)}{Colors.RESET}")
                print(f"📉 Default Worst: {Colors.YELLOW}{', '.join(self.worst_sectors)}{Colors.RESET}")
        else:
            # For backtesting, update based on current simulation time
            if self.current_simulation_time:
                sector_data = self.get_simulated_sectors_for_time(self.current_simulation_time)
                self.best_sectors = sector_data['best']
                self.worst_sectors = sector_data['worst']
                self.last_sectoral_update = self.current_simulation_time

                # Track sector updates
                self.backtest_tracker.add_sector_update(self.best_sectors, self.worst_sectors, self.current_simulation_time)

                print(f"✅ Simulated sector update for {self.current_simulation_time.strftime('%H:%M:%S')}")

        print(f"🏆 Top 3 Best: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"📉 Top 3 Worst: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
        return True

    def is_market_open(self, check_time=None):
        """Check if market is open for given time or current time"""
        if self.config.mode in ["BACKTEST", "REPLAY"]:
            if check_time:
                time_only = check_time.time()
                return self.market_start <= time_only <= self.market_end
            return True  # Always "open" for backtesting

        # Live mode logic
        now = datetime.now()
        current_time = now.time()

        if now.weekday() > 4:  # Weekend
            return False

        return self.market_start <= current_time <= self.market_end

    def fetch_live_sectoral_performance_3sector_debug(self):
        """Enhanced sectoral performance fetching for 3 best/worst sectors"""
        try:
            logger.info("🔍 Fetching live 3-sector performance...")

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

                if sectoral_performance:
                    sectoral_performance.sort(key=lambda x: x['change_percent'], reverse=True)

                    old_best = self.best_sectors[:]
                    old_worst = self.worst_sectors[:]

                    # CHANGE: Update top 3 best and worst sectors
                    if len(sectoral_performance) >= 6:
                        self.best_sectors = [sectoral_performance[0]['sector'], 
                                           sectoral_performance[1]['sector'], 
                                           sectoral_performance[2]['sector']]
                        self.worst_sectors = [sectoral_performance[-1]['sector'], 
                                            sectoral_performance[-2]['sector'], 
                                            sectoral_performance[-3]['sector']]
                    elif len(sectoral_performance) >= 3:
                        self.best_sectors = [sectoral_performance[0]['sector'], 
                                           sectoral_performance[1]['sector']]
                        self.worst_sectors = [sectoral_performance[-1]['sector'], 
                                            sectoral_performance[-2]['sector']]

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
                    print(f"❌ No sectoral data matched.")
                    return False

        except Exception as e:
            logger.error(f"❌ Error fetching sectoral data: {e}")
            self.api_errors.append(f"{datetime.now()}: {e}")
            return False

    def display_3sector_update(self, sectoral_performance, old_best, old_worst):
        """Display enhanced 3-sector performance update"""
        if self.config.mode in ["BACKTEST", "REPLAY"]:
            current_time = self.current_simulation_time or datetime.now()
        else:
            current_time = datetime.now()

        print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'='*100}")
        print(f"📊 3-SECTOR PERFORMANCE UPDATE - {current_time.strftime('%H:%M:%S')} IST")
        print(f"{'='*100}{Colors.RESET}")

        print(f"🏆 Top 3 Best: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"📉 Top 3 Worst: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")

        if sectoral_performance:
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

    def normalize_live_data(self, df, symbol):
        """Fast data normalization for all timeframes"""
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
        """Check if stock opened with gap down (Open < Previous Close by 1%+)"""
        try:
            if df is None or len(df) < 2:
                return False

            current_open = df['Open'].iloc[-1]
            previous_close = df['Close'].iloc[-2]

            if pd.isna(current_open) or pd.isna(previous_close) or previous_close <= 0:
                return False

            gap_percentage = ((current_open - previous_close) / previous_close) * 100

            # Consider gap down if opening is 1% or more below previous close
            return gap_percentage <= -1.0

        except Exception as e:
            logger.error(f"Error checking gap down: {e}")
            return False

    def fetch_live_data_for_backtest(self, symbol, timeframe, target_time):
        """Fetch historical data up to target_time for backtesting"""
        try:
            tf_map = {
                5: '5 min', 
                15: '15 min', 
                30: '30 min',
                60: '60 mins',
                'daily': 'EOD'
            }
            bar_size = tf_map.get(timeframe)
            if not bar_size: 
                return None, False

            # Adjust duration based on timeframe
            if timeframe == 5 or timeframe == 15:
                duration = '10 D'
            elif timeframe == 30:
                duration = '20 D' 
            elif timeframe == 60:
                duration = '60 D'
            elif timeframe == 'daily':
                duration = '365 D'
            else:
                duration = '10 D'

            # Use cache key for this symbol+timeframe combination
            cache_key = f"{symbol}_{timeframe}"

            if cache_key not in self.historical_data_cache:
                raw_df = td_hist.get_historic_data(symbol, duration=duration, bar_size=bar_size)
                if raw_df is not None and len(raw_df) > 0:
                    normalized_df = self.normalize_live_data(raw_df, symbol)
                    if normalized_df is not None:
                        self.historical_data_cache[cache_key] = normalized_df

            if cache_key in self.historical_data_cache:
                full_df = self.historical_data_cache[cache_key]

                # Filter data up to target_time for realistic backtesting
                filtered_df = full_df[full_df.index <= pd.Timestamp(target_time)]

                if len(filtered_df) >= 20:
                    # Check for gap down only on intraday timeframes
                    is_gap_down = False
                    if timeframe in [5, 15, 30]:
                        is_gap_down = self.check_gap_down(filtered_df)

                    if timeframe == 'daily':
                        return filtered_df.tail(250), is_gap_down
                    elif timeframe == 60:
                        return filtered_df.tail(200), is_gap_down
                    else:
                        return filtered_df.tail(100), is_gap_down

            return None, False

        except Exception as e:
            logger.error(f"Backtest data fetch error {symbol}_{timeframe}: {e}")
            return None, False

    def fetch_live_data(self, symbol, timeframe):
        """Enhanced fetch live data with gap-down detection"""
        if self.config.mode in ["BACKTEST", "REPLAY"]:
            return self.fetch_live_data_for_backtest(symbol, timeframe, self.current_simulation_time)

        # Original live data fetching logic
        try:
            tf_map = {
                5: '5 min', 
                15: '15 min', 
                30: '30 min',
                60: '60 mins',
                'daily': 'EOD'
            }
            bar_size = tf_map.get(timeframe)
            if not bar_size: return None, False

            if timeframe == 5 or timeframe == 15:
                duration = '10 D'
            elif timeframe == 30:
                duration = '20 D' 
            elif timeframe == 60:
                duration = '60 D'
            elif timeframe == 'daily':
                duration = '365 D'
            else:
                duration = '10 D'

            raw_df = td_hist.get_historic_data(symbol, duration=duration, bar_size=bar_size)

            if raw_df is not None and len(raw_df) > 0:
                normalized_df = self.normalize_live_data(raw_df, symbol)
                if normalized_df is not None and len(normalized_df) >= 20:
                    # Check for gap down only on intraday timeframes
                    is_gap_down = False
                    if timeframe in [5, 15, 30]:
                        is_gap_down = self.check_gap_down(normalized_df)

                    if timeframe == 'daily':
                        return normalized_df.tail(250), is_gap_down
                    elif timeframe == 60:
                        return normalized_df.tail(200), is_gap_down
                    else:
                        return normalized_df.tail(100), is_gap_down
            return None, False
        except Exception as e:
            logger.error(f"Live data fetch error {symbol}_{timeframe}: {e}")
            return None, False

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
            elif indicator_name == 'VWAP': 
                return 50
            else:
                return 50
        except: 
            return 50

    def calculate_multi_indicator_signals(self, symbol, timeframes_data):
        """Enhanced calculate signals for 3-sector strategy"""
        try:
            if not timeframes_data: return 'Neutral', 0

            sector = next((s for s, stocks in SECTOR_STOCKS.items() if symbol in stocks), None)
            if not sector: return 'Neutral', 0

            total_weighted_score, total_weight = 0, 0
            timeframe_scores = {}
            timeframe_details = []

            for tf, df in timeframes_data.items():
                if df is None or len(df) < 20: 
                    continue

                indicators = TechnicalIndicators.calculate_all_indicators(df)
                if not indicators: 
                    continue

                tf_score, tf_weight = 0, 0
                current_price = df['Close'].iloc[-1]

                for name, weight in INDICATOR_WEIGHTS.items():
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
                    timeframe_details.append(tf)

                    tf_multiplier = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                    total_weighted_score += tf_final_score * tf_multiplier
                    total_weight += tf_multiplier

            if total_weight == 0: 
                return 'Neutral', 0
            base_score = total_weighted_score / total_weight

            # CHANGE: Enhanced 3-sector boost system
            sector_boost = 0
            has_longer_tf = 'daily' in timeframes_data or 60 in timeframes_data

            if sector in self.best_sectors:
                rank = self.best_sectors.index(sector) + 1
                if has_longer_tf:
                    sector_boost = [25, 20, 15][rank-1] if rank <= 3 else 10  # 25, 20, 15 for ranks 1, 2, 3
                else:
                    sector_boost = [20, 15, 10][rank-1] if rank <= 3 else 5   # 20, 15, 10 for ranks 1, 2, 3

            elif sector in self.worst_sectors:
                rank = self.worst_sectors.index(sector) + 1
                if has_longer_tf:
                    sector_boost = [-25, -20, -15][rank-1] if rank <= 3 else -10
                else:
                    sector_boost = [-20, -15, -10][rank-1] if rank <= 3 else -5

            base_score += sector_boost

            # Multi-timeframe confirmation bonus
            num_timeframes = len(timeframes_data)
            if num_timeframes >= 4:
                bullish_count = sum(1 for score in timeframe_scores.values() if score > 55)
                bearish_count = sum(1 for score in timeframe_scores.values() if score < 45)

                if bullish_count >= 3:
                    base_score += 7  # Enhanced bonus for 3-sector strategy
                elif bearish_count >= 3:
                    base_score -= 7

            # Enhanced signal classification
            if base_score >= 80: return 'Very Strong Buy', base_score
            elif base_score >= 70: return 'Strong Buy', base_score
            elif base_score >= 58: return 'Buy', base_score
            elif base_score <= 20: return 'Very Strong Sell', base_score
            elif base_score <= 30: return 'Strong Sell', base_score
            elif base_score <= 42: return 'Sell', base_score
            else: return 'Neutral', base_score

        except Exception as e:
            logger.error(f"Signal calculation error for {symbol}: {e}")
            return 'Neutral', 0

    def enhanced_3sector_scan_cycle(self):
        """Enhanced 3-sector scan with gap-down filtering and backtesting support"""
        # Market check for live mode
        if self.config.mode == "LIVE" and not self.is_market_open():
            logger.info("🕐 Market closed. Next scan in 5 minutes...")
            return

        start_time = time_module.time()

        if self.config.mode in ["BACKTEST", "REPLAY"]:
            current_time = self.current_simulation_time
        else:
            current_time = datetime.now()

        print(f"\n{Colors.CYAN}🔄 Starting 3-sector enhanced scan at {current_time.strftime('%H:%M:%S')}{Colors.RESET}")
        print(f"⏰ Analyzing: {Colors.YELLOW}5min → 15min → 30min → 60min → Daily{Colors.RESET}")
        print(f"🎯 Strategy: {Colors.GREEN}3 Best{Colors.RESET} + {Colors.RED}3 Worst{Colors.RESET} sectors")

        # Update sectors
        if self.config.mode == "LIVE":
            if not self.fetch_live_sectoral_performance_3sector_debug():
                print(f"⚠️ Sectoral update failed, continuing with previous sectors")
        else:
            # Update sectors for backtesting
            self.force_sector_update()

        try:
            # CHANGE: Get target stocks from top 3 best and top 3 worst sectors
            target_stocks_set = set()

            # From 3 best sectors - more stocks from better performing sectors
            for i, sector in enumerate(self.best_sectors):
                if sector in SECTOR_STOCKS:
                    if i == 0:  # Top sector gets 12 stocks
                        target_stocks_set.update(SECTOR_STOCKS[sector][:12])
                    elif i == 1:  # 2nd best gets 10 stocks
                        target_stocks_set.update(SECTOR_STOCKS[sector][:10])
                    else:  # 3rd best gets 8 stocks
                        target_stocks_set.update(SECTOR_STOCKS[sector][:8])

            # From 3 worst sectors - similar distribution for short opportunities
            for i, sector in enumerate(self.worst_sectors):
                if sector in SECTOR_STOCKS:
                    if i == 0:  # Worst sector gets 12 stocks
                        target_stocks_set.update(SECTOR_STOCKS[sector][:12])
                    elif i == 1:  # 2nd worst gets 10 stocks
                        target_stocks_set.update(SECTOR_STOCKS[sector][:10])
                    else:  # 3rd worst gets 8 stocks
                        target_stocks_set.update(SECTOR_STOCKS[sector][:8])

            target_stocks = list(target_stocks_set)

            if not target_stocks:
                print(f"⚠️ No target stocks found.")
                return

            print(f"🎯 3-Sector scanning {len(target_stocks)} stocks from 6 sectors")

            live_signals = []
            gap_down_filtered = 0

            with ThreadPoolExecutor(max_workers=3) as executor:
                def process_stock(symbol):
                    try:
                        timeframes_data = {}
                        has_gap_down = False

                        timeframes_to_fetch = [5, 15, 30, 60, 'daily']

                        for tf in timeframes_to_fetch:
                            df, is_gap_down = self.fetch_live_data(symbol, tf)
                            if df is not None: 
                                timeframes_data[tf] = df
                                # Flag gap down if detected on any intraday timeframe
                                if is_gap_down and tf in [5, 15, 30]:
                                    has_gap_down = True

                            # Sleep only in live mode to avoid overwhelming API
                            if self.config.mode == "LIVE":
                                time_module.sleep(1.0)

                        # CHANGE: Exclude gap-down stocks
                        if has_gap_down:
                            logger.info(f"🚫 {symbol} filtered out due to gap-down")
                            return None, True  # Return gap-down flag

                        if len(timeframes_data) >= 3:
                            signal, score = self.calculate_multi_indicator_signals(symbol, timeframes_data)
                            if abs(score - 50) > 15:
                                sector = next((s for s, st in SECTOR_STOCKS.items() if symbol in st), 'N/A')
                                signal_data = {
                                    'symbol': symbol, 'signal': signal, 'score': score, 'sector': sector,
                                    'timeframes': len(timeframes_data), 'timestamp': current_time,
                                    'tf_details': list(timeframes_data.keys())
                                }

                                # Track signal in backtesting mode
                                if self.config.mode in ["BACKTEST", "REPLAY"]:
                                    self.backtest_tracker.add_signal(signal_data, current_time)

                                return signal_data, False
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
            logger.info(f"⚡ Enhanced 3-sector scan completed in {scan_time:.2f}s - {len(live_signals)} signals, {gap_down_filtered} gap-down filtered")

            # Increment cycle count for backtesting
            if self.config.mode in ["BACKTEST", "REPLAY"]:
                self.backtest_tracker.cycle_count += 1

            self.display_enhanced_3sector_signals(live_signals, scan_time, gap_down_filtered)

        except Exception as e:
            logger.error(f"Error in 3-sector enhanced scan: {e}")

    def display_enhanced_3sector_signals(self, signals, scan_time, gap_down_filtered):
        """Display enhanced signals with 3-sector context and gap-down info"""
        # Clear screen only in live mode
        if self.config.mode == "LIVE":
            os.system('clear' if os.name == 'posix' else 'cls')

        if self.config.mode in ["BACKTEST", "REPLAY"]:
            current_time = self.current_simulation_time
        else:
            current_time = datetime.now()

        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*140}")
        if self.config.mode in ["BACKTEST", "REPLAY"]:
            print(f"🎬 ENHANCED 3-SECTOR BACKTEST SCANNER - {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
            print(f"📊 Mode: {self.config.mode} | Speed: {self.config.replay_speed}x | Cycle: {self.backtest_tracker.cycle_count}")
        else:
            print(f"🎯 ENHANCED 3-SECTOR SCANNER WITH GAP-DOWN FILTER - {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
        print(f"{'='*140}{Colors.RESET}")

        print(f"{Colors.BLUE}📊 Analysis: {Colors.YELLOW}5m{Colors.RESET} + {Colors.YELLOW}15m{Colors.RESET} + "
              f"{Colors.YELLOW}30m{Colors.RESET} + {Colors.CYAN}60m{Colors.RESET} + {Colors.MAGENTA}Daily{Colors.RESET}")
        print(f"🎯 Strategy: {Colors.GREEN}Top 3 Best{Colors.RESET} + {Colors.RED}Top 3 Worst{Colors.RESET} sectors")

        if self.last_sectoral_update:
            best_str = ', '.join(self.best_sectors)
            worst_str = ', '.join(self.worst_sectors)
            print(f"{Colors.MAGENTA}📈 Sectoral Update:{Colors.RESET} {Colors.YELLOW}{self.last_sectoral_update.strftime('%H:%M:%S')}{Colors.RESET}")
            print(f"🏆 Top 3 Best: {Colors.GREEN}{Colors.BOLD}{best_str}{Colors.RESET}")
            print(f"📉 Top 3 Worst: {Colors.RED}{Colors.BOLD}{worst_str}{Colors.RESET}")

        # Show different stats for backtest vs live
        if self.config.mode in ["BACKTEST", "REPLAY"]:
            print(f"{Colors.BLUE}📊 Backtest Stats:{Colors.RESET} Cycle: {self.backtest_tracker.cycle_count} | "
                  f"Total Signals: {self.backtest_tracker.total_signals} | "
                  f"⚡ Scan Time: {scan_time:.2f}s | 🚫 Gap-down: {Colors.MAGENTA}{gap_down_filtered}{Colors.RESET}")
        else:
            print(f"{Colors.BLUE}📈 Updates:{Colors.RESET} {self.successful_updates}/{self.sector_update_attempts} | "
                  f"⚡ Scan Time: {scan_time:.2f}s | 🚫 Gap-down Filtered: {Colors.MAGENTA}{gap_down_filtered}{Colors.RESET}")

        if not signals:
            print(f"\n{Colors.YELLOW}📭 No significant 3-sector signals found in this cycle.{Colors.RESET}")
            print(f"{Colors.CYAN}💡 {gap_down_filtered} stocks filtered due to gap-down opening{Colors.RESET}")
        else:
            # CHANGE: Separate bullish and bearish signals for top 10 each
            bullish_signals = [s for s in signals if 'Buy' in s['signal']]
            bearish_signals = [s for s in signals if 'Sell' in s['signal']]

            bullish_signals.sort(key=lambda x: x['score'], reverse=True)
            bearish_signals.sort(key=lambda x: x['score'])

            print(f"\n{Colors.GREEN}{Colors.BOLD}🚀 TOP 10 BULLISH SIGNALS (3-SECTOR STRATEGY):{Colors.RESET}")
            print(f"{'Stock':<10} {'Sector':<18} {'Signal':<18} {'Score':>8} {'TFs':>4} {'TF Coverage':<20} {'Strength':<12}")
            print(f"{Colors.GREEN}{'-' * 140}{Colors.RESET}")

            for s in bullish_signals[:10]:
                sector_color, sector_name = Colors.YELLOW, s['sector']
                if s['sector'] in self.best_sectors:
                    rank = self.best_sectors.index(s['sector']) + 1
                    stars = "★" * rank
                    sector_color, sector_name = Colors.GREEN, f"{stars}{s['sector']}"

                signal_color = Colors.GREEN + (Colors.BOLD if 'Very' in s['signal'] else "")

                deviation = abs(s['score'] - 50)
                if deviation >= 35:
                    strength = f"{Colors.GREEN}{Colors.BOLD}Exceptional{Colors.RESET}"
                elif deviation >= 25:
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
                      f"{signal_color}{s['signal']:<18}{Colors.RESET} "
                      f"{Colors.WHITE}{s['score']:>8.1f}{Colors.RESET} "
                      f"{Colors.CYAN}{s['timeframes']:>4}{Colors.RESET} "
                      f"{Colors.MAGENTA}{tf_coverage:<20}{Colors.RESET} "
                      f"{strength}")

            if bearish_signals:
                print(f"\n{Colors.RED}{Colors.BOLD}📉 TOP 10 BEARISH SIGNALS (3-SECTOR STRATEGY):{Colors.RESET}")
                print(f"{'Stock':<10} {'Sector':<18} {'Signal':<18} {'Score':>8} {'TFs':>4} {'TF Coverage':<20} {'Strength':<12}")
                print(f"{Colors.RED}{'-' * 140}{Colors.RESET}")

                for s in bearish_signals[:10]:
                    sector_color, sector_name = Colors.YELLOW, s['sector']
                    if s['sector'] in self.worst_sectors:
                        rank = self.worst_sectors.index(s['sector']) + 1
                        stars = "★" * rank
                        sector_color, sector_name = Colors.RED, f"{stars}{s['sector']}"

                    signal_color = Colors.RED + (Colors.BOLD if 'Very' in s['signal'] else "")

                    deviation = abs(s['score'] - 50)
                    if deviation >= 35:
                        strength = f"{Colors.RED}{Colors.BOLD}Exceptional{Colors.RESET}"
                    elif deviation >= 25:
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
                          f"{signal_color}{s['signal']:<18}{Colors.RESET} "
                          f"{Colors.WHITE}{s['score']:>8.1f}{Colors.RESET} "
                          f"{Colors.CYAN}{s['timeframes']:>4}{Colors.RESET} "
                          f"{Colors.MAGENTA}{tf_coverage:<20}{Colors.RESET} "
                          f"{strength}")

        # Show next scan info based on mode
        if self.config.mode == "LIVE":
            next_scan_time = (current_time + timedelta(minutes=5)).strftime('%H:%M:%S')
            print(f"\n{Colors.CYAN}{Colors.BOLD}⏰ Next 3-sector scan at {next_scan_time}{Colors.RESET}")
        elif self.config.mode in ["BACKTEST", "REPLAY"]:
            print(f"\n{Colors.CYAN}{Colors.BOLD}🎬 Backtest Progress: {self.backtest_tracker.cycle_count} cycles completed{Colors.RESET}")

        print(f"{Colors.BLUE}💡 3-sector strategy with gap-down filter for higher probability trades{Colors.RESET}")
        if gap_down_filtered > 0:
            print(f"{Colors.MAGENTA}🚫 Gap-down filter excluded {gap_down_filtered} stocks for risk management{Colors.RESET}")

    def run_backtest_simulation(self):
        """Run the complete backtesting simulation from 9:15 AM to 3:30 PM"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}🎬 STARTING BACKTEST SIMULATION{Colors.RESET}")
        print(f"📅 Date: {self.config.backtest_date}")
        print(f"⚡ Speed: {self.config.replay_speed}x")
        print(f"⏰ Time slots: {len(self.trading_time_slots)}")
        print(f"{Colors.CYAN}{'='*75}{Colors.RESET}")

        self.is_running = True
        simulation_start = time_module.time()

        try:
            for i, time_slot in enumerate(self.trading_time_slots):
                if not self.is_running:
                    break

                self.current_simulation_time = time_slot

                # Show progress
                progress = (i + 1) / len(self.trading_time_slots) * 100
                print(f"\n{Colors.BLUE}📊 Progress: {progress:.1f}% ({i+1}/{len(self.trading_time_slots)}) - {time_slot.strftime('%H:%M:%S')}{Colors.RESET}")

                # Run scan cycle for this time slot
                self.enhanced_3sector_scan_cycle()

                # Wait according to replay speed (simulate real-time intervals)
                sleep_time = self.scan_interval
                if sleep_time > 0:
                    time_module.sleep(sleep_time)

            # Generate final report
            simulation_time = time_module.time() - simulation_start
            print(f"\n{Colors.GREEN}{Colors.BOLD}✅ BACKTEST SIMULATION COMPLETED{Colors.RESET}")
            print(f"⏱️ Total simulation time: {simulation_time:.2f} seconds")
            print(f"📊 Simulated {len(self.trading_time_slots)} time slots at {self.config.replay_speed}x speed")

            # Save results to CSV
            if self.config.save_results and self.config.results_file:
                self.backtest_tracker.save_to_csv(self.config.results_file)

            # Print comprehensive report
            report = self.backtest_tracker.generate_report()
            print(f"\n{Colors.YELLOW}{report}{Colors.RESET}")

        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}🛑 Backtest simulation stopped by user{Colors.RESET}")
        finally:
            self.stop()

    def run_enhanced_3sector_scanner(self):
        """Main enhanced 3-sector scanner with mode detection"""
        self.is_running = True

        if self.config.mode in ["BACKTEST", "REPLAY"]:
            logger.info(f"🎬 Starting {self.config.mode} simulation...")
            self.run_backtest_simulation()
        else:
            logger.info("🚀 Starting Enhanced 3-Sector Scanner with Gap-Down Filter...")
            self.force_sector_update()

            try:
                while self.is_running:
                    self.enhanced_3sector_scan_cycle()
                    if self.is_running:
                        logger.info(f"💤 Waiting 5 minutes for next 3-sector cycle...")
                        time_module.sleep(self.scan_interval)
            except KeyboardInterrupt:
                logger.info("🛑 Enhanced 3-sector scanner stopped by user")
            finally:
                self.stop()

    def stop(self):
        """Stop the scanner"""
        self.is_running = False
        print(f"{Colors.YELLOW}🛑 3-sector scanner stopped{Colors.RESET}")


# --- MAIN EXECUTION FUNCTIONS ---
def run_live_scanner():
    """Run the live scanner"""
    config = BacktestConfig()  # Default is LIVE mode
    scanner = Enhanced3SectorBacktestScanner(config)
    scanner.run_enhanced_3sector_scanner()

def run_backtest(date_str, replay_speed=1.0):
    """Run backtesting for specific date"""
    config = BacktestConfig()
    config.set_backtest_mode(date_str, replay_speed)
    scanner = Enhanced3SectorBacktestScanner(config)
    scanner.run_enhanced_3sector_scanner()

def run_replay(date_str, replay_speed=5.0):
    """Run fast replay simulation for specific date"""
    config = BacktestConfig()
    config.set_replay_mode(date_str, replay_speed)
    scanner = Enhanced3SectorBacktestScanner(config)
    scanner.run_enhanced_3sector_scanner()

def main():
    """Main function with mode selection"""
    print(f"{Colors.CYAN}{Colors.BOLD}🎯 ENHANCED 3-SECTOR SCANNER WITH BACKTESTING{Colors.RESET}")
    print(f"{Colors.YELLOW}📊 Timeframes: 5min, 15min, 30min, 60min, Daily (EOD){Colors.RESET}")
    print(f"{Colors.CYAN}🔧 Features: 3 Best + 3 Worst sectors, Top 10 Bullish/Bearish, Gap-down filter{Colors.RESET}")
    print(f"{Colors.MAGENTA}⚡ Modes: Live Trading, Backtesting, Fast Replay{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*75}{Colors.RESET}")

    print("\nSelect Mode:")
    print("1. Live Trading (Real-time)")
    print("2. Backtest (Historical date)")
    print("3. Fast Replay (Quick simulation)")
    print("4. Today's Backtest (Current date)")

    try:
        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            # Live mode
            run_live_scanner()

        elif choice == "2":
            # Backtest mode
            date_input = input("Enter date (YYYY-MM-DD) or press Enter for yesterday: ").strip()
            if not date_input:
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                date_input = yesterday

            speed_input = input("Enter replay speed (1.0=normal, 2.0=2x, etc.) or press Enter for 1.0: ").strip()
            speed = float(speed_input) if speed_input else 1.0

            print(f"\n🎬 Starting backtest for {date_input} at {speed}x speed...")
            run_backtest(date_input, speed)

        elif choice == "3":
            # Fast replay mode
            date_input = input("Enter date (YYYY-MM-DD) or press Enter for yesterday: ").strip()
            if not date_input:
                yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                date_input = yesterday

            speed_input = input("Enter replay speed (5.0=5x, 10.0=10x, etc.) or press Enter for 5.0: ").strip()
            speed = float(speed_input) if speed_input else 5.0

            print(f"\n⚡ Starting fast replay for {date_input} at {speed}x speed...")
            run_replay(date_input, speed)

        elif choice == "4":
            # Today's backtest
            today = datetime.now().strftime('%Y-%m-%d')
            speed_input = input("Enter replay speed (1.0=normal, 2.0=2x, etc.) or press Enter for 2.0: ").strip()
            speed = float(speed_input) if speed_input else 2.0

            print(f"\n📅 Starting today's backtest for {today} at {speed}x speed...")
            run_backtest(today, speed)

        else:
            print("Invalid choice. Defaulting to live mode...")
            run_live_scanner()

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Exiting Enhanced 3-Sector Scanner...{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.RESET}")
        print("Defaulting to live mode...")
        run_live_scanner()


if __name__ == "__main__":
    main()
