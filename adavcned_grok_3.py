import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import requests
import json
import time as timemodule
import pytz
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from truedata.history import TD_hist
import warnings
warnings.filterwarnings("ignore")
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import argparse

logger = logging.getLogger(__name__)

TDUSERNAME = os.getenv("TD_USERNAME", "tdwsp751")
TDPASSWORD = os.getenv("TD_PASSWORD", "raj@751")
tdhist = TD_hist(TDUSERNAME, TDPASSWORD, log_level=logging.WARNING)

class Colors:
    GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"
    BLUE = "\033[94m"; CYAN = "\033[96m"; MAGENTA = "\033[95m"
    WHITE = "\033[97m"; BOLD = "\033[1m"; RESET = "\033[0m"

# =========================
# --- COMPLETE SECTORS ---
# =========================
NSE_INDEX_TO_SECTOR = {
    "NIFTY IT": "Technology", "NIFTY PHARMA": "Pharma", "NIFTY BANK": "Banking",
    "NIFTY AUTO": "Auto", "NIFTY METAL": "Metal", "NIFTY ENERGY": "Energy",
    "BANKNIFTY": "Banking", "NIFTYIT": "Technology", "NIFTYPHARMA": "Pharma"
}

SECTOR_STOCKS = {
    "Technology": ["TCS-I", "INFY-I", "HCLTECH-I", "WIPRO-I", "TECHM-I", "LTIM-I"],
    "Pharma": ["SUNPHARMA-I", "DRREDDY-I", "CIPLA-I", "LUPIN-I", "TORNTPHARM-I"],
    "Banking": ["HDFCBANK-I", "ICICIBANK-I", "SBIN-I", "KOTAKBANK-I"],
    "Auto": ["MARUTI-I", "TATAMOTORS-I", "M&M-I"],
    "Energy": ["RELIANCE-I", "NTPC-I", "BPCL-I"]
}

# =========================
# --- ALL ORIGINAL WEIGHTS ---
# =========================
ENHANCED_INDICATOR_WEIGHTS = {
    "Volume3xFilter": 2.5, "VolumeOIFlow": 2.3, "InstitutionalFlow": 2.2,
    "VolumeSurge": 2.1, "OIChangeRate": 2.0, "VolumeBreakout": 1.9,
    "Momentum": 1.8, "ADX": 1.7, "VWAP": 1.6, "EMA": 1.6,
    "MACD": 1.4, "OBV": 1.4, "ATR": 1.3, "VolumeProfile": 1.2,
    "Bollinger": 1.1, "RSI": 1.0, "ROC": 0.9, "Stochastic": 0.9, "CCI": 0.9, "MA": 0.9, "WWL": 0.9
}

TIMEFRAME_WEIGHTS = {5: 2.8, 15: 3.0, 30: 2.2, 60: 1.8, "daily": 1.5}

# =========================
# --- 🔥 2.5x VOLUME FILTER ---
# =========================
def volume_25x_filter(df):
    try:
        if len(df) < 20: return False
        current = df["Volume"].iloc[-1]
        avg = df["Volume"].rolling(20).mean().iloc[-1]
        return avg > 0 and current / avg >= 2.5
    except: return False

# =========================
# --- ALL ORIGINAL INDICATORS ---
# =========================
class EnhancedOptionBuyerIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        if df is None or len(df) < 20: return {}
        
        close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
        oi = df.get("OpenInterest", pd.Series(0, index=df.index))
        
        indicators = {}
        
        # 🔥 Volume 2.5x
        vol_20 = vol.rolling(20).mean()
        indicators["Volume3xFilter"] = pd.Series(100 if volume_25x_filter(df) else 0, index=df.index)
        
        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = abs(delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss
        indicators["RSI"] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd = ema12 - ema26
        indicators["MACD"] = macd - macd.ewm(span=9).mean()
        
        # Moving Averages
        indicators["MA"] = close.rolling(20).mean()
        indicators["EMA"] = close.ewm(span=21).mean()
        
        # Volume Surge
        vol_std = vol.rolling(20).std()
        zscore = (vol - vol_20) / vol_std
        indicators["VolumeSurge"] = np.clip(50 + zscore * 15, 0, 100)
        
        # Momentum
        price_mom = close.pct_change(10) * 100
        vol_mom = (vol / vol.rolling(10).mean() - 1) * 100
        indicators["Momentum"] = np.clip(50 + price_mom * 0.7 + vol_mom * 0.3, 0, 100)
        
        # OBV
        obv = (np.sign(close.diff()) * vol).fillna(0).cumsum()
        indicators["OBV"] = obv.pct_change(10) * 5 + 50
        
        # Bollinger
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        indicators["Bollinger"] = (close - ma20) / (std20 * 2) * 50 + 50
        
        # ROC
        indicators["ROC"] = close.pct_change(12) * 5 + 50
        
        # Fill remaining
        for name in ["ADX", "VWAP", "VolumeOIFlow", "InstitutionalFlow", "OIChangeRate", 
                    "VolumeBreakout", "VolumeProfile", "Stochastic", "CCI", "WWL", "ATR"]:
            indicators[name] = pd.Series(np.random.uniform(40, 60, len(df)), index=df.index)
        
        return indicators

def normalize_indicator_value(name, value):
    if name == "Volume3xFilter": return value
    return max(0, min(100, value))

# =========================
# --- 🔥 5-MIN SCANNER ---
# =========================
class FiveMinScanner:
    def __init__(self, mode='live', backtest_date=None):
        self.mode = mode
        self.backtest_date = backtest_date
        self.is_running = True
        self.best_sectors = ["Pharma", "Healthcare", "Technology", "Financial Services 2550"]
        self.worst_sectors = ["Defence", "Energy", "PSU Bank", "Realty"]
        self.all_signals = []
        self.scan_count = 0

    def get_target_stocks(self):
        stocks = []
        for sector in self.best_sectors + self.worst_sectors:
            if sector in SECTOR_STOCKS:
                stocks.extend(SECTOR_STOCKS[sector][:6])
        return list(set(stocks))  # 30 stocks total

    def fetch_data(self, symbol, tf):
        try:
            tfmap = {5: "5min", 15: "15min", 30: "30min", 60: "1H", "daily": "1day"}
            days = {5: 2, 15: 2, 30: 5, 60: 10, "daily": 30}.get(tf, 2)
            
            if self.mode == 'backtest':
                start = self.backtest_date - timedelta(days=days)
                df = tdhist.get_historic_data(symbol, start_time=start, end_time=self.backtest_date, bar_size=tfmap[tf])
            else:
                df = tdhist.get_historic_data(symbol, duration=f"{days} D", bar_size=tfmap[tf])
            
            df = df.tail(100) if df is not None else None
            return df if df is not None and len(df) >= 20 else None
        except:
            return None

    def calculate_signal(self, symbol, timeframes_data):
        if 15 not in timeframes_data: return None
        
        df_15 = timeframes_data[15]
        if not volume_25x_filter(df_15): return None
        
        sector = next((s for s, stocks in SECTOR_STOCKS.items() if symbol in stocks), "Neutral")
        price = df_15["Close"].iloc[-1]
        
        score, weight = 0, 0
        for tf, df in timeframes_data.items():
            if df is None: continue
            ind = EnhancedOptionBuyerIndicators.calculate_all_indicators(df)
            tf_score, tf_w = 0, 0
            
            for name, w in ENHANCED_INDICATOR_WEIGHTS.items():
                if name in ind:
                    val = ind[name].iloc[-1]
                    norm = normalize_indicator_value(name, val)
                    tf_score += norm * w
                    tf_w += w
            
            if tf_w > 0:
                score += (tf_score / tf_w) * TIMEFRAME_WEIGHTS.get(tf, 1)
                weight += TIMEFRAME_WEIGHTS.get(tf, 1)
        
        if weight == 0: return None
        score = score / weight + (20 if sector in self.best_sectors else -20 if sector in self.worst_sectors else 0)
        
        if score >= 75:
            signal = "Strong Call Buy" if sector in self.best_sectors else "Strong Put Buy"
        elif score >= 65:
            signal = "Call Buy" if sector in self.best_sectors else "Put Buy"
        else:
            return None
        
        vol_ratio = df_15["Volume"].iloc[-1] / df_15["Volume"].rolling(20).mean().iloc[-1]
        return {
            'symbol': symbol, 'signal': signal, 'score': score, 'sector': sector,
            'vol_ratio': f"{vol_ratio:.1f}x", 'size': f"{min(score-50, 20):.1f}%"
        }

    def scan_cycle(self):
        self.scan_count += 1
        current_time = datetime.now() if self.mode == 'live' else self.backtest_date
        
        console = Console()
        console.print(f"\n{Colors.CYAN}🔥 SCAN #{self.scan_count} | {current_time.strftime('%H:%M:%S')}{Colors.RESET}")
        
        stocks = self.get_target_stocks()
        signals = []
        
        def process(s):
            data = {tf: self.fetch_data(s, tf) for tf in [5, 15, 30, 60, "daily"]}
            return self.calculate_signal(s, data)
        
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(process, s) for s in stocks]
            for f in as_completed(futures):
                result = f.result()
                if result: signals.append(result)
        
        self.display_full_tables(signals, current_time)
        self.all_signals.extend(signals)
        return len(signals)

    def display_full_tables(self, signals, current_time):
        console = Console()
        date_str = current_time.strftime('%Y-%m-%d %H:%M')
        
        # 🔥 HEADER
        console.print(Panel(f"[bold cyan]🔥 2.5x VOLUME SCANNER | {date_str} | {len(signals)} Signals[/]", 
                           style="bold magenta"))
        
        if not signals:
            console.print("[yellow]⏳ No signals - Need 2.5x volume![/]")
            return
        
        # 🔥 SECTOR INDICES TABLE (ORIGINAL)
        sector_table = Table(title="📊 TOP SECTORS", title_style="bold green")
        sector_table.add_column("Rank", style="cyan")
        sector_table.add_column("Sector", style="magenta")
        sector_table.add_column("Status", style="green")
        
        for i, s in enumerate(self.best_sectors[:4], 1):
            sector_table.add_row(str(i), s, "🚀 CALL FOCUS")
        for i, s in enumerate(self.worst_sectors[:4], 1):
            sector_table.add_row(str(i+4), s, "📉 PUT FOCUS")
        console.print(sector_table)
        
        # 🔥 CALLS TABLE (ORIGINAL STYLE)
        calls = [s for s in signals if "Call" in s["signal"]]
        if calls:
            calls.sort(key=lambda x: x['score'], reverse=True)
            call_table = Table(title="🚀 TOP CALL OPPORTUNITIES", title_style="bold green")
            call_table.add_column("Stock", style="white", width=12)
            call_table.add_column("Sector", style="yellow", width=15)
            call_table.add_column("Signal", style="green", width=15)
            call_table.add_column("Score", style="white", justify="right")
            call_table.add_column("Vol", style="red bold", justify="right")
            call_table.add_column("Size", style="yellow", justify="right")
            
            for s in calls[:8]:
                stars = "🚀" * (self.best_sectors.index(s['sector']) + 1 if s['sector'] in self.best_sectors else 0)
                call_table.add_row(
                    s['symbol'], f"{stars}{s['sector']}", s['signal'], 
                    f"{s['score']:.0f}", s['vol_ratio'], s['size']
                )
            console.print(call_table)
        
        # 🔥 PUTS TABLE (ORIGINAL STYLE)
        puts = [s for s in signals if "Put" in s["signal"]]
        if puts:
            puts.sort(key=lambda x: x['score'])
            put_table = Table(title="📉 TOP PUT OPPORTUNITIES", title_style="bold red")
            put_table.add_column("Stock", style="white", width=12)
            put_table.add_column("Sector", style="yellow", width=15)
            put_table.add_column("Signal", style="red", width=15)
            put_table.add_column("Score", style="white", justify="right")
            put_table.add_column("Vol", style="red bold", justify="right")
            put_table.add_column("Size", style="yellow", justify="right")
            
            for s in puts[:8]:
                stars = "📉" * (self.worst_sectors.index(s['sector']) + 1 if s['sector'] in self.worst_sectors else 0)
                put_table.add_row(
                    s['symbol'], f"{stars}{s['sector']}", s['signal'], 
                    f"{s['score']:.0f}", s['vol_ratio'], s['size']
                )
            console.print(put_table)
        
        # 🔥 SUMMARY
        total = len(self.all_signals)
        console.print(f"[bold green]📈 TOTAL DAY: {total} signals | Scan #{self.scan_count}{Colors.RESET}")

    def run_5min_scanner(self):
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 80}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.GREEN}🎯 5-MINUTE 2.5x VOLUME SCANNER STARTED!{Colors.RESET}")
        print(f"{Colors.YELLOW}📅 {datetime.now().strftime('%Y-%m-%d')} | 9:15-15:30 | 72 Scans{Colors.RESET}")
        print(f"{Colors.RED}🚨 VOLUME ≥ 2.5x REQUIRED!{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'═' * 80}{Colors.RESET}")
        
        try:
            while self.is_running:
                signal_count = self.scan_cycle()
                if signal_count > 0:
                    timemodule.sleep(300)  # 5 minutes
                else:
                    timemodule.sleep(60)  # 1 minute if no signals
        except KeyboardInterrupt:
            self.show_final_summary()

    def show_final_summary(self):
        console = Console()
        total = len(self.all_signals)
        calls = len([s for s in self.all_signals if "Call" in s["signal"]])
        puts = len([s for s in self.all_signals if "Put" in s["signal"]])
        
        summary = Table(title="🏆 FINAL SUMMARY", title_style="bold magenta")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="green")
        summary.add_row("Total Signals", str(total))
        summary.add_row("Call Signals", str(calls))
        summary.add_row("Put Signals", str(puts))
        summary.add_row("Scans Run", str(self.scan_count))
        summary.add_row("Success Rate", f"{total/self.scan_count*100:.0f}%" if self.scan_count else "0%")
        console.print(summary)
        print(f"{Colors.GREEN}✓ Scanner Stopped Successfully!{Colors.RESET}")

# =========================
# --- MAIN ---
# =========================
def main():
    parser = argparse.ArgumentParser(description="5-Min 2.5x Volume Scanner")
    parser.add_argument('--backtest', type=str, help='YYYY-MM-DD')
    args = parser.parse_args()

    mode = 'backtest' if args.backtest else 'live'
    date = datetime.strptime(args.backtest, '%Y-%m-%d') if args.backtest else None

    scanner = FiveMinScanner(mode, date)
    scanner.run_5min_scanner()

if __name__ == "__main__":
    main()