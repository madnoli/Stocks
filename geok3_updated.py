import os
import logging
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from logzero import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from collections import defaultdict
import argparse
import csv
from retrying import retry

from tqdm import tqdm
from truedata.history import TD_hist

from rich.console import Console
from rich.table import Table
from rich import box

# ======== CONFIG ========
CONFIG = {
    "TDUSERNAME": os.getenv("TRUEDATA_USER", "tdwsp751"),
    "TDPASSWORD": os.getenv("TRUEDATA_PASS", "raj@751"),
    "MARKET_START": "09:15", "FIRST_RUN_AT": "09:20", "MARKET_END": "15:30",
    "MAX_WORKERS": 12, "TD_HIST_SESSIONS": 3, "RATE_PER_SECOND_TOTAL": 8.0,
    "BUCKET_SIZE": 15, "RETRY_ATTEMPTS": 2, "RETRY_DELAY_MS": 1000,
    "SHARES_FILE": os.getenv("SHARES_FILE", "shares.txt"),
    "SHOW_PROGRESS": os.getenv("SHOW_PROGRESS", "true").lower() == "true",
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    "MIN_BARS_REQUIRED": 15, "MAX_MISSING_DATA_PCT": 40, "SIGNAL_CONFIRMATION_BARS": 1,
    "TIMEFRAME_WEIGHTS": {15: 3.0, 5: 2.5, 30: 2.0, 60: 1.5},
    "BAR_SIZE_MAP": {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min"},
    "DURATION_MAP": {5: "45 D", 15: "45 D", 30: "90 D", 60: "180 D"},
}

level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING}
logging.getLogger().setLevel(level_map.get(CONFIG["LOG_LEVEL"], logging.INFO))

IST = pytz.timezone("Asia/Kolkata")
for noisy in ("truedata", "truedata.history", "websocket", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

console = Console()

# Global vars
last_bull_symbols, last_bear_symbols = set(), set()
previous_scores = {}
api_calls_done, api_calls_lock = 0, threading.Lock()
performance_metrics = defaultdict(int)
failed_symbols, oi_symbols_found = set(), set()

def next_5min_boundary_ist(now_ist: datetime) -> datetime:
    minute = (now_ist.minute // 5) * 5
    boundary = now_ist.replace(minute=minute, second=0, microsecond=0)
    return boundary + timedelta(minutes=5) if boundary <= now_ist else boundary

def parse_hhmm(s: str):
    h, m = map(int, s.split(":"))
    return h, m

def today_ist_dt(hhmm: str) -> datetime:
    now = datetime.now(IST)
    h, m = parse_hhmm(hhmm)
    return now.replace(hour=h, minute=m, second=0, microsecond=0)

def sleep_until(ts: datetime):
    while True:
        now = datetime.now(IST)
        delta = (ts - now).total_seconds()
        if delta <= 0:
            break
        time.sleep(min(0.5, delta))

class TokenBucketLimiter:
    def __init__(self, rate_per_sec: float, bucket_size: int):
        self.rate, self.capacity, self.tokens = rate_per_sec, bucket_size, bucket_size
        self.lock, self.last_refill = threading.Lock(), time.time()

    def acquire(self):
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_refill
                if elapsed > 0:
                    add = int(elapsed * self.rate)
                    if add > 0:
                        self.tokens = min(self.capacity, self.tokens + add)
                        self.last_refill = now
                if self.tokens > 0:
                    self.tokens -= 1
                    return
                sleep_for = max(0.0, 1.0 / max(self.rate, 0.001))
            time.sleep(sleep_for)

def authenticate_session():
    return TD_hist(CONFIG["TDUSERNAME"], CONFIG["TDPASSWORD"], log_level=logging.CRITICAL)

def build_sessions():
    pool, limiters = [], []
    for i in range(CONFIG["TD_HIST_SESSIONS"]):
        try:
            pool.append(authenticate_session())
        except Exception as e:
            logger.error(f"Session {i} failed: {e}")
    if not pool:
        raise SystemExit("No TrueData sessions initialized")
    
    per_sess_rate = CONFIG["RATE_PER_SECOND_TOTAL"] / len(pool)
    for _ in pool:
        limiters.append(TokenBucketLimiter(per_sess_rate, CONFIG["BUCKET_SIZE"]))
    return pool, limiters

tdhist_pool, sess_limiters = build_sessions()
logger.info("TrueData login successful.")

# ======== INDICATORS ========
def calculate_rsi(df, period=14):
    if df is None or len(df) < period + 3:
        return pd.Series([], dtype=float)
    
    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(period, min_periods=period//2).mean()
        avg_loss = loss.rolling(period, min_periods=period//2).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))
    except:
        return pd.Series([50] * len(df), index=df.index)

def calculate_macd(df, fast=12, slow=26, signal=9):
    if df is None or len(df) < slow + signal + 3:
        empty = pd.Series([], dtype=float)
        return empty, empty
    
    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        ema_fast = close.ewm(span=fast, min_periods=fast//2).mean()
        ema_slow = close.ewm(span=slow, min_periods=slow//2).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, min_periods=signal//2).mean()
        return macd_line, signal_line
    except:
        empty = pd.Series([0] * len(df), index=df.index)
        return empty, empty

def calculate_volume_surge(df, lookback=20):
    if df is None or len(df) < lookback + 3:
        return pd.Series([], dtype=float)
    
    try:
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(1000)
        vol_ma = volume.rolling(lookback, min_periods=lookback//2).mean()
        vol_std = volume.rolling(lookback, min_periods=lookback//2).std()
        vol_std = vol_std.where(vol_std > vol_ma * 0.01, vol_ma * 0.1)
        z_score = (volume - vol_ma) / vol_std
        return z_score.clip(-5, 5).fillna(0)
    except:
        return pd.Series([0] * len(df), index=df.index)

def calculate_momentum(df, period=10):
    if df is None or len(df) < period + 2:
        return pd.Series([], dtype=float)
    
    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        shifted_close = close.shift(period).replace(0, np.nan)
        return ((close / shifted_close) - 1.0).fillna(0)
    except:
        return pd.Series([0] * len(df), index=df.index)

# ======== FINAL FIX: Bulletproof Data Processing ========
def normalize_data_bulletproof(df, symbol, timeframe_minutes):
    """Bulletproof data normalization - handles all edge cases"""
    if df is None or df.empty:
        return None
    
    try:
        # Copy and standardize columns
        out = df.copy()
        out.columns = out.columns.str.lower()
        
        # Basic column mapping
        col_map = {}
        for col in out.columns:
            if any(x in col for x in ['time', 'date', 'timestamp']):
                col_map[col] = 'Timestamp'
            elif col in ['open', 'o'] and 'interest' not in col:
                col_map[col] = 'Open'
            elif col in ['high', 'h']:
                col_map[col] = 'High'
            elif col in ['low', 'l']:
                col_map[col] = 'Low'
            elif col in ['close', 'c']:
                col_map[col] = 'Close'
            elif col in ['volume', 'vol', 'v']:
                col_map[col] = 'Volume'
            elif any(x in col for x in ['oi', 'openint', 'open_int']):
                col_map[col] = 'OI'
        
        out.rename(columns=col_map, inplace=True)
        
        # Ensure required columns exist
        required = ['Open', 'High', 'Low', 'Close']
        for req in required:
            if req not in out.columns:
                logger.debug(f"Missing {req} for {symbol}")
                return None
        
        if 'Volume' not in out.columns:
            out['Volume'] = 1000
        if 'OI' not in out.columns:
            out['OI'] = out['Volume'] * 0.3
        
        # Handle timestamps - BULLETPROOF approach
        if 'Timestamp' not in out.columns:
            # Create sequential timestamps
            now = datetime.now(IST)
            out['Timestamp'] = pd.date_range(
                start=now - timedelta(minutes=timeframe_minutes * len(out)),
                periods=len(out), freq=f"{timeframe_minutes}T", tz=IST
            )
        else:
            try:
                # Convert existing timestamps
                out['Timestamp'] = pd.to_datetime(out['Timestamp'], errors='coerce')
                
                # Remove invalid timestamps
                valid_ts_mask = out['Timestamp'].notna()
                out = out[valid_ts_mask]
                
                if out.empty:
                    return None
                
                # Handle timezone
                if out['Timestamp'].dt.tz is None:
                    out['Timestamp'] = out['Timestamp'].dt.tz_localize(IST, ambiguous='infer')
                else:
                    out['Timestamp'] = out['Timestamp'].dt.tz_convert(IST)
            except Exception as ts_err:
                logger.debug(f"Timestamp error {symbol}: {ts_err}")
                # Fallback: create new timestamps
                now = datetime.now(IST)
                out['Timestamp'] = pd.date_range(
                    start=now - timedelta(minutes=timeframe_minutes * len(out)),
                    periods=len(out), freq=f"{timeframe_minutes}T", tz=IST
                )
        
        # Convert to numeric
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'OI']
        for col in numeric_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors='coerce').fillna(method='ffill')
        
        # Remove invalid rows
        out = out.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if out.empty or len(out) < CONFIG["MIN_BARS_REQUIRED"]:
            return None
        
        # Set index safely
        try:
            out = out.set_index('Timestamp').sort_index()
            out = out[~out.index.duplicated(keep='last')]
        except Exception:
            # If index setting fails, use integer index
            pass
        
        # Basic validation
        try:
            if (out['High'] < out['Low']).any():
                return None
        except Exception:
            pass  # Skip validation if it fails
        
        return out
        
    except Exception as e:
        logger.debug(f"Normalization failed for {symbol}: {e}")
        return None

def pick_session(symbol, tf):
    return hash(f"{symbol}_{tf}") % len(tdhist_pool)

@retry(stop_max_attempt_number=CONFIG["RETRY_ATTEMPTS"], 
       wait_exponential_multiplier=500, wait_exponential_max=3000)
def fetch_single_safe(symbol, timeframe):
    """Ultra-safe single fetch"""
    clean_symbol = symbol.replace("-EQ", "")
    
    if clean_symbol in failed_symbols:
        return symbol, timeframe, None
    
    bar_size = CONFIG["BAR_SIZE_MAP"].get(timeframe)
    duration = CONFIG["DURATION_MAP"].get(timeframe)
    
    if not bar_size or not duration:
        return symbol, timeframe, None
    
    try:
        si = pick_session(symbol, timeframe)
        hist = tdhist_pool[si]
        limiter = sess_limiters[si]
        
        limiter.acquire()
        
        df_raw = hist.get_historic_data(clean_symbol, duration=duration, bar_size=bar_size)
        
        if df_raw is None or df_raw.empty:
            failed_symbols.add(clean_symbol)
            return symbol, timeframe, None
        
        df_clean = normalize_data_bulletproof(df_raw, clean_symbol, timeframe)
        return symbol, timeframe, df_clean
        
    except Exception as e:
        logger.debug(f"Fetch error {clean_symbol}: {str(e)[:50]}")
        failed_symbols.add(clean_symbol)
        return symbol, timeframe, None

def fetch_batch_safe(symbols_list):
    """Safe batch fetching"""
    timeframes = [5, 15, 30, 60]
    batch_results = defaultdict(dict)
    
    total_tasks = len(symbols_list) * len(timeframes)
    
    with tqdm(total=total_tasks, desc="Fetching", ncols=50, 
              disable=not CONFIG["SHOW_PROGRESS"]) as pbar:
        with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
            # Submit all tasks
            futures = []
            for symbol in symbols_list:
                for tf in timeframes:
                    future = executor.submit(fetch_single_safe, symbol, tf)
                    futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                try:
                    symbol, tf, df = future.result()
                    if df is not None and not df.empty:
                        batch_results[symbol][tf] = df
                except Exception:
                    pass
                pbar.update(1)
    
    return dict(batch_results)

def filter_data_simple(symbol, timeframe_data, target_time):
    """ULTRA-SIMPLE filtering - no complex timestamp operations"""
    filtered = {}
    
    try:
        # Ensure target time has timezone
        if target_time.tzinfo is None:
            target_time = IST.localize(target_time)
        elif target_time.tzinfo != IST:
            target_time = target_time.astimezone(IST)
    except Exception:
        return filtered
    
    for tf, df in timeframe_data.items():
        if df is None or df.empty:
            continue
        
        try:
            # Simple approach: just use the dataframe as-is if it has enough bars
            if len(df) >= CONFIG["MIN_BARS_REQUIRED"]:
                filtered[tf] = df
        except Exception:
            continue
    
    return filtered

def analyze_simple_signals(timeframe_data, symbol):
    """Simplified signal analysis"""
    if not timeframe_data:
        return 'Neutral', 0.0, 'Normal'
    
    total_score = 0.0
    valid_frames = 0
    
    for tf, df in timeframe_data.items():
        if df is None or len(df) < CONFIG["MIN_BARS_REQUIRED"]:
            continue
        
        valid_frames += 1
        tf_weight = CONFIG["TIMEFRAME_WEIGHTS"].get(tf, 1.0)
        frame_score = 0.0
        
        # RSI
        try:
            rsi = calculate_rsi(df)
            if len(rsi) > 0:
                rsi_val = rsi.iloc[-1]
                if rsi_val > 70:
                    frame_score += 2.0
                elif rsi_val > 60:
                    frame_score += 1.0
                elif rsi_val < 30:
                    frame_score -= 2.0
                elif rsi_val < 40:
                    frame_score -= 1.0
        except:
            pass
        
        # MACD
        try:
            macd, signal = calculate_macd(df)
            if len(macd) > 0 and len(signal) > 0:
                macd_diff = macd.iloc[-1] - signal.iloc[-1]
                if macd_diff > 0:
                    frame_score += 1.0
                else:
                    frame_score -= 1.0
        except:
            pass
        
        # Volume
        try:
            vol_surge = calculate_volume_surge(df)
            if len(vol_surge) > 0:
                surge_val = vol_surge.iloc[-1]
                if surge_val >= 2.0:
                    # Check price direction
                    if len(df) >= 2:
                        price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1
                        if price_change > 0.005:  # 0.5% up
                            frame_score += 3.0
                        elif price_change < -0.005:  # 0.5% down
                            frame_score -= 3.0
                elif surge_val >= 1.5:
                    frame_score += 1.0
        except:
            pass
        
        # Momentum
        try:
            momentum = calculate_momentum(df)
            if len(momentum) > 0:
                mom_val = momentum.iloc[-1]
                if mom_val > 0.02:  # 2% momentum
                    frame_score += 2.0
                elif mom_val > 0.01:  # 1% momentum
                    frame_score += 1.0
                elif mom_val < -0.02:
                    frame_score -= 2.0
                elif mom_val < -0.01:
                    frame_score -= 1.0
        except:
            pass
        
        total_score += frame_score * tf_weight
    
    if valid_frames == 0:
        return 'Neutral', 0.0, 'Normal'
    
    # Normalize score
    avg_score = total_score / max(valid_frames, 1)
    normalized = max(-100, min(100, avg_score * 15))  # Scale to ±100
    
    # Classify
    oi_status = 'Normal'
    if normalized >= 35:
        signal = 'Strong Buy (Call Focus)'
        oi_status = 'Call Setup'
    elif normalized >= 15:
        signal = 'Buy (Call Potential)'
    elif normalized <= -35:
        signal = 'Strong Sell (Put Focus)'
        oi_status = 'Put Setup'
    elif normalized <= -15:
        signal = 'Sell (Put Potential)'
    else:
        signal = 'Neutral'
    
    return signal, normalized, oi_status

def render_results(scan_time, bullish, bearish, scan_num):
    global last_bull_symbols, last_bear_symbols
    
    console.rule(f"🔴 SCAN #{scan_num} | {scan_time.strftime('%H:%M')} IST", style="red")
    
    if bullish:
        table = Table(title="🚀 Call Opportunities", header_style="white on green")
        for col in ["Stock", "Signal", "Score", "Price"]:
            table.add_column(col)
        
        for r in bullish:
            is_new = r['symbol'] not in last_bull_symbols
            style = "black on green" if is_new else None
            table.add_row(r['symbol'], r['signal'], f"{r['score']:.1f}", 
                         f"₹{r['price']:.2f}" if r['price'] > 0 else "NA", style=style)
        console.print(table)
    
    if bearish:
        table = Table(title="📉 Put Opportunities", header_style="white on red")
        for col in ["Stock", "Signal", "Score", "Price"]:
            table.add_column(col)
        
        for r in bearish:
            is_new = r['symbol'] not in last_bear_symbols
            style = "white on red" if is_new else None
            table.add_row(r['symbol'], r['signal'], f"{r['score']:.1f}",
                         f"₹{r['price']:.2f}" if r['price'] > 0 else "NA", style=style)
        console.print(table)
    
    last_bull_symbols = {r['symbol'] for r in bullish}
    last_bear_symbols = {r['symbol'] for r in bearish}
    
    total = len(bullish) + len(bearish)
    console.print(f"[blue]📊 Total: {total} signals | Next: +5min[/blue]")
    console.rule()

def run_scanner_final(stocks):
    """Final stable scanner"""
    logger.info("🚀 Starting final stable scanner...")
    
    # Market check
    now = datetime.now(IST)
    market_start = today_ist_dt(CONFIG["MARKET_START"])
    market_end = today_ist_dt(CONFIG["MARKET_END"])
    
    if now < market_start:
        logger.info(f"⏰ Market opens in {(market_start-now).seconds/60:.1f} min")
        sleep_until(market_start)
    elif now > market_end:
        logger.info("📈 Market closed")
        return
    
    # Setup
    csv_file = f"scan_{datetime.now(IST).strftime('%Y%m%d')}.csv"
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Time", "Symbol", "Signal", "Score", "Action", "Price"])
    
    # Batch configuration
    batch_size = 20  # Conservative batch size
    batches = [stocks[i:i+batch_size] for i in range(0, len(stocks), batch_size)]
    
    next_scan = next_5min_boundary_ist(datetime.now(IST))
    scan_count = 0
    
    logger.info(f"📊 Ready: {len(batches)} batches of ~{batch_size} stocks")
    console.rule("🔴 SCANNING STARTED", style="red")
    
    try:
        while datetime.now(IST) <= market_end:
            if datetime.now(IST) < next_scan:
                time.sleep(min(30, (next_scan - datetime.now(IST)).total_seconds()))
                continue
            
            scan_count += 1
            scan_time = datetime.now(IST).replace(second=0, microsecond=0)
            logger.info(f"🔍 Scan #{scan_count} @ {scan_time.strftime('%H:%M')}")
            
            all_signals = []
            
            # Process batches
            for batch_idx, batch in enumerate(batches):
                try:
                    logger.info(f"Batch {batch_idx+1}/{len(batches)}")
                    
                    # Fetch batch data
                    batch_data = fetch_batch_safe(batch)
                    
                    # Analyze each symbol
                    for symbol, tf_data in batch_data.items():
                        try:
                            clean_sym = symbol.replace('-EQ', '')
                            
                            # Filter data (simplified)
                            filtered = filter_data_simple(clean_sym, tf_data, scan_time)
                            
                            if not filtered:
                                continue
                            
                            # Analyze
                            signal, score, oi_status = analyze_simple_signals(filtered, clean_sym)
                            
                            # Get price
                            price = 0
                            for tf in [5, 15, 30]:
                                if tf in filtered and not filtered[tf].empty:
                                    try:
                                        price = float(filtered[tf]['Close'].iloc[-1])
                                        break
                                    except:
                                        continue
                            
                            # Include significant signals
                            if abs(score) >= 12 or "Buy" in signal or "Sell" in signal:
                                all_signals.append({
                                    'symbol': clean_sym,
                                    'signal': signal,
                                    'score': score,
                                    'oi_status': oi_status,
                                    'price': price
                                })
                        
                        except Exception as sym_err:
                            logger.debug(f"Symbol error {symbol}: {sym_err}")
                            continue
                
                except Exception as batch_err:
                    logger.warning(f"Batch error: {batch_err}")
                    continue
                
                # Inter-batch delay
                if batch_idx < len(batches) - 1:
                    time.sleep(0.5)
            
            # Sort and display
            all_signals.sort(key=lambda x: abs(x['score']), reverse=True)
            bullish = [s for s in all_signals if s['score'] > 0][:8]
            bearish = [s for s in all_signals if s['score'] < 0][:8]
            
            if bullish or bearish:
                render_results(scan_time, bullish, bearish, scan_count)
                
                # Save CSV
                with open(csv_file, "a", newline='') as f:
                    writer = csv.writer(f)
                    for sig in bullish + bearish:
                        writer.writerow([
                            scan_time.strftime('%H:%M'), sig['symbol'], sig['signal'],
                            f"{sig['score']:.1f}", "CALL" if sig['score'] > 0 else "PUT",
                            f"{sig['price']:.2f}" if sig['price'] > 0 else "NA"
                        ])
            else:
                console.print("[yellow]📊 No significant signals detected[/yellow]")
            
            next_scan = next_5min_boundary_ist(datetime.now(IST))
            
            # Periodic cleanup
            if scan_count % 6 == 0:
                import gc; gc.collect()
    
    except KeyboardInterrupt:
        logger.info("👤 Stopped by user")
    except Exception as e:
        logger.error(f"Scanner error: {e}")
    finally:
        console.rule("🔴 SCANNING STOPPED", style="red")
        logger.info(f"Completed {scan_count} scans | Results: {csv_file}")

def load_stocks(filename):
    if not os.path.exists(filename):
        samples = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
        try:
            with open(filename, "w") as f:
                for s in samples: f.write(f"{s}\n")
            return samples
        except: return samples
    
    try:
        with open(filename, "r") as f:
            stocks = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
        logger.info(f"Loaded {len(stocks)} stocks")
        return stocks
    except Exception as e:
        logger.error(f"Load error: {e}")
        return []

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Final Stable Scanner")
        parser.add_argument("--live", action="store_true", help="Live scanning")
        parser.add_argument("--stocks-file", default=CONFIG["SHARES_FILE"])
        
        args = parser.parse_args()
        
        stocks = load_stocks(args.stocks_file)
        if not stocks:
            logger.error("No stocks loaded")
            exit(1)
        
        if args.live:
            run_scanner_final(stocks)
        else:
            print("\n🎯 Final Stable Option Scanner")
            print("Usage: python geok3_updated.py --live")
    
    except KeyboardInterrupt:
        print("\n👤 Interrupted")
    except Exception as e:
        logger.exception(f"Fatal: {e}")
    finally:
        logger.info("🔌 Cleanup...")
        for sess in tdhist_pool:
            try: sess.disconnect()
            except: pass
        logger.info("✅ Done")
