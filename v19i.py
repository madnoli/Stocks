# Live Trading Signal Script - v22 (Merged)
# Features:
# - Data Engine: Uses the high-performance, multi-session TrueData logic from the options scanner.
# - Core Analysis: Retains the original scoring, weights, and signal logic from v20 for consistency.
# - Live Operation: Runs a continuous 5-minute loop during market hours with precise candle timing.
# - Output: Preserves the original table format with momentum tracking.

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import time as time_module
import pytz
from logzero import logger
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import logging
import warnings
from tqdm import tqdm

# --- Suppress third-party warnings and logs ---
warnings.filterwarnings("ignore")
for noisy in ("truedata", "truedata.history", "truedata_ws", "websocket", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

# --- CONFIG ---
from config import username as TDUSERNAME, password as TDPASSWORD
from truedata.history import TD_hist

# --- COLOR CODES FOR TERMINAL OUTPUT ---
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

# --- CONFIGURABLE PARAMETERS (FROM SCRIPT 1) ---
TIMEFRAME_WEIGHTS = {
    5: 0.5, 10: 0.75, 15: 1.0, 30: 1.25, 60: 1.5, 1440: 2.0
}
INDICATOR_WEIGHTS = {
    'RSI': 1.0, 'MACD': 1.2, 'Stochastic': 0.8, 'MA': 1.5,
    'ADX': 1.2, 'Bollinger': 1.0, 'ROC': 0.7, 'OBV': 1.3, 'CCI': 0.9
}
TIMEFRAMES_IN_MINUTES = [5, 10, 15, 30, 60, 1440]
IST = pytz.timezone("Asia/Kolkata")

# --- TRUEDATA SESSION MANAGEMENT (FROM SCRIPT 2) ---
class TokenBucketLimiter:
    def __init__(self, rate_per_sec: float, bucket_size: int):
        self.rate = rate_per_sec
        self.capacity = bucket_size
        self.tokens = bucket_size
        self.lock = threading.Lock()
        self.last_refill = time_module.time()

    def acquire(self):
        while True:
            with self.lock:
                now = time_module.time()
                elapsed = now - self.last_refill
                if elapsed > 0:
                    add = int(elapsed * self.rate)
                    if add > 0:
                        self.tokens = min(self.capacity, self.tokens + add)
                        self.last_refill = now
                if self.tokens > 0:
                    self.tokens -= 1
                    return
                sleep_for = max(0.0, 1.0 / self.rate)
            time_module.sleep(sleep_for)

def authenticate_session():
    return TD_hist(TDUSERNAME, TDPASSWORD, log_level=logging.CRITICAL)

def build_sessions(session_count=3):
    pool = []
    for i in range(session_count):
        try:
            pool.append(authenticate_session())
        except Exception as e:
            logger.error(f"Session {i} init failed: {e}")
    if not pool:
        raise SystemExit("Failed to initialize TrueData sessions.")
    per_sess_rate = 10.0 / len(pool)
    limiters = [TokenBucketLimiter(rate_per_sec=per_sess_rate, bucket_size=10) for _ in pool]
    return pool, limiters

tdhist_pool, sess_limiters = build_sessions()
logger.info("TrueData login successful.")

# --- DATA FETCHING (FROM SCRIPT 2, ADAPTED) ---
BAR_SIZE_MAP = {5: "5 min", 10: "10 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"}
DURATION_MAP = {5: "30 D", 10: "30 D", 15: "30 D", 30: "60 D", 60: "120 D", 1440: "365 D"}

def normalize_hist_df(df, symbol):
    if df is None or df.empty: return None
    try:
        out = df.copy()
        out.rename(columns={c: str(c).lower() for c in out.columns}, inplace=True)
        rename_map = {"timestamp": "Date", "time": "Date", "datetime": "Date", "date": "Date",
                      "open": "Open", "high": "High", "low": "Low", "close": "Close",
                      "volume": "Volume", "vol": "Volume"}
        out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns}, inplace=True)
        if "Date" not in out.columns: return None
        if "Volume" not in out.columns: out["Volume"] = 0
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"])
        out["Date"] = out["Date"].dt.tz_localize(IST) if not pd.api.types.is_datetime64tz_dtype(out["Date"]) else out["Date"].dt.tz_convert(IST)
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")
        out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_values("Date").set_index("Date")
        return out if len(out) >= 50 else None
    except Exception as e:
        logger.error(f"Normalize error {symbol}: {e}")
        return None

def fetch_one(symbol_orig, timeframe_minutes, limiter, hist):
    td_symbol = symbol_orig.replace('-EQ', '')
    bar_size = BAR_SIZE_MAP.get(timeframe_minutes)
    duration = DURATION_MAP.get(timeframe_minutes)
    if not bar_size or not duration: return symbol_orig, timeframe_minutes, None
    try:
        limiter.acquire()
        df_raw = hist.get_historic_data(td_symbol, duration=duration, bar_size=bar_size)
        df = normalize_hist_df(df_raw, td_symbol)
        return symbol_orig, timeframe_minutes, df
    except Exception:
        return symbol_orig, timeframe_minutes, None

def prefetch_all(stocks, max_workers=64):
    total_calls = len(stocks) * len(TIMEFRAMES_IN_MINUTES)
    stock_multi_data = defaultdict(dict)
    with tqdm(total=total_calls, desc="Fetching Data", ncols=100) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for s in stocks:
                for tf in TIMEFRAMES_IN_MINUTES:
                    si = (hash(s) ^ tf) % len(tdhist_pool)
                    futures.append(executor.submit(fetch_one, s, tf, sess_limiters[si], tdhist_pool[si]))
            for fut in as_completed(futures):
                symbol_orig, tf, df = fut.result()
                if df is not None:
                    stock_multi_data[symbol_orig][tf] = df
                pbar.update(1)
    return {s: d for s, d in stock_multi_data.items() if len(d) >= 4}


# --- INDICATOR & SCORING FUNCTIONS (UNCHANGED FROM SCRIPT 1) ---
def calculate_rsi(df, period=14):
    if len(df) < period + 1: return pd.Series(dtype='float64')
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan); return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    if len(df) < slow + signal: return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean(); return macd, signal_line

def calculate_stochastic(df, period=14, smooth_d=3):
    if len(df) < period + smooth_d: return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan))
    d = k.rolling(window=smooth_d).mean(); return k, d

def calculate_moving_averages(df, short=50, long=200):
    if len(df) < long: return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    return df['Close'].rolling(window=short).mean(), df['Close'].rolling(window=long).mean()

def calculate_adx(df, period=14):
    if len(df) < period * 2: return pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64')
    df_adx = df.copy(); df_adx['H-L'] = df_adx['High'] - df_adx['Low']; df_adx['H-C'] = abs(df_adx['High'] - df_adx['Close'].shift(1)); df_adx['L-C'] = abs(df_adx['Low'] - df_adx['Close'].shift(1)); df_adx['TR'] = df_adx[['H-L', 'H-C', 'L-C']].max(axis=1)
    df_adx['+DM'] = np.where((df_adx['High'] - df_adx['High'].shift(1)) > (df_adx['Low'].shift(1) - df_adx['Low']), df_adx['High'] - df_adx['High'].shift(1), 0)
    df_adx['-DM'] = np.where((df_adx['Low'].shift(1) - df_adx['Low']) > (df_adx['High'] - df_adx['High'].shift(1)), df_adx['Low'].shift(1) - df_adx['Low'], 0)
    atr = df_adx['TR'].ewm(com=period - 1, adjust=False).mean().replace(0, np.nan); pdi = (df_adx['+DM'].ewm(com=period - 1, adjust=False).mean() / atr) * 100
    ndi = (df_adx['-DM'].ewm(com=period - 1, adjust=False).mean() / atr) * 100; adx = (abs(pdi - ndi) / (pdi + ndi).replace(0, np.nan)).ewm(com=period - 1, adjust=False).mean() * 100
    return adx, pdi, ndi

def calculate_bollinger_bands(df, period=20):
    if len(df) < period: return pd.Series(dtype='float64')
    return df['Close'].rolling(window=period).mean()

def calculate_roc(df, period=12):
    if len(df) < period + 1: return pd.Series(dtype='float64')
    shifted_close = df['Close'].shift(period).replace(0, np.nan)
    return ((df['Close'] - shifted_close) / shifted_close) * 100

def calculate_obv(df):
    if len(df) < 2: return pd.Series(dtype='float64')
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

def calculate_cci(df, period=20):
    if len(df) < period: return pd.Series(dtype='float64')
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).replace(0, np.nan)
    return (tp - sma_tp) / (0.015 * mad)

def get_indicator_scores(df):
    scores = {}
    try:
        rsi = calculate_rsi(df).iloc[-1]
        if pd.notna(rsi):
            if rsi > 70: scores['RSI'] = -1.5
            elif rsi > 55: scores['RSI'] = 1.0
            elif rsi < 30: scores['RSI'] = 1.5
            elif rsi < 45: scores['RSI'] = -1.0
            else: scores['RSI'] = 0.0
        else: scores['RSI'] = 0.0
        macd, signal = calculate_macd(df); scores['MACD'] = 1.0 if pd.notna(macd.iloc[-1]) and pd.notna(signal.iloc[-1]) and macd.iloc[-1] > signal.iloc[-1] else -1.0
        k, d = calculate_stochastic(df)
        if pd.notna(k.iloc[-1]) and pd.notna(d.iloc[-1]):
            if k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80: scores['Stochastic'] = 1.0
            elif k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 20: scores['Stochastic'] = -1.0
            else: scores['Stochastic'] = 0.0
        else: scores['Stochastic'] = 0.0
        ma_short, ma_long = calculate_moving_averages(df); scores['MA'] = 1.0 if pd.notna(ma_short.iloc[-1]) and pd.notna(ma_long.iloc[-1]) and ma_short.iloc[-1] > ma_long.iloc[-1] else -1.0
        adx, pdi, ndi = calculate_adx(df)
        if pd.notna(adx.iloc[-1]) and adx.iloc[-1] > 25: scores['ADX'] = 1.5 if pdi.iloc[-1] > ndi.iloc[-1] else -1.5
        else: scores['ADX'] = 0.0
        middle = calculate_bollinger_bands(df); scores['Bollinger'] = 1.0 if pd.notna(middle.iloc[-1]) and pd.notna(df['Close'].iloc[-1]) and df['Close'].iloc[-1] > middle.iloc[-1] else -1.0
        roc = calculate_roc(df).iloc[-1]; scores['ROC'] = 1.0 if pd.notna(roc) and roc > 0 else -1.0
        obv = calculate_obv(df); scores['OBV'] = 1.0 if pd.notna(obv.iloc[-1]) and pd.notna(obv.iloc[-2]) and obv.iloc[-1] > obv.iloc[-2] else -1.0
        cci = calculate_cci(df).iloc[-1]
        if pd.notna(cci):
            if cci > 100: scores['CCI'] = 1.5
            elif cci > 0: scores['CCI'] = 1.0
            elif cci < -100: scores['CCI'] = -1.5
            elif cci < 0: scores['CCI'] = -1.0
            else: scores['CCI'] = 0.0
        else: scores['CCI'] = 0.0
    except (IndexError, KeyError):
        return {key: 0.0 for key in INDICATOR_WEIGHTS.keys()}
    
    # Ensure all indicators have a score
    for key in INDICATOR_WEIGHTS.keys():
        scores.setdefault(key, 0.0)
    return scores

def analyze_signals(timeframe_dataframes):
    final_score, max_possible_score = 0.0, 0.0
    for tf_min, df in timeframe_dataframes.items():
        if df is None or len(df) < 50: continue
        indicator_scores = get_indicator_scores(df)
        tf_weight = TIMEFRAME_WEIGHTS.get(tf_min, 1.0)
        for indicator, score in indicator_scores.items():
            ind_weight = INDICATOR_WEIGHTS.get(indicator, 1.0)
            final_score += score * tf_weight * ind_weight
            max_possible_score += max(abs(score), 1.0) * tf_weight * ind_weight
    if max_possible_score == 0: return 'Neutral', 0
    normalized_score = (final_score / max_possible_score) * 100
    if normalized_score >= 70: signal_text = 'Very Strong Buy'
    elif normalized_score >= 15: signal_text = 'Strong Buy'
    elif normalized_score <= -70: signal_text = 'Very Strong Sell'
    elif normalized_score <= -15: signal_text = 'Strong Sell'
    else: signal_text = 'Neutral'
    return signal_text, normalized_score

# --- LIVE SCANNER & MAIN EXECUTION ---
def main_live_scanner(interval_minutes=5):
    try:
        with open('shares.txt', 'r') as f:
            stocks = [line.strip().upper() for line in f.readlines() if line.strip()]

        logger.info(f"Loaded {len(stocks)} stocks to track. Priming initial historical data...")
        stock_multi_data = prefetch_all(stocks)
        
        if not stock_multi_data:
            logger.error("Could not fetch initial data for any stock. Exiting.")
            return

        print(f"\n--- 📡 REAL-TIME SCANNER (Starting for {datetime.now(IST).date()}) 📡 ---")
        previous_scores = {}
        market_open = time(9, 15)
        market_close = time(15, 30)

        while True:
            now = datetime.now(IST)
            if not (market_open <= now.time() <= market_close):
                print(f"Market is closed. Waiting... Current time: {now.strftime('%H:%M:%S')}", end="\r")
                time_module.sleep(30)
                continue

            # Calculate next 5-min candle completion time
            next_run_minute = (now.minute // interval_minutes + 1) * interval_minutes
            if next_run_minute >= 60:
                next_run_time = now.replace(hour=now.hour + 1, minute=0, second=0, microsecond=0)
            else:
                next_run_time = now.replace(minute=next_run_minute, second=0, microsecond=0)
            
            wakeup_time = next_run_time + timedelta(seconds=5) # 5-second buffer
            sleep_duration = (wakeup_time - now).total_seconds()
            if sleep_duration > 0:
                logger.info(f"Next scan at {wakeup_time.strftime('%H:%M:%S')}. Sleeping for {sleep_duration:.1f} seconds.")
                time_module.sleep(sleep_duration)

            scan_time = datetime.now(IST)
            logger.info(f"Scan triggered at {scan_time.strftime('%H:%M:%S')}. Refreshing data...")
            stock_multi_data = prefetch_all(list(stock_multi_data.keys()))
            logger.info("Data refresh complete. Analyzing signals...")
            
            signals_this_interval = []
            current_scores = {}

            for symbol, timeframe_data in stock_multi_data.items():
                clean_symbol = symbol.replace('-EQ', '')
                time_point_aware = datetime.now(IST)
                filtered_timeframes = {tf: df[df.index <= time_point_aware] for tf, df in timeframe_data.items()}
                if len(filtered_timeframes) < 4: continue

                signal, score = analyze_signals(filtered_timeframes)
                current_scores[clean_symbol] = score

                df_60min = filtered_timeframes.get(60)
                if df_60min is None or len(df_60min) < 200: continue
                _, ma_long_series = calculate_moving_averages(df_60min)
                if ma_long_series.empty or pd.isna(ma_long_series.iloc[-1]): continue
                
                long_term_trend = 'bullish' if df_60min['Close'].iloc[-1] > ma_long_series.iloc[-1] else 'bearish'

                if 'Strong' in signal:
                    change = score - previous_scores.get(clean_symbol, 0.0) if clean_symbol in previous_scores else 'NA'
                    if (long_term_trend == 'bullish' and 'Buy' in signal) or \
                       (long_term_trend == 'bearish' and 'Sell' in signal):
                        signals_this_interval.append({'symbol': clean_symbol, 'signal': signal, 'score': score, 'trend': long_term_trend, 'change': change})

            print("\n" + "="*92)
            print(f"| SIGNALS AT {scan_time.strftime('%H:%M')} IST".center(90) + " |")
            print("="*92)
            if not signals_this_interval:
                print("| No strong, trend-aligned signals found at this time.".center(90) + " |")
            else:
                print(f"| {'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Action':<19} |")
                print("-"*92)
                signals_this_interval.sort(key=lambda x: abs(x['score']), reverse=True)
                for r in signals_this_interval:
                    change_val = r['change']
                    if isinstance(change_val, (int, float)):
                        sign = '+' if change_val >= 0 else ''
                        color = Colors.GREEN if change_val >= 0 else Colors.RED
                        change_str = f"{color}{sign}{change_val:>.2f}{Colors.RESET}"
                        padding = 19
                    else:
                        change_str, padding = "NA", 10
                    
                    if 'Buy' in r['signal']:
                        colored_signal, action = f"{Colors.GREEN}{r['signal']:<18}{Colors.RESET}", f"{Colors.GREEN}Consider Long{Colors.RESET}"
                    else:
                        colored_signal, action = f"{Colors.RED}{r['signal']:<18}{Colors.RESET}", f"{Colors.RED}Consider Short{Colors.RESET}"
                    
                    print(f"| {r['symbol']:<15} | {colored_signal} | {r['score']:>7.2f} | {change_str:>{padding}} | {r['trend']:<10} | {action:<29} |")
            print("="*92)
            previous_scores = current_scores.copy()

    except KeyboardInterrupt:
        logger.info("Scanner stopped by user.")
    except Exception as e:
        logger.error(f"A critical error occurred: {e}", exc_info=True)
    finally:
        logger.info("Disconnecting TrueData sessions...")
        for sess in tdhist_pool:
            try: sess.disconnect()
            except Exception: pass
        logger.info("Shutdown complete.")

if __name__ == "__main__":
    main_live_scanner(interval_minutes=5)