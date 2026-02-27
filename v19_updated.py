# Final Trading Signal Script - v20 (MOMENTUM TRACKING) - TrueData Live Market
# Keeps original scoring, tables, momentum "Change" column, and 60m 200MA trend alignment.
# Swaps SmartApi for TrueData multi-session fetching with rate limiting and IST tz-awareness.

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from logzero import logger
import logging
import threading
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm

# TrueData
from truedata.history import TD_hist
from config import username as TDUSERNAME, password as TDPASSWORD

# Terminal colors
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

try:
    GREEN = Colors.GREEN; RED = Colors.RED; RESET = Colors.RESET
except Exception:
    GREEN = RED = RESET = ""

# Silence noisy loggers
for noisy in ("truedata", "truedata.history", "truedata_ws", "websocket", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

IST = pytz.timezone("Asia/Kolkata")

# ---------------- User-configurable ----------------
TIMEFRAME_WEIGHTS = {5: 0.5, 10: 0.75, 15: 1.0, 30: 1.25, 60: 1.5, 1440: 2.0}
INDICATOR_WEIGHTS = {
    'RSI': 1.0, 'MACD': 1.2, 'Stochastic': 0.8, 'MA': 1.5,
    'ADX': 1.2, 'Bollinger': 1.0, 'ROC': 0.7, 'OBV': 1.3, 'CCI': 0.9
}

# Live settings
SCAN_SECONDS = int(os.getenv("SCAN_SECONDS", "60"))
MARKET_START = "09:15"  # IST
MARKET_END   = "15:30"  # IST
RUN_ONCE_AND_EXIT = False
ENABLE_WATCHLIST = False  # set True to print Moderate watchlist (|score| in [10,20))

# TrueData bars & durations
BAR_SIZE_MAP = {5: "5 min", 10: "10 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"}
DURATION_MAP = {5: "50 D", 10: "50 D", 15: "50 D", 30: "60 D", 60: "120 D", 1440: "400 D"}

# ---------- Token-bucket limiter ----------
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

# Global API call counter
api_calls_done = 0
api_calls_lock = threading.Lock()

# ---------------- TrueData sessions ----------------
def authenticate_session():
    return TD_hist(TDUSERNAME, TDPASSWORD, log_level=logging.CRITICAL)

def build_sessions():
    sess_count = int(os.getenv("TD_HIST_SESSIONS", "3"))  # tune 2–5
    pool = []
    for i in range(sess_count):
        try:
            pool.append(authenticate_session())
        except Exception as e:
            logger.error(f"Session {i} init failed: {e}")
    if not pool:
        raise SystemExit("Failed to initialize TrueData sessions.")
    per_sess_rate = 10.0 / len(pool)  # 10 rps aggregate target
    limiters = [TokenBucketLimiter(rate_per_sec=per_sess_rate, bucket_size=10) for _ in pool]
    return pool, limiters

tdhist_pool, sess_limiters = build_sessions()
logger.info("TrueData login successful.")

# ---------------- Indicators (original logic) ----------------
def calculate_rsi(df, period=14):
    if len(df) < period + 1: return pd.Series(dtype='float64')
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    if len(df) < slow + signal: return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    ema_fast = df['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_stochastic(df, period=14, smooth_d=3):
    if len(df) < period + smooth_d: return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    low_min = df['Low'].rolling(window=period).min()
    high_max = df['High'].rolling(window=period).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_moving_averages(df, short=50, long=200):
    if len(df) < long: return pd.Series(dtype='float64'), pd.Series(dtype='float64')
    return df['Close'].rolling(window=short).mean(), df['Close'].rolling(window=long).mean()

def calculate_adx(df, period=14):
    if len(df) < period * 2: return pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64')
    df_adx = df.copy()
    df_adx['H-L'] = df_adx['High'] - df_adx['Low']
    df_adx['H-C'] = abs(df_adx['High'] - df_adx['Close'].shift(1))
    df_adx['L-C'] = abs(df_adx['Low'] - df_adx['Close'].shift(1))
    df_adx['TR'] = df_adx[['H-L', 'H-C', 'L-C']].max(axis=1)
    df_adx['+DM'] = np.where(
        (df_adx['High'] - df_adx['High'].shift(1)) > (df_adx['Low'].shift(1) - df_adx['Low']),
        df_adx['High'] - df_adx['High'].shift(1),
        0
    )
    df_adx['-DM'] = np.where(
        (df_adx['Low'].shift(1) - df_adx['Low']) > (df_adx['High'] - df_adx['High'].shift(1)),
        df_adx['Low'].shift(1) - df_adx['Low'],
        0
    )
    atr = df_adx['TR'].ewm(com=period - 1, adjust=False).mean()
    pdi = (df_adx['+DM'].ewm(com=period - 1, adjust=False).mean() / atr) * 100
    ndi = (df_adx['-DM'].ewm(com=period - 1, adjust=False).mean() / atr) * 100
    adx = (abs(pdi - ndi) / (pdi + ndi)).ewm(com=period - 1, adjust=False).mean() * 100
    return adx, pdi, ndi

def calculate_bollinger_bands(df, period=20):
    if len(df) < period: return pd.Series(dtype='float64')
    return df['Close'].rolling(window=period).mean()

def calculate_roc(df, period=12):
    if len(df) < period + 1: return pd.Series(dtype='float64')
    return ((df['Close'] - df['Close'].shift(period)) / df['Close'].shift(period)) * 100

def calculate_obv(df):
    if len(df) < 2: return pd.Series(dtype='float64')
    return (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

def calculate_cci(df, period=20):
    if len(df) < period: return pd.Series(dtype='float64')
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma_tp) / (0.015 * mad)

# ---------------- Scoring & analysis ----------------
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

        macd, signal = calculate_macd(df)
        if pd.notna(macd.iloc[-1]) and pd.notna(signal.iloc[-1]):
            scores['MACD'] = 1.0 if macd.iloc[-1] > signal.iloc[-1] else -1.0
        else: scores['MACD'] = 0.0

        k, d = calculate_stochastic(df)
        if pd.notna(k.iloc[-1]) and pd.notna(d.iloc[-1]):
            if k.iloc[-1] > d.iloc[-1] and k.iloc[-1] < 80: scores['Stochastic'] = 1.0
            elif k.iloc[-1] < d.iloc[-1] and k.iloc[-1] > 20: scores['Stochastic'] = -1.0
            else: scores['Stochastic'] = 0.0
        else: scores['Stochastic'] = 0.0

        ma_short, ma_long = calculate_moving_averages(df)
        if pd.notna(ma_short.iloc[-1]) and pd.notna(ma_long.iloc[-1]):
            scores['MA'] = 1.0 if ma_short.iloc[-1] > ma_long.iloc[-1] else -1.0
        else: scores['MA'] = 0.0

        adx, pdi, ndi = calculate_adx(df)
        if pd.notna(adx.iloc[-1]) and adx.iloc[-1] > 25:
            scores['ADX'] = 1.5 if pdi.iloc[-1] > ndi.iloc[-1] else -1.5
        else: scores['ADX'] = 0.0

        middle = calculate_bollinger_bands(df)
        if pd.notna(middle.iloc[-1]) and pd.notna(df['Close'].iloc[-1]):
            scores['Bollinger'] = 1.0 if df['Close'].iloc[-1] > middle.iloc[-1] else -1.0
        else: scores['Bollinger'] = 0.0

        roc = calculate_roc(df).iloc[-1]
        if pd.notna(roc):
            scores['ROC'] = 1.0 if roc > 0 else -1.0
        else: scores['ROC'] = 0.0

        obv = calculate_obv(df)
        if len(obv) >= 2 and pd.notna(obv.iloc[-1]) and pd.notna(obv.iloc[-2]):
            scores['OBV'] = 1.0 if obv.iloc[-1] > obv.iloc[-2] else -1.0
        else: scores['OBV'] = 0.0

        cci = calculate_cci(df).iloc[-1]
        if pd.notna(cci):
            if cci > 100: scores['CCI'] = 1.5
            elif cci > 0: scores['CCI'] = 1.0
            elif cci < -100: scores['CCI'] = -1.5
            elif cci < 0: scores['CCI'] = -1.0
            else: scores['CCI'] = 0.0
        else: scores['CCI'] = 0.0

        return scores
    except IndexError:
        return {key: 0.0 for key in INDICATOR_WEIGHTS.keys()}

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
    if max_possible_score == 0: return 'Neutral', 0.0
    normalized_score = (final_score / max_possible_score) * 100.0
    if normalized_score >= 70: signal_text = 'Very Strong Buy'
    elif normalized_score >= 20: signal_text = 'Strong Buy'
    elif normalized_score <= -70: signal_text = 'Very Strong Sell'
    elif normalized_score <= -20: signal_text = 'Strong Sell'
    else: signal_text = 'Neutral'
    return signal_text, normalized_score

# ---------------- Fetch + normalize (TrueData) ----------------
def normalize_hist_df(df, symbol):
    if df is None or len(df) == 0:
        return None
    try:
        out = df.copy()
        out.rename(columns={c: str(c).lower() for c in out.columns}, inplace=True)
        rename_map = {}
        for src, tgt in (
            ("timestamp", "Date"), ("time", "Date"), ("datetime", "Date"), ("date", "Date"),
            ("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"),
            ("volume", "Volume"), ("vol", "Volume")
        ):
            if src in out.columns: rename_map[src] = tgt
        out.rename(columns=rename_map, inplace=True)

        if "Date" not in out.columns:
            if isinstance(out.index, pd.DatetimeIndex):
                out["Date"] = out.index
            else:
                return None

        if "Volume" not in out.columns:
            out["Volume"] = 0

        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"])

        # tz-aware IST
        if pd.api.types.is_datetime64tz_dtype(out["Date"]):
            out["Date"] = out["Date"].dt.tz_convert(IST)
        else:
            out["Date"] = out["Date"].dt.tz_localize(IST)

        for c in ["Open", "High", "Low", "Close", "Volume"]:
            out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")

        out = out.dropna(subset=["Open", "High", "Low", "Close"])
        out = out.sort_values("Date").set_index("Date")
        return out if len(out) >= 50 else None
    except Exception as e:
        logger.error(f"Normalize error {symbol}: {e}")
        return None

def pick_session(symbol_orig, timeframe_minutes):
    return (hash(symbol_orig) ^ timeframe_minutes) % len(tdhist_pool)

def fetch_one(symbol_orig, timeframe_minutes, limiter, hist):
    td_symbol = symbol_orig.replace('-EQ', '')
    bar_size = BAR_SIZE_MAP.get(timeframe_minutes)
    duration = DURATION_MAP.get(timeframe_minutes)
    if not bar_size or not duration:
        return symbol_orig, timeframe_minutes, None
    try:
        limiter.acquire()
        df_raw = hist.get_historic_data(td_symbol, duration=duration, bar_size=bar_size)
        df = normalize_hist_df(df_raw, td_symbol)
        global api_calls_done
        with api_calls_lock:
            api_calls_done += 1
        return symbol_orig, timeframe_minutes, df
    except Exception:
        return symbol_orig, timeframe_minutes, None

def prefetch_all(stocks, max_workers=64):
    tfs = [5, 10, 15, 30, 60, 1440]
    total_calls = len(stocks) * len(tfs)
    stock_multi_data = defaultdict(dict)

    global api_calls_done
    with api_calls_lock:
        api_calls_done = 0

    with tqdm(total=total_calls, desc="Prefetching Data", ncols=100) as api_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for s in stocks:
                for tf in tfs:
                    si = pick_session(s, tf)
                    futures.append(executor.submit(fetch_one, s, tf, sess_limiters[si], tdhist_pool[si]))
            for fut in as_completed(futures):
                symbol_orig, tf, df = fut.result()
                if df is not None:
                    stock_multi_data[symbol_orig][tf] = df
                api_bar.update(1)

    # Require at least 4 TFs (as original minimum depth)
    return {s: d for s, d in stock_multi_data.items() if len(d) >= 4}

# ---------------- Time helpers ----------------
def in_market_hours(now_ist: datetime) -> bool:
    start_t = datetime.strptime(MARKET_START, "%H:%M").time()
    end_t = datetime.strptime(MARKET_END, "%H:%M").time()
    return start_t <= now_ist.time() <= end_t

# ---------------- Live scanner ----------------
def live_market_scanner(interval_minutes=5, max_workers=64):
    logger.info("Starting TrueData live market scanner (original scoring/table layout)...")

    # Read stocks
    try:
        with open('shares.txt', 'r') as f:
            stocks = [line.strip().upper() for line in f.readlines() if line.strip()]
        logger.info(f"Loaded {len(stocks)} stocks from shares.txt.")
    except Exception as e:
        logger.error(f"Could not read shares.txt: {e}")
        return

    # Initial prefetch
    stock_multi_data = prefetch_all(stocks, max_workers=max_workers)
    if not stock_multi_data:
        print("No stocks with sufficient data found on initial load. Exiting.")
        return

    previous_scores = {}
    while True:
        now_ist = datetime.now(IST)

        if not RUN_ONCE_AND_EXIT and not in_market_hours(now_ist):
            print(f"Waiting for market hours ({MARKET_START}-{MARKET_END} IST). Current time: {now_ist.strftime('%H:%M:%S')}", end="\r")
            time_module.sleep(30)
            continue

        logger.info(f"[{now_ist.strftime('%H:%M:%S')}] Refreshing data for {len(stock_multi_data)} stocks...")
        stock_multi_data = prefetch_all(list(stock_multi_data.keys()), max_workers=max_workers)
        logger.info("Data refresh complete. Analyzing signals...")

        time_point_aware = now_ist.replace(second=0, microsecond=0)
        signals_this_scan = []
        moderates = []  # optional watchlist
        current_scores = {}

        for symbol, timeframe_data in stock_multi_data.items():
            clean_symbol = symbol.replace('-EQ', '')

            # Slice timeframes to current minute
            filtered_timeframes = {}
            for tf, df in timeframe_data.items():
                if df is None or df.empty: 
                    continue
                df_slice = df[df.index <= time_point_aware]
                if not df_slice.empty and len(df_slice) >= 50:
                    filtered_timeframes[tf] = df_slice
            if len(filtered_timeframes) < 4:
                continue

            signal, score = analyze_signals(filtered_timeframes)
            current_scores[clean_symbol] = score

            # Long-term trend filter (60m vs 200MA)
            full_df_60min = timeframe_data.get(60)
            if full_df_60min is None or len(full_df_60min) < 200:
                continue
            current_df_60min = full_df_60min[full_df_60min.index <= time_point_aware]
            if current_df_60min.empty:
                continue
            _, ma_long_series = calculate_moving_averages(full_df_60min)
            ma_long_value = ma_long_series.loc[current_df_60min.index[-1]]
            if pd.isna(ma_long_value):
                continue
            latest_close_60min = current_df_60min['Close'].iloc[-1]
            long_term_trend = 'bullish' if latest_close_60min > ma_long_value else 'bearish'

            if 'Strong' in signal:
                change = 'NA' if clean_symbol not in previous_scores else score - previous_scores.get(clean_symbol, 0.0)
                if long_term_trend == 'bullish' and 'Buy' in signal:
                    signals_this_scan.append({'symbol': clean_symbol, 'signal': signal, 'score': score, 'trend': long_term_trend, 'change': change})
                elif long_term_trend == 'bearish' and 'Sell' in signal:
                    signals_this_scan.append({'symbol': clean_symbol, 'signal': signal, 'score': score, 'trend': long_term_trend, 'change': change})
            elif ENABLE_WATCHLIST:
                if 10 <= abs(score) < 20:
                    change = 'NA' if clean_symbol not in previous_scores else score - previous_scores.get(clean_symbol, 0.0)
                    moderates.append({'symbol': clean_symbol, 'signal': signal, 'score': score, 'trend': long_term_trend, 'change': change})

        # Output tables
        print("\n" + "="*92)
        print(f"| SIGNALS AT {now_ist.strftime('%H:%M')} IST".center(90) + " |")
        print("="*92)

        if not signals_this_scan:
            print("| No strong, trend-aligned signals found at this time.".center(90) + " |")
        else:
            print(f"| {'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Action':<19} |")
            print("-"*92)
            signals_this_scan.sort(key=lambda x: x['score'], reverse=True)
            for result in signals_this_scan:
                signal_text = result['signal']
                change_val = result['change']
                if isinstance(change_val, (int, float, np.floating)):
                    sign = '+' if change_val > 0 else ''
                    color = GREEN if change_val > 0 else RED
                    change_str = f"{color}{sign}{change_val:>.2f}{RESET}"
                    pad = 19
                else:
                    change_str = "NA"
                    pad = 10
                if 'Buy' in signal_text:
                    colored_signal = f"{GREEN}{signal_text:<18}{RESET}"
                    action = f"{GREEN}Consider Long{RESET}"
                else:
                    colored_signal = f"{RED}{signal_text:<18}{RESET}"
                    action = f"{RED}Consider Short{RESET}"
                print(f"| {result['symbol']:<15} | {colored_signal} | {result['score']:>7.2f} | {change_str:>{pad}} | {result['trend']:<10} | {action:<29} |")

        print("="*92)

        # Optional watchlist
        if ENABLE_WATCHLIST:
            print(f"| {'WATCHLIST (Moderate, |score| 10–20)':<88} |")
            print("-"*92)
            if not moderates:
                print("| No moderate candidates at this time.".center(90) + " |")
            else:
                print(f"| {'Stock':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<10} | {'Action':<19} |")
                print("-"*92)
                moderates.sort(key=lambda x: abs(x['score']), reverse=True)
                for r in moderates:
                    change_val = r['change']
                    if isinstance(change_val, (int, float, np.floating)):
                        sign = '+' if change_val > 0 else ''
                        change_str = f"{sign}{change_val:>.2f}"
                    else:
                        change_str = "NA"
                    action = "Consider Long" if 'Buy' in r['signal'] else "Consider Short"
                    print(f"| {r['symbol']:<15} | {r['signal']:<18} | {r['score']:>7.2f} | {change_str:>10} | {r['trend']:<10} | {action:<19} |")
            print("="*92)

        previous_scores = current_scores.copy()
        if RUN_ONCE_AND_EXIT:
            logger.info("Scan complete. Exiting as RUN_ONCE_AND_EXIT is True.")
            break

        logger.info(f"Waiting for {SCAN_SECONDS} seconds until the next scan...")
        time_module.sleep(SCAN_SECONDS)

if __name__ == "__main__":
    try:
        live_market_scanner(interval_minutes=5, max_workers=64)
    except KeyboardInterrupt:
        print("\nScan interrupted by user. Shutting down.")
    finally:
        logger.info("Disconnecting TrueData sessions...")
        for sess in tdhist_pool:
            try: sess.disconnect()
            except Exception: pass
        logger.info("Shutdown complete.")
