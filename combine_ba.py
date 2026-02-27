# v22_truedata_scanner_optimized.py
# Description:
# This version introduces a decoupled, asynchronous data fetching architecture.
# - A background "producer" thread pool continuously fetches historical data without blocking the main scanner.
# - A separate "refresher" thread keeps 5-minute data fresh during live scanning.
# - The main "consumer" (scanner) thread starts analysis immediately, using whatever data is available in the cache.
# This results in a much faster time-to-first-signal and a more responsive feel,
# while still strictly adhering to the 10 API calls/sec global rate limit.
# It now scans the full list of 200 NSE stocks.

import os
import time as time_module
from datetime import datetime, timedelta, time
import threading
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

import pandas as pd
import numpy as np
import pytz
from logzero import logger
import logging
import warnings

warnings.filterwarnings("ignore")

# ---- TrueData import (fix: TD_hist) ----
try:
    from truedata.history import TD_hist as TDhist  # alias to keep code consistent
except Exception as e:
    raise ImportError(
        "Failed to import TD_hist. Ensure 'truedata' package is installed. "
        "pip install truedata"
    )

# ---- Credentials ----
TDUSERNAME = os.getenv("TDUSERNAME", "Trial106")
TDPASSWORD = os.getenv("TDPASSWORD", "raj106")
tdhist = TDhist(TDUSERNAME, TDPASSWORD, log_level=logging.WARNING)

# ---- Colors ----
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

# ---- Timeframes and Weights ----
TIMEFRAME_MAP = {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 mins"}
TIMEFRAME_WEIGHTS = {5: 1.0, 15: 1.5, 30: 2.0, 60: 2.5}
ENHANCED_INDICATOR_WEIGHTS = {
    'RSI': 1.3, 'MACD': 1.6, 'Stochastic': 1.0, 'MA': 1.8, 'EMA': 1.7,
    'VWAP': 1.5, 'ADX': 1.5, 'Bollinger': 1.4, 'ROC': 1.2, 'OBV': 1.6,
    'CCI': 1.1, 'WWL': 1.0, 'ATR': 1.4, 'VolumeSurge': 2.0, 'Momentum': 1.9
}

# ---- Expanded Stock Universe (200 Symbols) ----
ALL_NSE_STOCKS = [
    "CHOLAFIN", "GMRAIRPORT", "CYIENT", "HFCL", "AMBER", "KOTAKBANK", "PERSISTENT", "NHPC", "LT",
    "PAGEIND", "M&M", "RVNL", "SUPREMEIND", "BHARATFORG", "TATAPOWER", "KEI", "MARUTI", "POLYCAB",
    "PRESTIGE", "MOTHERSON", "OFSS", "NCC", "EICHERMOT", "BLUESTARCO", "BHARTIARTL", "PHOENIXLTD",
    "NBCC", "MUTHOOTFIN", "LTF", "MANAPPURAM", "TATASTEEL", "IIFL", "SUZLON", "AXISBANK", "VEDL",
    "UNOMINDA", "JSWENERGY", "TIINDIA", "CUMMINSIND", "CONCOR", "GRASIM", "COFORGE", "DLF", "UPL",
    "JSWSTEEL", "GAIL", "ASTRAL", "HAVELLS", "ONGC", "BOSCHLTD", "GODREJPROP", "NTPC",
    "ULTRACEMCO", "NYKAA", "HCLTECH", "UNITDSPR", "360ONE", "BEL", "BHEL", "TCS", "LODHA", "WIPRO",
    "SHREECEM", "DELHIVERY", "OIL", "DMART", "CAMS", "PPLPHARMA", "HAL", "ADANIPORTS", "SOLARINDS",
    "AMBUJACEM", "POLICYBZR", "SBIN", "TECHM", "KALYANKJIL", "KAYNES", "DRREDDY", "POWERGRID",
    "MAZDOCK", "DIXON", "DIVISLAB", "CIPLA", "IOC", "ADANIENT", "JINDALSTEL", "CROMPTON",
    "TVSMOTOR", "ICICIGI", "TITAN", "CANBK", "HDFCAMC", "SIEMENS", "EXIDEIND", "IRFC", "PETRONET",
    "HINDPETRO", "RECLTD", "BIOCON", "BAJAJ-AUTO", "LTIM", "DALBHARAT", "SUNPHARMA", "HEROMOTOCO",
    "HUDCO", "APOLLOHOSP", "HINDZINC", "ASHOKLEY", "RELIANCE", "IGL", "TATAELXSI", "MPHASIS",
    "IREDA", "LUPIN", "INDUSINDBK", "HINDALCO", "PFC", "TRENT", "PAYTM", "IRCTC", "COALINDIA",
    "ABB", "INFY", "OBEROIRLTY", "JUBLFOOD", "ICICIBANK", "BPCL", "ADANIGREEN", "IEX", "SRF",
    "CGPOWER", "ITC", "SAIL", "FEDERALBNK", "KFINTECH", "ALKEM", "TATAMOTORS", "JIOFIN", "BDL",
    "BAJAJFINSV", "HINDUNILVR", "INOXWIND", "INDIGO", "HDFCBANK", "LAURUSLABS", "TORNTPHARM",
    "TATATECH", "PNB", "ADANIENSOL", "VOLTAS", "NMDC", "IDFCFIRSTB", "LICI", "NATIONALUM",
    "BRITANNIA", "APLAPOLLO", "SBILIFE", "ZYDUSLIFE", "ICICIPRULI", "ABCAPITAL", "CDSL", "KPITTECH",
    "PIIND", "LICHSGFIN", "AUBANK", "SONACOMS", "TORNTPOWER", "HDFCLIFE", "SBICARD", "BANKINDIA",
    "COLPAL", "INDUSTOWER", "NUVAMA", "MARICO", "PNBHOUSING", "PGEL", "MANKIND", "BAJFINANCE",
    "NESTLEIND", "NAUKRI", "AUROPHARMA", "ASIANPAINT", "SHRIRAMFIN", "TATACONSUM", "ANGELONE",
    "MFSL", "DABUR", "TITAGARH", "GLENMARK", "FORTIS", "BSE", "MAXHEALTH", "MCX", "INDHOTEL",
    "VBL", "SYNGENE", "GODREJCP",
    # Note: Removed ETERNAL, SAMMAANCAP, PATANJALI as they may not be standard symbols
]

# ---- Sector Map (for reference, can be extended) ----
# Using a pre-defined map for sectors. Stocks not found here will be labeled 'Unknown'.
SECTOR_STOCKS = {
    "Technology": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "CYIENT", "KPITTECH", "TATAELXSI", "OFSS", "KAYNES", "TATATECH"],
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "PNB", "BANKBARODA", "CANBK", "IDFCFIRSTB", "INDUSINDBK", "AUBANK", "FEDERALBNK"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA", "TORNTPHARM", "GLENMARK", "ALKEM", "LAURUSLABS", "BIOCON", "ZYDUSLIFE", "MANKIND", "SYNGENE", "PPLPHARMA"],
    "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR", "EICHERMOT", "ASHOKLEY", "BOSCHLTD", "TIINDIA", "MOTHERSON", "UNOMINDA", "SONACOMS"],
    "Metal": ["TATASTEEL", "JSWSTEEL", "SAIL", "JINDALSTEL", "HINDALCO", "NMDC", "HINDZINC", "NATIONALUM"],
    "Energy": ["RELIANCE", "NTPC", "BPCL", "IOC", "ONGC", "GAIL", "HINDPETRO", "ADANIGREEN", "ADANIENSOL", "JSWENERGY", "COALINDIA", "TATAPOWER", "POWERGRID", "IEX", "TORNTPOWER", "IGL", "PETRONET"],
    "Consumer": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR", "MARICO", "COLPAL", "UPL", "VBL", "GODREJCP", "PAGEIND", "JUBLFOOD"],
    "Realty": ["DLF", "LODHA", "PRESTIGE", "GODREJPROP", "OBEROIRLTY", "PHOENIXLTD", "NCC", "NBCC"],
    "Finance": ["CHOLAFIN", "MUTHOOTFIN", "LTF", "MANAPPURAM", "IIFL", "HDFCAMC", "IRFC", "RECLTD", "PFC", "JIOFIN", "BAJAJFINSV", "LICI", "SBILIFE", "ICICIPRULI", "ABCAPITAL", "LICHSGFIN", "HDFCLIFE", "SBICARD", "PNBHOUSING", "BAJFINANCE", "SHRIRAMFIN", "ANGELONE", "MFSL", "ICICIGI"],
    "Infra": ["LT", "GMRAIRPORT", "ADANIPORTS", "RVNL", "CONCOR", "HUDCO", "INDUSTOWER"],
    "Chemicals": ["SRF", "PIIND", "DEEPAKNTR"],
    "PSU": ["BEL", "BHEL", "NHPC", "GAIL", "IOC", "NTPC", "POWERGRID", "HINDPETRO", "OIL", "RECLTD", "ONGC", "NMDC", "BPCL", "HAL", "RVNL", "PFC", "COALINDIA", "IRCTC", "IRFC", "MAZDOCK", "BDL", "SBIN", "PNB", "BANKBARODA", "CANBK", "UNIONBANK", "BANKINDIA"],
}
SYMBOL_TO_SECTOR = {s: sec for sec, symbols in SECTOR_STOCKS.items() for s in symbols}
for s in ALL_NSE_STOCKS:
    SYMBOL_TO_SECTOR.setdefault(s, "Unknown")


# ---- Global request rate limiter (exactly 10 calls/sec) ----
class TokenBucket:
    def __init__(self, rate_per_sec=10, capacity=10):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.lock = threading.Lock()
        self.last_refill = time_module.time()

    def acquire(self):
        while True:
            with self.lock:
                now = time_module.time()
                elapsed = now - self.last_refill
                refill = elapsed * self.rate
                if refill >= 1:
                    self.tokens = min(self.capacity, self.tokens + int(refill))
                    self.last_refill = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
            time_module.sleep(0.01)

bucket = TokenBucket(rate_per_sec=10, capacity=10)

# ---- Shared State for Asynchronous Operations ----
GLOBAL_DATA_CACHE = {}
CACHE_LOCK = threading.Lock()
FETCH_QUEUE = queue.Queue()
STOP_EVENT = threading.Event()

# ---- Data Normalization & Fetching (largely unchanged) ----
def normalize_truedata_df(df):
    try:
        if df is None or len(df) == 0: return None
        dfc = df.copy()
        cmap = {c: c.lower() for c in dfc.columns}
        dfc = dfc.rename(columns={
            k: 'Date' if 'date' in v or 'time' in v else
               'Open' if 'open' in v else
               'High' if 'high' in v else
               'Low' if 'low' in v else
               'Close' if 'close' in v else
               'Volume' if 'vol' in v else v
            for k, v in cmap.items()
        })
        if not all(col in dfc.columns for col in ['Date','Open','High','Low','Close']): return None
        if 'Volume' not in dfc.columns: dfc['Volume'] = 1000
        dfc['Date'] = pd.to_datetime(dfc['Date'], errors='coerce')
        dfc.set_index('Date', inplace=True)
        for col in ['Open','High','Low','Close','Volume']:
            dfc[col] = pd.to_numeric(dfc[col], errors='coerce')
        dfc = dfc.dropna().sort_index()
        return dfc if len(dfc) >= 200 else None # Increased requirement for 200MA
    except Exception as e:
        logger.error(f"Normalize error: {e}")
        return None

def td_fetch(symbol, timeframe_minutes):
    barsize = TIMEFRAME_MAP.get(timeframe_minutes)
    if not barsize: return None
    duration = "10 D" if timeframe_minutes in (5, 15) else "20 D" if timeframe_minutes == 30 else "60 D"
    try:
        bucket.acquire()
        method = getattr(tdhist, "get_historic_data", getattr(tdhist, "get_hist_data", None))
        if not method: return None
        raw = method(symbol, duration=duration, bar_size=barsize)
        return normalize_truedata_df(raw) if raw is not None and len(raw) > 0 else None
    except Exception as e:
        logger.error(f"TD fetch error {symbol} {barsize}: {e}")
        return None

# ---- Indicator Calculation (Optimized) ----
def enhanced_indicators(df):
    ind = {}
    try:
        if df is None or len(df) < 20: return ind
        close, high, low, vol = df['Close'], df['High'], df['Low'], df['Volume']
        
        # Reusable calculations
        ma20 = close.rolling(20).mean()
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        ind['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        ind['MACD'] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

        # Other indicators (logic remains the same as original)
        low14, high14 = low.rolling(14).min(), high.rolling(14).max()
        ind['Stochastic'] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
        ind['MA'] = ma20
        ind['EMA'] = close.ewm(span=21, adjust=False).mean()
        tr = pd.concat([(high - low), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1/14, adjust=False).mean() # Using ewm for smoother ATR
        ind['ATR'] = atr
        std20 = close.rolling(20).std()
        upper, lower = ma20 + 2 * std20, ma20 - 2 * std20
        ind['Bollinger'] = 100 * (close - ma20) / (upper - lower).replace(0, np.nan)
        ind['ROC'] = close.pct_change(12) * 100
        ind['OBV'] = (np.sign(close.diff()) * vol.fillna(0)).cumsum().pct_change(10) * 100
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        ind['CCI'] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
        wpr_high, wpr_low = high.rolling(14).max(), low.rolling(14).min()
        ind['WWL'] = -100 * (wpr_high - close) / (wpr_high - wpr_low).replace(0, np.nan)
        vwap_den = vol.rolling(20).sum().replace(0, np.nan)
        ind['VWAP'] = (tp * vol).rolling(20).sum() / vwap_den
        avg_vol20 = vol.rolling(20).mean().replace(0, np.nan)
        ind['VolumeSurge'] = np.clip(((vol / avg_vol20) - 0.5) * 40, 0, 100)
        price_mom = close.pct_change(10) * 100
        vol_mom = ((vol / vol.rolling(10).mean().replace(0, np.nan)) - 1) * 100
        ind['Momentum'] = np.clip((price_mom * 0.7 + vol_mom * 0.3) * 1.5, -50, 50) + 50
        
        return ind
    except Exception as e:
        logger.error(f"Indicator error: {e}")
        return ind

# ---- Scoring and Analysis (Unchanged) ----
def normalize_indicator_value(name, value, price=None):
    try:
        if pd.isna(value): return 50
        if name in ('RSI', 'Stochastic', 'Bollinger', 'VolumeSurge', 'Momentum'):
            return float(np.clip(value, 0, 100))
        if name == 'MACD': return 50 + float(np.clip(value * 2, -40, 40))
        if name in ('ROC', 'OBV'): return 50 + float(np.clip(value, -40, 40))
        if name == 'CCI': return 50 + float(np.clip(value / 4, -50, 50))
        if name == 'WWL': return float(np.clip(100 + value, 0, 100))
        if name in ('MA', 'EMA', 'VWAP') and price is not None and value not in (None, 0):
            diff_pct = (price - value) / value * 100
            return 80 if diff_pct > 2 else 65 if diff_pct > 0 else 40 if diff_pct < -2 else 20
        return 50
    except Exception: return 50

def compute_timeframe_score(df, tf_minutes):
    if df is None or len(df) < 50: return None
    inds = enhanced_indicators(df)
    if not inds: return None
    price = df['Close'].iloc[-1]
    score_sum, weight_sum = 0.0, 0.0
    for name, w in ENHANCED_INDICATOR_WEIGHTS.items():
        series = inds.get(name)
        if series is None or series.empty or pd.isna(series.iloc[-1]): continue
        norm = normalize_indicator_value(name, series.iloc[-1], price=price)
        score_sum += norm * w
        weight_sum += w
    if weight_sum == 0: return None
    return ((score_sum / weight_sum) - 50) * 2 * TIMEFRAME_WEIGHTS.get(tf_minutes, 1.0)

def analyze_signals_multitf(timeframe_data):
    if not timeframe_data: return 'Neutral', 0.0
    final_sum, weight_den = 0.0, 0.0
    for tf, df in timeframe_data.items():
        s = compute_timeframe_score(df, tf)
        if s is not None:
            final_sum += s
            weight_den += TIMEFRAME_WEIGHTS.get(tf, 1.0)
    if weight_den == 0: return 'Neutral', 0.0
    norm = final_sum / weight_den
    sig = 'Very Strong Buy' if norm >= 65 else 'Strong Buy' if norm >= 20 else \
          'Very Strong Sell' if norm <= -65 else 'Strong Sell' if norm <= -20 else 'Neutral'
    return sig, norm

# ---- Background Data Fetching/Refreshing ----
def fetch_and_cache_stock(symbol, timeframes, use_file_cache, analysis_date):
    """Fetches data for one stock, all TFs, and updates global cache."""
    CACHE_DIR = "data_cache_td_v22"
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{analysis_date.strftime('%Y-%m-%d')}.pkl")
    
    if use_file_cache and os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                if data:
                    with CACHE_LOCK:
                        GLOBAL_DATA_CACHE[symbol] = data
                    return symbol, True
        except Exception: pass

    timeframe_data = {}
    for tf in timeframes:
        if STOP_EVENT.is_set(): return symbol, False
        df = td_fetch(symbol, tf)
        if df is not None and not df.empty:
            timeframe_data[tf] = df
    
    if len(timeframe_data) >= 3:
        with CACHE_LOCK:
            GLOBAL_DATA_CACHE[symbol] = timeframe_data
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(timeframe_data, f)
        except Exception: pass
        return symbol, True
    return symbol, False


def background_worker(use_file_cache, analysis_date):
    """Worker thread that processes symbols from the queue."""
    while not STOP_EVENT.is_set():
        try:
            symbol, timeframes, is_refresh = FETCH_QUEUE.get(timeout=1)
            if is_refresh:
                df_5m = td_fetch(symbol, 5)
                if df_5m is not None:
                    with CACHE_LOCK:
                        if symbol in GLOBAL_DATA_CACHE:
                            GLOBAL_DATA_CACHE[symbol][5] = df_5m
            else:
                fetch_and_cache_stock(symbol, timeframes, use_file_cache, analysis_date)
            FETCH_QUEUE.task_done()
        except queue.Empty:
            continue

# ---- Main Scanner Logic ----
def main_interval_scanner(run_date_str=None, interval_minutes=5, max_workers=10, continuous=True, simulation_delay_seconds=0):
    analysis_date = datetime.strptime(run_date_str, "%Y-%m-%d") if run_date_str else datetime.now()
    logger.info(f"STARTING SCANNER FOR: {analysis_date.date()} | Symbols: {len(ALL_NSE_STOCKS)}")
    
    # Start background workers for data fetching
    for stock in ALL_NSE_STOCKS:
        FETCH_QUEUE.put((stock, [5, 15, 30, 60], False)) # False = initial fetch

    threads = []
    for _ in range(max_workers):
        thread = threading.Thread(target=background_worker, args=(not continuous, analysis_date), daemon=True)
        thread.start()
        threads.append(thread)
    
    print(f"\n--- 📡 REAL-TIME SCANNER ({analysis_date.date()}) | {len(ALL_NSE_STOCKS)} Stocks ---")
    print("--- Initial data fetch is running in the background. Results will appear as data arrives. ---")
    
    ist = pytz.timezone('Asia/Kolkata')
    session_starts, session_ends = time(9, 15), time(15, 25)
    previous_scores = {}

    def process_at(time_point):
        nonlocal previous_scores
        time_point_aware = ist.localize(time_point)
        signals = []
        current_scores = {}
        
        with CACHE_LOCK:
            # Create a copy to avoid holding lock during long analysis
            local_cache_copy = GLOBAL_DATA_CACHE.copy()

        for symbol, timeframe_data in local_cache_copy.items():
            filtered_tfs = {tf: df[df.index <= time_point_aware] for tf, df in timeframe_data.items()}
            filtered_tfs = {tf: df for tf, df in filtered_tfs.items() if not df.empty}
            if len(filtered_tfs) < 3: continue

            signal, score = analyze_signals_multitf(filtered_tfs)
            current_scores[symbol] = score
            
            # Trend alignment via 60m 200MA
            df60 = filtered_tfs.get(60)
            if df60 is None or len(df60) < 200: continue
            ma_long = df60['Close'].rolling(200).mean().iloc[-1]
            if pd.isna(ma_long): continue
            
            latest_close = df60['Close'].iloc[-1]
            long_term_trend = 'bullish' if latest_close > ma_long else 'bearish'

            if ('Buy' in signal and long_term_trend == 'bullish') or \
               ('Sell' in signal and long_term_trend == 'bearish'):
                change = score - previous_scores.get(symbol, score)
                sector = SYMBOL_TO_SECTOR.get(symbol, 'NA')
                signals.append({'symbol': symbol, 'sector': sector, 'signal': signal, 
                                'score': score, 'trend': long_term_trend, 'change': change})

        # Render table
        print("\n" + "="*120)
        print(f"| SIGNALS AT {time_point.strftime('%H:%M')} IST | Scanned: {len(local_cache_copy)}/{len(ALL_NSE_STOCKS)}".center(118) + "|")
        print("="*120)
        if not signals:
            print("| No strong, trend-aligned signals found at this time.".center(118) + "|")
        else:
            print(f"| {'Stock':<12} | {'Sector':<15} | {'Signal':<18} | {'Score':>7} | {'Change':>15} | {'Trend':<8} |")
            print("-"*120)
            signals.sort(key=lambda x: abs(x['score']), reverse=True)
            for res in signals:
                color = Colors.GREEN if 'Buy' in res['signal'] else Colors.RED
                ch_val = res['change']
                ch_color = Colors.GREEN if ch_val > 0.1 else Colors.RED if ch_val < -0.1 else Colors.YELLOW
                change_str = f"{ch_color}{ch_val:+.2f}{Colors.RESET}"
                print(f"| {res['symbol']:<12} | {res['sector']:<15} | {color}{res['signal']:<18}{Colors.RESET} | {res['score']:>7.2f} | {change_str:>25} | {res['trend']:<8} |")
        print("="*120)
        previous_scores = current_scores.copy()

    # --- Main Control Loop: Switches between Live and Backtest modes ---
    if continuous:
        # This is the live, real-time scanning mode for the current day.
        logger.info("Running in CONTINUOUS LIVE mode.")
        last_refresh_time = time_module.time()
        while True:
            now = datetime.now(ist)
            if now.time() > session_ends:
                logger.info("Market session ended. Shutting down.")
                break
            
            # Periodically queue 5-min data for a refresh to keep data live
            if time_module.time() - last_refresh_time > 60:
                logger.info("Queueing 5-min data refresh for all stocks.")
                for stock in ALL_NSE_STOCKS:
                    FETCH_QUEUE.put((stock, [5], True)) # True = is_refresh
                last_refresh_time = time_module.time()
    
            # Calculate the next 5-minute boundary to scan on
            next_boundary = (now + timedelta(minutes=interval_minutes)).replace(
                minute=(now.minute // interval_minutes) * interval_minutes, second=1, microsecond=0)
            if next_boundary <= now:
                next_boundary += timedelta(minutes=interval_minutes)
    
            # If before market hours, wait until the session starts
            if now.time() < session_starts:
                start_of_session = ist.localize(analysis_date.replace(hour=session_starts.hour, minute=session_starts.minute))
                sleep_duration = (start_of_session - now).total_seconds()
                if sleep_duration > 0:
                    logger.info(f"Before session. Sleeping for {sleep_duration:.0f}s until 09:15.")
                    time_module.sleep(min(sleep_duration, 60))
                continue
            
            # Wait until the next candle close
            sleep_duration = (next_boundary - now).total_seconds()
            if sleep_duration > 0:
                logger.info(f"Next scan at {next_boundary.strftime('%H:%M:%S')}. Sleeping for {sleep_duration:.0f}s.")
                time_module.sleep(sleep_duration)
            
            process_at(next_boundary.replace(tzinfo=None))
    else:
        # This is the backtesting mode for a specific historical date.
        logger.info(f"Running in BACKTEST mode for date: {analysis_date.date()}.")
        print("\n--- Waiting for initial data pre-fetch to complete for backtesting... ---")
        FETCH_QUEUE.join()  # Block until all initial data for the day is fetched
        print("--- Data pre-fetch complete. Starting backtest simulation. ---")

        # Generate all 5-minute intervals for the historical trading day
        intervals = []
        start_dt = analysis_date.replace(hour=session_starts.hour, minute=session_starts.minute)
        end_dt = analysis_date.replace(hour=session_ends.hour, minute=session_ends.minute)
        current_dt = start_dt
        while current_dt <= end_dt:
            intervals.append(current_dt)
            current_dt += timedelta(minutes=interval_minutes)
            
        # Process each historical interval sequentially to simulate the day
        for tp in intervals:
            process_at(tp)
            if simulation_delay_seconds > 0:
                # This delay makes the backtest output easier to follow, like a real-time feed.
                time_module.sleep(simulation_delay_seconds)

        logger.info("Backtest finished.")


    STOP_EVENT.set()
    for t in threads:
        t.join()
    logger.info("All worker threads terminated.")


if __name__ == "__main__":
    # ---- USAGE EXAMPLES ----

    # --- 1. To run in LIVE, REAL-TIME mode for today's market session: ---
    # Description: This is the default mode. It will track the market live,
    #              scan at every 5-minute candle close, and refresh 5m data periodically.
    # How to run: Set continuous=True and run_date_str=None (or just omit it).
    # main_interval_scanner(
    #     run_date_str=None,
    #     interval_minutes=5,
    #     max_workers=10,
    #     continuous=True
    # )

    # --- 2. To BACKTEST a specific day (e.g., yesterday) in a simulated feed: ---
    # Description: This mode simulates a live trading day using historical data.
    #              It will wait for all data to download, then step through the day's
    #              5-minute candles from 9:15 to 15:25 with a short delay to mimic real-time.
    # How to run: Set continuous=False and provide the date string in "YYYY-MM-DD" format.
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    main_interval_scanner(
        run_date_str=yesterday,
        interval_minutes=5,
        max_workers=10,
        continuous=False,
        simulation_delay_seconds=2 # Delay in seconds between each interval print
    )

