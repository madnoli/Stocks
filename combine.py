# v21_truedata_scanner.py
# Data source: TrueData TD_hist (import fix).
# Timeframes: 5, 15, 30, 60 minutes.
# Features: sector column, enhanced indicators, composite scoring, momentum "Change",
# trend alignment via 60m 200MA, table output.
# Global rate limit: exactly 10 TD calls/sec across all threads.

import os
import time as time_module
from datetime import datetime, timedelta, time
import threading
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

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

# ---- Credentials (from attached script pattern) ----
# Prefer environment variables. Replace defaults with actual from attached script if needed.
TDUSERNAME = os.getenv("TDUSERNAME", "Trial106")   # replace with actual
TDPASSWORD = os.getenv("TDPASSWORD", "raj106")     # replace with actual
tdhist = TDhist(TDUSERNAME, TDPASSWORD, log_level=logging.WARNING)  # single shared client [attached_file:1]

# ---- Colors ----
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

# ---- Timeframes ----
TIMEFRAME_MAP = {
    5: "5 min",
    15: "15 min",
    30: "30 min",
    60: "60 mins",
}  # [attached_file:1]

TIMEFRAME_WEIGHTS = {
    5: 1.0,
    15: 1.5,
    30: 2.0,
    60: 2.5,
}  # [attached_file:1]

# ---- Enhanced indicator weights (attached) ----
ENHANCED_INDICATOR_WEIGHTS = {
    'RSI': 1.3, 'MACD': 1.6, 'Stochastic': 1.0, 'MA': 1.8,
    'EMA': 1.7, 'VWAP': 1.5, 'ADX': 1.5, 'Bollinger': 1.4,
    'ROC': 1.2, 'OBV': 1.6, 'CCI': 1.1, 'WWL': 1.0,
    'ATR': 1.4, 'VolumeSurge': 2.0, 'Momentum': 1.9
}  # [attached_file:1]

# ---- Sector maps and stock universe (subset; extend as needed) ----
SECTOR_STOCKS = {
    "Technology": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "CYIENT", "KPITTECH", "TATAELXSI", "OFSS", "KAYNES"],
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "PNB", "BANKBARODA", "CANBK", "IDFCFIRSTB", "INDUSINDBK", "AUBANK", "FEDERALBNK"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA", "TORNTPHARM", "GLENMARK", "ALKEM", "LAURUSLABS", "BIOCON", "ZYDUSLIFE", "MANKIND", "SYNGENE"],
    "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR", "EICHERMOT", "ASHOKLEY", "BOSCHLTD", "TIINDIA", "MOTHERSON"],
    "Metal": ["TATASTEEL", "JSWSTEEL", "SAIL", "JINDALSTEL", "HINDALCO", "NMDC"],
    "Energy": ["RELIANCE", "NTPC", "BPCL", "IOC", "ONGC", "GAIL", "HINDPETRO", "ADANIGREEN", "ADANIENSOL", "JSWENERGY", "COALINDIA", "TATAPOWER", "POWERGRID"],
    "Consumer": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR", "MARICO", "COLPAL", "UPL", "VBL"],
    "Realty": ["DLF", "LODHA", "PRESTIGE", "GODREJPROP", "OBEROIRLTY", "PHOENIXLTD", "NCC", "NBCC"],
    "PSU Bank": ["SBIN", "PNB", "BANKBARODA", "CANBK", "UNIONBANK", "BANKINDIA"],
    "PSE": ["BEL", "BHEL", "NHPC", "GAIL", "IOC", "NTPC", "POWERGRID", "HINDPETRO", "OIL", "RECLTD", "ONGC", "NMDC", "BPCL", "HAL", "RVNL", "PFC", "COALINDIA", "IRCTC", "IRFC"],
    "Commodities": ["AMBUJACEM", "APLAPOLLO", "ULTRACEMCO", "SHREECEM", "JSWSTEEL", "HINDALCO", "IOC", "NTPC", "HINDPETRO", "OIL", "VEDL", "UPL", "BPCL", "JSWENERGY", "GRASIM", "RELIANCE", "TATAPOWER", "COALINDIA"],
    "Consumer Durables": ["TITAN", "DIXON", "HAVELLS", "CROMPTON", "POLYCAB", "EXIDEIND", "KAYNES", "VOLTAS", "PGEL", "BLUESTARCO"],
    "Healthcare": ["SUNPHARMA", "DIVISLAB", "CIPLA", "TORNTPHARM", "MAXHEALTH", "APOLLOHOSP", "DRREDDY", "MANKIND", "ZYDUSLIFE", "LUPIN", "FORTIS", "ALKEM", "AUROPHARMA", "GLENMARK", "BIOCON", "LAURUSLABS", "SYNGENE", "GRANULES"],
}  # [attached_file:1]

SYMBOL_TO_SECTOR = {}
for sector, symbols in SECTOR_STOCKS.items():
    for s in symbols:
        SYMBOL_TO_SECTOR.setdefault(s.upper(), sector)

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

bucket = TokenBucket(rate_per_sec=10, capacity=10)  # [attached_file:1]

# ---- Normalize fetched data like attachment ----
def normalize_truedata_df(df):
    try:
        if df is None or len(df) == 0:
            return None
        dfc = df.copy()
        cmap = {}
        for c in dfc.columns:
            lc = c.lower()
            if 'date' in lc or 'time' in lc:
                cmap[c] = 'Date'
            elif 'open' in lc:
                cmap[c] = 'Open'
            elif 'high' in lc:
                cmap[c] = 'High'
            elif 'low' in lc:
                cmap[c] = 'Low'
            elif 'close' in lc:
                cmap[c] = 'Close'
            elif 'vol' in lc:
                cmap[c] = 'Volume'
        dfc = dfc.rename(columns=cmap)
        for col in ['Date','Open','High','Low','Close']:
            if col not in dfc.columns:
                return None
        if 'Volume' not in dfc.columns:
            dfc['Volume'] = 1000  # default fallback [attached_file:1]
        dfc['Date'] = pd.to_datetime(dfc['Date'], errors='coerce')
        dfc.set_index('Date', inplace=True)
        for col in ['Open','High','Low','Close','Volume']:
            dfc[col] = pd.to_numeric(dfc[col], errors='coerce')
        dfc = dfc.dropna().sort_index()
        return dfc if len(dfc) >= 20 else None
    except Exception as e:
        logger.error(f"Normalize error: {e}")
        return None

# ---- Optional filter (gap-down), from attachment logic ----
def check_gap_down(dfi):
    try:
        if dfi is None or len(dfi) < 2:
            return False
        current_open = dfi['Open'].iloc[-1]
        prev_close = dfi['Close'].iloc[-2]
        if pd.isna(current_open) or pd.isna(prev_close) or prev_close == 0:
            return False
        gap_pct = (current_open - prev_close) / prev_close * 100
        return gap_pct < -1.0
    except Exception:
        return False

# ---- TrueData fetch with limiter ----
def td_fetch(symbol, timeframe_minutes):
    barsize = TIMEFRAME_MAP.get(timeframe_minutes)
    if not barsize:
        return None
    # durations aligned with attached script
    if timeframe_minutes in (5, 15):
        duration = "10 D"
    elif timeframe_minutes == 30:
        duration = "20 D"
    elif timeframe_minutes == 60:
        duration = "60 D"
    else:
        duration = "10 D"

    try:
        bucket.acquire()  # enforce 10 rps
        if hasattr(tdhist, "get_historic_data"):
            raw = tdhist.get_historic_data(symbol, duration=duration, bar_size=barsize)
        else:
            raw = tdhist.get_hist_data(symbol, duration=duration, bar_size=barsize)
        if raw is None or len(raw) == 0:
            return None
        return normalize_truedata_df(raw)
    except Exception as e:
        logger.error(f"TD fetch error {symbol} {barsize}: {e}")
        return None

# ---- Enhanced indicators (attached) ----
def enhanced_indicators(df):
    ind = {}
    try:
        if df is None or len(df) < 20:
            return ind
        close = df['Close']; high = df['High']; low = df['Low']; vol = df['Volume']

        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        ind['RSI'] = 100 - (100 / (1 + rs))

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        ind['MACD'] = macd_line - signal_line

        low14 = low.rolling(14).min()
        high14 = high.rolling(14).max()
        ind['Stochastic'] = 100 * (close - low14) / (high14 - low14)

        ind['MA'] = close.rolling(20).mean()
        ind['EMA'] = close.ewm(span=21, adjust=False).mean()

        high_diff = high.diff(); low_diff = low.diff()
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(14).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(14).mean() / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        ind['ADX'] = dx.rolling(14).mean()

        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + 2 * std20
        lower = ma20 - 2 * std20
        width = (upper - lower).replace(0, np.nan)
        ind['Bollinger'] = 100 * (close - ma20) / width

        ind['ROC'] = close.pct_change(12) * 100

        obv = (np.sign(close.diff()) * vol.fillna(0)).cumsum()
        ind['OBV'] = obv.pct_change(10) * 100

        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        ind['CCI'] = (tp - sma_tp) / (0.015 * mad)

        highest_high = high.rolling(14).max()
        lowest_low = low.rolling(14).min()
        wpr = -100 * (highest_high - close) / (highest_high - lowest_low)
        ind['WWL'] = wpr

        typical = (high + low + close) / 3
        vwap_num = (typical * vol).rolling(20).sum()
        vwap_den = vol.rolling(20).sum().replace(0, np.nan)
        ind['VWAP'] = vwap_num / vwap_den

        ind['ATR'] = atr

        avg_vol20 = vol.rolling(20).mean()
        vol_ratio = (vol / avg_vol20)
        ind['VolumeSurge'] = np.clip((vol_ratio - 0.5) * 40, 0, 100)

        price_mom = close.pct_change(10) * 100
        avg_vol10 = vol.rolling(10).mean().replace(0, np.nan)
        vol_mom = (vol / avg_vol10 - 1) * 100
        mom_score = price_mom * 0.7 + vol_mom * 0.3
        ind['Momentum'] = np.clip(mom_score * 1.5, -50, 50) + 50

        return ind
    except Exception as e:
        logger.error(f"Indicator error: {e}")
        return ind

def normalize_indicator_value(name, value, price=None):
    try:
        if name in ('RSI', 'Stochastic', 'ADX', 'Bollinger'):
            return float(np.clip(value, 0, 100))
        if name == 'MACD':
            return 50 + float(np.clip(value / 10, -25, 25))
        if name in ('ROC', 'OBV'):
            return 50 + float(np.clip(value / 2, -25, 25))
        if name == 'CCI':
            return 50 + float(np.clip(value / 4, -50, 50))
        if name == 'WWL':
            return float(np.clip(100 + value, 0, 100))
        if name in ('MA', 'EMA', 'VWAP') and price is not None and value not in (None, 0):
            diff_pct = (price - value) / value * 100
            if diff_pct > 2: return 75
            elif diff_pct > 0: return 60
            elif diff_pct > -2: return 50
            elif diff_pct > -5: return 40
            else: return 25
        if name in ('ATR',):
            return 50
        if name in ('VolumeSurge', 'Momentum'):
            return float(np.clip(value, 0, 100))
        return 50
    except Exception:
        return 50

def compute_timeframe_score(df, tf_minutes):
    if df is None or len(df) < 50:
        return None
    inds = enhanced_indicators(df)
    if not inds:
        return None
    price = df['Close'].iloc[-1]
    score_sum = 0.0
    weight_sum = 0.0
    for name, w in ENHANCED_INDICATOR_WEIGHTS.items():
        series = inds.get(name)
        if series is None or len(series) == 0 or pd.isna(series.iloc[-1]):
            continue
        latest = series.iloc[-1]
        norm = normalize_indicator_value(name, latest, price=price)
        score_sum += norm * w
        weight_sum += w
    if weight_sum == 0:
        return None
    normalized_0_100 = score_sum / weight_sum
    symmetric = (normalized_0_100 - 50) * 2
    return symmetric * TIMEFRAME_WEIGHTS.get(tf_minutes, 1.0)

def analyze_signals_multitf(timeframe_dataframes):
    if not timeframe_dataframes:
        return 'Neutral', 0.0
    final_sum = 0.0
    weight_den = 0.0
    for tf, df in timeframe_dataframes.items():
        s = compute_timeframe_score(df, tf)
        if s is None:
            continue
        final_sum += s
        weight_den += TIMEFRAME_WEIGHTS.get(tf, 1.0)
    if weight_den == 0:
        return 'Neutral', 0.0
    normalized = final_sum / weight_den
    if normalized >= 70:
        sig = 'Very Strong Buy'
    elif normalized >= 20:
        sig = 'Strong Buy'
    elif normalized <= -70:
        sig = 'Very Strong Sell'
    elif normalized <= -20:
        sig = 'Strong Sell'
    else:
        sig = 'Neutral'
    return sig, normalized

# ---- Cache and Prefetch ----
def fetch_data_for_stock(symbol, analysis_date):
    CACHE_DIR = "data_cache_td"
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{symbol}_{analysis_date.strftime('%Y-%m-%d')}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                logger.info(f"CACHE HIT: {symbol}")
                return symbol, pickle.load(f)
        except Exception:
            pass
    logger.info(f"CACHE MISS: {symbol}")
    timeframe_data = {}
    gap_down_flag = False
    for tf in [5, 15, 30, 60]:
        df = td_fetch(symbol, tf)
        if df is not None and not df.empty:
            timeframe_data[tf] = df
            if tf in (5, 15) and check_gap_down(df):
                gap_down_flag = True
    if gap_down_flag:
        logger.info(f"{symbol} filtered due to gap-down")
        return symbol, None
    if len(timeframe_data) >= 3:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(timeframe_data, f)
        except Exception:
            pass
        return symbol, timeframe_data
    return symbol, None

# ---- Interval generation ----
def generate_time_intervals(target_date, start_time="09:15", end_time="15:25", interval_minutes=5):
    intervals = []
    start = datetime.strptime(start_time, "%H:%M").time()
    end = datetime.strptime(end_time, "%H:%M").time()
    current_time = target_date.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
    end_datetime = target_date.replace(hour=end.hour, minute=end.minute)
    while current_time <= end_datetime:
        intervals.append(current_time)
        current_time += timedelta(minutes=interval_minutes)
    return intervals

# ---- Main scanner with table output ----
def main_interval_scanner(run_date_str=None, interval_minutes=5, max_workers=10, sectors_to_use=None, continuous=True, poll_seconds=0):
    """
    continuous=True -> loop across the session time grid (5-min boundaries).
    poll_seconds>0  -> optional intra-bar refresh seconds (for 5m TF live feel).
    """
    analysis_date = datetime.strptime(run_date_str, "%Y-%m-%d") if run_date_str else datetime.now()
    logger.info(f"Running interval scanner for: {analysis_date.date()}")

    if sectors_to_use:
        stocks = sorted({s for sec in sectors_to_use for s in SECTOR_STOCKS.get(sec, [])})
    else:
        stocks = sorted({s for arr in SECTOR_STOCKS.values() for s in arr})

    if not stocks:
        print("No symbols found from sector mapping.")
        return

    # Prefetch historical data
    stock_multi_data = {}
    with ThreadPoolExecutor(max_workers=max(2, max_workers)) as executor:
        tasks = [executor.submit(fetch_data_for_stock, s, analysis_date) for s in stocks]
        for fut in as_completed(tasks):
            symbol, data = fut.result()
            if data:
                stock_multi_data[symbol] = data

    if not stock_multi_data:
        print("No stocks with sufficient data found.")
        return

    print(f"\n--- 📡 REAL-TIME SCANNER ({analysis_date.date()}) 📡 ---")
    ist = pytz.timezone('Asia/Kolkata')

    # Build session grid
    def next_5min_boundary(now_dt):
        minute = (now_dt.minute // 5) * 5
        base = now_dt.replace(minute=minute, second=0, microsecond=0)
        if base < now_dt:
            base = base + timedelta(minutes=5)
        return base

    session_starts = time(9, 15)
    session_ends = time(15, 25)

    previous_scores = {}

    def process_at(time_point):
        nonlocal previous_scores
        time_point_aware = ist.localize(time_point)
        signals_this_interval = []
        current_scores = {}

        for symbol, timeframe_data in stock_multi_data.items():
            # Optional: refresh just 5m for live polling
            if poll_seconds > 0:
                d5 = td_fetch(symbol, 5)
                if d5 is not None and not d5.empty:
                    timeframe_data[5] = d5

            filtered_timeframes = {}
            for tf, df in timeframe_data.items():
                dff = df[df.index <= time_point_aware]
                if not dff.empty:
                    filtered_timeframes[tf] = dff
            if len(filtered_timeframes) < 3:
                continue

            signal, score = analyze_signals_multitf(filtered_timeframes)
            current_scores[symbol] = score

            # long-term trend via 60m 200MA
            full_60 = timeframe_data.get(60)
            if full_60 is None or len(full_60) < 200:
                continue
            current_60 = full_60[full_60.index <= time_point_aware]
            if current_60.empty:
                continue
            ma_long = current_60['Close'].rolling(200).mean().iloc[-1]
            if pd.isna(ma_long):
                continue
            latest_close = current_60['Close'].iloc[-1]
            long_term_trend = 'bullish' if latest_close > ma_long else 'bearish'

            if 'Strong' in signal:
                change = 'NA'
                if symbol in previous_scores:
                    change = score - previous_scores[symbol]
                sector = SYMBOL_TO_SECTOR.get(symbol, 'NA')
                if long_term_trend == 'bullish' and 'Buy' in signal:
                    signals_this_interval.append({'symbol': symbol, 'sector': sector, 'signal': signal, 'score': score, 'trend': long_term_trend, 'change': change})
                elif long_term_trend == 'bearish' and 'Sell' in signal:
                    signals_this_interval.append({'symbol': symbol, 'sector': sector, 'signal': signal, 'score': score, 'trend': long_term_trend, 'change': change})

        # Render table
        print("\n" + "="*128)
        print(f"| SIGNALS AT {time_point.strftime('%H:%M')} IST".center(126) + " |")
        print("="*128)
        if not signals_this_interval:
            print("| No strong, trend-aligned signals found at this time.".center(126) + " |")
        else:
            print(f"| {'Stock':<12} | {'Sector':<16} | {'Signal':<18} | {'Score':>7} | {'Change':>10} | {'Trend':<8} | {'Action':<16} |")
            print("-"*128)
            signals_this_interval.sort(key=lambda x: x['score'], reverse=True)
            for res in signals_this_interval:
                sig_text = res['signal']
                ch = res['change']
                if isinstance(ch, (int, float)):
                    sign = '+' if ch > 0 else ''
                    color = Colors.GREEN if ch > 0 else Colors.RED
                    change_str = f"{color}{sign}{ch:>.2f}{Colors.RESET}"
                else:
                    change_str = "NA"
                if 'Buy' in sig_text:
                    colored_signal = f"{Colors.GREEN}{sig_text:<18}{Colors.RESET}"
                    action = f"{Colors.GREEN}Consider Long{Colors.RESET}"
                else:
                    colored_signal = f"{Colors.RED}{sig_text:<18}{Colors.RESET}"
                    action = f"{Colors.RED}Consider Short{Colors.RESET}"
                print(f"| {res['symbol']:<12} | {res['sector']:<16} | {colored_signal} | {res['score']:>7.2f} | {change_str:>10} | {res['trend']:<8} | {action:<16} |")
        print("="*128)

        previous_scores = current_scores.copy()

    # Run once over session grid or continuously
    if not continuous:
        # Non-continuous: build grid and iterate once (historical)
        intervals = []
        start_dt = analysis_date.replace(hour=session_starts.hour, minute=session_starts.minute, second=0, microsecond=0)
        end_dt = analysis_date.replace(hour=session_ends.hour, minute=session_ends.minute, second=0, microsecond=0)
        cur = start_dt
        while cur <= end_dt:
            intervals.append(cur)
            cur += timedelta(minutes=interval_minutes)
        for tp in intervals:
            process_at(tp)
    else:
        # Continuous: loop until end of session, processing each 5-min boundary
        tz = pytz.timezone('Asia/Kolkata')
        while True:
            now = datetime.now(tz)
            if now.time() > session_ends:
                break
            # wait to the next boundary
            next_boundary = next_5min_boundary(now)
            # If inside session, process at boundary
            if next_boundary.time() >= session_starts and next_boundary.time() <= session_ends:
                sleep_sec = (next_boundary - now).total_seconds()
                if sleep_sec > 0:
                    time_module.sleep(sleep_sec)
                process_at(next_boundary.replace(tzinfo=None))
            else:
                # sleep until session start
                target = analysis_date.replace(hour=session_starts.hour, minute=session_starts.minute, second=0, microsecond=0)
                target = tz.localize(target)
                sleep_sec = (target - now).total_seconds()
                if sleep_sec > 0:
                    time_module.sleep(min(sleep_sec, 60))
                else:
                    # past end
                    break

if __name__ == "__main__":
    # Example: full universe, continuous real-time bar-close scans.
    # For intra-bar live feel on 5m, set poll_seconds>0 (e.g., 10) to refresh 5m within each boundary.
    main_interval_scanner(
        run_date_str=None,            # default: today
        interval_minutes=5,           # scan frequency grid
        max_workers=10,               # thread pool for initial prefetch
        sectors_to_use=None,          # e.g., ["Banking","Technology"]
        continuous=True,              # continuous during market hours
        poll_seconds=0                # set >0 for intra-bar 5m polling
    )
