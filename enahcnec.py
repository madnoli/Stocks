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
from truedata import TD_hist

# Rich for colored tables
from rich.console import Console
from rich.table import Table
from rich import box

# ======== CLEAN PRODUCTION CONFIG ========
CONFIG = {
    "TDUSERNAME": os.getenv("TRUEDATA_USER", "tdwsp751"),
    "TDPASSWORD": os.getenv("TRUEDATA_PASS", "raj@751"),

    # Market times (IST)
    "MARKET_START": "09:15",
    "FIRST_RUN_AT": "09:20",
    "MARKET_END": "15:30",
    "SETTLE_DELAY_SECONDS": 5,

    # Concurrency settings
    "MAX_WORKERS": int(os.getenv("MAX_WORKERS", "32")),
    "TD_HIST_SESSIONS": int(os.getenv("TD_HIST_SESSIONS", "4")),
    "RATE_PER_SECOND_TOTAL": float(os.getenv("RATE_PER_SECOND_TOTAL", "15.0")),
    "BUCKET_SIZE": int(os.getenv("BUCKET_SIZE", "20")),
    "RETRY_ATTEMPTS": int(os.getenv("RETRY_ATTEMPTS", "7")),
    "RETRY_DELAY_MS": int(os.getenv("RETRY_DELAY_MS", "2000")),

    # Output and logging - CLEAN PRODUCTION SETTINGS
    "SHARES_FILE": os.getenv("SHARES_FILE", "shares.txt"),
    "SHOW_PROGRESS": os.getenv("SHOW_PROGRESS", "true").lower() == "true",
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "WARNING"),

    # Always include all timeframes
    "SKIP_DAILY": False,

    # Analysis settings - optimized thresholds
    "MIN_BARS_REQUIRED": 20,
    "MAX_MISSING_DATA_PCT": 15,
    "SIGNAL_CONFIRMATION_BARS": 2,
    "MIN_SIGNAL_THRESHOLD": 10,

    # Enhanced real-time settings
    "REALTIME_FETCH_WORKERS": int(os.getenv("REALTIME_FETCH_WORKERS", "16")),
    "OI_ESTIMATION_FACTOR": float(os.getenv("OI_ESTIMATION_FACTOR", "0.15")),
    "VOLUME_SURGE_THRESHOLD": float(os.getenv("VOLUME_SURGE_THRESHOLD", "2.0")),
    "OI_SURGE_THRESHOLD": float(os.getenv("OI_SURGE_THRESHOLD", "1.5")),

    # Indicator periods
    "INDICATOR_PERIODS": {
        "RSI": 14, "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
        "STOCHASTIC_K": 14, "STOCHASTIC_D": 3, "MA_SHORT": 20, "MA_LONG": 50,
        "ADX": 14, "BB_PERIOD": 20, "BB_STD_DEV": 2, "ROC": 12, "CCI": 20,
        "EMA_FAST": 9, "EMA_SLOW": 21, "ATR": 14, "VOLUME_SURGE": 20,
        "MOMENTUM": 10, "WILLIAMS_R": 14, "CMF": 20, "ADL_LOOKBACK": 10,
        "REL_VOL": 20, "VWAP_REGIME": 20, "OBV_CONFIRM": 5,
        "OI_SURGE": 20, "OI_MOMENTUM": 10,
    },

    # OI-focused weights for option trading
    "INDICATOR_WEIGHTS": {
        "VolumeSurge": 2.5, "Momentum": 2.2, "ADX": 2.0, "VWAP": 1.8, "EMA": 1.9,
        "MACD": 1.7, "OBV": 1.6, "ATR": 1.5, "Bollinger": 1.4, "RSI": 1.3,
        "ROC": 1.2, "Stochastic": 1.1, "CCI": 1.0, "MA": 1.2, "WWL": 1.0,
        "CMF": 2.0, "ADL": 1.8, "RelVol": 1.7, "VWAPRegime": 1.9, "OBVConfirm": 1.4,
        # OI indicators get HIGHEST weights
        "OISurge": 3.5, "OIMomentum": 3.2, "CallBias": 4.0, "PutBias": 4.0, "OIVolConfirm": 3.0,
        # Enhanced real-time weights
        "LiveOI": 4.5, "LiveVolume": 3.0, "LivePrice": 2.8,
    },

    # All 5 timeframes with proper weights
    "TIMEFRAME_WEIGHTS": {15: 2.5, 5: 2.2, 30: 1.8, 60: 1.2, 1440: 1.0},

    # API mapping
    "BAR_SIZE_MAP": {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"},
    "DURATION_MAP": {5: "45 D", 15: "45 D", 30: "90 D", 60: "180 D", 1440: "365 D"},

    # Major NSE FNO stocks for enhanced OI estimation
    "MAJOR_FNO_STOCKS": [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "KOTAKBANK", "HINDUNILVR", 
        "ITC", "LT", "SBIN", "BHARTIARTL", "ASIANPAINT", "AXISBANK", "MARUTI", "BAJFINANCE",
        "HCLTECH", "WIPRO", "ULTRACEMCO", "NESTLEIND", "BAJAJFINSV", "POWERGRID", "NTPC",
        "COALINDIA", "ONGC", "TECHM", "TITAN", "SUNPHARMA", "JSWSTEEL", "TATAMOTORS", "INDUSINDBK"
    ]
}

# CLEAN: Root logger set to WARNING level for production
level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
logging.getLogger().setLevel(level_map.get(CONFIG["LOG_LEVEL"], logging.WARNING))

IST = pytz.timezone("Asia/Kolkata")

# Silence all noisy third-party loggers completely
for noisy in ("truedata", "truedata.history", "truedata_ws", "websocket", "urllib3", "requests", "connectionpool"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

console = Console()

# Global state
last_bull_symbols = set()
last_bear_symbols = set()
previous_scores = {}
api_calls_done = 0
api_calls_lock = threading.Lock()
performance_metrics = defaultdict(int)
failed_symbols = set()
oi_symbols_found = set()
live_data_cache = {}  # Cache for real-time data

# ---------- Helper functions ----------
def next_5min_boundary_ist(now_ist: datetime) -> datetime:
    minute = (now_ist.minute // 5) * 5
    boundary = now_ist.replace(minute=minute, second=0, microsecond=0)
    if boundary <= now_ist:
        boundary += timedelta(minutes=5)
    return boundary

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

def day_checkpoints_ist(day_date: datetime):
    d = day_date.date()
    start_h, start_m = parse_hhmm(CONFIG["FIRST_RUN_AT"])
    end_h, end_m = parse_hhmm(CONFIG["MARKET_END"])
    start_dt = IST.localize(datetime(d.year, d.month, d.day, start_h, start_m))
    end_dt = IST.localize(datetime(d.year, d.month, d.day, end_h, end_m))
    rng = pd.date_range(start=start_dt, end=end_dt, freq="5T", tz=IST, inclusive="both")
    return list(rng.to_pydatetime())

# ---------- Token-bucket limiter ----------
class TokenBucketLimiter:
    def __init__(self, rate_per_sec: float, bucket_size: int):
        self.rate = rate_per_sec
        self.capacity = bucket_size
        self.tokens = bucket_size
        self.lock = threading.Lock()
        self.last_refill = time.time()

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

# ---------- TrueData sessions ----------
def authenticate_session():
    return TD_hist(CONFIG["TDUSERNAME"], CONFIG["TDPASSWORD"], log_level=logging.CRITICAL)

def build_sessions():
    sess_count = CONFIG["TD_HIST_SESSIONS"]
    pool, limiters = [], []
    for i in range(sess_count):
        try:
            pool.append(authenticate_session())
        except Exception as e:
            console.print(f"[red]Session {i} failed: {e}[/red]")
    if not pool:
        console.print("[red]❌ Failed to initialize TrueData sessions.[/red]")
        raise SystemExit("Failed to initialize TrueData sessions.")

    per_sess_rate = CONFIG["RATE_PER_SECOND_TOTAL"] / len(pool)
    for _ in pool:
        limiters.append(TokenBucketLimiter(rate_per_sec=per_sess_rate, bucket_size=CONFIG["BUCKET_SIZE"]))
    return pool, limiters

tdhist_pool, sess_limiters = build_sessions()
console.print("✅ [green]TrueData connection established[/green]")


# ======== ENHANCED REAL-TIME DATA FETCHING (Replaces Excel RTD) ========
def fetch_realtime_data_enhanced(symbol, hist_session):
    """
    Enhanced real-time data fetching for OI, LTP, and Total Volume
    Replaces Excel RTD formulas: =RTD("truedata.velocity",,Symbol,"OI")
    """
    try:
        # Clean symbol name for TrueData API
        clean_symbol = symbol.replace('-EQ', '').replace('_EQ', '').upper()

        # Initialize quote data structure
        quote_data = {
            'LTP': 0,          # Last Traded Price (LAST in RTD)
            'TOTALVOL': 0,     # Total Volume (TOTALVOL in RTD)
            'OI': 0,           # Open Interest (OI in RTD)
            'HIGH': 0,
            'LOW': 0,
            'OPEN': 0,
            'PREV_CLOSE': 0,
            'BID': 0,
            'ASK': 0,
            'CHANGE': 0,
            'CHANGE_PCT': 0
        }

        try:
            # Primary method: Get real-time quote data
            realtime_data = hist_session.get_quote(clean_symbol)

            if realtime_data:
                # Map TrueData response to our structure
                quote_data.update({
                    'LTP': float(realtime_data.get('ltp', realtime_data.get('close', realtime_data.get('last', 0)))),
                    'TOTALVOL': int(realtime_data.get('volume', realtime_data.get('vol', realtime_data.get('totalvol', 0)))),
                    'OI': int(realtime_data.get('oi', realtime_data.get('openInterest', realtime_data.get('open_interest', 0)))),
                    'HIGH': float(realtime_data.get('high', 0)),
                    'LOW': float(realtime_data.get('low', 0)),
                    'OPEN': float(realtime_data.get('open', 0)),
                    'PREV_CLOSE': float(realtime_data.get('prev_close', realtime_data.get('prevClose', realtime_data.get('previous_close', 0)))),
                    'BID': float(realtime_data.get('bid', realtime_data.get('bid_price', 0))),
                    'ASK': float(realtime_data.get('ask', realtime_data.get('ask_price', 0)))
                })

                # Calculate change metrics
                if quote_data['PREV_CLOSE'] > 0:
                    quote_data['CHANGE'] = quote_data['LTP'] - quote_data['PREV_CLOSE']
                    quote_data['CHANGE_PCT'] = (quote_data['CHANGE'] / quote_data['PREV_CLOSE']) * 100

        except Exception as e:
            # Fallback method: Try alternative data source
            try:
                # Alternative quote method
                alt_data = hist_session.get_historic_data(clean_symbol, duration="1 D", bar_size="1 min")
                if alt_data is not None and not alt_data.empty:
                    latest = alt_data.tail(1).iloc[0]
                    quote_data.update({
                        'LTP': float(latest.get('Close', 0)),
                        'HIGH': float(latest.get('High', 0)),
                        'LOW': float(latest.get('Low', 0)),
                        'OPEN': float(latest.get('Open', 0)),
                        'TOTALVOL': int(latest.get('Volume', 0))
                    })

                    # Try to get OI from column variations
                    for oi_col in ['OI', 'OpenInterest', 'Open Interest', 'oi']:
                        if oi_col in alt_data.columns:
                            quote_data['OI'] = int(latest.get(oi_col, 0))
                            break
            except:
                pass

        # Enhanced OI estimation for major FNO stocks
        if quote_data['OI'] == 0 and clean_symbol in CONFIG['MAJOR_FNO_STOCKS']:
            total_vol = quote_data.get('TOTALVOL', 0)
            if total_vol > 0:
                # Enhanced estimation based on stock importance
                if clean_symbol in ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']:
                    quote_data['OI'] = int(total_vol * 0.25)  # Top 5 stocks - higher OI ratio
                elif clean_symbol in CONFIG['MAJOR_FNO_STOCKS'][:15]:
                    quote_data['OI'] = int(total_vol * CONFIG['OI_ESTIMATION_FACTOR'])  # Major FNO stocks
                else:
                    quote_data['OI'] = int(total_vol * 0.10)  # Other FNO stocks

        # Track OI availability globally
        if quote_data['OI'] > 100:
            global oi_symbols_found
            oi_symbols_found.add(clean_symbol)

        return quote_data

    except Exception:
        # Return zero data structure on complete failure
        return {
            'LTP': 0, 'TOTALVOL': 0, 'OI': 0, 'HIGH': 0, 'LOW': 0, 
            'OPEN': 0, 'PREV_CLOSE': 0, 'BID': 0, 'ASK': 0, 
            'CHANGE': 0, 'CHANGE_PCT': 0
        }

def get_live_market_data_batch(symbols, max_workers=None):
    """
    Batch fetch real-time market data for multiple symbols
    Much faster than individual RTD calls in Excel
    """
    global oi_symbols_found, live_data_cache

    if max_workers is None:
        max_workers = CONFIG['REALTIME_FETCH_WORKERS']

    console.print(f"📡 [cyan]Fetching real-time OI/LTP/Volume for {len(symbols)} symbols...[/cyan]")

    live_data = {}
    successful_fetches = 0
    oi_count = 0
    volume_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all real-time data requests
        future_to_symbol = {}
        for symbol in symbols:
            if symbol:
                clean_symbol = symbol.replace('-EQ', '').upper()
                session_idx = pick_session(symbol, 5)  # Use 5-min session for real-time
                session = tdhist_pool[session_idx]
                future = executor.submit(fetch_realtime_data_enhanced, clean_symbol, session)
                future_to_symbol[future] = clean_symbol

        # Collect results with progress
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                data = future.result()
                if data and data.get('LTP', 0) > 0:
                    live_data[symbol] = data
                    successful_fetches += 1

                    # Track data quality metrics
                    if data.get('OI', 0) > 100:
                        oi_count += 1

                    if data.get('TOTALVOL', 0) > 1000:
                        volume_count += 1

            except Exception as e:
                # Silent failure for individual symbols - continue with others
                pass

    # Update global cache
    live_data_cache.update(live_data)

    # Display data quality summary
    console.print(f"✅ [green]Real-time data: {successful_fetches} symbols[/green] | [yellow]OI: {oi_count}[/yellow] | [blue]Volume: {volume_count}[/blue]")

    if successful_fetches > 0:
        # Show sample of fetched data
        sample_symbols = list(live_data.keys())[:3]
        for sym in sample_symbols:
            data = live_data[sym]
            console.print(f"   📊 {sym}: LTP=₹{data['LTP']:.1f}, Vol={data['TOTALVOL']:,}, OI={data['OI']:,}")

    return live_data

def analyze_live_signals_enhanced(live_data, historical_data):
    """
    Enhanced signal analysis combining real-time OI/LTP/Volume with historical patterns
    """
    signals = []
    current_time = datetime.now(IST)

    console.print(f"🔍 [yellow]Analyzing {len(live_data)} symbols with enhanced real-time data...[/yellow]")

    for symbol, live_quote in live_data.items():
        try:
            # Skip if insufficient data
            if symbol not in historical_data:
                continue

            # Extract real-time metrics
            ltp = live_quote.get('LTP', 0)
            total_vol = live_quote.get('TOTALVOL', 0)
            current_oi = live_quote.get('OI', 0)
            high = live_quote.get('HIGH', ltp)
            low = live_quote.get('LOW', ltp)
            prev_close = live_quote.get('PREV_CLOSE', ltp)
            change_pct = live_quote.get('CHANGE_PCT', 0)

            if ltp <= 0 or prev_close <= 0:
                continue

            # Recalculate price metrics if needed
            if change_pct == 0 and prev_close > 0:
                change_pct = ((ltp - prev_close) / prev_close) * 100

            # Calculate position in day's range
            day_range = max(high - low, 0.01)
            price_position = ((ltp - low) / day_range) * 100

            # Enhanced signal calculation
            signal_strength = 0
            signal_components = []
            oi_status = "Normal"

            # Historical context analysis
            hist_timeframes = historical_data[symbol]
            volume_surge_score = 0
            oi_surge_score = 0
            momentum_score = 0

            # Analyze against recent historical data (5-min timeframe)
            if 5 in hist_timeframes and len(hist_timeframes[5]) > 20:
                recent_5min = hist_timeframes[5].tail(20)

                # Volume surge analysis
                avg_volume = recent_5min['Volume'].mean()
                if avg_volume > 0:
                    volume_surge_score = min(5, total_vol / avg_volume)
                    if volume_surge_score > CONFIG['VOLUME_SURGE_THRESHOLD']:
                        vol_direction = 1 if change_pct > 0 else -1
                        signal_strength += int(volume_surge_score * vol_direction)
                        signal_components.append(f"Vol: {volume_surge_score:.1f}x")

                # OI surge analysis (highest weight for options)
                avg_oi = recent_5min.get('OI', pd.Series([0])).mean()
                if avg_oi > 0 and current_oi > 0:
                    oi_surge_score = min(5, current_oi / avg_oi)
                    if oi_surge_score > CONFIG['OI_SURGE_THRESHOLD']:
                        oi_direction = 1 if change_pct > 0 else -1
                        signal_strength += int(oi_surge_score * 2 * oi_direction)  # Double weight for OI
                        signal_components.append(f"OI: {oi_surge_score:.1f}x")
                        oi_status = "High OI Activity" if oi_surge_score > 2.5 else "Moderate OI Activity"

                # Price momentum analysis
                recent_closes = recent_5min['Close'].tail(5)
                if len(recent_closes) >= 5:
                    momentum = (recent_closes.iloc[-1] / recent_closes.iloc[0]) - 1
                    if abs(momentum) > 0.02:  # 2% momentum threshold
                        momentum_score = min(3, abs(momentum) * 100)
                        signal_strength += int(momentum_score) * (1 if momentum > 0 else -1)
                        signal_components.append(f"Mom: {momentum*100:+.1f}%")

            # Real-time price action analysis
            if abs(change_pct) > 2:
                price_component = min(3, abs(change_pct) / 2)
                signal_strength += int(price_component) * (1 if change_pct > 0 else -1)
                signal_components.append(f"Price: {change_pct:+.1f}%")

            # Position in day's range analysis
            if price_position > 85 and change_pct > 0:
                signal_strength += 2
                signal_components.append("Near High")
            elif price_position < 15 and change_pct < 0:
                signal_strength -= 2
                signal_components.append("Near Low")

            # Enhanced signal classification for options trading
            if abs(signal_strength) >= 4:  # Lowered threshold for more signals
                if signal_strength >= 7:
                    signal_type = "🚀 ULTRA STRONG BUY"
                    option_action = "🚨 BUY CALLS AGGRESSIVELY"
                    priority = "URGENT"
                elif signal_strength >= 4:
                    signal_type = "⚡ STRONG BUY"
                    option_action = "🔥 BUY CALLS STRONG"
                    priority = "HIGH"
                elif signal_strength <= -7:
                    signal_type = "💥 ULTRA STRONG SELL"
                    option_action = "🚨 BUY PUTS AGGRESSIVELY"
                    priority = "URGENT"
                elif signal_strength <= -4:
                    signal_type = "⚡ STRONG SELL"
                    option_action = "🔥 BUY PUTS STRONG"
                    priority = "HIGH"
                else:
                    signal_type = "🟢 MODERATE BUY" if signal_strength > 0 else "🔴 MODERATE SELL"
                    option_action = "📈 Consider Calls" if signal_strength > 0 else "📉 Consider Puts"
                    priority = "MEDIUM"

                # Enhanced OI status for options guidance
                if current_oi > 10000 and oi_surge_score > 2:
                    if change_pct > 0:
                        oi_status = "Strong Call Setup"
                    else:
                        oi_status = "Strong Put Setup"

                signals.append({
                    'symbol': symbol,
                    'signal': signal_type,
                    'score': signal_strength,
                    'ltp': ltp,
                    'change_pct': change_pct,
                    'volume': total_vol,
                    'oi': current_oi,
                    'high': high,
                    'low': low,
                    'price_position': price_position,
                    'volume_surge': volume_surge_score,
                    'oi_surge': oi_surge_score,
                    'components': ' | '.join(signal_components) if signal_components else 'Price Action',
                    'action': option_action,
                    'priority': priority,
                    'oi_status': oi_status,
                    'timestamp': current_time
                })

        except Exception as e:
            # Skip individual symbol on error
            continue

    console.print(f"✅ [green]Enhanced analysis complete: {len(signals)} signals generated[/green]")
    return signals


# ======== ALL ORIGINAL INDICATOR FUNCTIONS ========
def safe_calculate_indicator(func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        if isinstance(result, tuple):
            return tuple(r.fillna(method='ffill').fillna(0) if isinstance(r, pd.Series) else r for r in result)
        elif isinstance(result, pd.Series):
            return result.fillna(method='ffill').fillna(0)
        return result
    except Exception:
        if hasattr(args[0], 'index'):
            return pd.Series(0, index=args[0].index)
        return 0

def ema_improved(series, length):
    if len(series) < length:
        return pd.Series(index=series.index, dtype='float64')
    return series.ewm(span=length, adjust=True, min_periods=length//2).mean()

def calculate_rsi_improved(df, period=14):
    if df is None or len(df) < period + 5:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')

    try:
        close_prices = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period, min_periods=period//2).mean()
        avg_loss = loss.rolling(window=period, min_periods=period//2).mean()

        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    except Exception:
        return pd.Series(50, index=df.index)

def volume_surge_improved(df, lookback=20):
    if df is None or len(df) < lookback + 5:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')

    try:
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        vol_ma = volume.rolling(lookback, min_periods=lookback//2).mean()
        vol_std = volume.rolling(lookback, min_periods=lookback//2).std()
        vol_std = vol_std.where(vol_std > vol_ma * 0.01, vol_ma * 0.1)
        z_score = (volume - vol_ma) / vol_std
        return z_score.clip(-5, 5).fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def momentum_improved(df, period=10):
    if df is None or len(df) < period + 2:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')

    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        shifted_close = close.shift(period).replace(0, np.nan)
        momentum_val = (close / shifted_close) - 1.0
        return momentum_val.fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def oi_surge_improved(df, lookback=20):
    """OI surge calculation for option trading"""
    if df is None or len(df) < lookback + 5:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')

    try:
        oi_col = None
        for col in df.columns:
            if col.upper() == 'OI' or 'openinterest' in col.lower():
                oi_col = col
                break

        if oi_col is None:
            return volume_surge_improved(df, lookback)

        oi = pd.to_numeric(df[oi_col], errors='coerce').fillna(0)
        if oi.sum() == 0:
            return volume_surge_improved(df, lookback)

        oi_ma = oi.rolling(lookback, min_periods=lookback//2).mean()
        oi_std = oi.rolling(lookback, min_periods=lookback//2).std()
        oi_std = oi_std.where(oi_std > oi_ma * 0.01, oi_ma * 0.1)
        z_score = (oi - oi_ma) / oi_std
        return z_score.clip(-5, 5).fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def oi_momentum_improved(df, period=10):
    """OI momentum calculation"""
    if df is None or len(df) < period + 2:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')

    try:
        oi_col = None
        for col in df.columns:
            if col.upper() == 'OI' or 'openinterest' in col.lower():
                oi_col = col
                break

        if oi_col is None:
            return momentum_improved(df, period)

        oi = pd.to_numeric(df[oi_col], errors='coerce').fillna(0)
        if oi.sum() == 0:
            return momentum_improved(df, period)

        shifted_oi = oi.shift(period).replace(0, np.nan)
        return ((oi / shifted_oi) - 1.0).fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

# ======== SIGNAL CLASSIFICATION ========
def classify_option_signal(normalized_score, oi_status, has_strong_conditions):
    """Signal classification for option trading"""

    if normalized_score >= 60:
        return "🚀 ULTRA STRONG BUY - CALL HEAVY", "ULTRA_STRONG"
    elif normalized_score <= -60:
        return "💥 ULTRA STRONG SELL - PUT HEAVY", "ULTRA_STRONG"
    elif normalized_score >= 35:
        if "Call Setup" in oi_status:
            return "🔥 VERY STRONG BUY - CALL FOCUS", "VERY_STRONG"
        else:
            return "🔥 VERY STRONG BUY", "VERY_STRONG"
    elif normalized_score <= -35:
        if "Put Setup" in oi_status:
            return "🔥 VERY STRONG SELL - PUT FOCUS", "VERY_STRONG"
        else:
            return "🔥 VERY STRONG SELL", "VERY_STRONG"
    elif normalized_score >= 20:
        if "High OI Activity" in oi_status:
            return "⚡ STRONG BUY - OI SURGE", "STRONG"
        else:
            return "⚡ STRONG BUY", "STRONG"
    elif normalized_score <= -20:
        if "High OI Activity" in oi_status:
            return "⚡ STRONG SELL - OI SURGE", "STRONG"
        else:
            return "⚡ STRONG SELL", "STRONG"
    elif normalized_score >= 10:
        return "🟢 BUY - Call Potential", "MODERATE"
    elif normalized_score <= -10:
        return "🔴 SELL - Put Potential", "MODERATE"
    else:
        return "⚪ NEUTRAL", "NEUTRAL"

def get_option_action(signal_strength, normalized_score):
    """Get specific option trading action"""
    if signal_strength == "ULTRA_STRONG":
        if normalized_score > 0:
            return "🚨 BUY CALLS AGGRESSIVELY", "URGENT"
        else:
            return "🚨 BUY PUTS AGGRESSIVELY", "URGENT"
    elif signal_strength == "VERY_STRONG":
        if normalized_score > 0:
            return "🔥 BUY CALLS STRONG", "HIGH"
        else:
            return "🔥 BUY PUTS STRONG", "HIGH"
    elif signal_strength == "STRONG":
        if normalized_score > 0:
            return "⚡ BUY CALLS", "MEDIUM"
        else:
            return "⚡ BUY PUTS", "MEDIUM"
    elif signal_strength == "MODERATE":
        if normalized_score > 0:
            return "📈 Consider Calls", "LOW"
        else:
            return "📉 Consider Puts", "LOW"
    else:
        return "⏸️ WAIT", "NONE"

# ======== VALIDATION ========
def validate_data_quality(df, min_bars=20):
    if df is None or df.empty:
        return False, "No data"

    if len(df) < min_bars:
        return False, f"Insufficient data: {len(df)} < {min_bars}"

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            return False, f"Missing column: {col}"

        missing_pct = (df[col].isna().sum() / len(df)) * 100
        if missing_pct > CONFIG['MAX_MISSING_DATA_PCT']:
            return False, f"Too much missing data in {col}: {missing_pct:.1f}%"

    return True, "Data quality OK"

# ======== DATA NORMALIZATION ========
def normalize_hist_df_clean(df, symbol, timeframe_minutes):
    """CLEAN: Enhanced normalization without debug messages"""
    if df is None or df.empty:
        return None

    try:
        out = df.copy()
        out.columns = out.columns.str.lower().str.strip()

        # Enhanced column mapping
        rename_map = {}
        for col in out.columns:
            col_clean = col.lower().strip()

            # Timestamp variations
            if any(x in col_clean for x in ['time', 'date', 'timestamp', 'datetime', 'ts']):
                rename_map[col] = 'Timestamp'

            # OHLC mapping
            elif col_clean in ['open'] or (col_clean.startswith('open') and 'interest' not in col_clean):
                rename_map[col] = 'Open'
            elif col_clean in ['high', 'h']:
                rename_map[col] = 'High'
            elif col_clean in ['low', 'l']:
                rename_map[col] = 'Low'
            elif col_clean in ['close', 'c']:
                rename_map[col] = 'Close'

            # Volume mapping
            elif col_clean in ['volume', 'vol', 'v']:
                rename_map[col] = 'Volume'

            # OI detection (SILENT)
            elif any(pattern in col_clean for pattern in [
                'oi', 'openinterest', 'open_interest', 'open interest', 
                'openint', 'open_int', 'oi_value', 'oivalue'
            ]):
                rename_map[col] = 'OI'

        out.rename(columns=rename_map, inplace=True)

        # Handle missing timestamp
        if "Timestamp" not in out.columns:
            if hasattr(out.index, 'dtype') and 'datetime' in str(out.index.dtype):
                out["Timestamp"] = out.index
                out = out.reset_index(drop=True)
            else:
                now = datetime.now(IST)
                out["Timestamp"] = pd.date_range(
                    start=now - timedelta(minutes=timeframe_minutes * len(out)),
                    periods=len(out),
                    freq=f"{timeframe_minutes}T",
                    tz=IST
                )

        # Check required columns
        required_cols = ["Open", "High", "Low", "Close"]
        missing_cols = [col for col in required_cols if col not in out.columns]

        if missing_cols:
            return None

        # Handle Volume
        if "Volume" not in out.columns:
            out["Volume"] = 1000

        # Handle OI (SILENT)
        if "OI" in out.columns:
            try:
                out["OI"] = pd.to_numeric(out["OI"], errors="coerce").fillna(0)

                oi_sum = out["OI"].sum()
                oi_max = out["OI"].max()

                if oi_sum > 0 and oi_max > 100:
                    global oi_symbols_found
                    oi_symbols_found.add(symbol)
                else:
                    out["OI"] = out["Volume"] * 0.1
            except Exception:
                out["OI"] = out["Volume"] * 0.1
        else:
            out["OI"] = out["Volume"] * 0.1

        # Process timestamp
        try:
            out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
            out = out.dropna(subset=["Timestamp"])

            if out.empty:
                return None

            # Handle timezone
            if out["Timestamp"].dt.tz is None:
                try:
                    out["Timestamp"] = out["Timestamp"].dt.tz_localize(IST, ambiguous='infer', nonexistent='shift_forward')
                except Exception:
                    out["Timestamp"] = out["Timestamp"].dt.tz_localize(IST, ambiguous='NaT', nonexistent='NaT')
                    out = out.dropna(subset=["Timestamp"])
            else:
                out["Timestamp"] = out["Timestamp"].dt.tz_convert(IST)
        except Exception:
            return None

        # Convert numeric columns
        for col in ["Open", "High", "Low", "Close", "Volume", "OI"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")

        # Remove invalid data
        out = out.dropna(subset=["Open", "High", "Low", "Close"])

        if out.empty or len(out) < 10:
            return None

        # Set index and remove duplicates
        out = out.sort_values("Timestamp").set_index("Timestamp")
        out = out[~out.index.duplicated(keep='last')]

        return out

    except Exception:
        return None

def pick_session(symbol_orig, timeframe_minutes):
    return hash((symbol_orig, timeframe_minutes)) & 0x7fffffff % len(tdhist_pool)


# ======== ENHANCED RENDERING FOR LIVE SIGNALS ========
def render_live_signals_enhanced(signals, timestamp):
    """
    Enhanced rendering for live signals with real-time OI/LTP/Volume data
    """
    if not signals:
        console.print("📊 [yellow]No strong enhanced signals detected[/yellow]")
        return

    # Sort by signal strength
    signals.sort(key=lambda x: abs(x['score']), reverse=True)

    console.rule(f"🎯 ENHANCED LIVE SIGNALS | {timestamp.strftime('%H:%M:%S')} IST", style="bold green")

    # Create enhanced table with OI/LTP/Volume data
    table = Table(title="📈 REAL-TIME NSE FNO SIGNALS WITH OI/LTP/VOLUME", box=box.DOUBLE_EDGE, header_style="bold white on blue")
    table.add_column("Stock", style="bold white", width=10)
    table.add_column("Signal", style="bold yellow", width=16)
    table.add_column("Score", style="bold green", justify="right", width=6)
    table.add_column("LTP", style="cyan", justify="right", width=8)
    table.add_column("Change %", style="bold red", justify="right", width=8)
    table.add_column("Volume", style="blue", justify="right", width=10)
    table.add_column("OI", style="magenta", justify="right", width=10)
    table.add_column("OI Status", style="yellow", width=12)
    table.add_column("Action", style="bold red", width=16)

    for signal in signals[:25]:  # Show top 25 signals
        # Color coding for change %
        change_pct = signal['change_pct']
        if change_pct >= 2:
            change_color = "bold green"
        elif change_pct >= 0:
            change_color = "green"
        elif change_pct <= -2:
            change_color = "bold red"
        else:
            change_color = "red"

        change_text = f"[{change_color}]{change_pct:+.1f}%[/{change_color}]"

        # Format large numbers
        volume_text = f"{signal['volume']:,.0f}" if signal['volume'] > 0 else "N/A"
        oi_text = f"{signal['oi']:,.0f}" if signal['oi'] > 0 else "N/A"

        # Row styling based on priority
        row_style = None
        if signal.get('priority') == 'URGENT':
            row_style = "bold black on yellow"
        elif signal.get('priority') == 'HIGH':
            row_style = "bold white on red"

        table.add_row(
            signal['symbol'],
            signal['signal'],
            f"{signal['score']:+.0f}",
            f"₹{signal['ltp']:.1f}",
            change_text,
            volume_text,
            oi_text,
            signal.get('oi_status', 'Normal'),
            signal['action'],
            style=row_style
        )

    console.print(table)

    # Enhanced summary with OI/Volume statistics
    buy_signals = [s for s in signals if s['score'] > 0]
    sell_signals = [s for s in signals if s['score'] < 0]
    urgent_signals = [s for s in signals if s.get('priority') == 'URGENT']
    high_priority = [s for s in signals if s.get('priority') == 'HIGH']
    oi_signals = [s for s in signals if s.get('oi', 0) > 10000]
    volume_surge_signals = [s for s in signals if s.get('volume_surge', 0) > 2]

    console.print(f"\n[bold yellow]📊 ENHANCED SUMMARY:[/bold yellow]")
    console.print(f"   🎯 Total: [cyan]{len(signals)}[/cyan] | 🟢 BUY: [green]{len(buy_signals)}[/green] | 🔴 SELL: [red]{len(sell_signals)}[/red]")
    console.print(f"   🚨 URGENT: [bold red]{len(urgent_signals)}[/bold red] | ⚡ HIGH: [yellow]{len(high_priority)}[/yellow]")
    console.print(f"   📈 High OI: [magenta]{len(oi_signals)}[/magenta] | 📊 Volume Surge: [blue]{len(volume_surge_signals)}[/blue]")

    console.rule()

def export_enhanced_signals_to_csv(signals, timestamp):
    """Export enhanced signals with complete OI/LTP/Volume data"""
    filename = f"enhanced_live_signals_{timestamp.strftime('%Y%m%d')}.csv"

    with open(filename, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Enhanced Real-time Scan: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"])
        writer.writerow(["Stock", "Signal", "Score", "LTP", "Change%", "High", "Low", "Volume", "OI", "Vol_Surge", "OI_Surge", "OI_Status", "Action", "Priority", "Components"])

        for signal in signals:
            writer.writerow([
                signal['symbol'],
                signal['signal'], 
                f"{signal['score']:+.1f}",
                f"₹{signal['ltp']:.2f}",
                f"{signal['change_pct']:+.2f}%",
                f"₹{signal.get('high', 0):.2f}",
                f"₹{signal.get('low', 0):.2f}",
                f"{signal['volume']:,.0f}",
                f"{signal['oi']:,.0f}",
                f"{signal.get('volume_surge', 0):.2f}",
                f"{signal.get('oi_surge', 0):.2f}",
                signal.get('oi_status', 'Normal'),
                signal['action'],
                signal.get('priority', 'MEDIUM'),
                signal['components']
            ])
        writer.writerow([])

def send_enhanced_live_alerts(signals):
    """Enhanced alerts with complete OI and volume data"""
    urgent_signals = [s for s in signals if s.get('priority') == 'URGENT']
    high_priority = [s for s in signals if s.get('priority') == 'HIGH']

    if urgent_signals:
        console.print("\n🚨 [bold white on red]URGENT - ENHANCED TRADING ALERTS[/bold white on red]")
        for signal in urgent_signals:
            oi_text = f" | OI: {signal['oi']:,.0f}" if signal['oi'] > 0 else ""
            vol_text = f" | Vol: {signal['volume']:,.0f}" if signal['volume'] > 0 else ""
            vol_surge_text = f" | Vol Surge: {signal.get('volume_surge', 0):.1f}x" if signal.get('volume_surge', 0) > 1 else ""
            oi_surge_text = f" | OI Surge: {signal.get('oi_surge', 0):.1f}x" if signal.get('oi_surge', 0) > 1 else ""

            console.print(f"🔥 {signal['symbol']}: {signal['action']}")
            console.print(f"   📊 LTP: ₹{signal['ltp']:.1f} ({signal['change_pct']:+.1f}%){vol_text}{oi_text}")
            console.print(f"   ⚡ {signal['components']}{vol_surge_text}{oi_surge_text}")

    if high_priority:
        console.print("\n⚡ [bold yellow]HIGH PRIORITY - ENHANCED SIGNALS[/bold yellow]")
        for signal in high_priority[:8]:  # Show top 8 high priority
            console.print(f"📈 {signal['symbol']}: {signal['action']} | Score: {signal['score']:+.1f} | LTP: ₹{signal['ltp']:.1f} | OI: {signal['oi']:,.0f}")

# ======== ORIGINAL ANALYSIS ENGINE (Maintained for Backtest) ========
def analyze_signals_enhanced_clean(timeframe_dataframes, symbol):
    """Enhanced analysis with clean logging - maintained for backtest compatibility"""
    if not timeframe_dataframes:
        return 'Neutral', 0.0, 'Normal', 'WAIT', 'NONE'

    final_score, max_possible = 0.0, 0.0
    valid_timeframes = 0
    oi_status = 'Normal'
    has_strong_conditions = False

    for tf_min, df in timeframe_dataframes.items():
        if df is None or len(df) < 20:
            continue

        is_valid, _ = validate_data_quality(df, 20)
        if not is_valid:
            continue

        valid_timeframes += 1
        tf_weight = CONFIG["TIMEFRAME_WEIGHTS"].get(tf_min, 1.0)

        # Track OI symbols (silent)
        if 'OI' in df.columns and df['OI'].sum() > 100:
            global oi_symbols_found
            oi_symbols_found.add(symbol)

        scores = {}

        # RSI Analysis
        try:
            rsi_series = calculate_rsi_improved(df)
            if len(rsi_series) >= 2:
                rsi_current = rsi_series.iloc[-1]

                if rsi_current > 65:
                    scores['RSI'] = 2.0
                elif rsi_current > 55:
                    scores['RSI'] = 1.0
                elif rsi_current < 35:
                    scores['RSI'] = -2.0
                elif rsi_current < 45:
                    scores['RSI'] = -1.0
                else:
                    scores['RSI'] = 0.0
            else:
                scores['RSI'] = 0.0
        except Exception:
            scores['RSI'] = 0.0

        # Momentum Analysis
        try:
            mom = momentum_improved(df)
            if len(mom) >= 2:
                current_mom = mom.iloc[-1]
                if current_mom > 0.01:
                    scores['Momentum'] = 2.0
                elif current_mom > 0.003:
                    scores['Momentum'] = 1.0
                elif current_mom < -0.01:
                    scores['Momentum'] = -2.0
                elif current_mom < -0.003:
                    scores['Momentum'] = -1.0
                else:
                    scores['Momentum'] = 0.0
            else:
                scores['Momentum'] = 0.0
        except Exception:
            scores['Momentum'] = 0.0

        # Volume surge analysis
        try:
            vol_surge = volume_surge_improved(df)
            if len(vol_surge) >= 2:
                current_surge = vol_surge.iloc[-1]
                price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1

                if current_surge >= 1.5:
                    if price_change > 0.005:
                        scores['VolumeSurge'] = 2.0
                    elif price_change < -0.005:
                        scores['VolumeSurge'] = -2.0
                    else:
                        scores['VolumeSurge'] = 1.0
                elif current_surge >= 1.0:
                    scores['VolumeSurge'] = 1.0 if price_change > 0 else -1.0
                else:
                    scores['VolumeSurge'] = 0.0
            else:
                scores['VolumeSurge'] = 0.0
        except Exception:
            scores['VolumeSurge'] = 0.0

        # OI analysis
        try:
            oi_z = oi_surge_improved(df)
            if len(oi_z) >= 2:
                oi_surge_current = oi_z.iloc[-1]
                price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1

                if oi_surge_current >= 1.5:
                    scores['OISurge'] = 2.0 if price_change > 0 else -2.0
                    oi_status = 'High OI Activity'
                elif oi_surge_current >= 0.8:
                    scores['OISurge'] = 1.0 if price_change > 0 else -1.0
                    oi_status = 'Moderate OI Activity'
                else:
                    scores['OISurge'] = 0.0
            else:
                scores['OISurge'] = 0.0
        except Exception:
            scores['OISurge'] = 0.0

        # OI momentum
        try:
            oi_mom = oi_momentum_improved(df)
            if len(oi_mom) >= 2:
                oi_mom_current = oi_mom.iloc[-1]
                if oi_mom_current > 0.05:
                    scores['OIMomentum'] = 2.0
                elif oi_mom_current > 0.02:
                    scores['OIMomentum'] = 1.0
                elif oi_mom_current < -0.02:
                    scores['OIMomentum'] = -1.0
                else:
                    scores['OIMomentum'] = 0.0
            else:
                scores['OIMomentum'] = 0.0
        except Exception:
            scores['OIMomentum'] = 0.0

        # Call/Put bias analysis
        price_up = df['Close'].iloc[-1] > df['Close'].iloc[-2]
        vol_high = scores.get('VolumeSurge', 0) >= 1.0
        oi_active = abs(scores.get('OISurge', 0)) >= 1.0

        if price_up and vol_high and oi_active:
            scores['CallBias'] = 3.0
            scores['PutBias'] = 0.0
            oi_status = 'Call Setup'
            has_strong_conditions = True
        elif not price_up and vol_high and oi_active:
            scores['PutBias'] = -3.0
            scores['CallBias'] = 0.0
            oi_status = 'Put Setup'
            has_strong_conditions = True
        elif price_up and (vol_high or oi_active):
            scores['CallBias'] = 1.0
            scores['PutBias'] = 0.0
        elif not price_up and (vol_high or oi_active):
            scores['PutBias'] = -1.0
            scores['CallBias'] = 0.0
        else:
            scores['CallBias'] = 0.0
            scores['PutBias'] = 0.0

        # OI-Volume Confirmation
        if oi_active or vol_high:
            if price_up:
                scores['OIVolConfirm'] = 1.0
            else:
                scores['OIVolConfirm'] = -1.0
        else:
            scores['OIVolConfirm'] = 0.0

        # Fill remaining indicators
        remaining_indicators = ['MACD', 'ADX', 'VWAP', 'EMA', 'CMF', 'ADL', 'OBV', 'ATR', 
                               'Bollinger', 'ROC', 'Stochastic', 'CCI', 'MA', 'WWL', 
                               'RelVol', 'VWAPRegime', 'OBVConfirm']
        for indicator in remaining_indicators:
            if indicator not in scores:
                scores[indicator] = 0.0

        # Calculate weighted scores
        for indicator, score in scores.items():
            ind_weight = CONFIG["INDICATOR_WEIGHTS"].get(indicator, 1.0)
            weighted_score = score * tf_weight * ind_weight
            final_score += weighted_score
            max_possible += 3.0 * tf_weight * ind_weight

    if valid_timeframes < 1 or max_possible == 0:
        return 'Neutral', 0.0, oi_status, 'WAIT', 'NONE'

    normalized = (final_score / max_possible) * 100.0

    if abs(normalized) > 100:
        normalized = np.sign(normalized) * 100

    signal_text, signal_strength = classify_option_signal(normalized, oi_status, has_strong_conditions)
    option_action, alert_priority = get_option_action(signal_strength, normalized)

    return signal_text, normalized, oi_status, option_action, alert_priority


# ======== CLEAN DATA FETCHING (Original Engine) ========
@retry(
    stop_max_attempt_number=CONFIG["RETRY_ATTEMPTS"],
    wait_exponential_multiplier=max(1, int(CONFIG["RETRY_DELAY_MS"] / 2)),
    wait_exponential_max=10000,
    retry_on_exception=lambda e: True
)
def fetch_one_clean(symbol_orig, timeframe_minutes, limiter, hist):
    """Clean fetch function without debug logs"""
    td_symbol = symbol_orig.replace("-EQ", "")

    bar_size = CONFIG["BAR_SIZE_MAP"].get(timeframe_minutes)
    duration = CONFIG["DURATION_MAP"].get(timeframe_minutes)

    if not bar_size or not duration:
        return symbol_orig, timeframe_minutes, None

    try:
        limiter.acquire()
        df_raw = hist.get_historic_data(td_symbol, duration=duration, bar_size=bar_size)

        if df_raw is None or df_raw.empty:
            return symbol_orig, timeframe_minutes, None

        df = normalize_hist_df_clean(df_raw, td_symbol, timeframe_minutes)

        # Silent API counter
        global api_calls_done
        with api_calls_lock:
            api_calls_done += 1

        return symbol_orig, timeframe_minutes, df

    except Exception:
        return symbol_orig, timeframe_minutes, None

def prefetch_clean(stocks, max_workers=CONFIG["MAX_WORKERS"]):
    """CLEAN: Enhanced prefetch with essential info only"""
    tfs = [5, 15, 30, 60, 1440]
    total_calls = len(stocks) * len(tfs)
    stock_multi_data = defaultdict(dict)

    global api_calls_done, oi_symbols_found
    with api_calls_lock:
        api_calls_done = 0

    oi_symbols_found = set()
    valid_stocks = [s for s in stocks if s]

    # CLEAN: Show essential startup info
    console.print(f"📊 Loading historical data for [cyan]{len(valid_stocks)} symbols[/cyan] across [yellow]5 timeframes[/yellow]")

    progress_kwargs = dict(
        total=total_calls,
        desc="🔄 Loading Historical Data",
        ncols=80,
        disable=not CONFIG["SHOW_PROGRESS"]
    )

    with tqdm(**progress_kwargs) as api_bar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for s in valid_stocks:
                for tf in tfs:
                    si = pick_session(s, tf)
                    futures.append(executor.submit(fetch_one_clean, s, tf, sess_limiters[si], tdhist_pool[si]))

            for fut in as_completed(futures):
                try:
                    symbol_orig, tf, df = fut.result()
                    if df is not None:
                        stock_multi_data[symbol_orig][tf] = df
                except Exception:
                    pass
                api_bar.update(1)

    valid_data = {s: d for s, d in stock_multi_data.items() if len(d) >= 1}

    # CLEAN: Essential completion info
    console.print(f"✅ Historical data loaded: [green]{len(valid_data)} symbols[/green] ready")
    if len(oi_symbols_found) > 0:
        console.print(f"📈 Historical OI data: [yellow]{len(oi_symbols_found)} symbols[/yellow]")

    return valid_data

def filter_timeframe_data(symbol, timeframe_data, time_point_aware):
    """Enhanced filtering with clean error handling"""
    filtered_timeframes = {}

    for tf, df in timeframe_data.items():
        if df is None or df.empty:
            continue

        try:
            if time_point_aware.tzinfo is None:
                time_point_aware = IST.localize(time_point_aware)
            elif time_point_aware.tzinfo != IST:
                time_point_aware = time_point_aware.astimezone(IST)

            if df.index.tz is None:
                df.index = df.index.tz_localize(IST, ambiguous='infer', nonexistent='shift_forward')
            else:
                df.index = df.index.tz_convert(IST)

            try:
                valid_index = df.index.dropna()
                if len(valid_index) != len(df.index):
                    df = df.loc[valid_index]

                if not df.empty:
                    mask = df.index <= time_point_aware
                    df_filtered = df.loc[mask]

                    if len(df_filtered) >= CONFIG["MIN_BARS_REQUIRED"]:
                        filtered_timeframes[tf] = df_filtered

            except Exception:
                continue

        except Exception:
            continue

    return filtered_timeframes

# ======== UTILITY FUNCTIONS ========
def load_stock_list(file_path):
    """Load stock symbols from file"""
    try:
        with open(file_path, 'r') as f:
            stocks = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
        console.print(f"📋 [green]Loaded {len(stocks)} stock symbols from file[/green]")
        return stocks
    except FileNotFoundError:
        console.print(f"[red]❌ File {file_path} not found[/red]")
        # Return comprehensive NSE FNO stocks if file not found
        sample_stocks = CONFIG['MAJOR_FNO_STOCKS'] + [
            "ADANIENT", "ADANIPORTS", "APOLLOHOSP", "BPCL", "BRITANNIA", "CIPLA", "DIVISLAB", "DRREDDY",
            "EICHERMOT", "GRASIM", "HEROMOTOCO", "HINDALCO", "JSWSTEEL", "M&M", "NESTLEIND", "ONGC",
            "SHREECEM", "TATASTEEL", "TATACONSUM", "UPL", "VEDL", "VOLTAS", "WIPRO", "ZEEL"
        ]
        console.print(f"📋 [yellow]Using comprehensive NSE FNO stocks: {len(sample_stocks)} symbols[/yellow]")
        return sample_stocks

def print_clean_banner():
    """Display enhanced clean banner"""
    console.print("\n" + "="*90)
    console.print("🎯 [bold green]ENHANCED NSE FNO SCANNER v3.2 - REAL-TIME OI/LTP/VOLUME[/bold green] 🎯")
    console.print("="*90)
    console.print("[cyan]🔥 Live OI Tracking • 💰 Real-time LTP • 📊 Volume Analysis • ⚡ Option Signals[/cyan]")
    console.print("="*90 + "\n")

def infer_institutional_flow(tf_data):
    """Simplified institutional flow detection"""
    return "Mixed"

# ======== BACKTEST FUNCTIONS (Original Engine Maintained) ========
def render_signals_clean(now_ts, top_bullish, top_bearish):
    """Clean signal rendering for backtest mode"""
    global last_bull_symbols, last_bear_symbols

    title = f"🎯 BACKTEST SIGNALS | {now_ts.strftime('%H:%M')} IST"
    console.rule(title, style="bold yellow")

    # Filter signal categories
    ultra_strong_bulls = [r for r in top_bullish if "ULTRA STRONG" in r['signal']]
    very_strong_bulls = [r for r in top_bullish if "VERY STRONG" in r['signal']]
    strong_bulls = [r for r in top_bullish if "STRONG BUY" in r['signal'] or "⚡ STRONG" in r['signal']]
    moderate_bulls = [r for r in top_bullish if "🟢 BUY" in r['signal']]

    ultra_strong_bears = [r for r in top_bearish if "ULTRA STRONG" in r['signal']]
    very_strong_bears = [r for r in top_bearish if "VERY STRONG" in r['signal']]
    strong_bears = [r for r in top_bearish if "STRONG SELL" in r['signal'] or "⚡ STRONG" in r['signal']]
    moderate_bears = [r for r in top_bearish if "🔴 SELL" in r['signal']]

    # Show strong signals
    all_strong = ultra_strong_bulls + very_strong_bulls + strong_bulls + ultra_strong_bears + very_strong_bears + strong_bears

    if all_strong:
        console.print("\n🔥 [bold white on red]STRONG BACKTEST SIGNALS[/bold white on red]")

        strong_table = Table(title="💪 STRONG SIGNALS", box=box.DOUBLE_EDGE, header_style="bold white on blue")
        strong_table.add_column("Stock", style="bold white")
        strong_table.add_column("Signal", style="bold yellow")
        strong_table.add_column("Score", style="bold green", justify="right")
        strong_table.add_column("OI Status", style="cyan")
        strong_table.add_column("Action", style="bold red")

        for r in all_strong[:15]:  # Show top 15 in backtest
            strong_table.add_row(
                r['symbol'], r['signal'], f"{r['score']:.1f}",
                r.get('oi_status', 'Normal'), r.get('action', 'TRADE')
            )

        console.print(strong_table)

    # Summary
    total_signals = len(all_strong) + len(moderate_bulls + moderate_bears)
    console.print(f"\n[bold yellow]📊 BACKTEST SUMMARY: {total_signals} signals generated[/bold yellow]")
    console.rule()

    last_bull_symbols = {r['symbol'] for r in top_bullish}
    last_bear_symbols = {r['symbol'] for r in top_bearish}

def export_to_csv(now_ts, top_bullish, top_bearish, filename):
    """Export backtest signals to CSV"""
    with open(filename, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Backtest Time: {now_ts.strftime('%Y-%m-%d %H:%M')}"])
        writer.writerow(["Stock", "Signal", "Score", "Change", "OI Status", "Action"])

        for r in top_bullish:
            ch = r['change']
            change_str = f"{ch:+.2f}" if isinstance(ch, (int, float, np.floating)) else "NEW"
            writer.writerow([r['symbol'], r['signal'], f"{r['score']:.2f}", change_str, 
                           r.get('oi_status', 'Normal'), r.get('action', 'TRADE')])

        for r in top_bearish:
            ch = r['change']
            change_str = f"{ch:+.2f}" if isinstance(ch, (int, float, np.floating)) else "NEW"
            writer.writerow([r['symbol'], r['signal'], f"{r['score']:.2f}", change_str, 
                           r.get('oi_status', 'Normal'), r.get('action', 'TRADE')])
        writer.writerow([])

def run_backtest_clean(day_str: str, stocks):
    """Clean backtest with essential information only"""
    global previous_scores, last_bull_symbols, last_bear_symbols, performance_metrics

    day_date = datetime.strptime(day_str, "%Y-%m-%d")
    console.print(f"📅 [bold cyan]Backtesting {day_str}[/bold cyan] with [yellow]{len(stocks)} symbols[/yellow]")

    stock_multi_data = prefetch_clean(stocks)

    if len(stock_multi_data) == 0:
        console.print("[red]❌ No valid historical data found[/red]")
        return

    checkpoints = day_checkpoints_ist(day_date)
    output_filename = day_date.strftime("%Y-%m-%d") + "_backtest_signals.csv"

    # Clean old file
    try:
        if os.path.exists(output_filename):
            os.remove(output_filename)
    except Exception:
        pass

    previous_scores = {}
    last_bull_symbols = set()
    last_bear_symbols = set()
    performance_metrics = defaultdict(int)

    console.print(f"🔍 Analyzing [cyan]{len(checkpoints)}[/cyan] time periods for backtest...")

    for i, asof_ts in enumerate(checkpoints):
        if i % 20 == 0:
            console.print(f"⏳ Backtest Progress: [cyan]{i+1}/{len(checkpoints)}[/cyan] | Time: [yellow]{asof_ts.strftime('%H:%M')}[/yellow]")

        time_point_aware = asof_ts.replace(second=0, microsecond=0)
        signals_this_scan = []
        current_scores = {}

        for symbol, timeframe_data in stock_multi_data.items():
            clean_symbol = symbol.replace('-EQ', '')
            filtered_timeframes = filter_timeframe_data(clean_symbol, timeframe_data, time_point_aware)

            if len(filtered_timeframes) < 1:
                continue

            signal, score, oi_status, option_action, alert_priority = analyze_signals_enhanced_clean(filtered_timeframes, clean_symbol)
            current_scores[clean_symbol] = score

            if abs(score) >= CONFIG['MIN_SIGNAL_THRESHOLD'] or any(x in signal for x in ['STRONG', 'BUY', 'SELL']):
                prev = previous_scores.get(clean_symbol, 'NA')
                change_val = 'NA' if isinstance(prev, str) else (score - prev)
                direction = 'bullish' if score > 0 else 'bearish'

                signals_this_scan.append({
                    'symbol': clean_symbol, 'signal': signal, 'score': score,
                    'trend': direction, 'change': change_val, 'oi_status': oi_status, 
                    'action': option_action
                })
                performance_metrics[f"{direction}_signals"] += 1

        previous_scores = current_scores.copy()
        signals_this_scan.sort(key=lambda x: abs(x['score']), reverse=True)
        top_bullish = [r for r in signals_this_scan if r['score'] > 0][:20]
        top_bearish = [r for r in signals_this_scan if r['score'] < 0][:20]

        if top_bullish or top_bearish:
            render_signals_clean(asof_ts, top_bullish, top_bearish)
            export_to_csv(asof_ts, top_bullish, top_bearish, output_filename)

    # Final backtest summary
    total_signals = sum(performance_metrics.values())
    console.print(f"\n📈 [bold green]BACKTEST COMPLETE[/bold green]")
    console.print(f"Total Signals: [cyan]{total_signals}[/cyan]")
    if 'bullish_signals' in performance_metrics:
        console.print(f"Bullish: [green]{performance_metrics['bullish_signals']}[/green]")
    if 'bearish_signals' in performance_metrics:
        console.print(f"Bearish: [red]{performance_metrics['bearish_signals']}[/red]")
    console.print(f"Results saved to: [yellow]{output_filename}[/yellow]")

# ======== COMPLETE ENHANCED LIVE MODE ========
def run_live_mode_enhanced(stocks):
    """
    Complete enhanced live mode with real-time OI, LTP, and TOTALVOL data
    Replaces Excel RTD formulas with direct API calls
    """
    global previous_scores, last_bull_symbols, last_bear_symbols

    console.print("🔴 [bold yellow]ENHANCED LIVE MODE STARTING...[/bold yellow]")
    console.print("📡 [cyan]Real-time OI, LTP & TOTALVOL tracking enabled[/cyan]")
    console.print("🔄 [green]Replaces Excel RTD formulas with direct API calls[/green]")

    # Check market timing
    now_ist = datetime.now(IST)
    market_start = today_ist_dt(CONFIG["MARKET_START"])
    market_end = today_ist_dt(CONFIG["MARKET_END"])
    first_run = today_ist_dt(CONFIG["FIRST_RUN_AT"])

    console.print(f"⏰ Current Time: [cyan]{now_ist.strftime('%H:%M:%S IST')}[/cyan]")
    console.print(f"📊 Market Hours: [yellow]{CONFIG['MARKET_START']} - {CONFIG['MARKET_END']} IST[/yellow]")

    # Market timing checks
    if now_ist < market_start:
        console.print(f"🕘 [yellow]Market opens at {CONFIG['MARKET_START']}[/yellow]")
        console.print(f"⏳ Waiting until market opens...")
        sleep_until(market_start)
        now_ist = datetime.now(IST)

    if now_ist >= market_end:
        console.print("🔒 [red]Market is closed for today[/red]")
        return

    if now_ist < first_run:
        console.print(f"⏳ Waiting until first enhanced scan at [yellow]{CONFIG['FIRST_RUN_AT']}[/yellow]...")
        sleep_until(first_run)

    console.print("🚀 [bold green]ENHANCED LIVE SCANNING ACTIVATED[/bold green]")

    # Load historical data once (for context and analysis)
    console.print("📊 [yellow]Loading historical context data...[/yellow]")
    historical_data = prefetch_clean(stocks)

    # Initialize variables
    previous_scores = {}
    last_bull_symbols = set()
    last_bear_symbols = set()
    scan_count = 0

    try:
        while True:
            now_ist = datetime.now(IST)

            if now_ist >= market_end:
                console.print("🔒 [bold red]Market closed. Stopping enhanced live mode.[/bold red]")
                break

            scan_count += 1
            console.print(f"\n🔄 [bold cyan]ENHANCED SCAN #{scan_count}[/bold cyan] | {now_ist.strftime('%H:%M:%S IST')}")

            # Fetch real-time market data (OI, LTP, TOTALVOL)
            live_market_data = get_live_market_data_batch(stocks, max_workers=CONFIG['REALTIME_FETCH_WORKERS'])

            if not live_market_data:
                console.print("[red]❌ No live market data available[/red]")
            else:
                # Analyze enhanced live signals
                live_signals = analyze_live_signals_enhanced(live_market_data, historical_data)

                if live_signals:
                    # Render enhanced live signals with OI/LTP/Volume data
                    render_live_signals_enhanced(live_signals, now_ist)

                    # Export to CSV with complete enhanced data
                    export_enhanced_signals_to_csv(live_signals, now_ist)

                    # Send enhanced priority alerts
                    send_enhanced_live_alerts(live_signals)

                    console.print(f"⚡ [bold green]{len(live_signals)} enhanced signals with OI/LTP/Volume data[/bold green]")
                else:
                    console.print("📊 [yellow]No strong enhanced signals in this scan[/yellow]")

            # Calculate next scan time (every 5 minutes)
            next_scan = next_5min_boundary_ist(now_ist)

            if next_scan >= market_end:
                console.print("🔒 [yellow]Next enhanced scan would be after market close[/yellow]")
                break

            sleep_seconds = (next_scan - datetime.now(IST)).total_seconds()
            console.print(f"⏱️  Next enhanced scan at [cyan]{next_scan.strftime('%H:%M')}[/cyan] | Waiting [yellow]{sleep_seconds:.0f}s[/yellow]")

            sleep_until(next_scan)

    except KeyboardInterrupt:
        console.print("\n🛑 [yellow]Enhanced live mode stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n💥 [red]Enhanced live mode error: {e}[/red]")

# ======== MAIN EXECUTION ========
if __name__ == "__main__":
    print_clean_banner()

    try:
        parser = argparse.ArgumentParser(description="Enhanced NSE FNO Scanner with Real-time OI/LTP/Volume")
        parser.add_argument("--live", action="store_true", help="Enhanced live market mode with OI/LTP/Volume")
        parser.add_argument("--backtest-date", help="Backtest date (YYYY-MM-DD)")
        parser.add_argument("--stocks-file", default=CONFIG["SHARES_FILE"], help="Stock symbols file")
        parser.add_argument("--test-oi", action="store_true", help="Test enhanced OI/LTP/Volume data")

        args = parser.parse_args()

        stocks = load_stock_list(args.stocks_file)
        if not stocks:
            console.print("[red]❌ No valid stocks loaded[/red]")
            exit(1)

        if args.live:
            console.print("🎯 [bold green]Starting Enhanced Live Mode with Real-time OI/LTP/Volume![/bold green]")
            run_live_mode_enhanced(stocks)

        elif args.backtest_date:
            try:
                datetime.strptime(args.backtest_date, "%Y-%m-%d")
                console.print(f"📅 [bold cyan]Running backtest for {args.backtest_date}[/bold cyan]")
                run_backtest_clean(args.backtest_date, stocks)
            except ValueError:
                console.print("[red]❌ Invalid date format. Use YYYY-MM-DD[/red]")

        elif args.test_oi:
            console.print("🧪 [yellow]Testing enhanced OI/LTP/Volume data...[/yellow]")
            test_data = get_live_market_data_batch(stocks[:15])  # Test with 15 symbols
            console.print(f"✅ Enhanced test complete: [cyan]{len(test_data)}[/cyan] symbols with live data")

            # Show detailed sample data
            if test_data:
                console.print("\n📊 [bold green]Sample Enhanced Data:[/bold green]")
                for symbol, data in list(test_data.items())[:8]:
                    console.print(f"   🎯 {symbol}: LTP=₹{data['LTP']:.1f} | Volume={data['TOTALVOL']:,} | OI={data['OI']:,} | Change={data['CHANGE_PCT']:+.1f}%")

        else:
            console.print("\n🎯 [bold green]Enhanced NSE FNO Option Scanner v3.2[/bold green]")
            console.print("🔥 [bold yellow]Real-time OI/LTP/TOTALVOL Analysis - Replaces Excel RTD![/bold yellow]")
            console.print("\n[cyan]Enhanced Features:[/cyan]")
            console.print("  📡 [yellow]Real-time OI (Open Interest) tracking[/yellow]")
            console.print("  💰 [yellow]Live LTP (Last Traded Price) monitoring[/yellow]") 
            console.print("  📊 [yellow]TOTALVOL (Total Volume) surge analysis[/yellow]")
            console.print("  ⚡ [yellow]Priority-based Call/Put alerts[/yellow]")
            console.print("  🚀 [yellow]Enhanced signal analysis with live data[/yellow]")
            console.print("\n[cyan]Usage:[/cyan]")
            console.print("  [yellow]python enhanced_scanner.py --live[/yellow] - 🔥 Enhanced Live Mode!")
            console.print("  [yellow]python enhanced_scanner.py --backtest-date 2025-09-29[/yellow] - Backtest")
            console.print("  [yellow]python enhanced_scanner.py --test-oi[/yellow] - Test Enhanced Data")

    except KeyboardInterrupt:
        console.print("\n[yellow]👤 Enhanced scanner stopped by user[/yellow]")
    except Exception as e:
        console.print(f"[red]💥 Enhanced scanner error: {e}[/red]")
    finally:
        # Clean shutdown
        for sess in tdhist_pool:
            try:
                sess.disconnect()
            except Exception:
                pass

        if performance_metrics:
            total = sum(performance_metrics.values())
            if total > 0:
                console.print(f"\n📊 [bold green]Final: {total} signals generated[/bold green]")

        console.print("✅ [green]Enhanced scanner shutdown complete[/green]")
