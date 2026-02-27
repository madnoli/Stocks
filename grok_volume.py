#!/usr/bin/env python3
"""
COMPLETE NSE OPTION SCANNER WITH ALL INDICATORS
================================================
Production version with 25+ indicators + Real option chain
"""
import os, logging, warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed
import time, threading
from collections import defaultdict
import argparse, csv
from retrying import retry
import requests
from tqdm import tqdm
from truedata import TD_hist
from rich.console import Console
from rich.table import Table
from rich import box

CONFIG = {
    "TDUSERNAME": os.getenv("TRUEDATA_USER", "tdwsp751"),
    "TDPASSWORD": os.getenv("TRUEDATA_PASS", "raj@751"),
    "OPTION_CHAIN_API": "http://localhost:3000/api/equity/options/",
    "OPTION_CHAIN_TIMEOUT": 3,
    "OPTION_CHAIN_CACHE_TTL": 30,
    "MARKET_START": "09:15",
    "FIRST_RUN_AT": "09:20",
    "MARKET_END": "15:30",
    "SETTLE_DELAY_SECONDS": 5,
    "MAX_WORKERS": 32,
    "TD_HIST_SESSIONS": 4,
    "RATE_PER_SECOND_TOTAL": 15.0,
    "BUCKET_SIZE": 20,
    "RETRY_ATTEMPTS": 7,
    "RETRY_DELAY_MS": 2000,
    "SHARES_FILE": "shares.txt",
    "SHOW_PROGRESS": True,
    "LOG_LEVEL": "WARNING",
    "MIN_BARS_REQUIRED": 20,
    "MAX_MISSING_DATA_PCT": 15,
    "MIN_SIGNAL_THRESHOLD": 10,
    "INDICATOR_PERIODS": {
        "RSI": 14, "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
        "STOCHASTIC_K": 14, "STOCHASTIC_D": 3, "MA_SHORT": 20, "MA_LONG": 50,
        "ADX": 14, "BB_PERIOD": 20, "BB_STD_DEV": 2, "ROC": 12, "CCI": 20,
        "EMA_FAST": 9, "EMA_SLOW": 21, "ATR": 14, "VOLUME_SURGE": 20,
        "MOMENTUM": 10, "WILLIAMS_R": 14, "CMF": 20, "ADL_LOOKBACK": 10,
        "REL_VOL": 20, "VWAP_REGIME": 20, "OBV_CONFIRM": 5,
    },
    "INDICATOR_WEIGHTS": {
        "VolumeSurge": 2.5, "Momentum": 2.2, "ADX": 2.0, "VWAP": 1.8, "EMA": 1.9,
        "MACD": 1.7, "OBV": 1.6, "ATR": 1.5, "Bollinger": 1.4, "RSI": 1.3,
        "ROC": 1.2, "Stochastic": 1.1, "CCI": 1.0, "MA": 1.2, "WWL": 1.0,
        "CMF": 2.0, "ADL": 1.8, "RelVol": 1.7, "VWAPRegime": 1.9, "OBVConfirm": 1.4,
        "PCR": 5.0, "CallOIBuild": 4.5, "PutOIBuild": 4.5,
        "CallVolume": 4.0, "PutVolume": 4.0, "OIChangeRatio": 4.2, "NetBuyPressure": 4.8,
    },
    "VOLUME_SMA_PERIOD": 20,
    "VOLUME_SMA_MULTIPLIER": 5.0,
    "TIMEFRAME_WEIGHTS": {15: 2.5, 5: 2.2, 30: 1.8, 60: 1.2, 1440: 1.0},
    "BAR_SIZE_MAP": {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"},
    "DURATION_MAP": {5: "45 D", 15: "45 D", 30: "90 D", 60: "180 D", 1440: "365 D"},
}

logging.getLogger().setLevel(getattr(logging, CONFIG["LOG_LEVEL"]))
IST = pytz.timezone("Asia/Kolkata")
for n in ("truedata", "websocket", "urllib3", "requests"):
    logging.getLogger(n).setLevel(logging.CRITICAL)

console = Console()
last_bull_symbols, last_bear_symbols = set(), set()
previous_scores = {}
api_calls_done, api_calls_lock = 0, threading.Lock()
performance_metrics = defaultdict(int)
option_chain_cache, option_chain_lock = {}, threading.Lock()
option_chain_stats = defaultdict(int)

# ========== OPTION CHAIN ==========
def fetch_option_chain(symbol, timeout=None):
    global option_chain_cache, option_chain_stats
    if timeout is None:
        timeout = CONFIG['OPTION_CHAIN_TIMEOUT']
    try:
        now = time.time()
        with option_chain_lock:
            if symbol in option_chain_cache:
                cached_data, cache_time = option_chain_cache[symbol]
                if (now - cache_time) < CONFIG['OPTION_CHAIN_CACHE_TTL']:
                    option_chain_stats['cache_hits'] += 1
                    return cached_data
        url = f"{CONFIG['OPTION_CHAIN_API']}{symbol}"
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            data = response.json()
            with option_chain_lock:
                option_chain_cache[symbol] = (data, now)
                option_chain_stats['api_calls'] += 1
                option_chain_stats['success'] += 1
            return data
        else:
            option_chain_stats['failures'] += 1
    except requests.exceptions.Timeout:
        option_chain_stats['timeouts'] += 1
    except Exception:
        option_chain_stats['errors'] += 1
    return None

def analyze_option_chain(symbol):
    option_data = fetch_option_chain(symbol)
    if not option_data or 'records' not in option_data:
        return None
    try:
        data_list = option_data['records'].get('data', [])
        if not data_list:
            return None
        expiry_dates = option_data['records'].get('expiryDates', [])
        if not expiry_dates:
            return None
        nearest_expiry = expiry_dates[0]
        expiry_data = [d for d in data_list if d.get('expiryDate') == nearest_expiry]
        if not expiry_data:
            return None

        total_call_oi = total_put_oi = total_call_volume = total_put_volume = 0
        total_call_oi_change = total_put_oi_change = 0
        call_buy_qty = call_sell_qty = put_buy_qty = put_sell_qty = 0
        underlying_price = 0

        for strike_data in expiry_data:
            ce_data, pe_data = strike_data.get('CE', {}), strike_data.get('PE', {})
            if 'underlyingValue' in ce_data and ce_data['underlyingValue']:
                underlying_price = ce_data['underlyingValue']
            elif 'underlyingValue' in pe_data and pe_data['underlyingValue']:
                underlying_price = pe_data['underlyingValue']
            if ce_data:
                total_call_oi += ce_data.get('openInterest', 0)
                total_call_volume += ce_data.get('totalTradedVolume', 0)
                total_call_oi_change += ce_data.get('changeinOpenInterest', 0)
                call_buy_qty += ce_data.get('totalBuyQuantity', 0)
                call_sell_qty += ce_data.get('totalSellQuantity', 0)
            if pe_data:
                total_put_oi += pe_data.get('openInterest', 0)
                total_put_volume += pe_data.get('totalTradedVolume', 0)
                total_put_oi_change += pe_data.get('changeinOpenInterest', 0)
                put_buy_qty += pe_data.get('totalBuyQuantity', 0)
                put_sell_qty += pe_data.get('totalSellQuantity', 0)

        if total_call_oi == 0 and total_put_oi == 0:
            return None

        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 99.0
        pcr_volume = total_put_volume / total_call_volume if total_call_volume > 0 else 0
        oi_change_ratio = total_call_oi_change / total_put_oi_change if total_put_oi_change != 0 else 0
        call_net_buy, put_net_buy = call_buy_qty - call_sell_qty, put_buy_qty - put_sell_qty

        signal, signal_strength, recommendation, oi_status = "NEUTRAL", 0.0, "WAIT", "Normal"

        if pcr_oi < 0.7:
            signal, signal_strength, recommendation = "CALL_HEAVY", 4.0, "BUY CALLS - Strong bullish"
            oi_status = f"Call Setup (PCR: {pcr_oi:.2f})"
        elif pcr_oi < 0.85:
            signal, signal_strength, recommendation = "CALL_BIAS", 2.5, "BUY CALLS - Moderate"
            oi_status = f"Call Bias (PCR: {pcr_oi:.2f})"
        elif pcr_oi > 1.3:
            signal, signal_strength, recommendation = "PUT_HEAVY", -4.0, "BUY PUTS - Strong bearish"
            oi_status = f"Put Setup (PCR: {pcr_oi:.2f})"
        elif pcr_oi > 1.15:
            signal, signal_strength, recommendation = "PUT_BIAS", -2.5, "BUY PUTS - Moderate"
            oi_status = f"Put Bias (PCR: {pcr_oi:.2f})"
        else:
            oi_status = f"Neutral (PCR: {pcr_oi:.2f})"

        if total_call_oi_change > total_put_oi_change * 1.5 and total_call_oi_change > 100:
            signal_strength += 1.5
            recommendation += " + Call OI↑"
            oi_status += " | Call OI Building"
        elif total_put_oi_change > total_call_oi_change * 1.5 and total_put_oi_change > 100:
            signal_strength -= 1.5
            recommendation += " + Put OI↑"
            oi_status += " | Put OI Building"

        if call_net_buy > 0 and call_net_buy > put_net_buy * 1.2:
            signal_strength += 1.0
        elif put_net_buy > 0 and put_net_buy > call_net_buy * 1.2:
            signal_strength -= 1.0

        return {
            'symbol': symbol, 'pcr_oi': pcr_oi, 'pcr_volume': pcr_volume,
            'total_call_oi': total_call_oi, 'total_put_oi': total_put_oi,
            'total_call_volume': total_call_volume, 'total_put_volume': total_put_volume,
            'call_oi_change': total_call_oi_change, 'put_oi_change': total_put_oi_change,
            'oi_change_ratio': oi_change_ratio, 'call_net_buy': call_net_buy,
            'put_net_buy': put_net_buy, 'underlying_price': underlying_price,
            'signal': signal, 'signal_strength': signal_strength,
            'recommendation': recommendation, 'oi_status': oi_status, 'expiry': nearest_expiry,
        }
    except Exception:
        return None

# ========== TIME HELPERS ==========
def next_5min_boundary_ist(now_ist):
    minute = (now_ist.minute // 5) * 5
    boundary = now_ist.replace(minute=minute, second=0, microsecond=0)
    return boundary + timedelta(minutes=5) if boundary <= now_ist else boundary

def parse_hhmm(s):
    h, m = map(int, s.split(":"))
    return h, m

def today_ist_dt(hhmm):
    now = datetime.now(IST)
    h, m = parse_hhmm(hhmm)
    return now.replace(hour=h, minute=m, second=0, microsecond=0)

def sleep_until(ts):
    while True:
        delta = (ts - datetime.now(IST)).total_seconds()
        if delta <= 0:
            break
        time.sleep(min(0.5, delta))

def day_checkpoints_ist(day_date):
    d = day_date.date()
    start_h, start_m = parse_hhmm(CONFIG["FIRST_RUN_AT"])
    end_h, end_m = parse_hhmm(CONFIG["MARKET_END"])
    start_dt = IST.localize(datetime(d.year, d.month, d.day, start_h, start_m))
    end_dt = IST.localize(datetime(d.year, d.month, d.day, end_h, end_m))
    return list(pd.date_range(start=start_dt, end=end_dt, freq="5T", tz=IST, inclusive="both").to_pydatetime())

# ========== TOKEN BUCKET ==========
class TokenBucketLimiter:
    def __init__(self, rate_per_sec, bucket_size):
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
    sess_count, pool, limiters = CONFIG["TD_HIST_SESSIONS"], [], []
    for i in range(sess_count):
        try:
            pool.append(authenticate_session())
            console.print(f"✅ Session {i+1}/{sess_count}")
        except Exception as e:
            console.print(f"[red]Session {i+1} failed[/red]")
    if not pool:
        raise SystemExit("❌ No TrueData sessions")
    per_sess_rate = CONFIG["RATE_PER_SECOND_TOTAL"] / len(pool)
    for _ in pool:
        limiters.append(TokenBucketLimiter(per_sess_rate, CONFIG["BUCKET_SIZE"]))
    return pool, limiters

tdhist_pool, sess_limiters = build_sessions()
# ========== ALL INDICATOR FUNCTIONS ==========
def calculate_rsi_improved(df, period=14):
    if df is None or len(df) < period + 5:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    try:
        close_prices = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        delta = close_prices.diff()
        gain, loss = delta.where(delta > 0, 0), -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period, min_periods=period//2).mean()
        avg_loss = loss.rolling(window=period, min_periods=period//2).mean()
        rs = avg_gain / avg_loss.replace(0, np.inf)
        return (100 - (100 / (1 + rs))).fillna(50)
    except:
        return pd.Series(50, index=df.index)

def calculate_macd(df, fast=12, slow=26, signal=9):
    if df is None or len(df) < slow + signal:
        return pd.Series(0, index=df.index if df is not None else []), pd.Series(0, index=df.index if df is not None else [])
    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line.fillna(0), signal_line.fillna(0)
    except:
        return pd.Series(0, index=df.index), pd.Series(0, index=df.index)

def calculate_stochastic(df, k_period=14, d_period=3):
    if df is None or len(df) < k_period + d_period:
        return pd.Series(50, index=df.index if df is not None else [])
    try:
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
        d = k.rolling(window=d_period).mean()
        return d.fillna(50)
    except:
        return pd.Series(50, index=df.index)

def calculate_adx(df, period=14):
    if df is None or len(df) < period * 2:
        return pd.Series(25, index=df.index if df is not None else [])
    try:
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        plus_dm = high.diff()
        minus_dm = low.diff().abs()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.rolling(window=period).mean()
        return adx.fillna(25)
    except:
        return pd.Series(25, index=df.index)

def calculate_atr(df, period=14):
    if df is None or len(df) < period + 5:
        return pd.Series(1, index=df.index if df is not None else [])
    try:
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().fillna(1)
    except:
        return pd.Series(1, index=df.index)

def calculate_bollinger_bands(df, period=20, std_dev=2):
    if df is None or len(df) < period:
        return pd.Series(0, index=df.index if df is not None else [])
    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        # Return position relative to bands
        bb_position = (close - lower_band) / (upper_band - lower_band).replace(0, np.nan)
        return bb_position.fillna(0.5)
    except:
        return pd.Series(0.5, index=df.index)

def calculate_cci(df, period=20):
    if df is None or len(df) < period:
        return pd.Series(0, index=df.index if df is not None else [])
    try:
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = (tp - sma_tp).abs().rolling(window=period).mean()
        cci = (tp - sma_tp) / (0.015 * mad).replace(0, np.nan)
        return cci.fillna(0)
    except:
        return pd.Series(0, index=df.index)

def calculate_roc(df, period=12):
    if df is None or len(df) < period + 2:
        return pd.Series(0, index=df.index if df is not None else [])
    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        roc = ((close - close.shift(period)) / close.shift(period).replace(0, np.nan)) * 100
        return roc.fillna(0)
    except:
        return pd.Series(0, index=df.index)

def calculate_williams_r(df, period=14):
    if df is None or len(df) < period:
        return pd.Series(-50, index=df.index if df is not None else [])
    try:
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        wr = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)
        return wr.fillna(-50)
    except:
        return pd.Series(-50, index=df.index)

def calculate_vwap(df):
    if df is None or len(df) < 5:
        return pd.Series(0, index=df.index if df is not None else [])
    try:
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        tp = (high + low + close) / 3
        vwap = (tp * volume).cumsum() / volume.cumsum().replace(0, np.nan)
        return vwap.fillna(close)
    except:
        return pd.Series(0, index=df.index)

def calculate_ema(df, period=9):
    if df is None or len(df) < period:
        return pd.Series(0, index=df.index if df is not None else [])
    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        return close.ewm(span=period, adjust=False).mean().fillna(close)
    except:
        return pd.Series(0, index=df.index)

def calculate_sma(df, period=20):
    if df is None or len(df) < period:
        return pd.Series(0, index=df.index if df is not None else [])
    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        return close.rolling(window=period).mean().fillna(close)
    except:
        return pd.Series(0, index=df.index)

def calculate_obv(df):
    if df is None or len(df) < 2:
        return pd.Series(0, index=df.index if df is not None else [])
    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv
    except:
        return pd.Series(0, index=df.index)

def calculate_cmf(df, period=20):
    if df is None or len(df) < period:
        return pd.Series(0, index=df.index if df is not None else [])
    try:
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        mfv = mfm * volume
        cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum().replace(0, np.nan)
        return cmf.fillna(0)
    except:
        return pd.Series(0, index=df.index)

def calculate_adl(df):
    if df is None or len(df) < 2:
        return pd.Series(0, index=df.index if df is not None else [])
    try:
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        mfm = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
        mfv = mfm * volume
        adl = mfv.fillna(0).cumsum()
        return adl
    except:
        return pd.Series(0, index=df.index)

def volume_surge_improved(df, lookback=20):
    if df is None or len(df) < lookback + 5:
        return pd.Series(0, index=df.index if df is not None else [])
    try:
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        vol_ma = volume.rolling(lookback, min_periods=lookback//2).mean()
        vol_std = volume.rolling(lookback, min_periods=lookback//2).std()
        vol_std = vol_std.where(vol_std > vol_ma * 0.01, vol_ma * 0.1)
        return ((volume - vol_ma) / vol_std).clip(-5, 5).fillna(0)
    except:
        return pd.Series(0, index=df.index)

def momentum_improved(df, period=10):
    if df is None or len(df) < period + 2:
        return pd.Series(0, index=df.index if df is not None else [])
    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        shifted_close = close.shift(period).replace(0, np.nan)
        return ((close / shifted_close) - 1.0).fillna(0)
    except:
        return pd.Series(0, index=df.index)

def calculate_volume_sma_filter(df, period=20, min_multiplier=5.0):
    if df is None or len(df) < period + 5:
        return False, 0.0, 0.0
    try:
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        vol_sma = volume.rolling(period, min_periods=period//2).mean()
        if vol_sma.empty or vol_sma.iloc[-1] == 0:
            return False, 0.0, 0.0
        current_sma, baseline_volume = vol_sma.iloc[-1], volume.median()
        min_threshold = baseline_volume * min_multiplier
        return current_sma >= min_threshold, current_sma, min_threshold
    except:
        return False, 0.0, 0.0

# ========== SIGNAL CLASSIFICATION ==========
def classify_option_signal(normalized_score, oi_status, has_strong_conditions):
    if normalized_score >= 35:
        return "🚀 ULTRA STRONG BUY - CALL HEAVY", "ULTRA_STRONG"
    elif normalized_score <= -35:
        return "💥 ULTRA STRONG SELL - PUT HEAVY", "ULTRA_STRONG"
    elif normalized_score >= 20:
        return ("🔥 VERY STRONG BUY - CALL FOCUS" if "Call" in oi_status else "🔥 VERY STRONG BUY"), "VERY_STRONG"
    elif normalized_score <= -20:
        return ("🔥 VERY STRONG SELL - PUT FOCUS" if "Put" in oi_status else "🔥 VERY STRONG SELL"), "VERY_STRONG"
    elif normalized_score >= 12:
        return ("⚡ STRONG BUY - OI SURGE" if "OI Building" in oi_status else "⚡ STRONG BUY"), "STRONG"
    elif normalized_score <= -12:
        return ("⚡ STRONG SELL - OI SURGE" if "OI Building" in oi_status else "⚡ STRONG SELL"), "STRONG"
    elif normalized_score >= 5:
        return "🟢 BUY - Call Potential", "MODERATE"
    elif normalized_score <= -5:
        return "🔴 SELL - Put Potential", "MODERATE"
    return "⚪ NEUTRAL", "NEUTRAL"

def get_option_action(signal_strength, normalized_score):
    actions = {
        "ULTRA_STRONG": ("🚨 BUY CALLS AGGRESSIVELY", "URGENT") if normalized_score > 0 else ("🚨 BUY PUTS AGGRESSIVELY", "URGENT"),
        "VERY_STRONG": ("🔥 BUY CALLS STRONG", "HIGH") if normalized_score > 0 else ("🔥 BUY PUTS STRONG", "HIGH"),
        "STRONG": ("⚡ BUY CALLS", "MEDIUM") if normalized_score > 0 else ("⚡ BUY PUTS", "MEDIUM"),
        "MODERATE": ("📈 Consider Calls", "LOW") if normalized_score > 0 else ("📉 Consider Puts", "LOW"),
    }
    return actions.get(signal_strength, ("⏸️ WAIT", "NONE"))

def validate_data_quality(df, min_bars=20):
    if df is None or df.empty or len(df) < min_bars:
        return False, "No data"
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            return False, f"Missing {col}"
        if (df[col].isna().sum() / len(df)) * 100 > CONFIG['MAX_MISSING_DATA_PCT']:
            return False, f"Too much missing"
    return True, "OK"

def normalize_hist_df_clean(df, symbol, timeframe_minutes):
    if df is None or df.empty:
        return None
    try:
        out = df.copy()
        out.columns = out.columns.str.lower().str.strip()
        rename_map = {}
        for col in out.columns:
            col_clean = col.lower().strip()
            if any(x in col_clean for x in ['time', 'date', 'timestamp']):
                rename_map[col] = 'Timestamp'
            elif col_clean in ['open'] or (col_clean.startswith('open') and 'interest' not in col_clean):
                rename_map[col] = 'Open'
            elif col_clean in ['high', 'h']:
                rename_map[col] = 'High'
            elif col_clean in ['low', 'l']:
                rename_map[col] = 'Low'
            elif col_clean in ['close', 'c']:
                rename_map[col] = 'Close'
            elif col_clean in ['volume', 'vol', 'v']:
                rename_map[col] = 'Volume'
        out.rename(columns=rename_map, inplace=True)
        if "Timestamp" not in out.columns:
            if hasattr(out.index, 'dtype') and 'datetime' in str(out.index.dtype):
                out["Timestamp"] = out.index
                out = out.reset_index(drop=True)
            else:
                now = datetime.now(IST)
                out["Timestamp"] = pd.date_range(start=now - timedelta(minutes=timeframe_minutes * len(out)),
                    periods=len(out), freq=f"{timeframe_minutes}T", tz=IST)
        if any(col not in out.columns for col in ["Open", "High", "Low", "Close"]):
            return None
        if "Volume" not in out.columns:
            out["Volume"] = 1000
        out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
        out = out.dropna(subset=["Timestamp"])
        if out.empty:
            return None
        if out["Timestamp"].dt.tz is None:
            out["Timestamp"] = out["Timestamp"].dt.tz_localize(IST, ambiguous='NaT', nonexistent='NaT')
            out = out.dropna(subset=["Timestamp"])
        else:
            out["Timestamp"] = out["Timestamp"].dt.tz_convert(IST)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["Open", "High", "Low", "Close"])
        if out.empty or len(out) < 10:
            return None
        out = out.sort_values("Timestamp").set_index("Timestamp")
        return out[~out.index.duplicated(keep='last')]
    except:
        return None

def pick_session(symbol_orig, timeframe_minutes):
    return hash((symbol_orig, timeframe_minutes)) & 0x7fffffff % len(tdhist_pool)
# ========== ENHANCED ANALYSIS WITH ALL INDICATORS ==========
def analyze_signals_enhanced_with_options(timeframe_dataframes, symbol):
    if not timeframe_dataframes:
        return 'Neutral', 0.0, 'Normal', 'WAIT', 'NONE', None
    
    # Volume SMA filter
    volume_filter_passed = False
    for tf_min, df in timeframe_dataframes.items():
        if df is None or len(df) < 20:
            continue
        meets_vol_threshold, _, _ = calculate_volume_sma_filter(df, period=20, min_multiplier=5.0)
        if meets_vol_threshold:
            volume_filter_passed = True
            break
    
    if not volume_filter_passed:
        return 'Neutral - Low Volume', 0.0, 'Insufficient Volume', 'WAIT', 'NONE', None
    
    # Fetch REAL option chain
    option_chain_analysis = analyze_option_chain(symbol)
    
    final_score, max_possible, valid_timeframes = 0.0, 0.0, 0
    oi_status, has_strong_conditions = 'Normal', False
    
    for tf_min, df in timeframe_dataframes.items():
        if df is None or len(df) < 20:
            continue
        
        is_valid, _ = validate_data_quality(df, 20)
        if not is_valid:
            continue
        
        valid_timeframes += 1
        tf_weight = CONFIG["TIMEFRAME_WEIGHTS"].get(tf_min, 1.0)
        scores = {}
        
        # RSI
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
        except:
            scores['RSI'] = 0.0
        
        # MACD
        try:
            macd_line, signal_line = calculate_macd(df)
            if len(macd_line) >= 2:
                if macd_line.iloc[-1] > signal_line.iloc[-1] and macd_line.iloc[-2] <= signal_line.iloc[-2]:
                    scores['MACD'] = 2.0
                elif macd_line.iloc[-1] > signal_line.iloc[-1]:
                    scores['MACD'] = 1.0
                elif macd_line.iloc[-1] < signal_line.iloc[-1] and macd_line.iloc[-2] >= signal_line.iloc[-2]:
                    scores['MACD'] = -2.0
                elif macd_line.iloc[-1] < signal_line.iloc[-1]:
                    scores['MACD'] = -1.0
                else:
                    scores['MACD'] = 0.0
            else:
                scores['MACD'] = 0.0
        except:
            scores['MACD'] = 0.0
        
        # Stochastic
        try:
            stoch = calculate_stochastic(df)
            if len(stoch) >= 2:
                if stoch.iloc[-1] > 80:
                    scores['Stochastic'] = -1.0
                elif stoch.iloc[-1] > 60:
                    scores['Stochastic'] = 1.0
                elif stoch.iloc[-1] < 20:
                    scores['Stochastic'] = -1.0
                elif stoch.iloc[-1] < 40:
                    scores['Stochastic'] = 1.0
                else:
                    scores['Stochastic'] = 0.0
            else:
                scores['Stochastic'] = 0.0
        except:
            scores['Stochastic'] = 0.0
        
        # ADX
        try:
            adx = calculate_adx(df)
            if len(adx) >= 2:
                adx_val = adx.iloc[-1]
                if adx_val > 25:
                    scores['ADX'] = 2.0 if df['Close'].iloc[-1] > df['Close'].iloc[-2] else -2.0
                elif adx_val > 20:
                    scores['ADX'] = 1.0 if df['Close'].iloc[-1] > df['Close'].iloc[-2] else -1.0
                else:
                    scores['ADX'] = 0.0
            else:
                scores['ADX'] = 0.0
        except:
            scores['ADX'] = 0.0
        
        # ATR
        try:
            atr = calculate_atr(df)
            if len(atr) >= 2:
                atr_ratio = atr.iloc[-1] / atr.mean() if atr.mean() > 0 else 1
                if atr_ratio > 1.5:
                    scores['ATR'] = 1.5
                elif atr_ratio > 1.2:
                    scores['ATR'] = 1.0
                else:
                    scores['ATR'] = 0.0
            else:
                scores['ATR'] = 0.0
        except:
            scores['ATR'] = 0.0
        
        # Bollinger Bands
        try:
            bb_pos = calculate_bollinger_bands(df)
            if len(bb_pos) >= 1:
                pos = bb_pos.iloc[-1]
                if pos > 0.9:
                    scores['Bollinger'] = -1.0
                elif pos > 0.7:
                    scores['Bollinger'] = 1.0
                elif pos < 0.1:
                    scores['Bollinger'] = -1.0
                elif pos < 0.3:
                    scores['Bollinger'] = 1.0
                else:
                    scores['Bollinger'] = 0.0
            else:
                scores['Bollinger'] = 0.0
        except:
            scores['Bollinger'] = 0.0
        
        # CCI
        try:
            cci = calculate_cci(df)
            if len(cci) >= 2:
                cci_val = cci.iloc[-1]
                if cci_val > 100:
                    scores['CCI'] = 2.0
                elif cci_val > 0:
                    scores['CCI'] = 1.0
                elif cci_val < -100:
                    scores['CCI'] = -2.0
                elif cci_val < 0:
                    scores['CCI'] = -1.0
                else:
                    scores['CCI'] = 0.0
            else:
                scores['CCI'] = 0.0
        except:
            scores['CCI'] = 0.0
        
        # ROC
        try:
            roc = calculate_roc(df)
            if len(roc) >= 2:
                roc_val = roc.iloc[-1]
                if roc_val > 5:
                    scores['ROC'] = 2.0
                elif roc_val > 2:
                    scores['ROC'] = 1.0
                elif roc_val < -5:
                    scores['ROC'] = -2.0
                elif roc_val < -2:
                    scores['ROC'] = -1.0
                else:
                    scores['ROC'] = 0.0
            else:
                scores['ROC'] = 0.0
        except:
            scores['ROC'] = 0.0
        
        # Williams %R
        try:
            wr = calculate_williams_r(df)
            if len(wr) >= 2:
                wr_val = wr.iloc[-1]
                if wr_val > -20:
                    scores['WWL'] = -1.0
                elif wr_val > -50:
                    scores['WWL'] = 1.0
                elif wr_val < -80:
                    scores['WWL'] = 2.0
                else:
                    scores['WWL'] = 0.0
            else:
                scores['WWL'] = 0.0
        except:
            scores['WWL'] = 0.0
        
        # VWAP
        try:
            vwap = calculate_vwap(df)
            close = df['Close']
            if len(vwap) >= 2 and len(close) >= 2:
                if close.iloc[-1] > vwap.iloc[-1]:
                    scores['VWAP'] = 1.5
                elif close.iloc[-1] < vwap.iloc[-1]:
                    scores['VWAP'] = -1.5
                else:
                    scores['VWAP'] = 0.0
            else:
                scores['VWAP'] = 0.0
        except:
            scores['VWAP'] = 0.0
        
        # EMA
        try:
            ema_fast = calculate_ema(df, CONFIG['INDICATOR_PERIODS']['EMA_FAST'])
            ema_slow = calculate_ema(df, CONFIG['INDICATOR_PERIODS']['EMA_SLOW'])
            if len(ema_fast) >= 2 and len(ema_slow) >= 2:
                if ema_fast.iloc[-1] > ema_slow.iloc[-1] and ema_fast.iloc[-2] <= ema_slow.iloc[-2]:
                    scores['EMA'] = 2.0
                elif ema_fast.iloc[-1] > ema_slow.iloc[-1]:
                    scores['EMA'] = 1.0
                elif ema_fast.iloc[-1] < ema_slow.iloc[-1] and ema_fast.iloc[-2] >= ema_slow.iloc[-2]:
                    scores['EMA'] = -2.0
                elif ema_fast.iloc[-1] < ema_slow.iloc[-1]:
                    scores['EMA'] = -1.0
                else:
                    scores['EMA'] = 0.0
            else:
                scores['EMA'] = 0.0
        except:
            scores['EMA'] = 0.0
        
        # SMA
        try:
            sma_short = calculate_sma(df, CONFIG['INDICATOR_PERIODS']['MA_SHORT'])
            sma_long = calculate_sma(df, CONFIG['INDICATOR_PERIODS']['MA_LONG'])
            if len(sma_short) >= 2 and len(sma_long) >= 2:
                if sma_short.iloc[-1] > sma_long.iloc[-1]:
                    scores['MA'] = 1.0
                elif sma_short.iloc[-1] < sma_long.iloc[-1]:
                    scores['MA'] = -1.0
                else:
                    scores['MA'] = 0.0
            else:
                scores['MA'] = 0.0
        except:
            scores['MA'] = 0.0
        
        # OBV
        try:
            obv = calculate_obv(df)
            if len(obv) >= 5:
                obv_trend = obv.iloc[-1] - obv.iloc[-5]
                if obv_trend > 0:
                    scores['OBV'] = 1.5
                elif obv_trend < 0:
                    scores['OBV'] = -1.5
                else:
                    scores['OBV'] = 0.0
            else:
                scores['OBV'] = 0.0
        except:
            scores['OBV'] = 0.0
        
        # CMF
        try:
            cmf = calculate_cmf(df)
            if len(cmf) >= 2:
                cmf_val = cmf.iloc[-1]
                if cmf_val > 0.1:
                    scores['CMF'] = 2.0
                elif cmf_val > 0:
                    scores['CMF'] = 1.0
                elif cmf_val < -0.1:
                    scores['CMF'] = -2.0
                elif cmf_val < 0:
                    scores['CMF'] = -1.0
                else:
                    scores['CMF'] = 0.0
            else:
                scores['CMF'] = 0.0
        except:
            scores['CMF'] = 0.0
        
        # ADL
        try:
            adl = calculate_adl(df)
            if len(adl) >= 5:
                adl_trend = adl.iloc[-1] - adl.iloc[-5]
                if adl_trend > 0:
                    scores['ADL'] = 1.5
                elif adl_trend < 0:
                    scores['ADL'] = -1.5
                else:
                    scores['ADL'] = 0.0
            else:
                scores['ADL'] = 0.0
        except:
            scores['ADL'] = 0.0
        
        # Momentum
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
        except:
            scores['Momentum'] = 0.0
        
        # Volume Surge
        try:
            vol_surge = volume_surge_improved(df)
            if len(vol_surge) >= 2:
                current_surge = vol_surge.iloc[-1]
                price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1
                if current_surge >= 1.5:
                    scores['VolumeSurge'] = 2.0 if price_change > 0.005 else (-2.0 if price_change < -0.005 else 1.0)
                elif current_surge >= 1.0:
                    scores['VolumeSurge'] = 1.0 if price_change > 0 else -1.0
                else:
                    scores['VolumeSurge'] = 0.0
            else:
                scores['VolumeSurge'] = 0.0
        except:
            scores['VolumeSurge'] = 0.0
        
        # RelVol, VWAPRegime, OBVConfirm - simplified scoring
        scores['RelVol'] = 0.0
        scores['VWAPRegime'] = 0.0
        scores['OBVConfirm'] = 0.0
        
        # REAL OPTION CHAIN INTEGRATION
        if option_chain_analysis:
            scores['PCR'] = option_chain_analysis['signal_strength']
            
            if option_chain_analysis['call_oi_change'] > option_chain_analysis['put_oi_change'] * 1.5:
                scores['CallOIBuild'] = 3.0
            elif option_chain_analysis['call_oi_change'] > option_chain_analysis['put_oi_change']:
                scores['CallOIBuild'] = 1.5
            elif option_chain_analysis['put_oi_change'] > option_chain_analysis['call_oi_change'] * 1.5:
                scores['CallOIBuild'] = -3.0
            elif option_chain_analysis['put_oi_change'] > option_chain_analysis['call_oi_change']:
                scores['CallOIBuild'] = -1.5
            else:
                scores['CallOIBuild'] = 0.0
            
            scores['PutOIBuild'] = -scores['CallOIBuild']
            
            if option_chain_analysis['total_call_volume'] > option_chain_analysis['total_put_volume'] * 1.3:
                scores['CallVolume'], scores['PutVolume'] = 2.0, 0.0
            elif option_chain_analysis['total_put_volume'] > option_chain_analysis['total_call_volume'] * 1.3:
                scores['CallVolume'], scores['PutVolume'] = 0.0, -2.0
            else:
                scores['CallVolume'], scores['PutVolume'] = 0.0, 0.0
            
            if option_chain_analysis['oi_change_ratio'] > 1.5:
                scores['OIChangeRatio'] = 2.0
            elif option_chain_analysis['oi_change_ratio'] < 0.5:
                scores['OIChangeRatio'] = -2.0
            else:
                scores['OIChangeRatio'] = 0.0
            
            if option_chain_analysis['call_net_buy'] > option_chain_analysis['put_net_buy'] * 1.5:
                scores['NetBuyPressure'] = 3.0
            elif option_chain_analysis['put_net_buy'] > option_chain_analysis['call_net_buy'] * 1.5:
                scores['NetBuyPressure'] = -3.0
            else:
                scores['NetBuyPressure'] = 0.0
            
            oi_status = option_chain_analysis['oi_status']
            if option_chain_analysis['signal'] in ['CALL_HEAVY', 'PUT_HEAVY']:
                has_strong_conditions = True
        else:
            for k in ['PCR', 'CallOIBuild', 'PutOIBuild', 'CallVolume', 'PutVolume', 'OIChangeRatio', 'NetBuyPressure']:
                scores[k] = 0.0
        
        # Calculate weighted scores
        for indicator, score in scores.items():
            ind_weight = CONFIG["INDICATOR_WEIGHTS"].get(indicator, 1.0)
            final_score += score * tf_weight * ind_weight
            max_possible += 3.0 * tf_weight * ind_weight
    
    if valid_timeframes < 1 or max_possible == 0:
        return 'Neutral', 0.0, oi_status, 'WAIT', 'NONE', option_chain_analysis
    
    normalized = np.clip((final_score / max_possible) * 100.0, -100, 100)
    signal_text, signal_strength = classify_option_signal(normalized, oi_status, has_strong_conditions)
    option_action, alert_priority = get_option_action(signal_strength, normalized)
    
    return signal_text, normalized, oi_status, option_action, alert_priority, option_chain_analysis

# ========== DATA FETCHING ==========
@retry(stop_max_attempt_number=CONFIG["RETRY_ATTEMPTS"], 
       wait_exponential_multiplier=max(1, CONFIG["RETRY_DELAY_MS"]//2),
       wait_exponential_max=10000, retry_on_exception=lambda e: True)
def fetch_one_clean(symbol_orig, timeframe_minutes, limiter, hist):
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
        global api_calls_done
        with api_calls_lock:
            api_calls_done += 1
        return symbol_orig, timeframe_minutes, df
    except:
        return symbol_orig, timeframe_minutes, None

def prefetch_clean(stocks, max_workers=CONFIG["MAX_WORKERS"]):
    tfs, stock_multi_data = [5, 15, 30, 60, 1440], defaultdict(dict)
    total_calls = len(stocks) * len(tfs)
    global api_calls_done
    with api_calls_lock:
        api_calls_done = 0
    valid_stocks = [s for s in stocks if s]
    console.print(f"📊 Analyzing [cyan]{len(valid_stocks)} symbols[/cyan]")
    with tqdm(total=total_calls, desc="🔄 Loading", ncols=80, disable=not CONFIG["SHOW_PROGRESS"]) as pbar:
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
                except:
                    pass
                pbar.update(1)
    valid_data = {s: d for s, d in stock_multi_data.items() if len(d) >= 1}
    console.print(f"✅ Loaded: [green]{len(valid_data)} symbols[/green]")
    return valid_data

def filter_timeframe_data(symbol, timeframe_data, time_point_aware):
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
            valid_index = df.index.dropna()
            if len(valid_index) != len(df.index):
                df = df.loc[valid_index]
            if not df.empty:
                df_filtered = df.loc[df.index <= time_point_aware]
                if len(df_filtered) >= CONFIG["MIN_BARS_REQUIRED"]:
                    filtered_timeframes[tf] = df_filtered
        except:
            continue
    return filtered_timeframes

# ========== RENDERING ==========
def render_signals_with_options(now_ts, top_bullish, top_bearish):
    global last_bull_symbols, last_bear_symbols
    console.rule(f"🎯 SIGNALS | {now_ts.strftime('%H:%M')} IST", style="bold yellow")
    all_strong = [r for r in top_bullish + top_bearish if any(x in r['signal'] for x in ['ULTRA', 'VERY', 'STRONG'])]
    all_moderate = [r for r in top_bullish + top_bearish if '🟢' in r['signal'] or '🔴' in r['signal']]
    
    if all_strong:
        console.print("\n🔥 [bold white on red]STRONG SIGNALS[/bold white on red]")
        t = Table(title="💪 STRONG", box=box.DOUBLE_EDGE, header_style="bold white on blue")
        t.add_column("Stock", style="bold white", width=12)
        t.add_column("Signal", style="bold yellow", width=30)
        t.add_column("Score", style="bold green", justify="right", width=8)
        t.add_column("PCR", style="magenta", justify="right", width=7)
        t.add_column("OI Status", style="cyan", width=25)
        t.add_column("Action", style="bold red", width=25)
        for r in all_strong:
            row_style = "bold black on yellow" if r['symbol'] not in (last_bull_symbols | last_bear_symbols) else None
            pcr_str = f"{r.get('pcr', 0):.2f}" if r.get('pcr') else "N/A"
            t.add_row(r['symbol'], r['signal'], f"{r['score']:.1f}", pcr_str,
                r.get('oi_status', 'Normal'), r.get('action', 'TRADE'), style=row_style)
        console.print(t)
    
    if all_moderate:
        console.print("\n📊 [bold blue]MODERATE[/bold blue]")
        t = Table(title="📈 MODERATE", box=box.SIMPLE, header_style="bold white on green")
        t.add_column("Stock", style="cyan")
        t.add_column("Signal", style="white")
        t.add_column("Score", style="yellow", justify="right")
        t.add_column("PCR", style="magenta", justify="right")
        for r in all_moderate[:10]:
            pcr_str = f"{r.get('pcr', 0):.2f}" if r.get('pcr') else "N/A"
            t.add_row(r['symbol'], r['signal'], f"{r['score']:.1f}", pcr_str)
        console.print(t)
    
    if option_chain_stats['api_calls'] > 0:
        success_rate = (option_chain_stats['success'] / option_chain_stats['api_calls']) * 100
        console.print(f"📡 Option Chain: [cyan]{option_chain_stats['success']}[/cyan] | Success: [green]{success_rate:.1f}%[/green]")
    console.rule()
    last_bull_symbols = {r['symbol'] for r in top_bullish}
    last_bear_symbols = {r['symbol'] for r in top_bearish}

def export_to_csv_with_options(now_ts, top_bullish, top_bearish, filename):
    with open(filename, "a", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([f"Time: {now_ts.strftime('%Y-%m-%d %H:%M')}"])
        writer.writerow(["Stock", "Signal", "Score", "PCR", "Call_OI", "Put_OI", "OI Status", "Action"])
        for r in top_bullish + top_bearish:
            pcr, call_oi, put_oi = r.get('pcr', 0), r.get('call_oi', 0), r.get('put_oi', 0)
            writer.writerow([r['symbol'], r['signal'], f"{r['score']:.2f}",
                f"{pcr:.2f}" if pcr else "N/A", call_oi, put_oi,
                r.get('oi_status', 'Normal'), r.get('action', 'TRADE')])
        writer.writerow([])

# ========== BACKTEST ==========
def run_backtest_clean(day_str, stocks):
    global previous_scores, last_bull_symbols, last_bear_symbols, performance_metrics
    
    day_date = datetime.strptime(day_str, "%Y-%m-%d")
    console.print(f"📅 [bold cyan]Backtesting {day_str}[/bold cyan]")
    
    stock_multi_data = prefetch_clean(stocks)
    if not stock_multi_data:
        console.print("[red]❌ No data[/red]")
        return
    
    checkpoints = day_checkpoints_ist(day_date)
    output_filename = day_date.strftime("%Y-%m-%d") + "_signals_options.csv"
    
    if os.path.exists(output_filename):
        try:
            os.remove(output_filename)
        except:
            pass
    
    previous_scores, last_bull_symbols, last_bear_symbols = {}, set(), set()
    performance_metrics = defaultdict(int)
    
    console.print(f"🔍 Analyzing [cyan]{len(checkpoints)}[/cyan] periods...")
    
    for i, asof_ts in enumerate(checkpoints):
        if i % 20 == 0:
            console.print(f"⏳ {i+1}/{len(checkpoints)} | {asof_ts.strftime('%H:%M')}")
        
        time_point_aware = asof_ts.replace(second=0, microsecond=0)
        signals_this_scan, current_scores = [], {}
        
        for symbol, timeframe_data in stock_multi_data.items():
            clean_symbol = symbol.replace('-EQ', '')
            filtered_timeframes = filter_timeframe_data(clean_symbol, timeframe_data, time_point_aware)
            
            if len(filtered_timeframes) < 1:
                continue
            
            signal, score, oi_status, option_action, alert_priority, option_data = \
                analyze_signals_enhanced_with_options(filtered_timeframes, clean_symbol)
            
            current_scores[clean_symbol] = score
            
            if abs(score) >= CONFIG['MIN_SIGNAL_THRESHOLD'] or any(x in signal for x in ['STRONG', 'BUY', 'SELL']):
                prev = previous_scores.get(clean_symbol, 'NA')
                change_val = 'NA' if isinstance(prev, str) else (score - prev)
                direction = 'bullish' if score > 0 else 'bearish'
                
                signal_record = {
                    'symbol': clean_symbol, 'signal': signal, 'score': score,
                    'trend': direction, 'change': change_val, 'oi_status': oi_status,
                    'action': option_action, 'priority': alert_priority
                }
                
                if option_data:
                    signal_record.update({
                        'pcr': option_data.get('pcr_oi', 0),
                        'call_oi': option_data.get('total_call_oi', 0),
                        'put_oi': option_data.get('total_put_oi', 0)
                    })
                
                signals_this_scan.append(signal_record)
                performance_metrics[f"{direction}_signals"] += 1
        
        previous_scores = current_scores.copy()
        signals_this_scan.sort(key=lambda x: abs(x['score']), reverse=True)
        top_bullish = [r for r in signals_this_scan if r['score'] > 0][:20]
        top_bearish = [r for r in signals_this_scan if r['score'] < 0][:20]
        
        if top_bullish or top_bearish:
            render_signals_with_options(asof_ts, top_bullish, top_bearish)
            export_to_csv_with_options(asof_ts, top_bullish, top_bearish, output_filename)
    
    console.print(f"\n📈 [bold green]BACKTEST COMPLETE[/bold green]")
    console.print(f"Results: [yellow]{output_filename}[/yellow]")

# ========== UTILITIES ==========
def load_stock_list(file_name):
    if not os.path.exists(file_name):
        sample = [
            "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "BHARTIARTL",
            "ITC", "LT", "AXISBANK", "MARUTI", "SUNPHARMA", "WIPRO", "HCLTECH",
            "BAJFINANCE", "TITAN", "NTPC", "ONGC", "TECHM", "TATASTEEL", "HINDALCO",
            "ADANIPORTS", "TATAMOTORS", "JSWSTEEL", "VEDL", "M&M", "KOTAKBANK",
            "ASIANPAINT", "ULTRACEMCO", "NESTLEIND", "BAJAJFINSV", "INDUSINDBK",
            "CIPLA", "COALINDIA", "GRASIM", "BPCL", "BRITANNIA", "DIVISLAB",
            "HEROMOTOCO", "SHREECEM", "UPL", "APOLLOHOSP", "BAJAJ-AUTO",
            "EICHERMOT", "SBILIFE", "HDFCLIFE", "POWERGRID", "DRREDDY"
        ]
        try:
            with open(file_name, "w") as f:
                f.write("\n".join(sample))
            console.print(f"📝 Created [yellow]{file_name}[/yellow]")
            return sample
        except:
            return []
    
    try:
        with open(file_name, "r") as f:
            stocks = [line.strip().split(',')[0].strip().upper() 
                for line in f if line.strip() and not line.startswith('#')]
        console.print(f"📈 Loaded [cyan]{len(stocks)}[/cyan] symbols")
        return stocks
    except:
        return []

def print_banner():
    console.print("\n" + "="*80, style="bold blue")
    console.print("🎯 [bold cyan]NSE OPTION SCANNER v4.0 - ALL INDICATORS[/bold cyan] 🎯", justify="center")
    console.print("="*80, style="bold blue")
    console.print(f"🕐 {datetime.now(IST).strftime('%H:%M:%S IST')}")
    console.print(f"📡 [yellow]{CONFIG['OPTION_CHAIN_API']}[/yellow]")
    console.print("="*80, style="bold blue")

# ========== MAIN ==========
if __name__ == "__main__":
    print_banner()
    
    try:
        parser = argparse.ArgumentParser(description="NSE Option Scanner - Complete")
        parser.add_argument("--backtest-date", help="YYYY-MM-DD")
        parser.add_argument("--stocks-file", default=CONFIG["SHARES_FILE"])
        parser.add_argument("--live", action="store_true")
        parser.add_argument("--test-symbol", help="Test single symbol")
        
        args = parser.parse_args()
        
        stocks = load_stock_list(args.stocks_file)
        if not stocks:
            console.print("[red]❌ No stocks[/red]")
            exit(1)
        
        if args.test_symbol:
            console.print(f"🧪 [yellow]Testing {args.test_symbol}[/yellow]")
            opt = analyze_option_chain(args.test_symbol)
            if opt:
                console.print(f"\n✅ [bold green]Option Chain Data:[/bold green]")
                console.print(f"   Symbol: {opt['symbol']}")
                console.print(f"   PCR (OI): [cyan]{opt['pcr_oi']:.2f}[/cyan]")
                console.print(f"   PCR (Vol): [cyan]{opt['pcr_volume']:.2f}[/cyan]")
                console.print(f"   Call OI: [green]{opt['total_call_oi']:,}[/green]")
                console.print(f"   Put OI: [red]{opt['total_put_oi']:,}[/red]")
                console.print(f"   Call Volume: [green]{opt['total_call_volume']:,}[/green]")
                console.print(f"   Put Volume: [red]{opt['total_put_volume']:,}[/red]")
                console.print(f"   Call OI Change: {opt['call_oi_change']:+,}")
                console.print(f"   Put OI Change: {opt['put_oi_change']:+,}")
                console.print(f"   Signal: [bold yellow]{opt['signal']}[/bold yellow]")
                console.print(f"   Strength: [bold cyan]{opt['signal_strength']:.1f}[/bold cyan]")
                console.print(f"   Recommendation: [bold magenta]{opt['recommendation']}[/bold magenta]")
                console.print(f"   Expiry: {opt['expiry']}")
            else:
                console.print("[red]❌ No option chain data[/red]")
                console.print(f"[yellow]⚠️  Ensure API is accessible[/yellow]")
        
        elif args.live:
           # global previous_scores, last_bull_symbols, last_bear_symbols  # ✅ MOVE HERE - FIRST LINE!
            
            console.print("🔴 [bold green]LIVE MODE[/bold green]")
            stocks_list = load_stock_list(args.stocks_file)
            
            now_ist = datetime.now(IST)
            if now_ist.weekday() >= 5:
                console.print("[yellow]⚠️  Weekend - Markets closed[/yellow]")
                exit(0)
            
            market_end = today_ist_dt(CONFIG["MARKET_END"])
            if now_ist >= market_end:
                console.print("[yellow]⚠️  Market closed[/yellow]")
                exit(0)
            
            first_run = today_ist_dt(CONFIG["FIRST_RUN_AT"])
            if now_ist < first_run:
                console.print(f"⏰ Waiting for {CONFIG['FIRST_RUN_AT']}")
                sleep_until(first_run)
            
            console.print("🔄 [cyan]Fetching initial data...[/cyan]")
            stock_multi_data = prefetch_clean(stocks_list)
            
            if not stock_multi_data:
                console.print("[red]❌ No data[/red]")
                exit(1)

            previous_scores = {}
            last_bull_symbols = set()
            last_bear_symbols = set()
            output_filename = datetime.now(IST).strftime("%Y-%m-%d") + "_live_signals.csv"
            scan_count = 0
            
            console.print("✅ [green]Live scanner started[/green]")
            console.rule("🚀 LIVE SCANNING", style="bold green")
            
            try:
                while True:
                    scan_count += 1
                    now_ist = datetime.now(IST)
                    
                    if now_ist >= today_ist_dt(CONFIG["MARKET_END"]):
                        console.print("\n🔔 [bold yellow]Market closed[/bold yellow]")
                        break
                    
                    next_scan = next_5min_boundary_ist(now_ist)
                    wait_seconds = (next_scan - now_ist).total_seconds()
                    
                    if wait_seconds > 0:
                        console.print(f"⏳ Scan #{scan_count} | Next: [yellow]{next_scan.strftime('%H:%M:%S')}[/yellow]")
                        sleep_until(next_scan)
                    
                    time.sleep(CONFIG["SETTLE_DELAY_SECONDS"])
                    
                    console.print(f"\n🔍 [bold cyan]SCANNING {datetime.now(IST).strftime('%H:%M:%S')}[/bold cyan]")
                    
                    stock_multi_data = prefetch_clean(stocks)
                    
                    if not stock_multi_data:
                        console.print("[red]⚠️  No data, retry[/red]")
                        continue
                    
                    time_point_aware = datetime.now(IST).replace(second=0, microsecond=0)
                    signals_this_scan, current_scores = [], {}
                    
                    for symbol, timeframe_data in stock_multi_data.items():
                        clean_symbol = symbol.replace('-EQ', '')
                        filtered_timeframes = filter_timeframe_data(clean_symbol, timeframe_data, time_point_aware)
                        
                        if len(filtered_timeframes) < 1:
                            continue
                        
                        signal, score, oi_status, option_action, alert_priority, option_data = \
                            analyze_signals_enhanced_with_options(filtered_timeframes, clean_symbol)
                        
                        current_scores[clean_symbol] = score
                        
                        if abs(score) >= CONFIG['MIN_SIGNAL_THRESHOLD'] or \
                           any(x in signal for x in ['STRONG', 'BUY', 'SELL']):
                            prev = previous_scores.get(clean_symbol, 'NA')
                            change_val = 'NA' if isinstance(prev, str) else (score - prev)
                            
                            signal_record = {
                                'symbol': clean_symbol, 'signal': signal, 'score': score,
                                'trend': 'bullish' if score > 0 else 'bearish',
                                'change': change_val, 'oi_status': oi_status,
                                'action': option_action, 'priority': alert_priority
                            }
                            
                            if option_data:
                                signal_record.update({
                                    'pcr': option_data.get('pcr_oi', 0),
                                    'call_oi': option_data.get('total_call_oi', 0),
                                    'put_oi': option_data.get('total_put_oi', 0)
                                })
                            
                            signals_this_scan.append(signal_record)
                    
                    previous_scores = current_scores.copy()
                    
                    signals_this_scan.sort(key=lambda x: abs(x['score']), reverse=True)
                    top_bullish = [r for r in signals_this_scan if r['score'] > 0][:20]
                    top_bearish = [r for r in signals_this_scan if r['score'] < 0][:20]
                    
                    if top_bullish or top_bearish:
                        render_signals_with_options(time_point_aware, top_bullish, top_bearish)
                        export_to_csv_with_options(time_point_aware, top_bullish, top_bearish, output_filename)
                        console.print(f"💾 Saved: [yellow]{output_filename}[/yellow]")
                    else:
                        console.print("⚪ [white]No strong signals[/white]")
                    
                    console.rule(style="dim")
            
            except KeyboardInterrupt:
                console.print("\n[yellow]👤 Stopped by user[/yellow]")
            
            console.print(f"\n📊 [bold green]LIVE SESSION SUMMARY[/bold green]")
            console.print(f"Total scans: [cyan]{scan_count}[/cyan]")
            console.print(f"Results: [yellow]{output_filename}[/yellow]")
            
            if option_chain_stats['api_calls'] > 0:
                console.print(f"\n📡 [bold cyan]Option Chain Stats:[/bold cyan]")
                console.print(f"   Total calls: {option_chain_stats['api_calls']}")
                console.print(f"   Success: [green]{option_chain_stats['success']}[/green]")
                console.print(f"   Cache hits: [yellow]{option_chain_stats.get('cache_hits', 0)}[/yellow]")
        
        elif args.backtest_date:
            try:
                datetime.strptime(args.backtest_date, "%Y-%m-%d")
                run_backtest_clean(args.backtest_date, stocks)
            except ValueError:
                console.print("[red]❌ Invalid date format. Use YYYY-MM-DD[/red]")
        
        else:
            console.print("\n🎯 [bold green]NSE Option Scanner v4.0[/bold green]")
            console.print("\n[cyan]Usage:[/cyan]")
            console.print("  [yellow]python scanner.py --live[/yellow]                      # Live mode")
            console.print("  [yellow]python scanner.py --backtest-date 2025-10-10[/yellow]  # Backtest")
            console.print("  [yellow]python scanner.py --test-symbol VEDL[/yellow]          # Test option chain")
            console.print("\n[cyan]Features:[/cyan]")
            console.print("  ✅ 25+ Technical Indicators")
            console.print("  ✅ Real Option Chain (PCR, OI, Volume)")
            console.print("  ✅ Volume SMA 5x Filter")
            console.print("  ✅ Multi-timeframe Analysis")
            console.print("  ✅ Live Every 5 Minutes")
            console.print("  ✅ CSV Export with Metrics")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]👤 Stopped[/yellow]")
    except Exception as e:
        console.print(f"[red]💥 Error: {e}[/red]")
        import traceback
        traceback.print_exc()
    finally:
        for sess in tdhist_pool:
            try:
                sess.disconnect()
            except:
                pass
        
        if performance_metrics:
            total = sum(performance_metrics.values())
            if total > 0:
                console.print(f"\n📊 [bold green]Total signals: {total}[/bold green]")
        
        console.print("✅ [green]Shutdown complete[/green]")

