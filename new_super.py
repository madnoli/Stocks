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

# ======== ENHANCED CONFIG FOR EXPERIENCED OPTION BUYER ========
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
    "SKIP_DAILY": False,

    # Analysis settings - optimized thresholds for option trading
    "MIN_BARS_REQUIRED": 20,
    "MAX_MISSING_DATA_PCT": 15,
    "SIGNAL_CONFIRMATION_BARS": 2,
    "MIN_SIGNAL_THRESHOLD": 8,

    # Enhanced indicator periods for option trading
    "INDICATOR_PERIODS": {
        "RSI": 14, "MACD_FAST": 12, "MACD_SLOW": 26, "MACD_SIGNAL": 9,
        "STOCHASTIC_K": 14, "STOCHASTIC_D": 3, "MA_SHORT": 20, "MA_LONG": 50,
        "ADX": 14, "BB_PERIOD": 20, "BB_STD_DEV": 2, "ROC": 12, "CCI": 20,
        "EMA_FAST": 9, "EMA_SLOW": 21, "ATR": 14, "VOLUME_SURGE": 20,
        "MOMENTUM": 10, "WILLIAMS_R": 14, "CMF": 20, "ADL_LOOKBACK": 10,
        "REL_VOL": 20, "VWAP_REGIME": 20, "OBV_CONFIRM": 5,
        "OI_SURGE": 20, "OI_MOMENTUM": 10, "PRICE_VELOCITY": 5,
        "GAMMA_SQUEEZE": 3, "IV_CRUSH": 10, "OPTION_FLOW": 15,
        "SUPERTREND_ATR": 10, "SUPERTREND_MULT": 3.0
    },

    # ======== EXPERIENCED OPTION BUYER WEIGHTS ========
    "INDICATOR_WEIGHTS": {
        "SuperTrendKDE": 6.0, "SuperTrendBreak": 5.5, "VolumeKDE": 5.0,
        "OISurge": 4.5, "OIMomentum": 4.0, "VolumeSurge": 4.2, "OIVolConfirm": 4.0,
        "CallBias": 4.8, "PutBias": 4.8, "OptionFlow": 4.5,
        "Momentum": 3.8, "PriceVelocity": 3.5, "ADX": 3.2, "GammaSqueezeRisk": 3.0, "VWAP": 2.8,
        "EMA": 2.5, "MACD": 2.3, "ATR": 2.2, "RSI": 2.0, "Bollinger": 1.8, "Stochastic": 1.5,
        "CMF": 2.8, "OBV": 2.5, "RelVol": 2.3, "ADL": 2.2, "VWAPRegime": 2.0, "OBVConfirm": 1.8,
        "ROC": 1.8, "CCI": 1.5, "MA": 1.5, "WWL": 1.3, "IVCrushRisk": 2.5,
    },

    # Timeframe weights and duration map
    "TIMEFRAME_WEIGHTS": {5: 3.0, 15: 2.8, 30: 2.2, 60: 1.8, 1440: 1.5},
    "BAR_SIZE_MAP": {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"},
    "DURATION_MAP": {5: "45 D", 15: "45 D", 30: "90 D", 60: "180 D", 1440: "365 D"},

    # SuperTrend KDE Settings
    "SUPERTREND_KDE_THRESHOLD": 0.90,
    "SUPERTREND_BANDWIDTH": 0.10,
    "SUPERTREND_BINS": 100,
    "SUPERTREND_LOOKBACK": 25,
}

# Setup logging and timezone
level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
logging.getLogger().setLevel(level_map.get(CONFIG["LOG_LEVEL"], logging.WARNING))
IST = pytz.timezone("Asia/Kolkata")

# Silence noisy loggers
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
supertrend_bullish_volumes = {}
supertrend_bearish_volumes = {}

# ======== SUPERTREND + KDE FUNCTIONS ========
def gaussian_kernel(distance, bandwidth=1.0):
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * (distance / bandwidth) ** 2)

def kde_probability(arr, bandwidth=0.10, steps=100):
    if len(arr) < 5:
        return None, None
    arr_mean = np.mean(arr)
    arr_std = np.std(arr)
    if arr_std == 0:
        return None, None
    normalized_arr = [(x - arr_mean) / arr_std for x in arr]
    arr_min, arr_max = -3.0, 3.0
    step_size = (arr_max - arr_min) / steps
    x_points = [arr_min + i * step_size for i in range(steps)]
    y_points = []
    for x in x_points:
        density = sum(gaussian_kernel(x - val, bandwidth) for val in normalized_arr)
        y_points.append(density)
    total_density = sum(y_points)
    if total_density > 0:
        y_points = [y / total_density for y in y_points]
    y_cumsum = []
    cumsum = 0.0
    for y in y_points:
        cumsum += y
        y_cumsum.append(cumsum)
    return x_points, y_cumsum

def calculate_supertrend(df, atr_length=10, multiplier=3.0):
    if df is None or len(df) < atr_length + 5:
        return None, None
    try:
        high = pd.to_numeric(df['High'], errors='coerce')
        low = pd.to_numeric(df['Low'], errors='coerce')
        close = pd.to_numeric(df['Close'], errors='coerce')
        tr1 = high - low
        tr2 = np.abs(high - close.shift(1))
        tr3 = np.abs(low - close.shift(1))
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = true_range.rolling(window=atr_length).mean()
        hl2 = (high + low) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        supertrend = pd.Series(index=df.index, dtype='float64')
        direction = pd.Series(index=df.index, dtype='int64')
        for i in range(1, len(df)):
            if pd.isna(upper_band.iloc[i]) or pd.isna(lower_band.iloc[i]):
                continue
            if upper_band.iloc[i] >= upper_band.iloc[i-1] and not (close.iloc[i-1] > upper_band.iloc[i-1]):
                upper_band.iloc[i] = upper_band.iloc[i-1]
            if lower_band.iloc[i] <= lower_band.iloc[i-1] and not (close.iloc[i-1] < lower_band.iloc[i-1]):
                lower_band.iloc[i] = lower_band.iloc[i-1]
            if i == 1:
                direction.iloc[i] = 1 if close.iloc[i] <= upper_band.iloc[i] else -1
            else:
                if close.iloc[i] <= upper_band.iloc[i]:
                    direction.iloc[i] = 1
                elif close.iloc[i] >= lower_band.iloc[i]:
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = direction.iloc[i-1]
            supertrend.iloc[i] = upper_band.iloc[i] if direction.iloc[i] == 1 else lower_band.iloc[i]
        return supertrend, direction
    except Exception:
        return None, None

def calculate_buy_sell_volume(df):
    try:
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        high = pd.to_numeric(df['High'], errors='coerce')
        low = pd.to_numeric(df['Low'], errors='coerce')
        close = pd.to_numeric(df['Close'], errors='coerce')
        range_val = high - low
        range_val = range_val.where(range_val > 0, 0.01)
        buy_volume = volume * (close - low) / range_val
        sell_volume = volume * (high - close) / range_val
        return buy_volume, sell_volume
    except Exception:
        volume = pd.to_numeric(df['Volume'], errors='coerce').fillna(1000)
        return volume * 0.6, volume * 0.4

def supertrend_kde_analysis(df, symbol):
    global supertrend_bullish_volumes, supertrend_bearish_volumes
    if df is None or len(df) < 30:
        return 0.0, 0.0, 0.0, "Normal"
    try:
        supertrend, direction = calculate_supertrend(df, CONFIG["INDICATOR_PERIODS"]["SUPERTREND_ATR"], CONFIG["INDICATOR_PERIODS"]["SUPERTREND_MULT"])
        if supertrend is None or direction is None:
            return 0.0, 0.0, 0.0, "No Data"
        close = pd.to_numeric(df['Close'], errors='coerce')
        buy_volume, sell_volume = calculate_buy_sell_volume(df)
        lookback = CONFIG["SUPERTREND_LOOKBACK"]
        buy_vol_avg = buy_volume.rolling(lookback).mean()
        buy_vol_short = buy_volume.rolling(10).mean()
        sell_vol_avg = sell_volume.rolling(lookback).mean()
        sell_vol_short = sell_volume.rolling(10).mean()
        buy_vol_ratio = buy_vol_short / buy_vol_avg.where(buy_vol_avg > 0, 1)
        sell_vol_ratio = sell_vol_short / sell_vol_avg.where(sell_vol_avg > 0, 1)
        bullish_break = bearish_break = False
        if len(close) >= 2 and len(supertrend) >= 2:
            prev_close = close.iloc[-2]; curr_close = close.iloc[-1]
            prev_st = supertrend.iloc[-2]; curr_st = supertrend.iloc[-1]
            bullish_break = (prev_close <= prev_st) and (curr_close > curr_st)
            bearish_break = (prev_close >= prev_st) and (curr_close < curr_st)
        if symbol not in supertrend_bullish_volumes:
            supertrend_bullish_volumes[symbol] = []
        if symbol not in supertrend_bearish_volumes:
            supertrend_bearish_volumes[symbol] = []
        kde_prob_bull = 0.0; kde_prob_bear = 0.0; supertrend_signal = 0.0; status = "Normal"
        if bullish_break and not pd.isna(buy_vol_ratio.iloc[-1]):
            vol_list = supertrend_bullish_volumes[symbol]; vol_list.append(float(buy_vol_ratio.iloc[-1]))
            if len(vol_list) > CONFIG["SUPERTREND_BINS"]:
                vol_list.pop(0)
            if len(vol_list) >= 10:
                x_points, y_cumsum = kde_probability(vol_list, CONFIG["SUPERTREND_BANDWIDTH"], CONFIG["SUPERTREND_BINS"])
                if x_points and y_cumsum:
                    vol_mean = np.mean(vol_list); vol_std = np.std(vol_list)
                    if vol_std > 0:
                        current_vol_std = (buy_vol_ratio.iloc[-1] - vol_mean) / vol_std
                        nearest_idx = int(np.argmin([abs(x - current_vol_std) for x in x_points]))
                        kde_prob_bull = y_cumsum[nearest_idx]
                        if kde_prob_bull >= CONFIG["SUPERTREND_KDE_THRESHOLD"]:
                            supertrend_signal = 4.0; status = f"🚀 SUPERTREND BULL KDE {kde_prob_bull*100:.1f}%"
                        elif kde_prob_bull >= 0.80:
                            supertrend_signal = 2.5; status = f"⬆️ ST Bull {kde_prob_bull*100:.1f}%"
                        else:
                            supertrend_signal = 1.0; status = f"📈 ST Break {kde_prob_bull*100:.1f}%"
        elif bearish_break and not pd.isna(sell_vol_ratio.iloc[-1]):
            vol_list = supertrend_bearish_volumes[symbol]; vol_list.append(float(sell_vol_ratio.iloc[-1]))
            if len(vol_list) > CONFIG["SUPERTREND_BINS"]:
                vol_list.pop(0)
            if len(vol_list) >= 10:
                x_points, y_cumsum = kde_probability(vol_list, CONFIG["SUPERTREND_BANDWIDTH"], CONFIG["SUPERTREND_BINS"])
                if x_points and y_cumsum:
                    vol_mean = np.mean(vol_list); vol_std = np.std(vol_list)
                    if vol_std > 0:
                        current_vol_std = (sell_vol_ratio.iloc[-1] - vol_mean) / vol_std
                        nearest_idx = int(np.argmin([abs(x - current_vol_std) for x in x_points]))
                        kde_prob_bear = y_cumsum[nearest_idx]
                        if kde_prob_bear >= CONFIG["SUPERTREND_KDE_THRESHOLD"]:
                            supertrend_signal = -4.0; status = f"💥 SUPERTREND BEAR KDE {kde_prob_bear*100:.1f}%"
                        elif kde_prob_bear >= 0.80:
                            supertrend_signal = -2.5; status = f"⬇️ ST Bear {kde_prob_bear*100:.1f}%"
                        else:
                            supertrend_signal = -1.0; status = f"📉 ST Break {kde_prob_bear*100:.1f}%"
        elif len(direction) >= 1:
            current_dir = direction.iloc[-1]
            if current_dir < 0:
                supertrend_signal = 0.5; status = "📈 ST Bullish Trend"
            elif current_dir > 0:
                supertrend_signal = -0.5; status = "📉 ST Bearish Trend"
        return supertrend_signal, kde_prob_bull, kde_prob_bear, status
    except Exception:
        return 0.0, 0.0, 0.0, "Error"

# ======== UNIFIED BOUNDARY FUNCTIONS ========
def get_unified_analysis_cutoff(mode="live", checkpoint_time=None):
    if mode == "backtest" and checkpoint_time:
        return checkpoint_time.replace(second=0, microsecond=0)
    now_ist = datetime.now(IST)
    minute = (now_ist.minute // 5) * 5
    boundary = now_ist.replace(minute=minute, second=0, microsecond=0)
    if now_ist.minute % 5 != 0 or now_ist.second < 30:
        boundary = boundary - timedelta(minutes=5)
    return boundary

def wait_for_next_completed_5min_candle():
    now_ist = datetime.now(IST)
    current_minute = now_ist.minute
    next_5min_mark = ((current_minute // 5) + 1) * 5
    if next_5min_mark >= 60:
        next_boundary = now_ist.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    else:
        next_boundary = now_ist.replace(minute=next_5min_mark, second=0, microsecond=0)
    next_analysis_time = next_boundary + timedelta(seconds=30)
    current_time = datetime.now(IST)
    if next_analysis_time > current_time:
        wait_seconds = int((next_analysis_time - current_time).total_seconds())
        console.print(f"⏳ Next analysis at [cyan]{next_analysis_time.strftime('%H:%M:%S')}[/cyan] IST ({wait_seconds}s)")
        while datetime.now(IST) < next_analysis_time:
            time.sleep(1)

def get_analysis_cutoff_time():
    return get_unified_analysis_cutoff("live")

def parse_hhmm(s: str):
    h, m = map(int, s.split(":")); return h, m

def today_ist_dt(hhmm: str) -> datetime:
    now = datetime.now(IST); h, m = parse_hhmm(hhmm)
    return now.replace(hour=h, minute=m, second=0, microsecond=0)

def sleep_until(ts: datetime):
    while True:
        now = datetime.now(IST)
        delta = (ts - now).total_seconds()
        if delta <= 0: break
        time.sleep(min(0.5, delta))

def day_checkpoints_ist(day_date: datetime):
    d = day_date.date()
    start_h, start_m = parse_hhmm(CONFIG["FIRST_RUN_AT"])
    end_h, end_m = parse_hhmm(CONFIG["MARKET_END"])
    start_dt = IST.localize(datetime(d.year, d.month, d.day, start_h, start_m))
    end_dt = IST.localize(datetime(d.year, d.month, d.day, end_h, end_m))
    rng = pd.date_range(start=start_dt, end=end_dt, freq="5T", tz=IST, inclusive="both")
    return list(rng.to_pydatetime())

# ======== RATE LIMITING AND SESSIONS ========
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

def authenticate_session():
    return TD_hist(CONFIG["TDUSERNAME"], CONFIG["TDPASSWORD"], log_level=logging.CRITICAL)

def build_sessions():
    sess_count = CONFIG["TD_HIST_SESSIONS"]
    pool, limiters = [], []
    successful_sessions = 0
    for i in range(sess_count):
        try:
            session = authenticate_session()
            pool.append(session)
            successful_sessions += 1
            console.print(f"✅ Session {i+1}/{sess_count} connected")
        except Exception as e:
            console.print(f"[red]Session {i+1} failed: {e}[/red]")
    if not pool:
        console.print("[red]❌ Failed to initialize ANY TrueData sessions.[/red]")
        raise SystemExit("Failed to initialize TrueData sessions.")
    if successful_sessions < sess_count:
        console.print(f"[yellow]⚠️  Only {successful_sessions}/{sess_count} sessions connected[/yellow]")
    per_sess_rate = CONFIG["RATE_PER_SECOND_TOTAL"] / len(pool)
    for _ in pool:
        limiters.append(TokenBucketLimiter(rate_per_sec=per_sess_rate, bucket_size=CONFIG["BUCKET_SIZE"]))
    return pool, limiters

tdhist_pool, sess_limiters = build_sessions()
console.print("✅ [green]TrueData connection established[/green]")

# ======== INDICATORS (RSI/Volume/Momentum/OI) ========
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
        z_score = z_score.where(volume > vol_ma * 0.5, z_score * 0.3)
        return z_score.clip(-5, 5).fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def momentum_improved(df, period=10):
    if df is None or len(df) < period + 2:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    try:
        close = pd.to_numeric(df['Close'], errors='coerce').fillna(method='ffill')
        high = pd.to_numeric(df['High'], errors='coerce').fillna(method='ffill')
        low = pd.to_numeric(df['Low'], errors='coerce').fillna(method='ffill')
        hl2 = (high + low) / 2
        shifted_close = close.shift(period).replace(0, np.nan)
        momentum_val = (hl2 / shifted_close) - 1.0
        return momentum_val.fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

def oi_surge_improved(df, lookback=20):
    if df is None or len(df) < lookback + 5:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    try:
            # OI detection if available
    oi_col = None
    for c in df.columns:
        cu = c.upper()
        if "OI" in cu or "OPENINTEREST" in cu.replace("_", ""):
            oi_col = c
            break

    oi_ok = True
    oi_z_val = 0.0
    if require_oi:
        if oi_col is None:
            oi_ok = False
        else:
            oi = pd.to_numeric(df[oi_col], errors="coerce").fillna(0)
            oima = oi.rolling(CONFIG["INDICATOR_PERIODS"]["OI_SURGE"], min_periods=8).mean()
            oisd = oi.rolling(CONFIG["INDICATOR_PERIODS"]["OI_SURGE"], min_periods=8).std(ddof=0)
            oisd = oisd.where(oisd > 0, oima * 0.10)
            oi_z = ((oi - oima) / oisd).clip(-5, 5)
            oi_z_val = float(oi_z.iloc[-1])
            oi_ok = oi_z_val >= 1.0

    # Guard: need enough bars
    if len(df.index) < 3:
        return {"signal": None, "reason": "insufficient_bars", "strength": 0.0}

    # Last completed candle
    c = float(close.iloc[-1])
    u = float(upper.iloc[-1])
    l = float(lower.iloc[-1])

    # Must be coming out of a recent squeeze
    prev_in_squeeze = bool(in_squeeze.rolling(20, min_periods=1).max().iloc[-2])

    # Volume confirmation
    vol_z_val = float(vol_z.iloc[-1])

    band_w = max(1e-9, (u - l))
    bull_break = (c > u) and ((c - u) >= min_break_close_frac * band_w)
    bear_break = (c < l) and ((l - c) >= min_break_close_frac * band_w)

    if prev_in_squeeze and (bull_break or bear_break):
        vol_ok = vol_z_val >= min_volume_z
        if not vol_ok:
            return {"signal": None, "reason": "no_volume_confirmation", "strength": 0.0}
        if not oi_ok:
            return {"signal": None, "reason": "no_oi_confirmation", "strength": 0.0}

        dist_frac = ((c - u) / band_w) if bull_break else ((l - c) / band_w)
        strength = float(2.0 * dist_frac + 0.6 * vol_z_val + 0.4 * max(0.0, oi_z_val))
        return {
            "signal": "bull" if bull_break else "bear",
            "reason": "squeeze_breakout_confirmed",
            "strength": round(strength, 2)
        }

    return {"signal": None, "reason": "no_breakout", "strength": 0.0}


def oi_momentum_improved(df, period=10):
    if df is None or len(df) < period + 2:
        return pd.Series(index=df.index if df is not None else [], dtype='float64')
    try:
        oi_col = None
        for col in df.columns:
            if col.upper() == 'OI' or 'openinterest' in col.lower():
                oi_col = col; break
        if oi_col is None:
            return momentum_improved(df, period)
        oi = pd.to_numeric(df[oi_col], errors='coerce').fillna(0)
        if oi.sum() == 0:
            return momentum_improved(df, period)
        oim_s = (oi / oi.shift(period//2)).replace([np.inf, -np.inf], 1.0) - 1.0
        oim_l = (oi / oi.shift(period)).replace([np.inf, -np.inf], 1.0) - 1.0
        combined_mom = (oim_s * 0.7) + (oim_l * 0.3)
        return combined_mom.fillna(0)
    except Exception:
        return pd.Series(0, index=df.index)

# ======== DATA QUALITY AND NORMALIZATION ========
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

def normalize_hist_df_clean(df, symbol, timeframe_minutes):
    if df is None or df.empty:
        return None
    try:
        out = df.copy()
        out.columns = out.columns.str.lower().str.strip()
        rename_map = {}
        for col in out.columns:
            col_clean = col.lower().strip()
            if any(x in col_clean for x in ['time', 'date', 'timestamp', 'datetime', 'ts']):
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
            elif any(p in col_clean for p in ['oi','openinterest','open_interest','open interest','openint','open_int','oi_value','oivalue']):
                rename_map[col] = 'OI'
        out.rename(columns=rename_map, inplace=True)
        if "Timestamp" not in out.columns:
            if hasattr(out.index, 'dtype') and 'datetime' in str(out.index.dtype):
                out["Timestamp"] = out.index; out = out.reset_index(drop=True)
            else:
                now = datetime.now(IST)
                out["Timestamp"] = pd.date_range(
                    start=now - timedelta(minutes=timeframe_minutes * len(out)),
                    periods=len(out),
                    freq=f"{timeframe_minutes}T",
                    tz=IST
                )
        required_cols = ["Open", "High", "Low", "Close"]
        if any(col not in out.columns for col in required_cols):
            return None
        if "Volume" not in out.columns:
            out["Volume"] = 1000
        if "OI" in out.columns:
            try:
                out["OI"] = pd.to_numeric(out["OI"], errors="coerce").fillna(0)
                oi_sum = out["OI"].sum(); oi_max = out["OI"].max()
                if oi_sum > 0 and oi_max > 100:
                    global oi_symbols_found; oi_symbols_found.add(symbol)
                else:
                    out["OI"] = out["Volume"] * 0.1
            except Exception:
                out["OI"] = out["Volume"] * 0.1
        else:
            out["OI"] = out["Volume"] * 0.1
        out["Timestamp"] = pd.to_datetime(out["Timestamp"], errors="coerce")
        out = out.dropna(subset=["Timestamp"])
        if out.empty:
            return None
        if out["Timestamp"].dt.tz is None:
            try:
                out["Timestamp"] = out["Timestamp"].dt.tz_localize(IST, ambiguous='infer', nonexistent='shift_forward')
            except Exception:
                out["Timestamp"] = out["Timestamp"].dt.tz_localize(IST, ambiguous='NaT', nonexistent='NaT')
                out = out.dropna(subset=["Timestamp"])
        else:
            out["Timestamp"] = out["Timestamp"].dt.tz_convert(IST)
        for col in ["Open", "High", "Low", "Close", "Volume", "OI"]:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["Open", "High", "Low", "Close"])
        if out.empty or len(out) < 10:
            return None
        out = out.sort_values("Timestamp").set_index("Timestamp")
        out = out[~out.index.duplicated(keep='last')]
        return out
    except Exception:
        return None

def pick_session(symbol_orig, timeframe_minutes):
    return hash((symbol_orig, timeframe_minutes)) & 0x7fffffff % len(tdhist_pool)

# ======== COMPOSITE ANALYSIS (weights aggregation) ========
def analyze_signals_enhanced_clean(timeframe_dataframes, symbol):
    if not timeframe_dataframes:
        return 'Neutral', 0.0, 'Normal', 'WAIT', 'NONE'
    final_score, max_possible = 0.0, 0.0
    valid_timeframes = 0
    oi_status = 'Normal'
    has_strong_conditions = False
    supertrend_status = 'Normal'
    for tf_min, df in timeframe_dataframes.items():
        if df is None or len(df) < 20:
            continue
        is_valid, _ = validate_data_quality(df, 20)
        if not is_valid:
            continue
        valid_timeframes += 1
        tf_weight = CONFIG["TIMEFRAME_WEIGHTS"].get(tf_min, 1.0)
        if 'OI' in df.columns and df['OI'].sum() > 100:
            global oi_symbols_found; oi_symbols_found.add(symbol)
        scores = {}
        try:
            st_signal, kde_bull, kde_bear, st_status = supertrend_kde_analysis(df, symbol)
            if abs(st_signal) >= 3.5:
                scores['SuperTrendKDE'] = st_signal
                supertrend_status = st_status
                has_strong_conditions = True
                if st_signal > 0:
                    oi_status = 'Strong Call Setup + SuperTrend'
                else:
                    oi_status = 'Strong Put Setup + SuperTrend'
            elif abs(st_signal) >= 1.0:
                scores['SuperTrendKDE'] = st_signal * 0.7
                supertrend_status = st_status
            else:
                scores['SuperTrendKDE'] = 0.0
            if abs(st_signal) >= 1.0:
                scores['SuperTrendBreak'] = st_signal * 0.8
            else:
                scores['SuperTrendBreak'] = 0.0
            max_kde = max(kde_bull, kde_bear)
            if max_kde >= CONFIG["SUPERTREND_KDE_THRESHOLD"]:
                scores['VolumeKDE'] = 3.0 if kde_bull > kde_bear else -3.0
            elif max_kde >= 0.80:
                scores['VolumeKDE'] = 2.0 if kde_bull > kde_bear else -2.0
            elif max_kde >= 0.70:
                scores['VolumeKDE'] = 1.0 if kde_bull > kde_bear else -1.0
            else:
                scores['VolumeKDE'] = 0.0
        except Exception:
            scores['SuperTrendKDE'] = 0.0; scores['SuperTrendBreak'] = 0.0; scores['VolumeKDE'] = 0.0
        try:
            volsurge = volume_surge_improved(df, CONFIG["INDICATOR_PERIODS"]["VOLUME_SURGE"])
            if len(volsurge) > 2:
                currentsurge = float(volsurge.iloc[-1])
                pricechange = float(pd.to_numeric(df['Close']).iloc[-1] / pd.to_numeric(df['Close']).iloc[-5] - 1) if len(df) >= 6 else 0.0
                if currentsurge >= 2.5:
                    scores['VolumeSurge'] = 3.0 if pricechange >= 0.008 else -3.0; has_strong_conditions = True
                elif currentsurge >= 1.8:
                    scores['VolumeSurge'] = 2.5 if pricechange >= 0.005 else -2.5
                elif currentsurge >= 1.2:
                    scores['VolumeSurge'] = 1.5 if pricechange >= 0.003 else -1.5
                else:
                    scores['VolumeSurge'] = 0.0
            else:
                scores['VolumeSurge'] = 0.0
        except Exception:
            scores['VolumeSurge'] = 0.0
        try:
            scores['OISurge'] = float(oi_surge_improved(df, CONFIG["INDICATOR_PERIODS"]["OI_SURGE"]).iloc[-1])
            scores['OIMomentum'] = float(oi_momentum_improved(df, CONFIG["INDICATOR_PERIODS"]["OI_MOMENTUM"]).iloc[-1])
        except Exception:
            scores['OISurge'] = 0.0; scores['OIMomentum'] = 0.0
        try:
            mom = momentum_improved(df, CONFIG["INDICATOR_PERIODS"]["MOMENTUM"]).iloc[-1]
            scores['Momentum'] = float(mom)
        except Exception:
            scores['Momentum'] = 0.0
        try:
            close = pd.to_numeric(df['Close'], errors='coerce')
            price_up = close.iloc[-1] > close.iloc[-2]
            vol_high = abs(scores.get('VolumeSurge', 0)) >= 1.5
            oi_active = abs(scores.get('OISurge', 0)) >= 1.5
            momentum_strong = abs(scores.get('Momentum', 0)) >= 1.0
            supertrend_strong = abs(scores.get('SuperTrendKDE', 0)) >= 2.0
            if price_up and vol_high and oi_active and momentum_strong and supertrend_strong:
                scores['CallBias'] = 5.0; scores['PutBias'] = 0.0
                if "SuperTrend" not in oi_status: oi_status = "Ultra Strong Call Setup"; has_strong_conditions = True
            elif (not price_up) and vol_high and oi_active and momentum_strong and supertrend_strong:
                scores['PutBias'] = -5.0; scores['CallBias'] = 0.0
                if "SuperTrend" not in oi_status: oi_status = "Ultra Strong Put Setup"; has_strong_conditions = True
            elif price_up and (vol_high and oi_active and momentum_strong):
                scores['CallBias'] = 4.0; scores['PutBias'] = 0.0
                if "SuperTrend" not in oi_status: oi_status = "Strong Call Setup"; has_strong_conditions = True
            elif (not price_up) and (vol_high and oi_active and momentum_strong):
                scores['PutBias'] = -4.0; scores['CallBias'] = 0.0
                if "SuperTrend" not in oi_status: oi_status = "Strong Put Setup"; has_strong_conditions = True
            elif price_up and (vol_high or oi_active):
                scores['CallBias'] = 2.0; scores['PutBias'] = 0.0
                if oi_active and "SuperTrend" not in oi_status: oi_status = "Call Setup"
            elif (not price_up) and (vol_high or oi_active):
                scores['PutBias'] = -2.0; scores['CallBias'] = 0.0
                if oi_active and "SuperTrend" not in oi_status: oi_status = "Put Setup"
            else:
                scores['CallBias'] = 0.0; scores['PutBias'] = 0.0
        except Exception:
            scores['CallBias'] = 0.0; scores['PutBias'] = 0.0
        remaining = ["ADX","VWAP","MACD","EMA","CMF","ADL","OBV","ATR","Bollinger","ROC","Stochastic","CCI","MA","WWL","RelVol","VWAPRegime","OBVConfirm","PriceVelocity","GammaSqueezeRisk","IVCrushRisk","OptionFlow"]
        for indicator in remaining:
            if indicator not in scores:
                try:
                    pricechange = float(pd.to_numeric(df['Close']).iloc[-1] / pd.to_numeric(df['Close']).iloc[-5] - 1) if len(df) >= 6 else 0.0
                    if abs(pricechange) >= 0.02:
                        scores[indicator] = 1.0 * np.sign(pricechange)
                    elif abs(pricechange) >= 0.01:
                        scores[indicator] = 0.5 * np.sign(pricechange)
                    else:
                        scores[indicator] = 0.0
                except Exception:
                    scores[indicator] = 0.0
        for indicator, score in scores.items():
            ind_weight = CONFIG["INDICATOR_WEIGHTS"].get(indicator, 1.0)
            weighted_score = score * tf_weight * ind_weight
            final_score += weighted_score
            max_possible += 4.0 * tf_weight * ind_weight
    if valid_timeframes < 1 or max_possible == 0:
        return 'Neutral', 0.0, oi_status, 'WAIT', 'NONE'
    normalized = (final_score / max_possible) * 100.0
    def classify_option_signal(normalized_score, oi_status, has_strong):
        if normalized_score >= 35:
            return "🚀 ULTRA STRONG BUY - AGGRESSIVE CALLS", "ULTRA_STRONG"
        if normalized_score <= -35:
            return "💥 ULTRA STRONG SELL - AGGRESSIVE PUTS", "ULTRA_STRONG"
        if normalized_score >= 20:
            return "🔥 VERY STRONG BUY - CALL FOCUS", "VERY_STRONG"
        if normalized_score <= -20:
            return "🔥 VERY STRONG SELL - PUT FOCUS", "VERY_STRONG"
        if normalized_score >= 12:
            return "⚡ STRONG BUY", "STRONG"
        if normalized_score <= -12:
            return "⚡ STRONG SELL", "STRONG"
        if normalized_score >= 8:
            return "🟢 BUY - Call Potential", "MODERATE"
        if normalized_score <= -8:
            return "🔴 SELL - Put Potential", "MODERATE"
        return "⚪ NEUTRAL", "NEUTRAL"
    def get_option_action(signal_strength, normalized_score, oi_status=""):
        if signal_strength == "ULTRA_STRONG":
            return ("🚨 BUY CALLS AGGRESSIVELY - ATM/ITM", "URGENT") if normalized_score > 0 else ("🚨 BUY PUTS AGGRESSIVELY - ATM/ITM", "URGENT")
        if signal_strength == "VERY_STRONG":
            return ("🔥 BUY CALLS STRONG", "HIGH") if normalized_score > 0 else ("🔥 BUY PUTS STRONG", "HIGH")
        if signal_strength == "STRONG":
            return ("⚡ BUY CALLS - ATM/OTM", "MEDIUM") if normalized_score > 0 else ("⚡ BUY PUTS - ATM/OTM", "MEDIUM")
        if signal_strength == "MODERATE":
            return ("📈 Consider Calls - OTM Safe", "LOW") if normalized_score > 0 else ("📉 Consider Puts - OTM Safe", "LOW")
        return ("⏸️ WAIT - No Clear Direction", "NONE")
    label, bucket = classify_option_signal(normalized, oi_status, has_strong_conditions)
    action, priority = get_option_action(bucket, normalized, oi_status)
    return label, normalized, oi_status, action, priority

# ======== BOLLINGER SQUEEZE BREAKOUT ========
def calculate_bollinger(df, period=None, std_dev=None):
    try:
        if period is None:
            period = CONFIG["INDICATOR_PERIODS"]["BB_PERIOD"]
        if std_dev is None:
            std_dev = CONFIG["INDICATOR_PERIODS"]["BB_STD_DEV"]
        close = pd.to_numeric(df["Close"], errors="coerce").ffill()
        if len(close) < max(20, period) + 2:
            return None, None, None, None, None
        ma = close.rolling(window=period, min_periods=period//2).mean()
        sd = close.rolling(window=period, min_periods=period//2).std(ddof=0)
        upper = ma + std_dev * sd
        lower = ma - std_dev * sd
        raw_width = (upper - lower)
        bandwidth = raw_width / ma.replace(0, np.nan)
        pct_b = (close - lower) / (upper - lower)
        return ma, upper, lower, bandwidth, pct_b
    except Exception:
        return None, None, None, None, None

def bollinger_squeeze_flags(df, squeeze_lookback=120, squeeze_percentile=0.15, keltner=True, atr_period=None):
    try:
        ma, upper, lower, bandwidth, _ = calculate_bollinger(df)
        if ma is None:
            return pd.Series(False, index=df.index)
        bw = bandwidth.replace([np.inf, -np.inf], np.nan).ffill()
        bw_q = bw.rolling(window=squeeze_lookback, min_periods=max(20, squeeze_lookback//4)).quantile(squeeze_percentile)
        squeeze_band = (bw <= bw_q)
        if keltner:
            if atr_period is None:
                atr_period = CONFIG["INDICATOR_PERIODS"]["ATR"]
            high = pd.to_numeric(df["High"], errors="coerce")
            low = pd.to_numeric(df["Low"], errors="coerce")
            close = pd.to_numeric(df["Close"], errors="coerce")
            tr1 = (high - low).abs()
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=atr_period, min_periods=max(5, atr_period//2)).mean()
            kel_upper = ma + 1.5 * atr
            kel_lower = ma - 1.5 * atr
            bb_inside_kc = (upper < kel_upper) & (lower > kel_lower)
            return (squeeze_band & bb_inside_kc.fillna(False)).fillna(False)
        return squeeze_band.fillna(False)
    except Exception:
        return pd.Series(False, index=df.index)

def bollinger_breakout_signal(df, min_break_close_frac=0.25, min_volume_z=1.5, require_oi=True):
    try:
        ma, upper, lower, bandwidth, pct_b = calculate_bollinger(df)
        if ma is None:
            return {"signal": None, "reason": "insufficient_bars", "strength": 0.0}
        in_squeeze = bollinger_squeeze_flags(df)
        close = pd.to_numeric(df["Close"], errors="coerce")
        vol = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
        vol_ma = vol.rolling(CONFIG["INDICATOR_PERIODS"]["VOLUME_SURGE"], min_periods=8).mean()
        vol_sd = vol.rolling(CONFIG["INDICATOR_PERIODS"]["VOLUME_SURGE"], min_periods=8).std(ddof=0)
        vol_sd = vol_sd.where(vol_sd > 0, vol_ma * 0.10)
        vol_z = ((vol - vol_ma) / vol_sd).clip(-5, 5)
        oi_col = None
        for c in df.columns:
            cu = c.upper()
            if "OI" in cu or "OPENINTEREST" in cu.replace("_", ""):
                oi_col = c; break
        oi_ok = True; oi_z_val = 0.0
        if require_oi:
            if oi_col is
                oi = pd.to_numeric(df[oi_col], errors="coerce").fillna(0)
                oima = oi.rolling(CONFIG["INDICATOR_PERIODS"]["OI_SURGE"], min_periods=8).mean()
                oisd = oi.rolling(CONFIG["INDICATOR_PERIODS"]["OI_SURGE"], min_periods=8).std(ddof=0)
                oisd = oisd.where(oisd > 0, oima * 0.10)
                oi_z = ((oi - oima) / oisd).clip(-5, 5)
                oi_z_val = float(oi_z.iloc[-1])
                oi_ok = oi_z_val >= 1.0

        if len(df.index) < 3:
            return {"signal": None, "reason": "insufficient_bars", "strength": 0.0}

        c = float(close.iloc[-1])
        u = float(upper.iloc[-1])
        l = float(lower.iloc[-1])
        prev_in_squeeze = bool(in_squeeze.rolling(20, min_periods=1).max().iloc[-2])
        vol_z_val = float(vol_z.iloc[-1])

        band_w = max(1e-9, (u - l))
        bull_break = (c > u) and ((c - u) >= min_break_close_frac * band_w)
        bear_break = (c < l) and ((l - c) >= min_break_close_frac * band_w)

        if prev_in_squeeze and (bull_break or bear_break):
            vol_ok = vol_z_val >= min_volume_z
            if not vol_ok:
                return {"signal": None, "reason": "no_volume_confirmation", "strength": 0.0}
            if not oi_ok:
                return {"signal": None, "reason": "no_oi_confirmation", "strength": 0.0}

            dist_frac = ((c - u) / band_w) if bull_break else ((l - c) / band_w)
            strength = float(2.0 * dist_frac + 0.6 * vol_z_val + 0.4 * max(0.0, oi_z_val))
            return {"signal": "bull" if bull_break else "bear", "reason": "squeeze_breakout_confirmed", "strength": round(strength, 2)}

        return {"signal": None, "reason": "no_breakout", "strength": 0.0}
    except Exception:
        return {"signal": None, "reason": "error", "strength": 0.0}

def find_bollinger_squeeze_breakouts(symbol_to_df, require_oi=True):
    results = []
    for symbol, df in symbol_to_df.items():
        try:
            sig = bollinger_breakout_signal(df, require_oi=require_oi)
            if sig["signal"] == "bull":
                results.append((symbol, "CALL", sig["strength"], f"squeeze breakout ↑, strength {sig['strength']}"))
            elif sig["signal"] == "bear":
                results.append((symbol, "PUT", sig["strength"], f"squeeze breakout ↓, strength {sig['strength']}"))
        except Exception:
            continue
    return sorted(results, key=lambda x: x[2], reverse=True)

def show_bollinger_breakout_table(candidates, max_rows=25):
    if not candidates:
        console.print("[yellow]No Bollinger squeeze breakouts on the latest completed candle[/yellow]")
        return
    table = Table(show_header=True, header_style="bold cyan", box=box.SIMPLE_HEAVY, title="🎯 Bollinger Squeeze Breakouts (5m)")
    table.add_column("Symbol", justify="left")
    table.add_column("Side", justify="center")
    table.add_column("Strength", justify="right")
    table.add_column("Note", justify="left")
    for sym, side, strength, note in candidates[:max_rows]:
        table.add_row(sym, side, f"{strength:.2f}", note)
    console.print(table)

# ======== DATA FETCH: MULTI-TIMEFRAME FROM TRUEDATA ========
def load_symbols():
    f = CONFIG["SHARES_FILE"]
    if not os.path.exists(f):
        console.print(f"[red]Missing {f}. Create it with one NSE symbol per line (e.g., RELIANCE-EQ).[/red]")
        raise SystemExit(1)
    with open(f, "r") as fh:
        syms = [line.strip() for line in fh if line.strip()]
    return syms

def fetch_hist_for_symbol(session, limiter, symbol, bar_minutes, duration_label):
    try:
        limiter.acquire()
        bars = session.get_historical_data(
            symbol, duration=duration_label, bar_size=CONFIG["BAR_SIZE_MAP"][bar_minutes]
        )
        df = pd.DataFrame(bars)
        df = normalize_hist_df_clean(df, symbol, bar_minutes)
        return symbol, bar_minutes, df, None
    except Exception as e:
        return symbol, bar_minutes, None, str(e)

def fetch_all_timeframes(symbols):
    tf_list = [5, 15, 30, 60]
    if not CONFIG["SKIP_DAILY"]:
        tf_list.append(1440)
    symbol_to_tf_df = defaultdict(dict)
    futures = []
    with ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as ex:
        for sym in symbols:
            for tf in tf_list:
                sess_idx = pick_session(sym, tf)
                session = tdhist_pool[sess_idx]
                limiter = sess_limiters[sess_idx]
                futures.append(ex.submit(fetch_hist_for_symbol, session, limiter, sym, tf, CONFIG["DURATION_MAP"][tf]))
        it = tqdm(as_completed(futures), total=len(futures), disable=not CONFIG["SHOW_PROGRESS"])
        for fut in it:
            symbol, tf, df, err = fut.result()
            if err:
                failed_symbols.add(symbol)
            else:
                symbol_to_tf_df[symbol][tf] = df
    return symbol_to_tf_df

# ======== SCAN AND RENDER ========
def scan_and_display(symbol_to_tf_df, analysis_cutoff=None, mode="live"):
    rows = []
    symbol_5m_map = {}
    for symbol, tf_map in symbol_to_tf_df.items():
        cleaned = {}
        for tf, df in tf_map.items():
            if df is None or df.empty:
                continue
            use_df = df.copy()
            if analysis_cutoff is not None:
                if mode == "live":
                    use_df = use_df[use_df.index < analysis_cutoff]
                else:
                    use_df = use_df[use_df.index <= analysis_cutoff]
            if len(use_df) >= CONFIG["MIN_BARS_REQUIRED"]:
                cleaned[tf] = use_df
        if not cleaned:
            continue
        label, score, oi_status, action, priority = analyze_signals_enhanced_clean(cleaned, symbol)
        rows.append((symbol, score, label, oi_status, action, priority))
        if 5 in cleaned:
            symbol_5m_map[symbol.replace("-EQ", "")] = cleaned[5]

    # Render main table
    if rows:
        rows.sort(key=lambda x: abs(x[1]), reverse=True)
        table = Table(show_header=True, header_style="bold white", box=box.SIMPLE_HEAVY, title="📊 Option Buyer Scanner")
        table.add_column("Symbol", justify="left")
        table.add_column("Score", justify="right")
        table.add_column("Signal", justify="left")
        table.add_column("OI/Status", justify="left")
        table.add_column("Action", justify="left")
        table.add_column("Priority", justify="center")
        for sym, score, label, oi_status, action, priority in rows[:60]:
            table.add_row(sym, f"{score:.2f}", label, oi_status, action, priority)
        console.print(table)
    else:
        console.print("[yellow]No symbols produced signals.[/yellow]")

    # Render Bollinger squeeze table
    try:
        bb_cands = find_bollinger_squeeze_breakouts(symbol_5m_map, require_oi=True)
        if bb_cands:
            console.print("\n[bold green]Bollinger Squeeze Breakouts[/bold green]")
            show_bollinger_breakout_table(bb_cands, max_rows=25)
        else:
            console.print("[yellow]No Bollinger squeeze breakouts on the latest completed 5m candle.[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Bollinger scan skipped: {e}[/yellow]")

# ======== LIVE LOOP ========
def run_live():
    symbols = load_symbols()
    console.print(f"[cyan]Loaded {len(symbols)} symbols from {CONFIG['SHARES_FILE']}[/cyan]")
    # Align to first run time
    first_run_dt = today_ist_dt(CONFIG["FIRST_RUN_AT"])
    now = datetime.now(IST)
    if now < first_run_dt:
        console.print(f"[blue]Waiting until first run at {CONFIG['FIRST_RUN_AT']} IST...[/blue]")
        sleep_until(first_run_dt + timedelta(seconds=30))
    # Main loop
    while True:
        analysis_cutoff = get_unified_analysis_cutoff("live")
        market_end_dt = today_ist_dt(CONFIG["MARKET_END"])
        if datetime.now(IST) > market_end_dt:
            console.print("[green]Market closed. Exiting live loop.[/green]")
            break
        console.print(f"\n[bold]Scanning up to completed 5m boundary: {analysis_cutoff.strftime('%H:%M')} IST[/bold]")
        symbol_to_tf_df = fetch_all_timeframes(symbols)
        scan_and_display(symbol_to_tf_df, analysis_cutoff=analysis_cutoff, mode="live")
        wait_for_next_completed_5min_candle()

# ======== BACKTEST (intraday checkpoints) ========
def run_backtest(backtest_date_str):
    try:
        base_date = datetime.strptime(backtest_date_str, "%Y-%m-%d")
        base_date = IST.localize(datetime(base_date.year, base_date.month, base_date.day, 0, 0))
    except Exception:
        console.print("[red]Invalid --backtest-date. Use YYYY-MM-DD.[/red]")
        return
    symbols = load_symbols()
    checkpoints = day_checkpoints_ist(base_date)
    for cp in checkpoints:
        console.print(f"\n[bold magenta]Backtest checkpoint: {cp.strftime('%Y-%m-%d %H:%M')} IST[/bold magenta]")
        symbol_to_tf_df = fetch_all_timeframes(symbols)
        scan_and_display(symbol_to_tf_df, analysis_cutoff=cp, mode="backtest")

# ======== ARGPARSE / ENTRYPOINT ========
def main():
    parser = argparse.ArgumentParser(description="Option Buyer Scanner with Bollinger Squeeze Breakout (TrueData)")
    parser.add_argument("--live", action="store_true", help="Run in live mode")
    parser.add_argument("--backtest-date", type=str, default=None, help="Backtest for a given date YYYY-MM-DD")
    args = parser.parse_args()

    if args.live:
        run_live()
    elif args.backtest_date:
        run_backtest(args.backtest_date)
    else:
        console.print("[yellow]No mode specified. Use --live or --backtest-date YYYY-MM-DD[/yellow]")

if __name__ == "__main__":
    main()
