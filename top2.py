# ==============================================================================
# COMPLETE PRESSURE SCANNER - PICKS TOP 1-2 BUY/SELL PRESSURE STOCKS EVERY 5 MIN
# Based on Enhanced Option Buyer Scanner v3.2 - Real OI/Volume Enforced
# ==============================================================================
import os
import logging
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import time
import threading
from collections import defaultdict
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm
from truedata.history import TD_hist

# Enhanced table formatting libraries
try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Install rich: pip install rich")

try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    print("Install colorama: pip install colorama")

# Initialize console
if RICH_AVAILABLE:
    console = Console()

# Logger
class Logger:
    def info(self, msg): print(f"[INFO] {msg}")
    def error(self, msg): print(f"[ERROR] {msg}")
    def warning(self, msg): print(f"[WARNING] {msg}")
logger = Logger()

# Configuration
class Config:
    TDUSERNAME = os.getenv("TRUEDATA_USER", "tdwsp751")
    TDPASSWORD = os.getenv("TRUEDATA_PASS", "raj@751")
    MARKET_START = "09:15"  # IST
    FIRST_RUN_AT = "09:20"  # IST
    FIRST_SCAN_DELAY = 15  # seconds
    MARKET_END = "15:30"  # IST
    SETTLE_DELAY_SECONDS = 15
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "64"))
    TD_HIST_SESSIONS = int(os.getenv("TD_HIST_SESSIONS", "5"))
    SHARES_FILE = os.getenv("SHARES_FILE", "shares.txt")
    BENCHMARK_INDEX = "NIFTY 50"
    GROUP_WEIGHTS = {
        "Trend": 2.5, "Momentum": 2.0, "Volume": 2.2, "Volatility": 1.8, "OI": 2.5,
    }
    INDICATOR_WEIGHTS = {
        "MA_Slope": 2.0, "ADX": 1.8, "VWAP": 1.7, "EMA": 1.7, "MACD_Trend": 1.5,
        "RSI": 2.0, "Stochastic": 1.2, "CCI": 1.2, "ROC": 1.1, "WilliamsR": 1.0,
        "VolumeSurge": 2.5, "OBV": 1.8, "CMF": 1.8, "RelVol": 1.5,
        "VolatilityExpansion": 2.5, "Bollinger": 1.3,
        "OptionBuyerMomentum": 2.8, "OIChange": 2.5, "VolumeOISync": 2.2,
    }
    SCORE_THRESHOLD_MIN = 10.0
    SIGNAL_THRESHOLDS = {
        'Very Strong Buy': 55.0, 'Strong Buy': 30.0, 'Buy Signal': 15.0,
        'Very Strong Sell': -55.0, 'Strong Sell': -30.0, 'Sell Signal': -15.0,
    }
    REGIME_MULTIPLIERS = {
        'bullish_in_bull_market': 1.15, 'bearish_in_bear_market': 1.15,
        'bullish_in_bear_market': 0.8, 'bearish_in_bull_market': 0.8,
    }

# Constants
IST = pytz.timezone("Asia/Kolkata")
BAR_SIZE_MAP = {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"}
DURATION_MAP = {5: "30 D", 15: "30 D", 30: "60 D", 60: "120 D", 1440: "365 D"}
TIMEFRAME_WEIGHTS = {5: 2.5, 15: 3.0, 30: 2.0, 60: 1.5, 1440: 1.0}

# State management
intraday_volume_data = {}
intraday_oi_data = {}
scan_count = 0
previous_scan_results = {}

# Colors
class Colors:
    HEADER = '\033[95m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_colored(text, color):
    if COLORAMA_AVAILABLE:
        print(color + text + Style.RESET_ALL)
    else:
        print(f"{color}{text}{Colors.END}")

# Table creation
def create_simple_table(data, title, new_stocks=None):
    if not data:
        print_colored(f"{title}: No stocks found.", Colors.YELLOW)
        return
    if RICH_AVAILABLE:
        table = Table(title=title, box=box.ROUNDED)
        table.add_column("Stock", style="bold")
        table.add_column("Signal")
        table.add_column("Score")
        table.add_column("Vol Δ%")
        table.add_column("OI Δ%")
        table.add_column("Vol Multi")
        for item in data:
            stock_style = item['symbol'] + " ✨" if new_stocks and item['symbol'] in new_stocks else item['symbol']
            table.add_row(
                stock_style,
                item['signal'],
                f"{item['score']:.2f}",
                f"{item.get('volume_change_pct', 0):+.1f}%",
                f"{item.get('oi_change_pct', 0):+.1f}%",
                f"{item.get('vol_multi', 0):.1f}x"
            )
        console.print(table)
    else:
        print_colored(title, Colors.BOLD)
        for item in data:
            marker = " ✨" if new_stocks and item['symbol'] in new_stocks else ""
            print(f"{item['symbol']}{marker} | {item['signal']} | Score: {item['score']:.2f} | VolΔ: {item.get('volume_change_pct', 0):+.1f}% | OIΔ: {item.get('oi_change_pct', 0):+.1f}% | VolMulti: {item.get('vol_multi', 0):.1f}x")

# Technical Indicators
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def vwap(df, period=None):
    price = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = price * df["volume"]
    if period:
        pv_sum = pv.rolling(period).sum()
        vol_sum = df["volume"].rolling(period).sum()
    else:
        pv_sum = pv.cumsum()
        vol_sum = df["volume"].cumsum()
    return pv_sum / vol_sum.replace(0, np.nan)

def atr(df, period=14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def williams_r(df, period=14):
    highest = df["high"].rolling(period).max()
    lowest = df["low"].rolling(period).min()
    return -100 * (highest - df["close"]) / (highest - lowest).replace(0, np.nan)

def volume_surge(df, lookback=20):
    vol_ma = df["volume"].rolling(lookback).mean()
    vol_std = df["volume"].rolling(lookback).std()
    z_score = (df["volume"] - vol_ma) / vol_std.replace(0, np.nan)
    return z_score.fillna(0)

def calculate_rsi(df, period=14):
    if len(df) < period + 1:
        return pd.Series(dtype='float64', index=df.index)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).ewm(com=period - 1, adjust=False).mean()
    loss = -delta.where(delta < 0, 0).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    rs.fillna(100, inplace=True)
    return 100 - (100 / (1 + rs))

def calculate_macd(df, fast=12, slow=26, signal=9):
    if len(df) < slow + signal:
        return pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index)
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_stochastic(df, period=14, smooth_d=3):
    if len(df) < period + smooth_d:
        return pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index)
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    k = 100 * ((df['close'] - low_min) / (high_max - low_min).replace(0, np.nan))
    k.fillna(50, inplace=True)
    d = k.rolling(window=smooth_d).mean()
    return k, d

def calculate_adx(df, period=14):
    if len(df) < period * 2:
        return pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index)
    df_adx = df.copy()
    df_adx['h-l'] = df_adx['high'] - df_adx['low']
    df_adx['h-c'] = abs(df_adx['high'] - df_adx['close'].shift(1))
    df_adx['l-c'] = abs(df_adx['low'] - df_adx['close'].shift(1))
    df_adx['tr'] = df_adx[['h-l', 'h-c', 'l-c']].max(axis=1)
    df_adx['+dm'] = np.where((df_adx['high'] - df_adx['high'].shift(1)) > (df_adx['low'].shift(1) - df_adx['low']),
                             df_adx['high'] - df_adx['high'].shift(1), 0)
    df_adx['-dm'] = np.where((df_adx['low'].shift(1) - df_adx['low']) > (df_adx['high'] - df_adx['high'].shift(1)),
                             df_adx['low'].shift(1) - df_adx['low'], 0)
    atr_val = df_adx['tr'].ewm(com=period - 1, adjust=False).mean().replace(0, np.nan)
    pdi = (df_adx['+dm'].ewm(com=period - 1, adjust=False).mean() / atr_val) * 100
    ndi = (df_adx['-dm'].ewm(com=period - 1, adjust=False).mean() / atr_val) * 100
    adx = (abs(pdi - ndi) / (pdi + ndi).replace(0, np.nan)).ewm(com=period - 1, adjust=False).mean() * 100
    return adx.fillna(20), pdi.fillna(20), ndi.fillna(20)

def calculate_bollinger_bands(df, period=20, std_dev=2):
    if len(df) < period:
        return pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index), pd.Series(dtype='float64', index=df.index)
    middle = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    return middle, upper, lower

def calculate_roc(df, period=12):
    if len(df) < period + 1:
        return pd.Series(dtype='float64', index=df.index)
    shifted_close = df['close'].shift(period).replace(0, np.nan)
    return ((df['close'] - df['close'].shift(period)) / shifted_close) * 100

def calculate_obv(df):
    if len(df) < 2:
        return pd.Series(dtype='float64', index=df.index)
    return (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()

def calculate_cci(df, period=20):
    if len(df) < period:
        return pd.Series(dtype='float64', index=df.index)
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).replace(0, np.nan)
    return (tp - sma_tp) / (0.015 * mad)

def cmf(df, period=20):
    mfm = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (df["high"] - df["low"]).replace(0, np.nan)
    mfm.fillna(0, inplace=True)
    mfv = mfm * df["volume"]
    mfv_sum = mfv.rolling(period).sum()
    vol_sum = df["volume"].rolling(period).sum().replace(0, np.nan)
    return (mfv_sum / vol_sum).fillna(0)

def relative_volume(df, lookback=50):
    vol_ma = df["volume"].rolling(lookback).mean()
    return (df["volume"] / vol_ma.replace(0, np.nan)).fillna(1.0)

def _has_real_oi(df):
    return ('openinterest' in df.columns) and (df['openinterest'].notna().sum() >= 2)

def detect_oi_buildup(df, lookback=20):
    if not _has_real_oi(df) or len(df) < lookback:
        return None
    oi_ma = df['openinterest'].rolling(lookback).mean()
    if len(oi_ma) == 0 or pd.isna(oi_ma.iloc[-1]):
        return None
    current_oi = df['openinterest'].iloc[-1]
    avg_oi = oi_ma.iloc[-1]
    if avg_oi > 0 and pd.notna(current_oi):
        oi_strength = (current_oi - avg_oi) / avg_oi
        return max(min(oi_strength * 100, 100), -100)
    return None

def volume_oi_sync_analysis(df):
    if len(df) < 10 or not _has_real_oi(df):
        return None
    vol_change = df['volume'].pct_change(5).fillna(0)
    oi_change = df['openinterest'].pct_change(5).fillna(0)
    sync_score = vol_change.iloc[-1] + oi_change.iloc[-1]
    return min(max(sync_score * 50, -100), 100)

def option_buyer_momentum(df):
    if len(df) < 20:
        return None
    price_mom = (df['close'].iloc[-1] / df['close'].iloc[-5] - 1) * 100 if len(df) >= 5 else 0
    vol_surge_val = volume_surge(df, lookback=20).iloc[-1] if len(df) > 20 else 0
    oi_buildup = detect_oi_buildup(df, lookback=20)
    if oi_buildup is None:
        return None
    combined_score = (price_mom * 0.4) + (vol_surge_val * 0.3) + (oi_buildup * 0.3)
    return min(max(combined_score, -100), 100)

def slope(series, lookback=10):
    if len(series) < lookback: return 0.0
    y = series.tail(lookback).values
    x = np.arange(len(y))
    if len(y) < 2: return 0.0
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0]

# Scoring Engine
def normalize_score(value, bullish_range, bearish_range, score_range=(-2.0, 2.0)):
    low_score, high_score = score_range
    bull_min, bull_max = bullish_range
    if value >= bull_max: return high_score
    if value > bull_min:
        return high_score * ((value - bull_min) / (bull_max - bull_min))
    bear_max, bear_min = bearish_range
    if value <= bear_min: return low_score
    if value < bear_max:
        return low_score * ((bear_max - value) / (bear_max - bear_min))
    return 0.0

def calculate_indicator_scores(df):
    scores = defaultdict(float)
    if df is None or len(df) < 50: return scores
    try:
        # Trend
        adx, pdi, ndi = calculate_adx(df)
        if not adx.empty and len(adx) > 3 and adx.iloc[-1] > 20 and adx.iloc[-1] > adx.iloc[-3]:
            scores['ADX'] = 2.0 if pdi.iloc[-1] > ndi.iloc[-1] else -2.0
        ema20 = ema(df['close'], 20)
        ema50 = ema(df['close'], 50)
        if not ema20.empty and not ema50.empty:
            ema_ratio = ema20.iloc[-1] / ema50.iloc[-1] if ema50.iloc[-1] != 0 else 1
            scores['EMA'] = normalize_score(ema_ratio, (1.001, 1.02), (0.999, 0.98))
        vwap_line = vwap(df)
        if not vwap_line.empty:
            vwap_ratio = df['close'].iloc[-1] / vwap_line.iloc[-1] if vwap_line.iloc[-1] != 0 else 1
            scores['VWAP'] = normalize_score(vwap_ratio, (1.002, 1.025), (0.998, 0.975))
        macd, signal = calculate_macd(df)
        if not macd.empty and not signal.empty:
            if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-1] > 0:
                scores['MACD_Trend'] = 2.0
            elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-1] < 0:
                scores['MACD_Trend'] = -2.0
        if not ema20.empty and len(ema20) >= 5:
            ma20_slope = slope(ema20, 5)
            price_norm_slope = ma20_slope / df['close'].iloc[-1] * 1000 if df['close'].iloc[-1] != 0 else 0
            scores['MA_Slope'] = normalize_score(price_norm_slope, (0.1, 0.5), (-0.1, -0.5), (-2.5, 2.5))
        # Momentum
        rsi = calculate_rsi(df)
        if not rsi.empty:
            scores['RSI'] = normalize_score(rsi.iloc[-1], (60, 85), (40, 15))
        k, d = calculate_stochastic(df)
        if not k.empty and not d.empty:
            if k.iloc[-1] > d.iloc[-1]:
                scores['Stochastic'] = normalize_score(k.iloc[-1], (20, 80), (100, 100))
            elif k.iloc[-1] < d.iloc[-1]:
                scores['Stochastic'] = normalize_score(k.iloc[-1], (0,0), (80, 20))
        cci = calculate_cci(df)
        if not cci.empty:
            scores['CCI'] = normalize_score(cci.iloc[-1], (100, 200), (-100, -200))
        roc = calculate_roc(df)
        if not roc.empty:
            scores['ROC'] = normalize_score(roc.iloc[-1], (0.5, 2.0), (-0.5, -2.0))
        wr = williams_r(df)
        if not wr.empty:
            scores['WilliamsR'] = normalize_score(wr.iloc[-1], (-100, -80), (-20, 0))
        # Volume
        zscore = volume_surge(df, 20)
        if not zscore.empty:
            price_up = df['close'].iloc[-1] > df['close'].iloc[-2]
            if price_up:
                scores['VolumeSurge'] = normalize_score(zscore.iloc[-1], (1.5, 3.0), (0,0))
            else:
                scores['VolumeSurge'] = normalize_score(zscore.iloc[-1], (0,0), (-1.5, -3.0))
        obv_line = calculate_obv(df)
        if len(obv_line) > 5:
            obv_slope = slope(obv_line, 5)
            scores['OBV'] = normalize_score(obv_slope, (1, 1e9), (-1, -1e9))
        cmf20 = cmf(df, 20)
        if not cmf20.empty:
            scores['CMF'] = normalize_score(cmf20.iloc[-1], (0.1, 0.25), (-0.1, -0.25))
        rv = relative_volume(df, 50)
        if not rv.empty:
            scores['RelVol'] = normalize_score(rv.iloc[-1], (1.5, 3.0), (0.5, 0.5))
        # Volatility
        atr_val = atr(df, 14)
        if len(atr_val) > 20:
            atr_ma = atr_val.rolling(20).mean()
            if len(atr_ma) > 0 and atr_ma.iloc[-1] != 0:
                atr_ratio = atr_val.iloc[-1] / atr_ma.iloc[-1]
                atr_slope_ratio = atr_val.iloc[-1] / atr_val.iloc[-5] if len(atr_val) >= 5 and atr_val.iloc[-5] > 0 else 1
                if atr_ratio > 1.1 and atr_slope_ratio > 1.1:
                    price_direction = 1 if df['close'].iloc[-1] > df['close'].iloc[-5] else -1
                    scores['VolatilityExpansion'] = 2.5 * price_direction
        _, bb_upper, bb_lower = calculate_bollinger_bands(df)
        if not bb_upper.empty and not bb_lower.empty:
            if df['close'].iloc[-1] > bb_upper.iloc[-1]:
                scores['Bollinger'] = 2.0
            elif df['close'].iloc[-1] < bb_lower.iloc[-1]:
                scores['Bollinger'] = -2.0
        # OI
        oi_bu = detect_oi_buildup(df, 20)
        if oi_bu is not None:
            scores['OIChange'] = normalize_score(oi_bu, (10, 30), (-10, -30))
        sync = volume_oi_sync_analysis(df)
        if sync is not None:
            scores['VolumeOISync'] = normalize_score(sync, (15, 40), (-15, -40))
        obm = option_buyer_momentum(df)
        if obm is not None:
            scores['OptionBuyerMomentum'] = normalize_score(obm, (20, 50), (-20, -50), (-3.0, 3.0))
    except Exception as e:
        logger.error(f"Error in indicator scores: {e}")
    return scores

def analyze_signals_pro(timeframe_data, market_regime='neutral'):
    total_score, total_weight = 0.0, 0.0
    group_scores = defaultdict(float)
    group_weights = defaultdict(float)
    for tf_min, df in timeframe_data.items():
        if df is None or len(df) < 50: continue
        indicator_scores = calculate_indicator_scores(df)
        tf_weight = TIMEFRAME_WEIGHTS.get(tf_min, 1.0)
        for group, weight in Config.GROUP_WEIGHTS.items():
            grp_score, grp_weight = 0.0, 0.0
            for indicator, ind_weight in Config.INDICATOR_WEIGHTS.items():
                if indicator in indicator_scores:
                    belongs_to_group = (
                        (group == 'Trend' and any(term in indicator for term in ['MA', 'ADX', 'VWAP', 'EMA', 'MACD'])) or
                        (group == 'Momentum' and any(term in indicator for term in ['RSI', 'Stochastic', 'CCI', 'ROC', 'Williams'])) or
                        (group == 'Volume' and any(term in indicator for term in ['Vol', 'OBV', 'CMF'])) or
                        (group == 'Volatility' and any(term in indicator for term in ['Volatility', 'Bollinger'])) or
                        (group == 'OI' and any(term in indicator for term in ['OI', 'Option']))
                    )
                    if belongs_to_group:
                        grp_score += indicator_scores[indicator] * ind_weight
                        grp_weight += abs(indicator_scores[indicator]) * ind_weight
            if grp_weight > 0:
                norm_grp_score = (grp_score / grp_weight) * weight * tf_weight
                group_scores[group] += norm_grp_score
                group_weights[group] += weight * tf_weight
    final_score = sum(group_scores.values())
    max_possible_score = sum(group_weights.values())
    if max_possible_score == 0: return 'Neutral', 0.0, {}
    normalized_score = (final_score / max_possible_score) * 100
    if normalized_score > 0 and market_regime == 'bullish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bullish_in_bull_market']
    elif normalized_score > 0 and market_regime == 'bearish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bullish_in_bear_market']
    elif normalized_score < 0 and market_regime == 'bearish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bearish_in_bear_market']
    elif normalized_score < 0 and market_regime == 'bullish':
        normalized_score *= Config.REGIME_MULTIPLIERS['bearish_in_bull_market']
    if normalized_score >= Config.SIGNAL_THRESHOLDS['Very Strong Buy']:
        signal = 'Very Strong Buy'
    elif normalized_score >= Config.SIGNAL_THRESHOLDS['Strong Buy']:
        signal = 'Strong Buy'
    elif normalized_score >= Config.SIGNAL_THRESHOLDS['Buy Signal']:
        signal = 'Buy Signal'
    elif normalized_score <= Config.SIGNAL_THRESHOLDS['Very Strong Sell']:
        signal = 'Very Strong Sell'
    elif normalized_score <= Config.SIGNAL_THRESHOLDS['Strong Sell']:
        signal = 'Strong Sell'
    elif normalized_score <= Config.SIGNAL_THRESHOLDS['Sell Signal']:
        signal = 'Sell Signal'
    else:
        signal = 'Neutral'
    final_sub_scores = {group: group_scores[group] / group_weights[group] * 10 if group_weights[group] > 0 else 0 for group in group_scores}
    return signal, normalized_score, final_sub_scores

# 5-min Volume/OI Tracking
def calculate_5min_volume_oi_changes(df, symbol, scan_time):
    try:
        df_5min = df[df.index <= scan_time]
        if len(df_5min) < 2:
            return 0, None, 0, 0
        current_volume = int(df_5min['volume'].iloc[-1]) if 'volume' in df_5min.columns else 0
        previous_volume = int(df_5min['volume'].iloc[-2]) if 'volume' in df_5min.columns else 0
        vol_change_pct = ((current_volume - previous_volume) / previous_volume * 100) if previous_volume > 0 else 0
        if _has_real_oi(df_5min):
            current_oi = int(df_5min['openinterest'].iloc[-1])
            previous_oi = int(df_5min['openinterest'].iloc[-2])
            oi_change_pct = ((current_oi - previous_oi) / previous_oi * 100) if previous_oi > 0 else 0
        else:
            current_oi, oi_change_pct = None, 0
        return current_volume, current_oi, vol_change_pct, oi_change_pct
    except Exception as e:
        logger.error(f"Error in 5-min changes for {symbol}: {e}")
        return 0, None, 0, 0

def extract_5min_volume_oi_data(df, symbol, time_point=None, is_live=False):
    try:
        df_slice = df if is_live or time_point is None else df[df.index <= time_point]
        if df_slice.empty:
            return {'current_volume': 'N/A', 'current_oi': 'N/A', 'volume_change_pct': 0, 'oi_change_pct': 0}
        current_volume, current_oi, vol_change_pct, oi_change_pct = calculate_5min_volume_oi_changes(df_slice, symbol, df_slice.index[-1])
        if abs(vol_change_pct) < 0.1 and abs(oi_change_pct) < 0.1:
            prev_volume = intraday_volume_data.get(symbol, None)
            prev_oi = intraday_oi_data.get(symbol, None)
            if prev_volume is not None and prev_volume > 0 and current_volume > 0:
                vol_change_pct = ((current_volume - prev_volume) / prev_volume) * 100
            if prev_oi is not None and prev_oi > 0 and current_oi > 0:
                oi_change_pct = ((current_oi - prev_oi) / prev_oi) * 100
        intraday_volume_data[symbol] = current_volume if isinstance(current_volume, int) else 0
        intraday_oi_data[symbol] = current_oi if isinstance(current_oi, int) else 0
        return {
            'current_volume': f"{current_volume:,}" if current_volume > 999 else str(current_volume),
            'current_oi': f"{current_oi:,}" if current_oi and current_oi > 999 else str(current_oi) if current_oi is not None else 'N/A',
            'volume_change_pct': vol_change_pct if abs(vol_change_pct) > 0.1 else 0,
            'oi_change_pct': oi_change_pct if abs(oi_change_pct) > 0.1 else 0,
            '_raw_volume': current_volume if isinstance(current_volume, int) else 0,
            '_raw_oi': current_oi if isinstance(current_oi, int) else 0
        }
    except Exception as e:
        logger.error(f"Error extracting 5-min data for {symbol}: {e}")
        return {'current_volume': 'N/A', 'current_oi': 'N/A', 'volume_change_pct': 0, 'oi_change_pct': 0}

# Timing Functions
def next_5min_boundary_ist(now_ist):
    minute = (now_ist.minute // 5) * 5
    boundary = now_ist.replace(minute=minute, second=0, microsecond=0)
    if boundary <= now_ist:
        boundary += timedelta(minutes=5)
    return boundary

def get_exact_candle_close_time(now_ist):
    next_boundary = next_5min_boundary_ist(now_ist)
    return next_boundary + timedelta(seconds=Config.SETTLE_DELAY_SECONDS)

def today_ist_dt(hhmm):
    now = datetime.now(IST)
    h, m = map(int, hhmm.split(":"))
    return now.replace(hour=h, minute=m, second=0, microsecond=0)

def sleep_until(ts):
    while True:
        now = datetime.now(IST)
        delta = (ts - now).total_seconds()
        if delta <= 0:
            break
        time.sleep(min(0.5, delta))

# Data Fetching
class TokenBucketLimiter:
    def __init__(self, rate_per_sec, bucket_size):
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
            time.sleep(max(0.0, 1.0 / self.rate))

def authenticate_session():
    return TD_hist(Config.TDUSERNAME, Config.TDPASSWORD, log_level=logging.CRITICAL)

def build_sessions():
    pool = []
    for i in range(Config.TD_HIST_SESSIONS):
        try:
            pool.append(authenticate_session())
        except Exception as e:
            logger.error(f"Session {i} failed: {e}")
    if not pool:
        raise SystemExit("No TrueData sessions.")
    per_sess_rate = 10.0 / len(pool)
    limiters = [TokenBucketLimiter(per_sess_rate, 10) for _ in pool]
    return pool, limiters

tdhist_pool, sess_limiters = build_sessions()

def normalize_hist_df(df, symbol):
    if df is None or len(df) == 0: return None
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    rename_map = {
        "timestamp": "date", "time": "date", "datetime": "date", "date": "date",
        "open": "open", "high": "high", "low": "low", "close": "close",
        "volume": "volume", "vol": "volume",
        "oi": "openinterest", "openinterest": "openinterest", "open_interest": "openinterest"
    }
    out.rename(columns={k: v for k, v in rename_map.items() if k in out.columns}, inplace=True)
    if "date" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
        out["date"] = out.index
    if "volume" not in out.columns:
        out["volume"] = 0
    if "openinterest" in out.columns:
        out["openinterest"] = pd.to_numeric(out["openinterest"], errors="coerce")
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out.dropna(subset=["date"], inplace=True)
    out["date"] = out["date"].dt.tz_localize(IST) if not pd.api.types.is_datetime64tz_dtype(out["date"]) else out["date"].dt.tz_convert(IST)
    for c in ["open", "high", "low", "close", "volume"]:
        out[c] = pd.to_numeric(out.get(c, np.nan), errors="coerce")
    out.dropna(subset=["open", "high", "low", "close"], inplace=True)
    out.sort_values("date", inplace=True)
    out.set_index("date", inplace=True)
    out = out[~out.index.duplicated(keep='last')]
    if len(out) == 0:
        return None
    return out

def pick_session(symbol, timeframe_minutes):
    return hash(symbol) ^ timeframe_minutes % len(tdhist_pool)

def fetch_one_timeaware(symbol_orig, timeframe_minutes, limiter, hist, up_to_time):
    td_symbol = symbol_orig.replace('-EQ', '')
    bar_size = BAR_SIZE_MAP.get(timeframe_minutes)
    duration_str = DURATION_MAP.get(timeframe_minutes)
    if not bar_size or not duration_str:
        return symbol_orig, timeframe_minutes, None
    try:
        limiter.acquire()
        if up_to_time:
            dur_num, dur_unit = duration_str.split()
            dur_num = int(dur_num)
            delta = timedelta(days=dur_num) if dur_unit == 'D' else timedelta(days=dur_num)
            start_time = up_to_time - delta
            df_raw = hist.get_historic_data(td_symbol, start_time=start_time, end_time=up_to_time, bar_size=bar_size)
        else:
            df_raw = hist.get_historic_data(td_symbol, duration=duration_str, bar_size=bar_size)
        df = normalize_hist_df(df_raw, td_symbol)
        return symbol_orig, timeframe_minutes, df
    except Exception as e:
        logger.error(f"Fetch error {symbol_orig} {timeframe_minutes}min: {e}")
        return symbol_orig, timeframe_minutes, None

def prefetch_all_timeaware(stocks, up_to_time=None, max_workers=Config.MAX_WORKERS):
    tfs = [5, 15, 30, 60, 1440]
    total_calls = len(stocks) * len(tfs)
    stock_multi_data = defaultdict(dict)
    with tqdm(total=total_calls, desc="Fetching data", leave=False) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for s in stocks:
                for tf in tfs:
                    session_idx = pick_session(s, tf)
                    futures.append(executor.submit(fetch_one_timeaware, s, tf, sess_limiters[session_idx], tdhist_pool[session_idx], up_to_time))
            for fut in as_completed(futures):
                symbol_orig, tf, df = fut.result()
                if df is not None and len(df) > 0:
                    stock_multi_data[symbol_orig][tf] = df
                pbar.update(1)
    return {s: d for s, d in stock_multi_data.items() if len(d) >= 2}

def get_market_regime(index_symbol="NIFTY 50"):
    try:
        si = pick_session(index_symbol, 1440)
        df_raw = tdhist_pool[si].get_historic_data(index_symbol, duration="200 D", bar_size="1 day")
        df = normalize_hist_df(df_raw, index_symbol)
        if df is None or len(df) < 50: return 'neutral'
        ema20 = ema(df['close'], 20)
        ema50 = ema(df['close'], 50)
        if ema20.empty or ema50.empty: return 'neutral'
        close = df['close'].iloc[-1]
        if close > ema20.iloc[-1] and ema20.iloc[-1] > ema50.iloc[-1]:
            return 'bullish'
        elif close < ema20.iloc[-1] and ema20.iloc[-1] < ema50.iloc[-1]:
            return 'bearish'
        return 'neutral'
    except Exception as e:
        logger.warning(f"Market regime error: {e}")
        return 'neutral'

def enhanced_institutional_flow_analysis(tf_data):
    frames = [tf_data.get(t) for t in (5, 15, 30) if tf_data.get(t) is not None and len(tf_data.get(t)) >= 60]
    if not frames: return "Unknown"
    votes = 0
    for df in frames:
        cmf_series = cmf(df, 20)
        rv_series = relative_volume(df, 50)
        if cmf_series.empty or rv_series.empty: continue
        cmf_last = cmf_series.iloc[-1]
        rv_last = rv_series.iloc[-1]
        if cmf_last > 0.05 and rv_last > 1.2: votes += 1
        elif cmf_last < -0.05 and rv_last > 1.2: votes -= 1
    if votes >= 2: return "Accumulation"
    elif votes <= -2: return "Distribution"
    return "Mixed"

# Main Scanner Logic
def run_scan_at_time_5min_fixed(time_point_aware, stocks, market_regime, is_live=False):
    stock_multi_data = prefetch_all_timeaware(stocks, time_point_aware if not is_live else None)
    signals_this_scan = []
    current_symbols = set()
    for symbol, timeframe_data in stock_multi_data.items():
        clean_symbol = symbol.replace('-EQ', '')
        current_symbols.add(clean_symbol)
        filtered_timeframes = {}
        for tf, df in timeframe_data.items():
            if df is not None and not df.empty:
                df_slice = df if is_live else df[df.index <= time_point_aware]
                if not df_slice.empty and len(df_slice) >= 50:
                    filtered_timeframes[tf] = df_slice
        if len(filtered_timeframes) < 2: continue
        signal, score, sub_scores = analyze_signals_pro(filtered_timeframes, market_regime)
        if abs(score) >= Config.SCORE_THRESHOLD_MIN:
            flow_tag = enhanced_institutional_flow_analysis(filtered_timeframes)
            tf_5min = filtered_timeframes.get(5)
            if tf_5min is not None:
                oi_vol_data = extract_5min_volume_oi_data(tf_5min, clean_symbol, time_point_aware, is_live)
            else:
                main_tf = filtered_timeframes.get(15, filtered_timeframes.get(30, list(filtered_timeframes.values())[0]))
                oi_vol_data = extract_5min_volume_oi_data(main_tf, clean_symbol, time_point_aware, is_live)
            vol_multi = 0.0
            if tf_5min is not None and len(tf_5min) >= 20:
                sma20_vol = tf_5min['volume'].rolling(20).mean().iloc[-1]
                current_vol = oi_vol_data['_raw_volume']
                vol_multi = current_vol / sma20_vol if sma20_vol > 0 else 0.0
            if vol_multi < 2.5: continue
            action = "Consider Call" if score > 0 else "Consider Put"
            if 'Strong' in signal: action = f"Strong {'Call' if score > 0 else 'Put'} Buy"
            signals_this_scan.append({
                'symbol': clean_symbol,
                'signal': signal,
                'score': score,
                'sub_scores': sub_scores,
                'flow': flow_tag,
                'action': action,
                'vol_multi': round(vol_multi, 1),
                **oi_vol_data
            })
    return signals_this_scan, current_symbols

# Main Function
def main():
    parser = argparse.ArgumentParser(description="Pressure Scanner")
    args = parser.parse_args()
    with open(Config.SHARES_FILE, 'r') as f:
        stocks = [line.strip().upper() for line in f if line.strip()]
    logger.info(f"Loaded {len(stocks)} stocks")
    global scan_count, previous_scan_results, intraday_volume_data, intraday_oi_data
    scan_count = 0
    previous_scan_results = {}
    intraday_volume_data = {}
    intraday_oi_data = {}
    now_ist = datetime.now(IST)
    first_run_time = today_ist_dt(Config.FIRST_RUN_AT) + timedelta(seconds=Config.FIRST_SCAN_DELAY)
    if now_ist < first_run_time:
        sleep_until(first_run_time)
    while True:
        scan_count += 1
        now_ist = datetime.now(IST)
        market_end = today_ist_dt(Config.MARKET_END)
        if now_ist > market_end:
            logger.info("Market closed.")
            break
        print_colored(f"[{now_ist}] LIVE SCAN #{scan_count}", Colors.HEADER)
        market_regime = get_market_regime(Config.BENCHMARK_INDEX)
        signals, current_symbols = run_scan_at_time_5min_fixed(now_ist, stocks, market_regime, True)
        new_stocks = current_symbols - set(previous_scan_results.keys())
        previous_scan_results = {s: True for s in current_symbols}
        signals.sort(key=lambda x: abs(x['score']), reverse=True)
        top_bullish = [s for s in signals if s['score'] > 0][:2]
        top_bearish = [s for s in signals if s['score'] < 0][:2]
        create_simple_table(top_bullish, "Top 1-2 Buying Pressure", new_stocks)
        create_simple_table(top_bearish, "Top 1-2 Selling Pressure", new_stocks)
        next_scan = get_exact_candle_close_time(now_ist)
        sleep_until(next_scan)

if __name__ == "__main__":
    main()