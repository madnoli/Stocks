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

    # Output and logging
    "SHARES_FILE": os.getenv("SHARES_FILE", "shares.txt"),
    "SHOW_PROGRESS": os.getenv("SHOW_PROGRESS", "true").lower() == "true",
    "LOG_LEVEL": os.getenv("LOG_LEVEL", "WARNING"),

    "SKIP_DAILY": False,

    # Analysis settings
    "MIN_BARS_REQUIRED": 20,
    "MAX_MISSING_DATA_PCT": 15,
    "SIGNAL_CONFIRMATION_BARS": 2,
    "MIN_SIGNAL_THRESHOLD": 10,

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
        "OISurge": 3.5, "OIMomentum": 3.2, "CallBias": 4.0, "PutBias": 4.0, "OIVolConfirm": 3.0,
    },

    "TIMEFRAME_WEIGHTS": {15: 2.5, 5: 2.2, 30: 1.8, 60: 1.2, 1440: 1.0},

    # API mapping - FIXED FOR LIVE MODE
    "BAR_SIZE_MAP": {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 min", 1440: "1 day"},
    # LIVE MODE: Reduced duration for real-time data
    "DURATION_MAP": {5: "5 D", 15: "10 D", 30: "15 D", 60: "30 D", 1440: "90 D"},
    # BACKTEST MODE: Full duration
    "BACKTEST_DURATION_MAP": {5: "45 D", 15: "45 D", 30: "90 D", 60: "180 D", 1440: "365 D"},
}

level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR, "CRITICAL": logging.CRITICAL}
logging.getLogger().setLevel(level_map.get(CONFIG["LOG_LEVEL"], logging.WARNING))

IST = pytz.timezone("Asia/Kolkata")

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
is_live_mode = False  # Global flag for live mode

# Helper functions
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

def get_live_cutoff_time():
    """Get the cutoff time for live data - current completed 5-minute bar"""
    now_ist = datetime.now(IST)
    minute = (now_ist.minute // 5) * 5
    cutoff = now_ist.replace(minute=minute, second=0, microsecond=0)
    if cutoff == now_ist.replace(second=0, microsecond=0):
        cutoff -= timedelta(minutes=5)
    return cutoff

# Token-bucket limiter
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

# TrueData sessions
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

# [Continue with all remaining functions - indicator calculations, analysis, rendering, live mode, etc.]
