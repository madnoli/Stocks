import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import requests
import json
import time as timemodule
from logzero import logger
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from truedata.history import TD_hist
import logging
import warnings
warnings.filterwarnings("ignore")

# =========================
# --- TRUE DATA CONFIG ---
# =========================
TDUSERNAME = os.getenv("TD_USERNAME", "tdwsp751")
TDPASSWORD = os.getenv("TD_PASSWORD", "raj@751")
tdhist = TD_hist(TDUSERNAME, TDPASSWORD, log_level=logging.WARNING)  # Uses TrueData historical API for live/replay windows [web:41][web:38]

# =========================
# --- RUN MODES ---
# =========================
RUN_MODE = os.getenv("RUN_MODE", "replay")          # "live" or "replay" [web:38]
REPLAY_DATE = os.getenv("REPLAY_DATE", "2025-08-20") # YYYY-MM-DD [web:41]
REPLAY_START = os.getenv("REPLAY_START", "09:25")    # HH:MM IST [web:41]
REPLAY_END = os.getenv("REPLAY_END", "15:00")        # HH:MM IST [web:41]
REPLAY_SPEED = float(os.getenv("REPLAY_SPEED", "1.0"))  # 1.0 = real-time pacing; 0 = fastest [web:41]

# =========================
# --- HTTP SESSION & RATE LIMIT ---
# =========================
HTTP_SESSION = requests.Session()  # Reuses TCP connections to reduce overhead [web:91][web:95]
API_SEMAPHORE = Semaphore(6)       # Limits concurrent HTTP calls for stability [web:93]

# =========================
# --- COLOR CODES ---
# =========================
class Colors:
    GREEN = "\033[92m"
    RED   = "\033[91m"
    YELLOW= "\033[93m"
    BLUE  = "\033[94m"
    CYAN  = "\033[96m"
    MAGENTA="\033[95m"
    WHITE = "\033[97m"
    BOLD  = "\033[1m"
    RESET = "\033[0m"

# =========================
# --- CONTROL FLAGS ---
# =========================
BUYER_MODE = True
RELAXED_MODE = True

# =========================
# --- INDICATOR WEIGHTS ---
# =========================
ENHANCED_INDICATOR_WEIGHTS = {
    "VolumeSurge": 2.0, "Momentum": 1.9, "ADX": 1.8, "VWAP": 1.7, "EMA": 1.7,
    "MACD": 1.5, "OBV": 1.5, "ATR": 1.4, "Bollinger": 1.3, "RSI": 1.2,
    "ROC": 1.1, "Stochastic": 1.0, "CCI": 1.0, "MA": 1.0, "WWL": 1.0
}
TIMEFRAME_WEIGHTS = {15: 3.0, 5: 2.5, 30: 2.0, 60: 1.5, "daily": 1.0}

# =========================
# --- SECTOR MAPS (CURATED) ---
# =========================
NSE_INDEX_TO_SECTOR = {
    "NIFTY FMCG": "Consumer",
    "NIFTY PHARMA": "Pharma",
    "NIFTY IT": "Technology",
    "NIFTY BANK": "Finance",
    "NIFTY PSU BANK": "PSU Bank",
    "NIFTY ENERGY": "Energy",
    "NIFTY REALTY": "Realty",
}  # Filters to equity sectors only for meaningful momentum boosts [web:38][web:41]

SECTOR_STOCKS = {
    "Consumer":  ["HINDUNILVR","NESTLEIND","TATACONSUM","DABUR","GODREJCP","MARICO","ITC","BRITANNIA"],
    "Pharma":    ["SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","LUPIN","AUROPHARMA","TORNTPHARM","BIOCON"],
    "Technology":["TCS","INFY","HCLTECH","LTIM","TECHM","WIPRO","PERSISTENT","COFORGE"],
    "Finance":   ["HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","SBIN","BAJFINANCE","BAJAJFINSV","HDFCLIFE"],
    "PSU Bank":  ["SBIN","PNB","BANKBARODA","CANBK","UNIONBANK","INDIANB","BANKINDIA","IDBI"],
    "Energy":    ["RELIANCE","ONGC","COALINDIA","BPCL","IOC","NTPC","POWERGRID","TATAPOWER"],
    "Realty":    ["DLF","LODHA","OBEROIRLTY","GODREJPROP","PRESTIGE","PHOENIXLTD","BRIGADE","SOBHA"],
}  # Expanded F&O baskets increase candidate throughput and signal odds per cycle [web:38][web:41]

# =========================
# --- BUY/SELL/DELTA (OHLCV) ---
# =========================
def add_buy_sell_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Intrabar Buying/Selling Pressure proxy:
      BuyVol ~ Volume * (Close-Low)/(High-Low)
      SellVol ~ Volume * (High-Close)/(High-Low)
      Tie/zero-range bars split with slight bias by direction vs prior close.
    """
    o = df["Open"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    c = df["Close"].astype(float)
    v = df["Volume"].fillna(0).astype(float)

    rng = (h - l).replace(0, np.nan)
    up_pressure = (c - l) / rng
    down_pressure = (h - c) / rng

    up_pressure = up_pressure.fillna(0.5).clip(0, 1)
    down_pressure = down_pressure.fillna(0.5).clip(0, 1)

    equal_mask = (abs(up_pressure - down_pressure) < 1e-12)
    prev_c = c.shift(1)
    dir_up = (c > prev_c).astype(float).fillna(0.0)
    dir_down = (c < prev_c).astype(float).fillna(0.0)
    neutral = (1.0 - dir_up - dir_down).clip(0, 1)

    up_adj = np.where(equal_mask, 0.6 * dir_up + 0.5 * neutral + 0.4 * dir_down, up_pressure)
    down_adj = np.where(equal_mask, 0.4 * dir_up + 0.5 * neutral + 0.6 * dir_down, down_pressure)

    total = up_adj + down_adj
    total = np.where(total == 0, 1.0, total)
    up_share = up_adj / total
    down_share = down_adj / total

    df["BuyVol"] = v * up_share
    df["SellVol"] = v * down_share
    df["DeltaVol"] = df["BuyVol"] - df["SellVol"]
    return df  # Standard OHLCV-only Volume Delta approximation when tick bid/ask not available [web:1][web:8][web:5]

# =========================
# --- INDICATORS ---
# =========================
class EnhancedTechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        indicators = {}
        if df is None or len(df) < 20: return indicators
        try:
            close = df["Close"]; high = df["High"]; low = df["Low"]; vol = df["Volume"]

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            indicators["RSI"] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            indicators["MACD"] = macd_line - signal_line

            # Stochastic
            low14 = low.rolling(14).min()
            high14 = high.rolling(14).max()
            indicators["Stochastic"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)

            # MAs
            indicators["MA"] = close.rolling(20).mean()
            indicators["EMA"] = close.ewm(span=21).mean()

            # ADX and ATR
            high_diff = high.diff(); low_diff = low.diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
            minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff > 0), 0.0)
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(14).mean()
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr.replace(0, np.nan))
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr.replace(0, np.nan))
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
            indicators["ADX"] = dx.rolling(14).mean()
            indicators["ATR"] = atr

            # Bollinger position
            ma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20
            indicators["Bollinger"] = (close - ma20) / (upper - lower).replace(0, np.nan) * 100

            # ROC, OBV, CCI, WWL
            indicators["ROC"] = close.pct_change(12) * 100
            obv = np.sign(close.diff().fillna(0)) * vol.fillna(0); obv = obv.cumsum()
            indicators["OBV"] = obv.pct_change(10) * 100
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(20).mean()
            mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
            indicators["CCI"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
            hh = high.rolling(14).max()
            ll = low.rolling(14).min()
            indicators["WWL"] = (hh - close) / (hh - ll).replace(0, np.nan) * -100

            # VWAP and VolumeSurge
            if len(df) >= 20:
                tpv = (high + low + close) / 3
                vwap_num = (tpv * vol).rolling(20).sum()
                vwap_den = vol.rolling(20).sum().replace(0, np.nan)
                indicators["VWAP"] = vwap_num / vwap_den
                avg20 = vol.rolling(20).mean()
                vr = (vol / avg20.replace(0, np.nan))
                indicators["VolumeSurge"] = np.clip((vr - 0.5) * 40, 0, 100)

            # Momentum
            if len(df) >= 10:
                price_mom = close.pct_change(10) * 100
                avg10 = vol.rolling(10).mean()
                vol_mom = (vol / avg10.replace(0, np.nan) - 1) * 100
                mom_score = price_mom * 0.7 + vol_mom * 0.3
                indicators["Momentum"] = np.clip(50 + mom_score * 1.5, -50, 50)

            return indicators
        except Exception as e:
            logger.error(f"Indicator calc error: {e}")
            return indicators  # Indicator stability is essential for replay/live parity [web:98][web:92]

def normalize_indicator_value(indicator_name, value):
    try:
        if indicator_name == "RSI": return max(0, min(100, value))
        if indicator_name == "MACD": return 50 + max(-25, min(25, value / 10))
        if indicator_name == "Stochastic": return max(0, min(100, value))
        if indicator_name in ("MA", "EMA", "VWAP"): return 50
        if indicator_name == "ADX": return max(0, min(100, value))
        if indicator_name == "Bollinger": return max(0, min(100, (value + 100) / 2))
        if indicator_name == "ROC": return 50 + max(-25, min(25, value / 2))
        if indicator_name == "OBV": return 50 + max(-25, min(25, value))
        if indicator_name == "CCI": return max(0, min(100, (value + 200) / 4))
        if indicator_name == "WWL": return max(0, min(100, (value + 100)))
        if indicator_name == "ATR": return 50
        if indicator_name == "VolumeSurge": return max(0, min(100, value))
        if indicator_name == "Momentum": return max(0, min(100, value + 50))
        return 50
    except Exception:
        return 50  # Bounded normalization improves robustness across TFs [web:98][web:92]

# =========================
# --- OPTION CHAIN + PCR ---
# =========================
def get_option_chain(symbol: str, when: datetime | None = None):
    """
    Wire to live broker API (Angel/Dhan/Kite) or a historical archive keyed by 'when'.
    Return iterable rows with CE/PE OI; function returns None if unavailable.
    """
    # Example: with API_SEMAPHORE: HTTP_SESSION.get(url).json()
    return None  # PCR hook; compute as Put OI / Call OI when chain available [web:16][web:36]

def compute_pcr_from_option_chain(option_chain) -> float | None:
    try:
        if option_chain is None: return None
        total_put_oi = 0.0; total_call_oi = 0.0
        for row in option_chain:
            p_oi = None; c_oi = None
            for f in ("put_oi","putOI","PE_oi","PE.oi","p_oi","putOpenInterest"):
                if isinstance(row, dict) and f in row and row[f] is not None: p_oi = float(row[f]); break
            for f in ("call_oi","callOI","CE_oi","CE.oi","c_oi","callOpenInterest"):
                if isinstance(row, dict) and f in row and row[f] is not None: c_oi = float(row[f]); break
            if p_oi is None and isinstance(row, dict) and "PE" in row and isinstance(row["PE"], dict):
                for f in ("oi","openInterest"):
                    if f in row["PE"] and row["PE"][f] is not None: p_oi = float(row["PE"][f]); break
            if c_oi is None and isinstance(row, dict) and "CE" in row and isinstance(row["CE"], dict):
                for f in ("oi","openInterest"):
                    if f in row["CE"] and row["CE"][f] is not None: c_oi = float(row["CE"][f]); break
            if p_oi is not None: total_put_oi += p_oi
            if c_oi is not None: total_call_oi += c_oi
        if total_call_oi <= 0: return None
        return total_put_oi / total_call_oi  # PCR(OI) = Put OI / Call OI, standard definition [web:36][web:16]
    except Exception as e:
        logger.warning(f"PCR compute error: {e}")
        return None

# =========================
# --- SCANNER ---
# =========================
class Enhanced3SectorScanner:
    def __init__(self):
        self.is_running = False
        self.current_signals = []
        self.best_sectors = ["Pharma", "Healthcare", "Technology", "Finance"]
        self.worst_sectors = ["Defence", "Energy", "PSU Bank", "Realty"]
        self.sectoral_history = []
        self.last_sectoral_update = None
        self.api_errors = []
        self.sector_update_attempts = 0
        self.successful_updates = 0
        self.gapdown_filtered_count = 0
        self.last_cycle_scores = {}
        self.current_cycle_scores = {}
        self.market_start = time(9, 15)
        self.market_end = time(15, 30)
        self.scan_interval = 300  # 5 min
        self.sector_snapshots = []
        self.sector_snapshot_retention_min = 180

        # Replay state
        self.replay_mode = (RUN_MODE.lower() == "replay")
        self.replay_clock = None
        self.replay_start_dt = None
        self.replay_end_dt = None
        if self.replay_mode:
            self.init_replay_clock()  # Initialize early so now_dt() is valid in init flows [web:41][web:38]

        print(f"{Colors.CYAN}{Colors.BOLD}Scanner initialized{Colors.RESET}")

    # --- Replay clock helpers ---
    def init_replay_clock(self):
        d = datetime.strptime(REPLAY_DATE, "%Y-%m-%d").date()
        s_hour, s_min = map(int, REPLAY_START.split(":"))
        e_hour, e_min = map(int, REPLAY_END.split(":"))
        self.replay_start_dt = datetime(d.year, d.month, d.day, s_hour, s_min)
        self.replay_end_dt   = datetime(d.year, d.month, d.day, e_hour, e_min)
        self.replay_clock = self.replay_start_dt
        print(f"{Colors.MAGENTA}Replay initialized: {self.replay_start_dt} -> {self.replay_end_dt}{Colors.RESET}")  # Stable replay window [web:41]

    def advance_replay_clock(self, seconds: int):
        if not self.replay_mode: return
        self.replay_clock += timedelta(seconds=seconds)
        if self.replay_clock > self.replay_end_dt:
            self.replay_clock = self.replay_end_dt

    def now_dt(self):
        if self.replay_mode:
            return self.replay_clock or self.replay_start_dt or datetime.now()
        return datetime.now()

    # --- Sector snapshots ---
    def _trim_sector_snapshots(self):
        try:
            if not self.sector_snapshots: return
            cutoff = self.now_dt() - timedelta(minutes=self.sector_snapshot_retention_min)
            self.sector_snapshots = [
                s for s in self.sector_snapshots
                if isinstance(s.get("timestamp"), datetime) and s["timestamp"] >= cutoff
            ]
        except Exception as e:
            logger.warning(f"Trim snapshots error: {e}")  # Guards against None timestamps [web:77]

    def _get_sector_change_at(self, sector: str, ref_time: datetime, offset_min: int):
        try:
            target = ref_time - timedelta(minutes=offset_min)
            cands = [
                s for s in self.sector_snapshots
                if s.get("sector") == sector and isinstance(s.get("timestamp"), datetime) and s["timestamp"] <= target
            ]
            if not cands: return None
            return max(cands, key=lambda s: s["timestamp"])["change"]
        except Exception as e:
            logger.warning(f"Snapshot lookup error {sector}: {e}")
            return None  # Safe lookup with null checks [web:77]

    # --- Display helpers ---
    def show_initialization_status(self):
        print(f"{Colors.CYAN}{Colors.BOLD}ENHANCED SECTOR SCANNER (API + Buyer Gates){Colors.RESET}")
        print(f"Timeframes: 5m, 15m, 30m, 60m, Daily")
        print(f"Sector API: http://localhost:3001/api/allIndices")
        print(f"Buyer windows: 09:25–11:30, 13:45–15:00 IST")
        self.show_sector_status()
        self.test_api_connection()
        print(f"{Colors.YELLOW}Running initial sector update...{Colors.RESET}")
        if not self.replay_mode:
            self.force_sector_update()
        else:
            print(f"{Colors.YELLOW}Skipping initial sector update in replay (clock initialized).{Colors.RESET}")
        print("-"*88)

    def test_api_connection(self):
        print(f"{Colors.BLUE}API CONNECTION TEST{Colors.RESET}")
        try:
            with API_SEMAPHORE:
                r = HTTP_SESSION.get("http://localhost:3001/api/allIndices", timeout=5)
            print(f"API HTTP: {r.status_code}")
        except Exception as e:
            print(f"{Colors.RED}API ERROR: {e}{Colors.RESET}")

    def show_sector_status(self):
        print(f"{Colors.MAGENTA}Best: {', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"{Colors.MAGENTA}Worst: {', '.join(self.worst_sectors)}{Colors.RESET}")

    # --- Sector API fetch (filtered to curated indices) ---
    def fetch_live_sectoral_performance(self):
        try:
            with API_SEMAPHORE:
                r = HTTP_SESSION.get("http://localhost:3001/api/allIndices", timeout=8)
            print(f"{Colors.BLUE}API Status: {r.status_code}{Colors.RESET}")
            if r.status_code != 200:
                nowt = self.now_dt()
                data = [{"index": s, "sector": s, "changepercent": 0.0, "timestamp": nowt}
                        for s in (self.best_sectors + self.worst_sectors)]
                self.display_sector_update(data)
                return False

            indices_data = r.json()
            if isinstance(indices_data, str):
                indices_data = json.loads(indices_data)
            if isinstance(indices_data, dict):
                indices_data = indices_data.get("data") or indices_data.get("indices") or indices_data.get("results") or indices_data
            if not isinstance(indices_data, list):
                print(f"{Colors.RED}Unexpected API payload{Colors.RESET}")
                return False

            curtime = self.now_dt()
            sectoral = []
            for idx in indices_data:
                if not isinstance(idx, dict): continue
                name = next((str(idx[f]).strip().upper() for f in ("name","symbol","index","indexName") if f in idx and idx[f]), None)
                if not name or name not in NSE_INDEX_TO_SECTOR:
                    continue
                sector_name = NSE_INDEX_TO_SECTOR[name]
                change_percent = 0.0
                for f in ("changepercent","changePercent","pChange","percentChange","change","pchg"):
                    if f in idx and idx[f] is not None:
                        try:
                            change_percent = float(idx[f]); break
                        except: pass
                sectoral.append({"index": name, "sector": sector_name, "changepercent": change_percent, "timestamp": curtime})

            if not sectoral:
                print("No curated sectors resolved from API.")
                return False

            for row in sectoral:
                self.sector_snapshots.append({"timestamp": curtime, "sector": row["sector"], "change": float(row["changepercent"])})
            self._trim_sector_snapshots()

            sectoral.sort(key=lambda x: x["changepercent"], reverse=True)
            n = len(sectoral)
            self.best_sectors = [sectoral[i]["sector"] for i in range(min(4,n))]
            self.worst_sectors = [sectoral[-i]["sector"] for i in range(1, min(4,n)+1)]
            self.last_sectoral_update = curtime
            self.sectoral_history.append({"timestamp": curtime, "best": self.best_sectors[:], "worst": self.worst_sectors[:], "fulldata": sectoral[:]})
            self.sectoral_history = self.sectoral_history[-24:]

            self.display_sector_update(sectoral)
            return True
        except Exception as e:
            logger.error(f"API sector fetch error: {e}")
            self.api_errors.append((self.now_dt(), str(e)))
            return False  # Curated filter avoids inverse/bond indices for equity scans [web:38][web:41]

    def _print_top_sector_delta_table(self, sectoral):
        try:
            now = self.now_dt()
            curr_map = {r["sector"]: float(r["changepercent"]) for r in sectoral}
            def row_line(sector):
                cur = curr_map.get(sector)
                chg_5 = self._get_sector_change_at(sector, now, 5)
                chg_15 = self._get_sector_change_at(sector, now, 15)
                d5 = None if (cur is None or chg_5 is None) else (cur - chg_5)
                d15 = None if (cur is None or chg_15 is None) else (cur - chg_15)
                def fmt(x):
                    if x is None: return "na"
                    arrow = "↑" if x > 0 else ("↓" if x < 0 else "→")
                    return f"{x:+.2f}% {arrow}"
                now_str = "na" if cur is None else f"{cur:+.2f}%"
                color = Colors.WHITE
                if d5 is not None:
                    if d5 >= 0.15: color = Colors.GREEN + Colors.BOLD
                    elif d5 <= -0.15: color = Colors.RED + Colors.BOLD
                    else: color = Colors.YELLOW
                return f"{color}{sector:16s}{Colors.RESET} | {now_str:6s} | {fmt(d5):8s} | {fmt(d15):9s}"

            print(f"{Colors.CYAN}{Colors.BOLD}SECTOR MOMENTUM TABLE (Top 4 Best & Worst){Colors.RESET}")
            print(f"{Colors.BLUE}Sector            | Now %  | Δ vs 5m | Δ vs 15m{Colors.RESET}")
            print("-"*52)
            if self.best_sectors:
                print(f"{Colors.GREEN}{Colors.BOLD}Top 4 Best{Colors.RESET}")
                for s in self.best_sectors[:4]:
                    print(row_line(s))
            print("-"*52)
            if self.worst_sectors:
                print(f"{Colors.RED}{Colors.BOLD}Top 4 Worst{Colors.RESET}")
                for s in self.worst_sectors[:4]:
                    print(row_line(s))
            print("-"*52)
            print(f"{Colors.MAGENTA}Tip: Best with +Δ and Worst with -Δ improves option-buyer alignment.{Colors.RESET}")
        except Exception as e:
            logger.warning(f"Momentum table error: {e}")  # First cycle shows na deltas; fill from next snapshot [web:41]

    def display_sector_update(self, sectoral):
        ct = self.now_dt().strftime('%H:%M:%S')
        print(f"{Colors.MAGENTA}{Colors.BOLD}{'-'*100}{Colors.RESET}")
        print(f"SECTOR PERFORMANCE UPDATE - {ct} IST")
        print(f"Best: {Colors.GREEN}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"Worst: {Colors.RED}{', '.join(self.worst_sectors)}{Colors.RESET}")
        topn = min(6, len(sectoral))
        print("Top 6 Sectors")
        for i, s in enumerate(sectoral[:topn]):
            color = Colors.GREEN if s["sector"] in self.best_sectors else Colors.YELLOW
            print(f"{i+1}. {color}{s['sector']:<18}{Colors.RESET} {s['changepercent']:>6.2f}  {s['index']}")
        print("Bottom 6 Sectors")
        for i, s in enumerate(sectoral[-topn:]):
            color = Colors.RED if s["sector"] in self.worst_sectors else Colors.YELLOW
            pos = len(sectoral) - topn + i + 1
            print(f"{pos}. {color}{s['sector']:<18}{Colors.RESET} {s['changepercent']:>6.2f}  {s['index']}")
        self._print_top_sector_delta_table(sectoral)

    def force_sector_update(self):
        print(f"{Colors.YELLOW}FORCING LIVE SECTOR UPDATE...{Colors.RESET}")
        self.sector_update_attempts += 1
        ok = self.fetch_live_sectoral_performance()
        if ok: self.successful_updates += 1
        else: print(f"{Colors.RED}Sector update failed; using last known lists{Colors.RESET}")
        return ok

    # --- Market/time guards ---
    def is_market_open(self):
        now = self.now_dt()
        ct = now.time()
        return now.weekday() < 5 and (self.market_start <= ct <= self.market_end)  # Shared for live and replay windows [web:41]

    # --- Data utilities ---
    def normalize_live_data(self, df, symbol):
        try:
            if df is None or len(df) == 0: return None
            d = df.copy()
            d.rename(columns={c: c.lower() for c in d.columns}, inplace=True)
            cmap = {}
            for src, tgt in (("time","Date"),("timestamp","Date"),("date","Date"),
                             ("open","Open"),("high","High"),("low","Low"),("close","Close"),
                             ("vol","Volume"),("volume","Volume")):
                if src in d.columns: cmap[src] = tgt
            d.rename(columns=cmap, inplace=True)
            if "Date" not in d.columns:
                if isinstance(d.index, pd.DatetimeIndex): d["Date"] = d.index
                else:
                    for cand in ("datetime","barstarttime","bartime","time"):
                        if cand in d.columns: d.rename(columns={cand:"Date"}, inplace=True); break
            req = ["Open","High","Low","Close"]
            if not all(c in d.columns for c in req): return None
            if "Volume" not in d.columns: d["Volume"] = 0
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce", utc=False)
            d = d.dropna(subset=["Date","Open","High","Low","Close"])
            for c in ["Open","High","Low","Close","Volume"]:
                d[c] = pd.to_numeric(d[c], errors="coerce")
            d = d.dropna(subset=["Open","High","Low","Close"])
            d.set_index("Date", inplace=True, drop=True)
            d = d.sort_index()
            if len(d) < 20: return None
            try:
                d = add_buy_sell_delta_columns(d)
            except Exception as e:
                logger.warning(f"Buy/Sell/Delta calc failed {symbol}: {e}")
                d["BuyVol"] = 0.0; d["SellVol"] = 0.0; d["DeltaVol"] = 0.0
            return d
        except Exception as e:
            logger.error(f"Normalize error {symbol}: {e}")
            return None  # Robust normalization for indicator stability [web:98][web:92]

    def _ohlc_resample(self, df, rule):
        agg = {"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum",
               "BuyVol":"sum","SellVol":"sum","DeltaVol":"sum"}
        df2 = df
        if "Open" not in df2.columns:
            # normalize title-case if needed
            cn = {c: c.title() for c in df2.columns}
            df2 = df2.rename(columns=cn)
        rs = df2.resample(rule, label="right", closed="right").agg(agg).dropna(subset=["Open","High","Low","Close"])
        return rs

    def fetch_symbol_bars_once(self, symbol: str, end_dt: datetime):
        """
        Pull 1-minute bars once per cycle, compute Buy/Sell/Delta, then downsample locally to 5/15/30/60/daily.
        Greatly reduces API calls and cycle time for I/O-bound scanning.
        """
        try:
            start_dt = end_dt - timedelta(days=10)
            raw = tdhist.get_historic_data(symbol, bar_size="1 min", start_time=start_dt, end_time=end_dt)
            if raw is None or len(raw) == 0:
                return {}
            df1 = self.normalize_live_data(raw, symbol)
            if df1 is None or len(df1) < 60:
                return {}
            mdf = df1.copy()
            if not isinstance(mdf.index, pd.DatetimeIndex):
                mdf.index = pd.to_datetime(mdf.index)
            out = {}
            for tf, rule in [(5,"5min"),(15,"15min"),(30,"30min"),(60,"60min")]:
                d = self._ohlc_resample(mdf, rule)
                if d is not None and len(d) >= 20:
                    out[tf] = d.tail(400 if tf==5 else 200)
            dly = self._ohlc_resample(mdf, "1D")
            if dly is not None and len(dly) >= 20:
                out["daily"] = dly.tail(400)
            return out
        except Exception as e:
            logger.error(f"One-shot fetch error {symbol}@{end_dt}: {e}")
            return {}  # One-shot + resample is a standard optimization for API-bound screeners [web:88][web:95]

    def fetch_live_data(self, symbol, timeframe):
        # Retained for compatibility; we prefer fetch_symbol_bars_once in process_symbol
        end_dt = self.now_dt()
        try:
            bar_size, st, en = self._tf_params_for_window(timeframe, end_dt)
            raw = tdhist.get_historic_data(symbol, bar_size=bar_size, start_time=st, end_time=en)
            if raw is None or len(raw) == 0: return None, False
            df = self.normalize_live_data(raw, symbol)
            if df is None or len(df) < 20: return None, False
            if timeframe == "daily": return df.tail(250), False
            if timeframe == 60: return df.tail(200), False
            return df.tail(100), False
        except Exception as e:
            logger.error(f"Live fetch error {symbol}@{timeframe}: {e}")
            return None, False

    def _tf_params_for_window(self, timeframe, end_dt: datetime):
        tfmap = {5:"5 min",15:"15 min",30:"30 min",60:"60 mins","daily":"EOD"}
        bar_size = tfmap.get(timeframe)
        if timeframe in (5,15): lookback_days = 10
        elif timeframe == 30: lookback_days = 20
        elif timeframe == 60: lookback_days = 60
        elif timeframe == "daily": lookback_days = 365
        else: lookback_days = 10
        start_dt = end_dt - timedelta(days=lookback_days)
        return bar_size, start_dt, end_dt

    # --- Scoring + buyer gates ---
    def calculate_enhanced_signals(self, symbol, timeframes_data):
        try:
            if not timeframes_data: return "Neutral", 0
            sector = next((s for s, lst in SECTOR_STOCKS.items() if symbol in lst), None)
            if not sector: return "Neutral", 0

            total_weighted = 0.0; total_w = 0.0; tf_scores = {}
            for tf, df in timeframes_data.items():
                if df is None or len(df) < 20: continue
                ind = EnhancedTechnicalIndicators.calculate_all_indicators(df)
                if not ind: continue
                tf_score = 0.0; tf_w = 0.0
                price = df["Close"].iloc[-1]
                for name, weight in ENHANCED_INDICATOR_WEIGHTS.items():
                    if name in ind and not ind[name].empty:
                        val = ind[name].iloc[-1]
                        if pd.isna(val): continue
                        if name in ("MA","EMA","VWAP"):
                            base = val
                            if pd.isna(base) or base == 0: norm = 50
                            else:
                                p = (price - base) / base * 100
                                norm = 75 if p >= 2 else 60 if p >= 0 else 50 if p >= -2 else 40 if p >= -5 else 25
                        else:
                            norm = normalize_indicator_value(name, val)
                        tf_score += norm * weight; tf_w += weight
                if tf_w <= 0: continue
                tf_final = tf_score / tf_w
                tf_scores[tf] = tf_final
                mult = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                total_weighted += tf_final * mult
                total_w += mult
            if total_w <= 0: return "Neutral", 0
            base_score = total_weighted / total_w

            # Confirmation bonus
            if len(tf_scores) >= 4:
                bull = sum(1 for v in tf_scores.values() if v >= 55)
                bear = sum(1 for v in tf_scores.values() if v <= 45)
                if bull >= 3: base_score += 8
                elif bear >= 3: base_score -= 8

            # Sector boost
            boost = 0
            has_long = ("daily" in timeframes_data) or (60 in timeframes_data)
            if sector in self.best_sectors:
                rank = self.best_sectors.index(sector) + 1
                boost_map = {1:25,2:20,3:15,4:10} if has_long else {1:20,2:15,3:10,4:5}
                boost = boost_map.get(rank,0)
            elif sector in self.worst_sectors:
                rank = self.worst_sectors.index(sector) + 1
                boost_map = {1:-25,2:-20,3:-15,4:-10} if has_long else {1:-20,2:-15,3:-10,4:-5}
                boost = boost_map.get(rank,0)
            base_score += boost

            # Buyer gates and thresholds
            if RELAXED_MODE:
                intended_buy = base_score >= 58
                intended_sell = base_score <= 42
            else:
                intended_buy = base_score >= 60
                intended_sell = base_score <= 40

            if not BUYER_MODE:
                if base_score >= 82: return "Very Strong Buy", base_score
                if base_score >= 72: return "Strong Buy", base_score
                if base_score >= 60: return "Buy", base_score
                if base_score <= 18: return "Very Strong Sell", base_score
                if base_score <= 28: return "Strong Sell", base_score
                if base_score <= 40: return "Sell", base_score
                return "Neutral", base_score

            try:
                tf_primary = 15 if 15 in timeframes_data else 5
                dfp = timeframes_data.get(tf_primary)
                indp = EnhancedTechnicalIndicators.calculate_all_indicators(dfp) if dfp is not None else {}
                price = dfp["Close"].iloc[-1] if dfp is not None else None
                ema = indp.get("EMA").iloc[-1] if "EMA" in indp and not indp["EMA"].empty else None
                vwap = indp.get("VWAP").iloc[-1] if "VWAP" in indp and not indp["VWAP"].empty else None

                adx5 = adx15 = None
                if 5 in timeframes_data:
                    i5 = EnhancedTechnicalIndicators.calculate_all_indicators(timeframes_data[5])
                    adx5 = i5.get("ADX").iloc[-1] if "ADX" in i5 and not i5["ADX"].empty else None
                if 15 in timeframes_data:
                    i15 = EnhancedTechnicalIndicators.calculate_all_indicators(timeframes_data[15])
                    adx15 = i15.get("ADX").iloc[-1] if "ADX" in i15 and not i15["ADX"].empty else None

                adx_thr = 18 if RELAXED_MODE else 22
                strong = (adx5 is not None and adx5 >= adx_thr) and (adx15 is not None and adx15 >= adx_thr)
                above = (price is not None and ema is not None and vwap is not None and price > ema and price > vwap)
                below = (price is not None and ema is not None and vwap is not None and price < ema and price < vwap)

                now = self.now_dt()
                d5 = d15 = None
                if self.sector_snapshots and sector:
                    ch5 = self._get_sector_change_at(sector, now, 5)
                    ch15 = self._get_sector_change_at(sector, now, 15)
                    rec = [s for s in self.sector_snapshots if s.get("sector") == sector and isinstance(s.get("timestamp"), datetime)]
                    cur = rec[-1]["change"] if rec else None
                    if cur is not None:
                        d5 = None if ch5 is None else (cur - ch5)
                        d15 = None if ch15 is None else (cur - ch15)

                # ATR/Volume gates on 5m
                atr_ok = vol_ok = False
                if 5 in timeframes_data:
                    df5 = timeframes_data[5]
                    i5b = EnhancedTechnicalIndicators.calculate_all_indicators(df5)
                    atrs = i5b.get("ATR")
                    if atrs is not None and not atrs.empty:
                        tail = atrs.tail(50).dropna()
                        if len(tail) >= 10:
                            atr_ok = (tail.rank(pct=True).iloc[-1]) >= (0.5 if RELAXED_MODE else 0.6)
                    vs = i5b.get("VolumeSurge")
                    if vs is not None and not vs.empty:
                        vol_ok = vs.iloc[-1] >= (50 if RELAXED_MODE else 60)

                # Buyer window (IST)
                t = self.now_dt().time()
                buyer_window = (time(9,25) <= t <= time(11,30)) or (time(13,45) <= t <= time(15,0))

                # TF agreement
                small_tfs = [tf_scores[k] for k in (5,15,30) if k in tf_scores]
                if RELAXED_MODE:
                    tf_buy_ok = (len(small_tfs) >= 3) and (sum(1 for v in small_tfs if v >= 56) >= 2)
                    tf_sell_ok = (len(small_tfs) >= 3) and (sum(1 for v in small_tfs if v <= 44) >= 2)
                else:
                    tflist = [tf_scores[k] for k in (5,15,30,60) if k in tf_scores]
                    tf_buy_ok = (len(tflist) >= 3) and all(v >= 58 for v in tflist[:3])
                    tf_sell_ok = (len(tflist) >= 3) and all(v <= 42 for v in tflist[:3])

                # Sector deltas
                if RELAXED_MODE:
                    d5_ok = (d5 is None) or (d5 > 0); d15_ok = (d15 is None) or (d15 > 0)
                    d5_ok_s = (d5 is None) or (d5 < 0); d15_ok_s = (d15 is None) or (d15 < 0)
                else:
                    d5_ok = (d5 is not None and d5 > 0); d15_ok = (d15 is not None and d15 > 0)
                    d5_ok_s = (d5 is not None and d5 < 0); d15_ok_s = (d15 is not None and d15 < 0)

                if intended_buy:
                    gates_pass = (strong and above and atr_ok and vol_ok and buyer_window and d5_ok and d15_ok and tf_buy_ok)
                    if not gates_pass: return "Neutral", min(base_score, 59.9)
                if intended_sell:
                    gates_pass = (strong and below and atr_ok and vol_ok and buyer_window and d5_ok_s and d15_ok_s and tf_sell_ok)
                    if not gates_pass: return "Neutral", max(base_score, 40.1)
            except Exception as e:
                logger.warning(f"Buyer gates error: {e}")

            if base_score >= 82: return "Very Strong Buy", base_score
            if base_score >= 72: return "Strong Buy", base_score
            if base_score >= 60: return "Buy", base_score
            if base_score <= 18: return "Very Strong Sell", base_score
            if base_score <= 28: return "Strong Sell", base_score
            if base_score <= 40: return "Sell", base_score
            return "Neutral", base_score
        except Exception as e:
            logger.error(f"Signal calc error {symbol}: {e}")
            return "Neutral", 0

    # --- One scan cycle ---
    def enhanced_scan_cycle(self):
        nowdt = self.now_dt()
        if not self.is_market_open():
            print(f"{Colors.YELLOW}Market closed at {nowdt.strftime('%H:%M:%S')} IST. Waiting {self.scan_interval//60}m...{Colors.RESET}")
            if self.replay_mode:
                self.advance_replay_clock(self.scan_interval)
                return
            timemodule.sleep(self.scan_interval); return

        t = nowdt.time()
        if BUYER_MODE and not ((time(9,25) <= t <= time(11,30)) or (time(13,45) <= t <= time(15,0))):
            print(f"{Colors.YELLOW}Outside buyer window; signals may be neutral-gated{Colors.RESET}")

        start = timemodule.time()
        print(f"{Colors.CYAN}Starting enhanced scan {nowdt.strftime('%H:%M:%S')}{Colors.RESET}")
        if not self.fetch_live_sectoral_performance():
            print("Sector update failed; proceeding with previous lists.")

        # Build targets from curated sectors
        targets = []
        for sec in (self.best_sectors + self.worst_sectors):
            if sec in SECTOR_STOCKS:
                targets.extend(SECTOR_STOCKS[sec][:12])  # scan 12 per sector
        if not targets:
            print(f"{Colors.YELLOW}No SECTOR_STOCKS configured; using sample symbols{Colors.RESET}")
            targets = ["RELIANCE","HDFCBANK","TCS","ICICIBANK","INFY","SBIN","AXISBANK","LTIM"]
        targets = sorted(set(targets))
        print(f"Scanning {len(targets)} symbols...")

        signals = []
        def process_symbol(sym):
            try:
                # One-shot fetch and local resampling to reduce API calls
                tfs_all = self.fetch_symbol_bars_once(sym, self.now_dt())
                tfs = {}
                for tf, df in tfs_all.items():
                    cut = df.index <= self.now_dt()
                    if cut.any():
                        d2 = df.loc[cut]
                        if len(d2) >= 20:
                            tfs[tf] = d2

                # PCR hook
                pcr_val = None
                try:
                    with API_SEMAPHORE:
                        oc = get_option_chain(sym, when=self.now_dt())
                    pcr_val = compute_pcr_from_option_chain(oc)
                except Exception as e:
                    logger.warning(f"PCR fetch/compute failed {sym}: {e}")

                if len(tfs) >= 3:
                    sig, score = self.calculate_enhanced_signals(sym, tfs)
                    if abs(score - 50) >= 15:
                        sector = next((s for s, lst in SECTOR_STOCKS.items() if sym in lst), "NA")
                        tf_pick = 5 if 5 in tfs else (min([k for k in tfs if isinstance(k, (int,float))]) if any(isinstance(k, (int,float)) for k in tfs) else "daily")
                        last_bar = tfs[tf_pick].iloc[-1]
                        buyv = float(last_bar.get("BuyVol", 0.0))
                        sellv = float(last_bar.get("SellVol", 0.0))
                        deltav = float(last_bar.get("DeltaVol", 0.0))
                        return {"symbol": sym, "signal": sig, "score": score, "sector": sector,
                                "tfcount": len(tfs), "time": self.now_dt(), "pcr": pcr_val,
                                "buyv": buyv, "sellv": sellv, "deltav": deltav}
                return None
            except Exception as e:
                logger.error(f"Proc error {sym}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = [ex.submit(process_symbol, s) for s in targets]
            for f in as_completed(futs):
                r = f.result()
                if r: signals.append(r)

        scan_time = timemodule.time() - start
        self.display_signals(signals, scan_time)
        next_scan_dt = (self.now_dt() + timedelta(seconds=self.scan_interval))
        print(f"{Colors.CYAN}Heartbeat: next scan at {next_scan_dt.strftime('%H:%M:%S')} IST{Colors.RESET}")

        if self.replay_mode:
            self.advance_replay_clock(self.scan_interval)
            sleep_s = int(self.scan_interval / max(REPLAY_SPEED, 1e-6))
            if REPLAY_SPEED > 0:
                timemodule.sleep(min(sleep_s, 2))
        else:
            timemodule.sleep(self.scan_interval)

    def display_signals(self, signals, scan_time):
        print(f"{Colors.BLUE}Scan time: {scan_time:.2f}s  Signals: {len(signals)}{Colors.RESET}")
        if not signals:
            print(f"{Colors.YELLOW}No significant signals this cycle.{Colors.RESET}")
            return
        bulls = [s for s in signals if "Buy" in s["signal"]]
        bears = [s for s in signals if "Sell" in s["signal"]]
        bulls.sort(key=lambda x: x["score"], reverse=True)
        bears.sort(key=lambda x: x["score"])
        print(f"{Colors.GREEN}{Colors.BOLD}BULLISH{Colors.RESET}")
        for s in bulls[:20]:
            pcr_str = f" PCR:{s['pcr']:.2f}" if s.get("pcr") is not None else ""
            dstr = f" Δ:{s['deltav']:.0f}" if s.get("deltav") is not None else ""
            print(f"{s['symbol']:<12} {s['sector']:<14} {s['signal']:<18} {s['score']:>6.1f} ({s['tfcount']} TFs){pcr_str}{dstr}")
        print(f"{Colors.RED}{Colors.BOLD}BEARISH{Colors.RESET}")
        for s in bears[:20]:
            pcr_str = f" PCR:{s['pcr']:.2f}" if s.get("pcr") is not None else ""
            dstr = f" Δ:{s['deltav']:.0f}" if s.get("deltav") is not None else ""
            print(f"{s['symbol']:<12} {s['sector']:<14} {s['signal']:<18} {s['score']:>6.1f} ({s['tfcount']} TFs){pcr_str}{dstr}")

    # --- Loop ---
    def run(self):
        print(f"{Colors.CYAN}{Colors.BOLD}Starting loop...{Colors.RESET}")
        if self.replay_mode and self.replay_clock is None:
            self.init_replay_clock()
        try:
            while True:
                if self.replay_mode and self.replay_clock >= self.replay_end_dt:
                    print(f"{Colors.YELLOW}Replay finished at {self.replay_clock.strftime('%H:%M:%S')}{Colors.RESET}")
                    break
                self.enhanced_scan_cycle()
        except KeyboardInterrupt:
            print(f"{Colors.YELLOW}Stopped by user{Colors.RESET}")

# =========================
# --- MAIN ---
# =========================
def main():
    print(f"{Colors.CYAN}{Colors.BOLD}Launching Enhanced Sector Scanner (Live/Replay)...{Colors.RESET}")
    sc = Enhanced3SectorScanner()
    sc.show_initialization_status()
    try:
        sc.enhanced_scan_cycle()
    except Exception as e:
        logger.error(f"Initial cycle error: {e}")
    sc.run()

if __name__ == "__main__":
    main()
