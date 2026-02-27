import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import requests
import json
import time as timemodule
from logzero import logger
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from truedata.history import TD_hist
import logging
import warnings
warnings.filterwarnings("ignore")

# =========================
# --- TRUE DATA CONFIG ---
# =========================
TDUSERNAME = os.getenv("TD_USERNAME", "tdwsp751")
TDPASSWORD = os.getenv("TD_PASSWORD", "raj@751")
tdhist = TD_hist(TDUSERNAME, TDPASSWORD, log_level=logging.WARNING)  # [attached_file:1]

# =========================
# --- COLOR CODES ---
# =========================
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"  # [attached_file:1]

# =========================
# --- INDICATOR WEIGHTS ---
# =========================
ENHANCED_INDICATOR_WEIGHTS = {
    "VolumeSurge": 2.0,
    "Momentum": 1.9,
    "ADX": 1.8,
    "VWAP": 1.7,
    "EMA": 1.7,
    "MACD": 1.5,
    "OBV": 1.5,
    "ATR": 1.4,
    "Bollinger": 1.3,
    "RSI": 1.2,
    "ROC": 1.1,
    "Stochastic": 1.0,
    "CCI": 1.0,
    "MA": 1.0,
    "WWL": 1.0,
}  # [attached_file:1]

TIMEFRAME_WEIGHTS = {15: 3.0, 5: 2.5, 30: 2.0, 60: 1.5, "daily": 1.0}  # [attached_file:1]

# =========================
# --- PLACEHOLDER MAPS (fill with real data) ---
# =========================
# Replace with the user’s actual dicts later.
NSE_INDEX_TO_SECTOR = {
    "NIFTY IT": "Technology",
    "NIFTY PHARMA": "Pharma",
    "NIFTY FMCG": "Consumer",
    "NIFTY BANK": "Banking",
    "NIFTY AUTO": "Auto",
    "NIFTY METAL": "Metal",
    "NIFTY ENERGY": "Energy",
    "NIFTY REALTY": "Realty",
    "NIFTY INFRA": "Infrastructure",
    "NIFTY PSU BANK": "PSU Bank",
    "NIFTY PSE": "PSE",
    "NIFTY COMMODITIES": "Commodities",
    "NIFTY MNC": "Finance",
    "NIFTY FINANCIAL SERVICES": "Finance",
    "NIFTY INFRASTRUCTURE": "Infrastructure",
    "BANKNIFTY": "Banking",
    "NIFTYAUTO": "Auto",
    "NIFTYIT": "Technology",
    "NIFTYPHARMA": "Pharma",
    "NIFTY CONSUMER DURABLES": "Consumer Durables",
    "NIFTY HEALTHCARE INDEX": "Healthcare",
    "NIFTY CAPITAL MARKETS": "Capital Market",
    "NIFTY PRIVATE BANK": "Private Bank",
    "NIFTY OIL & GAS": "Oil and Gas",
    "NIFTY INDIA DEFENCE": "Defence",
    "NIFTY CORE HOUSING": "Core Housing",
    "NIFTY SERVICES SECTOR": "Services Sector",
    "NIFTY FINANCIAL SERVICES 25/50": "Financial Services 2550",
    "NIFTY INDIA TOURISM": "Tourism",
}  # [attached_file:1]

# Representative sector->stocks mapping (extend with full lists as needed)
SECTOR_STOCKS = {
    "Technology": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "CYIENT", "KPITTECH", "TATAELXSI","SONACOMS","KAYNES","OFSS"],
    "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR","BHARATFORG", "EICHERMOT", "ASHOKLEY", "BOSCHLTD","TIINDIA","MOTHERSON"],
    "Banking": ["HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","PNB","BANKBARODA","CANBK","IDFCFIRSTB","INDUSINDBK","AUBANK","FEDERALBNK"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA", "TORNTPHARM","GLENMARK","ALKEM","LAURUSLABS","BIOCON","ZYDUSLIFE","MANKIND","SYNGENE","PPLPHARMA"],
    "Energy": ["RELIANCE","NTPC","BPCL","IOC","ONGC","GAIL","HINDPETRO","ADANIGREEN","ADANIENSOL","JSWENERGY","COALINDIA","TATAPOWER","SUZLON","PETRONET","OIL","POWERGRID","NHPC","ADANIPORTS","ABB","SIEMENS","CGPOWER","INOXWIND"],
    "Metal": ["TATASTEEL", "JSWSTEEL", "SAIL", "JINDALSTEL", "HINDALCO", "NMDC"],
    "Consumer": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR","AMBER","UNITDSPR","GODREJCP","MARICO","COLPAL","UPL","VBL"],
    "PSU Bank": ["SBIN", "PNB", "BANKBARODA", "CANBK", "UNIONBANK", "BANKINDIA"],
    "Finance": ["BAJFINANCE", "SHRIRAMFIN", "CHOLAFIN", "HDFCLIFE", "ICICIPRULI","ETERNAL"],
    "Realty": ["DLF","LODHA","PRESTIGE","GODREJPROP","OBEROIRLTY","PHOENIXLTD","NCC","NBCC"],
    "PSE": ["BEL","BHEL","NHPC","GAIL","IOC","NTPC","POWERGRID","HINDPETRO","OIL","RECLTD","ONGC","NMDC","BPCL","HAL","RVNL","PFC","COALINDIA","IRCTC","IRFC"],
    "Commodities": ["AMBUJACEM","APLAPOLLO","ULTRACEMCO","SHREECEM","JSWSTEEL","HINDALCO","NHPC","IOC","NTPC","HINDPETRO","ADANIGREEN","OIL","VEDL","PIIND","ONGC","NMDC","UPL","BPCL","JSWENERGY","GRASIM","RELIANCE","TORNTPOWER","TATAPOWER","COALINDIA","PIDILITIND","SRF","ADANIENSOL","JINDALSTEL","TATASTEEL","HINDALCO"],
    "Consumer Durables": ["TITAN","DIXON","HAVELLS","CROMPTON","POLYCAB","EXIDEIND","AMBER","KAYNES","VOLTAS","PGEL","BLUESTARCO"],
    "Healthcare": ["SUNPHARMA","DIVISLAB","CIPLA","TORNTPHARM","MAXHEALTH","APOLLOHOSP","DRREDDY","MANKIND","ZYDUSLIFE","LUPIN","FORTIS","ALKEM","AUROPHARMA","GLENMARK","BIOCON","LAURUSLABS","SYNGENE","GRANULES"],
    "Capital Market": ["HDFCAMC","BSE","360ONE","MCX","CDSL","NUVAMA","ANGELONE","KFINTECH","CAMS","IEX"],
    "Private Bank": ["HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","YESBANK","IDFCFIRSTB","INDUSINDBK","FEDERALBNK","BANDHANBNK","RBLBANK"],
    "Oil and Gas": ["RELIANCE","ONGC","IOC","BPCL","GAIL","HINDPETRO","OIL","PETRONET","IGL"],
    "Defence": ["HAL","BEL","SOLARINDS","MAZDOCK","BDL"],
    "Core Housing": ["ULTRACEMCO","ASIANPAINT","GRASIM","DLF","AMBUJACEM","LODHA","DIXON","POLYCAB","SHREECEM","HAVELLS","PRESTIGE","GODREJPROP","OBEROIRLTY","PHOENIXLTD","VOLTAS","DALBHARAT","KEI","BLUESTARCO","LICHSGFIN","PNBHOUSING","CROMPTON"],
    "Services Sector": ["HDFCBANK", "BHARTIARTL", "TCS", "ICICIBANK", "SBIN", "INFY", "BAJFINANCE", "HCLTECH", "KOTAKBANK", "AXISBANK", "BAJAJFINSV", "NTPC", "ZOMATO", "ADANIPORTS", "DMART", "POWERGRID", "WIPRO", "INDIGO", "JIOFINSERV", "SBILIFE", "HDFCLIFE", "LTIM", "TECHM", "TATAPOWER", "SHRIRAMFIN", "GAIL", "MAXHEALTH", "APOLLOHOSP", "NAUKRI", "INDUSINDBK"],
    "Financial Services 25/50": ["HDFCBANK", "ICICIBANK", "SBIN", "BAJFINANCE", "KOTAKBANK", "AXISBANK", "BAJAJFINSV", "JIOFIN", "SBILIFE", "HDFCLIFE", "PFC", "CHOLAFIN", "HDFCAMC", "SHRIRAMFIN", "MUTHOOTFIN", "RECLTD", "ICICIGI", "ICICIPRULI", "SBICARD", "LICHSGFIN"],
    "Tourism": ["INDIGO","INDHOTEL","IRCTC","JUBLFOOD"]
}  # [attached_file:1]

# =========================
# --- INDICATORS ---
# =========================
class EnhancedTechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        indicators = {}
        if df is None or len(df) < 20:
            return indicators
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

            # Stoch
            low14 = low.rolling(14).min()
            high14 = high.rolling(14).max()
            indicators["Stochastic"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)

            # MAs
            indicators["MA"] = close.rolling(20).mean()
            indicators["EMA"] = close.ewm(span=21).mean()

            # ADX
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

            # ROC
            indicators["ROC"] = close.pct_change(12) * 100

            # OBV change
            obv = np.sign(close.diff().fillna(0)) * vol.fillna(0)
            obv = obv.cumsum()
            indicators["OBV"] = obv.pct_change(10) * 100

            # CCI
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(20).mean()
            mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
            indicators["CCI"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

            # WWL (%R)
            hh = high.rolling(14).max()
            ll = low.rolling(14).min()
            indicators["WWL"] = (hh - close) / (hh - ll).replace(0, np.nan) * -100

            # VWAP (rolling proxy)
            if len(df) >= 20:
                tpv = (high + low + close) / 3
                vwap_num = (tpv * vol).rolling(20).sum()
                vwap_den = vol.rolling(20).sum().replace(0, np.nan)
                indicators["VWAP"] = vwap_num / vwap_den

            # VolumeSurge
            if len(df) >= 20:
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
            return indicators  # [attached_file:1]

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
        return 50  # [attached_file:1]

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

        # Sector snapshots for momentum table
        self.sector_snapshots = []   # {timestamp, sector, change}
        self.sector_snapshot_retention_min = 120
        print(f"{Colors.CYAN}{Colors.BOLD}Scanner initialized{Colors.RESET}")  # [attached_file:1]

    # --- Snapshot helpers ---
    def _trim_sector_snapshots(self):
        try:
            if not self.sector_snapshots: return
            cutoff = datetime.now() - timedelta(minutes=self.sector_snapshot_retention_min)
            self.sector_snapshots = [s for s in self.sector_snapshots if s["timestamp"] >= cutoff]
        except Exception as e:
            logger.warning(f"Trim snapshots error: {e}")  # [attached_file:1]

    def _get_sector_change_at(self, sector: str, ref_time: datetime, offset_min: int):
        try:
            target = ref_time - timedelta(minutes=offset_min)
            cands = [s for s in self.sector_snapshots if s["sector"] == sector and s["timestamp"] <= target]
            if not cands: return None
            return max(cands, key=lambda s: s["timestamp"])["change"]
        except Exception as e:
            logger.warning(f"Snapshot lookup error {sector}: {e}")
            return None  # [attached_file:1]

    # --- Display helpers ---
    def show_initialization_status(self):
        print(f"{Colors.CYAN}{Colors.BOLD}ENHANCED SECTOR SCANNER (API + Buyer Gates){Colors.RESET}")
        print(f"Timeframes: 5m, 15m, 30m, 60m, Daily")
        print(f"Top/Worst sectors used for weighting and stock picks")
        print(f"Sector API: http://localhost:3001/api/allIndices")
        print(f"Buyer windows: 09:25–11:30, 13:45–15:00 IST")
        self.show_sector_status()
        self.test_api_connection()
        print(f"{Colors.YELLOW}Running initial sector update...{Colors.RESET}")
        self.force_sector_update()
        print("-"*80)  # [attached_file:1]

    def test_api_connection(self):
        print(f"{Colors.BLUE}API CONNECTION TEST{Colors.RESET}")
        try:
            r = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            if r.status_code == 200:
                print(f"API OK, HTTP {r.status_code}")
            else:
                print(f"{Colors.RED}API FAIL HTTP {r.status_code}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}API ERROR: {e}{Colors.RESET}")  # [attached_file:1]

    def show_sector_status(self):
        print(f"{Colors.MAGENTA}Current Best: {', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"{Colors.MAGENTA}Current Worst: {', '.join(self.worst_sectors)}{Colors.RESET}")  # [attached_file:1]

    # --- API sectoral fetch ---
    def fetch_live_sectoral_performance(self):
        try:
            r = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            print(f"{Colors.BLUE}API Status: {r.status_code}{Colors.RESET}")
            if r.status_code != 200:
                # Show placeholder so UI isn’t blank
                nowt = datetime.now()
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

            curtime = datetime.now()
            sectoral = []
            for idx in indices_data:
                if not isinstance(idx, dict): continue
                name = next((str(idx[f]).strip().upper() for f in ("name","symbol","index","indexName") if f in idx and idx[f]), None)
                if not name or name not in NSE_INDEX_TO_SECTOR: continue
                change_percent = 0.0
                for f in ("changepercent","changePercent","pChange","percentChange","change","pchg"):
                    if f in idx and idx[f] is not None:
                        try:
                            change_percent = float(idx[f])
                            break
                        except: pass
                sectoral.append({
                    "index": name,
                    "sector": NSE_INDEX_TO_SECTOR[name],
                    "changepercent": change_percent,
                    "timestamp": curtime
                })

            if not sectoral:
                print("No mapped sectoral rows from API.")
                return False

            # snapshots
            for row in sectoral:
                self.sector_snapshots.append({"timestamp": row["timestamp"], "sector": row["sector"], "change": float(row["changepercent"])})
            self._trim_sector_snapshots()

            sectoral.sort(key=lambda x: x["changepercent"], reverse=True)
            n = len(sectoral)
            self.best_sectors = [sectoral[i]["sector"] for i in range(min(4,n))]
            self.worst_sectors = [sectoral[-i]["sector"] for i in range(1, min(4,n)+1)]
            self.last_sectoral_update = curtime
            self.sectoral_history.append({"timestamp": curtime, "best": self.best_sectors[:], "worst": self.worst_sectors[:], "fulldata": sectoral[:]})
            self.sectoral_history = self.sectoral_history[-20:]

            self.display_sector_update(sectoral)
            return True
        except Exception as e:
            logger.error(f"API sector fetch error: {e}")
            self.api_errors.append((datetime.now(), str(e)))
            return False  # [attached_file:1]

    def _print_top_sector_delta_table(self, sectoral):
        try:
            if not self.best_sectors: return
            now = datetime.now()
            curr_map = {r["sector"]: float(r["changepercent"]) for r in sectoral}
            print(f"{Colors.CYAN}{Colors.BOLD}TOP SECTORS MOMENTUM TABLE{Colors.RESET}")
            print(f"{Colors.BLUE}Sector            | Now %  | Δ vs 5m | Δ vs 15m{Colors.RESET}")
            print("-"*52)
            for sector in self.best_sectors:
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

                print(f"{color}{sector:16s}{Colors.RESET} | {now_str:6s} | {fmt(d5):8s} | {fmt(d15):9s}")
            print("-"*52)
            print(f"{Colors.MAGENTA}Tip: positive 5m and 15m deltas suggest sustained build-up.{Colors.RESET}")
        except Exception as e:
            logger.warning(f"Momentum table error: {e}")  # [attached_file:1]

    def display_sector_update(self, sectoral):
        ct = datetime.now().strftime('%H:%M:%S')
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
        self._print_top_sector_delta_table(sectoral)  # [attached_file:1]

    def force_sector_update(self):
        print(f"{Colors.YELLOW}FORCING LIVE SECTOR UPDATE...{Colors.RESET}")
        self.sector_update_attempts += 1
        ok = self.fetch_live_sectoral_performance()
        if ok:
            self.successful_updates += 1
        else:
            print(f"{Colors.RED}Sector update failed; using last known lists{Colors.RESET}")
        return ok  # [attached_file:1]

    # --- Market/time guards ---
    def is_market_open(self):
        now = datetime.now()
        ct = now.time()
        return now.weekday() < 5 and (self.market_start <= ct <= self.market_end)  # [attached_file:1]

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
                if isinstance(d.index, pd.DatetimeIndex):
                    d["Date"] = d.index
                else:
                    for cand in ("datetime","barstarttime","bartime","time"):
                        if cand in d.columns:
                            d.rename(columns={cand:"Date"}, inplace=True); break

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
            return d if len(d) >= 20 else None
        except Exception as e:
            logger.error(f"Normalize error {symbol}: {e}")
            return None  # [attached_file:1]

    def check_gapdown(self, df):
        try:
            if df is None or len(df) < 2: return False
            current_open = df["Open"].iloc[-1]; previous_close = df["Close"].iloc[-2]
            if previous_close == 0: return False
            gap = (current_open - previous_close) / previous_close * 100
            return gap <= -1.0
        except Exception:
            return False  # [attached_file:1]

    def fetch_live_data(self, symbol, timeframe):
        try:
            tfmap = {5:"5 min",15:"15 min",30:"30 min",60:"60 mins","daily":"EOD"}
            bar_size = tfmap.get(timeframe)
            if not bar_size: return None, False

            if timeframe in (5,15): duration = "10 D"
            elif timeframe == 30: duration = "20 D"
            elif timeframe == 60: duration = "60 D"
            elif timeframe == "daily": duration = "365 D"
            else: duration = "10 D"

            raw = tdhist.get_historic_data(symbol, duration=duration, bar_size=bar_size)
            if raw is None or len(raw) == 0: return None, False
            df = self.normalize_live_data(raw, symbol)
            if df is None or len(df) < 20: return None, False

            is_gap = False
            if timeframe in (5,15,30): is_gap = self.check_gapdown(df)

            if timeframe == "daily": return df.tail(250), is_gap
            if timeframe == 60: return df.tail(200), is_gap
            return df.tail(100), is_gap
        except Exception as e:
            logger.error(f"Live fetch error {symbol}@{timeframe}: {e}")
            return None, False  # [attached_file:1]

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
                        tf_score += norm * weight
                        tf_w += weight
                if tf_w <= 0: continue
                tf_final = tf_score / tf_w
                tf_scores[tf] = tf_final
                mult = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                total_weighted += tf_final * mult
                total_w += mult

            if total_w <= 0: return "Neutral", 0
            base_score = total_weighted / total_w

            # confirmation bonus
            if len(tf_scores) >= 4:
                bull = sum(1 for v in tf_scores.values() if v >= 55)
                bear = sum(1 for v in tf_scores.values() if v <= 45)
                if bull >= 3: base_score += 8
                elif bear >= 3: base_score -= 8

            # sector boost
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

            # Buyer gates
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

                strong = (adx5 is not None and adx5 >= 22) and (adx15 is not None and adx15 >= 22)
                above = (price is not None and ema is not None and vwap is not None and price > ema and price > vwap)
                below = (price is not None and ema is not None and vwap is not None and price < ema and price < vwap)

                now = datetime.now()
                d5 = d15 = None
                if self.sector_snapshots and sector:
                    ch5 = self._get_sector_change_at(sector, now, 5)
                    ch15 = self._get_sector_change_at(sector, now, 15)
                    rec = [s for s in self.sector_snapshots if s["sector"] == sector]
                    cur = rec[-1]["change"] if rec else None
                    if cur is not None:
                        d5 = None if ch5 is None else (cur - ch5)
                        d15 = None if ch15 is None else (cur - ch15)

                atr_ok = vol_ok = False
                if 5 in timeframes_data:
                    df5 = timeframes_data[5]
                    i5b = EnhancedTechnicalIndicators.calculate_all_indicators(df5)
                    atrs = i5b.get("ATR")
                    if atrs is not None and not atrs.empty:
                        tail = atrs.tail(50).dropna()
                        if len(tail) >= 10:
                            atr_ok = (tail.rank(pct=True).iloc[-1]) >= 0.6
                    vs = i5b.get("VolumeSurge")
                    if vs is not None and not vs.empty:
                        vol_ok = vs.iloc[-1] >= 60

                t = datetime.now().time()
                buyer_window = (time(9,25) <= t <= time(11,30)) or (time(13,45) <= t <= time(15,0))

                tflist = [tf_scores[k] for k in (5,15,30,60) if k in tf_scores]
                enough = len(tflist) >= 3
                tf_buy_ok = enough and all(v >= 58 for v in tflist[:3])
                tf_sell_ok = enough and all(v <= 42 for v in tflist[:3])

                intend_buy = base_score >= 60
                intend_sell = base_score <= 40

                if intend_buy:
                    pass_gates = (strong and above and atr_ok and vol_ok and buyer_window and (d5 is not None and d5 > 0) and (d15 is not None and d15 > 0) and tf_buy_ok)
                    if not pass_gates: return "Neutral", min(base_score, 59.9)
                if intend_sell:
                    pass_gates = (strong and below and atr_ok and vol_ok and buyer_window and (d5 is not None and d5 < 0) and (d15 is not None and d15 < 0) and tf_sell_ok)
                    if not pass_gates: return "Neutral", max(base_score, 40.1)
            except Exception as e:
                logger.warning(f"Buyer gates error: {e}")

            # Classification if not gated
            if base_score >= 82: return "Very Strong Buy", base_score
            if base_score >= 72: return "Strong Buy", base_score
            if base_score >= 60: return "Buy", base_score
            if base_score <= 18: return "Very Strong Sell", base_score
            if base_score <= 28: return "Strong Sell", base_score
            if base_score <= 40: return "Sell", base_score
            return "Neutral", base_score
        except Exception as e:
            logger.error(f"Signal calc error {symbol}: {e}")
            return "Neutral", 0  # [attached_file:1]

    # --- One scan cycle ---
    def enhanced_scan_cycle(self):
        nowdt = datetime.now()
        if not self.is_market_open():
            print(f"{Colors.YELLOW}Market closed at {nowdt.strftime('%H:%M:%S')} IST. Waiting {self.scan_interval//60}m...{Colors.RESET}")
            timemodule.sleep(self.scan_interval)
            return

        t = nowdt.time()
        if not ((time(9,25) <= t <= time(11,30)) or (time(13,45) <= t <= time(15,0))):
            print(f"{Colors.YELLOW}Outside buyer window; signals may be neutral-gated{Colors.RESET}")

        start = timemodule.time()
        print(f"{Colors.CYAN}Starting enhanced scan {nowdt.strftime('%H:%M:%S')}{Colors.RESET}")
        if not self.fetch_live_sectoral_performance():
            print("Sector update failed; proceeding with previous lists.")

        # Build target list (sample if SECTOR_STOCKS empty)
        targets = []
        for sec in (self.best_sectors + self.worst_sectors):
            if sec in SECTOR_STOCKS:
                targets.extend(SECTOR_STOCKS[sec][:6])
        if not targets:
            print(f"{Colors.YELLOW}No SECTOR_STOCKS configured; using sample symbols{Colors.RESET}")
            targets = ["RELIANCE","HDFCBANK","TCS","ICICIBANK","INFY","SBIN","AXISBANK","LTIM"]  # sample

        targets = sorted(set(targets))
        print(f"Scanning {len(targets)} symbols...")

        signals = []
        def process_symbol(sym):
            try:
                tfs = {}
                for tf in [5,15,30,60,"daily"]:
                    df, _ = self.fetch_live_data(sym, tf)
                    if df is not None:
                        tfs[tf] = df
                    timemodule.sleep(0.5)
                if len(tfs) >= 3:
                    sig, score = self.calculate_enhanced_signals(sym, tfs)
                    if abs(score - 50) >= 15:
                        sector = next((s for s, lst in SECTOR_STOCKS.items() if sym in lst), "NA")
                        return {"symbol": sym, "signal": sig, "score": score, "sector": sector, "tfcount": len(tfs), "time": datetime.now()}
                return None
            except Exception as e:
                logger.error(f"Proc error {sym}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=3) as ex:
            futs = [ex.submit(process_symbol, s) for s in targets]
            for f in as_completed(futs):
                r = f.result()
                if r: signals.append(r)

        scan_time = timemodule.time() - start
        self.display_signals(signals, scan_time)
        print(f"{Colors.CYAN}Heartbeat: next scan at {(datetime.now()+timedelta(seconds=self.scan_interval)).strftime('%H:%M:%S')} IST{Colors.RESET}")  # [attached_file:1]

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
        for s in bulls[:15]:
            print(f"{s['symbol']:<10} {s['sector']:<16} {s['signal']:<18} {s['score']:>6.1f} ({s['tfcount']} TFs)")
        print(f"{Colors.RED}{Colors.BOLD}BEARISH{Colors.RESET}")
        for s in bears[:15]:
            print(f"{s['symbol']:<10} {s['sector']:<16} {s['signal']:<18} {s['score']:>6.1f} ({s['tfcount']} TFs)")  # [attached_file:1]

    # --- Loop ---
    def run(self):
        print(f"{Colors.CYAN}{Colors.BOLD}Starting loop...{Colors.RESET}")
        try:
            while True:
                self.enhanced_scan_cycle()
                timemodule.sleep(self.scan_interval)
        except KeyboardInterrupt:
            print(f"{Colors.YELLOW}Stopped by user{Colors.RESET}")  # [attached_file:1]

# =========================
# --- MAIN ---
# =========================
def main():
    print(f"{Colors.CYAN}{Colors.BOLD}Launching Enhanced Sector Scanner...{Colors.RESET}")
    sc = Enhanced3SectorScanner()
    sc.show_initialization_status()
    # Run one quick visible cycle immediately
    try:
        sc.enhanced_scan_cycle()
    except Exception as e:
        logger.error(f"Initial cycle error: {e}")
    # Continue loop
    sc.run()

if __name__ == "__main__":
    main()
