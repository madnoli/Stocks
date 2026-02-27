import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import requests
import json
import time as timemodule
import pytz
from logzero import logger
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from truedata.history import TD_hist  # Correct class
import logging
import warnings
warnings.filterwarnings("ignore")

# =========================
# --- TRUE DATA CONFIG ---
# =========================
TDUSERNAME = os.getenv("TD_USERNAME", "tdwsp751")
TDPASSWORD = os.getenv("TD_PASSWORD", "raj@751")
tdhist = TD_hist(TDUSERNAME, TDPASSWORD, log_level=logging.WARNING)

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
    RESET = "\033[0m"

# =========================
# --- ENHANCED INDICATOR WEIGHTS ---
# =========================
ENHANCED_INDICATOR_WEIGHTS = {
# Tier 1: Primary Drivers for Momentum & Confirmation (Highest Weight)
    "VolumeSurge": 2.0,   # Crucial for confirming breakout conviction.
    "Momentum": 1.9,      # Directly measures the velocity of price change.
    "ADX": 1.8,           # Best for identifying strong trend strength (avoids choppy markets).
    "VWAP": 1.7,          # A key intraday level; crossing it is a powerful signal.
    "EMA": 1.7,           # More responsive to recent price changes than a simple MA.

    # Tier 2: Strong Secondary Confirmation
    "MACD": 1.5,          # Excellent for momentum shifts and trend direction.
    "OBV": 1.5,           # Confirms price trends with volume flow.
    "ATR": 1.4,           # Essential for gauging volatility and setting realistic profit targets/stops.
    
    # Tier 3: Supporting Indicators
    "Bollinger": 1.3,     # Good for identifying volatility breakouts (squeezes).
    "RSI": 1.2,           # Useful for entries, but less critical than pure momentum in a strong trend.
    "ROC": 1.1,           # Good momentum indicator, but somewhat redundant with "Momentum".

    # Tier 4: Lower Priority
    "Stochastic": 1.0,    # Can be too sensitive and give false signals in strong trends.
    "CCI": 1.0,           # Similar to RSI but can be noisier for this specific purpose.
    "MA": 1.0,            # EMA is generally superior for intraday responsiveness.
    "WWL": 1.0,           # Assuming this is a custom or less standard indicator.
}  # [attached_file:1]

# =========================
# --- TIMEFRAME WEIGHTS ---
# =========================
TIMEFRAME_WEIGHTS = {
    # Highest weight for the primary analysis timeframe.
    15: 3.0,
    
    # Second-highest for fine-tuning entries and exits.
    5: 2.5,
    
    # Medium weight for confirming the short-term trend direction.
    30: 2.0,
    60: 1.5,
    
    # Lowest weight for understanding the overall daily context.
    "daily": 1.0,
}  # [attached_file:1]

# =========================
# --- NSE INDEX TO SECTOR MAPPING ---
# =========================
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
# --- ENHANCED TECHNICAL INDICATORS ---
# =========================
class EnhancedTechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        indicators = {}
        if df is None or len(df) < 20:
            return indicators
        try:
            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            vol = df["Volume"]

            # 1. RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            indicators["RSI"] = 100 - (100 / (1 + rs))

            # 2. MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            indicators["MACD"] = macd_line - signal_line

            # 3. Stochastic
            low14 = low.rolling(window=14).min()
            high14 = high.rolling(window=14).max()
            indicators["Stochastic"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)

            # 4. MA 20
            indicators["MA"] = close.rolling(window=20).mean()

            # 5. EMA 21
            indicators["EMA"] = close.ewm(span=21).mean()

            # 6. ADX 14
            high_diff = high.diff()
            low_diff = low.diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
            minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff > 0), 0.0)
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr.replace(0, np.nan))
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr.replace(0, np.nan))
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
            indicators["ADX"] = dx.rolling(window=14).mean()

            # 7. Bollinger position
            ma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20
            indicators["Bollinger"] = (close - ma20) / (upper - lower).replace(0, np.nan) * 100

            # 8. ROC 12
            indicators["ROC"] = close.pct_change(periods=12) * 100

            # 9. OBV change
            obv = np.sign(close.diff().fillna(0)) * vol.fillna(0)
            obv = obv.cumsum()
            indicators["OBV"] = obv.pct_change(periods=10) * 100

            # 10. CCI
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(window=20).mean()
            mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
            indicators["CCI"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

            # 11. Williams %R
            hh = high.rolling(window=14).max()
            ll = low.rolling(window=14).min()
            indicators["WWL"] = (hh - close) / (hh - ll).replace(0, np.nan) * -100

            # 12. VWAP (rolling 20)
            if len(df) >= 20:
                tpv = (high + low + close) / 3
                vwap_num = (tpv * vol).rolling(window=20).sum()
                vwap_den = vol.rolling(window=20).sum().replace(0, np.nan)
                indicators["VWAP"] = vwap_num / vwap_den

            # 13. ATR (reuse)
            indicators["ATR"] = atr

            # 14. Volume Surge
            if len(df) >= 20:
                avg20 = vol.rolling(window=20).mean()
                current = vol
                vr = (current / avg20.replace(0, np.nan))
                indicators["VolumeSurge"] = np.clip((vr - 0.5) * 40, 0, 100)

            # 15. Momentum
            if len(df) >= 10:
                price_mom = close.pct_change(periods=10) * 100
                avg10 = vol.rolling(window=10).mean()
                vol_mom = (vol / avg10.replace(0, np.nan) - 1) * 100
                mom_score = price_mom * 0.7 + vol_mom * 0.3
                indicators["Momentum"] = np.clip(50 + mom_score * 1.5, -50, 50)

            return indicators
        except Exception as e:
            logger.error(f"Error calculating enhanced indicators: {e}")
            return indicators  # [attached_file:1]

# =========================
# --- NORMALIZATION HELPERS ---
# =========================
def normalize_indicator_value(indicator_name, value):
    try:
        if indicator_name == "RSI":
            return max(0, min(100, value))
        elif indicator_name == "MACD":
            return 50 + max(-25, min(25, value / 10))
        elif indicator_name == "Stochastic":
            return max(0, min(100, value))
        elif indicator_name in ("MA", "EMA", "VWAP"):
            return 50
        elif indicator_name == "ADX":
            return max(0, min(100, value))
        elif indicator_name == "Bollinger":
            return max(0, min(100, (value + 100) / 2))
        elif indicator_name == "ROC":
            return 50 + max(-25, min(25, value / 2))
        elif indicator_name == "OBV":
            return 50 + max(-25, min(25, value))
        elif indicator_name == "CCI":
            return max(0, min(100, (value + 200) / 4))
        elif indicator_name == "WWL":
            return max(0, min(100, (value + 100)))
        elif indicator_name == "ATR":
            return 50
        elif indicator_name == "VolumeSurge":
            return max(0, min(100, value))
        elif indicator_name == "Momentum":
            return max(0, min(100, value + 50))
        else:
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
        # Now 4 best/worst by default
        self.best_sectors = ["Pharma", "Healthcare", "Technology", "Financial Services 2550"]
        self.worst_sectors = ["Defence", "Energy", "PSU Bank", "Realty"]
        self.sectoral_history = []
        self.last_sectoral_update = None
        self.api_errors = []
        self.sector_update_attempts = 0
        self.successful_updates = 0
        self.gapdown_filtered_count = 0

        # Score delta tracking
        self.last_cycle_scores = {}     # symbol -> previous cycle score
        self.current_cycle_scores = {}  # symbol -> current cycle score

        # Market hours
        self.market_start = time(9, 15)
        self.market_end = time(15, 30)
        self.scan_interval = 300  # 5 min

        # NEW: Sector change snapshot memory for momentum table
        self.sector_snapshots = []  # list of dicts: {"timestamp": dt, "sector": str, "change": float}
        self.sector_snapshot_retention_min = 120  # keep ~2h

        logger.info("Enhanced 3-Sector Scanner with API Sectoral Data initialized")  # [attached_file:1]

    # --- Snapshot helpers ---
    def _trim_sector_snapshots(self):
        try:
            if not self.sector_snapshots:
                return
            cutoff = datetime.now() - timedelta(minutes=self.sector_snapshot_retention_min)
            self.sector_snapshots = [s for s in self.sector_snapshots if s["timestamp"] >= cutoff]
        except Exception as e:
            logger.warning(f"Trim snapshots error: {e}")  # [attached_file:1]

    def _get_sector_change_at(self, sector: str, ref_time: datetime, offset_min: int):
        try:
            target_time = ref_time - timedelta(minutes=offset_min)
            candidates = [s for s in self.sector_snapshots
                          if s["sector"] == sector and s["timestamp"] <= target_time]
            if not candidates:
                return None
            return max(candidates, key=lambda s: s["timestamp"])["change"]
        except Exception as e:
            logger.warning(f"Lookup snapshot error for {sector}: {e}")
            return None  # [attached_file:1]

    # --- Display helpers ---
    def show_initialization_status(self):
        print(f"{Colors.CYAN}{Colors.BOLD}ENHANCED 3-SECTOR SCANNER WITH API SECTORAL DATA{Colors.RESET}")
        print(f"{Colors.CYAN}{'-'*80}{Colors.RESET}")
        print(f"Timeframes: {Colors.YELLOW}5min, 15min, 30min, 60min, Daily{Colors.RESET}")
        print(f"Strategy: {Colors.GREEN}Top 4 Best{Colors.RESET} / {Colors.RED}Top 4 Worst{Colors.RESET} sectors")
        print(f"Filter: {Colors.MAGENTA}Gap-down exclusion{Colors.RESET}")
        print(f"Sectoral Data: {Colors.GREEN}API http://localhost:3001/api/allIndices{Colors.RESET}")
        print(f"{Colors.YELLOW}NEW INDICATORS{Colors.RESET}")
        print("ATR (1.4), Volume Surge (2.0), Momentum (1.9)")
        print(f"{Colors.BLUE}ENHANCED WEIGHTS{Colors.RESET}")
        for indicator, weight in ENHANCED_INDICATOR_WEIGHTS.items():
            color = Colors.GREEN if weight >= 1.5 else Colors.YELLOW if weight >= 1.2 else Colors.WHITE
            print(f" - {indicator}: {color}{weight}{Colors.RESET}")
        self.show_sector_status()
        self.test_api_connection()
        print(f"{Colors.YELLOW}Running initial sector update...{Colors.RESET}")
        self.force_sector_update()
        print(f"{Colors.CYAN}{'-'*80}{Colors.RESET}")  # [attached_file:1]

    def test_api_connection(self):
        print(f"{Colors.BLUE}API CONNECTION TEST{Colors.RESET}")
        try:
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            if response.status_code == 200:
                print(f"API Connection {Colors.GREEN}SUCCESS{Colors.RESET}")
                data = response.json()
                if isinstance(data, list):
                    print(f"Items Count: {len(data)}")
                elif isinstance(data, dict):
                    print(f"Dict Keys: {list(data.keys())}")
            else:
                print(f"API Connection {Colors.RED}FAILED{Colors.RESET} - Status {response.status_code}")
        except Exception as e:
            print(f"API Connection {Colors.RED}ERROR{Colors.RESET} - {str(e)}")  # [attached_file:1]

    def show_sector_status(self):
        print(f"{Colors.MAGENTA}CURRENT 4-SECTOR STATUS{Colors.RESET}")
        print(f"Top 4 Best Sectors: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"Top 4 Worst Sectors: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
        print(f"Last Update: {self.last_sectoral_update or 'Never'}")
        print(f"Gap-down Filtered: {self.gapdown_filtered_count}")  # [attached_file:1]

    # --- API sectoral fetch (now 4/4) ---
    def fetch_live_sectoral_performance_3sector_debug(self):
        try:
            logger.info("Fetching live sector performance from API...")
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            print(f"{Colors.BLUE}API RESPONSE DEBUG{Colors.RESET}")
            print(f"Status Code: {response.status_code}")
            if response.status_code != 200:
                print(f"API request failed with status code {response.status_code}")
                return False

            indices_data = response.json()
            if isinstance(indices_data, str):
                indices_data = json.loads(indices_data)

            # Normalize container
            if isinstance(indices_data, dict):
                if "data" in indices_data:
                    indices_data = indices_data["data"]
                elif "indices" in indices_data:
                    indices_data = indices_data["indices"]
                elif "results" in indices_data:
                    indices_data = indices_data["results"]

            if not isinstance(indices_data, list):
                logger.error("Processed API data is not a list.")
                return False

            sectoral_performance = []
            current_time = datetime.now()
            for index in indices_data:
                if not isinstance(index, dict):
                    continue
                index_name = next((str(index[field]).strip().upper()
                                   for field in ("name", "symbol", "index", "indexName")
                                   if field in index and index[field]), None)
                if index_name and index_name in NSE_INDEX_TO_SECTOR:
                    change_percent = 0.0
                    for field in ("changepercent", "changePercent", "pChange", "percentChange", "change", "pchg"):
                        if field in index and index[field] is not None:
                            try:
                                change_percent = float(index[field])
                                break
                            except (ValueError, TypeError):
                                continue
                    sectoral_performance.append({
                        "index": index_name,
                        "sector": NSE_INDEX_TO_SECTOR[index_name],
                        "changepercent": change_percent,
                        "timestamp": current_time,
                    })
                    logger.info(f"{index_name} {NSE_INDEX_TO_SECTOR[index_name]} {change_percent:.2f}")

            if not sectoral_performance:
                print("No sectoral data matched from API response.")
                return False

            # NEW: append snapshots for each sector reading
            for row in sectoral_performance:
                self.sector_snapshots.append({
                    "timestamp": row["timestamp"],
                    "sector": row["sector"],
                    "change": float(row["changepercent"])
                })
            self._trim_sector_snapshots()

            sectoral_performance.sort(key=lambda x: x["changepercent"], reverse=True)
            old_best = self.best_sectors[:]
            old_worst = self.worst_sectors[:]

            n = len(sectoral_performance)
            best_count = min(4, n)
            worst_count = min(4, n)
            self.best_sectors = [sectoral_performance[i]["sector"] for i in range(best_count)]
            self.worst_sectors = [sectoral_performance[-i]["sector"] for i in range(1, worst_count + 1)]

            self.last_sectoral_update = current_time
            self.sectoral_history.append({
                "timestamp": current_time,
                "best": self.best_sectors[:],
                "worst": self.worst_sectors[:],
                "fulldata": sectoral_performance[:],
            })
            if len(self.sectoral_history) > 20:
                self.sectoral_history = self.sectoral_history[-20:]

            self.display_3sector_update(sectoral_performance, old_best, old_worst)
            return True
        except Exception as e:
            logger.error(f"Error fetching API sectoral data: {e}")
            self.api_errors.append((datetime.now(), str(e)))
            return False  # [attached_file:1]

    def _print_top_sector_delta_table(self, sectoral_performance):
        try:
            if not self.best_sectors:
                return
            now = datetime.now()
            curr_map = {r["sector"]: float(r["changepercent"]) for r in sectoral_performance}

            print(f"{Colors.CYAN}{Colors.BOLD}TOP SECTORS MOMENTUM TABLE{Colors.RESET}")
            print(f"{Colors.BLUE}Sector            | Now %  | Δ vs 5m | Δ vs 15m{Colors.RESET}")
            print("-" * 52)
            for sector in self.best_sectors:
                cur = curr_map.get(sector)
                chg_5 = self._get_sector_change_at(sector, now, 5)
                chg_15 = self._get_sector_change_at(sector, now, 15)
                d5 = None if (cur is None or chg_5 is None) else (cur - chg_5)
                d15 = None if (cur is None or chg_15 is None) else (cur - chg_15)

                def fnum(x):
                    if x is None:
                        return "na"
                    arrow = "↑" if x > 0 else ("↓" if x < 0 else "→")
                    return f"{x:+.2f}% {arrow}"

                now_str = "na" if cur is None else f"{cur:+.2f}%"
                color = Colors.WHITE
                if d5 is not None:
                    if d5 >= 0.15:
                        color = Colors.GREEN + Colors.BOLD
                    elif d5 <= -0.15:
                        color = Colors.RED + Colors.BOLD
                    else:
                        color = Colors.YELLOW
                print(f"{color}{sector:16s}{Colors.RESET} | {now_str:6s} | {fnum(d5):8s} | {fnum(d15):9s}")
            print("-" * 52)
            print(f"{Colors.MAGENTA}Tip: Positive 5m and 15m deltas suggest sustained build-up; divergence can hint at early reversals.{Colors.RESET}")
        except Exception as e:
            logger.warning(f"Top sector delta table error: {e}")  # [attached_file:1]

    def display_3sector_update(self, sectoral_performance, old_best, old_worst):
        current_time = datetime.now()
        print(f"{Colors.MAGENTA}{Colors.BOLD}{'-'*100}")
        print(f"4-SECTOR PERFORMANCE UPDATE - {current_time.strftime('%H:%M:%S')} IST")
        print(f"{'-'*100}{Colors.RESET}")
        print(f"Top 4 Best: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"Top 4 Worst: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")

        topn = min(6, len(sectoral_performance))
        print("Top 6 Performing Sectors")
        for i, sectordata in enumerate(sectoral_performance[:topn]):
            color = Colors.YELLOW
            if sectordata["sector"] in self.best_sectors:
                rank = self.best_sectors.index(sectordata["sector"]) + 1
                color = Colors.GREEN + (Colors.BOLD if rank == 1 else "")
            print(f"{i+1}. {color}{sectordata['sector']:<20}{Colors.RESET} {sectordata['changepercent']:>6.2f}  {sectordata['index']}")

        print("Bottom 6 Performing Sectors")
        bottom_slice = sectoral_performance[-topn:]
        for i, sectordata in enumerate(bottom_slice):
            color = Colors.YELLOW
            if sectordata["sector"] in self.worst_sectors:
                rank = self.worst_sectors.index(sectordata["sector"]) + 1
                color = Colors.RED + (Colors.BOLD if rank == 1 else "")
            pos = len(sectoral_performance) - topn + i + 1
            print(f"{pos}. {color}{sectordata['sector']:<20}{Colors.RESET} {sectordata['changepercent']:>6.2f}  {sectordata['index']}")
        print(f"{Colors.MAGENTA}{'-'*100}{Colors.RESET}")

        # NEW: Momentum table for current top sectors
        self._print_top_sector_delta_table(sectoral_performance)  # [attached_file:1]

    def force_sector_update(self):
        print(f"{Colors.YELLOW}FORCING REAL SECTOR UPDATE WITH API...{Colors.RESET}")
        self.sector_update_attempts += 1
        success = self.fetch_live_sectoral_performance_3sector_debug()
        if success:
            self.successful_updates += 1
            print("API sectoral update successful!")
        else:
            print("API sectoral update failed - using defaults")
            print(f"Top 4 Best: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
            print(f"Top 4 Worst: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
        return success  # [attached_file:1]

    # --- Market state ---
    def is_market_open(self):
        now = datetime.now()
        ct = now.time()
        if now.weekday() >= 5:
            return False
        return self.market_start <= ct <= self.market_end  # [attached_file:1]

    # --- Data normalization/fetch ---
    def normalize_live_data(self, df, symbol):
        try:
            if df is None or len(df) == 0:
                return None
            dfc = df.copy()

            # Lowercase headers for mapping
            dfc.rename(columns={c: c.lower() for c in dfc.columns}, inplace=True)

            # Map common fields
            col_map = {}
            for src, tgt in (
                ("time", "Date"), ("timestamp", "Date"), ("date", "Date"),
                ("open", "Open"), ("high", "High"), ("low", "Low"),
                ("close", "Close"), ("vol", "Volume"), ("volume", "Volume"),
            ):
                if src in dfc.columns:
                    col_map[src] = tgt
            dfc.rename(columns=col_map, inplace=True)

            # Recover Date from index if needed
            if "Date" not in dfc.columns:
                if isinstance(dfc.index, pd.DatetimeIndex):
                    dfc["Date"] = dfc.index
                else:
                    for cand in ("datetime", "barstarttime", "bartime", "time"):
                        if cand in dfc.columns:
                            dfc.rename(columns={cand: "Date"}, inplace=True)
                            break

            # Require OHLC
            required = ["Open", "High", "Low", "Close"]
            if not all(col in dfc.columns for col in required):
                return None

            if "Volume" not in dfc.columns:
                dfc["Volume"] = 0

            # Parse Date
            if "Date" in dfc.columns:
                dfc["Date"] = pd.to_datetime(dfc["Date"], errors="coerce", utc=False)
            else:
                idx = pd.to_datetime(dfc.index, errors="coerce", utc=False)
                dfc["Date"] = idx

            # Drop bad rows
            dfc = dfc.dropna(subset=["Date", "Open", "High", "Low", "Close"])

            # Ensure numeric
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                dfc[col] = pd.to_numeric(dfc[col], errors="coerce")
            dfc = dfc.dropna(subset=["Open", "High", "Low", "Close"])

            # Remove timezone
            if pd.api.types.is_datetime64tz_dtype(dfc["Date"]):
                dfc["Date"] = dfc["Date"].dt.tz_convert(None)

            # Index and sort
            dfc.set_index("Date", inplace=True, drop=True)
            if not isinstance(dfc.index, pd.DatetimeIndex):
                new_idx = pd.to_datetime(dfc.index, errors="coerce", utc=False)
                dfc = dfc[~new_idx.isna()]
                dfc.index = pd.to_datetime(dfc.index, errors="coerce", utc=False)
            dfc = dfc.sort_index()

            return dfc if len(dfc) >= 20 else None
        except Exception as e:
            logger.error(f"Normalize error {symbol}: {e}")
            return None  # [attached_file:1]

    def check_gapdown(self, df):
        try:
            if df is None or len(df) < 2:
                return False
            current_open = df["Open"].iloc[-1]
            previous_close = df["Close"].iloc[-2]
            if pd.isna(current_open) or pd.isna(previous_close) or previous_close == 0:
                return False
            gap_percentage = (current_open - previous_close) / previous_close * 100
            return gap_percentage <= -1.0
        except Exception as e:
            logger.error(f"Error checking gap down: {e}")
            return False  # [attached_file:1]

    def fetch_live_data(self, symbol, timeframe):
        try:
            # Map internal timeframe to TrueData bar_size strings
            tfmap = {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 mins", "daily": "EOD"}
            bar_size = tfmap.get(timeframe)
            if not bar_size:
                return None, False

            # Duration per timeframe
            if timeframe in (5, 15):
                duration = "10 D"
            elif timeframe == 30:
                duration = "20 D"
            elif timeframe == 60:
                duration = "60 D"
            elif timeframe == "daily":
                duration = "365 D"
            else:
                duration = "10 D"

            rawdf = tdhist.get_historic_data(symbol, duration=duration, bar_size=bar_size)
            if rawdf is None or len(rawdf) == 0:
                return None, False

            normalized_df = self.normalize_live_data(rawdf, symbol)
            if normalized_df is None or len(normalized_df) < 20:
                return None, False

            is_gapdown = False
            if timeframe in (5, 15, 30):
                is_gapdown = self.check_gapdown(normalized_df)

            if timeframe == "daily":
                return normalized_df.tail(250), is_gapdown
            elif timeframe == 60:
                return normalized_df.tail(200), is_gapdown
            else:
                return normalized_df.tail(100), is_gapdown
        except Exception as e:
            logger.error(f"Live data fetch error {symbol}@{timeframe}: {e}")
            return None, False  # [attached_file:1]

    # --- Signal calculation ---
    def calculate_enhanced_signals(self, symbol, timeframes_data):
        try:
            if not timeframes_data:
                return "Neutral", 0
            sector = next((s for s, stocks in SECTOR_STOCKS.items() if symbol in stocks), None)
            if not sector:
                return "Neutral", 0

            total_weighted_score = 0.0
            total_weight = 0.0
            timeframe_scores = {}

            for tf, df in timeframes_data.items():
                if df is None or len(df) < 20:
                    continue
                indicators = EnhancedTechnicalIndicators.calculate_all_indicators(df)
                if not indicators:
                    continue

                tf_score = 0.0
                tf_weight = 0.0
                current_price = df["Close"].iloc[-1]

                for name, weight in ENHANCED_INDICATOR_WEIGHTS.items():
                    if name in indicators and indicators[name] is not None and not indicators[name].empty:
                        latest_val = indicators[name].iloc[-1]
                        if pd.isna(latest_val):
                            continue
                        if name in ("MA", "EMA", "VWAP"):
                            base = latest_val
                            if pd.isna(base) or base == 0:
                                norm_score = 50
                            else:
                                price_vs = (current_price - base) / base * 100
                                if price_vs >= 2:
                                    norm_score = 75
                                elif price_vs >= 0:
                                    norm_score = 60
                                elif price_vs >= -2:
                                    norm_score = 50
                                elif price_vs >= -5:
                                    norm_score = 40
                                else:
                                    norm_score = 25
                        else:
                            norm_score = normalize_indicator_value(name, latest_val)
                        tf_score += norm_score * weight
                        tf_weight += weight

                if tf_weight <= 0:
                    continue

                tf_final_score = tf_score / tf_weight
                tf_multiplier = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                timeframe_scores[tf] = tf_final_score
                total_weighted_score += tf_final_score * tf_multiplier
                total_weight += tf_multiplier

            if total_weight <= 0:
                return "Neutral", 0

            base_score = total_weighted_score / total_weight

            # Multi-timeframe confirmation bonus
            num_timeframes = len(timeframe_scores)
            if num_timeframes >= 4:
                bullish_count = sum(1 for v in timeframe_scores.values() if v >= 55)
                bearish_count = sum(1 for v in timeframe_scores.values() if v <= 45)
                if bullish_count >= 3:
                    base_score += 8
                elif bearish_count >= 3:
                    base_score -= 8

            # Sector boost extended to rank 1-4
            sector_boost = 0
            has_longer_tf = ("daily" in timeframes_data) or (60 in timeframes_data)

            if sector in self.best_sectors:
                rank = self.best_sectors.index(sector) + 1
                if has_longer_tf:
                    boost_map = {1: 25, 2: 20, 3: 15, 4: 10}
                else:
                    boost_map = {1: 20, 2: 15, 3: 10, 4: 5}
                sector_boost = boost_map.get(rank, 0)
            elif sector in self.worst_sectors:
                rank = self.worst_sectors.index(sector) + 1
                if has_longer_tf:
                    boost_map = {1: -25, 2: -20, 3: -15, 4: -10}
                else:
                    boost_map = {1: -20, 2: -15, 3: -10, 4: -5}
                sector_boost = boost_map.get(rank, 0)

            base_score += sector_boost

            # Classification
            if base_score >= 82:
                return "Very Strong Buy", base_score
            elif base_score >= 72:
                return "Strong Buy", base_score
            elif base_score >= 60:
                return "Buy", base_score
            elif base_score <= 18:
                return "Very Strong Sell", base_score
            elif base_score <= 28:
                return "Strong Sell", base_score
            elif base_score <= 40:
                return "Sell", base_score
            else:
                return "Neutral", base_score
        except Exception as e:
            logger.error(f"Enhanced signal calculation error for {symbol}: {e}")
            return "Neutral", 0  # [attached_file:1]

    # --- Single-cycle processing ---
    def enhanced_scan_cycle(self):
        if not self.is_market_open():
            logger.info("Market closed. Next scan in 5 minutes...")
            return

        start_time = timemodule.time()
        current_time = datetime.now()
        print(f"{Colors.CYAN}Starting ENHANCED 4-sector scan at {current_time.strftime('%H:%M:%S')}{Colors.RESET}")
        print("Analyzing: 5min 15min 30min 60min Daily")
        print(f"Strategy: {Colors.GREEN}Top 4 Best{Colors.RESET} / {Colors.RED}Top 4 Worst{Colors.RESET} sectors")
        print(f"New Indicators: {Colors.MAGENTA}ATR1.4, VolumeSurge2.0, Momentum1.9{Colors.RESET}")
        print(f"Sectoral Source: {Colors.GREEN}API localhost:3001/api/allIndices{Colors.RESET}")

        # Update sectors
        if not self.fetch_live_sectoral_performance_3sector_debug():
            print("API sectoral update failed, continuing with previous sectors")

        # Build target stocks from top 4 best and top 4 worst
        target_stocks_set = set()

        # Best sectors allocations: 12/10/8/6
        for i, sector in enumerate(self.best_sectors):
            if sector in SECTOR_STOCKS:
                if i == 0:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:12])
                elif i == 1:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:10])
                elif i == 2:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:8])
                elif i == 3:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:6])

        # Worst sectors allocations: 12/10/8/6
        for i, sector in enumerate(self.worst_sectors):
            if sector in SECTOR_STOCKS:
                if i == 0:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:12])
                elif i == 1:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:10])
                elif i == 2:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:8])
                elif i == 3:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:6])

        target_stocks = list(target_stocks_set)
        if not target_stocks:
            print("No target stocks found.")
            return

        print(f"Enhanced scanning {len(target_stocks)} stocks from up to 8 sectors")
        live_signals = []
        gapdown_filtered = 0

        def process_stock(symbol):
            try:
                timeframes_data = {}
                has_gapdown = False
                timeframes_to_fetch = [5, 15, 30, 60, "daily"]
                for tf in timeframes_to_fetch:
                    df, is_gapdown = self.fetch_live_data(symbol, tf)
                    if df is not None:
                        timeframes_data[tf] = df
                    timemodule.sleep(1.0)

                if len(timeframes_data) >= 3:
                    signal, score = self.calculate_enhanced_signals(symbol, timeframes_data)
                    # Only significant signals
                    if abs(score - 50) >= 15:
                        sector = next((s for s, st in SECTOR_STOCKS.items() if symbol in st), "NA")
                        result = {
                            "symbol": symbol,
                            "signal": signal,
                            "score": score,
                            "sector": sector,
                            "timeframes": len(timeframes_data),
                            "timestamp": datetime.now(),
                            "tfdetails": list(timeframes_data.keys()),
                        }
                        # Track current cycle score for delta
                        self.current_cycle_scores[symbol] = score
                        return result, False
                return None, False
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                return None, False

        try:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(process_stock, symbol) for symbol in target_stocks]
                for future in as_completed(futures):
                    result, is_gap = future.result()
                    if is_gap:
                        gapdown_filtered += 1
                    elif result:
                        live_signals.append(result)
            self.gapdown_filtered_count += gapdown_filtered
            scan_time = timemodule.time() - start_time
            logger.info(f"Enhanced scan completed in {scan_time:.2f}s - {len(live_signals)} signals, {gapdown_filtered} gap-down filtered")
            self.display_enhanced_signals(live_signals, scan_time, gapdown_filtered)
        except Exception as e:
            logger.error(f"Error in enhanced scan: {e}")  # [attached_file:1]

    def display_enhanced_signals(self, signals, scan_time, gapdown_filtered):
        os.system("clear" if os.name == "posix" else "cls")
        current_time = datetime.now()
        print(f"{Colors.CYAN}{Colors.BOLD}{'-'*150}{Colors.RESET}")
        print(f"ENHANCED 4-SECTOR SCANNER WITH API SECTORAL DATA - {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
        print(f"{'-'*150}")
        print(f"Analysis: {Colors.YELLOW}5m{Colors.RESET} {Colors.YELLOW}15m{Colors.RESET} {Colors.YELLOW}30m{Colors.RESET} {Colors.CYAN}60m{Colors.RESET} {Colors.MAGENTA}Daily{Colors.RESET}")
        best_str = ", ".join(self.best_sectors)
        worst_str = ", ".join(self.worst_sectors)
        if self.last_sectoral_update:
            print(f"{Colors.MAGENTA}API Sectoral Update{Colors.RESET} {Colors.YELLOW}{self.last_sectoral_update.strftime('%H:%M:%S')}{Colors.RESET}")
            print(f"Top 4 Best: {Colors.GREEN}{Colors.BOLD}{best_str}{Colors.RESET}")
            print(f"Top 4 Worst: {Colors.RED}{Colors.BOLD}{worst_str}{Colors.RESET}")
        print(f"{Colors.BLUE}Updates{Colors.RESET}: {self.successful_updates}/{self.sector_update_attempts}  Scan Time: {scan_time:.2f}s  Gap-down Filtered: {Colors.MAGENTA}{gapdown_filtered}{Colors.RESET}")

        if not signals:
            print(f"{Colors.YELLOW}No significant enhanced signals found in this cycle.{Colors.RESET}")
        else:
            # Separate bullish/bearish
            bullish = [s for s in signals if "Buy" in s["signal"]]
            bearish = [s for s in signals if "Sell" in s["signal"]]
            bullish.sort(key=lambda x: x["score"], reverse=True)
            bearish.sort(key=lambda x: x["score"])

            def strength_str(score, bullish_side=True):
                deviation = abs(score - 50)
                if deviation >= 40:
                    return f"{(Colors.GREEN if bullish_side else Colors.RED)}{Colors.BOLD}Exceptional{Colors.RESET}"
                elif deviation >= 30:
                    return f"{(Colors.GREEN if bullish_side else Colors.RED)}{Colors.BOLD}Very Strong{Colors.RESET}"
                elif deviation >= 20:
                    return f"{(Colors.GREEN if bullish_side else Colors.RED)}Strong{Colors.RESET}"
                else:
                    return f"{Colors.YELLOW}Moderate{Colors.RESET}"

            # Header with Score Δ
            print(f"{Colors.GREEN}{Colors.BOLD}TOP 10 BULLISH SIGNALS (ENHANCED + API SECTORAL){Colors.RESET}")
            print(f"{'Stock':<10} {'Sector':<18} {'Signal':<20} {'Score':>8} {'Score Δ':>8} {'TFs':>4} {'TF Coverage':<20} {'Strength':<15}")
            print(f"{Colors.GREEN}{'-'*150}{Colors.RESET}")
            for s in bullish[:20]:
                sector_name = s["sector"]
                sector_color = Colors.YELLOW
                if sector_name in self.best_sectors:
                    rank = self.best_sectors.index(sector_name) + 1
                    stars = "*" * rank
                    sector_color = Colors.GREEN
                    sector_display = f"{stars}{sector_name}"
                else:
                    sector_display = sector_name

                prev = self.last_cycle_scores.get(s["symbol"])
                delta_display = "n/a" if prev is None else f"{s['score'] - prev:+.1f}"

                signal_color = Colors.GREEN + (Colors.BOLD if "Very" in s["signal"] else "")
                tf_details = s.get("tfdetails", [])
                tf_display = ",".join([str(tf) if isinstance(tf, int) else "D" for tf in tf_details])[:20]
                print(
                    f"{Colors.WHITE}{s['symbol']:<10}{Colors.RESET} "
                    f"{sector_color}{sector_display:<18}{Colors.RESET} "
                    f"{signal_color}{s['signal']:<20}{Colors.RESET} "
                    f"{Colors.WHITE}{s['score']:>8.1f}{Colors.RESET} "
                    f"{Colors.CYAN}{delta_display:>8}{Colors.RESET} "
                    f"{Colors.CYAN}{s['timeframes']:>4}{Colors.RESET} "
                    f"{Colors.MAGENTA}{tf_display:<20}{Colors.RESET} "
                    f"{strength_str(s['score'], bullish_side=True)}"
                )

            print(f"{Colors.RED}{Colors.BOLD}TOP 10 BEARISH SIGNALS (ENHANCED + API SECTORAL){Colors.RESET}")
            print(f"{'Stock':<10} {'Sector':<18} {'Signal':<20} {'Score':>8} {'Score Δ':>8} {'TFs':>4} {'TF Coverage':<20} {'Strength':<15}")
            print(f"{Colors.RED}{'-'*150}{Colors.RESET}")
            for s in bearish[:20]:
                sector_name = s["sector"]
                sector_color = Colors.YELLOW
                if sector_name in self.worst_sectors:
                    rank = self.worst_sectors.index(sector_name) + 1
                    stars = "*" * rank
                    sector_color = Colors.RED
                    sector_display = f"{stars}{sector_name}"
                else:
                    sector_display = sector_name

                prev = self.last_cycle_scores.get(s["symbol"])
                delta_display = "n/a" if prev is None else f"{s['score'] - prev:+.1f}"

                signal_color = Colors.RED + (Colors.BOLD if "Very" in s["signal"] else "")
                tf_details = s.get("tfdetails", [])
                tf_display = ",".join([str(tf) if isinstance(tf, int) else "D" for tf in tf_details])[:20]
                print(
                    f"{Colors.WHITE}{s['symbol']:<10}{Colors.RESET} "
                    f"{sector_color}{sector_display:<18}{Colors.RESET} "
                    f"{signal_color}{s['signal']:<20}{Colors.RESET} "
                    f"{Colors.WHITE}{s['score']:>8.1f}{Colors.RESET} "
                    f"{Colors.CYAN}{delta_display:>8}{Colors.RESET} "
                    f"{Colors.CYAN}{s['timeframes']:>4}{Colors.RESET} "
                    f"{Colors.MAGENTA}{tf_display:<20}{Colors.RESET} "
                    f"{strength_str(s['score'], bullish_side=False)}"
                )

        next_scan_time = (current_time + timedelta(minutes=5)).strftime("%H:%M:%S")
        print(f"{Colors.CYAN}{Colors.BOLD}Next enhanced scan at {next_scan_time}{Colors.RESET}")
        print(f"{Colors.BLUE}Enhanced 4-sector strategy with Volume Surge, Momentum, LIVE API sectoral data{Colors.RESET}")
        if gapdown_filtered > 0:
            print(f"{Colors.MAGENTA}Gap-down filter excluded {gapdown_filtered} stocks for risk management{Colors.RESET}")

        # Rotate score maps for delta tracking
        self.last_cycle_scores = self.current_cycle_scores
        self.current_cycle_scores = {}  # [attached_file:1]

    # --- Run loop ---
    def run_enhanced_scanner(self):
        self.is_running = True
        logger.info("Starting Enhanced 4-Sector Scanner with API Sectoral Data...")
        self.show_initialization_status()
        try:
            while self.is_running:
                self.enhanced_scan_cycle()
                if self.is_running:
                    logger.info("Waiting 5 minutes for next enhanced cycle...")
                    timemodule.sleep(self.scan_interval)
        except KeyboardInterrupt:
            logger.info("Enhanced scanner stopped by user")
        finally:
            self.stop()  # [attached_file:1]

    def stop(self):
        self.is_running = False
        print(f"{Colors.YELLOW}Enhanced 4-sector scanner stopped{Colors.RESET}")  # [attached_file:1]

# =========================
# --- MAIN EXECUTION ---
# =========================
def main():
    print(f"{Colors.CYAN}{Colors.BOLD}ENHANCED 3-SECTOR SCANNER WITH API SECTORAL DATA{Colors.RESET}")
    print(f"{Colors.YELLOW}Timeframes: 5min, 15min, 30min, 60min, Daily EOD{Colors.RESET}")
    print(f"{Colors.CYAN}Features: 4 Best / 4 Worst sectors, Enhanced indicators, API sectoral data{Colors.RESET}")
    print(f"{Colors.MAGENTA}NEW: ATR (1.4), Volume Surge (2.0), Momentum (1.9){Colors.RESET}")
    print(f"{Colors.GREEN}LIVE API sectoral updates from http://localhost:3001/api/allIndices{Colors.RESET}")
    print(f"{Colors.BLUE}Updates every 5 minutes with REAL sectoral performance{Colors.RESET}")
    scanner = Enhanced3SectorScanner()
    try:
        scanner.run_enhanced_scanner()
    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Shutting down enhanced scanner...{Colors.RESET}")
        scanner.stop()

if __name__ == "__main__":
    main()
