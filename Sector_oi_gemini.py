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
# --- CONTROL FLAGS ---
# =========================
## --- OPTION BUYER ENHANCEMENT --- ##
# BUYER_MODE is now permanently enabled for this script's purpose.
BUYER_MODE = True
# RELAXED_MODE is set to False for stricter filtering.
RELAXED_MODE = False

# =========================
# --- INDICATOR WEIGHTS ---
# =========================
## --- OPTION BUYER ENHANCEMENT --- ##
# Weights are adjusted to prioritize momentum, trend, and volume.
ENHANCED_INDICATOR_WEIGHTS = {
    "Momentum": 2.2,
    "ADX": 2.1,
    "VolumeSurge": 2.0,
    "VWAP": 1.8,
    "EMA": 1.7,
    "ATR": 1.6,
    "MACD": 1.5,
    "Bollinger": 1.5, # Given more weight due to its role in volatility checks
    "OBV": 1.4,
    "RSI": 1.2,
    "ROC": 1.1,
    "Stochastic": 1.0,
    "CCI": 1.0,
    "MA": 1.0,
    "WWL": 1.0,
}
TIMEFRAME_WEIGHTS = {15: 3.0, 5: 2.5, 30: 2.0, 60: 1.5, "daily": 1.0}

# =========================
# --- PLACEHOLDER MAPS ---
# =========================
NSE_INDEX_TO_SECTOR = {
    "NIFTY IT": "Technology", "NIFTY PHARMA": "Pharma", "NIFTY FMCG": "Consumer",
    "NIFTY BANK": "Banking", "NIFTY AUTO": "Auto", "NIFTY METAL": "Metal",
    "NIFTY ENERGY": "Energy", "NIFTY REALTY": "Realty", "NIFTY INFRA": "Infrastructure",
    "NIFTY PSU BANK": "PSU Bank", "NIFTY PSE": "PSE", "NIFTY COMMODITIES": "Commodities",
    "NIFTY MNC": "Finance", "NIFTY FINANCIAL SERVICES": "Finance",
    "NIFTY INFRASTRUCTURE": "Infrastructure", "BANKNIFTY": "Banking", "NIFTYAUTO": "Auto",
    "NIFTYIT": "Technology", "NIFTYPHARMA": "Pharma", "NIFTY CONSUMER DURABLES": "Consumer Durables",
    "NIFTY HEALTHCARE INDEX": "Healthcare", "NIFTY CAPITAL MARKETS": "Capital Market",
    "NIFTY PRIVATE BANK": "Private Bank", "NIFTY OIL & GAS": "Oil and Gas",
    "NIFTY INDIA DEFENCE": "Defence", "NIFTY CORE HOUSING": "Core Housing",
    "NIFTY SERVICES SECTOR": "Services Sector", "NIFTY FINANCIAL SERVICES 25/50": "Financial Services 2550",
    "NIFTY INDIA TOURISM": "Tourism",
}

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
}

# =========================
# --- BUY/SELL/DELTA (OHLCV) ---
# =========================
def add_buy_sell_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    df["BuyVol"] = v * up_share
    df["SellVol"] = v * (1 - up_share)
    df["DeltaVol"] = df["BuyVol"] - df["SellVol"]
    return df

# =========================
# --- INDICATORS ---
# =========================
class EnhancedTechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        indicators = {}
        if df is None or len(df) < 26: # Increased requirement for stable EMAs
            return indicators
        try:
            close = df["Close"]; high = df["High"]; low = df["Low"]; vol = df["Volume"]
            
            # Common MAs and prices
            ma20 = close.rolling(20).mean()
            indicators["MA"] = ma20
            indicators["EMA"] = close.ewm(span=21).mean()
            
            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss.replace(0, np.nan)
            indicators["RSI"] = 100 - (100 / (1 + rs))

            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            indicators["MACD"] = macd_line - signal_line

            # Stochastic
            low14 = low.rolling(14).min()
            high14 = high.rolling(14).max()
            indicators["Stochastic"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)

            # ADX
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1/14, adjust=False).mean()
            high_diff = high.diff(); low_diff = low.diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
            minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff > 0), 0.0)
            plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr.replace(0, np.nan))
            minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr.replace(0, np.nan))
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
            indicators["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean()
            indicators["ATR"] = atr
            
            # Bollinger Bands & Squeeze ## --- OPTION BUYER ENHANCEMENT --- ##
            std20 = close.rolling(20).std()
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20
            indicators["Bollinger"] = (close - ma20) / (upper - lower).replace(0, np.nan) * 100
            bb_width = (upper - lower) / ma20.replace(0, np.nan)
            indicators["BB_Squeeze"] = bb_width < bb_width.rolling(120).min() * 1.5 # True if in a squeeze

            # ROC
            indicators["ROC"] = close.pct_change(12) * 100

            # OBV change
            obv = (np.sign(close.diff().fillna(0)) * vol.fillna(0)).cumsum()
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
            tpv = (high + low + close) / 3
            vwap_num = (tpv * vol).rolling(20).sum()
            vwap_den = vol.rolling(20).sum().replace(0, np.nan)
            indicators["VWAP"] = vwap_num / vwap_den

            # VolumeSurge
            avg_vol = vol.rolling(20).mean()
            indicators["VolumeSurge"] = (vol / avg_vol.replace(0, np.nan) - 1) * 100
            
            # Momentum
            price_mom = close.pct_change(10) * 100
            vol_mom = (vol / avg_vol.replace(0, np.nan) - 1) * 50
            indicators["Momentum"] = np.clip(50 + (price_mom * 0.7 + vol_mom * 0.3), 0, 100)

            return indicators
        except Exception as e:
            logger.error(f"Indicator calc error: {e}")
            return indicators

# ... [The normalize_indicator_value, get_option_chain, and compute_pcr_from_option_chain functions remain unchanged] ...
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
        if indicator_name == "Momentum": return max(0, min(100, value))
        return 50
    except Exception:
        return 50
def get_option_chain(symbol: str): return None
def compute_pcr_from_option_chain(option_chain) -> float | None: return None

# =========================
# --- SCANNER ---
# =========================
class Enhanced3SectorScanner:
    # ... [__init__, snapshot helpers, display helpers, API fetch logic remain largely the same] ...
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
        print(f"{Colors.CYAN}{Colors.BOLD}Scanner initialized for OPTION BUYING{Colors.RESET}")
    def _trim_sector_snapshots(self):
        try:
            if not self.sector_snapshots: return
            cutoff = datetime.now() - timedelta(minutes=self.sector_snapshot_retention_min)
            self.sector_snapshots = [s for s in self.sector_snapshots if s["timestamp"] >= cutoff]
        except Exception as e: logger.warning(f"Trim snapshots error: {e}")
    def _get_sector_change_at(self, sector: str, ref_time: datetime, offset_min: int):
        try:
            target = ref_time - timedelta(minutes=offset_min)
            cands = [s for s in self.sector_snapshots if s["sector"] == sector and s["timestamp"] <= target]
            if not cands: return None
            return max(cands, key=lambda s: s["timestamp"])["change"]
        except Exception as e:
            logger.warning(f"Snapshot lookup error {sector}: {e}"); return None
    def show_initialization_status(self):
        print(f"{Colors.CYAN}{Colors.BOLD}ENHANCED OPTION BUYER SCANNER{Colors.RESET}")
        print(f"Timeframes: 5m, 15m, 30m, 60m, Daily")
        print(f"Buyer Gates: {Colors.GREEN}Strictly Enforced{Colors.RESET}")
        self.show_sector_status()
        self.test_api_connection()
        print(f"{Colors.YELLOW}Running initial sector update...{Colors.RESET}")
        self.force_sector_update()
        print("-"*88)
    def test_api_connection(self):
        print(f"{Colors.BLUE}API CONNECTION TEST{Colors.RESET}")
        try:
            r = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            print(f"API HTTP: {r.status_code}")
        except Exception as e: print(f"{Colors.RED}API ERROR: {e}{Colors.RESET}")
    def show_sector_status(self):
        print(f"{Colors.MAGENTA}Best: {', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"{Colors.MAGENTA}Worst: {', '.join(self.worst_sectors)}{Colors.RESET}")
    def fetch_live_sectoral_performance(self):
        try:
            r = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            if r.status_code != 200: return False
            indices_data = r.json().get("data", r.json())
            curtime = datetime.now()
            sectoral = []
            for idx in indices_data:
                name = str(idx.get("symbol", idx.get("index", ""))).strip().upper()
                if name in NSE_INDEX_TO_SECTOR:
                    change_percent = float(idx.get("pChange", idx.get("changepercent", 0.0)))
                    sectoral.append({"index": name, "sector": NSE_INDEX_TO_SECTOR[name], "changepercent": change_percent, "timestamp": curtime})
            for row in sectoral: self.sector_snapshots.append({"timestamp": row["timestamp"], "sector": row["sector"], "change": float(row["changepercent"])})
            self._trim_sector_snapshots()
            sectoral.sort(key=lambda x: x["changepercent"], reverse=True)
            n = len(sectoral)
            self.best_sectors = [s["sector"] for s in sectoral[:min(4, n)]]
            self.worst_sectors = [s["sector"] for s in sectoral[-min(4, n):]][::-1]
            self.last_sectoral_update = curtime
            self.sectoral_history.append({"timestamp": curtime, "best": self.best_sectors[:], "worst": self.worst_sectors[:], "fulldata": sectoral[:]})
            self.display_sector_update(sectoral)
            return True
        except Exception as e:
            logger.error(f"API sector fetch error: {e}"); return False
    def _print_top_sector_delta_table(self, sectoral): pass # Can be skipped for brevity
    def display_sector_update(self, sectoral):
        ct = datetime.now().strftime('%H:%M:%S')
        print(f"{Colors.MAGENTA}{Colors.BOLD}{'-'*100}{Colors.RESET}")
        print(f"SECTOR PERFORMANCE UPDATE - {ct} IST")
        print(f"Best: {Colors.GREEN}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"Worst: {Colors.RED}{', '.join(self.worst_sectors)}{Colors.RESET}")
    def force_sector_update(self):
        ok = self.fetch_live_sectoral_performance()
        if not ok: print(f"{Colors.RED}Sector update failed; using last known lists{Colors.RESET}")
        return ok
    def is_market_open(self):
        now = datetime.now()
        ct = now.time()
        return now.weekday() < 5 and (self.market_start <= ct <= self.market_end)
    def normalize_live_data(self, df, symbol):
        try:
            d = df.copy()
            d.rename(columns={c: c.capitalize() for c in d.columns}, inplace=True)
            if 'Timestamp' in d.columns: d.rename(columns={'Timestamp': 'Date'}, inplace=True)
            d.set_index(pd.to_datetime(d.index), inplace=True)
            d = add_buy_sell_delta_columns(d)
            return d
        except Exception as e:
            logger.error(f"Normalize error {symbol}: {e}"); return None
    def check_gapdown(self, df): return False # Disable gapdown filter for option buying focus
    def fetch_live_data(self, symbol, timeframe):
        try:
            tfmap = {5:"5 min",15:"15 min",30:"30 min",60:"60 mins","daily":"EOD"}
            bar_size = tfmap.get(timeframe)
            if not bar_size: return None, False
            duration = {"daily": "365 D", 60: "60 D", 30: "20 D"}.get(timeframe, "10 D")
            raw = tdhist.get_historic_data(symbol, duration=duration, bar_size=bar_size)
            if raw is None or len(raw) < 26: return None, False
            df = self.normalize_live_data(raw, symbol)
            return df.tail(250), False
        except Exception as e:
            logger.error(f"Live fetch error {symbol}@{timeframe}: {e}"); return None, False
    
    ## --- OPTION BUYER ENHANCEMENT --- ##
    def check_option_buyer_gates(self, symbol, timeframes_data, sector, base_score):
        """
        Applies a very strict set of rules to find explosive moves suitable for option buying.
        Returns (passes_gates: bool, reason_checklist: dict)
        """
        checklist = {}
        try:
            # We need at least the 5m and 15m timeframes
            if 5 not in timeframes_data or 15 not in timeframes_data:
                return False, {"Setup": "Missing primary timeframes (5m, 15m)"}
            
            df5 = timeframes_data[5]
            df15 = timeframes_data[15]
            ind5 = EnhancedTechnicalIndicators.calculate_all_indicators(df5)
            ind15 = EnhancedTechnicalIndicators.calculate_all_indicators(df15)

            # Gate 1: Strong Trend (ADX)
            adx5 = ind5.get("ADX", pd.Series([0])).iloc[-1]
            adx15 = ind15.get("ADX", pd.Series([0])).iloc[-1]
            checklist['ADX > 25'] = adx5 > 25 and adx15 > 25

            # Gate 2: Price Action Confirmation (Above/Below Key Averages)
            price = df5["Close"].iloc[-1]
            ema21_5m = ind5.get("EMA", pd.Series([price+1])).iloc[-1]
            vwap_5m = ind5.get("VWAP", pd.Series([price+1])).iloc[-1]
            is_buy = base_score > 50
            if is_buy:
                checklist['Price > EMA/VWAP'] = price > ema21_5m and price > vwap_5m
            else: # is_sell
                checklist['Price < EMA/VWAP'] = price < ema21_5m and price < vwap_5m

            # Gate 3: Volatility Breakout (Bollinger Band Squeeze)
            # Looks for a breakout from a recent low-volatility period
            is_in_squeeze = ind5.get("BB_Squeeze", pd.Series([False])).iloc[-2]
            broke_upper_band = df5["Close"].iloc[-1] > (ind5["MA"].iloc[-1] + 2 * df5["Close"].rolling(20).std().iloc[-1])
            broke_lower_band = df5["Close"].iloc[-1] < (ind5["MA"].iloc[-1] - 2 * df5["Close"].rolling(20).std().iloc[-1])
            if is_buy:
                checklist['BB Squeeze Breakout'] = is_in_squeeze and broke_upper_band
            else:
                checklist['BB Squeeze Breakout'] = is_in_squeeze and broke_lower_band
            
            # Gate 4: Volume & Immediate Momentum Confirmation
            vol_surge = ind5.get("VolumeSurge", pd.Series([0])).iloc[-1]
            checklist['Volume Surge > 75%'] = vol_surge > 75
            
            # Check last candle's buying/selling pressure
            last_delta = df5["DeltaVol"].iloc[-1]
            last_vol = df5["Volume"].iloc[-1]
            if is_buy:
                checklist['Recent Buy Delta'] = last_delta > 0.2 * last_vol # At least 60% buy volume
            else:
                checklist['Recent Sell Delta'] = last_delta < -0.2 * last_vol # At least 60% sell volume

            # Gate 5: Sector Momentum
            now = datetime.now()
            curr_map = {s["sector"]: float(s["changepercent"]) for s in self.sectoral_history[-1]['fulldata']} if self.sectoral_history else {}
            cur = curr_map.get(sector, 0.0)
            chg_15 = self._get_sector_change_at(sector, now, 15)
            if chg_15 is not None:
                d15 = cur - chg_15
                if is_buy:
                    checklist['Sector Momentum ↑'] = d15 > 0.10 # Sector accelerating upwards
                else:
                    checklist['Sector Momentum ↓'] = d15 < -0.10 # Sector accelerating downwards
            else:
                checklist['Sector Momentum'] = False

            # Gate 6: Multi-Timeframe Agreement
            tf_scores = self.last_cycle_scores.get(symbol, {})
            s5, s15, s30 = tf_scores.get(5, 50), tf_scores.get(15, 50), tf_scores.get(30, 50)
            if is_buy:
                checklist['TF Align (5,15,30 > 65)'] = all(s > 65 for s in [s5, s15, s30])
            else:
                checklist['TF Align (5,15,30 < 35)'] = all(s < 35 for s in [s5, s15, s30])

            all_gates_passed = all(checklist.values())
            return all_gates_passed, checklist

        except Exception as e:
            logger.warning(f"Buyer gates error for {symbol}: {e}")
            return False, {"Error": str(e)}

    def calculate_enhanced_signals(self, symbol, timeframes_data):
        try:
            sector = next((s for s, lst in SECTOR_STOCKS.items() if symbol in lst), "NA")
            total_weighted = 0.0; total_w = 0.0; tf_scores = {}
            for tf, df in timeframes_data.items():
                if df is None or len(df) < 26: continue
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
                            p = (price - base) / base * 100 if base != 0 else 0
                            norm = 75 if p > 0.5 else 60 if p > 0 else 40 if p < 0 else 25
                        else:
                            norm = normalize_indicator_value(name, val)
                        tf_score += norm * weight; tf_w += weight
                if tf_w > 0:
                    tf_final = tf_score / tf_w
                    tf_scores[tf] = tf_final
                    mult = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                    total_weighted += tf_final * mult
                    total_w += mult
            
            self.last_cycle_scores[symbol] = tf_scores
            if total_w <= 0: return "Neutral", 0, {}

            base_score = total_weighted / total_w
            
            # Sector boost
            if sector in self.best_sectors: base_score += 15
            elif sector in self.worst_sectors: base_score -= 15

            # --- OPTION BUYER GATE CHECK --- ##
            passes_gates, checklist = self.check_option_buyer_gates(symbol, timeframes_data, sector, base_score)
            
            if not passes_gates:
                return "Neutral", base_score, checklist

            if base_score >= 75: return "EXPLOSIVE BUY", base_score, checklist
            if base_score >= 65: return "Strong Buy", base_score, checklist
            if base_score <= 25: return "EXPLOSIVE SELL", base_score, checklist
            if base_score <= 35: return "Strong Sell", base_score, checklist
            
            return "Neutral", base_score, checklist
            
        except Exception as e:
            logger.error(f"Signal calc error {symbol}: {e}")
            return "Neutral", 0, {}

    # --- One scan cycle ---
    def enhanced_scan_cycle(self):
        nowdt = datetime.now()
        if not self.is_market_open():
            print(f"{Colors.YELLOW}Market closed. Waiting...{Colors.RESET}")
            timemodule.sleep(self.scan_interval); return

        start = timemodule.time()
        print(f"{Colors.CYAN}Starting Option Buyer scan {nowdt.strftime('%H:%M:%S')}{Colors.RESET}")
        self.force_sector_update()

        targets = sorted(set(stock for sector in self.best_sectors + self.worst_sectors for stock in SECTOR_STOCKS.get(sector, [])))
        print(f"Scanning {len(targets)} symbols...")

        signals = []
        def process_symbol(sym):
            try:
                tfs = {tf: self.fetch_live_data(sym, tf)[0] for tf in [5, 15, 30, 60]}
                tfs = {k: v for k, v in tfs.items() if v is not None}
                
                if len(tfs) >= 3:
                    sig, score, checklist = self.calculate_enhanced_signals(sym, tfs)
                    if sig != "Neutral":
                        sector = next((s for s, lst in SECTOR_STOCKS.items() if sym in lst), "NA")
                        df5 = tfs.get(5)
                        price = df5["Close"].iloc[-1] if df5 is not None else 0
                        atr = EnhancedTechnicalIndicators.calculate_all_indicators(df5)['ATR'].iloc[-1] if df5 is not None else 0
                        return {"symbol": sym, "signal": sig, "score": score, "sector": sector,
                                "price": price, "atr": atr, "checklist": checklist}
                return None
            except Exception as e:
                logger.error(f"Proc error {sym}: {e}"); return None

        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(process_symbol, s): s for s in targets}
            for f in as_completed(futs):
                r = f.result()
                if r: signals.append(r)

        scan_time = timemodule.time() - start
        self.display_signals(signals, scan_time)
        print(f"{Colors.CYAN}Heartbeat: next scan at {(datetime.now()+timedelta(seconds=self.scan_interval)).strftime('%H:%M:%S')} IST{Colors.RESET}")

    def display_signals(self, signals, scan_time):
        print(f"\n{Colors.BLUE}Scan time: {scan_time:.2f}s | Signals Found: {len(signals)}{Colors.RESET}")
        if not signals:
            print(f"{Colors.YELLOW}No high-probability setups found this cycle.{Colors.RESET}"); return
        
        bulls = sorted([s for s in signals if "Buy" in s["signal"]], key=lambda x: x["score"], reverse=True)
        bears = sorted([s for s in signals if "Sell" in s["signal"]], key=lambda x: x["score"])

        def print_signal(s, color):
            signal_color = Colors.GREEN if "Buy" in s["signal"] else Colors.RED
            stop_loss = s['price'] - 2 * s['atr'] if "Buy" in s['signal'] else s['price'] + 2 * s['atr']
            print(f"{color}{Colors.BOLD}{s['symbol']:<12}{Colors.RESET} | {signal_color}{s['signal']:<16}{Colors.RESET} | Score: {s['score']:.1f} @ ₹{s['price']:.2f} | SL Suggestion: ~₹{stop_loss:.2f}")
            checklist_str = []
            for reason, passed in s['checklist'].items():
                if passed:
                    checklist_str.append(f"{Colors.GREEN}✓ {reason}{Colors.RESET}")
            print(f"  └─ Gates Passed: {' | '.join(checklist_str)}")

        if bulls:
            print(f"\n{Colors.GREEN}{Colors.BOLD}--- BULLISH SIGNALS (Calls) ---{Colors.RESET}")
            for s in bulls: print_signal(s, Colors.GREEN)
        
        if bears:
            print(f"\n{Colors.RED}{Colors.BOLD}--- BEARISH SIGNALS (Puts) ---{Colors.RESET}")
            for s in bears: print_signal(s, Colors.RED)

    # --- Loop ---
    def run(self):
        try:
            while True:
                self.enhanced_scan_cycle()
                timemodule.sleep(self.scan_interval)
        except KeyboardInterrupt:
            print(f"{Colors.YELLOW}Stopped by user{Colors.RESET}")
# =========================
# --- MAIN ---
# =========================
def main():
    scanner = Enhanced3SectorScanner()
    scanner.show_initialization_status()
    scanner.run()

if __name__ == "__main__":
    main()