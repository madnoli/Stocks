import os
import logging
import warnings
from datetime import datetime, time as dtime, timedelta

import numpy as np
import pandas as pd
import pytz
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from logzero import logger
from truedata.history import TD_hist

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
}

TIMEFRAME_WEIGHTS = {15: 3.0, 5: 2.5, 30: 2.0, 60: 1.5, "daily": 1.0}

# =========================
# --- NSE SECTOR MAPS ---
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
}

SECTOR_STOCKS = {
    "Technology": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "COFORGE", "PERSISTENT", "KPITTECH", "TATAELXSI"],
    "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR", "EICHERMOT", "ASHOKLEY", "BOSCHLTD"],
    "Banking": ["HDFCBANK", "ICICIBANK", "SBIN", "KOTAKBANK", "AXISBANK", "PNB", "BANKBARODA", "CANBK", "INDUSINDBK", "FEDERALBNK"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA", "TORNTPHARM", "ALKEM", "LAURUSLABS", "BIOCON", "ZYDUSLIFE", "MANKIND"],
    "Energy": ["RELIANCE", "NTPC", "BPCL", "IOC", "ONGC", "GAIL", "HINDPETRO", "JSWENERGY", "TATAPOWER", "COALINDIA", "POWERGRID"],
    "Metal": ["TATASTEEL", "JSWSTEEL", "SAIL", "JINDALSTEL", "HINDALCO", "NMDC"],
    "Consumer": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR", "MARICO", "COLPAL"],
    "PSU Bank": ["SBIN", "PNB", "BANKBARODA", "CANBK", "UNIONBANK", "BANKINDIA"],
    "Finance": ["BAJFINANCE", "SHRIRAMFIN", "CHOLAFIN", "HDFCLIFE", "ICICIPRULI"],
    "Realty": ["DLF", "LODHA", "PRESTIGE", "GODREJPROP", "OBEROIRLTY", "PHOENIXLTD"],
    "PSE": ["BEL", "BHEL", "NHPC", "GAIL", "IOC", "NTPC", "POWERGRID", "OIL", "ONGC", "NMDC", "BPCL", "HAL", "RVNL", "PFC", "COALINDIA"],
    "Commodities": ["AMBUJACEM", "ULTRACEMCO", "JSWSTEEL", "HINDALCO", "RELIANCE", "GRASIM", "TATASTEEL", "COALINDIA"],
    "Consumer Durables": ["TITAN", "DIXON", "HAVELLS", "POLYCAB", "VOLTAS"],
    "Healthcare": ["SUNPHARMA", "DIVISLAB", "CIPLA", "DRREDDY", "MANKIND", "ZYDUSLIFE", "LUPIN", "APOLLOHOSP"],
    "Capital Market": ["HDFCAMC", "BSE", "MCX", "CDSL", "ANGELONE"],
    "Private Bank": ["HDFCBANK", "ICICIBANK", "KOTAKBANK", "AXISBANK", "IDFCFIRSTB", "INDUSINDBK", "FEDERALBNK"],
    "Oil and Gas": ["RELIANCE", "ONGC", "IOC", "BPCL", "GAIL", "HINDPETRO", "OIL", "PETRONET", "IGL"],
    "Defence": ["HAL", "BEL", "SOLARINDS", "MAZDOCK", "BDL"],
    "Core Housing": ["ULTRACEMCO", "ASIANPAINT", "DLF", "AMBUJACEM", "LODHA", "DIXON", "POLYCAB", "SHREECEM", "HAVELLS"],
    "Services Sector": ["HDFCBANK", "BHARTIARTL", "TCS", "ICICIBANK", "SBIN", "INFY", "BAJFINANCE", "HCLTECH"],
    "Financial Services 2550": ["HDFCBANK", "ICICIBANK", "SBIN", "BAJFINANCE", "KOTAKBANK", "AXISBANK"],
    "Tourism": ["INDIGO", "INDHOTEL", "IRCTC", "JUBLFOOD"],
}

# =========================
# --- INDICATORS ---
# =========================
class EnhancedTechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame):
        indicators = {}
        if df is None or len(df) < 20:
            return indicators
        try:
            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            vol = df["Volume"]

            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            indicators["RSI"] = 100 - (100 / (1 + rs))

            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            indicators["MACD"] = macd_line - signal_line

            low14 = low.rolling(window=14).min()
            high14 = high.rolling(window=14).max()
            indicators["Stochastic"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)

            indicators["MA"] = close.rolling(window=20).mean()
            indicators["EMA"] = close.ewm(span=21).mean()

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

            ma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20
            indicators["Bollinger"] = (close - ma20) / (upper - lower).replace(0, np.nan) * 100

            indicators["ROC"] = close.pct_change(periods=12) * 100

            obv = np.sign(close.diff().fillna(0)) * vol.fillna(0)
            obv = obv.cumsum()
            indicators["OBV"] = obv.pct_change(periods=10) * 100

            tp = (high + low + close) / 3
            sma_tp = tp.rolling(window=20).mean()
            mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
            indicators["CCI"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

            if len(df) >= 20:
                tpv = (high + low + close) / 3
                vwap_num = (tpv * vol).rolling(window=20).sum()
                vwap_den = vol.rolling(window=20).sum().replace(0, np.nan)
                indicators["VWAP"] = vwap_num / vwap_den

            indicators["ATR"] = atr

            if len(df) >= 20:
                avg20 = vol.rolling(window=20).mean()
                vr = (vol / avg20.replace(0, np.nan))
                indicators["VolumeSurge"] = np.clip((vr - 0.5) * 40, 0, 100)

            if len(df) >= 10:
                price_mom = close.pct_change(periods=10) * 100
                avg10 = vol.rolling(window=10).mean()
                vol_mom = (vol / avg10.replace(0, np.nan) - 1) * 100
                mom_score = price_mom * 0.7 + vol_mom * 0.3
                indicators["Momentum"] = np.clip(50 + mom_score * 1.5, -50, 50)

            return indicators
        except Exception as e:
            logger.error(f"Indicators error: {e}")
            return indicators

def normalize_indicator_value(name, value):
    try:
        if name == "RSI":
            return max(0, min(100, value))
        elif name == "MACD":
            return 50 + max(-25, min(25, value / 10))
        elif name == "Stochastic":
            return max(0, min(100, value))
        elif name in ("MA", "EMA", "VWAP"):
            return 50
        elif name == "ADX":
            return max(0, min(100, value))
        elif name == "Bollinger":
            return max(0, min(100, (value + 100) / 2))
        elif name == "ROC":
            return 50 + max(-25, min(25, value / 2))
        elif name == "OBV":
            return 50 + max(-25, min(25, value))
        elif name == "CCI":
            return max(0, min(100, (value + 200) / 4))
        elif name == "WWL":
            return max(0, min(100, (value + 100)))
        elif name == "ATR":
            return 50
        elif name == "VolumeSurge":
            return max(0, min(100, value))
        elif name == "Momentum":
            return max(0, min(100, value + 50))
        else:
            return 50
    except Exception:
        return 50

# =========================
# --- SAFE FORMATTER ---
# =========================
def safe_fmt(val, width=0, prec=2):
    if val is None:
        s = "n/a"
    else:
        try:
            if isinstance(val, str):
                s = val
            else:
                v = float(val)
                if np.isnan(v) or np.isinf(v):
                    s = "n/a"
                else:
                    s = f"{v:.{prec}f}"
        except Exception:
            s = "n/a"
    return f"{s:>{width}}"

# =========================
# --- OI / PCR HELPERS ---
# =========================
def pcr_bucket(pcr: float) -> str:
    if pcr is None or (isinstance(pcr, float) and np.isnan(pcr)):
        return "n/a"
    if pcr < 0.6:
        return "Ext Bearish"
    if pcr < 0.8:
        return "Bearish"
    if pcr <= 1.2:
        return "Neutral"
    if pcr <= 1.4:
        return "Bullish"
    return "Ext Bullish"

def spot_from_symbol(sym: str) -> str:
    return sym.replace("-I", "").replace("-II", "")

def build_option_symbol(underlying: str, expiry: str, strike: int, side: str) -> str:
    return f"{underlying}{expiry}{strike}{side}"  # adapt if provider uses different format

def get_current_expiries_for_underlying(underlying: str):
    today = datetime.now(pytz.timezone("Asia/Kolkata")).date()
    mon = today.strftime("%b").upper()
    yy = today.strftime("%y")
    return [f"{yy}{mon}"]  # extend with weekly codes if needed

def get_relevant_strikes(underlying: str, ltp: float):
    step = 50 if ltp <= 500 else 100
    center = int(round(ltp / step) * step)
    return list(range(center - 4 * step, center + 5 * step, step))

def fetch_series(symbol: str, start, end, bar_size="1 min"):
    try:
        df = tdhist.get_series(symbol, start_time=start, end_time=end, duration=end - start, bar_size="1 min")
        if isinstance(df, pd.DataFrame):
            low = {c.lower(): c for c in df.columns}
            if "oi" in low:
                df.rename(columns={low["oi"]: "oi"}, inplace=True)
            return df
    except Exception:
        return None
    return None

def get_option_chain_OI(underlying: str, timeframe_min: int, approx_ltp: float = None):
    window = pd.Timedelta(minutes=timeframe_min)
    now_ts = pd.Timestamp.now(tz="Asia/Kolkata")
    start = now_ts - window
    if approx_ltp is None:
        try:
            s = tdhist.get_series(underlying, start_time=now_ts - pd.Timedelta(minutes=60),
                                  end_time=now_ts, duration=pd.Timedelta(minutes=60), bar_size="1 min")
            if isinstance(s, pd.DataFrame) and not s.empty:
                ccol = [c for c in s.columns if str(c).lower() in ("close", "ltp", "last")]
                if ccol:
                    approx_ltp = float(s[ccol[0]].iloc[-1])
        except Exception:
            approx_ltp = None

    expiries = get_current_expiries_for_underlying(underlying)
    strikes = get_relevant_strikes(underlying, approx_ltp or 1000.0)

    ce_sum = 0.0
    pe_sum = 0.0
    for expiry in expiries:
        for k in strikes:
            ce_sym = build_option_symbol(underlying, expiry, k, "CE")
            pe_sym = build_option_symbol(underlying, expiry, k, "PE")
            try:
                ce_df = fetch_series(ce_sym, start, now_ts, "1 min")
                pe_df = fetch_series(pe_sym, start, now_ts, "1 min")
                if isinstance(ce_df, pd.DataFrame) and "oi" in ce_df.columns and not ce_df.empty:
                    ce_sum += float(ce_df["oi"].iloc[-1])
                if isinstance(pe_df, pd.DataFrame) and "oi" in pe_df.columns and not pe_df.empty:
                    pe_sum += float(pe_df["oi"].iloc[-1])
            except Exception:
                continue
    if ce_sum == 0 and pe_sum == 0:
        return np.nan, np.nan, np.nan
    pcr = (pe_sum / ce_sum) if ce_sum else np.nan
    return pe_sum, ce_sum, pcr

def compute_multi_tf_pcr(underlying: str, approx_ltp: float = None):
    def only_pcr(x):
        try:
            v = x[2]
            return float(v) if v is not None and not np.isnan(v) else np.nan
        except Exception:
            return np.nan
    p5 = only_pcr(get_option_chain_OI(underlying, 5, approx_ltp))
    p15 = only_pcr(get_option_chain_OI(underlying, 15, approx_ltp))
    p30 = only_pcr(get_option_chain_OI(underlying, 30, approx_ltp))
    p60 = only_pcr(get_option_chain_OI(underlying, 60, approx_ltp))
    return {"5": p5, "15": p15, "30": p30, "60": p60}

# =========================
# --- SCANNER CORE ---
# =========================
class EnhancedScannerPCR:
    def __init__(self):
        self.is_running = False
        self.best_sectors = ["Pharma", "Healthcare", "Technology", "Financial Services 2550"]
        self.worst_sectors = ["Defence", "Energy", "PSU Bank", "Realty"]
        self.last_sectoral_update = None
        self.scan_interval = 300
        self.last_cycle_scores = {}
        self.current_cycle_scores = {}
        self.market_start = dtime(9, 15)
        self.market_end = dtime(15, 30)

    def normalize_live_data(self, df, symbol):
        try:
            if df is None or len(df) == 0:
                return None
            dfc = df.copy()
            dfc.rename(columns={c: c.capitalize() for c in dfc.columns}, inplace=True)
            rename = {}
            for c in dfc.columns:
                lc = str(c).lower()
                if lc in ("timestamp", "time", "date"):
                    rename[c] = "Date"
                elif lc == "open": rename[c] = "Open"
                elif lc == "high": rename[c] = "High"
                elif lc == "low": rename[c] = "Low"
                elif lc in ("close", "last", "ltp"): rename[c] = "Close"
                elif lc in ("volume", "vol"): rename[c] = "Volume"
            if rename:
                dfc.rename(columns=rename, inplace=True)

            if "Date" not in dfc.columns:
                if isinstance(dfc.index, pd.DatetimeIndex):
                    dfc["Date"] = dfc.index
                else:
                    return None

            dfc["Date"] = pd.to_datetime(dfc["Date"], errors="coerce")
            dfc = dfc.dropna(subset=["Date"])
            dfc.set_index("Date", inplace=True)
            for col in ("Open", "High", "Low", "Close", "Volume"):
                if col in dfc.columns:
                    dfc[col] = pd.to_numeric(dfc[col], errors="coerce")
            dfc = dfc.dropna(subset=["Open", "High", "Low", "Close"])
            return dfc.sort_index()
        except Exception as e:
            logger.error(f"Normalize error {symbol}: {e}")
            return None

    def fetch_live_data(self, symbol, timeframe):
        try:
            tfmap = {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 mins", "daily": "EOD"}
            bar_size = tfmap.get(timeframe)
            if not bar_size:
                return None
            duration = "10 D" if timeframe in (5, 15) else "20 D" if timeframe == 30 else "60 D" if timeframe == 60 else "365 D"
            rawdf = tdhist.get_historic_data(symbol, duration=duration, bar_size=bar_size)
            return self.normalize_live_data(rawdf, symbol)
        except Exception as e:
            logger.error(f"Fetch error {symbol}@{timeframe}: {e}")
            return None

    def update_sectors_from_api(self):
        try:
            resp = requests.get("http://localhost:3001/api/allIndices", timeout=8)
            if resp.status_code != 200:
                return False
            data = resp.json()
            if isinstance(data, dict):
                data = data.get("data") or data.get("indices") or data.get("results") or data
            perf = []
            now = datetime.now()
            for row in data:
                if not isinstance(row, dict):
                    continue
                name = None
                for f in ("name", "symbol", "index", "indexName"):
                    if f in row and row[f]:
                        name = str(row[f]).strip().upper()
                        break
                if not name or name not in NSE_INDEX_TO_SECTOR:
                    continue
                chg = 0.0
                for f in ("changepercent", "changePercent", "pChange", "percentChange", "pchg", "change"):
                    if f in row and row[f] is not None:
                        try:
                            chg = float(row[f])
                            break
                        except Exception:
                            pass
                perf.append({"index": name, "sector": NSE_INDEX_TO_SECTOR[name], "chg": chg})
            if not perf:
                return False
            perf.sort(key=lambda x: x["chg"], reverse=True)
            n = len(perf)
            self.best_sectors = [perf[i]["sector"] for i in range(min(4, n))]
            self.worst_sectors = [perf[-i]["sector"] for i in range(1, min(4, n) + 1)]
            self.last_sectoral_update = now
            return True
        except Exception:
            return False

    def score_symbol(self, symbol, tf_data):
        try:
            if not tf_data:
                return "Neutral", 0
            sector = next((s for s, lst in SECTOR_STOCKS.items() if symbol in lst), None)
            if not sector:
                return "Neutral", 0
            total_w = 0.0
            total_ws = 0.0
            tf_scores = {}
            for tf, df in tf_data.items():
                if df is None or len(df) < 20:
                    continue
                indicators = EnhancedTechnicalIndicators.calculate_all_indicators(df)
                if not indicators:
                    continue
                tf_score = 0.0
                tf_weight_sum = 0.0
                cp = df["Close"].iloc[-1]
                for name, wt in ENHANCED_INDICATOR_WEIGHTS.items():
                    if name in indicators and indicators[name] is not None and not indicators[name].empty:
                        v = indicators[name].iloc[-1]
                        if pd.isna(v):
                            continue
                        if name in ("MA", "EMA", "VWAP"):
                            base = v
                            if pd.isna(base) or base == 0:
                                norm = 50
                            else:
                                diff = (cp - base) / base * 100
                                if diff >= 2:
                                    norm = 75
                                elif diff >= 0:
                                    norm = 60
                                elif diff >= -2:
                                    norm = 50
                                elif diff >= -5:
                                    norm = 40
                                else:
                                    norm = 25
                        else:
                            norm = normalize_indicator_value(name, v)
                        tf_score += norm * wt
                        tf_weight_sum += wt
                if tf_weight_sum <= 0:
                    continue
                tf_final = tf_score / tf_weight_sum
                tf_scores[tf] = tf_final
                mult = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                total_ws += tf_final * mult
                total_w += mult
            if total_w <= 0:
                return "Neutral", 0
            score = total_ws / total_w
            if len(tf_scores) >= 4:
                bulls = sum(1 for v in tf_scores.values() if v >= 55)
                bears = sum(1 for v in tf_scores.values() if v <= 45)
                if bulls >= 3:
                    score += 8
                elif bears >= 3:
                    score -= 8
            has_long = ("daily" in tf_data) or (60 in tf_data)
            if sector in self.best_sectors:
                r = self.best_sectors.index(sector) + 1
                boost_map = {1: 25, 2: 20, 3: 15, 4: 10} if has_long else {1: 20, 2: 15, 3: 10, 4: 5}
                score += boost_map.get(r, 0)
            elif sector in self.worst_sectors:
                r = self.worst_sectors.index(sector) + 1
                drag_map = {1: -25, 2: -20, 3: -15, 4: -10} if has_long else {1: -20, 2: -15, 3: -10, 4: -5}
                score += drag_map.get(r, 0)

            if score >= 82: return "Very Strong Buy", score
            if score >= 72: return "Strong Buy", score
            if score >= 60: return "Buy", score
            if score <= 18: return "Very Strong Sell", score
            if score <= 28: return "Strong Sell", score
            if score <= 40: return "Sell", score
            return "Neutral", score
        except Exception as e:
            logger.error(f"Score error {symbol}: {e}")
            return "Neutral", 0

    def enrich_tape(self, symbol):
        try:
            now_ts = pd.Timestamp.now(tz="Asia/Kolkata")
            df = tdhist.get_series(symbol, start_time=now_ts - pd.Timedelta(minutes=60),
                                   end_time=now_ts, duration=pd.Timedelta(minutes=60), bar_size="1 min")
            if not isinstance(df, pd.DataFrame) or df.empty:
                return np.nan, np.nan, np.nan, np.nan
            cols = {c.lower(): c for c in df.columns}
            close_col = cols.get("close") or cols.get("last") or cols.get("ltp")
            ltp = float(df[close_col].iloc[-1]) if close_col else np.nan
            byv = float(df[cols.get("buy_volume")].iloc[-1]) if "buy_volume" in cols else np.nan
            selv = float(df[cols.get("sell_volume")].iloc[-1]) if "sell_volume" in cols else np.nan
            dvol = (byv - selv) if (isinstance(byv, (int, float)) and isinstance(selv, (int, float))) else np.nan
            return ltp, byv, selv, dvol
        except Exception:
            return np.nan, np.nan, np.nan, np.nan

    def print_sheet_like(self, title: str, rows: list, bullish=True):
        print()
        print(f"{Colors.BOLD}{title}{Colors.RESET}")
        header = f"{'Sr No':<5} {'Stock Name':<12} {'Sector':<14} {'Signal':<18} {'Score':>6} {'Score Δ':>8} {'TF':>3} {'TF Coverrage':<14} {'Strenght':<10} {'By Volume':>10} {'Sell Volume':>12} {'Delta Volume':>12} {'5 min PCR':>10} {'15min PCR':>10} {'30 Min PCR':>12} {'60 Min PCr':>12} {'PCR remark':<12}"
        print(header)
        print("-" * len(header))
        for i, s in enumerate(rows, 1):
            tf_details = s.get("tfdetails", [])
            tf_display = ",".join([str(tf) if isinstance(tf, int) else "D" for tf in tf_details])[:14]
            deviation = abs(s["score"] - 50)
            strength = "Exceptional" if deviation >= 40 else "Very Strong" if deviation >= 30 else "Strong" if deviation >= 20 else "Moderate"
            print(
                f"{i:<5} "
                f"{s['symbol']:<12} "
                f"{s['sector']:<14} "
                f"{s['signal']:<18} "
                f"{safe_fmt(s.get('score'),6,1)} "
                f"{f'{s.get('score_delta','n/a')}'.rjust(8)} "
                f"{s.get('timeframes',0):>3} "
                f"{tf_display:<14} "
                f"{strength:<10} "
                f"{safe_fmt(s.get('by_volume'),10,0)} "
                f"{safe_fmt(s.get('sell_volume'),12,0)} "
                f"{safe_fmt(s.get('delta_volume'),12,0)} "
                f"{safe_fmt(s.get('pcr_5'),10,2)} "
                f"{safe_fmt(s.get('pcr_15'),10,2)} "
                f"{safe_fmt(s.get('pcr_30'),12,2)} "
                f"{safe_fmt(s.get('pcr_60'),12,2)} "
                f"{(s.get('pcr_remark','n/a')):<12}"
            )

    def print_sector_tables(self, best_df: pd.DataFrame, worst_df: pd.DataFrame):
        print()
        print("Sector Details")
        print()
        print("Top 5 Best Performing sector")
        hdr = f"{'Sr No':<5} {'Sector Name':<20} {'Now %':>8} {'Δ vs 5m':>10} {'Δ vs 15m':>10} {'Δ vs 30m':>10}"
        print(hdr)
        print("-" * len(hdr))
        for i, r in best_df.head(5).reset_index(drop=True).iterrows():
            print(f"{i+1:<5} {r['sector']:<20} {r['now_pct']:>8.2f} {r['d5']:>10.2f} {r['d15']:>10.2f} {r['d30']:>10.2f}")
        print()
        print("Top 5 Worst Performing sector")
        print(hdr)
        print("-" * len(hdr))
        for i, r in worst_df.head(5).reset_index(drop=True).iterrows():
            print(f"{i+1:<5} {r['sector']:<20} {r['now_pct']:>8.2f} {r['d5']:>10.2f} {r['d15']:>10.2f} {r['d30']:>10.2f}")

    def is_market_open(self):
        now = datetime.now()
        if now.weekday() >= 5:
            return False
        return self.market_start <= now.time() <= self.market_end

    def enhanced_scan_cycle(self):
        if not self.is_market_open():
            print("Market closed, skipping cycle.")
            return

        print(f"{Colors.CYAN}Starting scan {datetime.now().strftime('%H:%M:%S')}{Colors.RESET}")
        self.update_sectors_from_api()

        target = set()
        for i, sec in enumerate(self.best_sectors):
            if sec in SECTOR_STOCKS:
                target.update(SECTOR_STOCKS[sec][:12 if i == 0 else 10 if i == 1 else 8 if i == 2 else 6])
        for i, sec in enumerate(self.worst_sectors):
            if sec in SECTOR_STOCKS:
                target.update(SECTOR_STOCKS[sec][:12 if i == 0 else 10 if i == 1 else 8 if i == 2 else 6])
        symbols = list(target)
        if not symbols:
            print("No symbols gathered from sectors.")
            return

        signals = []

        def work(sym):
            tf_data = {}
            for tf in [5, 15, 30, 60, "daily"]:
                df = self.fetch_live_data(sym, tf)
                if df is not None:
                    tf_data[tf] = df
            if len(tf_data) < 3:
                return None
            signal, score = self.score_symbol(sym, tf_data)
            if abs(score - 50) < 15:
                return None
            sector = next((s for s, lst in SECTOR_STOCKS.items() if sym in lst), "NA")
            ltp, byv, selv, dvol = self.enrich_tape(sym)
            root = spot_from_symbol(sym)
            pcrs = compute_multi_tf_pcr(root, ltp if not np.isnan(ltp) else None)
            row = {
                "symbol": sym,
                "sector": sector,
                "signal": signal,
                "score": score,
                "timeframes": len(tf_data),
                "tfdetails": list(tf_data.keys()),
                "ltp": ltp,
                "by_volume": byv,
                "sell_volume": selv,
                "delta_volume": dvol,
                "pcr_5": pcrs.get("5"),
                "pcr_15": pcrs.get("15"),
                "pcr_30": pcrs.get("30"),
                "pcr_60": pcrs.get("60"),
            }
            prev = self.last_cycle_scores.get(sym)
            row["score_delta"] = "n/a" if prev is None else f"{score - prev:+.1f}"
            row["pcr_remark"] = pcr_bucket(row["pcr_15"] if row["pcr_15"] is not None else row["pcr_30"])
            self.current_cycle_scores[sym] = score
            return row

        with ThreadPoolExecutor(max_workers=4) as exe:
            futs = [exe.submit(work, s) for s in symbols]
            for f in as_completed(futs):
                r = f.result()
                if r:
                    signals.append(r)

        bullish = [s for s in signals if "Buy" in s["signal"]]
        bearish = [s for s in signals if "Sell" in s["signal"]]
        bullish.sort(key=lambda x: x["score"], reverse=True)
        bearish.sort(key=lambda x: x["score"])

        self.print_sheet_like("Bullish Stocks", bullish[:20], bullish=True)
        self.print_sheet_like("BearishStocks", bearish[:20], bullish=False)

        best_df = pd.DataFrame([{"sector": sec, "now_pct": 0.0, "d5": 0.0, "d15": 0.0, "d30": 0.0} for sec in self.best_sectors])
        worst_df = pd.DataFrame([{"sector": sec, "now_pct": 0.0, "d5": 0.0, "d15": 0.0, "d30": 0.0} for sec in self.worst_sectors])
        self.print_sector_tables(best_df, worst_df)

        self.last_cycle_scores = self.current_cycle_scores
        self.current_cycle_scores = {}

    def run(self):
        self.is_running = True
        print(f"{Colors.CYAN}{Colors.BOLD}ENHANCED SECTOR SCANNER WITH OI-PCR OUTPUT GRID{Colors.RESET}")
        print(f"Sectors Best: {', '.join(self.best_sectors)} | Worst: {', '.join(self.worst_sectors)}")
        try:
            while self.is_running:
                self.enhanced_scan_cycle()
                if not self.is_running:
                    break
                import time as t
                t.sleep(self.scan_interval)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.is_running = False

# =========================
# --- MAIN ---
# =========================
if __name__ == "__main__":
    EnhancedScannerPCR().run()
