#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import time as timemodule
import logging
import warnings
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, time

import numpy as np
import pandas as pd
import pytz
import requests
from logzero import logger

warnings.filterwarnings("ignore")

# =========================
# Console colors
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

def printf(*args, **kwargs):
    print(*args, **kwargs, flush=True)

# =========================
# Daily file append helper
# =========================
def append_table_row_with_time_date():
    """
    Appends a row with current local time and date to a file named as today's date (YYYY-MM-DD.txt).
    Ensures a single header line: 'Time,Date'.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    filename = f"{date_str}.txt"

    # Write header if new or empty
    write_header = not os.path.exists(filename) or os.path.getsize(filename) == 0
    with open(filename, "a", encoding="utf-8") as f:
        if write_header:
            f.write("Time,Date\n")
        f.write(f"{time_str},{date_str}\n")

# =========================
# Config: Indicators & Weights
# =========================
ENHANCED_INDICATOR_WEIGHTS = {
    "RSI": 1.3,
    "MACD": 1.6,
    "Stochastic": 1.0,
    "MA": 1.8,
    "ADX": 1.5,
    "Bollinger": 1.4,
    "ROC": 1.2,
    "OBV": 1.6,
    "CCI": 1.1,
    "WWL": 1.0,
    "EMA": 1.7,
    "VWAP": 1.5,
    "ATR": 1.4,
    "VolumeSurge": 2.0,
    "Momentum": 1.9,
}

TIMEFRAME_WEIGHTS = {
    5: 1.0,
    15: 1.5,
    30: 2.0,
    60: 2.5,
    "daily": 3.0,
}

# =========================
# NSE Index -> Sector map
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

# =========================
# Sector -> Stocks universe (extend as needed)
# =========================
SECTOR_STOCKS = {
    "Technology": ["TCS","INFY","HCLTECH","WIPRO","TECHM","LTIM","MPHASIS","COFORGE","PERSISTENT","CYIENT","KPITTECH","TATAELXSI","SONACOMS","KAYNES","OFSS"],
    "Auto": ["MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","HEROMOTOCO","TVSMOTOR","BHARATFORG","EICHERMOT","ASHOKLEY","BOSCHLTD","TIINDIA","MOTHERSON"],
    "Banking": ["HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","PNB","BANKBARODA","CANBK","IDFCFIRSTB","INDUSINDBK","AUBANK","FEDERALBNK"],
    "Pharma": ["SUNPHARMA","DRREDDY","CIPLA","LUPIN","AUROPHARMA","TORNTPHARM","GLENMARK","ALKEM","LAURUSLABS","BIOCON","ZYDUSLIFE","MANKIND","SYNGENE","PPLPHARMA"],
    "Energy": ["RELIANCE","NTPC","BPCL","IOC","ONGC","GAIL","HINDPETRO","ADANIGREEN","ADANIENSOL","JSWENERGY","COALINDIA","TATAPOWER","SUZLON","PETRONET","OIL","POWERGRID","NHPC","ADANIPORTS","ABB","SIEMENS","CGPOWER","INOXWIND"],
    "Metal": ["TATASTEEL","JSWSTEEL","SAIL","JINDALSTEL","HINDALCO","NMDC"],
    "Consumer": ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","TATACONSUM","DABUR","AMBER","UNITDSPR","GODREJCP","MARICO","COLPAL","UPL","VBL"],
    "PSU Bank": ["SBIN","PNB","BANKBARODA","CANBK","UNIONBANK","BANKINDIA"],
    "Realty": ["DLF","LODHA","PRESTIGE","GODREJPROP","OBEROIRLTY","PHOENIXLTD","NCC","NBCC"],
    # Extend mapping for other sectors as needed
}

# =========================
# Indicator calculations
# =========================
class EnhancedTechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> dict:
        indicators = {}
        try:
            if df is None or len(df) < 20:
                return indicators

            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            vol = df["Volume"]

            # 1. RSI 14
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            indicators["RSI"] = 100 - (100 / (1 + rs))

            # 2. MACD (12,26,9)
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            indicators["MACD"] = macd_line - signal_line

            # 3. Stochastic %K
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
            indicators["Bollinger"] = 100 * ((close - ma20) / (upper - lower).replace(0, np.nan))

            # 8. ROC 12
            indicators["ROC"] = close.pct_change(periods=12) * 100

            # 9. OBV change
            obv = np.sign(close.diff().fillna(0)) * vol.fillna(0)
            obv = obv.cumsum()
            indicators["OBV"] = obv.pct_change(periods=10) * 100

            # 10. CCI 20
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(window=20).mean()
            mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
            indicators["CCI"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

            # 11. Williams %R 14
            hh = high.rolling(window=14).max()
            ll = low.rolling(window=14).min()
            indicators["WWL"] = -100 * (hh - close) / (hh - ll).replace(0, np.nan)

            # 12. VWAP (rolling 20 approximation)
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

            # 15. Momentum (price 10 + vol 10)
            if len(df) >= 10:
                price_mom = close.pct_change(periods=10) * 100
                avg10 = vol.rolling(window=10).mean()
                vol_mom = (vol / avg10.replace(0, np.nan) - 1) * 100
                mom_score = price_mom * 0.7 + vol_mom * 0.3
                indicators["Momentum"] = np.clip(50 + mom_score * 1.5, -50, 50)

            return indicators
        except Exception as e:
            logger.error(f"Error calculating enhanced indicators: {e}")
            return indicators

def normalize_indicator_value(indicator_name: str, value: float) -> float:
    try:
        if indicator_name == "RSI":
            return max(0, min(100, value))
        elif indicator_name == "MACD":
            return 50 + max(-25, min(25, value)) * 10
        elif indicator_name == "Stochastic":
            return max(0, min(100, value))
        elif indicator_name in ["MA", "EMA", "VWAP"]:
            return 50
        elif indicator_name == "ADX":
            return max(0, min(100, value))
        elif indicator_name == "Bollinger":
            return max(0, min(100, (value + 100) / 2))
        elif indicator_name == "ROC":
            return 50 + max(-25, min(25, value)) * 2
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
        return 50

# =========================
# Scanner
# =========================
class EnhancedSectorScanner:
    def __init__(self):
        self.is_running = False
        self.current_signals = []
        # Defaults, updated from API
        self.best_sectors = ["Pharma", "Healthcare", "Technology", "Financial Services 2550"]
        self.worst_sectors = ["Defence", "Energy", "PSU Bank", "Realty"]
        self.sectoral_history = []
        self.last_sectoral_update = None
        self.api_errors = []
        self.sector_update_attempts = 0
        self.successful_updates = 0
        self.gapdown_filtered_count = 0
        # Score tracking
        self.last_cycle_scores = {}  # symbol -> previous score
        self.current_cycle_scores = {}  # symbol -> current score
        # Market window (IST)
        self.market_start = time(9, 15)
        self.market_end = time(15, 30)
        self.scan_interval = 300  # seconds

        logger.info("Enhanced Sector Scanner with API Sectoral Data initialized")

    # ----------- Display helpers -----------
    def show_initialization_status(self):
        printf(f"{Colors.CYAN}{Colors.BOLD}ENHANCED 5+5 SECTOR SCANNER WITH API SECTORAL DATA{Colors.RESET}")
        printf(f"{Colors.CYAN}{'-'*80}{Colors.RESET}")
        printf(f"Timeframes: {Colors.YELLOW}5min, 15min, 30min, 60min, Daily{Colors.RESET}")
        printf(f"Strategy: {Colors.GREEN}Top 5 Best{Colors.RESET} / {Colors.RED}Top 5 Worst{Colors.RESET} sectors")
        printf(f"Filter: {Colors.MAGENTA}Gap-down exclusion{Colors.RESET}")
        printf(f"Sectoral Data: {Colors.GREEN}API http://localhost:3001/api/allIndices{Colors.RESET}")
        printf(f"{Colors.YELLOW}NEW INDICATORS{Colors.RESET}")
        printf("ATR 1.4, Volume Surge 2.0, Momentum 1.9")
        printf(f"{Colors.BLUE}ENHANCED WEIGHTS{Colors.RESET}")
        for indicator, weight in ENHANCED_INDICATOR_WEIGHTS.items():
            color = Colors.GREEN if weight >= 1.5 else (Colors.YELLOW if weight >= 1.2 else Colors.WHITE)
            printf(f" - {indicator}: {color}{weight}{Colors.RESET}")
        self.show_sector_status()
        self.test_api_connection()
        printf(f"{Colors.YELLOW}Running initial sector update...{Colors.RESET}")
        self.force_sector_update()
        printf(f"{Colors.CYAN}{'-'*80}{Colors.RESET}")

    def test_api_connection(self):
        printf(f"{Colors.BLUE}API CONNECTION TEST{Colors.RESET}")
        try:
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            if response.status_code == 200:
                printf("API Connection: SUCCESS")
                data = response.json()
                if isinstance(data, list):
                    printf(f"Items Count: {len(data)}")
                elif isinstance(data, dict):
                    printf(f"Dict Keys: {list(data.keys())}")
            else:
                printf(f"API Connection FAILED - Status {response.status_code}")
        except Exception as e:
            printf(f"API Connection ERROR - {str(e)}")

    def show_sector_status(self):
        printf(f"{Colors.MAGENTA}CURRENT SECTOR STATUS{Colors.RESET}")
        printf(f"Top 5 Best Sectors: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        printf(f"Top 5 Worst Sectors: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
        printf(f"Last Update: {self.last_sectoral_update or 'Never'}")
        printf(f"Gap-down Filtered: {self.gapdown_filtered_count}")

    # ----------- Sector update via API -----------
    def fetch_live_sectoral_performance(self) -> bool:
        try:
            logger.info("Fetching live sector performance from API...")
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            printf(f"{Colors.BLUE}API RESPONSE DEBUG{Colors.RESET}")
            printf(f"Status Code: {response.status_code}")
            if response.status_code != 200:
                printf(f"API request failed with status code {response.status_code}")
                return False

            indices_data = response.json()
            if isinstance(indices_data, str):
                indices_data = json.loads(indices_data)
            # Normalize container to list of dicts
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
                index_name = next((str(index.get(field)).strip().upper()
                                   for field in ["name","symbol","index","indexName"]
                                   if index.get(field) is not None), None)
                if not index_name:
                    continue
                if index_name not in NSE_INDEX_TO_SECTOR:
                    continue
                change_percent = 0.0
                for field in ["changepercent","changePercent","pChange","percentChange","change","pchg"]:
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

            sectoral_performance.sort(key=lambda x: x["changepercent"], reverse=True)
            old_best = self.best_sectors[:]
            old_worst = self.worst_sectors[:]

            n = len(sectoral_performance)
            best_count = min(5, n)
            worst_count = min(5, n)

            self.best_sectors = [sectoral_performance[i]["sector"] for i in range(best_count)]
            self.worst_sectors = [sectoral_performance[-i]["sector"] for i in range(1, worst_count+1)]

            self.last_sectoral_update = current_time
            self.sectoral_history.append({
                "timestamp": current_time,
                "best": self.best_sectors[:],
                "worst": self.worst_sectors[:],
                "fulldata": sectoral_performance[:],
            })
            if len(self.sectoral_history) > 50:
                self.sectoral_history = self.sectoral_history[-50:]

            self.display_sector_update(sectoral_performance, old_best, old_worst)
            return True
        except Exception as e:
            logger.error(f"Error fetching API sectoral data: {e}")
            self.api_errors.append((datetime.now(), str(e)))
            return False

    def display_sector_update(self, sectoral_performance, old_best, old_worst):
        current_time = datetime.now()
        printf(f"{Colors.MAGENTA}{Colors.BOLD}{'-'*100}")
        printf(f"SECTOR PERFORMANCE UPDATE - {current_time.strftime('%H:%M:%S')} IST")
        printf(f"{'-'*100}{Colors.RESET}")

        printf(f"Top 5 Best: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        printf(f"Top 5 Worst: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")

        topn = min(6, len(sectoral_performance))
        print("Top 6 Performing Sectors")
        for i, sectordata in enumerate(sectoral_performance[:topn]):
            color = Colors.YELLOW
            if sectordata["sector"] in self.best_sectors:
                rank = self.best_sectors.index(sectordata["sector"]) + 1
                color = Colors.GREEN + Colors.BOLD if rank == 1 else Colors.GREEN
            printf(f"{i+1}. {color}{sectordata['sector']:<20}{Colors.RESET} {sectordata['changepercent']:>6.2f}  {sectordata['index']}")

        print("Bottom 6 Performing Sectors")
        bottomslice = sectoral_performance[-topn:]
        for i, sectordata in enumerate(bottomslice):
            color = Colors.YELLOW
            if sectordata["sector"] in self.worst_sectors:
                rank = self.worst_sectors.index(sectordata["sector"]) + 1
                color = Colors.RED + Colors.BOLD if rank == 1 else Colors.RED
            pos = len(sectoral_performance) - topn + i + 1
            printf(f"{pos}. {color}{sectordata['sector']:<20}{Colors.RESET} {sectordata['changepercent']:>6.2f}  {sectordata['index']}")
        printf(f"{Colors.MAGENTA}{'-'*100}{Colors.RESET}")

    def force_sector_update(self) -> bool:
        printf(f"{Colors.YELLOW}FORCING REAL SECTOR UPDATE WITH API...{Colors.RESET}")
        self.sector_update_attempts += 1
        success = self.fetch_live_sectoral_performance()
        if success:
            self.successful_updates += 1
            print("API sectoral update successful!")
        else:
            print("API sectoral update failed - using defaults")
            printf(f"Top 5 Best: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
            printf(f"Top 5 Worst: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
        return success

    # ----------- Market window -----------
    def is_market_open(self) -> bool:
        now = datetime.now()
        ct = now.time()
        if now.weekday() >= 5:
            return False
        return self.market_start <= ct <= self.market_end

    # ----------- Data normalization -----------
    def normalize_live_data(self, df: pd.DataFrame, symbol: str):
        try:
            if df is None or len(df) == 0:
                return None
            dfc = df.copy()

            # Lowercase headers
            dfc.rename(columns={c: c.lower() for c in dfc.columns}, inplace=True)

            # Map common fields to standardized
            colmap = {}
            for src, tgt in [
                ("time", "Date"), ("timestamp", "Date"), ("date", "Date"),
                ("open","Open"), ("high","High"), ("low","Low"), ("close","Close"),
                ("vol","Volume"), ("volume","Volume")
            ]:
                if src in dfc.columns:
                    colmap[src] = tgt
            dfc.rename(columns=colmap, inplace=True)

            if "Date" not in dfc.columns:
                if isinstance(dfc.index, pd.DatetimeIndex):
                    dfc["Date"] = dfc.index
                else:
                    for cand in ["datetime","barstarttime","bartime","time"]:
                        if cand in dfc.columns:
                            dfc.rename(columns={cand: "Date"}, inplace=True)
                            break

            required = ["Open","High","Low","Close"]
            for col in required:
                if col not in dfc.columns:
                    return None
            if "Volume" not in dfc.columns:
                dfc["Volume"] = 0

            # Ensure numeric
            for col in ["Open","High","Low","Close","Volume"]:
                dfc[col] = pd.to_numeric(dfc[col], errors="coerce")

            dfc.set_index("Date", inplace=True, drop=True)
            if not isinstance(dfc.index, pd.DatetimeIndex):
                newidx = pd.to_datetime(dfc.index, errors="coerce", utc=False)
                dfc = dfc[~newidx.isna()]
                dfc.index = pd.to_datetime(dfc.index, errors="coerce", utc=False)

            dfc = dfc.sort_index()
            return dfc if len(dfc) >= 20 else None
        except Exception as e:
            logger.error(f"Normalize error {symbol}: {e}")
            return None

    def check_gap_down(self, df: pd.DataFrame) -> bool:
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
            return False

    # ----------- Data fetch stub (replace with real feeds) -----------
    def fetch_live_data(self, symbol: str, timeframe):
        """
        Replace this stub with actual data source calls.
        Must return a pandas DataFrame with columns: Date, Open, High, Low, Close, Volume.
        """
        # Example stub: return empty to skip
        return None, False

    # ----------- Signal computation -----------
    def calculate_enhanced_signals(self, symbol: str, timeframes_data: dict):
        try:
            if not timeframes_data:
                return "Neutral", 0
            sector = next((s for s, stocks in SECTOR_STOCKS.items() if symbol in stocks), None)
            if not sector:
                return "Neutral", 0

            total_weighted_score = 0.0
            total_weight = 0.0
            timeframe_scores = {}

            # Per-timeframe composite score
            for tf, df in timeframes_data.items():
                if df is None or len(df) < 20:
                    continue
                indicators = EnhancedTechnicalIndicators.calculate_all_indicators(df)
                if not indicators:
                    continue

                tf_score = 0.0
                tf_weight_sum = 0.0

                current_price = df["Close"].iloc[-1]
                for name, weight in ENHANCED_INDICATOR_WEIGHTS.items():
                    series = indicators.get(name)
                    if series is None or len(series) == 0:
                        continue
                    latest_val = series.iloc[-1]
                    if pd.isna(latest_val):
                        continue

                    if name in ["MA","EMA","VWAP"]:
                        base = latest_val
                        if pd.isna(base) or base == 0:
                            norm_score = 50
                        else:
                            pricevs = (current_price - base) / base * 100
                            if pricevs >= 2:
                                norm_score = 75
                            elif pricevs >= 0:
                                norm_score = 60
                            elif pricevs >= -2:
                                norm_score = 50
                            elif pricevs >= -5:
                                norm_score = 40
                            else:
                                norm_score = 30
                    else:
                        norm_score = normalize_indicator_value(name, latest_val)

                    tf_score += norm_score * weight
                    tf_weight_sum += weight

                if tf_weight_sum == 0:
                    continue
                tf_composite = tf_score / tf_weight_sum
                tf_multiplier = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                tffinal = tf_composite * tf_multiplier

                timeframe_scores[tf] = tffinal
                total_weighted_score += tffinal
                total_weight += tf_multiplier

            if total_weight == 0:
                return "Neutral", 0

            base_score = total_weighted_score / total_weight  # centered ~50

            # Multi-timeframe confirmation bonus
            if len(timeframe_scores) >= 4:
                bullish_count = sum(1 for v in timeframe_scores.values() if v >= 55)
                bearish_count = sum(1 for v in timeframe_scores.values() if v <= 45)
                if bullish_count >= 3:
                    base_score += 8
                elif bearish_count >= 3:
                    base_score -= 8

            # Sector boost by rank among best/worst 5
            sector_boost = 0
            has_longer_tf = ("daily" in timeframes_data) or (60 in timeframes_data)
            if sector in self.best_sectors:
                rank = self.best_sectors.index(sector) + 1
                if has_longer_tf:
                    boost_map = {1: 25, 2: 20, 3: 15, 4: 10, 5: 8}
                else:
                    boost_map = {1: 20, 2: 15, 3: 10, 4: 5, 5: 4}
                sector_boost = boost_map.get(rank, 0)
            elif sector in self.worst_sectors:
                rank = self.worst_sectors.index(sector) + 1
                if has_longer_tf:
                    boost_map = {1: -25, 2: -20, 3: -15, 4: -10, 5: -8}
                else:
                    boost_map = {1: -20, 2: -15, 3: -10, 4: -5, 5: -4}
                sector_boost = boost_map.get(rank, 0)

            final_score = base_score + sector_boost

            # Classification
            if final_score >= 82:
                signal = "Very Strong Buy"
            elif final_score >= 72:
                signal = "Strong Buy"
            elif final_score >= 60:
                signal = "Buy"
            elif final_score <= 18:
                signal = "Very Strong Sell"
            elif final_score <= 28:
                signal = "Strong Sell"
            elif final_score <= 40:
                signal = "Sell"
            else:
                signal = "Neutral"

            return signal, final_score
        except Exception as e:
            logger.error(f"Enhanced signal calculation error for {symbol}: {e}")
            return "Neutral", 0

    # ----------- Target universe from 5 best + 5 worst -----------
    def build_target_universe(self):
        target_set = set()
        # Allocate from best sectors by rank: 1->12, 2->10, 3->8, 4->6, 5->6
        for i, sector in enumerate(self.best_sectors[:5]):
            if sector in SECTOR_STOCKS:
                if i == 0:
                    target_set.update(SECTOR_STOCKS[sector][:12])
                elif i == 1:
                    target_set.update(SECTOR_STOCKS[sector][:10])
                elif i == 2:
                    target_set.update(SECTOR_STOCKS[sector][:8])
                elif i == 3:
                    target_set.update(SECTOR_STOCKS[sector][:6])
                elif i == 4:
                    target_set.update(SECTOR_STOCKS[sector][:6])

        # Allocate from worst sectors by rank: 1->12, 2->10, 3->8, 4->6, 5->6
        for i, sector in enumerate(self.worst_sectors[:5]):
            if sector in SECTOR_STOCKS:
                if i == 0:
                    target_set.update(SECTOR_STOCKS[sector][:12])
                elif i == 1:
                    target_set.update(SECTOR_STOCKS[sector][:10])
                elif i == 2:
                    target_set.update(SECTOR_STOCKS[sector][:8])
                elif i == 3:
                    target_set.update(SECTOR_STOCKS[sector][:6])
                elif i == 4:
                    target_set.update(SECTOR_STOCKS[sector][:6])

        targets = list(target_set)
        return targets

    # ----------- Per cycle scan -----------
    def enhanced_scan_cycle(self):
        if not self.is_market_open():
            logger.info("Market closed. Next scan in 5 minutes...")
            return

        start_time = timemodule.time()
        current_time = datetime.now()

        printf(f"{Colors.CYAN}Starting ENHANCED 5+5 sector scan at {current_time.strftime('%H:%M:%S')}{Colors.RESET}")
        printf("Analyzing 5min 15min 30min 60min Daily")
        printf(f"Strategy: {Colors.GREEN}Top 5 Best{Colors.RESET} / {Colors.RED}Top 5 Worst{Colors.RESET} sectors")
        printf(f"New Indicators: {Colors.MAGENTA}ATR1.4, VolumeSurge2.0, Momentum1.9{Colors.RESET}")
        printf(f"Sectoral Source: {Colors.GREEN}API localhost:3001/api/allIndices{Colors.RESET}")

        # Update sectors
        self.fetch_live_sectoral_performance()

        # Build universe
        target_stocks = self.build_target_universe()
        if not target_stocks:
            print("No target stocks found.")
            return

        printf(f"Enhanced scanning {len(target_stocks)} stocks from up to 10 sectors")
        live_signals = []
        gapdown_filtered = 0

        def process_stock(symbol: str):
            try:
                timeframes_data = {}
                has_gapdown = False
                for tf in [5, 15, 30, 60, "daily"]:
                    df_raw, is_gap = self.fetch_live_data(symbol, tf)
                    if is_gap:
                        has_gapdown = True
                    if df_raw is not None:
                        df = self.normalize_live_data(df_raw, symbol)
                        if df is not None:
                            timeframes_data[tf] = df
                    timemodule.sleep(0.8)  # throttle

                if len(timeframes_data) >= 3:
                    signal, score = self.calculate_enhanced_signals(symbol, timeframes_data)
                    tf_details = list(timeframes_data.keys())
                    return {
                        "symbol": symbol,
                        "signal": signal,
                        "score": score,
                        "sector": next((s for s, st in SECTOR_STOCKS.items() if symbol in st), "NA"),
                        "timeframes": len(timeframes_data),
                        "timestamp": datetime.now(),
                        "tfdetails": tf_details,
                        "gap": has_gapdown,
                    }, has_gapdown
                return None, has_gapdown
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
                return None, False

        try:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(process_stock, symbol) for symbol in target_stocks]
                for future in as_completed(futures):
                    result, is_gap = future.result()
                    if is_gap:
                        gapdown_filtered += 1
                    if result:
                        live_signals.append(result)
        except Exception as e:
            logger.error(f"Error in enhanced scan: {e}")

        self.gapdown_filtered_count = gapdown_filtered
        scan_time = timemodule.time() - start_time
        logger.info(f"Enhanced scan completed in {scan_time:.2f}s - {len(live_signals)} signals, {gapdown_filtered} gap-down filtered")

        self.display_enhanced_signals(live_signals, scan_time, gapdown_filtered)

    def strength_str(self, score: float, bullish_side: bool = True) -> str:
        deviation = abs(score - 50)
        if deviation >= 40:
            return f"{Colors.GREEN if bullish_side else Colors.RED}{Colors.BOLD}Exceptional{Colors.RESET}"
        elif deviation >= 30:
            return f"{Colors.GREEN if bullish_side else Colors.RED}{Colors.BOLD}Very Strong{Colors.RESET}"
        elif deviation >= 20:
            return f"{Colors.GREEN if bullish_side else Colors.RED}Strong{Colors.RESET}"
        else:
            return f"{Colors.YELLOW}Moderate{Colors.RESET}"

    def display_enhanced_signals(self, signals: list, scan_time: float, gapdown_filtered: int):
        # Append a row to the date-named file with header "Time,Date"
        append_table_row_with_time_date()

        # Track current cycle scores for delta
        self.last_cycle_scores = self.current_cycle_scores
        self.current_cycle_scores = {}

        os.system("clear" if os.name == "posix" else "cls")
        current_time = datetime.now()
        printf(f"{Colors.CYAN}{Colors.BOLD}{'-'*150}{Colors.RESET}")
        printf(f"ENHANCED 5+5 SECTOR SCANNER WITH API SECTORAL DATA - {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
        printf(f"{'-'*150}")
        printf(f"Analysis: {Colors.YELLOW}5m{Colors.RESET} {Colors.YELLOW}15m{Colors.RESET} {Colors.YELLOW}30m{Colors.RESET} {Colors.CYAN}60m{Colors.RESET} {Colors.MAGENTA}Daily{Colors.RESET}")
        beststr = ", ".join(self.best_sectors)
        worststr = ", ".join(self.worst_sectors)
        if self.last_sectoral_update:
            printf(f"{Colors.MAGENTA}API Sectoral Update{Colors.RESET}: {Colors.YELLOW}{self.last_sectoral_update.strftime('%H:%M:%S')}{Colors.RESET}")
        printf(f"Top 5 Best: {Colors.GREEN}{Colors.BOLD}{beststr}{Colors.RESET}")
        printf(f"Top 5 Worst: {Colors.RED}{Colors.BOLD}{worststr}{Colors.RESET}")
        printf(f"{Colors.BLUE}Updates{Colors.RESET}: {self.successful_updates}/{self.sector_update_attempts}  Scan Time: {scan_time:.2f}s  Gap-down Filtered: {Colors.MAGENTA}{gapdown_filtered}{Colors.RESET}")

        if not signals:
            printf(f"{Colors.YELLOW}No significant enhanced signals found in this cycle.{Colors.RESET}")
            nextscan = (datetime.now() + timedelta(minutes=5)).strftime('%H:%M:%S')
            printf(f"{Colors.CYAN}{Colors.BOLD}Next enhanced scan at {nextscan}{Colors.RESET}")
            return

        # Keep only significant
        significant = []
        for s in signals:
            score = s["score"]
            symbol = s["symbol"]
            if abs(score - 50) >= 15:
                s["tfdetails"] = s.get("tfdetails", [])
                s["timeframes"] = len(s["tfdetails"])
                sector = s.get("sector","NA")
                self.current_cycle_scores[symbol] = score
                significant.append(s)

        # Split lists
        bullish = [s for s in significant if "Buy" in s["signal"]]
        bearish = [s for s in significant if "Sell" in s["signal"]]

        # Sort and take top 20
        bullish.sort(key=lambda x: x["score"], reverse=True)
        bearish.sort(key=lambda x: x["score"])
        bullish = bullish[:20]
        bearish = bearish[:20]

        # Header
        def hdr():
            printf(f"{'Stock':<10} {'Sector':<18} {'Signal':<20} {'Score':>8} {'ΔScore':>8} {'TFs':>4} {'TF Coverage':<20} {'Strength':<15}")

        # Print Bullish
        printf(f"{Colors.GREEN}{Colors.BOLD}TOP 20 BULLISH SIGNALS (5 Best sectors focus){Colors.RESET}")
        hdr()
        printf(f"{Colors.GREEN}{'-'*150}{Colors.RESET}")
        for s in bullish:
            sector_name = s["sector"]
            sector_color = Colors.YELLOW
            stars = ""
            if sector_name in self.best_sectors:
                rank = self.best_sectors.index(sector_name) + 1
                stars = "★" * rank
                sector_color = Colors.GREEN if rank == 1 else Colors.GREEN
                sector_display = f"{stars}{sector_name}"
            else:
                sector_display = sector_name
            prev = self.last_cycle_scores.get(s["symbol"])
            delta_display = "na" if prev is None else f"{s['score'] - prev:0.1f}"
            signal_color = Colors.GREEN + Colors.BOLD if "Very" in s["signal"] else Colors.GREEN
            tfdetails = s.get("tfdetails", [])
            tfdisplay = ", ".join([str(tf) if isinstance(tf, int) else "D" for tf in tfdetails])[:20]
            strength = self.strength_str(s["score"], bullish_side=True)
            printf(f"{Colors.WHITE}{s['symbol']:<10}{Colors.RESET} "
                   f"{sector_color}{sector_display:<18}{Colors.RESET} "
                   f"{signal_color}{s['signal']:<20}{Colors.RESET} "
                   f"{Colors.WHITE}{s['score']:>8.1f}{Colors.RESET} "
                   f"{Colors.CYAN}{delta_display:>8}{Colors.RESET} "
                   f"{Colors.CYAN}{s['timeframes']:>4}{Colors.RESET} "
                   f"{Colors.MAGENTA}{tfdisplay:<20}{Colors.RESET} "
                   f"{strength:<15}")

        # Print Bearish
        printf(f"{Colors.RED}{Colors.BOLD}TOP 20 BEARISH SIGNALS (5 Worst sectors focus){Colors.RESET}")
        hdr()
        printf(f"{Colors.RED}{'-'*150}{Colors.RESET}")
        for s in bearish:
            sector_name = s["sector"]
            sector_color = Colors.YELLOW
            stars = ""
            if sector_name in self.worst_sectors:
                rank = self.worst_sectors.index(sector_name) + 1
                stars = "★" * rank
                sector_color = Colors.RED if rank == 1 else Colors.RED
                sector_display = f"{stars}{sector_name}"
            else:
                sector_display = sector_name
            prev = self.last_cycle_scores.get(s["symbol"])
            delta_display = "na" if prev is None else f"{s['score'] - prev:0.1f}"
            signal_color = Colors.RED + Colors.BOLD if "Very" in s["signal"] else Colors.RED
            tfdetails = s.get("tfdetails", [])
            tfdisplay = ", ".join([str(tf) if isinstance(tf, int) else "D" for tf in tfdetails])[:20]
            strength = self.strength_str(s["score"], bullish_side=False)
            printf(f"{Colors.WHITE}{s['symbol']:<10}{Colors.RESET} "
                   f"{sector_color}{sector_display:<18}{Colors.RESET} "
                   f"{signal_color}{s['signal']:<20}{Colors.RESET} "
                   f"{Colors.WHITE}{s['score']:>8.1f}{Colors.RESET} "
                   f"{Colors.CYAN}{delta_display:>8}{Colors.RESET} "
                   f"{Colors.CYAN}{s['timeframes']:>4}{Colors.RESET} "
                   f"{Colors.MAGENTA}{tfdisplay:<20}{Colors.RESET} "
                   f"{strength:<15}")

        next_scan_time = (datetime.now() + timedelta(minutes=5)).strftime('%H:%M:%S')
        printf(f"{Colors.CYAN}{Colors.BOLD}Next enhanced scan at {next_scan_time}{Colors.RESET}")

    # ----------- Runner -----------
    def run(self):
        self.is_running = True
        self.show_initialization_status()
        try:
            while self.is_running:
                self.enhanced_scan_cycle()
                timemodule.sleep(self.scan_interval)
        except KeyboardInterrupt:
            printf(f"{Colors.YELLOW}Shutting down enhanced scanner...{Colors.RESET}")
        finally:
            self.stop()

    def stop(self):
        self.is_running = False

def main():
    printf(f"{Colors.CYAN}{Colors.BOLD}ENHANCED 5+5 SECTOR SCANNER WITH API SECTORAL DATA{Colors.RESET}")
    printf(f"{Colors.YELLOW}Timeframes: 5min, 15min, 30min, 60min, Daily EOD{Colors.RESET}")
    printf(f"{Colors.CYAN}Features: 5 Best + 5 Worst sectors, Enhanced indicators, API sectoral data{Colors.RESET}")
    printf(f"{Colors.MAGENTA}NEW: ATR 1.4, Volume Surge 2.0, Momentum 1.9{Colors.RESET}")
    printf(f"{Colors.GREEN}LIVE API sectoral updates from http://localhost:3001/api/allIndices{Colors.RESET}")
    printf(f"{Colors.BLUE}Updates every 5 minutes with REAL sectoral performance{Colors.RESET}")

    scanner = EnhancedSectorScanner()

    # IMPORTANT: Plug in actual data feed by overriding fetch_live_data
    # Example:
    # scanner.fetch_live_data = your_fetch_function

    scanner.run()

if __name__ == "__main__":
    main()
