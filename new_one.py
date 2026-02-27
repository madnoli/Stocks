import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import requests
import json
import time as timemodule
import pytz
import logging
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from truedata.history import TD_hist
import warnings
warnings.filterwarnings("ignore")
from rich.console import Console
from rich.table import Table
import argparse

logger = logging.getLogger(__name__)

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
# --- ENHANCED INDICATOR WEIGHTS FOR OPTION BUYERS ---
# =========================
ENHANCED_INDICATOR_WEIGHTS = {
    # Tier 1: Critical for Option Buyers (Highest Priority)
    "VolumeOIFlow": 2.5,      # Volume-OI divergence analysis
    "InstitutionalFlow": 2.3,  # Large volume + OI changes
    "VolumeSurge": 2.2,       # Volume surge detection
    "OIChangeRate": 2.1,      # Open Interest momentum
    "VolumeBreakout": 2.0,    # Volume confirmation on breakouts
    
    # Tier 2: Strong momentum indicators
    "Momentum": 1.9,          # Volume-price momentum
    "ADX": 1.8,               # Trend strength
    "VWAP": 1.7,              # Volume-weighted levels
    "EMA": 1.7,               # Price momentum
    
    # Tier 3: Supporting confirmation
    "MACD": 1.5,
    "OBV": 1.5,               # With OI correlation
    "ATR": 1.4,               # Volatility for option premium
    "VolumeProfile": 1.3,     # Volume distribution
    
    # Tier 4: Traditional indicators
    "Bollinger": 1.2,
    "RSI": 1.1,
    "ROC": 1.0,
    "Stochastic": 1.0,
    "CCI": 1.0,
    "MA": 1.0,
    "WWL": 1.0,
}

# =========================
# --- TIMEFRAME WEIGHTS ---
# =========================
TIMEFRAME_WEIGHTS = {
    5: 3.0,     # Highest priority for intraday entry/exit
    15: 2.8,    # Primary trend confirmation
    30: 2.2,    # Supporting trend
    60: 1.8,    # Medium-term view
    "daily": 1.5,  # Overall context
}

# =========================
# --- 200 FNO STOCKS LIST ---
# =========================
FNO_STOCKS_200 = [
    "RELIANCE-I", "TCS-I", "HDFCBANK-I", "INFY-I", "ICICIBANK-I", "HINDUNILVR-I", 
    "ITC-I", "SBIN-I", "BHARTIARTL-I", "KOTAKBANK-I", "LT-I", "AXISBANK-I",
    "BAJFINANCE-I", "ASIANPAINT-I", "MARUTI-I", "HCLTECH-I", "SUNPHARMA-I",
    "TITAN-I", "ULTRACEMCO-I", "NESTLEIND-I", "WIPRO-I", "TECHM-I", "POWERGRID-I",
    "NTPC-I", "BAJAJFINSV-I", "ONGC-I", "TATAMOTORS-I", "TATASTEEL-I", "ADANIENT-I",
    "ADANIPORTS-I", "COALINDIA-I", "JSWSTEEL-I", "INDUSINDBK-I", "HINDALCO-I",
    "GRASIM-I", "M&M-I", "DIVISLAB-I", "CIPLA-I", "BRITANNIA-I", "DRREDDY-I",
    "EICHERMOT-I", "APOLLOHOSP-I", "BAJAJ-AUTO-I", "HEROMOTOCO-I", "SHRIRAMFIN-I",
    "SBILIFE-I", "HDFCLIFE-I", "PIDILITIND-I", "HAVELLS-I", "GODREJCP-I",
    "DABUR-I", "MARICO-I", "MCDOWELL-N-I", "AMBUJACEM-I", "SIEMENS-I", "DLF-I",
    "LTIM-I", "TRENT-I", "BAJAJHLDNG-I", "BERGEPAINT-I", "COLPAL-I", "TATACONSUM-I",
    "VEDL-I", "GAIL-I", "IOC-I", "BPCL-I", "BEL-I", "ADANIGREEN-I", "TVSMOTOR-I",
    "LUPIN-I", "ZOMATO-I", "NYKAA-I", "PAYTM-I", "PNB-I", "BANKBARODA-I",
    "CANBK-I", "INDIGO-I", "DMART-I", "CHOLAFIN-I", "MUTHOOTFIN-I", "LICHSGFIN-I",
    "MOTHERSON-I", "BOSCHLTD-I", "TORNTPHARM-I", "AUROPHARMA-I", "ALKEM-I",
    "BIOCON-I", "LAURUSLABS-I", "GLENMARK-I", "MANKIND-I", "ZYDUSLIFE-I",
    "DIXON-I", "VOLTAS-I", "CROMPTON-I", "POLYCAB-I", "AMBER-I", "KAYNES-I",
    "PERSISTENT-I", "COFORGE-I", "MPHASIS-I", "LTTS-I", "INOXWIND-I", "SUZLON-I",
    "TATAPOWER-I", "ADANIENSOL-I", "JSWENERGY-I", "NHPC-I", "RECLTD-I", "PFC-I",
    "IRCTC-I", "IRFC-I", "RVNL-I", "HAL-I", "BHEL-I", "SAIL-I", "NMDC-I",
    "JINDALSTEL-I", "HINDZINC-I", "NATIONALUM-I", "APLAPOLLO-I", "PIIND-I",
    "SRF-I", "ATUL-I", "UPL-I", "DALBHARAT-I", "SHREECEM-I", "RAMCOCEM-I",
    "JKCEMENT-I", "INDIACEM-I", "NESTLEIND-I", "BRITANNIA-I", "JUBLFOOD-I",
    "VBL-I", "TATAELXSI-I", "SONACOMS-I", "EXIDEIND-I", "ASHOKLEY-I", "ESCORTS-I",
    "TIINDIA-I", "BHARATFORG-I", "SOLARINDS-I", "KPITTECH-I", "CYIENT-I",
    "UNIONBANK-I", "BANKINDIA-I", "IDFCFIRSTB-I", "FEDERALBNK-I", "AUBANK-I",
    "BANDHANBNK-I", "RBLBANK-I", "YESBANK-I", "IDFC-I", "M&MFIN-I", "PEL-I",
    "CUMMINSIND-I", "ABB-I", "THERMAX-I", "SCHNEIDER-I", "HONAUT-I", "ENDURANCE-I",
    "FORTIS-I", "MAXHEALTH-I", "SYNGENE-I", "GRANULES-I", "LALPATHLAB-I",
    "METROPOLIS-I", "THYROCARE-I", "CONCOR-I", "AEGISCHEM-I", "AARTIIND-I",
    "NAVINFLUOR-I", "DEEPAKNTR-I", "GNFC-I", "CHAMBLFERT-I", "COROMANDEL-I",
    "BALRAMCHIN-I", "HINDALCO-I", "JSW-I", "JSWENERGY-I", "APOLLOTYRE-I",
    "MRF-I", "CEAT-I", "BALKRISIND-I", "JK TYRE-I", "CGPOWER-I", "KEI-I",
    "AIAENG-I", "INDHOTEL-I", "LEMONTREE-I", "EIH-I", "JUBILANT-I", "PAGEIND-I",
    "PETRONET-I", "GUJGASLTD-I", "IGL-I", "MGL-I", "ATGL-I"
]

# =========================
# --- NSE SECTOR MAPPING ---
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
    "NIFTY FINANCIAL SERVICES": "Finance",
}

SECTOR_STOCKS = {
    "Technology": ["TCS-I", "INFY-I", "HCLTECH-I", "WIPRO-I", "TECHM-I", "LTIM-I", "MPHASIS-I", "COFORGE-I", "PERSISTENT-I"],
    "Auto": ["MARUTI-I", "TATAMOTORS-I", "M&M-I", "BAJAJ-AUTO-I", "HEROMOTOCO-I", "TVSMOTOR-I", "EICHERMOT-I"],
    "Banking": ["HDFCBANK-I", "ICICIBANK-I", "SBIN-I", "KOTAKBANK-I", "AXISBANK-I", "PNB-I", "BANKBARODA-I"],
    "Pharma": ["SUNPHARMA-I", "DRREDDY-I", "CIPLA-I", "LUPIN-I", "AUROPHARMA-I", "TORNTPHARM-I", "BIOCON-I"],
    "Energy": ["RELIANCE-I", "NTPC-I", "BPCL-I", "IOC-I", "ONGC-I", "GAIL-I", "TATAPOWER-I"],
    "Metal": ["TATASTEEL-I", "JSWSTEEL-I", "SAIL-I", "JINDALSTEL-I", "HINDALCO-I", "NMDC-I"],
    "Consumer": ["HINDUNILVR-I", "ITC-I", "NESTLEIND-I", "BRITANNIA-I", "TATACONSUM-I", "DABUR-I"],
    "PSU Bank": ["SBIN-I", "PNB-I", "BANKBARODA-I", "CANBK-I", "UNIONBANK-I"],
    "Finance": ["BAJFINANCE-I", "SHRIRAMFIN-I", "CHOLAFIN-I", "HDFCLIFE-I"],
    "Realty": ["DLF-I", "PRESTIGE-I", "GODREJPROP-I", "OBEROIRLTY-I"],
    "PSE": ["BEL-I", "BHEL-I", "HAL-I", "RVNL-I", "IRCTC-I"],
    "Commodities": ["AMBUJACEM-I", "ULTRACEMCO-I", "SHREECEM-I"],
    "Infrastructure": ["LT-I", "POWERGRID-I", "ADANIPORTS-I"],
}

# =========================
# --- TECHNICAL INDICATORS ---
# =========================
class EnhancedOptionBuyerIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        """Calculate all indicators including volume and open interest based ones"""
        indicators = {}
        if df is None or len(df) < 20:
            return indicators
        
        try:
            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            vol = df["Volume"]
            
            # Get Open Interest if available
            oi = df.get("OpenInterest", pd.Series([0] * len(df), index=df.index))
            
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

            # 6. ADX
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

            # 8. ROC
            indicators["ROC"] = close.pct_change(periods=12) * 100

            # 9. OBV
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

            # 12. VWAP
            if len(df) >= 20:
                tpv = (high + low + close) / 3
                vwap_num = (tpv * vol).rolling(window=20).sum()
                vwap_den = vol.rolling(window=20).sum().replace(0, np.nan)
                indicators["VWAP"] = vwap_num / vwap_den

            # 13. ATR
            indicators["ATR"] = atr

            # ===== NEW INDICATORS FOR OPTION BUYERS =====
            
            # 14. Volume Surge (Z-score based)
            if len(df) >= 20:
                avg_vol_20 = vol.rolling(window=20).mean()
                vol_std = vol.rolling(window=20).std()
                current_vol = vol
                vol_zscore = (current_vol - avg_vol_20) / vol_std.replace(0, np.nan)
                indicators["VolumeSurge"] = np.clip(50 + vol_zscore * 15, 0, 100)

            # 15. Open Interest Change Rate
            if oi.sum() > 0:
                oi_change = oi.pct_change(periods=1) * 100
                oi_momentum = oi.pct_change(periods=5) * 100
                indicators["OIChangeRate"] = np.clip(50 + (oi_change * 0.3 + oi_momentum * 0.7) * 2, 0, 100)
            else:
                indicators["OIChangeRate"] = pd.Series([50] * len(df), index=df.index)

            # 16. Volume-OI Flow Analysis
            if oi.sum() > 0:
                vol_trend = vol.rolling(window=10).mean()
                oi_trend = oi.rolling(window=10).mean()
                vol_direction = np.where(vol > vol_trend, 1, -1)
                oi_direction = np.where(oi > oi_trend, 1, -1)
                flow_score = (vol_direction + oi_direction) / 2 * 50 + 50
                indicators["VolumeOIFlow"] = pd.Series(flow_score, index=df.index)
            else:
                indicators["VolumeOIFlow"] = pd.Series([50] * len(df), index=df.index)

            # 17. Institutional Activity Score
            if len(df) >= 20:
                price_change = close.pct_change() * 100
                vol_percentile = vol.rolling(window=20).rank(pct=True) * 100
                institutional_score = np.where(
                    (vol_percentile > 80) & (abs(price_change) > 1.5),
                    75 + (vol_percentile - 80) * 1.25,
                    50 + (vol_percentile - 50) * 0.3
                )
                indicators["InstitutionalFlow"] = pd.Series(np.clip(institutional_score, 0, 100), index=df.index)

            # 18. Volume Profile Analysis
            if len(df) >= 20:
                current_price_level = close.iloc[-1]
                recent_high = high.rolling(window=10).max().iloc[-1]
                recent_low = low.rolling(window=10).min().iloc[-1]
                if recent_high > recent_low:
                    price_position = (current_price_level - recent_low) / (recent_high - recent_low)
                    volume_profile_score = 50 + (price_position - 0.5) * 100
                else:
                    volume_profile_score = 50
                indicators["VolumeProfile"] = pd.Series([np.clip(volume_profile_score, 0, 100)] * len(df), index=df.index)

            # 19. Volume Breakout Confirmation
            if len(df) >= 20:
                price_ma = close.rolling(window=20).mean()
                vol_ma = vol.rolling(window=20).mean()
                price_breakout = (close - price_ma) / price_ma * 100
                volume_confirmation = vol / vol_ma
                breakout_score = np.where(
                    abs(price_breakout) > 2,
                    50 + price_breakout * volume_confirmation * 5,
                    50 + price_breakout * 10
                )
                indicators["VolumeBreakout"] = pd.Series(np.clip(breakout_score, 0, 100), index=df.index)

            # 20. Enhanced Momentum (Price + Volume + OI)
            if len(df) >= 10:
                price_mom = close.pct_change(periods=10) * 100
                vol_mom = (vol / vol.rolling(window=10).mean() - 1) * 100
                if oi.sum() > 0:
                    oi_mom = (oi / oi.rolling(window=10).mean() - 1) * 100
                    combined_momentum = price_mom * 0.5 + vol_mom * 0.3 + oi_mom * 0.2
                else:
                    combined_momentum = price_mom * 0.7 + vol_mom * 0.3
                indicators["Momentum"] = pd.Series(np.clip(50 + combined_momentum * 1.2, 0, 100), index=df.index)

            return indicators
        
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return indicators

# =========================
# --- NORMALIZATION ---
# =========================
def normalize_indicator_value(indicator_name, value):
    """Normalize indicator values to 0-100 scale"""
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
        elif indicator_name in ("VolumeSurge", "OIChangeRate", "VolumeOIFlow", 
                               "InstitutionalFlow", "VolumeProfile", "VolumeBreakout", "Momentum"):
            return max(0, min(100, value))
        else:
            return 50
    except Exception:
        return 50

# =========================
# --- SCANNER CLASS ---
# =========================
class IntradayOptionScanner:
    def __init__(self):
        self.is_running = False
        self.current_signals = []
        self.best_sectors = ["Technology", "Pharma", "Banking", "Finance"]
        self.worst_sectors = ["PSE", "Metal", "Realty", "PSU Bank"]
        self.sectoral_history = []
        self.last_sectoral_update = None
        self.last_cycle_scores = {}
        self.current_cycle_scores = {}
        
        # Market hours
        self.market_start = time(9, 15)
        self.market_end = time(15, 30)
        self.scan_interval = 300  # 5 minutes

        logger.info("Intraday Option Scanner initialized")

    def show_initialization_status(self):
        print(f"{Colors.CYAN}{Colors.BOLD}INTRADAY OPTION BUYER SCANNER - 200 STOCKS{Colors.RESET}")
        print(f"{Colors.CYAN}{'-'*80}{Colors.RESET}")
        print(f"Scan Interval: {Colors.YELLOW}5 Minutes{Colors.RESET}")
        print(f"Timeframes: {Colors.YELLOW}5min, 15min, 30min, 60min, Daily{Colors.RESET}")
        print(f"Strategy: {Colors.GREEN}Volume+OI Flow Analysis{Colors.RESET}")
        print(f"Target: {Colors.BLUE}200 FNO Stocks{Colors.RESET}")
        print(f"{Colors.CYAN}{'-'*80}{Colors.RESET}")

    def fetch_live_sectoral_performance(self):
        """Fetch real-time sector performance"""
        try:
            response = requests.get("http://localhost:3000/api/allIndices", timeout=10)
            if response.status_code != 200:
                return False

            indices_data = response.json()
            if isinstance(indices_data, dict) and "data" in indices_data:
                indices_data = indices_data["data"]

            sectoral_performance = []
            for index in indices_data:
                if not isinstance(index, dict):
                    continue
                    
                index_name = next((str(index[field]).strip().upper()
                                 for field in ("name", "symbol", "index")
                                 if field in index and index[field]), None)
                
                if index_name and index_name in NSE_INDEX_TO_SECTOR:
                    change_percent = 0.0
                    for field in ("changepercent", "pChange", "change"):
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
                        "timestamp": datetime.now(),
                    })

            if not sectoral_performance:
                return False

            sectoral_performance.sort(key=lambda x: x["changepercent"], reverse=True)
            
            self.best_sectors = [sectoral_performance[i]["sector"] for i in range(min(4, len(sectoral_performance)))]
            self.worst_sectors = [sectoral_performance[-i]["sector"] for i in range(1, min(5, len(sectoral_performance) + 1))]
            
            self.last_sectoral_update = datetime.now()
            self.sectoral_history.append({
                "timestamp": datetime.now(),
                "best": self.best_sectors[:],
                "worst": self.worst_sectors[:],
                "fulldata": sectoral_performance[:],
            })
            
            return True
        except Exception as e:
            logger.error(f"Error fetching sector data: {e}")
            return False

    def is_market_open(self):
        """Check if market is open"""
        now = datetime.now()
        if now.weekday() >= 5:  # Weekend
            return False
        return self.market_start <= now.time() <= self.market_end

    def normalize_live_data(self, df, symbol):
        """Normalize TrueData response to standard OHLCV format"""
        try:
            if df is None or len(df) == 0:
                return None
            
            dfc = df.copy()
            dfc.rename(columns={c: c.lower() for c in dfc.columns}, inplace=True)

            col_map = {}
            for src, tgt in (
                ("time", "Date"), ("timestamp", "Date"), ("date", "Date"),
                ("open", "Open"), ("high", "High"), ("low", "Low"),
                ("close", "Close"), ("vol", "Volume"), ("volume", "Volume"),
                ("oi", "OpenInterest"), ("openinterest", "OpenInterest")
            ):
                if src in dfc.columns:
                    col_map[src] = tgt
            dfc.rename(columns=col_map, inplace=True)

            if "Date" not in dfc.columns and isinstance(dfc.index, pd.DatetimeIndex):
                dfc["Date"] = dfc.index

            required = ["Open", "High", "Low", "Close"]
            if not all(col in dfc.columns for col in required):
                return None

            if "Volume" not in dfc.columns:
                dfc["Volume"] = 0
            if "OpenInterest" not in dfc.columns:
                dfc["OpenInterest"] = 0

            if "Date" in dfc.columns:
                dfc["Date"] = pd.to_datetime(dfc["Date"], errors="coerce", utc=False)
            else:
                dfc["Date"] = pd.to_datetime(dfc.index, errors="coerce", utc=False)

            dfc = dfc.dropna(subset=["Date", "Open", "High", "Low", "Close"])

            for col in ["Open", "High", "Low", "Close", "Volume", "OpenInterest"]:
                if col in dfc.columns:
                    dfc[col] = pd.to_numeric(dfc[col], errors="coerce")
            dfc = dfc.dropna(subset=["Open", "High", "Low", "Close"])

            if pd.api.types.is_datetime64tz_dtype(dfc["Date"]):
                dfc["Date"] = dfc["Date"].dt.tz_convert(None)

            dfc.set_index("Date", inplace=True, drop=True)
            dfc = dfc.sort_index()

            return dfc if len(dfc) >= 20 else None
        except Exception as e:
            logger.error(f"Normalize error {symbol}: {e}")
            return None

    def fetch_live_data(self, symbol, timeframe):
        """Fetch historical data for given symbol and timeframe"""
        try:
            tfmap = {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 mins", "daily": "EOD"}
            bar_size = tfmap.get(timeframe)
            if not bar_size:
                return None

            tf_days_map = {5: 10, 15: 10, 30: 20, 60: 60, "daily": 365}
            days = tf_days_map.get(timeframe, 10)

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
                return None

            normalized_df = self.normalize_live_data(rawdf, symbol)
            if normalized_df is None or len(normalized_df) < 20:
                return None

            if timeframe == "daily":
                return normalized_df.tail(250)
            elif timeframe == 60:
                return normalized_df.tail(200)
            else:
                return normalized_df.tail(100)
        except Exception as e:
            logger.error(f"Data fetch error {symbol}@{timeframe}: {e}")
            return None

    def calculate_option_signals(self, symbol, timeframes_data):
        """Calculate buy/sell signals for option trading"""
        try:
            if not timeframes_data:
                return "Neutral", 0

            sector = next((s for s, stocks in SECTOR_STOCKS.items() if symbol in stocks), "Other")

            total_weighted_score = 0.0
            total_weight = 0.0
            timeframe_scores = {}

            for tf, df in timeframes_data.items():
                if df is None or len(df) < 20:
                    continue

                indicators = EnhancedOptionBuyerIndicators.calculate_all_indicators(df)
                if not indicators:
                    continue

                tf_score = 0.0
                tf_weight = 0.0
                current_price = df["Close"].iloc[-1]

                for name, weight in ENHANCED_INDICATOR_WEIGHTS.items():
                    if name in indicators and indicators[name] is not None:
                        indicator_data = indicators[name]
                        
                        has_data = False
                        if hasattr(indicator_data, 'empty'):
                            has_data = not indicator_data.empty
                        elif hasattr(indicator_data, 'size'):
                            has_data = indicator_data.size > 0
                        else:
                            has_data = indicator_data is not None
                            
                        if has_data:
                            try:
                                if hasattr(indicator_data, 'iloc'):
                                    latest_val = indicator_data.iloc[-1]
                                elif hasattr(indicator_data, '__getitem__'):
                                    latest_val = indicator_data[-1]
                                else:
                                    latest_val = float(indicator_data)
                            except (IndexError, TypeError, ValueError):
                                continue
                                
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

            # Multi-timeframe confirmation
            num_timeframes = len(timeframe_scores)
            if num_timeframes >= 4:
                bullish_count = sum(1 for v in timeframe_scores.values() if v >= 55)
                bearish_count = sum(1 for v in timeframe_scores.values() if v <= 45)
                if bullish_count >= 3:
                    base_score += 12
                elif bearish_count >= 3:
                    base_score -= 12

            # Sector boost
            sector_boost = 0
            has_longer_tf = ("daily" in timeframes_data) or (60 in timeframes_data)

            if sector in self.best_sectors:
                rank = self.best_sectors.index(sector) + 1
                boost_map = {1: 30, 2: 25, 3: 20, 4: 15} if has_longer_tf else {1: 25, 2: 20, 3: 15, 4: 10}
                sector_boost = boost_map.get(rank, 0)
            elif sector in self.worst_sectors:
                rank = self.worst_sectors.index(sector) + 1
                boost_map = {1: -30, 2: -25, 3: -20, 4: -15} if has_longer_tf else {1: -25, 2: -20, 3: -15, 4: -10}
                sector_boost = boost_map.get(rank, 0)

            base_score += sector_boost

            # Signal classification
            if base_score >= 85:
                return "Strong Call Buy", base_score
            elif base_score >= 75:
                return "Call Buy", base_score
            elif base_score >= 60:
                return "Moderate Call", base_score
            elif base_score <= 15:
                return "Strong Put Buy", base_score
            elif base_score <= 25:
                return "Put Buy", base_score
            elif base_score <= 40:
                return "Moderate Put", base_score
            else:
                return "Neutral", base_score

        except Exception as e:
            logger.error(f"Signal calculation error for {symbol}: {e}")
            return "Neutral", 0

    def scan_cycle(self):
        """Main scanning cycle every 5 minutes"""
        if not self.is_market_open():
            logger.info("Market closed. Waiting...")
            return

        start_time = timemodule.time()
        current_time = datetime.now()
        print(f"\n{Colors.CYAN}Starting scan at {current_time.strftime('%H:%M:%S')}{Colors.RESET}")
        print(f"Analyzing 200 stocks across 5 timeframes with Volume+OI analysis")

        # Update sectors
        self.fetch_live_sectoral_performance()

        live_signals = []

        def process_stock(symbol):
            try:
                timeframes_data = {}
                timeframes_to_fetch = [5, 15, 30, 60, "daily"]
                
                for tf in timeframes_to_fetch:
                    df = self.fetch_live_data(symbol, tf)
                    if df is not None:
                        timeframes_data[tf] = df
                    timemodule.sleep(0.5)  # Rate limiting

                if len(timeframes_data) >= 3:
                    signal, score = self.calculate_option_signals(symbol, timeframes_data)
                    
                    # Filter: only significant signals
                    if abs(score - 50) >= 15:  # Stronger threshold
                        sector = next((s for s, st in SECTOR_STOCKS.items() if symbol in st), "Other")

                        # Extract volume metrics
                        small_df = timeframes_data.get(5) or timeframes_data.get(15) or list(timeframes_data.values())[0]

                        current_vol = small_df["Volume"].iloc[-1] if "Volume" in small_df.columns else 0
                        
                        # Calculate VolumeX
                        if len(small_df) >= 20:
                            sma20_vol = small_df["Volume"].rolling(window=20).mean().iloc[-1]
                            volume_x = current_vol / sma20_vol if sma20_vol > 0 else 0.0
                        else:
                            volume_x = 0.0

                        # Only include if VolumeX >= 2.0 (strong volume surge)
                        if volume_x >= 2.0:
                            result = {
                                "symbol": symbol,
                                "signal": signal,
                                "score": score,
                                "sector": sector,
                                "timeframes": len(timeframes_data),
                                "timestamp": current_time,
                                "volume_x": volume_x,
                                "ltp": small_df["Close"].iloc[-1]
                            }
                            self.current_cycle_scores[symbol] = score
                            return result
                return None
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                return None

        try:
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(process_stock, symbol) for symbol in FNO_STOCKS_200]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        live_signals.append(result)
                        
            scan_time = timemodule.time() - start_time
            logger.info(f"Scan completed in {scan_time:.2f}s - {len(live_signals)} signals")
            self.display_signals(live_signals, scan_time, current_time)
        except Exception as e:
            logger.error(f"Error in scan cycle: {e}")

    def display_signals(self, signals, scan_time, current_time):
        """Display trading signals in formatted tables"""
        console = Console()
        console.print(f"[cyan bold]{'-'*120}[/]")
        console.print(f"INTRADAY OPTION SCANNER - {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
        console.print(f"{'-'*120}")
        console.print(f"Scan Time: {scan_time:.2f}s | Signals: {len(signals)}")
        
        best_str = ", ".join(self.best_sectors)
        worst_str = ", ".join(self.worst_sectors)
        console.print(f"Call Focus: [green bold]{best_str}[/]")
        console.print(f"Put Focus: [red bold]{worst_str}[/]")

        if not signals:
            console.print(f"[yellow]No significant opportunities found.[/]")
        else:
            # Separate call and put
            call_signals = [s for s in signals if "Call" in s["signal"]]
            put_signals = [s for s in signals if "Put" in s["signal"]]
            
            call_signals.sort(key=lambda x: x["score"], reverse=True)
            put_signals.sort(key=lambda x: x["score"])

            # Call table
            call_table = Table(title="🔥 CALL BUYING OPPORTUNITIES", title_style="bold green")
            call_table.add_column("Stock", style="white")
            call_table.add_column("LTP", justify="right", style="cyan")
            call_table.add_column("Sector", style="yellow")
            call_table.add_column("Signal", style="green")
            call_table.add_column("Score", justify="right", style="white")
            call_table.add_column("VolumeX", justify="right", style="blue")
            call_table.add_column("TFs", justify="right", style="cyan")

            for s in call_signals[:20]:  # Top 20
                signal_style = "green bold" if "Strong" in s["signal"] else "green"
                vol_x_style = "green bold" if s['volume_x'] >= 3 else "white"
                
                call_table.add_row(
                    s['symbol'],
                    f"{s['ltp']:.2f}",
                    s['sector'],
                    f"[{signal_style}]{s['signal']}[/]",
                    f"{s['score']:.1f}",
                    f"[{vol_x_style}]{s['volume_x']:.1f}x[/]",
                    str(s['timeframes'])
                )

            console.print(call_table)

            # Put table
            put_table = Table(title="🔻 PUT BUYING OPPORTUNITIES", title_style="bold red")
            put_table.add_column("Stock", style="white")
            put_table.add_column("LTP", justify="right", style="cyan")
            put_table.add_column("Sector", style="yellow")
            put_table.add_column("Signal", style="red")
            put_table.add_column("Score", justify="right", style="white")
            put_table.add_column("VolumeX", justify="right", style="blue")
            put_table.add_column("TFs", justify="right", style="cyan")

            for s in put_signals[:20]:  # Top 20
                signal_style = "red bold" if "Strong" in s["signal"] else "red"
                vol_x_style = "green bold" if s['volume_x'] >= 3 else "white"
                
                put_table.add_row(
                    s['symbol'],
                    f"{s['ltp']:.2f}",
                    s['sector'],
                    f"[{signal_style}]{s['signal']}[/]",
                    f"{s['score']:.1f}",
                    f"[{vol_x_style}]{s['volume_x']:.1f}x[/]",
                    str(s['timeframes'])
                )

            console.print(put_table)

        next_scan_time = (current_time + timedelta(minutes=5)).strftime("%H:%M:%S")
        console.print(f"[cyan bold]Next scan at {next_scan_time}[/]")

        # Rotate scores
        self.last_cycle_scores = self.current_cycle_scores
        self.current_cycle_scores = {}

    def run(self):
        """Main run loop"""
        self.is_running = True
        logger.info("Starting Intraday Option Scanner...")
        self.show_initialization_status()
        try:
            while self.is_running:
                self.scan_cycle()
                if self.is_running:
                    logger.info("Waiting 5 minutes for next scan...")
                    timemodule.sleep(self.scan_interval)
        except KeyboardInterrupt:
            logger.info("Scanner stopped by user")
        finally:
            self.stop()

    def stop(self):
        self.is_running = False
        print(f"{Colors.YELLOW}Scanner stopped{Colors.RESET}")

# =========================
# --- MAIN EXECUTION ---
# =========================
def main():
    print(f"{Colors.CYAN}{Colors.BOLD}INTRADAY OPTION SCANNER - 200 FNO STOCKS{Colors.RESET}")
    print(f"{Colors.YELLOW}🎯 Specialized for Option Buyers - Call & Put Side{Colors.RESET}")
    print(f"{Colors.GREEN}📊 Volume+OI Flow | 5-Min Scanning | Multi-Timeframe{Colors.RESET}")
    
    scanner = IntradayOptionScanner()
    try:
        scanner.run()
    except KeyboardInterrupt:
        print(f"{Colors.YELLOW}Shutting down...{Colors.RESET}")
        scanner.stop()

if __name__ == "__main__":
    main()
