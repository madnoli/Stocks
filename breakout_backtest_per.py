import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import requests
import json
import time as time_module
import pytz
from logzero import logger
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from truedata.history import TD_hist
import logging
import warnings
warnings.filterwarnings('ignore')

# ---------------- BACKTEST CONFIG ----------------
BACKTEST_MODE = True
# Set to yesterday or today as needed
BACKTEST_DATE = datetime(2025, 9, 16).date()
# Limit symbols for speed when testing
REPLAY_SYMBOLS = [
      "CHOLAFIN", "GMRAIRPORT", "CYIENT", "HFCL", "AMBER", "KOTAKBANK", "PERSISTENT", "NHPC",
    "LT", "PAGEIND", "M&M", "RVNL", "SUPREMEIND", "BHARATFORG", "TATAPOWER", "KEI",
    "MARUTI", "POLYCAB", "PRESTIGE", "MOTHERSON", "OFSS", "NCC", "EICHERMOT", "BLUESTARCO",
    "BHARTIARTL", "PHOENIXLTD", "NBCC", "MUTHOOTFIN", "LTF", "MANAPPURAM", "TATASTEEL",
    "IIFL", "SUZLON", "AXISBANK", "VEDL", "UNOMINDA", "JSWENERGY", "TIINDIA", "CUMMINSIND",
    "CONCOR", "GRASIM", "COFORGE", "DLF", "UPL", "JSWSTEEL", "GAIL", "ASTRAL", "ETERNAL",
    "HAVELLS", "ONGC", "BOSCHLTD", "GODREJPROP", "NTPC", "ULTRACEMCO", "NYKAA", "HCLTECH",
    "UNITDSPR", "360ONE", "BEL", "BHEL", "TCS", "LODHA", "WIPRO", "SHREECEM", "DELHIVERY",
    "OIL", "DMART", "CAMS", "PPLPHARMA", "HAL", "ADANIPORTS", "SOLARINDS", "AMBUJACEM",
    "POLICYBZR", "SBIN", "TECHM", "KALYANKJIL", "KAYNES", "DRREDDY", "POWERGRID",
    "MAZDOCK", "DIXON", "DIVISLAB", "CIPLA", "IOC", "ADANIENT", "JINDALSTEL",
    "CROMPTON", "TVSMOTOR", "ICICIGI", "TITAN", "CANBK", "HDFCAMC", "SIEMENS",
    "EXIDEIND", "IRFC", "PETRONET", "HINDPETRO", "RECLTD", "BIOCON", "BAJAJ-AUTO",
    "LTIM", "DALBHARAT", "SUNPHARMA", "HEROMOTOCO", "HUDCO",  "APOLLOHOSP",
    "HINDZINC", "ASHOKLEY", "RELIANCE", "IGL", "TATAELXSI", "MPHASIS", "IREDA", "LUPIN",
    "INDUSINDBK", "HINDALCO", "PFC", "TRENT", "PAYTM", "IRCTC", "COALINDIA",
    "SAMMAANCAP", "PATANJALI", "ABB", "INFY", "OBEROIRLTY", "JUBLFOOD", "ICICIBANK", "BPCL",
    "ADANIGREEN", "IEX", "SRF", "CGPOWER", "ITC", "SAIL", "FEDERALBNK", "KFINTECH", "ALKEM",
    "TATAMOTORS", "JIOFIN", "BDL", "BAJAJFINSV", "HINDUNILVR","INOXWIND", "INDIGO", "HDFCBANK", "LAURUSLABS", "TORNTPHARM", "TATATECH", "PNB",
    "ADANIENSOL", "VOLTAS", "NMDC", "IDFCFIRSTB", "LICI", "NATIONALUM", "BRITANNIA",
    "APLAPOLLO", "SBILIFE", "ZYDUSLIFE", "ICICIPRULI", "ABCAPITAL",
    "CDSL", "KPITTECH", "PIIND", "LICHSGFIN", "AUBANK", "SONACOMS", "TORNTPOWER", "HDFCLIFE",
    "SBICARD", "BANKINDIA", "COLPAL", "INDUSTOWER", "NUVAMA", "MARICO", "PNBHOUSING", "PGEL",
    "MANKIND", "BAJFINANCE", "NESTLEIND", "NAUKRI", "AUROPHARMA", "ASIANPAINT", "SHRIRAMFIN",
    "TATACONSUM", "ANGELONE", "MFSL", "DABUR", "TITAGARH", "GLENMARK", "FORTIS", "BSE",
    "MAXHEALTH", "MCX", "INDHOTEL", "VBL", "SYNGENE", "GODREJCP"
]

IST = pytz.timezone("Asia/Kolkata")

# ---------------- TRUEDATA CONFIG ----------------
TD_USERNAME = "tdwsp751"
TD_PASSWORD = "raj@751"
try:
    td_hist = TD_hist(TD_USERNAME, TD_PASSWORD, log_level=logging.WARNING)
except Exception as e:
    print(f"Failed to initialize Truedata history client: {e}")
    td_hist = None

# ---------------- COLORS ----------------
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

# ---------------- SECTOR MAPS (same as before) ----------------
NSE_INDEX_TO_SECTOR = {
    "NIFTY IT": "Technology", "NIFTY PHARMA": "Pharma", "NIFTY FMCG": "Consumer",
    "NIFTY BANK": "Banking", "NIFTY AUTO": "Auto", "NIFTY METAL": "Metal",
    "NIFTY ENERGY": "Energy", "NIFTY REALTY": "Realty", "NIFTY INFRA": "Infrastructure",
    "NIFTY PSU BANK": "PSU Bank", "NIFTY PSE": "PSE", "NIFTY COMMODITIES": "Commodities",
    "NIFTY MNC": "Finance", "NIFTY FINANCIAL SERVICES": "Finance",
    "NIFTY INFRASTRUCTURE": "Infrastructure", "BANKNIFTY": "Banking",
    "NIFTYAUTO": "Auto", "NIFTYIT": "Technology", "NIFTYPHARMA": "Pharma",
    "NIFTY CONSUMER DURABLES": "Consumer Durables", "NIFTY HEALTHCARE INDEX": "Healthcare",
    "NIFTY CAPITAL MARKETS": "Capital Market", "NIFTY PRIVATE BANK": "Private Bank",
    "NIFTY OIL & GAS": "Oil and Gas", "NIFTY INDIA DEFENCE": "Defence",
    "NIFTY CORE HOUSING": "Core Housing", "NIFTY SERVICES SECTOR": "Services Sector",
    "NIFTY FINANCIAL SERVICES 25/50": "Financial Services 25/50", "NIFTY INDIA TOURISM": "Tourism",
}
SECTOR_STOCKS = {
    "Technology": ["TCS","INFY","HCLTECH","WIPRO","TECHM","LTIM","MPHASIS","COFORGE","PERSISTENT","CYIENT","KPITTECH","TATAELXSI","SONACOMS","KAYNES","OFSS"],
    "Auto": ["MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","HEROMOTOCO","TVSMOTOR","BHARATFORG","EICHERMOT","ASHOKLEY","BOSCHLTD","TIINDIA","MOTHERSON"],
    "Banking": ["HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","PNB","BANKBARODA","CANBK","IDFCFIRSTB","INDUSINDBK","AUBANK","FEDERALBNK"],
    "Pharma": ["SUNPHARMA","DRREDDY","CIPLA","LUPIN","AUROPHARMA","TORNTPHARM","GLENMARK","ALKEM","LAURUSLABS","BIOCON","ZYDUSLIFE","MANKIND","SYNGENE","PPLPHARMA"],
    "Energy": ["RELIANCE","NTPC","BPCL","IOC","ONGC","GAIL","HINDPETRO","ADANIGREEN","ADANIENSOL","JSWENERGY","COALINDIA","TATAPOWER","SUZLON","PETRONET","OIL","POWERGRID","NHPC","ADANIPORTS","ABB","SIEMENS","CGPOWER","INOXWIND"],
    "Metal": ["TATASTEEL","JSWSTEEL","SAIL","JINDALSTEL","HINDALCO","NMDC"],
    "Consumer": ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","TATACONSUM","DABUR","AMBER","UNITDSPR","GODREJCP","MARICO","COLPAL","UPL","VBL"],
    "PSU Bank": ["SBIN","PNB","BANKBARODA","CANBK","UNIONBANK","BANKINDIA"],
    "Finance": ["BAJFINANCE","SHRIRAMFIN","CHOLAFIN","HDFCLIFE","ICICIPRULI"],
    "Realty": ["DLF","LODHA","PRESTIGE","GODREJPROP","OBEROIRLTY","PHOENIXLTD","NCC","NBCC"],
    "PSE": ["BEL","BHEL","NHPC","GAIL","IOC","NTPC","POWERGRID","HINDPETRO","OIL","RECLTD","ONGC","NMDC","BPCL","HAL","RVNL","PFC","COALINDIA","IRCTC","IRFC"],
    "Commodities": ["AMBUJACEM","APLAPOLLO","ULTRACEMCO","SHREECEM","JSWSTEEL","HINDALCO","NHPC","IOC","NTPC","HINDPETRO","ADANIGREEN","OIL","VEDL","PIIND","ONGC","NMDC","UPL","BPCL","JSWENERGY","GRASIM","RELIANCE","TORNTPOWER","TATAPOWER","COALINDIA","PIDILITIND","SRF","ADANIENSOL","JINDALSTEL","TATASTEEL"],
    "Consumer Durables": ["TITAN","DIXON","HAVELLS","CROMPTON","POLYCAB","EXIDEIND","AMBER","KAYNES","VOLTAS","PGEL","BLUESTARCO"],
    "Healthcare": ["SUNPHARMA","DIVISLAB","CIPLA","TORNTPHARM","MAXHEALTH","APOLLOHOSP","DRREDDY","MANKIND","ZYDUSLIFE","LUPIN","FORTIS","ALKEM","AUROPHARMA","GLENMARK","BIOCON","LAURUSLABS","SYNGENE","GRANULES"],
    "Capital Market": ["HDFCAMC","BSE","360ONE","MCX","CDSL","NUVAMA","ANGELONE","KFINTECH","CAMS","IEX"],
    "Private Bank": ["HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","YESBANK","IDFCFIRSTB","INDUSINDBK","FEDERALBNK","BANDHANBNK","RBLBANK"],
    "Oil and Gas": ["RELIANCE","ONGC","IOC","BPCL","GAIL","HINDPETRO","OIL","PETRONET","IGL"],
    "Defence": ["HAL","BEL","SOLARINDS","MAZDOCK","BDL"],
    "Core Housing": ["ULTRACEMCO","ASIANPAINT","GRASIM","DLF","AMBUJACEM","LODHA","DIXON","POLYCAB","SHREECEM","HAVELLS","PRESTIGE","GODREJPROP","OBEROIRLTY","PHOENIXLTD","VOLTAS","DALBHARAT","KEI","BLUESTARCO","LICHSGFIN","PNBHOUSING","CROMPTON"],
    "Services Sector": ["HDFCBANK","BHARTIARTL","TCS","ICICIBANK","SBIN","INFY","BAJFINANCE","HCLTECH","KOTAKBANK","AXISBANK","BAJAJFINSV","NTPC","ZOMATO","ADANIPORTS","DMART","POWERGRID","WIPRO","INDIGO","JIOFIN","SBILIFE","HDFCLIFE","LTIM","TECHM","TATAPOWER","SHRIRAMFIN","GAIL","MAXHEALTH","APOLLOHOSP","NAUKRI","INDUSINDBK"],
    "Financial Services 25/50": ["HDFCBANK","ICICIBANK","SBIN","BAJFINANCE","KOTAKBANK","AXISBANK","BAJAJFINSV","JIOFIN","SBILIFE","HDFCLIFE","PFC","CHOLAFIN","HDFCAMC","SHRIRAMFIN","MUTHOOTFIN","RECLTD","ICICIGI","ICICIPRULI","SBICARD","LICHSGFIN"],
    "Tourism": ["INDIGO","INDHOTEL","IRCTC","JUBLFOOD"]
}

# ---------------- RATE LIMITER ----------------
class TokenBucket:
    def __init__(self, rate_per_sec=9.0, bucket_size=12, per_min_ceiling=300):
        self.rate = rate_per_sec
        self.bucket_size = bucket_size
        self.tokens = bucket_size
        self.last_refill = time_module.time()
        self.lock = threading.Lock()
        self.per_min_ceiling = per_min_ceiling
        self.min_window = deque()

    def acquire(self):
        while True:
            with self.lock:
                now = time_module.time()
                elapsed = now - self.last_refill
                refill = elapsed * self.rate
                if refill > 0:
                    self.tokens = min(self.bucket_size, self.tokens + refill)
                    self.last_refill = now
                while self.min_window and self.min_window[0] <= now - 60.0:
                    self.min_window.popleft()
                if self.tokens >= 1.0 and len(self.min_window) < self.per_min_ceiling:
                    self.tokens -= 1.0
                    self.min_window.append(now)
                    return
            time_module.sleep(0.03)

api_limiter = TokenBucket(rate_per_sec=8.0, bucket_size=12, per_min_ceiling=300)

# ---------------- INDICATORS ----------------
class EnhancedTechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        indicators = {}
        if df is None or len(df) < 30:
            return indicators
        try:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            indicators['RSI'] = 100 - (100 / (1 + rs))
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            indicators['MACD'] = macd_line - signal_line
            tr1 = df['High'] - df['Low']
            tr2 = (df['High'] - df['Close'].shift()).abs()
            tr3 = (df['Low'] - df['Close'].shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            ATR = true_range.rolling(window=14).mean()
            indicators['ATR'] = ATR
            avg_volume_20 = df['Volume'].rolling(window=20).mean().replace(0, 1)
            volume_ratio = df['Volume'] / avg_volume_20
            indicators['Volume_Surge'] = np.clip((volume_ratio - 0.5) * 40, 0, 100)
            price_momentum = df['Close'].pct_change(periods=10) * 100
            avg_volume_10 = df['Volume'].rolling(window=10).mean().replace(0, 1)
            volume_momentum = (df['Volume'] / avg_volume_10 - 1) * 100
            momentum_score = (price_momentum * 0.7 + volume_momentum * 0.3)
            indicators['Momentum'] = 50 + np.clip(momentum_score * 1.5, -50, 50)
        except Exception as e:
            logger.error(f"Error in base indicators: {e}")
        return indicators

class OptionsReadyIndicators(EnhancedTechnicalIndicators):
    @staticmethod
    def calculate_all_indicators(
        df,
        squeeze_window=120,
        squeeze_quantile=0.1,
        require_two_bar_atr_accel=True
    ):
        indicators = super(OptionsReadyIndicators, OptionsReadyIndicators).calculate_all_indicators(df)
        if not indicators:
            return indicators
        try:
            ma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            upper_band = ma20 + (std20 * 2.0)
            lower_band = ma20 - (std20 * 2.0)
            bbw = (upper_band - lower_band) / ma20.replace(0, np.nan)
            indicators['BBW'] = bbw
            bbw_quantile = bbw.rolling(window=squeeze_window, min_periods=20).quantile(
                squeeze_quantile, interpolation='linear'
            )
            indicators['BBW_q'] = bbw_quantile
            in_squeeze = bbw <= bbw_quantile
            indicators['in_squeeze'] = in_squeeze
            breaks_upper = df['Close'] > upper_band.shift(1)
            breaks_lower = df['Close'] < lower_band.shift(1)
            vol_surge_now = indicators.get('Volume_Surge', pd.Series(index=df.index, data=0))
            vol_confirm = vol_surge_now > 60.0
            squeeze_on = in_squeeze.shift(1).fillna(False)
            indicators['squeeze_fire_up'] = squeeze_on & breaks_upper & vol_confirm
            indicators['squeeze_fire_down'] = squeeze_on & breaks_lower & vol_confirm
            ATR = indicators.get('ATR', pd.Series(index=df.index, data=np.nan))
            ATR_EMA10 = ATR.ewm(span=10, adjust=False).mean()
            atr_gt = ATR > ATR_EMA10
            atr_rising = ATR > ATR.shift(1)
            if require_two_bar_atr_accel:
                atr_accel = atr_gt & atr_gt.shift(1).fillna(False) & atr_rising
            else:
                atr_accel = atr_gt & atr_rising
            indicators['ATR_EMA10'] = ATR_EMA10
            indicators['ATR_accel'] = atr_accel.astype(float)
        except Exception as e:
            logger.error(f"Error in options indicators: {e}")
        return indicators

# ---------------- SCANNER CORE ----------------
class OptionsBreakoutScanner:
    def __init__(self, backtest_mode=False):
        self.backtest_mode = backtest_mode
        self.best_sectors = []
        self.worst_sectors = []
        self.min_avg_vol_5m = 50000
        self.min_avg_vol_15m = 30000
        self.use_mtf_confirmation = True

    def normalize_df(self, df, symbol):
        if df is None or len(df) == 0:
            return None
        if isinstance(df, pd.DataFrame):
            out = df.copy()
        else:
            out = pd.DataFrame(df)
        col_lookup = {c.lower(): c for c in out.columns}
        date_col = col_lookup.get('timestamp') or col_lookup.get('time') or list(out.columns)[0]
        open_col = col_lookup.get('open', 'Open')
        high_col = col_lookup.get('high', 'High')
        low_col = col_lookup.get('low', 'Low')
        close_col = col_lookup.get('close', 'Close')
        vol_col = col_lookup.get('volume') or col_lookup.get('vol') or 'Volume'
        out = out.rename(columns={
            date_col: 'Date', open_col: 'Open', high_col: 'High',
            low_col: 'Low', close_col: 'Close', vol_col: 'Volume'
        })
        out['Date'] = pd.to_datetime(out['Date'])
        out.set_index('Date', inplace=True)
        if out.index.tz is None:
            out = out.tz_localize(IST, ambiguous='infer')
        out = out[['Open','High','Low','Close','Volume']].dropna().sort_index()
        return out

    def fetch_tf(self, symbol, date_obj, bar_size):
        # Fetch bars with TrueData for a single day in a chosen timeframe
        try:
            # Use duration='1 D' with end_time at 15:30 IST to get the session day
            session_end_ist = IST.localize(datetime.combine(date_obj, time(15,30)))
            session_end_naive = session_end_ist.replace(tzinfo=None)
            api_limiter.acquire()
            df = td_hist.get_historic_data(
                symbol,
                duration='1 D',
                end_time=session_end_naive,
                bar_size=bar_size
            )
            if df is None or len(df) == 0:
                return None
            df = self.normalize_df(df, symbol)
            # Clamp to 09:15–15:30 IST
            df = df.between_time("09:15","15:30")
            return df
        except Exception as e:
            logger.error(f"{symbol} fetch_tf error {bar_size}: {e}")
            return None

    def fetch_all_tfs_for_symbol(self, symbol, date_obj):
        # Direct TF pulls; no 1-min resampling
        tf_map = {}
        for bar_size in ['5 mins','15 mins','30 mins','60 mins']:
            d = self.fetch_tf(symbol, date_obj, bar_size)
            if d is not None and len(d) >= 30:
                tf = int(bar_size.split()[0])
                tf_map[tf] = d
        return tf_map

    def _passes_liquidity(self, tfs):
        df5 = tfs.get(5, None)
        df15 = tfs.get(15, None)
        if df5 is not None:
            avg5 = df5['Volume'].tail(20).mean()
            if not np.isnan(avg5) and avg5 < self.min_avg_vol_5m:
                return False
        if df15 is not None:
            avg15 = df15['Volume'].tail(20).mean()
            if not np.isnan(avg15) and avg15 < self.min_avg_vol_15m:
                return False
        return True

    def _mtf_trend_bias(self, tfs):
        df30 = tfs.get(30, None)
        if df30 is None or len(df30) < 30:
            return 0
        inds = OptionsReadyIndicators.calculate_all_indicators(df30)
        if not inds:
            return 0
        macd = inds.get('MACD', None)
        ma20 = df30['Close'].rolling(20).mean()
        bias = 0
        if macd is not None and not macd.empty and pd.notna(macd.iloc[-1]):
            bias += 1 if macd.iloc[-1] > 0 else -1
        if not ma20.empty and pd.notna(ma20.iloc[-1]):
            bias += 1 if df30['Close'].iloc[-1] > ma20.iloc[-1] else -1
        return bias

    def calculate_options_signals(self, symbol, tfs):
        final_score = 50.0
        strongest_signal = "Neutral"
        squeeze_status = "No Squeeze"
        try:
            if not self._passes_liquidity(tfs):
                return 'Neutral', 50.0, "Liquidity Fail"
            mtf_bias = self._mtf_trend_bias(tfs) if self.use_mtf_confirmation else 0
            for tf in [15,30,60]:
                df = tfs.get(tf, None)
                if df is None or len(df) < 30:
                    continue
                inds = OptionsReadyIndicators.calculate_all_indicators(df)
                if not inds:
                    continue
                latest = {k: v.iloc[-1] for k, v in inds.items() if hasattr(v,'iloc') and len(v)>0 and pd.notna(v.iloc[-1])}
                is_up = bool(latest.get('squeeze_fire_up', False))
                is_dn = bool(latest.get('squeeze_fire_down', False))
                vol_surge = float(latest.get('Volume_Surge', 0))
                momentum = float(latest.get('Momentum', 50))
                atr_accel = float(latest.get('ATR_accel', 0)) > 0
                bbw = float(latest.get('BBW', np.nan))
                bbw_q = float(latest.get('BBW_q', np.nan))
                sector = next((s for s, st in SECTOR_STOCKS.items() if symbol in st), 'N/A')
                sector_boost = 0
                if sector in self.best_sectors:
                    sector_boost = 5
                if sector in self.worst_sectors:
                    sector_boost = -5
                if is_up:
                    score = 80 + sector_boost
                    if momentum > 65: score += 10
                    if vol_surge > 60: score += 15
                    if atr_accel: score += 10
                    if np.isfinite(bbw) and np.isfinite(bbw_q) and bbw <= bbw_q * 1.05: score += 5
                    if self.use_mtf_confirmation and mtf_bias < 0: score -= 10
                    if score > final_score:
                        final_score, strongest_signal, squeeze_status = score, "Explosive Buy", f"{tf}m Squeeze FIRE UP"
                elif is_dn:
                    score = 20 + sector_boost
                    if momentum < 35: score -= 10
                    if vol_surge > 60: score -= 15
                    if atr_accel: score -= 10
                    if np.isfinite(bbw) and np.isfinite(bbw_q) and bbw <= bbw_q * 1.05: score -= 5
                    if self.use_mtf_confirmation and mtf_bias > 0: score += 10
                    if score < final_score:
                        final_score, strongest_signal, squeeze_status = score, "Explosive Sell", f"{tf}m Squeeze FIRE DOWN"
                elif bool(latest.get('in_squeeze', False)) and squeeze_status == "No Squeeze":
                    squeeze_status = f"{tf}m Squeeze Coiling"
            return strongest_signal, float(np.clip(final_score, 0, 100)), squeeze_status
        except Exception as e:
            logger.error(f"Signal error {symbol}: {e}")
            return 'Neutral', 50.0, "Error"

# ---------------- BACKTEST ENGINE USING API TFs ----------------
class TFBacktester:
    def __init__(self, symbols, date_to_test):
        self.symbols = symbols
        self.date = date_to_test
        self.scanner = OptionsBreakoutScanner(backtest_mode=True)
        self.tf_data = {}  # symbol -> {5: df5, 15: df15, 30: df30, 60: df60}
        self.trades = []
        self.open_positions = {}

    def _preload(self):
        print(f"{Colors.CYAN}Loading TrueData TF bars (5/15/30/60) for {self.date}...{Colors.RESET}")
        def load_one(sym):
            tfs = self.scanner.fetch_all_tfs_for_symbol(sym, self.date)
            return sym, tfs
        with ThreadPoolExecutor(max_workers=10) as ex:
            futures = [ex.submit(load_one, s) for s in self.symbols]
            for i, fut in enumerate(as_completed(futures), 1):
                sym, tfs = fut.result()
                if tfs:
                    self.tf_data[sym] = tfs
                print(f"Fetched {i}/{len(self.symbols)}: {sym}  TFs={list(tfs.keys())}", end="\r")
        print(f"\n{Colors.GREEN}Ready: {len(self.tf_data)} symbols with TF data.{Colors.RESET}")

    def _iter_5m_clock(self):
        start = IST.localize(datetime.combine(self.date, time(9,15)))
        end = IST.localize(datetime.combine(self.date, time(15,30)))
        t = start
        while t <= end:
            yield t
            t += timedelta(minutes=5)

    def _available_up_to(self, df, t):
        if df is None or df.empty:
            return None
        return df[df.index <= t]

    def run(self):
        self._preload()
        if not self.tf_data:
            print(f"{Colors.RED}No TF data loaded; check plan/timeframe availability or symbols.{Colors.RESET}")
            return
        print(f"{Colors.BOLD}{Colors.YELLOW}--- Replay {self.date} (API TF bars) ---{Colors.RESET}")
        for t in self._iter_5m_clock():
            print(f"Time {t.strftime('%H:%M')} ", end="\r")
            for sym, tfs in self.tf_data.items():
                if sym in self.open_positions:
                    continue
                # Slice each TF to bars that have "closed" by time t
                have = {}
                for tf, df in tfs.items():
                    d = self._available_up_to(df, t)
                    if d is not None and len(d) >= 30:
                        have[tf] = d
                if len(have) < 2:
                    continue
                signal, score, squeeze = self.scanner.calculate_options_signals(sym, have)
                if "Explosive" in signal:
                    # Enter at last available 5m close if present, else use lowest tf available
                    entry_px = None
                    if 5 in have:
                        entry_px = float(have[5]['Close'].iloc[-1])
                    else:
                        tf_low = sorted(have.keys())[0]
                        entry_px = float(have[tf_low]['Close'].iloc[-1])
                    side = "LONG" if "Buy" in signal else "SHORT"
                    self.open_positions[sym] = {
                        'symbol': sym, 'side': side, 'entry_time': t, 'entry_px': entry_px,
                        'reason': squeeze
                    }
                    print(f"\n{Colors.GREEN}ENTRY{Colors.RESET} {sym} {side} {entry_px:.2f} at {t.strftime('%H:%M')} | {squeeze}")
        # Exit everything at session close
        session_close = IST.localize(datetime.combine(self.date, time(15,30)))
        for sym, pos in list(self.open_positions.items()):
            # Use last 5m close if available, else lowest tf
            tfs = self.tf_data.get(sym, {})
            exit_px = None
            exit_ts = None
            if 5 in tfs and not tfs[5].empty:
                d = tfs[5][tfs[5].index <= session_close]
                if not d.empty:
                    exit_px = float(d['Close'].iloc[-1])
                    exit_ts = d.index[-1]
            if exit_px is None:
                if tfs:
                    tf_low = sorted(tfs.keys())[0]
                    d = tfs[tf_low][tfs[tf_low].index <= session_close]
                    if not d.empty:
                        exit_px = float(d['Close'].iloc[-1])
                        exit_ts = d.index[-1]
            if exit_px is None:
                continue
            pnl = exit_px - pos['entry_px'] if pos['side']=="LONG" else pos['entry_px'] - exit_px
            self.trades.append({
                'symbol': sym, 'side': pos['side'], 'entry_time': pos['entry_time'],
                'entry_px': pos['entry_px'], 'exit_time': exit_ts, 'exit_px': exit_px,
                'pnl_points': pnl, 'reason': pos['reason']
            })
            print(f"\n{Colors.YELLOW}EXIT{Colors.RESET}  {sym} {exit_px:.2f} at {exit_ts.strftime('%H:%M')} | PnL {pnl:.2f}")
            del self.open_positions[sym]
        self.report()

    def report(self):
        print(f"\n{Colors.BOLD}{Colors.CYAN}======== BACKTEST SUMMARY ========{Colors.RESET}")
        if not self.trades:
            print(f"{Colors.YELLOW}No trades executed.{Colors.RESET}")
            return
        df = pd.DataFrame(self.trades)
        df['entry_time'] = df['entry_time'].dt.tz_convert(IST).dt.strftime('%H:%M')
        df['exit_time'] = pd.to_datetime(df['exit_time']).dt.tz_convert(IST).dt.strftime('%H:%M')
        print(df[['symbol','side','reason','entry_time','entry_px','exit_time','exit_px','pnl_points']].to_string(index=False))
        total = len(df)
        wins = int((df['pnl_points']>0).sum())
        losses = total - wins
        win_rate = (wins/total*100.0) if total else 0.0
        net = float(df['pnl_points'].sum())
        print(f"\nTrades: {total} | Wins: {wins} | Losses: {losses} | Win%: {win_rate:.1f} | Net PnL (pts): {net:.2f}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    if td_hist is None:
        print(f"{Colors.RED}Could not start because Truedata client failed to initialize.{Colors.RESET}")
        print("Please check credentials and network connection.")
    else:
        if BACKTEST_MODE:
            bt = TFBacktester(REPLAY_SYMBOLS, BACKTEST_DATE)
            bt.run()
        else:
            print(f"{Colors.YELLOW}Live mode with API TF bars not implemented in this file.{Colors.RESET}")
