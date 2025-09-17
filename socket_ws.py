# SECTOR-FOCUSED ENHANCED SCANNER
# WS: 5m bars (bar5min), REST: 15m/30m/60m/1D with resilient fallback
# Universe: Top 3 Best + Bottom 3 Worst sectors (live API)
# Scoring: 15 indicators, sector-agnostic, multi-TF confirmation

import pandas as pd
import numpy as np
import json
import os
import threading
import time as time_module
import asyncio
import websockets
from datetime import datetime, timedelta, time
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from logzero import logger
from truedata.history import TD_hist
import warnings
warnings.filterwarnings('ignore')

# ===== TrueData Auth/Config =====
TD_USERNAME = "tdwsp751"
TD_PASSWORD = "raj@751"
td_hist = TD_hist(TD_USERNAME, TD_PASSWORD)

# WebSocket endpoints (confirm with provider)
WS_HOST = "wss://realtime.truedata.in"
WS_PORT_STREAM = 8084
WS_PATH = "/ws"
WS_URL = f"{WS_HOST}:{WS_PORT_STREAM}{WS_PATH}"

# Sector API endpoint
SECTOR_API_URL = "http://localhost:3001/api/allIndices"

# ===== Master Stock List =====
ALL_NSE_STOCKS = [
    "CHOLAFIN","GMRAIRPORT","CYIENT","HFCL","AMBER","KOTAKBANK","PERSISTENT","NHPC",
    "LT","PAGEIND","M&M","RVNL","SUPREMEIND","BHARATFORG","TATAPOWER","KEI",
    "MARUTI","POLYCAB","PRESTIGE","MOTHERSON","OFSS","NCC","EICHERMOT","BLUESTARCO",
    "BHARTIARTL","PHOENIXLTD","NBCC","MUTHOOTFIN","LTF","MANAPPURAM","TATASTEEL",
    "IIFL","SUZLON","AXISBANK","VEDL","UNOMINDA","JSWENERGY","TIINDIA","CUMMINSIND",
    "CONCOR","GRASIM","COFORGE","DLF","UPL","JSWSTEEL","GAIL","ASTRAL","ETERNAL",
    "HAVELLS","ONGC","BOSCHLTD","GODREJPROP","NTPC","ULTRACEMCO","NYKAA","HCLTECH",
    "UNITDSPR","360ONE","BEL","BHEL","TCS","LODHA","WIPRO","SHREECEM","DELHIVERY",
    "OIL","DMART","CAMS","PPLPHARMA","HAL","ADANIPORTS","SOLARINDS","AMBUJACEM",
    "POLICYBZR","SBIN","TECHM","KALYANKJIL","KAYNES","DRREDDY","POWERGRID",
    "MAZDOCK","DIXON","DIVISLAB","CIPLA","IOC","ADANIENT","JINDALSTEL",
    "CROMPTON","TVSMOTOR","ICICIGI","TITAN","CANBK","HDFCAMC","SIEMENS",
    "EXIDEIND","IRFC","PETRONET","HINDPETRO","RECLTD","BIOCON","BAJAJ-AUTO",
    "LTIM","DALBHARAT","SUNPHARMA","HEROMOTOCO","HUDCO","APOLLOHOSP",
    "HINDZINC","ASHOKLEY","RELIANCE","IGL","TATAELXSI","MPHASIS","IREDA","LUPIN",
    "INDUSINDBK","HINDALCO","PFC","TRENT","PAYTM","IRCTC","COALINDIA",
    "SAMMAANCAP","PATANJALI","ABB","INFY","OBEROIRLTY","JUBLFOOD","ICICIBANK","BPCL",
    "ADANIGREEN","IEX","SRF","CGPOWER","ITC","SAIL","FEDERALBNK","KFINTECH","ALKEM",
    "TATAMOTORS","JIOFIN","BDL","BAJAJFINSV","HINDUNILVR","INOXWIND","INDIGO","HDFCBANK",
    "LAURUSLABS","TORNTPHARM","TATATECH","PNB","ADANIENSOL","VOLTAS","NMDC","IDFCFIRSTB",
    "LICI","NATIONALUM","BRITANNIA","APLAPOLLO","SBILIFE","ZYDUSLIFE","ICICIPRULI","ABCAPITAL",
    "CDSL","KPITTECH","PIIND","LICHSGFIN","AUBANK","SONACOMS","TORNTPOWER","HDFCLIFE",
    "SBICARD","BANKINDIA","COLPAL","INDUSTOWER","NUVAMA","MARICO","PNBHOUSING","PGEL",
    "MANKIND","BAJFINANCE","NESTLEIND","NAUKRI","AUROPHARMA","ASIANPAINT","SHRIRAMFIN",
    "TATACONSUM","ANGELONE","MFSL","DABUR","TITAGARH","GLENMARK","FORTIS","BSE",
    "MAXHEALTH","MCX","INDHOTEL","VBL","SYNGENE","GODREJCP"
]

# ===== CLI Colors =====
class Colors:
    GREEN = '\033[92m'; RED = '\033[91m'; YELLOW = '\033[93m'; BLUE = '\033[94m'
    CYAN = '\033[96m'; MAGENTA = '\033[95m'; WHITE = '\033[97m'; BOLD = '\033[1m'
    RESET = '\033[0m'

# ===== Indicator Weights =====
ENHANCED_INDICATOR_WEIGHTS = {
    'RSI': 1.3, 'MACD': 1.6, 'Stochastic': 1.0, 'MA': 1.8,
    'ADX': 1.5, 'Bollinger': 1.4, 'ROC': 1.2, 'OBV': 1.6,
    'CCI': 1.1, 'WWL': 1.0, 'EMA': 1.7, 'VWAP': 1.5,
    'ATR': 1.4, 'Volume_Surge': 2.0, 'Momentum': 1.9
}

# ===== Timeframe Weights (WS 5m + REST 15/30/60/1D) =====
TIMEFRAME_WEIGHTS = {5: 1.0, 15: 1.5, 30: 2.0, 60: 2.5, 'D': 3.0}

# ===== Index → Sector =====
NSE_INDEX_TO_SECTOR = {
    "NIFTY IT":"Technology","NIFTY PHARMA":"Pharma","NIFTY FMCG":"Consumer",
    "NIFTY BANK":"Banking","NIFTY AUTO":"Auto","NIFTY METAL":"Metal",
    "NIFTY ENERGY":"Energy","NIFTY REALTY":"Realty","NIFTY INFRA":"Infrastructure",
    "NIFTY PSU BANK":"PSU Bank","NIFTY PSE":"PSE","NIFTY COMMODITIES":"Commodities",
    "NIFTY MNC":"Finance","NIFTY FINANCIAL SERVICES":"Finance","BANKNIFTY":"Banking",
    "NIFTYAUTO":"Auto","NIFTYIT":"Technology","NIFTYPHARMA":"Pharma",
    "NIFTY CONSUMER DURABLES":"Consumer Durables","NIFTY HEALTHCARE INDEX":"Healthcare",
    "NIFTY CAPITAL MARKETS":"Capital Market","NIFTY PRIVATE BANK":"Private Bank",
    "NIFTY OIL & GAS":"Oil and Gas","NIFTY INDIA DEFENCE":"Defence",
    "NIFTY CORE HOUSING":"Core Housing","NIFTY SERVICES SECTOR":"Services Sector",
    "NIFTY FINANCIAL SERVICES 25/50":"Financial Services 25/50","NIFTY INDIA TOURISM":"Tourism"
}

# ===== Sector Stocks =====
SECTOR_STOCKS = {
    "Technology": ["TCS","INFY","HCLTECH","WIPRO","TECHM","LTIM","MPHASIS","COFORGE","PERSISTENT","CYIENT","KPITTECH","TATAELXSI","SONACOMS","KAYNES","OFSS"],
    "Auto": ["MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","HEROMOTOCO","TVSMOTOR","BHARATFORG","EICHERMOT","ASHOKLEY","BOSCHLTD","TIINDIA","MOTHERSON"],
    "Banking": ["HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","PNB","BANKBARODA","CANBK","IDFCFIRSTB","INDUSINDBK","AUBANK","FEDERALBNK"],
    "Pharma": ["SUNPHARMA","DRREDDY","CIPLA","LUPIN","AUROPHARMA","TORNTPHARM","GLENMARK","ALKEM","LAURUSLABS","BIOCON","ZYDUSLIFE","MANKIND","SYNGENE","PPLPHARMA"],
    "Energy": ["RELIANCE","NTPC","BPCL","IOC","ONGC","GAIL","HINDPETRO","ADANIGREEN","ADANIENSOL","JSWENERGY","COALINDIA","TATAPOWER","SUZLON","PETRONET","OIL","POWERGRID","NHPC","ADANIPORTS","ABB","SIEMENS","CGPOWER","INOXWIND"],
    "Metal": ["TATASTEEL","JSWSTEEL","SAIL","JINDALSTEL","HINDALCO","NMDC"],
    "Consumer": ["HINDUNILVR","ITC","NESTLEIND","BRITANNIA","TATACONSUM","DABUR","AMBER","UNITDSPR","GODREJCP","MARICO","COLPAL","UPL","VBL"],
    "PSU Bank": ["SBIN","PNB","BANKBARODA","CANBK","UNIONBANK","BANKINDIA"],
    "Finance": ["BAJFINANCE","SHRIRAMFIN","CHOLAFIN","HDFCLIFE","ICICIPRULI","ETERNAL"],
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
    "Services Sector": ["HDFCBANK","BHARTIARTL","TCS","ICICIBANK","SBIN","INFY","BAJFINANCE","HCLTECH","KOTAKBANK","AXISBANK","BAJAJFINSV","NTPC","ZOMATO","ADANIPORTS","DMART","POWERGRID","WIPRO","INDIGO","JIOFINSERV","SBILIFE","HDFCLIFE","LTIM","TECHM","TATAPOWER","SHRIRAMFIN","GAIL","MAXHEALTH","APOLLOHOSP","NAUKRI","INDUSINDBK"],
    "Financial Services 25/50": ["HDFCBANK","ICICIBANK","SBIN","BAJFINANCE","KOTAKBANK","AXISBANK","BAJAJFINSV","JIOFIN","SBILIFE","HDFCLIFE","PFC","CHOLAFIN","HDFCAMC","SHRIRAMFIN","MUTHOOTFIN","RECLTD","ICICIGI","ICICIPRULI","SBICARD","LICHSGFIN"],
    "Tourism": ["INDIGO","INDHOTEL","IRCTC","JUBLFOOD"]
}

# ===== Indicators =====
class Indicators:
    @staticmethod
    def calc(df):
        ind = {}
        if df is None or len(df) < 20: return ind
        try:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = -delta.where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            ind['RSI'] = 100 - (100 / (1 + rs))

            ema12 = df['Close'].ewm(span=12).mean(); ema26 = df['Close'].ewm(span=26).mean()
            macd = ema12 - ema26; sig = macd.ewm(span=9).mean()
            ind['MACD'] = macd - sig

            low14 = df['Low'].rolling(14).min(); high14 = df['High'].rolling(14).max()
            ind['Stochastic'] = 100 * (df['Close'] - low14) / (high14 - low14)

            ind['MA'] = df['Close'].rolling(20).mean()
            ind['EMA'] = df['Close'].ewm(span=21).mean()

            hd = df['High'].diff(); ld = df['Low'].diff()
            plus_dm = hd.where((hd > ld) & (hd > 0), 0)
            minus_dm = (-ld).where((ld < hd) & (ld < 0), 0)
            tr1 = df['High'] - df['Low']
            tr2 = (df['High'] - df['Close'].shift()).abs()
            tr3 = (df['Low'] - df['Close'].shift()).abs()
            tr = pd.concat([tr1,tr2,tr3],axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            ind['ADX'] = dx.rolling(14).mean()

            ma20 = df['Close'].rolling(20).mean(); std20 = df['Close'].rolling(20).std()
            ub = ma20 + 2*std20; lb = ma20 - 2*std20
            ind['Bollinger'] = (df['Close'] - ma20)/(ub - lb)*100

            ind['ROC'] = df['Close'].pct_change(12) * 100

            obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            ind['OBV'] = obv.pct_change(10) * 100

            tp = (df['High'] + df['Low'] + df['Close'])/3
            sma_tp = tp.rolling(20).mean()
            mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
            ind['CCI'] = (tp - sma_tp) / (0.015 * mad)

            hh = df['High'].rolling(14).max(); ll = df['Low'].rolling(14).min()
            ind['WWL'] = (hh - df['Close']) / (hh - ll) * -100

            if len(df) >= 20:
                tpv = (df['High'] + df['Low'] + df['Close'])/3
                num = (tpv * df['Volume']).rolling(20).sum()
                den = df['Volume'].rolling(20).sum()
                ind['VWAP'] = num / den

            ind['ATR'] = atr

            if len(df) >= 20:
                avgv20 = df['Volume'].rolling(20).mean()
                ratio = df['Volume'] / avgv20
                ind['Volume_Surge'] = np.clip((ratio - 0.5) * 40, 0, 100)

            if len(df) >= 10:
                pm = df['Close'].pct_change(10) * 100
                avgv10 = df['Volume'].rolling(10).mean()
                vm = (df['Volume'] / avgv10 - 1) * 100
                ind['Momentum'] = 50 + np.clip((pm * 0.7 + vm * 0.3) * 1.5, -50, 50)
        except Exception as e:
            logger.error(f"Indicator calc error: {e}")
        return ind

def norm_indicator(name, value):
    try:
        if name == 'RSI': return max(0, min(100, value))
        if name == 'MACD': return 50 + min(25, max(-25, value * 10))
        if name == 'Stochastic': return max(0, min(100, value))
        if name in ['MA','EMA','VWAP']: return 50
        if name == 'ADX': return max(0, min(100, value))
        if name == 'Bollinger': return max(0, min(100, (value + 100) / 2))
        if name == 'ROC': return 50 + min(25, max(-25, value * 2))
        if name == 'OBV': return 50 + min(25, max(-25, value))
        if name == 'CCI': return max(0, min(100, (value + 200) / 4))
        if name == 'WWL': return max(0, min(100, value + 100))
        if name == 'ATR': return min(100, max(0, value * 20))
        if name == 'Volume_Surge': return max(0, min(100, value))
        if name == 'Momentum': return max(0, min(100, value))
        return 50
    except:
        return 50

# ===== WebSocket 5m Client =====
class FiveMinStream:
    def __init__(self):
        self.ws = None
        self.connected = False
        self.lock = threading.Lock()
        self.buffers = defaultdict(lambda: deque(maxlen=200))
        self.symbol_to_id = {}
        self.id_to_symbol = {}

    def build_df(self, symbol):
        buf = list(self.buffers[symbol])
        if not buf or len(buf) < 20: return None
        df = pd.DataFrame(buf)
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        for c in ['Open','High','Low','Close','Volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna().sort_index()
        return df if len(df) >= 20 else None

    async def send(self, ws, msg):
        await ws.send(json.dumps(msg))

    async def subscribe(self, ws, symbols):
        await self.send(ws, {"method":"addsymbol","symbols": symbols})

    def handle_add_or_touchline(self, payload):
        symbollist = payload.get('symbollist') or payload.get('symbols') or []
        for row in symbollist:
            if not isinstance(row, list) or len(row) < 2: continue
            symbol_name = str(row[0]).upper().replace(" ", "").replace("-I","")
            symbol_id = str(row[1])
            self.symbol_to_id[symbol_name] = symbol_id
            self.id_to_symbol[symbol_id] = symbol_name

    def handle_bar5min(self, arr):
        # {"bar5min":[symbolid,timestamp,open,high,low,close,volume,oi]}
        try:
            sid = str(arr[0]); ts = arr[1]
            o = float(arr[2]); h = float(arr[3]); l = float(arr[4]); c = float(arr[5])
            v = float(arr[6]) if len(arr) > 6 and arr[6] is not None else 0.0
            symbol = self.id_to_symbol.get(sid)
            if not symbol: return
            rec = {'Date': ts, 'Open': o, 'High': h, 'Low': l, 'Close': c, 'Volume': v}
            with self.lock:
                self.buffers[symbol].append(rec)
        except Exception as e:
            logger.error(f"bar5min parse error: {e}")

    async def run(self, symbols):
        self.connected = True
        headers = {
            "user": TD_USERNAME,
            "password": TD_PASSWORD,
            "subscription": "5min"  # Ensure account includes 5min streaming
        }
        async for ws in websockets.connect(WS_URL, extra_headers=headers, ping_interval=20, ping_timeout=20):
            try:
                self.ws = ws
                await self.subscribe(ws, symbols)
                async for msg in ws:
                    try:
                        payload = json.loads(msg)
                    except:
                        continue
                    if isinstance(payload, dict):
                        m = payload.get('message') or ''
                        if m in ('symbols added','touchline'):
                            self.handle_add_or_touchline(payload)
                        if 'bar5min' in payload:
                            self.handle_bar5min(payload['bar5min'])
                    elif isinstance(payload, list):
                        # handle batches if required
                        pass
            except Exception as e:
                logger.error(f"WS error: {e}")
                await asyncio.sleep(3)
            finally:
                self.ws = None

# ===== Sector API =====
def fetch_top_worst_sectors():
    try:
        r = requests.get(SECTOR_API_URL, timeout=10)
        if r.status_code != 200: return None, None, None
        data = r.json()
        if isinstance(data, dict):
            data = data.get('data') or data.get('indices') or data.get('results') or data
        perf = []
        now = datetime.now()
        for idx in data if isinstance(data, list) else []:
            name = None
            for f in ['name','symbol','index','indexName']:
                if f in idx and idx[f]:
                    name = str(idx[f]).strip().upper(); break
            if not name or name not in NSE_INDEX_TO_SECTOR: continue
            chg = 0.0
            for f in ['change_percent','changePercent','pChange','percentChange','change','pchg']:
                if f in idx and idx[f] is not None:
                    try: chg = float(idx[f]); break
                    except: pass
            perf.append({'index': name, 'sector': NSE_INDEX_TO_SECTOR[name], 'change_percent': chg, 'timestamp': now})
        if not perf: return None, None, None
        perf.sort(key=lambda x: x['change_percent'], reverse=True)
        best = [p['sector'] for p in perf[:3]]
        worst = [p['sector'] for p in perf[-3:]]
        return best, worst, perf
    except Exception as e:
        logger.error(f"Sector API error: {e}")
        return None, None, None

def sector_filtered_universe(best, worst):
    selected = set((best or []) + (worst or []))
    chosen = set()
    for s in selected:
        for sym in SECTOR_STOCKS.get(s, []):
            chosen.add(sym)
    master = set(ALL_NSE_STOCKS)
    return sorted(list(chosen & master))

# ===== REST History (15/30/60/1D) with fallback =====
class HistoryFetcher:
    def __init__(self, count=9, period=1):
        self.calls = deque(); self.lock = threading.Lock()
        self.count = count; self.period = period

    def throttle(self):
        with self.lock:
            now = time_module.time()
            while self.calls and self.calls[0] <= now - self.period:
                self.calls.popleft()
            if len(self.calls) >= self.count:
                wait = (self.calls[0] + self.period) - now
                if wait > 0: time_module.sleep(wait)
            self.calls.append(time_module.time())

    def get(self, symbol, tf):
        tf_map = {15: '15 min', 30: '30 min', 60: '60 min', 'D': '1 day'}
        bar = tf_map.get(tf)
        if not bar: return None

        if tf == 15: duration = '10 D'
        elif tf == 30: duration = '20 D'
        elif tf == 60: duration = '40 D'
        elif tf == 'D': duration = '365 D'
        else: duration = '10 D'

        def normalize_df(df):
            if df is None or len(df) == 0: return None
            df2 = df.copy()
            cols = [c.lower() for c in df2.columns]
            cmap = {df2.columns[i]:
                    new for i, col in enumerate(cols)
                    for key, new in [('time','Date'),('open','Open'),('high','High'),
                                     ('low','Low'),('close','Close'),('vol','Volume')]
                    if key in col}
            df2 = df2.rename(columns=cmap)
            req = ['Date','Open','High','Low','Close']
            if not all(c in df2.columns for c in req): return None
            if 'Volume' not in df2.columns: df2['Volume'] = 1000
            df2['Date'] = pd.to_datetime(df2['Date'], errors='coerce')
            if df2['Date'].dt.tz is not None:
                df2['Date'] = df2['Date'].dt.tz_localize(None)
            df2.set_index('Date', inplace=True)
            for c in ['Open','High','Low','Close','Volume']:
                df2[c] = pd.to_numeric(df2[c], errors='coerce')
            df2 = df2.dropna().sort_index()
            return df2.tail(200) if tf in (60,'D') else df2.tail(100)

        # Retry TD_hist up to 3 times
        for attempt in range(1, 4):
            try:
                self.throttle()
                df = td_hist.get_historic_data(symbol, duration=duration, bar_size=bar)
                out = normalize_df(df)
                if out is not None:
                    return out
            except Exception as e:
                if attempt == 3 or "LZ4BlockError" not in str(type(e)):
                    logger.error(f"History {symbol}_{tf} attempt {attempt} error: {e}")
            time_module.sleep(0.25 * attempt)

        # CSV fallback (no compression)
        try:
            self.throttle()
            end = datetime.now()
            if tf == 'D':
                start = end - timedelta(days=365); interval = '1day'
            else:
                start = end - timedelta(days=40 if tf == 60 else 20); interval = f"{tf}min"
            url = (
                "https://history.truedata.in/get"
                f"?user={TD_USERNAME}"
                f"&password={TD_PASSWORD}"
                f"&symbol={symbol}"
                f"&interval={interval}"
                f"&start={start.strftime('%Y-%m-%d')}"
                f"&end={end.strftime('%Y-%m-%d')}"
                "&format=csv&compress=false"
            )
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and r.text:
                header = r.text.splitlines()[0].lower()
                if 'timestamp' in header or 'date' in header:
                    from io import StringIO
                    csv_df = pd.read_csv(StringIO(r.text))
                    return normalize_df(csv_df)
        except Exception as e:
            logger.error(f"CSV fallback {symbol}_{tf} error: {e}")

        return None

# ===== Scoring =====
def norm_score_for_ma(price, val):
    if val <= 0: return 50
    pct = (price - val) / val * 100
    if pct > 2: return 75
    if pct > 0: return 60
    if pct > -2: return 50
    if pct > -5: return 40
    return 25

def score_symbol(timeframes_dfs):
    if not timeframes_dfs: return 'Neutral', 0.0
    total_w, total = 0.0, 0.0
    tf_scores = {}
    for tf, df in timeframes_dfs.items():
        if df is None or len(df) < 20: continue
        inds = Indicators.calc(df)
        if not inds: continue
        tf_sum, tf_w = 0.0, 0.0
        price = df['Close'].iloc[-1]
        for name, w in ENHANCED_INDICATOR_WEIGHTS.items():
            if name in inds and inds[name] is not None and not getattr(inds[name], 'empty', False):
                val = inds[name].iloc[-1]
                if pd.isna(val): continue
                ns = norm_score_for_ma(price, val) if name in ['MA','EMA','VWAP'] else norm_indicator(name, val)
                tf_sum += ns * w; tf_w += w
        if tf_w > 0:
            tf_score = tf_sum / tf_w
            tf_scores[tf] = tf_score
            mult = TIMEFRAME_WEIGHTS.get(tf, 1.0)
            total += tf_score * mult
            total_w += mult
    if total_w == 0: return 'Neutral', 0.0
    base = total / total_w
    if len(tf_scores) >= 2:
        bull = sum(1 for s in tf_scores.values() if s > 55)
        bear = sum(1 for s in tf_scores.values() if s < 45)
        if bull >= 2: base += 8
        elif bear >= 2: base -= 8
    if base >= 82: return 'Very Strong Buy', base
    if base >= 72: return 'Strong Buy', base
    if base >= 60: return 'Buy', base
    if base <= 18: return 'Very Strong Sell', base
    if base <= 28: return 'Strong Sell', base
    if base <= 40: return 'Sell', base
    return 'Neutral', base

# ===== Main Scanner =====
class Scanner:
    def __init__(self):
        self.best = ["Technology","Pharma","Banking"]
        self.worst = ["Auto","Metal","Energy"]
        self.history = HistoryFetcher()
        self.five = FiveMinStream()
        self.market_start = time(9,15)
        self.market_end = time(15,30)
        self.scan_interval = 300  # 5 minutes

    def is_market_open(self):
        now = datetime.now()
        return now.weekday() <= 4 and self.market_start <= now.time() <= self.market_end

    def build_tf_data(self, symbol):
        data = {}
        # 5m WS
        df5 = self.five.build_df(symbol)
        if df5 is not None: data[5] = df5
        # REST 15/30/60/1D
        for tf in [15, 30, 60, 'D']:
            df = self.history.get(symbol, tf)
            if df is not None:
                data[tf] = df
        return data

    def tf_label(self, tf):
        return "1D" if tf == 'D' else f"{tf}m"

    async def run(self):
        print(f"{Colors.CYAN}{Colors.BOLD}🚀 Sector-Focused Enhanced Scanner (WS 5m + REST 15/30/60/1D){Colors.RESET}")
        b,w,_ = fetch_top_worst_sectors()
        if b and w: self.best, self.worst = b, w
        print(f"🏆 Best: {', '.join(self.best)} | 📉 Worst: {', '.join(self.worst)}")
        universe = sector_filtered_universe(self.best, self.worst)
        if not universe: universe = ALL_NSE_STOCKS[:100]

        ws_symbols = universe[:200]
        loop = asyncio.get_running_loop()
        ws_task = loop.create_task(self.five.run(ws_symbols))

        try:
            while True:
                if not self.is_market_open():
                    print(f"{Colors.YELLOW}Market closed. Waiting...{Colors.RESET}")
                    await asyncio.sleep(60)
                    continue

                b2,w2,_ = fetch_top_worst_sectors()
                if b2 and w2:
                    self.best, self.worst = b2, w2
                    universe = sector_filtered_universe(self.best, self.worst)

                print(f"\n{Colors.CYAN}🔄 Scan at {datetime.now().strftime('%H:%M:%S')} | Universe: {len(universe)}{Colors.RESET}")
                signals = []
                # REST concurrency kept low to reduce compression issues
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = {pool.submit(self.build_tf_data, sym): sym for sym in universe}
                    for fut in as_completed(futures):
                        sym = futures[fut]
                        try:
                            tfd = fut.result()
                            if len(tfd) < 2: continue
                            sig, score = score_symbol(tfd)
                            if abs(score - 50) > 15:
                                sector = next((s for s, lst in SECTOR_STOCKS.items() if sym in lst), 'N/A')
                                signals.append({
                                    'symbol': sym, 'sector': sector, 'signal': sig, 'score': score,
                                    'tf_details': list(tfd.keys()), 'timeframes': len(tfd)
                                })
                        except Exception as e:
                            logger.error(f"Process {sym} error: {e}")

                bull = [s for s in signals if 'Buy' in s['signal']]
                bear = [s for s in signals if 'Sell' in s['signal']]
                bull.sort(key=lambda x: x['score'], reverse=True)
                bear.sort(key=lambda x: x['score'])

                print(f"{Colors.GREEN}{Colors.BOLD}\nTOP 10 BULLISH:{Colors.RESET}")
                print(f"{'Stock':<10} {'Sector':<18} {'Signal':<18} {'Score':>7} {'TFs':>3} {'Coverage':<24}")
                print("-"*90)
                for s in bull[:10]:
                    cov = ",".join(self.tf_label(tf) for tf in s['tf_details'][:6])
                    print(f"{s['symbol']:<10} {s['sector']:<18} {s['signal']:<18} {s['score']:>7.1f} {s['timeframes']:>3} {cov:<24}")

                print(f"{Colors.RED}{Colors.BOLD}\nTOP 10 BEARISH:{Colors.RESET}")
                print(f"{'Stock':<10} {'Sector':<18} {'Signal':<18} {'Score':>7} {'TFs':>3} {'Coverage':<24}")
                print("-"*90)
                for s in bear[:10]:
                    cov = ",".join(self.tf_label(tf) for tf in s['tf_details'][:6])
                    print(f"{s['symbol']:<10} {s['sector']:<18} {s['signal']:<18} {s['score']:>7.1f} {s['timeframes']:>3} {cov:<24}")

                print(f"{Colors.BLUE}Next scan in 5 minutes...{Colors.RESET}")
                await asyncio.sleep(self.scan_interval)
        finally:
            ws_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ws_task

# ===== Entrypoint =====
if __name__ == "__main__":
    import contextlib
    try:
        asyncio.run(Scanner().run())
    except KeyboardInterrupt:
        print("\nShutting down...")
