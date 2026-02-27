# scan_13of15_5mloop_session_rich.py

import os, logging, warnings
warnings.filterwarnings("ignore")
from datetime import datetime, date
from time import monotonic, perf_counter, sleep
from collections import defaultdict
import threading
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from tqdm import tqdm
import requests
from truedata import TD_hist

# Rich console/table for output + export
from rich.console import Console
from rich.table import Table

# ----- Persistent HTTP session for connection reuse -----
SESSION = requests.Session()
adapter = requests.adapters.HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=0)
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)

# ----- TD creds -----
TD_USERNAME = "tdwsp751"
TD_PASSWORD = "raj@751"
td_hist = TD_hist(TD_USERNAME, TD_PASSWORD, log_level=logging.WARNING)

def td_get_historic(symbol, duration, bar_size):
    # If the SDK exposes session injection, pass SESSION; otherwise call directly.
    return td_hist.get_historic_data(symbol, duration=duration, bar_size=bar_size)

UNIVERSE_FNO = [
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

TIMEFRAMES = [5, 15, 30]
TF_MAP = {5:'5 min', 15:'15 min', 30:'30 mins'}
def tf_duration(tf):
    return {5:"1 D", 15:"1 D", 30:"3 D"}[tf]

INDICATOR_LIST = [
    "Stochastic","MA","EMA","ADX","Bollinger","ROC","OBV","CCI","WWL",
    "VWAP","ATR","Volume_Surge","Momentum","RSI","MACD"
]

class TokenBucket:
    def __init__(self, rate_per_sec=9.0, burst=12):
        self.rate=float(rate_per_sec); self.capacity=float(burst)
        self.tokens=float(burst); self.updated=monotonic(); self.lock=threading.Lock()
    def acquire(self):
        while True:
            with self.lock:
                now=monotonic(); dt=now-self.updated; self.updated=now
                self.tokens=min(self.capacity, self.tokens+dt*self.rate)
                if self.tokens>=1.0:
                    self.tokens-=1.0; return
            sleep(0.008)

BUCKET = TokenBucket(rate_per_sec=9.0, burst=12)

def _fetch_with_retry(symbol, duration, bar_size, retries=3, backoff=0.3):
    for attempt in range(1, retries+1):
        try:
            BUCKET.acquire()
            return td_get_historic(symbol, duration, bar_size)
        except Exception:
            if attempt == retries:
                return None
            sleep(backoff * attempt)

def normalize_df(df):
    if df is None or len(df)==0: return None
    d=df.copy()
    mapping={}
    for c in d.columns:
        lc=c.lower()
        if "time" in lc: mapping[c]="Date"
        elif "open" in lc: mapping[c]="Open"
        elif "high" in lc: mapping[c]="High"
        elif "low" in lc: mapping[c]="Low"
        elif "close" in lc: mapping[c]="Close"
        elif "vol" in lc: mapping[c]="Volume"
    d=d.rename(columns=mapping)
    req=["Date","Open","High","Low","Close"]
    if not all(r in d.columns for r in req): return None
    if "Volume" not in d.columns: d["Volume"]=1000
    d["Date"]=pd.to_datetime(d["Date"], errors="coerce")
    d=d.dropna(subset=["Date"]).set_index("Date").sort_index()
    for c in ["Open","High","Low","Close","Volume"]:
        d[c]=pd.to_numeric(d[c], errors="coerce")
    d=d.dropna()
    return d if len(d)>=20 else None

def fetch_timeframe(symbol, tf):
    raw=_fetch_with_retry(symbol, tf_duration(tf), TF_MAP[tf])
    if raw is None: return None
    d=normalize_df(raw)
    if d is None: return None
    return d.tail(100)

def calc_indicators(df: pd.DataFrame) -> dict:
    ind={}
    if df is None or len(df)<20: return ind
    c=df["Close"]; h=df["High"]; l=df["Low"]; v=df["Volume"]
    delta=c.diff(); gain=delta.where(delta>0,0).rolling(14).mean()
    loss=(-delta.where(delta<0,0)).rolling(14).mean()
    rs=gain/loss; ind["RSI"]=100-(100/(6+rs))
    ema12=c.ewm(span=12).mean(); ema26=c.ewm(span=26).mean()
    macd=ema12-ema26; signal=macd.ewm(span=9).mean(); ind["MACD"]=macd-signal
    low14=l.rolling(14).min(); high14=h.rolling(14).max()
    ind["Stochastic"]=(c-low14)/(high14-low14)*100
    ind["MA"]=c.rolling(20).mean(); ind["EMA"]=c.ewm(span=21).mean()
    hd=h.diff(); ld=l.diff()
    plus_dm=hd.where((hd>ld)&(hd>0),0.0); minus_dm=(-ld).where((ld<hd)&(ld<0),0.0)
    tr1=h-l; tr2=(h-c.shift()).abs(); tr3=(l-c.shift()).abs()
    tr=pd.concat([tr1,tr2,tr3],axis=1).max(axis=1); atr=tr.rolling(14).mean()
    plus_di=100*(plus_dm.rolling(14).mean()/atr); minus_di=100*(minus_dm.rolling(14).mean()/atr)
    dx=100*(plus_di-minus_di).abs()/(plus_di+minus_di); ind["ADX"]=dx.rolling(14).mean()
    ma20=c.rolling(20).mean(); std20=c.rolling(20).std()
    upper=ma20+2*std20; lower=ma20-2*std20; ind["Bollinger"]=(c-ma20)/(upper-lower)*100
    ind["ROC"]=c.pct_change(12)*100
    obv=(np.sign(c.diff())*v).fillna(0).cumsum(); ind["OBV"]=obv.pct_change(10)*100
    tp=(h+l+c)/3.0; sma_tp=tp.rolling(20).mean()
    mad=tp.rolling(20).apply(lambda x: np.abs(x-x.mean()).mean()); ind["CCI"]=(tp-sma_tp)/(0.015*mad)
    hh=h.rolling(14).max(); ll=l.rolling(14).min(); ind["WWL"]=(hh-c)/(hh-ll)*-100
    tpv=(h+l+c)/3.0; ind["VWAP"]=(tpv*v).rolling(20).sum()/v.rolling(20).sum()
    ind["ATR"]=atr
    avg20=v.rolling(20).mean(); ind["Volume_Surge"]=np.clip((v/avg20-0.5)*40,0,100)
    price_mom=c.pct_change(10)*100; avg10=v.rolling(10).mean()
    vol_mom=(v/avg10-1)*100; ind["Momentum"]=50+np.clip(price_mom*0.7+vol_mom*0.3,-50,50)
    return ind

def _price_vs(close_val, ref):
    if pd.isna(ref) or ref==0: return 0
    pct=(close_val-ref)/ref*100.0
    if pct>0.2: return +1
    if pct<-0.2: return -1
    return 0

def dir_rule(name, series: pd.Series, df: pd.DataFrame) -> int:
    if series is None or len(series)==0 or pd.isna(series.iloc[-1]): return 0
    x=float(series.iloc[-1]); close=float(df["Close"].iloc[-1])
    if name=="Stochastic": return +1 if x>=80 else (-1 if x<=20 else 0)
    if name=="MA": return _price_vs(close,x)
    if name=="EMA": return _price_vs(close,x)
    if name=="ADX":
        tail=series.tail(3).dropna()
        if len(tail)>=2 and tail.iloc[-1]>=20:
            return +1 if tail.iloc[-1]>tail.iloc[0] else (-1 if tail.iloc[-1]<tail.iloc[0] else 0)
        return 0
    if name=="Bollinger": return +1 if x>=65 else (-1 if x<=35 else 0)
    if name=="ROC": return +1 if x>0 else (-1 if x<0 else 0)
    if name=="OBV": return +1 if x>0 else (-1 if x<0 else 0)
    if name=="CCI": return +1 if x>=100 else (-1 if x<=-100 else 0)
    if name=="WWL": return +1 if x>=-20 else (-1 if x<=-80 else 0)
    if name=="VWAP": return _price_vs(close,x)
    if name=="ATR":
        tail=series.tail(3).dropna()
        if len(tail)>=2:
            atr_up=tail.iloc[-1]>tail.iloc[0]; price_up=close>df["Close"].iloc[-3]
            if atr_up and not price_up: return -1
            if (not atr_up) and price_up: return +1
        return 0
    if name=="Volume_Surge": return +1 if x>=60 else (-1 if x<=20 else 0)
    if name=="Momentum": return +1 if x>=55 else (-1 if x<=45 else 0)
    if name=="RSI": return +1 if x>=55 else (-1 if x<=45 else 0)
    if name=="MACD": return +1 if x>0 else (-1 if x<0 else 0)
    return 0

def majority_13of15(df: pd.DataFrame):
    inds=calc_indicators(df)
    up=dn=0
    for name in INDICATOR_LIST:
        d=dir_rule(name, inds.get(name), df)
        if d>0: up+=1
        elif d<0: dn+=1
    res="UP" if up>=13 else ("DN" if dn>=13 else "NEU")
    return res

def chunked(iterable, n):
    it=iter(iterable)
    while True:
        block=tuple(islice(it, n))
        if not block: break
        yield block

# Rich console with recording enabled for export_text()
CONSOLE = Console(record=True)

def build_and_print_table(qualified_rows, scan_dt_str):
    """
    qualified_rows: list of tuples (serial_no, symbol, signal) where signal in {"Bullish","Bearish"}.
    scan_dt_str: string like "2025-09-18 14:13:00" used as Rich table title.
    """
    table = Table(title=f"{scan_dt_str}")
    table.add_column("S.No.", justify="right")
    table.add_column("Stock Name", justify="left")
    table.add_column("Signal", justify="left")

    for sno, sym, sig in qualified_rows:
        table.add_row(str(sno), sym, sig)

    CONSOLE.print(table)

    out_file = f"{date.today().isoformat()}_output.txt"
    # Export accumulated console text; keep only last rendered table using the title as anchor.
    text_dump = CONSOLE.export_text(clear=False)
    anchor = f"{scan_dt_str}"
    block = text_dump[text_dump.rfind(anchor):] if anchor in text_dump else text_dump
    with open(out_file, "a", encoding="utf-8") as f:
        f.write(block.strip() + "\n\n")

def run_scan_once(universe):
    jobs=[(s, tf) for s in universe for tf in TIMEFRAMES]
    results=defaultdict(dict)
    with ThreadPoolExecutor(max_workers=44) as pool:
        for idx, batch in enumerate(chunked(jobs, 300), 1):
            futs={pool.submit(fetch_timeframe, s, tf):(s, tf) for s, tf in batch}
            pbar=tqdm(as_completed(futs), total=len(futs), desc=f"Fetching batch {idx}", unit="req", smoothing=0)
            last_t=perf_counter(); ctr=0
            for fut in pbar:
                s, tf=futs[fut]
                try:
                    df=fut.result()
                    if df is not None: results[s][tf]=df
                except Exception:
                    pass
                ctr+=1; now=perf_counter()
                if now-last_t>=0.5:
                    pbar.set_postfix_str(f"{ctr/(now-last_t):0.2f} req/s"); last_t=now; ctr=0
            pbar.close()

    # Build qualified list: only fully bullish or fully bearish across 5/15/30
    qualified_rows=[]
    sno=1
    for sym in tqdm(universe, desc="Evaluating", unit="stk", smoothing=0):
        frames=results.get(sym,{})
        if set(frames.keys())!=set(TIMEFRAMES): continue
        tf_res={tf: majority_13of15(frames[tf]) for tf in TIMEFRAMES}
        all_up=all(v=="UP" for v in tf_res.values())
        all_dn=all(v=="DN" for v in tf_res.values())
        if all_up:
            qualified_rows.append((sno, sym, "Bullish"))
            sno+=1
        elif all_dn:
            qualified_rows.append((sno, sym, "Bearish"))
            sno+=1

    scan_dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    build_and_print_table(qualified_rows, scan_dt_str)

def sleep_until_next_5_min():
    now=datetime.now()
    minute = (now.minute // 5) * 5 + 5
    next_time = now.replace(minute=0 if minute==60 else minute, second=0, microsecond=0)
    if minute==60: next_time = next_time.replace(hour=(now.hour+1)%24)
    delta=(next_time-now).total_seconds()
    if delta < 1: delta = 300
    tqdm.write(f"Sleeping {int(delta)}s until {next_time.strftime('%H:%M')}")
    sleep(delta)

def main_loop():
    CONSOLE.print("13-of-15 (5/15/30) — 5-minute loop with persistent connections")
    while True:
        CONSOLE.print(f"\n=== Scan start {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
        run_scan_once(UNIVERSE_FNO)
        sleep_until_next_5_min()

if __name__ == "__main__":
    main_loop()
