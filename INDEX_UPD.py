#!/usr/bin/env python3
# sector_dashboard_final.py
"""
Final Sector Strength Heatmap & Dashboard
- Polls /api/allIndices for index percentChange
- Polls /api/equity/{SYMBOL} for current stock prices
- Attempts to fetch OHLC (multiple common endpoints); falls back to mock candles if unavailable
- Single unified Dash callback to update everything (avoids duplicate-output errors)
- Auto-refresh every 2 minutes
"""

import math
import random
import traceback
from datetime import datetime, timedelta
from statistics import mean
from typing import Dict, List, Tuple, Optional

import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go

from dash import Dash, dcc, html, dash_table, Input, Output, State

# ---------- Configuration ----------
API_INDEX_URL = "http://localhost:3001/api/allIndices"
API_STOCK_URL = "http://localhost:3001/api/equity/{}"  # supply symbol without -I
POSSIBLE_HISTORY_ENDPOINTS = [
    "http://localhost:3001/api/equity/{}/history",                  # common
    "http://localhost:3001/api/equity/{}/historical",               # alt
    "http://localhost:3001/api/equity/{}/candles",                  # alt
    "http://localhost:3001/api/historical/{}?interval={}&limit={}", # alt pattern
]
REQUEST_TIMEOUT = 5.0
POLL_INTERVAL_MS = 120000  # 2 minutes
STOCK_WEIGHT = 0.7
INDEX_WEIGHT = 0.3

# ---------- Sector -> Stocks (cleaned, no -I) ----------
RAW_SECTOR_STOCKS = {
    'NIFTY IT': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM', 'LTIM', 'MPHASIS', 'COFORGE', 'PERSISTENT', 'CYIENT', 'KPITTECH', 'TATAELXSI', 'SONACOMS', 'KAYNES', 'OFSS'],
    'NIFTY AUTO': ['MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO', 'TVSMOTOR', 'BHARATFORG', 'EICHERMOT', 'ASHOKLEY', 'BOSCHLTD', 'TIINDIA', 'MOTHERSON'],
    'NIFTY BANK': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'PNB', 'BANKBARODA', 'CANBK', 'IDFCFIRSTB', 'INDUSINDBK', 'AUBANK', 'FEDERALBNK'],
    'NIFTY PHARMA': ['SUNPHARMA', 'DRREDDY', 'CIPLA', 'LUPIN', 'AUROPHARMA', 'TORNTPHARM', 'GLENMARK', 'ALKEM', 'LAURUSLABS', 'BIOCON', 'ZYDUSLIFE', 'MANKIND', 'SYNGENE', 'PPLPHARMA'],
    'NIFTY ENERGY': ['RELIANCE', 'NTPC', 'BPCL', 'IOC', 'ONGC', 'GAIL', 'HINDPETRO', 'ADANIGREEN', 'ADANIENSOL', 'JSWENERGY', 'COALINDIA', 'TATAPOWER', 'SUZLON', 'PETRONET', 'OIL', 'POWERGRID', 'NHPC', 'ADANIPORTS', 'ABB', 'SIEMENS', 'CGPOWER', 'INOXWIND'],
    'NIFTY METAL': ['TATASTEEL', 'JSWSTEEL', 'SAIL', 'JINDALSTEL', 'HINDALCO', 'NMDC'],
    'NIFTY NON-CYCLICAL CONSUMER': ['HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'TATACONSUM', 'DABUR', 'AMBER', 'UNITDSPR', 'GODREJCP', 'MARICO', 'COLPAL', 'UPL', 'VBL'],
    'NIFTY PSU BANK': ['SBIN', 'PNB', 'BANKBARODA', 'CANBK', 'UNIONBANK', 'BANKINDIA'],
    'NIFTY FINANCIAL SERVICES': ['BAJFINANCE', 'SHRIRAMFIN', 'CHOLAFIN', 'HDFCLIFE', 'ICICIPRULI', 'ETERNAL'],
    'NIFTY REALTY': ['DLF', 'LODHA', 'PRESTIGE', 'GODREJPROP', 'OBEROIRLTY', 'PHOENIXLDT', 'NCC', 'NBCC'],
    'NIFTY PSE': ['BEL', 'BHEL', 'NHPC', 'GAIL', 'IOC', 'NTPC', 'POWERGRID', 'HINDPETRO', 'OIL', 'RECLTD', 'ONGC', 'NMDC', 'BPCL', 'HAL', 'RVNL', 'PFC', 'COALINDIA', 'IRCTC', 'IRFC'],
    'NIFTY COMMODITIES': ['AMBUJACEM', 'APLAPOLLO', 'ULTRACEMCO', 'SHREECEM', 'JSWSTEEL', 'HINDALCO', 'NHPC', 'IOC', 'NTPC', 'HINDPETRO', 'ADANIGREEN', 'OIL', 'VEDL', 'PIIND', 'ONGC', 'NMDC', 'UPL', 'BPCL', 'JSWENERGY', 'GRASIM', 'RELIANCE', 'TORNTPOWER', 'TATAPOWER', 'COALINDIA', 'PIDILITIND', 'SRF', 'ADANIENSOL', 'JINDALSTEL', 'TATASTEEL', 'HINDALCO'],
    'NIFTY CONSUMER DURABLES': ['TITAN', 'DIXON', 'HAVELLS', 'CROMPTON', 'POLYCAB', 'EXIDEIND', 'AMBER', 'KAYNES', 'VOLTAS', 'PGEL', 'BLUESTARCO'],
    'NIFTY HEALTHCARE INDEX': ['SUNPHARMA', 'DIVISLAB', 'CIPLA', 'TORNTPHARM', 'MAXHEALTH', 'APOLLOHOSP', 'DRREDDY', 'MANKIND', 'ZYDUSLIFE', 'LUPIN', 'FORTIS', 'ALKEM', 'AUROPHARMA', 'GLENMARK', 'BIOCON', 'LAURUSLABS', 'SYNGENE', 'GRANULES'],
    'NIFTY CAPITAL MARKETS': ['HDFCAMC', 'BSE', '360ONE', 'MCX', 'CDSL', 'NUVAMA', 'ANGELONE', 'KFINTECH', 'CAMS', 'IEX'],
    'NIFTY PRIVATE BANK': ['HDFCBANK', 'ICICIBANK', 'KOTAKBANK', 'AXISBANK', 'YESBANK', 'IDFCFIRSTB', 'INDUSINDBK', 'FEDERALBNK', 'BANDHANBNK', 'RBLBANK'],
    'NIFTY OIL AND GAS': ['RELIANCE', 'ONGC', 'IOC', 'BPCL', 'GAIL', 'HINDPETRO', 'OIL', 'PETRONET', 'IGL'],
    'NIFTY INDIA DEFENCE': ['HAL', 'BEL', 'SOLARINDS', 'MAZDOCK', 'BDL'],
    'NIFTY CORE HOUSING': ['ULTRACEMCO', 'ASIANPAINT', 'GRASIM', 'DLF', 'AMBUJACEM', 'LODHA', 'DIXON', 'POLYCAB', 'SHREECEM', 'HAVELLS', 'PRESTIGE', 'GODREJPROP', 'OBEROIRLTY', 'PHOENIXLTD', 'VOLTAS', 'DALBHARAT', 'KEI', 'BLUESTARCO', 'LICHSGFIN', 'PNBHOUSING', 'CROMPTON'],
    'NIFTY SERVICES SECTOR': ['HDFCBANK', 'BHARTIARTL', 'TCS', 'ICICIBANK', 'SBIN', 'INFY', 'BAJFINANCE', 'HCLTECH', 'KOTAKBANK', 'AXISBANK', 'BAJAJFINSV', 'NTPC', 'ZOMATO', 'ADANIPORTS', 'DMART', 'POWERGRID', 'WIPRO', 'INDIGO', 'JIOFINSERV', 'SBILIFE', 'HDFCLIFE', 'LTIM', 'TECHM', 'TATAPOWER', 'SHRIRAMFIN', 'GAIL', 'MAXHEALTH', 'APOLLOHOSP', 'NAUKRI', 'INDUSINDBK'],
    'Financial Services 2550': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'BAJFINANCE', 'KOTAKBANK', 'AXISBANK', 'BAJAJFINSV', 'JIOFIN', 'SBILIFE', 'HDFCLIFE', 'PFC', 'CHOLAFIN', 'HDFCAMC', 'SHRIRAMFIN', 'MUTHOOTFIN', 'RECLTD', 'ICICIGI', 'ICICIPRULI', 'SBICARD', 'LICHSGFIN'],
    'NIFTY INDIA TOURISM': ['INDIGO', 'INDHOTEL', 'IRCTC', 'JUBLFOOD']
}

SECTOR_STOCKS = {k: list(dict.fromkeys(v)) for k, v in RAW_SECTOR_STOCKS.items()}  # dedupe preserve order

# ---------- HTTP helpers ----------
def fetch_indices() -> List[Dict]:
    """Fetch /api/allIndices. Try to parse both {data: [...]} and list responses."""
    try:
        r = requests.get(API_INDEX_URL, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        payload = r.json()
        if isinstance(payload, dict) and "data" in payload:
            return payload["data"] if isinstance(payload["data"], list) else []
        if isinstance(payload, list):
            return payload
    except Exception:
        # fallback: try to load a local file (helpful in offline/debug)
        try:
            import json, os
            local_paths = ["./all_indices.json", "/mnt/data/response_1764827980343 (2).json"]
            for p in local_paths:
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as fh:
                        content = json.load(fh)
                        if isinstance(content, dict) and "data" in content:
                            return content["data"]
                        if isinstance(content, list):
                            return content
        except Exception:
            pass
    return []


def fetch_stock(symbol: str) -> Optional[Dict]:
    """Fetch /api/equity/{SYMBOL} (no -I). Return JSON or None."""
    try:
        r = requests.get(API_STOCK_URL.format(symbol), timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def try_fetch_candles(symbol: str, interval: str = "5m", limit: int = 200) -> Optional[List[Dict]]:
    """
    Try several plausible endpoints for OHLC candles.
    Expected return: list of dicts with keys: time, open, high, low, close, volume (time as ISO or ms)
    If none available, return None (caller will mock).
    """
    # try patterns
    for pattern in POSSIBLE_HISTORY_ENDPOINTS:
        try:
            url = pattern.format(symbol, interval, limit)
            r = requests.get(url, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                continue
            data = r.json()
            # try common wrappers
            if isinstance(data, dict):
                # some apis: {"data": [{"o":..., "c":...}, ...]}
                if "data" in data and isinstance(data["data"], list):
                    return normalize_candles(data["data"])
                # maybe returns list as value of 'candles' or 'history'
                for key in ("candles", "history", "candlestick", "data"):
                    if key in data and isinstance(data[key], list):
                        return normalize_candles(data[key])
                # some quote structure: { "candles": [[ts,o,h,l,c,v], ...] }
                if "candles" in data and isinstance(data["candles"], list) and data["candles"] and isinstance(data["candles"][0], list):
                    # convert list-of-lists
                    return list_of_lists_to_candles(data["candles"])
            elif isinstance(data, list):
                return normalize_candles(data)
        except Exception:
            continue
    return None


def normalize_candles(raw_list: List[Dict]) -> List[Dict]:
    """
    Convert various candle dict shapes into a standard list of dicts:
    { "time": "...", "open": x, "high": x, "low": x, "close": x, "volume": x }
    """
    out = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        # common key combos
        t = item.get("time") or item.get("date") or item.get("dt") or item.get("timestamp") or item.get("ts") or item.get("datetime")
        o = item.get("open") or item.get("o")
        h = item.get("high") or item.get("h")
        l = item.get("low") or item.get("l")
        c = item.get("close") or item.get("c")
        v = item.get("volume") or item.get("v") or item.get("vol")
        if None in (o, h, l, c):
            # skip items missing core OHLC
            continue
        out.append({"time": t, "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": float(v) if v is not None else 0.0})
    return out


def list_of_lists_to_candles(lst: List[List]) -> List[Dict]:
    """Convert list-of-lists like [ [ts, o, h, l, c, v], ... ] to standard dicts."""
    out = []
    for row in lst:
        if len(row) >= 5:
            ts = row[0]
            o = float(row[1])
            h = float(row[2])
            l = float(row[3])
            c = float(row[4])
            v = float(row[5]) if len(row) > 5 else 0.0
            out.append({"time": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
    return out


def mock_candles_from_last(last_price: float, n: int = 60, interval_minutes: int = 5) -> List[Dict]:
    """Create a small synthetic candle series from last price (useful when history endpoint unavailable)."""
    now = datetime.utcnow()
    price = float(last_price or 100.0)
    candles = []
    for i in range(n):
        t = now - timedelta(minutes=(n - i) * interval_minutes)
        # small random walk
        o = price * (1 + random.uniform(-0.0025, 0.0025))
        c = o * (1 + random.uniform(-0.005, 0.005))
        h = max(o, c) * (1 + random.uniform(0, 0.0015))
        l = min(o, c) * (1 - random.uniform(0, 0.0015))
        vol = random.uniform(1000, 20000)
        candles.append({"time": t.isoformat(), "open": o, "high": h, "low": l, "close": c, "volume": vol})
        price = c
    return candles

# ---------- Parsing helpers ----------
def get_stock_pct(payload: Optional[Dict]) -> Optional[float]:
    """
    Extract percent change from equity payload.
    Prefer priceInfo.pChange, else compute from lastPrice/previousClose.
    """
    if not isinstance(payload, dict):
        return None
    priceInfo = payload.get("priceInfo") or (payload.get("data") or {}).get("priceInfo") if isinstance(payload.get("data"), dict) else payload.get("priceinfo") or {}
    if isinstance(priceInfo, dict) and "pChange" in priceInfo and priceInfo["pChange"] is not None:
        try:
            return float(priceInfo["pChange"])
        except Exception:
            pass
    # fallback compute
    try:
        # payload may be flattened
        pi = payload.get("priceInfo") or payload
        last = pi.get("lastPrice") or pi.get("last") or payload.get("lastPrice") or payload.get("last")
        prev = pi.get("previousClose") or pi.get("previousDayVal") or payload.get("previousClose")
        if last is None or prev is None:
            return None
        last = float(last)
        prev = float(prev)
        if prev == 0:
            return None
        return ((last - prev) / prev) * 100.0
    except Exception:
        return None


def get_index_pct(index_name: str, all_indices: List[Dict]) -> Optional[float]:
    """Search all_indices for exact index name and return percentChange."""
    for entry in all_indices:
        if not isinstance(entry, dict):
            continue
        if entry.get("index") == index_name or entry.get("indexSymbol") == index_name:
            v = entry.get("percentChange") or entry.get("perChange") or entry.get("perChange365d")
            try:
                return float(v)
            except Exception:
                # fallback compute from last/previousDayVal
                last = entry.get("last")
                prev = entry.get("previousDayVal") or entry.get("previousClose")
                try:
                    if last is not None and prev:
                        return ((float(last) - float(prev)) / float(prev)) * 100.0
                except Exception:
                    return None
    return None

# ---------- Sector stats ----------
def compute_sector_stats(all_indices: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
    """
    For each sector:
      - fetch each stock's % change
      - compute avg stock %, get index %
      - sector_score = STOCK_WEIGHT*avg_stock_pct + INDEX_WEIGHT*index_pct
    Returns DataFrame sorted desc by sector_score and dict of best/worst pairs.
    """
    rows = []
    best_worst = {}
    for sector, symbols in SECTOR_STOCKS.items():
        stock_pcts = []
        pairs = []
        for s in symbols:
            payload = fetch_stock(s)
            pct = get_stock_pct(payload)
            if pct is None:
                continue
            stock_pcts.append(pct)
            pairs.append((s, pct))
        if not stock_pcts:
            continue
        avg_stock_pct = mean(stock_pcts)
        idx_pct = get_index_pct(sector, all_indices)
        sector_score = (STOCK_WEIGHT * avg_stock_pct + INDEX_WEIGHT * idx_pct) if idx_pct is not None else avg_stock_pct
        pairs.sort(key=lambda x: x[1], reverse=True)
        best_sym, best_pct = pairs[0]
        worst_sym, worst_pct = pairs[-1]
        rows.append({
            "sector": sector,
            "sector_score": sector_score,
            "index_pct": idx_pct if idx_pct is not None else float("nan"),
            "avg_stock_pct": avg_stock_pct,
            "best_stock": best_sym,
            "best_stock_pct": best_pct,
            "worst_stock": worst_sym,
            "worst_stock_pct": worst_pct,
            "num_stocks": len(stock_pcts),
        })
        best_worst[sector] = (best_sym, best_pct, worst_sym, worst_pct)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, best_worst
    df.sort_values("sector_score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df, best_worst

# ---------- DASH App ----------
app = Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        html.H2("Sector Strength Heatmap & Dashboard", style={"textAlign": "center"}),
        html.Div(id="last-updated", style={"textAlign": "center", "marginBottom": 8}),
        dcc.Interval(id="refresh-interval", interval=POLL_INTERVAL_MS, n_intervals=0),

        # Heatmap area
        html.Div([
            dcc.Graph(id="sector-heatmap", config={"displayModeBar": True}),
        ], style={"width": "100%"}),

        html.Div(style={"display": "flex", "gap": "24px", "marginTop": "8px"}, children=[
            # Sector details column
            html.Div([
                html.H3("Sector Details", style={"marginTop": 6}),
                html.Div(id="sector-description", children="Click a sector tile to see stocks", style={"marginBottom": 6}),
                dash_table.DataTable(
                    id="sector-stock-table",
                    columns=[
                        {"name": "Rank", "id": "rank"},
                        {"name": "Stock", "id": "stock"},
                        {"name": "%Change", "id": "pct"},
                        {"name": "VWAP %", "id": "vwap_pct"},
                        {"name": "Rank (sector)", "id": "sector_rank"},
                    ],
                    style_cell={"textAlign": "center", "padding": "6px"},
                    style_header={"fontWeight": "bold"},
                    page_size=25,
                ),
            ], style={"flex": "1 1 60%"}),

            # Right column: top/worst and candlestick
            html.Div([
                html.H3("Top / Worst Sectors"),
                dash_table.DataTable(
                    id="top-worst-table",
                    columns=[
                        {"name": "Type", "id": "type"},
                        {"name": "Rank", "id": "rank"},
                        {"name": "Sector", "id": "sector"},
                        {"name": "Score (%)", "id": "score"},
                        {"name": "Index %", "id": "index_pct"},
                        {"name": "Avg Stock %", "id": "avg_stock_pct"},
                    ],
                    style_cell={"textAlign": "left", "padding": "6px"},
                    style_header={"fontWeight": "bold"},
                    page_size=8,
                ),
                html.Hr(),
                html.H4(id="candlestick-title", children="Select a stock row to show candlestick"),
                dcc.Loading(dcc.Graph(id="candlestick"), type="default"),
            ], style={"flex": "0 0 38%"}),
        ]),

        # hidden store to carry last sector_df if needed
        dcc.Store(id="last-sector-df"),
    ],
    style={"width": "95%", "margin": "8px auto", "fontFamily": "Arial"}
)


# ---------- Single Unified Callback ----------
@app.callback(
    Output("sector-heatmap", "figure"),
    Output("top-worst-table", "data"),
    Output("sector-stock-table", "data"),
    Output("sector-description", "children"),
    Output("candlestick", "figure"),
    Output("candlestick-title", "children"),
    Output("last-updated", "children"),
    Input("refresh-interval", "n_intervals"),
    State("sector-heatmap", "clickData"),
    prevent_initial_call=False,
)
def update_all(n_intervals, clickData):
    """
    Unified update:
      - refresh indices & per-stock percentages
      - build heatmap
      - populate top/worst tables
      - if clickData contains a sector, generate sector stock table
      - candlestick: if user clicked a row previously, we cannot detect that here; to keep single callback we will
        show candlestick for best stock of clicked sector (if any). This keeps logic simple and avoids duplicate outputs.
    """
    try:
        all_indices = fetch_indices()
        sector_df, best_worst = compute_sector_stats(all_indices)

        if sector_df.empty:
            empty_fig = px.imshow([[0]], text_auto=True, title="No data")
            return empty_fig, [], [], "No sector data available", go.Figure(), "Candlestick", f"Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}"

        # Heatmap: single row of sector scores (we'll make a horizontal heatmap)
        z = sector_df["sector_score"].tolist()
        sectors = sector_df["sector"].tolist()
        # Create a 2D array with shape (1, n)
        fig = go.Figure(data=go.Heatmap(
            z=[z],
            x=sectors,
            y=["Score"],
            colorscale="RdYlGn",
            reversescale=False,
            colorbar=dict(title="Score"),
            zmid=0.0
        ))
        fig.update_layout(title="Sector Scores (higher = stronger)", height=420, margin=dict(l=60, r=60, t=60, b=140))
        fig.update_xaxes(tickangle=-45)

        # Top 3 & Worst 5 table data
        top3 = sector_df.head(3)
        top_data = []
        for idx, row in top3.iterrows():
            top_data.append({
                "type": "Top",
                "rank": int(idx + 1),
                "sector": row["sector"],
                "score": f"{row['sector_score']:+.2f}",
                "index_pct": f"{row['index_pct']:+.2f}" if not math.isnan(row["index_pct"]) else "N/A",
                "avg_stock_pct": f"{row['avg_stock_pct']:+.2f}",
            })
        worst5 = sector_df.tail(5).sort_values("sector_score", ascending=True)
        for i, row in enumerate(worst5.itertuples(), start=1):
            top_data.append({
                "type": "Worst",
                "rank": i,
                "sector": row.sector,
                "score": f"{row.sector_score:+.2f}",
                "index_pct": f"{row.index_pct:+.2f}" if not math.isnan(row.index_pct) else "N/A",
                "avg_stock_pct": f"{row.avg_stock_pct:+.2f}",
            })

        # Determine selected sector from clickData (if any)
        selected_sector = None
        if clickData and isinstance(clickData, dict):
            # clickData for heatmap includes 'points' with 'x' sector name
            pts = clickData.get("points")
            if pts and isinstance(pts, list) and len(pts) > 0:
                selected_sector = pts[0].get("x")

        # Build sector stock table for selected sector (default to top sector)
        if not selected_sector:
            selected_sector = sector_df.iloc[0]["sector"]
        sector_descr = f"Stocks in {selected_sector}"
        # find row
        sector_row = sector_df[sector_df["sector"] == selected_sector]
        if sector_row.empty:
            sector_rows = []
        else:
            # rebuild list of stock details for selected sector
            stocks = SECTOR_STOCKS.get(selected_sector, [])
            stock_rows = []
            for i, s in enumerate(stocks, start=1):
                payload = fetch_stock(s)
                pct = get_stock_pct(payload) or 0.0
                # vwap%: compare vwap with previousClose if available (not always present)
                vwap_pct = None
                try:
                    pi = payload.get("priceInfo") if isinstance(payload, dict) else {}
                    vwap = pi.get("vwap") if isinstance(pi, dict) else None
                    prev = pi.get("previousClose") if isinstance(pi, dict) else None
                    if vwap is not None and prev:
                        vwap_pct = ((float(vwap) - float(prev)) / float(prev)) * 100.0
                except Exception:
                    vwap_pct = None
                stock_rows.append({
                    "rank": i,
                    "stock": s,
                    "pct": f"{pct:+.2f}",
                    "vwap_pct": f"{vwap_pct:+.2f}" if vwap_pct is not None else "N/A",
                    "sector_rank": i
                })
            sector_rows = stock_rows

        # Build candlestick: choose best stock of selected sector (guaranteed exists)
        candlestick_fig = go.Figure()
        candlestick_title = "Candlestick"
        try:
            # pick best stock per data (if available)
            best_stock = sector_row.iloc[0]["best_stock"] if not sector_row.empty else (SECTOR_STOCKS[selected_sector][0] if SECTOR_STOCKS.get(selected_sector) else None)
            if best_stock:
                # try fetch candles
                candles = try_fetch_candles(best_stock, interval="5m", limit=200)
                if not candles:
                    # fallback to mock series using lastPrice
                    payload = fetch_stock(best_stock)
                    last = None
                    if payload and isinstance(payload, dict):
                        last_price = None
                        try:
                            last_price = payload.get("priceInfo", {}).get("lastPrice") or payload.get("priceInfo", {}).get("last")
                        except Exception:
                            last_price = None
                        last = float(last_price) if last_price is not None else None
                    candles = mock_candles_from_last(last or 100.0, n=90, interval_minutes=5)
                # convert to dataframe and plot
                dfc = pd.DataFrame(candles)
                # ensure time is datetime
                try:
                    dfc["time"] = pd.to_datetime(dfc["time"])
                except Exception:
                    pass
                candlestick_fig = go.Figure(data=[go.Candlestick(
                    x=dfc["time"],
                    open=dfc["open"],
                    high=dfc["high"],
                    low=dfc["low"],
                    close=dfc["close"],
                    increasing_line_color="green",
                    decreasing_line_color="red"
                )])
                candlestick_fig.update_layout(title=f"{best_stock} - Recent candles (5m)", height=360, margin=dict(l=40, r=20, t=40, b=40))
                candlestick_title = f"Candlestick: {best_stock}"
        except Exception:
            candlestick_fig = go.Figure()
            candlestick_title = "Candlestick (error)"

        last = f"Last updated: {datetime.now():%Y-%m-%d %H:%M:%S}"
        return fig, top_data, sector_rows, sector_descr, candlestick_fig, candlestick_title, last

    except Exception as e:
        traceback.print_exc()
        empty_fig = px.imshow([[0]], text_auto=True, title="Error")
        return empty_fig, [], [], "Error updating", go.Figure(), "Candlestick", f"Error: {e}"


# ---------- Run ----------
if __name__ == "__main__":
    # Dash 3.x uses app.run()
    app.run(debug=True, port=8050, host="0.0.0.0")
