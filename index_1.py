# sector_dashboard_full.py
import os
import json
import urllib.parse
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import requests
import pandas as pd
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px

# ----------------------------
# CONFIG
# ----------------------------
API_INDEX_URL = "http://localhost:3001/api/allIndices"
API_STOCK_URL = "http://localhost:3001/api/equity/{}"
REFRESH_MS = 120_000  # 2 minutes
VOLUME_CACHE_FILE = "volume_cache.json"
VOLUME_HISTORY_LIMIT = 20

# ----------------------------
# RAW_SECTOR_STOCKS (unchanged)
# ----------------------------
RAW_SECTOR_STOCKS = {
    'NIFTY IT': [
        'TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM', 'LTIM', 'MPHASIS',
        'COFORGE', 'PERSISTENT', 'CYIENT', 'KPITTECH', 'TATAELXSI',
        'SONACOMS', 'KAYNES', 'OFSS'
    ],
    'NIFTY AUTO': [
        'MARUTI', 'TATAMOTORS', 'M&M', 'BAJAJ-AUTO', 'HEROMOTOCO',
        'TVSMOTOR', 'BHARATFORG', 'EICHERMOT', 'ASHOKLEY', 'BOSCHLTD',
        'TIINDIA', 'MOTHERSON'
    ],
    'NIFTY BANK': [
        'HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK', 'PNB',
        'BANKBARODA', 'CANBK', 'IDFCFIRSTB', 'INDUSINDBK', 'AUBANK',
        'FEDERALBNK'
    ],
    'NIFTY PHARMA': [
        'SUNPHARMA', 'DRREDDY', 'CIPLA', 'LUPIN', 'AUROPHARMA',
        'TORNTPHARM', 'GLENMARK', 'ALKEM', 'LAURUSLABS', 'BIOCON',
        'ZYDUSLIFE', 'MANKIND', 'SYNGENE', 'PPLPHARMA'
    ],
    'NIFTY ENERGY': [
        'RELIANCE', 'NTPC', 'BPCL', 'IOC', 'ONGC', 'GAIL', 'HINDPETRO',
        'ADANIGREEN', 'ADANIENSOL', 'JSWENERGY', 'COALINDIA', 'TATAPOWER',
        'SUZLON', 'PETRONET', 'OIL', 'POWERGRID', 'NHPC', 'ADANIPORTS',
        'ABB', 'SIEMENS', 'CGPOWER', 'INOXWIND'
    ],
    'NIFTY METAL': [
        'TATASTEEL', 'JSWSTEEL', 'SAIL', 'JINDALSTEL', 'HINDALCO', 'NMDC'
    ],
    'NIFTY NON-CYCLICAL CONSUMER': [
        'HINDUNILVR', 'ITC', 'NESTLEIND', 'BRITANNIA', 'TATACONSUM',
        'DABUR', 'AMBER', 'UNITDSPR', 'GODREJCP', 'MARICO', 'COLPAL',
        'UPL', 'VBL'
    ],
    'NIFTY PSU BANK': [
        'SBIN', 'PNB', 'BANKBARODA', 'CANBK', 'UNIONBANK', 'BANKINDIA'
    ],
    'NIFTY FINANCIAL SERVICES': [
        'BAJFINANCE', 'SHRIRAMFIN', 'CHOLAFIN', 'HDFCLIFE',
        'ICICIPRULI', 'ETERNAL'
    ],
    'NIFTY REALTY': [
        'DLF', 'LODHA', 'PRESTIGE', 'GODREJPROP', 'OBEROIRLTY',
        'PHOENIXLTD', 'NCC', 'NBCC'
    ],
    'NIFTY PSE': [
        'BEL', 'BHEL', 'NHPC', 'GAIL', 'IOC', 'NTPC', 'POWERGRID',
        'HINDPETRO', 'OIL', 'RECLTD', 'ONGC', 'NMDC', 'BPCL', 'HAL',
        'RVNL', 'PFC', 'COALINDIA', 'IRCTC', 'IRFC'
    ],
    'NIFTY COMMODITIES': [
        'AMBUJACEM', 'APLAPOLLO', 'ULTRACEMCO', 'SHREECEM', 'JSWSTEEL',
        'HINDALCO', 'IOC', 'NTPC', 'HINDPETRO', 'ADANIGREEN',
        'OIL', 'VEDL', 'PIIND', 'ONGC', 'NMDC', 'UPL', 'BPCL',
        'JSWENERGY', 'GRASIM', 'RELIANCE', 'TORNTPOWER', 'TATAPOWER',
        'COALINDIA', 'PIDILITIND', 'SRF', 'ADANIENSOL', 'JINDALSTEL',
        'TATASTEEL'
    ],
    'NIFTY CONSUMER DURABLES': [
        'TITAN', 'DIXON', 'HAVELLS', 'CROMPTON', 'POLYCAB', 'EXIDEIND',
        'AMBER', 'KAYNES', 'VOLTAS', 'PGEL', 'BLUESTARCO'
    ]
}

# ----------------------------
# Volume cache helpers
# ----------------------------
def load_volume_cache() -> Dict[str, List[Dict[str, Any]]]:
    try:
        if os.path.exists(VOLUME_CACHE_FILE):
            with open(VOLUME_CACHE_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def save_volume_cache(cache: Dict[str, List[Dict[str, Any]]]):
    try:
        with open(VOLUME_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass

def append_volume_to_cache(symbol: str, volume: int):
    cache = load_volume_cache()
    lst = cache.get(symbol, [])
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if lst and lst[-1].get("date") == today:
        lst[-1]["volume"] = volume
    else:
        lst.append({"date": today, "volume": volume})
    if len(lst) > VOLUME_HISTORY_LIMIT:
        lst = lst[-VOLUME_HISTORY_LIMIT:]
    cache[symbol] = lst
    save_volume_cache(cache)

def get_avg20_from_cache(symbol: str) -> float:
    cache = load_volume_cache()
    lst = cache.get(symbol, [])
    if not lst:
        return 0.0
    vols = [entry.get("volume", 0) for entry in lst if entry.get("volume") is not None]
    return float(sum(vols) / len(vols)) if vols else 0.0

# ----------------------------
# API helpers (robust + debug)
# ----------------------------
def get_indices() -> Dict[str, float]:
    try:
        r = requests.get(API_INDEX_URL, timeout=6)
        if r.status_code != 200:
            return {}
        j = r.json()
        if "data" not in j:
            return {}
        return {item.get("index"): float(item.get("percentChange", 0) or 0) for item in j["data"]}
    except Exception:
        return {}

def get_stock_info(sym: str) -> Optional[Dict[str, Any]]:
    try:
        url = API_STOCK_URL.format(urllib.parse.quote(sym, safe=''))
        r = requests.get(url, timeout=6)
        if r.status_code != 200:
            return None
        j = r.json()
        if "priceInfo" not in j:
            return None

        price = j.get("priceInfo", {})
        pre = j.get("preOpenMarket", {}) or {}
        volume = pre.get("totalTradedVolume", 0) or 0

        # persist today's volume for avg20 calculation across days
        try:
            append_volume_to_cache(sym, int(volume))
        except Exception:
            pass

        avg20 = get_avg20_from_cache(sym)

        return {
            "symbol": sym,
            "pchange": float(price.get("pChange", 0.0) or 0.0),
            "price": price.get("lastPrice"),
            "vwap": price.get("vwap"),
            "volume": int(volume),
            "day_high": price.get("intraDayHighLow", {}).get("max"),
            "day_low": price.get("intraDayHighLow", {}).get("min"),
            "avg20": float(avg20)
        }
    except Exception:
        return None

# ----------------------------
# Sector computations (always create row per RAW sector)
# ----------------------------
def compute_sector_table():
    index_map = get_indices()
    rows = []

    for sector, stocks in RAW_SECTOR_STOCKS.items():
        stock_changes = []
        for s in stocks:
            info = get_stock_info(s)
            if info:
                stock_changes.append(info.get("pchange", 0.0))

        avg_stock = (sum(stock_changes) / len(stock_changes)) if stock_changes else 0.0
        index_pct = index_map.get(sector, 0.0)
        score = 0.7 * avg_stock + 0.3 * index_pct

        rows.append({
            "Sector": sector,
            "Index %": round(index_pct, 2),
            "Avg Stock %": round(avg_stock, 2),
            "Score": round(score, 2)
        })

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame([{"Sector": s, "Index %": 0.0, "Avg Stock %": 0.0, "Score": 0.0} for s in RAW_SECTOR_STOCKS.keys()])
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    return df

def get_sector_volume_stats(sector: str):
    stocks = RAW_SECTOR_STOCKS.get(sector, [])
    vols = []
    for s in stocks:
        info = get_stock_info(s)
        if info:
            vols.append(info.get("volume", 0))
    if not vols:
        return 0.0, 0
    return float(sum(vols) / len(vols)), int(max(vols))

# ----------------------------
# Stock / Scanner tables
# ----------------------------
def get_sector_stock_table(sector: str):
    stocks = RAW_SECTOR_STOCKS.get(sector, [])
    rows = []
    for s in stocks:
        info = get_stock_info(s)
        if not info:
            continue
        rows.append({
            "Stock": s,
            "%Change": round(info.get("pchange", 0.0), 2),
            "LTP": info.get("price"),
            "VWAP": info.get("vwap"),
            "Volume": info.get("volume"),
            "Avg20": int(info.get("avg20", 0.0))
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["Rank"] = df["%Change"].rank(ascending=False, method="min")
    return df.sort_values("%Change", ascending=False).reset_index(drop=True)

def get_high_volume_rising_for_sector(sector: str, pchange_threshold=0.8, vol_multiplier=1.5):
    avg_vol, _ = get_sector_volume_stats(sector)
    rows = []
    for s in RAW_SECTOR_STOCKS.get(sector, []):
        info = get_stock_info(s)
        if not info:
            continue
        pch = info.get("pchange", 0.0)
        vol = info.get("volume", 0)
        avg20 = info.get("avg20", 0.0)
        cond_sector = (avg_vol > 0 and vol > avg_vol * vol_multiplier)
        cond_avg20 = (avg20 > 0 and vol > avg20 * vol_multiplier)
        if pch > pchange_threshold and (cond_sector or cond_avg20):
            rows.append({
                "Stock": s,
                "%Change": round(pch, 2),
                "Volume": vol,
                "SectorAvgVol": int(avg_vol) if avg_vol else None,
                "Avg20": int(avg20) if avg20 else None,
                "VWAP": info.get("vwap"),
                "LTP": info.get("price")
            })
    return pd.DataFrame(rows).sort_values("%Change", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()

def get_high_volume_falling_for_sector(sector: str, pchange_threshold=-0.8, vol_multiplier=1.5):
    avg_vol, _ = get_sector_volume_stats(sector)
    rows = []
    for s in RAW_SECTOR_STOCKS.get(sector, []):
        info = get_stock_info(s)
        if not info:
            continue
        pch = info.get("pchange", 0.0)
        vol = info.get("volume", 0)
        avg20 = info.get("avg20", 0.0)
        cond_sector = (avg_vol > 0 and vol > avg_vol * vol_multiplier)
        cond_avg20 = (avg20 > 0 and vol > avg20 * vol_multiplier)
        if pch < pchange_threshold and (cond_sector or cond_avg20):
            rows.append({
                "Stock": s,
                "%Change": round(pch, 2),
                "Volume": vol,
                "SectorAvgVol": int(avg_vol) if avg_vol else None,
                "Avg20": int(avg20) if avg20 else None,
                "VWAP": info.get("vwap"),
                "LTP": info.get("price")
            })
    return pd.DataFrame(rows).sort_values("%Change", ascending=True).reset_index(drop=True) if rows else pd.DataFrame()

def get_volume_shocks_all(vol_multiplier=2.0):
    rows = []
    sector_avg_map = {}
    for sector in RAW_SECTOR_STOCKS.keys():
        sector_avg_map[sector] = get_sector_volume_stats(sector)[0]

    for sector, stocks in RAW_SECTOR_STOCKS.items():
        avg_vol = sector_avg_map.get(sector, 0)
        for s in stocks:
            info = get_stock_info(s)
            if not info:
                continue
            vol = info.get("volume", 0)
            avg20 = info.get("avg20", 0.0)
            ratio_sector = (vol / avg_vol) if avg_vol and avg_vol > 0 else None
            ratio_avg20 = (vol / avg20) if avg20 and avg20 > 0 else None
            triggered = False
            if ratio_sector and ratio_sector >= vol_multiplier:
                triggered = True
            if ratio_avg20 and ratio_avg20 >= vol_multiplier:
                triggered = True
            if triggered:
                rows.append({
                    "Stock": s,
                    "Sector": sector,
                    "Volume": vol,
                    "SectorAvg": int(avg_vol) if avg_vol else None,
                    "Avg20": int(avg20) if avg20 else None,
                    "RatioSector": round(ratio_sector, 2) if ratio_sector else None,
                    "RatioAvg20": round(ratio_avg20, 2) if ratio_avg20 else None,
                    "%Change": round(info.get("pchange", 0.0), 2)
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(["RatioAvg20", "RatioSector"], ascending=False, na_position="last").reset_index(drop=True)

# ----------------------------
# Market Movers (Gainers/Losers)
# ----------------------------
def get_market_movers_dataframe():
    rows = []
    for sector, stocks in RAW_SECTOR_STOCKS.items():
        for s in stocks:
            info = get_stock_info(s)
            if not info:
                continue
            rows.append({
                "Stock": s,
                "%Change": info["pchange"],
                "LTP": info["price"],
                "Volume": info["volume"],
                "Avg20": info["avg20"],
                "Sector": sector
            })
    return pd.DataFrame(rows)

def get_top_gainers(limit=10):
    df = get_market_movers_dataframe()
    if df.empty:
        return df
    return df.sort_values("%Change", ascending=False).head(limit).reset_index(drop=True)

def get_top_losers(limit=10):
    df = get_market_movers_dataframe()
    if df.empty:
        return df
    return df.sort_values("%Change", ascending=True).head(limit).reset_index(drop=True)

def get_high_volume_losers(limit=10, vol_multiplier=1.5):
    df = get_market_movers_dataframe()
    if df.empty:
        return df
    df["HighVol"] = (df["Volume"] > df["Avg20"] * vol_multiplier) & (df["Avg20"] > 0)
    df = df[df["HighVol"] & (df["%Change"] < 0)]
    return df.sort_values("%Change", ascending=True).head(limit).reset_index(drop=True)

# ----------------------------
# Dash App
# ----------------------------
app = dash.Dash(__name__)
app.title = "Sector Strength + Market Movers"

app.layout = html.Div([
    html.H1("Sector Strength Heatmap", style={"textAlign": "center"}),
    dcc.Interval(id="interval", interval=REFRESH_MS, n_intervals=0),
    dcc.Graph(id="heatmap"),
    html.Div([
        html.Div([
            html.H2("Sector Details"),
            html.Div(id="stock-table")
        ], style={"width": "60%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"}),

        html.Div([
            html.H2("Scanners"),
            dcc.Tabs(id="extraTabs", value="rise", children=[
                dcc.Tab(label="High-Volume Rising", value="rise"),
                dcc.Tab(label="High-Volume Falling", value="fall"),
                dcc.Tab(label="Volume Shocks", value="shock"),
            ]),
            html.Div(id="extraScannerOutput"),
            html.Br(),
            html.Div(id="lastUpdated", style={"fontSize": "0.9em", "color": "#666"})
        ], style={"width": "35%", "display": "inline-block", "verticalAlign": "top", "padding": "10px"})
    ]),
    html.Hr(),
    # Market Movers section (Top gainers / losers / HV losers)
    html.Div([
        html.H2("Market Movers"),
        dcc.Tabs(id="moverTabs", value="gainers", children=[
            dcc.Tab(label="Top Gainers", value="gainers"),
            dcc.Tab(label="Top Losers", value="losers"),
            dcc.Tab(label="High-Volume Losers", value="hv_losers"),
        ]),
        html.Div(id="marketMoverTable")
    ], style={"padding": "10px"}),
], style={"fontFamily": "Arial, sans-serif", "margin": "10px"})

# ----------------------------
# Callbacks
# ----------------------------
@app.callback(
    Output("heatmap", "figure"),
    Output("lastUpdated", "children"),
    Input("interval", "n_intervals")
)
def update_heatmap(n):
    df = compute_sector_table()
    if df.empty:
        fig = px.imshow([[0]], x=["No Data"], y=["Score"], color_continuous_scale="RdYlGn", aspect="auto")
        fig.add_annotation(text="No sector data available", x=0, y=0, showarrow=False)
    else:
        fig = px.imshow([df["Score"]], x=df["Sector"], y=["Score"], color_continuous_scale="RdYlGn", aspect="auto")
        fig.update_xaxes(tickangle=45)
    fig.update_layout(height=420, margin=dict(l=40, r=10, t=40, b=120), coloraxis_colorbar=dict(title="Score"))
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return fig, f"Last refresh: {ts}"

@app.callback(
    Output("stock-table", "children"),
    Input("heatmap", "clickData")
)
def show_sector_details(clickData):
    if not clickData:
        return "Click a sector on the heatmap to view stock performance."
    sector = clickData["points"][0]["x"]
    df = get_sector_stock_table(sector)
    if df.empty:
        return html.Div([html.H3(f"Stocks in {sector}"), "No data available for that sector right now."])
    cols = [{"name": i, "id": i} for i in df.columns]
    style_data_conditional = [
        {"if": {"filter_query": "{%Change} >= 0"}, "backgroundColor": "#e8f8e8"},
        {"if": {"filter_query": "{%Change} < 0"}, "backgroundColor": "#fde6e6"}
    ]
    return html.Div([
        html.H3(f"Stocks in {sector}"),
        dash_table.DataTable(
            data=df.to_dict("records"),
            columns=cols,
            page_size=20,
            style_cell={"textAlign": "center", "minWidth": "90px", "maxWidth": "220px"},
            style_data_conditional=style_data_conditional,
            sort_action="native",
            filter_action="native",
        )
    ])

@app.callback(
    Output("extraScannerOutput", "children"),
    Input("extraTabs", "value"),
    State("heatmap", "clickData")
)
def update_scanner(tab, heatmap_click):
    sector_context = None
    if heatmap_click:
        try:
            sector_context = heatmap_click["points"][0]["x"]
        except Exception:
            sector_context = None

    if tab == "rise":
        title = "High-Volume Rising Stocks"
        if sector_context:
            df = get_high_volume_rising_for_sector(sector_context)
        else:
            frames = []
            for sec in RAW_SECTOR_STOCKS.keys():
                dfsec = get_high_volume_rising_for_sector(sec)
                if not dfsec.empty:
                    dfsec["Sector"] = sec
                    frames.append(dfsec)
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    elif tab == "fall":
        title = "High-Volume Falling Stocks"
        if sector_context:
            df = get_high_volume_falling_for_sector(sector_context)
        else:
            frames = []
            for sec in RAW_SECTOR_STOCKS.keys():
                dfsec = get_high_volume_falling_for_sector(sec)
                if not dfsec.empty:
                    dfsec["Sector"] = sec
                    frames.append(dfsec)
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        title = "Volume Shocks (sector-relative or vs avg20)"
        df = get_volume_shocks_all()

    if df is None or df.empty:
        return html.Div([html.H3(title), "No matches found. Try after some market activity or broaden the thresholds."])

    cols = [{"name": i, "id": i} for i in df.columns]
    return html.Div([
        html.H3(title),
        dash_table.DataTable(
            data=df.to_dict("records"),
            columns=cols,
            page_size=20,
            style_cell={"textAlign": "center"},
            sort_action="native",
            filter_action="native",
        )
    ])

@app.callback(
    Output("marketMoverTable", "children"),
    Input("moverTabs", "value"),
    Input("interval", "n_intervals")
)
def update_market_movers(tab, _):
    if tab == "gainers":
        title = "Top 10 Gainers"
        df = get_top_gainers()
    elif tab == "losers":
        title = "Top 10 Losers"
        df = get_top_losers()
    else:
        title = "High-Volume Losers (Volume > 1.5 × avg20)"
        df = get_high_volume_losers()

    if df is None or df.empty:
        return html.Div([html.H3(title), "No data available."])

    columns = [{"name": i, "id": i} for i in df.columns]
    return html.Div([
        html.H3(title),
        dash_table.DataTable(
            data=df.to_dict("records"),
            columns=columns,
            page_size=10,
            sort_action="native",
            filter_action="native",
            style_cell={"textAlign": "center"},
            style_data_conditional=[
                {"if": {"filter_query": "{%Change} < 0"}, "backgroundColor": "#FFDDDD"},
                {"if": {"filter_query": "{%Change} >= 0"}, "backgroundColor": "#DDFFDD"}
            ]
        )
    ])

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    # ensure cache exists
    if not os.path.exists(VOLUME_CACHE_FILE):
        try:
            with open(VOLUME_CACHE_FILE, "w") as f:
                json.dump({}, f)
        except Exception:
            pass
    app.run(debug=True, port=8050)
