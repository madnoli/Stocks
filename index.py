import json
import requests
import urllib.parse
import dash
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd

###############################################################
# CONFIG
###############################################################

API_INDEX_URL = "http://localhost:3001/api/allIndices"
API_STOCK_URL = "http://localhost:3001/api/equity/{}"

REFRESH_MS = 120000   # 2 minutes auto-refresh

###############################################################
# RAW SECTOR → STOCK MAPPING
###############################################################
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
        'HINDALCO', 'IOC', 'NTPC',        'HINDPETRO', 'ADANIGREEN',
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

###############################################################
# HELPERS: Fetching APIs
###############################################################

def get_indices():
    try:
        r = requests.get(API_INDEX_URL, timeout=5)
        j = r.json()
        return {item["index"]: item["percentChange"] for item in j["data"]}
    except:
        return {}

def get_stock_change(sym):
    try:
        encoded = urllib.parse.quote(sym, safe='')
        r = requests.get(API_STOCK_URL.format(encoded), timeout=5)
        j = r.json()
        return j["priceInfo"]["pChange"]
    except:
        return None

###############################################################
# MAIN DATA ASSEMBLY
###############################################################

def compute_sector_table():
    index_map = get_indices()
    rows = []

    for sector, stocks in RAW_SECTOR_STOCKS.items():
        stock_changes = []

        for s in stocks:
            ch = get_stock_change(s)
            if ch is not None:
                stock_changes.append(ch)

        if not stock_changes:
            avg_stock = 0
        else:
            avg_stock = sum(stock_changes) / len(stock_changes)

        index_pct = index_map.get(sector, 0)

        score = 0.7 * avg_stock + 0.3 * index_pct

        rows.append({
            "Sector": sector,
            "Index %": round(index_pct, 2),
            "Avg Stock %": round(avg_stock, 2),
            "Score": round(score, 2)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Score", ascending=False)
    return df

def get_sector_stock_table(sector):
    stocks = RAW_SECTOR_STOCKS.get(sector, [])
    rows = []

    for s in stocks:
        pct = get_stock_change(s)
        if pct is None:
            continue
        rows.append({
            "Stock": s,
            "%Change": round(pct, 2)
        })

    df = pd.DataFrame(rows)
    df["Rank"] = df["%Change"].rank(ascending=False)
    df = df.sort_values("%Change", ascending=False)
    return df

###############################################################
# DASH UI
###############################################################

app = Dash(__name__)

app.layout = html.Div([
    html.H1("Sector Strength Heatmap", style={"textAlign": "center"}),

    dcc.Interval(id="interval", interval=REFRESH_MS, n_intervals=0),

    dcc.Graph(id="heatmap"),

    html.H2("Sector Details"),
    html.Div(id="stock-table")
])

###############################################################
# CALLBACKS
###############################################################

@app.callback(
    Output("heatmap", "figure"),
    Input("interval", "n_intervals")
)
def update_heatmap(_):
    df = compute_sector_table()

    fig = px.imshow(
        [df["Score"]],
        x=df["Sector"],
        y=["Score"],
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )

    fig.update_layout(
        height=400,
        coloraxis_colorbar=dict(title="Score")
    )

    return fig


@app.callback(
    Output("stock-table", "children"),
    Input("heatmap", "clickData")
)
def show_sector_details(clickData):
    if not clickData:
        return "Click a sector to view stock performance."

    sector = clickData["points"][0]["x"]
    df = get_sector_stock_table(sector)

    return html.Div([
        html.H3(f"Stocks in {sector}"),
        dash_table.DataTable(
            data=df.to_dict("records"),
            columns=[{"name": i, "id": i} for i in df.columns],
            page_size=20,
            style_cell={"textAlign": "center"},
            style_data_conditional=[
                {
                    "if": {"filter_query": "{%Change} > 0"},
                    "backgroundColor": "#d4ffd4"
                },
                {
                    "if": {"filter_query": "{%Change} < 0"},
                    "backgroundColor": "#ffd6d6"
                }
            ]
        )
    ])


###############################################################
# RUN
###############################################################

if __name__ == "__main__":
    app.run(debug=True, port=8050)
