import pandas as pd
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from rich.console import Console
from rich.table import Table
import os
import time
from truedata_ws.websocket.TD import TD

logger = logging.getLogger(__name__)

# TrueData config
TDUSERNAME = os.getenv("TD_USERNAME", "tdwsp751")
TDPASSWORD = os.getenv("TD_PASSWORD", "raj@751")

# Sector stocks (flattened to ~200 stocks, without -I)
stocks = list(set([stock.replace('-I', '') for sector in SECTOR_STOCKS.values() for stock in sector]))  # Assume SECTOR_STOCKS is defined as before

td = TD(TDUSERNAME, TDPASSWORD, log_level=logging.WARNING)
td.start()
time.sleep(2)  # Wait for connection

def fetch_option_chain(symbol):
    try:
        expiries = td.get_option_exp(symbol)
        if not expiries:
            return None
        expiry = expiries[0]  # Nearest expiry
        chain = td.get_option_chain(symbol, expiry=expiry)
        if chain is None or len(chain) == 0:
            return None
        return chain
    except Exception as e:
        logger.error(f"Error fetching option chain for {symbol}: {e}")
        return None

def calculate_pressure(df):
    buying = 0.0
    selling = 0.0
    if df is None or len(df) == 0:
        return 0, 0
    
    # Calls
    if 'call_net_chng' in df.columns and 'call_chng_oi' in df.columns:
        call_net_change = df['call_net_chng']
        call_oi_change = df['call_chng_oi']
        for i in range(len(df)):
            cn = call_net_change.iloc[i] if not pd.isna(call_net_change.iloc[i]) else 0
            co = call_oi_change.iloc[i] if not pd.isna(call_oi_change.iloc[i]) else 0
            if cn > 0 and co > 0:
                buying += co
            elif cn > 0 and co < 0:
                buying += abs(co)
            elif cn < 0 and co > 0:
                selling += co
            elif cn < 0 and co < 0:
                selling += abs(co)
    
    # Puts (corrected logic)
    if 'put_net_chng' in df.columns and 'put_chng_oi' in df.columns:
        put_net_change = df['put_net_chng']
        put_oi_change = df['put_chng_oi']
        for i in range(len(df)):
            pn = put_net_change.iloc[i] if not pd.isna(put_net_change.iloc[i]) else 0
            po = put_oi_change.iloc[i] if not pd.isna(put_oi_change.iloc[i]) else 0
            if pn > 0 and po > 0:
                selling += po  # long buildup, bearish
            elif pn > 0 and po < 0:
                selling += abs(po)  # short covering, bearish
            elif pn < 0 and po > 0:
                buying += po  # short buildup, bullish
            elif pn < 0 and po < 0:
                buying += abs(po)  # long unwinding, bullish
    
    return buying, selling

def process_stock(symbol):
    df = fetch_option_chain(symbol)
    if df is None:
        return None
    
    buying, selling = calculate_pressure(df)
    total = buying + selling
    if total == 0:
        return None
    
    buying_pct = (buying / total) * 100
    selling_pct = (selling / total) * 100
    
    if buying_pct >= 90:
        return {"symbol": symbol, "pressure": "90%+ Buying", "buying_pct": buying_pct}
    elif selling_pct >= 90:
        return {"symbol": symbol, "pressure": "90%+ Selling", "selling_pct": selling_pct}
    return None

def main():
    console = Console()
    console.print("[bold cyan]Scanning ~200 stocks for 90% Buying/Selling Pressure from Option Chain (using TrueData)[/bold cyan]")
    
    results = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_stock, symbol) for symbol in stocks]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)
    
    if not results:
        console.print("[yellow]No stocks found with 90% buying or selling pressure.[/yellow]")
    else:
        table = Table(title="Stocks with 90% Pressure")
        table.add_column("Symbol")
        table.add_column("Pressure")
        table.add_column("Percentage")
        
        for r in results:
            if "buying_pct" in r:
                table.add_row(r["symbol"], r["pressure"], f"{r['buying_pct']:.2f}%", style="green")
            else:
                table.add_row(r["symbol"], r["pressure"], f"{r['selling_pct']:.2f}%", style="red")
        
        console.print(table)
    
    td.disconnect()

if __name__ == "__main__":
    main()