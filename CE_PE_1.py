import json
import time
import requests
import pandas as pd
from datetime import datetime
from great_tables import GT
from rich.console import Console
from rich.table import Table
from pathlib import Path

# --- Configuration ---
API_TMPL = "http://localhost:3000/api/equity/options/{symbol}"
TIMEOUT = 20
STOCKS_FILE = "stocks.txt"
HTML_OUT = "option_chain_summary.html"
CACHE_FILE = "option_chain_cache.json"

# --- Tunables ---
MIN_TOTAL_OI = 2000
MIN_TOTAL_VOL = 200
PCR_TOL = 0.03
SLEEP_SECONDS = 300
EPS = 1e-6

console = Console()

def safe_div(a, b):
    if b is None or abs(b) < EPS:
        return float('inf') if a > 0 else 0.0
    return a / b

def pct_change(now, prev):
    if now is None or prev is None:
        return None
    denom = prev if abs(prev) > EPS else EPS
    return ((now - prev) / denom) * 100.0

def parse_expiry(s):
    try:
        return datetime.strptime(s, "%d-%b-%Y")
    except (ValueError, TypeError):
        return None

def choose_current_expiry(records):
    exps = records.get("expiryDates") or []
    exps_parsed = [(e, parse_expiry(e)) for e in exps]
    now = datetime.now()
    future = [e for e in exps_parsed if e[1] and e[1] >= now]
    if future:
        return min(future, key=lambda x: x[1])[0]
    past = [e for e in exps_parsed if e[1]]
    if past:
        return max(past, key=lambda x: x[1])[0]
    return None

def fetch_symbol_metrics(symbol):
    url = API_TMPL.format(symbol=symbol)
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    obj = r.json()

    recs = obj.get("records", {})
    curr_exp = choose_current_expiry(recs)
    if not curr_exp:
        raise ValueError("No valid expiry")

    rows = [row for row in recs.get("data", []) if row.get("expiryDate") == curr_exp]
    if not rows:
        raise ValueError("No rows for current expiry")

    underlying = next((val for row in rows for val in [row.get("CE", {}).get("underlyingValue"), row.get("PE", {}).get("underlyingValue")] if isinstance(val, (int, float))), None)
    if underlying is None:
        raise ValueError("Underlying price not found")

    ce_oi_sum, pe_oi_sum = 0, 0
    ce_vol_sum, pe_vol_sum = 0, 0
    ce_oi_wsum, pe_oi_wsum = 0.0, 0.0
    ce_oi_w, pe_oi_w = 0.0, 0.0
    ce_iv_wsum, pe_iv_wsum = 0.0, 0.0
    ce_iv_w, pe_iv_w = 0.0, 0.0

    for row in rows:
        ce = row.get("CE") or {}
        pe = row.get("PE") or {}
        ce_oi = ce.get("openInterest") or 0
        pe_oi = pe.get("openInterest") or 0
        ce_vol = ce.get("totalTradedVolume") or 0
        pe_vol = pe.get("totalTradedVolume") or 0

        ce_oi_sum += ce_oi
        pe_oi_sum += pe_oi
        ce_vol_sum += ce_vol
        pe_vol_sum += pe_vol

        if isinstance(ce.get("pchangeinOpenInterest"), (int, float)) and ce_oi > 0:
            ce_oi_wsum += ce.get("pchangeinOpenInterest") * ce_oi
            ce_oi_w += ce_oi
        if isinstance(pe.get("pchangeinOpenInterest"), (int, float)) and pe_oi > 0:
            pe_oi_wsum += pe.get("pchangeinOpenInterest") * pe_oi
            pe_oi_w += pe_oi

        ce_iv = ce.get("impliedVolatility") or 0
        pe_iv = pe.get("impliedVolatility") or 0
        if ce_iv > 0 and ce_oi > 0:
            ce_iv_wsum += ce_iv * ce_oi
            ce_iv_w += ce_oi
        if pe_iv > 0 and pe_oi > 0:
            pe_iv_wsum += pe_iv * pe_oi
            pe_iv_w += pe_oi

    total_oi = ce_oi_sum + pe_oi_sum
    total_vol = ce_vol_sum + pe_vol_sum
    pcr = safe_div(pe_oi_sum, ce_oi_sum)

    ce_oi_chg_pct = safe_div(ce_oi_wsum, ce_oi_w)
    pe_oi_chg_pct = safe_div(pe_oi_wsum, pe_oi_w)
    blended_oi_chg = safe_div((ce_oi_chg_pct * ce_oi_sum) + (pe_oi_chg_pct * pe_oi_sum), total_oi)

    avg_ce_iv = safe_div(ce_iv_wsum, ce_iv_w)
    avg_pe_iv = safe_div(pe_iv_wsum, pe_iv_w)
    avg_iv = safe_div((avg_ce_iv * ce_oi_sum) + (avg_pe_iv * pe_oi_sum), total_oi) * 100

    vol_oi_ratio = safe_div(total_vol, total_oi)

    atm_strike_row = min(rows, key=lambda r: abs(r.get("strikePrice", float('inf')) - underlying))
    atm_ce = atm_strike_row.get("CE", {})
    atm_pe = atm_strike_row.get("PE", {})
    atm_pcr = safe_div(atm_pe.get("openInterest", 0), atm_ce.get("openInterest", 0))
    atm_ce_vol = atm_ce.get("totalTradedVolume", 0)
    atm_pe_vol = atm_pe.get("totalTradedVolume", 0)
    atm_vol_dom = "CALLS" if atm_ce_vol > atm_pe_vol else ("PUTS" if atm_pe_vol > atm_ce_vol else "NEUTRAL")
    atm_signal = f"PCR:{atm_pcr:.2f}|VOL:{atm_vol_dom}"

    def classify(pcr, ce_oi, pe_oi, ce_vol, pe_vol):
        is_low_liq = (ce_oi + pe_oi < MIN_TOTAL_OI) or (ce_vol + pe_vol < MIN_TOTAL_VOL)
        if abs(pcr - 1.0) <= PCR_TOL: return "Neutral"
        ce_oi_dom, pe_oi_dom = ce_oi > pe_oi, pe_oi > ce_oi
        ce_vol_dom, pe_vol_dom = ce_vol >= pe_vol, pe_vol >= ce_vol
        if pcr < 0.8 and ce_oi_dom and ce_vol_dom: return "Strong Bullish" if not is_low_liq else "Mild Bullish"
        if pcr > 1.2 and pe_oi_dom and pe_vol_dom: return "Strong Bearish" if not is_low_liq else "Mild Bearish"
        if pcr < 1.0 and (ce_oi_dom or ce_vol_dom): return "Mild Bullish"
        if pcr > 1.0 and (pe_oi_dom or pe_vol_dom): return "Mild Bearish"
        return "Neutral"

    remark = classify(pcr, ce_oi_sum, pe_oi_sum, ce_vol_sum, pe_vol_sum)

    return {
        "Stock": symbol, "Price": underlying, "Volume": total_vol, "OI": total_oi,
        "OI Chg %": blended_oi_chg, "PCR": pcr, "Avg IV %": avg_iv, "V/OI Ratio": vol_oi_ratio,
        "ATM Signal": atm_signal, "Remark": remark, "Expiry": curr_exp
    }

def load_symbols(path=STOCKS_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip()]

def load_cache():
    p = Path(CACHE_FILE)
    if not p.exists(): return {}
    try: return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError: return {}

def save_cache(cache):
    Path(CACHE_FILE).write_text(json.dumps(cache, indent=2), encoding="utf-8")

def compute_cycle(symbols):
    results = []
    for s in symbols:
        try:
            results.append(fetch_symbol_metrics(s))
        except Exception as e:
            results.append({
                "Stock": s, "Price": None, "Volume": None, "OI": None, "OI Chg %": None,
                "PCR": None, "Avg IV %": None, "V/OI Ratio": None, "ATM Signal": None,
                "Remark": f"Error: {e}", "Expiry": None
            })
    return results

def add_volume_cycle_delta(results, prev_cache):
    new_cache = {}
    for rec in results:
        if "Stock" not in rec or "Expiry" not in rec or rec["Expiry"] is None: continue
        key = f"{rec['Stock']}|{rec['Expiry']}"
        prev_vol = prev_cache.get(key, {}).get("Volume")
        rec["Vol Chg %"] = pct_change(rec.get("Volume"), prev_vol)
        if rec.get("Expiry"):
            new_cache[key] = {"Volume": rec.get("Volume"), "ts": datetime.now().isoformat()}
    return results, new_cache

def render_tables(df):
    if df.empty:
        console.print("[yellow]DataFrame is empty. No tables to render.[/yellow]")
        return
        
    bull_mask = df["Remark"].isin(["Strong Bullish", "Mild Bullish"])
    bear_mask = df["Remark"].isin(["Strong Bearish", "Mild Bearish"])

    bullish_df = df[bull_mask].sort_values("PCR", ascending=True).head(20)
    bearish_df = df[bear_mask].sort_values("PCR", ascending=False).head(20)
    
    bullish_df = bullish_df.assign(Group="Top 20 Bullish")
    bearish_df = bearish_df.assign(Group="Top 20 Bearish")

    table_df = pd.concat([bullish_df, bearish_df], ignore_index=True)
    
    if table_df.empty:
        console.print("[yellow]No bullish or bearish stocks found to display.[/yellow]")
        return

    COLS = ["Group", "Stock", "Price", "Remark", "PCR", "ATM Signal", "V/OI Ratio", "Avg IV %", "Vol Chg %", "OI Chg %", "Volume", "OI", "Expiry"]
    table_df = table_df[[col for col in COLS if col in table_df.columns]]

    # --- HTML Table (Great Tables) ---
    gt = (
        GT(table_df)
        .tab_header(
            title="Options Dashboard",
            subtitle="Sentiment & Activity Analysis for Current Expiry"
        )
        .fmt_number(columns=["Volume", "OI"], decimals=0, use_seps=True)
        .fmt_number(columns=["Price", "PCR", "V/OI Ratio"], decimals=2)
        .fmt_percent(columns=["Avg IV %", "Vol Chg %", "OI Chg %"], decimals=2)
        .cols_label(
            **{
                "OI Chg %": "OI Δ %", "Vol Chg %": "Vol Δ %",
                "Avg IV %": "Avg IV %", "V/OI Ratio": "V/OI"
            }
        )
    )
    # --- GUARANTEED FIX: Manually generate HTML and write to file ---
    html_content = gt.as_raw_html()
    Path(HTML_OUT).write_text(html_content, encoding="utf-8")

    # --- Console Table (Rich) ---
    table = Table(title="Options Dashboard", show_lines=True, expand=True)
    for c in table_df.columns:
        justify = "right" if c not in ["Group", "Stock", "Remark", "ATM Signal", "Expiry"] else "left"
        table.add_column(c, justify=justify, overflow="fold")
    
    for _, row in table_df.iterrows():
        style_str = ""
        remark_val = row.get("Remark", "")
        if "Bullish" in remark_val: style_str = "green"
        if "Bearish" in remark_val: style_str = "red"
        
        row_data = []
        for col_name in table_df.columns:
            val = row[col_name]
            if pd.isna(val):
                row_data.append("-")
            elif col_name == "Price":
                row_data.append(f"{val:,.2f}")
            elif col_name in ["Volume", "OI"]:
                 row_data.append(f"{val:,.0f}")
            elif col_name in ["Avg IV %", "Vol Chg %", "OI Chg %"]:
                row_data.append(f"{val:.2f}%")
            elif col_name in ["PCR", "V/OI Ratio"]:
                row_data.append(f"{val:.2f}")
            else:
                 row_data.append(str(val))

        # Apply style to the first few columns for emphasis
        row_data[0] = f"[{style_str}]{row_data[0]}[/{style_str}]"
        row_data[1] = f"[{style_str}]{row_data[1]}[/{style_str}]"
        row_data[3] = f"[{style_str}]{row_data[3]}[/{style_str}]"
        
        table.add_row(*row_data)

    console.print(table)

def run_once():
    """Executes one full data collection and rendering cycle."""
    symbols = load_symbols()
    prev_cache = load_cache()
    
    console.print(f"Fetching data for {len(symbols)} symbols...")
    results = compute_cycle(symbols)
    
    df = pd.DataFrame(results)
    df = df[df["Expiry"].notna()]
    
    results_with_vol_delta, new_cache = add_volume_cycle_delta(df.to_dict("records"), prev_cache)
    
    df_final = pd.DataFrame(results_with_vol_delta)
    
    render_tables(df_final)
    save_cache(new_cache)
    console.print(f"Successfully generated HTML report: [link=file://{Path(HTML_OUT).resolve()}]'{HTML_OUT}'[/link]")

if __name__ == "__main__":
    while True:
        try:
            ts = datetime.now().strftime('%H:%M:%S')
            console.rule(f"[bold blue]Cycle Start: {ts}")
            run_once()
            ts_end = datetime.now().strftime('%H:%M:%S')
            console.rule(f"[bold green]Cycle Complete: {ts_end}. Sleeping for {SLEEP_SECONDS}s.")
        except Exception as e:
            console.print_exception(show_locals=False)
            console.log(f"[bold red]An error occurred: {e}. Retrying after sleep.")
        time.sleep(SLEEP_SECONDS)

        