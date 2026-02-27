import json
import time
import requests
import pandas as pd
from datetime import datetime
from great_tables import GT
from rich.console import Console
from rich.table import Table
from pathlib import Path

API_TMPL = "http://localhost:3000/api/equity/options/{symbol}"
TIMEOUT = 20
STOCKS_FILE = "stocks.txt"
HTML_OUT = "option_chain_summary.html"
CACHE_FILE = "option_chain_cache.json"   # for cycle-over-cycle Volume only

# Tunables
MIN_TOTAL_OI = 2000
MIN_TOTAL_VOL = 200
PCR_TOL = 0.03
SLEEP_SECONDS = 300
EPS = 1e-6

console = Console()

def safe_div(a, b):
    if b == 0:
        return float('inf') if a > 0 else 0.0
    return a / b

def pct_change(now, prev):
    if prev is None:
        return None
    denom = prev if abs(prev) > EPS else EPS
    return ((now - prev) / denom) * 100.0

def parse_expiry(s):
    try:
        return datetime.strptime(s, "%d-%b-%Y")
    except Exception:
        return None

def choose_current_expiry(records):
    exps = records.get("expiryDates") or []
    exps_parsed = [(e, parse_expiry(e)) for e in exps]
    now = datetime.now()
    future = [e for e in exps_parsed if e[1] and e[1] >= now]
    chosen = min(future, key=lambda x: x[1]) if future else (min([e for e in exps_parsed if e[1]], key=lambda x: x[1]) if exps_parsed else (None, None))
    return chosen[0] if chosen else None

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

    # Underlying price
    underlying = None
    for row in rows:
        ce = row.get("CE") or {}
        pe = row.get("PE") or {}
        underlying = ce.get("underlyingValue") or pe.get("underlyingValue")
        if isinstance(underlying, (int, float)):
            break

    # Aggregate CE/PE OI and Volume; compute OI change % FROM OPTION CHAIN directly
    ce_oi_sum = pe_oi_sum = 0
    ce_vol_sum = pe_vol_sum = 0

    # for side-weighted pchangeinOpenInterest
    ce_oi_wsum = pe_oi_wsum = 0.0
    ce_oi_w = pe_oi_w = 0.0

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

        # weight pchangeinOpenInterest by OI on each side
        if isinstance(ce.get("pchangeinOpenInterest"), (int, float)) and ce_oi > 0:
            ce_oi_wsum += ce.get("pchangeinOpenInterest") * ce_oi
            ce_oi_w += ce_oi
        if isinstance(pe.get("pchangeinOpenInterest"), (int, float)) and pe_oi > 0:
            pe_oi_wsum += pe.get("pchangeinOpenInterest") * pe_oi
            pe_oi_w += pe_oi

    total_oi = ce_oi_sum + pe_oi_sum
    total_vol = ce_vol_sum + pe_vol_sum

    # Side-weighted OI % change
    ce_oi_chg_pct = (ce_oi_wsum / ce_oi_w) if ce_oi_w > 0 else None
    pe_oi_chg_pct = (pe_oi_wsum / pe_oi_w) if pe_oi_w > 0 else None

    # Blended OI % change across CE+PE (weighted by side OI share)
    blended_oi_chg = None
    if total_oi > 0:
        ce_wt = ce_oi_sum / total_oi
        pe_wt = pe_oi_sum / total_oi
        wsum = 0.0
        wt = 0.0
        if ce_oi_chg_pct is not None:
            wsum += ce_oi_chg_pct * ce_wt
            wt += ce_wt
        if pe_oi_chg_pct is not None:
            wsum += pe_oi_chg_pct * pe_wt
            wt += pe_wt
        if wt > 0:
            blended_oi_chg = wsum / wt

    # PCR and classification
    pcr = safe_div(pe_oi_sum, ce_oi_sum)

    def classify(pcr, ce_oi, pe_oi, ce_vol, pe_vol):
        total_oi = ce_oi + pe_oi
        total_vol = ce_vol + pe_vol
        low_liq = (total_oi < MIN_TOTAL_OI or total_vol < MIN_TOTAL_VOL)

        if abs(pcr - 1.0) <= PCR_TOL:
            return "Neutral"

        ce_oi_dom = ce_oi > pe_oi
        pe_oi_dom = pe_oi > ce_oi
        ce_vol_dom = ce_vol >= pe_vol
        pe_vol_dom = pe_vol >= ce_vol

        if pcr < 0.8 and ce_oi_dom and ce_vol_dom:
            return "Strong Bullish" if not low_liq else "Mild Bullish"
        if pcr > 1.2 and pe_oi_dom and pe_vol_dom:
            return "Strong Bearish" if not low_liq else "Mild Bearish"

        if pcr < 1.0 and (ce_oi_dom or ce_vol_dom):
            return "Mild Bullish"
        if pcr > 1.0 and (pe_oi_dom or pe_vol_dom):
            return "Mild Bearish"

        return "Neutral"

    remark = classify(pcr, ce_oi_sum, pe_oi_sum, ce_vol_sum, pe_vol_sum)

    return {
        "Stock": symbol,
        "Price": underlying,
        "Volume": total_vol,
        "OI": total_oi,
        "OI change %": blended_oi_chg,   # DIRECT from option-chain pchangeinOpenInterest
        "PCR": pcr,
        "Remark": remark,
        "Current Expiry": curr_exp
    }

def load_symbols(path=STOCKS_FILE):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip().upper() for line in f if line.strip()]

def load_cache():
    p = Path(CACHE_FILE)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_cache(cache):
    Path(CACHE_FILE).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

def compute_cycle(symbols):
    results = []
    for s in symbols:
        try:
            results.append(fetch_symbol_metrics(s))
        except Exception as e:
            results.append({
                "Stock": s, "Price": None, "Volume": None, "OI": None,
                "OI change %": None, "PCR": None, "Remark": f"Error: {e}",
                "Current Expiry": None
            })
    return results

def add_volume_cycle_delta(results, prev_cache):
    # Compute cycle-over-cycle Volume change % only
    new_cache = {}
    for rec in results:
        sym = rec["Stock"]
        exp = rec["Current Expiry"]
        key = f"{sym}|{exp}" if exp else f"{sym}|"
        prev = prev_cache.get(key, {})
        prev_vol = prev.get("Volume")

        vol_chg_pct = pct_change(rec["Volume"], prev_vol) if rec["Volume"] is not None and prev_vol is not None else None
        rec["Volume change %"] = vol_chg_pct

        if rec["Current Expiry"]:
            new_cache[key] = {
                "Volume": rec["Volume"],
                "ts": datetime.now().isoformat(timespec="seconds")
            }
    return results, new_cache

def render_tables(df):
    bull_mask = df["Remark"].isin(["Strong Bullish", "Mild Bullish"])
    bear_mask = df["Remark"].isin(["Strong Bearish", "Mild Bearish"])

    bullish_df = df[bull_mask].sort_values(["PCR", "OI"], ascending=[True, False]).head(20)
    bearish_df = df[bear_mask].sort_values(["PCR", "OI"], ascending=[False, False])

    if len(bearish_df) < 20:
        neutral_pad = df[df["Remark"] == "Neutral"].sort_values(["PCR","OI"], ascending=[False, False])
        pad_needed = 20 - len(bearish_df)
        bearish_df = pd.concat([bearish_df, neutral_pad.head(pad_needed)], ignore_index=True)

    bearish_df = bearish_df.head(20)

    bullish_df = bullish_df.assign(Group="Bullish Top 20")
    bearish_df = bearish_df.assign(Group="Bearish Top 20")

    table_df = pd.concat([bullish_df, bearish_df], ignore_index=True)

    gt = (
        GT(table_df[["Group","Stock","Volume","Price","OI","OI change %","Volume change %","PCR","Current Expiry"]])
        .tab_header(title="Option-Chain Snapshot (Current Expiry)",
                    subtitle="PCR classification; OI% from option chain; Volume% cycle-over-cycle (5 min)")
        .fmt_number(columns=["Volume", "OI"], decimals=0, use_seps=True)
        .fmt_number(columns=["Price"], decimals=2, use_seps=True)
        .fmt_number(columns=["PCR"], decimals=2)
        .fmt_percent(columns=["OI change %", "Volume change %"], decimals=2)
    )
    gt.write_raw_html(HTML_OUT)

    table = Table(title="Option-Chain Snapshot (Current Expiry)")
    cols = ["Group","Stock","Volume","Price","OI","OI change %","Volume change %","PCR","Current Expiry"]
    from rich.console import Console
    console = Console()
    for c in cols:
        table.add_column(c, justify="right" if c not in ["Group","Stock","Current Expiry"] else "left")
    for _, row in table_df.iterrows():
        table.add_row(
            str(row["Group"]), str(row["Stock"]),
            f"{row['Volume']:,.0f}" if pd.notna(row["Volume"]) else "-",
            f"{row['Price']:,.2f}" if pd.notna(row["Price"]) else "-",
            f"{row['OI']:,.0f}" if pd.notna(row["OI"]) else "-",
            f"{row['OI change %']:.2f}%" if pd.notna(row["OI change %"]) else "-",
            f"{row['Volume change %']:.2f}%" if pd.notna(row["Volume change %"]) else "-",
            f"{row['PCR']:.2f}" if pd.notna(row["PCR"]) else "-",
            str(row["Current Expiry"])
        )
    console.print(table)

def run_once():
    symbols = load_symbols(STOCKS_FILE)
    prev_cache = load_cache()
    results = compute_cycle(symbols)
    df = pd.DataFrame(results)
    df = df[~df["Current Expiry"].isna()]
    # Add cycle-over-cycle Volume% only; OI change % already from chain
    results2, new_cache = add_volume_cycle_delta(df.to_dict(orient="records"), prev_cache)
    df = pd.DataFrame(results2)
    render_tables(df)
    save_cache(new_cache)

if __name__ == "__main__":
    while True:
        try:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle start")
            run_once()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Cycle complete; sleeping {SLEEP_SECONDS}s")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")
        time.sleep(SLEEP_SECONDS)
