#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import time
from datetime import datetime, timedelta
from collections import deque, OrderedDict

from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich import box

# ---------------------------------
# NIFTY CONFIG
# ---------------------------------
API_URL = "http://localhost:3001/api/index/options/NIFTY"

ROLLING_INTERVAL_MIN = 2          # SAMPLE EVERY 2 MINUTES
ROLLING_MAX_ROWS = 15             # LAST 30 MINUTES (15 samples × 2 min)

console = Console()

# -------------------------------
# File writing helper
# -------------------------------
def write_to_file(ts, atm, metrics):
    line = (
        f"{ts.strftime('%Y-%m-%d %H:%M:%S')},"
        f"ATM={atm},"
        f"CallOI={metrics['total_call_oi']},"
        f"PutOI={metrics['total_put_oi']},"
        f"TotalOI={metrics['total_oi']},"
        f"Diff={metrics['diff']},"
        f"PCR={metrics['pcr']:.3f},"
        f"Sentiment={metrics['sentiment']}\n"
    )
    # You can rename this file if you want a separate file per index
    with open("OI_data_NIFTY.txt", "a") as f:
        f.write(line)

# -------------------------------
# API Fetch
# -------------------------------
def fetch_option_chain(url=API_URL, timeout=10):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()

def parse_expiry_date(ed):
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(ed, fmt)
        except ValueError:
            pass
    return None

def filter_current_expiry(data):
    records = data.get("records", {})
    all_rows = records.get("data", []) or []

    expiry_set = set()
    for r in all_rows:
        ed = r.get("expiryDate")
        if ed:
            expiry_set.add(ed)

    if not expiry_set:
        return data, None

    parsed = [(ed, parse_expiry_date(ed)) for ed in expiry_set]
    parsed = [(ed, dt) for ed, dt in parsed if dt]
    if not parsed:
        return data, None

    current_expiry = min(parsed, key=lambda x: x[1])[0]
    filtered_rows = [r for r in all_rows if r.get("expiryDate") == current_expiry]

    trimmed = dict(data)
    trimmed_records = dict(records)
    trimmed_records["data"] = filtered_rows
    trimmed_records["expiryDates"] = [current_expiry]
    trimmed["records"] = trimmed_records

    return trimmed, current_expiry

# -------------------------------
# ATM & Strike Slice
# -------------------------------
def get_atm_and_slice(data, strikes_each_side=3):
    records = data.get("records", {}).get("data", [])
    spot = data.get("records", {}).get("underlyingValue", None)
    if spot is None or not records:
        return None, OrderedDict(), spot

    strike_list = sorted({rec.get("strikePrice") for rec in records if rec.get("strikePrice") is not None})
    if not strike_list:
        return None, OrderedDict(), spot

    atm = min(strike_list, key=lambda k: abs(k - spot))

    idx = strike_list.index(atm)
    lo = max(0, idx - strikes_each_side)
    hi = min(len(strike_list), idx + strikes_each_side + 1)
    window = strike_list[lo:hi]

    target_len = strikes_each_side * 2 + 1
    while len(window) < target_len and lo > 0:
        lo -= 1
        window = strike_list[lo:hi]
    while len(window) < target_len and hi < len(strike_list):
        hi += 1
        window = strike_list[lo:hi]

    by_strike = OrderedDict((s, {"CE": None, "PE": None}) for s in sorted(window))
    for rec in records:
        s = rec.get("strikePrice")
        if s in by_strike:
            if "CE" in rec:
                by_strike[s]["CE"] = rec["CE"]
            if "PE" in rec:
                by_strike[s]["PE"] = rec["PE"]

    return atm, by_strike, spot

# -------------------------------
# PCR & Metrics
# -------------------------------
def pcr_value(total_put_oi, total_call_oi):
    return round((total_put_oi / total_call_oi), 3) if total_call_oi > 0 else 0.0

def pcr_sentiment(pcr):
    if pcr > 1.3:
        return "Bullish"
    if pcr < 0.7:
        return "Bearish"
    return "Neutral"

def compute_metrics(by_strike):
    total_call_oi = 0
    total_put_oi = 0
    per_strike = {}

    for s, legs in by_strike.items():
        ce_oi = (legs["CE"] or {}).get("openInterest", 0) or 0
        pe_oi = (legs["PE"] or {}).get("openInterest", 0) or 0

        total_call_oi += ce_oi
        total_put_oi += pe_oi

        per_strike[s] = {"call_oi": ce_oi, "put_oi": pe_oi}

    pcr = pcr_value(total_put_oi, total_call_oi)
    sentiment = pcr_sentiment(pcr)

    return {
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "total_oi": total_call_oi + total_put_oi,
        "diff": total_put_oi - total_call_oi,
        "pcr": pcr,
        "sentiment": sentiment,
        "per_strike": per_strike,
    }

# -------------------------------
# Display Helpers
# -------------------------------
def arrow_trend(curr, prev1, prev2):
    if prev1 is None or prev2 is None:
        return "➖"
    if curr > prev1 and curr > prev2:
        return "▲"
    if curr < prev1 and curr < prev2:
        return "▼"
    return "➖"

def colorize_arrow(value, arrow):
    if arrow == "▲":
        return f"[green]{value:,} {arrow}[/]"
    if arrow == "▼":
        return f"[red]{value:,} {arrow}[/]"
    return f"{value:,} {arrow}"

def badge(sentiment, pcr):
    if sentiment == "Bullish":
        return f"[white on green] Bullish [/]\n[green]PCR {pcr:.3f}[/]"
    if sentiment == "Bearish":
        return f"[white on red] Bearish [/]\n[red]PCR {pcr:.3f}[/]"
    return f"[black on yellow] Neutral [/]\n[yellow]PCR {pcr:.3f}[/]"

def diff_cell(diff):
    if diff > 0:
        return f"[green]{diff:,}[/] ✅"
    if diff < 0:
        return f"[red]{diff:,}[/] ❌"
    return f"{diff:,} ➖"

def per_strike_table(by_strike):
    t = Table(title="ATM ±3 Strikes Detail", box=box.MINIMAL_DOUBLE_HEAD, show_lines=True, header_style="bold cyan")
    t.add_column("Strike", justify="right")
    t.add_column("Call OI", justify="right")
    t.add_column("Put OI", justify="right")

    for s, d in by_strike.items():
        ce = (d["CE"] or {}).get("openInterest", 0) or 0
        pe = (d["PE"] or {}).get("openInterest", 0) or 0
        t.add_row(str(s), f"[bright_cyan]{ce:,}[/]", f"[bright_magenta]{pe:,}[/]")

    return t

def build_snapshot_row(ts, atm, metrics, hist_rows):
    prev1 = hist_rows[-1] if len(hist_rows) >= 1 else None
    prev2 = hist_rows[-2] if len(hist_rows) >= 2 else None

    curr_call = metrics["total_call_oi"]
    curr_put = metrics["total_put_oi"]

    call_arrow = arrow_trend(curr_call, prev1["total_call_oi"] if prev1 else None,
                             prev2["total_call_oi"] if prev2 else None)
    put_arrow = arrow_trend(curr_put, prev1["total_put_oi"] if prev1 else None,
                            prev2["total_put_oi"] if prev2 else None)

    return [
        ts.strftime("%H:%M:%S"),
        str(atm),
        f"{metrics['total_oi']:,}",
        colorize_arrow(curr_call, call_arrow),
        colorize_arrow(curr_put, put_arrow),
        diff_cell(metrics["diff"]),
        f"{metrics['pcr']:.3f}",
        badge(metrics["sentiment"], metrics["pcr"]),
    ]

# -------------------------------
# RENDER DASHBOARD
# -------------------------------
def render_rich_dashboard(ts, atm, spot, expiry_str, metrics, by_strike, last_rows):
    header = Panel.fit(
        f"[bold white]NIFTY OI Dashboard[/bold white]\n"
        f"[cyan]Time:[/] {ts.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"[cyan]Spot:[/] {spot:.2f}  "
        f"[cyan]ATM:[/] {atm}  "
        f"[cyan]Expiry:[/] {expiry_str or 'N/A'}",
        border_style="bright_blue",
    )

    snap = Table(title="Snapshot (Current Expiry)", box=box.MINIMAL_DOUBLE_HEAD, show_lines=True, header_style="bold green")
    snap.add_column("Time")
    snap.add_column("ATM", justify="right")
    snap.add_column("Total OI", justify="right")
    snap.add_column("Call OI", justify="right")
    snap.add_column("Put OI", justify="right")
    snap.add_column("Difference", justify="right")
    snap.add_column("PCR", justify="right")
    snap.add_column("Sentiment", justify="center")
    snap.add_row(*build_snapshot_row(ts, atm, metrics, last_rows))

    hist = Table(title="Last 30 Minutes (2-Min)", box=box.MINIMAL_DOUBLE_HEAD, show_lines=True, header_style="bold magenta")
    hist.add_column("Time")
    hist.add_column("ATM", justify="right")
    hist.add_column("Total OI", justify="right")
    hist.add_column("Call OI", justify="right")
    hist.add_column("Put OI", justify="right")
    hist.add_column("Difference", justify="right")
    hist.add_column("PCR", justify="right")
    hist.add_column("Sentiment", justify="center")

    for i, row in enumerate(last_rows):
        prev1 = last_rows[i-1] if i > 0 else None
        prev2 = last_rows[i-2] if i > 1 else None

        call_arrow = arrow_trend(row["total_call_oi"],
                                 prev1["total_call_oi"] if prev1 else None,
                                 prev2["total_call_oi"] if prev2 else None)
        put_arrow = arrow_trend(row["total_put_oi"],
                                prev1["total_put_oi"] if prev1 else None,
                                prev2["total_put_oi"] if prev2 else None)

        hist.add_row(
            row["time"].strftime("%H:%M:%S"),
            str(row["atm"]),
            f"{row['total_oi']:,}",
            colorize_arrow(row["total_call_oi"], call_arrow),
            colorize_arrow(row["total_put_oi"], put_arrow),
            diff_cell(row["diff"]),
            f"{row['pcr']:.3f}",
            badge(row["sentiment"], row["pcr"]),
        )

    console.print(header)
    console.print(snap)
    console.print(per_strike_table(by_strike))
    console.print(hist)

# -------------------------------
# MAIN LOOP
# -------------------------------
def main(poll_seconds=ROLLING_INTERVAL_MIN * 60):
    console.rule("[bold]NIFTY Option Chain OI Monitor (ATM ±3)")

    last_rows = deque(maxlen=ROLLING_MAX_ROWS)
    last_sample_time = None

    while True:
        try:
            raw = fetch_option_chain()
            trimmed, expiry_str = filter_current_expiry(raw)

            ts = datetime.now()
            atm, by_strike, spot = get_atm_and_slice(trimmed)

            if atm is None:
                console.print("[red]No ATM or strike data found[/red]")
                time.sleep(poll_seconds)
                continue

            metrics = compute_metrics(by_strike)

            # ------------------------------------
            # Sampling every 2 minutes
            # ------------------------------------
            if (last_sample_time is None) or (ts - last_sample_time >= timedelta(minutes=ROLLING_INTERVAL_MIN)):

                last_rows.append({
                    "time": ts,
                    "atm": atm,
                    "total_call_oi": metrics["total_call_oi"],
                    "total_put_oi": metrics["total_put_oi"],
                    "total_oi": metrics["total_oi"],
                    "diff": metrics["diff"],
                    "pcr": metrics["pcr"],
                    "sentiment": metrics["sentiment"],
                })

                # write to file
                write_to_file(ts, atm, metrics)

                last_sample_time = ts

            render_rich_dashboard(ts, atm, spot, expiry_str, metrics, by_strike, list(last_rows))

            console.print(f"[grey62]Next update in {ROLLING_INTERVAL_MIN} minutes...[/grey62]")
            time.sleep(poll_seconds)

        except KeyboardInterrupt:
            console.print("[bold yellow]Stopped by user[/bold yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            time.sleep(poll_seconds)

if __name__ == "__main__":
    main()
