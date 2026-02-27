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

API_URL = "http://localhost:3000/api/index/options/BANKNIFTY"  # <- BANKNIFTY

ROLLING_INTERVAL_MIN = 3
ROLLING_MAX_ROWS = 20

console = Console()

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
    parsed = [(ed, dt) for ed, dt in parsed if dt is not None]
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

    total_oi = total_call_oi + total_put_oi
    diff = total_put_oi - total_call_oi
    pcr = pcr_value(total_put_oi, total_call_oi)
    sentiment = pcr_sentiment(pcr)

    return {
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "total_oi": total_oi,
        "diff": diff,
        "pcr": pcr,
        "sentiment": sentiment,
        "per_strike": per_strike,
    }

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
        ce_txt = f"[bright_cyan]{ce:,}[/]"
        pe_txt = f"[bright_magenta]{pe:,}[/]"
        t.add_row(str(s), ce_txt, pe_txt)
    return t

def build_snapshot_row(ts, atm, metrics, hist_rows):
    prev1 = hist_rows[-1] if len(hist_rows) >= 1 else None
    prev2 = hist_rows[-2] if len(hist_rows) >= 2 else None

    cur_call = metrics["total_call_oi"]
    cur_put = metrics["total_put_oi"]

    prev1_call = prev1["total_call_oi"] if prev1 else None
    prev2_call = prev2["total_call_oi"] if prev2 else None
    prev1_put = prev1["total_put_oi"] if prev1 else None
    prev2_put = prev2["total_put_oi"] if prev2 else None

    call_arrow = arrow_trend(cur_call, prev1_call, prev2_call)
    put_arrow = arrow_trend(cur_put, prev1_put, prev2_put)

    return [
        ts.strftime("%H:%M:%S"),
        str(atm),
        f"{metrics['total_oi']:,}",
        colorize_arrow(cur_call, call_arrow),
        colorize_arrow(cur_put, put_arrow),
        diff_cell(metrics["diff"]),
        f"{metrics['pcr']:.3f}",
        badge(metrics["sentiment"], metrics["pcr"]),
    ]

def render_rich_dashboard(ts, atm, spot, expiry_str, metrics, by_strike, last_hour_rows):
    header = Panel.fit(
        f"[bold white]BANKNIFTY OI Dashboard[/bold white]\n"
        f"[cyan]Time:[/] {ts.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"[cyan]Spot:[/] {spot:.2f}  "
        f"[cyan]ATM:[/] {atm}  "
        f"[cyan]Expiry:[/] {expiry_str or 'N/A'}",
        border_style="bright_blue",
    )

    snap = Table(title="Snapshot (Current Expiry)", box=box.MINIMAL_DOUBLE_HEAD, show_lines=True, header_style="bold green")
    snap.add_column("Time", no_wrap=True)
    snap.add_column("ATM", justify="right")
    snap.add_column("Total OI", justify="right")
    snap.add_column("Call OI", justify="right")
    snap.add_column("Put OI", justify="right")
    snap.add_column("Difference", justify="right")
    snap.add_column("PCR", justify="right")
    snap.add_column("Sentiment", justify="center")
    snap.add_row(*build_snapshot_row(ts, atm, metrics, last_hour_rows))

    hist = Table(title="Last 1 Hour (3-min)", box=box.MINIMAL_DOUBLE_HEAD, show_lines=True, header_style="bold magenta")
    hist.add_column("Time", no_wrap=True)
    hist.add_column("ATM", justify="right")
    hist.add_column("Total OI", justify="right")
    hist.add_column("Call OI", justify="right")
    hist.add_column("Put OI", justify="right")
    hist.add_column("Difference", justify="right")
    hist.add_column("PCR", justify="right")
    hist.add_column("Sentiment", justify="center")

    for i, row in enumerate(last_hour_rows):
        prev1 = last_hour_rows[i-1] if i-1 >= 0 else None
        prev2 = last_hour_rows[i-2] if i-2 >= 0 else None

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

def main(poll_seconds=ROLLING_INTERVAL_MIN * 60):
    console.rule("[bold]BANKNIFTY Option Chain OI Monitor (Current Expiry, ATM ±3 strikes)")
    last_hour = deque(maxlen=ROLLING_MAX_ROWS)
    last_sample_time = None

    while True:
        try:
            raw = fetch_option_chain()
            trimmed, expiry_str = filter_current_expiry(raw)
            ts = datetime.now()
            atm, by_strike, spot = get_atm_and_slice(trimmed, strikes_each_side=3)

            if atm is None:
                console.print("[red]No ATM or strikes found for current expiry[/red]")
                time.sleep(poll_seconds)
                continue

            metrics = compute_metrics(by_strike)

            if (last_sample_time is None) or (ts - last_sample_time >= timedelta(minutes=ROLLING_INTERVAL_MIN)):
                last_hour.append({
                    "time": ts,
                    "atm": atm,
                    "total_call_oi": metrics["total_call_oi"],
                    "total_put_oi": metrics["total_put_oi"],
                    "total_oi": metrics["total_oi"],
                    "diff": metrics["diff"],
                    "pcr": metrics["pcr"],
                    "sentiment": metrics["sentiment"],
                })
                last_sample_time = ts

            render_rich_dashboard(ts, atm, spot, expiry_str, metrics, by_strike, list(last_hour))

            console.print(f"[grey62]Next update in {ROLLING_INTERVAL_MIN} minutes...[/grey62]")
            time.sleep(poll_seconds)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Stopped by user[/bold yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            time.sleep(poll_seconds)

if __name__ == "__main__":
    main()
