#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NIFTY Option Chain OI Monitor + Fast Scalper Mode (5–20 pts)

Requirements:
    pip install requests rich
"""

import requests
import time
from datetime import datetime, timedelta
from collections import deque, OrderedDict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# ---------------------------------
# CONFIG
# ---------------------------------
API_URL = "http://localhost:3001/api/index/options/NIFTY"

ROLLING_INTERVAL_MIN = 2          # Sample every N minutes
ROLLING_MAX_ROWS = 15             # ~30 minutes history (15 x 2min)
VWAP_WINDOW_SAMPLES = 30          # ~1 hour of spot history for VWAP/SR

console = Console()


# =====================================================================
# 1. BASIC HELPERS (FILE WRITE, API, EXPIRY FILTER)
# =====================================================================

def write_to_file(ts, atm, metrics):
    """
    Append a simple CSV-style line for later analysis.
    """
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
    with open("OI_data_NIFTY.txt", "a") as f:
        f.write(line)


def fetch_option_chain(url=API_URL, timeout=10):
    """
    Call your local API server for latest NIFTY option chain.
    """
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def parse_expiry_date(ed):
    """
    Parse expiry date in either 01-Jan-2025 or 2025-01-01 format.
    """
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(ed, fmt)
        except ValueError:
            pass
    return None


def filter_current_expiry(data):
    """
    Select nearest expiry series and filter records belonging to it.
    """
    records = data.get("records", {})
    all_rows = records.get("data", []) or []

    expiry_set = set()
    for r in all_rows:
        ed = r.get("expiryDate")
        if ed:
            expiry_set.add(ed)

    if not expiry_set:
        return data, None

    parsed = [
        (ed, parse_expiry_date(ed))
        for ed in expiry_set
        if parse_expiry_date(ed)
    ]
    if not parsed:
        return data, None

    current_expiry = min(parsed, key=lambda x: x[1])[0]
    filtered_rows = [r for r in all_rows if r.get("expiryDate") == current_expiry]

    new_data = dict(data)
    new_rec = dict(records)
    new_rec["data"] = filtered_rows
    new_rec["expiryDates"] = [current_expiry]
    new_data["records"] = new_rec

    return new_data, current_expiry


# =====================================================================
# 2. ATM & OI METRICS
# =====================================================================

def get_atm_and_slice(data, strikes_each_side=3):
    """
    Determine ATM strike & capture CE/PE legs +/- N strikes around ATM.
    """
    records = data.get("records", {}).get("data", [])
    spot = data.get("records", {}).get("underlyingValue", None)

    if spot is None or not records:
        return None, OrderedDict(), spot

    strike_list = sorted({
        rec.get("strikePrice")
        for rec in records
        if rec.get("strikePrice") is not None
    })

    if not strike_list:
        return None, OrderedDict(), spot

    atm = min(strike_list, key=lambda k: abs(k - spot))

    idx = strike_list.index(atm)
    lo = max(0, idx - strikes_each_side)
    hi = min(len(strike_list), idx + strikes_each_side + 1)

    window = strike_list[lo:hi]

    expected = strikes_each_side * 2 + 1
    while len(window) < expected and lo > 0:
        lo -= 1
        window = strike_list[lo:hi]
    while len(window) < expected and hi < len(strike_list):
        hi += 1
        window = strike_list[lo:hi]

    by_strike = OrderedDict((s, {"CE": None, "PE": None}) for s in window)

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

    for _, legs in by_strike.items():
        ce_oi = (legs["CE"] or {}).get("openInterest") or 0
        pe_oi = (legs["PE"] or {}).get("openInterest") or 0
        total_call_oi += ce_oi
        total_put_oi += pe_oi

    pcr = pcr_value(total_put_oi, total_call_oi)
    sent = pcr_sentiment(pcr)

    return {
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "total_oi": total_call_oi + total_put_oi,
        "diff": total_put_oi - total_call_oi,
        "pcr": pcr,
        "sentiment": sent,
    }


# =====================================================================
# 3. DISPLAY HELPERS + VWAP
# =====================================================================

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


def badge(sent, pcr):
    if sent == "Bullish":
        return f"[white on green] Bullish [/]\n[green]PCR {pcr:.3f}[/]"
    if sent == "Bearish":
        return f"[white on red] Bearish [/]\n[red]PCR {pcr:.3f}[/]"
    return f"[black on yellow] Neutral [/]\n[yellow]PCR {pcr:.3f}[/]"


def diff_cell(diff):
    if diff > 0:
        return f"[green]{diff:,}[/] ✅"
    if diff < 0:
        return f"[red]{diff:,}[/] ❌"
    return f"{diff:,} ➖"


def compute_vwap_and_sr(spot_history):
    if not spot_history:
        return None, None, None
    vwap = sum(spot_history) / len(spot_history)
    return vwap, min(spot_history), max(spot_history)
# =====================================================================
# 3.5 NEW: SIGNAL TABLE HELPERS (incl. VWAP signal)
# =====================================================================

def signal_from_metrics(metrics):
    diff = metrics.get("diff", 0)
    pcr = metrics.get("pcr", 0.0)
    if diff < 0 and pcr < 1:
        return "SELL"
    if diff > 0 and pcr > 1:
        return "BUY"
    return "NO TRADE"


def vwap_signal(spot, vwap):
    if vwap is None or spot is None:
        return "NEUTRAL"
    if spot > vwap:
        return "BUY"
    if spot < vwap:
        return "SELL"
    return "NEUTRAL"


def build_signal_table(history_rows):
    t = Table(
        title="INTRADAY DATA - NIFTY",
        box=box.MINIMAL_DOUBLE_HEAD,
        header_style="bold yellow",
        show_lines=True,
    )

    t.add_column("Time")
    t.add_column("Diff", justify="right")
    t.add_column("PCR", justify="center")
    t.add_column("Option Signal", justify="center")
    t.add_column("VWAP Signal", justify="center")

    for row in reversed(history_rows):
        diff = row["diff"]
        pcr = row["pcr"]
        signal = signal_from_metrics(row)

        spot = row["spot"]
        vwap = row["vwap"]
        vw = vwap_signal(spot, vwap)

        diff_text = f"[red]{diff:,}[/]" if diff < 0 else f"[green]{diff:,}[/]"

        sig = "[red]SELL[/]" if signal == "SELL" else "[green]BUY[/]" if signal == "BUY" else "[grey]NO TRADE[/]"
        vws = "[green]BUY[/]" if vw == "BUY" else "[red]SELL[/]" if vw == "SELL" else "[grey]NEUTRAL[/]"

        t.add_row(
            row["time"].strftime("%H:%M"),
            diff_text,
            f"{pcr:.2f}",
            sig,
            vws
        )

    return t


# =====================================================================
# 5. VWAP FORMATTING FOR HISTORY TABLE (FULL B VERSION)
# =====================================================================

def format_vwap_cell(row_vwap, row_spot):
    """
    Returns full VWAP display:
        BUY ↑ 25918.45 (+0.22%)
        SELL ↓ 25914.90 (-0.18%)
        NEUTRAL → 25918.45 (0.00%)
    """
    if row_vwap is None or row_spot is None:
        return "—"

    try:
        pct_diff = ((row_spot - row_vwap) / row_vwap) * 100

        if row_spot > row_vwap:
            return (
                f"[green]BUY ↑ {row_vwap:.2f} "
                f"(+{pct_diff:.2f}%)[/]"
            )
        elif row_spot < row_vwap:
            return (
                f"[red]SELL ↓ {row_vwap:.2f} "
                f"({pct_diff:.2f}%)[/]"
            )
        else:
            return (
                f"[yellow]NEUTRAL → {row_vwap:.2f} "
                f"(0.00%)[/]"
            )

    except:
        return f"{row_vwap:.2f}"


# =====================================================================
# 6. SNAPSHOT ROW BUILDER
# =====================================================================

def build_snapshot_row(ts, atm, metrics, hist_rows):
    prev1 = hist_rows[-1] if len(hist_rows) >= 1 else None
    prev2 = hist_rows[-2] if len(hist_rows) >= 2 else None

    curr_call = metrics["total_call_oi"]
    curr_put = metrics["total_put_oi"]

    call_arrow = arrow_trend(
        curr_call,
        prev1["total_call_oi"] if prev1 else None,
        prev2["total_call_oi"] if prev2 else None,
    )
    put_arrow = arrow_trend(
        curr_put,
        prev1["total_put_oi"] if prev1 else None,
        prev2["total_put_oi"] if prev2 else None,
    )

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
# =====================================================================
# 7. MAIN RENDERING DASHBOARD
# =====================================================================

def render_rich_dashboard(
    ts, atm, spot, expiry_str,
    metrics, by_strike,
    last_rows, spot_history,
    is_expiry_day
):
    vwap, support, resistance = compute_vwap_and_sr(spot_history)

    header_text = (
        f"[bold white]NIFTY OI Dashboard[/bold white]\n"
        f"[cyan]Time:[/] {ts.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"[cyan]Spot:[/] {spot:.2f}  "
        f"[cyan]ATM:[/] {atm}  "
        f"[cyan]Expiry:[/] {expiry_str or 'N/A'}\n"
    )

    if vwap is not None:
        header_text += (
            f"[cyan]VWAP:[/] {vwap:.2f}  "
            f"[cyan]Support:[/] {support:.2f}  "
            f"[cyan]Resistance:[/] {resistance:.2f}"
        )

    console.print(Panel.fit(header_text, border_style="bright_blue"))

    # intraday signal table
    try:
        console.print(build_signal_table(last_rows))
    except:
        console.print("[grey]No intraday data yet...[/grey]")

    # snapshot
    snap_table = Table(
        title="Snapshot (Current Expiry)",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_lines=True,
        header_style="bold green",
    )
    for c in ["Time","ATM","Total OI","Call OI","Put OI","Difference","PCR","Sentiment"]:
        snap_table.add_column(c)
    snap_table.add_row(*build_snapshot_row(ts, atm, metrics, last_rows))
    console.print(snap_table)

    # history table
    hist = Table(
        title="Last 30 Minutes (2-Min)",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_lines=True,
        header_style="bold magenta"
    )

    hist.add_column("Time")
    hist.add_column("ATM")
    hist.add_column("VWAP")
    hist.add_column("Total OI", justify="right")
    hist.add_column("Call OI", justify="right")
    hist.add_column("Put OI", justify="right")
    hist.add_column("Difference", justify="right")
    hist.add_column("PCR")
    hist.add_column("Sentiment")

    for i, row in enumerate(last_rows):
        prev1 = last_rows[i - 1] if i > 0 else None
        prev2 = last_rows[i - 2] if i > 1 else None

        call_arrow = arrow_trend(
            row["total_call_oi"],
            prev1["total_call_oi"] if prev1 else None,
            prev2["total_call_oi"] if prev2 else None,
        )
        put_arrow = arrow_trend(
            row["total_put_oi"],
            prev1["total_put_oi"] if prev1 else None,
            prev2["total_put_oi"] if prev2 else None,
        )

        vwap_cell = format_vwap_cell(row["vwap"], row["spot"])

        hist.add_row(
            row["time"].strftime("%H:%M:%S"),
            str(row["atm"]),
            vwap_cell,
            f"{row['total_oi']:,}",
            colorize_arrow(row["total_call_oi"], call_arrow),
            colorize_arrow(row["total_put_oi"], put_arrow),
            diff_cell(row["diff"]),
            f"{row['pcr']:.3f}",
            badge(row["sentiment"], row["pcr"]),
        )

    console.print(hist)


# =====================================================================
# 8. MAIN LOOP
# =====================================================================

def main(poll_seconds=ROLLING_INTERVAL_MIN * 60):
    console.rule("[bold]NIFTY OI Monitor + Scalper + VWAP Engine[/bold]")

    last_rows = deque(maxlen=ROLLING_MAX_ROWS)
    last_sample_time = None
    spot_history = deque(maxlen=VWAP_WINDOW_SAMPLES)

    while True:
        try:
            raw = fetch_option_chain()
            trimmed, expiry_str = filter_current_expiry(raw)

            ts = datetime.now()

            atm, by_strike, spot = get_atm_and_slice(trimmed)
            if atm is None or spot is None:
                console.print("[red]No ATM / Spot available[/red]")
                time.sleep(poll_seconds)
                continue

            spot = float(spot)
            spot_history.append(spot)
            metrics = compute_metrics(by_strike)

            expiry_dt = parse_expiry_date(expiry_str) if expiry_str else None
            is_expiry_day = expiry_dt and expiry_dt.date() == ts.date()

            if (
                last_sample_time is None or
                ts - last_sample_time >= timedelta(minutes=ROLLING_INTERVAL_MIN)
            ):
                vwap, _, _ = compute_vwap_and_sr(spot_history)

                legs = by_strike.get(atm, {})
                ce = legs.get("CE") or {}
                pe = legs.get("PE") or {}

                ce_iv = float(ce.get("impliedVolatility") or 0.0)
                pe_iv = float(pe.get("impliedVolatility") or 0.0)

                last_rows.append({
                    "time": ts,
                    "atm": atm,
                    "spot": spot,
                    "vwap": vwap,
                    "total_call_oi": metrics["total_call_oi"],
                    "total_put_oi": metrics["total_put_oi"],
                    "total_oi": metrics["total_oi"],
                    "diff": metrics["diff"],
                    "pcr": metrics["pcr"],
                    "sentiment": metrics["sentiment"],
                    "ce_iv": ce_iv,
                    "pe_iv": pe_iv,
                })

                write_to_file(ts, atm, metrics)
                last_sample_time = ts

            render_rich_dashboard(
                ts, atm, spot, expiry_str,
                metrics, by_strike, last_rows,
                spot_history, is_expiry_day
            )

            console.print(f"[grey]Next update in {ROLLING_INTERVAL_MIN} minutes...[/grey]")
            time.sleep(poll_seconds)

        except KeyboardInterrupt:
            console.print("[yellow]Stopped by user[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
