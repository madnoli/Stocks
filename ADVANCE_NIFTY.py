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
    Choose the nearest expiry from all expiries present in API response
    and filter the data to only that expiry.
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


# =====================================================================
# 2. ATM & OI METRICS
# =====================================================================

def get_atm_and_slice(data, strikes_each_side=3):
    """
    Find ATM strike and collect CE/PE legs for ATM ± N strikes.
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

    # Ensure full window size if possible
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
    """
    Sum CE/PE OI across the ATM-window and compute PCR.
    """
    total_call_oi = 0
    total_put_oi = 0

    for _, legs in by_strike.items():
        ce_oi = (legs["CE"] or {}).get("openInterest", 0) or 0
        pe_oi = (legs["PE"] or {}).get("openInterest", 0) or 0

        total_call_oi += ce_oi
        total_put_oi += pe_oi

    pcr = pcr_value(total_put_oi, total_call_oi)
    sentiment = pcr_sentiment(pcr)

    return {
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "total_oi": total_call_oi + total_put_oi,
        "diff": total_put_oi - total_call_oi,
        "pcr": pcr,
        "sentiment": sentiment,
    }


# =====================================================================
# 3. DISPLAY HELPERS + VWAP
# =====================================================================

def arrow_trend(curr, prev1, prev2):
    """
    Simple trend arrow based on last 2 samples.
    """
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
    """
    Table for ATM ±3 OI detail.
    """
    t = Table(
        title="ATM ±3 Strikes Detail",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_lines=True,
        header_style="bold cyan",
    )
    t.add_column("Strike", justify="right")
    t.add_column("Call OI", justify="right")
    t.add_column("Put OI", justify="right")

    for s, d in by_strike.items():
        ce = (d["CE"] or {}).get("openInterest", 0) or 0
        pe = (d["PE"] or {}).get("openInterest", 0) or 0
        t.add_row(str(s), f"[bright_cyan]{ce:,}[/]", f"[bright_magenta]{pe:,}[/]")

    return t


def compute_vwap_and_sr(spot_history):
    """
    Approximate VWAP using just spot prices => average(spot)
    + basic support / resistance from recent low / high.
    """
    if not spot_history:
        return None, None, None
    vwap = sum(spot_history) / len(spot_history)
    support = min(spot_history)
    resistance = max(spot_history)
    return vwap, support, resistance


# =====================================================================
# 3.5 NEW: SIGNAL TABLE HELPERS
# =====================================================================

def signal_from_metrics(metrics):
    """
    Decide BUY / SELL / NO TRADE based on Diff & PCR
    (Rules adapted to match the screenshot behavior).
    - diff negative & PCR < 1 -> SELL
    - diff positive & PCR > 1 -> BUY
    - else NO TRADE
    """
    diff = metrics.get("diff", 0)
    pcr = metrics.get("pcr", 0.0)

    if diff < 0 and pcr < 1:
        return "SELL"
    if diff > 0 and pcr > 1:
        return "BUY"
    return "NO TRADE"


def build_signal_table(history_rows):
    """
    Create the SIGNAL TABLE like screenshot:
    Time | Diff | PCR | Option Signal
    Newest rows first.
    """
    table = Table(
        title="INTRADAY DATA - NIFTY",
        box=box.MINIMAL_DOUBLE_HEAD,
        header_style="bold yellow",
        show_lines=True,
    )

    table.add_column("Time", justify="center")
    table.add_column("Diff", justify="right")
    table.add_column("PCR", justify="center")
    table.add_column("Option Signal", justify="center")

    # Show newest first (reverse chronological)
    for row in reversed(history_rows):
        diff = row.get("diff", 0)
        pcr = row.get("pcr", 0.0)
        signal = signal_from_metrics(row)

        # Format diff with sign color
        diff_cell_text = f"[red]{diff:,}[/]" if diff < 0 else f"[green]{diff:,}[/]" if diff > 0 else f"{diff:,}"

        # Format signal color
        if signal == "SELL":
            sig_cell = "[red]SELL[/]"
        elif signal == "BUY":
            sig_cell = "[green]BUY[/]"
        else:
            sig_cell = "[grey]NO TRADE[/]"

        # Time formatting like screenshot (HH:MM)
        time_str = row["time"].strftime("%H:%M")

        table.add_row(time_str, diff_cell_text, f"{pcr:.2f}", sig_cell)

    return table


# =====================================================================
# 4. OPTION BUYER BIAS + FAST SCALPER MODE
# =====================================================================

def option_buyer_bias_panel(
    ts,
    atm,
    by_strike,
    pcr,
    sentiment,
    prev_ce_iv,
    prev_pe_iv,
    is_expiry_day,
    spot,
    vwap,
):
    """
    Decide:
      - Directional bias (BUY CE/PE or no trade)
      - Fast scalper opportunities (5–20 pts)
      - And print extra columns: Scalper, Target, SL, Comment for CE/PE.
    """
    if not by_strike:
        return Panel.fit(
            "[red]No strike data for bias[/red]",
            border_style="red",
            title="Option Buyers Bias + Fast Scalper Mode",
        )

    # Use ATM; if missing, fall back to first key
    if atm not in by_strike:
        strike_used = next(iter(by_strike.keys()))
    else:
        strike_used = atm

    legs = by_strike[strike_used]
    ce = legs.get("CE") or {}
    pe = legs.get("PE") or {}

    ce_chg_oi = ce.get("changeinOpenInterest", 0) or 0
    pe_chg_oi = pe.get("changeinOpenInterest", 0) or 0

    ce_iv = float(ce.get("impliedVolatility", 0.0) or 0.0)
    pe_iv = float(pe.get("impliedVolatility", 0.0) or 0.0)

    ce_price_chg = float(ce.get("change", 0.0) or 0.0)
    pe_price_chg = float(pe.get("change", 0.0) or 0.0)

    ce_ltp = float(ce.get("lastPrice", 0.0) or 0.0)
    pe_ltp = float(pe.get("lastPrice", 0.0) or 0.0)

    # ---- IV Trend Detection (Rising / Falling / Flat) ----
    ce_iv_trend = "Flat"
    pe_iv_trend = "Flat"
    if prev_ce_iv is not None:
        if ce_iv > prev_ce_iv * 1.02:
            ce_iv_trend = "Rising"
        elif ce_iv < prev_ce_iv * 0.98:
            ce_iv_trend = "Falling"
    if prev_pe_iv is not None:
        if pe_iv > prev_pe_iv * 1.02:
            pe_iv_trend = "Rising"
        elif pe_iv < prev_pe_iv * 0.98:
            pe_iv_trend = "Falling"

    # ---- Buildup Type (Long / Short / Short Covering / Neutral) ----
    def side_state(chg_oi, price_chg):
        if chg_oi > 0 and price_chg > 0:
            return "Long Buildup", "[green]Favour Buying[/green]"
        if chg_oi > 0 and price_chg < 0:
            return "Short Buildup", "[red]Avoid Buying[/red]"
        if chg_oi < 0 and price_chg > 0:
            return "Short Covering", "[yellow]Scalp Only[/yellow]"
        return "Neutral", "[white]No Clear Trade[/white]"

    ce_note, ce_bias = side_state(ce_chg_oi, ce_price_chg)
    pe_note, pe_bias = side_state(pe_chg_oi, pe_price_chg)

    # ---- VWAP Filters ----
    ce_above_vwap = vwap is not None and spot > vwap
    pe_below_vwap = vwap is not None and spot < vwap

    # ---- Directional Bias (for normal intraday) ----
    best_side = "No high-probability option buy setup"
    reasons = []

    # Directional BUY CE logic
    if (
        ce_note == "Long Buildup"
        and sentiment != "Bearish"
        and ce_iv_trend != "Falling"
        and ce_above_vwap
    ):
        best_side = "✅ [bold green]Bias: BUY CE[/bold green]"
        reasons.append("CE long buildup + spot above VWAP + IV not falling")

    # Directional BUY PE logic
    if (
        pe_note == "Long Buildup"
        and sentiment != "Bullish"
        and pe_iv_trend != "Falling"
        and pe_below_vwap
    ):
        if "BUY CE" in best_side:
            best_side += " / [bold green]Also BUY PE possible[/bold green]"
        else:
            best_side = "✅ [bold green]Bias: BUY PE[/bold green]"
        reasons.append("PE long buildup + spot below VWAP + IV not falling")

    # ---- FAST SCALPER MODE (5–20 pts) ----
    def scalp_flag(note, iv_trend, price_chg, vwap_ok, sentiment_ok, ltp):
        """
        Returns True when a high-energy 1–3 candle scalp is ok.
        """
        if ltp <= 0:
            return False
        strong_move = abs(price_chg) >= max(3, ltp * 0.01)  # >= 3 pts or ~1%
        return (
            note == "Long Buildup"
            and iv_trend == "Rising"
            and strong_move
            and vwap_ok
            and sentiment_ok
        )

    scalp_ce = scalp_flag(
        ce_note,
        ce_iv_trend,
        ce_price_chg,
        ce_above_vwap,
        sentiment != "Bearish",
        ce_ltp,
    )
    scalp_pe = scalp_flag(
        pe_note,
        pe_iv_trend,
        pe_price_chg,
        pe_below_vwap,
        sentiment != "Bullish",
        pe_ltp,
    )

    # ---- Text generator for CE/PE scalper columns ----
    def scalper_text(flag, option, iv_trend, note, vwap_ok):
        """
        Build nice text block showing:
            - Scalper action
            - Target / SL
            - Comment (buildup, IV, VWAP).
        """
        if not flag:
            return (
                f"[red]⛔ Avoid {option}[/red]\n"
                f"[grey]Comment:[/] {note}, IV: {iv_trend}, "
                f"VWAP: {'OK' if vwap_ok else 'No'}"
            )

        return (
            f"[bold green]⚡ Scalp {option}[/bold green]\n"
            f"[bold yellow]Target:[/] 5–20 pts  |  "
            f"[bold red]SL:[/] 5–10 pts\n"
            f"[grey]Comment:[/] {note}, IV: {iv_trend}, "
            f"VWAP: {'OK' if vwap_ok else 'No'}"
        )

    ce_scalper = scalper_text(scalp_ce, "CE", ce_iv_trend, ce_note, ce_above_vwap)
    pe_scalper = scalper_text(scalp_pe, "PE", pe_iv_trend, pe_note, pe_below_vwap)

    # ---- High-level text if NO scalper at all ----
    scalper_summary = []
    if scalp_ce:
        scalper_summary.append(
            "⚡ CE scalp possible (look for 5–20 pts move within 1–3 candles)."
        )
    if scalp_pe:
        scalper_summary.append(
            "⚡ PE scalp possible (look for 5–20 pts move within 1–3 candles)."
        )
    if not scalper_summary:
        scalper_summary.append(
            "No high-conviction scalper setup right now "
            "(wait for clean long buildup + IV spike)."
        )
    scalper_summary_text = "\n".join(scalper_summary)

    # ---- Expiry scalper mode note ----
    mode_line = ""
    if is_expiry_day:
        mode_line = (
            "[bold magenta]Expiry Fast Scalper Mode:[/] "
            "focus on 5–10 pt quick moves only.\n"
        )
        if ts.hour >= 14 and ts.minute >= 30:
            mode_line += (
                "[red]Post 14:30 on expiry: only ultra-quick scalps; "
                "avoid holding options for more than a few minutes.[/red]\n"
            )

    reason_text = (
        "\n".join(f"- {r}" for r in reasons)
        if reasons
        else "- Conditions not fully aligned for a strong directional bias."
    )

    text = (
        f"[bold cyan]ATM Strike Used:[/] {strike_used}\n"
        f"{best_side}\n\n"
        f"{mode_line}"
        f"[bold]Fast Scalper Mode (5–20 pts):[/]\n"
        f"{scalper_summary_text}\n\n"
        f"[bold]CE Side:[/]\n"
        f"{ce_scalper}\n\n"
        f"[bold]PE Side:[/]\n"
        f"{pe_scalper}\n\n"
        f"[bold]Market-Wide Bias:[/]\n"
        f"  Sentiment: {sentiment} | PCR: {pcr:.3f}\n\n"
        f"[bold]Reasoning:[/]\n"
        f"{reason_text}"
    )

    return Panel.fit(
        text,
        border_style="yellow",
        title="Option Buyers Bias + Fast Scalper Mode",
    )


# =====================================================================
# 5. SNAPSHOT & HISTORY TABLES
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


def render_rich_dashboard(
    ts,
    atm,
    spot,
    expiry_str,
    metrics,
    by_strike,
    last_rows,
    spot_history,
    is_expiry_day,
):
    """
    Main UI: header, snapshot, scalper panel, ATM-table, history.
    The user's requested intraday signal table is printed immediately after the header (Option B).
    """
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
            f"[cyan]VWAP(approx):[/] {vwap:.2f}  "
            f"[cyan]Support:[/] {support:.2f}  "
            f"[cyan]Resistance:[/] {resistance:.2f}"
        )

    header = Panel.fit(header_text, border_style="bright_blue")

    console.print(header)

    # --------------------------
    # NEW: Print the INTRADAY SIGNAL TABLE immediately after header (Option B)
    # --------------------------
    try:
        signal_table = build_signal_table(last_rows)
        console.print(signal_table)
    except Exception:
        # safety: if last_rows empty or error, print a small notice
        console.print(Panel.fit("[grey70]No intraday signal history yet[/grey70]"))

    # --------------------------
    # Continue with existing UI
    # --------------------------
    snap = Table(
        title="Snapshot (Current Expiry)",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_lines=True,
        header_style="bold green",
    )
    snap.add_column("Time")
    snap.add_column("ATM", justify="right")
    snap.add_column("Total OI", justify="right")
    snap.add_column("Call OI", justify="right")
    snap.add_column("Put OI", justify="right")
    snap.add_column("Difference", justify="right")
    snap.add_column("PCR", justify="right")
    snap.add_column("Sentiment", justify="center")
    snap.add_row(*build_snapshot_row(ts, atm, metrics, last_rows))

    hist = Table(
        title="Last 30 Minutes (2-Min)",
        box=box.MINIMAL_DOUBLE_HEAD,
        show_lines=True,
        header_style="bold magenta",
    )
    hist.add_column("Time")
    hist.add_column("ATM", justify="right")
    hist.add_column("Total OI", justify="right")
    hist.add_column("Call OI", justify="right")
    hist.add_column("Put OI", justify="right")
    hist.add_column("Difference", justify="right")
    hist.add_column("PCR", justify="right")
    hist.add_column("Sentiment", justify="center")

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

    prev_ce_iv = last_rows[-1]["ce_iv"] if last_rows else None
    prev_pe_iv = last_rows[-1]["pe_iv"] if last_rows else None

    bias_panel = option_buyer_bias_panel(
        ts,
        atm,
        by_strike,
        metrics["pcr"],
        metrics["sentiment"],
        prev_ce_iv,
        prev_pe_iv,
        is_expiry_day,
        spot,
        vwap,
    )

    console.print(snap)
    console.print(bias_panel)
    console.print(per_strike_table(by_strike))
    console.print(hist)


# =====================================================================
# 6. MAIN LOOP
# =====================================================================

def main(poll_seconds=ROLLING_INTERVAL_MIN * 60):
    console.rule(
        "[bold]NIFTY OI Monitor + Option Buyers Bias + Fast Scalper Mode (5–20 pts)"
    )

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
                console.print("[red]No ATM or strike/spot data found[/red]")
                time.sleep(poll_seconds)
                continue

            spot = float(spot)
            spot_history.append(spot)

            metrics = compute_metrics(by_strike)

            expiry_dt = parse_expiry_date(expiry_str) if expiry_str else None
            is_expiry_day = expiry_dt.date() == ts.date() if expiry_dt else False

            # Sample history every ROLLING_INTERVAL_MIN minutes
            if (
                last_sample_time is None
                or ts - last_sample_time >= timedelta(minutes=ROLLING_INTERVAL_MIN)
            ):
                legs_atm = by_strike.get(atm, {})
                ce = legs_atm.get("CE") or {}
                pe = legs_atm.get("PE") or {}
                ce_iv = float(ce.get("impliedVolatility", 0.0) or 0.0)
                pe_iv = float(pe.get("impliedVolatility", 0.0) or 0.0)

                last_rows.append(
                    {
                        "time": ts,
                        "atm": atm,
                        "total_call_oi": metrics["total_call_oi"],
                        "total_put_oi": metrics["total_put_oi"],
                        "total_oi": metrics["total_oi"],
                        "diff": metrics["diff"],
                        "pcr": metrics["pcr"],
                        "sentiment": metrics["sentiment"],
                        "ce_iv": ce_iv,
                        "pe_iv": pe_iv,
                    }
                )

                write_to_file(ts, atm, metrics)
                last_sample_time = ts

            render_rich_dashboard(
                ts,
                atm,
                spot,
                expiry_str,
                metrics,
                by_strike,
                list(last_rows),
                spot_history,
                is_expiry_day,
            )

            console.print(
                f"[grey62]Next update in {ROLLING_INTERVAL_MIN} minutes...[/grey62]"
            )
            time.sleep(poll_seconds)

        except KeyboardInterrupt:
            console.print("[bold yellow]Stopped by user[/bold yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
