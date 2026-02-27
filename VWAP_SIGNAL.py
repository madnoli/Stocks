#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NIFTY OI Monitor — Full feature set:
- VWAP in Snapshot + Intraday + History (color + signal + % + arrow)
- CE/PE IV and Theta columns in Snapshot & History
- CE/PE bias uses VWAP + IV trend
- Terminal bell alerts on new BUY/SELL signals
Requirements: pip install requests rich
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
ROLLING_INTERVAL_MIN = 2          # sample every 2 minutes
ROLLING_MAX_ROWS = 15             # ~30 minutes history
VWAP_WINDOW_SAMPLES = 30          # samples used to compute VWAP

console = Console()

# ---------------------------------------------------------------------
# Helpers: API, expiry parsing, write to file
# ---------------------------------------------------------------------
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
    with open("OI_data_NIFTY.txt", "a") as f:
        f.write(line)


def fetch_option_chain(url=API_URL, timeout=10):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def parse_expiry_date(ed):
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(ed, fmt)
        except Exception:
            continue
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


# ---------------------------------------------------------------------
# ATM slice and metrics
# ---------------------------------------------------------------------
def get_atm_and_slice(data, strikes_each_side=3):
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
    return round((total_put_oi / total_call_oi), 3) if total_call_oi and total_call_oi > 0 else 0.0


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
# ---------------------------------------------------------------------
# Display helpers + VWAP
# ---------------------------------------------------------------------
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


def compute_vwap_and_sr(spot_history):
    if not spot_history:
        return None, None, None
    vwap = sum(spot_history) / len(spot_history)
    support = min(spot_history)
    resistance = max(spot_history)
    return vwap, support, resistance


# ---------------------------------------------------------------------
# Signals, VWAP formatting, and helpers for IV/Theta
# ---------------------------------------------------------------------
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


def format_vwap_cell_history(row_vwap, row_spot):
    """
    For History table: color-coded, arrow, pct diff
    BUY ↑ 25918.45 (+0.22%)
    """
    if row_vwap is None or row_spot is None:
        return "—"
    try:
        pct_diff = ((row_spot - row_vwap) / row_vwap) * 100
        if row_spot > row_vwap:
            return f"[green]BUY ↑ {row_vwap:.2f} (+{pct_diff:.2f}%)[/]"
        elif row_spot < row_vwap:
            return f"[red]SELL ↓ {row_vwap:.2f} ({pct_diff:.2f}%)[/]"
        else:
            return f"[yellow]NEUTRAL → {row_vwap:.2f} (0.00%)[/]"
    except Exception:
        return f"{row_vwap:.2f}"


def format_vwap_cell_intraday(vwap, spot):
    """
    For Intraday table show numeric VWAP and signal label
    e.g. [green]BUY 25918.45[/]
    """
    if vwap is None:
        return "—"
    sig = vwap_signal(spot, vwap)
    try:
        if sig == "BUY":
            return f"[green]BUY {vwap:.2f}[/]"
        elif sig == "SELL":
            return f"[red]SELL {vwap:.2f}[/]"
        else:
            return f"[yellow]NEUTRAL {vwap:.2f}[/]"
    except:
        return f"{vwap:.2f}"


def extract_theta(leg):
    """
    Try a few common keys for theta; return None if not present.
    """
    if not leg:
        return None
    for key in ("theta", "thetas", "thetaValue"):
        if key in leg and leg.get(key) is not None:
            try:
                return float(leg.get(key))
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------
# Build Intraday signal table (with VWAP numeric + VWAP Signal)
# ---------------------------------------------------------------------
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
    t.add_column("VWAP", justify="right")  # numeric VWAP

    for row in reversed(history_rows):
        diff = row.get("diff", 0)
        pcr = row.get("pcr", 0.0)
        opt_sig = signal_from_metrics(row)
        v_sig = vwap_signal(row.get("spot"), row.get("vwap"))
        vwap_numeric = row.get("vwap")

        diff_text = f"[red]{diff:,}[/]" if diff < 0 else f"[green]{diff:,}[/]"
        opt_cell = "[red]SELL[/]" if opt_sig == "SELL" else "[green]BUY[/]" if opt_sig == "BUY" else "[grey]NO TRADE[/]"
        vws = "[green]BUY[/]" if v_sig == "BUY" else "[red]SELL[/]" if v_sig == "SELL" else "[grey]NEUTRAL[/]"
        vwap_num_cell = f"{vwap_numeric:.2f}" if vwap_numeric is not None else "—"

        vwap_num_cell = format_vwap_cell_intraday(vwap_numeric, row.get("spot")) if vwap_numeric is not None else "—"

        t.add_row(
            row["time"].strftime("%H:%M"),
            diff_text,
            f"{pcr:.2f}",
            opt_cell,
            vws,
            vwap_num_cell
        )

    return t


# ---------------------------------------------------------------------
# Snapshot builder includes CE/PE IV & Theta display
# ---------------------------------------------------------------------
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

    # last sample CE/PE IV and theta if present
    last = hist_rows[-1] if hist_rows else None
    ce_iv = f"{last['ce_iv']:.2f}" if last and last.get("ce_iv") is not None else "—"
    pe_iv = f"{last['pe_iv']:.2f}" if last and last.get("pe_iv") is not None else "—"
    ce_theta = f"{last['ce_theta']:.2f}" if last and last.get("ce_theta") is not None else "—"
    pe_theta = f"{last['pe_theta']:.2f}" if last and last.get("pe_theta") is not None else "—"

    return [
        ts.strftime("%H:%M:%S"),
        str(atm),
        f"{metrics['total_oi']:,}",
        colorize_arrow(curr_call, call_arrow),
        colorize_arrow(curr_put, put_arrow),
        diff_cell(metrics["diff"]),
        f"{metrics['pcr']:.3f}",
        badge(metrics["sentiment"], metrics["pcr"]),
        ce_iv,
        pe_iv,
        ce_theta,
        pe_theta
    ]
# ---------------------------------------------------------------------
# Render dashboard + history table (with color VWAP + IV + Theta)
# ---------------------------------------------------------------------
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
    # compute simple IV trends
    ce_iv_trend = "Flat"
    pe_iv_trend = "Flat"
    # prev_xx_iv are floats or None
    # we compute trends using latest ATM legs if available
    legs = by_strike.get(atm, {}) if by_strike else {}
    ce = legs.get("CE") or {}
    pe = legs.get("PE") or {}

    ce_iv = float(ce.get("impliedVolatility") or 0.0)
    pe_iv = float(pe.get("impliedVolatility") or 0.0)
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

    # buildup detection (re-using prior logic)
    def side_state(chg_oi, price_chg):
        if chg_oi > 0 and price_chg > 0:
            return "Long Buildup", "[green]Favour Buying[/green]"
        if chg_oi > 0 and price_chg < 0:
            return "Short Buildup", "[red]Avoid Buying[/red]"
        if chg_oi < 0 and price_chg > 0:
            return "Short Covering", "[yellow]Scalp Only[/yellow]"
        return "Neutral", "[white]No Clear Trade[/white]"

    ce_chg_oi = (ce.get("changeinOpenInterest") or 0) or 0
    pe_chg_oi = (pe.get("changeinOpenInterest") or 0) or 0
    ce_price_chg = float(ce.get("change") or 0.0)
    pe_price_chg = float(pe.get("change") or 0.0)

    ce_note, ce_bias = side_state(ce_chg_oi, ce_price_chg)
    pe_note, pe_bias = side_state(pe_chg_oi, pe_price_chg)

    # VWAP direction booleans
    ce_above_vwap = vwap is not None and spot > vwap
    pe_below_vwap = vwap is not None and spot < vwap

    # stronger bias rules: require buildup + VWAP alignment + IV not falling
    reasons = []
    best_side = "No high-probability option buy setup"

    if ce_note == "Long Buildup" and ce_above_vwap and ce_iv_trend != "Falling" and sentiment != "Bearish":
        best_side = "✅ [bold green]Bias: BUY CE[/bold green]"
        reasons.append("CE long buildup + spot above VWAP + IV not falling")
    if pe_note == "Long Buildup" and pe_below_vwap and pe_iv_trend != "Falling" and sentiment != "Bullish":
        if "BUY CE" in best_side:
            best_side += " / [bold green]Also BUY PE possible[/bold green]"
        else:
            best_side = "✅ [bold green]Bias: BUY PE[/bold green]"
        reasons.append("PE long buildup + spot below VWAP + IV not falling")

    if not reasons:
        reasons.append("Conditions not fully aligned for a strong directional bias.")

    # scalper text remains similar
    scalp_ce = ce_note == "Long Buildup" and ce_iv_trend == "Rising" and abs(ce_price_chg) >= max(3, (float(ce.get("lastPrice") or 0.0) * 0.01)) and ce_above_vwap
    scalp_pe = pe_note == "Long Buildup" and pe_iv_trend == "Rising" and abs(pe_price_chg) >= max(3, (float(pe.get("lastPrice") or 0.0) * 0.01)) and pe_below_vwap

    ce_scalper = f"[bold green]⚡ Scalp CE[/bold green]" if scalp_ce else "[red]⛔ Avoid CE[/red]"
    pe_scalper = f"[bold green]⚡ Scalp PE[/bold green]" if scalp_pe else "[red]⛔ Avoid PE[/red]"

    text = (
        f"[bold cyan]ATM Strike Used:[/] {atm}\n"
        f"{best_side}\n\n"
        f"[bold]Fast Scalper Mode (5–20 pts):[/]\n"
        f"{'⚡ CE scalp possible' if scalp_ce else ''} {'⚡ PE scalp possible' if scalp_pe else ''}\n\n"
        f"[bold]CE Side:[/] IV: {ce_iv:.2f} | Trend: {ce_iv_trend}\n"
        f"{ce_scalper}\n\n"
        f"[bold]PE Side:[/] IV: {pe_iv:.2f} | Trend: {pe_iv_trend}\n"
        f"{pe_scalper}\n\n"
        f"[bold]Market-Wide Bias:[/]\n"
        f"  Sentiment: {sentiment} | PCR: {pcr:.3f}\n\n"
        f"[bold]Reasoning:[/]\n"
        + "\n".join(f"- {r}" for r in reasons)
    )
    return Panel.fit(text, border_style="yellow", title="Option Buyers Bias + Fast Scalper Mode")


def render_rich_dashboard(ts, atm, spot, expiry_str, metrics, by_strike, last_rows, spot_history, is_expiry_day):
    # compute current VWAP for header
    vwap, support, resistance = compute_vwap_and_sr(spot_history)
    header_text = (
        f"[bold white]NIFTY OI Dashboard[/bold white]\n"
        f"[cyan]Time:[/] {ts.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"[cyan]Spot:[/] {spot:.2f}  "
        f"[cyan]ATM:[/] {atm}  "
        f"[cyan]Expiry:[/] {expiry_str or 'N/A'}\n"
    )

    if vwap is not None:
        # header VWAP with signal (current spot vs current vwap)
        cur_vs = vwap_signal(spot, vwap)
        if cur_vs == "BUY":
            header_text += f"[green]VWAP: BUY {vwap:.2f}[/green]  "
        elif cur_vs == "SELL":
            header_text += f"[red]VWAP: SELL {vwap:.2f}[/red]  "
        else:
            header_text += f"[yellow]VWAP: NEUTRAL {vwap:.2f}[/yellow]  "
        header_text += f"[cyan]Support:[/] {support:.2f}  [cyan]Resistance:[/] {resistance:.2f}"

    console.print(Panel.fit(header_text, border_style="bright_blue"))

    # Intraday signal table (with numeric VWAP column + VWAP Signal)
    try:
        console.print(build_signal_table(last_rows))
    except Exception:
        console.print(Panel.fit("[grey]No intraday signal history yet[/grey]"))

    # Snapshot table: include CE/PE IV and Theta
    snap = Table(title="Snapshot (Current Expiry)", box=box.MINIMAL_DOUBLE_HEAD, show_lines=True, header_style="bold green")
    snap.add_column("Time")
    snap.add_column("ATM", justify="right")
    snap.add_column("Total OI", justify="right")
    snap.add_column("Call OI", justify="right")
    snap.add_column("Put OI", justify="right")
    snap.add_column("Difference", justify="right")
    snap.add_column("PCR", justify="right")
    snap.add_column("Sentiment", justify="center")
    snap.add_column("CE_IV", justify="right")
    snap.add_column("PE_IV", justify="right")
    snap.add_column("CE_Theta", justify="right")
    snap.add_column("PE_Theta", justify="right")

    snap.add_row(*build_snapshot_row(ts, atm, metrics, last_rows))
    console.print(snap)

    # History table with VWAP color cell + CE/PE IV + Theta
    hist = Table(title="Last 30 Minutes (2-Min)", box=box.MINIMAL_DOUBLE_HEAD, show_lines=True, header_style="bold magenta")
    hist.add_column("Time")
    hist.add_column("ATM", justify="right")
    hist.add_column("VWAP", justify="right")   # formatted (BUY/SELL/NEUTRAL + pct + arrow)
    hist.add_column("Total OI", justify="right")
    hist.add_column("Call OI", justify="right")
    hist.add_column("Put OI", justify="right")
    hist.add_column("CE_IV", justify="right")
    hist.add_column("PE_IV", justify="right")
    hist.add_column("CE_Theta", justify="right")
    hist.add_column("PE_Theta", justify="right")
    hist.add_column("Difference", justify="right")
    hist.add_column("PCR", justify="right")
    hist.add_column("Sentiment", justify="center")

    for i, row in enumerate(last_rows):
        prev1 = last_rows[i - 1] if i > 0 else None
        prev2 = last_rows[i - 2] if i > 1 else None

        call_arrow = arrow_trend(row["total_call_oi"], prev1["total_call_oi"] if prev1 else None, prev2["total_call_oi"] if prev2 else None)
        put_arrow = arrow_trend(row["total_put_oi"], prev1["total_put_oi"] if prev1 else None, prev2["total_put_oi"] if prev2 else None)

        vwap_cell = format_vwap_cell_history(row.get("vwap"), row.get("spot"))

        # CE/PE IV & Theta for this row
        ce_iv_cell = f"{row.get('ce_iv'):.2f}" if row.get("ce_iv") is not None else "—"
        pe_iv_cell = f"{row.get('pe_iv'):.2f}" if row.get("pe_iv") is not None else "—"
        ce_theta_cell = f"{row.get('ce_theta'):.2f}" if row.get("ce_theta") is not None else "—"
        pe_theta_cell = f"{row.get('pe_theta'):.2f}" if row.get("pe_theta") is not None else "—"

        hist.add_row(
            row["time"].strftime("%H:%M:%S"),
            str(row["atm"]),
            vwap_cell,
            f"{row['total_oi']:,}",
            colorize_arrow(row["total_call_oi"], call_arrow),
            colorize_arrow(row["total_put_oi"], put_arrow),
            ce_iv_cell,
            pe_iv_cell,
            ce_theta_cell,
            pe_theta_cell,
            diff_cell(row["diff"]),
            f"{row['pcr']:.3f}",
            badge(row["sentiment"], row["pcr"])
        )

    console.print(hist)


# ---------------------------------------------------------------------
# Main loop: sampling, storing vwap + ce/pe iv/theta, and alerts
# ---------------------------------------------------------------------
def main(poll_seconds=ROLLING_INTERVAL_MIN * 60):
    console.rule("[bold]NIFTY OI Monitor + VWAP/IV/Theta + Alerts[/bold]")

    last_rows = deque(maxlen=ROLLING_MAX_ROWS)
    last_sample_time = None
    spot_history = deque(maxlen=VWAP_WINDOW_SAMPLES)

    # store previous signals to detect changes and fire alerts
    prev_option_signal = None
    prev_vwap_signal = None

    while True:
        try:
            raw = fetch_option_chain()
            trimmed, expiry_str = filter_current_expiry(raw)
            ts = datetime.now()
            atm, by_strike, spot = get_atm_and_slice(trimmed)

            if atm is None or spot is None:
                console.print("[red]No ATM or spot data found; retrying...[/red]")
                time.sleep(poll_seconds)
                continue

            spot = float(spot)
            spot_history.append(spot)

            metrics = compute_metrics(by_strike)
            expiry_dt = parse_expiry_date(expiry_str) if expiry_str else None
            is_expiry_day = expiry_dt and expiry_dt.date() == ts.date()

            # sample per interval
            if last_sample_time is None or ts - last_sample_time >= timedelta(minutes=ROLLING_INTERVAL_MIN):
                vwap, support, resistance = compute_vwap_and_sr(spot_history)

                legs_atm = by_strike.get(atm, {}) or {}
                ce = legs_atm.get("CE") or {}
                pe = legs_atm.get("PE") or {}

                ce_iv = None
                pe_iv = None
                try:
                    ce_iv = float(ce.get("impliedVolatility")) if ce.get("impliedVolatility") is not None else None
                except Exception:
                    ce_iv = None
                try:
                    pe_iv = float(pe.get("impliedVolatility")) if pe.get("impliedVolatility") is not None else None
                except Exception:
                    pe_iv = None

                # theta extraction attempt
                ce_theta = extract_theta(ce)
                pe_theta = extract_theta(pe)

                row = {
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
                    "ce_theta": ce_theta,
                    "pe_theta": pe_theta
                }

                # append sample
                last_rows.append(row)
                write_to_file(ts, atm, metrics)
                last_sample_time = ts

                # --- Alerts: compare latest two samples (if exist)
                if len(last_rows) >= 2:
                    prev = last_rows[-2]
                    curr = last_rows[-1]

                    # option signal alert
                    prev_opt = signal_from_metrics(prev)
                    curr_opt = signal_from_metrics(curr)
                    if prev_opt != curr_opt and curr_opt in ("BUY", "SELL"):
                        console.print(f"[bold red]*** ALERT: Option Signal changed to {curr_opt} at {curr['time'].strftime('%H:%M:%S')} ***[/bold red]")
                        print("\a", end="")  # terminal bell

                    # vwap signal alert
                    prev_vw = vwap_signal(prev.get("spot"), prev.get("vwap"))
                    curr_vw = vwap_signal(curr.get("spot"), curr.get("vwap"))
                    if prev_vw != curr_vw and curr_vw in ("BUY", "SELL"):
                        console.print(f"[bold magenta]*** ALERT: VWAP Signal changed to {curr_vw} at {curr['time'].strftime('%H:%M:%S')} ***[/bold magenta]")
                        print("\a", end="")  # terminal bell

            # render UI
            render_rich_dashboard(ts, atm, spot, expiry_str, metrics, by_strike, list(last_rows), spot_history, is_expiry_day)

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
