#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NIFTY OI Monitor + VWAP/IV/Theta + XGBoost ML integration

- Requires: pip install requests rich xgboost scikit-learn pandas
- Model file expected (optional): pcr_xgb_model.json
- If model not found, script runs without ML predictions.
"""

import os
import requests
import time
from datetime import datetime, timedelta
from collections import deque, OrderedDict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# -------------------------
# Config
# -------------------------
API_URL = "http://localhost:3001/api/index/options/NIFTY"
ROLLING_INTERVAL_MIN = 2
ROLLING_MAX_ROWS = 15
VWAP_WINDOW_SAMPLES = 30
MODEL_PATH = "pcr_xgb_model.json"  # XGBoost model file

# -------------------------
# Try import XGBoost
# -------------------------
ML_AVAILABLE = False
model = None
try:
    from xgboost import XGBClassifier
    ML_AVAILABLE = True
except Exception as e:
    ML_AVAILABLE = False
    console.print("[yellow]XGBoost not installed or import failed. ML disabled.[/yellow]")

# -------------------------
# Basic helpers
# -------------------------
def write_to_file(ts, atm, metrics, extra=None):
    line = (
        f"{ts.strftime('%Y-%m-%d %H:%M:%S')},"
        f"ATM={atm},CallOI={metrics['total_call_oi']},PutOI={metrics['total_put_oi']},"
        f"TotalOI={metrics['total_oi']},Diff={metrics['diff']},PCR={metrics['pcr']:.3f},"
        f"Sentiment={metrics['sentiment']}"
    )
    if extra:
        # include ml signal/prob if present
        if "ml_signal" in extra:
            line += f",ML={extra.get('ml_signal')},ML_prob={extra.get('ml_prob'):.3f}"
    line += "\n"
    with open("OI_data_NIFTY_with_ML.txt", "a") as f:
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


# -------------------------
# ATM & metrics
# -------------------------
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
# -------------------------
# Display helpers and VWAP
# -------------------------
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
    return vwap, min(spot_history), max(spot_history)


# -------------------------
# Theta / IV / VWAP helpers
# -------------------------
def extract_theta(leg):
    if not leg:
        return None
    for key in ("theta", "thetas", "thetaValue"):
        if key in leg and leg.get(key) is not None:
            try:
                return float(leg.get(key))
            except Exception:
                pass
    return None


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


# -------------------------
# Feature building for ML
# -------------------------
def build_ml_features(curr_row, prev_row, prev2_row):
    """
    curr_row: dict with keys spot, vwap, diff, pcr, total_call_oi, total_put_oi, ce_iv, pe_iv
    prev_row / prev2_row: previous sampled rows or None
    Returns a list in the model feature order:
    [pcr, pcr_change, pcr_roc_3, total_call_oi, total_put_oi,
     diff, oi_diff_change, spot, spot_change, spot_vwap_diff, ce_iv, pe_iv]
    """
    pcr = curr_row.get("pcr", 0.0)
    pcr_change = 0.0
    pcr_roc_3 = 0.0
    if prev_row:
        pcr_change = pcr - prev_row.get("pcr", 0.0)
    # compute pcr_roc_3 as curr - prev2
    if prev2_row:
        pcr_roc_3 = pcr - prev2_row.get("pcr", 0.0)
    total_call_oi = curr_row.get("total_call_oi", 0)
    total_put_oi = curr_row.get("total_put_oi", 0)
    diff = curr_row.get("diff", 0)
    oi_diff_change = 0
    if prev_row:
        oi_diff_change = diff - prev_row.get("diff", 0)
    spot = curr_row.get("spot", 0.0)
    spot_change = 0.0
    if prev_row:
        spot_change = spot - prev_row.get("spot", 0.0)
    spot_vwap_diff = spot - (curr_row.get("vwap") or spot)
    ce_iv = curr_row.get("ce_iv") or 0.0
    pe_iv = curr_row.get("pe_iv") or 0.0

    features = [
        pcr,
        pcr_change,
        pcr_roc_3,
        total_call_oi,
        total_put_oi,
        diff,
        oi_diff_change,
        spot,
        spot_change,
        spot_vwap_diff,
        ce_iv,
        pe_iv,
    ]
    return features


# -------------------------
# ML: load model if present
# -------------------------
if ML_AVAILABLE:
    if os.path.exists(MODEL_PATH):
        try:
            model = XGBClassifier()
            model.load_model(MODEL_PATH)
            console.print(f"[green]Loaded ML model from {MODEL_PATH}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to load XGBoost model: {e}[/red]")
            model = None
            ML_AVAILABLE = False
    else:
        console.print(f"[yellow]Model file {MODEL_PATH} not found. ML disabled.[/yellow]")
        model = None
        ML_AVAILABLE = False
# -------------------------
# UI builders: snapshot, intraday table, history
# -------------------------
def build_snapshot_row(ts, atm, metrics, hist_rows):
    prev1 = hist_rows[-1] if len(hist_rows) >= 1 else None
    prev2 = hist_rows[-2] if len(hist_rows) >= 2 else None

    curr_call = metrics["total_call_oi"]
    curr_put = metrics["total_put_oi"]

    call_arrow = arrow_trend(curr_call, prev1["total_call_oi"] if prev1 else None, prev2["total_call_oi"] if prev2 else None)
    put_arrow = arrow_trend(curr_put, prev1["total_put_oi"] if prev1 else None, prev2["total_put_oi"] if prev2 else None)

    last = hist_rows[-1] if hist_rows else None
    ce_iv = f"{last['ce_iv']:.2f}" if last and last.get("ce_iv") is not None else "—"
    pe_iv = f"{last['pe_iv']:.2f}" if last and last.get("pe_iv") is not None else "—"
    ce_theta = f"{last['ce_theta']:.2f}" if last and last.get("ce_theta") is not None else "—"
    pe_theta = f"{last['pe_theta']:.2f}" if last and last.get("pe_theta") is not None else "—"
    ml_sig = f"{last['ml_signal']}" if last and last.get("ml_signal") is not None else "—"
    ml_prob = f"{last['ml_prob']:.2f}" if last and last.get("ml_prob") is not None else "—"

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
        pe_theta,
        ml_sig,
        ml_prob
    ]


def build_signal_table(history_rows):
    t = Table(title="INTRADAY DATA - NIFTY", box=box.MINIMAL_DOUBLE_HEAD, header_style="bold yellow", show_lines=True)
    t.add_column("Time")
    t.add_column("Diff", justify="right")
    t.add_column("PCR", justify="center")
    t.add_column("Option Signal", justify="center")
    t.add_column("VWAP Signal", justify="center")
    t.add_column("VWAP", justify="right")
    t.add_column("ML Signal", justify="center")
    t.add_column("ML Prob", justify="center")

    for row in reversed(history_rows):
        diff = row.get("diff", 0)
        pcr = row.get("pcr", 0.0)
        opt_sig = signal_from_metrics(row)
        v_sig = vwap_signal(row.get("spot"), row.get("vwap"))
        vwap_num_display = format_vwap_cell_intraday(row.get("vwap"), row.get("spot")) if row.get("vwap") is not None else "—"
        ml_sig = row.get("ml_signal") or "—"
        ml_prob = f"{row['ml_prob']:.2f}" if row.get("ml_prob") is not None else "—"

        diff_text = f"[red]{diff:,}[/]" if diff < 0 else f"[green]{diff:,}[/]"
        opt_cell = "[red]SELL[/]" if opt_sig == "SELL" else "[green]BUY[/]" if opt_sig == "BUY" else "[grey]NO TRADE[/]"
        vws = "[green]BUY[/]" if v_sig == "BUY" else "[red]SELL[/]" if v_sig == "SELL" else "[grey]NEUTRAL[/]"

        # color ML signal
        if ml_sig == "BUY":
            ml_cell = "[green]ML BUY[/]"
        elif ml_sig == "SELL":
            ml_cell = "[red]ML SELL[/]"
        else:
            ml_cell = "[grey]ML —[/]"

        t.add_row(
            row["time"].strftime("%H:%M"),
            diff_text,
            f"{pcr:.2f}",
            opt_cell,
            vws,
            vwap_num_display,
            ml_cell,
            ml_prob
        )
    return t


def render_rich_dashboard(ts, atm, spot, expiry_str, metrics, by_strike, last_rows, spot_history, is_expiry_day):
    vwap, support, resistance = compute_vwap_and_sr(spot_history)
    header_text = (
        f"[bold white]NIFTY OI Dashboard[/bold white]\n"
        f"[cyan]Time:[/] {ts.strftime('%Y-%m-%d %H:%M:%S')}  "
        f"[cyan]Spot:[/] {spot:.2f}  "
        f"[cyan]ATM:[/] {atm}  "
        f"[cyan]Expiry:[/] {expiry_str or 'N/A'}\n"
    )
    if vwap is not None:
        cur_vs = vwap_signal(spot, vwap)
        if cur_vs == "BUY":
            header_text += f"[green]VWAP: BUY {vwap:.2f}[/green]  "
        elif cur_vs == "SELL":
            header_text += f"[red]VWAP: SELL {vwap:.2f}[/red]  "
        else:
            header_text += f"[yellow]VWAP: NEUTRAL {vwap:.2f}[/yellow]  "
        header_text += f"[cyan]Support:[/] {support:.2f}  [cyan]Resistance:[/] {resistance:.2f}"
    console.print(Panel.fit(header_text, border_style="bright_blue"))

    # Intraday table
    try:
        console.print(build_signal_table(last_rows))
    except Exception:
        console.print(Panel.fit("[grey]No intraday data yet[/grey]"))

    # Snapshot table
    snap = Table(title="Snapshot (Current Expiry)", box=box.MINIMAL_DOUBLE_HEAD, show_lines=True, header_style="bold green")
    cols = ["Time","ATM","Total OI","Call OI","Put OI","Difference","PCR","Sentiment","CE_IV","PE_IV","CE_Theta","PE_Theta","ML_Sig","ML_Prob"]
    for c in cols:
        snap.add_column(c)
    snap.add_row(*build_snapshot_row(ts, atm, metrics, last_rows))
    console.print(snap)

    # History table with VWAP + IV + Theta + ML
    hist = Table(title="Last 30 Minutes (2-Min)", box=box.MINIMAL_DOUBLE_HEAD, show_lines=True, header_style="bold magenta")
    hist.add_column("Time")
    hist.add_column("ATM", justify="right")
    hist.add_column("VWAP", justify="right")
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
    hist.add_column("ML_Sig", justify="center")
    hist.add_column("ML_Prob", justify="center")

    for i, row in enumerate(last_rows):
        prev1 = last_rows[i-1] if i>0 else None
        prev2 = last_rows[i-2] if i>1 else None
        call_arrow = arrow_trend(row["total_call_oi"], prev1["total_call_oi"] if prev1 else None, prev2["total_call_oi"] if prev2 else None)
        put_arrow = arrow_trend(row["total_put_oi"], prev1["total_put_oi"] if prev1 else None, prev2["total_put_oi"] if prev2 else None)

        vwap_cell = format_vwap_cell_history(row.get("vwap"), row.get("spot"))
        ce_iv_cell = f"{row.get('ce_iv'):.2f}" if row.get("ce_iv") is not None else "—"
        pe_iv_cell = f"{row.get('pe_iv'):.2f}" if row.get("pe_iv") is not None else "—"
        ce_theta_cell = f"{row.get('ce_theta'):.2f}" if row.get("ce_theta") is not None else "—"
        pe_theta_cell = f"{row.get('pe_theta'):.2f}" if row.get("pe_theta") is not None else "—"
        ml_sig = row.get("ml_signal") or "—"
        ml_prob = f"{row['ml_prob']:.2f}" if row.get("ml_prob") is not None else "—"

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
            badge(row["sentiment"], row["pcr"]),
            "[green]ML BUY[/]" if ml_sig=="BUY" else "[red]ML SELL[/]" if ml_sig=="SELL" else "[grey]ML —[/]",
            ml_prob
        )
    console.print(hist)


# -------------------------
# Main loop: sampling, ML, alerts
# -------------------------
def main(poll_seconds=ROLLING_INTERVAL_MIN * 60):
    console.rule("[bold]NIFTY OI Monitor + ML[/bold]")
    last_rows = deque(maxlen=ROLLING_MAX_ROWS)
    last_sample_time = None
    spot_history = deque(maxlen=VWAP_WINDOW_SAMPLES)

    prev_ml_signal = None

    while True:
        try:
            raw = fetch_option_chain()
            trimmed, expiry_str = filter_current_expiry(raw)
            ts = datetime.now()
            atm, by_strike, spot = get_atm_and_slice(trimmed)
            if atm is None or spot is None:
                console.print("[red]No ATM/spot data; retrying...[/red]")
                time.sleep(poll_seconds)
                continue

            spot = float(spot)
            spot_history.append(spot)
            metrics = compute_metrics(by_strike)
            expiry_dt = parse_expiry_date(expiry_str) if expiry_str else None
            is_expiry_day = expiry_dt and expiry_dt.date() == ts.date()

            if last_sample_time is None or ts - last_sample_time >= timedelta(minutes=ROLLING_INTERVAL_MIN):
                # compute VWAP and ATM-leg metrics
                vwap, support, resistance = compute_vwap_and_sr(spot_history)
                legs_atm = by_strike.get(atm, {}) or {}
                ce = legs_atm.get("CE") or {}
                pe = legs_atm.get("PE") or {}

                ce_iv = None
                pe_iv = None
                try:
                    ce_iv = float(ce.get("impliedVolatility")) if ce.get("impliedVolatility") is not None else None
                except:
                    ce_iv = None
                try:
                    pe_iv = float(pe.get("impliedVolatility")) if pe.get("impliedVolatility") is not None else None
                except:
                    pe_iv = None

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
                    "pe_theta": pe_theta,
                    # ml fields to be filled below
                    "ml_signal": None,
                    "ml_prob": None
                }

                # ML prediction if model available
                if ML_AVAILABLE and model is not None:
                    prev_row = last_rows[-1] if len(last_rows) >= 1 else None
                    prev2_row = last_rows[-2] if len(last_rows) >= 2 else None
                    features = build_ml_features(row, prev_row, prev2_row)
                    try:
                        # XGBoost expects 2D
                        prob = model.predict_proba([features])[0]
                        # binary: prob[1] is probability of class 1 (BUY)
                        buy_prob = float(prob[1])
                        pred = int(model.predict([features])[0])
                        ml_sig = "BUY" if pred == 1 else "SELL"
                        row["ml_signal"] = ml_sig
                        row["ml_prob"] = buy_prob
                    except Exception as e:
                        # Model call failed; disable ML to avoid repeated exceptions
                        console.print(f"[red]ML prediction failed: {e}[/red]")
                        row["ml_signal"] = None
                        row["ml_prob"] = None
                # append row and persist
                last_rows.append(row)
                write_to_file(ts, atm, metrics, extra={"ml_signal": row.get("ml_signal"), "ml_prob": row.get("ml_prob") or 0.0})
                last_sample_time = ts

                # Alerts: ML flips
                if ML_AVAILABLE and model is not None and len(last_rows) >= 2:
                    prev = last_rows[-2]
                    curr = last_rows[-1]
                    prev_ml = prev.get("ml_signal")
                    curr_ml = curr.get("ml_signal")
                    if prev_ml != curr_ml and curr_ml in ("BUY", "SELL"):
                        console.print(f"[bold magenta]*** ALERT: ML changed to {curr_ml} at {curr['time'].strftime('%H:%M:%S')} (p={curr.get('ml_prob'):.2f}) ***[/bold magenta]")
                        # terminal bell
                        print("\a", end="")

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
