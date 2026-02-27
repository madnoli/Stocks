#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BANKNIFTY OI monitor with dynamic strike selection (ATM ±700, step 100),
2-minute sampling, last 30 minutes (15 samples), Indian number formatting,
writes OI_data.txt (append) and OI_report.html (overwrite).

Requirements:
 - requests
 - rich

Run with: python3 this_script.py
"""

import requests
import time
from datetime import datetime, timedelta
from collections import deque, OrderedDict
import html
from rich.console import Console

API_URL = "http://localhost:3001/api/index/options/BANKNIFTY"

ROLLING_INTERVAL_MIN = 2
ROLLING_MAX_ROWS = 15   # 30 minutes / 2-min samples

# dynamic strike band params
STRIKE_STEP = 100
STRIKE_SPAN = 700  # +/- 700 -> 15 strikes (inclusive)

console = Console()


# ---------------------------
# Utilities
# ---------------------------
def indian_format(n):
    """Format integer or float using Indian number grouping (lakhs/crores)."""
    try:
        neg = False
        if n is None:
            return "-"
        if isinstance(n, (int, float)) and n < 0:
            neg = True
            n = abs(n)
        s = f"{int(round(n)):,}"  # standard grouping first
        # convert to indian grouping
        parts = s.split(",")
        if len(parts) <= 3:
            out = s
        else:
            # join first (most significant) part with last groups of two
            leading = "".join(parts[: len(parts) - 2])
            last_two = parts[-2:]
            out = f"{leading},{last_two[0]},{last_two[1]}"
            # However above approach may fail for larger; use custom below instead:
            s_digits = str(int(round(n)))
            if len(s_digits) <= 3:
                out = s_digits
            else:
                # last 3 digits
                last3 = s_digits[-3:]
                rem = s_digits[:-3]
                rev = rem[::-1]
                grouped = [rev[i : i + 2] for i in range(0, len(rev), 2)]
                rem_grouped = ",".join(g[::-1] for g in grouped[::-1])
                out = rem_grouped + "," + last3
        if neg:
            return "-" + out
        return out
    except Exception:
        # fallback
        try:
            return f"{n:,}"
        except Exception:
            return str(n)


def pretty_float(v, decimals=2):
    if v is None:
        return "-"
    try:
        if isinstance(v, float):
            # keep two decimals for price
            return f"{v:.{decimals}f}"
        return str(v)
    except Exception:
        return str(v)


# ---------------------------
# API + parsing helpers
# ---------------------------
def fetch_option_chain(url=API_URL, timeout=10):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def parse_expiry_date(ed):
    for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(ed, fmt)
        except Exception:
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


def get_atm_and_slice_for_display(data, strikes_each_side=3):
    """
    Returns atm, an ordered dict of ATM±strikes_each_side for per-strike display,
    and spot (underlying).
    This is only for UI per-strike table; NOT used for total OI aggregation.
    """
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


def get_selected_strikes(atm, step=STRIKE_STEP, span=STRIKE_SPAN):
    """Return list of strikes centered at ATM with total ((span/step)*2)+1 strikes."""
    if atm is None:
        return []
    low = atm - span
    high = atm + span
    strikes = list(range(int(low), int(high) + 1, step))
    return strikes


def aggregate_oi_for_strikes(records, selected_strikes):
    """
    Sum call and put OI for the provided selected_strikes across the records.
    Returns (total_call_oi, total_put_oi, mapping_per_strike)
    """
    total_call_oi = 0
    total_put_oi = 0
    per_strike = {}
    # build a map from strike to rec for fast lookup (records contain CE and PE fields per strike)
    strike_map = {}
    for rec in records:
        s = rec.get("strikePrice")
        if s is not None:
            strike_map[s] = rec

    for s in selected_strikes:
        rec = strike_map.get(s)
        ce_oi = 0
        pe_oi = 0
        if rec:
            ce = rec.get("CE") or {}
            pe = rec.get("PE") or {}
            ce_oi = int(ce.get("openInterest") or 0)
            pe_oi = int(pe.get("openInterest") or 0)
        per_strike[s] = {"call_oi": ce_oi, "put_oi": pe_oi}
        total_call_oi += ce_oi
        total_put_oi += pe_oi

    return total_call_oi, total_put_oi, per_strike


def pcr_value(total_put_oi, total_call_oi):
    return round((total_put_oi / total_call_oi), 3) if total_call_oi > 0 else 0.0


def pcr_sentiment(pcr):
    if pcr > 1.3:
        return "Bullish"
    if pcr < 0.7:
        return "Bearish"
    return "Neutral"


# ---------------------------
# Text logging
# ---------------------------
def write_to_file(ts, atm, spot, metrics, selected_strikes):
    """Append CSV-like line for audit."""
    line = (
        f"{ts.strftime('%Y-%m-%d %H:%M:%S')},"
        f"ATM={atm},"
        f"LTP={spot:.2f},"
        f"CallOI={metrics['total_call_oi']},"
        f"PutOI={metrics['total_put_oi']},"
        f"TotalOI={metrics['total_oi']},"
        f"Diff={metrics['diff']},"
        f"PCR={metrics['pcr']:.3f},"
        f"Sentiment={metrics['sentiment']},"
        f"Strikes={'|'.join(map(str, selected_strikes))}\n"
    )
    with open("OI_data.txt", "a") as f:
        f.write(line)


# ---------------------------
# HTML report
# ---------------------------
def render_html_report(rows, day_high, day_low, per_strike_display, output_file="OI_report.html"):
    """
    rows: list of sample dicts, newest-first
    per_strike_display: OrderedDict of ATM±3 strikes for table display (can be empty)
    """

    day_high_txt = f"{day_high:.2f}" if day_high is not None else "N/A"
    day_low_txt = f"{day_low:.2f}" if day_low is not None else "N/A"

    html_header = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>OI Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; background:#f7f7f7; padding:20px; }}
.container {{ background:white; padding:18px; border-radius:8px; max-width:1280px; margin:auto; box-shadow:0 6px 20px rgba(0,0,0,0.06); }}
h2 {{ margin:0 0 10px 0; }}
.small {{ color:#666; margin-bottom:12px; }}
.table {{ width:100%; border-collapse:collapse; font-size:13px; }}
.table th {{ background:#f4f6f8; text-align:left; padding:10px; border-bottom:1px solid #eaeef2; font-weight:700; }}
.table td {{ padding:10px; border-bottom:1px solid #f0f3f6; vertical-align:middle; }}
.center {{ text-align:center; }}
.right {{ text-align:right; }}
.badge {{ padding:6px 10px; border-radius:6px; font-weight:700; font-size:12px; display:inline-block; }}
.green {{ background:#e6f7ec; color:#0a7a37; }}
.red {{ background:#fdecea; color:#b31b1b; }}
.yellow {{ background:#fff6db; color:#8a6b00; }}
.up {{ color:#0b9b3a; font-weight:700; }}
.down {{ color:#c30b0b; font-weight:700; }}
.smallmuted {{ font-size:12px; color:#888; }}
.perstrike {{ margin-top:12px; font-size:13px; }}
</style>
</head>
<body>
<div class="container">
<h2>OI Pulse - Last {len(rows)} Samples (Latest on Top)</h2>
<p class="small"><strong>Day High:</strong> {day_high_txt} &nbsp;&nbsp; <strong>Day Low:</strong> {day_low_txt}</p>

<table class="table" id="oi-table">
<thead>
<tr>
<th>Date</th><th>Time</th><th>LTP</th><th>Day H/L Break</th>
<th class="right">Δ Call OI</th><th class="right">Δ Put OI</th><th class="right">Diff. in OI</th>
<th class="center">Direction</th><th class="right">Chng. In Direction</th><th class="right">Direction of chng. %</th>
<th class="right">Net PCR</th><th class="center">Day High/Low OI</th><th class="center">Sentiment</th>
</tr>
</thead>
<tbody>
"""

    html_rows = []
    for r in rows:
        date_s = r["time"].strftime("%d-%m-%Y")
        time_s = r["time"].strftime("%H:%M:%S")
        ltp_txt = pretty_float(r["spot"], 2)

        # Day H/L Break
        if r.get("day_break") == "high":
            dhl = f'<span class="badge green">Day High Break ({pretty_float(r["spot"],2)})</span>'
        elif r.get("day_break") == "low":
            dhl = f'<span class="badge red">Day Low Break ({pretty_float(r["spot"],2)})</span>'
        else:
            dhl = "-"

        dcall = "-" if r["delta_call_oi"] is None else indian_format(r["delta_call_oi"])
        dput = "-" if r["delta_put_oi"] is None else indian_format(r["delta_put_oi"])
        diff_txt = "-" if r["diff_in_oi"] is None else indian_format(r["diff_in_oi"])

        # direction arrow
        if r["diff_in_oi"] is None:
            direction = "-"
            dir_cls = ""
        elif r["diff_in_oi"] > 0:
            direction = "↑"
            dir_cls = "up"
        elif r["diff_in_oi"] < 0:
            direction = "↓"
            dir_cls = "down"
        else:
            direction = "→"
            dir_cls = ""

        chng_dir = "-" if r["diff_in_oi"] is None else indian_format(abs(r["diff_in_oi"]))
        dir_pct = "-" if r["dir_pct"] is None else f"{r['dir_pct']:.2f} %"
        net_pcr = "-" if r.get("pcr") is None else f"{r['pcr']:.3f}"
        day_hl_txt = r.get("day_hl_diff_text", "-")
        sent = r.get("sentiment", "-")
        if sent == "Bullish":
            sent_html = '<span class="badge green">Bullish</span>'
        elif sent == "Bearish":
            sent_html = '<span class="badge red">Bearish</span>'
        else:
            sent_html = '<span class="badge yellow">Neutral</span>'

        # styled diff cell (green for positive diff, red for negative)
        diff_cell_html = diff_txt
        if diff_txt != "-" and r["diff_in_oi"] is not None:
            if r["diff_in_oi"] > 0:
                diff_cell_html = f'<span class="up">{diff_txt}</span>'
            elif r["diff_in_oi"] < 0:
                diff_cell_html = f'<span class="down">{diff_txt}</span>'

        html_rows.append(
            f"<tr>"
            f"<td>{html.escape(date_s)}</td>"
            f"<td>{html.escape(time_s)}</td>"
            f"<td class='right'>{html.escape(ltp_txt)}</td>"
            f"<td class='center'>{dhl}</td>"
            f"<td class='right'>{dcall}</td>"
            f"<td class='right'>{dput}</td>"
            f"<td class='right'>{diff_cell_html}</td>"
            f"<td class='center {dir_cls}'>{direction}</td>"
            f"<td class='right'>{chng_dir}</td>"
            f"<td class='right'>{dir_pct}</td>"
            f"<td class='right'>{net_pcr}</td>"
            f"<td class='center smallmuted'>{html.escape(day_hl_txt)}</td>"
            f"<td class='center'>{sent_html}</td>"
            f"</tr>"
        )

    html_footer = """
</tbody>
</table>
"""

    # Optionally show per-strike (ATM ±3) table for quick view
    per_strike_html = ""
    if per_strike_display:
        per_strike_html = """
<div class="perstrike">
<h4>ATM ±3 strikes (per-strike OI)</h4>
<table class="table">
<tr><th>Strike</th><th class="right">Call OI</th><th class="right">Put OI</th></tr>
"""
        for s, v in per_strike_display.items():
            per_strike_html += (
                f"<tr><td>{s}</td>"
                f"<td class='right'>{indian_format(v.get('CE_oi', 0))}</td>"
                f"<td class='right'>{indian_format(v.get('PE_oi', 0))}</td></tr>"
            )
        per_strike_html += "</table></div>"

    html_end = """
</div>
</body>
</html>
"""
    content = html_header + "".join(html_rows) + html_footer + per_strike_html + html_end
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)


# ---------------------------
# Main loop
# ---------------------------
def main(poll_seconds=ROLLING_INTERVAL_MIN * 60):
    console.rule("[bold]BANKNIFTY OI Monitor (Dynamic 15 strikes)")

    last_rows = deque(maxlen=ROLLING_MAX_ROWS)  # chronological oldest->newest (we will reverse for HTML)
    last_sample_time = None

    day_high = None
    day_low = None

    while True:
        try:
            raw = fetch_option_chain()
            trimmed, expiry = filter_current_expiry(raw)
            ts = datetime.now()

            # For per-strike display (ATM ±3)
            atm_disp, per_strike_disp, spot = get_atm_and_slice_for_display(trimmed, strikes_each_side=3)
            # For aggregation, we pick dynamic strikes around ATM (B)
            atm_for_selection = atm_disp  # use same ATM detected
            selected_strikes = get_selected_strikes(atm_for_selection, step=STRIKE_STEP, span=STRIKE_SPAN)

            # aggregate across selected strikes
            records = trimmed.get("records", {}).get("data", []) or []
            total_call_oi, total_put_oi, per_strike_map = aggregate_oi_for_strikes(records, selected_strikes)

            # compute metrics using aggregated OI
            total_oi = total_call_oi + total_put_oi
            diff = total_put_oi - total_call_oi
            pcr = pcr_value(total_put_oi, total_call_oi)
            sentiment = pcr_sentiment(pcr)

            metrics = {
                "total_call_oi": total_call_oi,
                "total_put_oi": total_put_oi,
                "total_oi": total_oi,
                "diff": diff,
                "pcr": pcr,
                "sentiment": sentiment,
            }

            # update day high/low using spot (LTP)
            if spot is not None:
                if day_high is None or spot > day_high:
                    day_high = spot
                if day_low is None or spot < day_low:
                    day_low = spot

            # sampling window check
            if (last_sample_time is None) or (ts - last_sample_time >= timedelta(minutes=ROLLING_INTERVAL_MIN)):
                prev = last_rows[-1] if len(last_rows) > 0 else None

                if prev:
                    delta_call = metrics["total_call_oi"] - prev["total_call_oi"]
                    delta_put = metrics["total_put_oi"] - prev["total_put_oi"]
                    diff_in_oi = (delta_put or 0) - (delta_call or 0)
                    prev_total_oi = prev.get("total_oi", 0) or 0
                    dir_pct = (abs(diff_in_oi) / prev_total_oi) * 100 if prev_total_oi > 0 else None
                else:
                    delta_call = None
                    delta_put = None
                    diff_in_oi = None
                    dir_pct = None

                # day high/low break detection compared to previously seen spots in last_rows
                prev_spots = [r["spot"] for r in last_rows] if len(last_rows) > 0 else []
                prev_max = max(prev_spots) if prev_spots else None
                prev_min = min(prev_spots) if prev_spots else None

                if prev_max is None:
                    day_break = None
                else:
                    if spot is not None and spot > prev_max:
                        day_break = "high"
                    elif spot is not None and spot < prev_min:
                        day_break = "low"
                    else:
                        day_break = None

                sample = {
                    "time": ts,
                    "atm": atm_for_selection,
                    "spot": spot,
                    "total_call_oi": metrics["total_call_oi"],
                    "total_put_oi": metrics["total_put_oi"],
                    "total_oi": metrics["total_oi"],
                    "diff": metrics["diff"],
                    "pcr": metrics["pcr"],
                    "sentiment": metrics["sentiment"],
                    "delta_call_oi": delta_call,
                    "delta_put_oi": delta_put,
                    "diff_in_oi": diff_in_oi,
                    "dir_pct": dir_pct,
                    "day_break": day_break,
                    "day_hl_diff_text": f"H:{pretty_float(day_high,2)}/L:{pretty_float(day_low,2)}"
                }

                last_rows.append(sample)
                last_sample_time = ts

                # write to text log
                write_to_file(ts, atm_for_selection, spot, metrics, selected_strikes)

                # Build small per-strike display aggregator for ATM±3 table (CE_oi / PE_oi)
                per_strike_display = OrderedDict()
                # per_strike_disp comes from get_atm_and_slice_for_display and contains CE/PE nested dicts or None
                for s, legs in (per_strike_disp.items() if per_strike_disp else []):
                    ce_oi = (legs["CE"] or {}).get("openInterest", 0) if legs else 0
                    pe_oi = (legs["PE"] or {}).get("openInterest", 0) if legs else 0
                    per_strike_display[s] = {"CE_oi": int(ce_oi or 0), "PE_oi": int(pe_oi or 0)}

                # render HTML newest-first
                rows_html = list(reversed(last_rows))
                render_html_report(rows_html, day_high, day_low, per_strike_display)

                console.print(f"[green]Sampled {ts.strftime('%Y-%m-%d %H:%M:%S')} | ATM {atm_for_selection} | Spot {pretty_float(spot,2)} | PCR {pcr:.3f}[/green]")
                console.print("[green]OI_report.html updated; OI_data.txt appended[/green]")

            time.sleep(poll_seconds)

        except KeyboardInterrupt:
            console.print("[yellow]Stopped by user[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
