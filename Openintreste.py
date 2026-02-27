# Openintreste.py — Rich panels: Intraday PCR (latest expiry) for NIFTY & BANKNIFTY
# Data: NSE option-chain (OI + underlyingValue)
# Features:
#   - Dynamic Index View badges with arrows: 🔼 Positive, 🔽 Negative, — Neutral
#   - Last 10 rows per index at strict 3‑minute cadence, newest on top
#   - Buyer signal via PCR bands + slope + momentum proxy

import warnings, logging, time as time_module, requests, random
from datetime import datetime
from typing import Dict, List, Tuple
import config

# Silence warnings/logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

# Settings (override via config.py if desired)
POLL_SECONDS   = getattr(config, "POLL_SECONDS", 180)   # 3 minutes
PCR_LOW_BAND   = getattr(config, "PCR_LOW_BAND", 0.80)
PCR_HIGH_BAND  = getattr(config, "PCR_HIGH_BAND", 1.20)
SLOPE_MIN_MOVE = getattr(config, "SLOPE_MIN_MOVE", 0.02)
INDEX_BADGE_UP = getattr(config, "INDEX_BADGE_UP", 1.02)  # tighter to reduce flicker
INDEX_BADGE_DN = getattr(config, "INDEX_BADGE_DN", 0.98)

# Rich UI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich.live import Live
from rich import box

# Hardened NSE session
HEADERS = {
    "user-agent":"Mozilla/5.0",
    "accept":"application/json, text/plain, */*",
    "referer":"https://www.nseindia.com",
    "accept-language":"en-US,en;q=0.9"
}
sess = requests.Session(); sess.headers.update(HEADERS)

def warmup():
    try:
        sess.get("https://www.nseindia.com", timeout=10)
        sess.get("https://www.nseindia.com/option-chain", timeout=10)
    except Exception: pass
warmup()

def fetch_chain(idx: str) -> Dict:
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={idx.upper()}"
    for _ in range(2):
        try:
            sess.headers.update({"accept-language": random.choice(["en-US,en;q=0.9","en-GB,en;q=0.8"])})
            r = sess.get(url, timeout=12); r.raise_for_status(); return r.json()
        except Exception:
            warmup(); time_module.sleep(1.0)
    return {}

def latest_expiry_summary(chain_json):
    rec = chain_json.get("records", {})
    exps = rec.get("expiryDates", [])
    if not exps: return "", 0, 0.0, 0.0
    exp = exps[0]
    call_oi = put_oi = 0
    for item in rec.get("data,") or rec.get("data", []):
        pass  # defensive; line handled below

    # Proper iteration
    call_oi = put_oi = 0
    for item in rec.get("data", []):
        if item.get("expiryDate") != exp: continue
        ce = item.get("CE"); pe = item.get("PE")
        if not ce or not pe: continue
        try:
            call_oi += int(ce.get("openInterest", 0))
            put_oi  += int(pe.get("openInterest", 0))
        except Exception:
            continue
    diff = put_oi - call_oi
    pcr = round((put_oi / call_oi) if call_oi else 0.0, 2)
    uv = float(rec.get("underlyingValue", 0) or 0)
    return exp, diff, pcr, uv

# PCR slope + momentum proxy from underlyingValue
def slope_ok(series: List[float], up: bool) -> bool:
    if len(series) < 3: return True
    d = series[-1] - series[-3]
    return d >= SLOPE_MIN_MOVE if up else d <= -SLOPE_MIN_MOVE

def buyer_signal(series: List[float], pcr: float, mom_up: bool) -> str:
    if pcr >= PCR_HIGH_BAND and slope_ok(series, True) and mom_up: return "BUY"
    if pcr <= PCR_LOW_BAND and slope_ok(series, False) and (not mom_up): return "SELL"
    return "NEUTRAL"

def classify_index_view(pcr: float, uv_series: List[float]) -> str:
    mom_up = (len(uv_series) < 3) or ((uv_series[-1] - uv_series[-3]) >= 0.0)
    if mom_up and pcr >= INDEX_BADGE_UP:
        return "Positive"
    if (not mom_up) and pcr <= INDEX_BADGE_DN:
        return "Negative"
    return "Neutral"

def header_badge(name: str, view: str) -> str:
    arrow = "🔼" if view == "Positive" else ("🔽" if view == "Negative" else "—")
    color = "green" if view == "Positive" else ("red" if view == "Negative" else "yellow")
    return f"[white]{name}[/] {arrow} ([{color}]{view.lower()}[/{color}])"

def is_new_3min_slot(prev_stamp: str, now_dt: datetime) -> Tuple[bool, str]:
    stamp = now_dt.strftime("%H%M")
    return (now_dt.minute % 3 == 0) and (stamp != prev_stamp), stamp

# Rich rendering
console = Console()

def build_panel(title: str, hist_rows: List[Tuple[str,int,float,str]]):
    t = Table(show_header=True, header_style="bold white", box=box.SIMPLE_HEAVY)
    t.add_column("Time", justify="center", width=6)
    t.add_column("Diff", justify="right", width=14)
    t.add_column("PCR", justify="center", width=6)
    t.add_column("Option Signal", justify="center", width=12)
    for tm, diff, pcr, sig in hist_rows[-10:][::-1]:  # last 10, newest on top
        diff_str = f"{diff:,}"
        pcr_str = f"{pcr:.2f}"
        color = "red" if pcr < 1.0 else "green"
        sig_color = "red" if sig == "SELL" else ("green" if sig == "BUY" else "yellow")
        t.add_row(f"[white]{tm}[/]", f"[{color}]{diff_str}[/]", f"[{color}]{pcr_str}[/]", f"[{sig_color}]{sig}[/]")
    return Panel(Align.center(t), title=title, border_style="yellow", box=box.ROUNDED)

def run():
    nifty_pcr_series: List[float] = []
    bank_pcr_series:  List[float] = []
    nifty_uv_series:  List[float] = []
    bank_uv_series:   List[float] = []
    nifty_hist:       List[Tuple[str,int,float,str]] = []
    bank_hist:        List[Tuple[str,int,float,str]] = []
    last_stamp_n = ""
    last_stamp_b = ""

    with Live(refresh_per_second=4, console=console, screen=False) as live:
        while True:
            now_dt = datetime.now()
            allow_n, stamp_n = is_new_3min_slot(last_stamp_n, now_dt)
            allow_b, stamp_b = is_new_3min_slot(last_stamp_b, now_dt)

            # Fetch chains once per cycle
            n_chain = fetch_chain("NIFTY")
            b_chain = fetch_chain("BANKNIFTY")

            # NIFTY
            n_exp, n_diff, n_pcr, n_uv = latest_expiry_summary(n_chain)
            if n_exp:
                nifty_pcr_series.append(n_pcr)
                if len(nifty_pcr_series) > 200: nifty_pcr_series = nifty_pcr_series[-200:]
                nifty_uv_series.append(n_uv)
                if len(nifty_uv_series) > 5: nifty_uv_series = nifty_uv_series[-5:]
                n_mom_up = (len(nifty_uv_series) < 3) or ((nifty_uv_series[-1] - nifty_uv_series[-3]) >= 0.0)
                if allow_n:
                    n_sig = buyer_signal(nifty_pcr_series, n_pcr, n_mom_up)
                    nifty_hist.append((stamp_n, n_diff, n_pcr, n_sig))
                    if len(nifty_hist) > 180: nifty_hist = nifty_hist[-180:]
                    last_stamp_n = stamp_n
                n_view = classify_index_view(n_pcr, nifty_uv_series)
            else:
                n_view = "Neutral"

            # BANKNIFTY
            b_exp, b_diff, b_pcr, b_uv = latest_expiry_summary(b_chain)
            if b_exp:
                bank_pcr_series.append(b_pcr)
                if len(bank_pcr_series) > 200: bank_pcr_series = bank_pcr_series[-200:]
                bank_uv_series.append(b_uv)
                if len(bank_uv_series) > 5: bank_uv_series = bank_uv_series[-5:]
                b_mom_up = (len(bank_uv_series) < 3) or ((bank_uv_series[-1] - bank_uv_series[-3]) >= 0.0)
                if allow_b:
                    b_sig = buyer_signal(bank_pcr_series, b_pcr, b_mom_up)
                    bank_hist.append((stamp_b, b_diff, b_pcr, b_sig))
                    if len(bank_hist) > 180: bank_hist = bank_hist[-180:]
                    last_stamp_b = stamp_b
                b_view = classify_index_view(b_pcr, bank_uv_series)
            else:
                b_view = "Neutral"

            header = f"\n[bold]INDEX VIEW[/bold]  {header_badge('Nifty', n_view)}    {header_badge('BankNifty', b_view)}\n"

            layout = Table.grid(expand=True)
            layout.add_row(header)
            layout.add_row(build_panel(f"INTRADAY DATA - NIFTY  [dim]{n_exp}[/]", nifty_hist))
            layout.add_row(build_panel(f"INTRADAY DATA - BANKNIFTY  [dim]{b_exp}[/]", bank_hist))
            live.update(layout)

            time_module.sleep(POLL_SECONDS)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="(%(asctime)s) %(levelname)s :: %(message)s")
    run()
