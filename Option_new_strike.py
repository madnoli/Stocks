# openintreste_entry_last5_fixed.py
# Intraday PCR (latest expiry) for NIFTY & BANKNIFTY with entries + smart strike + robust fetch
# Improvements:
#   - Commit first sample immediately; then enforce 3-minute cadence
#   - Stronger warmup/retries for NSE endpoint
#   - underlyingValue fallback path
#   - Keep & display only last 5 rows per index
#   - Trade blotter + smart ATM/ITM strike selection

import warnings, logging, time as time_module, requests, random
from datetime import datetime, time as dt_time
from typing import Dict, List, Tuple, Optional

# Optional config.py overrides
try:
    import config  # user-defined overrides
except Exception:
    class config:
        pass

# Base settings
POLL_SECONDS    = getattr(config, "POLL_SECONDS", 180)      # 3 minutes
PCR_LOW_BAND    = getattr(config, "PCR_LOW_BAND", 0.80)
PCR_HIGH_BAND   = getattr(config, "PCR_HIGH_BAND", 1.20)
SLOPE_MIN_MOVE  = getattr(config, "SLOPE_MIN_MOVE", 0.02)
INDEX_BADGE_UP  = getattr(config, "INDEX_BADGE_UP", 1.02)
INDEX_BADGE_DN  = getattr(config, "INDEX_BADGE_DN", 0.98)

# Entry/Exit Tunables
ENABLE_TIME_FILTER   = getattr(config, "ENABLE_TIME_FILTER", True)
START_TIME           = getattr(config, "START_TIME", "11:15")   # HH:MM local
END_TIME             = getattr(config, "END_TIME", "14:45")
NIFTY_OPTION_SL      = getattr(config, "NIFTY_OPTION_SL", 18)   # premium points
NIFTY_OPTION_TP_R    = getattr(config, "NIFTY_OPTION_TP_R", 1.8)
BANK_OPTION_SL       = getattr(config, "BANK_OPTION_SL", 28)
BANK_OPTION_TP_R     = getattr(config, "BANK_OPTION_TP_R", 1.8)
ALLOW_FIRST_SIGNAL_ONLY = getattr(config, "ALLOW_FIRST_SIGNAL_ONLY", False)

EMA_FAST_LEN         = getattr(config, "EMA_FAST_LEN", 9)
EMA_SLOW_LEN         = getattr(config, "EMA_SLOW_LEN", 21)
PCR_BUY_ZONE         = getattr(config, "PCR_BUY_ZONE", (0.70, 1.15))   # for calls
PCR_SELL_ZONE        = getattr(config, "PCR_SELL_ZONE", (0.90, 1.50))  # for puts
USE_RECLAIM_CANDLE   = getattr(config, "USE_RECLAIM_CANDLE", True)

# Strike step sizes
NIFTY_STRIKE_STEP = 50
BANK_STRIKE_STEP  = 100

# Silence warnings/logging
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.WARNING)

# Rich UI
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.align import Align
from rich.live import Live
from rich import box

# Hardened NSE session (community patterns) [web:34][web:26]
HEADERS = {
    "user-agent":"Mozilla/5.0",
    "accept":"application/json, text/plain, */*",
    "referer":"https://www.nseindia.com",
    "accept-language":"en-US,en;q=0.9"
}
sess = requests.Session(); sess.headers.update(HEADERS)

def warmup():
    # stronger warmup with small sleeps
    for _ in range(2):
        try:
            sess.get("https://www.nseindia.com", timeout=20)
            time_module.sleep(1.0)
            sess.get("https://www.nseindia.com/option-chain", timeout=20)
            time_module.sleep(1.0)
        except Exception:
            time_module.sleep(1.0)

warmup()

def fetch_chain(idx: str) -> Dict:
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={idx.upper()}"
    for backoff in [0, 1, 2, 3]:
        try:
            sess.headers.update({"accept-language": random.choice(["en-US,en;q=0.9","en-GB,en;q=0.8"])})
            r = sess.get(url, timeout=20); r.raise_for_status(); return r.json()
        except Exception:
            warmup(); time_module.sleep(1.0 + backoff)
    return {}

def latest_expiry_summary(chain_json):
    rec = chain_json.get("records", {}) or {}
    exps = rec.get("expiryDates", []) or []
    if not exps: return "", 0, 0.0, 0.0
    exp = exps[0]
    call_oi = put_oi = 0
    for item in rec.get("data", []) or []:
        if item.get("expiryDate") != exp: continue
        ce = item.get("CE"); pe = item.get("PE")
        if not ce or not pe: continue
        call_oi += int((ce.get("openInterest") or 0) or 0)
        put_oi  += int((pe.get("openInterest") or 0) or 0)
    diff = put_oi - call_oi
    pcr = round((put_oi / call_oi), 2) if call_oi else 0.0
    uv = rec.get("underlyingValue")
    if uv is None:
        # fallback seen in some responses
        uv = chain_json.get("filtered", {}).get("data", [{}])[0].get("CE", {}).get("underlyingValue", 0)
    uv = float(uv or 0)
    return exp, diff, pcr, uv

# EMA helpers (on small lists)
def ema_series(values: List[float], length: int) -> List[float]:
    if length <= 1 or len(values) == 0:
        return values[:]
    k = 2.0 / (length + 1)
    out = []
    ema_val = values[0]
    out.append(ema_val)
    for v in values[1:]:
        ema_val = v * k + ema_val * (1 - k)
        out.append(ema_val)
    return out

def last_ema(values: List[float], length: int) -> Optional[float]:
    if not values: return None
    return ema_series(values, length)[-1]

# PCR slope + momentum proxy
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
    if mom_up and pcr >= INDEX_BADGE_UP: return "Positive"
    if (not mom_up) and pcr <= INDEX_BADGE_DN: return "Negative"
    return "Neutral"

def header_badge(name: str, view: str) -> str:
    arrow = "🔼" if view == "Positive" else ("🔽" if view == "Negative" else "—")
    color = "green" if view == "Positive" else ("red" if view == "Negative" else "yellow")
    return f"[white]{name}[/] {arrow} ([{color}]{view.lower()}[/{color}])"

# Commit first sample immediately, then every 3 minutes
def is_new_3min_slot(prev_stamp: Optional[str], now_dt: datetime):
    stamp = now_dt.strftime("%H%M")
    if prev_stamp is None:
        return True, stamp
    return (now_dt.minute % 3 == 0) and (stamp != prev_stamp), stamp

# Utilities
def parse_hhmm(hhmm: str) -> dt_time:
    h, m = hhmm.split(":")
    return dt_time(int(h), int(m))

def in_time_window(now_dt: datetime) -> bool:
    if not ENABLE_TIME_FILTER: return True
    st = parse_hhmm(START_TIME); et = parse_hhmm(END_TIME)
    return st <= now_dt.time() <= et

def round_to_step(x: float, step: int) -> int:
    return int(round(x / step) * step)

# Candle/EMA proxy state for reclaim/reject triggers
class PriceState:
    def __init__(self):
        self.uv_closes: List[float] = []  # last slot closes
    def add_close(self, uv: float):
        self.uv_closes.append(uv)
        if len(self.uv_closes) > 50: self.uv_closes = self.uv_closes[-50:]

# Smart strike selection: ATM default; 1-step ITM on strong trends. [web:43][web:48]
def uv_volatility(uv_series: List[float]) -> float:
    if len(uv_series) < 5: return 0.0
    diffs = [abs(uv_series[i] - uv_series[i-1]) for i in range(1, len(uv_series))]
    return sum(diffs) / len(diffs)

def smart_strike(symbol: str,
                 side: str,
                 spot: float,
                 step: int,
                 uv_series: List[float],
                 view: str,
                 pcr: float) -> int:
    atm = round_to_step(spot, step)
    ema_fast = last_ema(uv_series, EMA_FAST_LEN) or spot
    ema_slow = last_ema(uv_series, EMA_SLOW_LEN) or spot
    strong_up = (view == "Positive") and (spot > ema_slow) and (spot > ema_fast)
    strong_dn = (view == "Negative") and (spot < ema_slow) and (spot < ema_fast)
    if side == "CALL":
        pick = atm - step if strong_up else atm
    else:
        pick = atm + step if strong_dn else atm
    return pick

def trend_bias_adjustment(spot: float, ema_slow: float, vol: float, step: int) -> int:
    if ema_slow == 0: return 0
    dist = abs(spot - ema_slow)
    if vol > 0 and dist > 1.5 * vol:
        return step
    return 0

# Trade state
class Trade:
    def __init__(self, symbol: str, side: str, entry_uv: float, entry_pcr: float, strike: int, sl_pts: int, tp_r: float, premium_est: float):
        self.symbol = symbol
        self.side = side  # "CALL" or "PUT"
        self.entry_uv = entry_uv
        self.entry_pcr = entry_pcr
        self.strike = strike
        self.sl_pts = sl_pts
        self.tp_r = tp_r
        self.premium_est = premium_est
        self.open_time = datetime.now().strftime("%H:%M")
        self.exit_time: Optional[str] = None
        self.exit_reason: Optional[str] = None
        self.active = True
    def target_points(self) -> int:
        return int(round(self.sl_pts * self.tp_r))

def blotter_table(rows: List[List[str]]) -> Panel:
    t = Table(show_header=True, header_style="bold white", box=box.SIMPLE_HEAVY, expand=True)
    t.add_column("Time", width=5)
    t.add_column("Symbol", width=10)
    t.add_column("Side", width=6)
    t.add_column("Strike", width=8)
    t.add_column("SL", width=5)
    t.add_column("TP", width=5)
    t.add_column("PCR", width=5)
    t.add_column("Reason", overflow="fold")
    for r in rows[-10:][::-1]:
        t.add_row(*r)
    return Panel(t, title="TRADE BLOTTER", border_style="cyan", box=box.ROUNDED)

def try_entry(symbol: str,
              pcr: float,
              nview: str,
              uv_series: List[float],
              price_state: PriceState,
              pcr_zone: Tuple[float,float],
              sl_pts: int,
              tp_r: float,
              step: int,
              allow_first_only_flag: bool,
              last_side_taken: Optional[str]) -> Tuple[Optional[Trade], Optional[str]]:
    if nview == "Neutral": return None, None
    low, high = pcr_zone
    if not (low <= pcr <= high): return None, None

    ema_fast = last_ema(uv_series, EMA_FAST_LEN)
    ema_slow = last_ema(uv_series, EMA_SLOW_LEN)
    if ema_fast is None or ema_slow is None: return None, None
    last_uv = uv_series[-1]

    trigger_long = trigger_short = False
    if USE_RECLAIM_CANDLE and len(price_state.uv_closes) >= 2:
        prev_close = price_state.uv_closes[-2]
        curr_close = price_state.uv_closes[-1]
        if nview == "Positive" and prev_close <= ema_fast and curr_close > ema_fast and last_uv > ema_slow:
            trigger_long = True
        if nview == "Negative" and prev_close >= ema_fast and curr_close < ema_fast and last_uv < ema_slow:
            trigger_short = True
    else:
        if nview == "Positive" and last_uv > ema_fast >= ema_slow:
            trigger_long = True
        if nview == "Negative" and last_uv < ema_fast <= ema_slow:
            trigger_short = True

    if allow_first_only_flag and last_side_taken is not None:
        if last_side_taken == "CALL" and trigger_long: return None, None
        if last_side_taken == "PUT" and trigger_short: return None, None

    if trigger_long and nview == "Positive":
        base_strike = smart_strike(symbol, "CALL", last_uv, step, uv_series, nview, pcr)
        ema_s = ema_slow or last_uv
        vol = uv_volatility(uv_series)
        bias = trend_bias_adjustment(last_uv, ema_s, vol, step)
        strike = round_to_step(base_strike - bias, step)
        tr = Trade(symbol, "CALL", last_uv, pcr, strike, sl_pts, tp_r, sl_pts * 2)
        reason = f"Positive + PCR zone {low}-{high}, 9/21 reclaim; strike={strike} (ATM/ITM bias)"
        return tr, reason

    if trigger_short and nview == "Negative":
        base_strike = smart_strike(symbol, "PUT", last_uv, step, uv_series, nview, pcr)
        ema_s = ema_slow or last_uv
        vol = uv_volatility(uv_series)
        bias = trend_bias_adjustment(last_uv, ema_s, vol, step)
        strike = round_to_step(base_strike + bias, step)
        tr = Trade(symbol, "PUT", last_uv, pcr, strike, sl_pts, tp_r, sl_pts * 2)
        reason = f"Negative + PCR zone {low}-{high}, 9/21 reject; strike={strike} (ATM/ITM bias)"
        return tr, reason

    return None, None

def should_exit(tr: Trade,
                pcr: float,
                nview: str,
                uv_series: List[float]) -> Optional[str]:
    if tr.side == "CALL" and nview == "Negative":
        return "View flip against (to Negative)"
    if tr.side == "PUT" and nview == "Positive":
        return "View flip against (to Positive)"
    ema_fast = last_ema(uv_series, EMA_FAST_LEN)
    if ema_fast is None: return None
    last_uv = uv_series[-1]
    if tr.side == "CALL" and last_uv < ema_fast:
        return "Close under 9 EMA (call invalidation)"
    if tr.side == "PUT" and last_uv > ema_fast:
        return "Close over 9 EMA (put invalidation)"
    if tr.side == "CALL" and pcr < PCR_LOW_BAND:
        return "PCR dropped under low band"
    if tr.side == "PUT" and pcr > PCR_HIGH_BAND:
        return "PCR rose above high band"
    return None

# Rich components
console = Console()

def build_panel(title: str, hist_rows: List[Tuple[str,int,float,str]]):
    t = Table(show_header=True, header_style="bold white", box=box.SIMPLE_HEAVY)
    t.add_column("Time", justify="center", width=6)
    t.add_column("Diff", justify="right", width=14)
    t.add_column("PCR", justify="center", width=6)
    t.add_column("Option Signal", justify="center", width=12)
    # Show last 5 rows only, newest on top [web:73][web:74]
    for tm, diff, pcr, sig in hist_rows[-5:][::-1]:
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
    last_stamp_n: Optional[str] = None  # first-sample commit
    last_stamp_b: Optional[str] = None

    ps_n = PriceState()
    ps_b = PriceState()

    open_trade_n: Optional[Trade] = None
    open_trade_b: Optional[Trade] = None
    last_side_taken_n: Optional[str] = None
    last_side_taken_b: Optional[str] = None
    blotter_rows: List[List[str]] = []

    with Live(refresh_per_second=4, console=console, screen=False) as live:
        while True:
            now_dt = datetime.now()
            allow_n, stamp_n = is_new_3min_slot(last_stamp_n, now_dt)
            allow_b, stamp_b = is_new_3min_slot(last_stamp_b, now_dt)

            # Fetch once per cycle
            n_chain = fetch_chain("NIFTY")
            b_chain = fetch_chain("BANKNIFTY")

            # NIFTY
            n_exp, n_diff, n_pcr, n_uv = latest_expiry_summary(n_chain)
            if n_exp:
                nifty_pcr_series.append(n_pcr)
                if len(nifty_pcr_series) > 200: nifty_pcr_series = nifty_pcr_series[-200:]
                nifty_uv_series.append(n_uv)
                if len(nifty_uv_series) > 60: nifty_uv_series = nifty_uv_series[-60:]
                if allow_n:
                    ps_n.add_close(n_uv)
                n_mom_up = (len(nifty_uv_series) < 3) or ((nifty_uv_series[-1] - nifty_uv_series[-3]) >= 0.0)
                if allow_n:
                    n_sig = buyer_signal(nifty_pcr_series, n_pcr, n_mom_up)
                    nifty_hist.append((stamp_n, n_diff, n_pcr, n_sig))
                    if len(nifty_hist) > 5: nifty_hist = nifty_hist[-5:]  # keep last 5
                    last_stamp_n = stamp_n
                n_view = classify_index_view(n_pcr, nifty_uv_series)
            else:
                n_view = "Neutral"
                n_sig = "NEUTRAL"

            # BANKNIFTY
            b_exp, b_diff, b_pcr, b_uv = latest_expiry_summary(b_chain)
            if b_exp:
                bank_pcr_series.append(b_pcr)
                if len(bank_pcr_series) > 200: bank_pcr_series = bank_pcr_series[-200:]
                bank_uv_series.append(b_uv)
                if len(bank_uv_series) > 60: bank_uv_series = bank_uv_series[-60:]
                if allow_b:
                    ps_b.add_close(b_uv)
                b_mom_up = (len(bank_uv_series) < 3) or ((bank_uv_series[-1] - bank_uv_series[-3]) >= 0.0)
                if allow_b:
                    b_sig = buyer_signal(bank_pcr_series, b_pcr, b_mom_up)
                    bank_hist.append((stamp_b, b_diff, b_pcr, b_sig))
                    if len(bank_hist) > 5: bank_hist = bank_hist[-5:]  # keep last 5
                    last_stamp_b = stamp_b
                b_view = classify_index_view(b_pcr, bank_uv_series)
            else:
                b_view = "Neutral"
                b_sig = "NEUTRAL"

            # Layout
            header = f"\n[bold]INDEX VIEW[/bold]  {header_badge('Nifty', n_view)}    {header_badge('BankNifty', b_view)}\n"
            layout = Table.grid(expand=True)
            layout.add_row(header)
            layout.add_row(build_panel(f"INTRADAY DATA - NIFTY  [dim]{n_exp}[/]", nifty_hist))
            layout.add_row(build_panel(f"INTRADAY DATA - BANKNIFTY  [dim]{b_exp}[/]", bank_hist))

            # Quick sanity output (optional): comment out after testing
            # if allow_n: console.print(f"NIFTY tick: exp={n_exp} pcr={n_pcr} uv={n_uv}", style="yellow")
            # if allow_b: console.print(f"BANKNIFTY tick: exp={b_exp} pcr={b_pcr} uv={b_uv}", style="yellow")

            # NIFTY entries/exits
            if allow_n and in_time_window(now_dt) and n_exp:
                if open_trade_n and open_trade_n.active:
                    why = should_exit(open_trade_n, n_pcr, n_view, nifty_uv_series)
                    if why:
                        open_trade_n.active = False
                        open_trade_n.exit_time = now_dt.strftime("%H:%M")
                        open_trade_n.exit_reason = why
                        blotter_rows.append([
                            open_trade_n.exit_time, "NIFTY", f"EXIT-{open_trade_n.side}",
                            str(open_trade_n.strike), str(open_trade_n.sl_pts),
                            str(open_trade_n.target_points()), f"{n_pcr:.2f}", why
                        ])
                        last_side_taken_n = open_trade_n.side
                        open_trade_n = None

                if (open_trade_n is None) and (n_view != "Neutral") and (n_sig in ["BUY", "SELL"]):
                    zone = PCR_BUY_ZONE if n_sig == "BUY" else PCR_SELL_ZONE
                    tr, reason = try_entry(
                        symbol="NIFTY", pcr=n_pcr, nview=n_view, uv_series=nifty_uv_series,
                        price_state=ps_n, pcr_zone=zone, sl_pts=NIFTY_OPTION_SL, tp_r=NIFTY_OPTION_TP_R,
                        step=NIFTY_STRIKE_STEP, allow_first_only_flag=ALLOW_FIRST_SIGNAL_ONLY,
                        last_side_taken=last_side_taken_n
                    )
                    if tr:
                        open_trade_n = tr
                        blotter_rows.append([
                            now_dt.strftime("%H:%M"), "NIFTY", tr.side, str(tr.strike),
                            str(tr.sl_pts), str(tr.target_points()), f"{n_pcr:.2f}", reason
                        ])

            # BANKNIFTY entries/exits
            if allow_b and in_time_window(now_dt) and b_exp:
                if open_trade_b and open_trade_b.active:
                    why = should_exit(open_trade_b, b_pcr, b_view, bank_uv_series)
                    if why:
                        open_trade_b.active = False
                        open_trade_b.exit_time = now_dt.strftime("%H:%M")
                        open_trade_b.exit_reason = why
                        blotter_rows.append([
                            open_trade_b.exit_time, "BANKNIFTY", f"EXIT-{open_trade_b.side}",
                            str(open_trade_b.strike), str(open_trade_b.sl_pts),
                            str(open_trade_b.target_points()), f"{b_pcr:.2f}", why
                        ])
                        last_side_taken_b = open_trade_b.side
                        open_trade_b = None

                if (open_trade_b is None) and (b_view != "Neutral") and (b_sig in ["BUY", "SELL"]):
                    zone = PCR_BUY_ZONE if b_sig == "BUY" else PCR_SELL_ZONE
                    tr, reason = try_entry(
                        symbol="BANKNIFTY", pcr=b_pcr, nview=b_view, uv_series=bank_uv_series,
                        price_state=ps_b, pcr_zone=zone, sl_pts=BANK_OPTION_SL, tp_r=BANK_OPTION_TP_R,
                        step=BANK_STRIKE_STEP, allow_first_only_flag=ALLOW_FIRST_SIGNAL_ONLY,
                        last_side_taken=last_side_taken_b
                    )
                    if tr:
                        open_trade_b = tr
                        blotter_rows.append([
                            now_dt.strftime("%H:%M"), "BANKNIFTY", tr.side, str(tr.strike),
                            str(tr.sl_pts), str(tr.target_points()), f"{b_pcr:.2f}", reason
                        ])

            # Attach blotter
            layout.add_row(Panel(blotter_table(blotter_rows), border_style="cyan", box=box.ROUNDED))

            # Render
            # Using Live, but also print the layout to force initial draw even before Live refresh cycles
            console.print(layout)
            time_module.sleep(POLL_SECONDS)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="(%(asctime)s) %(levelname)s :: %(message)s")
    run()
