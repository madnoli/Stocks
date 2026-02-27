import json
import time
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from great_tables import GT
from rich.console import Console
from rich.table import Table
from pathlib import Path

# --- Configuration ---
API_TMPL = "http://localhost:3000/api/equity/options/{symbol}"
TIMEOUT = 20
STOCKS_FILE = "stocks.txt"
HTML_OUT = "option_chain_summary.html"
CACHE_FILE = "option_chain_cache.json"

# --- Enhanced Tunables for Better Detection ---
MIN_TOTAL_OI = 10000             # 10K contracts (relaxed)
MIN_TOTAL_VOL = 1000             # 1K volume (relaxed)
PCR_TOL = 0.1                    # ±10% PCR band (wider)
SLEEP_SECONDS = 300
EPS = 1e-6

# Enhanced filtering parameters
VOLUME_SPIKE_X = 1.5             # 1.5x volume threshold (realistic)
PRICE_MOMENTUM_MIN = 0.002       # 0.2% price movement (achievable)
LIQ_SPREAD_GOOD = 0.15           # 15% relative spread cutoff
LIQ_SPREAD_EXCELLENT = 0.08      # 8% relative spread cutoff
INSTITUTIONAL_BLOCK_RS = 25_00_000  # Rs 25 lakh as institutional (realistic)
OI_VOLUME_NEW_POS_RATIO = 0.3    # 30% Volume/OI ratio for new positions
IV_RANK_LOOKBACK = 30            # 30-day IV rank lookback
VPIN_BUCKETS = 50                # VPIN averaging buckets
GEX_CONTRACT_SIZE = 1            # Contract multiplier
ATM_WINDOW_PCT = 0.03            # ±3% of spot for ATM consideration
DEBUG_MODE = True                # Enable debug output

console = Console()

# -------------------- Utilities --------------------

def safe_div(a, b):
    if b is None or abs(b) < EPS:
        return float('inf') if (a is not None and a > 0) else 0.0
    return a / b

def pct_change(now, prev):
    if now is None or prev is None:
        return None
    denom = prev if abs(prev) > EPS else EPS
    return ((now - prev) / denom) * 100.0

def parse_expiry(s):
    try:
        return datetime.strptime(s, "%d-%b-%Y")
    except (ValueError, TypeError):
        return None

def choose_current_expiry(records):
    exps = records.get("expiryDates") or []
    exps_parsed = [(e, parse_expiry(e)) for e in exps]
    now = datetime.now()
    future = [e for e in exps_parsed if e[1] and e[1] >= now]
    if future:
        return min(future, key=lambda x: x[1])[0]
    past = [e for e in exps_parsed if e[1]]
    if past:
        return max(past, key=lambda x: x[1])[0]
    return None

# -------------------- Market Session Analysis --------------------

def get_market_session_context():
    """Adjust thresholds based on market session timing"""
    now = datetime.now()
    hour = now.hour

    if 9 <= hour <= 10:  # Opening hour
        return {
            'session': 'OPENING',
            'volume_multiplier': 0.8,
            'oi_multiplier': 0.9,
            'momentum_threshold': 0.003,
            'weight': 1.4
        }
    elif 10 <= hour <= 11:  # Mid-morning
        return {
            'session': 'MID_MORNING', 
            'volume_multiplier': 1.0,
            'oi_multiplier': 1.0,
            'momentum_threshold': 0.005,
            'weight': 1.2
        }
    elif 14 <= hour <= 15:  # Closing session
        return {
            'session': 'CLOSING',
            'volume_multiplier': 0.7,
            'oi_multiplier': 0.8, 
            'momentum_threshold': 0.004,
            'weight': 1.3
        }
    else:  # Normal session
        return {
            'session': 'NORMAL',
            'volume_multiplier': 1.1,
            'oi_multiplier': 1.0,
            'momentum_threshold': 0.006,
            'weight': 1.0
        }

def get_market_session_weight(current_time):
    context = get_market_session_context()
    return context['weight']

# -------------------- Advanced Metrics --------------------

def assess_option_liquidity(option_quote):
    """Enhanced liquidity assessment with multiple bid-ask sources"""
    # Handle multiple possible field names
    bid = (option_quote.get('bidprice') or 
           option_quote.get('bid') or 
           option_quote.get('bidPrice') or 0)
    ask = (option_quote.get('askPrice') or 
           option_quote.get('ask') or 
           option_quote.get('askprice') or 0)

    if bid is None: bid = 0
    if ask is None: ask = 0

    # Use last traded price as fallback
    ltp = (option_quote.get('lastPrice') or 
           option_quote.get('ltp') or 
           option_quote.get('last_traded_price') or 0)

    # Calculate mid price
    if bid > 0 and ask > 0:
        mid = (bid + ask) / 2
    elif ltp > 0:
        mid = ltp
    else:
        return {'quality': 'POOR', 'score': 0, 'spread_pct': None}

    if mid <= 0:
        return {'quality': 'POOR', 'score': 0, 'spread_pct': None}

    # Calculate relative spread
    if bid > 0 and ask > 0:
        rel_spread = (ask - bid) / mid
    else:
        rel_spread = 0.20  # Assume 20% spread if no bid-ask data

    # Score the liquidity
    if rel_spread <= LIQ_SPREAD_EXCELLENT:
        return {'quality': 'EXCELLENT', 'score': 5, 'spread_pct': rel_spread * 100}
    elif rel_spread <= LIQ_SPREAD_GOOD:
        return {'quality': 'GOOD', 'score': 4, 'spread_pct': rel_spread * 100}
    elif rel_spread <= 0.25:
        return {'quality': 'FAIR', 'score': 3, 'spread_pct': rel_spread * 100}
    else:
        return {'quality': 'POOR', 'score': 1, 'spread_pct': rel_spread * 100}

def detect_institutional_activity(option_quote):
    """Detect smart money flow and institutional activity"""
    # Robust field extraction
    volume = (option_quote.get('totalTradedVolume') or 
              option_quote.get('volume') or 
              option_quote.get('traded_volume') or 0)
    oi = (option_quote.get('openInterest') or 
          option_quote.get('oi') or 
          option_quote.get('open_interest') or 0)
    last_price = (option_quote.get('lastPrice') or 
                  option_quote.get('ltp') or 
                  option_quote.get('last_traded_price') or 0)
    bid = (option_quote.get('bidprice') or 
           option_quote.get('bid') or 0)
    ask = (option_quote.get('askPrice') or 
           option_quote.get('ask') or 0)

    # Calculate institutional indicators
    avg_contract_value = last_price * volume if (last_price and volume) else 0
    large_block = avg_contract_value >= INSTITUTIONAL_BLOCK_RS

    # New position detection
    new_position_signal = False
    if oi > 0:
        new_position_signal = (volume / oi) > OI_VOLUME_NEW_POS_RATIO

    # Urgency detection (trades at or above ask)
    urgency_signal = False
    if ask > 0 and last_price > 0:
        urgency_signal = last_price >= (ask * 0.98)  # Within 2% of ask

    # Smart money composite score
    smart_money_score = sum([large_block, new_position_signal, urgency_signal])

    return {
        'institutional_size': large_block,
        'new_positions': new_position_signal,
        'urgent_execution': urgency_signal,
        'smart_money_score': smart_money_score,
        'avg_contract_value': avg_contract_value
    }

def enhanced_volume_signal(spot_price, hist_avg_volume, recent_volume, price_change_pct, session_context):
    """Enhanced volume analysis with session context"""
    if not hist_avg_volume or hist_avg_volume <= 0:
        hist_avg_volume = max(recent_volume * 0.5, 1000)  # Fallback estimate

    # Apply session-specific multipliers
    adjusted_threshold = VOLUME_SPIKE_X * session_context['volume_multiplier']
    adjusted_momentum = session_context['momentum_threshold']

    vol_ratio = safe_div(recent_volume, hist_avg_volume)
    price_momentum_abs = abs(price_change_pct) if price_change_pct else 0

    # Volume significance
    volume_significant = vol_ratio >= adjusted_threshold
    # Price momentum confirmation
    momentum_confirmed = price_momentum_abs >= adjusted_momentum
    # Combined signal
    valid_signal = volume_significant and momentum_confirmed

    return {
        'volume_ratio': vol_ratio,
        'price_momentum': price_change_pct,
        'momentum_abs': price_momentum_abs,
        'valid_signal': valid_signal,
        'volume_significant': volume_significant,
        'momentum_confirmed': momentum_confirmed
    }

def compute_iv_rank(symbol, current_avg_iv, cache):
    """Calculate IV rank from historical data"""
    hist_key = f"IVHIST|{symbol}"
    arr = cache.get(hist_key, [])
    ts_now = datetime.now().isoformat()

    # Append current IV
    if current_avg_iv and isinstance(current_avg_iv, (int, float)):
        arr.append({'ts': ts_now, 'iv': float(current_avg_iv)})

    # Keep last IV_RANK_LOOKBACK entries
    arr = arr[-IV_RANK_LOOKBACK:]
    cache[hist_key] = arr

    # Calculate IV rank
    iv_values = [x['iv'] for x in arr if isinstance(x.get('iv'), (int, float)) and x['iv'] > 0]

    if len(iv_values) < 5:
        return None, cache  # Insufficient history

    current_iv = float(current_avg_iv) if current_avg_iv else 0
    iv_min = min(iv_values)
    iv_max = max(iv_values)

    if abs(iv_max - iv_min) < EPS:
        return 50.0, cache

    iv_rank = 100.0 * safe_div((current_iv - iv_min), (iv_max - iv_min))
    return max(0.0, min(100.0, iv_rank)), cache

def calculate_gamma_exposure(spot, rows, expiry_dt):
    """Estimate gamma exposure for squeeze detection"""
    if spot is None or not rows:
        return 0.0

    # Focus on near-ATM strikes
    lower = spot * (1 - ATM_WINDOW_PCT)
    upper = spot * (1 + ATM_WINDOW_PCT)

    total_gex = 0.0
    for row in rows:
        K = row.get('strikePrice')
        if K is None or not (lower <= K <= upper):
            continue

        ce = row.get('CE') or {}
        pe = row.get('PE') or {}
        ce_oi = ce.get('openInterest') or 0
        pe_oi = pe.get('openInterest') or 0

        # Get gamma values or estimate
        ce_gamma = ce.get('gamma')
        pe_gamma = pe.get('gamma')

        # Time to expiry approximation
        T = 0.1  # Default 36 days approximation
        if isinstance(expiry_dt, datetime):
            days_to_expiry = (expiry_dt - datetime.now()).days
            T = max(days_to_expiry, 1) / 365.0

        # Estimate gamma if not provided
        if not isinstance(ce_gamma, (int, float)):
            ce_iv = ce.get('impliedVolatility') or 0.20
            ce_gamma = (1.0 / (spot * ce_iv * math.sqrt(T))) if (spot and ce_iv and T) else 0.01

        if not isinstance(pe_gamma, (int, float)):
            pe_iv = pe.get('impliedVolatility') or 0.20
            pe_gamma = (1.0 / (spot * pe_iv * math.sqrt(T))) if (spot and pe_iv and T) else 0.01

        # Net gamma exposure calculation
        net_gamma = (ce_oi * ce_gamma) - (pe_oi * pe_gamma)
        distance_weight = max(0.1, 1.0 - abs(spot - K) / (spot * ATM_WINDOW_PCT))
        total_gex += net_gamma * distance_weight

    return total_gex

def prefilter_fno_stocks(symbol, base_metrics):
    """Pre-filter for F&O eligibility and basic liquidity"""
    volume = base_metrics.get('Volume', 0)
    oi = base_metrics.get('OI', 0) 
    price = base_metrics.get('Price', 0)

    # Get session context for dynamic filtering
    session_ctx = get_market_session_context()

    # Apply session-adjusted thresholds
    min_vol_threshold = MIN_TOTAL_VOL * session_ctx['volume_multiplier']
    min_oi_threshold = MIN_TOTAL_OI * session_ctx['oi_multiplier']

    # Basic filters
    volume_ok = volume >= min_vol_threshold
    oi_ok = oi >= min_oi_threshold  
    price_ok = price >= 10.0  # Avoid penny stocks

    # Activity score
    activity_score = (volume * price) if (volume and price) else 0
    activity_ok = activity_score >= 50_000  # Rs 50K daily activity (relaxed)

    return {
        'passes': all([volume_ok, oi_ok, price_ok, activity_ok]),
        'volume_ok': volume_ok,
        'oi_ok': oi_ok,
        'price_ok': price_ok,
        'activity_ok': activity_ok,
        'thresholds': {
            'min_vol': min_vol_threshold,
            'min_oi': min_oi_threshold,
            'min_activity': 50_000
        }
    }

# -------------------- Core Data Functions --------------------

def fetch_symbol_metrics(symbol, cache):
    """Enhanced symbol metrics with comprehensive analysis"""
    url = API_TMPL.format(symbol=symbol)

    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        obj = r.json()
    except Exception as e:
        raise ValueError(f"API request failed: {e}")

    recs = obj.get("records", {})
    curr_exp = choose_current_expiry(recs)
    if not curr_exp:
        raise ValueError("No valid expiry found")

    rows = [row for row in recs.get("data", []) if row.get("expiryDate") == curr_exp]
    if not rows:
        raise ValueError("No option chain data for current expiry")

    # Extract underlying price
    underlying = None
    for row in rows:
        for side in ['CE', 'PE']:
            val = row.get(side, {}).get("underlyingValue")
            if isinstance(val, (int, float)) and val > 0:
                underlying = val
                break
        if underlying:
            break

    if underlying is None:
        raise ValueError("Underlying price not found")

    # Get market session context
    session_ctx = get_market_session_context()

    # Initialize aggregation variables
    ce_oi_sum = pe_oi_sum = 0
    ce_vol_sum = pe_vol_sum = 0
    ce_oi_wsum = pe_oi_wsum = 0.0
    ce_oi_w = pe_oi_w = 0.0
    ce_iv_wsum = pe_iv_wsum = 0.0
    ce_iv_w = pe_iv_w = 0.0

    # Collect sample quotes for analysis
    sample_quotes = []

    # Process each strike
    for row in rows:
        ce = row.get("CE") or {}
        pe = row.get("PE") or {}

        # Basic metrics
        ce_oi = ce.get("openInterest") or 0
        pe_oi = pe.get("openInterest") or 0
        ce_vol = ce.get("totalTradedVolume") or 0
        pe_vol = pe.get("totalTradedVolume") or 0

        # Aggregate totals
        ce_oi_sum += ce_oi
        pe_oi_sum += pe_oi
        ce_vol_sum += ce_vol
        pe_vol_sum += pe_vol

        # OI change calculations
        ce_oi_chg = ce.get("pchangeinOpenInterest")
        pe_oi_chg = pe.get("pchangeinOpenInterest")

        if isinstance(ce_oi_chg, (int, float)) and ce_oi > 0:
            ce_oi_wsum += ce_oi_chg * ce_oi
            ce_oi_w += ce_oi
        if isinstance(pe_oi_chg, (int, float)) and pe_oi > 0:
            pe_oi_wsum += pe_oi_chg * pe_oi
            pe_oi_w += pe_oi

        # IV calculations
        ce_iv = ce.get("impliedVolatility") or 0
        pe_iv = pe.get("impliedVolatility") or 0
        if ce_iv > 0 and ce_oi > 0:
            ce_iv_wsum += ce_iv * ce_oi
            ce_iv_w += ce_oi
        if pe_iv > 0 and pe_oi > 0:
            pe_iv_wsum += pe_iv * pe_oi
            pe_iv_w += pe_oi

        # Collect near-ATM quotes for detailed analysis
        K = row.get("strikePrice")
        if K and abs(K - underlying) / underlying <= ATM_WINDOW_PCT:
            if ce_oi > 0 or ce_vol > 0:
                sample_quotes.append(ce)
            if pe_oi > 0 or pe_vol > 0:
                sample_quotes.append(pe)

    # Calculate derived metrics
    total_oi = ce_oi_sum + pe_oi_sum
    total_vol = ce_vol_sum + pe_vol_sum
    pcr = safe_div(pe_oi_sum, ce_oi_sum)

    # OI change percentages
    ce_oi_chg_pct = safe_div(ce_oi_wsum, ce_oi_w)
    pe_oi_chg_pct = safe_div(pe_oi_wsum, pe_oi_w)
    blended_oi_chg = safe_div((ce_oi_chg_pct * ce_oi_sum) + (pe_oi_chg_pct * pe_oi_sum), total_oi)

    # IV calculations
    avg_ce_iv = safe_div(ce_iv_wsum, ce_iv_w)
    avg_pe_iv = safe_div(pe_iv_wsum, pe_iv_w)
    avg_iv = safe_div((avg_ce_iv * ce_oi_sum) + (avg_pe_iv * pe_oi_sum), total_oi)

    # Volume to OI ratio
    vol_oi_ratio = safe_div(total_vol, total_oi)

    # ATM analysis
    if rows:
        atm_row = min(rows, key=lambda r: abs(r.get("strikePrice", float('inf')) - underlying))
        atm_ce = atm_row.get("CE", {})
        atm_pe = atm_row.get("PE", {})
        atm_pcr = safe_div(atm_pe.get("openInterest", 0), atm_ce.get("openInterest", 0))
        atm_ce_vol = atm_ce.get("totalTradedVolume", 0)
        atm_pe_vol = atm_pe.get("totalTradedVolume", 0)

        if atm_ce_vol > atm_pe_vol:
            atm_vol_dom = "CALLS"
        elif atm_pe_vol > atm_ce_vol:
            atm_vol_dom = "PUTS"  
        else:
            atm_vol_dom = "NEUTRAL"

        atm_signal = f"PCR:{atm_pcr:.2f}|VOL:{atm_vol_dom}"
    else:
        atm_signal = "N/A"

    # Advanced analytics on sample quotes
    liq_scores = []
    smart_scores = []

    for quote in sample_quotes:
        if not quote:
            continue
        liq = assess_option_liquidity(quote)
        smart = detect_institutional_activity(quote)
        liq_scores.append(liq.get('score', 0))
        smart_scores.append(smart.get('smart_money_score', 0))

    liq_score_avg = float(np.mean(liq_scores)) if liq_scores else 0.0
    smart_money_avg = float(np.mean(smart_scores)) if smart_scores else 0.0

    # Volume analysis with historical context
    vol_hist_key = f"VOLHIST|{symbol}"
    vol_hist = cache.get(vol_hist_key, [])
    vol_hist.append({'ts': datetime.now().isoformat(), 'vol': int(total_vol)})
    vol_hist = vol_hist[-50:]  # Keep last 50 data points
    cache[vol_hist_key] = vol_hist

    hist_vol_values = [x['vol'] for x in vol_hist[:-1] if isinstance(x.get('vol'), (int, float))]
    if hist_vol_values:
        hist_avg_volume = float(np.mean(hist_vol_values))
    else:
        hist_avg_volume = max(total_vol * 0.7, 1000)  # Estimate if no history

    # Price change estimation (simplified)
    price_change_pct = 0.0
    try:
        # Try to get from metadata if available
        meta_change = recs.get('underlyingValue', {}).get('pChange', 0)
        if isinstance(meta_change, (int, float)):
            price_change_pct = meta_change / 100.0
    except:
        price_change_pct = 0.0

    vol_sig = enhanced_volume_signal(underlying, hist_avg_volume, total_vol, price_change_pct, session_ctx)

    # IV rank calculation
    iv_rank, cache = compute_iv_rank(symbol, avg_iv, cache)

    # Gamma exposure
    expiry_dt = parse_expiry(curr_exp)
    gex = calculate_gamma_exposure(underlying, rows, expiry_dt)

    # Enhanced scoring system
    metrics_for_score = {
        'institutional_flow': min(1.0, smart_money_avg / 3.0),
        'gamma_exposure': min(1.0, abs(gex) / (abs(gex) + 100.0)) if gex else 0.0,
        'volume_quality': 1.0 if vol_sig['valid_signal'] else min(1.0, vol_sig['volume_ratio'] / VOLUME_SPIKE_X),
        'liquidity_score': min(1.0, liq_score_avg / 5.0),
        'pcr_deviation': min(1.0, abs(pcr - 1.0)),
        'iv_rank': (iv_rank / 100.0) if isinstance(iv_rank, (int, float)) else 0.5
    }

    weights = {
        'institutional_flow': 0.25,
        'gamma_exposure': 0.20,
        'volume_quality': 0.20,
        'liquidity_score': 0.15,
        'pcr_deviation': 0.10,
        'iv_rank': 0.10
    }

    # Apply session weighting
    session_weight = session_ctx['weight']
    total_score = sum(metrics_for_score[k] * weights[k] for k in weights) * session_weight

    # Enhanced classification
    if total_score >= 0.75:
        enhanced_remark = "STRONG_BUY"
    elif total_score >= 0.6:
        enhanced_remark = "BUY"
    elif total_score >= 0.45:
        enhanced_remark = "WATCH"
    elif total_score >= 0.3:
        enhanced_remark = "NEUTRAL"
    else:
        enhanced_remark = "AVOID"

    # Legacy classification for comparison
    def legacy_classify(pcr_val, ce_oi, pe_oi, ce_vol, pe_vol):
        is_low_liq = (ce_oi + pe_oi < MIN_TOTAL_OI) or (ce_vol + pe_vol < MIN_TOTAL_VOL)

        if abs(pcr_val - 1.0) <= PCR_TOL:
            return "Neutral"

        ce_oi_dom = ce_oi > pe_oi
        pe_oi_dom = pe_oi > ce_oi
        ce_vol_dom = ce_vol >= pe_vol
        pe_vol_dom = pe_vol >= ce_vol

        if pcr_val < 0.8 and ce_oi_dom and ce_vol_dom:
            return "Strong Bullish" if not is_low_liq else "Mild Bullish"
        elif pcr_val > 1.2 and pe_oi_dom and pe_vol_dom:
            return "Strong Bearish" if not is_low_liq else "Mild Bearish"
        elif pcr_val < 1.0 and (ce_oi_dom or ce_vol_dom):
            return "Mild Bullish"
        elif pcr_val > 1.0 and (pe_oi_dom or pe_vol_dom):
            return "Mild Bearish"
        else:
            return "Neutral"

    legacy_remark = legacy_classify(pcr, ce_oi_sum, pe_oi_sum, ce_vol_sum, pe_vol_sum)
    combined_remark = f"{enhanced_remark}"

    # Compile final metrics
    result = {
        "Stock": symbol,
        "Price": underlying,
        "Volume": total_vol,
        "OI": total_oi,
        "OI Chg %": blended_oi_chg,
        "PCR": pcr,
        "Avg IV %": (avg_iv * 100.0 if avg_iv else None),
        "V/OI Ratio": vol_oi_ratio,
        "ATM Signal": atm_signal,
        "Remark": combined_remark,
        "Expiry": curr_exp,
        "Liq Score": liq_score_avg,
        "Smart Flow": smart_money_avg,
        "Vol Ratio": vol_sig.get('volume_ratio'),
        "GEX": gex,
        "IV Rank": iv_rank,
        "Score": total_score,
        "Session": session_ctx['session']
    }

    return result, cache

def fetch_symbol_metrics_debug(symbol, cache):
    """Debug wrapper for symbol metrics"""
    try:
        metrics, cache = fetch_symbol_metrics(symbol, cache)

        if DEBUG_MODE:
            console.print(f"\n[blue]═══ DEBUG: {symbol} ═══[/blue]")
            console.print(f"  💰 Price: ₹{metrics.get('Price', 0):,.2f}")
            console.print(f"  📊 Volume: {metrics.get('Volume', 0):,}")
            console.print(f"  🎯 OI: {metrics.get('OI', 0):,}")  
            console.print(f"  ⚖️  PCR: {metrics.get('PCR', 0):.3f}")
            console.print(f"  📈 Score: {metrics.get('Score', 0):.3f}")
            console.print(f"  🏷️  Remark: {metrics.get('Remark', 'N/A')}")
            console.print(f"  🌊 Vol Ratio: {metrics.get('Vol Ratio', 0):.2f}x")
            console.print(f"  🎪 Session: {metrics.get('Session', 'N/A')}")

            # Pre-filter check
            filter_result = prefilter_fno_stocks(symbol, metrics)
            status = "✅ PASS" if filter_result['passes'] else "❌ FILTERED"
            console.print(f"  🔍 Filter: {status}")

            if not filter_result['passes']:
                console.print(f"    └── Vol OK: {filter_result['volume_ok']} (need {filter_result['thresholds']['min_vol']:,.0f})")
                console.print(f"    └── OI OK: {filter_result['oi_ok']} (need {filter_result['thresholds']['min_oi']:,.0f})")
                console.print(f"    └── Price OK: {filter_result['price_ok']}")
                console.print(f"    └── Activity OK: {filter_result['activity_ok']}")

        return metrics, cache

    except Exception as e:
        if DEBUG_MODE:
            console.print(f"[red]💥 ERROR {symbol}: {str(e)[:100]}...[/red]")
        return None, cache

def load_symbols(path=STOCKS_FILE):
    """Load symbols from file"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            symbols = [line.strip().upper() for line in f if line.strip() and not line.startswith('#')]
        console.print(f"📁 Loaded {len(symbols)} symbols from {path}")
        return symbols
    except FileNotFoundError:
        console.print(f"[red]📁 File {path} not found. Creating sample file...[/red]")
        # Create sample stocks file
        sample_stocks = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "KOTAKBANK", "SBIN", "BHARTIARTL", "ITC", "LT"]
        with open(path, "w") as f:
            for stock in sample_stocks:
                f.write(f"{stock}\n")
        console.print(f"✅ Created {path} with {len(sample_stocks)} sample stocks")
        return sample_stocks

def load_cache():
    """Load cache from file"""
    p = Path(CACHE_FILE)
    if not p.exists(): 
        return {}
    try:
        cache = json.loads(p.read_text(encoding="utf-8"))
        cache_size = len(cache)
        console.print(f"💾 Loaded cache with {cache_size} entries")
        return cache
    except json.JSONDecodeError:
        console.print(f"[yellow]💾 Cache file corrupted, starting fresh[/yellow]")
        return {}

def save_cache(cache):
    """Save cache to file"""
    try:
        Path(CACHE_FILE).write_text(json.dumps(cache, indent=2), encoding="utf-8")
        console.print(f"💾 Cache saved with {len(cache)} entries")
    except Exception as e:
        console.print(f"[yellow]💾 Cache save failed: {e}[/yellow]")

def compute_cycle(symbols, cache):
    """Process all symbols and return results"""
    results = []
    session_ctx = get_market_session_context()

    console.print(f"🚀 Processing {len(symbols)} symbols in {session_ctx['session']} session...")

    for i, symbol in enumerate(symbols, 1):
        console.print(f"[{i:2d}/{len(symbols)}] Processing {symbol}...", end="")

        try:
            metrics, cache = fetch_symbol_metrics_debug(symbol, cache)
            if metrics:
                # Apply pre-filtering
                filter_result = prefilter_fno_stocks(symbol, metrics)
                if filter_result['passes']:
                    results.append(metrics)
                    console.print(" ✅")
                else:
                    console.print(" 🚫 (filtered)")
            else:
                console.print(" ❌")
        except Exception as e:
            error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
            console.print(f" ❌ ({error_msg})")

            # Add error record for completeness
            results.append({
                "Stock": symbol, "Price": None, "Volume": None, "OI": None, 
                "OI Chg %": None, "PCR": None, "Avg IV %": None, "V/OI Ratio": None,
                "ATM Signal": None, "Remark": f"Error: {error_msg}", "Expiry": None,
                "Liq Score": None, "Smart Flow": None, "Vol Ratio": None, 
                "GEX": None, "IV Rank": None, "Score": None, "Session": None
            })

    console.print(f"\n📊 Found {len([r for r in results if r.get('Score')])} valid stocks out of {len(symbols)} processed")
    return results, cache

def add_volume_cycle_delta(results, prev_cache):
    """Add volume change calculations"""
    new_cache = prev_cache.copy()

    for rec in results:
        if not rec.get("Stock") or not rec.get("Expiry"):
            continue

        key = f"{rec['Stock']}|{rec['Expiry']}"
        prev_vol = prev_cache.get(key, {}).get("Volume")
        rec["Vol Chg %"] = pct_change(rec.get("Volume"), prev_vol)

        if rec.get("Volume"):
            new_cache[key] = {
                "Volume": rec["Volume"], 
                "ts": datetime.now().isoformat()
            }

    return results, new_cache

def render_tables(df):
    """Render console and HTML tables"""
    if df.empty:
        console.print("[yellow]📊 No data to display - all stocks filtered out[/yellow]")
        return

    # Filter valid stocks only
    valid_df = df[df["Score"].notna() & (df["Score"] > 0)].copy()

    if valid_df.empty:
        console.print("[yellow]📊 No valid stocks found after filtering[/yellow]")
        return

    # Categorize stocks
    def get_category(row):
        remark = str(row.get("Remark", ""))
        score = row.get("Score", 0)

        if "STRONG_BUY" in remark or score >= 0.75:
            return "Strong Buy"
        elif "BUY" in remark or score >= 0.6:
            return "Buy"
        elif "WATCH" in remark or score >= 0.45:
            return "Watch"
        else:
            return "Monitor"

    valid_df["Category"] = valid_df.apply(get_category, axis=1)

    # Sort by score descending
    valid_df = valid_df.sort_values("Score", ascending=False)

    # Take top performers
    display_df = valid_df.head(25)

    console.print(f"\n🎯 Found {len(valid_df)} target stocks:")

    # Group by category for summary
    category_counts = valid_df["Category"].value_counts()
    for category, count in category_counts.items():
        console.print(f"  • {category}: {count} stocks")

    # Display columns
    DISPLAY_COLS = [
        "Stock", "Category", "Score", "Price", "PCR", "Vol Ratio", "V/OI Ratio",
        "Smart Flow", "Liq Score", "IV Rank", "Volume", "OI", "Expiry", "Session"
    ]

    table_df = display_df[[c for c in DISPLAY_COLS if c in display_df.columns]].copy()

    # HTML Export
    try:
        gt = (
            GT(table_df)
            .tab_header(
                title="🎯 Enhanced Option Buyer Scanner",
                subtitle=f"Target Analysis • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Found {len(valid_df)} opportunities"
            )
            .fmt_number(columns=["Volume", "OI"], decimals=0, use_seps=True)
            .fmt_number(columns=["Price", "PCR", "Vol Ratio", "V/OI Ratio", "Score", "Smart Flow", "Liq Score"], decimals=2)
            .fmt_number(columns=["IV Rank"], decimals=1)
        )

        html_content = gt.as_raw_html()
        Path(HTML_OUT).write_text(html_content, encoding="utf-8")
        console.print(f"📄 HTML report saved: {HTML_OUT}")
    except Exception as e:
        console.print(f"[yellow]📄 HTML export failed: {e}[/yellow]")

    # Console Table
    table = Table(title="🎯 Enhanced Option Buyer Targets", show_lines=True, expand=True)

    for col in table_df.columns:
        justify = "right" if col not in ["Stock", "Category", "Expiry", "Session"] else "left"
        table.add_column(col, justify=justify, overflow="fold")

    for _, row in table_df.iterrows():
        # Style based on category
        category = str(row.get("Category", ""))
        if "Strong Buy" in category:
            style = "bold green"
        elif "Buy" in category:
            style = "green"
        elif "Watch" in category:
            style = "yellow"
        else:
            style = "white"

        row_data = []
        for col_name in table_df.columns:
            val = row[col_name]

            if pd.isna(val) or val is None:
                formatted = "-"
            elif col_name == "Price":
                formatted = f"₹{val:,.1f}"
            elif col_name in ["Volume", "OI"]:
                formatted = f"{val:,.0f}"
            elif col_name in ["Score", "PCR", "Vol Ratio", "V/OI Ratio", "Smart Flow", "Liq Score"]:
                formatted = f"{val:.2f}"
            elif col_name == "IV Rank":
                formatted = f"{val:.0f}%" if val else "-"
            else:
                formatted = str(val)

            row_data.append(formatted)

        # Apply styling to key columns
        if len(row_data) >= 2:
            row_data[0] = f"[{style}]{row_data[0]}[/{style}]"  # Stock
            row_data[1] = f"[{style}]{row_data[1]}[/{style}]"  # Category

        table.add_row(*row_data)

    console.print(table)

    # Summary statistics
    if len(valid_df) > 0:
        console.print(f"\n📈 Summary Statistics:")
        console.print(f"  • Average Score: {valid_df['Score'].mean():.2f}")
        console.print(f"  • Median PCR: {valid_df['PCR'].median():.2f}")
        console.print(f"  • Total Volume: {valid_df['Volume'].sum():,.0f}")
        console.print(f"  • Average Vol Ratio: {valid_df['Vol Ratio'].mean():.1f}x")

def run_once():
    """Execute one complete scan cycle"""
    symbols = load_symbols()
    if not symbols:
        console.print("[red]❌ No symbols to process[/red]")
        return

    prev_cache = load_cache()

    # Show session info
    session_ctx = get_market_session_context()
    console.print(f"\n🕐 Market Session: {session_ctx['session']} (Weight: {session_ctx['weight']:.1f}x)")
    console.print(f"📊 Volume Threshold: {VOLUME_SPIKE_X * session_ctx['volume_multiplier']:.1f}x")
    console.print(f"📈 Momentum Threshold: {session_ctx['momentum_threshold']*100:.1f}%\n")

    # Process symbols
    results, updated_cache = compute_cycle(symbols, prev_cache)

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Filter valid expiry data
    valid_df = df[df["Expiry"].notna()]

    # Add volume deltas
    results_with_deltas, final_cache = add_volume_cycle_delta(
        valid_df.to_dict("records"), updated_cache
    )

    # Final DataFrame
    final_df = pd.DataFrame(results_with_deltas)

    # Render results
    render_tables(final_df)

    # Save cache
    save_cache(final_cache)

    console.print(f"\n✅ Cycle completed at {datetime.now().strftime('%H:%M:%S')}")

def main():
    """Main execution function"""
    console.print("\n" + "="*60)
    console.print("🚀 ENHANCED OPTION BUYER SCANNER v2.0")
    console.print("    Advanced Flow Detection & Smart Money Analysis")
    console.print("="*60)

    # Configuration summary
    console.print(f"\n⚙️  Configuration:")
    console.print(f"   • Min Volume: {MIN_TOTAL_VOL:,} contracts")
    console.print(f"   • Min OI: {MIN_TOTAL_OI:,} contracts") 
    console.print(f"   • Volume Spike: {VOLUME_SPIKE_X}x threshold")
    console.print(f"   • Momentum: {PRICE_MOMENTUM_MIN*100:.1f}% minimum")
    console.print(f"   • Institutional Block: ₹{INSTITUTIONAL_BLOCK_RS:,}")
    console.print(f"   • Debug Mode: {'ON' if DEBUG_MODE else 'OFF'}")

    if DEBUG_MODE:
        console.print(f"\n🔧 Running in DEBUG mode - detailed analysis enabled")

    console.print(f"\n⏰ Starting scan at {datetime.now().strftime('%H:%M:%S')}")
    console.print(f"🔄 Refresh interval: {SLEEP_SECONDS} seconds\n")

if __name__ == "__main__":
    main()

    while True:
        try:
            ts = datetime.now().strftime('%H:%M:%S')
            console.rule(f"[bold blue]🔄 Scan Cycle: {ts}[/bold blue]")

            run_once()

            ts_end = datetime.now().strftime('%H:%M:%S')
            console.rule(f"[bold green]✅ Cycle Complete: {ts_end}[/bold green]")
            console.print(f"😴 Sleeping for {SLEEP_SECONDS} seconds...\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]🛑 Scan stopped by user[/yellow]")
            break
        except Exception as e:
            console.print_exception(show_locals=False)
            console.print(f"[bold red]💥 Error occurred: {e}[/bold red]")
            console.print(f"🔄 Retrying after {SLEEP_SECONDS} seconds...\n")

        time.sleep(SLEEP_SECONDS)
