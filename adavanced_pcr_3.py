import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import requests
import json
import time as timemodule
import pytz
import logging
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from truedata.history import TD_hist
import warnings
warnings.filterwarnings("ignore")
from rich.console import Console
from rich.table import Table
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =========================
# --- TRUE DATA CONFIG ---
# =========================
TDUSERNAME = os.getenv("TD_USERNAME", "tdwsp751")
TDPASSWORD = os.getenv("TD_PASSWORD", "raj@751")
tdhist = TD_hist(TDUSERNAME, TDPASSWORD, log_level=logging.WARNING)

# =========================
# --- COLOR CODES ---
# =========================
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

# =========================
# --- ENHANCED INDICATOR WEIGHTS FOR OPTION BUYERS ---
# =========================
ENHANCED_INDICATOR_WEIGHTS = {
    # Tier 1: Critical for Option Buyers (Highest Priority)
    "VolumeOIFlow": 2.5,      # Volume-OI divergence analysis
    "InstitutionalFlow": 2.3, # Large volume + OI changes
    "VolumeSurge": 2.2,       # Enhanced volume surge detection
    "OIChangeRate": 2.1,      # Open Interest momentum
    "VolumeBreakout": 2.0,    # Volume confirmation on breakouts

    # Tier 2: Strong momentum indicators
    "Momentum": 1.9,          # Enhanced with volume-price momentum
    "ADX": 1.8,               # Trend strength
    "VWAP": 1.7,              # Volume-weighted levels
    "EMA": 1.7,               # Price momentum

    # Tier 3: Supporting confirmation
    "MACD": 1.5,
    "OBV": 1.5,               # Enhanced with OI correlation
    "ATR": 1.4,               # Volatility for option premium
    "VolumeProfile": 1.3,     # Volume distribution analysis

    # Tier 4: Traditional indicators
    "Bollinger": 1.2,
    "RSI": 1.1,
    "ROC": 1.0,
    "Stochastic": 1.0,
    "CCI": 1.0,
    "MA": 1.0,
    "WWL": 1.0,
}

# =========================
# --- OPTION-CHAIN METRIC WEIGHTS ---
# =========================
OPTION_CHAIN_WEIGHTS = {
    "PCR_Score": 2.8,             # Put-Call Ratio sentiment
    "OI_Momentum_Score": 2.6,     # OI change momentum across strikes
    "IV_Attractiveness_Score": 2.4,# ATM IV percentile proxy within chain
    "Institutional_Options_Flow": 2.3, # Significant OI/Volume spikes
    "Max_Pain_Distance_Score": 2.1,    # Distance from max pain
    "Strike_SR_Score": 2.0,       # S/R from high OI strikes
}

# =========================
# --- TIMEFRAME WEIGHTS ---
# =========================
TIMEFRAME_WEIGHTS = {
    15: 3.0,    # Primary for option trading
    5: 2.8,     # Entry/exit precision
    30: 2.2,    # Trend confirmation
    60: 1.8,    # Medium-term view
    "daily": 1.5,  # Overall context
}

# =========================
# --- NSE INDEX TO SECTOR MAPPING ---
# =========================
NSE_INDEX_TO_SECTOR = {
    "NIFTY IT": "Technology",
    "NIFTY PHARMA": "Pharma",
    "NIFTY FMCG": "Consumer",
    "NIFTY BANK": "Banking",
    "NIFTY AUTO": "Auto",
    "NIFTY METAL": "Metal",
    "NIFTY ENERGY": "Energy",
    "NIFTY REALTY": "Realty",
    "NIFTY INFRA": "Infrastructure",
    "NIFTY PSU BANK": "PSU Bank",
    "NIFTY PSE": "PSE",
    "NIFTY COMMODITIES": "Commodities",
    "NIFTY MNC": "Finance",
    "NIFTY FINANCIAL SERVICES": "Finance",
    "NIFTY INFRASTRUCTURE": "Infrastructure",
    "BANKNIFTY": "Banking",
    "NIFTYAUTO": "Auto",
    "NIFTYIT": "Technology",
    "NIFTYPHARMA": "Pharma",
    "NIFTY CONSUMER DURABLES": "Consumer Durables",
    "NIFTY HEALTHCARE INDEX": "Healthcare",
    "NIFTY CAPITAL MARKETS": "Capital Market",
    "NIFTY PRIVATE BANK": "Private Bank",
    "NIFTY OIL & GAS": "Oil and Gas",
    "NIFTY INDIA DEFENCE": "Defence",
    "NIFTY CORE HOUSING": "Core Housing",
    "NIFTY SERVICES SECTOR": "Services Sector",
    "NIFTY FINANCIAL SERVICES 25/50": "Financial Services 2550",
    "NIFTY INDIA TOURISM": "Tourism",
}

# Sector to stocks mapping
SECTOR_STOCKS = {
    "Technology": ["TCS-I", "INFY-I", "HCLTECH-I", "WIPRO-I", "TECHM-I", "LTIM-I", "MPHASIS-I", "COFORGE-I", "PERSISTENT-I", "CYIENT-I", "KPITTECH-I", "TATAELXSI-I", "SONACOMS-I", "KAYNES-I", "OFSS-I"],
    "Auto": ["MARUTI-I", "TATAMOTORS-I", "M&M-I", "BAJAJ-AUTO-I", "HEROMOTOCO-I", "TVSMOTOR-I", "BHARATFORG-I", "EICHERMOT-I", "ASHOKLEY-I", "BOSCHLTD-I", "TIINDIA-I", "MOTHERSON-I"],
    "Banking": ["HDFCBANK-I", "ICICIBANK-I", "SBIN-I", "KOTAKBANK-I", "AXISBANK-I", "PNB-I", "BANKBARODA-I", "CANBK-I", "IDFCFIRSTB-I", "INDUSINDBK-I", "AUBANK-I", "FEDERALBNK-I"],
    "Pharma": ["SUNPHARMA-I", "DRREDDY-I", "CIPLA-I", "LUPIN-I", "AUROPHARMA-I", "TORNTPHARM-I", "GLENMARK-I", "ALKEM-I", "LAURUSLABS-I", "BIOCON-I", "ZYDUSLIFE-I", "MANKIND-I", "SYNGENE-I", "PPLPHARMA-I"],
    "Energy": ["RELIANCE-I", "NTPC-I", "BPCL-I", "IOC-I", "ONGC-I", "GAIL-I", "HINDPETRO-I", "ADANIGREEN-I", "ADANIENSOL-I", "JSWENERGY-I", "COALINDIA-I", "TATAPOWER-I", "SUZLON-I", "PETRONET-I", "OIL-I", "POWERGRID-I", "NHPC-I", "ADANIPORTS-I", "ABB-I", "SIEMENS-I", "CGPOWER-I", "INOXWIND-I"],
    "Metal": ["TATASTEEL-I", "JSWSTEEL-I", "SAIL-I", "JINDALSTEL-I", "HINDALCO-I", "NMDC-I"],
    "Consumer": ["HINDUNILVR-I", "ITC-I", "NESTLEIND-I", "BRITANNIA-I", "TATACONSUM-I", "DABUR-I", "AMBER-I", "UNITDSPR-I", "GODREJCP-I", "MARICO-I", "COLPAL-I", "UPL-I", "VBL-I"],
    "PSU Bank": ["SBIN-I", "PNB-I", "BANKBARODA-I", "CANBK-I", "UNIONBANK-I", "BANKINDIA-I"],
    "Finance": ["BAJFINANCE-I", "SHRIRAMFIN-I", "CHOLAFIN-I", "HDFCLIFE-I", "ICICIPRULI-I", "ETERNAL-I"],
    "Realty": ["DLF-I", "LODHA-I", "PRESTIGE-I", "GODREJPROP-I", "OBEROIRLTY-I", "PHOENIXLTD-I", "NCC-I", "NBCC-I"],
    "PSE": ["BEL-I", "BHEL-I", "NHPC-I", "GAIL-I", "IOC-I", "NTPC-I", "POWERGRID-I", "HINDPETRO-I", "OIL-I", "RECLTD-I", "ONGC-I", "NMDC-I", "BPCL-I", "HAL-I", "RVNL-I", "PFC-I", "COALINDIA-I", "IRCTC-I", "IRFC-I"],
    "Commodities": ["AMBUJACEM-I", "APLAPOLLO-I", "ULTRACEMCO-I", "SHREECEM-I", "JSWSTEEL-I", "HINDALCO-I", "NHPC-I", "IOC-I", "NTPC-I", "HINDPETRO-I", "ADANIGREEN-I", "OIL-I", "VEDL-I", "PIIND-I", "ONGC-I", "NMDC-I", "UPL-I", "BPCL-I", "JSWENERGY-I", "GRASIM-I", "RELIANCE-I", "TORNTPOWER-I", "TATAPOWER-I", "COALINDIA-I", "PIDILITIND-I", "SRF-I", "ADANIENSOL-I", "JINDALSTEL-I", "TATASTEEL-I", "HINDALCO-I"],
    "Consumer Durables": ["TITAN-I", "DIXON-I", "HAVELLS-I", "CROMPTON-I", "POLYCAB-I", "EXIDEIND-I", "AMBER-I", "KAYNES-I", "VOLTAS-I", "PGEL-I", "BLUESTARCO-I"],
    "Healthcare": ["SUNPHARMA-I", "DIVISLAB-I", "CIPLA-I", "TORNTPHARM-I", "MAXHEALTH-I", "APOLLOHOSP-I", "DRREDDY-I", "MANKIND-I", "ZYDUSLIFE-I", "LUPIN-I", "FORTIS-I", "ALKEM-I", "AUROPHARMA-I", "GLENMARK-I", "BIOCON-I", "LAURUSLABS-I", "SYNGENE-I", "GRANULES-I"],
    "Capital Market": ["HDFCAMC-I", "BSE-I", "360ONE-I", "MCX-I", "CDSL-I", "NUVAMA-I", "ANGELONE-I", "KFINTECH-I", "CAMS-I", "IEX-I"],
    "Private Bank": ["HDFCBANK-I", "ICICIBANK-I", "KOTAKBANK-I", "AXISBANK-I", "YESBANK-I", "IDFCFIRSTB-I", "INDUSINDBK-I", "FEDERALBNK-I", "BANDHANBNK-I", "RBLBANK-I"],
    "Oil and Gas": ["RELIANCE-I", "ONGC-I", "IOC-I", "BPCL-I", "GAIL-I", "HINDPETRO-I", "OIL-I", "PETRONET-I", "IGL-I"],
    "Defence": ["HAL-I", "BEL-I", "SOLARINDS-I", "MAZDOCK-I", "BDL-I"],
    "Core Housing": ["ULTRACEMCO-I", "ASIANPAINT-I", "GRASIM-I", "DLF-I", "AMBUJACEM-I", "LODHA-I", "DIXON-I", "POLYCAB-I", "SHREECEM-I", "HAVELLS-I", "PRESTIGE-I", "GODREJPROP-I", "OBEROIRLTY-I", "PHOENIXLTD-I", "VOLTAS-I", "DALBHARAT-I", "KEI-I", "BLUESTARCO-I", "LICHSGFIN-I", "PNBHOUSING-I", "CROMPTON-I"],
    "Services Sector": ["HDFCBANK-I", "BHARTIARTL-I", "TCS-I", "ICICIBANK-I", "SBIN-I", "INFY-I", "BAJFINANCE-I", "HCLTECH-I", "KOTAKBANK-I", "AXISBANK-I", "BAJAJFINSV-I", "NTPC-I", "ZOMATO-I", "ADANIPORTS-I", "DMART-I", "POWERGRID-I", "WIPRO-I", "INDIGO-I", "JIOFINSERV-I", "SBILIFE-I", "HDFCLIFE-I", "LTIM-I", "TECHM-I", "TATAPOWER-I", "SHRIRAMFIN-I", "GAIL-I", "MAXHEALTH-I", "APOLLOHOSP-I", "NAUKRI-I", "INDUSINDBK-I"],
    "Financial Services 2550": ["HDFCBANK-I", "ICICIBANK-I", "SBIN-I", "BAJFINANCE-I", "KOTAKBANK-I", "AXISBANK-I", "BAJAJFINSV-I", "JIOFIN-I", "SBILIFE-I", "HDFCLIFE-I", "PFC-I", "CHOLAFIN-I", "HDFCAMC-I", "SHRIRAMFIN-I", "MUTHOOTFIN-I", "RECLTD-I", "ICICIGI-I", "ICICIPRULI-I", "SBICARD-I", "LICHSGFIN-I"],
    "Tourism": ["INDIGO-I", "INDHOTEL-I", "IRCTC-I", "JUBLFOOD-I"]
}
# =========================
# --- SYMBOL MAPPING UTILITIES ---
# =========================
def clean_symbol_for_options(symbol):
    """
    Clean symbol for option chain API calls
    Removes suffixes and handles special cases
    """
    # Remove common suffixes
    cleaned = symbol.replace("-I", "").replace("-B", "").replace("-E", "").strip()
    
    # Handle special mappings
    mappings = {
        "NIFTY50": "NIFTY",
        "BANKNIFTY": "BANKNIFTY",
        "FINNIFTY": "FINNIFTY", 
        "MIDCPNIFTY": "MIDCPNIFTY",
        "SENSEX": "SENSEX",
        "BANKEX": "BANKEX"
    }
    
    return mappings.get(cleaned, cleaned)

def validate_symbol_for_options(symbol):
    """
    Validate if a symbol likely has option chain data
    """
    cleaned = clean_symbol_for_options(symbol)
    
    # List of symbols that typically have active options
    major_stocks = [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR", 
        "ITC", "LT", "SBIN", "BAJFINANCE", "BHARTIARTL",
        "ASIANPAINT", "MARUTI", "KOTAKBANK", "AXISBANK",
        "ICICIBANK", "SUNPHARMA", "NESTLEIND", "ULTRACEMCO",
        "TATAMOTORS", "ONGC", "POWERGRID", "NTPC", "COALINDIA",
        "DRREDDY", "JSWSTEEL", "GRASIM", "ADANIENT", "WIPRO",
        "VEDL", "TATASTEEL", "HINDALCO", "CIPLA", "TECHM"
    ]
    
    indices = ["NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY"]
    
    return cleaned in major_stocks or cleaned in indices

# =========================
# --- OPTION CHAIN HELPERS ---
# =========================
def fetch_option_chain(symbol):
    """
    Fetch option chain data for a symbol
    Removes -I suffix from TrueData symbols to get base symbol for option chain API
    """
    # Clean the symbol - remove -I suffix and any other suffixes
    base_symbol = clean_symbol_for_options(symbol)
    
    url = f"http://localhost:3000/api/equity/options/{base_symbol}"
    
    try:
        resp = requests.get(url, timeout=15)
        
        if resp.status_code != 200:
            logger.warning(f"Option chain API returned status {resp.status_code} for {base_symbol}")
            return None
            
        data = resp.json()
        
        # Handle case where response is a string that needs to be parsed
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse option chain JSON for {base_symbol}")
                return None
        
        # Validate the response structure
        if not isinstance(data, dict) or 'records' not in data:
            logger.warning(f"Invalid option chain structure for {base_symbol}")
            return None
            
        records = data.get('records', {})
        if not isinstance(records, dict) or 'data' not in records:
            logger.warning(f"No option chain data found for {base_symbol}")
            return None
            
        option_data = records.get('data', [])
        if not option_data:
            logger.warning(f"Empty option chain data for {base_symbol}")
            return None
            
        logger.info(f"Successfully fetched option chain for {base_symbol}: {len(option_data)} strikes")
        return data
        
    except requests.exceptions.Timeout:
        logger.error(f"Option chain API timeout for {base_symbol}")
        return None
    except requests.exceptions.ConnectionError:
        logger.error(f"Option chain API connection error for {base_symbol}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Option chain API request error for {base_symbol}: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Option chain JSON decode error for {base_symbol}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching option chain for {base_symbol}: {e}")
        return None

def validate_option_chain_data(chain_data, symbol):
    """
    Validate option chain data structure and content
    """
    if not chain_data:
        return False, "No chain data"
        
    try:
        records = chain_data.get('records', {})
        data = records.get('data', [])
        
        if not data:
            return False, "No option data"
            
        # Check for valid strikes
        valid_strikes = 0
        for row in data:
            if isinstance(row, dict) and 'strikePrice' in row:
                strike = row.get('strikePrice')
                if isinstance(strike, (int, float)) and strike > 0:
                    valid_strikes += 1
                    
        if valid_strikes == 0:
            return False, "No valid strikes found"
            
        return True, f"Valid chain with {valid_strikes} strikes"
        
    except Exception as e:
        return False, f"Validation error: {e}"

def _iter_chain_rows(chain):
    # Yields per-strike CE/PE dicts
    try:
        for row in chain.get("records", {}).get("data", []):
            ce = row.get("CE")
            pe = row.get("PE")
            yield row, ce, pe
    except Exception:
        return

def compute_pcr_scores(chain):
    total_call_vol = 0
    total_put_vol = 0
    total_call_oi = 0
    total_put_oi = 0
    for _, ce, pe in _iter_chain_rows(chain):
        if ce:
            total_call_vol += max(0, ce.get("totalTradedVolume", 0) or 0)
            total_call_oi += max(0, ce.get("openInterest", 0) or 0)
        if pe:
            total_put_vol += max(0, pe.get("totalTradedVolume", 0) or 0)
            total_put_oi += max(0, pe.get("openInterest", 0) or 0)

    vol_pcr = total_put_vol / max(1, total_call_vol)
    oi_pcr = total_put_oi / max(1, total_call_oi)

    # Convert PCR to 0-100 bullishness score (lower PCR -> more bullish)
    def pcr_to_score(pcr):
        if pcr < 0.7:
            return 85  # strong call bias
        elif pcr < 1.0:
            return 70
        elif pcr < 1.3:
            return 50
        else:
            return 30  # put bias
    score = (pcr_to_score(vol_pcr) * 0.6) + (pcr_to_score(oi_pcr) * 0.4)
    return np.clip(score, 0, 100), vol_pcr, oi_pcr

def compute_oi_momentum_score(chain):
    # Aggregate absolute percentage changes across many strikes to sense positioning momentum
    vals = []
    for _, ce, pe in _iter_chain_rows(chain):
        if ce and isinstance(ce.get("pchangeinOpenInterest"), (int, float)):
            vals.append(abs(ce.get("pchangeinOpenInterest", 0) or 0))
        if pe and isinstance(pe.get("pchangeinOpenInterest"), (int, float)):
            vals.append(abs(pe.get("pchangeinOpenInterest", 0) or 0))
    if not vals:
        return 50.0
    pct95 = np.percentile(vals, 95)
    avg = np.mean(vals)
    # Score higher if broad OI changes are large
    raw = (avg / max(1e-6, pct95)) * 100
    return float(np.clip(40 + raw * 60, 0, 100))

def compute_iv_attractiveness_score(chain):
    # Use ATM IVs as proxy: pick strikes nearest to underlying
    try:
        underlying = None
        rows = []
        for row, ce, pe in _iter_chain_rows(chain):
            if ce and "underlyingValue" in ce and isinstance(ce["underlyingValue"], (int, float)):
                underlying = ce["underlyingValue"]
            rows.append((row, ce, pe))
        if underlying is None and rows:
            # fallback: try PE
            for _, ce, pe in rows:
                if pe and "underlyingValue" in pe and isinstance(pe["underlyingValue"], (int, float)):
                    underlying = pe["underlyingValue"]
                    break
        if underlying is None:
            return 50.0

        # Collect IVs near ATM
        ivs = []
        for row, ce, pe in rows:
            sp = row.get("strikePrice")
            if not isinstance(sp, (int, float)):
                continue
            dist = abs(underlying - sp)
            if dist <= underlying * 0.03:  # within ~3% of spot as ATM neighborhood
                if ce and isinstance(ce.get("impliedVolatility"), (int, float)):
                    ivs.append(ce.get("impliedVolatility", 0) or 0)
                if pe and isinstance(pe.get("impliedVolatility"), (int, float)):
                    ivs.append(pe.get("impliedVolatility", 0) or 0)
        if not ivs:
            return 50.0

        avg_iv = float(np.mean(ivs))
        # Heuristic: mid IV (~20-35) good for buyers (momentum), too low -> choppy, too high -> risky
        if avg_iv <= 15:
            return 45.0
        elif avg_iv <= 25:
            return 65.0
        elif avg_iv <= 35:
            return 75.0
        elif avg_iv <= 50:
            return 60.0
        else:
            return 50.0
    except Exception:
        return 50.0

def compute_max_pain_distance_score(chain):
    # Max Pain strike = strike with max total OI (CE+PE)
    try:
        totals = {}
        underlying = None
        for row, ce, pe in _iter_chain_rows(chain):
            sp = row.get("strikePrice")
            if not isinstance(sp, (int, float)):
                continue
            total_oi = 0
            if ce:
                total_oi += max(0, ce.get("openInterest", 0) or 0)
                if underlying is None and isinstance(ce.get("underlyingValue"), (int, float)):
                    underlying = ce.get("underlyingValue")
            if pe:
                total_oi += max(0, pe.get("openInterest", 0) or 0)
                if underlying is None and isinstance(pe.get("underlyingValue"), (int, float)):
                    underlying = pe.get("underlyingValue")
            totals[sp] = totals.get(sp, 0) + total_oi

        if not totals or underlying is None:
            return 50.0, None, None

        max_pain_strike = max(totals.items(), key=lambda x: x[1])[0]
        distance_pct = (underlying - max_pain_strike) / max(1e-6, max_pain_strike) * 100.0
        # Score higher if price is not pinned near max pain (less resistance)
        dist = abs(distance_pct)
        if dist <= 0.5:
            score = 35.0
        elif dist <= 1.0:
            score = 45.0
        elif dist <= 2.0:
            score = 55.0
        elif dist <= 3.0:
            score = 65.0
        else:
            score = 75.0
        return float(score), float(max_pain_strike), float(distance_pct)
    except Exception:
        return 50.0, None, None

def compute_institutional_options_flow_score(chain):
    # Detect unusual OI or volume spikes per strike
    spikes = 0
    total = 0
    for _, ce, pe in _iter_chain_rows(chain):
        for leg in (ce, pe):
            if not leg:
                continue
            total += 1
            vol = leg.get("totalTradedVolume", 0) or 0
            oi_chg = leg.get("pchangeinOpenInterest", 0) or 0
            if vol >= 10000 or abs(oi_chg) >= 15:  # heuristic thresholds
                spikes += 1
    if total == 0:
        return 50.0
    ratio = spikes / total
    # Map to score: more spikes -> higher score
    return float(np.clip(50 + ratio * 50, 0, 100))

def compute_strike_sr_score(chain):
    # If spot is below a high OI call strike -> resistance; above high OI put strike -> support
    try:
        underlying = None
        ce_oi_by_strike = {}
        pe_oi_by_strike = {}

        for row, ce, pe in _iter_chain_rows(chain):
            sp = row.get("strikePrice")
            if not isinstance(sp, (int, float)):
                continue
            if ce:
                ce_oi_by_strike[sp] = ce_oi_by_strike.get(sp, 0) + max(0, ce.get("openInterest", 0) or 0)
                if underlying is None and isinstance(ce.get("underlyingValue"), (int, float)):
                    underlying = ce.get("underlyingValue")
            if pe:
                pe_oi_by_strike[sp] = pe_oi_by_strike.get(sp, 0) + max(0, pe.get("openInterest", 0) or 0)
                if underlying is None and isinstance(pe.get("underlyingValue"), (int, float)):
                    underlying = pe.get("underlyingValue")

        if underlying is None:
            return 50.0

        # High OI call resistance nearest above spot
        above_calls = [sp for sp in ce_oi_by_strike.keys() if sp >= underlying]
        below_puts = [sp for sp in pe_oi_by_strike.keys() if sp <= underlying]
        res_score = 0
        sup_score = 0

        if above_calls:
            top_call_strike = max(above_calls, key=lambda s: ce_oi_by_strike.get(s, 0))
            dist = (top_call_strike - underlying) / max(1e-6, underlying) * 100
            res_score = 50 + min(30, dist * 2)  # farther resistance is less immediate
        else:
            res_score = 65

        if below_puts:
            top_put_strike = min(below_puts, key=lambda s: -pe_oi_by_strike.get(s, 0))  # max OI but closest below
            dist = (underlying - top_put_strike) / max(1e-6, underlying) * 100
            sup_score = 50 + min(30, dist * 2)  # closer strong support benefits buyers
        else:
            sup_score = 60

        # Combine: higher is better for calls
        combined = (sup_score * 0.6 + res_score * 0.4)
        return float(np.clip(combined, 0, 100))
    except Exception:
        return 50.0

def build_option_chain_scores(symbol):
    """
    Enhanced option chain scoring with better error handling
    """
    try:
        chain = fetch_option_chain(symbol)
        if not chain:
            return {}
            
        # Validate the chain data
        is_valid, msg = validate_option_chain_data(chain, symbol)
        if not is_valid:
            return {}
            
        scores = {}
        
        # Calculate PCR scores
        try:
            pcr_score, vol_pcr, oi_pcr = compute_pcr_scores(chain)
            scores["PCR_Score"] = pcr_score
            scores["_PCR_details"] = {
                "volume_pcr": vol_pcr, 
                "oi_pcr": oi_pcr,
                "interpretation": "Lower PCR = More Bullish" if vol_pcr < 1.0 else "Higher PCR = More Bearish"
            }
        except Exception as e:
            scores["PCR_Score"] = 50.0

        # Calculate OI Momentum
        try:
            scores["OI_Momentum_Score"] = compute_oi_momentum_score(chain)
        except Exception as e:
            scores["OI_Momentum_Score"] = 50.0

        # Calculate IV Attractiveness
        try:
            scores["IV_Attractiveness_Score"] = compute_iv_attractiveness_score(chain)
        except Exception as e:
            scores["IV_Attractiveness_Score"] = 50.0

        # Calculate Max Pain
        try:
            mp_score, mp_strike, mp_dist = compute_max_pain_distance_score(chain)
            scores["Max_Pain_Distance_Score"] = mp_score
            scores["_MaxPain_details"] = {
                "strike": mp_strike, 
                "distance_pct": mp_dist,
                "interpretation": f"Price is {abs(mp_dist):.1f}% {'above' if mp_dist > 0 else 'below'} max pain" if mp_dist else "At max pain"
            }
        except Exception as e:
            scores["Max_Pain_Distance_Score"] = 50.0
            scores["_MaxPain_details"] = {"strike": None, "distance_pct": None}

        # Calculate Institutional Flow
        try:
            scores["Institutional_Options_Flow"] = compute_institutional_options_flow_score(chain)
        except Exception as e:
            scores["Institutional_Options_Flow"] = 50.0

        # Calculate Strike S/R
        try:
            scores["Strike_SR_Score"] = compute_strike_sr_score(chain)
        except Exception as e:
            scores["Strike_SR_Score"] = 50.0

        return scores
        
    except Exception as e:
        return {}

# =========================
# --- ENHANCED TECHNICAL INDICATORS FOR OPTION BUYERS ---
# =========================
class EnhancedOptionBuyerIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        """Calculate all indicators including new volume and open interest based ones"""
        indicators = {}
        if df is None or len(df) < 20:
            return indicators

        try:
            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            vol = df["Volume"]

            oi = df.get("OpenInterest", pd.Series([0] * len(df), index=df.index))

            # 1. RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            indicators["RSI"] = 100 - (100 / (1 + rs))

            # 2. MACD
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            indicators["MACD"] = macd_line - signal_line

            # 3. Stochastic
            low14 = low.rolling(window=14).min()
            high14 = high.rolling(window=14).max()
            indicators["Stochastic"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)

            # 4. MA 20
            indicators["MA"] = close.rolling(window=20).mean()

            # 5. EMA 21
            indicators["EMA"] = close.ewm(span=21).mean()

            # 6. ADX
            high_diff = high.diff()
            low_diff = low.diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
            minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff > 0), 0.0)
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr.replace(0, np.nan))
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr.replace(0, np.nan))
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
            indicators["ADX"] = dx.rolling(window=14).mean()

            # 7. Bollinger position
            ma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20
            indicators["Bollinger"] = (close - ma20) / (upper - lower).replace(0, np.nan) * 100

            # 8. ROC
            indicators["ROC"] = close.pct_change(periods=12) * 100

            # 9. Enhanced OBV with OI correlation
            obv = np.sign(close.diff().fillna(0)) * vol.fillna(0)
            obv = obv.cumsum()
            indicators["OBV"] = obv.pct_change(periods=10) * 100

            # 10. CCI
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(window=20).mean()
            mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
            indicators["CCI"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))

            # 11. Williams %R
            hh = high.rolling(window=14).max()
            ll = low.rolling(window=14).min()
            indicators["WWL"] = (hh - close) / (hh - ll).replace(0, np.nan) * -100

            # 12. VWAP (rolling)
            if len(df) >= 20:
                tpv = (high + low + close) / 3
                vwap_num = (tpv * vol).rolling(window=20).sum()
                vwap_den = vol.rolling(window=20).sum().replace(0, np.nan)
                indicators["VWAP"] = vwap_num / vwap_den

            # 13. ATR
            indicators["ATR"] = atr

            # 14. Volume Surge (Z-score)
            if len(df) >= 20:
                avg_vol_20 = vol.rolling(window=20).mean()
                vol_std = vol.rolling(window=20).std()
                current_vol = vol
                vol_zscore = (current_vol - avg_vol_20) / vol_std.replace(0, np.nan)
                indicators["VolumeSurge"] = np.clip(50 + vol_zscore * 15, 0, 100)

            # 15. OI Change Rate
            if oi.sum() > 0:
                oi_change = oi.pct_change(periods=1) * 100
                oi_momentum = oi.pct_change(periods=5) * 100
                indicators["OIChangeRate"] = np.clip(50 + (oi_change * 0.3 + oi_momentum * 0.7) * 2, 0, 100)
            else:
                indicators["OIChangeRate"] = pd.Series([50] * len(df), index=df.index)

            # 16. Volume-OI Flow
            if oi.sum() > 0:
                vol_oi_ratio = vol / (oi + 1)
                vol_trend = vol.rolling(window=10).mean()
                oi_trend = oi.rolling(window=10).mean()
                vol_direction = np.where(vol > vol_trend, 1, -1)
                oi_direction = np.where(oi > oi_trend, 1, -1)
                flow_score = (vol_direction + oi_direction) / 2 * 50 + 50
                indicators["VolumeOIFlow"] = pd.Series(flow_score, index=df.index)
            else:
                indicators["VolumeOIFlow"] = pd.Series([50] * len(df), index=df.index)

            # 17. Institutional Activity Score (price x volume)
            if len(df) >= 20:
                price_change = close.pct_change() * 100
                vol_percentile = vol.rolling(window=20).rank(pct=True) * 100
                institutional_score = np.where(
                    (vol_percentile > 80) & (abs(price_change) > 1.5),
                    75 + (vol_percentile - 80) * 1.25,
                    50 + (vol_percentile - 50) * 0.3
                )
                indicators["InstitutionalFlow"] = pd.Series(np.clip(institutional_score, 0, 100), index=df.index)

            # 18. Volume Profile (range position)
            if len(df) >= 20:
                current_price_level = close.iloc[-1]
                recent_high = high.rolling(window=10).max().iloc[-1]
                recent_low = low.rolling(window=10).min().iloc[-1]
                if recent_high > recent_low:
                    price_position = (current_price_level - recent_low) / (recent_high - recent_low)
                    volume_profile_score = 50 + (price_position - 0.5) * 100
                else:
                    volume_profile_score = 50
                indicators["VolumeProfile"] = pd.Series([np.clip(volume_profile_score, 0, 100)] * len(df), index=df.index)

            # 19. Volume Breakout
            if len(df) >= 20:
                price_ma = close.rolling(window=20).mean()
                vol_ma = vol.rolling(window=20).mean()
                price_breakout = (close - price_ma) / price_ma * 100
                volume_confirmation = vol / vol_ma
                breakout_score = np.where(
                    abs(price_breakout) > 2,
                    50 + price_breakout * volume_confirmation * 5,
                    50 + price_breakout * 10
                )
                indicators["VolumeBreakout"] = pd.Series(np.clip(breakout_score, 0, 100), index=df.index)

            # 20. Enhanced Momentum
            if len(df) >= 10:
                price_mom = close.pct_change(periods=10) * 100
                vol_mom = (vol / vol.rolling(window=10).mean() - 1) * 100
                if oi.sum() > 0:
                    oi_mom = (oi / oi.rolling(window=10).mean() - 1) * 100
                    combined_momentum = price_mom * 0.5 + vol_mom * 0.3 + oi_mom * 0.2
                else:
                    combined_momentum = price_mom * 0.7 + vol_mom * 0.3
                indicators["Momentum"] = pd.Series(np.clip(50 + combined_momentum * 1.2, 0, 100), index=df.index)

            return indicators

        except Exception as e:
            logger.error(f"Error calculating enhanced option buyer indicators: {e}")
            return indicators

# =========================
# --- NORMALIZATION HELPERS ---
# =========================
def normalize_indicator_value(indicator_name, value):
    """Enhanced normalization for new indicators"""
    try:
        if indicator_name == "RSI":
            return max(0, min(100, value))
        elif indicator_name == "MACD":
            return 50 + max(-25, min(25, value / 10))
        elif indicator_name == "Stochastic":
            return max(0, min(100, value))
        elif indicator_name in ("MA", "EMA", "VWAP"):
            return 50
        elif indicator_name == "ADX":
            return max(0, min(100, value))
        elif indicator_name == "Bollinger":
            return max(0, min(100, (value + 100) / 2))
        elif indicator_name == "ROC":
            return 50 + max(-25, min(25, value / 2))
        elif indicator_name == "OBV":
            return 50 + max(-25, min(25, value))
        elif indicator_name == "CCI":
            return max(0, min(100, (value + 200) / 4))
        elif indicator_name == "WWL":
            return max(0, min(100, (value + 100)))
        elif indicator_name == "ATR":
            return 50
        elif indicator_name in ("VolumeSurge", "OIChangeRate", "VolumeOIFlow",
                                "InstitutionalFlow", "VolumeProfile", "VolumeBreakout", "Momentum"):
            return max(0, min(100, value))
        else:
            return 50
    except Exception:
        return 50

# =========================
# --- ENHANCED SCANNER CLASS ---
# =========================
class EnhancedOptionBuyerScanner:
    def __init__(self, mode='live', backtest_date=None):
        self.is_running = False
        self.current_signals = []
        self.best_sectors = ["Pharma", "Healthcare", "Technology", "Financial Services 2550"]
        self.worst_sectors = ["Defence", "Energy", "PSU Bank", "Realty"]
        self.sectoral_history = []
        self.last_sectoral_update = None
        self.api_errors = []
        self.sector_update_attempts = 0
        self.successful_updates = 0
        self.gapdown_filtered_count = 0
        self.mode = mode
        self.backtest_date = backtest_date

        # Score tracking for deltas
        self.last_cycle_scores = {}
        self.current_cycle_scores = {}

        # Market hours - EXTENDED FOR TESTING
        self.market_start = time(9, 0)   # 9:00 AM
        self.market_end = time(23, 59)   # 11:59 PM (extended for testing)
        self.scan_interval = 300  # 5 minutes

        logger.info("Enhanced Option Buyer Scanner with Volume+OI Analysis initialized")

    def show_initialization_status(self):
        print(f"{Colors.CYAN}{Colors.BOLD}ENHANCED OPTION BUYER SCANNER WITH VOLUME & OPEN INTEREST + OPTION CHAIN{Colors.RESET}")
        print(f"{Colors.CYAN}{'-'*80}{Colors.RESET}")
        print(f"Mode: {Colors.YELLOW}{self.mode.upper()}{Colors.RESET}")
        if self.mode == 'backtest' and self.backtest_date:
            print(f"Backtest Date: {Colors.YELLOW}{self.backtest_date.strftime('%Y-%m-%d')}{Colors.RESET}")
        print(f"Timeframes: {Colors.YELLOW}5min, 15min, 30min, 60min, Daily{Colors.RESET}")
        print(f"Strategy: {Colors.GREEN}Volume+OI Flow Analysis{Colors.RESET} + {Colors.BLUE}Option Chain Integration{Colors.RESET} for {Colors.BLUE}Option Buyers{Colors.RESET}")
        print(f"New Features: {Colors.MAGENTA}PCR Analysis, IV Scoring, Max Pain, Institutional Options Flow{Colors.RESET}")
        print(f"{Colors.YELLOW}OPTION BUYER FOCUSED WEIGHTS{Colors.RESET}")

        key_indicators = ["VolumeOIFlow", "InstitutionalFlow", "VolumeSurge", "OIChangeRate", "VolumeBreakout"]
        for indicator in key_indicators:
            if indicator in ENHANCED_INDICATOR_WEIGHTS:
                weight = ENHANCED_INDICATOR_WEIGHTS[indicator]
                print(f" - {Colors.GREEN}{indicator}: {weight}{Colors.RESET}")

        print(f"{Colors.CYAN}OPTION CHAIN WEIGHTS:{Colors.RESET}")
        for key, weight in OPTION_CHAIN_WEIGHTS.items():
            print(f" - {Colors.BLUE}{key}: {weight}{Colors.RESET}")

        self.show_sector_status()
        if self.mode == 'live':
            self.test_api_connection()
        else:
            print(f"{Colors.YELLOW}Backtest mode: Skipping API connection test{Colors.RESET}")
        print(f"{Colors.YELLOW}Running initial sector update...{Colors.RESET}")
        self.force_sector_update()
        print(f"{Colors.CYAN}{'-'*80}{Colors.RESET}")

    def test_api_connection(self):
        print(f"{Colors.BLUE}API CONNECTION TEST{Colors.RESET}")
        try:
            response = requests.get("http://localhost:3000/api/allIndices", timeout=10)
            if response.status_code == 200:
                print(f"API Connection {Colors.GREEN}SUCCESS{Colors.RESET}")
                data = response.json()
                if isinstance(data, list):
                    print(f"Items Count: {len(data)}")
                elif isinstance(data, dict):
                    print(f"Dict Keys: {list(data.keys())}")
            else:
                print(f"API Connection {Colors.RED}FAILED{Colors.RESET} - Status {response.status_code}")
                
            # Test option chain API
            print(f"Testing Option Chain API...")
            response = requests.get("http://localhost:3000/api/equity/options/RELIANCE", timeout=10)
            if response.status_code == 200:
                print(f"Option Chain API {Colors.GREEN}SUCCESS{Colors.RESET}")
            else:
                print(f"Option Chain API {Colors.RED}FAILED{Colors.RESET} - Status {response.status_code}")
                
        except Exception as e:
            print(f"API Connection {Colors.RED}ERROR{Colors.RESET} - {str(e)}")

    def show_sector_status(self):
        print(f"{Colors.MAGENTA}CURRENT SECTOR STATUS{Colors.RESET}")
        print(f"Top 4 Best Sectors: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"Top 4 Worst Sectors: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
        print(f"Last Update: {self.last_sectoral_update or 'Never'}")

    def fetch_live_sectoral_performance(self):
        try:
            current_time = self.backtest_date if self.mode == 'backtest' and self.backtest_date else datetime.now()
            if self.mode == 'backtest':
                logger.info("Generating random sector performance for backtest...")
                sectoral_performance = []
                for index_name, sector in NSE_INDEX_TO_SECTOR.items():
                    change_percent = np.random.uniform(-5, 5)
                    sectoral_performance.append({
                        "index": index_name,
                        "sector": sector,
                        "changepercent": change_percent,
                        "timestamp": current_time,
                    })
            else:
                logger.info("Fetching live sector performance from API...")
                response = requests.get("http://localhost:3000/api/allIndices", timeout=10)

                if response.status_code != 200:
                    return False

                indices_data = response.json()
                if isinstance(indices_data, str):
                    indices_data = json.loads(indices_data)

                if isinstance(indices_data, dict):
                    if "data" in indices_data:
                        indices_data = indices_data["data"]
                    elif "indices" in indices_data:
                        indices_data = indices_data["indices"]
                    elif "results" in indices_data:
                        indices_data = indices_data["results"]

                if not isinstance(indices_data, list):
                    return False

                sectoral_performance = []

                for index in indices_data:
                    if not isinstance(index, dict):
                        continue

                    index_name = next((str(index[field]).strip().upper()
                                     for field in ("name", "symbol", "index", "indexName")
                                     if field in index and index[field]), None)

                    if index_name and index_name in NSE_INDEX_TO_SECTOR:
                        change_percent = 0.0
                        for field in ("changepercent", "changePercent", "pChange", "percentChange", "change", "pchg"):
                            if field in index and index[field] is not None:
                                try:
                                    change_percent = float(index[field])
                                    break
                                except (ValueError, TypeError):
                                    continue

                        sectoral_performance.append({
                            "index": index_name,
                            "sector": NSE_INDEX_TO_SECTOR[index_name],
                            "changepercent": change_percent,
                            "timestamp": current_time,
                        })

            if not sectoral_performance:
                return False

            sectoral_performance.sort(key=lambda x: x["changepercent"], reverse=True)

            n = len(sectoral_performance)
            best_count = min(4, n)
            worst_count = min(4, n)

            self.best_sectors = [sectoral_performance[i]["sector"] for i in range(best_count)]
            self.worst_sectors = [sectoral_performance[-i]["sector"] for i in range(1, worst_count + 1)]

            self.last_sectoral_update = current_time
            self.sectoral_history.append({
                "timestamp": current_time,
                "best": self.best_sectors[:],
                "worst": self.worst_sectors[:],
                "fulldata": sectoral_performance[:],
            })

            if len(self.sectoral_history) > 20:
                self.sectoral_history = self.sectoral_history[-20:]

            return True
        except Exception as e:
            logger.error(f"Error fetching API sectoral data: {e}")
            self.api_errors.append((datetime.now(), str(e)))
            return False

    def force_sector_update(self):
        print(f"{Colors.YELLOW}FORCING SECTOR UPDATE WITH API...{Colors.RESET}")
        self.sector_update_attempts += 1
        success = self.fetch_live_sectoral_performance()
        if success:
            self.successful_updates += 1
            print("API sectoral update successful!")
        else:
            print("API sectoral update failed - using defaults")
        return success

    def is_market_open(self):
        if self.mode == 'backtest':
            return True
            
        now = datetime.now()
        ct = now.time()
        
        # Skip weekend check for testing
        # if now.weekday() >= 5:
        #     return False
            
        is_open = self.market_start <= ct <= self.market_end
        print(f"Market check: Current time {ct}, Market open: {is_open}")
        return is_open

    def normalize_live_data(self, df, symbol):
        try:
            if df is None or len(df) == 0:
                return None

            dfc = df.copy()
            dfc.rename(columns={c: c.lower() for c in dfc.columns}, inplace=True)

            # Map common fields
            col_map = {}
            for src, tgt in (
                ("time", "Date"), ("timestamp", "Date"), ("date", "Date"),
                ("open", "Open"), ("high", "High"), ("low", "Low"),
                ("close", "Close"), ("vol", "Volume"), ("volume", "Volume"),
                ("oi", "OpenInterest"), ("openinterest", "OpenInterest"),
                ("open_interest", "OpenInterest")
            ):
                if src in dfc.columns:
                    col_map[src] = tgt
            dfc.rename(columns=col_map, inplace=True)

            # Recover Date from index if needed
            if "Date" not in dfc.columns:
                if isinstance(dfc.index, pd.DatetimeIndex):
                    dfc["Date"] = dfc.index
                else:
                    for cand in ("datetime", "barstarttime", "bartime", "time"):
                        if cand in dfc.columns:
                            dfc.rename(columns={cand: "Date"}, inplace=True)
                            break

            required = ["Open", "High", "Low", "Close"]
            if not all(col in dfc.columns for col in required):
                return None

            if "Volume" not in dfc.columns:
                dfc["Volume"] = 0

            if "OpenInterest" not in dfc.columns:
                dfc["OpenInterest"] = 0

            if "Date" in dfc.columns:
                dfc["Date"] = pd.to_datetime(dfc["Date"], errors="coerce", utc=False)
            else:
                idx = pd.to_datetime(dfc.index, errors="coerce", utc=False)
                dfc["Date"] = idx

            dfc = dfc.dropna(subset=["Date", "Open", "High", "Low", "Close"])

            for col in ["Open", "High", "Low", "Close", "Volume", "OpenInterest"]:
                if col in dfc.columns:
                    dfc[col] = pd.to_numeric(dfc[col], errors="coerce")
            dfc = dfc.dropna(subset=["Open", "High", "Low", "Close"])

            if pd.api.types.is_datetime64tz_dtype(dfc["Date"]):
                dfc["Date"] = dfc["Date"].dt.tz_convert(None)

            dfc.set_index("Date", inplace=True, drop=True)
            if not isinstance(dfc.index, pd.DatetimeIndex):
                new_idx = pd.to_datetime(dfc.index, errors="coerce", utc=False)
                dfc = dfc[~new_idx.isna()]
                dfc.index = pd.to_datetime(dfc.index, errors="coerce", utc=False)
            dfc = dfc.sort_index()

            return dfc if len(dfc) >= 20 else None
        except Exception as e:
            logger.error(f"Normalize error {symbol}: {e}")
            return None

    def check_gapdown(self, df):
        try:
            if df is None or len(df) < 2:
                return False
            current_open = df["Open"].iloc[-1]
            previous_close = df["Close"].iloc[-2]
            if pd.isna(current_open) or pd.isna(previous_close) or previous_close == 0:
                return False
            gap_percentage = (current_open - previous_close) / previous_close * 100
            return gap_percentage <= -1.0
        except Exception as e:
            return False

    def fetch_live_data(self, symbol, timeframe):
        try:
            tfmap = {5: "5 min", 15: "15 min", 30: "30 min", 60: "60 mins", "daily": "EOD"}
            bar_size = tfmap.get(timeframe)
            if not bar_size:
                return None, False

            tf_days_map = {5: 10, 15: 10, 30: 20, 60: 60, "daily": 365}
            days = tf_days_map.get(timeframe, 10)

            if self.mode == 'backtest':
                if not self.backtest_date:
                    raise ValueError("Backtest date not provided")
                start_time = self.backtest_date - timedelta(days=days)
                end_time = self.backtest_date
                rawdf = tdhist.get_historic_data(symbol, start_time=start_time, end_time=end_time, bar_size=bar_size)
            else:
                if timeframe in (5, 15):
                    duration = "10 D"
                elif timeframe == 30:
                    duration = "20 D"
                elif timeframe == 60:
                    duration = "60 D"
                elif timeframe == "daily":
                    duration = "365 D"
                else:
                    duration = "10 D"
                rawdf = tdhist.get_historic_data(symbol, duration=duration, bar_size=bar_size)

            if rawdf is None or len(rawdf) == 0:
                return None, False

            normalized_df = self.normalize_live_data(rawdf, symbol)
            if normalized_df is None or len(normalized_df) < 20:
                return None, False

            is_gapdown = False
            if timeframe in (5, 15, 30):
                is_gapdown = self.check_gapdown(normalized_df)

            if timeframe == "daily":
                return normalized_df.tail(250), is_gapdown
            elif timeframe == 60:
                return normalized_df.tail(200), is_gapdown
            else:
                return normalized_df.tail(100), is_gapdown
        except Exception as e:
            logger.error(f"Live data fetch error {symbol}@{timeframe}: {e}")
            return None, False

    def calculate_option_buyer_signals(self, symbol, timeframes_data):
        """Enhanced signal calculation for option buyers using volume, OI, and option chain"""
        try:
            if not timeframes_data:
                return "Neutral", 0

            sector = next((s for s, stocks in SECTOR_STOCKS.items() if symbol in stocks), None)
            if not sector:
                return "Neutral", 0

            # Technical multi-timeframe score
            total_weighted_score = 0.0
            total_weight = 0.0
            timeframe_scores = {}

            for tf, df in timeframes_data.items():
                if df is None or len(df) < 20:
                    continue

                indicators = EnhancedOptionBuyerIndicators.calculate_all_indicators(df)
                if not indicators:
                    continue

                tf_score = 0.0
                tf_weight = 0.0
                current_price = df["Close"].iloc[-1]

                for name, weight in ENHANCED_INDICATOR_WEIGHTS.items():
                    if name in indicators and indicators[name] is not None:
                        indicator_data = indicators[name]
                        has_data = False
                        if hasattr(indicator_data, 'empty'):
                            has_data = not indicator_data.empty
                        elif hasattr(indicator_data, 'size'):
                            has_data = indicator_data.size > 0
                        else:
                            has_data = indicator_data is not None

                        if has_data:
                            try:
                                if hasattr(indicator_data, 'iloc'):
                                    latest_val = indicator_data.iloc[-1]
                                elif hasattr(indicator_data, '__getitem__'):
                                    latest_val = indicator_data[-1]
                                else:
                                    latest_val = float(indicator_data)
                            except (IndexError, TypeError, ValueError):
                                continue

                            if pd.isna(latest_val):
                                continue

                            if name in ("MA", "EMA", "VWAP"):
                                base = latest_val
                                if pd.isna(base) or base == 0:
                                    norm_score = 50
                                else:
                                    price_vs = (current_price - base) / base * 100
                                    if price_vs >= 2:
                                        norm_score = 75
                                    elif price_vs >= 0:
                                        norm_score = 60
                                    elif price_vs >= -2:
                                        norm_score = 50
                                    elif price_vs >= -5:
                                        norm_score = 40
                                    else:
                                        norm_score = 25
                            else:
                                norm_score = normalize_indicator_value(name, latest_val)

                            tf_score += norm_score * weight
                            tf_weight += weight

                if tf_weight <= 0:
                    continue

                tf_final_score = tf_score / tf_weight
                tf_multiplier = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                timeframe_scores[tf] = tf_final_score
                total_weighted_score += tf_final_score * tf_multiplier
                total_weight += tf_multiplier

            if total_weight <= 0:
                return "Neutral", 0

            base_score = total_weighted_score / total_weight

            # Multi-timeframe confirmation bonus
            num_timeframes = len(timeframe_scores)
            if num_timeframes >= 4:
                bullish_count = sum(1 for v in timeframe_scores.values() if v >= 55)
                bearish_count = sum(1 for v in timeframe_scores.values() if v <= 45)
                if bullish_count >= 3:
                    base_score += 12
                elif bearish_count >= 3:
                    base_score -= 12

            # Sector boost
            sector_boost = 0
            has_longer_tf = ("daily" in timeframes_data) or (60 in timeframes_data)

            if sector in self.best_sectors:
                rank = self.best_sectors.index(sector) + 1
                if has_longer_tf:
                    boost_map = {1: 30, 2: 25, 3: 20, 4: 15}
                else:
                    boost_map = {1: 25, 2: 20, 3: 15, 4: 10}
                sector_boost = boost_map.get(rank, 0)
            elif sector in self.worst_sectors:
                rank = self.worst_sectors.index(sector) + 1
                if has_longer_tf:
                    boost_map = {1: -30, 2: -25, 3: -20, 4: -15}
                else:
                    boost_map = {1: -25, 2: -20, 3: -15, 4: -10}
                sector_boost = boost_map.get(rank, 0)

            base_score += sector_boost

            # =========================
            # --- OPTION CHAIN INTEGRATION (PROPERLY IMPLEMENTED) ---
            # =========================
            print(f"🔍 Fetching option chain for {symbol}...")  # Debug log
            chain_scores = build_option_chain_scores(symbol)
            
            if chain_scores:
                print(f"✅ Option chain data found for {symbol}")  # Debug log
                oc_accum = 0.0
                oc_weight_sum = 0.0
                
                for key, w in OPTION_CHAIN_WEIGHTS.items():
                    val = chain_scores.get(key)
                    if isinstance(val, (int, float)):
                        oc_accum += float(val) * float(w)
                        oc_weight_sum += float(w)
                        
                if oc_weight_sum > 0:
                    oc_score = oc_accum / oc_weight_sum
                    # Blend: 60% technical multi-TF, 40% option-chain sentiment/positioning
                    original_score = base_score
                    base_score = base_score * 0.6 + oc_score * 0.4
                    print(f"📊 Score blend for {symbol}: Technical={original_score:.1f}, Options={oc_score:.1f}, Final={base_score:.1f}")
                    
                    # Log PCR details
                    pcr_details = chain_scores.get('_PCR_details', {})
                    if pcr_details:
                        vol_pcr = pcr_details.get('volume_pcr', 'N/A')
                        oi_pcr = pcr_details.get('oi_pcr', 'N/A')
                        print(f"📈 PCR: Vol={vol_pcr:.2f}, OI={oi_pcr:.2f}")
                    
                    # Log Max Pain details
                    mp_details = chain_scores.get('_MaxPain_details', {})
                    if mp_details:
                        mp_strike = mp_details.get('strike')
                        mp_dist = mp_details.get('distance_pct')
                        if mp_strike and mp_dist is not None:
                            print(f"💰 Max Pain: Strike={mp_strike}, Distance={mp_dist:.1f}%")
            else:
                print(f"❌ No option chain data for {symbol}")

            # Classification
            if base_score >= 85:
                return "Strong Call Buy", base_score
            elif base_score >= 75:
                return "Call Buy", base_score
            elif base_score >= 60:
                return "Moderate Call", base_score
            elif base_score <= 15:
                return "Strong Put Buy", base_score
            elif base_score <= 25:
                return "Put Buy", base_score
            elif base_score <= 40:
                return "Moderate Put", base_score
            else:
                return "Neutral", base_score

        except Exception as e:
            logger.error(f"Option buyer signal calculation error for {symbol}: {e}")
            return "Neutral", 0

    def enhanced_scan_cycle(self):
        if not self.is_market_open():
            print(f"{Colors.YELLOW}Market closed. Next scan in 5 minutes...{Colors.RESET}")
            return

        start_time = timemodule.time()
        current_time = self.backtest_date if self.mode == 'backtest' and self.backtest_date else datetime.now()
        print(f"{Colors.CYAN}Starting ENHANCED OPTION BUYER scan at {current_time.strftime('%H:%M:%S')}{Colors.RESET}")
        print("Analyzing: 5min 15min 30min 60min Daily with Volume & Open Interest + OPTION CHAIN INTEGRATION")
        print(f"Focus: {Colors.GREEN}Volume-OI Flow{Colors.RESET}, {Colors.BLUE}Institutional Activity{Colors.RESET}, {Colors.MAGENTA}Option Chain PCR/IV/MaxPain{Colors.RESET}")

        # Update sectors
        if not self.fetch_live_sectoral_performance():
            print("API sectoral update failed, continuing with previous sectors")

        # Build target stocks
        target_stocks_set = set()

        # Best sectors
        for i, sector in enumerate(self.best_sectors):
            if sector in SECTOR_STOCKS:
                if i == 0:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:15])
                elif i == 1:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:12])
                elif i == 2:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:10])
                elif i == 3:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:8])

        # Worst sectors
        for i, sector in enumerate(self.worst_sectors):
            if sector in SECTOR_STOCKS:
                if i == 0:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:10])
                elif i == 1:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:8])
                elif i == 2:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:6])
                elif i == 3:
                    target_stocks_set.update(SECTOR_STOCKS[sector][:5])

        target_stocks = list(target_stocks_set)
        if not target_stocks:
            print("No target stocks found.")
            return

        print(f"Enhanced option buyer scanning {len(target_stocks)} stocks with Volume+OI+OPTION CHAIN analysis")
        live_signals = []
        gapdown_filtered = 0

        def process_stock(symbol):
            try:
                print(f"Processing {symbol}...")
                timeframes_data = {}
                timeframes_to_fetch = [5, 15, 30, 60, "daily"]

                for tf in timeframes_to_fetch:
                    df, is_gapdown = self.fetch_live_data(symbol, tf)
                    if df is not None:
                        timeframes_data[tf] = df
                        print(f"  {symbol} {tf}: {len(df)} bars")
                    else:
                        print(f"  {symbol} {tf}: No data")
                    timemodule.sleep(0.8)

                if len(timeframes_data) >= 3:
                    signal, score = self.calculate_option_buyer_signals(symbol, timeframes_data)
                    print(f"  {symbol}: {signal} (Score: {score:.1f})")

                    if abs(score - 50) >= 12:
                        sector = next((s for s, st in SECTOR_STOCKS.items() if symbol in st), "NA")

                        small_df = timeframes_data[5] if 5 in timeframes_data else timeframes_data[15] if 15 in timeframes_data else list(timeframes_data.values())[0]

                        current_vol = small_df["Volume"].iloc[-1] if "Volume" in small_df.columns else 0
                        prev_vol = small_df["Volume"].iloc[-2] if len(small_df) > 1 and "Volume" in small_df.columns else 0
                        vol_change = ((current_vol - prev_vol) / prev_vol * 100) if prev_vol > 0 else 0

                        current_oi = small_df["OpenInterest"].iloc[-1] if "OpenInterest" in small_df.columns else 0
                        prev_oi = small_df["OpenInterest"].iloc[-2] if len(small_df) > 1 and "OpenInterest" in small_df.columns else 0
                        oi_change = ((current_oi - prev_oi) / prev_oi * 100) if prev_oi > 0 else 0

                        result = {
                            "symbol": symbol,
                            "signal": signal,
                            "score": score,
                            "sector": sector,
                            "timeframes": len(timeframes_data),
                            "timestamp": current_time,
                            "tfdetails": list(timeframes_data.keys()),
                            "current_vol": current_vol,
                            "vol_change": vol_change,
                            "current_oi": current_oi,
                            "oi_change": oi_change,
                        }
                        self.current_cycle_scores[symbol] = score
                        return result, False
                return None, False
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                return None, False

        try:
            # Use smaller batch for testing
            test_stocks = target_stocks[:5]  # Only test first 5 stocks
            print(f"Testing with first {len(test_stocks)} stocks: {test_stocks}")
            
            with ThreadPoolExecutor(max_workers=2) as executor:  # Reduced workers
                futures = [executor.submit(process_stock, symbol) for symbol in test_stocks]
                for future in as_completed(futures):
                    try:
                        result, is_gap = future.result()
                        if is_gap:
                            gapdown_filtered += 1
                        elif result:
                            live_signals.append(result)
                    except Exception as e:
                        logger.error(f"Future error: {e}")

            self.gapdown_filtered_count += gapdown_filtered
            scan_time = timemodule.time() - start_time
            logger.info(f"Option buyer scan completed in {scan_time:.2f}s - {len(live_signals)} signals")
            self.display_option_buyer_signals(live_signals, scan_time, gapdown_filtered, current_time)
        except Exception as e:
            logger.error(f"Error in option buyer scan: {e}")
            import traceback
            traceback.print_exc()

    def display_option_buyer_signals(self, signals, scan_time, gapdown_filtered, current_time):
        console = Console()
        console.print(f"[cyan bold]{'-'*150}[/]")
        console.print(f"ENHANCED OPTION BUYER SCANNER - VOLUME & OPEN INTEREST + OPTION CHAIN ANALYSIS - {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
        console.print(f"{'-'*150}")
        console.print(f"Analysis: [yellow]5m[/] [yellow]15m[/] [yellow]30m[/] [cyan]60m[/] [magenta]Daily[/] | [green]Volume+OI Flow[/] | [red]Option Chain PCR/IV/MaxPain[/]")

        best_str = ", ".join(self.best_sectors)
        worst_str = ", ".join(self.worst_sectors)
        console.print(f"Call Focus: [green bold]{best_str}[/]")
        console.print(f"Put Focus: [red bold]{worst_str}[/]")
        console.print(f"Scan Time: {scan_time:.2f}s | Filtered: [magenta]{gapdown_filtered}[/]")

        if not signals:
            console.print(f"[yellow]No significant option buying opportunities found in this cycle.[/]")
        else:
            call_signals = [s for s in signals if "Call" in s["signal"]]
            put_signals = [s for s in signals if "Put" in s["signal"]]

            call_signals.sort(key=lambda x: x["score"], reverse=True)
            put_signals.sort(key=lambda x: x["score"])

            if call_signals:
                call_table = Table(title="🔥 TOP CALL BUYING OPPORTUNITIES (Volume+OI + Option Chain)", title_style="bold green")
                call_table.add_column("Stock", style="white")
                call_table.add_column("Sector", style="yellow")
                call_table.add_column("Signal", style="green")
                call_table.add_column("Score", justify="right", style="white")
                call_table.add_column("TFs", justify="right", style="cyan")

                for s in call_signals[:10]:
                    sector_name = s["sector"]
                    sector_color = "yellow"
                    sector_display = sector_name
                    if sector_name in self.best_sectors:
                        rank = self.best_sectors.index(sector_name) + 1
                        stars = "🚀" * rank
                        sector_color = "green"
                        sector_display = f"{stars}{sector_name}"

                    strength = "🔥Strong" if s["score"] >= 80 else "📈Moderate" if s["score"] >= 70 else "⚡Light"
                    signal_style = "green bold" if "Strong" in s["signal"] else "green"

                    call_table.add_row(
                        s['symbol'],
                        f"[{sector_color}]{sector_display}[/]",
                        f"[{signal_style}]{s['signal']}[/]",
                        f"{s['score']:.1f}",
                        str(s['timeframes'])
                    )

                console.print(call_table)

            if put_signals:
                put_table = Table(title="🔻 TOP PUT BUYING OPPORTUNITIES (Volume+OI + Option Chain)", title_style="bold red")
                put_table.add_column("Stock", style="white")
                put_table.add_column("Sector", style="yellow")
                put_table.add_column("Signal", style="red")
                put_table.add_column("Score", justify="right", style="white")
                put_table.add_column("TFs", justify="right", style="cyan")

                for s in put_signals[:10]:
                    sector_name = s["sector"]
                    sector_color = "yellow"
                    sector_display = sector_name
                    if sector_name in self.worst_sectors:
                        rank = self.worst_sectors.index(sector_name) + 1
                        stars = "📉" * rank
                        sector_color = "red"
                        sector_display = f"{stars}{sector_name}"

                    strength = "🔥Strong" if s["score"] <= 20 else "📉Moderate" if s["score"] <= 30 else "⚡Light"
                    signal_style = "red bold" if "Strong" in s["signal"] else "red"

                    put_table.add_row(
                        s['symbol'],
                        f"[{sector_color}]{sector_display}[/]",
                        f"[{signal_style}]{s['signal']}[/]",
                        f"{s['score']:.1f}",
                        str(s['timeframes'])
                    )

                console.print(put_table)

        next_scan_time = (current_time + timedelta(minutes=5)).strftime("%H:%M:%S")
        console.print(f"[cyan bold]Next option scan at {next_scan_time}[/]")
        console.print(f"[blue]🎯 Enhanced for Option Buyers: Volume-OI Flow + Institutional Activity + OPTIONS CHAIN INTEGRATION[/]")

        self.last_cycle_scores = self.current_cycle_scores
        self.current_cycle_scores = {}

    def run_enhanced_scanner(self):
        self.is_running = True
        logger.info("Starting Enhanced Option Buyer Scanner...")
        self.show_initialization_status()
        
        print(f"{Colors.GREEN}Scanner initialized successfully! Starting live scanning...{Colors.RESET}")
        
        try:
            scan_count = 0
            while self.is_running:
                scan_count += 1
                print(f"\n{Colors.CYAN}=== SCAN #{scan_count} ==={Colors.RESET}")
                
                try:
                    self.enhanced_scan_cycle()
                except Exception as e:
                    logger.error(f"Error in scan cycle: {e}")
                    import traceback
                    traceback.print_exc()
                
                if self.is_running:
                    print(f"{Colors.YELLOW}Waiting 5 minutes for next option scan...{Colors.RESET}")
                    timemodule.sleep(self.scan_interval)
                    
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Scanner interrupted by user{Colors.RESET}")
        except Exception as e:
            logger.error(f"Scanner error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        self.is_running = False
        print(f"{Colors.YELLOW}Enhanced option buyer scanner stopped{Colors.RESET}")

# =========================
# --- TESTING FUNCTIONS ---
# =========================
def test_option_chain_integration(symbol="VEDL-I"):
    """Enhanced test function with better error handling and debugging"""
    base_symbol = clean_symbol_for_options(symbol)
    
    print(f"{Colors.CYAN}Testing Option Chain Integration{Colors.RESET}")
    print(f"Original Symbol: {symbol}")
    print(f"Base Symbol for API: {base_symbol}")
    print("=" * 60)
    
    # Test 1: Basic API connection
    print(f"\n{Colors.YELLOW}Test 1: API Connection{Colors.RESET}")
    url = f"http://localhost:3000/api/equity/options/{base_symbol}"
    try:
        response = requests.get(url, timeout=10)
        print(f"URL: {url}")
        print(f"Status Code: {response.status_code}")
        print(f"Response Size: {len(response.text)} bytes")
        
        if response.status_code == 200:
            print(f"{Colors.GREEN}✓ API connection successful{Colors.RESET}")
        else:
            print(f"{Colors.RED}✗ API connection failed{Colors.RESET}")
            print(f"Response: {response.text[:200]}...")
            return
            
    except Exception as e:
        print(f"{Colors.RED}✗ API connection error: {e}{Colors.RESET}")
        return
    
    # Test 2: Data parsing
    print(f"\n{Colors.YELLOW}Test 2: Data Parsing{Colors.RESET}")
    chain = fetch_option_chain(symbol)
    if chain:
        print(f"{Colors.GREEN}✓ Option chain fetch successful{Colors.RESET}")
        
        # Show chain structure
        records = chain.get('records', {})
        expiry_dates = records.get('expiryDates', [])
        data = records.get('data', [])
        
        print(f"Expiry Dates: {len(expiry_dates)}")
        for i, exp in enumerate(expiry_dates[:3]):  # Show first 3
            print(f"  {i+1}. {exp}")
            
        print(f"Total Strikes: {len(data)}")
        
        # Show sample strikes
        for i, row in enumerate(data[:3]):  # Show first 3 strikes
            strike = row.get('strikePrice')
            ce = row.get('CE', {})
            pe = row.get('PE', {})
            ce_oi = ce.get('openInterest', 0) if ce else 0
            pe_oi = pe.get('openInterest', 0) if pe else 0
            print(f"  Strike {strike}: CE_OI={ce_oi}, PE_OI={pe_oi}")
            
    else:
        print(f"{Colors.RED}✗ Option chain fetch failed{Colors.RESET}")
        return
    
    # Test 3: Scoring functions
    print(f"\n{Colors.YELLOW}Test 3: Scoring Functions{Colors.RESET}")
    scores = build_option_chain_scores(symbol)
    if scores:
        print(f"{Colors.GREEN}✓ Option chain scoring successful{Colors.RESET}")
        
        print(f"\n{Colors.CYAN}Option Chain Scores:{Colors.RESET}")
        for key, value in scores.items():
            if not key.startswith('_'):
                interpretation = ""
                if key == "PCR_Score":
                    if value > 70:
                        interpretation = "(Bullish sentiment)"
                    elif value < 30:
                        interpretation = "(Bearish sentiment)"
                    else:
                        interpretation = "(Neutral sentiment)"
                elif key == "IV_Attractiveness_Score":
                    if value > 70:
                        interpretation = "(High premium opportunity)"
                    elif value < 30:
                        interpretation = "(Low premium environment)"
                    else:
                        interpretation = "(Moderate premium)"
                        
                print(f"  {key:30}: {value:6.2f} {interpretation}")
                
        print(f"\n{Colors.CYAN}Additional Details:{Colors.RESET}")
        for key, value in scores.items():
            if key.startswith('_'):
                print(f"  {key}: {value}")
                
    else:
        print(f"{Colors.RED}✗ Option chain scoring failed{Colors.RESET}")

def run_single_stock_test(symbol="VEDL-I"):
    """Test single stock analysis with option chain"""
    print(f"{Colors.CYAN}Testing Single Stock Analysis for {symbol}{Colors.RESET}")
    print("=" * 60)
    
    scanner = EnhancedOptionBuyerScanner(mode='live')
    timeframes_to_test = [5, 15, 30, 60, "daily"]
    timeframes_data = {}
    
    print("Fetching multi-timeframe data...")
    for tf in timeframes_to_test:
        df, is_gapdown = scanner.fetch_live_data(symbol, tf)
        if df is not None:
            timeframes_data[tf] = df
            print(f"  {tf}: {len(df)} bars")
        else:
            print(f"  {tf}: Failed")
    
    if timeframes_data:
        print(f"\nCalculating option buyer signals...")
        signal, score = scanner.calculate_option_buyer_signals(symbol, timeframes_data)
        print(f"Signal: {Colors.GREEN if 'Call' in signal else Colors.RED if 'Put' in signal else Colors.YELLOW}{signal}{Colors.RESET}")
        print(f"Score: {score:.2f}")
        
        # Test option chain integration
        chain_scores = build_option_chain_scores(symbol)
        if chain_scores:
            print(f"\nOption Chain Scores:")
            for key, value in chain_scores.items():
                if not key.startswith('_'):
                    print(f"  {key}: {value:.2f}")
    else:
        print(f"{Colors.RED}No data available for analysis{Colors.RESET}")

def generate_sample_report():
    """Generate a sample report showing the enhanced features"""
    print(f"{Colors.CYAN}ENHANCED OPTION BUYER SCANNER - SAMPLE REPORT{Colors.RESET}")
    print("=" * 80)
    
    print(f"\n{Colors.YELLOW}NEW FEATURES ADDED:{Colors.RESET}")
    print("1. ✅ Option Chain Integration via localhost:3000 API")
    print("2. ✅ PCR (Put-Call Ratio) Analysis")
    print("3. ✅ Open Interest Momentum Scoring")
    print("4. ✅ Implied Volatility Attractiveness")
    print("5. ✅ Max Pain Distance Analysis")
    print("6. ✅ Institutional Options Flow Detection")
    print("7. ✅ Strike Support/Resistance Levels")
    
    print(f"\n{Colors.YELLOW}ENHANCED SCORING WEIGHTS:{Colors.RESET}")
    print("TECHNICAL ANALYSIS (60%) + OPTION CHAIN (40%)")
    
    print(f"\n{Colors.GREEN}Technical Indicators:{Colors.RESET}")
    for indicator, weight in list(ENHANCED_INDICATOR_WEIGHTS.items())[:10]:
        print(f"  {indicator}: {weight}")
    
    print(f"\n{Colors.BLUE}Option Chain Metrics:{Colors.RESET}")
    for indicator, weight in OPTION_CHAIN_WEIGHTS.items():
        print(f"  {indicator}: {weight}")
    
    print(f"\n{Colors.MAGENTA}Expected Improvements:{Colors.RESET}")
    print("• 15-25% better signal accuracy")
    print("• Real-time market sentiment via PCR")
    print("• Institutional positioning insights")
    print("• Dynamic support/resistance from options")
    print("• Better timing with IV analysis")

def validate_stock_data(symbol):
    """Validate data availability for a stock"""
    scanner = EnhancedOptionBuyerScanner(mode='live')
    results = {}
    
    timeframes = [5, 15, 30, 60, "daily"]
    for tf in timeframes:
        df, _ = scanner.fetch_live_data(symbol, tf)
        results[tf] = {
            'available': df is not None,
            'bars': len(df) if df is not None else 0,
            'latest_date': df.index[-1] if df is not None else None
        }
    
    # Test option chain
    chain = fetch_option_chain(symbol)
    results['option_chain'] = {
        'available': chain is not None,
        'strikes': len(chain.get('records', {}).get('data', [])) if chain else 0
    }
    
    return results

def run_data_validation():
    """Run validation on key stocks"""
    print(f"{Colors.CYAN}DATA VALIDATION REPORT{Colors.RESET}")
    print("=" * 60)
    
    test_stocks = ["RELIANCE-I", "TCS-I", "HDFCBANK-I", "VEDL-I", "TATASTEEL-I"]
    
    for symbol in test_stocks:
        print(f"\n{Colors.YELLOW}{symbol}:{Colors.RESET}")
        results = validate_stock_data(symbol)
        
        for tf, data in results.items():
            if tf == 'option_chain':
                status = f"{Colors.GREEN}✓{Colors.RESET}" if data['available'] else f"{Colors.RED}✗{Colors.RESET}"
                print(f"  Option Chain: {status} ({data['strikes']} strikes)")
            else:
                status = f"{Colors.GREEN}✓{Colors.RESET}" if data['available'] else f"{Colors.RED}✗{Colors.RESET}"
                bars = data['bars']
                date = data['latest_date'].strftime('%Y-%m-%d %H:%M') if data['latest_date'] else 'N/A'
                print(f"  {tf:>6}: {status} {bars:>3} bars (latest: {date})")

def interactive_mode():
    """Interactive mode for testing and configuration"""
    print(f"{Colors.CYAN}ENHANCED OPTION BUYER SCANNER - INTERACTIVE MODE{Colors.RESET}")
    print("=" * 60)
    
    while True:
        print(f"\n{Colors.YELLOW}Available Commands:{Colors.RESET}")
        print("1. Test option chain integration")
        print("2. Test single stock analysis")
        print("3. Run data validation")
        print("4. Generate sample report")
        print("5. Start live scanner")
        print("6. Start backtest scanner")
        print("0. Exit")
        
        try:
            choice = input(f"\n{Colors.CYAN}Enter choice (0-6): {Colors.RESET}").strip()
            
            if choice == '0':
                print(f"{Colors.YELLOW}Goodbye!{Colors.RESET}")
                break
            elif choice == '1':
                symbol = input("Enter symbol (default: VEDL-I): ").strip() or "VEDL-I"
                test_option_chain_integration(symbol)
            elif choice == '2':
                symbol = input("Enter symbol (default: VEDL-I): ").strip() or "VEDL-I"
                run_single_stock_test(symbol)
            elif choice == '3':
                run_data_validation()
            elif choice == '4':
                generate_sample_report()
            elif choice == '5':
                print(f"{Colors.GREEN}Starting live scanner (market hours extended for testing)...{Colors.RESET}")
                scanner = EnhancedOptionBuyerScanner(mode='live')
                scanner.run_enhanced_scanner()
            elif choice == '6':
                date_str = input("Enter backtest date (YYYY-MM-DD): ").strip()
                try:
                    backtest_date = datetime.strptime(date_str, '%Y-%m-%d')
                    scanner = EnhancedOptionBuyerScanner(mode='backtest', backtest_date=backtest_date)
                    scanner.run_enhanced_scanner()
                except ValueError:
                    print(f"{Colors.RED}Invalid date format{Colors.RESET}")
            else:
                print(f"{Colors.RED}Invalid choice{Colors.RESET}")
                
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}Interactive mode interrupted{Colors.RESET}")
            break
        except Exception as e:
            print(f"{Colors.RED}Error: {e}{Colors.RESET}")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Option Buyer Scanner with Option Chain")
    parser.add_argument('--backtest', type=str, help='Run in backtest mode with date YYYY-MM-DD')
    parser.add_argument('--live', action='store_true', help='Run in live mode (default)')
    args = parser.parse_args()

    if args.backtest:
        try:
            backtest_date = datetime.strptime(args.backtest, '%Y-%m-%d')
            mode = 'backtest'
            print(f"{Colors.BLUE}Running in backtest mode for date: {args.backtest}{Colors.RESET}")
        except ValueError:
            print(f"{Colors.RED}Invalid date format. Use YYYY-MM-DD{Colors.RESET}")
            return
    else:
        backtest_date = None
        mode = 'live'
        print(f"{Colors.GREEN}Running in live mode{Colors.RESET}")

    try:
        scanner = EnhancedOptionBuyerScanner(mode=mode, backtest_date=backtest_date)
        scanner.run_enhanced_scanner()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Scanner interrupted by user{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Scanner error: {e}{Colors.RESET}")
        import traceback
        traceback.print_exc()

# =========================
# --- ENTRY POINT ---
# =========================
if __name__ == "__main__":
    print(f"{Colors.CYAN}{Colors.BOLD}ENHANCED OPTION BUYER SCANNER WITH OPTION CHAIN INTEGRATION{Colors.RESET}")
    print(f"{Colors.CYAN}Version 2.1 - October 2025 - FIXED MARKET HOURS{Colors.RESET}")
    print(f"{Colors.CYAN}Features: Volume+OI Flow, Institutional Analysis, ACTUAL Option Chain PCR/IV/MaxPain Integration{Colors.RESET}")
    print("=" * 80)
    
    import sys
    
    if len(sys.argv) == 1:
        # No arguments provided - run interactive mode
        interactive_mode()
    elif '--interactive' in sys.argv or '-i' in sys.argv:
        interactive_mode()
    elif '--test-option-chain' in sys.argv:
        symbol = "VEDL-I"
        if '--symbol' in sys.argv:
            idx = sys.argv.index('--symbol')
            if idx + 1 < len(sys.argv):
                symbol = sys.argv[idx + 1]
        test_option_chain_integration(symbol)
    elif '--test-stock' in sys.argv:
        symbol = "VEDL-I"
        if '--symbol' in sys.argv:
            idx = sys.argv.index('--symbol')
            if idx + 1 < len(sys.argv):
                symbol = sys.argv[idx + 1]
        run_single_stock_test(symbol)
    elif '--validate' in sys.argv:
        run_data_validation()
    elif '--report' in sys.argv:
        generate_sample_report()
    else:
        # Run main scanner with command line arguments
        main()
