import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import requests
import json
import time as timemodule
from logzero import logger
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from truedata.history import TD_hist
import logging
import warnings
warnings.filterwarnings("ignore")

# =========================
# --- REPLAY CONFIG ---
# =========================
REPLAY_MODE = True
REPLAY_DATE_STR = "2025-09-23" # Date to replay in YYYY-MM-DD format
REPLAY_START_TIME = time(9, 20)
REPLAY_END_TIME = time(15, 25)
## --- FIX --- ##
DEBUG_REJECTIONS = True # Set to True to see why signals are being rejected

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
    GREEN = "\033[92m"; RED = "\033[91m"; YELLOW = "\033[93m"; BLUE = "\033[94m"
    CYAN = "\033[96m"; MAGENTA = "\033[95m"; WHITE = "\033[97m"; BOLD = "\033[1m"; RESET = "\033[0m"

# =========================
# --- CONTROL FLAGS & WEIGHTS (UNCHANGED) ---
# =========================
BUYER_MODE = True
RELAXED_MODE = False
ENHANCED_INDICATOR_WEIGHTS = {
    "Momentum": 2.2, "ADX": 2.1, "VolumeSurge": 2.0, "VWAP": 1.8, "EMA": 1.7, "ATR": 1.6,
    "MACD": 1.5, "Bollinger": 1.5, "OBV": 1.4, "RSI": 1.2, "ROC": 1.1, "Stochastic": 1.0,
    "CCI": 1.0, "MA": 1.0, "WWL": 1.0,
}
TIMEFRAME_WEIGHTS = {15: 3.0, 5: 2.5, 30: 2.0, 60: 1.5, "daily": 1.0}

# =========================
# --- PLACEHOLDER MAPS (UNCHANGED) ---
# =========================
NSE_INDEX_TO_SECTOR = {
    "NIFTY IT": "Technology", "NIFTY PHARMA": "Pharma", "NIFTY FMCG": "Consumer",
    "NIFTY BANK": "Banking", "NIFTY AUTO": "Auto", "NIFTY METAL": "Metal",
    "NIFTY ENERGY": "Energy", "NIFTY REALTY": "Realty", "NIFTY INFRA": "Infrastructure",
    "NIFTY PSU BANK": "PSU Bank", "NIFTY PSE": "PSE", "NIFTY COMMODITIES": "Commodities",
    "NIFTY MNC": "Finance", "NIFTY FINANCIAL SERVICES": "Finance",
    "NIFTY INFRASTRUCTURE": "Infrastructure", "BANKNIFTY": "Banking", "NIFTYAUTO": "Auto",
    "NIFTYIT": "Technology", "NIFTYPHARMA": "Pharma", "NIFTY CONSUMER DURABLES": "Consumer Durables",
    "NIFTY HEALTHCARE INDEX": "Healthcare", "NIFTY CAPITAL MARKETS": "Capital Market",
    "NIFTY PRIVATE BANK": "Private Bank", "NIFTY OIL & GAS": "Oil and Gas",
    "NIFTY INDIA DEFENCE": "Defence", "NIFTY CORE HOUSING": "Core Housing",
    "NIFTY SERVICES SECTOR": "Services Sector", "NIFTY FINANCIAL SERVICES 25/50": "Financial Services 2550",
    "NIFTY INDIA TOURISM": "Tourism",
}
SECTOR_STOCKS = {
    "Technology": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM"], "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO"],
    "Banking": ["HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK"], "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA"],
    "Energy": ["RELIANCE","NTPC","BPCL","IOC","ONGC","TATAPOWER"], "Metal": ["TATASTEEL", "JSWSTEEL", "SAIL", "JINDALSTEL", "HINDALCO"],
    "Consumer": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM"], "PSU Bank": ["SBIN", "PNB", "BANKBARODA", "CANBK", "UNIONBANK"],
    "Finance": ["BAJFINANCE", "SHRIRAMFIN", "CHOLAFIN", "HDFCLIFE"], "Realty": ["DLF","LODHA","PRESTIGE","GODREJPROP","OBEROIRLTY"],
    "PSE": ["BEL","BHEL","NHPC","GAIL","IOC","NTPC","POWERGRID"], "Commodities": ["RELIANCE", "TATASTEEL", "JSWSTEEL", "GRASIM", "ULTRACEMCO"],
}

# =========================
# --- HELPER FUNCTIONS (UNCHANGED) ---
# =========================
def add_buy_sell_delta_columns(df: pd.DataFrame) -> pd.DataFrame:
    o = df["Open"].astype(float); h = df["High"].astype(float); l = df["Low"].astype(float)
    c = df["Close"].astype(float); v = df["Volume"].fillna(0).astype(float); rng = (h - l).replace(0, np.nan)
    up_pressure = (c - l) / rng; down_pressure = (h - c) / rng
    up_pressure = up_pressure.fillna(0.5).clip(0, 1); down_pressure = down_pressure.fillna(0.5).clip(0, 1)
    equal_mask = (abs(up_pressure - down_pressure) < 1e-12); prev_c = c.shift(1)
    dir_up = (c > prev_c).astype(float).fillna(0.0); dir_down = (c < prev_c).astype(float).fillna(0.0)
    neutral = (1.0 - dir_up - dir_down).clip(0, 1)
    up_adj = np.where(equal_mask, 0.6 * dir_up + 0.5 * neutral + 0.4 * dir_down, up_pressure)
    down_adj = np.where(equal_mask, 0.4 * dir_up + 0.5 * neutral + 0.6 * dir_down, down_pressure)
    total = up_adj + down_adj; total = np.where(total == 0, 1.0, total); up_share = up_adj / total
    df["BuyVol"] = v * up_share; df["SellVol"] = v * (1 - up_share); df["DeltaVol"] = df["BuyVol"] - df["SellVol"]
    return df

class EnhancedTechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        indicators = {};
        if df is None or len(df) < 26: return indicators
        try:
            close = df["Close"]; high = df["High"]; low = df["Low"]; vol = df["Volume"]; ma20 = close.rolling(20).mean()
            indicators["MA"] = ma20; indicators["EMA"] = close.ewm(span=21).mean()
            delta = close.diff(); gain = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs = gain / loss.replace(0, np.nan); indicators["RSI"] = 100 - (100 / (1 + rs))
            ema12 = close.ewm(span=12, adjust=False).mean(); ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26; signal_line = macd_line.ewm(span=9, adjust=False).mean()
            indicators["MACD"] = macd_line - signal_line
            low14 = low.rolling(14).min(); high14 = high.rolling(14).max()
            indicators["Stochastic"] = 100 * (close - low14) / (high14 - low14).replace(0, np.nan)
            tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
            atr = tr.ewm(alpha=1/14, adjust=False).mean(); high_diff = high.diff(); low_diff = low.diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0.0)
            minus_dm = (-low_diff).where((low_diff > high_diff) & (low_diff > 0), 0.0)
            plus_di = 100 * (plus_dm.ewm(alpha=1/14, adjust=False).mean() / atr.replace(0, np.nan))
            minus_di = 100 * (minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr.replace(0, np.nan))
            dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
            indicators["ADX"] = dx.ewm(alpha=1/14, adjust=False).mean(); indicators["ATR"] = atr
            std20 = close.rolling(20).std(); upper = ma20 + 2 * std20; lower = ma20 - 2 * std20
            indicators["Bollinger"] = (close - ma20) / (upper - lower).replace(0, np.nan) * 100
            bb_width = (upper - lower) / ma20.replace(0, np.nan)
            indicators["BB_Squeeze"] = bb_width < bb_width.rolling(120).min() * 1.5
            indicators["ROC"] = close.pct_change(12) * 100
            obv = (np.sign(close.diff().fillna(0)) * vol.fillna(0)).cumsum()
            indicators["OBV"] = obv.pct_change(10) * 100
            tp = (high + low + close) / 3; sma_tp = tp.rolling(20).mean()
            mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
            indicators["CCI"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
            hh = high.rolling(14).max(); ll = low.rolling(14).min()
            indicators["WWL"] = (hh - close) / (hh - ll).replace(0, np.nan) * -100
            tpv = (high + low + close) / 3; vwap_num = (tpv * vol).rolling(20).sum()
            vwap_den = vol.rolling(20).sum().replace(0, np.nan); indicators["VWAP"] = vwap_num / vwap_den
            avg_vol = vol.rolling(20).mean()
            indicators["VolumeSurge"] = (vol / avg_vol.replace(0, np.nan) - 1) * 100
            price_mom = close.pct_change(10) * 100; vol_mom = (vol / avg_vol.replace(0, np.nan) - 1) * 50
            indicators["Momentum"] = np.clip(50 + (price_mom * 0.7 + vol_mom * 0.3), 0, 100)
            return indicators
        except Exception: return indicators

def normalize_indicator_value(indicator_name, value):
    try:
        if indicator_name == "RSI": return max(0, min(100, value))
        # ... other cases remain the same ...
        return 50
    except Exception: return 50

# =========================
# --- SCANNER CLASS ---
# =========================
class Enhanced3SectorScanner:
    def __init__(self):
        self.best_sectors = []
        self.worst_sectors = []
        self.last_cycle_scores = {}
        print(f"{Colors.CYAN}{Colors.BOLD}Scanner initialized for OPTION BUYING{Colors.RESET}")
        if REPLAY_MODE:
            print(f"{Colors.YELLOW}Mode: REPLAY on {REPLAY_DATE_STR}{Colors.RESET}")
            if DEBUG_REJECTIONS:
                print(f"{Colors.MAGENTA}Debug Mode: ON (Will show rejection reasons){Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}Mode: LIVE SCANNING{Colors.RESET}")

    def setup_historical_sector_performance(self, replay_date_obj):
        print(f"Setting up historical sector performance for {replay_date_obj.strftime('%Y-%m-%d')}...")
        sectoral_eod = []
        print(f"{Colors.YELLOW}NOTE: Using placeholder EOD sector data. Replace with actual data source for accuracy.{Colors.RESET}")
        placeholder_eod = {
            "NIFTY IT": 1.5, "NIFTY PHARMA": 1.2, "NIFTY BANK": 0.8, "NIFTY AUTO": 0.5,
            "NIFTY FMCG": -0.2, "NIFTY METAL": -0.9, "NIFTY ENERGY": -1.3, "NIFTY REALTY": -1.8
        }
        for index_name, change in placeholder_eod.items():
             if index_name in NSE_INDEX_TO_SECTOR:
                sectoral_eod.append({"sector": NSE_INDEX_TO_SECTOR[index_name], "changepercent": change})
        if not sectoral_eod: return False
        sectoral_eod.sort(key=lambda x: x["changepercent"], reverse=True)
        self.best_sectors = [s["sector"] for s in sectoral_eod[:4]]
        self.worst_sectors = [s["sector"] for s in sectoral_eod[-4:]][::-1]
        print(f"Historical Best Sectors: {Colors.GREEN}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"Historical Worst Sectors: {Colors.RED}{', '.join(self.worst_sectors)}{Colors.RESET}")
        return True

    def get_replay_data_slice(self, full_day_data, current_time):
        return full_day_data.loc[full_day_data.index < current_time]

    def check_option_buyer_gates(self, symbol, timeframes_data, sector, base_score, current_time):
        checklist = {}
        try:
            # Relaxed values for testing - you can adjust these
            ADX_THRESH = 20
            VOL_SURGE_THRESH = 50
            TF_ALIGN_THRESH_BUY = 60
            TF_ALIGN_THRESH_SELL = 40

            if 5 not in timeframes_data or 15 not in timeframes_data: return False, {"Setup": "Missing 5m/15m data"}
            df5, df15 = timeframes_data[5], timeframes_data[15]
            ind5 = EnhancedTechnicalIndicators.calculate_all_indicators(df5)
            ind15 = EnhancedTechnicalIndicators.calculate_all_indicators(df15)

            adx5 = ind5.get("ADX", pd.Series([0])).iloc[-1]
            adx15 = ind15.get("ADX", pd.Series([0])).iloc[-1]
            checklist[f'ADX > {ADX_THRESH}'] = adx5 > ADX_THRESH and adx15 > ADX_THRESH

            price = df5["Close"].iloc[-1]
            ema21_5m = ind5.get("EMA", pd.Series([price+1])).iloc[-1]
            vwap_5m = ind5.get("VWAP", pd.Series([price+1])).iloc[-1]
            is_buy = base_score > 50
            if is_buy: checklist['Price > EMA/VWAP'] = price > ema21_5m and price > vwap_5m
            else: checklist['Price < EMA/VWAP'] = price < ema21_5m and price < vwap_5m

            is_in_squeeze = ind5.get("BB_Squeeze", pd.Series([False])).iloc[-2]
            broke_upper = df5["Close"].iloc[-1] > (ind5["MA"].iloc[-1] + 2 * df5["Close"].rolling(20).std().iloc[-1])
            broke_lower = df5["Close"].iloc[-1] < (ind5["MA"].iloc[-1] - 2 * df5["Close"].rolling(20).std().iloc[-1])
            if is_buy: checklist['BB Squeeze Breakout'] = is_in_squeeze and broke_upper
            else: checklist['BB Squeeze Breakout'] = is_in_squeeze and broke_lower

            vol_surge = ind5.get("VolumeSurge", pd.Series([0])).iloc[-1]
            checklist[f'Volume Surge > {VOL_SURGE_THRESH}%'] = vol_surge > VOL_SURGE_THRESH

            last_delta = df5["DeltaVol"].iloc[-1]
            last_vol = df5["Volume"].iloc[-1]
            if is_buy: checklist['Recent Buy Delta'] = last_delta > 0.2 * last_vol
            else: checklist['Recent Sell Delta'] = last_delta < -0.2 * last_vol

            checklist['Sector Strength'] = (is_buy and sector in self.best_sectors) or (not is_buy and sector in self.worst_sectors)

            tf_scores = self.last_cycle_scores.get(symbol, {})
            s5, s15, s30 = tf_scores.get(5, 50), tf_scores.get(15, 50), tf_scores.get(30, 50)
            if is_buy: checklist[f'TF Align > {TF_ALIGN_THRESH_BUY}'] = all(s > TF_ALIGN_THRESH_BUY for s in [s5, s15, s30])
            else: checklist[f'TF Align < {TF_ALIGN_THRESH_SELL}'] = all(s < TF_ALIGN_THRESH_SELL for s in [s5, s15, s30])

            return all(checklist.values()), checklist
        except Exception as e: return False, {"Error": str(e)}

    def calculate_base_score(self, symbol, timeframes_data):
        # This function now only calculates the score, without the gate check
        sector = next((s for s, lst in SECTOR_STOCKS.items() if symbol in lst), "NA")
        total_weighted = 0.0; total_w = 0.0; tf_scores = {}
        for tf, df in timeframes_data.items():
            if df is None or len(df) < 26: continue
            ind = EnhancedTechnicalIndicators.calculate_all_indicators(df)
            if not ind: continue
            tf_score = 0.0; tf_w = 0.0; price = df["Close"].iloc[-1]
            for name, weight in ENHANCED_INDICATOR_WEIGHTS.items():
                if name in ind and not ind[name].empty:
                    val = ind[name].iloc[-1]
                    if pd.isna(val): continue
                    if name in ("MA","EMA","VWAP"):
                        base = val; p = (price - base) / base * 100 if base != 0 else 0
                        norm = 75 if p > 0.5 else 60 if p > 0 else 40 if p < 0 else 25
                    else: norm = normalize_indicator_value(name, val)
                    tf_score += norm * weight; tf_w += weight
            if tf_w > 0:
                tf_final = tf_score / tf_w; tf_scores[tf] = tf_final
                mult = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                total_weighted += tf_final * mult; total_w += mult
        self.last_cycle_scores[symbol] = tf_scores
        if total_w <= 0: return 50 # Return neutral score if no data
        base_score = total_weighted / total_w
        if sector in self.best_sectors: base_score += 15
        elif sector in self.worst_sectors: base_score -= 15
        return base_score

    def replay_scan_cycle(self, current_time, preloaded_data, all_signals):
        print(f"\n{Colors.CYAN}--- Scanning at {current_time.strftime('%H:%M')} ---{Colors.RESET}")
        targets = sorted(set(stock for sector in self.best_sectors + self.worst_sectors for stock in SECTOR_STOCKS.get(sector, [])))
        
        for sym in targets:
            if sym not in preloaded_data: continue
            
            # Create data slices for each timeframe
            full_day_data = preloaded_data[sym]
            tfs = {}
            base_slice = self.get_replay_data_slice(full_day_data, current_time)
            if len(base_slice) < 100: continue

            for tf_min in [5, 15, 30, 60]:
                resampled = base_slice.resample(f'{tf_min}T').agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
                }).dropna()
                if len(resampled) > 26: tfs[tf_min] = add_buy_sell_delta_columns(resampled)

            if len(tfs) < 3: continue
            
            base_score = self.calculate_base_score(sym, tfs)
            passes_gates, checklist = self.check_option_buyer_gates(sym, tfs, next((s for s, lst in SECTOR_STOCKS.items() if sym in lst), "NA"), base_score, current_time)

            if passes_gates:
                # This is a valid signal
                sig = "EXPLOSIVE BUY" if base_score > 50 else "EXPLOSIVE SELL"
                price = tfs[5]["Close"].iloc[-1]
                signal_details = {"time": current_time.strftime('%H:%M'), "symbol": sym, "signal": sig, "score": base_score, "price": price}
                all_signals.append(signal_details)
                color = Colors.GREEN if "BUY" in sig else Colors.RED
                print(f"  {color}{Colors.BOLD}ALERT at {signal_details['time']}: {signal_details['symbol']} - {signal_details['signal']} @ {signal_details['price']:.2f} (Score: {signal_details['score']:.1f}){Colors.RESET}")
            
            elif DEBUG_REJECTIONS and abs(base_score - 50) > 15:
                # This is a potential signal that got rejected. Let's see why.
                reasons = {k: v for k, v in checklist.items() if not v}
                if reasons:
                     print(f"  {Colors.YELLOW}DEBUG [{sym}]: Rejected at {current_time.strftime('%H:%M')}. Score: {base_score:.1f}. Failed Gates: {reasons}{Colors.RESET}")

# =========================
# --- MAIN EXECUTION (UNCHANGED) ---
# =========================
def run_replay():
    scanner = Enhanced3SectorScanner()
    scan_interval = timedelta(minutes=5)
    replay_date = datetime.strptime(REPLAY_DATE_STR, "%Y-%m-%d").date()
    
    if not scanner.setup_historical_sector_performance(replay_date): return

    print("\nPre-loading 1-minute data for all target stocks...")
    all_targets = sorted(set(stock for sector in scanner.best_sectors + scanner.worst_sectors for stock in SECTOR_STOCKS.get(sector, [])))
    preloaded_data = {}
    start_fetch_dt = datetime.combine(replay_date - timedelta(days=20), time(9, 15))
    end_fetch_dt = datetime.combine(replay_date, time(15, 30))
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_sym = {executor.submit(tdhist.get_historic_data, sym, start_time=start_fetch_dt, end_time=end_fetch_dt, bar_size='1 min'): sym for sym in all_targets}
        for future in as_completed(future_to_sym):
            sym = future_to_sym[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    df.rename(columns={c: c.capitalize() for c in df.columns}, inplace=True)
                    df.set_index(pd.to_datetime(df.index), inplace=True)
                    preloaded_data[sym] = df
                    print(f"  {Colors.GREEN}Successfully loaded {sym}{Colors.RESET}")
                else: print(f"  {Colors.YELLOW}No data for {sym}{Colors.RESET}")
            except Exception as e: print(f"  {Colors.RED}Failed to load {sym}: {e}{Colors.RESET}")
    all_signals_found = []
    current_sim_time = datetime.combine(replay_date, REPLAY_START_TIME)
    end_sim_time = datetime.combine(replay_date, REPLAY_END_TIME)
    while current_sim_time <= end_sim_time:
        scanner.replay_scan_cycle(current_sim_time, preloaded_data, all_signals_found)
        current_sim_time += scan_interval
    print(f"\n\n{Colors.MAGENTA}{'='*30} REPLAY SUMMARY: {REPLAY_DATE_STR} {'='*30}{Colors.RESET}")
    if not all_signals_found:
        print(f"{Colors.YELLOW}No signals were generated during the replay.{Colors.RESET}")
    else:
        bulls = [s for s in all_signals_found if "BUY" in s['signal']]
        bears = [s for s in all_signals_found if "SELL" in s['signal']]
        print(f"\n{Colors.GREEN}{Colors.BOLD}--- {len(bulls)} Bullish Signals ---{Colors.RESET}")
        for s in bulls: print(f"  [{s['time']}] {s['symbol']:<12} {s['signal']:<16} @ ₹{s['price']:.2f} (Score: {s['score']:.1f})")
        print(f"\n{Colors.RED}{Colors.BOLD}--- {len(bears)} Bearish Signals ---{Colors.RESET}")
        for s in bears: print(f"  [{s['time']}] {s['symbol']:<12} {s['signal']:<16} @ ₹{s['price']:.2f} (Score: {s['score']:.1f})")
    print(f"{Colors.MAGENTA}{'='*78}{Colors.RESET}")

def main():
    if REPLAY_MODE: run_replay()
    else: print(f"{Colors.YELLOW}Live mode execution is not set up in this version. Set REPLAY_MODE = True.{Colors.RESET}")

if __name__ == "__main__":
    main()