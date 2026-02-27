import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import requests
import json
import time as time_module
import pytz
from logzero import logger
import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from truedata.history import TD_hist
import logging
import warnings
warnings.filterwarnings('ignore')

# --- TRUEDATA CONFIG ---
TD_USERNAME = "tdwsp751"
TD_PASSWORD = "raj@751"
try:
    td_hist = TD_hist(TD_USERNAME, TD_PASSWORD, log_level=logging.WARNING)
except Exception as e:
    print(f"Failed to initialize Truedata history client: {e}")
    td_hist = None

# --- MASTER STOCK LIST FOR SCANNING ---
ALL_NSE_STOCKS = [
    "CHOLAFIN", "GMRAIRPORT", "CYIENT", "HFCL", "AMBER", "KOTAKBANK", "PERSISTENT", "NHPC",
    "LT", "PAGEIND", "M&M", "RVNL", "SUPREMEIND", "BHARATFORG", "TATAPOWER", "KEI",
    "MARUTI", "POLYCAB", "PRESTIGE", "MOTHERSON", "OFSS", "NCC", "EICHERMOT", "BLUESTARCO",
    "BHARTIARTL", "PHOENIXLTD", "NBCC", "MUTHOOTFIN", "LTF", "MANAPPURAM", "TATASTEEL",
    "IIFL", "SUZLON", "AXISBANK", "VEDL", "UNOMINDA", "JSWENERGY", "TIINDIA", "CUMMINSIND",
    "CONCOR", "GRASIM", "COFORGE", "DLF", "UPL", "JSWSTEEL", "GAIL", "ASTRAL", "ETERNAL",
    "HAVELLS", "ONGC", "BOSCHLTD", "GODREJPROP", "NTPC", "ULTRACEMCO", "NYKAA", "HCLTECH",
    "UNITDSPR", "360ONE", "BEL", "BHEL", "TCS", "LODHA", "WIPRO", "SHREECEM", "DELHIVERY",
    "OIL", "DMART", "CAMS", "PPLPHARMA", "HAL", "ADANIPORTS", "SOLARINDS", "AMBUJACEM",
    "POLICYBZR", "SBIN", "TECHM", "KALYANKJIL", "KAYNES", "DRREDDY", "POWERGRID",
    "MAZDOCK", "DIXON", "DIVISLAB", "CIPLA", "IOC", "ADANIENT", "JINDALSTEL",
    "CROMPTON", "TVSMOTOR", "ICICIGI", "TITAN", "CANBK", "HDFCAMC", "SIEMENS",
    "EXIDEIND", "IRFC", "PETRONET", "HINDPETRO", "RECLTD", "BIOCON", "BAJAJ-AUTO",
    "LTIM", "DALBHARAT", "SUNPHARMA", "HEROMOTOCO", "HUDCO",  "APOLLOHOSP",
    "HINDZINC", "ASHOKLEY", "RELIANCE", "IGL", "TATAELXSI", "MPHASIS", "IREDA", "LUPIN",
    "INDUSINDBK", "HINDALCO", "PFC", "TRENT", "PAYTM", "IRCTC", "COALINDIA",
    "SAMMAANCAP", "PATANJALI", "ABB", "INFY", "OBEROIRLTY", "JUBLFOOD", "ICICIBANK", "BPCL",
    "ADANIGREEN", "IEX", "SRF", "CGPOWER", "ITC", "SAIL", "FEDERALBNK", "KFINTECH", "ALKEM",
    "TATAMOTORS", "JIOFIN", "BDL", "BAJAJFINSV", "HINDUNILVR","INOXWIND", "INDIGO", "HDFCBANK", "LAURUSLABS", "TORNTPHARM", "TATATECH", "PNB",
    "ADANIENSOL", "VOLTAS", "NMDC", "IDFCFIRSTB", "LICI", "NATIONALUM", "BRITANNIA",
    "APLAPOLLO", "SBILIFE", "ZYDUSLIFE", "ICICIPRULI", "ABCAPITAL",
    "CDSL", "KPITTECH", "PIIND", "LICHSGFIN", "AUBANK", "SONACOMS", "TORNTPOWER", "HDFCLIFE",
    "SBICARD", "BANKINDIA", "COLPAL", "INDUSTOWER", "NUVAMA", "MARICO", "PNBHOUSING", "PGEL",
    "MANKIND", "BAJFINANCE", "NESTLEIND", "NAUKRI", "AUROPHARMA", "ASIANPAINT", "SHRIRAMFIN",
    "TATACONSUM", "ANGELONE", "MFSL", "DABUR", "TITAGARH", "GLENMARK", "FORTIS", "BSE",
    "MAXHEALTH", "MCX", "INDHOTEL", "VBL", "SYNGENE", "GODREJCP"
]
# --- SECTOR TO STOCKS MAPPING (used for context, not for scoring in backtest) ---
SECTOR_STOCKS = {
    "Technology": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "CYIENT", "KPITTECH", "TATAELXSI","SONACOMS","KAYNES","OFSS"],
    "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR","BHARATFORG", "EICHERMOT", "ASHOKLEY", "BOSCHLTD","TIINDIA","MOTHERSON"],
    "Banking": ["HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","PNB","BANKBARODA","CANBK","IDFCFIRSTB","INDUSINDBK","AUBANK","FEDERALBNK"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA", "TORNTPHARM","GLENMARK","ALKEM","LAURUSLABS","BIOCON","ZYDUSLIFE","MANKIND","SYNGENE","PPLPHARMA"],
    "Energy": ["RELIANCE","NTPC","BPCL","IOC","ONGC","GAIL","HINDPETRO","ADANIGREEN","ADANIENSOL","JSWENERGY","COALINDIA","TATAPOWER","SUZLON","PETRONET","OIL","POWERGRID","NHPC","ADANIPORTS","ABB","SIEMENS","CGPOWER","INOXWIND"],
    "Metal": ["TATASTEEL", "JSWSTEEL", "SAIL", "JINDALSTEL", "HINDALCO", "NMDC"],
    "Consumer": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR","AMBER","UNITDSPR","GODREJCP","MARICO","COLPAL","UPL","VBL"],
    "Realty": ["DLF","LODHA","PRESTIGE","GODREJPROP","OBEROIRLTY","PHOENIXLTD","NCC","NBCC"],
} # Truncated for brevity, you can include the full list here

# --- COLOR CODES ---
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# --- RATE LIMITER (Good practice, even for backtesting) ---
api_limiter = threading.Semaphore(8) # Simplified for backtesting

# === INDICATOR ENGINES (UNCHANGED) ===
class EnhancedTechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        # This entire class is copied from your original script without changes
        indicators = {}
        if df is None or len(df) < 30:
            return indicators
        try:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            indicators['RSI'] = 100 - (100 / (1 + rs))
            ema12 = df['Close'].ewm(span=12, adjust=False).mean()
            ema26 = df['Close'].ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            indicators['MACD'] = macd_line - signal_line
            tr1 = df['High'] - df['Low']
            tr2 = (df['High'] - df['Close'].shift()).abs()
            tr3 = (df['Low'] - df['Close'].shift()).abs()
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            ATR = true_range.rolling(window=14).mean()
            indicators['ATR'] = ATR
            avg_volume_20 = df['Volume'].rolling(window=20).mean().replace(0, 1)
            volume_ratio = df['Volume'] / avg_volume_20
            indicators['Volume_Surge'] = np.clip((volume_ratio - 0.5) * 40, 0, 100)
            price_momentum = df['Close'].pct_change(periods=10) * 100
            avg_volume_10 = df['Volume'].rolling(window=10).mean().replace(0, 1)
            volume_momentum = (df['Volume'] / avg_volume_10 - 1) * 100
            momentum_score = (price_momentum * 0.7 + volume_momentum * 0.3)
            indicators['Momentum'] = 50 + np.clip(momentum_score * 1.5, -50, 50)
        except Exception as e:
            logger.error(f"Error in base indicators: {e}")
        return indicators

class OptionsReadyIndicators(EnhancedTechnicalIndicators):
    @staticmethod
    def calculate_all_indicators(
        df, squeeze_window=120, squeeze_quantile=0.1, require_two_bar_atr_accel=True
    ):
        # This entire class is copied from your original script without changes
        indicators = super(OptionsReadyIndicators, OptionsReadyIndicators).calculate_all_indicators(df)
        if not indicators:
            return indicators
        try:
            ma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            upper_band = ma20 + (std20 * 2.0)
            lower_band = ma20 - (std20 * 2.0)
            bbw = (upper_band - lower_band) / ma20.replace(0, np.nan)
            indicators['BBW'] = bbw
            bbw_quantile = bbw.rolling(window=squeeze_window, min_periods=20).quantile(
                squeeze_quantile, interpolation='linear'
            )
            indicators['BBW_q'] = bbw_quantile
            in_squeeze = bbw <= bbw_quantile
            indicators['in_squeeze'] = in_squeeze
            breaks_upper = df['Close'] > upper_band.shift(1)
            breaks_lower = df['Close'] < lower_band.shift(1)
            vol_surge_now = indicators.get('Volume_Surge', pd.Series(index=df.index, data=0))
            vol_confirm = vol_surge_now > 60.0
            squeeze_on = in_squeeze.shift(1).fillna(False)
            indicators['squeeze_fire_up'] = squeeze_on & breaks_upper & vol_confirm
            indicators['squeeze_fire_down'] = squeeze_on & breaks_lower & vol_confirm
            ATR = indicators.get('ATR', pd.Series(index=df.index, data=np.nan))
            ATR_EMA10 = ATR.ewm(span=10, adjust=False).mean()
            atr_gt = ATR > ATR_EMA10
            atr_rising = ATR > ATR.shift(1)
            if require_two_bar_atr_accel:
                atr_accel = atr_gt & atr_gt.shift(1).fillna(False) & atr_rising
            else:
                atr_accel = atr_gt & atr_rising
            indicators['ATR_EMA10'] = ATR_EMA10
            indicators['ATR_accel'] = atr_accel.astype(float)
        except Exception as e:
            logger.error(f"Error in options indicators: {e}")
        return indicators

# === SCANNER CLASS MODIFIED FOR BACKTESTING ===
class OptionsBreakoutScanner:
    def __init__(self):
        # Simplified for backtesting
        self.min_avg_vol_5m = 50000
        self.min_avg_vol_15m = 30000
        self.use_mtf_confirmation = True
        # Sector analysis is disabled for backtesting as we don't have historical sector data
        self.best_sectors = []
        self.worst_sectors = []

    def normalize_data(self, df, symbol):
        # Same normalization logic
        try:
            if df is None or df.empty: return None
            df_clean = df.copy()
            col_lookup = {col.lower(): col for col in df_clean.columns}
            date_col = col_lookup.get('timestamp') or col_lookup.get('time')
            final_df = pd.DataFrame({
                'Date': pd.to_datetime(df_clean[date_col]),
                'Open': pd.to_numeric(df_clean[col_lookup.get('open')], errors='coerce'),
                'High': pd.to_numeric(df_clean[col_lookup.get('high')], errors='coerce'),
                'Low': pd.to_numeric(df_clean[col_lookup.get('low')], errors='coerce'),
                'Close': pd.to_numeric(df_clean[col_lookup.get('close')], errors='coerce'),
                'Volume': pd.to_numeric(df_clean[col_lookup.get('volume') or col_lookup.get('vol')], errors='coerce')
            })
            final_df.set_index('Date', inplace=True)
            return final_df.dropna().sort_index()
        except Exception as e:
            logger.error(f"Normalize error for {symbol}: {e}")
            return None

    def fetch_backtest_data(self, symbol, end_date):
        if not td_hist:
            logger.error("Truedata client not initialized.")
            return None
        try:
            with api_limiter:
                # Fetch 15 days of data ending on the backtest date for accurate indicator calculation
                raw_df = td_hist.get_historic_data(symbol, duration='15 D', bar_size='1 min', end_date=end_date)
            
            if raw_df is None or raw_df.empty:
                return None
            
            df_1m = self.normalize_data(raw_df, symbol)
            return df_1m
        except Exception as e:
            logger.error(f"Data fetch error for {symbol} on {end_date}: {e}")
            return None

    def resample_timeframes(self, df_1m_slice):
        # Same resampling logic
        ohlc_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
        resampled_data = {}
        for tf in [5, 15, 30, 60]:
            df_resampled = df_1m_slice.resample(f'{tf}T', label='right', closed='right').agg(ohlc_dict).dropna()
            if len(df_resampled) >= 30: # Ensure enough data for indicators
                resampled_data[tf] = df_resampled
        return resampled_data

    # All helper methods like _passes_liquidity, _mtf_trend_bias, and calculate_options_signals
    # are copied from your original script without changes. They are self-contained.
    def _passes_liquidity(self, timeframes_data):
        df5 = timeframes_data.get(5, None)
        df15 = timeframes_data.get(15, None)
        if df5 is not None and not df5.empty:
            avg5 = df5['Volume'].tail(20).mean()
            if not np.isnan(avg5) and avg5 < self.min_avg_vol_5m: return False
        if df15 is not None and not df15.empty:
            avg15 = df15['Volume'].tail(20).mean()
            if not np.isnan(avg15) and avg15 < self.min_avg_vol_15m: return False
        return True

    def _mtf_trend_bias(self, timeframes_data):
        df30 = timeframes_data.get(30, None)
        if df30 is None or len(df30) < 30: return 0
        inds = OptionsReadyIndicators.calculate_all_indicators(df30)
        if not inds: return 0
        macd = inds.get('MACD', None)
        ma20 = df30['Close'].rolling(20).mean()
        bias = 0
        if macd is not None and not macd.empty and pd.notna(macd.iloc[-1]):
            bias += 1 if macd.iloc[-1] > 0 else -1
        if not ma20.empty and pd.notna(ma20.iloc[-1]):
            bias += 1 if df30['Close'].iloc[-1] > ma20.iloc[-1] else -1
        return bias

    def calculate_options_signals(self, symbol, timeframes_data):
        # The core signal logic remains UNCHANGED to ensure the backtest is valid.
        final_score = 50.0
        strongest_signal = "Neutral"
        squeeze_status = "No Squeeze"
        try:
            if not self._passes_liquidity(timeframes_data):
                return 'Neutral', 50.0, "Liquidity Fail"
            mtf_bias = self._mtf_trend_bias(timeframes_data) if self.use_mtf_confirmation else 0

            for tf in [15, 30, 60]:
                if tf not in timeframes_data: continue
                indicators = OptionsReadyIndicators.calculate_all_indicators(timeframes_data[tf])
                if not indicators: continue
                latest = {name: ind.iloc[-1] for name, ind in indicators.items() if hasattr(ind, 'iloc') and len(ind) > 0 and pd.notna(ind.iloc[-1])}
                
                is_firing_up = bool(latest.get('squeeze_fire_up', False))
                is_firing_down = bool(latest.get('squeeze_fire_down', False))
                
                # SECTOR BOOST IS 0 IN BACKTESTING
                sector_boost = 0

                if is_firing_up:
                    score = 80 + sector_boost # Base score
                    if float(latest.get('Momentum', 50)) > 65: score += 10
                    if float(latest.get('Volume_Surge', 0)) > 60: score += 15
                    if float(latest.get('ATR_accel', 0)) > 0: score += 10
                    if self.use_mtf_confirmation and mtf_bias < 0: score -= 10 # Penalty
                    if score > final_score:
                        final_score, strongest_signal, squeeze_status = score, "Explosive Buy", f"{tf}m Squeeze FIRE UP"
                
                elif is_firing_down:
                    score = 20 + sector_boost # Base score
                    if float(latest.get('Momentum', 50)) < 35: score -= 10
                    if float(latest.get('Volume_Surge', 0)) > 60: score -= 15
                    if float(latest.get('ATR_accel', 0)) > 0: score -= 10
                    if self.use_mtf_confirmation and mtf_bias > 0: score += 10 # Penalty
                    if score < final_score:
                        final_score, strongest_signal, squeeze_status = score, "Explosive Sell", f"{tf}m Squeeze FIRE DOWN"

            return strongest_signal, float(np.clip(final_score, 0, 100)), squeeze_status
        except Exception as e:
            logger.error(f"Signal calc error for {symbol}: {e}")
            return 'Neutral', 50.0, "Error"

    def run_backtest(self, backtest_date_str):
        print(f"\n{Colors.BOLD}🚀 RUNNING BACKTEST FOR DATE: {backtest_date_str}{Colors.RESET}")
        print(f"🔩 NOTE: Sectoral strength analysis is DISABLED for historical backtesting.")
        print("-" * 60)
        
        backtest_date = datetime.strptime(backtest_date_str, '%Y-%m-%d').date()
        all_signals = []

        for i, symbol in enumerate(ALL_NSE_STOCKS):
            print(f"Processing {symbol} ({i+1}/{len(ALL_NSE_STOCKS)})...", end='\r')
            
            # 1. Fetch historical data including the lookback period
            full_hist_df = self.fetch_backtest_data(symbol, backtest_date_str)
            if full_hist_df is None or full_hist_df.empty:
                continue

            # 2. Isolate the 1-minute data for the specific backtest day
            day_df_1m = full_hist_df[full_hist_df.index.date == backtest_date]
            if day_df_1m.empty:
                continue
            
            # Isolate the data *before* the backtest day for the initial lookback
            lookback_df = full_hist_df[full_hist_df.index.date < backtest_date]

            # 3. Simulate the day by iterating through each 1-minute candle
            for i in range(len(day_df_1m)):
                # Create a point-in-time view of the data
                current_minute_data = day_df_1m.iloc[:i+1]
                point_in_time_df = pd.concat([lookback_df, current_minute_data])
                
                # Resample based on data up to the current minute
                timeframes_data = self.resample_timeframes(point_in_time_df)
                
                if not timeframes_data:
                    continue

                # Check for signals using the exact same logic
                signal, score, squeeze = self.calculate_options_signals(symbol, timeframes_data)

                # If a strong signal is found, log it
                if "Explosive" in signal:
                    signal_time = day_df_1m.index[i]
                    
                    # Avoid duplicate signals for the same breakout event
                    is_duplicate = False
                    for s in all_signals:
                        if s['symbol'] == symbol and s['signal'] == signal and (signal_time - s['time']).total_seconds() < 900: # 15 min cooldown
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        found_signal = {
                            'time': signal_time,
                            'symbol': symbol,
                            'signal': signal,
                            'score': score,
                            'squeeze': squeeze
                        }
                        all_signals.append(found_signal)

        # 4. Display all found signals at the end
        print("\n" + "=" * 60)
        print(f"✅ Backtest Complete. Found {len(all_signals)} signals.")
        print("=" * 60)

        if not all_signals:
            print(f"{Colors.YELLOW}No explosive breakout signals found for {backtest_date_str}.{Colors.RESET}")
            return

        # Sort signals by time
        all_signals.sort(key=lambda x: x['time'])

        for s in all_signals:
            color = Colors.GREEN if 'Buy' in s['signal'] else Colors.RED
            time_str = s['time'].strftime('%H:%M')
            print(f"[{time_str}] {color}{s['symbol']:<12} | {s['signal']:<15} | Score: {s['score']:.1f} | {s['squeeze']}{Colors.RESET}")


if __name__ == "__main__":
    if td_hist is None:
        print(f"{Colors.RED}Could not start backtest because Truedata client failed to initialize.{Colors.RESET}")
    else:
        # --- CONFIGURATION FOR BACKTEST ---
        # Automatically set to yesterday's date. You can also hardcode a date like "2024-05-20"
        YESTERDAY = (datetime.now() - timedelta(days=1))
        BACKTEST_DATE = YESTERDAY.strftime('%Y-%m-%d')

        scanner = OptionsBreakoutScanner()
        scanner.run_backtest(BACKTEST_DATE)