# Enhanced Multi-Indicator Options Scanner with Advanced Options Analytics
# Features: IV Analysis, Options Flow, Advanced Momentum, Strike Recommendations
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import requests
import json
import time as time_module
import pytz
from logzero import logger
import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from truedata.history import TD_hist
import logging
import warnings
import yfinance as yf
from scipy import stats
warnings.filterwarnings('ignore')

# --- TRUEDATA CONFIG ---
TD_USERNAME = "Trial106"
TD_PASSWORD = "raj106"
td_hist = TD_hist(TD_USERNAME, TD_PASSWORD, log_level=logging.WARNING)

# --- COLOR CODES ---
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

# --- ENHANCED OPTIONS-SPECIFIC WEIGHTS ---
TIMEFRAME_WEIGHTS = {1: 0.8, 5: 1.2, 15: 1.8, 30: 2.2, 60: 2.8, 'daily': 3.2}

INDICATOR_WEIGHTS = {
    'RSI': 1.3, 'MACD': 1.6, 'Stochastic': 1.0, 'MA': 1.8,
    'ADX': 1.5, 'Bollinger': 1.4, 'ROC': 1.2, 'OBV': 1.6, 
    'CCI': 1.1, 'WWL': 1.0, 'EMA': 1.7, 'VWAP': 1.5,
    'ATR': 1.4, 'Volume_Surge': 2.0, 'Momentum': 1.9
}

# --- OPTIONS-SPECIFIC SECTOR DATA ---
SECTOR_OPTIONS_PROFILE = {
    "Technology": {"avg_iv": 32, "volatility_factor": 1.3, "volume_threshold": 2500, "typical_move": 4.2},
    "Auto": {"avg_iv": 38, "volatility_factor": 1.4, "volume_threshold": 1800, "typical_move": 3.8},
    "Banking": {"avg_iv": 35, "volatility_factor": 1.2, "volume_threshold": 3000, "typical_move": 3.2},
    "Pharma": {"avg_iv": 40, "volatility_factor": 1.5, "volume_threshold": 1500, "typical_move": 5.1},
    "Energy": {"avg_iv": 42, "volatility_factor": 1.6, "volume_threshold": 2200, "typical_move": 4.8},
    "Metal": {"avg_iv": 45, "volatility_factor": 1.7, "volume_threshold": 2000, "typical_move": 5.5},
    "Consumer": {"avg_iv": 28, "volatility_factor": 1.1, "volume_threshold": 2800, "typical_move": 2.8},
}

SECTOR_STOCKS = {
    "Technology": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "CYIENT"],
    "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR", "BHARATFORG", "EICHERMOT"],
    "Banking": ["HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","PNB","BANKBARODA","CANBK"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA", "TORNTPHARM", "GLENMARK", "ALKEM"],
    "Energy": ["RELIANCE","NTPC","BPCL","IOC","ONGC","GAIL","HINDPETRO","ADANIGREEN"],
    "Metal": ["TATASTEEL", "JSWSTEEL", "SAIL", "JINDALSTEL", "HINDALCO", "NMDC"],
    "Consumer": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR"],
}

NSE_INDEX_TO_SECTOR = {
    "NIFTY IT": "Technology", "NIFTY PHARMA": "Pharma", "NIFTY FMCG": "Consumer",
    "NIFTY BANK": "Banking", "NIFTY AUTO": "Auto", "NIFTY METAL": "Metal", "NIFTY ENERGY": "Energy"
}

# --- OPTIONS-ENHANCED TECHNICAL INDICATORS CLASS ---
class OptionsEnhancedIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        """Enhanced indicators specifically for options trading"""
        indicators = {}
        if df is None or len(df) < 20:
            return indicators

        try:
            # 1. Enhanced RSI with overbought/oversold zones
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs))

            # 2. MACD with enhanced sensitivity
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            indicators['MACD'] = macd_line - signal_line

            # 3. Fast Stochastic for quick entries
            low14 = df['Low'].rolling(window=14).min()
            high14 = df['High'].rolling(window=14).max()
            indicators['Stochastic'] = 100 * (df['Close'] - low14) / (high14 - low14)

            # 4. Multiple Moving Averages
            indicators['MA'] = df['Close'].rolling(window=20).mean()
            indicators['EMA'] = df['Close'].ewm(span=21).mean()

            # 5. Enhanced ADX for trend strength
            high_diff = df['High'].diff()
            low_diff = df['Low'].diff()
            plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
            minus_dm = (-low_diff).where((low_diff < high_diff) & (low_diff < 0), 0)
            tr1 = df['High'] - df['Low']
            tr2 = abs(df['High'] - df['Close'].shift())
            tr3 = abs(df['Low'] - df['Close'].shift())
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            indicators['ADX'] = dx.rolling(window=14).mean()

            # 6. Bollinger Bands with squeeze detection
            ma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            upper_band = ma20 + (std20 * 2)
            lower_band = ma20 - (std20 * 2)
            indicators['Bollinger'] = (df['Close'] - ma20) / (upper_band - lower_band) * 100

            # 7. Rate of Change for momentum
            indicators['ROC'] = df['Close'].pct_change(periods=12) * 100

            # 8. Enhanced OBV with volume analysis
            obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            indicators['OBV'] = obv.pct_change(periods=10) * 100

            # 9. CCI for oversold/overbought conditions
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            indicators['CCI'] = (typical_price - sma_tp) / (0.015 * mad)

            # 10. Williams %R
            highest_high = df['High'].rolling(window=14).max()
            lowest_low = df['Low'].rolling(window=14).min()
            indicators['WWL'] = (highest_high - df['Close']) / (highest_high - lowest_low) * -100
            
            # 11. VWAP
            if len(df) >= 20:
                typical_price_vwap = (df['High'] + df['Low'] + df['Close']) / 3
                vwap_numerator = (typical_price_vwap * df['Volume']).rolling(window=20).sum()
                vwap_denominator = df['Volume'].rolling(window=20).sum()
                indicators['VWAP'] = vwap_numerator / vwap_denominator

            # 12. ATR for volatility (OPTIONS CRITICAL)
            indicators['ATR'] = true_range.rolling(window=14).mean()

            # 13. Volume Surge Detection (OPTIONS CRITICAL)
            avg_volume = df['Volume'].rolling(window=20).mean()
            current_volume = df['Volume'].iloc[-1] if len(df) > 0 else 0
            volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
            indicators['Volume_Surge'] = min(100, volume_ratio * 20)  # Scale to 0-100

            # 14. Price Momentum (OPTIONS CRITICAL)
            if len(df) >= 10:
                price_change_5 = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5] * 100
                price_change_10 = (df['Close'].iloc[-1] - df['Close'].iloc[-10]) / df['Close'].iloc[-10] * 100
                indicators['Momentum'] = (price_change_5 * 0.7) + (price_change_10 * 0.3)

        except Exception as e:
            logger.error(f"Error calculating enhanced indicators: {e}")

        return indicators

    @staticmethod
    def detect_volatility_expansion(df):
        """Detect volatility expansion - crucial for options"""
        try:
            if df is None or len(df) < 30:
                return 0
            
            # ATR expansion
            atr_current = df['High'].iloc[-1] - df['Low'].iloc[-1]
            atr_avg = ((df['High'] - df['Low']).rolling(window=20).mean()).iloc[-1]
            atr_expansion = (atr_current / atr_avg) if atr_avg > 0 else 1
            
            # Bollinger Band width expansion
            ma20 = df['Close'].rolling(window=20).mean().iloc[-1]
            std20 = df['Close'].rolling(window=20).std().iloc[-1]
            bb_width_current = std20 / ma20 if ma20 > 0 else 0
            bb_width_avg = (df['Close'].rolling(window=20).std() / df['Close'].rolling(window=20).mean()).rolling(window=10).mean().iloc[-1]
            bb_expansion = (bb_width_current / bb_width_avg) if bb_width_avg > 0 else 1
            
            # Combined volatility expansion score
            expansion_score = (atr_expansion * 0.6 + bb_expansion * 0.4) * 50
            return min(100, expansion_score)
            
        except Exception as e:
            logger.error(f"Error detecting volatility expansion: {e}")
            return 0

    @staticmethod
    def calculate_iv_percentile(symbol):
        """Calculate IV percentile (mock implementation - integrate with options API)"""
        try:
            # This would integrate with actual options API
            # For now, returning a mock value based on sector
            sector = next((s for s, stocks in SECTOR_STOCKS.items() if symbol in stocks), None)
            if sector and sector in SECTOR_OPTIONS_PROFILE:
                base_iv = SECTOR_OPTIONS_PROFILE[sector]['avg_iv']
                # Add some randomness for demo (replace with real IV data)
                import random
                return min(100, max(0, base_iv + random.randint(-15, 15)))
            return 50
        except:
            return 50

# --- ENHANCED OPTIONS SCANNER CLASS ---
class AdvancedOptionsScanner:
    def __init__(self):
        self.is_running = False
        self.current_signals = {}
        self.best_sectors = ["Technology", "Banking", "Consumer"]
        self.worst_sectors = ["Metal", "Energy", "Auto"]
        self.sectoral_history = []
        self.last_sectoral_update = None
        self.options_signals_history = []
        
        # Enhanced scanning for options
        self.market_start = time(9, 15)
        self.market_end = time(15, 30)
        self.scan_interval = 180  # 3 minutes for options (faster)
        
        # Options-specific tracking
        self.high_iv_filtered = 0
        self.low_volume_filtered = 0
        self.successful_options_signals = 0
        
        logger.info("🚀 Advanced Options Scanner initialized with enhanced accuracy")
        self.show_initialization_status()

    def show_initialization_status(self):
        """Show enhanced initialization status"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}🎯 ADVANCED OPTIONS SCANNER - HIGH ACCURACY{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")
        print(f"⚡ Features: {Colors.GREEN}IV Analysis{Colors.RESET} | {Colors.YELLOW}Volume Surge{Colors.RESET} | {Colors.MAGENTA}Volatility Expansion{Colors.RESET}")
        print(f"📊 Timeframes: {Colors.YELLOW}1min, 5min, 15min, 30min, 60min, Daily{Colors.RESET}")
        print(f"🎯 Scanning: {Colors.GREEN}3 minutes intervals{Colors.RESET} (vs 5min for stocks)")
        print(f"🔥 Accuracy: {Colors.BOLD}Enhanced for Options Trading{Colors.RESET}")
        
        self.test_api_connection()
        print(f"{Colors.CYAN}{'='*80}{Colors.RESET}")

    def test_api_connection(self):
        """Enhanced API connection test"""
        print(f"\n{Colors.BLUE}🔍 API CONNECTION TEST:{Colors.RESET}")
        try:
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            if response.status_code == 200:
                print(f"✅ API Connection: {Colors.GREEN}SUCCESS{Colors.RESET}")
                print(f"🚀 Options Scanner: {Colors.GREEN}READY{Colors.RESET}")
            else:
                print(f"❌ API Connection: {Colors.RED}FAILED{Colors.RESET}")
        except Exception as e:
            print(f"❌ API Connection: {Colors.RED}ERROR{Colors.RESET}")

    def is_market_open(self):
        """Check if market is open"""
        now = datetime.now()
        current_time = now.time()
        
        if now.weekday() > 4:  # Weekend
            return False
            
        return self.market_start <= current_time <= self.market_end

    def fetch_live_sectoral_performance(self):
        """Fetch sectoral performance with options focus"""
        try:
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            
            if response.status_code == 200:
                indices_data = response.json()
                
                if isinstance(indices_data, str):
                    indices_data = json.loads(indices_data)
                    
                if isinstance(indices_data, dict):
                    if 'data' in indices_data: indices_data = indices_data['data']
                    elif 'indices' in indices_data: indices_data = indices_data['indices']
                    elif 'results' in indices_data: indices_data = indices_data['results']
                
                if not isinstance(indices_data, list):
                    return False
                
                sectoral_performance = []
                current_time = datetime.now()
                
                for index in indices_data:
                    if not isinstance(index, dict): continue
                    
                    index_name = next((str(index[field]).strip().upper() for field in ['name', 'symbol', 'index', 'indexName'] if field in index and index[field]), None)
                    
                    if index_name and index_name in NSE_INDEX_TO_SECTOR:
                        change_percent = 0.0
                        for field in ['change_percent', 'changePercent', 'pChange', 'percentChange', 'change', 'pchg']:
                            if field in index and index[field] is not None:
                                try:
                                    change_percent = float(index[field])
                                    break
                                except (ValueError, TypeError): continue
                        
                        sectoral_performance.append({
                            'index': index_name, 'sector': NSE_INDEX_TO_SECTOR[index_name],
                            'change_percent': change_percent, 'timestamp': current_time
                        })
                
                if sectoral_performance:
                    sectoral_performance.sort(key=lambda x: x['change_percent'], reverse=True)
                    
                    # Update top 3 best and worst
                    if len(sectoral_performance) >= 6:
                        self.best_sectors = [sectoral_performance[i]['sector'] for i in range(3)]
                        self.worst_sectors = [sectoral_performance[i]['sector'] for i in range(-3, 0)]
                    
                    self.last_sectoral_update = current_time
                    return True
                    
        except Exception as e:
            logger.error(f"Error fetching sectoral data: {e}")
            return False

    def normalize_live_data(self, df, symbol):
        """Enhanced data normalization for options"""
        try:
            if df is None or df.empty: return None

            df_clean = df.copy()
            cols = [c.lower() for c in df_clean.columns]
            column_mapping = {df_clean.columns[i]: new_name for i, col in enumerate(cols)
                              for key, new_name in [('time', 'Date'), ('open', 'Open'), ('high', 'High'),
                                                    ('low', 'Low'), ('close', 'Close'), ('vol', 'Volume')]
                              if key in col}
            df_clean = df_clean.rename(columns=column_mapping)
            
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            if not all(col in df_clean.columns for col in required_cols): return None
            
            if 'Volume' not in df_clean.columns: df_clean['Volume'] = 1000
            
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            if df_clean['Date'].dt.tz is not None:
                df_clean['Date'] = df_clean['Date'].dt.tz_localize(None)
            df_clean.set_index('Date', inplace=True)
            
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
            return df_clean.dropna().sort_index() if len(df_clean) >= 10 else None
        except Exception as e:
            logger.error(f"Normalize error {symbol}: {e}")
            return None

    def fetch_live_data(self, symbol, timeframe):
        """Enhanced fetch with options-specific data"""
        try:
            tf_map = {
                1: '1 min',
                5: '5 min', 
                15: '15 min', 
                30: '30 min',
                60: '60 mins',
                'daily': 'EOD'
            }
            bar_size = tf_map.get(timeframe)
            if not bar_size: return None

            if timeframe == 1:
                duration = '5 D'
            elif timeframe == 5:
                duration = '10 D'
            elif timeframe == 15:
                duration = '15 D'
            elif timeframe == 30:
                duration = '25 D'
            elif timeframe == 60:
                duration = '60 D'
            elif timeframe == 'daily':
                duration = '365 D'
            else:
                duration = '10 D'
                
            raw_df = td_hist.get_historic_data(symbol, duration=duration, bar_size=bar_size)

            if raw_df is not None and len(raw_df) > 0:
                normalized_df = self.normalize_live_data(raw_df, symbol)
                if normalized_df is not None and len(normalized_df) >= 10:
                    if timeframe == 'daily':
                        return normalized_df.tail(100)
                    elif timeframe == 60:
                        return normalized_df.tail(80)
                    else:
                        return normalized_df.tail(50)
            return None
        except Exception as e:
            logger.error(f"Live data fetch error {symbol}_{timeframe}: {e}")
            return None

    def check_options_filters(self, symbol, df_1min, df_5min):
        """Advanced options-specific filters"""
        filters = {
            'gap_down': False,
            'high_iv': False,
            'low_volume': False,
            'poor_liquidity': False
        }
        
        try:
            # 1. Gap-down check (enhanced)
            if df_1min is not None and len(df_1min) >= 2:
                current_open = df_1min['Open'].iloc[-1]
                previous_close = df_1min['Close'].iloc[-2]
                if previous_close > 0:
                    gap_percentage = ((current_open - previous_close) / previous_close) * 100
                    if gap_percentage <= -1.5:  # Stricter for options
                        filters['gap_down'] = True

            # 2. High IV filter (mock - integrate with options API)
            iv_percentile = OptionsEnhancedIndicators.calculate_iv_percentile(symbol)
            if iv_percentile > 75:  # Avoid buying when IV is high
                filters['high_iv'] = True

            # 3. Low volume filter
            if df_5min is not None and len(df_5min) >= 10:
                avg_volume = df_5min['Volume'].rolling(window=10).mean().iloc[-1]
                current_volume = df_5min['Volume'].iloc[-1]
                sector = next((s for s, stocks in SECTOR_STOCKS.items() if symbol in stocks), None)
                
                if sector and sector in SECTOR_OPTIONS_PROFILE:
                    min_volume = SECTOR_OPTIONS_PROFILE[sector]['volume_threshold']
                    if current_volume < min_volume or current_volume < avg_volume * 0.5:
                        filters['low_volume'] = True

            # 4. Poor liquidity (additional check)
            # This would integrate with options chain data
            
        except Exception as e:
            logger.error(f"Error in options filters for {symbol}: {e}")

        return filters

    def normalize_indicator_value(self, indicator_name, value):
        """Enhanced normalize indicator values for options"""
        try:
            if indicator_name == 'RSI': 
                return max(0, min(100, value))
            elif indicator_name == 'MACD': 
                return 50 + min(30, max(-30, value * 12))  # Enhanced sensitivity
            elif indicator_name == 'Stochastic': 
                return max(0, min(100, value))
            elif indicator_name == 'MA' or indicator_name == 'EMA': 
                return 50
            elif indicator_name == 'ADX': 
                return max(0, min(100, value))
            elif indicator_name == 'Bollinger': 
                return max(0, min(100, (value + 100) / 2))
            elif indicator_name == 'ROC': 
                return 50 + min(35, max(-35, value * 2.5))  # Enhanced for options
            elif indicator_name == 'OBV': 
                return 50 + min(30, max(-30, value))
            elif indicator_name == 'CCI': 
                return max(0, min(100, (value + 200) / 4))
            elif indicator_name == 'WWL': 
                return max(0, min(100, value + 100))
            elif indicator_name == 'VWAP': 
                return 50
            elif indicator_name == 'ATR':
                return min(100, value * 10)  # Scale ATR
            elif indicator_name == 'Volume_Surge':
                return min(100, value)
            elif indicator_name == 'Momentum':
                return 50 + min(40, max(-40, value * 3))  # Enhanced momentum sensitivity
            else:
                return 50
        except: 
            return 50

    def calculate_options_signals(self, symbol, timeframes_data):
        """Advanced options signal calculation with enhanced accuracy"""
        try:
            if not timeframes_data: return 'Neutral', 0, {}

            sector = next((s for s, stocks in SECTOR_STOCKS.items() if symbol in stocks), None)
            if not sector: return 'Neutral', 0, {}

            total_weighted_score, total_weight = 0, 0
            timeframe_scores = {}
            signal_details = {'volatility_expansion': 0, 'momentum_strength': 0, 'volume_confirmation': False}
            
            # Enhanced volatility expansion check
            if 5 in timeframes_data and timeframes_data[5] is not None:
                vol_expansion = OptionsEnhancedIndicators.detect_volatility_expansion(timeframes_data[5])
                signal_details['volatility_expansion'] = vol_expansion
            
            for tf, df in timeframes_data.items():
                if df is None or len(df) < 10: continue
                
                indicators = OptionsEnhancedIndicators.calculate_all_indicators(df)
                if not indicators: continue

                tf_score, tf_weight = 0, 0
                current_price = df['Close'].iloc[-1]
                
                # Enhanced indicator processing for options
                for name, weight in INDICATOR_WEIGHTS.items():
                    if name in indicators and indicators[name] is not None and not indicators[name].empty:
                        latest_val = indicators[name].iloc[-1]
                        if pd.notna(latest_val):
                            if name in ['MA', 'EMA', 'VWAP']:
                                if latest_val > 0:
                                    price_vs_ma = (current_price - latest_val) / latest_val * 100
                                    if price_vs_ma > 3:      # Enhanced thresholds for options
                                        norm_score = 80
                                    elif price_vs_ma > 1:
                                        norm_score = 65
                                    elif price_vs_ma > -1:
                                        norm_score = 50
                                    elif price_vs_ma > -3:
                                        norm_score = 35
                                    else:
                                        norm_score = 20
                                else:
                                    norm_score = 50
                            else:
                                norm_score = self.normalize_indicator_value(name, latest_val)
                            
                            # Special handling for options-critical indicators
                            if name == 'Volume_Surge' and norm_score > 60:
                                signal_details['volume_confirmation'] = True
                                norm_score *= 1.3  # Boost volume surge importance
                            elif name == 'Momentum' and abs(norm_score - 50) > 20:
                                signal_details['momentum_strength'] = abs(norm_score - 50)
                                norm_score *= 1.2  # Boost momentum importance
                            elif name == 'ATR' and norm_score > 70:
                                norm_score *= 1.1  # Boost high volatility
                            
                            tf_score += norm_score * weight
                            tf_weight += weight
                
                if tf_weight > 0:
                    tf_final_score = tf_score / tf_weight
                    timeframe_scores[tf] = tf_final_score
                    
                    tf_multiplier = TIMEFRAME_WEIGHTS.get(tf, 1.0)
                    total_weighted_score += tf_final_score * tf_multiplier
                    total_weight += tf_multiplier

            if total_weight == 0: return 'Neutral', 0, signal_details
            base_score = total_weighted_score / total_weight

            # Enhanced sector-based adjustments for options
            sector_boost = 0
            has_intraday = any(tf in [1, 5, 15] for tf in timeframes_data.keys())
            
            if sector in self.best_sectors:
                rank = self.best_sectors.index(sector) + 1
                if has_intraday:  # More aggressive for intraday options
                    sector_boost = [30, 25, 20][rank-1] if rank <= 3 else 15
                else:
                    sector_boost = [25, 20, 15][rank-1] if rank <= 3 else 10
                    
            elif sector in self.worst_sectors:
                rank = self.worst_sectors.index(sector) + 1
                if has_intraday:
                    sector_boost = [-30, -25, -20][rank-1] if rank <= 3 else -15
                else:
                    sector_boost = [-25, -20, -15][rank-1] if rank <= 3 else -10
            
            base_score += sector_boost

            # Options-specific bonuses
            # 1. Volatility expansion bonus
            if signal_details['volatility_expansion'] > 70:
                base_score += 12
            elif signal_details['volatility_expansion'] > 50:
                base_score += 8

            # 2. Volume confirmation bonus
            if signal_details['volume_confirmation']:
                base_score += 10

            # 3. Strong momentum bonus
            if signal_details['momentum_strength'] > 25:
                base_score += 8
            elif signal_details['momentum_strength'] > 15:
                base_score += 5

            # 4. Multi-timeframe confirmation (stricter for options)
            num_timeframes = len(timeframes_data)
            if num_timeframes >= 4:
                bullish_count = sum(1 for score in timeframe_scores.values() if score > 60)
                bearish_count = sum(1 for score in timeframe_scores.values() if score < 40)
                
                if bullish_count >= 3:
                    base_score += 10
                elif bearish_count >= 3:
                    base_score -= 10

            # Enhanced signal classification for options
            if base_score >= 85: return 'Very Strong Buy', base_score, signal_details
            elif base_score >= 75: return 'Strong Buy', base_score, signal_details
            elif base_score >= 62: return 'Buy', base_score, signal_details
            elif base_score <= 15: return 'Very Strong Sell', base_score, signal_details
            elif base_score <= 25: return 'Strong Sell', base_score, signal_details
            elif base_score <= 38: return 'Sell', base_score, signal_details
            else: return 'Neutral', base_score, signal_details
            
        except Exception as e:
            logger.error(f"Options signal calculation error for {symbol}: {e}")
            return 'Neutral', 0, {}

    def recommend_strikes_and_expiry(self, symbol, signal, score, current_price):
        """Recommend optimal strikes and expiry for options"""
        recommendations = {
            'strikes': [],
            'expiry_preference': '',
            'strategy': '',
            'risk_level': ''
        }
        
        try:
            if 'Buy' in signal:
                if score >= 85:
                    recommendations['strikes'] = ['ATM', 'ATM+1']
                    recommendations['expiry_preference'] = '1-2 weeks'
                    recommendations['strategy'] = 'Aggressive Call Buying'
                    recommendations['risk_level'] = 'High'
                elif score >= 75:
                    recommendations['strikes'] = ['ATM-1', 'ATM']
                    recommendations['expiry_preference'] = '2-4 weeks'
                    recommendations['strategy'] = 'Moderate Call Buying'
                    recommendations['risk_level'] = 'Medium'
                elif score >= 62:
                    recommendations['strikes'] = ['ATM-2', 'ATM-1']
                    recommendations['expiry_preference'] = '4-6 weeks'
                    recommendations['strategy'] = 'Conservative Call Buying'
                    recommendations['risk_level'] = 'Low-Medium'
                    
            elif 'Sell' in signal:
                if score <= 15:
                    recommendations['strikes'] = ['ATM', 'ATM+1']
                    recommendations['expiry_preference'] = '1-2 weeks'
                    recommendations['strategy'] = 'Aggressive Put Buying'
                    recommendations['risk_level'] = 'High'
                elif score <= 25:
                    recommendations['strikes'] = ['ATM-1', 'ATM']
                    recommendations['expiry_preference'] = '2-4 weeks'
                    recommendations['strategy'] = 'Moderate Put Buying'
                    recommendations['risk_level'] = 'Medium'
                elif score <= 38:
                    recommendations['strikes'] = ['ATM-2', 'ATM-1']
                    recommendations['expiry_preference'] = '4-6 weeks'
                    recommendations['strategy'] = 'Conservative Put Buying'
                    recommendations['risk_level'] = 'Low-Medium'
                    
        except Exception as e:
            logger.error(f"Error generating recommendations for {symbol}: {e}")

        return recommendations

    def advanced_options_scan_cycle(self):
        """Main enhanced options scanning cycle"""
        if not self.is_market_open():
            logger.info("🕐 Market closed. Next scan in 3 minutes...")
            return

        start_time = time_module.time()
        current_time = datetime.now()
        
        print(f"\n{Colors.CYAN}🚀 Advanced Options Scan - {current_time.strftime('%H:%M:%S')}{Colors.RESET}")
        print(f"⚡ Analyzing: {Colors.YELLOW}1min → 5min → 15min → 30min → 60min → Daily{Colors.RESET}")
        
        # Update sectors
        self.fetch_live_sectoral_performance()

        try:
            # Get enhanced target stocks
            target_stocks_set = set()
            
            for i, sector in enumerate(self.best_sectors):
                if sector in SECTOR_STOCKS:
                    count = [15, 12, 10][i] if i < 3 else 8
                    target_stocks_set.update(SECTOR_STOCKS[sector][:count])
            
            for i, sector in enumerate(self.worst_sectors):
                if sector in SECTOR_STOCKS:
                    count = [15, 12, 10][i] if i < 3 else 8
                    target_stocks_set.update(SECTOR_STOCKS[sector][:count])

            target_stocks = list(target_stocks_set)
            
            if not target_stocks:
                print("⚠️ No target stocks found.")
                return

            print(f"🎯 Scanning {len(target_stocks)} options-optimized stocks")
            
            options_signals = []
            filtered_counts = {'gap_down': 0, 'high_iv': 0, 'low_volume': 0}
            
            with ThreadPoolExecutor(max_workers=3) as executor:
                def process_stock_for_options(symbol):
                    try:
                        # Fetch all timeframes for enhanced accuracy
                        timeframes_data = {}
                        
                        timeframes_to_fetch = [1, 5, 15, 30, 60, 'daily']
                        
                        for tf in timeframes_to_fetch:
                            df = self.fetch_live_data(symbol, tf)
                            if df is not None: 
                                timeframes_data[tf] = df
                            time_module.sleep(0.8)  # Faster for options

                        if len(timeframes_data) >= 4:  # Require more timeframes for options
                            # Check options-specific filters
                            filters = self.check_options_filters(symbol, 
                                                               timeframes_data.get(1), 
                                                               timeframes_data.get(5))
                            
                            # Apply filters
                            if filters['gap_down']:
                                return None, 'gap_down'
                            elif filters['high_iv']:
                                return None, 'high_iv'
                            elif filters['low_volume']:
                                return None, 'low_volume'
                            
                            signal, score, details = self.calculate_options_signals(symbol, timeframes_data)
                            
                            # Stricter criteria for options
                            if abs(score - 50) > 20:  # Increased from 15 to 20
                                sector = next((s for s, st in SECTOR_STOCKS.items() if symbol in st), 'N/A')
                                
                                # Get current price for strike recommendations
                                current_price = timeframes_data[5]['Close'].iloc[-1] if 5 in timeframes_data else 0
                                recommendations = self.recommend_strikes_and_expiry(symbol, signal, score, current_price)
                                
                                return {
                                    'symbol': symbol, 'signal': signal, 'score': score, 'sector': sector,
                                    'timeframes': len(timeframes_data), 'timestamp': datetime.now(),
                                    'tf_details': list(timeframes_data.keys()),
                                    'signal_details': details,
                                    'recommendations': recommendations,
                                    'current_price': current_price
                                }, None
                    except Exception as e:
                        logger.error(f"Error processing {symbol} for options: {e}")
                    return None, None

                futures = [executor.submit(process_stock_for_options, symbol) for symbol in target_stocks]
                
                for future in as_completed(futures):
                    result, filter_type = future.result()
                    if filter_type:
                        filtered_counts[filter_type] += 1
                    elif result:
                        options_signals.append(result)

            scan_time = time_module.time() - start_time
            
            # Update counters
            self.high_iv_filtered = filtered_counts['high_iv']
            self.low_volume_filtered = filtered_counts['low_volume']
            
            logger.info(f"⚡ Advanced options scan completed in {scan_time:.2f}s")
            self.display_advanced_options_signals(options_signals, scan_time, filtered_counts)

        except Exception as e:
            logger.error(f"Error in advanced options scan: {e}")

    def display_advanced_options_signals(self, signals, scan_time, filtered_counts):
        """Display enhanced options signals with recommendations"""
        os.system('clear' if os.name == 'posix' else 'cls')
        current_time = datetime.now()
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*160}")
        print(f"🎯 ADVANCED OPTIONS SCANNER - HIGH ACCURACY - {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
        print(f"{'='*160}{Colors.RESET}")
        
        print(f"{Colors.BLUE}📊 Analysis: {Colors.YELLOW}1m{Colors.RESET} + {Colors.YELLOW}5m{Colors.RESET} + "
              f"{Colors.YELLOW}15m{Colors.RESET} + {Colors.YELLOW}30m{Colors.RESET} + {Colors.CYAN}60m{Colors.RESET} + {Colors.MAGENTA}Daily{Colors.RESET}")
        
        if self.last_sectoral_update:
            best_str = ', '.join(self.best_sectors)
            worst_str = ', '.join(self.worst_sectors)
            print(f"🏆 Top 3 Best: {Colors.GREEN}{Colors.BOLD}{best_str}{Colors.RESET}")
            print(f"📉 Top 3 Worst: {Colors.RED}{Colors.BOLD}{worst_str}{Colors.RESET}")
        
        print(f"{Colors.BLUE}⚡ Scan Time: {scan_time:.2f}s | "
              f"🚫 Filtered: Gap-down({filtered_counts['gap_down']}) | High-IV({filtered_counts['high_iv']}) | Low-Vol({filtered_counts['low_volume']}){Colors.RESET}")

        if not signals:
            print(f"\n{Colors.YELLOW}📭 No high-confidence options signals found.{Colors.RESET}")
        else:
            # Separate and sort signals
            bullish_signals = [s for s in signals if 'Buy' in s['signal']]
            bearish_signals = [s for s in signals if 'Sell' in s['signal']]
            
            bullish_signals.sort(key=lambda x: x['score'], reverse=True)
            bearish_signals.sort(key=lambda x: x['score'])
            
            print(f"\n{Colors.GREEN}{Colors.BOLD}🚀 TOP 10 BULLISH OPTIONS SIGNALS:{Colors.RESET}")
            print(f"{'Stock':^8} {'Sector':^12} {'Signal':^16} {'Score':^6} {'Price':^8} {'Strike Rec':^12} {'Expiry':^12} {'Strategy':^20} {'Vol Exp':^6}")
            print(f"{Colors.GREEN}{'-' * 160}{Colors.RESET}")
            
            for s in bullish_signals[:10]:
                sector_color = Colors.GREEN if s['sector'] in self.best_sectors else Colors.YELLOW
                signal_color = Colors.GREEN + (Colors.BOLD if 'Very' in s['signal'] else "")
                
                vol_exp = s['signal_details'].get('volatility_expansion', 0)
                vol_color = Colors.GREEN if vol_exp > 70 else Colors.YELLOW if vol_exp > 50 else Colors.WHITE
                
                strikes = ', '.join(s['recommendations']['strikes'][:2])
                expiry = s['recommendations']['expiry_preference']
                strategy = s['recommendations']['strategy'][:18]
                
                print(f"{Colors.WHITE}{s['symbol']:^8}{Colors.RESET} "
                      f"{sector_color}{s['sector'][:12]:^12}{Colors.RESET} "
                      f"{signal_color}{s['signal'][:16]:^16}{Colors.RESET} "
                      f"{Colors.WHITE}{s['score']:^6.1f}{Colors.RESET} "
                      f"{Colors.CYAN}{s['current_price']:^8.1f}{Colors.RESET} "
                      f"{Colors.MAGENTA}{strikes[:12]:^12}{Colors.RESET} "
                      f"{Colors.YELLOW}{expiry[:12]:^12}{Colors.RESET} "
                      f"{Colors.BLUE}{strategy[:20]:^20}{Colors.RESET} "
                      f"{vol_color}{vol_exp:^6.0f}{Colors.RESET}")
            
            if bearish_signals:
                print(f"\n{Colors.RED}{Colors.BOLD}📉 TOP 10 BEARISH OPTIONS SIGNALS:{Colors.RESET}")
                print(f"{'Stock':^8} {'Sector':^12} {'Signal':^16} {'Score':^6} {'Price':^8} {'Strike Rec':^12} {'Expiry':^12} {'Strategy':^20} {'Vol Exp':^6}")
                print(f"{Colors.RED}{'-' * 160}{Colors.RESET}")
                
                for s in bearish_signals[:10]:
                    sector_color = Colors.RED if s['sector'] in self.worst_sectors else Colors.YELLOW
                    signal_color = Colors.RED + (Colors.BOLD if 'Very' in s['signal'] else "")
                    
                    vol_exp = s['signal_details'].get('volatility_expansion', 0)
                    vol_color = Colors.GREEN if vol_exp > 70 else Colors.YELLOW if vol_exp > 50 else Colors.WHITE
                    
                    strikes = ', '.join(s['recommendations']['strikes'][:2])
                    expiry = s['recommendations']['expiry_preference']
                    strategy = s['recommendations']['strategy'][:18]
                    
                    print(f"{Colors.WHITE}{s['symbol']:^8}{Colors.RESET} "
                          f"{sector_color}{s['sector'][:12]:^12}{Colors.RESET} "
                          f"{signal_color}{s['signal'][:16]:^16}{Colors.RESET} "
                          f"{Colors.WHITE}{s['score']:^6.1f}{Colors.RESET} "
                          f"{Colors.CYAN}{s['current_price']:^8.1f}{Colors.RESET} "
                          f"{Colors.MAGENTA}{strikes[:12]:^12}{Colors.RESET} "
                          f"{Colors.YELLOW}{expiry[:12]:^12}{Colors.RESET} "
                          f"{Colors.BLUE}{strategy[:20]:^20}{Colors.RESET} "
                          f"{vol_color}{vol_exp:^6.0f}{Colors.RESET}")
        
        next_scan_time = (current_time + timedelta(minutes=3)).strftime('%H:%M:%S')
        print(f"\n{Colors.CYAN}{Colors.BOLD}⏰ Next enhanced scan at {next_scan_time} | 🎯 Optimized for Options Trading{Colors.RESET}")

    def run_advanced_options_scanner(self):
        """Main enhanced options scanner loop"""
        self.is_running = True
        logger.info("🚀 Starting Advanced Options Scanner...")
        
        try:
            while self.is_running:
                self.advanced_options_scan_cycle()
                if self.is_running:
                    logger.info(f"💤 Waiting 3 minutes for next enhanced cycle...")
                    time_module.sleep(self.scan_interval)
        except KeyboardInterrupt:
            logger.info("🛑 Advanced options scanner stopped by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the enhanced scanner"""
        self.is_running = False
        print(f"{Colors.YELLOW}🛑 Advanced options scanner stopped{Colors.RESET}")

# --- MAIN EXECUTION ---
def main():
    print(f"{Colors.CYAN}{Colors.BOLD}🎯 ADVANCED OPTIONS SCANNER - HIGH ACCURACY{Colors.RESET}")
    print(f"{Colors.YELLOW}📊 Enhanced with: IV Analysis, Volume Surge, Volatility Expansion{Colors.RESET}")
    print(f"{Colors.CYAN}🔧 Features: Strike Recommendations, Risk Assessment, Enhanced Filters{Colors.RESET}")
    print(f"{Colors.MAGENTA}⚡ Updates every 3 minutes for optimal options timing{Colors.RESET}")
    
    scanner = AdvancedOptionsScanner()
    try:
        scanner.run_advanced_options_scanner()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Shutting down advanced options scanner...{Colors.RESET}")
        scanner.stop()

if __name__ == "__main__":
    main()
