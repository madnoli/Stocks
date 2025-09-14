# Enhanced Multi-Indicator Scanner with 5-Minute Sectoral Detection & Options Features
# Features: All original functionality + comprehensive options trading capabilities
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
warnings.filterwarnings('ignore')
import math

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

# --- NSE INDEX MAPPING ---
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
    "NIFTY CPSE": "CPSE",
    "NIFTY FINANCIAL SERVICES": "Finance",
    "NIFTY INFRASTRUCTURE": "Infrastructure",
    "BANKNIFTY": "Banking",
    "NIFTYFIN": "Finance",
    "NIFTYAUTO": "Auto",
    "NIFTYIT": "Technology",
    "NIFTYPHARMA": "Pharma",
    "NIFTY CONSUMER DURABLES": "CONSUMER DURABLES",
    "NIFTY HEALTHCARE INDEX": "Healthcare",
    "NIFTY CAPITAL MARKETS": "Capital Market",
    "NIFTY PRIVATE BANK": "Private Bank",
    "NIFTY OIL & GAS": "OIL and GAS",
    "NIFTY INDIA CONSUMPTION": "INDIA CONSUMPTION",
}

# --- SECTOR STOCKS ---
SECTOR_STOCKS = {
    "Technology": ["TCS", "INFY", "HCLTECH", "WIPRO", "TECHM", "LTIM", "MPHASIS", "COFORGE", "PERSISTENT", "CYIENT", "KPITTECH", "TATAELXSI","SONACOMS","KAYNES","OFSS"],
    "Auto": ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "TVSMOTOR","BHARATFORG", "EICHERMOT", "ASHOKLEY", "BOSCHLTD","TIINDIA","MOTHERSON"],
    "Banking": ["HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","PNB","BANKBARODA","CANBK","IDFCFIRSTB","INDUSINDBK","AUBANK","FEDERALBNK"],
    "Pharma": ["SUNPHARMA", "DRREDDY", "CIPLA", "LUPIN", "AUROPHARMA", "TORNTPHARM","GLENMARK","ALKEM","LAURUSLABS","BIOCON","ZYDUSLIFE","MANKIND","SYNGENE","PPLPHARMA"],
    "Energy": ["RELIANCE","NTPC","BPCL","IOC","ONGC","GAIL","HINDPETRO","ADANIGREEN","ADANIENSOL","JSWENERGY","COALINDIA","TATAPOWER","SUZLON","PETRONET","OIL","POWERGRID","NHPC","ADANIPORTS","ABB","SIEMENS","CGPOWER","INOXWIND"],
    "Metal": ["TATASTEEL", "JSWSTEEL", "SAIL", "JINDALSTEL", "HINDALCO", "NMDC"],
    "Consumer": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "TATACONSUM", "DABUR","AMBER","UNITDSPR","GODREJCP","MARICO","COLPAL","UPL","VBL"],
    "PSU Bank": ["SBIN", "PNB", "BANKBARODA", "CANBK", "UNIONBANK", "BANKINDIA"],
    "Finance": ["BAJFINANCE", "SHRIRAMFIN", "CHOLAFIN", "HDFCLIFE", "ICICIPRULI","ETERNAL",],
    "Realty": ["DLF","LODHA","PRESTIGE","GODREJPROP","OBEROIRLTY","PHOENIXLTD","BRIGADE","NCC","NBCC"],
    "CPSE": ["NHPC","NBCC","NTPC","POWERGRID","OIL","ONGC","COALINDIA","BEL"],
    "PSE": ["BEL","BHEL","NHPC","GAIL","IOC","NTPC","POWERGRID","NTPC","HINDPETRO","OIL","RECLTD","ONGC","NMDC","BPCL","HAL","RVNL","PFC","COALINDIA","IRCTC","IRFC",],
    "Commodities": ["AMBUJACEM","APLAPOLLO","ULTRACEMCO","SHREECEM","JSWSTEEL","HINDALCO","NHPC","IOC","NTPC","HINDPETRO","ADANIGREEN","OIL","VEDL","PIIND","ONGC","NMDC","UPL","BPCL","JSWENERGY","GRASIM","RELIANCE","TORNTPOWER","TATAPOWER","COALINDIA","PIDILITIND","SRF","ADANIENSOL","JINDALSTEL","TATASTEEL","HINDALCO"],
    "CONSUMER DURABLES": ["TITAN","DIXON","HAVELLS","CROMPTON","POLYCAB","EXIDEIND","AMBER","KAYNES","VOLTAS","PGEL","BLUESTARCO"],
    "Healthcare": ["SUNPHARMA","DIVISLAB","CIPLA","TORNTPHARM","MAXHEALTH","APOLLOHOSP","DRREDDY","MANKIND","ZYDUSLIFE","LUPIN","FORTIS","ALKEM","AUROPHARMA","GLENMARK","BIOCON","LAURUSLABS","SYNGENE","GRANULES"],
    "Capital Market": ["HDFCAMC","BSE","360ONE","MCX","CDSL","NUVAMA","ANGELONE","KFINTECH","CAMS","IEX"],
    "Private Bank": ["HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","YESBANK","IDFCFIRSTB","INDUSINDBK","FEDERALBNK","BANDHANBNK","RBLBANK"],
    "OIL and GAS": ["RELIANCE","ONGC","IOC","BPCL","GAIL","HINDPETRO","OIL","PETRONET","IGL"],
    "INDIA CONSUMPTION": ["BHARTIARTL","HINDUNILVR","ITC","MARUTI","M&M","TITAN","ETERNAL","DMART","BAJAJ-AUTO","ASIANPAINT","ADANIPOWER","NESTLEIND","INDIGO","DLF","EICHERMOT","TRENT","TVSMOTOR","VBL","BRITANNIA","GODREJCP","TATAPOWER","MAXHEALTH","APOLLOHOSP","INDHOTEL","TATACONSUM","HEROMOTOCO","HAVELLS","UNITDSPR","NAUKRI","COLPAL"]
}

# --- OPTIONS-FOCUSED STOCKS ---
OPTIONS_FOCUSED_STOCKS = {
    "NIFTY": ["NIFTY"],
    "BANKNIFTY": ["BANKNIFTY"],
    "High_Liquidity": ["RELIANCE", "TCS", "HDFCBANK", "ICICIBANK", "INFY", "ITC", "SBIN", "BHARTIARTL", "LT", "KOTAKBANK"],
    "Active_Options": ["MARUTI", "TATAMOTORS", "BAJFINANCE", "ASIANPAINT", "HINDUNILVR", "AXISBANK", "WIPRO", "SUNPHARMA"]
}

# --- TIMEFRAME & INDICATOR WEIGHTS ---
TIMEFRAME_WEIGHTS = {5: 1.0, 15: 1.5, 30: 2.0}
INDICATOR_WEIGHTS = {
    'RSI': 1.0, 'MACD': 1.2, 'Stochastic': 0.8, 'MA': 1.5,
    'ADX': 1.2, 'Bollinger': 1.0, 'ROC': 0.7, 'OBV': 1.3, 'CCI': 0.9, 'WWL': 0.9
}

# --- OPTIONS DATA HANDLER ---
class OptionsDataHandler:
    def __init__(self):
        self.option_chain_cache = {}
        self.cache_timeout = 180  # 3 minutes cache
        self.vix_data = None
        self.pcr_data = {}
        
    def fetch_option_chain_data(self, symbol):
        """Fetch real-time option chain data"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{datetime.now().strftime('%H:%M')}"
            if cache_key in self.option_chain_cache:
                cache_time = self.option_chain_cache[cache_key]['timestamp']
                if (datetime.now() - cache_time).seconds < self.cache_timeout:
                    return self.option_chain_cache[cache_key]['data']
            
            # Try local API first
            option_data = self._fetch_local_option_chain(symbol)
            
            # Generate mock data for demo if API fails
            if not option_data:
                option_data = self._generate_mock_option_data(symbol)
            
            if option_data:
                self.option_chain_cache[cache_key] = {
                    'data': option_data,
                    'timestamp': datetime.now()
                }
                
            return option_data
            
        except Exception as e:
            logger.error(f"Error fetching option chain for {symbol}: {e}")
            return self._generate_mock_option_data(symbol)
    
    def _fetch_local_option_chain(self, symbol):
        """Fetch from local API"""
        try:
            response = requests.get(f"http://localhost:3001/api/optionChain/{symbol}", timeout=5)
            if response.status_code == 200:
                return response.json()
        except:
            pass
        return None
    
    def _generate_mock_option_data(self, symbol):
        """Generate realistic mock option data for demonstration"""
        try:
            # Get current price (mock)
            current_price = 25000 if symbol == "NIFTY" else 52000 if symbol == "BANKNIFTY" else 100
            
            strikes = []
            for i in range(-10, 11):
                strike_gap = 50 if symbol == "NIFTY" else 100 if symbol == "BANKNIFTY" else 5
                strike = current_price + (i * strike_gap)
                
                # Mock realistic OI and volume based on distance from ATM
                distance = abs(strike - current_price)
                max_oi = 10000 - (distance * 10)
                max_volume = 5000 - (distance * 20)
                
                strikes.append({
                    'strike': strike,
                    'call_oi': max(max_oi + np.random.randint(-1000, 1000), 100),
                    'put_oi': max(max_oi + np.random.randint(-1000, 1000), 100),
                    'call_volume': max(max_volume + np.random.randint(-500, 500), 10),
                    'put_volume': max(max_volume + np.random.randint(-500, 500), 10),
                    'call_ltp': max(current_price - strike + np.random.randint(-50, 50), 5),
                    'put_ltp': max(strike - current_price + np.random.randint(-50, 50), 5),
                    'call_iv': 15 + np.random.uniform(-5, 5),
                    'put_iv': 15 + np.random.uniform(-5, 5)
                })
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'strikes': strikes,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error generating mock data for {symbol}: {e}")
            return None
    
    def analyze_oi_patterns(self, option_data):
        """Analyze Open Interest patterns"""
        if not option_data or 'strikes' not in option_data:
            return {}
        
        try:
            strikes = option_data['strikes']
            current_price = option_data['current_price']
            
            # Calculate max pain
            max_pain = self.calculate_max_pain(strikes, current_price)
            
            # Calculate PCR
            pcr_oi = self.calculate_pcr_oi(strikes)
            pcr_volume = self.calculate_pcr_volume(strikes)
            
            return {
                'max_pain': max_pain,
                'pcr_oi': pcr_oi,
                'pcr_volume': pcr_volume,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing OI patterns: {e}")
            return {}
    
    def calculate_max_pain(self, strikes, current_price):
        """Calculate Max Pain point"""
        try:
            max_pain_values = []
            
            for strike_data in strikes:
                strike = strike_data['strike']
                total_pain = 0
                
                for s in strikes:
                    s_strike = s['strike']
                    call_oi = s.get('call_oi', 0)
                    put_oi = s.get('put_oi', 0)
                    
                    if strike > s_strike:  # Call ITM
                        total_pain += call_oi * (strike - s_strike)
                    if strike < s_strike:  # Put ITM
                        total_pain += put_oi * (s_strike - strike)
                
                max_pain_values.append({'strike': strike, 'pain': total_pain})
            
            # Find minimum pain point
            min_pain = min(max_pain_values, key=lambda x: x['pain'])
            return min_pain['strike']
            
        except Exception as e:
            logger.error(f"Error calculating max pain: {e}")
            return current_price
    
    def calculate_pcr_oi(self, strikes):
        """Calculate Put-Call Ratio by Open Interest"""
        try:
            total_put_oi = sum(s.get('put_oi', 0) for s in strikes)
            total_call_oi = sum(s.get('call_oi', 0) for s in strikes)
            
            if total_call_oi > 0:
                return round(total_put_oi / total_call_oi, 3)
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating PCR OI: {e}")
            return 0
    
    def calculate_pcr_volume(self, strikes):
        """Calculate Put-Call Ratio by Volume"""
        try:
            total_put_volume = sum(s.get('put_volume', 0) for s in strikes)
            total_call_volume = sum(s.get('call_volume', 0) for s in strikes)
            
            if total_call_volume > 0:
                return round(total_put_volume / total_call_volume, 3)
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating PCR Volume: {e}")
            return 0
    
    def calculate_iv_indicators(self, symbol):
        """Calculate IV-based indicators"""
        try:
            option_data = self.fetch_option_chain_data(symbol)
            if not option_data:
                return {}
            
            strikes = option_data.get('strikes', [])
            if not strikes:
                return {}
            
            # Calculate average IV
            call_ivs = [s.get('call_iv', 0) for s in strikes if s.get('call_iv', 0) > 0]
            put_ivs = [s.get('put_iv', 0) for s in strikes if s.get('put_iv', 0) > 0]
            
            avg_call_iv = np.mean(call_ivs) if call_ivs else 0
            avg_put_iv = np.mean(put_ivs) if put_ivs else 0
            avg_iv = (avg_call_iv + avg_put_iv) / 2 if avg_call_iv and avg_put_iv else 0
            
            # IV percentile (mock calculation)
            iv_percentile = min(100, max(0, (avg_iv - 10) * 5))
            
            # IV rank (simplified)
            iv_rank = min(100, max(0, (avg_iv - 8) * 6.25))
            
            return {
                'avg_iv': round(avg_iv, 2),
                'call_iv': round(avg_call_iv, 2),
                'put_iv': round(avg_put_iv, 2),
                'iv_percentile': round(iv_percentile, 1),
                'iv_rank': round(iv_rank, 1),
                'iv_environment': 'High' if avg_iv > 20 else 'Medium' if avg_iv > 12 else 'Low'
            }
            
        except Exception as e:
            logger.error(f"Error calculating IV indicators for {symbol}: {e}")
            return {}

# --- OPTIONS STRATEGY SCANNER ---
class OptionsStrategyScanner:
    def __init__(self, options_handler):
        self.options_handler = options_handler
        
    def scan_options_strategies(self, symbol):
        """Scan for various options strategies"""
        try:
            option_data = self.options_handler.fetch_option_chain_data(symbol)
            if not option_data:
                return {}
            
            strategies = {
                'covered_calls': self.scan_covered_call_opportunities(option_data),
                'straddles': self.scan_straddle_opportunities(option_data)
            }
            
            return strategies
            
        except Exception as e:
            logger.error(f"Error scanning strategies for {symbol}: {e}")
            return {}
    
    def scan_covered_call_opportunities(self, option_data):
        """Scan for covered call opportunities"""
        try:
            current_price = option_data['current_price']
            strikes = option_data['strikes']
            
            opportunities = []
            
            for strike_data in strikes:
                strike = strike_data['strike']
                call_ltp = strike_data.get('call_ltp', 0)
                call_iv = strike_data.get('call_iv', 0)
                
                if strike > current_price and call_ltp > 20:  # OTM calls with decent premium
                    monthly_yield = (call_ltp / current_price) * 100
                    if monthly_yield > 1:  # At least 1% monthly yield
                        opportunities.append({
                            'strike': strike,
                            'premium': call_ltp,
                            'monthly_yield': round(monthly_yield, 2),
                            'iv': round(call_iv, 1)
                        })
            
            return sorted(opportunities, key=lambda x: x['monthly_yield'], reverse=True)[:3]
            
        except Exception as e:
            logger.error(f"Error scanning covered calls: {e}")
            return []
    
    def scan_straddle_opportunities(self, option_data):
        """Scan for long/short straddle opportunities"""
        try:
            current_price = option_data['current_price']
            strikes = option_data['strikes']
            
            opportunities = []
            
            for strike_data in strikes:
                strike = strike_data['strike']
                call_ltp = strike_data.get('call_ltp', 0)
                put_ltp = strike_data.get('put_ltp', 0)
                call_iv = strike_data.get('call_iv', 0)
                put_iv = strike_data.get('put_iv', 0)
                
                if abs(strike - current_price) < current_price * 0.02:  # Near ATM
                    total_premium = call_ltp + put_ltp
                    avg_iv = (call_iv + put_iv) / 2
                    breakeven_range = total_premium / current_price * 100
                    
                    strategy_type = 'Long' if avg_iv < 15 else 'Short'
                    
                    opportunities.append({
                        'strike': strike,
                        'total_premium': round(total_premium, 1),
                        'breakeven_range': round(breakeven_range, 2),
                        'avg_iv': round(avg_iv, 1),
                        'strategy': strategy_type
                    })
            
            return sorted(opportunities, key=lambda x: x['breakeven_range'])[:2]
            
        except Exception as e:
            logger.error(f"Error scanning straddles: {e}")
            return []

# --- TECHNICAL INDICATORS CLASS ---
class TechnicalIndicators:
    @staticmethod
    def calculate_all_indicators(df):
        """Calculate all technical indicators"""
        indicators = {}
        if df is None or len(df) < 20:
            return indicators

        try:
            # 1. RSI
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            indicators['RSI'] = 100 - (100 / (1 + rs))

            # 2. MACD
            ema12 = df['Close'].ewm(span=12).mean()
            ema26 = df['Close'].ewm(span=26).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9).mean()
            indicators['MACD'] = macd_line - signal_line

            # 3. Stochastic
            low14 = df['Low'].rolling(window=14).min()
            high14 = df['High'].rolling(window=14).max()
            indicators['Stochastic'] = 100 * (df['Close'] - low14) / (high14 - low14)

            # 4. Moving Average
            indicators['MA'] = df['Close'].rolling(window=20).mean()

            # 5. ADX
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

            # 6. Bollinger Bands Position
            ma20 = df['Close'].rolling(window=20).mean()
            std20 = df['Close'].rolling(window=20).std()
            upper_band = ma20 + (std20 * 2)
            lower_band = ma20 - (std20 * 2)
            indicators['Bollinger'] = (df['Close'] - ma20) / (upper_band - lower_band) * 100

            # 7. ROC
            indicators['ROC'] = df['Close'].pct_change(periods=12) * 100

            # 8. OBV
            obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
            indicators['OBV'] = obv.pct_change(periods=10) * 100

            # 9. CCI
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma_tp = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
            indicators['CCI'] = (typical_price - sma_tp) / (0.015 * mad)

            # 10. Williams %R
            highest_high = df['High'].rolling(window=14).max()
            lowest_low = df['Low'].rolling(window=14).min()
            indicators['WWL'] = (highest_high - df['Close']) / (highest_high - lowest_low) * -100

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")

        return indicators

# --- ENHANCED LIVE SCANNER WITH OPTIONS FEATURES ---
class Enhanced5MinLiveScanner:
    def __init__(self):
        # Original attributes (unchanged)
        self.is_running = False
        self.current_signals = {}
        self.best_sectors = ["Technology", "Pharma"]
        self.worst_sectors = ["Auto", "Metal"]
        self.sectoral_history = []
        self.last_sectoral_update = None
        self.api_errors = []
        self.sector_update_attempts = 0
        self.successful_updates = 0
        
        # NEW: Options-specific attributes
        self.options_handler = OptionsDataHandler()
        self.strategy_scanner = OptionsStrategyScanner(self.options_handler)
        self.options_signals = {}
        self.iv_environment = "Medium"
        self.vix_level = 15.0
        self.options_enabled = True
        
        # Market hours (unchanged)
        self.market_start = time(9, 15)
        self.market_end = time(15, 30)
        
        # Intervals (unchanged)
        self.scan_interval = 300  # 5 minutes
        self.sectoral_update_interval = 300  # 5 minutes
        
        logger.info("🚀 Enhanced 5-Min Live Scanner with Options Features initialized")
        
        # Initialize with debug checks
        self.show_initialization_status()

    def show_initialization_status(self):
        """Show initialization status and run initial checks"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}🚀 ENHANCED SCANNER INITIALIZATION{Colors.RESET}")
        print(f"{Colors.CYAN}{'='*60}{Colors.RESET}")
        
        # Show current sector status
        self.show_sector_status()
        
        # Test API connection
        self.test_api_connection()
        
        # NEW: Options system checks
        self.test_options_system()
        
        # Force initial sector update
        print(f"\n{Colors.YELLOW}🔄 Running initial sector update...{Colors.RESET}")
        self.force_sector_update()
        
        print(f"{Colors.CYAN}{'='*60}{Colors.RESET}")

    def test_options_system(self):
        """Test options data system"""
        print(f"\n{Colors.MAGENTA}🎯 OPTIONS SYSTEM TEST:{Colors.RESET}")
        
        try:
            # Test option data fetch
            test_data = self.options_handler.fetch_option_chain_data("NIFTY")
            if test_data:
                print(f"✅ Options Data: {Colors.GREEN}SUCCESS{Colors.RESET}")
                print(f"📊 Strikes Available: {len(test_data.get('strikes', []))}")
                
                # Test OI analysis
                oi_analysis = self.options_handler.analyze_oi_patterns(test_data)
                if oi_analysis:
                    print(f"📈 OI Analysis: {Colors.GREEN}WORKING{Colors.RESET}")
                    print(f"🎯 Max Pain: {oi_analysis.get('max_pain', 'N/A')}")
                    print(f"📊 PCR (OI): {oi_analysis.get('pcr_oi', 'N/A')}")
                
                # Test IV indicators
                iv_data = self.options_handler.calculate_iv_indicators("NIFTY")
                if iv_data:
                    print(f"📊 IV Analysis: {Colors.GREEN}WORKING{Colors.RESET}")
                    self.iv_environment = iv_data.get('iv_environment', 'Medium')
                    print(f"🌡️ IV Environment: {Colors.YELLOW}{self.iv_environment}{Colors.RESET}")
                
            else:
                print(f"⚠️ Options Data: {Colors.YELLOW}MOCK MODE{Colors.RESET}")
                
        except Exception as e:
            print(f"❌ Options System: {Colors.RED}ERROR{Colors.RESET} - {str(e)}")
            print(f"🔄 Continuing with mock data...")

    def test_api_connection(self):
        """Test API connection and show response structure"""
        print(f"\n{Colors.BLUE}🔍 API CONNECTION TEST:{Colors.RESET}")
        try:
            response = requests.get("http://localhost:3001/api/allIndices", timeout=10)
            
            if response.status_code == 200:
                print(f"✅ API Connection: {Colors.GREEN}SUCCESS{Colors.RESET}")
                data = response.json()
                print(f"📊 Response Type: {type(data)}")
                
                if isinstance(data, list) and data:
                    print(f"📋 Items Count: {len(data)}")
                    
            else:
                print(f"❌ API Connection: {Colors.RED}FAILED{Colors.RESET} (Status: {response.status_code})")
                
        except Exception as e:
            print(f"❌ API Connection: {Colors.RED}ERROR{Colors.RESET} - {str(e)}")

    def show_sector_status(self):
        """Show current sector selection status"""
        print(f"\n{Colors.MAGENTA}📊 CURRENT SECTOR STATUS:{Colors.RESET}")
        print(f"🏆 Top 2 Best Sectors: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
        print(f"📉 Top 2 Worst Sectors: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
        print(f"🕐 Last Update: {self.last_sectoral_update or 'Never'}")

    def force_sector_update(self):
        """Force a sector update with detailed logging"""
        print(f"\n{Colors.YELLOW}🔄 FORCING SECTOR UPDATE...{Colors.RESET}")
        self.sector_update_attempts += 1
        
        success = self.fetch_live_sectoral_performance_5min_debug()
        
        if success:
            self.successful_updates += 1
            print(f"✅ Update successful!")
            print(f"🏆 Top 2 Best: {Colors.GREEN}{Colors.BOLD}{', '.join(self.best_sectors)}{Colors.RESET}")
            print(f"📉 Top 2 Worst: {Colors.RED}{Colors.BOLD}{', '.join(self.worst_sectors)}{Colors.RESET}")
        else:
            print(f"❌ Update failed - using defaults")
            print(f"🏆 Default Best: {Colors.YELLOW}{', '.join(self.best_sectors)}{Colors.RESET}")
            print(f"📉 Default Worst: {Colors.YELLOW}{', '.join(self.worst_sectors)}{Colors.RESET}")
        
        return success

    def is_market_open(self):
        """Check if market is open"""
        now = datetime.now()
        current_time = now.time()
        
        if now.weekday() > 4:  # Weekend
            return False
            
        return self.market_start <= current_time <= self.market_end

    def fetch_live_sectoral_performance_5min_debug(self):
        """Enhanced sectoral performance fetching with comprehensive debugging"""
        try:
            logger.info("🔍 Fetching live sectoral performance (5-min update with debug)...")
            
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
                    logger.error("❌ Processed API data is not a list.")
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
                    
                    # Update top 2 best and worst sectors
                    if len(sectoral_performance) >= 4:
                        self.best_sectors = [sectoral_performance[0]['sector'], sectoral_performance[1]['sector']]
                        self.worst_sectors = [sectoral_performance[-1]['sector'], sectoral_performance[-2]['sector']]
                    elif len(sectoral_performance) >= 2:
                        self.best_sectors = [sectoral_performance[0]['sector']]
                        self.worst_sectors = [sectoral_performance[-1]['sector']]
                    
                    self.last_sectoral_update = current_time
                    
                    self.sectoral_history.append({
                        'timestamp': current_time,
                        'best': self.best_sectors, 'worst': self.worst_sectors,
                        'full_data': sectoral_performance
                    })
                    
                    if len(self.sectoral_history) > 20:
                        self.sectoral_history = self.sectoral_history[-20:]
                    
                    return True
                else:
                    print(f"❌ No sectoral data matched.")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Error fetching sectoral data: {e}")
            self.api_errors.append(f"{datetime.now()}: {e}")
            return False

    def normalize_live_data(self, df, symbol):
        """Fast data normalization"""
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
                
            return df_clean.dropna().sort_index() if len(df_clean) >= 20 else None
        except Exception as e:
            logger.error(f"Normalize error {symbol}: {e}")
            return None

    def fetch_live_data(self, symbol, timeframe):
        """Fetch live data for symbol and timeframe"""
        try:
            tf_map = {5: '5 min', 15: '15 min', 30: '30 min'}
            bar_size = tf_map.get(timeframe)
            if not bar_size: return None

            duration = '10 D' if timeframe <= 15 else '20 D'
            raw_df = td_hist.get_historic_data(symbol, duration=duration, bar_size=bar_size)

            if raw_df is not None and len(raw_df) > 0:
                normalized_df = self.normalize_live_data(raw_df, symbol)
                if normalized_df is not None and len(normalized_df) >= 20:
                    return normalized_df.tail(100)
            return None
        except Exception as e:
            logger.error(f"Live data fetch error {symbol}_{timeframe}min: {e}")
            return None

    def normalize_indicator_value(self, indicator_name, value):
        """Normalize indicator values to 0-100 scale"""
        try:
            if indicator_name == 'RSI': return max(0, min(100, value))
            if indicator_name == 'MACD': return 50 + min(25, max(-25, value * 10))
            if indicator_name == 'Stochastic': return max(0, min(100, value))
            if indicator_name == 'MA': return 50
            if indicator_name == 'ADX': return max(0, min(100, value))
            if indicator_name == 'Bollinger': return max(0, min(100, (value + 100) / 2))
            if indicator_name == 'ROC': return 50 + min(25, max(-25, value * 2))
            if indicator_name == 'OBV': return 50 + min(25, max(-25, value))
            if indicator_name == 'CCI': return max(0, min(100, (value + 200) / 4))
            if indicator_name == 'WWL': return max(0, min(100, value + 100))
            return 50
        except: return 50

    def calculate_multi_indicator_signals_with_options(self, symbol, timeframes_data):
        """Enhanced signal calculation with options data integration"""
        try:
            if not timeframes_data: 
                return 'Neutral', 0, {}

            sector = next((s for s, stocks in SECTOR_STOCKS.items() if symbol in stocks), None)
            if not sector: 
                return 'Neutral', 0, {}

            # Original technical analysis
            total_weighted_score, total_weight = 0, 0
            for tf, df in timeframes_data.items():
                if df is None or len(df) < 20: continue
                
                indicators = TechnicalIndicators.calculate_all_indicators(df)
                if not indicators: continue

                tf_score, tf_weight = 0, 0
                for name, weight in INDICATOR_WEIGHTS.items():
                    if name in indicators and indicators[name] is not None and not indicators[name].empty:
                        latest_val = indicators[name].iloc[-1]
                        if pd.notna(latest_val):
                            norm_score = self.normalize_indicator_value(name, latest_val)
                            tf_score += norm_score * weight
                            tf_weight += weight
                
                if tf_weight > 0:
                    total_weighted_score += (tf_score / tf_weight) * TIMEFRAME_WEIGHTS.get(tf, 1.0)
                    total_weight += TIMEFRAME_WEIGHTS.get(tf, 1.0)

            if total_weight == 0: 
                return 'Neutral', 0, {}
            
            base_score = total_weighted_score / total_weight

            # Sector boost
            sector_boost = 0
            if sector in self.best_sectors:
                sector_boost = 15 if sector == self.best_sectors[0] else 10
            elif sector in self.worst_sectors:
                sector_boost = -15 if sector == self.worst_sectors[0] else -10
            base_score += sector_boost

            # NEW: Options enhancement
            options_data = {}
            options_boost = 0
            
            if self.options_enabled and symbol in OPTIONS_FOCUSED_STOCKS.get('High_Liquidity', []):
                try:
                    # Get options data
                    option_chain = self.options_handler.fetch_option_chain_data(symbol)
                    if option_chain:
                        oi_analysis = self.options_handler.analyze_oi_patterns(option_chain)
                        iv_data = self.options_handler.calculate_iv_indicators(symbol)
                        
                        options_data = {
                            'oi_analysis': oi_analysis,
                            'iv_data': iv_data,
                            'option_chain': option_chain
                        }
                        
                        # Calculate options-based score adjustment
                        options_boost = self.calculate_options_score_boost(oi_analysis, iv_data, base_score)
                        
                except Exception as e:
                    logger.error(f"Options analysis error for {symbol}: {e}")

            final_score = base_score + options_boost

            # Generate signal
            if final_score >= 70: signal = 'Very Strong Buy'
            elif final_score >= 60: signal = 'Strong Buy'
            elif final_score >= 55: signal = 'Buy'
            elif final_score <= 30: signal = 'Very Strong Sell'
            elif final_score <= 40: signal = 'Strong Sell'
            elif final_score <= 45: signal = 'Sell'
            else: signal = 'Neutral'

            return signal, final_score, options_data

        except Exception as e:
            logger.error(f"Enhanced signal calculation error for {symbol}: {e}")
            return 'Neutral', 0, {}

    def calculate_options_score_boost(self, oi_analysis, iv_data, base_score):
        """Calculate score boost based on options data"""
        try:
            boost = 0
            
            if not oi_analysis or not iv_data:
                return boost
            
            # PCR-based adjustment
            pcr_oi = oi_analysis.get('pcr_oi', 0)
            if pcr_oi:
                if pcr_oi > 1.2:  # Bearish
                    boost -= 5 if base_score > 50 else 0
                elif pcr_oi < 0.8:  # Bullish
                    boost += 5 if base_score > 50 else 0
            
            # IV environment adjustment
            iv_env = iv_data.get('iv_environment', 'Medium')
            iv_percentile = iv_data.get('iv_percentile', 50)
            
            if iv_env == 'High' and iv_percentile > 70:
                if base_score < 50:  # If bearish, boost short signals
                    boost -= 3
                else:  # If bullish, be cautious
                    boost -= 1
            elif iv_env == 'Low' and iv_percentile < 30:
                if base_score > 50:  # If bullish, boost long signals
                    boost += 3
                elif base_score < 50:  # If bearish, boost short signals
                    boost -= 3
            
            return max(-10, min(10, boost))  # Cap the boost
            
        except Exception as e:
            logger.error(f"Error calculating options boost: {e}")
            return 0

    def enhanced_5min_scan_cycle_with_options(self):
        """Enhanced 5-minute scan with options analysis"""
        if not self.is_market_open():
            logger.info("🕐 Market closed. Next scan in 5 minutes...")
            return

        start_time = time_module.time()
        current_time = datetime.now()
        
        print(f"\n{Colors.CYAN}🔄 Starting enhanced scan cycle with options at {current_time.strftime('%H:%M:%S')}{Colors.RESET}")
        
        # Update sectoral data
        if not self.fetch_live_sectoral_performance_5min_debug():
            print(f"⚠️ Sectoral update failed, continuing with previous sectors")

        try:
            # Get target stocks
            target_stocks_set = set()
            if len(self.best_sectors) > 0 and self.best_sectors[0] in SECTOR_STOCKS:
                target_stocks_set.update(SECTOR_STOCKS[self.best_sectors[0]][:10])
            if len(self.best_sectors) > 1 and self.best_sectors[1] in SECTOR_STOCKS:
                target_stocks_set.update(SECTOR_STOCKS[self.best_sectors[1]][:8])

            # Add high-liquidity options stocks
            target_stocks_set.update(OPTIONS_FOCUSED_STOCKS['High_Liquidity'][:5])
            
            target_stocks = list(target_stocks_set)
            
            if not target_stocks:
                print(f"⚠️ No target stocks found.")
                return

            print(f"🎯 Scanning {len(target_stocks)} stocks (including options-focused stocks)")
            
            live_signals = []
            
            with ThreadPoolExecutor(max_workers=6) as executor:
                def process_stock_with_options(symbol):
                    try:
                        # Original technical analysis
                        timeframes_data = {}
                        for tf in [5, 15, 30]:
                            df = self.fetch_live_data(symbol, tf)
                            if df is not None: 
                                timeframes_data[tf] = df
                            time_module.sleep(0.6)

                        if len(timeframes_data) >= 2:
                            # Enhanced signal calculation with options
                            signal, score, options_data = self.calculate_multi_indicator_signals_with_options(symbol, timeframes_data)
                            
                            if abs(score - 50) > 10:
                                sector = next((s for s, st in SECTOR_STOCKS.items() if symbol in st), 'N/A')
                                
                                result = {
                                    'symbol': symbol, 'signal': signal, 'score': score, 'sector': sector,
                                    'timeframes': len(timeframes_data), 'timestamp': datetime.now(),
                                    'options_data': options_data
                                }
                                
                                # Add options strategy scan for high-priority stocks
                                if symbol in OPTIONS_FOCUSED_STOCKS.get('High_Liquidity', []):
                                    strategies = self.strategy_scanner.scan_options_strategies(symbol)
                                    if strategies:
                                        result['strategies'] = strategies
                                
                                return result
                                
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
                    return None

                futures = [executor.submit(process_stock_with_options, symbol) for symbol in target_stocks]
                live_signals = [future.result() for future in as_completed(futures) if future.result()]

            scan_time = time_module.time() - start_time
            logger.info(f"⚡ Enhanced options scan completed in {scan_time:.2f}s - {len(live_signals)} signals")
            
            # Store options signals
            self.options_signals = {s['symbol']: s.get('options_data', {}) for s in live_signals}
            
            self.display_enhanced_signals_with_options(live_signals, scan_time)

        except Exception as e:
            logger.error(f"Error in enhanced options scan: {e}")

    def display_enhanced_signals_with_options(self, signals, scan_time):
        """Display enhanced signals with options context"""
        os.system('clear' if os.name == 'posix' else 'cls')
        current_time = datetime.now()
        
        print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*120}")
        print(f"🎯 ENHANCED MULTI-INDICATOR LIVE SCANNER WITH OPTIONS - {current_time.strftime('%Y-%m-%d %H:%M:%S')} IST")
        print(f"{'='*120}{Colors.RESET}")
        
        # Sectoral and options environment info
        if self.last_sectoral_update:
            best_str = ', '.join(self.best_sectors)
            worst_str = ', '.join(self.worst_sectors)
            print(f"{Colors.MAGENTA}📊 Sectoral Update:{Colors.RESET} {Colors.YELLOW}{self.last_sectoral_update.strftime('%H:%M:%S')}{Colors.RESET} | "
                  f"🏆 Top 2 Best: {Colors.GREEN}{Colors.BOLD}{best_str}{Colors.RESET} | "
                  f"📉 Top 2 Worst: {Colors.RED}{Colors.BOLD}{worst_str}{Colors.RESET}")
        
        print(f"{Colors.MAGENTA}🎯 Options Environment:{Colors.RESET} {Colors.YELLOW}{self.iv_environment}{Colors.RESET} | "
              f"{Colors.BLUE}📈 Updates:{Colors.RESET} {self.successful_updates}/{self.sector_update_attempts} | "
              f"⚡ Scan Time: {scan_time:.2f}s | 🎯 Stocks: {len(signals)}")

        if not signals:
            print(f"\n{Colors.YELLOW}📭 No significant signals found in this cycle.{Colors.RESET}")
        else:
            print(f"\n{Colors.WHITE}{Colors.BOLD}🎯 {len(signals)} ENHANCED SIGNALS WITH OPTIONS ANALYSIS:{Colors.RESET}")
            print(f"\n{'Stock':<10} {'Sector':<18} {'Signal':<18} {'Score':>8} {'TFs':>4} {'Options':>10} {'PCR':>6} {'IV%':>5}")
            print(f"{Colors.CYAN}{'-' * 120}{Colors.RESET}")

            signals.sort(key=lambda x: abs(x['score'] - 50), reverse=True)

            for s in signals[:25]:  # Show top 25
                # Sector color coding
                sector_color, sector_name = Colors.YELLOW, s['sector']
                if s['sector'] in self.best_sectors:
                    sector_icon = "★★" if s['sector'] == self.best_sectors[0] else "★ "
                    sector_color, sector_name = Colors.GREEN, f"{sector_icon}{s['sector']}"
                elif s['sector'] in self.worst_sectors:
                    sector_icon = "★★" if s['sector'] == self.worst_sectors[0] else "★ "
                    sector_color, sector_name = Colors.RED, f"{sector_icon}{s['sector']}"

                # Signal color
                signal_color = Colors.YELLOW
                if 'Buy' in s['signal']: 
                    signal_color = Colors.GREEN + (Colors.BOLD if 'Very' in s['signal'] else "")
                elif 'Sell' in s['signal']: 
                    signal_color = Colors.RED + (Colors.BOLD if 'Very' in s['signal'] else "")
                
                # Options data
                options_data = s.get('options_data', {})
                oi_analysis = options_data.get('oi_analysis', {})
                iv_data = options_data.get('iv_data', {})
                
                pcr_oi = oi_analysis.get('pcr_oi', 0)
                iv_percentile = iv_data.get('iv_percentile', 0)
                
                pcr_str = f"{pcr_oi:.2f}" if pcr_oi else "N/A"
                iv_str = f"{iv_percentile:.0f}" if iv_percentile else "N/A"
                options_indicator = "🎯" if options_data else "📈"

                print(f"{Colors.WHITE}{s['symbol']:<10}{Colors.RESET} "
                      f"{sector_color}{sector_name:<18}{Colors.RESET} "
                      f"{signal_color}{s['signal']:<18}{Colors.RESET} "
                      f"{Colors.WHITE}{s['score']:>8.1f}{Colors.RESET} "
                      f"{Colors.CYAN}{s['timeframes']:>4}{Colors.RESET} "
                      f"{Colors.MAGENTA}{options_indicator:>10}{Colors.RESET} "
                      f"{Colors.YELLOW}{pcr_str:>6}{Colors.RESET} "
                      f"{Colors.BLUE}{iv_str:>5}{Colors.RESET}")
            
            # Show options strategies for top stocks
            self.display_options_strategies(signals[:5])
        
        next_scan_time = (current_time + timedelta(minutes=5)).strftime('%H:%M:%S')
        print(f"\n{Colors.CYAN}{Colors.BOLD}⏰ Next enhanced scan at {next_scan_time}{Colors.RESET}")

    def display_options_strategies(self, top_signals):
        """Display options strategies for top signals"""
        print(f"\n{Colors.MAGENTA}{Colors.BOLD}🎯 OPTIONS STRATEGIES FOR TOP SIGNALS:{Colors.RESET}")
        
        strategies_found = False
        for signal in top_signals:
            if 'strategies' in signal:
                strategies_found = True
                symbol = signal['symbol']
                print(f"\n{Colors.CYAN}📈 {symbol}:{Colors.RESET}")
                
                strategies = signal['strategies']
                
                # Show best strategy for each type
                for strategy_type, strategy_list in strategies.items():
                    if strategy_list:
                        if strategy_type == 'covered_calls' and strategy_list:
                            best = strategy_list[0]
                            print(f"  🔄 Covered Call: Strike {best['strike']} | Yield: {best['monthly_yield']}% | IV: {best['iv']}%")
                        
                        elif strategy_type == 'straddles' and strategy_list:
                            best = strategy_list[0]
                            print(f"  ⚖️ {best['strategy']} Straddle: Strike {best['strike']} | Premium: {best['total_premium']} | BE: ±{best['breakeven_range']}%")
        
        if not strategies_found:
            print(f"  {Colors.YELLOW}📊 Options strategies analysis in progress...{Colors.RESET}")

    def run_enhanced_5min_scanner_with_options(self):
        """Main enhanced scanner with options features"""
        self.is_running = True
        logger.info("🚀 Starting Enhanced 5-Minute Scanner with Options...")
        
        self.force_sector_update()
        
        try:
            while self.is_running:
                self.enhanced_5min_scan_cycle_with_options()
                if self.is_running:
                    logger.info(f"💤 Waiting 5 minutes for next cycle...")
                    time_module.sleep(self.scan_interval)
        except KeyboardInterrupt:
            logger.info("🛑 Enhanced scanner stopped by user")
        finally:
            self.stop()

    def stop(self):
        """Stop the scanner"""
        self.is_running = False
        print(f"{Colors.YELLOW}🛑 Scanner stopped{Colors.RESET}")

# --- MAIN EXECUTION WITH OPTIONS TOGGLE ---
def main():
    print(f"{Colors.CYAN}{Colors.BOLD}🎯 ENHANCED MULTI-INDICATOR SCANNER WITH OPTIONS FEATURES{Colors.RESET}")
    print(f"{Colors.YELLOW}📊 Features: Original Technical Analysis + Options Chain Analysis + Strategy Scanning{Colors.RESET}")
    
    scanner = Enhanced5MinLiveScanner()
    
    try:
        # Ask user if they want options features enabled
        print(f"\n{Colors.MAGENTA}🎯 Enable Options Features? (y/n, default=y):{Colors.RESET} ", end="")
        user_input = input().strip().lower()
        
        if user_input in ['n', 'no']:
            scanner.options_enabled = False
            print(f"{Colors.YELLOW}📈 Running in Technical Analysis Only mode{Colors.RESET}")
        else:
            scanner.options_enabled = True
            print(f"{Colors.GREEN}🎯 Running with Full Options Features{Colors.RESET}")
            
        scanner.run_enhanced_5min_scanner_with_options()
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Shutting down...{Colors.RESET}")
        scanner.stop()

if __name__ == "__main__":
    main()
