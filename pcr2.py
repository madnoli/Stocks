# NIFTY Live Intraday PCR Analyzer - Market Hours (9:15 AM - 3:30 PM)
# Real NSE Option Chain Data with Buy/Sell Signals

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import time as time_module
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import threading
import os
import schedule
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import pickle
warnings.filterwarnings('ignore')

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
    ORANGE = '\033[38;5;208m'

@dataclass
class MarketData:
    """Live market data structure"""
    time: str
    price: float
    diff: float
    pcr: float
    option_signal: str
    call_oi_total: int = 0
    put_oi_total: int = 0
    oi_change: float = 0.0
    volume_ratio: float = 0.0

@dataclass 
class LiveStrikeData:
    """Live strike data with OI changes"""
    strike: int
    call_oi: int
    put_oi: int
    call_oi_change: int
    put_oi_change: int
    call_volume: int
    put_volume: int
    call_ltp: float
    put_ltp: float
    call_iv: float
    put_iv: float
    position: str
    signal_strength: float = 0.0

class LiveNiftyAnalyzer:
    def __init__(self):
        self.live_data = []
        self.strike_history = {}
        self.current_price = 0.0
        self.previous_price = 0.0
        self.is_running = False
        self.market_session_active = False
        
        # Enhanced session with cookies and headers
        self.session = requests.Session()
        self.setup_session()
        
        # Strike parameters
        self.atm_strike = 25100
        self.strike_gap = 50
        self.target_strikes = []
        
        # Market timing
        self.market_start = time(9, 15)
        self.market_end = time(15, 30)
        self.scan_interval = 300  # 5 minutes
        
        # Signal calculation parameters
        self.signal_thresholds = {
            'strong_buy': {'pcr': 0.7, 'oi_buildup': 0.15},
            'buy': {'pcr': 0.8, 'oi_buildup': 0.10},
            'neutral': {'pcr': 1.0, 'oi_buildup': 0.05},
            'sell': {'pcr': 1.2, 'oi_buildup': -0.10},
            'strong_sell': {'pcr': 1.4, 'oi_buildup': -0.15}
        }
        
        print(f"{Colors.ORANGE}{Colors.BOLD}📊 LIVE NIFTY INTRADAY PCR ANALYZER{Colors.RESET}")
        print(f"{Colors.CYAN}⏰ Market Hours: 9:15 AM - 3:30 PM (Every 5 minutes){Colors.RESET}")
        print(f"{Colors.YELLOW}🎯 Real NSE Option Chain Data with Buy/Sell Signals{Colors.RESET}")
        
        self.initialize_live_system()

    def setup_session(self):
        """Setup enhanced session for NSE data"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin',
            'Referer': 'https://www.nseindia.com/option-chain'
        }
        self.session.headers.update(self.headers)
        
        # Initialize session with NSE homepage
        try:
            homepage = self.session.get('https://www.nseindia.com', timeout=10)
            print(f"✅ NSE Session initialized: {homepage.status_code}")
        except Exception as e:
            print(f"⚠️ NSE Session warning: {e}")

    def initialize_live_system(self):
        """Initialize live trading system"""
        try:
            print(f"\n{Colors.MAGENTA}🔧 INITIALIZING LIVE SYSTEM:{Colors.RESET}")
            
            # Get current price and setup strikes
            self.current_price = self.get_live_nifty_price()
            self.setup_target_strikes()
            
            # Load previous session data if available
            self.load_previous_session()
            
            # Test live data connection
            self.test_live_connection()
            
            print(f"✅ Live system ready for market hours")
            
        except Exception as e:
            print(f"❌ Error initializing live system: {e}")

    def get_live_nifty_price(self):
        """Get real-time NIFTY price from NSE"""
        try:
            # Method 1: NSE Indices API
            url = "https://www.nseindia.com/api/allIndices"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                for index in data.get('data', []):
                    if index.get('index') == 'NIFTY 50':
                        price = float(index.get('last', 0))
                        if price > 0:
                            self.previous_price = self.current_price
                            self.current_price = price
                            return price
            
            # Method 2: Option Chain API
            url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                price = data.get('records', {}).get('underlyingValue', 0)
                if price > 0:
                    self.previous_price = self.current_price
                    self.current_price = price
                    return price
            
            print(f"⚠️ Using fallback price calculation")
            return self.calculate_fallback_price()
            
        except Exception as e:
            print(f"❌ Error getting live price: {e}")
            return self.calculate_fallback_price()

    def calculate_fallback_price(self):
        """Calculate realistic fallback price"""
        base_price = self.current_price if self.current_price > 0 else 25100
        
        # Market hours-based movement
        now = datetime.now()
        if self.is_market_hours():
            # Simulate realistic intraday movement
            time_factor = (now.hour - 9) + (now.minute / 60.0)
            volatility = 0.2 + (0.3 * np.sin(time_factor * 0.7))
            change_pct = np.random.uniform(-volatility, volatility)
            new_price = base_price * (1 + change_pct / 100)
            
            self.previous_price = self.current_price
            self.current_price = new_price
            return new_price
        
        return base_price

    def setup_target_strikes(self):
        """Setup target strikes around ATM"""
        try:
            # Find ATM strike
            self.atm_strike = round(self.current_price / self.strike_gap) * self.strike_gap
            
            # Generate target strikes (ATM ± 5 strikes for better coverage)
            self.target_strikes = []
            for i in range(-5, 6):  # -5 to +5 strikes
                strike = self.atm_strike + (i * self.strike_gap)
                self.target_strikes.append(strike)
            
            print(f"🎯 ATM Strike: {Colors.WHITE}{Colors.BOLD}{self.atm_strike}{Colors.RESET}")
            print(f"📊 Target Strikes: {len(self.target_strikes)} strikes ({min(self.target_strikes)} to {max(self.target_strikes)})")
            
        except Exception as e:
            print(f"❌ Error setting up strikes: {e}")

    def fetch_live_option_chain(self):
        """Fetch live option chain data from NSE"""
        try:
            url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
            
            # Add current timestamp to avoid caching
            params = {'timestamp': int(time_module.time() * 1000)}
            
            response = self.session.get(url, params=params, timeout=20)
            
            if response.status_code == 200:
                data = response.json()
                
                # Update current price
                records = data.get('records', {})
                underlying_value = records.get('underlyingValue', 0)
                if underlying_value > 0:
                    self.previous_price = self.current_price
                    self.current_price = underlying_value
                
                return data
            else:
                print(f"⚠️ NSE API returned status: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching live option chain: {e}")
            return None

    def parse_live_option_data(self, nse_data):
        """Parse live NSE option chain data"""
        try:
            if not nse_data:
                return {}
            
            records = nse_data.get('records', {})
            option_data = records.get('data', [])
            
            live_strikes = {}
            
            for item in option_data:
                strike_price = item.get('strikePrice', 0)
                
                if strike_price in self.target_strikes:
                    # Call data
                    ce_data = item.get('CE', {})
                    call_oi = ce_data.get('openInterest', 0)
                    call_oi_change = ce_data.get('changeinOpenInterest', 0)
                    call_volume = ce_data.get('totalTradedVolume', 0)
                    call_ltp = ce_data.get('lastPrice', 0)
                    call_iv = ce_data.get('impliedVolatility', 0)
                    
                    # Put data
                    pe_data = item.get('PE', {})
                    put_oi = pe_data.get('openInterest', 0)
                    put_oi_change = pe_data.get('changeinOpenInterest', 0)
                    put_volume = pe_data.get('totalTradedVolume', 0)
                    put_ltp = pe_data.get('lastPrice', 0)
                    put_iv = pe_data.get('impliedVolatility', 0)
                    
                    # Determine position
                    if strike_price == self.atm_strike:
                        position = "ATM"
                    elif strike_price > self.atm_strike:
                        position = "Above"
                    else:
                        position = "Below"
                    
                    # Calculate signal strength
                    signal_strength = self.calculate_signal_strength(
                        call_oi, put_oi, call_oi_change, put_oi_change, 
                        call_volume, put_volume, position
                    )
                    
                    live_strikes[strike_price] = LiveStrikeData(
                        strike=strike_price,
                        call_oi=call_oi,
                        put_oi=put_oi,
                        call_oi_change=call_oi_change,
                        put_oi_change=put_oi_change,
                        call_volume=call_volume,
                        put_volume=put_volume,
                        call_ltp=call_ltp,
                        put_ltp=put_ltp,
                        call_iv=call_iv,
                        put_iv=put_iv,
                        position=position,
                        signal_strength=signal_strength
                    )
            
            return live_strikes
            
        except Exception as e:
            print(f"❌ Error parsing live option data: {e}")
            return {}

    def calculate_signal_strength(self, call_oi, put_oi, call_oi_change, put_oi_change, call_volume, put_volume, position):
        """Calculate signal strength based on OI changes and volumes"""
        try:
            # Base strength from OI changes
            total_oi_change = abs(call_oi_change) + abs(put_oi_change)
            oi_strength = min(1.0, total_oi_change / 10000)  # Normalize to 0-1
            
            # Volume strength
            total_volume = call_volume + put_volume
            volume_strength = min(1.0, total_volume / 50000)  # Normalize to 0-1
            
            # Position strength (ATM gets higher weight)
            position_multiplier = 1.5 if position == "ATM" else 1.0
            
            # Combined strength
            signal_strength = (oi_strength * 0.6 + volume_strength * 0.4) * position_multiplier
            
            return round(signal_strength, 3)
            
        except Exception as e:
            return 0.0

    def calculate_live_pcr_and_signals(self, live_strikes):
        """Calculate PCR and generate buy/sell signals from live data"""
        try:
            if not live_strikes:
                return None
            
            # Calculate overall PCR
            total_call_oi = sum(data.call_oi for data in live_strikes.values())
            total_put_oi = sum(data.put_oi for data in live_strikes.values())
            
            if total_call_oi == 0:
                return None
            
            overall_pcr = round(total_put_oi / total_call_oi, 2)
            
            # Calculate OI changes
            total_call_oi_change = sum(data.call_oi_change for data in live_strikes.values())
            total_put_oi_change = sum(data.put_oi_change for data in live_strikes.values())
            
            # OI change ratio
            total_oi = total_call_oi + total_put_oi
            oi_change_ratio = (total_put_oi_change - total_call_oi_change) / total_oi if total_oi > 0 else 0
            
            # Price difference
            price_diff = self.current_price - self.previous_price if self.previous_price > 0 else 0
            
            # Generate enhanced buy/sell signal
            option_signal = self.generate_enhanced_signal(overall_pcr, oi_change_ratio, price_diff, live_strikes)
            
            # Volume ratio
            total_call_volume = sum(data.call_volume for data in live_strikes.values())
            total_put_volume = sum(data.put_volume for data in live_strikes.values())
            volume_ratio = total_put_volume / total_call_volume if total_call_volume > 0 else 0
            
            return MarketData(
                time=datetime.now().strftime("%H%M"),
                price=self.current_price,
                diff=round(price_diff, 2),
                pcr=overall_pcr,
                option_signal=option_signal,
                call_oi_total=total_call_oi,
                put_oi_total=total_put_oi,
                oi_change=round(oi_change_ratio * 100, 2),
                volume_ratio=round(volume_ratio, 2)
            )
            
        except Exception as e:
            print(f"❌ Error calculating PCR and signals: {e}")
            return None

    def generate_enhanced_signal(self, pcr, oi_change_ratio, price_diff, live_strikes):
        """Generate enhanced buy/sell signals based on multiple factors"""
        try:
            signal_score = 0
            
            # PCR-based scoring
            if pcr <= 0.7:
                signal_score += 3  # Very bullish
            elif pcr <= 0.8:
                signal_score += 2  # Bullish
            elif pcr <= 1.0:
                signal_score += 1  # Slightly bullish
            elif pcr <= 1.2:
                signal_score -= 1  # Slightly bearish
            elif pcr <= 1.4:
                signal_score -= 2  # Bearish
            else:
                signal_score -= 3  # Very bearish
            
            # OI change-based scoring
            if oi_change_ratio > 0.15:
                signal_score += 2  # Strong put buildup (bearish for market)
            elif oi_change_ratio > 0.05:
                signal_score += 1  # Moderate put buildup
            elif oi_change_ratio < -0.15:
                signal_score -= 2  # Strong call buildup (bullish for market)
            elif oi_change_ratio < -0.05:
                signal_score -= 1  # Moderate call buildup
            
            # Price movement scoring
            if price_diff > 20:
                signal_score += 1  # Price rising
            elif price_diff < -20:
                signal_score -= 1  # Price falling
            
            # ATM strike analysis
            atm_data = live_strikes.get(self.atm_strike)
            if atm_data:
                atm_pcr = atm_data.put_oi / atm_data.call_oi if atm_data.call_oi > 0 else 1
                if atm_pcr > 1.5:
                    signal_score += 1  # ATM puts building up
                elif atm_pcr < 0.7:
                    signal_score -= 1  # ATM calls building up
            
            # Convert score to signal
            if signal_score >= 4:
                return "STRONG BUY"
            elif signal_score >= 2:
                return "BUY"
            elif signal_score >= 1:
                return "WEAK BUY"
            elif signal_score <= -4:
                return "STRONG SELL"
            elif signal_score <= -2:
                return "SELL"
            elif signal_score <= -1:
                return "WEAK SELL"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            print(f"❌ Error generating signal: {e}")
            return "NEUTRAL"

    def is_market_hours(self):
        """Check if current time is within market hours"""
        now = datetime.now()
        current_time = now.time()
        
        # Check if it's a weekday and within market hours
        if now.weekday() < 5:  # Monday = 0, Friday = 4
            return self.market_start <= current_time <= self.market_end
        
        return False

    def display_live_data(self):
        """Display live data in the format from your image"""
        try:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"{Colors.ORANGE}{Colors.BOLD}📊 INTRADAY DATA - NIFTY 💹{Colors.RESET}")
            print(f"{Colors.ORANGE}{'='*80}{Colors.RESET}")
            print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST | 💹 Live: {self.current_price:.2f}")
            
            # Timeframe tabs
            print(f"\n{Colors.ORANGE}{Colors.BOLD}[ 5 mins ]{Colors.RESET}  "
                  f"{Colors.WHITE}[ 15 mins ]{Colors.RESET}  "
                  f"{Colors.WHITE}[ 30 mins ]{Colors.RESET}  "
                  f"{Colors.BLUE}Show more{Colors.RESET}")
            
            print(f"\n{Colors.WHITE}{Colors.BOLD}{'Time':<8} {'Diff':<12} {'PCR':<8} {'Option Signal':<15} {'OI Chg%':<8}{Colors.RESET}")
            print(f"{Colors.ORANGE}{'-'*60}{Colors.RESET}")
            
            # Display recent data (last 20 records)
            recent_data = self.live_data[-20:] if len(self.live_data) >= 20 else self.live_data
            
            for data in recent_data:
                # Color coding
                diff_color = Colors.GREEN if data.diff >= 0 else Colors.RED
                diff_str = f"{data.diff:+.2f}"
                
                # Signal colors
                if "BUY" in data.option_signal:
                    signal_color = Colors.GREEN + Colors.BOLD
                elif "SELL" in data.option_signal:
                    signal_color = Colors.RED + Colors.BOLD
                else:
                    signal_color = Colors.YELLOW
                
                # OI change color
                oi_color = Colors.RED if data.oi_change > 0 else Colors.GREEN
                
                print(f"{Colors.YELLOW}{data.time:<8}{Colors.RESET} "
                      f"{diff_color}{diff_str:<12}{Colors.RESET} "
                      f"{Colors.CYAN}{data.pcr:<8}{Colors.RESET} "
                      f"{signal_color}{data.option_signal:<15}{Colors.RESET} "
                      f"{oi_color}{data.oi_change:+.1f}%{Colors.RESET}")
            
            # Display current strike analysis
            self.display_live_strike_analysis()
            
            # Display summary
            if self.live_data:
                latest = self.live_data[-1]
                print(f"\n{Colors.CYAN}📊 CURRENT ANALYSIS:{Colors.RESET}")
                print(f"🎯 Signal: {Colors.GREEN if 'BUY' in latest.option_signal else Colors.RED if 'SELL' in latest.option_signal else Colors.YELLOW}{Colors.BOLD}{latest.option_signal}{Colors.RESET}")
                print(f"📊 PCR: {Colors.MAGENTA}{Colors.BOLD}{latest.pcr}{Colors.RESET}")
                print(f"🔄 OI Change: {Colors.RED if latest.oi_change > 0 else Colors.GREEN}{latest.oi_change:+.1f}%{Colors.RESET}")
                print(f"📈 Total OI: {Colors.CYAN}{(latest.call_oi_total + latest.put_oi_total):,}{Colors.RESET}")
            
        except Exception as e:
            print(f"❌ Error displaying live data: {e}")

    def display_live_strike_analysis(self):
        """Display live strike analysis"""
        try:
            if not hasattr(self, 'current_strikes') or not self.current_strikes:
                return
            
            print(f"\n{Colors.MAGENTA}{Colors.BOLD}🎯 LIVE STRIKE ANALYSIS (Real OI Data):{Colors.RESET}")
            print(f"{Colors.WHITE}{'Strike':<8} {'Pos':<6} {'C-OI':<8} {'P-OI':<8} {'C-Chg':<8} {'P-Chg':<8} {'PCR':<6} {'Signal':<8}{Colors.RESET}")
            print(f"{Colors.MAGENTA}{'-'*70}{Colors.RESET}")
            
            # Sort strikes
            sorted_strikes = sorted(self.current_strikes.keys(), reverse=True)
            
            for strike in sorted_strikes:
                data = self.current_strikes[strike]
                
                if not data:
                    continue
                
                # Strike PCR
                strike_pcr = round(data.put_oi / data.call_oi, 2) if data.call_oi > 0 else 0
                
                # Position color
                if data.position == "ATM":
                    pos_color = Colors.WHITE + Colors.BOLD
                elif data.position == "Above":
                    pos_color = Colors.GREEN
                else:
                    pos_color = Colors.RED
                
                # OI change colors
                call_chg_color = Colors.GREEN if data.call_oi_change > 0 else Colors.RED
                put_chg_color = Colors.GREEN if data.put_oi_change > 0 else Colors.RED
                
                # Generate strike signal
                if data.position == "ATM":
                    if strike_pcr > 1.3:
                        strike_signal = "BEARISH"
                        signal_color = Colors.RED
                    elif strike_pcr < 0.8:
                        strike_signal = "BULLISH"
                        signal_color = Colors.GREEN
                    else:
                        strike_signal = "NEUTRAL"
                        signal_color = Colors.YELLOW
                else:
                    if data.put_oi_change > data.call_oi_change:
                        strike_signal = "PUT"
                        signal_color = Colors.RED
                    elif data.call_oi_change > data.put_oi_change:
                        strike_signal = "CALL"
                        signal_color = Colors.GREEN
                    else:
                        strike_signal = "FLAT"
                        signal_color = Colors.YELLOW
                
                print(f"{pos_color}{strike:<8}{Colors.RESET} "
                      f"{pos_color}{data.position:<6}{Colors.RESET} "
                      f"{Colors.CYAN}{data.call_oi//1000:>5}K{Colors.RESET} "
                      f"{Colors.MAGENTA}{data.put_oi//1000:>5}K{Colors.RESET} "
                      f"{call_chg_color}{data.call_oi_change:>+6}{Colors.RESET} "
                      f"{put_chg_color}{data.put_oi_change:>+6}{Colors.RESET} "
                      f"{Colors.WHITE}{strike_pcr:<6}{Colors.RESET} "
                      f"{signal_color}{strike_signal:<8}{Colors.RESET}")
            
        except Exception as e:
            print(f"❌ Error displaying strike analysis: {e}")

    def save_live_data(self):
        """Save live data to CSV with timestamp"""
        try:
            if not self.live_data:
                return
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            
            # Save main data
            df_main = pd.DataFrame([
                {
                    'Time': data.time,
                    'Price': data.price,
                    'Diff': data.diff,
                    'PCR': data.pcr,
                    'Option_Signal': data.option_signal,
                    'Call_OI_Total': data.call_oi_total,
                    'Put_OI_Total': data.put_oi_total,
                    'OI_Change_Pct': data.oi_change,
                    'Volume_Ratio': data.volume_ratio,
                    'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                for data in self.live_data
            ])
            
            filename_main = f'nifty_live_5min_{timestamp}.csv'
            df_main.to_csv(filename_main, index=False)
            print(f"💾 Live data saved: {filename_main}")
            
            # Save strike data if available
            if hasattr(self, 'current_strikes') and self.current_strikes:
                df_strikes = pd.DataFrame([
                    {
                        'Strike': strike,
                        'Position': data.position,
                        'Call_OI': data.call_oi,
                        'Put_OI': data.put_oi,
                        'Call_OI_Change': data.call_oi_change,
                        'Put_OI_Change': data.put_oi_change,
                        'Call_Volume': data.call_volume,
                        'Put_Volume': data.put_volume,
                        'Call_LTP': data.call_ltp,
                        'Put_LTP': data.put_ltp,
                        'Call_IV': data.call_iv,
                        'Put_IV': data.put_iv,
                        'Strike_PCR': round(data.put_oi / data.call_oi, 2) if data.call_oi > 0 else 0,
                        'Signal_Strength': data.signal_strength,
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    for strike, data in self.current_strikes.items()
                ])
                
                filename_strikes = f'nifty_strikes_5min_{timestamp}.csv'
                df_strikes.to_csv(filename_strikes, index=False)
                print(f"💾 Strike data saved: {filename_strikes}")
                
        except Exception as e:
            print(f"❌ Error saving data: {e}")

    def load_previous_session(self):
        """Load previous session data if available"""
        try:
            today = datetime.now().strftime('%Y%m%d')
            session_file = f'nifty_session_{today}.pkl'
            
            if os.path.exists(session_file):
                with open(session_file, 'rb') as f:
                    session_data = pickle.load(f)
                    self.live_data = session_data.get('live_data', [])
                    self.strike_history = session_data.get('strike_history', {})
                
                print(f"📂 Loaded previous session: {len(self.live_data)} records")
            
        except Exception as e:
            print(f"⚠️ Could not load previous session: {e}")

    def save_session(self):
        """Save current session data"""
        try:
            today = datetime.now().strftime('%Y%m%d')
            session_file = f'nifty_session_{today}.pkl'
            
            session_data = {
                'live_data': self.live_data,
                'strike_history': self.strike_history,
                'last_update': datetime.now()
            }
            
            with open(session_file, 'wb') as f:
                pickle.dump(session_data, f)
                
        except Exception as e:
            print(f"⚠️ Could not save session: {e}")

    def test_live_connection(self):
        """Test live data connection"""
        print(f"\n{Colors.BLUE}🔍 TESTING LIVE CONNECTION:{Colors.RESET}")
        
        try:
            # Test NSE connection
            test_data = self.fetch_live_option_chain()
            
            if test_data:
                print(f"✅ NSE Option Chain: {Colors.GREEN}CONNECTED{Colors.RESET}")
                
                records = test_data.get('records', {})
                data_count = len(records.get('data', []))
                print(f"📊 Option Strikes Available: {data_count}")
                
                # Test data parsing
                parsed_data = self.parse_live_option_data(test_data)
                print(f"📈 Parsed Strikes: {len(parsed_data)}")
                
                if parsed_data:
                    print(f"✅ Data parsing: {Colors.GREEN}WORKING{Colors.RESET}")
                else:
                    print(f"⚠️ Data parsing: {Colors.YELLOW}LIMITED{Colors.RESET}")
                
            else:
                print(f"❌ NSE Option Chain: {Colors.RED}FAILED{Colors.RESET}")
                print(f"🔄 Will use fallback data during market hours")
                
        except Exception as e:
            print(f"❌ Connection test error: {e}")

    def run_market_hours_analysis(self):
        """Main function to run analysis during market hours (9:15 AM - 3:30 PM)"""
        self.is_running = True
        print(f"\n{Colors.GREEN}🚀 STARTING LIVE MARKET ANALYSIS{Colors.RESET}")
        print(f"{Colors.YELLOW}⏰ Monitoring: 9:15 AM - 3:30 PM (Every 5 minutes){Colors.RESET}")
        print(f"{Colors.CYAN}📊 Real NSE Option Chain Data{Colors.RESET}")
        print(f"{Colors.MAGENTA}💡 Press Ctrl+C to stop{Colors.RESET}")
        
        try:
            cycle_count = 0
            
            while self.is_running:
                current_time = datetime.now()
                
                if self.is_market_hours():
                    if not self.market_session_active:
                        print(f"\n{Colors.GREEN}🔔 MARKET OPENED - Starting data collection{Colors.RESET}")
                        self.market_session_active = True
                    
                    cycle_count += 1
                    print(f"\n{Colors.CYAN}📊 Cycle #{cycle_count} - {current_time.strftime('%H:%M:%S')}{Colors.RESET}")
                    
                    # Fetch live option chain data
                    option_chain_data = self.fetch_live_option_chain()
                    
                    if option_chain_data:
                        # Parse strike data
                        live_strikes = self.parse_live_option_data(option_chain_data)
                        self.current_strikes = live_strikes
                        
                        # Calculate PCR and signals
                        market_data = self.calculate_live_pcr_and_signals(live_strikes)
                        
                        if market_data:
                            self.live_data.append(market_data)
                            
                            # Keep last 100 records
                            if len(self.live_data) > 100:
                                self.live_data = self.live_data[-100:]
                            
                            # Display updated data
                            self.display_live_data()
                            
                            # Auto-save every 10 cycles (50 minutes)
                            if cycle_count % 10 == 0:
                                self.save_live_data()
                                self.save_session()
                        else:
                            print(f"⚠️ Failed to calculate market data")
                    else:
                        print(f"⚠️ Failed to fetch option chain data")
                
                else:
                    if self.market_session_active:
                        print(f"\n{Colors.YELLOW}🔔 MARKET CLOSED - Saving final data{Colors.RESET}")
                        self.save_live_data()
                        self.save_session()
                        self.market_session_active = False
                    
                    next_market_open = datetime.combine(datetime.now().date(), self.market_start)
                    if datetime.now().time() > self.market_end:
                        next_market_open += timedelta(days=1)
                    
                    print(f"\n🕐 Market is closed. Next session: {next_market_open.strftime('%Y-%m-%d %H:%M')}")
                    print(f"📊 Today's Records: {len(self.live_data)}")
                
                # Wait for 5 minutes
                print(f"⏳ Next update in 5 minutes...")
                time_module.sleep(300)  # 5 minutes
                
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}🛑 Analysis stopped by user{Colors.RESET}")
            
        except Exception as e:
            print(f"\n❌ Error in analysis loop: {e}")
            
        finally:
            self.stop_analysis()

    def stop_analysis(self):
        """Stop analysis and save data"""
        self.is_running = False
        print(f"\n{Colors.CYAN}📊 FINAL SUMMARY:{Colors.RESET}")
        print(f"🔢 Total Records: {len(self.live_data)}")
        
        if self.live_data:
            print(f"🕐 Session Duration: {self.live_data[0].time} - {self.live_data[-1].time}")
            
            # Calculate session statistics
            buy_signals = sum(1 for d in self.live_data if 'BUY' in d.option_signal)
            sell_signals = sum(1 for d in self.live_data if 'SELL' in d.option_signal)
            neutral_signals = len(self.live_data) - buy_signals - sell_signals
            
            print(f"📈 Buy Signals: {Colors.GREEN}{buy_signals}{Colors.RESET}")
            print(f"📉 Sell Signals: {Colors.RED}{sell_signals}{Colors.RESET}")
            print(f"📊 Neutral Signals: {Colors.YELLOW}{neutral_signals}{Colors.RESET}")
            
            # Save final data
            self.save_live_data()
            self.save_session()
        
        print(f"{Colors.GREEN}✅ Live NIFTY Analysis completed successfully{Colors.RESET}")

def main():
    """Main function"""
    print(f"{Colors.ORANGE}{Colors.BOLD}📊 LIVE NIFTY INTRADAY PCR ANALYZER{Colors.RESET}")
    print(f"{Colors.CYAN}⏰ Market Hours Analysis: 9:15 AM - 3:30 PM (Every 5 minutes){Colors.RESET}")
    print(f"{Colors.YELLOW}🎯 Real NSE Option Chain Data with Enhanced Buy/Sell Signals{Colors.RESET}")
    
    analyzer = LiveNiftyAnalyzer()
    
    try:
        print(f"\n{Colors.MAGENTA}Select Mode:{Colors.RESET}")
        print(f"1. Live Market Analysis (9:15 AM - 3:30 PM, Every 5 min)")
        print(f"2. Current Market Snapshot (One-time)")
        print(f"3. Test Connection & Display Current Data")
        
        choice = input(f"{Colors.CYAN}Enter choice (1-3, default=1): {Colors.RESET}").strip()
        
        if choice == "2":
            # Single snapshot
            print(f"\n{Colors.YELLOW}📊 Getting current market snapshot...{Colors.RESET}")
            
            if analyzer.is_market_hours():
                option_data = analyzer.fetch_live_option_chain()
                if option_data:
                    live_strikes = analyzer.parse_live_option_data(option_data)
                    analyzer.current_strikes = live_strikes
                    
                    market_data = analyzer.calculate_live_pcr_and_signals(live_strikes)
                    if market_data:
                        analyzer.live_data.append(market_data)
                        analyzer.display_live_data()
                        
                        save_choice = input(f"\n{Colors.CYAN}Save snapshot to CSV? (y/n): {Colors.RESET}").strip().lower()
                        if save_choice == 'y':
                            analyzer.save_live_data()
                    else:
                        print(f"❌ Failed to get market data")
                else:
                    print(f"❌ Failed to fetch option chain")
            else:
                print(f"🕐 Market is currently closed")
                
        elif choice == "3":
            # Test and display
            print(f"\n{Colors.YELLOW}🔍 Testing connection and displaying current data...{Colors.RESET}")
            analyzer.test_live_connection()
            
            # Try to get current data
            option_data = analyzer.fetch_live_option_chain()
            if option_data:
                live_strikes = analyzer.parse_live_option_data(option_data)
                analyzer.current_strikes = live_strikes
                
                print(f"\n{Colors.GREEN}✅ Successfully fetched live data{Colors.RESET}")
                print(f"📊 Current NIFTY Price: {analyzer.current_price:.2f}")
                print(f"🎯 Strikes Analyzed: {len(live_strikes)}")
                
                # Display current analysis
                analyzer.display_live_strike_analysis()
            else:
                print(f"❌ Could not fetch current data")
        
        else:
            # Live analysis (default)
            analyzer.run_market_hours_analysis()
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Program interrupted by user{Colors.RESET}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
    finally:
        if analyzer.is_running:
            analyzer.stop_analysis()

if __name__ == "__main__":
    main()
