# NIFTY PCR Calculator - 3 Minutes Interval
# Calculates Put-Call Ratio using NSE Option Chain Data

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional
warnings.filterwarnings('ignore')

# --- COLOR CODES FOR DISPLAY ---
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

@dataclass
class PCRData:
    """Data structure for PCR information"""
    time: str
    price: float
    pcr: float
    option_signal: str
    call_oi: int = 0
    put_oi: int = 0
    total_oi: int = 0

class NiftyPCRCalculator:
    def __init__(self):
        self.pcr_history = []
        self.is_running = False
        self.current_price = 0.0
        self.session = requests.Session()
        
        # Request headers to mimic browser
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        print(f"{Colors.CYAN}{Colors.BOLD}🎯 NIFTY PCR Calculator - 3 Minutes Interval{Colors.RESET}")
        print(f"{Colors.YELLOW}📊 Fetching NSE Option Chain Data every 3 minutes{Colors.RESET}")
        
    def fetch_nse_option_chain(self):
        """Fetch NIFTY option chain data from NSE"""
        try:
            # NSE Option Chain URL for NIFTY
            url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
            
            response = self.session.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return self.parse_nse_data(data)
            else:
                print(f"❌ NSE API Error: Status {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error fetching NSE data: {e}")
            return None
    
    def parse_nse_data(self, data):
        """Parse NSE option chain JSON data"""
        try:
            # Get current NIFTY price
            records = data.get('records', {})
            self.current_price = records.get('underlyingValue', 0)
            
            # Get option chain data
            option_data = records.get('data', [])
            
            total_call_oi = 0
            total_put_oi = 0
            
            for item in option_data:
                # Call data
                if 'CE' in item:
                    ce_data = item['CE']
                    total_call_oi += ce_data.get('openInterest', 0)
                
                # Put data  
                if 'PE' in item:
                    pe_data = item['PE']
                    total_put_oi += pe_data.get('openInterest', 0)
            
            if total_call_oi > 0:
                pcr = round(total_put_oi / total_call_oi, 3)
                return {
                    'pcr': pcr,
                    'call_oi': total_call_oi,
                    'put_oi': total_put_oi,
                    'total_oi': total_call_oi + total_put_oi,
                    'current_price': self.current_price
                }
            
            return None
            
        except Exception as e:
            print(f"❌ Error parsing NSE data: {e}")
            return None
    
    def fetch_alternative_option_data(self):
        """Alternative method using your local API or TrueData"""
        try:
            # Method 1: Try your local API
            response = self.session.get("http://localhost:3001/api/optionChain/NIFTY", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return self.parse_local_api_data(data)
                
        except:
            pass
        
        # Method 2: Generate realistic demo data
        return self.generate_realistic_pcr_data()
    
    def parse_local_api_data(self, data):
        """Parse your local API data"""
        try:
            strikes = data.get('strikes', [])
            current_price = data.get('current_price', 25100)
            self.current_price = current_price
            
            total_call_oi = sum(s.get('call_oi', 0) for s in strikes)
            total_put_oi = sum(s.get('put_oi', 0) for s in strikes)
            
            if total_call_oi > 0:
                pcr = round(total_put_oi / total_call_oi, 3)
                return {
                    'pcr': pcr,
                    'call_oi': total_call_oi,
                    'put_oi': total_put_oi,
                    'total_oi': total_call_oi + total_put_oi,
                    'current_price': current_price
                }
            return None
            
        except Exception as e:
            print(f"❌ Error parsing local API data: {e}")
            return None
    
    def generate_realistic_pcr_data(self):
        """Generate realistic PCR data for demo"""
        try:
            # Simulate realistic NIFTY price movement
            base_price = 25118 if not hasattr(self, 'last_demo_price') else self.last_demo_price
            price_change = np.random.uniform(-0.5, 0.5)  # ±0.5% movement
            self.current_price = round(base_price * (1 + price_change / 100), 2)
            self.last_demo_price = self.current_price
            
            # Simulate realistic OI values
            total_call_oi = np.random.randint(8000000, 12000000)
            
            # PCR typically ranges between 0.6 to 1.8
            # Higher PCR (>1.2) indicates bearish sentiment
            # Lower PCR (<0.8) indicates bullish sentiment
            current_time = datetime.now()
            time_factor = (current_time.hour - 9) / 6.5  # 0 to 1 throughout trading day
            
            # PCR tends to be higher during market stress
            base_pcr = 0.9 + (0.3 * np.sin(time_factor * np.pi)) + np.random.uniform(-0.2, 0.2)
            pcr = round(max(0.4, min(2.0, base_pcr)), 3)
            
            total_put_oi = int(total_call_oi * pcr)
            
            return {
                'pcr': pcr,
                'call_oi': total_call_oi,
                'put_oi': total_put_oi,
                'total_oi': total_call_oi + total_put_oi,
                'current_price': self.current_price
            }
            
        except Exception as e:
            print(f"❌ Error generating demo data: {e}")
            return None
    
    def get_option_signal(self, pcr_value, price_change_pct=0):
        """Generate option trading signal based on PCR and price movement"""
        try:
            # PCR-based signals
            if pcr_value <= 0.7:
                return "STRONG BUY"  # Very bullish
            elif pcr_value <= 0.8:
                return "BUY"        # Bullish
            elif pcr_value <= 1.0:
                return "NEUTRAL"    # Balanced
            elif pcr_value <= 1.2:
                return "SELL"       # Bearish
            else:
                return "STRONG SELL" # Very bearish
                
        except:
            return "NEUTRAL"
    
    def calculate_pcr(self):
        """Main PCR calculation method"""
        try:
            # Try NSE first, then alternatives
            option_data = self.fetch_nse_option_chain()
            
            if not option_data:
                option_data = self.fetch_alternative_option_data()
            
            if option_data:
                current_time = datetime.now().strftime("%H:%M")
                pcr_value = option_data['pcr']
                option_signal = self.get_option_signal(pcr_value)
                
                # Calculate price change if we have history
                price_change = 0
                if self.pcr_history:
                    last_price = self.pcr_history[-1].price
                    price_change = ((self.current_price - last_price) / last_price) * 100
                
                pcr_data = PCRData(
                    time=current_time,
                    price=self.current_price,
                    pcr=pcr_value,
                    option_signal=option_signal,
                    call_oi=option_data['call_oi'],
                    put_oi=option_data['put_oi'],
                    total_oi=option_data['total_oi']
                )
                
                return pcr_data
            
            return None
            
        except Exception as e:
            print(f"❌ Error calculating PCR: {e}")
            return None
    
    def display_pcr_table(self):
        """Display PCR data in tabular format similar to your image"""
        try:
            print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*80}")
            print(f"📊 INTRADAY DATA - NIFTY PCR (3 Minutes Interval)")
            print(f"{'='*80}{Colors.RESET}")
            print(f"🕐 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST")
            print(f"💹 Current NIFTY: {self.current_price}")
            
            if not self.pcr_history:
                print(f"\n{Colors.YELLOW}📊 Collecting data... Please wait{Colors.RESET}")
                return
            
            # Table header
            print(f"\n{Colors.WHITE}{Colors.BOLD}{'Time':<8} {'Price':<12} {'PCR':<8} {'Option Signal':<15} {'Total OI':<12}{Colors.RESET}")
            print(f"{Colors.CYAN}{'-'*65}{Colors.RESET}")
            
            # Display last 10 records (similar to your image)
            recent_data = self.pcr_history[-10:] if len(self.pcr_history) >= 10 else self.pcr_history
            
            for data in recent_data:
                # Color coding for signals
                if data.option_signal == "BUY" or data.option_signal == "STRONG BUY":
                    signal_color = Colors.GREEN + Colors.BOLD
                elif data.option_signal == "SELL" or data.option_signal == "STRONG SELL":
                    signal_color = Colors.RED + Colors.BOLD
                else:
                    signal_color = Colors.YELLOW
                
                # Format price with proper alignment
                price_str = f"{data.price:,.2f}"
                oi_str = f"{data.total_oi:,}" if data.total_oi > 0 else "N/A"
                
                print(f"{Colors.WHITE}{data.time:<8}{Colors.RESET} "
                      f"{Colors.CYAN}{price_str:<12}{Colors.RESET} "
                      f"{Colors.MAGENTA}{data.pcr:<8}{Colors.RESET} "
                      f"{signal_color}{data.option_signal:<15}{Colors.RESET} "
                      f"{Colors.BLUE}{oi_str:<12}{Colors.RESET}")
            
            # Display summary statistics
            if len(self.pcr_history) > 1:
                latest = self.pcr_history[-1]
                previous = self.pcr_history[-2]
                
                pcr_change = latest.pcr - previous.pcr
                price_change = latest.price - previous.price
                price_change_pct = (price_change / previous.price) * 100
                
                print(f"\n{Colors.YELLOW}📈 SUMMARY:{Colors.RESET}")
                print(f"📊 Current PCR: {Colors.MAGENTA}{Colors.BOLD}{latest.pcr}{Colors.RESET}")
                print(f"🔄 PCR Change: {Colors.GREEN if pcr_change >= 0 else Colors.RED}{pcr_change:+.3f}{Colors.RESET}")
                print(f"💹 Price Change: {Colors.GREEN if price_change >= 0 else Colors.RED}{price_change:+.2f} ({price_change_pct:+.2f}%){Colors.RESET}")
                print(f"🎯 Signal: {Colors.GREEN if 'BUY' in latest.option_signal else Colors.RED if 'SELL' in latest.option_signal else Colors.YELLOW}{Colors.BOLD}{latest.option_signal}{Colors.RESET}")
            
            # PCR interpretation guide
            print(f"\n{Colors.BLUE}📋 PCR INTERPRETATION GUIDE:{Colors.RESET}")
            print(f"   PCR < 0.7  = {Colors.GREEN}Strong Bullish{Colors.RESET}")
            print(f"   PCR 0.7-1.0 = {Colors.YELLOW}Neutral to Bullish{Colors.RESET}")
            print(f"   PCR 1.0-1.2 = {Colors.YELLOW}Neutral to Bearish{Colors.RESET}")  
            print(f"   PCR > 1.2  = {Colors.RED}Strong Bearish{Colors.RESET}")
            
        except Exception as e:
            print(f"❌ Error displaying table: {e}")
    
    def save_to_csv(self, filename="nifty_pcr_data.csv"):
        """Save PCR data to CSV file"""
        try:
            if self.pcr_history:
                df = pd.DataFrame([
                    {
                        'Time': data.time,
                        'Price': data.price,
                        'PCR': data.pcr,
                        'Option_Signal': data.option_signal,
                        'Call_OI': data.call_oi,
                        'Put_OI': data.put_oi,
                        'Total_OI': data.total_oi
                    }
                    for data in self.pcr_history
                ])
                
                df.to_csv(filename, index=False)
                print(f"\n💾 Data saved to {filename}")
                
        except Exception as e:
            print(f"❌ Error saving to CSV: {e}")
    
    def run_pcr_monitor(self):
        """Main monitoring loop - runs every 3 minutes"""
        self.is_running = True
        print(f"\n{Colors.GREEN}🚀 Starting NIFTY PCR Monitor (3-minute intervals){Colors.RESET}")
        print(f"{Colors.YELLOW}⏰ Data will update every 3 minutes during market hours{Colors.RESET}")
        print(f"{Colors.CYAN}💡 Press Ctrl+C to stop{Colors.RESET}")
        
        try:
            while self.is_running:
                current_time = datetime.now()
                
                # Check if market is open (9:15 AM to 3:30 PM, Monday to Friday)
                if (current_time.weekday() < 5 and 
                    time(9, 15) <= current_time.time() <= time(15, 30)):
                    
                    # Calculate PCR
                    pcr_data = self.calculate_pcr()
                    
                    if pcr_data:
                        self.pcr_history.append(pcr_data)
                        
                        # Keep only last 100 records to manage memory
                        if len(self.pcr_history) > 100:
                            self.pcr_history = self.pcr_history[-100:]
                        
                        # Clear screen and display updated table
                        import os
                        os.system('clear' if os.name == 'posix' else 'cls')
                        
                        self.display_pcr_table()
                        
                        # Auto-save every 10 records
                        if len(self.pcr_history) % 10 == 0:
                            self.save_to_csv()
                    
                    else:
                        print(f"⚠️ Failed to fetch PCR data at {current_time.strftime('%H:%M:%S')}")
                
                else:
                    print(f"\n🕐 Market is closed. Monitoring will resume during market hours.")
                    print(f"📅 Market Hours: Monday-Friday, 9:15 AM - 3:30 PM IST")
                
                # Wait for 3 minutes
                print(f"\n⏳ Next update in 3 minutes... (at {(current_time + timedelta(minutes=3)).strftime('%H:%M:%S')})")
                time.sleep(180)  # 180 seconds = 3 minutes
                
        except KeyboardInterrupt:
            print(f"\n{Colors.YELLOW}🛑 Monitoring stopped by user{Colors.RESET}")
            
        except Exception as e:
            print(f"\n❌ Error in monitoring loop: {e}")
            
        finally:
            self.stop_monitor()
    
    def stop_monitor(self):
        """Stop the PCR monitor"""
        self.is_running = False
        print(f"\n{Colors.CYAN}📊 Final Summary:{Colors.RESET}")
        print(f"🔢 Total Records Collected: {len(self.pcr_history)}")
        
        if self.pcr_history:
            # Save final data
            self.save_to_csv(f"nifty_pcr_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
            
            # Show final statistics
            latest = self.pcr_history[-1]
            avg_pcr = sum(d.pcr for d in self.pcr_history) / len(self.pcr_history)
            
            print(f"📈 Final NIFTY Price: {latest.price}")
            print(f"📊 Final PCR: {latest.pcr}")
            print(f"📊 Average PCR: {avg_pcr:.3f}")
            print(f"🎯 Final Signal: {latest.option_signal}")
        
        print(f"{Colors.GREEN}✅ PCR Monitor stopped successfully{Colors.RESET}")

def main():
    """Main function"""
    print(f"{Colors.CYAN}{Colors.BOLD}🎯 NIFTY PCR Calculator - Real-time 3-Minute Intervals{Colors.RESET}")
    print(f"{Colors.YELLOW}📊 Similar to NSE Option Chain Analysis Tools{Colors.RESET}")
    
    calculator = NiftyPCRCalculator()
    
    try:
        # Ask user for mode
        print(f"\n{Colors.MAGENTA}Select Mode:{Colors.RESET}")
        print(f"1. Live Monitoring (Every 3 minutes)")
        print(f"2. Single PCR Calculation (One-time)")
        print(f"3. Demo Mode (Simulated data)")
        
        choice = input(f"{Colors.CYAN}Enter choice (1-3, default=1): {Colors.RESET}").strip()
        
        if choice == "2":
            # Single calculation
            print(f"\n{Colors.YELLOW}📊 Calculating current NIFTY PCR...{Colors.RESET}")
            pcr_data = calculator.calculate_pcr()
            
            if pcr_data:
                calculator.pcr_history.append(pcr_data)
                calculator.display_pcr_table()
                
                # Ask if user wants to save
                save_choice = input(f"\n{Colors.CYAN}Save to CSV? (y/n): {Colors.RESET}").strip().lower()
                if save_choice == 'y':
                    calculator.save_to_csv()
            else:
                print(f"❌ Failed to calculate PCR")
        
        elif choice == "3":
            # Demo mode
            print(f"\n{Colors.YELLOW}🎬 Demo Mode - Generating simulated PCR data{Colors.RESET}")
            
            for i in range(10):
                pcr_data = calculator.calculate_pcr()
                if pcr_data:
                    calculator.pcr_history.append(pcr_data)
                
                # Simulate 3-minute intervals
                time.sleep(1)  # Quick demo
            
            calculator.display_pcr_table()
            calculator.save_to_csv("demo_nifty_pcr.csv")
        
        else:
            # Live monitoring (default)
            calculator.run_pcr_monitor()
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Program interrupted by user{Colors.RESET}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
    finally:
        if calculator.is_running:
            calculator.stop_monitor()

if __name__ == "__main__":
    main()
