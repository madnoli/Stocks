import yfinance as yf
import pandas as pd
import requests
import json # Import the json library to handle the specific error
from datetime import date, timedelta

# --- COLOR CODES for better terminal output ---
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    ORANGE = '\033[38;5;208m'

class RealDataNiftyAnalyzer:
    """
    Combines CPR and PCR using live market data to find optimal Nifty entry points.
    """
    def __init__(self):
        print(f"{Colors.ORANGE}{Colors.BOLD}🚀 NIFTY LIVE DATA ANALYZER (CPR + PCR) 🚀{Colors.RESET}")
        print("-" * 55)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'application/json, text/javascript, */*; q=0.01'
        }
        self.session = requests.Session()

    def get_nifty_price_data(self) -> tuple:
        """
        Fetches previous day's OHLC for CPR and the latest Nifty price using yfinance.
        """
        print(f"{Colors.CYAN}Step 1: Fetching Nifty Price Data...{Colors.RESET}")
        try:
            nifty = yf.Ticker("^NSEI")
            hist = nifty.history(period="5d")
            
            if hist.empty:
                print(f"{Colors.RED}Error: Could not fetch Nifty historical data.{Colors.RESET}")
                return None, None, None, None

            prev_day = hist.iloc[-2]
            prev_high = prev_day['High']
            prev_low = prev_day['Low']
            prev_close = prev_day['Close']
            
            live_price = hist['Close'].iloc[-1]

            print(f"  - Previous Day's High:   {prev_high:.2f}")
            print(f"  - Previous Day's Low:    {prev_low:.2f}")
            print(f"  - Previous Day's Close:  {prev_close:.2f}")
            print(f"  - {Colors.BOLD}Current Nifty Price:     {live_price:.2f}{Colors.RESET}\n")

            return (prev_high, prev_low, prev_close, live_price)
        except Exception as e:
            print(f"{Colors.RED}An error occurred while fetching data from yfinance: {e}{Colors.RESET}")
            return None, None, None, None


    def calculate_cpr(self, high: float, low: float, close: float) -> dict:
        """Calculates the Central Pivot Range (CPR) levels."""
        pivot = (high + low + close) / 3
        bc = (high + low) / 2
        tc = (pivot - bc) + pivot

        if tc < bc:
            tc, bc = bc, tc
            
        cpr_width = abs(tc - bc) / pivot * 100
        
        print(f"{Colors.CYAN}Step 2: Calculating CPR Levels...{Colors.RESET}")
        print(f"  - {Colors.RED}Top Central (TC):    {tc:.2f}{Colors.RESET}")
        print(f"  - {Colors.WHITE}{Colors.BOLD}Central Pivot (PP):  {pivot:.2f}{Colors.RESET}")
        print(f"  - {Colors.GREEN}Bottom Central (BC): {bc:.2f}{Colors.RESET}")
        
        if cpr_width < 0.3:
            print(f"  - CPR Width: {cpr_width:.2f}% ({Colors.YELLOW}Narrow - Expecting a Trending Day{Colors.RESET})\n")
        else:
            print(f"  - CPR Width: {cpr_width:.2f}% ({Colors.BLUE}Wide - Expecting a Sideways Day{Colors.RESET})\n")
        
        return {'TC': tc, 'Pivot': pivot, 'BC': bc}

    def get_live_pcr(self) -> dict:
        """
        Fetches live option chain data from the NSE website and calculates PCR.
        IMPROVED with better error handling.
        """
        print(f"{Colors.CYAN}Step 3: Calculating Live PCR Sentiment...{Colors.RESET}")
        try:
            # "Warm-up" the session by visiting the main page first. This helps in getting cookies.
            main_url = "https://www.nseindia.com"
            api_url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
            
            self.session.get(main_url, headers=self.headers, timeout=10)
            response = self.session.get(api_url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            total_put_oi = data['filtered']['PE']['totOI']
            total_call_oi = data['filtered']['CE']['totOI']
            
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0

            print(f"  - Total PUT OI: {total_put_oi:,}")
            print(f"  - Total CALL OI: {total_call_oi:,}")
            print(f"  - {Colors.WHITE}{Colors.BOLD}Live PCR Value: {pcr:.2f}{Colors.RESET}")

            if pcr > 1.1:
                sentiment = "BULLISH"
                color = Colors.GREEN
            elif pcr < 0.9:
                sentiment = "BEARISH"
                color = Colors.RED
            else:
                sentiment = "NEUTRAL"
                color = Colors.YELLOW

            print(f"  - Market Sentiment: {color}{sentiment}{Colors.RESET}\n")
            return {'pcr': pcr, 'sentiment': sentiment}

        # Catch the specific JSON decoding error
        except json.JSONDecodeError:
            print(f"{Colors.RED}Error: The NSE server did not return valid JSON data.{Colors.RESET}")
            print(f"{Colors.YELLOW}This usually happens when the market is closed. Please run between 9:15 AM - 3:30 PM IST on a trading day.{Colors.RESET}")
            return None
        # Catch other network-related errors
        except requests.exceptions.RequestException as e:
            print(f"{Colors.RED}Error fetching NSE Option Chain: {e}. The server might be busy or blocking requests.{Colors.RESET}")
            return None


    def find_best_entry(self, cpr_levels: dict, pcr_info: dict, current_price: float):
        """Generates a trading signal based on combined CPR and PCR analysis."""
        cpr_tc = cpr_levels['TC']
        cpr_bc = cpr_levels['BC']
        pcr_sentiment = pcr_info['sentiment']

        print(f"{Colors.CYAN}Step 4: Analyzing Entry Point...{Colors.RESET}")
        print(f"  - Current Nifty Price: {Colors.WHITE}{Colors.BOLD}{current_price:.2f}{Colors.RESET}")

        signal = f"{Colors.YELLOW}{Colors.BOLD}WAIT ⏳{Colors.RESET}"
        reasoning = ""

        if pcr_sentiment == "BULLISH" and current_price > cpr_tc:
            signal = f"{Colors.GREEN}{Colors.BOLD}STRONG BUY CALL (CE) ✅{Colors.RESET}"
            reasoning = f"Dual confirmation: Sentiment is {Colors.GREEN}BULLISH{Colors.RESET} and price is above the {Colors.GREEN}bullish CPR zone{Colors.RESET} (> {cpr_tc:.2f})."
        elif pcr_sentiment == "BEARISH" and current_price < cpr_bc:
            signal = f"{Colors.RED}{Colors.BOLD}STRONG BUY PUT (PE) ✅{Colors.RESET}"
            reasoning = f"Dual confirmation: Sentiment is {Colors.RED}BEARISH{Colors.RESET} and price is below the {Colors.RED}bearish CPR zone{Colors.RESET} (< {cpr_bc:.2f})."
        else:
            if pcr_sentiment == "NEUTRAL":
                reasoning = "PCR is neutral. Market is indecisive. Avoid trading until a clear trend emerges."
            elif cpr_bc < current_price < cpr_tc:
                reasoning = f"Price is inside the CPR 'No-Trade Zone' ({cpr_bc:.2f}-{cpr_tc:.2f}). Wait for a breakout."
            else:
                reasoning = f"Indicators are contradictory. PCR sentiment ({pcr_sentiment}) does not match price location. No clear signal."

        print("\n" + "="*20 + " TRADING SIGNAL " + "="*20)
        print(f"Signal:         {signal}")
        print(f"Reasoning:      {reasoning}")
        print("="*55)
        print(f"{Colors.WHITE}\nDisclaimer: This is for educational purposes only. Always do your own research.{Colors.RESET}")

# --- Main Execution ---
if __name__ == "__main__":
    analyzer = RealDataNiftyAnalyzer()
    
    prev_high, prev_low, prev_close, live_price = analyzer.get_nifty_price_data()

    if live_price is not None:
        cpr = analyzer.calculate_cpr(prev_high, prev_low, prev_close)
        pcr = analyzer.get_live_pcr()

        if pcr is not None:
            analyzer.find_best_entry(cpr, pcr, live_price)