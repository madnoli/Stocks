
import requests
import json
import time
from datetime import datetime
import pandas as pd
import logging

class ProductionEntryScanner:
    def __init__(self, api_endpoint=None):
        """
        Production-ready real-time option scanner

        Args:
            api_endpoint: Your live option chain API endpoint
        """
        self.api_endpoint = api_endpoint or "http://localhost:3000/optionchain/BANKNIFTY"
        self.high_confidence_threshold = 80
        self.update_interval = 300  # 5 minutes
        self.running = False

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('live_trading_scanner.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def fetch_live_option_chain(self):
        """
        Fetch live option chain data from your API
        Customize this method based on your data source
        """
        try:
            # Example API calls - customize for your source:

            # Method 1: NSE Direct API (if available)
            # response = requests.get("https://www.nseindia.com/api/option-chain-indices?symbol=BANKNIFTY", 
            #                        headers={'User-Agent': 'Mozilla/5.0...'})

            # Method 2: Your local API server
            response = requests.get(self.api_endpoint, timeout=10)

            # Method 3: Third-party data provider (Zerodha, AngelOne, etc.)
            # response = requests.get(your_broker_api_endpoint, 
            #                        headers={'Authorization': 'Bearer your_token'})

            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"API Error: Status {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Network error fetching data: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error: {e}")
            return None

    def analyze_high_confidence_signals(self, option_data):
        """
        Enhanced signal analysis for production trading
        Returns only signals with score >= 80
        """
        signals = []

        try:
            underlying_value = self.get_underlying_value(option_data)
            atm_strike = round(underlying_value / 100) * 100

            # Focus on liquid strikes near ATM
            liquid_strikes = [
                atm_strike - 200, atm_strike - 100, atm_strike, 
                atm_strike + 100, atm_strike + 200
            ]

            for item in option_data['records']['data']:
                strike = item.get('strikePrice', 0)

                if strike not in liquid_strikes:
                    continue

                # Analyze CE options
                if 'CE' in item and self.is_tradeable_option(item['CE']):
                    ce_score = self.calculate_production_score(item['CE'], 'CE', underlying_value, strike)
                    if ce_score >= self.high_confidence_threshold:
                        signal = self.create_production_signal(strike, 'CE', item['CE'], ce_score, underlying_value)
                        signals.append(signal)

                # Analyze PE options
                if 'PE' in item and self.is_tradeable_option(item['PE']):
                    pe_score = self.calculate_production_score(item['PE'], 'PE', underlying_value, strike)
                    if pe_score >= self.high_confidence_threshold:
                        signal = self.create_production_signal(strike, 'PE', item['PE'], pe_score, underlying_value)
                        signals.append(signal)

            # Sort by score and return top 3
            signals.sort(key=lambda x: x['score'], reverse=True)
            return signals[:3]

        except Exception as e:
            self.logger.error(f"Error analyzing signals: {e}")
            return []

    def is_tradeable_option(self, option_data):
        """Check if option meets basic trading criteria"""
        last_price = option_data.get('lastPrice', 0)
        volume = option_data.get('totalTradedVolume', 0)
        bid = option_data.get('bidprice', 0)
        ask = option_data.get('askPrice', 0)

        return (
            last_price >= 20 and  # Minimum price for meaningful moves
            volume >= 1000 and   # Minimum volume for liquidity
            bid > 0 and ask > 0  # Must have valid bid-ask
        )

    def calculate_production_score(self, option_data, option_type, underlying_value, strike):
        """
        Production-grade scoring system
        Maximum score: 100
        """
        score = 0

        # 1. Volume Factor (25 points)
        volume = option_data.get('totalTradedVolume', 0)
        if volume >= 100000:
            score += 25
        elif volume >= 50000:
            score += 20
        elif volume >= 25000:
            score += 15
        elif volume >= 10000:
            score += 10
        elif volume >= 5000:
            score += 5

        # 2. Open Interest Change (25 points)
        oi_change = option_data.get('changeinOpenInterest', 0)
        if abs(oi_change) >= 5000:
            score += 25
        elif abs(oi_change) >= 2500:
            score += 20
        elif abs(oi_change) >= 1000:
            score += 15
        elif abs(oi_change) >= 500:
            score += 10

        # 3. Price Momentum (20 points)
        price_change = option_data.get('pChange', 0)
        if abs(price_change) >= 20:
            score += 20
        elif abs(price_change) >= 15:
            score += 16
        elif abs(price_change) >= 10:
            score += 12
        elif abs(price_change) >= 5:
            score += 8

        # 4. Implied Volatility (15 points)
        iv = option_data.get('impliedVolatility', 0)
        if 15 <= iv <= 25:  # Sweet spot
            score += 15
        elif 12 <= iv <= 30:
            score += 12
        elif 10 <= iv <= 35:
            score += 8

        # 5. Liquidity (10 points)
        bid = option_data.get('bidprice', 0)
        ask = option_data.get('askPrice', 0)
        last_price = option_data.get('lastPrice', 0)

        if bid > 0 and ask > 0 and last_price > 0:
            spread_pct = ((ask - bid) / last_price) * 100
            if spread_pct <= 2:
                score += 10
            elif spread_pct <= 5:
                score += 7
            elif spread_pct <= 10:
                score += 4

        # 6. Strike Position Bonus (5 points)
        distance = abs(strike - underlying_value)
        if distance <= 100:  # ATM
            score += 5
        elif distance <= 200:  # Near ATM
            score += 3

        return min(score, 100)  # Cap at 100

    def create_production_signal(self, strike, option_type, option_data, score, underlying_value):
        """Create production-ready trading signal"""
        last_price = option_data.get('lastPrice', 0)

        return {
            'timestamp': datetime.now().isoformat(),
            'signal_id': f"{strike}_{option_type}_{int(time.time())}",
            'instrument': f"BANKNIFTY {strike} {option_type}",
            'action': f"BUY {'CALL' if option_type == 'CE' else 'PUT'}",
            'strike_price': strike,
            'option_type': option_type,
            'entry_price': last_price,
            'confidence_score': score,
            'underlying_price': underlying_value,
            'volume': option_data.get('totalTradedVolume', 0),
            'oi_change': option_data.get('changeinOpenInterest', 0),
            'price_change_pct': option_data.get('pChange', 0),
            'implied_volatility': option_data.get('impliedVolatility', 0),
            'bid_price': option_data.get('bidprice', 0),
            'ask_price': option_data.get('askPrice', 0),
            'stop_loss': round(last_price * 0.7, 2),
            'target_1': round(last_price * 1.5, 2),
            'target_2': round(last_price * 2.0, 2),
            'risk_reward_1': 1.67,  # (1.5-1)/(1-0.7)
            'risk_reward_2': 3.33,  # (2.0-1)/(1-0.7)
            'distance_from_atm': strike - underlying_value,
            'recommendation': 'STRONG BUY' if score >= 90 else 'BUY'
        }

    def get_underlying_value(self, option_data):
        """Extract underlying value from option chain"""
        try:
            for item in option_data['records']['data']:
                if 'CE' in item and item['CE'].get('underlyingValue'):
                    return item['CE']['underlyingValue']
            return 0
        except:
            return 0

    def send_production_alert(self, signal):
        """Send production-ready alert"""
        alert_text = f"""
🚨 HIGH CONFIDENCE SIGNAL 🚨
ID: {signal['signal_id']}
Time: {datetime.fromisoformat(signal['timestamp']).strftime('%H:%M:%S')}

📈 {signal['action']} {signal['strike_price']} {signal['option_type']}
💰 Entry: ₹{signal['entry_price']}
🎯 Score: {signal['confidence_score']}/100
📊 Volume: {signal['volume']:,}
⚡ Change: {signal['price_change_pct']:+.1f}%

🎯 TARGETS:
   Target 1: ₹{signal['target_1']} (R:R 1:{signal['risk_reward_1']:.1f})
   Target 2: ₹{signal['target_2']} (R:R 1:{signal['risk_reward_2']:.1f})
   Stop Loss: ₹{signal['stop_loss']}

🏦 Bank Nifty: {signal['underlying_price']:.2f}
{'='*50}
"""

        print(alert_text)

        # Save alert
        alert_file = f"production_alerts/alert_{signal['signal_id']}.txt"
        os.makedirs('production_alerts', exist_ok=True)
        with open(alert_file, 'w') as f:
            f.write(alert_text)

        # Log
        self.logger.info(f"PRODUCTION SIGNAL: {signal['instrument']} @ ₹{signal['entry_price']}")

    def run_production_scanner(self):
        """Main production scanner loop"""
        self.running = True
        self.logger.info("PRODUCTION SCANNER STARTED")

        print(f"🔥 PRODUCTION HIGH-CONFIDENCE SCANNER ACTIVE")
        print(f"📊 API Endpoint: {self.api_endpoint}")
        print(f"⏰ Update Interval: {self.update_interval//60} minutes")
        print(f"🎯 Confidence Threshold: {self.high_confidence_threshold}/100")
        print("=" * 60)

        scan_count = 0

        while self.running:
            try:
                scan_count += 1
                current_time = datetime.now()

                # Market hours check
                if not self.is_market_open():
                    print(f"💤 Market closed - Scanner paused")
                    time.sleep(600)  # Check every 10 minutes
                    continue

                print(f"\n📊 Production Scan #{scan_count} - {current_time.strftime('%H:%M:%S')}")

                # Fetch live data
                live_data = self.fetch_live_option_chain()

                if live_data:
                    high_conf_signals = self.analyze_high_confidence_signals(live_data)

                    if high_conf_signals:
                        print(f"🚨 {len(high_conf_signals)} HIGH CONFIDENCE SIGNALS DETECTED!")

                        for signal in high_conf_signals:
                            self.send_production_alert(signal)

                            # Save to production CSV
                            self.save_production_signal(signal)
                    else:
                        print("💤 No high confidence signals - Monitoring...")
                else:
                    print("❌ Failed to fetch live data - Retrying...")

                # Wait for next scan
                time.sleep(self.update_interval)

            except KeyboardInterrupt:
                print("\n🛑 Production scanner stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Production scanner error: {e}")
                time.sleep(60)

        self.running = False
        self.logger.info("PRODUCTION SCANNER STOPPED")

    def is_market_open(self):
        """Check if market is currently open"""
        now = datetime.now()
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        # Check if current time is within market hours and it's a weekday
        return (market_open <= now <= market_close and now.weekday() < 5)

    def save_production_signal(self, signal):
        """Save signal to production CSV"""
        filename = f"production_signals/signals_{datetime.now().strftime('%Y%m%d')}.csv"
        os.makedirs('production_signals', exist_ok=True)

        # Convert to DataFrame and append
        df = pd.DataFrame([signal])

        # Append to existing file or create new
        if os.path.exists(filename):
            df.to_csv(filename, mode='a', header=False, index=False)
        else:
            df.to_csv(filename, index=False)

        print(f"💾 Signal saved to {filename}")

# Usage example:
if __name__ == "__main__":
    # Initialize with your API endpoint
    scanner = ProductionEntryScanner(api_endpoint="http://localhost:3000/optionchain/BANKNIFTY")

    # Start production scanning
    scanner.run_production_scanner()
