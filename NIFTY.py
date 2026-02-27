import requests
import time
import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class OptionChainAnalyzer:
    def __init__(self, api_url="http://localhost:3000/api/index/options/NIFTY"):
        self.api_url = api_url
        self.previous_data = None
        self.analysis_results = []
        print(f"🚀 NIFTY Option Chain Analyzer initialized")
        print(f"📡 API URL: {self.api_url}")
        
    def test_connection(self):
        """Test API connection"""
        print(f"\n🔍 Testing connection to {self.api_url}...")
        
        try:
            response = requests.get(self.api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print("✅ API connection successful!")
                
                # Check data structure
                if 'records' in data and 'data' in data['records']:
                    records_count = len(data['records']['data'])
                    spot_price = data['records'].get('underlyingValue', 'N/A')
                    print(f"📊 Found {records_count} option strikes")
                    print(f"📊 Current NIFTY spot: {spot_price}")
                    return True, data
                else:
                    print("❌ Invalid data structure received")
                    print(f"Available keys: {list(data.keys())}")
                    return False, data
            else:
                print(f"❌ API error: Status {response.status_code}")
                return False, None
                
        except requests.exceptions.ConnectionError:
            print("❌ CONNECTION FAILED!")
            print("\n🔧 SOLUTIONS:")
            print("1. Start your NSE API server: npm start or node server.js")
            print("2. Verify server runs on http://localhost:3000")
            print("3. Check endpoint: /api/index/options/NIFTY")
            return False, None
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return False, None

    def fetch_option_chain(self):
        """Fetch option chain data from localhost API"""
        try:
            response = requests.get(self.api_url, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"API Error: Status Code {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return None
    
    def calculate_pcr(self, data):
        """Calculate Put-Call Ratio"""
        try:
            records = data.get('records', {}).get('data', [])
            total_call_oi = sum([record.get('CE', {}).get('openInterest', 0) for record in records if 'CE' in record])
            total_put_oi = sum([record.get('PE', {}).get('openInterest', 0) for record in records if 'PE' in record])
            
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
            return round(pcr, 3), total_call_oi, total_put_oi
        except Exception as e:
            print(f"PCR calculation error: {e}")
            return 0, 0, 0
    
    def find_max_pain(self, data):
        """Calculate Max Pain level"""
        try:
            records = data.get('records', {}).get('data', [])
            max_pain_data = {}
            
            for record in records:
                strike = record.get('strikePrice')
                if not strike:
                    continue
                    
                # Calculate total pain for option writers at this strike
                call_pain = sum([max(0, strike - other_record.get('strikePrice', 0)) * other_record.get('CE', {}).get('openInterest', 0) 
                               for other_record in records 
                               if other_record.get('strikePrice', 0) < strike and 'CE' in other_record])
                
                put_pain = sum([max(0, other_record.get('strikePrice', 0) - strike) * other_record.get('PE', {}).get('openInterest', 0) 
                              for other_record in records 
                              if other_record.get('strikePrice', 0) > strike and 'PE' in other_record])
                
                max_pain_data[strike] = call_pain + put_pain
            
            if max_pain_data:
                max_pain_strike = min(max_pain_data, key=max_pain_data.get)
                return max_pain_strike
            return None
        except Exception as e:
            print(f"Max Pain calculation error: {e}")
            return None
    
    def analyze_oi_changes(self, current_data, previous_data):
        """Analyze Open Interest changes strike-wise between two snapshots"""
        if not previous_data:
            return {}
        
        try:
            current_records = {record.get('strikePrice'): record for record in current_data.get('records', {}).get('data', [])}
            previous_records = {record.get('strikePrice'): record for record in previous_data.get('records', {}).get('data', [])}
            
            oi_changes = {}
            
            for strike in current_records:
                if strike in previous_records:
                    current_call_oi = current_records[strike].get('CE', {}).get('openInterest', 0)
                    previous_call_oi = previous_records[strike].get('CE', {}).get('openInterest', 0)
                    call_oi_change = current_call_oi - previous_call_oi
                    
                    current_put_oi = current_records[strike].get('PE', {}).get('openInterest', 0)
                    previous_put_oi = previous_records[strike].get('PE', {}).get('openInterest', 0)
                    put_oi_change = current_put_oi - previous_put_oi
                    
                    if abs(call_oi_change) > 1000 or abs(put_oi_change) > 1000:  # Significant changes
                        oi_changes[strike] = {
                            'call_oi_change': call_oi_change,
                            'put_oi_change': put_oi_change,
                            'net_change': call_oi_change - put_oi_change,
                            'call_volume': current_records[strike].get('CE', {}).get('totalTradedVolume', 0),
                            'put_volume': current_records[strike].get('PE', {}).get('totalTradedVolume', 0)
                        }
            
            return oi_changes
        except Exception as e:
            print(f"OI change analysis error: {e}")
            return {}
    
    def identify_support_resistance(self, data):
        """Identify support and resistance levels based on OI"""
        try:
            records = data.get('records', {}).get('data', [])
            
            # Sort by Call OI (descending) for resistance
            call_oi_data = [(record.get('strikePrice'), record.get('CE', {}).get('openInterest', 0)) 
                           for record in records if 'CE' in record and record.get('CE', {}).get('openInterest', 0) > 0]
            call_oi_data.sort(key=lambda x: x[1], reverse=True)
            
            # Sort by Put OI (descending) for support
            put_oi_data = [(record.get('strikePrice'), record.get('PE', {}).get('openInterest', 0)) 
                          for record in records if 'PE' in record and record.get('PE', {}).get('openInterest', 0) > 0]
            put_oi_data.sort(key=lambda x: x[1], reverse=True)
            
            resistance_levels = [level[0] for level in call_oi_data[:3]]  # Top 3
            support_levels = [level[0] for level in put_oi_data[:3]]      # Top 3
            
            return {
                'resistance': resistance_levels,
                'support': support_levels,
                'max_call_oi_strike': call_oi_data[0][0] if call_oi_data else None,
                'max_put_oi_strike': put_oi_data[0][0] if put_oi_data else None,
                'max_call_oi_value': call_oi_data[0][1] if call_oi_data else 0,
                'max_put_oi_value': put_oi_data[0][1] if put_oi_data else 0
            }
        except Exception as e:
            print(f"Support/Resistance analysis error: {e}")
            return {}
    
    def calculate_iv_metrics(self, data):
        """Calculate Implied Volatility metrics around ATM strike"""
        try:
            records = data.get('records', {}).get('data', [])
            spot_price = data.get('records', {}).get('underlyingValue', 0)
            if not records:
                return {}
            
            # Identify ATM strike by minimum distance from spot
            atm_record = min(records, key=lambda x: abs(x.get('strikePrice', 0) - spot_price))
            atm_strike = atm_record.get('strikePrice')
            atm_call_iv = atm_record.get('CE', {}).get('impliedVolatility', 0) or 0
            atm_put_iv = atm_record.get('PE', {}).get('impliedVolatility', 0) or 0
            
            # Calculate IV skew
            iv_skew = atm_put_iv - atm_call_iv
            
            return {
                'atm_call_iv': round(atm_call_iv, 2),
                'atm_put_iv': round(atm_put_iv, 2),
                'iv_skew': round(iv_skew, 2),
                'atm_strike': atm_strike
            }
        except Exception as e:
            print(f"IV calculation error: {e}")
            return {}
    
    def generate_signals(self, analysis):
        """Generate trading signals based on analysis"""
        signals = []
        
        try:
            # PCR-based signals
            pcr = analysis.get('pcr', 0)
            if pcr > 1.3:
                signals.append("🔵 BULLISH: High PCR indicates oversold condition - Consider CALLS")
            elif pcr < 0.7:
                signals.append("🔴 BEARISH: Low PCR indicates overbought condition - Consider PUTS")
            elif 1.0 <= pcr <= 1.2:
                signals.append("⚪ NEUTRAL: PCR in normal range - Range-bound market")
            
            # OI change signals
            oi_changes = analysis.get('oi_changes', {})
            for strike, changes in oi_changes.items():
                if changes['call_oi_change'] > 5000 and changes['call_volume'] > 1000:
                    signals.append(f"⚡ STRONG RESISTANCE at {strike} (Call OI +{changes['call_oi_change']:,}, Vol: {changes['call_volume']:,})")
                if changes['put_oi_change'] > 5000 and changes['put_volume'] > 1000:
                    signals.append(f"⚡ STRONG SUPPORT at {strike} (Put OI +{changes['put_oi_change']:,}, Vol: {changes['put_volume']:,})")
            
            # Max Pain signal
            max_pain = analysis.get('max_pain')
            spot_price = analysis.get('spot_price', 0)
            if max_pain and spot_price:
                pain_diff = ((spot_price - max_pain) / max_pain) * 100
                if pain_diff > 0.5:
                    signals.append(f"📉 BEARISH BIAS: Price {pain_diff:.1f}% above Max Pain {max_pain}")
                elif pain_diff < -0.5:
                    signals.append(f"📈 BULLISH BIAS: Price {abs(pain_diff):.1f}% below Max Pain {max_pain}")
            
            # IV-based signals
            iv_metrics = analysis.get('iv_metrics', {})
            iv_skew = iv_metrics.get('iv_skew', 0)
            if iv_skew > 2:
                signals.append("📊 HIGH PUT IV SKEW: Fear premium, potential support")
            elif iv_skew < -2:
                signals.append("📊 HIGH CALL IV SKEW: Greed premium, potential resistance")
            
            return signals
        except Exception as e:
            print(f"Signal generation error: {e}")
            return []
    
    def analyze_current_data(self, data):
        """Perform comprehensive analysis on current data"""
        analysis = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'spot_price': data.get('records', {}).get('underlyingValue', 0)
        }
        
        # Calculate PCR
        pcr, total_call_oi, total_put_oi = self.calculate_pcr(data)
        analysis.update({
            'pcr': pcr,
            'total_call_oi': total_call_oi,
            'total_put_oi': total_put_oi
        })
        
        # Find Max Pain
        analysis['max_pain'] = self.find_max_pain(data)
        
        # Analyze OI changes
        analysis['oi_changes'] = self.analyze_oi_changes(data, self.previous_data)
        
        # Identify support/resistance
        analysis['levels'] = self.identify_support_resistance(data)
        
        # Calculate IV metrics
        analysis['iv_metrics'] = self.calculate_iv_metrics(data)
        
        # Generate signals
        analysis['signals'] = self.generate_signals(analysis)
        
        return analysis
    
    def display_analysis(self, analysis):
        """Display the analysis results"""
        print("\n" + "="*100)
        print(f"🎯 NIFTY OPTION CHAIN ANALYSIS - {analysis['timestamp']}")
        print("="*100)
        
        print(f"📊 Spot Price: {analysis['spot_price']}")
        print(f"📊 Put-Call Ratio: {analysis['pcr']}")
        print(f"📊 Max Pain Level: {analysis['max_pain']}")
        print(f"📊 Total Call OI: {analysis['total_call_oi']:,}")
        print(f"📊 Total Put OI: {analysis['total_put_oi']:,}")
        
        # IV Metrics
        iv_metrics = analysis.get('iv_metrics', {})
        if iv_metrics:
            print(f"📊 ATM Strike: {iv_metrics.get('atm_strike')}")
            print(f"📊 ATM Call IV: {iv_metrics.get('atm_call_iv')}%")
            print(f"📊 ATM Put IV: {iv_metrics.get('atm_put_iv')}%")
            print(f"📊 IV Skew: {iv_metrics.get('iv_skew')}%")
        
        levels = analysis.get('levels', {})
        if levels:
            print(f"\n🎯 KEY LEVELS:")
            print(f"🔴 Resistance: {levels.get('resistance', [])}")
            print(f"🟢 Support: {levels.get('support', [])}")
            print(f"🔴 Max Call OI: {levels.get('max_call_oi_strike')} ({levels.get('max_call_oi_value', 0):,})")
            print(f"🟢 Max Put OI: {levels.get('max_put_oi_strike')} ({levels.get('max_put_oi_value', 0):,})")
        
        # Display significant OI changes
        oi_changes = analysis.get('oi_changes', {})
        if oi_changes:
            print(f"\n⚡ SIGNIFICANT OI CHANGES:")
            for strike, changes in sorted(oi_changes.items()):
                print(f"Strike {strike}: Call OI {changes['call_oi_change']:+,} (Vol: {changes['call_volume']:,}), "
                      f"Put OI {changes['put_oi_change']:+,} (Vol: {changes['put_volume']:,})")
        
        # Display signals
        signals = analysis.get('signals', [])
        if signals:
            print(f"\n🚨 TRADING SIGNALS:")
            for signal in signals:
                print(f"   {signal}")
        else:
            print(f"\n⚪ No significant signals detected")
        
        print("="*100)
    
    def save_to_csv(self, analysis):
        """Save analysis to CSV file"""
        try:
            # Flatten the analysis for CSV
            csv_data = {
                'timestamp': analysis['timestamp'],
                'spot_price': analysis['spot_price'],
                'pcr': analysis['pcr'],
                'max_pain': analysis['max_pain'],
                'total_call_oi': analysis['total_call_oi'],
                'total_put_oi': analysis['total_put_oi'],
                'signals_count': len(analysis.get('signals', [])),
                'oi_changes_count': len(analysis.get('oi_changes', {})),
                'atm_call_iv': analysis.get('iv_metrics', {}).get('atm_call_iv', 0),
                'atm_put_iv': analysis.get('iv_metrics', {}).get('atm_put_iv', 0),
                'iv_skew': analysis.get('iv_metrics', {}).get('iv_skew', 0)
            }
            
            self.analysis_results.append(csv_data)
            
            # Save to CSV every 5 records
            if len(self.analysis_results) % 5 == 0:
                df = pd.DataFrame(self.analysis_results)
                filename = f"nifty_option_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(filename, index=False)
                print(f"💾 Data saved to {filename}")
                
        except Exception as e:
            print(f"CSV save error: {e}")
    
    def run_continuous_analysis(self, interval_minutes=2):
        """Run continuous analysis every specified minutes"""
        print(f"🚀 Starting NIFTY Option Chain Analysis (Every {interval_minutes} minutes)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                # Fetch current data
                current_data = self.fetch_option_chain()
                
                if current_data:
                    # Perform analysis
                    analysis = self.analyze_current_data(current_data)
                    
                    # Display results
                    self.display_analysis(analysis)
                    
                    # Save to CSV
                    self.save_to_csv(analysis)
                    
                    # Store current data for next iteration
                    self.previous_data = current_data
                else:
                    print(f"❌ Failed to fetch data at {datetime.now().strftime('%H:%M:%S')}")
                
                # Wait for next iteration
                print(f"⏰ Next update in {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
        except KeyboardInterrupt:
            print("\n🛑 Analysis stopped by user")
            # Save final data
            if self.analysis_results:
                df = pd.DataFrame(self.analysis_results)
                filename = f"nifty_option_analysis_final_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
                df.to_csv(filename, index=False)
                print(f"💾 Final data saved to {filename}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

# MAIN EXECUTION - THIS WILL ACTUALLY RUN
if __name__ == "__main__":
    print("🎯 NIFTY Option Chain Analyzer - Active Version")
    print("=" * 60)
    
    try:
        # Create analyzer
        analyzer = OptionChainAnalyzer()
        
        # Run single test first to validate connection and data
        success, data = analyzer.test_connection()
        if success and data:
            print("\n✅ Connection successful! Running single analysis...")
            analysis = analyzer.analyze_current_data(data)
            analyzer.display_analysis(analysis)
            
            # Ask for continuous mode
            try:
                user_input = input("\n❓ Start continuous analysis every 2 minutes? (y/n): ").lower().strip()
                if user_input == 'y':
                    analyzer.previous_data = data
                    analyzer.run_continuous_analysis(interval_minutes=2)
                else:
                    print("\n👋 Single analysis completed. Exiting.")
            except KeyboardInterrupt:
                print("\n\n🛑 Analysis stopped by user. Goodbye!")
        else:
            print("\n❌ Test failed. Please ensure your API server is running and endpoint is correct.")
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nPlease ensure:")
        print("• Your NSE API server is running on localhost:3000")
        print("• The endpoint /api/index/options/NIFTY is correct")
        print("• No firewall is blocking the connection")
