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

    def calculate_enhanced_pcr(self, data):
        """Calculate multiple PCR variants for comprehensive analysis"""
        records = data.get('records', {}).get('data', [])
        spot_price = data.get('records', {}).get('underlyingValue', 0)

        # Standard OI-based PCR
        total_call_oi = sum([r.get('CE', {}).get('openInterest', 0) for r in records if 'CE' in r])
        total_put_oi = sum([r.get('PE', {}).get('openInterest', 0) for r in records if 'PE' in r])
        pcr_oi = total_put_oi / total_call_oi if total_call_oi > 0 else 0

        # Volume-based PCR
        total_call_vol = sum([r.get('CE', {}).get('totalTradedVolume', 0) for r in records if 'CE' in r])
        total_put_vol = sum([r.get('PE', {}).get('totalTradedVolume', 0) for r in records if 'PE' in r])
        pcr_volume = total_put_vol / total_call_vol if total_call_vol > 0 else 0

        # ATM PCR (±5% from spot)
        atm_range = spot_price * 0.05
        atm_call_oi = sum([r.get('CE', {}).get('openInterest', 0) for r in records 
                           if 'CE' in r and abs(r.get('strikePrice', 0) - spot_price) <= atm_range])
        atm_put_oi = sum([r.get('PE', {}).get('openInterest', 0) for r in records 
                          if 'PE' in r and abs(r.get('strikePrice', 0) - spot_price) <= atm_range])
        pcr_atm = atm_put_oi / atm_call_oi if atm_call_oi > 0 else 0

        return {
            'pcr_oi': round(pcr_oi, 3),
            'pcr_volume': round(pcr_volume, 3),
            'pcr_atm': round(pcr_atm, 3),
            'total_call_oi': total_call_oi,
            'total_put_oi': total_put_oi,
            'total_call_volume': total_call_vol,
            'total_put_volume': total_put_vol,
            'interpretation': self._interpret_pcr(pcr_oi, pcr_volume, pcr_atm)
        }

    def _interpret_pcr(self, pcr_oi, pcr_vol, pcr_atm):
        """Interpret PCR signals"""
        signals = []

        # OI-based (longer term sentiment)
        if pcr_oi > 1.3:
            signals.append("OI PCR > 1.3: BULLISH (Oversold)")
        elif pcr_oi < 0.7:
            signals.append("OI PCR < 0.7: BEARISH (Overbought)")
        else:
            signals.append("OI PCR in neutral range")

        # Volume-based (immediate sentiment)
        if pcr_vol > 1.5:
            signals.append("Vol PCR > 1.5: STRONG BULLISH (Heavy Put Buying)")
        elif pcr_vol < 0.5:
            signals.append("Vol PCR < 0.5: STRONG BEARISH (Heavy Call Buying)")
        else:
            signals.append("Vol PCR in neutral range")

        # ATM PCR (most accurate for direction)
        if pcr_atm > 1.2:
            signals.append("ATM PCR > 1.2: UPSIDE EXPECTED")
        elif pcr_atm < 0.8:
            signals.append("ATM PCR < 0.8: DOWNSIDE EXPECTED")
        else:
            signals.append("ATM PCR neutral")

        return signals

    def detect_oi_buildup_patterns(self, current_data, previous_data):
        """Detect Long/Short Buildup and Unwinding patterns"""
        patterns = {
            'long_buildup': [],    # Price ↑ + OI ↑ = Bullish
            'short_buildup': [],   # Price ↓ + OI ↑ = Bearish
            'long_unwinding': [],  # Price ↓ + OI ↓ = Weak Bullish
            'short_unwinding': []  # Price ↑ + OI ↓ = Weak Bearish
        }

        if not previous_data:
            return patterns

        current_records = {r.get('strikePrice'): r for r in current_data.get('records', {}).get('data', [])}
        previous_records = {r.get('strikePrice'): r for r in previous_data.get('records', {}).get('data', [])}

        for strike in current_records:
            if strike not in previous_records:
                continue

            # Call Options Analysis
            curr_call_oi = current_records[strike].get('CE', {}).get('openInterest', 0)
            prev_call_oi = previous_records[strike].get('CE', {}).get('openInterest', 0)
            curr_call_price = current_records[strike].get('CE', {}).get('lastPrice', 0)
            prev_call_price = previous_records[strike].get('CE', {}).get('lastPrice', 1)

            call_oi_change = ((curr_call_oi - prev_call_oi) / prev_call_oi * 100) if prev_call_oi > 0 else 0
            call_price_change = ((curr_call_price - prev_call_price) / prev_call_price * 100) if prev_call_price > 0 else 0

            # Put Options Analysis
            curr_put_oi = current_records[strike].get('PE', {}).get('openInterest', 0)
            prev_put_oi = previous_records[strike].get('PE', {}).get('openInterest', 0)
            curr_put_price = current_records[strike].get('PE', {}).get('lastPrice', 0)
            prev_put_price = previous_records[strike].get('PE', {}).get('lastPrice', 1)

            put_oi_change = ((curr_put_oi - prev_put_oi) / prev_put_oi * 100) if prev_put_oi > 0 else 0
            put_price_change = ((curr_put_price - prev_put_price) / prev_put_price * 100) if prev_put_price > 0 else 0

            # Detect Call patterns (threshold: 5% OI change)
            if abs(call_oi_change) > 5:
                if call_price_change > 2 and call_oi_change > 5:
                    patterns['long_buildup'].append({
                        'strike': strike,
                        'type': 'CALL',
                        'oi_change': round(call_oi_change, 2),
                        'price_change': round(call_price_change, 2),
                        'signal': 'BULLISH'
                    })
                elif call_price_change < -2 and call_oi_change > 5:
                    patterns['short_buildup'].append({
                        'strike': strike,
                        'type': 'CALL',
                        'oi_change': round(call_oi_change, 2),
                        'price_change': round(call_price_change, 2),
                        'signal': 'RESISTANCE'
                    })
                elif call_price_change < -2 and call_oi_change < -5:
                    patterns['long_unwinding'].append({
                        'strike': strike,
                        'type': 'CALL',
                        'oi_change': round(call_oi_change, 2),
                        'price_change': round(call_price_change, 2),
                        'signal': 'BEARISH'
                    })
                elif call_price_change > 2 and call_oi_change < -5:
                    patterns['short_unwinding'].append({
                        'strike': strike,
                        'type': 'CALL',
                        'oi_change': round(call_oi_change, 2),
                        'price_change': round(call_price_change, 2),
                        'signal': 'COVERING'
                    })

            # Detect Put patterns
            if abs(put_oi_change) > 5:
                if put_price_change > 2 and put_oi_change > 5:
                    patterns['short_buildup'].append({
                        'strike': strike,
                        'type': 'PUT',
                        'oi_change': round(put_oi_change, 2),
                        'price_change': round(put_price_change, 2),
                        'signal': 'BEARISH'
                    })
                elif put_price_change < -2 and put_oi_change > 5:
                    patterns['long_buildup'].append({
                        'strike': strike,
                        'type': 'PUT',
                        'oi_change': round(put_oi_change, 2),
                        'price_change': round(put_price_change, 2),
                        'signal': 'SUPPORT'
                    })
                elif put_price_change < -2 and put_oi_change < -5:
                    patterns['short_unwinding'].append({
                        'strike': strike,
                        'type': 'PUT',
                        'oi_change': round(put_oi_change, 2),
                        'price_change': round(put_price_change, 2),
                        'signal': 'BULLISH'
                    })
                elif put_price_change > 2 and put_oi_change < -5:
                    patterns['long_unwinding'].append({
                        'strike': strike,
                        'type': 'PUT',
                        'oi_change': round(put_oi_change, 2),
                        'price_change': round(put_price_change, 2),
                        'signal': 'WEAK_SUPPORT'
                    })

        return patterns

    def analyze_volume_oi_ratio(self, data):
        """Analyze Volume to OI ratio for genuine vs speculative moves"""
        records = data.get('records', {}).get('data', [])
        analysis = {'high_conviction': [], 'speculative': []}

        for record in records:
            strike = record.get('strikePrice')

            # Call analysis
            if 'CE' in record:
                call_vol = record['CE'].get('totalTradedVolume', 0)
                call_oi = record['CE'].get('openInterest', 0)
                call_vol_oi_ratio = (call_vol / call_oi) if call_oi > 0 else 0

                if call_vol_oi_ratio < 0.3 and call_oi > 100000:  # Low ratio = institutional
                    analysis['high_conviction'].append({
                        'strike': strike,
                        'type': 'CALL',
                        'volume': call_vol,
                        'oi': call_oi,
                        'ratio': round(call_vol_oi_ratio, 3),
                        'signal': 'STRONG RESISTANCE'
                    })
                elif call_vol_oi_ratio > 2 and call_vol > 50000:  # High ratio = speculative
                    analysis['speculative'].append({
                        'strike': strike,
                        'type': 'CALL',
                        'volume': call_vol,
                        'oi': call_oi,
                        'ratio': round(call_vol_oi_ratio, 3),
                        'signal': 'INTRADAY ACTIVITY'
                    })

            # Put analysis
            if 'PE' in record:
                put_vol = record['PE'].get('totalTradedVolume', 0)
                put_oi = record['PE'].get('openInterest', 0)
                put_vol_oi_ratio = (put_vol / put_oi) if put_oi > 0 else 0

                if put_vol_oi_ratio < 0.3 and put_oi > 100000:
                    analysis['high_conviction'].append({
                        'strike': strike,
                        'type': 'PUT',
                        'volume': put_vol,
                        'oi': put_oi,
                        'ratio': round(put_vol_oi_ratio, 3),
                        'signal': 'STRONG SUPPORT'
                    })
                elif put_vol_oi_ratio > 2 and put_vol > 50000:
                    analysis['speculative'].append({
                        'strike': strike,
                        'type': 'PUT',
                        'volume': put_vol,
                        'oi': put_oi,
                        'ratio': round(put_vol_oi_ratio, 3),
                        'signal': 'INTRADAY ACTIVITY'
                    })

        return analysis

    def calculate_gamma_exposure(self, data):
        """Calculate Gamma Exposure at each strike to detect squeeze zones"""
        records = data.get('records', {}).get('data', [])
        spot_price = data.get('records', {}).get('underlyingValue', 0)
        gamma_exposure = {}

        for record in records:
            strike = record.get('strikePrice')

            # Simplified gamma calculation (ATM options have highest gamma)
            distance_from_spot = abs(strike - spot_price)
            gamma_factor = max(0, 100 - (distance_from_spot / spot_price * 100))

            call_oi = record.get('CE', {}).get('openInterest', 0)
            put_oi = record.get('PE', {}).get('openInterest', 0)

            # Net gamma exposure (positive = bullish pressure, negative = bearish)
            net_gamma = (call_oi - put_oi) * gamma_factor

            if abs(net_gamma) > 100000:  # Significant gamma exposure
                gamma_exposure[strike] = {
                    'net_gamma': round(net_gamma, 2),
                    'call_oi': call_oi,
                    'put_oi': put_oi,
                    'distance_from_spot': round(distance_from_spot, 2),
                    'squeeze_potential': 'HIGH' if abs(net_gamma) > 500000 else 'MODERATE'
                }

        return gamma_exposure

    def analyze_strike_concentration(self, data):
        """Identify strike prices with abnormal OI concentration"""
        records = data.get('records', {}).get('data', [])
        spot = data.get('records', {}).get('underlyingValue', 0)

        # Calculate average OI
        call_ois = [r.get('CE', {}).get('openInterest', 0) for r in records if 'CE' in r]
        put_ois = [r.get('PE', {}).get('openInterest', 0) for r in records if 'PE' in r]

        avg_call_oi = sum(call_ois) / len(call_ois) if call_ois else 0
        avg_put_oi = sum(put_ois) / len(put_ois) if put_ois else 0

        concentrations = []

        for record in records:
            strike = record.get('strikePrice')

            call_oi = record.get('CE', {}).get('openInterest', 0)
            put_oi = record.get('PE', {}).get('openInterest', 0)

            # Strikes with 2x average OI
            if call_oi > avg_call_oi * 2:
                concentrations.append({
                    'strike': strike,
                    'type': 'CALL',
                    'oi': call_oi,
                    'avg_ratio': round(call_oi / avg_call_oi, 2),
                    'distance_from_spot': round(((strike - spot) / spot) * 100, 2),
                    'level': 'RESISTANCE'
                })

            if put_oi > avg_put_oi * 2:
                concentrations.append({
                    'strike': strike,
                    'type': 'PUT',
                    'oi': put_oi,
                    'avg_ratio': round(put_oi / avg_put_oi, 2),
                    'distance_from_spot': round(((strike - spot) / spot) * 100, 2),
                    'level': 'SUPPORT'
                })

        return sorted(concentrations, key=lambda x: x['avg_ratio'], reverse=True)[:10]

    def calculate_momentum_score(self, data, patterns, gamma_data):
        """Calculate comprehensive momentum score (-5 to +5)"""
        score = 0
        factors = []

        # PCR contribution
        pcr_data = self.calculate_enhanced_pcr(data)
        pcr_oi = pcr_data['pcr_oi']

        if pcr_oi > 1.3:
            score += 2
            factors.append("PCR Bullish +2")
        elif pcr_oi < 0.7:
            score -= 2
            factors.append("PCR Bearish -2")

        # OI buildup patterns contribution
        long_buildup_count = len(patterns.get('long_buildup', []))
        short_buildup_count = len(patterns.get('short_buildup', []))

        if long_buildup_count > short_buildup_count:
            score += 1
            factors.append(f"Long Buildup Dominance +1 ({long_buildup_count} vs {short_buildup_count})")
        elif short_buildup_count > long_buildup_count:
            score -= 1
            factors.append(f"Short Buildup Dominance -1 ({short_buildup_count} vs {long_buildup_count})")

        # Unwinding patterns
        put_unwind = len([p for p in patterns.get('short_unwinding', []) if p['type'] == 'PUT'])
        call_unwind = len([p for p in patterns.get('long_unwinding', []) if p['type'] == 'CALL'])

        if put_unwind > 3:
            score += 1
            factors.append(f"Put Unwinding (Bullish) +1")
        if call_unwind > 3:
            score -= 1
            factors.append(f"Call Unwinding (Bearish) -1")

        # Gamma exposure bias
        total_gamma = sum([v['net_gamma'] for v in gamma_data.values()])
        if total_gamma > 1000000:
            score += 1
            factors.append("Positive Gamma Exposure +1")
        elif total_gamma < -1000000:
            score -= 1
            factors.append("Negative Gamma Exposure -1")

        return {
            'score': max(-5, min(5, score)),
            'interpretation': self._interpret_momentum(score),
            'factors': factors
        }

    def _interpret_momentum(self, score):
        """Interpret momentum score"""
        if score >= 3:
            return "🟢 STRONG BULLISH"
        elif score >= 1:
            return "🟢 BULLISH"
        elif score <= -3:
            return "🔴 STRONG BEARISH"
        elif score <= -1:
            return "🔴 BEARISH"
        else:
            return "⚪ NEUTRAL"

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
            # Momentum Score Signal (Primary)
            momentum = analysis.get('momentum', {})
            if momentum:
                signals.append(f"🎯 MOMENTUM: {momentum.get('interpretation')} (Score: {momentum.get('score')})")

            # PCR-based signals
            pcr_data = analysis.get('pcr_data', {})
            pcr = pcr_data.get('pcr_oi', 0)
            if pcr > 1.3:
                signals.append("🔵 BULLISH: High PCR indicates oversold condition - Consider CALLS")
            elif pcr < 0.7:
                signals.append("🔴 BEARISH: Low PCR indicates overbought condition - Consider PUTS")
            elif 1.0 <= pcr <= 1.2:
                signals.append("⚪ NEUTRAL: PCR in normal range - Range-bound market")

            # OI Buildup Patterns
            oi_patterns = analysis.get('oi_patterns', {})
            long_buildup = oi_patterns.get('long_buildup', [])
            short_buildup = oi_patterns.get('short_buildup', [])

            if len(long_buildup) > 3:
                signals.append(f"📈 {len(long_buildup)} LONG BUILDUP patterns detected - BULLISH")
            if len(short_buildup) > 3:
                signals.append(f"📉 {len(short_buildup)} SHORT BUILDUP patterns detected - BEARISH")

            # Gamma Squeeze Alert
            gamma_data = analysis.get('gamma_exposure', {})
            high_gamma_strikes = [s for s, d in gamma_data.items() if d.get('squeeze_potential') == 'HIGH']
            if high_gamma_strikes:
                signals.append(f"⚡ HIGH GAMMA SQUEEZE potential at strikes: {high_gamma_strikes[:3]}")

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
        """Enhanced comprehensive analysis"""
        analysis = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'spot_price': data.get('records', {}).get('underlyingValue', 0)
        }

        # Enhanced PCR
        analysis['pcr_data'] = self.calculate_enhanced_pcr(data)
        analysis['pcr'] = analysis['pcr_data']['pcr_oi']

        # OI Buildup/Unwinding Patterns
        analysis['oi_patterns'] = self.detect_oi_buildup_patterns(data, self.previous_data)

        # Volume/OI Ratio Analysis
        analysis['vol_oi_analysis'] = self.analyze_volume_oi_ratio(data)

        # Gamma Exposure
        analysis['gamma_exposure'] = self.calculate_gamma_exposure(data)

        # Strike Concentration
        analysis['strike_concentration'] = self.analyze_strike_concentration(data)

        # Original methods
        analysis['max_pain'] = self.find_max_pain(data)
        analysis['levels'] = self.identify_support_resistance(data)
        analysis['iv_metrics'] = self.calculate_iv_metrics(data)
        analysis['oi_changes'] = self.analyze_oi_changes(data, self.previous_data)

        # Momentum Score (uses above data)
        analysis['momentum'] = self.calculate_momentum_score(
            data, 
            analysis['oi_patterns'],
            analysis['gamma_exposure']
        )

        # Generate signals
        analysis['signals'] = self.generate_signals(analysis)

        return analysis

    def display_analysis(self, analysis):
        """Display the comprehensive analysis results"""
        print("\n" + "="*120)
        print(f"🎯 NIFTY OPTION CHAIN ANALYSIS - {analysis['timestamp']}")
        print("="*120)

        # Basic Info
        print(f"\n📊 SPOT PRICE: {analysis['spot_price']}")
        print(f"📊 MAX PAIN LEVEL: {analysis['max_pain']}")

        # Momentum Score
        momentum = analysis.get('momentum', {})
        print(f"\n{'='*50}")
        print(f"🎯 MARKET MOMENTUM: {momentum.get('interpretation')} (Score: {momentum.get('score')}/5)")
        print(f"{'='*50}")
        for factor in momentum.get('factors', []):
            print(f"   • {factor}")

        # Enhanced PCR Metrics
        pcr_data = analysis.get('pcr_data', {})
        print(f"\n📊 PUT-CALL RATIO ANALYSIS:")
        print(f"   • OI PCR: {pcr_data.get('pcr_oi')} (Total Call OI: {pcr_data.get('total_call_oi'):,}, Total Put OI: {pcr_data.get('total_put_oi'):,})")
        print(f"   • Volume PCR: {pcr_data.get('pcr_volume')} (Call Vol: {pcr_data.get('total_call_volume'):,}, Put Vol: {pcr_data.get('total_put_volume'):,})")
        print(f"   • ATM PCR: {pcr_data.get('pcr_atm')}")
        print(f"   Interpretation:")
        for interp in pcr_data.get('interpretation', []):
            print(f"      - {interp}")

        # OI Buildup Patterns
        oi_patterns = analysis.get('oi_patterns', {})
        print(f"\n📈 OI BUILDUP/UNWINDING PATTERNS:")
        print(f"   • Long Buildup: {len(oi_patterns.get('long_buildup', []))} strikes")
        if oi_patterns.get('long_buildup'):
            for pattern in oi_patterns['long_buildup'][:5]:
                print(f"      {pattern['strike']} {pattern['type']}: OI {pattern['oi_change']:+.1f}%, Price {pattern['price_change']:+.1f}% → {pattern['signal']}")

        print(f"   • Short Buildup: {len(oi_patterns.get('short_buildup', []))} strikes")
        if oi_patterns.get('short_buildup'):
            for pattern in oi_patterns['short_buildup'][:5]:
                print(f"      {pattern['strike']} {pattern['type']}: OI {pattern['oi_change']:+.1f}%, Price {pattern['price_change']:+.1f}% → {pattern['signal']}")

        print(f"   • Long Unwinding: {len(oi_patterns.get('long_unwinding', []))} strikes")
        print(f"   • Short Unwinding: {len(oi_patterns.get('short_unwinding', []))} strikes")

        # Gamma Exposure
        gamma_data = analysis.get('gamma_exposure', {})
        if gamma_data:
            print(f"\n⚡ GAMMA EXPOSURE (Top 5 Strikes):")
            sorted_gamma = sorted(gamma_data.items(), key=lambda x: abs(x[1]['net_gamma']), reverse=True)[:5]
            for strike, data in sorted_gamma:
                print(f"   • Strike {strike}: Net Gamma {data['net_gamma']:,.0f}, Squeeze: {data['squeeze_potential']}")

        # Volume/OI Ratio Analysis
        vol_oi_analysis = analysis.get('vol_oi_analysis', {})
        high_conviction = vol_oi_analysis.get('high_conviction', [])
        if high_conviction:
            print(f"\n💪 HIGH CONVICTION POSITIONS (Vol/OI < 0.3):")
            for pos in high_conviction[:5]:
                print(f"   • {pos['strike']} {pos['type']}: Ratio {pos['ratio']}, OI {pos['oi']:,} → {pos['signal']}")

        # Strike Concentration
        concentrations = analysis.get('strike_concentration', [])
        if concentrations:
            print(f"\n🎯 STRIKE CONCENTRATION (>2x Average OI):")
            for conc in concentrations[:5]:
                print(f"   • {conc['strike']} {conc['type']}: {conc['avg_ratio']}x average, {conc['distance_from_spot']:+.2f}% from spot → {conc['level']}")

        # IV Metrics
        iv_metrics = analysis.get('iv_metrics', {})
        if iv_metrics:
            print(f"\n📊 IMPLIED VOLATILITY:")
            print(f"   • ATM Strike: {iv_metrics.get('atm_strike')}")
            print(f"   • ATM Call IV: {iv_metrics.get('atm_call_iv')}%")
            print(f"   • ATM Put IV: {iv_metrics.get('atm_put_iv')}%")
            print(f"   • IV Skew: {iv_metrics.get('iv_skew')}%")

        # Support/Resistance Levels
        levels = analysis.get('levels', {})
        if levels:
            print(f"\n🎯 KEY LEVELS:")
            print(f"   🔴 Resistance: {levels.get('resistance', [])}")
            print(f"   🟢 Support: {levels.get('support', [])}")
            print(f"   🔴 Max Call OI: {levels.get('max_call_oi_strike')} ({levels.get('max_call_oi_value', 0):,})")
            print(f"   🟢 Max Put OI: {levels.get('max_put_oi_strike')} ({levels.get('max_put_oi_value', 0):,})")

        # Display significant OI changes
        oi_changes = analysis.get('oi_changes', {})
        if oi_changes:
            print(f"\n⚡ SIGNIFICANT OI CHANGES (Since Last Update):")
            for strike, changes in sorted(oi_changes.items())[:10]:
                print(f"   Strike {strike}: Call OI {changes['call_oi_change']:+,} (Vol: {changes['call_volume']:,}), "
                      f"Put OI {changes['put_oi_change']:+,} (Vol: {changes['put_volume']:,})")

        # Display signals
        signals = analysis.get('signals', [])
        if signals:
            print(f"\n{'='*50}")
            print(f"🚨 TRADING SIGNALS:")
            print(f"{'='*50}")
            for i, signal in enumerate(signals, 1):
                print(f"{i}. {signal}")
        else:
            print(f"\n⚪ No significant signals detected")

        print("="*120)

    def save_to_csv(self, analysis):
        """Save analysis to CSV file"""
        try:
            # Flatten the analysis for CSV
            momentum = analysis.get('momentum', {})
            pcr_data = analysis.get('pcr_data', {})

            csv_data = {
                'timestamp': analysis['timestamp'],
                'spot_price': analysis['spot_price'],
                'pcr_oi': pcr_data.get('pcr_oi', 0),
                'pcr_volume': pcr_data.get('pcr_volume', 0),
                'pcr_atm': pcr_data.get('pcr_atm', 0),
                'max_pain': analysis['max_pain'],
                'momentum_score': momentum.get('score', 0),
                'momentum_interpretation': momentum.get('interpretation', ''),
                'long_buildup_count': len(analysis.get('oi_patterns', {}).get('long_buildup', [])),
                'short_buildup_count': len(analysis.get('oi_patterns', {}).get('short_buildup', [])),
                'signals_count': len(analysis.get('signals', [])),
                'oi_changes_count': len(analysis.get('oi_changes', {})),
                'atm_call_iv': analysis.get('iv_metrics', {}).get('atm_call_iv', 0),
                'atm_put_iv': analysis.get('iv_metrics', {}).get('atm_put_iv', 0),
                'iv_skew': analysis.get('iv_metrics', {}).get('iv_skew', 0),
                'high_conviction_count': len(analysis.get('vol_oi_analysis', {}).get('high_conviction', [])),
                'gamma_squeeze_strikes': len([k for k, v in analysis.get('gamma_exposure', {}).items() if v.get('squeeze_potential') == 'HIGH'])
            }

            self.analysis_results.append(csv_data)

            # Save to CSV every 5 records
            if len(self.analysis_results) % 5 == 0:
                df = pd.DataFrame(self.analysis_results)
                filename = f"nifty_option_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
                df.to_csv(filename, index=False)
                print(f"\n💾 Data saved to {filename}")

        except Exception as e:
            print(f"CSV save error: {e}")

    def run_continuous_analysis(self, interval_minutes=2):
        """Run continuous analysis every specified minutes"""
        print(f"\n🚀 Starting NIFTY Option Chain Analysis (Every {interval_minutes} minutes)")
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
                print(f"\n⏰ Next update in {interval_minutes} minutes...")
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


# MAIN EXECUTION
if __name__ == "__main__":
    print("🎯 ENHANCED NIFTY Option Chain Analyzer v2.0")
    print("="*60)
    print("New Features:")
    print("  ✅ OI Buildup/Unwinding Pattern Detection")
    print("  ✅ Volume/OI Ratio Analysis")
    print("  ✅ Gamma Exposure Calculation")
    print("  ✅ Enhanced PCR (OI, Volume, ATM)")
    print("  ✅ Market Momentum Score (-5 to +5)")
    print("  ✅ Strike Concentration Analysis")
    print("="*60)

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
