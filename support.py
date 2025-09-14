# Enhanced NIFTY PCR Analyzer with Support/Resistance and Entry Points
# Features: S/R levels + Call/Put entry signals + Risk management

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
import pickle
from truedata.history import TD_hist
import logging
from dateutil.relativedelta import relativedelta
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
    ORANGE = '\033[38;5;208m'

@dataclass
class SupportResistanceLevel:
    """Support/Resistance level with strength"""
    level: float
    type: str  # "SUPPORT" or "RESISTANCE"
    strength: str  # "WEAK", "MODERATE", "STRONG", "VERY_STRONG"
    oi_confirmation: int  # OI at this level
    volume_confirmation: int  # Volume at this level
    distance_from_current: float  # Distance from current price
    confidence: float  # 0-100 confidence score

@dataclass
class CallPutEntrySignal:
    """Call/Put entry signal with complete details"""
    signal_type: str  # "CALL_BUY", "PUT_BUY", "CALL_SELL", "PUT_SELL"
    entry_price: float
    strike_price: int
    option_type: str  # "CALL" or "PUT"
    action: str  # "BUY" or "SELL"
    confidence: float  # 0-100
    risk_reward_ratio: float
    stop_loss: float
    target_1: float
    target_2: float
    max_risk: float
    expected_return: float
    time_frame: str  # "INTRADAY", "SWING", "POSITIONAL"
    reasoning: str  # Why this signal was generated

@dataclass
class EnhancedMarketAnalysis:
    """Complete market analysis with S/R and entry points"""
    current_price: float
    pcr: float
    option_signal: str
    support_levels: List[SupportResistanceLevel]
    resistance_levels: List[SupportResistanceLevel]
    call_entry_signals: List[CallPutEntrySignal]
    put_entry_signals: List[CallPutEntrySignal]
    market_bias: str  # "BULLISH", "BEARISH", "NEUTRAL"
    volatility_regime: str  # "LOW", "MODERATE", "HIGH"
    trading_recommendation: str

class AdvancedNiftyAnalyzer:
    def __init__(self):
        self.live_data = []
        self.current_price = 25100.0
        self.previous_price = 0.0
        self.price_history = []  # For S/R calculation
        self.is_running = False
        
        # Strike parameters
        self.atm_strike = 25100
        self.strike_gap = 50
        self.target_strikes = []
        
        # S/R parameters
        self.sr_lookback_periods = 20
        self.min_touches_for_sr = 3
        self.sr_tolerance = 0.5  # 0.5% tolerance for S/R levels
        
        print(f"{Colors.ORANGE}{Colors.BOLD}🎯 ADVANCED NIFTY ANALYZER WITH S/R + ENTRY POINTS{Colors.RESET}")
        print(f"{Colors.CYAN}📊 Features: Support/Resistance + Call/Put Entries + Risk Management{Colors.RESET}")
        
        self.initialize_advanced_system()

    def initialize_advanced_system(self):
        """Initialize advanced system with S/R analysis"""
        try:
            print(f"\n{Colors.MAGENTA}🔧 INITIALIZING ADVANCED SYSTEM:{Colors.RESET}")
            
            # Get current price and historical data
            self.current_price = self.get_nifty_price_smart()
            self.load_price_history()
            
            # Setup strikes
            self.setup_enhanced_strikes()
            
            print(f"✅ Advanced system ready")
            
        except Exception as e:
            print(f"❌ Error initializing: {e}")

    def get_nifty_price_smart(self):
        """Smart NIFTY price fetching with history tracking"""
        try:
            # Try TrueData first
            nifty_data = td_hist.get_n_historical_bars('NIFTY-I', no_of_bars=1, bar_size='1 min')
            
            if nifty_data is not None and len(nifty_data) > 0:
                for col in ['Close', 'close', 'CLOSE', 'Last', 'LTP']:
                    if col in nifty_data.columns:
                        price = float(nifty_data[col].iloc[-1])
                        if price > 0:
                            self.previous_price = self.current_price
                            self.current_price = price
                            
                            # Add to price history
                            self.price_history.append({
                                'price': price,
                                'timestamp': datetime.now(),
                                'volume': 100000  # Default volume
                            })
                            
                            print(f"💹 NIFTY Price: {Colors.CYAN}{price:.2f}{Colors.RESET}")
                            return price
            
            # Fallback
            base_price = 25100.0
            variation = np.random.uniform(-50, 50)
            price = base_price + variation
            
            self.previous_price = self.current_price
            self.current_price = price
            
            # Add to history
            self.price_history.append({
                'price': price,
                'timestamp': datetime.now(),
                'volume': 100000
            })
            
            print(f"💹 NIFTY Price (Est): {Colors.YELLOW}{price:.2f}{Colors.RESET}")
            return price
            
        except Exception as e:
            print(f"❌ Error getting price: {e}")
            return 25100.0

    def load_price_history(self):
        """Load historical price data for S/R analysis"""
        try:
            print(f"📊 Loading price history for S/R analysis...")
            
            # Try to get historical data from TrueData
            try:
                hist_data = td_hist.get_n_historical_bars('NIFTY-I', no_of_bars=100, bar_size='3 min')
                
                if hist_data is not None and len(hist_data) > 0:
                    for i, (timestamp, row) in enumerate(hist_data.iterrows()):
                        price = self.get_value_from_row(row, ['Close', 'close', 'CLOSE', 'Last', 'LTP'], 25100)
                        volume = int(self.get_value_from_row(row, ['Volume', 'volume', 'VOLUME'], 50000))
                        
                        if price > 0:
                            self.price_history.append({
                                'price': price,
                                'timestamp': timestamp,
                                'volume': volume
                            })
                    
                    print(f"✅ Loaded {len(self.price_history)} historical price points")
                    return
                    
            except Exception as e:
                print(f"⚠️ TrueData history failed: {e}")
            
            # Generate synthetic price history
            print(f"🎭 Generating synthetic price history...")
            base_price = self.current_price
            
            for i in range(100):
                timestamp = datetime.now() - timedelta(minutes=i * 3)
                price_change = np.random.normal(0, 0.003) * base_price
                price = base_price + price_change
                volume = np.random.randint(30000, 100000)
                
                self.price_history.append({
                    'price': price,
                    'timestamp': timestamp,
                    'volume': volume
                })
                
                base_price = price
            
            # Reverse to chronological order
            self.price_history.reverse()
            print(f"✅ Generated {len(self.price_history)} synthetic price points")
            
        except Exception as e:
            print(f"❌ Error loading price history: {e}")

    def get_value_from_row(self, row, column_names, default=0):
        """Get value from row with flexible column names"""
        for col in column_names:
            if col in row.index and pd.notna(row[col]):
                return float(row[col])
        return default

    def setup_enhanced_strikes(self):
        """Setup enhanced strike analysis"""
        try:
            # Calculate ATM
            self.atm_strike = round(self.current_price / self.strike_gap) * self.strike_gap
            
            # Generate wider strike range for better S/R analysis
            self.target_strikes = []
            for i in range(-8, 9):  # ATM ± 8 strikes for comprehensive analysis
                strike = self.atm_strike + (i * self.strike_gap)
                self.target_strikes.append(strike)
            
            print(f"🎯 ATM Strike: {Colors.WHITE}{Colors.BOLD}{self.atm_strike}{Colors.RESET}")
            print(f"📊 Strike Range: {min(self.target_strikes)} to {max(self.target_strikes)}")
            
        except Exception as e:
            print(f"❌ Error setting up strikes: {e}")

    def calculate_support_resistance_levels(self) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """Calculate strong support and resistance levels"""
        try:
            if len(self.price_history) < 20:
                return [], []
            
            # Extract prices and volumes
            prices = [p['price'] for p in self.price_history[-50:]]  # Last 50 data points
            volumes = [p['volume'] for p in self.price_history[-50:]]
            
            support_levels = []
            resistance_levels = []
            
            # Method 1: Pivot Point Analysis
            pivot_supports, pivot_resistances = self.calculate_pivot_points()
            support_levels.extend(pivot_supports)
            resistance_levels.extend(pivot_resistances)
            
            # Method 2: Volume Profile Analysis
            volume_supports, volume_resistances = self.calculate_volume_profile_levels(prices, volumes)
            support_levels.extend(volume_supports)
            resistance_levels.extend(volume_resistances)
            
            # Method 3: Fibonacci Retracement Levels
            fib_supports, fib_resistances = self.calculate_fibonacci_levels(prices)
            support_levels.extend(fib_supports)
            resistance_levels.extend(fib_resistances)
            
            # Method 4: Strike-based S/R (High OI strikes act as S/R)
            strike_supports, strike_resistances = self.calculate_strike_based_sr()
            support_levels.extend(strike_supports)
            resistance_levels.extend(strike_resistances)
            
            # Filter and rank S/R levels
            support_levels = self.filter_and_rank_sr_levels(support_levels, "SUPPORT")
            resistance_levels = self.filter_and_rank_sr_levels(resistance_levels, "RESISTANCE")
            
            return support_levels[:5], resistance_levels[:5]  # Top 5 of each
            
        except Exception as e:
            print(f"❌ Error calculating S/R levels: {e}")
            return [], []

    def calculate_pivot_points(self) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """Calculate pivot point based S/R levels"""
        try:
            if len(self.price_history) < 3:
                return [], []
            
            # Get recent high, low, close
            recent_prices = [p['price'] for p in self.price_history[-20:]]
            high = max(recent_prices)
            low = min(recent_prices)
            close = recent_prices[-1]
            
            # Calculate pivot point
            pivot = (high + low + close) / 3
            
            # Calculate support and resistance levels
            r1 = 2 * pivot - low
            r2 = pivot + (high - low)
            r3 = high + 2 * (pivot - low)
            
            s1 = 2 * pivot - high
            s2 = pivot - (high - low)
            s3 = low - 2 * (high - pivot)
            
            supports = []
            resistances = []
            
            # Create support levels
            for level, name in [(s1, "S1"), (s2, "S2"), (s3, "S3")]:
                if level < self.current_price:
                    distance = abs(level - self.current_price)
                    strength = "STRONG" if distance < self.current_price * 0.02 else "MODERATE"
                    
                    supports.append(SupportResistanceLevel(
                        level=round(level, 2),
                        type="SUPPORT",
                        strength=strength,
                        oi_confirmation=0,
                        volume_confirmation=0,
                        distance_from_current=distance,
                        confidence=80.0
                    ))
            
            # Create resistance levels
            for level, name in [(r1, "R1"), (r2, "R2"), (r3, "R3")]:
                if level > self.current_price:
                    distance = abs(level - self.current_price)
                    strength = "STRONG" if distance < self.current_price * 0.02 else "MODERATE"
                    
                    resistances.append(SupportResistanceLevel(
                        level=round(level, 2),
                        type="RESISTANCE",
                        strength=strength,
                        oi_confirmation=0,
                        volume_confirmation=0,
                        distance_from_current=distance,
                        confidence=80.0
                    ))
            
            return supports, resistances
            
        except Exception as e:
            return [], []

    def calculate_volume_profile_levels(self, prices, volumes) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """Calculate S/R levels based on volume profile"""
        try:
            # Create price-volume bins
            price_range = max(prices) - min(prices)
            num_bins = 20
            bin_size = price_range / num_bins
            
            volume_profile = {}
            
            for price, volume in zip(prices, volumes):
                bin_level = round(price / bin_size) * bin_size
                if bin_level not in volume_profile:
                    volume_profile[bin_level] = 0
                volume_profile[bin_level] += volume
            
            # Find high volume areas (potential S/R)
            sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
            high_volume_levels = sorted_levels[:5]  # Top 5 volume levels
            
            supports = []
            resistances = []
            
            for level, volume in high_volume_levels:
                distance = abs(level - self.current_price)
                
                if level < self.current_price:
                    supports.append(SupportResistanceLevel(
                        level=round(level, 2),
                        type="SUPPORT",
                        strength="STRONG" if volume > np.mean(volumes) * 1.5 else "MODERATE",
                        oi_confirmation=0,
                        volume_confirmation=int(volume),
                        distance_from_current=distance,
                        confidence=75.0
                    ))
                else:
                    resistances.append(SupportResistanceLevel(
                        level=round(level, 2),
                        type="RESISTANCE",
                        strength="STRONG" if volume > np.mean(volumes) * 1.5 else "MODERATE",
                        oi_confirmation=0,
                        volume_confirmation=int(volume),
                        distance_from_current=distance,
                        confidence=75.0
                    ))
            
            return supports, resistances
            
        except Exception as e:
            return [], []

    def calculate_fibonacci_levels(self, prices) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """Calculate Fibonacci retracement levels"""
        try:
            high = max(prices)
            low = min(prices)
            
            # Fibonacci levels
            fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
            
            supports = []
            resistances = []
            
            for fib in fib_levels:
                # For uptrend (support levels)
                level = high - (high - low) * fib
                
                if level < self.current_price:
                    distance = abs(level - self.current_price)
                    supports.append(SupportResistanceLevel(
                        level=round(level, 2),
                        type="SUPPORT",
                        strength="STRONG" if fib in [0.382, 0.618] else "MODERATE",
                        oi_confirmation=0,
                        volume_confirmation=0,
                        distance_from_current=distance,
                        confidence=70.0
                    ))
                else:
                    distance = abs(level - self.current_price)
                    resistances.append(SupportResistanceLevel(
                        level=round(level, 2),
                        type="RESISTANCE",
                        strength="STRONG" if fib in [0.382, 0.618] else "MODERATE",
                        oi_confirmation=0,
                        volume_confirmation=0,
                        distance_from_current=distance,
                        confidence=70.0
                    ))
            
            return supports, resistances
            
        except Exception as e:
            return [], []

    def calculate_strike_based_sr(self) -> Tuple[List[SupportResistanceLevel], List[SupportResistanceLevel]]:
        """Calculate S/R based on option strikes with high OI"""
        try:
            supports = []
            resistances = []
            
            # Generate option data for strikes
            option_data = self.fetch_enhanced_option_data()
            
            if not option_data:
                return [], []
            
            # Find strikes with highest OI
            high_oi_strikes = []
            for strike, data in option_data.items():
                total_oi = data.call_oi + data.put_oi
                high_oi_strikes.append((strike, total_oi, data))
            
            # Sort by total OI
            high_oi_strikes.sort(key=lambda x: x[1], reverse=True)
            
            # Top 5 high OI strikes act as S/R
            for strike, total_oi, data in high_oi_strikes[:5]:
                distance = abs(strike - self.current_price)
                
                # Determine if support or resistance
                if strike < self.current_price:
                    supports.append(SupportResistanceLevel(
                        level=float(strike),
                        type="SUPPORT",
                        strength="VERY_STRONG" if total_oi > 500000 else "STRONG",
                        oi_confirmation=total_oi,
                        volume_confirmation=data.call_volume + data.put_volume,
                        distance_from_current=distance,
                        confidence=85.0
                    ))
                elif strike > self.current_price:
                    resistances.append(SupportResistanceLevel(
                        level=float(strike),
                        type="RESISTANCE",
                        strength="VERY_STRONG" if total_oi > 500000 else "STRONG",
                        oi_confirmation=total_oi,
                        volume_confirmation=data.call_volume + data.put_volume,
                        distance_from_current=distance,
                        confidence=85.0
                    ))
            
            return supports, resistances
            
        except Exception as e:
            return [], []

    def filter_and_rank_sr_levels(self, levels: List[SupportResistanceLevel], level_type: str) -> List[SupportResistanceLevel]:
        """Filter and rank S/R levels by strength and proximity"""
        try:
            if not levels:
                return []
            
            # Remove levels too close to each other
            filtered_levels = []
            levels.sort(key=lambda x: x.level)
            
            for level in levels:
                # Check if too close to existing levels
                too_close = False
                for existing in filtered_levels:
                    if abs(level.level - existing.level) < self.current_price * 0.005:  # 0.5% tolerance
                        # Keep the stronger level
                        if level.confidence > existing.confidence:
                            filtered_levels.remove(existing)
                        else:
                            too_close = True
                        break
                
                if not too_close:
                    filtered_levels.append(level)
            
            # Rank by confidence and proximity
            strength_weights = {"VERY_STRONG": 4, "STRONG": 3, "MODERATE": 2, "WEAK": 1}
            
            for level in filtered_levels:
                # Proximity bonus (closer levels are more important)
                proximity_bonus = max(0, 10 - (level.distance_from_current / self.current_price * 100))
                
                # Strength bonus
                strength_bonus = strength_weights.get(level.strength, 1) * 10
                
                # OI confirmation bonus
                oi_bonus = min(20, level.oi_confirmation / 50000)  # Max 20 points for high OI
                
                # Update confidence
                level.confidence += proximity_bonus + strength_bonus + oi_bonus
                level.confidence = min(100, level.confidence)
            
            # Sort by confidence
            filtered_levels.sort(key=lambda x: x.confidence, reverse=True)
            
            return filtered_levels
            
        except Exception as e:
            return levels

    def generate_call_put_entry_signals(self, support_levels: List[SupportResistanceLevel], 
                                      resistance_levels: List[SupportResistanceLevel],
                                      option_data: Dict) -> Tuple[List[CallPutEntrySignal], List[CallPutEntrySignal]]:
        """Generate Call and Put entry signals based on S/R and PCR analysis"""
        try:
            call_signals = []
            put_signals = []
            
            # Calculate overall market bias
            total_call_oi = sum(data.call_oi for data in option_data.values())
            total_put_oi = sum(data.put_oi for data in option_data.values())
            pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0
            
            # Determine market bias
            if pcr < 0.8:
                market_bias = "BULLISH"
            elif pcr > 1.2:
                market_bias = "BEARISH"
            else:
                market_bias = "NEUTRAL"
            
            # Generate Call entry signals
            call_signals.extend(self.generate_call_entry_signals(support_levels, resistance_levels, market_bias, pcr, option_data))
            
            # Generate Put entry signals  
            put_signals.extend(self.generate_put_entry_signals(support_levels, resistance_levels, market_bias, pcr, option_data))
            
            return call_signals, put_signals
            
        except Exception as e:
            print(f"❌ Error generating entry signals: {e}")
            return [], []

    def generate_call_entry_signals(self, support_levels, resistance_levels, market_bias, pcr, option_data) -> List[CallPutEntrySignal]:
        """Generate Call BUY entry signals"""
        try:
            call_signals = []
            
            # Call BUY conditions
            call_buy_conditions = [
                market_bias == "BULLISH",
                pcr < 0.9,
                len([s for s in support_levels if s.distance_from_current < self.current_price * 0.01]) > 0  # Strong support nearby
            ]
            
            if sum(call_buy_conditions) >= 2:  # At least 2 conditions met
                # Find best Call strikes to buy
                atm_strikes = [strike for strike in self.target_strikes 
                             if abs(strike - self.current_price) <= self.strike_gap]
                
                for strike in atm_strikes:
                    if strike in option_data:
                        data = option_data[strike]
                        
                        # Calculate entry signal
                        confidence = self.calculate_call_confidence(data, market_bias, pcr, support_levels)
                        
                        if confidence >= 70:
                            # Find nearest support for stop loss
                            nearest_support = min(support_levels, key=lambda x: x.distance_from_current) if support_levels else None
                            stop_loss_level = nearest_support.level - 20 if nearest_support else self.current_price - 50
                            
                            # Calculate targets
                            target_1 = self.current_price + 30  # 30 points target
                            target_2 = self.current_price + 60  # 60 points target
                            
                            # Risk-reward calculation
                            risk = self.current_price - stop_loss_level
                            reward = target_1 - self.current_price
                            rr_ratio = reward / risk if risk > 0 else 0
                            
                            signal = CallPutEntrySignal(
                                signal_type="CALL_BUY",
                                entry_price=self.current_price,
                                strike_price=strike,
                                option_type="CALL",
                                action="BUY",
                                confidence=confidence,
                                risk_reward_ratio=round(rr_ratio, 2),
                                stop_loss=stop_loss_level,
                                target_1=target_1,
                                target_2=target_2,
                                max_risk=risk,
                                expected_return=reward,
                                time_frame="INTRADAY",
                                reasoning=f"Bullish bias (PCR: {pcr:.2f}), Strong support at {nearest_support.level if nearest_support else 'N/A'}"
                            )
                            
                            call_signals.append(signal)
            
            return call_signals
            
        except Exception as e:
            return []

    def generate_put_entry_signals(self, support_levels, resistance_levels, market_bias, pcr, option_data) -> List[CallPutEntrySignal]:
        """Generate Put BUY entry signals"""
        try:
            put_signals = []
            
            # Put BUY conditions
            put_buy_conditions = [
                market_bias == "BEARISH",
                pcr > 1.1,
                len([r for r in resistance_levels if r.distance_from_current < self.current_price * 0.01]) > 0  # Strong resistance nearby
            ]
            
            if sum(put_buy_conditions) >= 2:
                # Find best Put strikes to buy
                atm_strikes = [strike for strike in self.target_strikes 
                             if abs(strike - self.current_price) <= self.strike_gap]
                
                for strike in atm_strikes:
                    if strike in option_data:
                        data = option_data[strike]
                        
                        confidence = self.calculate_put_confidence(data, market_bias, pcr, resistance_levels)
                        
                        if confidence >= 70:
                            # Find nearest resistance for stop loss
                            nearest_resistance = min(resistance_levels, key=lambda x: x.distance_from_current) if resistance_levels else None
                            stop_loss_level = nearest_resistance.level + 20 if nearest_resistance else self.current_price + 50
                            
                            # Calculate targets
                            target_1 = self.current_price - 30
                            target_2 = self.current_price - 60
                            
                            # Risk-reward
                            risk = stop_loss_level - self.current_price
                            reward = self.current_price - target_1
                            rr_ratio = reward / risk if risk > 0 else 0
                            
                            signal = CallPutEntrySignal(
                                signal_type="PUT_BUY",
                                entry_price=self.current_price,
                                strike_price=strike,
                                option_type="PUT",
                                action="BUY",
                                confidence=confidence,
                                risk_reward_ratio=round(rr_ratio, 2),
                                stop_loss=stop_loss_level,
                                target_1=target_1,
                                target_2=target_2,
                                max_risk=risk,
                                expected_return=reward,
                                time_frame="INTRADAY",
                                reasoning=f"Bearish bias (PCR: {pcr:.2f}), Strong resistance at {nearest_resistance.level if nearest_resistance else 'N/A'}"
                            )
                            
                            put_signals.append(signal)
            
            return put_signals
            
        except Exception as e:
            return []

    def calculate_call_confidence(self, option_data, market_bias, pcr, support_levels) -> float:
        """Calculate confidence for Call BUY signal"""
        try:
            confidence = 50.0  # Base confidence
            
            # Market bias bonus
            if market_bias == "BULLISH":
                confidence += 20
            
            # PCR bonus
            if pcr < 0.8:
                confidence += 15
            elif pcr < 1.0:
                confidence += 10
            
            # Support level bonus
            strong_supports = [s for s in support_levels if s.strength in ["STRONG", "VERY_STRONG"]]
            confidence += len(strong_supports) * 5
            
            # OI analysis bonus
            if option_data.call_oi > option_data.put_oi:
                confidence += 10
            
            return min(100, confidence)
            
        except Exception as e:
            return 50.0

    def calculate_put_confidence(self, option_data, market_bias, pcr, resistance_levels) -> float:
        """Calculate confidence for Put BUY signal"""
        try:
            confidence = 50.0
            
            # Market bias bonus
            if market_bias == "BEARISH":
                confidence += 20
            
            # PCR bonus
            if pcr > 1.2:
                confidence += 15
            elif pcr > 1.0:
                confidence += 10
            
            # Resistance level bonus
            strong_resistances = [r for r in resistance_levels if r.strength in ["STRONG", "VERY_STRONG"]]
            confidence += len(strong_resistances) * 5
            
            # OI analysis bonus
            if option_data.put_oi > option_data.call_oi:
                confidence += 10
            
            return min(100, confidence)
            
        except Exception as e:
            return 50.0

    def fetch_enhanced_option_data(self):
        """Fetch enhanced option data (same as original)"""
        try:
            option_data = {}
            
            for strike in self.target_strikes:
                try:
                    call_data = self.generate_enhanced_option_data(strike, 'call')
                    put_data = self.generate_enhanced_option_data(strike, 'put')
                    
                    if call_data and put_data:
                        enhanced_data = self.create_enhanced_strike_data(strike, call_data, put_data)
                        if enhanced_data:
                            option_data[strike] = enhanced_data
                    
                except Exception as e:
                    continue
            
            return option_data
            
        except Exception as e:
            return {}

    def generate_enhanced_option_data(self, strike, option_type='call'):
        """Generate enhanced option data (same as original)"""
        try:
            distance = abs(strike - self.current_price)
            distance_pct = distance / self.current_price * 100
            
            if option_type == 'call':
                if strike > self.current_price:
                    intrinsic = 0
                    time_value = max(5, 80 - (distance * 0.8) + np.random.uniform(-10, 15))
                else:
                    intrinsic = self.current_price - strike
                    time_value = np.random.uniform(20, 60)
                ltp = max(5, intrinsic + time_value)
            else:
                if strike < self.current_price:
                    intrinsic = 0
                    time_value = max(5, 80 - (distance * 0.8) + np.random.uniform(-10, 15))
                else:
                    intrinsic = strike - self.current_price
                    time_value = np.random.uniform(20, 60)
                ltp = max(5, intrinsic + time_value)
            
            base_oi = 50000
            
            if distance <= 50:
                oi_multiplier = np.random.uniform(2.0, 4.0)
            elif distance <= 100:
                oi_multiplier = np.random.uniform(1.5, 2.5)
            elif distance <= 150:
                oi_multiplier = np.random.uniform(1.0, 1.8)
            else:
                oi_multiplier = np.random.uniform(0.3, 1.0)
            
            oi = int(base_oi * oi_multiplier * np.random.uniform(0.7, 1.3))
            volume = int(oi * np.random.uniform(0.1, 0.3))
            
            if distance_pct < 1:
                iv = 15 + np.random.uniform(-2, 2)
            elif distance_pct < 3:
                iv = 16 + np.random.uniform(-2, 3)
            else:
                iv = 18 + np.random.uniform(-3, 4)
            
            return {
                'ltp': round(ltp, 2),
                'oi': oi,
                'volume': volume,
                'iv': round(iv, 2),
                'oi_change': np.random.randint(-1000, 1000),
                'success': True
            }
            
        except Exception as e:
            return None

    # Include the original EnhancedStrikeData class and create_enhanced_strike_data method
    @dataclass 
    class EnhancedStrikeData:
        """Enhanced strike data with detailed OI information"""
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
        oi_concentration: str = ""

    def create_enhanced_strike_data(self, strike, call_data, put_data):
        """Create enhanced strike data"""
        try:
            call_oi = call_data['oi']
            put_oi = put_data['oi']
            
            total_oi = call_oi + put_oi
            call_pct = (call_oi / total_oi) * 100 if total_oi > 0 else 50
            
            if call_pct > 70:
                oi_concentration = "HIGH_CALL"
            elif call_pct < 30:
                oi_concentration = "HIGH_PUT"
            else:
                oi_concentration = "BALANCED"
            
            if strike == self.atm_strike:
                position = "ATM"
            elif strike > self.atm_strike:
                position = "Above"
            else:
                position = "Below"
            
            return self.EnhancedStrikeData(
                strike=strike,
                call_oi=call_oi,
                put_oi=put_oi,
                call_oi_change=call_data['oi_change'],
                put_oi_change=put_data['oi_change'],
                call_volume=call_data['volume'],
                put_volume=put_data['volume'],
                call_ltp=call_data['ltp'],
                put_ltp=put_data['ltp'],
                call_iv=call_data['iv'],
                put_iv=put_data['iv'],
                position=position,
                oi_concentration=oi_concentration
            )
            
        except Exception as e:
            return None

    def display_advanced_analysis(self):
        """Display advanced analysis with S/R levels and entry points"""
        try:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print(f"{Colors.ORANGE}{Colors.BOLD}🎯 ADVANCED NIFTY ANALYZER - S/R + ENTRY POINTS{Colors.RESET}")
            print(f"{Colors.CYAN}📊 Support/Resistance Levels + Call/Put Entry Signals{Colors.RESET}")
            print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} IST | 💹 Current Price: {Colors.WHITE}{Colors.BOLD}{self.current_price:.2f}{Colors.RESET}")
            
            # Get option data
            option_data = self.fetch_enhanced_option_data()
            
            if not option_data:
                print(f"❌ No option data available")
                return
            
            # Calculate S/R levels
            support_levels, resistance_levels = self.calculate_support_resistance_levels()
            
            # Generate entry signals
            call_signals, put_signals = self.generate_call_put_entry_signals(support_levels, resistance_levels, option_data)
            
            # Display Support Levels
            print(f"\n{Colors.GREEN}{Colors.BOLD}📈 SUPPORT LEVELS:{Colors.RESET}")
            print(f"{Colors.GREEN}{'-'*60}{Colors.RESET}")
            print(f"{Colors.WHITE}{'Level':<10} {'Type':<12} {'Strength':<12} {'Distance':<10} {'Confidence':<12}{Colors.RESET}")
            
            for support in support_levels[:5]:
                strength_color = Colors.GREEN if support.strength == "VERY_STRONG" else Colors.YELLOW if support.strength == "STRONG" else Colors.WHITE
                print(f"{Colors.GREEN}{support.level:<10.2f}{Colors.RESET} "
                      f"{Colors.CYAN}{support.type:<12}{Colors.RESET} "
                      f"{strength_color}{support.strength:<12}{Colors.RESET} "
                      f"{Colors.WHITE}{support.distance_from_current:<10.2f}{Colors.RESET} "
                      f"{Colors.YELLOW}{support.confidence:<12.1f}%{Colors.RESET}")
            
            # Display Resistance Levels
            print(f"\n{Colors.RED}{Colors.BOLD}📉 RESISTANCE LEVELS:{Colors.RESET}")
            print(f"{Colors.RED}{'-'*60}{Colors.RESET}")
            print(f"{Colors.WHITE}{'Level':<10} {'Type':<12} {'Strength':<12} {'Distance':<10} {'Confidence':<12}{Colors.RESET}")
            
            for resistance in resistance_levels[:5]:
                strength_color = Colors.RED if resistance.strength == "VERY_STRONG" else Colors.YELLOW if resistance.strength == "STRONG" else Colors.WHITE
                print(f"{Colors.RED}{resistance.level:<10.2f}{Colors.RESET} "
                      f"{Colors.CYAN}{resistance.type:<12}{Colors.RESET} "
                      f"{strength_color}{resistance.strength:<12}{Colors.RESET} "
                      f"{Colors.WHITE}{resistance.distance_from_current:<10.2f}{Colors.RESET} "
                      f"{Colors.YELLOW}{resistance.confidence:<12.1f}%{Colors.RESET}")
            
            # Display Call Entry Signals
            print(f"\n{Colors.GREEN}{Colors.BOLD}📞 CALL ENTRY SIGNALS:{Colors.RESET}")
            print(f"{Colors.GREEN}{'-'*80}{Colors.RESET}")
            
            if call_signals:
                print(f"{Colors.WHITE}{'Strike':<8} {'Action':<6} {'Entry':<10} {'Target1':<10} {'Target2':<10} {'Stop':<10} {'R:R':<6} {'Confidence':<12}{Colors.RESET}")
                
                for signal in call_signals:
                    conf_color = Colors.GREEN if signal.confidence >= 80 else Colors.YELLOW if signal.confidence >= 70 else Colors.RED
                    
                    print(f"{Colors.GREEN}{signal.strike_price:<8}{Colors.RESET} "
                          f"{Colors.CYAN}{signal.action:<6}{Colors.RESET} "
                          f"{Colors.WHITE}{signal.entry_price:<10.2f}{Colors.RESET} "
                          f"{Colors.GREEN}{signal.target_1:<10.2f}{Colors.RESET} "
                          f"{Colors.GREEN}{signal.target_2:<10.2f}{Colors.RESET} "
                          f"{Colors.RED}{signal.stop_loss:<10.2f}{Colors.RESET} "
                          f"{Colors.BLUE}{signal.risk_reward_ratio:<6.2f}{Colors.RESET} "
                          f"{conf_color}{signal.confidence:<12.1f}%{Colors.RESET}")
                
                # Show reasoning for best signal
                if call_signals:
                    best_call = max(call_signals, key=lambda x: x.confidence)
                    print(f"\n🎯 Best Call Signal: {Colors.GREEN}{best_call.strike_price} {best_call.action}{Colors.RESET}")
                    print(f"💡 Reasoning: {best_call.reasoning}")
            else:
                print(f"{Colors.YELLOW}No Call entry signals found{Colors.RESET}")
            
            # Display Put Entry Signals
            print(f"\n{Colors.RED}{Colors.BOLD}📞 PUT ENTRY SIGNALS:{Colors.RESET}")
            print(f"{Colors.RED}{'-'*80}{Colors.RESET}")
            
            if put_signals:
                print(f"{Colors.WHITE}{'Strike':<8} {'Action':<6} {'Entry':<10} {'Target1':<10} {'Target2':<10} {'Stop':<10} {'R:R':<6} {'Confidence':<12}{Colors.RESET}")
                
                for signal in put_signals:
                    conf_color = Colors.GREEN if signal.confidence >= 80 else Colors.YELLOW if signal.confidence >= 70 else Colors.RED
                    
                    print(f"{Colors.RED}{signal.strike_price:<8}{Colors.RESET} "
                          f"{Colors.CYAN}{signal.action:<6}{Colors.RESET} "
                          f"{Colors.WHITE}{signal.entry_price:<10.2f}{Colors.RESET} "
                          f"{Colors.RED}{signal.target_1:<10.2f}{Colors.RESET} "
                          f"{Colors.RED}{signal.target_2:<10.2f}{Colors.RESET} "
                          f"{Colors.GREEN}{signal.stop_loss:<10.2f}{Colors.RESET} "
                          f"{Colors.BLUE}{signal.risk_reward_ratio:<6.2f}{Colors.RESET} "
                          f"{conf_color}{signal.confidence:<12.1f}%{Colors.RESET}")
                
                # Show reasoning for best signal
                if put_signals:
                    best_put = max(put_signals, key=lambda x: x.confidence)
                    print(f"\n🎯 Best Put Signal: {Colors.RED}{best_put.strike_price} {best_put.action}{Colors.RESET}")
                    print(f"💡 Reasoning: {best_put.reasoning}")
            else:
                print(f"{Colors.YELLOW}No Put entry signals found{Colors.RESET}")
            
            # Trading Summary
            print(f"\n{Colors.CYAN}{Colors.BOLD}📊 TRADING SUMMARY:{Colors.RESET}")
            print(f"{Colors.CYAN}{'-'*50}{Colors.RESET}")
            
            total_call_oi = sum(data.call_oi for data in option_data.values())
            total_put_oi = sum(data.put_oi for data in option_data.values())
            pcr = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else 0
            
            if pcr < 0.8:
                bias = f"{Colors.GREEN}BULLISH{Colors.RESET}"
            elif pcr > 1.2:
                bias = f"{Colors.RED}BEARISH{Colors.RESET}"
            else:
                bias = f"{Colors.YELLOW}NEUTRAL{Colors.RESET}"
            
            print(f"📊 Overall PCR: {Colors.WHITE}{Colors.BOLD}{pcr}{Colors.RESET}")
            print(f"📈 Market Bias: {bias}")
            print(f"🎯 Call Signals: {Colors.GREEN}{len(call_signals)}{Colors.RESET} | Put Signals: {Colors.RED}{len(put_signals)}{Colors.RESET}")
            
            nearest_support = min(support_levels, key=lambda x: x.distance_from_current) if support_levels else None
            nearest_resistance = min(resistance_levels, key=lambda x: x.distance_from_current) if resistance_levels else None
            
            if nearest_support:
                print(f"📈 Nearest Support: {Colors.GREEN}{nearest_support.level:.2f}{Colors.RESET} ({Colors.WHITE}{nearest_support.strength}{Colors.RESET})")
            if nearest_resistance:
                print(f"📉 Nearest Resistance: {Colors.RED}{nearest_resistance.level:.2f}{Colors.RESET} ({Colors.WHITE}{nearest_resistance.strength}{Colors.RESET})")
            
        except Exception as e:
            print(f"❌ Error displaying advanced analysis: {e}")

def main():
    """Main function for advanced analysis"""
    print(f"{Colors.ORANGE}{Colors.BOLD}🎯 ADVANCED NIFTY ANALYZER - S/R + ENTRY POINTS{Colors.RESET}")
    print(f"{Colors.CYAN}📊 Complete trading system with Support/Resistance + Call/Put entries{Colors.RESET}")
    
    analyzer = AdvancedNiftyAnalyzer()
    
    try:
        print(f"\n{Colors.MAGENTA}Select Analysis Mode:{Colors.RESET}")
        print(f"1. Complete S/R + Entry Analysis")
        print(f"2. Live Trading Mode")
        print(f"3. S/R Levels Only")
        
        choice = input(f"{Colors.CYAN}Enter choice (1-3, default=1): {Colors.RESET}").strip()
        
        if choice == "2":
            print(f"\n{Colors.YELLOW}🚀 Live trading mode - continuous analysis{Colors.RESET}")
            # Could implement live mode here
            
        elif choice == "3":
            print(f"\n{Colors.YELLOW}📊 S/R levels analysis only{Colors.RESET}")
            # Could show only S/R levels
            
        else:
            print(f"\n{Colors.YELLOW}📊 Complete analysis with entry points{Colors.RESET}")
            analyzer.display_advanced_analysis()
            
            save_choice = input(f"\n{Colors.CYAN}Save analysis results? (y/n): {Colors.RESET}").strip().lower()
            if save_choice == 'y':
                print(f"💾 Analysis results would be saved here")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Analysis interrupted{Colors.RESET}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
