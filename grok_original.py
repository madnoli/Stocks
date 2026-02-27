# Program to Calculate Multi-Timeframe Bullish/Bearish Scores for NSE Stocks using TrueData API
# - Install dependencies: pip install truedata-ws pandas pandas_ta numpy schedule
# - Replace 'your_username' and 'your_password' with actual TrueData credentials
# - Assumes shares.txt contains one NSE stock symbol per line (e.g., RELIANCE, SBIN)
# - Fetches recent 1min intraday data (last 2 hours for sufficient bars)
# - Resamples to 5min, 15min, 30min time frames (3 frames as per query)
# - Computes 15+ indicators per frame and scores bullish/bearish pressure
# - Incorporates current options OI and volume for additional pressure
# - Runs the analysis every 5 minutes using schedule library

import pandas as pd
import pandas_ta as ta
import numpy as np
from truedata_ws.websocket.TD import TD
from datetime import datetime, timedelta
import schedule
import time

# Initialize TrueData API
td = TD('tdwsp751', 'raj@751', live_port=None)  # Set live_port if real-time needed

# Read stock symbols from shares.txt
with open('shares.txt', 'r') as f:
    stocks = [line.strip() for line in f if line.strip()]

# Time frames to analyze (3 as per query: 5min, 15min, 30min)
time_frames = ['5T', '15T', '30T']

# Function to fetch and process data for a symbol
def analyze_stock(symbol):
    try:
        # Fetch recent 1min historical data (last 2 hours for intraday analysis)
        start_time = datetime.now() - timedelta(hours=2)
        hist_data = td.get_historic_data(symbol, start_time=start_time, bar_size='1min')
        if not hist_data or len(hist_data) < 30:  # Need enough data for 30min frame
            print(f"Insufficient data for {symbol}")
            return None, None
        
        df = pd.DataFrame(hist_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Dictionary to hold scores per time frame
        bull_scores = {}
        bear_scores = {}
        
        for tf in time_frames:
            # Resample to the time frame
            df_tf = df.resample(tf).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'oi': 'last'  # Open interest if available, else sum or last
            }).dropna()
            
            if len(df_tf) < 14:  # Need at least 14 bars for RSI, etc.
                continue
            
            # Compute 15+ indicators using pandas_ta
            df_tf.ta.rsi(append=True, length=14)
            df_tf.ta.macd(append=True)
            df_tf.ta.sma(length=10, append=True)
            df_tf.ta.sma(length=20, append=True)
            df_tf.ta.sma(length=50, append=True)
            df_tf.ta.sma(length=200, append=True)  # May be limited by data length
            df_tf.ta.ema(length=12, append=True)
            df_tf.ta.ema(length=26, append=True)
            df_tf.ta.ema(length=50, append=True)
            df_tf.ta.bbands(append=True)
            df_tf.ta.stoch(append=True)
            df_tf.ta.obv(append=True)
            df_tf.ta.cmf(append=True)
            df_tf.ta.ad(append=True)
            df_tf.ta.atr(append=True)
            df_tf.ta.psar(append=True)
            df_tf.ta.ichimoku(append=True)
            
            # Latest values
            latest = df_tf.iloc[-1]
            
            # Bullish score for this frame
            bull_score = 0
            if latest['RSI_14'] > 50: bull_score += 1
            if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']: bull_score += 1
            if latest['close'] > latest['SMA_10']: bull_score += 1
            if latest['close'] > latest['SMA_20']: bull_score += 1
            if 'SMA_50' in latest and 'SMA_200' in latest and latest['SMA_50'] > latest['SMA_200']: bull_score += 1
            if latest['close'] > latest['EMA_12']: bull_score += 1
            if latest['close'] > latest['EMA_26']: bull_score += 1
            if latest['close'] > latest['EMA_50']: bull_score += 1
            if latest['close'] > latest['BBL_5_2.0']: bull_score += 1
            if latest['STOCHk_14_3_3'] > latest['STOCHd_14_3_3'] and latest['STOCHk_14_3_3'] > 20: bull_score += 1
            if df_tf['OBV'].iloc[-1] > df_tf['OBV'].iloc[-2]: bull_score += 1
            if latest['CMF_20'] > 0: bull_score += 1
            if df_tf['AD'].iloc[-1] > df_tf['AD'].iloc[-2]: bull_score += 1
            if latest['ATRr_14'] < df_tf['ATRr_14'].mean(): bull_score += 1
            if 'PSARl_0.02_0.2' in latest and latest['close'] > latest['PSARl_0.02_0.2']: bull_score += 1
            if 'ISA_9' in latest and latest['close'] > latest['ISA_9']: bull_score += 1
            
            # Bearish score
            bear_score = 0
            if latest['RSI_14'] < 50: bear_score += 1
            if latest['MACD_12_26_9'] < latest['MACDs_12_26_9']: bear_score += 1
            if latest['close'] < latest['SMA_10']: bear_score += 1
            if latest['close'] < latest['SMA_20']: bear_score += 1
            if 'SMA_50' in latest and 'SMA_200' in latest and latest['SMA_50'] < latest['SMA_200']: bear_score += 1
            if latest['close'] < latest['EMA_12']: bear_score += 1
            if latest['close'] < latest['EMA_26']: bear_score += 1
            if latest['close'] < latest['EMA_50']: bear_score += 1
            if latest['close'] < latest['BBU_5_2.0']: bear_score += 1
            if latest['STOCHk_14_3_3'] < latest['STOCHd_14_3_3'] and latest['STOCHk_14_3_3'] < 80: bear_score += 1
            if df_tf['OBV'].iloc[-1] < df_tf['OBV'].iloc[-2]: bear_score += 1
            if latest['CMF_20'] < 0: bear_score += 1
            if df_tf['AD'].iloc[-1] < df_tf['AD'].iloc[-2]: bear_score += 1
            if latest['ATRr_14'] > df_tf['ATRr_14'].mean(): bear_score += 1
            if 'PSARs_0.02_0.2' in latest and latest['close'] < latest['PSARs_0.02_0.2']: bear_score += 1
            if 'ISB_26' in latest and latest['close'] < latest['ISB_26']: bear_score += 1
            
            bull_scores[tf] = bull_score
            bear_scores[tf] = bear_score
        
        # Aggregate scores across frames (average)
        total_bull = sum(bull_scores.values()) / len(bull_scores) if bull_scores else 0
        total_bear = sum(bear_scores.values()) / len(bear_scores) if bear_scores else 0
        
        # Add options pressure (current, not time-frame dependent)
        try:
            td.start_option_chain(symbol)
            oc = td.get_option_chain(symbol)
            call_oi = oc.call_chain['oi'].sum()
            put_oi = oc.put_chain['oi'].sum()
            call_vol = oc.call_chain['volume'].sum()
            put_vol = oc.put_chain['volume'].sum()
            
            if call_oi > put_oi: total_bull += 2
            if call_vol > put_vol: total_bull += 1
            if put_oi > call_oi: total_bear += 2
            if put_vol > call_vol: total_bear += 1
        except:
            print(f"Options data not available for {symbol}")
        
        return total_bull, total_bear
    except Exception as e:
        print(f"Error for {symbol}: {e}")
        return None, None

# Main analysis function to run every 5 minutes
def run_analysis():
    print(f"Running analysis at {datetime.now()}")
    results = []
    for symbol in stocks:
        bull, bear = analyze_stock(symbol)
        if bull is not None:
            results.append({'symbol': symbol, 'bullish_score': bull, 'bearish_score': bear})
    
    df_results = pd.DataFrame(results)
    top_bullish = df_results.sort_values('bullish_score', ascending=False).head(5)
    top_bearish = df_results.sort_values('bearish_score', ascending=False).head(5)
    
    print("Top Bullish Stocks:")
    print(top_bullish)
    print("\nTop Bearish Stocks:")
    print(top_bearish)

# Schedule the analysis to run every 5 minutes
schedule.every(5).minutes.do(run_analysis)

# Run the scheduler
while True:
    schedule.run_pending()
    time.sleep(1)