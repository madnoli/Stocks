import yfinance as yf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema, savgol_filter
from sklearn.cluster import KMeans

class SupportResistanceScreener:
    def get_stock_data(self, symbol, period="1y"):
        """Download daily stock data from Yahoo Finance"""
        try:
            stock = yf.download(symbol, period=period, progress=False)
            return stock if len(stock) > 0 else None
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            return None
    
    def find_support_resistance_pivot(self, data):
        """Calculate pivot-based support/resistance levels"""
        pivot_point = (data['High'] + data['Low'] + data['Close']) / 3
        support_l1 = (pivot_point * 2) - data['High']
        resistance_l1 = (pivot_point * 2) - data['Low']
        
        return {
            'pivot_point': float(pivot_point.iloc[-1]),
            'support_l1': float(support_l1.iloc[-1]),
            'resistance_l1': float(resistance_l1.iloc[-1])
        }
    
    def find_support_resistance_fractal(self, data, window=5):
        """Find support/resistance using local extrema"""
        high_prices = data['High'].values
        low_prices = data['Low'].values
        
        resistance_indices = argrelextrema(high_prices, np.greater, order=window)
        support_indices = argrelextrema(low_prices, np.less, order=window)
        
        return {
            'resistance_levels': high_prices[resistance_indices].tolist(),
            'support_levels': low_prices[support_indices].tolist()
        }
    
    def screen_stocks(self, symbols):
        """Screen multiple stocks for strong S/R levels"""
        results = []
        for symbol in symbols:
            data = self.get_stock_data(symbol)
            if data is not None:
                pivot_result = self.find_support_resistance_pivot(data)
                fractal_result = self.find_support_resistance_fractal(data)
                
                results.append({
                    'symbol': symbol,
                    'current_price': float(data['Close'].iloc[-1]),
                    'pivot': pivot_result,
                    'fractal': fractal_result
                })
        return results
