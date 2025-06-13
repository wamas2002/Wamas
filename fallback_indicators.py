
"""
Fallback Indicator Calculator
Safe calculations that never fail
"""
import pandas as pd
import numpy as np

def safe_calculate_indicators(df):
    """Calculate indicators with safe fallbacks"""
    try:
        # Safe RSI calculation
        if 'close' in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
        else:
            df['RSI_14'] = 50.0
        
        # Safe Stochastic - fixed calculation
        if all(col in df.columns for col in ['high', 'low', 'close']):
            lowest_low = df['low'].rolling(window=14).min()
            highest_high = df['high'].rolling(window=14).max()
            k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            df['STOCHk_14'] = k_percent.fillna(50.0)
            df['STOCHd_14'] = df['STOCHk_14'].rolling(window=3).mean().fillna(50.0)
        else:
            df['STOCHk_14'] = 50.0
            df['STOCHd_14'] = 50.0
        
        # Safe EMA calculations
        if 'close' in df.columns:
            df['EMA_9'] = df['close'].ewm(span=9).mean()
            df['EMA_21'] = df['close'].ewm(span=21).mean()
            df['EMA_50'] = df['close'].ewm(span=50).mean()
        
        # Safe MACD
        if 'close' in df.columns:
            ema12 = df['close'].ewm(span=12).mean()
            ema26 = df['close'].ewm(span=26).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
        
        # Safe Volume indicators
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        else:
            df['volume_ratio'] = 1.0
        
        return df
        
    except Exception as e:
        print(f"Indicator calculation error: {e}")
        # Return safe defaults
        safe_columns = {
            'RSI_14': 50.0,
            'STOCHk_14': 50.0,
            'STOCHd_14': 50.0,
            'MACD': 0.0,
            'MACD_signal': 0.0,
            'volume_ratio': 1.0
        }
        
        for col, default_val in safe_columns.items():
            if col not in df.columns:
                df[col] = default_val
        
        return df

def generate_local_confidence_boost(technical_score, rsi, volume_ratio):
    """Generate confidence boost using local analysis"""
    boost = 0.0
    
    # RSI-based boost
    if rsi < 30:  # Oversold
        boost += 8.0
    elif rsi > 70:  # Overbought  
        boost += 5.0
    
    # Volume boost
    if volume_ratio > 1.5:
        boost += 6.0
    elif volume_ratio > 1.2:
        boost += 3.0
    
    # Technical score boost
    if technical_score > 80:
        boost += 10.0
    elif technical_score > 70:
        boost += 7.0
    
    return min(boost, 20.0)  # Cap at 20% boost
