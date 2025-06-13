
import pandas as pd

def safe_stochastic(df, k=14, d=3):
    """Safe stochastic oscillator calculation"""
    try:
        high = df['high']
        low = df['low'] 
        close = df['close']
        
        lowest_low = low.rolling(window=k).min()
        highest_high = high.rolling(window=k).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d).mean()
        
        return pd.DataFrame({
            'STOCHk_14_3_3': k_percent,
            'STOCHd_14_3_3': d_percent
        })
    except Exception:
        return pd.DataFrame({
            'STOCHk_14_3_3': [50.0] * len(df),
            'STOCHd_14_3_3': [50.0] * len(df)
        })

def safe_rsi(df, period=14):
    """Safe RSI calculation"""
    try:
        return df.ta.rsi(length=period)
    except Exception:
        return pd.Series([50.0] * len(df), index=df.index)

def safe_macd(df):
    """Safe MACD calculation"""
    try:
        return df.ta.macd()
    except Exception:
        return pd.DataFrame({
            'MACD_12_26_9': [0.0] * len(df),
            'MACDh_12_26_9': [0.0] * len(df),
            'MACDs_12_26_9': [0.0] * len(df)
        })
