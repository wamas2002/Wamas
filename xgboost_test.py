#!/usr/bin/env python3
"""Quick XGBoost verification with live OKX data"""

import xgboost as xgb
from trading.okx_data_service import OKXDataService
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_xgboost():
    print("Testing XGBoost with live OKX market data...")
    
    # Get live market data
    data_service = OKXDataService()
    df = data_service.get_historical_data('BTC-USDT', '1H', 100)
    
    if df is None or len(df) < 50:
        print("FAILED: Insufficient market data")
        return False
    
    print(f"Retrieved {len(df)} authentic data points")
    print(f"Latest BTC price: ${df['close'].iloc[-1]:.2f}")
    
    # Generate features
    df['returns'] = df['close'].pct_change()
    df['sma_10'] = df['close'].rolling(10).mean()
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(10).mean()
    df['price_change'] = df['close'] - df['open']
    df['target'] = df['returns'].shift(-1)
    
    # Clean data
    features = ['returns', 'sma_10', 'volume_ratio', 'price_change']
    df_clean = df[features + ['target']].dropna()
    
    if len(df_clean) < 20:
        print("FAILED: Insufficient clean data")
        return False
    
    X = df_clean[features].values
    y = df_clean['target'].values
    
    print(f"Training data shape: X{X.shape}, y{y.shape}")
    
    # Train XGBoost model
    try:
        model = xgb.XGBRegressor(
            n_estimators=50, 
            max_depth=3, 
            learning_rate=0.1,
            random_state=42, 
            verbosity=0
        )
        model.fit(X, y)
        
        # Test predictions
        predictions = model.predict(X[-5:])
        feature_importance = model.feature_importances_
        
        print(f"Sample predictions: {predictions[:3]}")
        print(f"Feature importance: {feature_importance}")
        print(f"XGBoost version: {xgb.__version__}")
        print("SUCCESS: XGBoost working with live OKX data")
        return True
        
    except Exception as e:
        print(f"FAILED: XGBoost error - {e}")
        return False

if __name__ == "__main__":
    test_xgboost()