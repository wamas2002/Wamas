"""
Comprehensive Feature Engineering for AI Trading Models
Generates 100+ technical indicators, price action metrics, and sentiment features
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sqlite3
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta

class AdvancedFeatureEngineer:
    """Generate comprehensive features for ML model training"""
    
    def __init__(self, db_path: str = "data/trading_data.db"):
        self.db_path = db_path
        self.scaler = StandardScaler()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def generate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate 100+ technical indicators using pandas_ta"""
        
        # Ensure we have OHLCV columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Create a copy to avoid modifying original
        data = df.copy()
        
        # === TREND INDICATORS ===
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            data[f'sma_{period}'] = ta.sma(data['close'], length=period)
            data[f'ema_{period}'] = ta.ema(data['close'], length=period)
            data[f'wma_{period}'] = ta.wma(data['close'], length=period)
            data[f'tema_{period}'] = ta.tema(data['close'], length=period)
            data[f'trima_{period}'] = ta.trima(data['close'], length=period)
        
        # MACD variations
        macd_12_26 = ta.macd(data['close'], fast=12, slow=26, signal=9)
        data = pd.concat([data, macd_12_26], axis=1)
        
        macd_5_35 = ta.macd(data['close'], fast=5, slow=35, signal=5)
        macd_5_35.columns = ['macd_5_35', 'macd_h_5_35', 'macd_s_5_35']
        data = pd.concat([data, macd_5_35], axis=1)
        
        # Parabolic SAR
        data['psar'] = ta.psar(data['high'], data['low'])['PSARl_0.02_0.2']
        
        # Average Directional Index
        adx_data = ta.adx(data['high'], data['low'], data['close'])
        data = pd.concat([data, adx_data], axis=1)
        
        # Aroon
        aroon = ta.aroon(data['high'], data['low'])
        data = pd.concat([data, aroon], axis=1)
        
        # === MOMENTUM INDICATORS ===
        # RSI variations
        for period in [7, 14, 21, 30]:
            data[f'rsi_{period}'] = ta.rsi(data['close'], length=period)
        
        # Stochastic
        stoch = ta.stoch(data['high'], data['low'], data['close'])
        data = pd.concat([data, stoch], axis=1)
        
        # Williams %R
        for period in [14, 21]:
            data[f'willr_{period}'] = ta.willr(data['high'], data['low'], data['close'], length=period)
        
        # Rate of Change
        for period in [5, 10, 20]:
            data[f'roc_{period}'] = ta.roc(data['close'], length=period)
        
        # Commodity Channel Index
        for period in [14, 20, 30]:
            data[f'cci_{period}'] = ta.cci(data['high'], data['low'], data['close'], length=period)
        
        # Money Flow Index
        data['mfi'] = ta.mfi(data['high'], data['low'], data['close'], data['volume'])
        
        # === VOLATILITY INDICATORS ===
        # Bollinger Bands
        for period in [20, 50]:
            for std in [1.5, 2.0, 2.5]:
                bb = ta.bbands(data['close'], length=period, std=std)
                bb.columns = [f'bb_lower_{period}_{std}', f'bb_mid_{period}_{std}', f'bb_upper_{period}_{std}', 
                             f'bb_bandwidth_{period}_{std}', f'bb_percent_{period}_{std}']
                data = pd.concat([data, bb], axis=1)
        
        # Average True Range
        for period in [7, 14, 21]:
            data[f'atr_{period}'] = ta.atr(data['high'], data['low'], data['close'], length=period)
        
        # Keltner Channels
        kc = ta.kc(data['high'], data['low'], data['close'])
        data = pd.concat([data, kc], axis=1)
        
        # Donchian Channels
        dc = ta.donchian(data['high'], data['low'])
        data = pd.concat([data, dc], axis=1)
        
        # === VOLUME INDICATORS ===
        # On-Balance Volume
        data['obv'] = ta.obv(data['close'], data['volume'])
        
        # Volume Weighted Average Price
        data['vwap'] = ta.vwap(data['high'], data['low'], data['close'], data['volume'])
        
        # Accumulation/Distribution Line
        data['ad'] = ta.ad(data['high'], data['low'], data['close'], data['volume'])
        
        # Chaikin Money Flow
        data['cmf'] = ta.cmf(data['high'], data['low'], data['close'], data['volume'])
        
        # Volume Price Trend
        data['vpt'] = ta.vpt(data['close'], data['volume'])
        
        # Positive/Negative Volume Index
        data['pvi'] = ta.pvi(data['close'], data['volume'])
        data['nvi'] = ta.nvi(data['close'], data['volume'])
        
        # Volume Rate of Change
        for period in [5, 10, 20]:
            data[f'vroc_{period}'] = ta.roc(data['volume'], length=period)
        
        return data
    
    def generate_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate price action and volatility metrics"""
        
        data = df.copy()
        
        # Basic price relationships
        data['body_size'] = abs(data['close'] - data['open'])
        data['upper_wick'] = data['high'] - np.maximum(data['close'], data['open'])
        data['lower_wick'] = np.minimum(data['close'], data['open']) - data['low']
        data['total_range'] = data['high'] - data['low']
        
        # Normalized ratios
        data['body_to_range'] = data['body_size'] / (data['total_range'] + 1e-8)
        data['upper_wick_to_range'] = data['upper_wick'] / (data['total_range'] + 1e-8)
        data['lower_wick_to_range'] = data['lower_wick'] / (data['total_range'] + 1e-8)
        
        # Price changes
        for period in [1, 2, 3, 5, 10, 20]:
            data[f'price_change_{period}'] = data['close'].pct_change(period)
            data[f'high_change_{period}'] = data['high'].pct_change(period)
            data[f'low_change_{period}'] = data['low'].pct_change(period)
            data[f'volume_change_{period}'] = data['volume'].pct_change(period)
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            data[f'price_std_{window}'] = data['close'].rolling(window).std()
            data[f'price_skew_{window}'] = data['close'].rolling(window).skew()
            data[f'price_kurt_{window}'] = data['close'].rolling(window).kurt()
            data[f'volume_std_{window}'] = data['volume'].rolling(window).std()
            
            # Rolling min/max
            data[f'price_min_{window}'] = data['close'].rolling(window).min()
            data[f'price_max_{window}'] = data['close'].rolling(window).max()
            data[f'price_position_{window}'] = (data['close'] - data[f'price_min_{window}']) / (
                data[f'price_max_{window}'] - data[f'price_min_{window}'] + 1e-8
            )
        
        # Gap analysis
        data['gap_up'] = (data['open'] > data['high'].shift(1)).astype(int)
        data['gap_down'] = (data['open'] < data['low'].shift(1)).astype(int)
        data['gap_size'] = np.where(
            data['gap_up'] == 1,
            (data['open'] - data['high'].shift(1)) / data['high'].shift(1),
            np.where(
                data['gap_down'] == 1,
                (data['low'].shift(1) - data['open']) / data['low'].shift(1),
                0
            )
        )
        
        # Doji patterns
        data['doji'] = (data['body_size'] < (data['total_range'] * 0.1)).astype(int)
        data['hammer'] = ((data['lower_wick'] > data['body_size'] * 2) & 
                         (data['upper_wick'] < data['body_size'])).astype(int)
        data['shooting_star'] = ((data['upper_wick'] > data['body_size'] * 2) & 
                                (data['lower_wick'] < data['body_size'])).astype(int)
        
        # Volume spikes
        data['volume_ma_20'] = data['volume'].rolling(20).mean()
        data['volume_spike'] = (data['volume'] > data['volume_ma_20'] * 2).astype(int)
        
        return data
    
    def generate_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate time-based cyclical features"""
        
        data = df.copy()
        
        if 'datetime' in data.columns:
            dt_col = pd.to_datetime(data['datetime'])
        elif data.index.name == 'datetime' or isinstance(data.index, pd.DatetimeIndex):
            dt_col = data.index
        else:
            # Try to create from timestamp
            if 'timestamp' in data.columns:
                dt_col = pd.to_datetime(data['timestamp'], unit='ms')
            else:
                self.logger.warning("No datetime information found, skipping cyclical features")
                return data
        
        # Extract time components
        data['hour'] = dt_col.hour
        data['day_of_week'] = dt_col.dayofweek
        data['day_of_month'] = dt_col.day
        data['month'] = dt_col.month
        data['quarter'] = dt_col.quarter
        
        # Create cyclical encodings
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # Market session indicators
        data['asian_session'] = ((data['hour'] >= 0) & (data['hour'] < 8)).astype(int)
        data['european_session'] = ((data['hour'] >= 8) & (data['hour'] < 16)).astype(int)
        data['american_session'] = ((data['hour'] >= 16) & (data['hour'] < 24)).astype(int)
        
        # Weekend indicator
        data['weekend'] = (data['day_of_week'] >= 5).astype(int)
        
        return data
    
    def add_sentiment_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add news sentiment features from database"""
        
        data = df.copy()
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Initialize sentiment columns
            data['sentiment_score'] = 0.0
            data['sentiment_momentum'] = 0.0
            data['positive_news_count'] = 0
            data['negative_news_count'] = 0
            
            # Get sentiment data for each timestamp
            for idx, row in data.iterrows():
                if 'timestamp' in row:
                    timestamp = row['timestamp']
                elif hasattr(idx, 'timestamp'):
                    timestamp = int(idx.timestamp() * 1000)
                else:
                    continue
                
                # Query sentiment within 5-minute window
                query = """
                    SELECT 
                        AVG(sentiment_score) as avg_sentiment,
                        AVG(sentiment_momentum) as avg_momentum,
                        SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as pos_count,
                        SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as neg_count
                    FROM news_sentiment
                    WHERE timestamp BETWEEN ? AND ?
                    AND (symbols LIKE ? OR symbols = '')
                """
                
                params = [timestamp - 300000, timestamp + 300000, f'%{symbol}%']
                result = conn.execute(query, params).fetchone()
                
                if result and result[0] is not None:
                    data.loc[idx, 'sentiment_score'] = result[0]
                    data.loc[idx, 'sentiment_momentum'] = result[1] or 0.0
                    data.loc[idx, 'positive_news_count'] = result[2] or 0
                    data.loc[idx, 'negative_news_count'] = result[3] or 0
            
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Could not add sentiment features: {e}")
            # Add zero columns if sentiment data unavailable
            data['sentiment_score'] = 0.0
            data['sentiment_momentum'] = 0.0
            data['positive_news_count'] = 0
            data['negative_news_count'] = 0
        
        return data
    
    def generate_lag_features(self, df: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """Generate lagged features for time series prediction"""
        
        data = df.copy()
        
        # Price lags
        for lag in [1, 2, 3, 5, 8, 13, 21]:
            data[f'{target_col}_lag_{lag}'] = data[target_col].shift(lag)
            data[f'volume_lag_{lag}'] = data['volume'].shift(lag)
            
            # Lag ratios
            data[f'{target_col}_ratio_lag_{lag}'] = data[target_col] / (data[f'{target_col}_lag_{lag}'] + 1e-8)
        
        # Rolling lag statistics
        for window in [5, 10, 20]:
            data[f'{target_col}_lag_mean_{window}'] = data[target_col].shift(1).rolling(window).mean()
            data[f'{target_col}_lag_std_{window}'] = data[target_col].shift(1).rolling(window).std()
        
        return data
    
    def generate_all_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate all features in sequence"""
        
        self.logger.info("Generating technical indicators...")
        data = self.generate_technical_indicators(df)
        
        self.logger.info("Generating price action features...")
        data = self.generate_price_action_features(data)
        
        self.logger.info("Generating cyclical features...")
        data = self.generate_cyclical_features(data)
        
        self.logger.info("Adding sentiment features...")
        data = self.add_sentiment_features(data, symbol)
        
        self.logger.info("Generating lag features...")
        data = self.generate_lag_features(data)
        
        # Remove infinite and NaN values
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Get feature count
        feature_count = len([col for col in data.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime', 'symbol', 'timeframe']])
        self.logger.info(f"Generated {feature_count} features")
        
        return data
    
    def create_labels(self, df: pd.DataFrame, prediction_horizon: int = 1, 
                     classification_type: str = 'binary') -> pd.DataFrame:
        """Create prediction labels for ML training"""
        
        data = df.copy()
        
        # Calculate future returns
        data['future_return'] = data['close'].shift(-prediction_horizon) / data['close'] - 1
        
        if classification_type == 'binary':
            # Binary: Buy (1) / No Buy (0)
            data['target'] = (data['future_return'] > 0.002).astype(int)  # 0.2% threshold
            
        elif classification_type == 'multiclass':
            # Multi-class: Buy (2) / Hold (1) / Sell (0)
            data['target'] = np.where(
                data['future_return'] > 0.005, 2,  # Buy: > 0.5%
                np.where(data['future_return'] < -0.005, 0, 1)  # Sell: < -0.5%, else Hold
            )
        
        elif classification_type == 'regression':
            # Regression: predict actual future return
            data['target'] = data['future_return']
        
        # Remove rows with NaN targets
        data = data.dropna(subset=['target'])
        
        return data
    
    def prepare_training_data(self, symbol: str, timeframe: str = '1m', 
                            classification_type: str = 'binary',
                            prediction_horizon: int = 1) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare complete training dataset"""
        
        # Load OHLCV data
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT * FROM ohlcv_data 
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=[symbol, timeframe])
        conn.close()
        
        if df.empty:
            raise ValueError(f"No data found for {symbol} {timeframe}")
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('datetime')
        
        # Generate all features
        df = self.generate_all_features(df, symbol)
        
        # Create labels
        df = self.create_labels(df, prediction_horizon, classification_type)
        
        # Select feature columns (exclude OHLCV and metadata)
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp', 'datetime', 
                       'symbol', 'timeframe', 'target', 'future_return', 'id']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Fill remaining NaN values
        df[feature_cols] = df[feature_cols].fillna(method='forward').fillna(0)
        
        self.logger.info(f"Prepared training data: {len(df)} samples, {len(feature_cols)} features")
        
        return df, feature_cols

if __name__ == "__main__":
    # Test feature engineering
    engineer = AdvancedFeatureEngineer()
    
    # Prepare data for BTC
    try:
        df, features = engineer.prepare_training_data('BTC/USDT', '1m', 'binary')
        print(f"Generated dataset with {len(df)} samples and {len(features)} features")
        print(f"Feature columns: {features[:10]}...")  # Show first 10 features
        print(f"Target distribution: {df['target'].value_counts()}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to run data collection first")