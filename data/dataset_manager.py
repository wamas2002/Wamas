"""
Dataset Manager for ML Training
Handles dataset export, CSV storage, and data merging by timestamp
"""

import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import os

class DatasetManager:
    """Manage datasets for ML training with CSV export and import capabilities"""
    
    def __init__(self, db_path: str = "data/trading_data.db", csv_path: str = "data/datasets/"):
        self.db_path = db_path
        self.csv_path = csv_path
        
        # Ensure CSV directory exists
        os.makedirs(csv_path, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def merge_data_sources_by_timestamp(self, symbol: str, timeframe: str = '1m') -> pd.DataFrame:
        """
        Merge OHLCV data with sentiment data by timestamp at 1-minute resolution
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Data timeframe ('1m', '5m', etc.)
        
        Returns:
            Merged DataFrame with OHLCV + sentiment data
        """
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get OHLCV data
            ohlcv_query = """
                SELECT 
                    timestamp,
                    datetime,
                    open,
                    high,
                    low,
                    close,
                    volume
                FROM ohlcv_data 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp
            """
            
            ohlcv_df = pd.read_sql_query(ohlcv_query, conn, params=[symbol, timeframe])
            
            if ohlcv_df.empty:
                self.logger.warning(f"No OHLCV data found for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert timestamp to minute-level for merging
            ohlcv_df['minute_timestamp'] = (ohlcv_df['timestamp'] // 60000) * 60000
            ohlcv_df['datetime'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
            
            # Get sentiment data for this symbol
            symbol_clean = symbol.replace('/', '').replace('USDT', '').replace('USD', '')
            
            sentiment_query = """
                SELECT 
                    (timestamp / 60000) * 60000 as minute_timestamp,
                    AVG(sentiment_score) as avg_sentiment,
                    COUNT(*) as sentiment_count,
                    SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
                    SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count
                FROM news_sentiment 
                WHERE symbols LIKE ? OR symbols = ''
                GROUP BY minute_timestamp
                ORDER BY minute_timestamp
            """
            
            sentiment_df = pd.read_sql_query(sentiment_query, conn, params=[f'%{symbol_clean}%'])
            
            # Merge OHLCV with sentiment data
            if not sentiment_df.empty:
                merged_df = pd.merge(
                    ohlcv_df, 
                    sentiment_df, 
                    on='minute_timestamp', 
                    how='left'
                )
                
                # Fill missing sentiment values with neutral
                sentiment_cols = ['avg_sentiment', 'sentiment_count', 'positive_count', 'negative_count', 'neutral_count']
                merged_df[sentiment_cols] = merged_df[sentiment_cols].fillna(0)
                
                self.logger.info(f"Merged {len(ohlcv_df)} OHLCV records with {len(sentiment_df)} sentiment records")
            else:
                # Add empty sentiment columns if no sentiment data
                merged_df = ohlcv_df.copy()
                merged_df['avg_sentiment'] = 0.0
                merged_df['sentiment_count'] = 0
                merged_df['positive_count'] = 0
                merged_df['negative_count'] = 0
                merged_df['neutral_count'] = 0
                
                self.logger.warning(f"No sentiment data found for {symbol}, using zero values")
            
            conn.close()
            
            # Clean up columns
            merged_df = merged_df.drop(['minute_timestamp'], axis=1, errors='ignore')
            merged_df['symbol'] = symbol
            merged_df['timeframe'] = timeframe
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error merging data sources for {symbol}: {e}")
            return pd.DataFrame()
    
    def create_labeled_dataset(self, symbol: str, prediction_horizon: int = 1, 
                              classification_type: str = 'binary') -> pd.DataFrame:
        """
        Create complete labeled dataset with features and targets
        
        Args:
            symbol: Trading pair
            prediction_horizon: Minutes ahead to predict
            classification_type: 'binary', 'multiclass', or 'regression'
        
        Returns:
            Complete dataset with features and labels
        """
        
        # Merge data sources
        df = self.merge_data_sources_by_timestamp(symbol)
        
        if df.empty:
            return pd.DataFrame()
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate future returns for labeling
        df['future_close'] = df['close'].shift(-prediction_horizon)
        df['future_return'] = (df['future_close'] / df['close']) - 1
        
        # Create labels based on classification type
        if classification_type == 'binary':
            # Binary: Buy (1) if return > threshold, else No Buy (0)
            threshold = 0.002  # 0.2%
            df['target'] = (df['future_return'] > threshold).astype(int)
            
        elif classification_type == 'multiclass':
            # Multi-class: Strong Buy (3), Buy (2), Hold (1), Sell (0)
            df['target'] = np.where(
                df['future_return'] > 0.01, 3,  # Strong Buy: > 1%
                np.where(
                    df['future_return'] > 0.003, 2,  # Buy: > 0.3%
                    np.where(
                        df['future_return'] > -0.003, 1, 0  # Hold: -0.3% to 0.3%, else Sell
                    )
                )
            )
            
        elif classification_type == 'regression':
            # Regression: predict actual future return
            df['target'] = df['future_return']
        
        # Add price action features
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - np.maximum(df['close'], df['open'])
        df['lower_wick'] = np.minimum(df['close'], df['open']) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Normalized ratios
        df['body_to_range'] = df['body_size'] / (df['total_range'] + 1e-8)
        df['upper_wick_ratio'] = df['upper_wick'] / (df['total_range'] + 1e-8)
        df['lower_wick_ratio'] = df['lower_wick'] / (df['total_range'] + 1e-8)
        
        # Price changes
        for period in [1, 2, 3, 5, 10, 20]:
            df[f'price_change_{period}'] = df['close'].pct_change(period)
            df[f'volume_change_{period}'] = df['volume'].pct_change(period)
        
        # Rolling statistics
        for window in [5, 10, 20, 50]:
            df[f'price_ma_{window}'] = df['close'].rolling(window).mean()
            df[f'price_std_{window}'] = df['close'].rolling(window).std()
            df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            
            # Price position within range
            df[f'price_min_{window}'] = df['close'].rolling(window).min()
            df[f'price_max_{window}'] = df['close'].rolling(window).max()
            df[f'price_position_{window}'] = (df['close'] - df[f'price_min_{window}']) / (
                df[f'price_max_{window}'] - df[f'price_min_{window}'] + 1e-8
            )
        
        # Volume spikes
        df['volume_spike'] = (df['volume'] > df['volume_ma_20'] * 2).astype(int)
        
        # Time features
        dt_series = pd.to_datetime(df['timestamp'], unit='ms')
        df['hour'] = dt_series.dt.hour
        df['day_of_week'] = dt_series.dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Market session indicators
        df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['american_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        
        # Lag features
        for lag in [1, 2, 3, 5, 8, 13]:
            df[f'close_lag_{lag}'] = df['close'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Remove rows with NaN targets
        df = df.dropna(subset=['target'])
        
        # Forward fill missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method='ffill').fillna(0)
        
        self.logger.info(f"Created labeled dataset: {len(df)} samples, {len(df.columns)} features")
        
        return df
    
    def export_to_csv(self, symbol: str, dataset_type: str = 'training', 
                      classification_type: str = 'binary') -> str:
        """
        Export complete dataset to CSV file
        
        Args:
            symbol: Trading pair
            dataset_type: 'training', 'validation', or 'full'
            classification_type: Label type for dataset
        
        Returns:
            Path to exported CSV file
        """
        
        # Create dataset
        df = self.create_labeled_dataset(symbol, classification_type=classification_type)
        
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Generate filename
        symbol_clean = symbol.replace('/', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{symbol_clean}_{dataset_type}_{classification_type}_{timestamp}.csv"
        filepath = os.path.join(self.csv_path, filename)
        
        # Export to CSV
        df.to_csv(filepath, index=False)
        
        self.logger.info(f"Exported {len(df)} records to {filepath}")
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'dataset_type': dataset_type,
            'classification_type': classification_type,
            'samples': len(df),
            'features': len(df.columns) - 1,  # Exclude target
            'export_date': datetime.now().isoformat(),
            'filepath': filepath
        }
        
        metadata_path = filepath.replace('.csv', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return filepath
    
    def load_from_csv(self, filepath: str) -> pd.DataFrame:
        """Load dataset from CSV file"""
        
        try:
            df = pd.read_csv(filepath)
            
            # Convert datetime column if present
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])
            
            self.logger.info(f"Loaded {len(df)} records from {filepath}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV {filepath}: {e}")
            return pd.DataFrame()
    
    def create_train_validation_split(self, symbol: str, validation_ratio: float = 0.2,
                                    classification_type: str = 'binary') -> Tuple[str, str]:
        """
        Create training and validation datasets and export to CSV
        
        Args:
            symbol: Trading pair
            validation_ratio: Fraction for validation set
            classification_type: Label type
        
        Returns:
            Tuple of (train_csv_path, val_csv_path)
        """
        
        # Create full dataset
        df = self.create_labeled_dataset(symbol, classification_type=classification_type)
        
        if df.empty:
            raise ValueError(f"No data available for {symbol}")
        
        # Time-based split (use last X% for validation)
        split_idx = int(len(df) * (1 - validation_ratio))
        
        train_df = df[:split_idx].copy()
        val_df = df[split_idx:].copy()
        
        # Export training set
        symbol_clean = symbol.replace('/', '_')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        train_filename = f"{symbol_clean}_training_{classification_type}_{timestamp}.csv"
        train_path = os.path.join(self.csv_path, train_filename)
        train_df.to_csv(train_path, index=False)
        
        # Export validation set
        val_filename = f"{symbol_clean}_validation_{classification_type}_{timestamp}.csv"
        val_path = os.path.join(self.csv_path, val_filename)
        val_df.to_csv(val_path, index=False)
        
        self.logger.info(f"Created training set: {len(train_df)} samples -> {train_path}")
        self.logger.info(f"Created validation set: {len(val_df)} samples -> {val_path}")
        
        return train_path, val_path
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding metadata and target)"""
        
        exclude_cols = [
            'timestamp', 'datetime', 'symbol', 'timeframe', 'target', 
            'future_close', 'future_return'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
    
    def create_datasets_for_multiple_symbols(self, symbols: List[str], 
                                           classification_type: str = 'binary') -> Dict[str, Tuple[str, str]]:
        """
        Create training datasets for multiple symbols
        
        Args:
            symbols: List of trading pairs
            classification_type: Label type
        
        Returns:
            Dict mapping symbol to (train_path, val_path)
        """
        
        results = {}
        
        for symbol in symbols:
            try:
                train_path, val_path = self.create_train_validation_split(
                    symbol, classification_type=classification_type
                )
                results[symbol] = (train_path, val_path)
                
            except Exception as e:
                self.logger.error(f"Failed to create dataset for {symbol}: {e}")
                continue
        
        return results
    
    def get_dataset_summary(self) -> pd.DataFrame:
        """Get summary of all available datasets"""
        
        try:
            # Get available CSV files
            csv_files = [f for f in os.listdir(self.csv_path) if f.endswith('.csv')]
            
            summaries = []
            for csv_file in csv_files:
                try:
                    filepath = os.path.join(self.csv_path, csv_file)
                    df = pd.read_csv(filepath, nrows=1)  # Just read header
                    
                    # Parse filename for metadata
                    parts = csv_file.replace('.csv', '').split('_')
                    symbol = parts[0] + '/' + parts[1] if len(parts) > 1 else 'Unknown'
                    
                    # Get file size
                    file_size = os.path.getsize(filepath)
                    
                    summaries.append({
                        'filename': csv_file,
                        'symbol': symbol,
                        'features': len(df.columns),
                        'file_size_mb': round(file_size / (1024 * 1024), 2),
                        'created': datetime.fromtimestamp(os.path.getctime(filepath)).isoformat()
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Could not read {csv_file}: {e}")
                    continue
            
            return pd.DataFrame(summaries)
            
        except Exception as e:
            self.logger.error(f"Error getting dataset summary: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    # Test dataset creation
    manager = DatasetManager()
    
    # Create datasets for major symbols
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
    
    try:
        results = manager.create_datasets_for_multiple_symbols(symbols, 'binary')
        
        print("Created datasets:")
        for symbol, (train_path, val_path) in results.items():
            print(f"{symbol}:")
            print(f"  Training: {train_path}")
            print(f"  Validation: {val_path}")
        
        # Show summary
        summary = manager.get_dataset_summary()
        print("\nDataset Summary:")
        print(summary.to_string(index=False))
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to collect data first using the data collectors")