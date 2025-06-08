import pandas as pd
import numpy as np
from typing import Dict, Any

class DataCleaner:
    """Utility class for cleaning and preparing market data"""
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Clean DataFrame for technical analysis"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        try:
            # Create a copy to avoid modifying original
            clean_df = df.copy()
            
            # Remove duplicate indices
            if clean_df.index.duplicated().any():
                clean_df = clean_df[~clean_df.index.duplicated(keep='last')]
            
            # Sort by index
            clean_df = clean_df.sort_index()
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in clean_df.columns:
                    if 'close' in clean_df.columns:
                        clean_df[col] = clean_df['close']
                    else:
                        clean_df[col] = 0.0
            
            # Remove infinite values
            clean_df = clean_df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill then backward fill missing values
            clean_df = clean_df.ffill().bfill()
            
            # Fill any remaining NaN with zeros
            clean_df = clean_df.fillna(0)
            
            return clean_df
            
        except Exception as e:
            print(f"Error cleaning dataframe: {e}")
            return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    
    @staticmethod
    def safe_concat(dfs_dict: Dict[str, pd.DataFrame], axis: int = 1) -> pd.DataFrame:
        """Safely concatenate DataFrames with duplicate index handling"""
        try:
            if not dfs_dict:
                return pd.DataFrame()
            
            # Clean all DataFrames first
            cleaned_dfs = {}
            base_index = None
            
            for name, df in dfs_dict.items():
                if df is not None and not df.empty:
                    cleaned_df = DataCleaner.clean_dataframe(df)
                    if not cleaned_df.empty:
                        cleaned_dfs[name] = cleaned_df
                        if base_index is None:
                            base_index = cleaned_df.index
            
            if not cleaned_dfs:
                return pd.DataFrame()
            
            # Concatenate with proper error handling
            if axis == 1:  # Column-wise concatenation
                result = pd.concat(list(cleaned_dfs.values()), axis=1, sort=False)
            else:  # Row-wise concatenation
                result = pd.concat(list(cleaned_dfs.values()), axis=0, sort=False)
            
            # Final cleanup
            result = DataCleaner.clean_dataframe(result)
            
            return result
            
        except Exception as e:
            print(f"Error in safe_concat: {e}")
            # Return the first valid DataFrame as fallback
            for df in dfs_dict.values():
                if df is not None and not df.empty:
                    return DataCleaner.clean_dataframe(df)
            return pd.DataFrame()
    
    @staticmethod
    def validate_ohlcv_data(df: pd.DataFrame) -> bool:
        """Validate OHLCV data integrity"""
        try:
            if df is None or df.empty:
                return False
            
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                return False
            
            # Check for logical price relationships
            valid_high = (df['high'] >= df['open']) & (df['high'] >= df['close']) & (df['high'] >= df['low'])
            valid_low = (df['low'] <= df['open']) & (df['low'] <= df['close']) & (df['low'] <= df['high'])
            
            # Allow some tolerance for data inconsistencies
            valid_ratio = (valid_high & valid_low).sum() / len(df)
            
            return valid_ratio > 0.8  # 80% of data should be valid
            
        except Exception as e:
            print(f"Error validating OHLCV data: {e}")
            return False