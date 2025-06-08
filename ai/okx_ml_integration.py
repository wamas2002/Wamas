"""
OKX ML Integration Module
Integrates ML predictions with existing OKX trading system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
import logging
from typing import Dict, List, Optional, Any
from ai.comprehensive_ml_pipeline import TradingMLPipeline, get_ml_signal

class OKXMLIntegration:
    """Integrates ML predictions with OKX live trading data"""
    
    def __init__(self, okx_connector=None):
        self.okx_connector = okx_connector
        self.ml_pipeline = TradingMLPipeline()
        self.db_path = "data/trading_data.db"
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Setup OKX data storage
        self._setup_okx_storage()
    
    def _setup_okx_storage(self):
        """Setup database for OKX live data storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # OKX live data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS okx_live_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    datetime TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    source TEXT DEFAULT 'okx',
                    UNIQUE(symbol, timestamp)
                )
            ''')
            
            # ML trading signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    datetime TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL,
                    prediction INTEGER,
                    model_used TEXT,
                    price_at_signal REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database setup error: {e}")
    
    def store_okx_candle(self, symbol: str, candle_data: Dict[str, Any]):
        """Store OKX candle data for ML training"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Extract candle data
            timestamp = int(candle_data.get('timestamp', datetime.now().timestamp() * 1000))
            datetime_str = datetime.fromtimestamp(timestamp / 1000).isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO okx_live_data 
                (symbol, timestamp, datetime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                timestamp,
                datetime_str,
                float(candle_data.get('open', 0)),
                float(candle_data.get('high', 0)),
                float(candle_data.get('low', 0)),
                float(candle_data.get('close', 0)),
                float(candle_data.get('volume', 0))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing OKX candle: {e}")
    
    def get_recent_okx_data(self, symbol: str, minutes: int = 100) -> pd.DataFrame:
        """Get recent OKX data for ML prediction"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_time = int((datetime.now() - timedelta(minutes=minutes)).timestamp() * 1000)
            
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM okx_live_data 
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(query, conn, params=[symbol, cutoff_time])
            conn.close()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting OKX data: {e}")
            return pd.DataFrame()
    
    def generate_ml_signal(self, symbol: str, current_price: float = None) -> Dict[str, Any]:
        """Generate ML trading signal using live OKX data"""
        try:
            # Get recent data
            recent_data = self.get_recent_okx_data(symbol, minutes=100)
            
            if recent_data.empty or len(recent_data) < 50:
                return {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'status': 'insufficient_data',
                    'prediction': 0
                }
            
            # Generate ML signal
            ml_result = get_ml_signal(symbol, recent_data, self.ml_pipeline)
            
            # Store signal in database
            self._store_ml_signal(symbol, ml_result, current_price)
            
            return ml_result
            
        except Exception as e:
            self.logger.error(f"Error generating ML signal: {e}")
            return {
                'signal': 'HOLD',
                'confidence': 0.0,
                'status': 'error',
                'prediction': 0
            }
    
    def _store_ml_signal(self, symbol: str, signal_data: Dict[str, Any], price: float = None):
        """Store ML signal in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = int(datetime.now().timestamp() * 1000)
            datetime_str = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT INTO ml_trading_signals 
                (symbol, timestamp, datetime, signal, confidence, prediction, model_used, price_at_signal)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                timestamp,
                datetime_str,
                signal_data.get('signal', 'HOLD'),
                signal_data.get('confidence', 0.0),
                signal_data.get('ml_prediction', 0),
                'ensemble',
                price or 0.0
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing ML signal: {e}")
    
    def get_signal_history(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """Get ML signal history for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)
            
            query = """
                SELECT timestamp, datetime, signal, confidence, prediction, price_at_signal
                FROM ml_trading_signals 
                WHERE symbol = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn, params=[symbol, cutoff_time])
            conn.close()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting signal history: {e}")
            return pd.DataFrame()
    
    def update_models_with_okx_data(self, symbols: List[str]):
        """Update ML models with latest OKX data"""
        self.logger.info("Updating ML models with OKX data...")
        
        for symbol in symbols:
            try:
                # Check if model needs retraining
                needs_retrain = self.ml_pipeline.retrain_if_needed(symbol, hours_threshold=6)
                
                if needs_retrain:
                    self.logger.info(f"Retraining model for {symbol}")
                    
                    # Train with latest data
                    results = self.ml_pipeline.train_models(symbol)
                    
                    if results:
                        best_f1 = max([r.get('f1_score', 0) for r in results.values() if isinstance(r, dict)])
                        self.logger.info(f"Model updated for {symbol} - Best F1: {best_f1:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error updating model for {symbol}: {e}")
    
    def get_ml_dashboard_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive ML data for dashboard display"""
        try:
            # Get recent signals
            signals = self.get_signal_history(symbol, hours=24)
            
            # Get model performance
            recent_data = self.get_recent_okx_data(symbol, minutes=1440)  # 24 hours
            
            # Calculate signal accuracy if we have enough data
            accuracy = 0.0
            if len(signals) > 10:
                # Simple accuracy calculation (you can enhance this)
                buy_signals = signals[signals['signal'] == 'BUY']
                if len(buy_signals) > 0:
                    accuracy = len(buy_signals) / len(signals)
            
            return {
                'total_signals_24h': len(signals),
                'buy_signals_24h': len(signals[signals['signal'] == 'BUY']) if not signals.empty else 0,
                'avg_confidence': signals['confidence'].mean() if not signals.empty else 0.0,
                'signal_accuracy': accuracy,
                'last_signal': {
                    'signal': signals.iloc[0]['signal'] if not signals.empty else 'NONE',
                    'confidence': signals.iloc[0]['confidence'] if not signals.empty else 0.0,
                    'timestamp': signals.iloc[0]['datetime'] if not signals.empty else 'N/A'
                },
                'data_points_available': len(recent_data),
                'model_status': 'active' if symbol in self.ml_pipeline.models else 'training'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dashboard data: {e}")
            return {
                'total_signals_24h': 0,
                'buy_signals_24h': 0,
                'avg_confidence': 0.0,
                'signal_accuracy': 0.0,
                'last_signal': {'signal': 'ERROR', 'confidence': 0.0, 'timestamp': 'N/A'},
                'data_points_available': 0,
                'model_status': 'error'
            }

# Integration function for existing trading strategies
def enhance_trading_signal_with_ml(base_signal: str, symbol: str, current_price: float, 
                                  ml_integration: OKXMLIntegration) -> Dict[str, Any]:
    """
    Enhance existing trading signals with ML predictions
    
    Args:
        base_signal: Original signal from existing strategy ('BUY', 'SELL', 'HOLD')
        symbol: Trading pair
        current_price: Current market price
        ml_integration: ML integration instance
    
    Returns:
        Enhanced signal with ML confidence
    """
    
    # Get ML signal
    ml_result = ml_integration.generate_ml_signal(symbol, current_price)
    
    # Combine signals
    ml_signal = ml_result['signal']
    ml_confidence = ml_result['confidence']
    
    # Signal combination logic
    if base_signal == 'BUY' and ml_signal == 'BUY':
        final_signal = 'STRONG_BUY'
        confidence = min(0.95, ml_confidence + 0.2)
    elif base_signal == 'BUY' and ml_signal == 'HOLD':
        final_signal = 'BUY'
        confidence = ml_confidence * 0.8
    elif base_signal == 'SELL' and ml_signal == 'BUY':
        final_signal = 'HOLD'  # Conflicting signals
        confidence = 0.5
    elif base_signal == 'HOLD' and ml_signal == 'BUY' and ml_confidence > 0.7:
        final_signal = 'BUY'
        confidence = ml_confidence
    else:
        final_signal = base_signal
        confidence = ml_confidence
    
    return {
        'signal': final_signal,
        'confidence': confidence,
        'base_signal': base_signal,
        'ml_signal': ml_signal,
        'ml_confidence': ml_confidence,
        'ml_status': ml_result['status']
    }

if __name__ == "__main__":
    # Test the integration
    integration = OKXMLIntegration()
    
    # Test with sample data
    sample_candle = {
        'timestamp': int(datetime.now().timestamp() * 1000),
        'open': 105000,
        'high': 105500,
        'low': 104800,
        'close': 105200,
        'volume': 150.5
    }
    
    integration.store_okx_candle('BTC-USDT', sample_candle)
    print("ML integration test completed successfully")