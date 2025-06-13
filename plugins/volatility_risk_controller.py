"""
Volatility-Adjusted Risk Control System
Uses ATR to auto-adjust stop-loss and trade size with max drawdown per trade ≤ 2%
"""

import os
import json
import sqlite3
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import ccxt
import pandas_ta as ta

logger = logging.getLogger(__name__)

class VolatilityRiskController:
    def __init__(self):
        self.db_path = "risk_control.db"
        self.config_file = "config/risk_config.json"
        self.exchange = None
        self.max_drawdown_per_trade = 2.0  # 2% maximum drawdown per trade
        self.base_position_size = 0.1  # 10% base position size
        self.atr_lookback = 14
        self.volatility_threshold_low = 1.0  # Low volatility threshold (%)
        self.volatility_threshold_high = 4.0  # High volatility threshold (%)
        
        self.ensure_directories()
        self.setup_database()
        self.initialize_exchange()
        self.load_risk_config()
    
    def ensure_directories(self):
        """Ensure required directories exist"""
        os.makedirs("config", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    
    def initialize_exchange(self):
        """Initialize OKX exchange for volatility data"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True
            })
            logger.info("Risk controller connected to OKX")
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
    
    def setup_database(self):
        """Setup risk control database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS volatility_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    atr_value REAL NOT NULL,
                    atr_percent REAL NOT NULL,
                    volatility_regime TEXT NOT NULL,
                    price REAL NOT NULL,
                    volume_24h REAL,
                    price_change_24h REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_adjustments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    original_position_size REAL NOT NULL,
                    adjusted_position_size REAL NOT NULL,
                    original_stop_loss REAL NOT NULL,
                    adjusted_stop_loss REAL NOT NULL,
                    atr_multiplier REAL NOT NULL,
                    volatility_factor REAL NOT NULL,
                    risk_reduction_pct REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    trade_id TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    description TEXT,
                    severity TEXT NOT NULL,
                    action_taken TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Risk control database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def load_risk_config(self):
        """Load risk management configuration"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    
                self.max_drawdown_per_trade = config.get('max_drawdown_per_trade', 2.0)
                self.base_position_size = config.get('base_position_size', 0.1)
                self.volatility_threshold_low = config.get('volatility_threshold_low', 1.0)
                self.volatility_threshold_high = config.get('volatility_threshold_high', 4.0)
                
            else:
                self.save_risk_config()
                
        except Exception as e:
            logger.error(f"Failed to load risk config: {e}")
    
    def save_risk_config(self):
        """Save risk management configuration"""
        try:
            config = {
                'max_drawdown_per_trade': self.max_drawdown_per_trade,
                'base_position_size': self.base_position_size,
                'volatility_threshold_low': self.volatility_threshold_low,
                'volatility_threshold_high': self.volatility_threshold_high,
                'atr_lookback': self.atr_lookback,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save risk config: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """Get market data for volatility analysis"""
        try:
            if not self.exchange:
                return None
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get market data for {symbol}: {e}")
            return None
    
    def calculate_volatility_metrics(self, symbol: str) -> Optional[Dict]:
        """Calculate comprehensive volatility metrics"""
        try:
            df = self.get_market_data(symbol)
            if df is None or len(df) < self.atr_lookback:
                return None
            
            # Calculate ATR
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=self.atr_lookback)
            
            # Get latest values
            latest = df.iloc[-1]
            atr_value = latest['atr']
            current_price = latest['close']
            
            # Calculate ATR as percentage of price
            atr_percent = (atr_value / current_price) * 100
            
            # Calculate additional volatility measures
            returns = df['close'].pct_change().dropna()
            rolling_std = returns.rolling(window=20).std() * np.sqrt(24) * 100  # Annualized volatility
            current_volatility = rolling_std.iloc[-1]
            
            # Price changes
            price_change_1h = ((current_price - df['close'].iloc[-2]) / df['close'].iloc[-2]) * 100
            price_change_24h = ((current_price - df['close'].iloc[-24]) / df['close'].iloc[-24]) * 100 if len(df) >= 24 else 0
            
            # Volume analysis
            volume_avg = df['volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = latest['volume'] / volume_avg if volume_avg > 0 else 1
            
            # Determine volatility regime
            if atr_percent <= self.volatility_threshold_low:
                regime = "LOW"
            elif atr_percent >= self.volatility_threshold_high:
                regime = "HIGH"
            else:
                regime = "MEDIUM"
            
            metrics = {
                'symbol': symbol,
                'atr_value': atr_value,
                'atr_percent': atr_percent,
                'volatility_regime': regime,
                'current_price': current_price,
                'rolling_volatility': current_volatility,
                'price_change_1h': price_change_1h,
                'price_change_24h': price_change_24h,
                'volume_ratio': volume_ratio,
                'timestamp': datetime.now()
            }
            
            # Save to database
            self.save_volatility_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate volatility metrics for {symbol}: {e}")
            return None
    
    def save_volatility_metrics(self, metrics: Dict):
        """Save volatility metrics to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO volatility_metrics 
                (symbol, atr_value, atr_percent, volatility_regime, price, 
                 volume_24h, price_change_24h)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                metrics['symbol'],
                metrics['atr_value'],
                metrics['atr_percent'],
                metrics['volatility_regime'],
                metrics['current_price'],
                metrics.get('volume_ratio', 0),
                metrics.get('price_change_24h', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save volatility metrics: {e}")
    
    def calculate_position_size_adjustment(self, symbol: str, base_size: float, confidence: float) -> Dict:
        """Calculate volatility-adjusted position size"""
        try:
            volatility_metrics = self.calculate_volatility_metrics(symbol)
            if not volatility_metrics:
                return {
                    'adjusted_size': base_size,
                    'adjustment_factor': 1.0,
                    'reason': 'No volatility data available'
                }
            
            atr_percent = volatility_metrics['atr_percent']
            regime = volatility_metrics['volatility_regime']
            
            # Base adjustment factor based on volatility regime
            if regime == "LOW":
                volatility_factor = 1.2  # Increase position size for low volatility
            elif regime == "HIGH":
                volatility_factor = 0.6  # Decrease position size for high volatility
            else:
                volatility_factor = 1.0  # Normal position size
            
            # Additional adjustment based on confidence
            confidence_factor = min(1.0, confidence / 75.0)  # Scale based on 75% confidence threshold
            
            # Combined adjustment factor
            total_factor = volatility_factor * confidence_factor
            
            # Calculate adjusted position size
            adjusted_size = base_size * total_factor
            
            # Ensure maximum risk per trade doesn't exceed 2%
            max_risk_size = self.max_drawdown_per_trade / (atr_percent * 2)  # Conservative ATR multiplier
            adjusted_size = min(adjusted_size, max_risk_size)
            
            # Ensure minimum viable position size
            min_size = 0.01  # 1% minimum
            adjusted_size = max(adjusted_size, min_size)
            
            return {
                'adjusted_size': adjusted_size,
                'adjustment_factor': total_factor,
                'volatility_factor': volatility_factor,
                'confidence_factor': confidence_factor,
                'atr_percent': atr_percent,
                'regime': regime,
                'reason': f'Volatility regime: {regime}, ATR: {atr_percent:.2f}%'
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate position size adjustment: {e}")
            return {'adjusted_size': base_size, 'adjustment_factor': 1.0, 'reason': 'Calculation error'}
    
    def calculate_dynamic_stop_loss(self, symbol: str, entry_price: float, side: str, confidence: float) -> Dict:
        """Calculate ATR-based dynamic stop loss"""
        try:
            volatility_metrics = self.calculate_volatility_metrics(symbol)
            if not volatility_metrics:
                # Fallback to 2% stop loss
                default_stop_pct = 2.0
                if side.upper() == 'BUY':
                    stop_price = entry_price * (1 - default_stop_pct / 100)
                else:
                    stop_price = entry_price * (1 + default_stop_pct / 100)
                
                return {
                    'stop_loss': stop_price,
                    'stop_distance_pct': default_stop_pct,
                    'atr_multiplier': 2.0,
                    'reason': 'Default stop loss - no volatility data'
                }
            
            atr_value = volatility_metrics['atr_value']
            atr_percent = volatility_metrics['atr_percent']
            regime = volatility_metrics['volatility_regime']
            
            # ATR multiplier based on volatility regime and confidence
            if regime == "LOW":
                base_multiplier = 1.5  # Tighter stops in low volatility
            elif regime == "HIGH":
                base_multiplier = 3.0  # Wider stops in high volatility
            else:
                base_multiplier = 2.0  # Standard multiplier
            
            # Adjust multiplier based on confidence
            confidence_adjustment = 1.0 + ((confidence - 70) / 100)  # Adjust around 70% confidence
            confidence_adjustment = max(0.8, min(1.5, confidence_adjustment))
            
            final_multiplier = base_multiplier * confidence_adjustment
            
            # Calculate stop loss distance
            stop_distance = atr_value * final_multiplier
            stop_distance_pct = (stop_distance / entry_price) * 100
            
            # Ensure stop loss doesn't exceed maximum drawdown
            max_stop_pct = self.max_drawdown_per_trade
            if stop_distance_pct > max_stop_pct:
                stop_distance_pct = max_stop_pct
                stop_distance = entry_price * (max_stop_pct / 100)
            
            # Calculate actual stop price
            if side.upper() == 'BUY':
                stop_price = entry_price - stop_distance
            else:  # SELL
                stop_price = entry_price + stop_distance
            
            return {
                'stop_loss': stop_price,
                'stop_distance_pct': stop_distance_pct,
                'atr_multiplier': final_multiplier,
                'atr_value': atr_value,
                'regime': regime,
                'reason': f'ATR-based stop ({regime} volatility, {final_multiplier:.1f}x ATR)'
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate dynamic stop loss: {e}")
            return {'stop_loss': entry_price * 0.98, 'stop_distance_pct': 2.0, 'reason': 'Error in calculation'}
    
    def assess_trade_risk(self, symbol: str, signal_data: Dict) -> Dict:
        """Comprehensive trade risk assessment"""
        try:
            volatility_metrics = self.calculate_volatility_metrics(symbol)
            if not volatility_metrics:
                return {
                    'risk_level': 'UNKNOWN',
                    'risk_score': 50,
                    'recommendations': ['Unable to assess risk - no volatility data']
                }
            
            risk_factors = []
            risk_score = 0
            
            # Volatility risk
            atr_percent = volatility_metrics['atr_percent']
            if atr_percent > self.volatility_threshold_high:
                risk_factors.append(f'High volatility: {atr_percent:.1f}%')
                risk_score += 30
            elif atr_percent < self.volatility_threshold_low:
                risk_factors.append(f'Low volatility: {atr_percent:.1f}%')
                risk_score += 10
            else:
                risk_score += 20
            
            # Price momentum risk
            price_change_24h = abs(volatility_metrics.get('price_change_24h', 0))
            if price_change_24h > 10:
                risk_factors.append(f'High 24h price movement: {price_change_24h:.1f}%')
                risk_score += 25
            elif price_change_24h > 5:
                risk_score += 15
            else:
                risk_score += 5
            
            # Volume risk
            volume_ratio = volatility_metrics.get('volume_ratio', 1)
            if volume_ratio > 3:
                risk_factors.append(f'Volume surge: {volume_ratio:.1f}x average')
                risk_score += 20
            elif volume_ratio < 0.5:
                risk_factors.append('Low volume')
                risk_score += 15
            else:
                risk_score += 5
            
            # Confidence risk
            confidence = signal_data.get('confidence', 50)
            if confidence < 60:
                risk_factors.append(f'Low signal confidence: {confidence:.1f}%')
                risk_score += 20
            elif confidence > 85:
                risk_score -= 10  # High confidence reduces risk
            
            # Determine risk level
            if risk_score >= 70:
                risk_level = 'HIGH'
            elif risk_score >= 40:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            # Generate recommendations
            recommendations = []
            if risk_level == 'HIGH':
                recommendations.extend([
                    'Consider reducing position size by 50%',
                    'Use tighter stop losses',
                    'Monitor position closely'
                ])
            elif risk_level == 'MEDIUM':
                recommendations.extend([
                    'Use standard risk management',
                    'Consider ATR-based stops'
                ])
            else:
                recommendations.extend([
                    'Normal position sizing acceptable',
                    'Standard stop loss distances'
                ])
            
            return {
                'risk_level': risk_level,
                'risk_score': risk_score,
                'risk_factors': risk_factors,
                'recommendations': recommendations,
                'volatility_regime': volatility_metrics['volatility_regime'],
                'atr_percent': atr_percent
            }
            
        except Exception as e:
            logger.error(f"Failed to assess trade risk: {e}")
            return {'risk_level': 'UNKNOWN', 'risk_score': 50, 'recommendations': ['Risk assessment failed']}
    
    def apply_risk_controls(self, signal_data: Dict) -> Dict:
        """Apply comprehensive risk controls to trading signal"""
        try:
            symbol = signal_data['symbol']
            entry_price = signal_data['current_price']
            side = signal_data['signal']
            confidence = signal_data['confidence']
            
            # Calculate position size adjustment
            base_size = signal_data.get('position_size', self.base_position_size)
            size_adjustment = self.calculate_position_size_adjustment(symbol, base_size, confidence)
            
            # Calculate dynamic stop loss
            stop_loss_calc = self.calculate_dynamic_stop_loss(symbol, entry_price, side, confidence)
            
            # Assess overall trade risk
            risk_assessment = self.assess_trade_risk(symbol, signal_data)
            
            # Apply additional risk controls based on assessment
            final_position_size = size_adjustment['adjusted_size']
            if risk_assessment['risk_level'] == 'HIGH':
                final_position_size *= 0.5  # Halve position size for high risk trades
            
            # Log risk adjustment
            self.log_risk_adjustment(
                symbol,
                base_size,
                final_position_size,
                signal_data.get('stop_loss', 0),
                stop_loss_calc['stop_loss'],
                stop_loss_calc['atr_multiplier'],
                size_adjustment['adjustment_factor'],
                risk_assessment['risk_score']
            )
            
            # Update signal data with risk controls
            enhanced_signal = signal_data.copy()
            enhanced_signal.update({
                'adjusted_position_size': final_position_size,
                'dynamic_stop_loss': stop_loss_calc['stop_loss'],
                'risk_level': risk_assessment['risk_level'],
                'risk_score': risk_assessment['risk_score'],
                'volatility_regime': risk_assessment.get('volatility_regime', 'UNKNOWN'),
                'atr_multiplier': stop_loss_calc['atr_multiplier'],
                'size_adjustment_factor': size_adjustment['adjustment_factor'],
                'risk_recommendations': risk_assessment['recommendations']
            })
            
            return enhanced_signal
            
        except Exception as e:
            logger.error(f"Failed to apply risk controls: {e}")
            return signal_data
    
    def log_risk_adjustment(self, symbol: str, original_size: float, adjusted_size: float,
                           original_stop: float, adjusted_stop: float, atr_multiplier: float,
                           volatility_factor: float, risk_score: float):
        """Log risk adjustment details"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            risk_reduction = ((original_size - adjusted_size) / original_size) * 100 if original_size > 0 else 0
            
            cursor.execute('''
                INSERT INTO risk_adjustments 
                (symbol, original_position_size, adjusted_position_size, 
                 original_stop_loss, adjusted_stop_loss, atr_multiplier,
                 volatility_factor, risk_reduction_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, original_size, adjusted_size, original_stop, adjusted_stop,
                atr_multiplier, volatility_factor, risk_reduction
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log risk adjustment: {e}")
    
    def log_risk_event(self, event_type: str, symbol: str, description: str, 
                      severity: str = 'MEDIUM', action_taken: str = None):
        """Log risk management events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO risk_events 
                (event_type, symbol, description, severity, action_taken)
                VALUES (?, ?, ?, ?, ?)
            ''', (event_type, symbol, description, severity, action_taken))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Risk event logged: {event_type} for {symbol} - {description}")
            
        except Exception as e:
            logger.error(f"Failed to log risk event: {e}")
    
    def get_risk_metrics(self, symbol: str = None, days: int = 7) -> Dict:
        """Get risk management metrics and statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Query risk adjustments
            if symbol:
                query = '''
                    SELECT * FROM risk_adjustments 
                    WHERE symbol = ? AND timestamp > datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                '''.format(days)
                df_adjustments = pd.read_sql_query(query, conn, params=[symbol])
            else:
                query = '''
                    SELECT * FROM risk_adjustments 
                    WHERE timestamp > datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                '''.format(days)
                df_adjustments = pd.read_sql_query(query, conn)
            
            # Query volatility metrics
            if symbol:
                query = '''
                    SELECT * FROM volatility_metrics 
                    WHERE symbol = ? AND timestamp > datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                '''.format(days)
                df_volatility = pd.read_sql_query(query, conn, params=[symbol])
            else:
                query = '''
                    SELECT * FROM volatility_metrics 
                    WHERE timestamp > datetime('now', '-{} days')
                    ORDER BY timestamp DESC
                '''.format(days)
                df_volatility = pd.read_sql_query(query, conn)
            
            conn.close()
            
            metrics = {
                'total_adjustments': len(df_adjustments),
                'avg_risk_reduction': df_adjustments['risk_reduction_pct'].mean() if not df_adjustments.empty else 0,
                'avg_atr_multiplier': df_adjustments['atr_multiplier'].mean() if not df_adjustments.empty else 0,
                'volatility_distribution': df_volatility['volatility_regime'].value_counts().to_dict() if not df_volatility.empty else {},
                'avg_atr_percent': df_volatility['atr_percent'].mean() if not df_volatility.empty else 0
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get risk metrics: {e}")
            return {}

# Global instance
volatility_risk_controller = VolatilityRiskController()