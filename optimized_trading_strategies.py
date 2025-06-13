"""
Optimized Trading Strategies Engine
Advanced algorithmic strategies with dynamic optimization and ML enhancement
"""

import os
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedTradingStrategies:
    def __init__(self):
        self.exchange = None
        self.db_path = 'optimized_strategies.db'
        self.strategies = {}
        self.models = {}
        self.scalers = {}
        self.performance_metrics = {}
        
        # Strategy parameters
        self.strategy_configs = {
            'momentum_breakout': {
                'min_confidence': 75.0,
                'rsi_oversold': 25,
                'rsi_overbought': 75,
                'volume_threshold': 1.5,
                'atr_multiplier': 2.0,
                'trend_strength_min': 1.5
            },
            'mean_reversion': {
                'min_confidence': 70.0,
                'bb_threshold': 0.1,
                'rsi_extreme_low': 20,
                'rsi_extreme_high': 80,
                'volume_confirmation': 1.2,
                'reversal_confirmation': 3
            },
            'trend_following': {
                'min_confidence': 72.0,
                'ema_fast': 12,
                'ema_slow': 26,
                'adx_threshold': 25,
                'atr_stop_multiplier': 1.5,
                'min_trend_bars': 5
            },
            'scalping': {
                'min_confidence': 80.0,
                'quick_profit_target': 0.8,
                'tight_stop_loss': 0.4,
                'volume_surge': 2.0,
                'max_hold_time': 15,  # minutes
                'spread_threshold': 0.1
            },
            'swing_trading': {
                'min_confidence': 68.0,
                'support_resistance_buffer': 0.5,
                'fibonacci_levels': [0.236, 0.382, 0.618],
                'swing_high_low_period': 20,
                'momentum_confirmation': 5
            }
        }
        
        self.initialize_exchange()
        self.setup_database()
    
    def initialize_exchange(self):
        """Initialize OKX exchange"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True
            })
            logger.info("Optimized strategies engine connected to OKX")
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
    
    def setup_database(self):
        """Setup optimized strategies database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Strategy signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    target_price REAL,
                    risk_reward_ratio REAL,
                    position_size REAL,
                    timeframe TEXT,
                    indicators_data TEXT,
                    market_conditions TEXT,
                    expected_duration INTEGER,
                    success_probability REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Strategy performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT,
                    trades_count INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    profit_factor REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    sharpe_ratio REAL DEFAULT 0,
                    total_return REAL DEFAULT 0,
                    avg_win REAL DEFAULT 0,
                    avg_loss REAL DEFAULT 0,
                    avg_trade_duration INTEGER DEFAULT 0,
                    last_optimization TIMESTAMP,
                    performance_score REAL DEFAULT 0,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Strategy optimization table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_optimization (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT NOT NULL,
                    optimization_type TEXT NOT NULL,
                    original_params TEXT,
                    optimized_params TEXT,
                    performance_improvement REAL,
                    optimization_score REAL,
                    backtest_results TEXT,
                    implementation_status TEXT DEFAULT 'PENDING',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Optimized strategies database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_enhanced_market_data(self, symbol: str, timeframes: List[str] = ['1h', '4h', '1d']) -> Dict:
        """Get multi-timeframe market data for comprehensive analysis"""
        try:
            market_data = {}
            
            for timeframe in timeframes:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=500)
                if not ohlcv:
                    continue
                
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Calculate comprehensive indicators
                df = self.calculate_advanced_technical_indicators(df)
                
                market_data[timeframe] = df
            
            return market_data
            
        except Exception as e:
            logger.error(f"Enhanced market data fetch failed for {symbol}: {e}")
            return {}
    
    def calculate_advanced_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators for strategy analysis"""
        try:
            # Trend indicators
            df['sma_10'] = ta.sma(df['close'], length=10)
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['sma_200'] = ta.sma(df['close'], length=200)
            df['ema_12'] = ta.ema(df['close'], length=12)
            df['ema_26'] = ta.ema(df['close'], length=26)
            
            # Momentum indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            df['rsi_fast'] = ta.rsi(df['close'], length=7)
            df['rsi_slow'] = ta.rsi(df['close'], length=21)
            
            # MACD
            macd = ta.macd(df['close'])
            if macd is not None and not macd.empty:
                df['macd'] = macd.iloc[:, 0] if len(macd.columns) > 0 else 0
                df['macd_signal'] = macd.iloc[:, 1] if len(macd.columns) > 1 else 0
                df['macd_histogram'] = macd.iloc[:, 2] if len(macd.columns) > 2 else 0
            
            # Bollinger Bands
            bb = ta.bbands(df['close'], length=20)
            if bb is not None and not bb.empty:
                df['bb_upper'] = bb.iloc[:, 0] if len(bb.columns) > 0 else df['close']
                df['bb_middle'] = bb.iloc[:, 1] if len(bb.columns) > 1 else df['close']
                df['bb_lower'] = bb.iloc[:, 2] if len(bb.columns) > 2 else df['close']
                df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
                df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volatility indicators
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['atr_percent'] = df['atr'] / df['close'] * 100
            
            # Volume indicators
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Support and resistance
            df['resistance'] = df['high'].rolling(window=20).max()
            df['support'] = df['low'].rolling(window=20).min()
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            
            # Trend strength
            df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
            df['trend_strength'] = abs(df['close'] - df['sma_20']) / df['atr']
            
            # Price patterns
            df['higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
            df['lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
            
            # Momentum
            df['momentum_5'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5) * 100
            df['momentum_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100
            df['momentum_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20) * 100
            
            # Stochastic
            stoch = ta.stoch(df['high'], df['low'], df['close'])
            if stoch is not None and not stoch.empty:
                df['stoch_k'] = stoch.iloc[:, 0] if len(stoch.columns) > 0 else 50
                df['stoch_d'] = stoch.iloc[:, 1] if len(stoch.columns) > 1 else 50
            
            return df.fillna(0)
            
        except Exception as e:
            logger.error(f"Advanced indicator calculation failed: {e}")
            return df
    
    def momentum_breakout_strategy(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Advanced momentum breakout strategy"""
        try:
            config = self.strategy_configs['momentum_breakout']
            df_1h = market_data.get('1h')
            df_4h = market_data.get('4h')
            
            if df_1h is None or len(df_1h) < 50:
                return None
            
            latest = df_1h.iloc[-1]
            latest_4h = df_4h.iloc[-1] if df_4h is not None else latest
            
            signals = []
            confidence_factors = []
            
            # Volume surge detection
            if latest['volume_ratio'] > config['volume_threshold']:
                signals.append('VOLUME_SURGE')
                confidence_factors.append(15)
            
            # Breakout detection
            resistance_break = latest['close'] > latest['resistance'] * 1.001
            support_break = latest['close'] < latest['support'] * 0.999
            
            if resistance_break:
                signals.append('RESISTANCE_BREAKOUT')
                confidence_factors.append(20)
            elif support_break:
                signals.append('SUPPORT_BREAKDOWN')
                confidence_factors.append(-20)
            
            # RSI momentum
            if latest['rsi'] > 50 and latest['rsi'] < config['rsi_overbought']:
                signals.append('RSI_BULLISH_MOMENTUM')
                confidence_factors.append(10)
            elif latest['rsi'] < 50 and latest['rsi'] > config['rsi_oversold']:
                signals.append('RSI_BEARISH_MOMENTUM')
                confidence_factors.append(-10)
            
            # MACD confirmation
            if latest['macd'] > latest['macd_signal'] and latest['macd_histogram'] > 0:
                signals.append('MACD_BULLISH')
                confidence_factors.append(12)
            elif latest['macd'] < latest['macd_signal'] and latest['macd_histogram'] < 0:
                signals.append('MACD_BEARISH')
                confidence_factors.append(-12)
            
            # Trend strength
            if latest['trend_strength'] > config['trend_strength_min']:
                signals.append('STRONG_TREND')
                confidence_factors.append(15)
            
            # Multi-timeframe alignment
            if df_4h is not None:
                if latest_4h['ema_12'] > latest_4h['ema_26'] and latest['ema_12'] > latest['ema_26']:
                    signals.append('MTF_BULL_ALIGNMENT')
                    confidence_factors.append(18)
                elif latest_4h['ema_12'] < latest_4h['ema_26'] and latest['ema_12'] < latest['ema_26']:
                    signals.append('MTF_BEAR_ALIGNMENT')
                    confidence_factors.append(-18)
            
            # Calculate confidence
            base_confidence = 50
            total_confidence = base_confidence + sum(confidence_factors)
            total_confidence = max(0, min(100, total_confidence))
            
            # Determine signal
            if total_confidence >= config['min_confidence'] and sum([f for f in confidence_factors if f > 0]) > abs(sum([f for f in confidence_factors if f < 0])):
                signal = 'BUY'
            elif total_confidence >= config['min_confidence'] and abs(sum([f for f in confidence_factors if f < 0])) > sum([f for f in confidence_factors if f > 0]):
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            if signal == 'HOLD':
                return None
            
            # Calculate targets
            atr = latest['atr']
            current_price = latest['close']
            
            if signal == 'BUY':
                stop_loss = current_price - (atr * config['atr_multiplier'])
                take_profit = current_price + (atr * config['atr_multiplier'] * 1.5)
            else:
                stop_loss = current_price + (atr * config['atr_multiplier'])
                take_profit = current_price - (atr * config['atr_multiplier'] * 1.5)
            
            risk_reward = abs(take_profit - current_price) / abs(current_price - stop_loss)
            
            return {
                'strategy': 'momentum_breakout',
                'symbol': symbol,
                'signal': signal,
                'confidence': round(total_confidence, 2),
                'entry_price': current_price,
                'stop_loss': round(stop_loss, 6),
                'take_profit': round(take_profit, 6),
                'risk_reward_ratio': round(risk_reward, 2),
                'signals': signals,
                'timeframe': '1h',
                'expected_duration': 240,  # 4 hours
                'market_conditions': {
                    'volatility': latest['atr_percent'],
                    'volume_ratio': latest['volume_ratio'],
                    'trend_strength': latest['trend_strength']
                }
            }
            
        except Exception as e:
            logger.error(f"Momentum breakout strategy failed for {symbol}: {e}")
            return None
    
    def mean_reversion_strategy(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Advanced mean reversion strategy"""
        try:
            config = self.strategy_configs['mean_reversion']
            df_1h = market_data.get('1h')
            
            if df_1h is None or len(df_1h) < 50:
                return None
            
            latest = df_1h.iloc[-1]
            recent = df_1h.iloc[-config['reversal_confirmation']:]
            
            signals = []
            confidence_factors = []
            
            # Bollinger Bands mean reversion
            if latest['bb_position'] < config['bb_threshold']:
                signals.append('BB_OVERSOLD')
                confidence_factors.append(18)
            elif latest['bb_position'] > (1 - config['bb_threshold']):
                signals.append('BB_OVERBOUGHT')
                confidence_factors.append(-18)
            
            # RSI extremes
            if latest['rsi'] < config['rsi_extreme_low']:
                signals.append('RSI_EXTREME_OVERSOLD')
                confidence_factors.append(20)
            elif latest['rsi'] > config['rsi_extreme_high']:
                signals.append('RSI_EXTREME_OVERBOUGHT')
                confidence_factors.append(-20)
            
            # Volume confirmation
            if latest['volume_ratio'] > config['volume_confirmation']:
                signals.append('VOLUME_CONFIRMATION')
                confidence_factors.append(10)
            
            # Reversal pattern detection
            hammer_pattern = (latest['close'] > latest['open']) and \
                           ((latest['close'] - latest['open']) > 2 * (latest['open'] - latest['low'])) and \
                           (latest['high'] - latest['close'] < 0.1 * (latest['close'] - latest['low']))
            
            shooting_star_pattern = (latest['open'] > latest['close']) and \
                                  ((latest['open'] - latest['close']) > 2 * (latest['high'] - latest['open'])) and \
                                  (latest['close'] - latest['low'] < 0.1 * (latest['high'] - latest['open']))
            
            if hammer_pattern:
                signals.append('HAMMER_REVERSAL')
                confidence_factors.append(15)
            elif shooting_star_pattern:
                signals.append('SHOOTING_STAR_REVERSAL')
                confidence_factors.append(-15)
            
            # Divergence detection (simplified)
            price_trend = recent['close'].iloc[-1] - recent['close'].iloc[0]
            rsi_trend = recent['rsi'].iloc[-1] - recent['rsi'].iloc[0]
            
            if price_trend < 0 and rsi_trend > 0:  # Bullish divergence
                signals.append('BULLISH_DIVERGENCE')
                confidence_factors.append(12)
            elif price_trend > 0 and rsi_trend < 0:  # Bearish divergence
                signals.append('BEARISH_DIVERGENCE')
                confidence_factors.append(-12)
            
            # Calculate confidence
            base_confidence = 50
            total_confidence = base_confidence + sum(confidence_factors)
            total_confidence = max(0, min(100, total_confidence))
            
            # Determine signal
            positive_factors = sum([f for f in confidence_factors if f > 0])
            negative_factors = abs(sum([f for f in confidence_factors if f < 0]))
            
            if total_confidence >= config['min_confidence'] and positive_factors > negative_factors:
                signal = 'BUY'
            elif total_confidence >= config['min_confidence'] and negative_factors > positive_factors:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            if signal == 'HOLD':
                return None
            
            # Calculate targets (tighter for mean reversion)
            atr = latest['atr']
            current_price = latest['close']
            
            if signal == 'BUY':
                stop_loss = current_price - (atr * 1.0)
                take_profit = latest['bb_middle']  # Target mean
            else:
                stop_loss = current_price + (atr * 1.0)
                take_profit = latest['bb_middle']  # Target mean
            
            risk_reward = abs(take_profit - current_price) / abs(current_price - stop_loss)
            
            return {
                'strategy': 'mean_reversion',
                'symbol': symbol,
                'signal': signal,
                'confidence': round(total_confidence, 2),
                'entry_price': current_price,
                'stop_loss': round(stop_loss, 6),
                'take_profit': round(take_profit, 6),
                'risk_reward_ratio': round(risk_reward, 2),
                'signals': signals,
                'timeframe': '1h',
                'expected_duration': 120,  # 2 hours
                'market_conditions': {
                    'bb_position': latest['bb_position'],
                    'rsi': latest['rsi'],
                    'volume_ratio': latest['volume_ratio']
                }
            }
            
        except Exception as e:
            logger.error(f"Mean reversion strategy failed for {symbol}: {e}")
            return None
    
    def trend_following_strategy(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Advanced trend following strategy"""
        try:
            config = self.strategy_configs['trend_following']
            df_1h = market_data.get('1h')
            df_4h = market_data.get('4h')
            
            if df_1h is None or len(df_1h) < 50:
                return None
            
            latest = df_1h.iloc[-1]
            latest_4h = df_4h.iloc[-1] if df_4h is not None else latest
            recent = df_1h.iloc[-config['min_trend_bars']:]
            
            signals = []
            confidence_factors = []
            
            # ADX trend strength
            if latest['adx'] > config['adx_threshold']:
                signals.append('STRONG_TREND_ADX')
                confidence_factors.append(15)
            
            # EMA alignment
            ema_bullish = latest['ema_12'] > latest['ema_26'] and latest['close'] > latest['ema_12']
            ema_bearish = latest['ema_12'] < latest['ema_26'] and latest['close'] < latest['ema_12']
            
            if ema_bullish:
                signals.append('EMA_BULLISH_ALIGNMENT')
                confidence_factors.append(18)
            elif ema_bearish:
                signals.append('EMA_BEARISH_ALIGNMENT')
                confidence_factors.append(-18)
            
            # SMA trend confirmation
            sma_bullish_hierarchy = latest['sma_10'] > latest['sma_20'] > latest['sma_50']
            sma_bearish_hierarchy = latest['sma_10'] < latest['sma_20'] < latest['sma_50']
            
            if sma_bullish_hierarchy:
                signals.append('SMA_BULLISH_HIERARCHY')
                confidence_factors.append(16)
            elif sma_bearish_hierarchy:
                signals.append('SMA_BEARISH_HIERARCHY')
                confidence_factors.append(-16)
            
            # Consistent trend bars
            consistent_up_trend = all(recent['close'] > recent['open'])
            consistent_down_trend = all(recent['close'] < recent['open'])
            
            if consistent_up_trend:
                signals.append('CONSISTENT_UPTREND')
                confidence_factors.append(12)
            elif consistent_down_trend:
                signals.append('CONSISTENT_DOWNTREND')
                confidence_factors.append(-12)
            
            # Volume trend confirmation
            volume_increasing = recent['volume'].iloc[-1] > recent['volume'].mean()
            if volume_increasing:
                signals.append('VOLUME_TREND_CONFIRMATION')
                confidence_factors.append(8)
            
            # Higher timeframe alignment
            if df_4h is not None:
                htf_bullish = latest_4h['ema_12'] > latest_4h['ema_26'] and latest_4h['adx'] > 20
                htf_bearish = latest_4h['ema_12'] < latest_4h['ema_26'] and latest_4h['adx'] > 20
                
                if htf_bullish and ema_bullish:
                    signals.append('HTF_BULLISH_CONFIRMATION')
                    confidence_factors.append(20)
                elif htf_bearish and ema_bearish:
                    signals.append('HTF_BEARISH_CONFIRMATION')
                    confidence_factors.append(-20)
            
            # Calculate confidence
            base_confidence = 50
            total_confidence = base_confidence + sum(confidence_factors)
            total_confidence = max(0, min(100, total_confidence))
            
            # Determine signal
            positive_factors = sum([f for f in confidence_factors if f > 0])
            negative_factors = abs(sum([f for f in confidence_factors if f < 0]))
            
            if total_confidence >= config['min_confidence'] and positive_factors > negative_factors:
                signal = 'BUY'
            elif total_confidence >= config['min_confidence'] and negative_factors > positive_factors:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            if signal == 'HOLD':
                return None
            
            # Calculate targets (wider for trend following)
            atr = latest['atr']
            current_price = latest['close']
            
            if signal == 'BUY':
                stop_loss = current_price - (atr * config['atr_stop_multiplier'])
                take_profit = current_price + (atr * config['atr_stop_multiplier'] * 2)
            else:
                stop_loss = current_price + (atr * config['atr_stop_multiplier'])
                take_profit = current_price - (atr * config['atr_stop_multiplier'] * 2)
            
            risk_reward = abs(take_profit - current_price) / abs(current_price - stop_loss)
            
            return {
                'strategy': 'trend_following',
                'symbol': symbol,
                'signal': signal,
                'confidence': round(total_confidence, 2),
                'entry_price': current_price,
                'stop_loss': round(stop_loss, 6),
                'take_profit': round(take_profit, 6),
                'risk_reward_ratio': round(risk_reward, 2),
                'signals': signals,
                'timeframe': '1h',
                'expected_duration': 480,  # 8 hours
                'market_conditions': {
                    'adx': latest['adx'],
                    'trend_strength': latest['trend_strength'],
                    'ema_alignment': 'bullish' if ema_bullish else 'bearish' if ema_bearish else 'neutral'
                }
            }
            
        except Exception as e:
            logger.error(f"Trend following strategy failed for {symbol}: {e}")
            return None
    
    def run_optimized_strategy_scan(self, symbols: List[str]) -> List[Dict]:
        """Run all optimized strategies across symbols"""
        all_signals = []
        
        for symbol in symbols:
            try:
                # Get multi-timeframe data
                market_data = self.get_enhanced_market_data(symbol)
                
                if not market_data:
                    continue
                
                # Run all strategies
                strategies = [
                    self.momentum_breakout_strategy(symbol, market_data),
                    self.mean_reversion_strategy(symbol, market_data),
                    self.trend_following_strategy(symbol, market_data)
                ]
                
                # Filter valid signals
                for signal in strategies:
                    if signal and signal['signal'] != 'HOLD':
                        all_signals.append(signal)
                        self.save_strategy_signal(signal)
                
            except Exception as e:
                logger.error(f"Strategy scan failed for {symbol}: {e}")
                continue
        
        # Sort by confidence
        all_signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return all_signals
    
    def save_strategy_signal(self, signal: Dict):
        """Save strategy signal to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO strategy_signals 
                (strategy_name, symbol, signal, confidence, entry_price, stop_loss, 
                 take_profit, risk_reward_ratio, timeframe, indicators_data, 
                 market_conditions, expected_duration)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['strategy'],
                signal['symbol'],
                signal['signal'],
                signal['confidence'],
                signal['entry_price'],
                signal['stop_loss'],
                signal['take_profit'],
                signal['risk_reward_ratio'],
                signal['timeframe'],
                json.dumps(signal['signals']),
                json.dumps(signal['market_conditions']),
                signal['expected_duration']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save strategy signal: {e}")
    
    def optimize_strategy_parameters(self, strategy_name: str, symbol: str = None) -> Dict:
        """Optimize strategy parameters using historical performance"""
        try:
            # Get historical signals and performance
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT * FROM strategy_signals 
                WHERE strategy_name = ?
            '''
            params = [strategy_name]
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            
            query += ' ORDER BY timestamp DESC LIMIT 1000'
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            if df.empty:
                return {'status': 'insufficient_data'}
            
            # Analyze performance patterns
            high_conf_signals = df[df['confidence'] >= 75]
            medium_conf_signals = df[(df['confidence'] >= 60) & (df['confidence'] < 75)]
            
            optimization_results = {
                'strategy_name': strategy_name,
                'total_signals': len(df),
                'high_confidence_signals': len(high_conf_signals),
                'medium_confidence_signals': len(medium_conf_signals),
                'avg_confidence': df['confidence'].mean(),
                'avg_risk_reward': df['risk_reward_ratio'].mean(),
                'recommended_adjustments': []
            }
            
            # Generate optimization recommendations
            if df['confidence'].mean() < 70:
                optimization_results['recommended_adjustments'].append({
                    'parameter': 'min_confidence',
                    'current': self.strategy_configs[strategy_name]['min_confidence'],
                    'recommended': max(65, df['confidence'].mean() - 5),
                    'reason': 'Lower threshold to capture more opportunities'
                })
            
            if df['risk_reward_ratio'].mean() < 1.5:
                optimization_results['recommended_adjustments'].append({
                    'parameter': 'profit_target_multiplier',
                    'current': 'varies',
                    'recommended': 'increase by 25%',
                    'reason': 'Improve risk-reward ratio'
                })
            
            return optimization_results
            
        except Exception as e:
            logger.error(f"Strategy optimization failed: {e}")
            return {'status': 'error', 'message': str(e)}

def main():
    """Main optimized strategies function"""
    try:
        strategies = OptimizedTradingStrategies()
        
        symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT',
            'XRP/USDT', 'DOT/USDT', 'AVAX/USDT', 'LINK/USDT', 'UNI/USDT'
        ]
        
        logger.info("🚀 Starting Optimized Trading Strategies Engine")
        
        # Run strategy scan
        signals = strategies.run_optimized_strategy_scan(symbols)
        
        logger.info(f"✅ Strategy scan complete: {len(signals)} high-quality signals generated")
        
        # Display top signals
        for i, signal in enumerate(signals[:5], 1):
            logger.info(f"{i}. {signal['strategy'].upper()}: {signal['symbol']} {signal['signal']} "
                       f"(Confidence: {signal['confidence']}%, R/R: {signal['risk_reward_ratio']})")
        
        # Run optimization analysis
        for strategy_name in ['momentum_breakout', 'mean_reversion', 'trend_following']:
            optimization = strategies.optimize_strategy_parameters(strategy_name)
            logger.info(f"📊 {strategy_name} optimization: {optimization.get('total_signals', 0)} signals analyzed")
        
    except Exception as e:
        logger.error(f"Optimized strategies engine failed: {e}")

if __name__ == "__main__":
    main()