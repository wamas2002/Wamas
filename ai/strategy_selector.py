"""
Strategy Selector with Performance Tracking
Allows switching between different trading strategies with real-time performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import sqlite3
from enum import Enum
from trading.okx_data_service import OKXDataService
from ai.comprehensive_ml_pipeline import TradingMLPipeline
from ai.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
import warnings
warnings.filterwarnings('ignore')

class StrategyType(Enum):
    SWING = "swing"
    DAY_TRADING = "day_trading"
    AI_ONLY = "ai_only"
    SCALPING = "scalping"
    MOMENTUM = "momentum"

class StrategySelector:
    """Advanced strategy selector with performance tracking and optimization"""
    
    def __init__(self, db_path: str = "data/strategy_performance.db"):
        self.db_path = db_path
        self.okx_data_service = OKXDataService()
        self.ml_pipeline = TradingMLPipeline()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        self.setup_database()
        
        # Strategy configurations
        self.strategies = {
            StrategyType.SWING: {
                'name': 'Swing Trading',
                'description': 'Medium-term positions (2-10 days)',
                'timeframe': '4h',
                'min_confidence': 0.7,
                'risk_per_trade': 0.02,
                'max_positions': 3,
                'stop_loss': 0.05,
                'take_profit': 0.15,
                'hold_time_min': 4,  # hours
                'hold_time_max': 240  # hours (10 days)
            },
            StrategyType.DAY_TRADING: {
                'name': 'Day Trading',
                'description': 'Intraday positions (minutes to hours)',
                'timeframe': '15m',
                'min_confidence': 0.6,
                'risk_per_trade': 0.01,
                'max_positions': 5,
                'stop_loss': 0.02,
                'take_profit': 0.06,
                'hold_time_min': 0.25,  # 15 minutes
                'hold_time_max': 8  # hours
            },
            StrategyType.AI_ONLY: {
                'name': 'AI-Only Mode',
                'description': 'Pure AI model decisions',
                'timeframe': '1h',
                'min_confidence': 0.8,
                'risk_per_trade': 0.015,
                'max_positions': 4,
                'stop_loss': 0.03,
                'take_profit': 0.10,
                'hold_time_min': 1,  # hours
                'hold_time_max': 72  # hours
            },
            StrategyType.SCALPING: {
                'name': 'Scalping',
                'description': 'Very short-term positions (seconds to minutes)',
                'timeframe': '1m',
                'min_confidence': 0.5,
                'risk_per_trade': 0.005,
                'max_positions': 8,
                'stop_loss': 0.008,
                'take_profit': 0.015,
                'hold_time_min': 0.02,  # 1 minute
                'hold_time_max': 0.5  # 30 minutes
            },
            StrategyType.MOMENTUM: {
                'name': 'Momentum Trading',
                'description': 'Trend-following with strong signals',
                'timeframe': '2h',
                'min_confidence': 0.75,
                'risk_per_trade': 0.025,
                'max_positions': 3,
                'stop_loss': 0.04,
                'take_profit': 0.12,
                'hold_time_min': 2,  # hours
                'hold_time_max': 120  # hours
            }
        }
    
    def setup_database(self):
        """Setup strategy performance database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Strategy performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_type TEXT NOT NULL,
                symbol TEXT NOT NULL,
                entry_time INTEGER NOT NULL,
                exit_time INTEGER,
                entry_price REAL NOT NULL,
                exit_price REAL,
                position_size REAL NOT NULL,
                pnl REAL,
                pnl_percentage REAL,
                win INTEGER,
                confidence REAL NOT NULL,
                signals_used TEXT,
                status TEXT DEFAULT 'OPEN',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Strategy statistics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_type TEXT NOT NULL,
                period_start INTEGER NOT NULL,
                period_end INTEGER NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                total_pnl REAL NOT NULL,
                avg_pnl_per_trade REAL NOT NULL,
                max_drawdown REAL NOT NULL,
                sharpe_ratio REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_strategy_config(self, strategy_type: StrategyType) -> Dict[str, Any]:
        """Get configuration for a specific strategy"""
        return self.strategies[strategy_type].copy()
    
    def generate_signal(self, symbol: str, strategy_type: StrategyType) -> Dict[str, Any]:
        """Generate trading signal based on selected strategy"""
        strategy_config = self.strategies[strategy_type]
        timeframe = strategy_config['timeframe']
        min_confidence = strategy_config['min_confidence']
        
        # Get market data
        data = self.okx_data_service.get_historical_data(symbol, timeframe, limit=200)
        if data.empty:
            return self._create_no_signal_response(symbol, strategy_type, "No market data")
        
        # Get ML prediction
        ml_signal = self._get_ml_signal(symbol, data, strategy_type)
        
        # Get technical analysis
        technical_signal = self._get_technical_signal(data, strategy_type)
        
        # Get sentiment signal
        sentiment_signal = self._get_sentiment_signal(symbol, strategy_type)
        
        # Combine signals based on strategy
        combined_signal = self._combine_signals(
            ml_signal, technical_signal, sentiment_signal, strategy_type
        )
        
        # Check if signal meets minimum confidence
        if combined_signal['confidence'] < min_confidence:
            return self._create_no_signal_response(
                symbol, strategy_type, f"Confidence {combined_signal['confidence']:.3f} below minimum {min_confidence}"
            )
        
        # Generate final signal
        return self._create_signal_response(symbol, strategy_type, combined_signal, data)
    
    def _get_ml_signal(self, symbol: str, data: pd.DataFrame, strategy_type: StrategyType) -> Dict[str, Any]:
        """Get ML signal optimized for strategy type"""
        try:
            prediction = self.ml_pipeline.predict(symbol, data.tail(1))
            
            # Adjust signal based on strategy type
            signal = prediction.get('signal', 0.5)
            confidence = prediction.get('confidence', 0.5)
            
            if strategy_type == StrategyType.SCALPING:
                # For scalping, reduce noise by requiring stronger signals
                if 0.4 < signal < 0.6:
                    signal = 0.5  # Neutralize weak signals
                confidence *= 0.9  # Slightly reduce confidence for high-frequency
            
            elif strategy_type == StrategyType.SWING:
                # For swing trading, smooth out short-term noise
                if len(data) >= 12:  # 12 periods for smoothing
                    recent_signals = []
                    for i in range(min(5, len(data) - 1)):
                        recent_pred = self.ml_pipeline.predict(symbol, data.iloc[-(i+2):-(i+1)])
                        recent_signals.append(recent_pred.get('signal', 0.5))
                    
                    if recent_signals:
                        signal = 0.7 * signal + 0.3 * np.mean(recent_signals)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'source': 'ml_pipeline'
            }
        except Exception as e:
            return {'signal': 0.5, 'confidence': 0.3, 'source': 'ml_pipeline', 'error': str(e)}
    
    def _get_technical_signal(self, data: pd.DataFrame, strategy_type: StrategyType) -> Dict[str, Any]:
        """Get technical analysis signal optimized for strategy"""
        try:
            if len(data) < 20:
                return {'signal': 0.5, 'confidence': 0.3, 'source': 'technical'}
            
            signal = 0.5
            confidence = 0.6
            
            # RSI
            rsi = self._calculate_rsi(data['close'])
            if not rsi.empty:
                rsi_value = rsi.iloc[-1]
                if strategy_type == StrategyType.SCALPING:
                    # More sensitive RSI for scalping
                    if rsi_value < 35:
                        signal += 0.15
                    elif rsi_value > 65:
                        signal -= 0.15
                else:
                    # Standard RSI levels
                    if rsi_value < 30:
                        signal += 0.2
                    elif rsi_value > 70:
                        signal -= 0.2
            
            # Moving average signals
            if len(data) >= 50:
                sma_20 = data['close'].rolling(20).mean()
                sma_50 = data['close'].rolling(50).mean()
                current_price = data['close'].iloc[-1]
                
                # Price vs moving averages
                if current_price > sma_20.iloc[-1]:
                    signal += 0.1
                if current_price > sma_50.iloc[-1]:
                    signal += 0.1
                
                # Golden/Death cross
                if sma_20.iloc[-1] > sma_50.iloc[-1] and sma_20.iloc[-2] <= sma_50.iloc[-2]:
                    signal += 0.15  # Golden cross
                elif sma_20.iloc[-1] < sma_50.iloc[-1] and sma_20.iloc[-2] >= sma_50.iloc[-2]:
                    signal -= 0.15  # Death cross
            
            # Volume analysis
            if strategy_type != StrategyType.SCALPING:  # Volume less important for scalping
                if len(data) >= 20:
                    volume_ma = data['volume'].rolling(20).mean()
                    current_volume = data['volume'].iloc[-1]
                    if current_volume > volume_ma.iloc[-1] * 1.5:
                        confidence += 0.1  # High volume increases confidence
            
            # MACD for momentum strategies
            if strategy_type in [StrategyType.MOMENTUM, StrategyType.SWING]:
                macd_line, macd_signal_line = self._calculate_macd(data['close'])
                if len(macd_line) > 1 and len(macd_signal_line) > 1:
                    if macd_line.iloc[-1] > macd_signal_line.iloc[-1] and macd_line.iloc[-2] <= macd_signal_line.iloc[-2]:
                        signal += 0.1  # MACD bullish crossover
                    elif macd_line.iloc[-1] < macd_signal_line.iloc[-1] and macd_line.iloc[-2] >= macd_signal_line.iloc[-2]:
                        signal -= 0.1  # MACD bearish crossover
            
            return {
                'signal': max(0, min(1, signal)),
                'confidence': max(0.3, min(1.0, confidence)),
                'source': 'technical'
            }
        except Exception as e:
            return {'signal': 0.5, 'confidence': 0.3, 'source': 'technical', 'error': str(e)}
    
    def _get_sentiment_signal(self, symbol: str, strategy_type: StrategyType) -> Dict[str, Any]:
        """Get sentiment signal with strategy-specific weighting"""
        try:
            sentiment_data = self.sentiment_analyzer.get_symbol_sentiment(symbol)
            
            sentiment_score = sentiment_data['overall_sentiment']
            confidence = sentiment_data['confidence']
            
            # Adjust sentiment weighting based on strategy
            if strategy_type == StrategyType.SCALPING:
                # Sentiment less important for scalping
                confidence *= 0.5
            elif strategy_type == StrategyType.SWING:
                # Sentiment more important for swing trading
                confidence *= 1.2
            
            return {
                'signal': sentiment_score,
                'confidence': min(1.0, confidence),
                'source': 'sentiment'
            }
        except Exception as e:
            return {'signal': 0.5, 'confidence': 0.4, 'source': 'sentiment', 'error': str(e)}
    
    def _combine_signals(self, ml_signal: Dict, technical_signal: Dict, 
                        sentiment_signal: Dict, strategy_type: StrategyType) -> Dict[str, Any]:
        """Combine signals based on strategy type"""
        
        # Strategy-specific weights
        weights = {
            StrategyType.AI_ONLY: {'ml': 0.8, 'technical': 0.15, 'sentiment': 0.05},
            StrategyType.DAY_TRADING: {'ml': 0.5, 'technical': 0.4, 'sentiment': 0.1},
            StrategyType.SWING: {'ml': 0.4, 'technical': 0.3, 'sentiment': 0.3},
            StrategyType.SCALPING: {'ml': 0.3, 'technical': 0.65, 'sentiment': 0.05},
            StrategyType.MOMENTUM: {'ml': 0.45, 'technical': 0.45, 'sentiment': 0.1}
        }
        
        strategy_weights = weights[strategy_type]
        
        # Calculate weighted signal
        combined_signal = (
            ml_signal['signal'] * strategy_weights['ml'] +
            technical_signal['signal'] * strategy_weights['technical'] +
            sentiment_signal['signal'] * strategy_weights['sentiment']
        )
        
        # Calculate weighted confidence
        combined_confidence = (
            ml_signal['confidence'] * strategy_weights['ml'] +
            technical_signal['confidence'] * strategy_weights['technical'] +
            sentiment_signal['confidence'] * strategy_weights['sentiment']
        )
        
        # Determine action
        if combined_signal >= 0.65:
            action = 'BUY'
        elif combined_signal <= 0.35:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        return {
            'signal': combined_signal,
            'confidence': combined_confidence,
            'action': action,
            'components': {
                'ml': ml_signal,
                'technical': technical_signal,
                'sentiment': sentiment_signal
            }
        }
    
    def _create_signal_response(self, symbol: str, strategy_type: StrategyType, 
                               combined_signal: Dict, data: pd.DataFrame) -> Dict[str, Any]:
        """Create complete signal response"""
        strategy_config = self.strategies[strategy_type]
        current_price = data['close'].iloc[-1]
        
        # Calculate position sizing
        position_size = self._calculate_position_size(strategy_config, combined_signal['confidence'])
        
        # Calculate stop loss and take profit
        stop_loss, take_profit = self._calculate_stop_take_levels(
            current_price, combined_signal['action'], strategy_config
        )
        
        return {
            'symbol': symbol,
            'strategy': strategy_type.value,
            'action': combined_signal['action'],
            'signal_strength': combined_signal['signal'],
            'confidence': combined_signal['confidence'],
            'current_price': current_price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'timeframe': strategy_config['timeframe'],
            'expected_hold_time': self._estimate_hold_time(strategy_config, combined_signal),
            'components': combined_signal['components'],
            'timestamp': datetime.now()
        }
    
    def _create_no_signal_response(self, symbol: str, strategy_type: StrategyType, reason: str) -> Dict[str, Any]:
        """Create no signal response"""
        return {
            'symbol': symbol,
            'strategy': strategy_type.value,
            'action': 'HOLD',
            'signal_strength': 0.5,
            'confidence': 0.0,
            'reason': reason,
            'timestamp': datetime.now()
        }
    
    def _calculate_position_size(self, strategy_config: Dict, confidence: float) -> float:
        """Calculate position size based on strategy and confidence"""
        base_risk = strategy_config['risk_per_trade']
        
        # Adjust risk based on confidence
        adjusted_risk = base_risk * confidence
        
        # Ensure minimum and maximum limits
        min_risk = base_risk * 0.5
        max_risk = base_risk * 1.5
        
        return max(min_risk, min(max_risk, adjusted_risk))
    
    def _calculate_stop_take_levels(self, current_price: float, action: str, 
                                   strategy_config: Dict) -> tuple:
        """Calculate stop loss and take profit levels"""
        if action == 'BUY':
            stop_loss = current_price * (1 - strategy_config['stop_loss'])
            take_profit = current_price * (1 + strategy_config['take_profit'])
        elif action == 'SELL':
            stop_loss = current_price * (1 + strategy_config['stop_loss'])
            take_profit = current_price * (1 - strategy_config['take_profit'])
        else:
            stop_loss = None
            take_profit = None
        
        return stop_loss, take_profit
    
    def _estimate_hold_time(self, strategy_config: Dict, combined_signal: Dict) -> str:
        """Estimate expected hold time based on signal strength"""
        min_hours = strategy_config['hold_time_min']
        max_hours = strategy_config['hold_time_max']
        
        signal_strength = abs(combined_signal['signal'] - 0.5) * 2  # 0 to 1
        
        # Stronger signals might be held longer
        estimated_hours = min_hours + (max_hours - min_hours) * signal_strength
        
        if estimated_hours < 1:
            return f"{int(estimated_hours * 60)} minutes"
        elif estimated_hours < 24:
            return f"{estimated_hours:.1f} hours"
        else:
            return f"{estimated_hours / 24:.1f} days"
    
    def get_strategy_performance(self, strategy_type: StrategyType, days: int = 7) -> Dict[str, Any]:
        """Get performance metrics for a strategy"""
        conn = sqlite3.connect(self.db_path)
        
        since_timestamp = int((datetime.now() - timedelta(days=days)).timestamp())
        
        query = '''
            SELECT * FROM strategy_performance 
            WHERE strategy_type = ? AND entry_time >= ?
            ORDER BY entry_time DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=(strategy_type.value, since_timestamp))
        conn.close()
        
        if df.empty:
            return self._get_default_performance_metrics(strategy_type, days)
        
        # Calculate performance metrics
        closed_trades = df[df['status'] == 'CLOSED']
        
        if closed_trades.empty:
            return self._get_default_performance_metrics(strategy_type, days)
        
        total_trades = len(closed_trades)
        winning_trades = len(closed_trades[closed_trades['win'] == 1])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = closed_trades['pnl'].sum()
        avg_return = closed_trades['pnl_percentage'].mean()
        
        return {
            'strategy': strategy_type.value,
            'period_days': days,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_return_pct': avg_return,
            'open_positions': len(df[df['status'] == 'OPEN']),
            'last_signal_time': df['entry_time'].max() if not df.empty else None
        }
    
    def _get_default_performance_metrics(self, strategy_type: StrategyType, days: int) -> Dict[str, Any]:
        """Get default performance metrics when no data available"""
        return {
            'strategy': strategy_type.value,
            'period_days': days,
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0.0,
            'total_pnl': 0.0,
            'avg_return_pct': 0.0,
            'open_positions': 0,
            'last_signal_time': None
        }
    
    def record_trade_entry(self, signal: Dict[str, Any]) -> int:
        """Record trade entry in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        entry_time = int(datetime.now().timestamp())
        
        cursor.execute('''
            INSERT INTO strategy_performance 
            (strategy_type, symbol, entry_time, entry_price, position_size, 
             confidence, signals_used, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['strategy'],
            signal['symbol'],
            entry_time,
            signal['current_price'],
            signal['position_size'],
            signal['confidence'],
            str(signal.get('components', {})),
            'OPEN'
        ))
        
        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return trade_id
    
    def record_trade_exit(self, trade_id: int, exit_price: float):
        """Record trade exit and calculate PnL"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get trade details
        cursor.execute('SELECT * FROM strategy_performance WHERE id = ?', (trade_id,))
        trade = cursor.fetchone()
        
        if trade:
            entry_price = trade[4]  # entry_price
            position_size = trade[5]  # position_size
            
            # Calculate PnL
            pnl = (exit_price - entry_price) * position_size
            pnl_percentage = (exit_price - entry_price) / entry_price * 100
            win = 1 if pnl > 0 else 0
            
            exit_time = int(datetime.now().timestamp())
            
            # Update trade record
            cursor.execute('''
                UPDATE strategy_performance 
                SET exit_time = ?, exit_price = ?, pnl = ?, 
                    pnl_percentage = ?, win = ?, status = ?
                WHERE id = ?
            ''', (exit_time, exit_price, pnl, pnl_percentage, win, 'CLOSED', trade_id))
            
            conn.commit()
        
        conn.close()
    
    def get_all_strategies_performance(self, days: int = 7) -> List[Dict[str, Any]]:
        """Get performance comparison for all strategies"""
        performances = []
        
        for strategy_type in StrategyType:
            performance = self.get_strategy_performance(strategy_type, days)
            performances.append(performance)
        
        # Sort by win rate
        performances.sort(key=lambda x: x['win_rate'], reverse=True)
        
        return performances
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        return macd_line, signal_line