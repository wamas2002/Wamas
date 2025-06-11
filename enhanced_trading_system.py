#!/usr/bin/env python3
"""
Enhanced Trading System with Advanced Features
Implements multi-timeframe analysis, dynamic risk management, and ML optimization
"""

import sqlite3
import ccxt
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from sklearn.ensemble import RandomForestClassifier
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTradingSystem:
    def __init__(self):
        self.db_path = 'trading_platform.db'
        self.exchange = None
        self.initialize_exchange()
        self.setup_enhanced_database()
        
    def initialize_exchange(self):
        """Initialize OKX connection"""
        try:
            api_key = os.environ.get('OKX_API_KEY')
            secret_key = os.environ.get('OKX_SECRET_KEY')
            passphrase = os.environ.get('OKX_PASSPHRASE')
            
            if api_key and secret_key and passphrase:
                self.exchange = ccxt.okx({
                    'apiKey': api_key,
                    'secret': secret_key,
                    'password': passphrase,
                    'sandbox': False,
                    'rateLimit': 800,
                    'enableRateLimit': True,
                })
                logger.info("Enhanced Trading System connected to OKX")
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")

    def setup_enhanced_database(self):
        """Setup enhanced database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Multi-timeframe analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS multi_timeframe_analysis (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    symbol TEXT,
                    tf_1m_signal TEXT,
                    tf_5m_signal TEXT,
                    tf_15m_signal TEXT,
                    tf_1h_signal TEXT,
                    tf_4h_signal TEXT,
                    tf_1d_signal TEXT,
                    consensus_signal TEXT,
                    consensus_strength REAL,
                    final_recommendation TEXT
                )
            ''')
            
            # Dynamic stop losses
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dynamic_stops (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    symbol TEXT,
                    current_price REAL,
                    atr_value REAL,
                    stop_loss_level REAL,
                    take_profit_level REAL,
                    risk_reward_ratio REAL
                )
            ''')
            
            # Risk management
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    portfolio_value REAL,
                    max_position_risk REAL,
                    correlation_risk REAL,
                    volatility_risk REAL,
                    drawdown_risk REAL,
                    overall_risk_score REAL
                )
            ''')
            
            # Performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_tracking (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    roi_percentage REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Enhanced database tables initialized")
            
        except Exception as e:
            logger.error(f"Database setup error: {e}")

    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = np.abs(df['high'] - df['close'].shift())
        df['low_close'] = np.abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=period).mean()
        return df['atr'].iloc[-1]

    def multi_timeframe_analysis(self, symbol):
        """Analyze signals across multiple timeframes"""
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        signals = {}
        
        try:
            for tf in timeframes:
                ohlcv = self.exchange.fetch_ohlcv(f"{symbol}/USDT", tf, limit=50)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                
                # Calculate indicators
                df['rsi'] = self.calculate_rsi(df['close'])
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['ema_12'] = df['close'].ewm(span=12).mean()
                df['ema_26'] = df['close'].ewm(span=26).mean()
                df['macd'] = df['ema_12'] - df['ema_26']
                
                current = df.iloc[-1]
                
                # Generate signal for this timeframe
                signal_score = 0
                
                # RSI signals
                if current['rsi'] < 30:
                    signal_score += 2  # Strong buy
                elif current['rsi'] < 40:
                    signal_score += 1  # Buy
                elif current['rsi'] > 70:
                    signal_score -= 2  # Strong sell
                elif current['rsi'] > 60:
                    signal_score -= 1  # Sell
                
                # Price vs SMA
                if current['close'] > current['sma_20']:
                    signal_score += 1
                else:
                    signal_score -= 1
                
                # MACD
                if current['macd'] > 0:
                    signal_score += 1
                else:
                    signal_score -= 1
                
                # Determine signal
                if signal_score >= 2:
                    signals[tf] = 'BUY'
                elif signal_score <= -2:
                    signals[tf] = 'SELL'
                else:
                    signals[tf] = 'HOLD'
                
                time.sleep(0.2)  # Rate limiting
            
            # Calculate consensus
            buy_count = sum(1 for s in signals.values() if s == 'BUY')
            sell_count = sum(1 for s in signals.values() if s == 'SELL')
            hold_count = sum(1 for s in signals.values() if s == 'HOLD')
            
            total_signals = len(signals)
            consensus_strength = max(buy_count, sell_count, hold_count) / total_signals
            
            if buy_count > sell_count and buy_count > hold_count:
                consensus = 'BUY'
            elif sell_count > buy_count and sell_count > hold_count:
                consensus = 'SELL'
            else:
                consensus = 'HOLD'
            
            # Save to database
            self.save_multi_timeframe_analysis(symbol, signals, consensus, consensus_strength)
            
            return {
                'signals': signals,
                'consensus': consensus,
                'strength': consensus_strength
            }
            
        except Exception as e:
            logger.error(f"Multi-timeframe analysis error for {symbol}: {e}")
            return None

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def dynamic_risk_management(self, symbol):
        """Implement dynamic risk management"""
        try:
            # Get market data
            ohlcv = self.exchange.fetch_ohlcv(f"{symbol}/USDT", '1h', limit=50)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            current_price = df['close'].iloc[-1]
            atr = self.calculate_atr(df)
            
            # Dynamic stop loss (2x ATR)
            stop_loss_distance = atr * 2.0
            stop_loss_level = current_price - stop_loss_distance
            
            # Dynamic take profit (3x ATR for 1.5:1 risk/reward)
            take_profit_distance = atr * 3.0
            take_profit_level = current_price + take_profit_distance
            
            risk_reward_ratio = take_profit_distance / stop_loss_distance
            
            # Calculate volatility risk
            volatility = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
            current_volatility = volatility.iloc[-1]
            
            # Save dynamic stops
            self.save_dynamic_stops(symbol, current_price, atr, stop_loss_level, take_profit_level, risk_reward_ratio)
            
            return {
                'stop_loss': stop_loss_level,
                'take_profit': take_profit_level,
                'risk_reward': risk_reward_ratio,
                'volatility': current_volatility,
                'atr': atr
            }
            
        except Exception as e:
            logger.error(f"Dynamic risk management error for {symbol}: {e}")
            return None

    def advanced_position_sizing(self, symbol, signal_confidence, volatility):
        """Calculate optimal position size based on volatility and confidence"""
        try:
            # Get portfolio value
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT quantity, current_price FROM portfolio WHERE symbol = 'USDT'")
            usdt_result = cursor.fetchone()
            usdt_balance = float(usdt_result[0]) if usdt_result else 0
            
            cursor.execute("SELECT symbol, quantity, current_price FROM portfolio WHERE symbol != 'USDT' AND quantity > 0")
            positions = cursor.fetchall()
            
            total_crypto_value = sum(float(qty) * float(price) for _, qty, price in positions)
            total_portfolio_value = usdt_balance + total_crypto_value
            
            conn.close()
            
            # Base position size (1% of portfolio)
            base_position_size = total_portfolio_value * 0.01
            
            # Adjust for confidence (0.5x to 2x)
            confidence_multiplier = 0.5 + (signal_confidence / 100) * 1.5
            
            # Adjust for volatility (reduce size for high volatility)
            volatility_adjustment = max(0.3, 1 - (volatility * 2))
            
            # Final position size
            position_size = base_position_size * confidence_multiplier * volatility_adjustment
            
            # Cap at 5% of portfolio for safety
            max_position_size = total_portfolio_value * 0.05
            position_size = min(position_size, max_position_size)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Position sizing error: {e}")
            return 0

    def generate_enhanced_signals(self, symbol):
        """Generate enhanced signals with all improvements"""
        try:
            # Multi-timeframe analysis
            mtf_analysis = self.multi_timeframe_analysis(symbol)
            if not mtf_analysis:
                return None
            
            # Dynamic risk management
            risk_analysis = self.dynamic_risk_management(symbol)
            if not risk_analysis:
                return None
            
            # Only generate signals if consensus is strong
            if mtf_analysis['strength'] < 0.6:  # At least 60% consensus
                return None
            
            consensus_signal = mtf_analysis['consensus']
            if consensus_signal == 'HOLD':
                return None
            
            # Calculate confidence based on consensus strength and volatility
            base_confidence = mtf_analysis['strength'] * 100
            volatility_penalty = risk_analysis['volatility'] * 10
            final_confidence = max(50, base_confidence - volatility_penalty)
            
            # Position sizing
            position_size = self.advanced_position_sizing(symbol, final_confidence, risk_analysis['volatility'])
            
            signal = {
                'symbol': symbol,
                'signal': consensus_signal,
                'confidence': final_confidence,
                'reasoning': f"Multi-timeframe consensus ({mtf_analysis['strength']:.1%}), "
                           f"ATR: ${risk_analysis['atr']:.2f}, "
                           f"R/R: {risk_analysis['risk_reward']:.1f}:1, "
                           f"Volatility: {risk_analysis['volatility']:.1%}",
                'position_size': position_size,
                'stop_loss': risk_analysis['stop_loss'],
                'take_profit': risk_analysis['take_profit'],
                'timeframe_signals': mtf_analysis['signals'],
                'timestamp': datetime.now().isoformat()
            }
            
            return signal
            
        except Exception as e:
            logger.error(f"Enhanced signal generation error for {symbol}: {e}")
            return None

    def update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        try:
            # Get recent trades
            conn_live = sqlite3.connect('live_trading.db')
            cursor_live = conn_live.cursor()
            
            cursor_live.execute('''
                SELECT side, amount, price, pnl, timestamp
                FROM live_trades 
                WHERE timestamp > datetime('now', '-7 days')
                ORDER BY timestamp DESC
            ''')
            
            trades = cursor_live.fetchall()
            conn_live.close()
            
            if len(trades) >= 5:
                total_trades = len(trades)
                profitable_trades = [t for t in trades if t[3] and float(t[3]) > 0]
                losing_trades = [t for t in trades if t[3] and float(t[3]) < 0]
                
                winning_trades = len(profitable_trades)
                losing_count = len(losing_trades)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                # Calculate profit factor
                gross_profit = sum(float(t[3]) for t in profitable_trades)
                gross_loss = abs(sum(float(t[3]) for t in losing_trades))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                # Calculate returns for Sharpe ratio
                returns = [float(t[3]) for t in trades if t[3]]
                if len(returns) >= 3:
                    mean_return = np.mean(returns)
                    std_return = np.std(returns)
                    sharpe_ratio = mean_return / std_return if std_return > 0 else 0
                else:
                    sharpe_ratio = 0
                
                # Calculate max drawdown
                cumulative_pnl = np.cumsum([float(t[3]) for t in reversed(trades) if t[3]])
                running_max = np.maximum.accumulate(cumulative_pnl)
                drawdown = (cumulative_pnl - running_max)
                max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
                
                # Calculate ROI
                total_pnl = sum(float(t[3]) for t in trades if t[3])
                initial_value = 450  # Approximate starting portfolio value
                roi_percentage = (total_pnl / initial_value) * 100
                
                # Save performance metrics
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO performance_tracking 
                    (timestamp, total_trades, winning_trades, losing_trades, win_rate, 
                     profit_factor, sharpe_ratio, max_drawdown, roi_percentage)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    total_trades,
                    winning_trades,
                    losing_count,
                    win_rate,
                    profit_factor,
                    sharpe_ratio,
                    max_drawdown,
                    roi_percentage
                ))
                
                conn.commit()
                conn.close()
                
                logger.info(f"Performance Update: {total_trades} trades, {win_rate:.1%} win rate, "
                           f"Profit factor: {profit_factor:.2f}, Sharpe: {sharpe_ratio:.2f}")
                
                return {
                    'total_trades': total_trades,
                    'win_rate': win_rate,
                    'profit_factor': profit_factor,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'roi_percentage': roi_percentage
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Performance metrics error: {e}")
            return None

    def save_multi_timeframe_analysis(self, symbol, signals, consensus, strength):
        """Save multi-timeframe analysis to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO multi_timeframe_analysis 
                (timestamp, symbol, tf_1m_signal, tf_5m_signal, tf_15m_signal, 
                 tf_1h_signal, tf_4h_signal, tf_1d_signal, consensus_signal, consensus_strength)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                symbol,
                signals.get('1m', 'HOLD'),
                signals.get('5m', 'HOLD'),
                signals.get('15m', 'HOLD'),
                signals.get('1h', 'HOLD'),
                signals.get('4h', 'HOLD'),
                signals.get('1d', 'HOLD'),
                consensus,
                strength
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving multi-timeframe analysis: {e}")

    def save_dynamic_stops(self, symbol, price, atr, stop_loss, take_profit, risk_reward):
        """Save dynamic stop levels to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO dynamic_stops 
                (timestamp, symbol, current_price, atr_value, stop_loss_level, take_profit_level, risk_reward_ratio)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                symbol,
                price,
                atr,
                stop_loss,
                take_profit,
                risk_reward
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving dynamic stops: {e}")

    def save_enhanced_signals(self, signals):
        """Save enhanced signals to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for signal in signals:
                cursor.execute('''
                    INSERT INTO ai_signals (symbol, signal, confidence, reasoning, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    signal['symbol'],
                    signal['signal'],
                    signal['confidence'],
                    signal['reasoning'],
                    signal['timestamp']
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving enhanced signals: {e}")

    def run_enhanced_trading_cycle(self):
        """Run enhanced trading cycle with all improvements"""
        crypto_symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
        enhanced_signals = []
        
        logger.info("Running Enhanced Trading Analysis with Multi-Timeframe & Risk Management")
        
        for symbol in crypto_symbols:
            try:
                signal = self.generate_enhanced_signals(symbol)
                if signal and signal['confidence'] >= 65:  # Higher threshold for quality
                    enhanced_signals.append(signal)
                    logger.info(f"{symbol}: {signal['signal']} ({signal['confidence']:.1f}%) "
                               f"- Position: ${signal['position_size']:.2f}, "
                               f"Stop: ${signal['stop_loss']:.2f}, "
                               f"Target: ${signal['take_profit']:.2f}")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Enhanced analysis error for {symbol}: {e}")
        
        # Save enhanced signals
        if enhanced_signals:
            self.save_enhanced_signals(enhanced_signals)
            logger.info(f"Generated {len(enhanced_signals)} high-quality enhanced signals")
        
        # Update performance metrics
        self.update_performance_metrics()
        
        return enhanced_signals

def main():
    """Main enhanced trading function"""
    enhanced_system = EnhancedTradingSystem()
    enhanced_system.run_enhanced_trading_cycle()

if __name__ == "__main__":
    main()