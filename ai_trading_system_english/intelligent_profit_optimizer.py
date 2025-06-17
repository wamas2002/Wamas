#!/usr/bin/env python3
"""
Intelligent Profit Optimizer
Automated profit-taking system with dynamic exit strategies and market timing
"""

import ccxt
import os
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IntelligentProfitOptimizer:
    def __init__(self):
        self.exchange = None
        self.db_path = 'intelligent_profit_optimizer.db'
        self.initialize_exchange()
        self.setup_database()
        
        # Profit optimization parameters
        self.profit_levels = {
            'quick_profit': 0.005,    # 0.5% quick profit for volatile markets
            'standard_profit': 0.015, # 1.5% standard profit target
            'extended_profit': 0.025, # 2.5% extended profit for trending markets
            'maximum_profit': 0.05    # 5% maximum profit before mandatory exit
        }
        
        # Risk management thresholds
        self.trailing_percentage = 0.008  # 0.8% trailing stop from peak
        self.time_decay_hours = 6         # Reduce profit targets after 6 hours
        self.market_momentum_threshold = 0.002  # 0.2% momentum for trend detection

    def initialize_exchange(self):
        """Initialize OKX exchange for profit optimization"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.environ.get('OKX_API_KEY'),
                'secret': os.environ.get('OKX_SECRET_KEY'),
                'password': os.environ.get('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            self.exchange.load_markets()
            logger.info("Intelligent profit optimizer connected to OKX")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")

    def setup_database(self):
        """Setup profit optimization database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Profit optimization tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS profit_optimization (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry_time TIMESTAMP,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    peak_price REAL,
                    peak_profit REAL DEFAULT 0,
                    current_profit REAL DEFAULT 0,
                    profit_level TEXT DEFAULT 'accumulating',
                    market_trend TEXT DEFAULT 'neutral',
                    momentum_score REAL DEFAULT 0,
                    optimal_exit_price REAL,
                    exit_confidence REAL DEFAULT 0,
                    last_analysis TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'ACTIVE'
                )
            ''')
            
            # Profit-taking decisions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS profit_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    decision_type TEXT NOT NULL,
                    profit_amount REAL,
                    profit_percentage REAL,
                    market_conditions TEXT,
                    confidence_score REAL,
                    execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    result TEXT DEFAULT 'PENDING'
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Intelligent profit optimization database ready")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")

    def get_live_positions(self) -> List[Dict]:
        """Get all profitable positions for optimization"""
        try:
            positions = self.exchange.fetch_positions()
            profitable_positions = []
            
            for position in positions:
                if position['contracts'] and float(position['contracts']) > 0:
                    unrealized_pnl = float(position.get('unrealizedPnl', 0))
                    percentage = float(position.get('percentage', 0))
                    
                    # Focus on positions with profit potential
                    if percentage > -0.5:  # Include slightly losing positions that might recover
                        profitable_positions.append({
                            'symbol': position['symbol'],
                            'side': position['side'],
                            'size': float(position['contracts']),
                            'entry_price': float(position['entryPrice']) if position['entryPrice'] else 0,
                            'mark_price': float(position['markPrice']) if position['markPrice'] else 0,
                            'unrealized_pnl': unrealized_pnl,
                            'percentage': percentage,
                            'notional': float(position['notional']) if position['notional'] else 0,
                            'timestamp': datetime.now()
                        })
            
            return profitable_positions
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def analyze_market_momentum(self, symbol: str) -> Tuple[float, str]:
        """Analyze market momentum for trend detection"""
        try:
            # Get recent price data for momentum analysis
            ohlcv = self.exchange.fetch_ohlcv(symbol, '5m', limit=20)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate momentum indicators
            price_change_5m = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
            price_change_15m = (df['close'].iloc[-1] - df['close'].iloc[-15]) / df['close'].iloc[-15]
            
            # Volume momentum
            recent_volume = df['volume'].tail(5).mean()
            historical_volume = df['volume'].head(15).mean()
            volume_momentum = (recent_volume - historical_volume) / historical_volume if historical_volume > 0 else 0
            
            # Combined momentum score
            momentum_score = (price_change_5m * 0.4 + price_change_15m * 0.4 + volume_momentum * 0.2)
            
            # Determine trend
            if momentum_score > self.market_momentum_threshold:
                trend = 'bullish'
            elif momentum_score < -self.market_momentum_threshold:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            return momentum_score, trend
            
        except Exception as e:
            logger.error(f"Failed to analyze momentum for {symbol}: {e}")
            return 0.0, 'neutral'

    def calculate_optimal_exit_strategy(self, position: Dict) -> Dict:
        """Calculate optimal exit strategy based on multiple factors"""
        try:
            symbol = position['symbol']
            current_profit_pct = position['percentage'] / 100
            
            # Get market momentum
            momentum_score, trend = self.analyze_market_momentum(symbol)
            
            # Determine profit level category
            if current_profit_pct >= self.profit_levels['maximum_profit']:
                profit_level = 'maximum'
                exit_confidence = 0.95
            elif current_profit_pct >= self.profit_levels['extended_profit']:
                profit_level = 'extended'
                exit_confidence = 0.8 if trend != 'bullish' else 0.6
            elif current_profit_pct >= self.profit_levels['standard_profit']:
                profit_level = 'standard'
                exit_confidence = 0.7 if trend == 'bearish' else 0.4
            elif current_profit_pct >= self.profit_levels['quick_profit']:
                profit_level = 'quick'
                exit_confidence = 0.6 if trend == 'bearish' else 0.2
            else:
                profit_level = 'accumulating'
                exit_confidence = 0.1
            
            # Adjust confidence based on momentum
            if trend == 'bullish' and momentum_score > 0.01:
                exit_confidence *= 0.7  # Reduce exit urgency in strong uptrend
            elif trend == 'bearish' and momentum_score < -0.01:
                exit_confidence *= 1.3  # Increase exit urgency in strong downtrend
            
            # Calculate optimal exit price
            current_price = position['mark_price']
            if trend == 'bullish':
                # Set higher target in uptrend
                optimal_exit_price = current_price * (1 + 0.01)
            elif trend == 'bearish':
                # Exit sooner in downtrend
                optimal_exit_price = current_price * (1 - 0.005)
            else:
                # Neutral market - exit at current profit level
                optimal_exit_price = current_price
            
            return {
                'symbol': symbol,
                'profit_level': profit_level,
                'current_profit_pct': current_profit_pct,
                'momentum_score': momentum_score,
                'trend': trend,
                'exit_confidence': min(exit_confidence, 1.0),
                'optimal_exit_price': optimal_exit_price,
                'should_exit': exit_confidence > 0.75,
                'exit_reason': self._determine_exit_reason(profit_level, trend, exit_confidence)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate exit strategy: {e}")
            return {'should_exit': False, 'exit_reason': f'Analysis error: {e}'}

    def _determine_exit_reason(self, profit_level: str, trend: str, confidence: float) -> str:
        """Determine specific reason for exit recommendation"""
        if profit_level == 'maximum':
            return 'Maximum profit target reached'
        elif profit_level == 'extended' and trend == 'bearish':
            return 'Extended profit secured before downtrend'
        elif profit_level == 'standard' and confidence > 0.75:
            return 'Standard profit target with market uncertainty'
        elif trend == 'bearish' and confidence > 0.75:
            return 'Bearish momentum detected - protect gains'
        else:
            return 'Optimal profit-taking opportunity'

    def execute_intelligent_exit(self, position: Dict, exit_strategy: Dict) -> bool:
        """Execute intelligent profit-taking exit"""
        try:
            symbol = position['symbol']
            side = 'sell' if position['side'] == 'long' else 'buy'
            amount = position['size']
            
            # Create market order for immediate execution
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount,
                params={'reduceOnly': True}
            )
            
            profit_amount = position['unrealized_pnl']
            profit_pct = position['percentage']
            
            logger.info(f"💰 PROFIT SECURED: {symbol} {position['side']} "
                       f"Profit: ${profit_amount:.2f} ({profit_pct:.2f}%) "
                       f"Reason: {exit_strategy['exit_reason']}")
            
            # Log the profit decision
            self.log_profit_decision(symbol, exit_strategy, profit_amount, profit_pct, 'EXECUTED')
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute profit exit for {position['symbol']}: {e}")
            self.log_profit_decision(position['symbol'], exit_strategy, 0, 0, 'FAILED')
            return False

    def log_profit_decision(self, symbol: str, strategy: Dict, profit_amount: float, profit_pct: float, status: str):
        """Log profit-taking decision to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO profit_decisions 
                (symbol, decision_type, profit_amount, profit_percentage, 
                 market_conditions, confidence_score, result)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, 'PROFIT_EXIT', profit_amount, profit_pct,
                  f"Trend: {strategy.get('trend', 'unknown')}, Level: {strategy.get('profit_level', 'unknown')}",
                  strategy.get('exit_confidence', 0), status))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log profit decision: {e}")

    def update_position_tracking(self, position: Dict, strategy: Dict):
        """Update position tracking with optimization data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if position exists
            cursor.execute('''
                SELECT id, peak_price, peak_profit FROM profit_optimization 
                WHERE symbol = ? AND status = 'ACTIVE'
            ''', (position['symbol'],))
            
            existing = cursor.fetchone()
            current_price = position['mark_price']
            current_profit = position['percentage'] / 100
            
            if existing:
                # Update existing position
                position_id, peak_price, peak_profit = existing
                new_peak_price = max(peak_price or 0, current_price)
                new_peak_profit = max(peak_profit or 0, current_profit)
                
                cursor.execute('''
                    UPDATE profit_optimization 
                    SET current_price = ?, peak_price = ?, peak_profit = ?,
                        current_profit = ?, profit_level = ?, market_trend = ?,
                        momentum_score = ?, optimal_exit_price = ?, 
                        exit_confidence = ?, last_analysis = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (current_price, new_peak_price, new_peak_profit, current_profit,
                      strategy.get('profit_level', 'unknown'), strategy.get('trend', 'neutral'),
                      strategy.get('momentum_score', 0), strategy.get('optimal_exit_price', current_price),
                      strategy.get('exit_confidence', 0), position_id))
            else:
                # Insert new position
                cursor.execute('''
                    INSERT INTO profit_optimization 
                    (symbol, entry_price, current_price, peak_price, peak_profit,
                     current_profit, profit_level, market_trend, momentum_score,
                     optimal_exit_price, exit_confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (position['symbol'], position['entry_price'], current_price,
                      current_price, current_profit, current_profit,
                      strategy.get('profit_level', 'accumulating'), strategy.get('trend', 'neutral'),
                      strategy.get('momentum_score', 0), strategy.get('optimal_exit_price', current_price),
                      strategy.get('exit_confidence', 0)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update position tracking: {e}")

    def optimize_portfolio_profits(self):
        """Main profit optimization loop"""
        try:
            logger.info("🎯 Running intelligent profit optimization...")
            
            positions = self.get_live_positions()
            
            if not positions:
                logger.info("📭 No positions available for profit optimization")
                return
            
            profitable_positions = [pos for pos in positions if pos['percentage'] > 0]
            total_profit = sum(pos['unrealized_pnl'] for pos in profitable_positions)
            
            logger.info(f"💹 Analyzing {len(positions)} positions (${total_profit:.2f} total unrealized profit)")
            
            exits_executed = 0
            profit_secured = 0
            
            for position in positions:
                # Calculate optimal exit strategy
                exit_strategy = self.calculate_optimal_exit_strategy(position)
                
                # Update tracking
                self.update_position_tracking(position, exit_strategy)
                
                profit_status = "🟢" if position['percentage'] > 0 else "🔴"
                logger.info(f"{profit_status} {position['symbol']}: {position['percentage']:.2f}% "
                           f"(Level: {exit_strategy.get('profit_level', 'unknown')}, "
                           f"Trend: {exit_strategy.get('trend', 'neutral')}, "
                           f"Exit Confidence: {exit_strategy.get('exit_confidence', 0):.1%})")
                
                # Execute exit if strategy recommends it
                if exit_strategy.get('should_exit', False):
                    logger.info(f"🚨 Profit exit signal for {position['symbol']}: "
                               f"{exit_strategy.get('exit_reason', 'Optimal timing')}")
                    
                    if exit_strategy.get('exit_confidence', 0) >= 0.8:
                        if self.execute_intelligent_exit(position, exit_strategy):
                            exits_executed += 1
                            profit_secured += position['unrealized_pnl']
                    else:
                        logger.info(f"⚠️ Moderate confidence exit signal - monitoring {position['symbol']}")
            
            if exits_executed > 0:
                logger.info(f"✅ Profit optimization complete: {exits_executed} positions closed, "
                           f"${profit_secured:.2f} profit secured")
            else:
                logger.info("✅ Profit optimization complete: All positions maintained for further gains")
                
        except Exception as e:
            logger.error(f"Profit optimization error: {e}")

def main():
    """Main profit optimization function"""
    optimizer = IntelligentProfitOptimizer()
    
    logger.info("🚀 Starting Intelligent Profit Optimizer")
    logger.info("💎 Automated profit-taking with market timing and trend analysis")
    
    while True:
        try:
            optimizer.optimize_portfolio_profits()
            
            logger.info("⏰ Next profit optimization in 3 minutes...")
            time.sleep(180)  # Check every 3 minutes for optimal timing
            
        except KeyboardInterrupt:
            logger.info("🛑 Profit optimizer stopped by user")
            break
        except Exception as e:
            logger.error(f"Optimizer error: {e}")
            time.sleep(180)

if __name__ == "__main__":
    main()