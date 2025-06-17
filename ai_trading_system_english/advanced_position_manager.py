#!/usr/bin/env python3
"""
Advanced Position Manager
Intelligent position management with dynamic exit strategies and risk optimization
"""

import ccxt
import os
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedPositionManager:
    def __init__(self):
        self.exchange = None
        self.db_path = 'advanced_position_management.db'
        self.initialize_exchange()
        self.setup_database()
        
        # Risk management parameters
        self.trailing_stop_distance = 0.03  # 3% trailing stop
        self.profit_target_adjustment = 0.02  # 2% profit target adjustment
        self.max_loss_tolerance = 0.15  # 15% maximum loss per position
        self.time_based_exit_hours = 24  # Exit positions after 24 hours regardless

    def initialize_exchange(self):
        """Initialize OKX exchange for position management"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.environ.get('OKX_API_KEY'),
                'secret': os.environ.get('OKX_SECRET_KEY'),
                'password': os.environ.get('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            self.exchange.load_markets()
            logger.info("Advanced position manager connected to OKX")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")

    def setup_database(self):
        """Setup position management database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Position management table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS position_management (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    position_size REAL NOT NULL,
                    leverage INTEGER NOT NULL,
                    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    highest_price REAL,
                    lowest_price REAL,
                    trailing_stop_price REAL,
                    profit_target_price REAL,
                    unrealized_pnl REAL,
                    max_profit REAL DEFAULT 0,
                    max_loss REAL DEFAULT 0,
                    exit_strategy TEXT DEFAULT 'ACTIVE',
                    risk_level TEXT DEFAULT 'MODERATE',
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Exit decisions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS exit_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    decision_type TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    current_price REAL,
                    profit_loss REAL,
                    confidence_score REAL,
                    execution_status TEXT DEFAULT 'PENDING',
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Advanced position management database ready")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")

    def get_active_positions(self) -> List[Dict]:
        """Get all active positions from OKX"""
        try:
            positions = self.exchange.fetch_positions()
            active_positions = []
            
            for position in positions:
                if position['contracts'] and float(position['contracts']) > 0:
                    active_positions.append({
                        'symbol': position['symbol'],
                        'side': position['side'],
                        'size': float(position['contracts']),
                        'entry_price': float(position['entryPrice']) if position['entryPrice'] else 0,
                        'mark_price': float(position['markPrice']) if position['markPrice'] else 0,
                        'unrealized_pnl': float(position['unrealizedPnl']) if position['unrealizedPnl'] else 0,
                        'percentage': float(position['percentage']) if position['percentage'] else 0,
                        'leverage': position.get('leverage', 1),
                        'notional': float(position['notional']) if position['notional'] else 0,
                        'timestamp': datetime.now()
                    })
            
            return active_positions
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def update_position_tracking(self, position: Dict):
        """Update or insert position tracking data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if position exists
            cursor.execute('''
                SELECT id, highest_price, lowest_price, max_profit, max_loss 
                FROM position_management 
                WHERE symbol = ? AND exit_strategy = 'ACTIVE'
            ''', (position['symbol'],))
            
            existing = cursor.fetchone()
            current_price = position['mark_price']
            
            if existing:
                # Update existing position
                position_id, highest_price, lowest_price, max_profit, max_loss = existing
                
                # Update price extremes
                new_highest = max(highest_price or 0, current_price)
                new_lowest = min(lowest_price or float('inf'), current_price)
                new_max_profit = max(max_profit or 0, position['unrealized_pnl'])
                new_max_loss = min(max_loss or 0, position['unrealized_pnl'])
                
                # Calculate trailing stop
                if position['side'] == 'long':
                    trailing_stop_price = new_highest * (1 - self.trailing_stop_distance)
                    profit_target_price = position['entry_price'] * (1 + self.profit_target_adjustment)
                else:
                    trailing_stop_price = new_lowest * (1 + self.trailing_stop_distance)
                    profit_target_price = position['entry_price'] * (1 - self.profit_target_adjustment)
                
                cursor.execute('''
                    UPDATE position_management 
                    SET current_price = ?, highest_price = ?, lowest_price = ?, 
                        trailing_stop_price = ?, profit_target_price = ?,
                        unrealized_pnl = ?, max_profit = ?, max_loss = ?,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (current_price, new_highest, new_lowest, trailing_stop_price,
                      profit_target_price, position['unrealized_pnl'], 
                      new_max_profit, new_max_loss, position_id))
            else:
                # Insert new position
                if position['side'] == 'long':
                    trailing_stop_price = current_price * (1 - self.trailing_stop_distance)
                    profit_target_price = position['entry_price'] * (1 + self.profit_target_adjustment)
                else:
                    trailing_stop_price = current_price * (1 + self.trailing_stop_distance)
                    profit_target_price = position['entry_price'] * (1 - self.profit_target_adjustment)
                
                cursor.execute('''
                    INSERT INTO position_management 
                    (symbol, side, entry_price, current_price, position_size, leverage,
                     highest_price, lowest_price, trailing_stop_price, profit_target_price,
                     unrealized_pnl, max_profit, max_loss)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (position['symbol'], position['side'], position['entry_price'],
                      current_price, position['size'], position['leverage'],
                      current_price, current_price, trailing_stop_price,
                      profit_target_price, position['unrealized_pnl'], 
                      position['unrealized_pnl'], position['unrealized_pnl']))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to update position tracking: {e}")

    def analyze_exit_conditions(self, position: Dict) -> Dict:
        """Analyze if position should be exited based on multiple conditions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get position management data
            cursor.execute('''
                SELECT * FROM position_management 
                WHERE symbol = ? AND exit_strategy = 'ACTIVE'
            ''', (position['symbol'],))
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return {'should_exit': False, 'reason': 'No tracking data'}
            
            # Parse row data
            columns = ['id', 'symbol', 'side', 'entry_price', 'current_price', 'position_size',
                      'leverage', 'entry_time', 'highest_price', 'lowest_price', 'trailing_stop_price',
                      'profit_target_price', 'unrealized_pnl', 'max_profit', 'max_loss',
                      'exit_strategy', 'risk_level', 'last_updated']
            
            pos_data = dict(zip(columns, row))
            current_price = position['mark_price']
            
            exit_conditions = []
            confidence_score = 0
            
            # 1. Trailing Stop Loss Condition
            if position['side'] == 'long' and current_price <= pos_data['trailing_stop_price']:
                exit_conditions.append('Trailing stop loss triggered')
                confidence_score += 0.9
            elif position['side'] == 'short' and current_price >= pos_data['trailing_stop_price']:
                exit_conditions.append('Trailing stop loss triggered')
                confidence_score += 0.9
            
            # 2. Profit Target Reached
            if position['side'] == 'long' and current_price >= pos_data['profit_target_price']:
                exit_conditions.append('Profit target reached')
                confidence_score += 0.8
            elif position['side'] == 'short' and current_price <= pos_data['profit_target_price']:
                exit_conditions.append('Profit target reached')
                confidence_score += 0.8
            
            # 3. Maximum Loss Tolerance
            if position['percentage'] <= -self.max_loss_tolerance * 100:
                exit_conditions.append('Maximum loss tolerance exceeded')
                confidence_score += 0.95
            
            # 4. Time-based Exit (positions older than 24 hours)
            entry_time = datetime.fromisoformat(pos_data['entry_time'])
            if datetime.now() - entry_time > timedelta(hours=self.time_based_exit_hours):
                exit_conditions.append('Time-based exit (24h limit)')
                confidence_score += 0.6
            
            # 5. Profit Protection (if position was profitable but now declining)
            if pos_data['max_profit'] > 2 and position['unrealized_pnl'] < pos_data['max_profit'] * 0.5:
                exit_conditions.append('Profit protection (50% retracement)')
                confidence_score += 0.7
            
            should_exit = len(exit_conditions) > 0 and confidence_score >= 0.7
            
            return {
                'should_exit': should_exit,
                'reasons': exit_conditions,
                'confidence_score': min(confidence_score, 1.0),
                'current_price': current_price,
                'profit_loss': position['unrealized_pnl'],
                'position_data': pos_data
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze exit conditions: {e}")
            return {'should_exit': False, 'reason': f'Analysis error: {e}'}

    def execute_position_exit(self, position: Dict, exit_decision: Dict) -> bool:
        """Execute position exit based on decision"""
        try:
            symbol = position['symbol']
            side = 'sell' if position['side'] == 'long' else 'buy'
            amount = position['size']
            
            # Create market order to close position
            order = self.exchange.create_market_order(
                symbol=symbol,
                side=side,
                amount=amount,
                params={'reduceOnly': True}  # Ensure we're closing position, not opening new one
            )
            
            logger.info(f"🚪 POSITION CLOSED: {symbol} {position['side']} "
                       f"Size: {amount} P&L: ${exit_decision['profit_loss']:.2f} "
                       f"Reason: {', '.join(exit_decision['reasons'])}")
            
            # Log exit decision
            self.log_exit_decision(symbol, exit_decision, 'EXECUTED')
            
            # Update position management to mark as closed
            self.mark_position_closed(symbol)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute position exit for {position['symbol']}: {e}")
            self.log_exit_decision(position['symbol'], exit_decision, 'FAILED')
            return False

    def log_exit_decision(self, symbol: str, decision: Dict, status: str):
        """Log exit decision to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO exit_decisions 
                (symbol, decision_type, reason, current_price, profit_loss, 
                 confidence_score, execution_status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, 'EXIT', ', '.join(decision.get('reasons', ['Unknown'])),
                  decision.get('current_price', 0), decision.get('profit_loss', 0),
                  decision.get('confidence_score', 0), status))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log exit decision: {e}")

    def mark_position_closed(self, symbol: str):
        """Mark position as closed in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE position_management 
                SET exit_strategy = 'CLOSED', last_updated = CURRENT_TIMESTAMP
                WHERE symbol = ? AND exit_strategy = 'ACTIVE'
            ''', (symbol,))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to mark position closed: {e}")

    def generate_position_report(self) -> Dict:
        """Generate comprehensive position management report"""
        try:
            active_positions = self.get_active_positions()
            
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_positions': len(active_positions),
                'total_unrealized_pnl': sum(pos['unrealized_pnl'] for pos in active_positions),
                'position_analysis': [],
                'exit_recommendations': []
            }
            
            for position in active_positions:
                # Update tracking
                self.update_position_tracking(position)
                
                # Analyze exit conditions
                exit_analysis = self.analyze_exit_conditions(position)
                
                position_report = {
                    'symbol': position['symbol'],
                    'side': position['side'],
                    'unrealized_pnl': position['unrealized_pnl'],
                    'percentage': position['percentage'],
                    'should_exit': exit_analysis['should_exit'],
                    'exit_confidence': exit_analysis.get('confidence_score', 0),
                    'exit_reasons': exit_analysis.get('reasons', [])
                }
                
                report['position_analysis'].append(position_report)
                
                if exit_analysis['should_exit']:
                    report['exit_recommendations'].append({
                        'symbol': position['symbol'],
                        'action': 'CLOSE_POSITION',
                        'reasons': exit_analysis['reasons'],
                        'confidence': exit_analysis['confidence_score']
                    })
            
            return report
            
        except Exception as e:
            logger.error(f"Failed to generate position report: {e}")
            return {}

    def manage_positions(self):
        """Main position management loop"""
        try:
            logger.info("🔄 Running position management analysis...")
            
            active_positions = self.get_active_positions()
            
            if not active_positions:
                logger.info("📭 No active positions to manage")
                return
            
            logger.info(f"📊 Managing {len(active_positions)} active positions")
            
            exits_executed = 0
            
            for position in active_positions:
                # Update position tracking
                self.update_position_tracking(position)
                
                # Analyze exit conditions
                exit_decision = self.analyze_exit_conditions(position)
                
                logger.info(f"📈 {position['symbol']}: {position['side']} "
                           f"P&L: ${position['unrealized_pnl']:.2f} ({position['percentage']:.2f}%)")
                
                if exit_decision['should_exit']:
                    logger.info(f"🚨 Exit signal for {position['symbol']}: "
                               f"{', '.join(exit_decision['reasons'])} "
                               f"(Confidence: {exit_decision['confidence_score']:.1%})")
                    
                    # Execute exit if confidence is high enough
                    if exit_decision['confidence_score'] >= 0.8:
                        if self.execute_position_exit(position, exit_decision):
                            exits_executed += 1
                    else:
                        logger.info(f"⚠️ Low confidence exit signal - monitoring {position['symbol']}")
                        self.log_exit_decision(position['symbol'], exit_decision, 'MONITORED')
            
            if exits_executed > 0:
                logger.info(f"✅ Position management complete: {exits_executed} positions closed")
            else:
                logger.info("✅ Position management complete: All positions maintained")
                
        except Exception as e:
            logger.error(f"Position management error: {e}")

def main():
    """Main position management function"""
    manager = AdvancedPositionManager()
    
    logger.info("🚀 Starting Advanced Position Manager")
    logger.info("🎯 Intelligent exit strategies with trailing stops and profit protection")
    
    while True:
        try:
            manager.manage_positions()
            
            # Generate detailed report every 30 minutes
            if int(time.time()) % 1800 == 0:  # Every 30 minutes
                report = manager.generate_position_report()
                if report:
                    logger.info("📊 Position management report generated")
            
            logger.info("⏰ Next position analysis in 5 minutes...")
            time.sleep(300)  # Check every 5 minutes
            
        except KeyboardInterrupt:
            logger.info("🛑 Position manager stopped by user")
            break
        except Exception as e:
            logger.error(f"Manager error: {e}")
            time.sleep(300)

if __name__ == "__main__":
    main()