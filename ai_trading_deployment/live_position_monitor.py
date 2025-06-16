#!/usr/bin/env python3
"""
Live Position Monitor
Real-time tracking of live futures positions with P&L analysis
"""

import ccxt
import os
import sqlite3
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LivePositionMonitor:
    def __init__(self):
        self.exchange = None
        self.db_path = 'live_trading_positions.db'
        self.initialize_exchange()
        self.setup_database()

    def initialize_exchange(self):
        """Initialize OKX exchange for live monitoring"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.environ.get('OKX_API_KEY'),
                'secret': os.environ.get('OKX_SECRET_KEY'),
                'password': os.environ.get('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            self.exchange.load_markets()
            logger.info("Live position monitor connected to OKX")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")

    def setup_database(self):
        """Setup position monitoring database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create position tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS position_tracking (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    position_id TEXT,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    position_size REAL NOT NULL,
                    leverage INTEGER NOT NULL,
                    entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    current_price REAL,
                    unrealized_pnl REAL,
                    percentage_change REAL,
                    status TEXT DEFAULT 'OPEN',
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Position monitoring database ready")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")

    def get_current_balance(self) -> float:
        """Get current USDT balance"""
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['USDT']['total'])
        except Exception as e:
            logger.error(f"Failed to fetch balance: {e}")
            return 0.0

    def get_live_positions(self) -> List[Dict]:
        """Get all live futures positions"""
        try:
            positions = self.exchange.fetch_positions()
            live_positions = []
            
            for position in positions:
                if position['contracts'] and float(position['contracts']) > 0:
                    live_positions.append({
                        'symbol': position['symbol'],
                        'side': position['side'],
                        'size': float(position['contracts']),
                        'entry_price': float(position['entryPrice']) if position['entryPrice'] else 0,
                        'mark_price': float(position['markPrice']) if position['markPrice'] else 0,
                        'unrealized_pnl': float(position['unrealizedPnl']) if position['unrealizedPnl'] else 0,
                        'percentage': float(position['percentage']) if position['percentage'] else 0,
                        'leverage': position.get('leverage', 1),
                        'notional': float(position['notional']) if position['notional'] else 0
                    })
            
            return live_positions
        except Exception as e:
            logger.error(f"Failed to fetch positions: {e}")
            return []

    def update_position_tracking(self, positions: List[Dict]):
        """Update position tracking in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for pos in positions:
                # Check if position exists
                cursor.execute('''
                    SELECT id FROM position_tracking 
                    WHERE symbol = ? AND status = 'OPEN'
                ''', (pos['symbol'],))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing position
                    cursor.execute('''
                        UPDATE position_tracking 
                        SET current_price = ?, unrealized_pnl = ?, 
                            percentage_change = ?, last_updated = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (pos['mark_price'], pos['unrealized_pnl'], 
                          pos['percentage'], existing[0]))
                else:
                    # Insert new position
                    cursor.execute('''
                        INSERT INTO position_tracking 
                        (symbol, side, entry_price, position_size, leverage, 
                         current_price, unrealized_pnl, percentage_change)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (pos['symbol'], pos['side'], pos['entry_price'], 
                          pos['size'], pos['leverage'], pos['mark_price'], 
                          pos['unrealized_pnl'], pos['percentage']))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to update position tracking: {e}")

    def calculate_portfolio_performance(self) -> Dict:
        """Calculate overall portfolio performance"""
        try:
            positions = self.get_live_positions()
            balance = self.get_current_balance()
            
            total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in positions)
            total_notional = sum(pos['notional'] for pos in positions)
            
            performance = {
                'current_balance': balance,
                'active_positions': len(positions),
                'total_unrealized_pnl': total_unrealized_pnl,
                'total_notional': total_notional,
                'portfolio_percentage': (total_unrealized_pnl / balance * 100) if balance > 0 else 0,
                'positions': positions,
                'timestamp': datetime.now().isoformat()
            }
            
            return performance
        except Exception as e:
            logger.error(f"Failed to calculate performance: {e}")
            return {}

    def get_position_history(self, hours: int = 24) -> List[Dict]:
        """Get position history from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            since_time = datetime.now() - timedelta(hours=hours)
            
            df = pd.read_sql_query('''
                SELECT * FROM position_tracking 
                WHERE last_updated >= ? 
                ORDER BY last_updated DESC
            ''', conn, params=(since_time,))
            
            conn.close()
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to get position history: {e}")
            return []

    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        try:
            performance = self.calculate_portfolio_performance()
            history = self.get_position_history(24)
            
            # Calculate daily statistics
            daily_pnl_changes = []
            for pos in history:
                if pos['unrealized_pnl']:
                    daily_pnl_changes.append(pos['unrealized_pnl'])
            
            report = {
                'current_performance': performance,
                'daily_statistics': {
                    'positions_tracked': len(history),
                    'avg_unrealized_pnl': sum(daily_pnl_changes) / len(daily_pnl_changes) if daily_pnl_changes else 0,
                    'max_unrealized_pnl': max(daily_pnl_changes) if daily_pnl_changes else 0,
                    'min_unrealized_pnl': min(daily_pnl_changes) if daily_pnl_changes else 0,
                },
                'risk_metrics': {
                    'exposure_ratio': (performance.get('total_notional', 0) / performance.get('current_balance', 1)) if performance.get('current_balance', 0) > 0 else 0,
                    'diversification': len(set(pos['symbol'] for pos in performance.get('positions', []))),
                    'average_leverage': sum(pos['leverage'] for pos in performance.get('positions', [])) / len(performance.get('positions', [])) if performance.get('positions') else 0
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return report
        except Exception as e:
            logger.error(f"Failed to generate performance report: {e}")
            return {}

    def monitor_positions(self):
        """Monitor positions and update tracking"""
        try:
            logger.info("🔄 Monitoring live positions...")
            
            # Get current positions
            positions = self.get_live_positions()
            
            if positions:
                logger.info(f"📊 Tracking {len(positions)} live positions")
                
                # Update database
                self.update_position_tracking(positions)
                
                # Generate performance summary
                performance = self.calculate_portfolio_performance()
                
                logger.info(f"💰 Portfolio Status:")
                logger.info(f"   Balance: ${performance.get('current_balance', 0):.2f}")
                logger.info(f"   Active Positions: {performance.get('active_positions', 0)}")
                logger.info(f"   Total P&L: ${performance.get('total_unrealized_pnl', 0):.2f}")
                logger.info(f"   Portfolio %: {performance.get('portfolio_percentage', 0):.3f}%")
                
                # Log individual positions
                for pos in positions:
                    pnl_color = "🟢" if pos['unrealized_pnl'] >= 0 else "🔴"
                    logger.info(f"   {pnl_color} {pos['symbol']}: {pos['side']} "
                               f"${pos['unrealized_pnl']:.2f} ({pos['percentage']:.2f}%)")
            else:
                logger.info("📭 No active positions found")
                
        except Exception as e:
            logger.error(f"Position monitoring failed: {e}")

def main():
    """Main monitoring function"""
    monitor = LivePositionMonitor()
    
    logger.info("🚀 Starting Live Position Monitor")
    logger.info("📈 Tracking real-time P&L and position performance")
    
    while True:
        try:
            monitor.monitor_positions()
            
            # Generate detailed report every 10 cycles
            if int(time.time()) % 600 == 0:  # Every 10 minutes
                report = monitor.generate_performance_report()
                if report:
                    logger.info("📊 Performance Report Generated")
            
            logger.info("⏰ Next update in 60 seconds...")
            time.sleep(60)  # Update every minute
            
        except KeyboardInterrupt:
            logger.info("🛑 Position monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()