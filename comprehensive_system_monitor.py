#!/usr/bin/env python3
"""
Comprehensive System Monitor
Real-time monitoring and analytics for all trading systems with performance optimization
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

class ComprehensiveSystemMonitor:
    def __init__(self):
        self.exchange = None
        self.db_path = 'comprehensive_system_monitor.db'
        self.initialize_exchange()
        self.setup_database()
        
        # System components to monitor
        self.systems = {
            'live_futures_engine': 'advanced_futures_trading.db',
            'position_manager': 'advanced_position_management.db',
            'profit_optimizer': 'intelligent_profit_optimizer.db',
            'signal_executor': 'advanced_signal_executor.db',
            'under50_engine': 'live_under50_futures_trading.db'
        }

    def initialize_exchange(self):
        """Initialize OKX exchange for real-time data"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.environ.get('OKX_API_KEY'),
                'secret': os.environ.get('OKX_SECRET_KEY'),
                'password': os.environ.get('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            self.exchange.load_markets()
            logger.info("System monitor connected to OKX")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")

    def setup_database(self):
        """Setup comprehensive monitoring database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # System performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_balance REAL,
                    available_balance REAL,
                    total_positions INTEGER,
                    profitable_positions INTEGER,
                    total_unrealized_pnl REAL,
                    portfolio_percentage REAL,
                    daily_trades INTEGER,
                    system_efficiency REAL
                )
            ''')
            
            # Component health tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS component_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    component_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    last_activity TIMESTAMP,
                    trades_today INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0,
                    error_count INTEGER DEFAULT 0
                )
            ''')
            
            # Performance alerts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    alert_type TEXT NOT NULL,
                    component TEXT,
                    message TEXT NOT NULL,
                    severity TEXT DEFAULT 'INFO',
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Comprehensive system monitor database ready")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")

    def get_real_time_portfolio_status(self) -> Dict:
        """Get comprehensive real-time portfolio status"""
        try:
            # Account balance
            balance = self.exchange.fetch_balance()
            total_balance = float(balance['USDT']['total'])
            available_balance = float(balance['USDT']['free'])
            
            # Active positions
            positions = self.exchange.fetch_positions()
            active_positions = []
            total_unrealized_pnl = 0
            
            for position in positions:
                if position['contracts'] and float(position['contracts']) > 0:
                    unrealized_pnl = float(position.get('unrealizedPnl', 0))
                    total_unrealized_pnl += unrealized_pnl
                    
                    active_positions.append({
                        'symbol': position['symbol'],
                        'side': position['side'],
                        'size': float(position['contracts']),
                        'unrealized_pnl': unrealized_pnl,
                        'percentage': float(position.get('percentage', 0)),
                        'leverage': position.get('leverage', 1)
                    })
            
            profitable_positions = len([p for p in active_positions if p['unrealized_pnl'] > 0])
            portfolio_percentage = (total_unrealized_pnl / total_balance * 100) if total_balance > 0 else 0
            
            return {
                'timestamp': datetime.now().isoformat(),
                'total_balance': total_balance,
                'available_balance': available_balance,
                'total_positions': len(active_positions),
                'profitable_positions': profitable_positions,
                'losing_positions': len(active_positions) - profitable_positions,
                'total_unrealized_pnl': total_unrealized_pnl,
                'portfolio_percentage': portfolio_percentage,
                'positions': active_positions,
                'balance_utilization': ((total_balance - available_balance) / total_balance * 100) if total_balance > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio status: {e}")
            return {}

    def monitor_system_components(self) -> Dict:
        """Monitor health and performance of all system components"""
        component_status = {}
        
        for system_name, db_path in self.systems.items():
            try:
                status = self.analyze_component_health(system_name, db_path)
                component_status[system_name] = status
                
                # Save component health data
                self.save_component_health(system_name, status)
                
            except Exception as e:
                logger.error(f"Failed to monitor {system_name}: {e}")
                component_status[system_name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
        
        return component_status

    def analyze_component_health(self, system_name: str, db_path: str) -> Dict:
        """Analyze individual component health and performance"""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check for recent activity
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            status = {
                'status': 'ACTIVE',
                'last_activity': None,
                'trades_today': 0,
                'success_rate': 0,
                'error_count': 0
            }
            
            # Analyze different table structures for activity
            if 'futures_signals' in tables:
                cursor.execute('''
                    SELECT COUNT(*), MAX(timestamp) 
                    FROM futures_signals 
                    WHERE timestamp >= date('now')
                ''')
                result = cursor.fetchone()
                status['trades_today'] = result[0] if result[0] else 0
                status['last_activity'] = result[1] if result[1] else 'No recent activity'
                
            elif 'signal_executions' in tables:
                cursor.execute('''
                    SELECT COUNT(*), MAX(execution_time),
                           SUM(CASE WHEN status = 'EXECUTED' THEN 1 ELSE 0 END)
                    FROM signal_executions 
                    WHERE execution_time >= date('now')
                ''')
                result = cursor.fetchone()
                status['trades_today'] = result[0] if result[0] else 0
                status['last_activity'] = result[1] if result[1] else 'No recent activity'
                if result[0] and result[0] > 0:
                    status['success_rate'] = (result[2] / result[0]) * 100
                
            elif 'profit_decisions' in tables:
                cursor.execute('''
                    SELECT COUNT(*), MAX(execution_time) 
                    FROM profit_decisions 
                    WHERE execution_time >= date('now')
                ''')
                result = cursor.fetchone()
                status['trades_today'] = result[0] if result[0] else 0
                status['last_activity'] = result[1] if result[1] else 'Monitoring active'
                
            conn.close()
            
            # Determine overall status
            if status['last_activity'] and 'No recent' not in status['last_activity']:
                try:
                    last_time = datetime.fromisoformat(status['last_activity'].replace('Z', '+00:00'))
                    time_diff = datetime.now() - last_time.replace(tzinfo=None)
                    if time_diff > timedelta(hours=2):
                        status['status'] = 'IDLE'
                except:
                    pass
            else:
                status['status'] = 'IDLE'
            
            return status
            
        except Exception as e:
            return {
                'status': 'ERROR',
                'error': str(e),
                'trades_today': 0,
                'success_rate': 0
            }

    def save_component_health(self, component_name: str, status: Dict):
        """Save component health status to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO component_health 
                (component_name, status, last_activity, trades_today, success_rate, error_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (component_name, status['status'], status.get('last_activity', ''),
                  status.get('trades_today', 0), status.get('success_rate', 0),
                  status.get('error_count', 0)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save component health: {e}")

    def calculate_system_efficiency(self, portfolio_status: Dict, component_status: Dict) -> float:
        """Calculate overall system efficiency score"""
        try:
            # Base efficiency factors
            balance_score = min(portfolio_status.get('total_balance', 0) / 200, 1.0) * 20  # Max 20 points
            position_score = min(portfolio_status.get('total_positions', 0) / 10, 1.0) * 15  # Max 15 points
            
            # Profitability score
            profit_ratio = portfolio_status.get('profitable_positions', 0) / max(portfolio_status.get('total_positions', 1), 1)
            profitability_score = profit_ratio * 25  # Max 25 points
            
            # PnL performance score
            pnl_percentage = portfolio_status.get('portfolio_percentage', 0)
            pnl_score = max(0, min(pnl_percentage * 2, 20))  # Max 20 points for positive PnL
            
            # System activity score
            active_systems = len([s for s in component_status.values() if s.get('status') == 'ACTIVE'])
            activity_score = (active_systems / len(component_status)) * 20  # Max 20 points
            
            total_score = balance_score + position_score + profitability_score + pnl_score + activity_score
            return min(total_score, 100)  # Cap at 100%
            
        except Exception as e:
            logger.error(f"Failed to calculate system efficiency: {e}")
            return 0

    def generate_performance_alerts(self, portfolio_status: Dict, component_status: Dict) -> List[Dict]:
        """Generate performance alerts based on system analysis"""
        alerts = []
        
        try:
            # Portfolio alerts
            if portfolio_status.get('total_unrealized_pnl', 0) < -10:
                alerts.append({
                    'type': 'PORTFOLIO_LOSS',
                    'component': 'Portfolio',
                    'message': f"Portfolio showing ${portfolio_status['total_unrealized_pnl']:.2f} unrealized loss",
                    'severity': 'WARNING'
                })
            
            if portfolio_status.get('total_positions', 0) < 3:
                alerts.append({
                    'type': 'LOW_ACTIVITY',
                    'component': 'Portfolio',
                    'message': f"Only {portfolio_status['total_positions']} active positions - opportunity for growth",
                    'severity': 'INFO'
                })
            
            # Component alerts
            for component, status in component_status.items():
                if status.get('status') == 'ERROR':
                    alerts.append({
                        'type': 'COMPONENT_ERROR',
                        'component': component,
                        'message': f"{component} experiencing errors: {status.get('error', 'Unknown')}",
                        'severity': 'HIGH'
                    })
                elif status.get('status') == 'IDLE' and status.get('trades_today', 0) == 0:
                    alerts.append({
                        'type': 'COMPONENT_IDLE',
                        'component': component,
                        'message': f"{component} has been idle today - no trades executed",
                        'severity': 'INFO'
                    })
            
            # Efficiency alerts
            efficiency = self.calculate_system_efficiency(portfolio_status, component_status)
            if efficiency < 60:
                alerts.append({
                    'type': 'LOW_EFFICIENCY',
                    'component': 'System',
                    'message': f"System efficiency at {efficiency:.1f}% - optimization recommended",
                    'severity': 'WARNING'
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"Failed to generate alerts: {e}")
            return []

    def save_performance_snapshot(self, portfolio_status: Dict, efficiency: float):
        """Save performance snapshot to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO system_performance 
                (total_balance, available_balance, total_positions, profitable_positions,
                 total_unrealized_pnl, portfolio_percentage, system_efficiency)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (portfolio_status.get('total_balance', 0),
                  portfolio_status.get('available_balance', 0),
                  portfolio_status.get('total_positions', 0),
                  portfolio_status.get('profitable_positions', 0),
                  portfolio_status.get('total_unrealized_pnl', 0),
                  portfolio_status.get('portfolio_percentage', 0),
                  efficiency))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save performance snapshot: {e}")

    def run_comprehensive_monitoring_cycle(self):
        """Run complete system monitoring and analysis cycle"""
        try:
            logger.info("🔍 Running comprehensive system monitoring...")
            
            # Get real-time portfolio status
            portfolio_status = self.get_real_time_portfolio_status()
            
            if portfolio_status:
                logger.info(f"💰 Portfolio: ${portfolio_status['total_balance']:.2f} "
                           f"({portfolio_status['total_positions']} positions, "
                           f"${portfolio_status['total_unrealized_pnl']:.2f} P&L)")
            
            # Monitor all system components
            component_status = self.monitor_system_components()
            
            active_components = [name for name, status in component_status.items() 
                               if status.get('status') == 'ACTIVE']
            
            logger.info(f"🔧 System Components: {len(active_components)}/{len(component_status)} active")
            
            # Calculate system efficiency
            efficiency = self.calculate_system_efficiency(portfolio_status, component_status)
            logger.info(f"⚡ System Efficiency: {efficiency:.1f}%")
            
            # Generate and log alerts
            alerts = self.generate_performance_alerts(portfolio_status, component_status)
            
            if alerts:
                high_priority_alerts = [a for a in alerts if a['severity'] in ['HIGH', 'WARNING']]
                if high_priority_alerts:
                    logger.warning(f"⚠️ {len(high_priority_alerts)} high-priority alerts detected")
                    for alert in high_priority_alerts[:3]:  # Show top 3
                        logger.warning(f"   • {alert['message']}")
            
            # Save performance snapshot
            if portfolio_status:
                self.save_performance_snapshot(portfolio_status, efficiency)
            
            logger.info("✅ System monitoring cycle complete")
            
        except Exception as e:
            logger.error(f"Monitoring cycle error: {e}")

def main():
    """Main monitoring function"""
    monitor = ComprehensiveSystemMonitor()
    
    logger.info("🚀 Starting Comprehensive System Monitor")
    logger.info("📊 Real-time monitoring of all trading systems and portfolio performance")
    
    while True:
        try:
            monitor.run_comprehensive_monitoring_cycle()
            
            logger.info("⏰ Next monitoring cycle in 2 minutes...")
            time.sleep(120)  # Monitor every 2 minutes
            
        except KeyboardInterrupt:
            logger.info("🛑 System monitor stopped by user")
            break
        except Exception as e:
            logger.error(f"Monitor error: {e}")
            time.sleep(120)

if __name__ == "__main__":
    main()