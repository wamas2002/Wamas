#!/usr/bin/env python3
"""
Master Portfolio Dashboard
Comprehensive real-time portfolio analytics with live trading performance tracking
"""

from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import ccxt
import os
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'portfolio_dashboard_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

class MasterPortfolioAnalytics:
    def __init__(self):
        self.exchange = None
        self.initialize_exchange()
        
        # Database paths for all trading systems
        self.db_paths = {
            'live_trading': 'live_under50_futures_trading.db',
            'position_monitor': 'live_trading_positions.db',
            'position_manager': 'advanced_position_management.db',
            'profit_optimizer': 'intelligent_profit_optimizer.db'
        }

    def initialize_exchange(self):
        """Initialize OKX exchange connection"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.environ.get('OKX_API_KEY'),
                'secret': os.environ.get('OKX_SECRET_KEY'),
                'password': os.environ.get('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            self.exchange.load_markets()
            logger.info("Master portfolio analytics connected to OKX")
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")

    def get_portfolio_overview(self) -> Dict:
        """Get comprehensive portfolio overview"""
        try:
            # Get account balance
            balance = self.exchange.fetch_balance()
            total_balance = float(balance['USDT']['total'])
            
            # Get all positions
            positions = self.exchange.fetch_positions()
            active_positions = []
            total_unrealized_pnl = 0
            total_notional = 0
            
            for position in positions:
                if position['contracts'] and float(position['contracts']) > 0:
                    unrealized_pnl = float(position.get('unrealizedPnl', 0))
                    total_unrealized_pnl += unrealized_pnl
                    total_notional += float(position.get('notional', 0))
                    
                    active_positions.append({
                        'symbol': position['symbol'],
                        'side': position['side'],
                        'size': float(position['contracts']),
                        'entry_price': float(position['entryPrice']) if position['entryPrice'] else 0,
                        'mark_price': float(position['markPrice']) if position['markPrice'] else 0,
                        'unrealized_pnl': unrealized_pnl,
                        'percentage': float(position.get('percentage', 0)),
                        'leverage': position.get('leverage', 1),
                        'notional': float(position.get('notional', 0))
                    })
            
            # Calculate key metrics
            portfolio_percentage = (total_unrealized_pnl / total_balance * 100) if total_balance > 0 else 0
            exposure_ratio = (total_notional / total_balance) if total_balance > 0 else 0
            
            # Position distribution
            profitable_positions = len([p for p in active_positions if p['unrealized_pnl'] > 0])
            losing_positions = len([p for p in active_positions if p['unrealized_pnl'] < 0])
            
            return {
                'timestamp': datetime.now().isoformat(),
                'account_balance': total_balance,
                'total_positions': len(active_positions),
                'profitable_positions': profitable_positions,
                'losing_positions': losing_positions,
                'total_unrealized_pnl': total_unrealized_pnl,
                'portfolio_percentage': portfolio_percentage,
                'exposure_ratio': exposure_ratio,
                'positions': active_positions,
                'risk_level': self._assess_risk_level(exposure_ratio, portfolio_percentage)
            }
            
        except Exception as e:
            logger.error(f"Failed to get portfolio overview: {e}")
            return {}

    def get_trading_performance_metrics(self) -> Dict:
        """Calculate comprehensive trading performance metrics"""
        try:
            # Get recent trading history from databases
            live_trades = self._get_live_trading_history()
            position_history = self._get_position_history()
            profit_decisions = self._get_profit_decisions()
            
            # Calculate performance metrics
            total_trades = len(live_trades)
            successful_trades = len([t for t in live_trades if t.get('success', False)])
            win_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Calculate daily P&L
            daily_pnl = self._calculate_daily_pnl(position_history)
            
            # Calculate profit secured
            total_profit_secured = sum(d.get('profit_amount', 0) for d in profit_decisions)
            
            return {
                'total_trades_executed': total_trades,
                'successful_trades': successful_trades,
                'win_rate': win_rate,
                'daily_pnl': daily_pnl,
                'profit_secured': total_profit_secured,
                'active_strategies': self._count_active_strategies(),
                'system_uptime': self._calculate_system_uptime(),
                'last_trade_time': max([t.get('timestamp', '') for t in live_trades], default=''),
                'performance_grade': self._calculate_performance_grade(win_rate, daily_pnl)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate performance metrics: {e}")
            return {}

    def get_risk_analytics(self) -> Dict:
        """Calculate comprehensive risk analytics"""
        try:
            portfolio = self.get_portfolio_overview()
            
            # Calculate VaR and risk metrics
            positions = portfolio.get('positions', [])
            if not positions:
                return {'status': 'no_positions'}
            
            # Portfolio concentration risk
            position_values = [abs(p['notional']) for p in positions]
            total_value = sum(position_values)
            max_position_ratio = max(position_values) / total_value if total_value > 0 else 0
            
            # Leverage analysis
            leverages = [p['leverage'] for p in positions]
            avg_leverage = sum(leverages) / len(leverages) if leverages else 0
            max_leverage = max(leverages) if leverages else 0
            
            # Correlation risk (simplified)
            correlation_risk = self._estimate_correlation_risk(positions)
            
            # Drawdown analysis
            unrealized_pnls = [p['unrealized_pnl'] for p in positions]
            current_drawdown = min(unrealized_pnls) if unrealized_pnls else 0
            
            return {
                'max_position_concentration': max_position_ratio,
                'average_leverage': avg_leverage,
                'maximum_leverage': max_leverage,
                'correlation_risk_score': correlation_risk,
                'current_drawdown': current_drawdown,
                'diversification_score': len(set(p['symbol'] for p in positions)),
                'risk_score': self._calculate_composite_risk_score(
                    max_position_ratio, avg_leverage, correlation_risk, current_drawdown
                ),
                'recommendations': self._generate_risk_recommendations(
                    max_position_ratio, avg_leverage, correlation_risk
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate risk analytics: {e}")
            return {}

    def _get_live_trading_history(self) -> List[Dict]:
        """Get live trading history from database"""
        try:
            conn = sqlite3.connect(self.db_paths['live_trading'])
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM live_trades 
                WHERE timestamp >= date('now', '-7 days')
                ORDER BY timestamp DESC LIMIT 100
            ''')
            trades = []
            for row in cursor.fetchall():
                trades.append({
                    'timestamp': row[1] if len(row) > 1 else '',
                    'symbol': row[2] if len(row) > 2 else '',
                    'success': row[7] if len(row) > 7 else False
                })
            conn.close()
            return trades
        except Exception as e:
            logger.error(f"Failed to get live trading history: {e}")
            return []

    def _get_position_history(self) -> List[Dict]:
        """Get position history from monitoring database"""
        try:
            conn = sqlite3.connect(self.db_paths['position_monitor'])
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM position_tracking 
                WHERE last_updated >= datetime('now', '-24 hours')
                ORDER BY last_updated DESC
            ''')
            history = []
            for row in cursor.fetchall():
                history.append({
                    'symbol': row[1] if len(row) > 1 else '',
                    'unrealized_pnl': row[9] if len(row) > 9 else 0,
                    'timestamp': row[12] if len(row) > 12 else ''
                })
            conn.close()
            return history
        except Exception as e:
            logger.error(f"Failed to get position history: {e}")
            return []

    def _get_profit_decisions(self) -> List[Dict]:
        """Get profit-taking decisions from optimizer database"""
        try:
            conn = sqlite3.connect(self.db_paths['profit_optimizer'])
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM profit_decisions 
                WHERE execution_time >= datetime('now', '-24 hours')
                ORDER BY execution_time DESC
            ''')
            decisions = []
            for row in cursor.fetchall():
                decisions.append({
                    'symbol': row[1] if len(row) > 1 else '',
                    'profit_amount': row[3] if len(row) > 3 else 0,
                    'timestamp': row[7] if len(row) > 7 else ''
                })
            conn.close()
            return decisions
        except Exception as e:
            logger.error(f"Failed to get profit decisions: {e}")
            return []

    def _assess_risk_level(self, exposure_ratio: float, portfolio_percentage: float) -> str:
        """Assess overall portfolio risk level"""
        if exposure_ratio > 0.8 or abs(portfolio_percentage) > 10:
            return 'HIGH'
        elif exposure_ratio > 0.5 or abs(portfolio_percentage) > 5:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _calculate_daily_pnl(self, history: List[Dict]) -> float:
        """Calculate daily P&L from position history"""
        if not history:
            return 0
        
        # Get latest P&L values for each symbol
        latest_pnl = {}
        for record in history:
            symbol = record.get('symbol', '')
            pnl = record.get('unrealized_pnl', 0)
            if symbol and symbol not in latest_pnl:
                latest_pnl[symbol] = pnl
        
        return sum(latest_pnl.values())

    def _count_active_strategies(self) -> int:
        """Count number of active trading strategies"""
        active_count = 0
        try:
            # Check each database for recent activity
            for db_name, db_path in self.db_paths.items():
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    if cursor.fetchall():
                        active_count += 1
                    conn.close()
                except:
                    pass
        except:
            pass
        return active_count

    def _calculate_system_uptime(self) -> str:
        """Calculate system uptime based on oldest database entry"""
        try:
            oldest_timestamp = datetime.now()
            for db_path in self.db_paths.values():
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT MIN(timestamp) FROM sqlite_master")
                    result = cursor.fetchone()
                    if result and result[0]:
                        db_time = datetime.fromisoformat(result[0])
                        if db_time < oldest_timestamp:
                            oldest_timestamp = db_time
                    conn.close()
                except:
                    pass
            
            uptime = datetime.now() - oldest_timestamp
            return f"{uptime.days}d {uptime.seconds//3600}h"
        except:
            return "Unknown"

    def _calculate_performance_grade(self, win_rate: float, daily_pnl: float) -> str:
        """Calculate overall performance grade"""
        if win_rate >= 70 and daily_pnl > 0:
            return 'A+'
        elif win_rate >= 60 and daily_pnl >= 0:
            return 'A'
        elif win_rate >= 50:
            return 'B'
        elif win_rate >= 40:
            return 'C'
        else:
            return 'D'

    def _estimate_correlation_risk(self, positions: List[Dict]) -> float:
        """Estimate correlation risk based on position symbols"""
        if len(positions) <= 1:
            return 0
        
        # Simplified correlation estimation based on common crypto correlations
        btc_related = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
        defi_related = ['UNI', 'AAVE', 'SUSHI', 'CRV', 'LINK']
        gaming_related = ['MANA', 'SAND', 'AXS', 'ENJ']
        
        categories = [btc_related, defi_related, gaming_related]
        max_category_count = 0
        
        for category in categories:
            count = sum(1 for pos in positions if any(token in pos['symbol'] for token in category))
            max_category_count = max(max_category_count, count)
        
        return min(max_category_count / len(positions), 1.0)

    def _calculate_composite_risk_score(self, concentration: float, leverage: float, correlation: float, drawdown: float) -> float:
        """Calculate composite risk score (0-100)"""
        concentration_risk = min(concentration * 100, 30)
        leverage_risk = min((leverage - 1) * 10, 25)
        correlation_risk = correlation * 20
        drawdown_risk = min(abs(drawdown) * 2, 25)
        
        return min(concentration_risk + leverage_risk + correlation_risk + drawdown_risk, 100)

    def _generate_risk_recommendations(self, concentration: float, leverage: float, correlation: float) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if concentration > 0.4:
            recommendations.append("Consider reducing position concentration")
        if leverage > 2.5:
            recommendations.append("High leverage detected - monitor closely")
        if correlation > 0.6:
            recommendations.append("Portfolio may be over-concentrated in correlated assets")
        if not recommendations:
            recommendations.append("Risk levels are within acceptable ranges")
            
        return recommendations

# Initialize analytics
analytics = MasterPortfolioAnalytics()

@app.route('/')
def dashboard():
    """Render master portfolio dashboard"""
    return render_template('master_portfolio_dashboard.html')

@app.route('/api/portfolio/overview')
def portfolio_overview():
    """Get portfolio overview data"""
    return jsonify(analytics.get_portfolio_overview())

@app.route('/api/portfolio/performance')
def portfolio_performance():
    """Get trading performance metrics"""
    return jsonify(analytics.get_trading_performance_metrics())

@app.route('/api/portfolio/risk')
def portfolio_risk():
    """Get risk analytics"""
    return jsonify(analytics.get_risk_analytics())

@app.route('/api/portfolio/comprehensive')
def comprehensive_data():
    """Get all portfolio data in one call"""
    return jsonify({
        'overview': analytics.get_portfolio_overview(),
        'performance': analytics.get_trading_performance_metrics(),
        'risk': analytics.get_risk_analytics(),
        'timestamp': datetime.now().isoformat()
    })

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info("Client connected to master portfolio dashboard")
    emit('status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info("Client disconnected from master portfolio dashboard")

def background_updates():
    """Send periodic updates to connected clients"""
    while True:
        try:
            # Get comprehensive data
            data = {
                'overview': analytics.get_portfolio_overview(),
                'performance': analytics.get_trading_performance_metrics(),
                'risk': analytics.get_risk_analytics(),
                'timestamp': datetime.now().isoformat()
            }
            
            # Emit to all connected clients
            socketio.emit('portfolio_update', data)
            
            time.sleep(30)  # Update every 30 seconds
        except Exception as e:
            logger.error(f"Background update error: {e}")
            time.sleep(30)

if __name__ == '__main__':
    logger.info("🚀 Starting Master Portfolio Dashboard")
    logger.info("📊 Comprehensive real-time portfolio analytics")
    logger.info("🌐 Access: http://localhost:5000")
    
    # Start background updates in a separate thread
    update_thread = threading.Thread(target=background_updates, daemon=True)
    update_thread.start()
    
    # Start the Flask-SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)