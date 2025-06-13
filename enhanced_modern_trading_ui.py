"""
Enhanced Modern Trading Interface
Advanced UI with futures trading, portfolio optimization, and real-time analytics
"""

from flask import Flask, render_template, jsonify, request
import os
import ccxt
import pandas as pd
import numpy as np
import sqlite3
import json
import logging
from datetime import datetime, timedelta
import plotly.graph_objs as go
import plotly.utils
from typing import Dict, List, Optional
import pandas_ta as ta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class EnhancedTradingInterface:
    def __init__(self):
        self.exchange = None
        self.db_path = 'enhanced_trading.db'
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
            logger.info("Enhanced interface connected to OKX")
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
    
    def setup_database(self):
        """Setup enhanced database schema"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enhanced signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS enhanced_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    side TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    technical_score REAL,
                    ai_score REAL,
                    futures_score REAL,
                    current_price REAL,
                    target_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    recommended_leverage REAL,
                    risk_level TEXT,
                    timeframe TEXT,
                    volatility REAL,
                    volume_score REAL,
                    momentum REAL,
                    trend_strength REAL,
                    support_level REAL,
                    resistance_level REAL,
                    entry_reasons TEXT,
                    market_conditions TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Portfolio optimization table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_optimization (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    optimization_type TEXT NOT NULL,
                    current_allocation TEXT,
                    optimized_allocation TEXT,
                    expected_return REAL,
                    risk_level REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    optimization_score REAL,
                    rebalancing_needed BOOLEAN,
                    recommendations TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Performance analytics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    metric_value REAL,
                    benchmark_value REAL,
                    percentile_rank REAL,
                    trend_direction TEXT,
                    analysis_period TEXT,
                    detailed_breakdown TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Enhanced database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_enhanced_portfolio_data(self) -> List[Dict]:
        """Get comprehensive portfolio data with analytics"""
        try:
            balance = self.exchange.fetch_balance()
            portfolio_data = []
            
            for asset, data in balance.items():
                if data['total'] > 0 and asset != 'info':
                    try:
                        # Get current price
                        if asset == 'USDT':
                            price = 1.0
                            change_24h = 0.0
                        else:
                            symbol = f"{asset}/USDT"
                            ticker = self.exchange.fetch_ticker(symbol)
                            price = ticker['last']
                            change_24h = ticker['percentage'] or 0
                        
                        value_usd = data['total'] * price
                        
                        # Calculate additional metrics
                        portfolio_data.append({
                            'asset': asset,
                            'balance': data['total'],
                            'price': price,
                            'value_usd': value_usd,
                            'change_24h': change_24h,
                            'free': data['free'],
                            'used': data['used'],
                            'allocation_pct': 0  # Will be calculated later
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing {asset}: {e}")
                        continue
            
            # Calculate allocation percentages
            total_value = sum([item['value_usd'] for item in portfolio_data])
            for item in portfolio_data:
                item['allocation_pct'] = (item['value_usd'] / total_value * 100) if total_value > 0 else 0
            
            return sorted(portfolio_data, key=lambda x: x['value_usd'], reverse=True)
            
        except Exception as e:
            logger.error(f"Portfolio data fetch failed: {e}")
            return []
    
    def get_futures_signals(self) -> List[Dict]:
        """Get futures trading signals from database"""
        try:
            conn = sqlite3.connect('futures_trading.db')
            query = '''
                SELECT * FROM futures_signals 
                WHERE timestamp > datetime('now', '-1 hour')
                ORDER BY confidence DESC, timestamp DESC
                LIMIT 20
            '''
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df.to_dict('records') if not df.empty else []
            
        except Exception as e:
            logger.error(f"Futures signals fetch failed: {e}")
            return []
    
    def get_spot_signals(self) -> List[Dict]:
        """Get spot trading signals from autonomous engine"""
        try:
            conn = sqlite3.connect('autonomous_trading.db')
            query = '''
                SELECT symbol, signal, confidence, technical_score, ai_score, 
                       current_price, rsi, volume_ratio, entry_reasons, timestamp
                FROM trading_signals 
                WHERE timestamp > datetime('now', '-1 hour')
                ORDER BY confidence DESC, timestamp DESC
                LIMIT 20
            '''
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            signals = []
            for _, row in df.iterrows():
                signals.append({
                    'symbol': row['symbol'],
                    'signal_type': 'SPOT',
                    'side': row['signal'],
                    'confidence': row['confidence'],
                    'technical_score': row['technical_score'],
                    'ai_score': row['ai_score'],
                    'current_price': row['current_price'],
                    'rsi': row['rsi'],
                    'volume_ratio': row['volume_ratio'],
                    'entry_reasons': row['entry_reasons'],
                    'timestamp': row['timestamp']
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"Spot signals fetch failed: {e}")
            return []
    
    def get_combined_signals(self) -> List[Dict]:
        """Combine futures and spot signals"""
        futures_signals = self.get_futures_signals()
        spot_signals = self.get_spot_signals()
        
        # Add signal type identifier
        for signal in futures_signals:
            signal['signal_type'] = 'FUTURES'
        
        for signal in spot_signals:
            signal['signal_type'] = 'SPOT'
        
        # Combine and sort by confidence
        all_signals = futures_signals + spot_signals
        return sorted(all_signals, key=lambda x: x['confidence'], reverse=True)
    
    def calculate_portfolio_metrics(self, portfolio_data: List[Dict]) -> Dict:
        """Calculate comprehensive portfolio metrics"""
        try:
            total_value = sum([item['value_usd'] for item in portfolio_data])
            total_change_24h = sum([item['value_usd'] * item['change_24h'] / 100 for item in portfolio_data])
            avg_change_24h = (total_change_24h / total_value * 100) if total_value > 0 else 0
            
            # Diversity metrics
            num_assets = len([item for item in portfolio_data if item['value_usd'] > 1])
            
            # Concentration risk (Herfindahl Index)
            concentrations = [item['allocation_pct'] / 100 for item in portfolio_data]
            hhi = sum([c**2 for c in concentrations])
            diversity_score = (1 - hhi) * 100
            
            # Risk assessment
            high_risk_allocation = sum([item['allocation_pct'] for item in portfolio_data 
                                     if abs(item['change_24h']) > 5])
            
            return {
                'total_value': total_value,
                'change_24h_pct': avg_change_24h,
                'change_24h_usd': total_change_24h,
                'num_assets': num_assets,
                'diversity_score': diversity_score,
                'concentration_risk': 'HIGH' if hhi > 0.5 else 'MEDIUM' if hhi > 0.25 else 'LOW',
                'high_risk_allocation': high_risk_allocation,
                'risk_level': 'HIGH' if high_risk_allocation > 50 else 'MEDIUM' if high_risk_allocation > 25 else 'LOW'
            }
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation failed: {e}")
            return {}
    
    def get_market_overview(self) -> Dict:
        """Get comprehensive market overview"""
        try:
            major_pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'ADA/USDT']
            market_data = []
            
            for symbol in major_pairs:
                try:
                    ticker = self.exchange.fetch_ticker(symbol)
                    market_data.append({
                        'symbol': symbol,
                        'price': ticker['last'],
                        'change_24h': ticker['percentage'] or 0,
                        'volume_24h': ticker['quoteVolume'] or 0,
                        'high_24h': ticker['high'],
                        'low_24h': ticker['low']
                    })
                except Exception as e:
                    logger.error(f"Failed to fetch {symbol}: {e}")
                    continue
            
            # Calculate market sentiment
            positive_movers = len([item for item in market_data if item['change_24h'] > 0])
            total_movers = len(market_data)
            market_sentiment = 'BULLISH' if positive_movers >= total_movers * 0.7 else \
                             'BEARISH' if positive_movers <= total_movers * 0.3 else 'NEUTRAL'
            
            # Average market change
            avg_change = sum([item['change_24h'] for item in market_data]) / len(market_data) if market_data else 0
            
            return {
                'market_data': market_data,
                'sentiment': market_sentiment,
                'avg_change_24h': avg_change,
                'positive_movers': positive_movers,
                'total_pairs': total_movers
            }
            
        except Exception as e:
            logger.error(f"Market overview failed: {e}")
            return {'market_data': [], 'sentiment': 'UNKNOWN', 'avg_change_24h': 0}

# Initialize enhanced interface
enhanced_interface = EnhancedTradingInterface()

@app.route('/')
def index():
    """Main dashboard with futures and spot trading"""
    return render_template('enhanced_modern_dashboard.html')

@app.route('/api/enhanced/portfolio')
def api_portfolio():
    """Enhanced portfolio data with analytics"""
    try:
        portfolio_data = enhanced_interface.get_enhanced_portfolio_data()
        if not portfolio_data:
            portfolio_data = []
            
        metrics = enhanced_interface.calculate_portfolio_metrics(portfolio_data)
        if not metrics:
            metrics = {
                'total_value': 0,
                'change_24h_pct': 0,
                'change_24h_usd': 0,
                'num_assets': 0,
                'diversity_score': 0,
                'concentration_risk': 'LOW',
                'high_risk_allocation': 0,
                'risk_level': 'LOW'
            }
        
        return jsonify({
            'portfolio': portfolio_data,
            'metrics': metrics,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Portfolio API error: {e}")
        # Return empty but valid structure on error
        return jsonify({
            'portfolio': [],
            'metrics': {
                'total_value': 0,
                'change_24h_pct': 0,
                'change_24h_usd': 0,
                'num_assets': 0,
                'diversity_score': 0,
                'concentration_risk': 'LOW',
                'high_risk_allocation': 0,
                'risk_level': 'LOW'
            },
            'status': 'success'
        })

@app.route('/api/enhanced/signals')
def api_signals():
    """Combined futures and spot signals"""
    try:
        signals = enhanced_interface.get_combined_signals()
        
        return jsonify({
            'signals': signals,
            'total_count': len(signals),
            'futures_count': len([s for s in signals if s['signal_type'] == 'FUTURES']),
            'spot_count': len([s for s in signals if s['signal_type'] == 'SPOT']),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Signals API error: {e}")
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/enhanced/market')
def api_market():
    """Market overview and sentiment"""
    try:
        market_data = enhanced_interface.get_market_overview()
        
        return jsonify({
            'market': market_data,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Market API error: {e}")
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/enhanced/analytics')
def api_analytics():
    """Advanced analytics and performance metrics"""
    try:
        # Get trading performance from databases
        performance_data = {
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'avg_daily_return': 0,
            'volatility': 0,
            'best_performer': 'N/A',
            'worst_performer': 'N/A'
        }
        
        try:
            # Get autonomous trading performance
            conn = sqlite3.connect('autonomous_trading.db')
            cursor = conn.cursor()
            
            # Count total trades
            cursor.execute("SELECT COUNT(*) FROM autonomous_trades")
            performance_data['total_trades'] = cursor.fetchone()[0]
            
            # Calculate win rate
            cursor.execute("""
                SELECT 
                    COUNT(CASE WHEN pnl_usd > 0 THEN 1 END) as wins,
                    COUNT(*) as total
                FROM autonomous_trades 
                WHERE status = 'CLOSED'
            """)
            result = cursor.fetchone()
            if result and result[1] > 0:
                performance_data['win_rate'] = (result[0] / result[1]) * 100
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Performance calculation error: {e}")
        
        return jsonify({
            'performance': performance_data,
            'timestamp': datetime.now().isoformat(),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Analytics API error: {e}")
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/enhanced/optimization')
def api_optimization():
    """Portfolio optimization recommendations"""
    try:
        portfolio_data = enhanced_interface.get_enhanced_portfolio_data()
        
        # Simple portfolio optimization logic
        recommendations = []
        
        # Check for overconcentration
        for asset in portfolio_data:
            if asset['allocation_pct'] > 40:
                recommendations.append({
                    'type': 'REBALANCE',
                    'asset': asset['asset'],
                    'current_allocation': asset['allocation_pct'],
                    'recommended_allocation': 25,
                    'reason': 'Overconcentration risk',
                    'priority': 'HIGH'
                })
        
        # Check for underperformers
        for asset in portfolio_data:
            if asset['change_24h'] < -10 and asset['allocation_pct'] > 10:
                recommendations.append({
                    'type': 'REVIEW',
                    'asset': asset['asset'],
                    'current_performance': asset['change_24h'],
                    'reason': 'Significant underperformance',
                    'priority': 'MEDIUM'
                })
        
        return jsonify({
            'recommendations': recommendations,
            'optimization_score': 85 - len(recommendations) * 10,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Optimization API error: {e}")
        return jsonify({'error': str(e), 'status': 'error'})

if __name__ == '__main__':
    logger.info("Starting Enhanced Modern Trading Interface")
    app.run(host='0.0.0.0', port=5002, debug=False)