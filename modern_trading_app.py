"""
Professional Trading Platform - Modern UI
Port 5001 - Frontend redesign with 3Commas/TradingView inspired interface
"""

from flask import Flask, render_template, jsonify, request
import logging
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from real_data_service import RealDataService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'modern-trading-platform-2024'

# Initialize data service
data_service = RealDataService()

class ModernTradingInterface:
    def __init__(self):
        self.data_service = data_service
        
    def get_dashboard_data(self):
        """Get comprehensive dashboard data for modern UI"""
        try:
            # Portfolio data
            portfolio = self.data_service.get_real_portfolio_data()
            
            # AI performance
            ai_performance = self.data_service.get_real_ai_performance()
            
            # Risk metrics
            risk_metrics = self.data_service.get_real_risk_metrics()
            
            # Technical signals
            technical_signals = self.data_service.get_real_technical_signals()
            
            # System status
            system_status = self._get_system_status()
            
            return {
                'portfolio': portfolio,
                'ai_performance': ai_performance,
                'risk_metrics': risk_metrics,
                'technical_signals': technical_signals,
                'system_status': system_status,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Dashboard data error: {e}")
            return {
                'error': str(e),
                'portfolio': {'total_value': 0, 'daily_pnl': 0, 'positions': []},
                'system_status': {'status': 'error', 'message': str(e)}
            }
    
    def _get_system_status(self):
        """Get system operational status"""
        try:
            # Check database connectivity
            conn = sqlite3.connect('data/portfolio_tracking.db')
            conn.execute('SELECT COUNT(*) FROM portfolio_positions').fetchone()
            conn.close()
            
            return {
                'status': 'live',
                'message': 'All systems operational',
                'uptime': '24h 15m',
                'last_sync': datetime.now().strftime('%H:%M:%S')
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'System error: {str(e)}',
                'uptime': '0h 0m',
                'last_sync': 'Never'
            }

# Initialize trading interface
trading_interface = ModernTradingInterface()

# Routes
@app.route('/')
def dashboard():
    """Modern dashboard page"""
    return render_template('modern/dashboard.html')

@app.route('/portfolio')
def portfolio():
    """Portfolio management page"""
    return render_template('modern/portfolio.html')

@app.route('/strategy-builder')
def strategy_builder():
    """Strategy builder page"""
    return render_template('modern/strategy_builder.html')

@app.route('/analytics')
def analytics():
    """Analytics and performance page"""
    return render_template('modern/analytics.html')

@app.route('/ai-panel')
def ai_panel():
    """AI models and performance page"""
    return render_template('modern/ai_panel.html')

@app.route('/settings')
def settings():
    """Settings and configuration page"""
    return render_template('modern/settings.html')

# API Routes
@app.route('/api/dashboard-data')
def api_dashboard_data():
    """Get dashboard data"""
    data = trading_interface.get_dashboard_data()
    return jsonify(data)

@app.route('/api/portfolio-data')
def api_portfolio_data():
    """Get portfolio data"""
    try:
        portfolio_data = data_service.get_real_portfolio_data()
        return jsonify(portfolio_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/trading-signals')
def api_trading_signals():
    """Get trading signals"""
    try:
        signals = data_service.get_real_technical_signals()
        return jsonify(signals)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/ai-performance')
def api_ai_performance():
    """Get AI model performance"""
    try:
        performance = data_service.get_real_ai_performance()
        return jsonify(performance)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/risk-metrics')
def api_risk_metrics():
    """Get risk management metrics"""
    try:
        risk_data = data_service.get_real_risk_metrics()
        return jsonify(risk_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/market-data/<symbol>')
def api_market_data(symbol):
    """Get market data for specific symbol"""
    try:
        # Get OHLCV data from database
        conn = sqlite3.connect('data/market_data.db')
        query = '''
            SELECT timestamp, open_price, high_price, low_price, close_price, volume
            FROM ohlcv_data 
            WHERE symbol = ? 
            ORDER BY timestamp DESC 
            LIMIT 100
        '''
        df = pd.read_sql_query(query, conn, params=[symbol])
        conn.close()
        
        if df.empty:
            return jsonify({'error': f'No data available for {symbol}'})
        
        # Convert to TradingView format
        data = {
            'symbol': symbol,
            'data': df.to_dict('records'),
            'last_updated': datetime.now().isoformat()
        }
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

def create_app():
    """Application factory"""
    return app

if __name__ == '__main__':
    import os
    import time
    import socket
    
    # Find available port starting from 5000
    def find_port():
        for port in [5000, 8080, 8081, 8082, 3000, 3001]:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                result = sock.connect_ex(('127.0.0.1', port))
                if result != 0:
                    return port
            finally:
                sock.close()
        return 5000  # fallback
    
    port = find_port()
    logger.info(f"Starting Modern Trading Platform on port {port}")
    logger.info("Professional UI with 3Commas/TradingView design")
    
    # Ensure proper server startup
    try:
        logger.info("Starting Flask server for production deployment")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        # Try alternative port if primary fails
        if port == 5000:
            port = 8080
            logger.info(f"Retrying on port {port}")
            app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)
        else:
            raise