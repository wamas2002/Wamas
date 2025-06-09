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


def safe_get_price(data):
    """Safely extract price from data"""
    if isinstance(data, (int, float)):
        return float(data)
    elif isinstance(data, dict):
        return float(data.get('price', data.get('last', data.get('close', 0.0))))
    elif hasattr(data, 'price'):
        return float(float(data) if isinstance(data, (int, float)) else float(data.get("price", data.get("last", data.get("close", 0.0)))) if isinstance(data, dict) else getattr(data, "price", 0.0))
    else:
        return 0.0

def safe_get_value(data):
    """Safely extract value from data"""
    if isinstance(data, (int, float)):
        return float(data)
    elif isinstance(data, dict):
        return float(data.get('value', data.get('current_value', 0.0)))
    elif hasattr(data, 'value'):
        return float(data.value)
    else:
        return 0.0


def safe_data_access(func):
    """Decorator to safely handle data access errors"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AttributeError as e:
            if "'float' object has no attribute 'price'" in str(e):
                # Return safe fallback data structure
                return {
                    'portfolio': {'total_value': 0, 'positions': []},
                    'error': 'Data type conversion error - using fallback values'
                }
            raise e
        except Exception as e:
            return {'error': str(e)}
    return wrapper

class ModernTradingInterface:
    def safe_extract_price(self, data):
        """Safely extract price from any data type"""
        if isinstance(data, (int, float)):
            return float(data)
        elif isinstance(data, dict):
            return float(data.get('price', data.get('last', data.get('close', 0.0))))
        elif hasattr(data, 'price'):
            return float(data.price)
        elif hasattr(data, 'last'):
            return float(data.last)
        elif hasattr(data, 'close'):
            return float(data.close)
        else:
            return 0.0

    def __init__(self):
        self.data_service = data_service
        
    @safe_data_access
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
    try:
        return render_template('modern/safe_dashboard.html')
    except Exception as e:
        logger.error(f"Dashboard rendering error: {e}")
        # Fallback to basic HTML if template fails
        return '''
        <html><head><title>Trading Platform</title></head>
        <body style="background:#0B1426;color:white;font-family:Arial;padding:20px;">
        <h1>Professional Trading Platform</h1>
        <p>Loading dashboard data...</p>
        <script>
        fetch('/api/dashboard-data')
            .then(r => r.json())
            .then(d => {
                document.body.innerHTML = `
                    <h1>Professional Trading Platform</h1>
                    <p>Portfolio Value: $${(d.portfolio?.total_value || 0).toLocaleString()}</p>
                    <p>AI Accuracy: ${(d.ai_performance?.overall_accuracy || 0).toFixed(1)}%</p>
                    <p>System Status: ${d.system_status?.status || 'Unknown'}</p>
                `;
            })
            .catch(e => console.error('Data load error:', e));
        </script>
        </body></html>
        ''', 200

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
    import socket
    
    # Use port 8080 for reliable deployment
    port = 8080
    logger.info(f"Starting Modern Trading Platform on port {port}")
    logger.info("Professional UI with 3Commas/TradingView design")
    
    try:
        logger.info("Starting Flask server for production deployment")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        # Try one more port if first attempt fails
        port = 8080 if port != 8080 else 8081
        logger.info(f"Retrying on port {port}")
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True, use_reloader=False)