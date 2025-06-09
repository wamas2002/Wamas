"""
Professional Trading Platform with Integrated TradingView Widgets
Real-time charts on every page using official TradingView widgets
"""

from flask import Flask, render_template, jsonify, request
import logging
import json
from datetime import datetime
from real_data_service import RealDataService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'tradingview-platform-2024'

# Initialize data service
data_service = RealDataService()

class TradingViewManager:
    """Manage TradingView widget configurations"""
    
    def __init__(self):
        self.default_symbols = {
            'BTC': 'OKX:BTCUSDT',
            'ETH': 'OKX:ETHUSDT',
            'BNB': 'OKX:BNBUSDT',
            'ADA': 'OKX:ADAUSDT',
            'SOL': 'OKX:SOLUSDT',
            'XRP': 'OKX:XRPUSDT',
            'DOT': 'OKX:DOTUSDT',
            'AVAX': 'OKX:AVAXUSDT',
            'PI': 'OKX:PIUSDT'
        }
        
    def get_widget_config(self, symbol='BTCUSDT', container_id='tradingview_widget', 
                         width='100%', height='500', theme='dark', studies=None):
        """Generate TradingView widget configuration"""
        
        # Convert symbol to TradingView format
        tv_symbol = self.convert_to_tv_symbol(symbol)
        
        config = {
            "symbol": tv_symbol,
            "interval": "15",
            "timezone": "Etc/UTC",
            "theme": theme,
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#f1f3f6",
            "enable_publishing": False,
            "withdateranges": True,
            "range": "YTD",
            "hide_side_toolbar": False,
            "allow_symbol_change": True,
            "save_image": False,
            "details": True,
            "hotlist": True,
            "calendar": True,
            "container_id": container_id,
            "width": width,
            "height": height
        }
        
        if studies:
            config["studies"] = studies
            
        return config
    
    def convert_to_tv_symbol(self, symbol):
        """Convert symbol to TradingView format"""
        if symbol.endswith('USDT'):
            base = symbol.replace('USDT', '')
            if base in self.default_symbols:
                return self.default_symbols[base]
        return f"OKX:{symbol}"
    
    def get_symbol_list(self):
        """Get available trading symbols"""
        return list(self.default_symbols.keys())

# Initialize TradingView manager
tv_manager = TradingViewManager()

@app.route('/')
def dashboard():
    """Dashboard with TradingView widget"""
    try:
        # Get dashboard data
        dashboard_data = data_service.get_real_portfolio_data()
        ai_data = data_service.get_real_ai_performance()
        
        # Get main symbol for chart
        main_symbol = 'BTCUSDT'
        positions = dashboard_data.get('positions', [])
        if positions:
            # Use largest position as main symbol
            largest_pos = max(positions, key=lambda x: x.get('current_value', 0))
            main_symbol = largest_pos.get('symbol', 'BTC') + 'USDT'
        
        # Configure TradingView widget
        widget_config = tv_manager.get_widget_config(
            symbol=main_symbol,
            container_id='dashboard_chart',
            height='400',
            studies=['RSI', 'MACD']
        )
        
        return render_template('tradingview/dashboard.html', 
                             widget_config=json.dumps(widget_config),
                             dashboard_data=dashboard_data,
                             ai_data=ai_data,
                             symbols=tv_manager.get_symbol_list())
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return render_template('tradingview/error.html', error=str(e))

@app.route('/portfolio')
def portfolio():
    """Portfolio page with charts for each holding"""
    try:
        portfolio_data = data_service.get_real_portfolio_data()
        positions = portfolio_data.get('positions', [])
        
        # Create widget configs for each position
        position_widgets = []
        for pos in positions:
            symbol = pos.get('symbol', 'BTC') + 'USDT'
            widget_config = tv_manager.get_widget_config(
                symbol=symbol,
                container_id=f"chart_{pos.get('symbol', 'BTC').lower()}",
                height='300'
            )
            position_widgets.append({
                'position': pos,
                'widget_config': json.dumps(widget_config)
            })
        
        return render_template('tradingview/portfolio.html',
                             portfolio_data=portfolio_data,
                             position_widgets=position_widgets)
    except Exception as e:
        logger.error(f"Portfolio error: {e}")
        return render_template('tradingview/error.html', error=str(e))

@app.route('/strategy')
def strategy_builder():
    """Strategy builder with TradingView integration"""
    try:
        # Default strategy symbol
        strategy_symbol = request.args.get('symbol', 'BTCUSDT')
        
        widget_config = tv_manager.get_widget_config(
            symbol=strategy_symbol,
            container_id='strategy_chart',
            height='500',
            studies=['RSI', 'EMA', 'BB']
        )
        
        return render_template('tradingview/strategy.html',
                             widget_config=json.dumps(widget_config),
                             current_symbol=strategy_symbol,
                             symbols=tv_manager.get_symbol_list())
    except Exception as e:
        logger.error(f"Strategy builder error: {e}")
        return render_template('tradingview/error.html', error=str(e))

@app.route('/analytics')
def analytics():
    """Analytics page with historical performance charts"""
    try:
        # Get performance data for multiple symbols
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        analytics_widgets = []
        
        for symbol in symbols:
            widget_config = tv_manager.get_widget_config(
                symbol=symbol,
                container_id=f"analytics_{symbol.lower()}",
                height='350'
            )
            analytics_widgets.append({
                'symbol': symbol,
                'widget_config': json.dumps(widget_config)
            })
        
        return render_template('tradingview/analytics.html',
                             analytics_widgets=analytics_widgets)
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return render_template('tradingview/error.html', error=str(e))

@app.route('/ai-panel')
def ai_panel():
    """AI panel with forecasting charts"""
    try:
        ai_data = data_service.get_real_ai_performance()
        technical_signals = data_service.get_real_technical_signals()
        
        # Get symbols with AI predictions
        ai_symbols = list(technical_signals.keys())
        ai_widgets = []
        
        for symbol in ai_symbols[:3]:  # Show top 3 AI-monitored symbols
            full_symbol = symbol + 'USDT' if not symbol.endswith('USDT') else symbol
            widget_config = tv_manager.get_widget_config(
                symbol=full_symbol,
                container_id=f"ai_{symbol.lower()}",
                height='350',
                studies=['RSI', 'MACD']
            )
            
            signal_data = technical_signals.get(symbol, {})
            ai_widgets.append({
                'symbol': symbol,
                'widget_config': json.dumps(widget_config),
                'signal': signal_data
            })
        
        return render_template('tradingview/ai_panel.html',
                             ai_data=ai_data,
                             ai_widgets=ai_widgets)
    except Exception as e:
        logger.error(f"AI panel error: {e}")
        return render_template('tradingview/error.html', error=str(e))

# API Endpoints
@app.route('/api/dashboard-data')
def api_dashboard_data():
    """Get dashboard data"""
    try:
        data = {
            'portfolio': data_service.get_real_portfolio_data(),
            'ai_performance': data_service.get_real_ai_performance(),
            'technical_signals': data_service.get_real_technical_signals(),
            'risk_metrics': data_service.get_real_risk_metrics(),
            'system_status': {
                'status': 'live',
                'last_sync': datetime.now().strftime('%H:%M:%S'),
                'message': 'All systems operational'
            },
            'last_updated': datetime.now().isoformat()
        }
        return jsonify(data)
    except Exception as e:
        logger.error(f"API dashboard data error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/symbol-data/<symbol>')
def api_symbol_data(symbol):
    """Get data for specific symbol"""
    try:
        technical_signals = data_service.get_real_technical_signals()
        symbol_data = technical_signals.get(symbol, {})
        
        return jsonify({
            'symbol': symbol,
            'data': symbol_data,
            'tv_symbol': tv_manager.convert_to_tv_symbol(symbol + 'USDT'),
            'last_updated': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"API symbol data error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/widget-config')
def api_widget_config():
    """Get TradingView widget configuration"""
    try:
        symbol = request.args.get('symbol', 'BTCUSDT')
        container_id = request.args.get('container_id', 'tradingview_widget')
        height = request.args.get('height', '500')
        studies = request.args.getlist('studies')
        
        config = tv_manager.get_widget_config(
            symbol=symbol,
            container_id=container_id,
            height=height,
            studies=studies if studies else None
        )
        
        return jsonify(config)
    except Exception as e:
        logger.error(f"API widget config error: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return render_template('tradingview/error.html', 
                         error='Page not found'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('tradingview/error.html', 
                         error='Internal server error'), 500

if __name__ == '__main__':
    logger.info("Starting Professional Trading Platform with TradingView Integration")
    logger.info("Real-time charts integrated across all pages")
    logger.info("Starting Flask server on port 5000")
    
    app.run(host='0.0.0.0', port=5000, debug=False)