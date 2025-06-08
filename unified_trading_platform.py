"""
Unified Trading Platform - Complete Flask Interface at Port 5001
Migrated system with all features integrated and authentic OKX data
"""

from flask import Flask, render_template, jsonify, request
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import plotly.graph_objs as go
import plotly.utils
import os

# Import only essential services
from real_data_service import real_data_service

app = Flask(__name__)
app.secret_key = 'intellectia_trading_2025'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedTradingPlatform:
    def __init__(self):
        self.data_service = real_data_service
        
    def get_dashboard_data(self):
        """Get comprehensive dashboard data"""
        try:
            # Portfolio data
            portfolio_data = self.data_service.get_real_portfolio_data()
            
            # Market data
            symbols = ['BTC', 'ETH', 'PI']
            market_prices = self.data_service.get_real_market_prices(symbols)
            
            # AI performance
            ai_performance = self.data_service.get_real_ai_performance()
            
            # Risk metrics
            risk_metrics = self.data_service.get_real_risk_metrics()
            
            # Technical signals
            technical_signals = self.data_service.get_real_technical_signals()
            
            return {
                'portfolio': portfolio_data,
                'market_prices': market_prices,
                'ai_performance': ai_performance,
                'risk_metrics': risk_metrics,
                'technical_signals': technical_signals,
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Dashboard data error: {e}")
            return {'error': str(e)}
    
    def get_portfolio_analytics(self):
        """Get portfolio analytics with authentic data"""
        try:
            conn = sqlite3.connect('data/portfolio_tracking.db')
            
            # Get portfolio positions
            positions_query = """
                SELECT symbol, quantity, current_value, percentage_of_portfolio 
                FROM positions 
                WHERE current_value > 0
                ORDER BY current_value DESC
            """
            
            try:
                positions_df = pd.read_sql_query(positions_query, conn)
                total_value = positions_df['current_value'].sum()
                
                analytics = {
                    'total_value': total_value,
                    'positions': positions_df.to_dict('records'),
                    'concentration_risk': positions_df['percentage_of_portfolio'].max(),
                    'diversification_score': 100 - positions_df['percentage_of_portfolio'].max(),
                    'risk_level': 'High' if positions_df['percentage_of_portfolio'].max() > 80 else 'Moderate'
                }
            except:
                # Use authenticated OKX portfolio data
                portfolio_data = self.data_service.get_real_portfolio_data()
                analytics = {
                    'total_value': portfolio_data.get('total_value', 0),
                    'positions': portfolio_data.get('positions', []),
                    'concentration_risk': portfolio_data.get('concentration_risk', 0),
                    'diversification_score': 100 - portfolio_data.get('concentration_risk', 0),
                    'risk_level': portfolio_data.get('risk_level', 'Unknown')
                }
            
            conn.close()
            return analytics
            
        except Exception as e:
            logger.error(f"Portfolio analytics error: {e}")
            return {'error': str(e)}
    
    def get_ai_models_status(self):
        """Get AI model performance from authenticated sources"""
        try:
            ai_performance = self.data_service.get_real_ai_performance()
            
            models_status = {
                'overall_accuracy': ai_performance.get('overall_accuracy', 0),
                'total_predictions': ai_performance.get('total_predictions', 0),
                'model_performance': ai_performance.get('model_performance', []),
                'active_models': len(ai_performance.get('model_performance', [])),
                'last_updated': datetime.now().isoformat()
            }
            
            return models_status
            
        except Exception as e:
            logger.error(f"AI models status error: {e}")
            return {'error': str(e)}

# Initialize platform
trading_platform = UnifiedTradingPlatform()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    try:
        dashboard_data = trading_platform.get_dashboard_data()
        return render_template('dashboard.html', data=dashboard_data)
    except Exception as e:
        logger.error(f"Dashboard route error: {e}")
        return render_template('error.html', error=str(e))

@app.route('/portfolio')
def portfolio():
    """Portfolio overview page"""
    try:
        portfolio_data = trading_platform.data_service.get_real_portfolio_data()
        analytics_data = trading_platform.get_portfolio_analytics()
        
        return render_template('portfolio.html', 
                             portfolio=portfolio_data,
                             analytics=analytics_data)
    except Exception as e:
        logger.error(f"Portfolio route error: {e}")
        return render_template('error.html', error=str(e))

@app.route('/analytics')
def analytics():
    """Advanced analytics page"""
    try:
        # Fundamental analysis
        fundamental_data = trading_platform.data_service.get_real_fundamental_analysis()
        
        # Technical analysis
        technical_data = trading_platform.data_service.get_real_technical_signals()
        
        return render_template('analytics.html',
                             fundamental=fundamental_data,
                             technical=technical_data)
    except Exception as e:
        logger.error(f"Analytics route error: {e}")
        return render_template('error.html', error=str(e))

@app.route('/ai-strategy')
def ai_strategy():
    """AI strategy and models page"""
    try:
        ai_status = trading_platform.get_ai_models_status()
        return render_template('ai_strategy.html', data=ai_status)
    except Exception as e:
        logger.error(f"AI strategy route error: {e}")
        return render_template('error.html', error=str(e))

@app.route('/risk-manager')
def risk_manager():
    """Risk management page"""
    try:
        risk_data = trading_platform.data_service.get_real_risk_metrics()
        portfolio_data = trading_platform.data_service.get_real_portfolio_data()
        
        return render_template('risk_manager.html',
                             risk=risk_data,
                             portfolio=portfolio_data)
    except Exception as e:
        logger.error(f"Risk manager route error: {e}")
        return render_template('error.html', error=str(e))

@app.route('/alerts')
def alerts():
    """Alerts and notifications page"""
    try:
        # Get portfolio alerts based on real data
        portfolio_data = trading_platform.data_service.get_real_portfolio_data()
        risk_metrics = trading_platform.data_service.get_real_risk_metrics()
        
        alerts = []
        
        # Concentration risk alert
        if risk_metrics.get('concentration_risk', 0) > 80:
            alerts.append({
                'type': 'CRITICAL',
                'message': f"Portfolio concentration risk: {risk_metrics['concentration_risk']:.1f}%",
                'symbol': risk_metrics.get('largest_position', 'Unknown'),
                'timestamp': datetime.now().isoformat(),
                'action': 'Rebalance portfolio immediately'
            })
        
        alerts_data = {'alerts': alerts, 'alert_count': len(alerts)}
        return render_template('alerts.html', data=alerts_data)
    except Exception as e:
        logger.error(f"Alerts route error: {e}")
        return render_template('error.html', error=str(e))

# API Routes for real-time data
@app.route('/api/dashboard')
def api_dashboard():
    """Dashboard data API"""
    return jsonify(trading_platform.get_dashboard_data())

@app.route('/api/portfolio')
def api_portfolio():
    """Portfolio data API"""
    try:
        portfolio_data = trading_platform.data_service.get_real_portfolio_data()
        return jsonify(portfolio_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/market-prices')
def api_market_prices():
    """Live market prices API"""
    try:
        symbols = request.args.getlist('symbols') or ['BTC', 'ETH', 'PI']
        prices = trading_platform.data_service.get_real_market_prices(symbols)
        return jsonify(prices)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/ai-performance')
def api_ai_performance():
    """AI model performance API"""
    try:
        performance = trading_platform.data_service.get_real_ai_performance()
        return jsonify(performance)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/risk-metrics')
def api_risk_metrics():
    """Risk metrics API"""
    try:
        risk_data = trading_platform.data_service.get_real_risk_metrics()
        return jsonify(risk_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/technical-signals')
def api_technical_signals():
    """Technical analysis signals API"""
    try:
        signals = trading_platform.data_service.get_real_technical_signals()
        return jsonify(signals)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/fundamental-analysis')
def api_fundamental_analysis():
    """Fundamental analysis API"""
    try:
        fundamental = trading_platform.data_service.get_real_fundamental_analysis()
        return jsonify(fundamental)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/portfolio-chart')
def api_portfolio_chart():
    """Portfolio composition chart data"""
    try:
        portfolio_data = trading_platform.data_service.get_real_portfolio_data()
        
        if not portfolio_data.get('positions'):
            return jsonify({'error': 'No portfolio positions'})
        
        # Create pie chart data
        labels = [pos['symbol'] for pos in portfolio_data['positions']]
        values = [pos['current_value'] for pos in portfolio_data['positions']]
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            textposition='auto'
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            showlegend=True,
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return jsonify(json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health_check():
    """System health check"""
    try:
        # Validate data authenticity
        validation = trading_platform.data_service.validate_data_authenticity()
        
        health_status = {
            'status': 'healthy' if validation.get('overall_authentic', False) else 'degraded',
            'timestamp': datetime.now().isoformat(),
            'data_authenticity': validation,
            'services': {
                'portfolio_data': 'operational',
                'market_data': 'operational',
                'ai_models': 'operational',
                'risk_engine': 'operational'
            }
        }
        
        return jsonify(health_status)
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'timestamp': datetime.now().isoformat(),
            'error': str(e)
        })

if __name__ == '__main__':
    logger.info("Starting Unified Trading Platform on port 5001")
    logger.info("All features migrated from Streamlit to Flask interface")
    app.run(host='0.0.0.0', port=5001, debug=True)