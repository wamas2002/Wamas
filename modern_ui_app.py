"""
Modern Trading Platform UI - Complete Flask Interface
Comprehensive migration of all features to replace Streamlit interface
"""

from flask import Flask, render_template, jsonify, request, redirect, url_for
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import plotly.graph_objs as go
import plotly.utils

# Import all authentic data services
from real_data_service import real_data_service
from advanced_portfolio_analytics import AdvancedPortfolioAnalytics
from advanced_sentiment_analysis import AdvancedSentimentAnalyzer
from advanced_technical_analysis import AdvancedTechnicalAnalysis
from advanced_rebalancing_engine import AdvancedRebalancingEngine

app = Flask(__name__)
app.secret_key = 'trading_platform_2025'

# Initialize components
portfolio_analytics = AdvancedPortfolioAnalytics()
sentiment_analyzer = AdvancedSentimentAnalyzer()
technical_analyzer = AdvancedTechnicalAnalysis()
rebalancing_engine = AdvancedRebalancingEngine()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModernTradingUI:
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
        """Get advanced portfolio analytics"""
        try:
            analytics = portfolio_analytics.generate_comprehensive_analytics()
            return analytics
        except Exception as e:
            logger.error(f"Portfolio analytics error: {e}")
            return {'error': str(e)}
    
    def get_strategy_data(self):
        """Get strategy and AI model data"""
        try:
            conn = sqlite3.connect('data/ai_performance.db')
            
            # Get model performance
            models_query = """
                SELECT model_name, symbol, accuracy, precision_score, total_trades, win_rate
                FROM model_performance
                WHERE last_updated > datetime('now', '-24 hours')
                ORDER BY accuracy DESC
            """
            models_df = pd.read_sql_query(models_query, conn)
            
            # Get recent predictions
            predictions_query = """
                SELECT symbol, prediction, confidence, model_used, timestamp
                FROM ai_predictions
                WHERE timestamp > datetime('now', '-6 hours')
                ORDER BY timestamp DESC
                LIMIT 50
            """
            predictions_df = pd.read_sql_query(predictions_query, conn)
            
            conn.close()
            
            return {
                'model_performance': models_df.to_dict('records'),
                'recent_predictions': predictions_df.to_dict('records'),
                'strategy_stats': {
                    'active_models': len(models_df['model_name'].unique()),
                    'total_predictions': len(predictions_df),
                    'avg_accuracy': models_df['accuracy'].mean() if not models_df.empty else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Strategy data error: {e}")
            return {'error': str(e)}
    
    def get_alerts_data(self):
        """Get active alerts and notifications"""
        try:
            # Get portfolio alerts
            portfolio_data = self.data_service.get_real_portfolio_data()
            risk_metrics = self.data_service.get_real_risk_metrics()
            
            alerts = []
            
            # Concentration risk alert
            if risk_metrics['concentration_risk'] > 80:
                alerts.append({
                    'type': 'CRITICAL',
                    'message': f"Portfolio concentration risk: {risk_metrics['concentration_risk']:.1f}%",
                    'symbol': risk_metrics['largest_position'],
                    'timestamp': datetime.now().isoformat(),
                    'action': 'Rebalance portfolio immediately'
                })
            
            # Volatility alert
            if risk_metrics['portfolio_volatility'] > 70:
                alerts.append({
                    'type': 'WARNING',
                    'message': f"High portfolio volatility: {risk_metrics['portfolio_volatility']:.1f}%",
                    'symbol': 'Portfolio',
                    'timestamp': datetime.now().isoformat(),
                    'action': 'Review risk exposure'
                })
            
            # AI performance alerts
            ai_performance = self.data_service.get_real_ai_performance()
            if ai_performance['overall_accuracy'] < 60:
                alerts.append({
                    'type': 'WARNING',
                    'message': f"AI accuracy below threshold: {ai_performance['overall_accuracy']:.1f}%",
                    'symbol': 'AI Models',
                    'timestamp': datetime.now().isoformat(),
                    'action': 'Review model performance'
                })
            
            return {'alerts': alerts, 'alert_count': len(alerts)}
            
        except Exception as e:
            logger.error(f"Alerts data error: {e}")
            return {'error': str(e)}

# Initialize UI handler
trading_ui = ModernTradingUI()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    try:
        dashboard_data = trading_ui.get_dashboard_data()
        return render_template('dashboard.html', data=dashboard_data)
    except Exception as e:
        logger.error(f"Dashboard route error: {e}")
        return render_template('error.html', error=str(e))

@app.route('/portfolio')
def portfolio():
    """Portfolio overview page"""
    try:
        portfolio_data = trading_ui.data_service.get_real_portfolio_data()
        analytics_data = trading_ui.get_portfolio_analytics()
        
        return render_template('portfolio.html', 
                             portfolio=portfolio_data,
                             analytics=analytics_data)
    except Exception as e:
        logger.error(f"Portfolio route error: {e}")
        return render_template('error.html', error=str(e))

@app.route('/strategy')
def strategy():
    """Strategy builder and AI models page"""
    try:
        strategy_data = trading_ui.get_strategy_data()
        return render_template('strategy.html', data=strategy_data)
    except Exception as e:
        logger.error(f"Strategy route error: {e}")
        return render_template('error.html', error=str(e))

@app.route('/analytics')
def analytics():
    """Advanced analytics page"""
    try:
        # Fundamental analysis
        fundamental_data = trading_ui.data_service.get_real_fundamental_analysis()
        
        # Technical analysis
        technical_data = trading_ui.data_service.get_real_technical_signals()
        
        # Sentiment analysis
        sentiment_data = sentiment_analyzer.generate_comprehensive_sentiment('BTC')
        
        return render_template('analytics.html',
                             fundamental=fundamental_data,
                             technical=technical_data,
                             sentiment=sentiment_data)
    except Exception as e:
        logger.error(f"Analytics route error: {e}")
        return render_template('error.html', error=str(e))

@app.route('/risk-manager')
def risk_manager():
    """Risk management page"""
    try:
        risk_data = trading_ui.data_service.get_real_risk_metrics()
        rebalancing_data = rebalancing_engine.generate_smart_rebalancing_plan()
        
        return render_template('risk_manager.html',
                             risk=risk_data,
                             rebalancing=rebalancing_data)
    except Exception as e:
        logger.error(f"Risk manager route error: {e}")
        return render_template('error.html', error=str(e))

@app.route('/alerts')
def alerts():
    """Alerts and notifications page"""
    try:
        alerts_data = trading_ui.get_alerts_data()
        return render_template('alerts.html', data=alerts_data)
    except Exception as e:
        logger.error(f"Alerts route error: {e}")
        return render_template('error.html', error=str(e))

# API Routes for real-time data
@app.route('/api/dashboard')
def api_dashboard():
    """Dashboard data API"""
    return jsonify(trading_ui.get_dashboard_data())

@app.route('/api/portfolio')
def api_portfolio():
    """Portfolio data API"""
    try:
        portfolio_data = trading_ui.data_service.get_real_portfolio_data()
        return jsonify(portfolio_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/market-prices')
def api_market_prices():
    """Live market prices API"""
    try:
        symbols = request.args.getlist('symbols') or ['BTC', 'ETH', 'PI']
        prices = trading_ui.data_service.get_real_market_prices(symbols)
        return jsonify(prices)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/ai-performance')
def api_ai_performance():
    """AI model performance API"""
    try:
        performance = trading_ui.data_service.get_real_ai_performance()
        return jsonify(performance)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/risk-metrics')
def api_risk_metrics():
    """Risk metrics API"""
    try:
        risk_data = trading_ui.data_service.get_real_risk_metrics()
        return jsonify(risk_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/technical-signals')
def api_technical_signals():
    """Technical analysis signals API"""
    try:
        signals = trading_ui.data_service.get_real_technical_signals()
        return jsonify(signals)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/fundamental-analysis')
def api_fundamental_analysis():
    """Fundamental analysis API"""
    try:
        fundamental = trading_ui.data_service.get_real_fundamental_analysis()
        return jsonify(fundamental)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/alerts')
def api_alerts():
    """Active alerts API"""
    try:
        alerts_data = trading_ui.get_alerts_data()
        return jsonify(alerts_data)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/portfolio-chart')
def api_portfolio_chart():
    """Portfolio composition chart data"""
    try:
        portfolio_data = trading_ui.data_service.get_real_portfolio_data()
        
        if not portfolio_data['positions']:
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
            height=400
        )
        
        return jsonify(json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/performance-chart')
def api_performance_chart():
    """Portfolio performance chart data"""
    try:
        analytics = portfolio_analytics.generate_comprehensive_analytics()
        
        # Create performance chart
        returns_data = analytics.get('portfolio_returns', [])
        
        if not returns_data:
            return jsonify({'error': 'No performance data available'})
        
        dates = [datetime.now() - timedelta(days=i) for i in range(len(returns_data), 0, -1)]
        cumulative_returns = np.cumsum(returns_data)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_returns,
            mode='lines',
            name='Portfolio Returns',
            line=dict(color='#1f77b4', width=2)
        ))
        
        fig.update_layout(
            title="Portfolio Performance",
            xaxis_title="Date",
            yaxis_title="Cumulative Returns (%)",
            height=400
        )
        
        return jsonify(json.loads(plotly.utils.PlotlyJSONEncoder().encode(fig)))
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/health')
def health_check():
    """System health check"""
    try:
        # Validate data authenticity
        validation = trading_ui.data_service.validate_data_authenticity()
        
        health_status = {
            'status': 'healthy' if validation['overall_authentic'] else 'degraded',
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
    app.run(host='0.0.0.0', port=5001, debug=True)