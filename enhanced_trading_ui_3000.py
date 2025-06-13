"""
Enhanced Trading UI on Port 3000
Modern Flask application with advanced analytics, signal attribution, and self-improving AI
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

# Import plugins with error handling
import sys
import os
sys.path.append('plugins')
sys.path.append('ai')

# Initialize plugin components
attribution_engine = None
volatility_risk_controller = None
model_evaluator = None

# Try to import plugins safely
try:
    if os.path.exists('plugins/signal_attribution_engine.py'):
        from plugins.signal_attribution_engine import SignalAttributionEngine
        attribution_engine = SignalAttributionEngine()
except Exception as e:
    logger.warning(f"Signal attribution engine not available: {e}")

try:
    if os.path.exists('plugins/volatility_risk_controller.py'):
        from plugins.volatility_risk_controller import VolatilityRiskController
        volatility_risk_controller = VolatilityRiskController()
except Exception as e:
    logger.warning(f"Volatility risk controller not available: {e}")

try:
    if os.path.exists('ai/model_evaluator.py'):
        from ai.model_evaluator import AIModelEvaluator
        model_evaluator = AIModelEvaluator()
except Exception as e:
    logger.warning(f"Model evaluator not available: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class EnhancedTradingUI:
    def __init__(self):
        self.exchange = None
        self.db_path = 'enhanced_ui.db'
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
            logger.info("Enhanced UI connected to OKX")
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
    
    def setup_database(self):
        """Setup enhanced UI database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # UI interactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ui_interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_action TEXT NOT NULL,
                    page TEXT NOT NULL,
                    interaction_data TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Dashboard metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dashboard_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_type TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Enhanced UI database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_combined_signals_data(self) -> List[Dict]:
        """Get combined signals from all trading engines with attribution"""
        signals = []
        
        try:
            # Get futures signals
            conn = sqlite3.connect('futures_trading.db')
            query = '''
                SELECT symbol, signal, confidence, technical_score, ai_score, 
                       current_price, rsi, volume_ratio, recommended_leverage,
                       risk_level, timestamp
                FROM futures_signals 
                WHERE timestamp > datetime('now', '-2 hours')
                ORDER BY confidence DESC
                LIMIT 20
            '''
            df_futures = pd.read_sql_query(query, conn)
            conn.close()
            
            for _, row in df_futures.iterrows():
                signal_data = {
                    'symbol': row['symbol'],
                    'signal': row['signal'],
                    'confidence': row['confidence'],
                    'technical_score': row.get('technical_score', 0),
                    'ai_score': row.get('ai_score', 0),
                    'current_price': row['current_price'],
                    'rsi': row.get('rsi', 0),
                    'volume_ratio': row.get('volume_ratio', 0),
                    'timestamp': row['timestamp'],
                    'signal_type': 'FUTURES',
                    'leverage': row.get('recommended_leverage', 1),
                    'risk_level': row.get('risk_level', 'MEDIUM')
                }
                
                # Add signal attribution
                attribution_id = attribution_engine.log_signal_origin(signal_data)
                signal_data['attribution_id'] = attribution_id
                
                # Apply risk controls
                enhanced_signal = volatility_risk_controller.apply_risk_controls(signal_data)
                signals.append(enhanced_signal)
            
        except Exception as e:
            logger.error(f"Failed to get futures signals: {e}")
        
        try:
            # Get spot signals from autonomous trading
            conn = sqlite3.connect('autonomous_trading.db')
            query = '''
                SELECT symbol, signal, confidence, technical_score, ai_score,
                       current_price, rsi, volume_ratio, entry_reasons, timestamp
                FROM ai_signals
                WHERE timestamp > datetime('now', '-2 hours')
                ORDER BY confidence DESC
                LIMIT 20
            '''
            df_spot = pd.read_sql_query(query, conn)
            conn.close()
            
            for _, row in df_spot.iterrows():
                signal_data = {
                    'symbol': row['symbol'],
                    'signal': row['signal'],
                    'confidence': row['confidence'],
                    'technical_score': row.get('technical_score', 0),
                    'ai_score': row.get('ai_score', 0),
                    'current_price': row['current_price'],
                    'rsi': row.get('rsi', 0),
                    'volume_ratio': row.get('volume_ratio', 0),
                    'entry_reasons': row.get('entry_reasons', ''),
                    'timestamp': row['timestamp'],
                    'signal_type': 'SPOT',
                    'leverage': 1,
                    'risk_level': 'MEDIUM'
                }
                
                # Add signal attribution
                attribution_id = attribution_engine.log_signal_origin(signal_data)
                signal_data['attribution_id'] = attribution_id
                
                # Apply risk controls
                enhanced_signal = volatility_risk_controller.apply_risk_controls(signal_data)
                signals.append(enhanced_signal)
                
        except Exception as e:
            logger.error(f"Failed to get spot signals: {e}")
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals[:30]  # Return top 30 signals
    
    def get_signal_attribution_data(self) -> Dict:
        """Get signal attribution analytics"""
        try:
            # Get source performance
            source_performance = attribution_engine.get_source_performance()
            
            # Get recent attributions
            recent_attributions = attribution_engine.get_recent_attributions(100)
            
            # Calculate attribution statistics
            attribution_stats = {}
            if recent_attributions:
                df = pd.DataFrame(recent_attributions)
                
                # Source distribution
                source_counts = df['origin_source'].value_counts().to_dict()
                
                # Success rates by source
                success_by_source = {}
                for source in df['origin_source'].unique():
                    source_data = df[df['origin_source'] == source]
                    completed = source_data[source_data['outcome'].notna()]
                    if len(completed) > 0:
                        success_rate = (completed['outcome'] == 'WIN').mean() * 100
                        success_by_source[source] = success_rate
                
                attribution_stats = {
                    'source_distribution': source_counts,
                    'success_rates': success_by_source,
                    'total_signals': len(recent_attributions)
                }
            
            return {
                'source_performance': source_performance,
                'recent_attributions': recent_attributions,
                'attribution_stats': attribution_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get attribution data: {e}")
            return {}
    
    def get_model_evaluation_data(self) -> Dict:
        """Get AI model evaluation data"""
        try:
            # Get current active model
            active_model = model_evaluator.get_current_active_model()
            
            # Get performance history
            performance_history = model_evaluator.get_model_performance_history()
            
            # Check if evaluation is due
            evaluation_due = model_evaluator.should_evaluate_models()
            
            return {
                'active_model': active_model,
                'performance_history': performance_history,
                'evaluation_due': evaluation_due,
                'last_evaluation': model_evaluator.active_model_config.get('last_evaluation'),
                'available_models': list(model_evaluator.available_models.keys())
            }
            
        except Exception as e:
            logger.error(f"Failed to get model evaluation data: {e}")
            return {}
    
    def get_risk_analytics_data(self) -> Dict:
        """Get risk management analytics"""
        try:
            # Get system-wide risk metrics
            risk_metrics = volatility_risk_controller.get_risk_metrics()
            
            # Get recent volatility data
            conn = sqlite3.connect('risk_control.db')
            query = '''
                SELECT symbol, atr_percent, volatility_regime, timestamp
                FROM volatility_metrics
                WHERE timestamp > datetime('now', '-24 hours')
                ORDER BY timestamp DESC
            '''
            volatility_data = pd.read_sql_query(query, conn)
            conn.close()
            
            # Calculate risk statistics
            risk_stats = {}
            if not volatility_data.empty:
                risk_stats = {
                    'avg_volatility': volatility_data['atr_percent'].mean(),
                    'volatility_distribution': volatility_data['volatility_regime'].value_counts().to_dict(),
                    'high_volatility_count': len(volatility_data[volatility_data['volatility_regime'] == 'HIGH']),
                    'recent_volatility': volatility_data.to_dict('records')[:50]
                }
            
            return {
                'risk_metrics': risk_metrics,
                'risk_stats': risk_stats,
                'volatility_data': volatility_data.to_dict('records') if not volatility_data.empty else []
            }
            
        except Exception as e:
            logger.error(f"Failed to get risk analytics: {e}")
            return {}
    
    def get_portfolio_data_with_analytics(self) -> Dict:
        """Get enhanced portfolio data with analytics"""
        try:
            balance = self.exchange.fetch_balance()
            portfolio_data = []
            total_value = 0
            
            for asset, data in balance.items():
                if data['total'] > 0 and asset != 'info':
                    try:
                        if asset == 'USDT':
                            price = 1.0
                            change_24h = 0.0
                        else:
                            symbol = f"{asset}/USDT"
                            ticker = self.exchange.fetch_ticker(symbol)
                            price = ticker['last']
                            change_24h = ticker['percentage'] or 0
                        
                        value_usd = data['total'] * price
                        total_value += value_usd
                        
                        # Get volatility data for the asset
                        volatility_metrics = volatility_risk_controller.calculate_volatility_metrics(f"{asset}/USDT" if asset != 'USDT' else 'BTC/USDT')
                        
                        portfolio_data.append({
                            'asset': asset,
                            'balance': data['total'],
                            'price': price,
                            'value_usd': value_usd,
                            'change_24h': change_24h,
                            'free': data['free'],
                            'used': data['used'],
                            'volatility_regime': volatility_metrics.get('volatility_regime', 'UNKNOWN') if volatility_metrics else 'UNKNOWN',
                            'atr_percent': volatility_metrics.get('atr_percent', 0) if volatility_metrics else 0
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing {asset}: {e}")
                        continue
            
            # Calculate allocation percentages
            for item in portfolio_data:
                item['allocation_pct'] = (item['value_usd'] / total_value * 100) if total_value > 0 else 0
            
            # Calculate portfolio analytics
            portfolio_analytics = {
                'total_value': total_value,
                'num_assets': len(portfolio_data),
                'largest_holding': max(portfolio_data, key=lambda x: x['allocation_pct'])['asset'] if portfolio_data else 'None',
                'high_volatility_allocation': sum(item['allocation_pct'] for item in portfolio_data if item['volatility_regime'] == 'HIGH'),
                'diversification_score': self._calculate_diversification_score(portfolio_data)
            }
            
            return {
                'portfolio': sorted(portfolio_data, key=lambda x: x['value_usd'], reverse=True),
                'analytics': portfolio_analytics
            }
            
        except Exception as e:
            logger.error(f"Portfolio data fetch failed: {e}")
            return {'portfolio': [], 'analytics': {}}
    
    def _calculate_diversification_score(self, portfolio_data: List[Dict]) -> float:
        """Calculate portfolio diversification score"""
        if not portfolio_data:
            return 0
        
        # Herfindahl-Hirschman Index for concentration
        allocations = [item['allocation_pct'] / 100 for item in portfolio_data]
        hhi = sum(alloc**2 for alloc in allocations)
        
        # Convert to diversification score (higher is better)
        diversification_score = (1 - hhi) * 100
        
        return diversification_score

# Initialize enhanced UI
enhanced_ui = EnhancedTradingUI()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('enhanced_dashboard_3000.html')

@app.route('/api/signals')
def api_signals():
    """Enhanced signals with attribution and risk controls"""
    try:
        signals = enhanced_ui.get_combined_signals_data()
        
        return jsonify({
            'signals': signals,
            'total_count': len(signals),
            'futures_count': len([s for s in signals if s.get('signal_type') == 'FUTURES']),
            'spot_count': len([s for s in signals if s.get('signal_type') == 'SPOT']),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Signals API error: {e}")
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/signal-attribution')
def api_signal_attribution():
    """Signal attribution analytics"""
    try:
        attribution_data = enhanced_ui.get_signal_attribution_data()
        
        return jsonify({
            'attribution': attribution_data,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Attribution API error: {e}")
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/model-evaluation')
def api_model_evaluation():
    """AI model evaluation data"""
    try:
        model_data = enhanced_ui.get_model_evaluation_data()
        
        return jsonify({
            'models': model_data,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Model evaluation API error: {e}")
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/risk-analytics')
def api_risk_analytics():
    """Risk management analytics"""
    try:
        risk_data = enhanced_ui.get_risk_analytics_data()
        
        return jsonify({
            'risk': risk_data,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Risk analytics API error: {e}")
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/portfolio-enhanced')
def api_portfolio_enhanced():
    """Enhanced portfolio with volatility analytics"""
    try:
        portfolio_data = enhanced_ui.get_portfolio_data_with_analytics()
        
        return jsonify({
            'portfolio_data': portfolio_data,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Enhanced portfolio API error: {e}")
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/trigger-model-evaluation', methods=['POST'])
def api_trigger_model_evaluation():
    """Manually trigger model evaluation"""
    try:
        # Run model evaluation cycle
        model_evaluator.run_evaluation_cycle()
        
        return jsonify({
            'message': 'Model evaluation completed',
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Model evaluation trigger failed: {e}")
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/api/audit-logs')
def api_audit_logs():
    """System audit logs"""
    try:
        # Get audit logs from multiple sources
        audit_logs = []
        
        # Risk events
        conn = sqlite3.connect('risk_control.db')
        query = '''
            SELECT 'RISK' as log_type, event_type, symbol, description, 
                   severity, timestamp
            FROM risk_events
            WHERE timestamp > datetime('now', '-7 days')
            ORDER BY timestamp DESC
            LIMIT 50
        '''
        risk_logs = pd.read_sql_query(query, conn)
        conn.close()
        
        audit_logs.extend(risk_logs.to_dict('records'))
        
        # Model switches
        conn = sqlite3.connect('ai/model_evaluation.db')
        query = '''
            SELECT 'MODEL_SWITCH' as log_type, from_model, to_model, 
                   reason, performance_improvement, switch_date as timestamp
            FROM model_switches
            WHERE switch_date > datetime('now', '-7 days')
            ORDER BY switch_date DESC
            LIMIT 20
        '''
        model_logs = pd.read_sql_query(query, conn)
        conn.close()
        
        audit_logs.extend(model_logs.to_dict('records'))
        
        # Sort by timestamp
        audit_logs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'audit_logs': audit_logs,
            'total_count': len(audit_logs),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Audit logs API error: {e}")
        return jsonify({'error': str(e), 'status': 'error'})

if __name__ == '__main__':
    logger.info("Starting Enhanced Trading UI on port 3000")
    logger.info("Features: Signal Attribution, AI Model Evaluation, Risk Analytics, Audit Logs")
    app.run(host='0.0.0.0', port=3000, debug=False)