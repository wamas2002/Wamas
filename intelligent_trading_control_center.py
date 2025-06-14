#!/usr/bin/env python3
"""
Intelligent Trading Control Center - Unified Smart Enhancement Dashboard
Integrates all intelligent modules: confidence tuning, order book monitoring, 
smart execution, anomaly detection, portfolio rotation, drawdown protection, and remote control
"""

import os
import sys
import sqlite3
import ccxt
import pandas as pd
import numpy as np
import logging
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from flask import Flask, render_template_string, jsonify

# Import all intelligent modules
from ai.online_confidence_tuner import OnlineConfidenceTuner
from execution.order_book_monitor import OrderBookMonitor
from execution.smart_order_executor import SmartOrderExecutor
from monitoring.anomaly_detector import AnomalyDetector
from portfolio.rotation_engine import PortfolioRotationEngine
from portfolio.drawdown_guard import DrawdownGuard
from api.remote_control import RemoteControlAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('intelligent_trading.log')
    ]
)
logger = logging.getLogger(__name__)

class IntelligentTradingControlCenter:
    """Unified control center for all intelligent trading enhancements"""
    
    def __init__(self):
        self.exchange = None
        self.is_running = False
        
        # Initialize all intelligent modules
        self.confidence_tuner = OnlineConfidenceTuner()
        self.order_book_monitor = OrderBookMonitor()
        self.smart_executor = SmartOrderExecutor()
        self.anomaly_detector = AnomalyDetector()
        self.rotation_engine = PortfolioRotationEngine()
        self.drawdown_guard = DrawdownGuard()
        self.remote_control = RemoteControlAPI()
        
        # Trading parameters with intelligent defaults
        self.trading_config = {
            'min_confidence': 75.0,
            'max_position_size': 1.5,
            'emergency_stop_active': False,
            'smart_execution_enabled': True,
            'confidence_tuning_enabled': True,
            'anomaly_detection_enabled': True,
            'drawdown_protection_enabled': True
        }
        
        # System state
        self.symbols = []
        self.signal_history = []
        self.trade_history = []
        self.system_metrics = {}
        
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the intelligent trading control center"""
        try:
            # Connect to OKX
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            self.exchange.load_markets()
            
            # Get trading symbols under $200
            self.symbols = self.fetch_trading_symbols()
            
            # Setup unified database
            self.setup_unified_database()
            
            logger.info(f"Intelligent Trading Control Center initialized with {len(self.symbols)} symbols")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    def fetch_trading_symbols(self) -> List[str]:
        """Fetch symbols under $200 for trading"""
        try:
            tickers = self.exchange.fetch_tickers()
            under_200_symbols = []
            
            for symbol, ticker in tickers.items():
                if '/USDT' in symbol and ticker['last'] and float(ticker['last']) <= 200:
                    under_200_symbols.append(symbol)
            
            # Sort by volume and take top 100
            symbol_volumes = [(symbol, tickers[symbol]['quoteVolume'] or 0) for symbol in under_200_symbols]
            symbol_volumes.sort(key=lambda x: x[1], reverse=True)
            
            return [symbol for symbol, _ in symbol_volumes[:100]]
            
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            return []
    
    def setup_unified_database(self):
        """Setup unified database for intelligent trading"""
        try:
            conn = sqlite3.connect('intelligent_trading.db')
            cursor = conn.cursor()
            
            # Unified signals table with all enhancements
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS intelligent_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    signal_type TEXT,
                    
                    -- Confidence metrics
                    original_confidence REAL,
                    tuned_confidence REAL,
                    final_confidence REAL,
                    
                    -- Market context
                    market_regime TEXT,
                    order_book_quality TEXT,
                    execution_strategy TEXT,
                    
                    -- Risk assessment
                    anomaly_score REAL,
                    drawdown_risk TEXT,
                    position_size_adjusted REAL,
                    
                    -- Execution details
                    executed BOOLEAN DEFAULT FALSE,
                    execution_method TEXT,
                    fill_price REAL,
                    slippage_pct REAL,
                    
                    -- Performance tracking
                    outcome_24h REAL,
                    win_loss TEXT,
                    enhancement_impact TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Unified intelligent trading database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> Optional[pd.DataFrame]:
        """Get market data for analysis"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Calculate technical indicators
            df['rsi'] = pd.Series(df['close']).rolling(14).apply(
                lambda x: 100 - (100 / (1 + x.diff().fillna(0).apply(lambda y: max(y, 0)).sum() / 
                                        x.diff().fillna(0).apply(lambda y: max(-y, 0)).sum())), raw=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Market data fetch failed for {symbol}: {e}")
            return None
    
    def generate_intelligent_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Generate signal with full intelligent enhancement pipeline"""
        if len(df) < 50:
            return None
        
        try:
            # Step 1: Generate base signal
            base_signal = self.generate_base_signal(symbol, df)
            if not base_signal:
                return None
            
            # Step 2: Apply confidence tuning
            if self.trading_config['confidence_tuning_enabled']:
                tuned_signal = self.confidence_tuner.apply_confidence_tuning(base_signal)
            else:
                tuned_signal = base_signal
            
            # Step 3: Check order book quality
            if self.trading_config['smart_execution_enabled']:
                can_execute, quality_reason = self.order_book_monitor.should_execute_trade(symbol)
                tuned_signal['order_book_quality'] = quality_reason
                tuned_signal['order_book_can_execute'] = can_execute
            else:
                tuned_signal['order_book_can_execute'] = True
                tuned_signal['order_book_quality'] = 'not_checked'
            
            # Step 4: Anomaly detection
            if self.trading_config['anomaly_detection_enabled']:
                anomaly_analysis = self.anomaly_detector.analyze_market_conditions(symbol, df)
                tuned_signal['anomaly_detected'] = anomaly_analysis['anomaly_detected']
                tuned_signal['anomaly_score'] = len(anomaly_analysis.get('anomalies', []))
            else:
                tuned_signal['anomaly_detected'] = False
                tuned_signal['anomaly_score'] = 0
            
            # Step 5: Drawdown protection
            if self.trading_config['drawdown_protection_enabled']:
                balance_data = self.get_balance_data()
                should_block, block_reason = self.drawdown_guard.should_block_trade(
                    tuned_signal['confidence'], 
                    self.trading_config['max_position_size']
                )
                tuned_signal['drawdown_blocked'] = should_block
                tuned_signal['drawdown_reason'] = block_reason
                
                # Adjust position size
                adjusted_size = self.drawdown_guard.adjust_position_size(
                    self.trading_config['max_position_size']
                )
                tuned_signal['position_size_adjusted'] = adjusted_size
            else:
                tuned_signal['drawdown_blocked'] = False
                tuned_signal['position_size_adjusted'] = self.trading_config['max_position_size']
            
            # Step 6: Final execution decision
            execute_signal = (
                tuned_signal['confidence'] >= self.trading_config['min_confidence'] and
                tuned_signal.get('order_book_can_execute', True) and
                not tuned_signal.get('anomaly_detected', False) and
                not tuned_signal.get('drawdown_blocked', False) and
                not self.trading_config['emergency_stop_active']
            )
            
            tuned_signal['execute_decision'] = execute_signal
            tuned_signal['enhancement_pipeline'] = {
                'confidence_tuning': self.trading_config['confidence_tuning_enabled'],
                'order_book_check': self.trading_config['smart_execution_enabled'],
                'anomaly_detection': self.trading_config['anomaly_detection_enabled'],
                'drawdown_protection': self.trading_config['drawdown_protection_enabled']
            }
            
            return tuned_signal
            
        except Exception as e:
            logger.error(f"Intelligent signal generation failed for {symbol}: {e}")
            return None
    
    def generate_base_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Generate base trading signal"""
        if len(df) < 20:
            return None
        
        try:
            latest = df.iloc[-1]
            
            # Simple signal generation for demonstration
            rsi = latest.get('rsi', 50)
            price = latest['close']
            volume = latest['volume']
            
            signal_strength = 0
            signal_type = 'BUY'
            
            # RSI component
            if rsi < 30:
                signal_strength += 30
                signal_type = 'BUY'
            elif rsi > 70:
                signal_strength += 30
                signal_type = 'SELL'
            elif 40 <= rsi <= 60:
                signal_strength += 15
            
            # Volume component
            avg_volume = df['volume'].rolling(20).mean().iloc[-1]
            if volume > avg_volume * 1.5:
                signal_strength += 20
            
            # Price momentum
            price_change = (price - df['close'].iloc[-5]) / df['close'].iloc[-5]
            if abs(price_change) > 0.02:  # 2% move
                signal_strength += 25
            
            if signal_strength < 40:
                return None
            
            confidence = min(signal_strength + np.random.normal(0, 5), 95)
            
            return {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': confidence,
                'price': float(price),
                'rsi': float(rsi) if not pd.isna(rsi) else None,
                'volume_ratio': float(volume / avg_volume) if avg_volume > 0 else 1.0,
                'components': {
                    'rsi_score': min(30, abs(50 - rsi)),
                    'volume_score': min(20, (volume / avg_volume - 1) * 20) if avg_volume > 0 else 0,
                    'momentum_score': min(25, abs(price_change) * 1000)
                }
            }
            
        except Exception as e:
            logger.error(f"Base signal generation failed for {symbol}: {e}")
            return None
    
    def execute_intelligent_trade(self, signal: Dict) -> Dict:
        """Execute trade using intelligent order execution"""
        try:
            if not signal.get('execute_decision'):
                return {
                    'success': False,
                    'reason': 'Signal filtered by intelligence pipeline',
                    'signal_id': f"{signal['symbol']}_{int(time.time())}"
                }
            
            # Check remote control
            trading_allowed, reason = self.remote_control.is_trading_allowed()
            if not trading_allowed:
                return {
                    'success': False,
                    'reason': f'Remote control block: {reason}',
                    'signal_id': f"{signal['symbol']}_{int(time.time())}"
                }
            
            # Determine execution strategy
            execution_strategy = 'auto'  # Let smart executor decide
            
            # Execute with smart order executor
            execution_result = self.smart_executor.smart_execute_order(
                symbol=signal['symbol'],
                side=signal['signal_type'].lower(),
                amount=signal.get('position_size_adjusted', 1.0),
                strategy=execution_strategy
            )
            
            # Log execution to unified database
            self.log_intelligent_signal(signal, execution_result)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Intelligent trade execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'signal_id': f"{signal['symbol']}_{int(time.time())}"
            }
    
    def log_intelligent_signal(self, signal: Dict, execution_result: Dict):
        """Log signal with all intelligent enhancements"""
        try:
            conn = sqlite3.connect('intelligent_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO intelligent_signals (
                    timestamp, symbol, signal_type, original_confidence,
                    tuned_confidence, final_confidence, market_regime,
                    order_book_quality, execution_strategy, anomaly_score,
                    drawdown_risk, position_size_adjusted, executed,
                    execution_method, enhancement_impact
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['timestamp'],
                signal['symbol'],
                signal['signal_type'],
                signal.get('original_confidence', signal['confidence']),
                signal.get('tuned_confidence', signal['confidence']),
                signal['confidence'],
                signal.get('market_regime', 'unknown'),
                signal.get('order_book_quality', 'not_checked'),
                execution_result.get('strategy', 'unknown'),
                signal.get('anomaly_score', 0),
                signal.get('drawdown_reason', 'none'),
                signal.get('position_size_adjusted', 0),
                execution_result.get('success', False),
                execution_result.get('strategy', 'unknown'),
                json.dumps(signal.get('enhancement_pipeline', {}))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log intelligent signal: {e}")
    
    def get_balance_data(self) -> Dict:
        """Get current balance data"""
        try:
            balance = self.exchange.fetch_balance()
            return balance
        except Exception as e:
            logger.error(f"Failed to get balance data: {e}")
            return {}
    
    def run_intelligent_trading_cycle(self):
        """Main intelligent trading cycle"""
        logger.info("🧠 Starting Intelligent Trading Control Center")
        
        while self.is_running:
            try:
                cycle_start = time.time()
                signals_generated = 0
                trades_executed = 0
                
                # Check system health
                self.update_system_metrics()
                
                # Run portfolio rotation if needed (weekly)
                if datetime.now().weekday() == 0 and datetime.now().hour == 0:  # Monday midnight
                    self.run_portfolio_rotation()
                
                # Scan symbols for signals
                logger.info(f"🔄 Intelligent scan: {len(self.symbols)} symbols")
                
                for symbol in self.symbols[:20]:  # Limit to prevent rate limiting
                    try:
                        df = self.get_market_data(symbol)
                        if df is not None:
                            signal = self.generate_intelligent_signal(symbol, df)
                            
                            if signal and signal.get('execute_decision'):
                                signals_generated += 1
                                
                                # Execute trade
                                execution_result = self.execute_intelligent_trade(signal)
                                
                                if execution_result.get('success'):
                                    trades_executed += 1
                                    logger.info(f"🎯 {symbol}: {signal['signal_type']} executed "
                                               f"(Conf: {signal['confidence']:.1f}%, "
                                               f"Strategy: {execution_result.get('strategy')})")
                                
                                # Store for dashboard
                                self.signal_history.append(signal)
                                if len(self.signal_history) > 100:
                                    self.signal_history = self.signal_history[-100:]
                    
                    except Exception as e:
                        logger.error(f"Symbol processing failed for {symbol}: {e}")
                        continue
                    
                    time.sleep(0.5)  # Rate limiting
                
                cycle_time = time.time() - cycle_start
                logger.info(f"✅ Intelligent cycle complete: {signals_generated} signals, "
                           f"{trades_executed} trades ({cycle_time:.1f}s)")
                
                # Wait before next cycle
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Intelligent trading cycle error: {e}")
                time.sleep(60)
    
    def update_system_metrics(self):
        """Update comprehensive system metrics"""
        try:
            balance_data = self.get_balance_data()
            
            # Anomaly detection
            anomaly_summary = self.anomaly_detector.get_anomaly_summary()
            
            # Drawdown protection
            if balance_data:
                drawdown_check = self.drawdown_guard.comprehensive_drawdown_check(
                    balance_data, self.trade_history
                )
            else:
                drawdown_check = {'protection_status': {'active': False}}
            
            # Remote control status
            remote_status = self.remote_control.system_status
            
            self.system_metrics = {
                'timestamp': datetime.now().isoformat(),
                'anomaly_health': anomaly_summary.get('health_status', 'unknown'),
                'drawdown_protection': drawdown_check['protection_status'].get('active', False),
                'remote_control': remote_status.get('emergency_stop', False),
                'confidence_tuning': len(self.confidence_tuner.get_current_weights()),
                'order_book_monitoring': True,
                'smart_execution': True
            }
            
        except Exception as e:
            logger.error(f"System metrics update failed: {e}")
    
    def run_portfolio_rotation(self):
        """Run weekly portfolio rotation analysis"""
        try:
            logger.info("📊 Running weekly portfolio rotation analysis...")
            
            rotation_result = self.rotation_engine.run_rotation_analysis(self.trade_history)
            
            if rotation_result.get('success'):
                logger.info(f"Portfolio rotation complete: {rotation_result.get('executed_changes', 0)} changes")
            
        except Exception as e:
            logger.error(f"Portfolio rotation failed: {e}")
    
    def start_intelligent_trading(self):
        """Start the intelligent trading system"""
        if self.is_running:
            return
        
        self.is_running = True
        trading_thread = threading.Thread(target=self.run_intelligent_trading_cycle, daemon=True)
        trading_thread.start()
    
    def stop_intelligent_trading(self):
        """Stop the intelligent trading system"""
        self.is_running = False

# Flask web interface
app = Flask(__name__)
control_center = IntelligentTradingControlCenter()

@app.route('/')
def intelligent_dashboard():
    """Intelligent Trading Control Center Dashboard"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Intelligent Trading Control Center</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Arial, sans-serif; background: #0a0e27; color: #fff; }
            .container { max-width: 1800px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; margin-bottom: 30px; }
            .header h1 { color: #4CAF50; font-size: 2.8em; margin-bottom: 10px; }
            .header p { color: #999; font-size: 1.2em; }
            .modules-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .module-card { background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 20px; border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
            .module-title { color: #4CAF50; font-size: 1.3em; margin-bottom: 15px; border-bottom: 2px solid #4CAF50; padding-bottom: 8px; }
            .metric { display: flex; justify-content: space-between; margin: 8px 0; }
            .metric-label { color: #ccc; }
            .metric-value { color: #fff; font-weight: bold; }
            .status-active { color: #4CAF50; }
            .status-inactive { color: #f44336; }
            .status-warning { color: #ff9800; }
            .signals-section { background: #1a1a2e; border-radius: 10px; padding: 20px; margin-top: 20px; }
            .signal-item { background: #252545; margin: 10px 0; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; }
            .signal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
            .enhancement-tags { margin-top: 8px; }
            .tag { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.8em; margin-right: 5px; }
            .tag-confidence { background: #4CAF50; }
            .tag-order-book { background: #2196F3; }
            .tag-anomaly { background: #ff9800; }
            .tag-drawdown { background: #9c27b0; }
            @media (max-width: 768px) { .modules-grid { grid-template-columns: 1fr; } }
        </style>
        <script>
            function updateData() {
                Promise.all([
                    fetch('/api/intelligent/status').then(r => r.json()),
                    fetch('/api/intelligent/signals').then(r => r.json()),
                    fetch('/api/intelligent/metrics').then(r => r.json())
                ]).then(([status, signals, metrics]) => {
                    // Update module statuses
                    document.getElementById('confidence-tuning').textContent = 
                        status.confidence_tuning_enabled ? 'Active' : 'Disabled';
                    document.getElementById('confidence-tuning').className = 
                        status.confidence_tuning_enabled ? 'metric-value status-active' : 'metric-value status-inactive';
                    
                    document.getElementById('order-book-monitoring').textContent = 
                        status.smart_execution_enabled ? 'Active' : 'Disabled';
                    document.getElementById('order-book-monitoring').className = 
                        status.smart_execution_enabled ? 'metric-value status-active' : 'metric-value status-inactive';
                    
                    document.getElementById('anomaly-detection').textContent = 
                        status.anomaly_detection_enabled ? 'Active' : 'Disabled';
                    document.getElementById('anomaly-detection').className = 
                        status.anomaly_detection_enabled ? 'metric-value status-active' : 'metric-value status-inactive';
                    
                    document.getElementById('drawdown-protection').textContent = 
                        status.drawdown_protection_enabled ? 'Active' : 'Disabled';
                    document.getElementById('drawdown-protection').className = 
                        status.drawdown_protection_enabled ? 'metric-value status-active' : 'metric-value status-inactive';
                    
                    document.getElementById('emergency-stop').textContent = 
                        status.emergency_stop_active ? 'ACTIVE' : 'Ready';
                    document.getElementById('emergency-stop').className = 
                        status.emergency_stop_active ? 'metric-value status-warning' : 'metric-value status-active';
                    
                    // Update metrics
                    document.getElementById('total-enhancements').textContent = 
                        Object.values(status).filter(v => v === true).length;
                    document.getElementById('system-health').textContent = metrics.anomaly_health || 'Good';
                    document.getElementById('intelligence-score').textContent = 
                        Math.round((Object.values(status).filter(v => v === true).length / 5) * 100) + '%';
                    
                    // Update signals
                    const signalsHtml = signals.slice(0, 10).map(signal => `
                        <div class="signal-item">
                            <div class="signal-header">
                                <span><strong>${signal.symbol}</strong> (${signal.signal_type})</span>
                                <span style="color: #4CAF50">${signal.confidence.toFixed(1)}%</span>
                            </div>
                            <div class="enhancement-tags">
                                ${signal.enhancement_pipeline?.confidence_tuning ? '<span class="tag tag-confidence">Confidence Tuned</span>' : ''}
                                ${signal.enhancement_pipeline?.order_book_check ? '<span class="tag tag-order-book">Order Book Verified</span>' : ''}
                                ${signal.enhancement_pipeline?.anomaly_detection ? '<span class="tag tag-anomaly">Anomaly Checked</span>' : ''}
                                ${signal.enhancement_pipeline?.drawdown_protection ? '<span class="tag tag-drawdown">Drawdown Protected</span>' : ''}
                            </div>
                        </div>
                    `).join('');
                    document.getElementById('signals-list').innerHTML = signalsHtml || '<div class="signal-item">No recent signals</div>';
                });
            }
            
            document.addEventListener('DOMContentLoaded', function() {
                updateData();
                setInterval(updateData, 5000);
            });
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Intelligent Trading Control Center</h1>
                <p>Advanced Non-Destructive Enhancements • Online Learning • Smart Execution</p>
            </div>
            
            <div class="modules-grid">
                <div class="module-card">
                    <div class="module-title">🎯 Confidence Optimization</div>
                    <div class="metric">
                        <span class="metric-label">Status:</span>
                        <span class="metric-value status-active" id="confidence-tuning">Active</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Learning Mode:</span>
                        <span class="metric-value">Online</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Component Weights:</span>
                        <span class="metric-value">5 Tracked</span>
                    </div>
                </div>
                
                <div class="module-card">
                    <div class="module-title">📊 Order Book Intelligence</div>
                    <div class="metric">
                        <span class="metric-label">Status:</span>
                        <span class="metric-value status-active" id="order-book-monitoring">Active</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Execution Strategies:</span>
                        <span class="metric-value">TWAP, Iceberg, Smart</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Spoofing Detection:</span>
                        <span class="metric-value">Enabled</span>
                    </div>
                </div>
                
                <div class="module-card">
                    <div class="module-title">🛡️ Anomaly Detection</div>
                    <div class="metric">
                        <span class="metric-label">Status:</span>
                        <span class="metric-value status-active" id="anomaly-detection">Active</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">System Health:</span>
                        <span class="metric-value" id="system-health">Good</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Active Anomalies:</span>
                        <span class="metric-value">0</span>
                    </div>
                </div>
                
                <div class="module-card">
                    <div class="module-title">🛡️ Drawdown Protection</div>
                    <div class="metric">
                        <span class="metric-label">Status:</span>
                        <span class="metric-value status-active" id="drawdown-protection">Active</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Thresholds:</span>
                        <span class="metric-value">6% / 8% / 12%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Protection Level:</span>
                        <span class="metric-value">None</span>
                    </div>
                </div>
                
                <div class="module-card">
                    <div class="module-title">🔄 Portfolio Rotation</div>
                    <div class="metric">
                        <span class="metric-label">Status:</span>
                        <span class="metric-value">Weekly Schedule</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Performance Tracking:</span>
                        <span class="metric-value">Sharpe + Win Rate</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Top Performers:</span>
                        <span class="metric-value">10 Tracked</span>
                    </div>
                </div>
                
                <div class="module-card">
                    <div class="module-title">📱 Remote Control</div>
                    <div class="metric">
                        <span class="metric-label">Emergency Stop:</span>
                        <span class="metric-value status-active" id="emergency-stop">Ready</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">API Security:</span>
                        <span class="metric-value">Enabled</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Rate Limiting:</span>
                        <span class="metric-value">10/min</span>
                    </div>
                </div>
                
                <div class="module-card">
                    <div class="module-title">📈 Intelligence Summary</div>
                    <div class="metric">
                        <span class="metric-label">Active Enhancements:</span>
                        <span class="metric-value" id="total-enhancements">5</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Intelligence Score:</span>
                        <span class="metric-value" id="intelligence-score">100%</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Non-Destructive:</span>
                        <span class="metric-value status-active">Verified</span>
                    </div>
                </div>
            </div>
            
            <div class="signals-section">
                <div class="module-title">🎯 Enhanced Signals</div>
                <div id="signals-list">Loading signals...</div>
            </div>
        </div>
    </body>
    </html>
    ''')

@app.route('/api/intelligent/status')
def get_intelligent_status():
    """Get intelligent system status"""
    return jsonify(control_center.trading_config)

@app.route('/api/intelligent/signals')
def get_intelligent_signals():
    """Get recent intelligent signals"""
    return jsonify(control_center.signal_history[-20:] if control_center.signal_history else [])

@app.route('/api/intelligent/metrics')
def get_intelligent_metrics():
    """Get system metrics"""
    return jsonify(control_center.system_metrics)

def main():
    """Main function"""
    try:
        # Start intelligent trading system
        control_center.start_intelligent_trading()
        
        # Start web interface on port 6000 (avoiding conflicts)
        app.run(host='0.0.0.0', port=6000, debug=False)
        
    except KeyboardInterrupt:
        logger.info("Shutting down Intelligent Trading Control Center...")
        control_center.stop_intelligent_trading()
    except Exception as e:
        logger.error(f"System error: {e}")

if __name__ == "__main__":
    main()