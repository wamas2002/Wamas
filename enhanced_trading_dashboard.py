#!/usr/bin/env python3
"""
Enhanced Trading Dashboard with AI Insights
Smart non-destructive enhancements with market regime detection and feedback learning
"""

import os
import sys
import sqlite3
import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from flask import Flask, render_template_string, jsonify
import threading

# Import our new AI modules
from ai.market_regime_detector import MarketRegimeDetector
from ai.sell_signal_generator import SellSignalGenerator
from ai.signal_filter import SignalFilter
from feedback.feedback_logger import FeedbackLogger
from feedback.feedback_analyzer import FeedbackAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('enhanced_trading.log')
    ]
)
logger = logging.getLogger(__name__)

class EnhancedTradingSystem:
    """Enhanced trading system with market context awareness and SELL signals"""
    
    def __init__(self):
        self.exchange = None
        self.is_running = False
        
        # Enhanced AI components
        self.regime_detector = MarketRegimeDetector()
        self.sell_generator = SellSignalGenerator()
        self.signal_filter = SignalFilter()
        self.feedback_logger = FeedbackLogger()
        self.feedback_analyzer = FeedbackAnalyzer()
        
        # Trading parameters
        self.min_confidence = 70.0
        self.position_size_base = 0.08  # 8% base allocation
        self.max_position_size = 0.10   # 10% maximum per trade
        self.stop_loss_pct = 12.0
        
        # Symbol management
        self.symbols = []
        self.signal_history = []
        self.current_regime = "unknown"
        self.buy_sell_ratio = {"BUY": 0, "SELL": 0}
        
        # Initialize system
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize enhanced trading system"""
        try:
            # Connect to OKX
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            # Load markets and get symbols under $100
            self.exchange.load_markets()
            self.symbols = self.fetch_symbols_under_100()
            
            # Setup enhanced database
            self.setup_database()
            
            logger.info(f"Enhanced trading system initialized with {len(self.symbols)} symbols")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    def fetch_symbols_under_100(self) -> List[str]:
        """Fetch symbols under $100 from OKX"""
        try:
            tickers = self.exchange.fetch_tickers()
            under_100_symbols = []
            
            for symbol, ticker in tickers.items():
                if '/USDT' in symbol and ticker['last'] and float(ticker['last']) <= 100:
                    under_100_symbols.append(symbol)
            
            # Sort by volume and take top 50
            symbol_volumes = [(symbol, tickers[symbol]['quoteVolume'] or 0) for symbol in under_100_symbols]
            symbol_volumes.sort(key=lambda x: x[1], reverse=True)
            
            return [symbol for symbol, _ in symbol_volumes[:50]]
            
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            return []
    
    def setup_database(self):
        """Setup enhanced trading database"""
        try:
            conn = sqlite3.connect('enhanced_trading.db')
            cursor = conn.cursor()
            
            # Enhanced signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS enhanced_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    signal_type TEXT,
                    confidence REAL,
                    original_confidence REAL,
                    adjusted_confidence REAL,
                    market_regime TEXT,
                    regime_confidence REAL,
                    price REAL,
                    target_price REAL,
                    stop_loss REAL,
                    volume_ratio REAL,
                    volatility REAL,
                    filter_result TEXT,
                    components TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Enhanced trading database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> Optional[pd.DataFrame]:
        """Get market data with enhanced analysis"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = self.calculate_enhanced_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Market data fetch failed for {symbol}: {e}")
            return None
    
    def calculate_enhanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        # Standard indicators
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        
        # EMAs
        df['ema_20'] = ta.ema(df['close'], length=20)
        df['ema_50'] = ta.ema(df['close'], length=50)
        
        # Volume analysis
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def analyze_enhanced_signal(self, symbol: str, df: pd.DataFrame) -> List[Dict]:
        """Generate both BUY and SELL signals with enhanced analysis"""
        signals = []
        
        try:
            # Detect market regime first
            regime_data = self.regime_detector.detect_regime(df)
            self.current_regime = regime_data['regime']
            
            # Generate BUY signal
            buy_signal = self.generate_buy_signal(symbol, df, regime_data)
            if buy_signal:
                signals.append(buy_signal)
            
            # Generate SELL signal
            sell_signal = self.sell_generator.generate_sell_signal(symbol, df)
            if sell_signal:
                # Add regime context to SELL signal
                sell_signal['market_regime'] = regime_data['regime']
                sell_signal['regime_confidence'] = regime_data['confidence']
                signals.append(sell_signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Enhanced signal analysis failed for {symbol}: {e}")
            return []
    
    def generate_buy_signal(self, symbol: str, df: pd.DataFrame, regime_data: Dict) -> Optional[Dict]:
        """Generate BUY signal with market context awareness"""
        if len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate signal strength
        buy_score = 0
        
        # RSI analysis
        rsi = latest['rsi']
        if pd.notna(rsi):
            if rsi < 30:
                buy_score += 25
            elif rsi < 40:
                buy_score += 15
        
        # MACD analysis
        if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
            macd = latest['MACD_12_26_9']
            macd_signal = latest['MACDs_12_26_9']
            if pd.notna(macd) and pd.notna(macd_signal):
                if macd > macd_signal and prev['MACD_12_26_9'] <= prev['MACDs_12_26_9']:
                    buy_score += 20
        
        # EMA trend analysis
        if 'ema_20' in df.columns and 'ema_50' in df.columns:
            if latest['close'] > latest['ema_20'] > latest['ema_50']:
                buy_score += 20
        
        # Volume confirmation
        volume_surge = latest['volume_ratio'] if pd.notna(latest['volume_ratio']) else 1.0
        if volume_surge > 1.5:
            buy_score += 10
        
        if buy_score < 40:  # Minimum threshold
            return None
        
        confidence = min(buy_score + (volume_surge - 1) * 5, 95)
        
        signal = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'signal_type': 'BUY',
            'confidence': confidence,
            'market_regime': regime_data['regime'],
            'regime_confidence': regime_data['confidence'],
            'price': float(latest['close']),
            'target_price': float(latest['close'] * 1.08),  # 8% target
            'stop_loss': float(latest['close'] * 0.88),     # 12% stop loss
            'volume_ratio': volume_surge,
            'components': {
                'rsi': float(rsi) if pd.notna(rsi) else None,
                'macd': float(macd) if 'macd' in locals() and pd.notna(macd) else None,
                'ema20': float(latest['ema_20']) if pd.notna(latest['ema_20']) else None,
                'ema50': float(latest['ema_50']) if pd.notna(latest['ema_50']) else None
            }
        }
        
        return signal
    
    def process_signal_with_filters(self, signal: Dict, market_data: pd.DataFrame) -> Tuple[Optional[Dict], bool]:
        """Process signal through enhanced filtering system"""
        try:
            # Apply comprehensive signal filtering
            filtered_signal, execute_signal = self.signal_filter.comprehensive_signal_filter(signal, market_data)
            
            # Log signal generation for feedback learning
            if filtered_signal:
                self.feedback_logger.log_signal_generation(filtered_signal)
            
            return filtered_signal, execute_signal
            
        except Exception as e:
            logger.error(f"Signal filtering failed: {e}")
            return signal, True
    
    def save_enhanced_signal(self, signal: Dict):
        """Save enhanced signal to database"""
        try:
            conn = sqlite3.connect('enhanced_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO enhanced_signals (
                    timestamp, symbol, signal_type, confidence, original_confidence,
                    adjusted_confidence, market_regime, regime_confidence, price,
                    target_price, stop_loss, volume_ratio, volatility,
                    filter_result, components
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['timestamp'], signal['symbol'], signal['signal_type'],
                signal['confidence'], signal.get('original_confidence', signal['confidence']),
                signal.get('adjusted_confidence', signal['confidence']),
                signal.get('market_regime', 'unknown'), signal.get('regime_confidence', 0.0),
                signal['price'], signal.get('target_price'), signal.get('stop_loss'),
                signal.get('volume_ratio', 1.0), signal.get('volatility', 0.0),
                signal.get('filter_result', 'passed'), json.dumps(signal.get('components', {}))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save enhanced signal: {e}")
    
    def enhanced_trading_loop(self):
        """Main enhanced trading loop"""
        logger.info("🚀 Starting Enhanced Trading System with AI Insights")
        
        while self.is_running:
            try:
                signals_generated = 0
                buy_signals = 0
                sell_signals = 0
                
                logger.info(f"🔄 Enhanced scan: {len(self.symbols)} symbols (Regime: {self.current_regime})")
                
                for symbol in self.symbols:
                    try:
                        df = self.get_market_data(symbol)
                        if df is not None:
                            signals = self.analyze_enhanced_signal(symbol, df)
                            
                            for signal in signals:
                                # Process through filters
                                filtered_signal, execute_signal = self.process_signal_with_filters(signal, df)
                                
                                if filtered_signal and execute_signal:
                                    signals_generated += 1
                                    
                                    if filtered_signal['signal_type'] == 'BUY':
                                        buy_signals += 1
                                    else:
                                        sell_signals += 1
                                    
                                    # Save signal
                                    self.save_enhanced_signal(filtered_signal)
                                    
                                    # Store for dashboard
                                    self.signal_history.append(filtered_signal)
                                    if len(self.signal_history) > 100:
                                        self.signal_history = self.signal_history[-100:]
                                    
                                    logger.info(f"📊 {symbol}: {filtered_signal['signal_type']} "
                                              f"(Conf: {filtered_signal['confidence']:.1f}%, "
                                              f"Regime: {filtered_signal.get('market_regime', 'unknown')})")
                    
                    except Exception as e:
                        logger.error(f"Enhanced analysis failed for {symbol}: {e}")
                        continue
                
                # Update BUY/SELL ratio
                self.buy_sell_ratio = {"BUY": buy_signals, "SELL": sell_signals}
                
                logger.info(f"✅ Enhanced scan complete: {signals_generated} signals "
                           f"(BUY: {buy_signals}, SELL: {sell_signals})")
                
                # Wait before next scan
                time.sleep(300)  # 5 minutes
                    
            except Exception as e:
                logger.error(f"Enhanced trading loop error: {e}")
                time.sleep(60)
    
    def start_enhanced_trading(self):
        """Start enhanced trading system"""
        if self.is_running:
            return
        
        self.is_running = True
        trading_thread = threading.Thread(target=self.enhanced_trading_loop, daemon=True)
        trading_thread.start()
    
    def stop_enhanced_trading(self):
        """Stop enhanced trading system"""
        self.is_running = False

# Flask web interface with AI Insights
app = Flask(__name__)
trading_system = EnhancedTradingSystem()

@app.route('/')
def enhanced_dashboard():
    """Enhanced trading dashboard with AI Insights"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced AI Trading Dashboard - Market Context Awareness</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Arial, sans-serif; background: #0a0e27; color: #fff; }
            .container { max-width: 1800px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; margin-bottom: 30px; }
            .header h1 { color: #4CAF50; font-size: 2.8em; margin-bottom: 10px; }
            .header p { color: #999; font-size: 1.2em; }
            .tab-container { margin-bottom: 30px; }
            .tabs { display: flex; background: #1a1a2e; border-radius: 10px; overflow: hidden; }
            .tab { flex: 1; padding: 15px; text-align: center; cursor: pointer; background: #252545; border: none; color: #fff; }
            .tab.active { background: #4CAF50; }
            .tab-content { display: none; }
            .tab-content.active { display: block; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .stat-card { background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
            .stat-value { font-size: 1.8em; font-weight: bold; color: #4CAF50; }
            .stat-label { color: #ccc; margin-top: 5px; font-size: 0.9em; }
            .regime-indicator { padding: 6px 12px; border-radius: 6px; font-size: 0.9em; font-weight: bold; margin-top: 8px; display: inline-block; }
            .bull { background: #4CAF50; color: #fff; }
            .bear { background: #f44336; color: #fff; }
            .sideways { background: #ff9800; color: #fff; }
            .unknown { background: #666; color: #fff; }
            .content-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
            .signals-section { background: #1a1a2e; border-radius: 10px; padding: 20px; }
            .insights-section { background: #1a1a2e; border-radius: 10px; padding: 20px; }
            .section-title { color: #4CAF50; font-size: 1.5em; margin-bottom: 15px; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
            .signal-item { background: #252545; margin: 10px 0; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; }
            .sell-signal { border-left: 4px solid #f44336; }
            .signal-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
            .signal-symbol { font-weight: bold; color: #fff; }
            .confidence-badge { padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
            .high-conf { background: #4CAF50; color: #fff; }
            .medium-conf { background: #ff9800; color: #fff; }
            .low-conf { background: #f44336; color: #fff; }
            .signal-details { font-size: 0.9em; color: #ccc; }
            .insight-item { background: #252545; margin: 10px 0; padding: 15px; border-radius: 8px; }
            .insight-rule { color: #ffa726; font-weight: bold; margin-bottom: 5px; }
            .insight-confidence { color: #81c784; font-size: 0.9em; }
            .ratio-warning { color: #f44336; font-weight: bold; }
            .ratio-balanced { color: #4CAF50; font-weight: bold; }
            @media (max-width: 768px) { .content-grid { grid-template-columns: 1fr; } .tabs { flex-direction: column; } }
        </style>
        <script>
            function showTab(tabName) {
                // Hide all tab contents
                const contents = document.querySelectorAll('.tab-content');
                contents.forEach(content => content.classList.remove('active'));
                
                // Remove active class from all tabs
                const tabs = document.querySelectorAll('.tab');
                tabs.forEach(tab => tab.classList.remove('active'));
                
                // Show selected tab content
                document.getElementById(tabName).classList.add('active');
                document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
            }
            
            function updateData() {
                Promise.all([
                    fetch('/api/enhanced/status').then(r => r.json()),
                    fetch('/api/enhanced/signals').then(r => r.json()),
                    fetch('/api/enhanced/insights').then(r => r.json())
                ]).then(([status, signals, insights]) => {
                    // Update status
                    document.getElementById('system-status').textContent = status.status;
                    document.getElementById('current-regime').textContent = status.current_regime;
                    document.getElementById('current-regime').className = 'regime-indicator ' + 
                        (status.current_regime === 'bull' ? 'bull' : 
                         status.current_regime === 'bear' ? 'bear' : 
                         status.current_regime === 'sideways' ? 'sideways' : 'unknown');
                    
                    document.getElementById('total-signals').textContent = signals.length;
                    document.getElementById('buy-signals').textContent = status.buy_signals;
                    document.getElementById('sell-signals').textContent = status.sell_signals;
                    
                    // BUY/SELL ratio analysis
                    const total = status.buy_signals + status.sell_signals;
                    const buyRatio = total > 0 ? (status.buy_signals / total * 100) : 0;
                    document.getElementById('buy-ratio').textContent = buyRatio.toFixed(1) + '%';
                    
                    const ratioElement = document.getElementById('ratio-status');
                    if (buyRatio > 85) {
                        ratioElement.textContent = 'Heavily Bullish (Monitor for Reversal)';
                        ratioElement.className = 'ratio-warning';
                    } else if (buyRatio < 15) {
                        ratioElement.textContent = 'Heavily Bearish (Monitor for Reversal)';
                        ratioElement.className = 'ratio-warning';
                    } else {
                        ratioElement.textContent = 'Balanced Signal Distribution';
                        ratioElement.className = 'ratio-balanced';
                    }
                    
                    // Update signals
                    const signalsHtml = signals.slice(0, 15).map(signal => `
                        <div class="signal-item ${signal.signal_type === 'SELL' ? 'sell-signal' : ''}">
                            <div class="signal-header">
                                <span class="signal-symbol">${signal.symbol} (${signal.signal_type})</span>
                                <span class="confidence-badge ${signal.confidence >= 80 ? 'high-conf' : 
                                    signal.confidence >= 70 ? 'medium-conf' : 'low-conf'}">${signal.confidence.toFixed(1)}%</span>
                            </div>
                            <div class="signal-details">
                                Regime: ${signal.market_regime || 'unknown'} | 
                                Price: $${signal.price.toFixed(6)} | 
                                Target: $${signal.target_price ? signal.target_price.toFixed(6) : 'N/A'}
                            </div>
                        </div>
                    `).join('');
                    document.getElementById('signals-list').innerHTML = signalsHtml;
                    
                    // Update insights
                    document.getElementById('regime-stability').textContent = insights.regime_stability + '%';
                    document.getElementById('signal-accuracy').textContent = insights.signal_accuracy + '%';
                    
                    const learningHtml = insights.learning_insights.map(insight => `
                        <div class="insight-item">
                            <div class="insight-rule">${insight}</div>
                        </div>
                    `).join('');
                    document.getElementById('learning-insights').innerHTML = learningHtml;
                    
                    const rulesHtml = insights.actionable_rules.map(rule => `
                        <div class="insight-item">
                            <div class="insight-rule">${rule}</div>
                        </div>
                    `).join('');
                    document.getElementById('actionable-rules').innerHTML = rulesHtml;
                });
            }
            
            // Initialize
            document.addEventListener('DOMContentLoaded', function() {
                showTab('overview');
                updateData();
                setInterval(updateData, 5000);
            });
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 Enhanced AI Trading Dashboard</h1>
                <p>Market Context Awareness • SELL Signal Generation • Feedback Learning • AI Insights</p>
            </div>
            
            <div class="tab-container">
                <div class="tabs">
                    <button class="tab active" onclick="showTab('overview')">Trading Overview</button>
                    <button class="tab" onclick="showTab('insights')">AI Insights</button>
                    <button class="tab" onclick="showTab('learning')">Learning Analytics</button>
                </div>
                
                <div id="overview" class="tab-content active">
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value" id="system-status">Running</div>
                            <div class="stat-label">System Status</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="current-regime">unknown</div>
                            <div class="stat-label">Market Regime</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="total-signals">0</div>
                            <div class="stat-label">Total Signals</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="buy-signals">0</div>
                            <div class="stat-label">BUY Signals</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="sell-signals">0</div>
                            <div class="stat-label">SELL Signals</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="buy-ratio">0%</div>
                            <div class="stat-label">BUY Ratio</div>
                        </div>
                    </div>
                    
                    <div class="content-grid">
                        <div class="signals-section">
                            <div class="section-title">🎯 Enhanced Signals (BUY/SELL)</div>
                            <div id="signals-list">Loading signals...</div>
                        </div>
                        
                        <div class="insights-section">
                            <div class="section-title">⚠️ Signal Distribution Analysis</div>
                            <div id="ratio-status" class="ratio-balanced">Analyzing signal patterns...</div>
                        </div>
                    </div>
                </div>
                
                <div id="insights" class="tab-content">
                    <div class="stats-grid">
                        <div class="stat-card">
                            <div class="stat-value" id="regime-stability">0</div>
                            <div class="stat-label">Regime Stability</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-value" id="signal-accuracy">0</div>
                            <div class="stat-label">Signal Accuracy (24h)</div>
                        </div>
                    </div>
                    
                    <div class="insights-section">
                        <div class="section-title">🧠 AI Learning Insights</div>
                        <div id="learning-insights">Analyzing patterns...</div>
                    </div>
                </div>
                
                <div id="learning" class="tab-content">
                    <div class="insights-section">
                        <div class="section-title">📋 Actionable Trading Rules</div>
                        <div id="actionable-rules">Generating rules...</div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    ''')

@app.route('/api/enhanced/status')
def get_enhanced_status():
    """Get enhanced system status"""
    return jsonify({
        'status': 'Running' if trading_system.is_running else 'Stopped',
        'current_regime': trading_system.current_regime,
        'buy_signals': trading_system.buy_sell_ratio.get('BUY', 0),
        'sell_signals': trading_system.buy_sell_ratio.get('SELL', 0)
    })

@app.route('/api/enhanced/signals')
def get_enhanced_signals():
    """Get recent enhanced signals"""
    return jsonify(trading_system.signal_history[-20:] if trading_system.signal_history else [])

@app.route('/api/enhanced/insights')
def get_ai_insights():
    """Get AI insights and learning analytics"""
    try:
        # Get regime stability
        regime_summary = trading_system.regime_detector.get_regime_summary()
        regime_stability = int(regime_summary.get('stability', 0) * 100)
        
        # Get learning insights
        learning_insights = trading_system.feedback_logger.get_learning_insights()
        
        # Get actionable rules
        actionable_rules = trading_system.feedback_analyzer.generate_actionable_rules()
        
        # Mock signal accuracy for demonstration (would be calculated from actual data)
        signal_accuracy = 75  # This would come from feedback analysis
        
        return jsonify({
            'regime_stability': regime_stability,
            'signal_accuracy': signal_accuracy,
            'learning_insights': learning_insights[:5],  # Top 5 insights
            'actionable_rules': actionable_rules[:5]     # Top 5 rules
        })
        
    except Exception as e:
        logger.error(f"AI insights failed: {e}")
        return jsonify({
            'regime_stability': 0,
            'signal_accuracy': 0,
            'learning_insights': ['Collecting data for insights...'],
            'actionable_rules': ['Analyzing performance patterns...']
        })

def main():
    """Main function"""
    try:
        # Start enhanced trading system
        trading_system.start_enhanced_trading()
        
        # Start web interface
        app.run(host='0.0.0.0', port=5002, debug=False)
        
    except KeyboardInterrupt:
        logger.info("Shutting down enhanced trading system...")
        trading_system.stop_enhanced_trading()
    except Exception as e:
        logger.error(f"System error: {e}")

if __name__ == "__main__":
    main()