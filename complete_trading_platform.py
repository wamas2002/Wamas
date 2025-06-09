"""
Complete AI-Powered Trading Platform with TradingView Integration
Full trading system functionality with real OKX data and TradingView widgets
"""

from flask import Flask, render_template, jsonify, request
import logging
import json
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import os
from typing import Dict, List, Optional
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'complete-trading-platform-2024'

class OKXDataService:
    """Real OKX data service for live market data"""
    
    def __init__(self):
        self.exchange = None
        self.initialize_exchange()
        
    def initialize_exchange(self):
        """Initialize OKX exchange connection"""
        try:
            self.exchange = ccxt.okx({
                'sandbox': False,
                'rateLimit': 1200,
                'enableRateLimit': True,
            })
            logger.info("OKX exchange connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OKX: {e}")
            self.exchange = None

    def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        try:
            if self.exchange:
                ticker = self.exchange.fetch_ticker(symbol)
                return float(ticker['last'])
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return 0.0

    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> List[Dict]:
        """Get OHLCV market data"""
        try:
            if self.exchange:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                data = []
                for candle in ohlcv:
                    data.append({
                        'timestamp': candle[0],
                        'open': candle[1],
                        'high': candle[2],
                        'low': candle[3],
                        'close': candle[4],
                        'volume': candle[5]
                    })
                return data
            return []
        except Exception as e:
            logger.error(f"Error fetching market data for {symbol}: {e}")
            return []

    def get_portfolio_balance(self) -> Dict:
        """Get portfolio balance (mock for demo)"""
        return {
            'total_balance': 125840.50,
            'available_balance': 23450.75,
            'positions': [
                {
                    'symbol': 'BTC/USDT',
                    'quantity': 1.85,
                    'current_price': 46800.0,
                    'current_value': 86580.0,
                    'pnl': 2980.50,
                    'pnl_percentage': 3.54
                },
                {
                    'symbol': 'ETH/USDT',
                    'quantity': 12.4,
                    'current_price': 2580.0,
                    'current_value': 31992.0,
                    'pnl': 1992.0,
                    'pnl_percentage': 6.61
                },
                {
                    'symbol': 'BNB/USDT',
                    'quantity': 15.2,
                    'current_price': 325.0,
                    'current_value': 4940.0,
                    'pnl': 228.0,
                    'pnl_percentage': 4.84
                }
            ]
        }

class DatabaseManager:
    """Database operations for trading platform"""
    
    def __init__(self):
        self.db_path = 'trading_platform.db'
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_price REAL NOT NULL,
                current_price REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'completed'
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                reasoning TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")

class AITradingEngine:
    """AI-powered trading signal generation"""
    
    def __init__(self, data_service):
        self.data_service = data_service
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
    
    def generate_signals(self) -> List[Dict]:
        """Generate AI trading signals"""
        signals = []
        
        for symbol in self.symbols:
            try:
                # Get market data
                market_data = self.data_service.get_market_data(symbol)
                if not market_data:
                    continue
                
                # Simple AI logic (in production, use sophisticated ML models)
                df = pd.DataFrame(market_data)
                if len(df) < 20:
                    continue
                
                # Calculate technical indicators
                df['sma_10'] = df['close'].rolling(10).mean()
                df['sma_20'] = df['close'].rolling(20).mean()
                df['rsi'] = self.calculate_rsi(df['close'])
                
                current = df.iloc[-1]
                prev = df.iloc[-2]
                
                # Generate signal
                signal_data = self.analyze_signals(current, prev, symbol)
                signals.append(signal_data)
                
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
        
        return signals
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def analyze_signals(self, current, prev, symbol):
        """Analyze technical signals and generate recommendation"""
        
        # Signal logic
        signal = "HOLD"
        confidence = 50.0
        reasoning = "Neutral market conditions"
        
        if current['close'] > current['sma_20'] and prev['close'] <= prev['sma_20']:
            signal = "BUY"
            confidence = 75.0
            reasoning = "Price broke above 20-period SMA"
        elif current['close'] < current['sma_20'] and prev['close'] >= prev['sma_20']:
            signal = "SELL"
            confidence = 70.0
            reasoning = "Price broke below 20-period SMA"
        elif current['rsi'] < 30:
            signal = "BUY"
            confidence = 80.0
            reasoning = "RSI indicates oversold conditions"
        elif current['rsi'] > 70:
            signal = "SELL"
            confidence = 85.0
            reasoning = "RSI indicates overbought conditions"
        
        return {
            'symbol': symbol.replace('/USDT', ''),
            'signal': signal,
            'confidence': confidence,
            'reasoning': reasoning,
            'current_price': current['close'],
            'rsi': current['rsi'],
            'timestamp': datetime.now()
        }

class TradingViewManager:
    """Manage TradingView widget configurations"""
    
    def __init__(self):
        self.symbols = {
            'BTC': 'OKX:BTCUSDT',
            'ETH': 'OKX:ETHUSDT',
            'BNB': 'OKX:BNBUSDT',
            'ADA': 'OKX:ADAUSDT',
            'SOL': 'OKX:SOLUSDT',
            'XRP': 'OKX:XRPUSDT',
            'DOT': 'OKX:DOTUSDT',
            'AVAX': 'OKX:AVAXUSDT'
        }
    
    def get_widget_config(self, symbol='BTCUSDT', **kwargs):
        """Generate TradingView widget configuration"""
        tv_symbol = f"OKX:{symbol}" if not symbol.startswith('OKX:') else symbol
        
        return {
            "symbol": tv_symbol,
            "interval": kwargs.get('interval', '15'),
            "timezone": "Etc/UTC",
            "theme": kwargs.get('theme', 'dark'),
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#252D3D",
            "enable_publishing": False,
            "withdateranges": True,
            "hide_side_toolbar": kwargs.get('hide_side_toolbar', False),
            "allow_symbol_change": True,
            "container_id": kwargs.get('container_id', 'tradingview_widget'),
            "width": kwargs.get('width', '100%'),
            "height": kwargs.get('height', 500)
        }
    
    def get_symbol_list(self):
        """Get available trading symbols"""
        return list(self.symbols.values())

# Initialize services
data_service = OKXDataService()
db_manager = DatabaseManager()
ai_engine = AITradingEngine(data_service)
tv_manager = TradingViewManager()

# Background AI signal generation
def generate_ai_signals_background():
    """Background task to generate AI signals"""
    while True:
        try:
            signals = ai_engine.generate_signals()
            # Save signals to database
            conn = sqlite3.connect(db_manager.db_path)
            cursor = conn.cursor()
            
            for signal in signals:
                cursor.execute('''
                    INSERT INTO ai_signals (symbol, signal, confidence, reasoning)
                    VALUES (?, ?, ?, ?)
                ''', (signal['symbol'], signal['signal'], signal['confidence'], signal['reasoning']))
            
            conn.commit()
            conn.close()
            logger.info(f"Generated {len(signals)} AI signals")
        except Exception as e:
            logger.error(f"Error in background AI signal generation: {e}")
        
        time.sleep(300)  # Run every 5 minutes

# Start background task
ai_thread = threading.Thread(target=generate_ai_signals_background, daemon=True)
ai_thread.start()

@app.route('/')
def dashboard():
    """Main dashboard with TradingView widgets"""
    try:
        widget_config = tv_manager.get_widget_config('BTCUSDT')
        return render_template('complete_dashboard.html', 
                             widget_config=json.dumps(widget_config),
                             symbols=tv_manager.get_symbol_list())
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return f"Error loading dashboard: {e}", 500

@app.route('/portfolio')
def portfolio():
    """Portfolio management page"""
    try:
        portfolio_data = data_service.get_portfolio_balance()
        widget_config = tv_manager.get_widget_config('BTCUSDT')
        return render_template('complete_portfolio.html',
                             portfolio_data=portfolio_data,
                             widget_config=json.dumps(widget_config),
                             symbols=tv_manager.get_symbol_list())
    except Exception as e:
        logger.error(f"Portfolio error: {e}")
        return f"Error loading portfolio: {e}", 500

@app.route('/trading')
def trading():
    """Trading interface page"""
    try:
        widget_config = tv_manager.get_widget_config('BTCUSDT')
        return render_template('complete_trading.html',
                             widget_config=json.dumps(widget_config),
                             symbols=tv_manager.get_symbol_list())
    except Exception as e:
        logger.error(f"Trading error: {e}")
        return f"Error loading trading interface: {e}", 500

@app.route('/ai-signals')
def ai_signals():
    """AI signals and analysis page"""
    try:
        widget_config = tv_manager.get_widget_config('BTCUSDT')
        return render_template('complete_ai_signals.html',
                             widget_config=json.dumps(widget_config),
                             symbols=tv_manager.get_symbol_list())
    except Exception as e:
        logger.error(f"AI signals error: {e}")
        return f"Error loading AI signals: {e}", 500

# API Endpoints
@app.route('/api/portfolio')
def api_portfolio():
    """Get portfolio data"""
    try:
        portfolio = data_service.get_portfolio_balance()
        return jsonify(portfolio)
    except Exception as e:
        logger.error(f"Portfolio API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals')
def api_signals():
    """Get latest AI signals"""
    try:
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, signal, confidence, reasoning, timestamp
            FROM ai_signals
            WHERE timestamp > datetime('now', '-1 hour')
            ORDER BY timestamp DESC
            LIMIT 20
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        signals = []
        for row in results:
            signals.append({
                'symbol': row[0],
                'signal': row[1],
                'confidence': row[2],
                'reasoning': row[3],
                'timestamp': row[4]
            })
        
        return jsonify(signals)
    except Exception as e:
        logger.error(f"Signals API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-data/<symbol>')
def api_market_data(symbol):
    """Get market data for symbol"""
    try:
        symbol_formatted = f"{symbol}/USDT" if not symbol.endswith('/USDT') else symbol
        market_data = data_service.get_market_data(symbol_formatted)
        return jsonify(market_data)
    except Exception as e:
        logger.error(f"Market data API error for {symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/place-order', methods=['POST'])
def api_place_order():
    """Place trading order (demo implementation)"""
    try:
        data = request.json
        symbol = data.get('symbol')
        side = data.get('side')
        quantity = float(data.get('quantity', 0))
        price = float(data.get('price', 0))
        
        # In production, this would place a real order
        # For demo, just save to database
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (symbol, side, quantity, price)
            VALUES (?, ?, ?, ?)
        ''', (symbol, side, quantity, price))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'{side.upper()} order placed for {quantity} {symbol} at ${price}',
            'order_id': cursor.lastrowid
        })
    except Exception as e:
        logger.error(f"Order placement error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def api_health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'complete-trading-platform',
        'timestamp': datetime.now().isoformat(),
        'features': [
            'Real OKX market data',
            'AI signal generation',
            'TradingView integration',
            'Portfolio management',
            'Order placement'
        ]
    })

if __name__ == '__main__':
    logger.info("Starting Complete AI-Powered Trading Platform")
    logger.info("Features: Real OKX data, AI signals, TradingView widgets, Portfolio management")
    logger.info("Starting server on port 5000")
    
    app.run(host='0.0.0.0', port=5000, debug=False)