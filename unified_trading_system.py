"""
Unified Trading System - All functions on port 5000
Complete trading platform with signal generation, execution, monitoring, and analytics
"""
import os
import sqlite3
import ccxt
import pandas as pd
import pandas_ta as ta
import numpy as np
from flask import Flask, render_template, jsonify, request
from datetime import datetime, timedelta
import threading
import time
import logging
from typing import Dict, List, Optional
import json

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedTradingSystem:
    """Complete trading system with all functions unified"""
    
    def __init__(self):
        self.exchange = None
        self.is_running = False
        
        # Trading parameters (aggressive settings)
        self.min_confidence = 70.0
        self.position_size_pct = 0.05  # 5%
        self.stop_loss_pct = 12.0
        self.take_profit_pct = 20.0
        self.min_trade_usd = 5.0
        
        # Initialize with expanded symbol list (100 symbols under $200)
        self.symbols = self.get_symbols_under_200_usdt()
        
        self.executed_signals = set()
        self.live_trading_enabled = True
        self.initialize_system()
    
    def get_symbols_under_200_usdt(self) -> List[str]:
        """Get 100 cryptocurrency symbols with price under $200 USDT"""
        try:
            if not self.exchange:
                # Initialize exchange temporarily for symbol fetching
                temp_exchange = ccxt.okx({
                    'apiKey': os.environ.get('OKX_API_KEY'),
                    'secret': os.environ.get('OKX_SECRET_KEY'),
                    'password': os.environ.get('OKX_PASSPHRASE'),
                    'sandbox': False,
                    'enableRateLimit': True,
                })
                markets = temp_exchange.load_markets()
                tickers = temp_exchange.fetch_tickers()
            else:
                markets = self.exchange.load_markets()
                tickers = self.exchange.fetch_tickers()
            
            # Filter USDT pairs with price under $200
            valid_symbols = []
            for symbol in markets:
                if '/USDT' in symbol and symbol in tickers:
                    ticker = tickers[symbol]
                    price = ticker['last']
                    
                    if price and 0.001 <= price <= 200:  # Price between $0.001 and $200
                        market = markets[symbol]
                        if market['active'] and market['spot']:  # Active spot trading
                            valid_symbols.append(symbol)
            
            # Sort by 24h volume (descending) and take top 100
            valid_symbols.sort(key=lambda s: tickers[s]['quoteVolume'] or 0, reverse=True)
            selected_symbols = valid_symbols[:100]
            
            logger.info(f"Selected {len(selected_symbols)} symbols under $200 for trading")
            return selected_symbols
            
        except Exception as e:
            logger.error(f"Error fetching symbols: {e}")
            # Fallback to expanded symbol list
            return [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT', 'SOL/USDT', 'DOGE/USDT',
                'LINK/USDT', 'LTC/USDT', 'DOT/USDT', 'AVAX/USDT', 'UNI/USDT', 'ATOM/USDT', 'NEAR/USDT',
                'TRX/USDT', 'ICP/USDT', 'ALGO/USDT', 'HBAR/USDT', 'XLM/USDT', 'SAND/USDT', 'MANA/USDT',
                'THETA/USDT', 'AXS/USDT', 'FIL/USDT', 'ETC/USDT', 'EGLD/USDT', 'FLOW/USDT', 'ENJ/USDT',
                'CHZ/USDT', 'CRV/USDT', 'MATIC/USDT', 'VET/USDT', 'FTM/USDT', 'ONE/USDT', 'LUNA/USDT',
                'ZEC/USDT', 'DASH/USDT', 'XMR/USDT', 'NEO/USDT', 'IOTA/USDT', 'ZIL/USDT', 'ONT/USDT',
                'ICX/USDT', 'QTUM/USDT', 'LSK/USDT', 'NANO/USDT', 'DGB/USDT', 'SC/USDT', 'ZEN/USDT',
                'DCR/USDT', 'BAT/USDT', 'REP/USDT', 'KNC/USDT', 'ZRX/USDT', 'LRC/USDT', 'REN/USDT',
                'STORJ/USDT', 'GNT/USDT', 'CVC/USDT', 'GTO/USDT', 'CTR/USDT', 'RCN/USDT', 'RDN/USDT',
                'MCO/USDT', 'ICN/USDT', 'AMB/USDT', 'BCPT/USDT', 'CND/USDT', 'DLT/USDT', 'GAS/USDT',
                'POWR/USDT', 'BQX/USDT', 'SNT/USDT', 'BNT/USDT', 'GAS/USDT', 'HSR/USDT', 'OAX/USDT',
                'DNT/USDT', 'MCO/USDT', 'ICN/USDT', 'WTC/USDT', 'LLT/USDT', 'YOYO/USDT', 'LRC/USDT',
                'OST/USDT', 'BRD/USDT', 'TNB/USDT', 'FUEL/USDT', 'MAID/USDT', 'AST/USDT', 'BTM/USDT',
                'BCPT/USDT', 'ARN/USDT', 'GVT/USDT', 'CDT/USDT', 'GXS/USDT', 'POE/USDT', 'QSP/USDT',
                'BTS/USDT', 'XZC/USDT', 'LSK/USDT', 'TNT/USDT', 'FUEL/USDT', 'MAID/USDT', 'AST/USDT'
            ]
    
    def initialize_system(self):
        """Initialize complete trading system"""
        try:
            # Initialize OKX exchange
            self.exchange = ccxt.okx({
                'apiKey': os.environ.get('OKX_API_KEY'),
                'secret': os.environ.get('OKX_SECRET_KEY'),
                'password': os.environ.get('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            # Test connection
            balance = self.exchange.fetch_balance()
            logger.info("Unified system connected to OKX")
            
            # Setup database
            self.setup_database()
            
            # Start trading engine
            self.start_trading_engine()
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.exchange = None
    
    def setup_database(self):
        """Setup unified database schema"""
        conn = sqlite3.connect('unified_trading.db')
        cursor = conn.cursor()
        
        # Signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                signal_type TEXT,
                confidence REAL,
                price REAL,
                target_price REAL,
                stop_loss REAL,
                technical_score REAL,
                volume_score REAL,
                momentum_score REAL,
                executed BOOLEAN DEFAULT 0
            )
        ''')
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                amount REAL,
                price REAL,
                order_id TEXT,
                confidence REAL,
                status TEXT,
                pnl REAL DEFAULT 0
            )
        ''')
        
        # Performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                total_trades INTEGER,
                winning_trades INTEGER,
                total_pnl REAL,
                win_rate REAL,
                avg_profit REAL,
                max_drawdown REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Unified database initialized")
    
    def start_trading_engine(self):
        """Start background trading engine"""
        self.is_running = True
        trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        trading_thread.start()
        logger.info("Unified trading engine started")
    
    def trading_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                # Generate signals
                signals = self.generate_signals()
                
                # Execute high confidence signals
                for signal in signals:
                    if signal['confidence'] >= self.min_confidence:
                        self.execute_signal(signal)
                
                # Wait before next scan  
                time.sleep(20)  # 20 second intervals for faster processing
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(60)
    
    def generate_signals(self) -> List[Dict]:
        """Generate trading signals for all symbols"""
        signals = []
        
        for symbol in self.symbols:
            try:
                # Get market data
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=200)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Calculate indicators
                df = self.calculate_indicators(df)
                
                # Generate signal
                signal = self.analyze_signal(symbol, df)
                
                if signal and signal['confidence'] >= self.min_confidence:
                    signals.append(signal)
                    self.save_signal(signal)
                
            except Exception as e:
                logger.error(f"Signal generation failed for {symbol}: {e}")
        
        return signals
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # MACD
        macd = ta.macd(df['close'])
        df = pd.concat([df, macd], axis=1)
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20)
        df = pd.concat([df, bb], axis=1)
        
        # EMA
        df['ema_20'] = ta.ema(df['close'], length=20)
        df['ema_50'] = ta.ema(df['close'], length=50)
        
        # Volume indicators
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        return df
    
    def analyze_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Analyze data to generate trading signal"""
        if len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Technical scoring
        technical_score = 0
        signals = []
        
        # RSI analysis
        if latest['rsi'] < 30:
            technical_score += 20
            signals.append("RSI oversold")
        elif latest['rsi'] > 70:
            technical_score -= 20
            signals.append("RSI overbought")
        
        # MACD analysis
        if latest['MACD_12_26_9'] > latest['MACDs_12_26_9']:
            technical_score += 15
            signals.append("MACD bullish")
        
        # Price vs EMA
        if latest['close'] > latest['ema_20'] > latest['ema_50']:
            technical_score += 20
            signals.append("Price above EMAs")
        
        # Bollinger Bands
        if latest['close'] < latest['BBL_20_2.0']:
            technical_score += 15
            signals.append("Below lower BB")
        
        # Volume analysis
        volume_score = min(latest['volume_ratio'] * 10, 20)
        technical_score += volume_score
        
        # Momentum analysis
        momentum_score = 0
        if latest['close'] > prev['close']:
            momentum_score += 10
        
        # Calculate final confidence
        total_score = technical_score + momentum_score
        confidence = min(max(total_score, 0), 100)
        
        if confidence >= self.min_confidence:
            signal = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'signal_type': 'BUY' if confidence >= self.min_confidence else 'HOLD',
                'confidence': confidence,
                'price': latest['close'],
                'target_price': latest['close'] * (1 + self.take_profit_pct / 100),
                'stop_loss': latest['close'] * (1 - self.stop_loss_pct / 100),
                'technical_score': technical_score,
                'volume_score': volume_score,
                'momentum_score': momentum_score,
                'signals': signals
            }
            
            return signal
        
        return None
    
    def execute_signal(self, signal: Dict) -> bool:
        """Execute trading signal"""
        if not self.exchange:
            return False
        
        try:
            symbol = signal['symbol']
            signal_key = f"{symbol}_{signal['timestamp']}"
            
            if signal_key in self.executed_signals:
                return False
            
            # Calculate position size
            balance = self.exchange.fetch_balance()
            usdt_balance = float(balance.get('USDT', {}).get('free', 0))
            position_value = usdt_balance * self.position_size_pct
            
            if position_value < self.min_trade_usd:
                return False
            
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate quantity
            quantity = position_value / current_price
            quantity = self.exchange.amount_to_precision(symbol, quantity)
            
            # Execute trade
            order = self.exchange.create_market_buy_order(symbol, quantity)
            
            # Record trade
            self.executed_signals.add(signal_key)
            self.save_trade({
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': 'BUY',
                'amount': float(quantity),
                'price': current_price,
                'order_id': order.get('id', ''),
                'confidence': signal['confidence'],
                'status': 'EXECUTED'
            })
            
            logger.info(f"✅ EXECUTED: BUY {quantity} {symbol} @ ${current_price} ({signal['confidence']:.1f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            return False
    
    def save_signal(self, signal: Dict):
        """Save signal to database"""
        try:
            conn = sqlite3.connect('unified_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO signals 
                (timestamp, symbol, signal_type, confidence, price, target_price, stop_loss, 
                 technical_score, volume_score, momentum_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['timestamp'], signal['symbol'], signal['signal_type'],
                signal['confidence'], signal['price'], signal['target_price'],
                signal['stop_loss'], signal['technical_score'],
                signal['volume_score'], signal['momentum_score']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving signal: {e}")
    
    def save_trade(self, trade: Dict):
        """Save trade to database"""
        try:
            conn = sqlite3.connect('unified_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades 
                (timestamp, symbol, side, amount, price, order_id, confidence, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade['timestamp'], trade['symbol'], trade['side'],
                trade['amount'], trade['price'], trade['order_id'],
                trade['confidence'], trade['status']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error saving trade: {e}")

# Initialize unified system
unified_system = UnifiedTradingSystem()

# Web interface routes
@app.route('/')
def dashboard():
    """Main dashboard"""
    return render_template('unified_dashboard.html')

@app.route('/api/signals')
def get_signals():
    """Get recent signals"""
    try:
        conn = sqlite3.connect('unified_trading.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, signal_type, confidence, price, target_price, timestamp
            FROM signals 
            WHERE timestamp > datetime('now', '-2 hours')
            ORDER BY confidence DESC LIMIT 20
        ''')
        
        signals = []
        for row in cursor.fetchall():
            signals.append({
                'symbol': row[0],
                'signal': row[1],
                'confidence': row[2],
                'price': row[3],
                'target': row[4],
                'timestamp': row[5]
            })
        
        conn.close()
        return jsonify(signals)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/trades')
def get_trades():
    """Get recent trades"""
    try:
        conn = sqlite3.connect('unified_trading.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, side, amount, price, confidence, timestamp, status
            FROM trades 
            ORDER BY timestamp DESC LIMIT 20
        ''')
        
        trades = []
        for row in cursor.fetchall():
            trades.append({
                'symbol': row[0],
                'side': row[1],
                'amount': row[2],
                'price': row[3],
                'confidence': row[4],
                'timestamp': row[5],
                'status': row[6]
            })
        
        conn.close()
        return jsonify(trades)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/portfolio')
def get_portfolio():
    """Get portfolio data"""
    try:
        if unified_system.exchange:
            balance = unified_system.exchange.fetch_balance()
            portfolio = []
            
            for currency, data in balance.items():
                if data['total'] > 0:
                    portfolio.append({
                        'currency': currency,
                        'total': data['total'],
                        'free': data['free'],
                        'used': data['used']
                    })
            
            return jsonify(portfolio)
        else:
            return jsonify([])
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/performance')
def get_performance():
    """Get trading performance"""
    try:
        conn = sqlite3.connect('unified_trading.db')
        cursor = conn.cursor()
        
        # Calculate basic performance metrics
        cursor.execute('SELECT COUNT(*) FROM trades')
        total_trades = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(confidence) FROM trades')
        avg_confidence = cursor.fetchone()[0] or 0
        
        # Get recent performance
        cursor.execute('''
            SELECT COUNT(*) FROM trades 
            WHERE timestamp > datetime('now', '-24 hours')
        ''')
        daily_trades = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_trades': total_trades,
            'daily_trades': daily_trades,
            'avg_confidence': round(avg_confidence, 1),
            'system_status': 'ACTIVE' if unified_system.is_running else 'INACTIVE'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'exchange_connected': unified_system.exchange is not None,
        'trading_active': unified_system.is_running,
        'symbols_monitored': len(unified_system.symbols),
        'min_confidence': unified_system.min_confidence,
        'position_size': f"{unified_system.position_size_pct * 100}%",
        'last_update': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Create templates directory and basic template
    os.makedirs('templates', exist_ok=True)
    
    # Create basic dashboard template
    dashboard_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>Unified Trading System</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a1a; color: #fff; }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { background: #2a2a2a; border-radius: 8px; padding: 20px; border-left: 4px solid #00ff88; }
        .metric { font-size: 24px; font-weight: bold; color: #00ff88; }
        .label { color: #ccc; margin-bottom: 5px; }
        .signal { padding: 10px; margin: 5px 0; background: #333; border-radius: 4px; }
        .buy { border-left: 4px solid #00ff88; }
        .status-active { color: #00ff88; }
        .status-inactive { color: #ff4444; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #444; }
        th { background: #333; }
        .refresh-btn { background: #00ff88; color: #000; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Unified Trading System</h1>
            <button class="refresh-btn" onclick="refreshAll()">Refresh All</button>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3>System Status</h3>
                <div id="status-content">Loading...</div>
            </div>
            
            <div class="card">
                <h3>Performance</h3>
                <div id="performance-content">Loading...</div>
            </div>
            
            <div class="card">
                <h3>Recent Signals</h3>
                <div id="signals-content">Loading...</div>
            </div>
            
            <div class="card">
                <h3>Recent Trades</h3>
                <div id="trades-content">Loading...</div>
            </div>
            
            <div class="card">
                <h3>Portfolio</h3>
                <div id="portfolio-content">Loading...</div>
            </div>
        </div>
    </div>

    <script>
        function refreshAll() {
            loadStatus();
            loadPerformance();
            loadSignals();
            loadTrades();
            loadPortfolio();
        }
        
        function loadStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status-content').innerHTML = `
                        <div class="label">Exchange Connected:</div>
                        <div class="metric ${data.exchange_connected ? 'status-active' : 'status-inactive'}">
                            ${data.exchange_connected ? 'CONNECTED' : 'DISCONNECTED'}
                        </div>
                        <div class="label">Trading Active:</div>
                        <div class="metric ${data.trading_active ? 'status-active' : 'status-inactive'}">
                            ${data.trading_active ? 'ACTIVE' : 'INACTIVE'}
                        </div>
                        <div class="label">Symbols Monitored:</div>
                        <div class="metric">${data.symbols_monitored}</div>
                        <div class="label">Min Confidence:</div>
                        <div class="metric">${data.min_confidence}%</div>
                        <div class="label">Position Size:</div>
                        <div class="metric">${data.position_size}</div>
                    `;
                })
                .catch(error => {
                    document.getElementById('status-content').innerHTML = 'Error loading status';
                });
        }
        
        function loadPerformance() {
            fetch('/api/performance')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('performance-content').innerHTML = `
                        <div class="label">Total Trades:</div>
                        <div class="metric">${data.total_trades}</div>
                        <div class="label">Today's Trades:</div>
                        <div class="metric">${data.daily_trades}</div>
                        <div class="label">Avg Confidence:</div>
                        <div class="metric">${data.avg_confidence}%</div>
                        <div class="label">System Status:</div>
                        <div class="metric status-active">${data.system_status}</div>
                    `;
                });
        }
        
        function loadSignals() {
            fetch('/api/signals')
                .then(response => response.json())
                .then(data => {
                    let html = '';
                    data.forEach(signal => {
                        html += `
                            <div class="signal buy">
                                <strong>${signal.symbol}</strong> - ${signal.signal}<br>
                                Confidence: ${signal.confidence.toFixed(1)}%<br>
                                Price: $${signal.price.toFixed(4)}
                            </div>
                        `;
                    });
                    document.getElementById('signals-content').innerHTML = html || 'No recent signals';
                });
        }
        
        function loadTrades() {
            fetch('/api/trades')
                .then(response => response.json())
                .then(data => {
                    let html = '<table><tr><th>Symbol</th><th>Side</th><th>Amount</th><th>Price</th><th>Confidence</th></tr>';
                    data.forEach(trade => {
                        html += `
                            <tr>
                                <td>${trade.symbol}</td>
                                <td>${trade.side}</td>
                                <td>${trade.amount.toFixed(4)}</td>
                                <td>$${trade.price.toFixed(4)}</td>
                                <td>${trade.confidence.toFixed(1)}%</td>
                            </tr>
                        `;
                    });
                    html += '</table>';
                    document.getElementById('trades-content').innerHTML = html;
                });
        }
        
        function loadPortfolio() {
            fetch('/api/portfolio')
                .then(response => response.json())
                .then(data => {
                    let html = '<table><tr><th>Currency</th><th>Total</th><th>Free</th></tr>';
                    data.forEach(item => {
                        html += `
                            <tr>
                                <td>${item.currency}</td>
                                <td>${item.total.toFixed(4)}</td>
                                <td>${item.free.toFixed(4)}</td>
                            </tr>
                        `;
                    });
                    html += '</table>';
                    document.getElementById('portfolio-content').innerHTML = html;
                });
        }
        
        // Auto-refresh every 30 seconds
        setInterval(refreshAll, 30000);
        
        // Initial load
        refreshAll();
    </script>
</body>
</html>
    '''
    
    with open('templates/unified_dashboard.html', 'w') as f:
        f.write(dashboard_html)
    
    logger.info("🚀 Starting Unified Trading System on port 5000")
    logger.info("Features: Signal Generation, Auto Execution, Portfolio, Analytics")
    app.run(host='0.0.0.0', port=5000, debug=False)