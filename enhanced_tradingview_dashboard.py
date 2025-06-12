#!/usr/bin/env python3
"""
Enhanced TradingView-Style Dashboard
Professional charting and technical analysis with real-time data
"""

import sqlite3
import json
import os
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from flask import Flask, render_template, jsonify, request
import pandas_ta as ta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class TradingViewDashboard:
    def __init__(self):
        self.exchange = None
        self.initialize_exchange()
        self.setup_database()
        
        # TradingView-style configuration
        self.timeframes = ['5m', '15m', '1h', '4h', '1d']
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
        
    def initialize_exchange(self):
        """Initialize OKX exchange connection"""
        try:
            api_key = os.getenv('OKX_API_KEY')
            secret = os.getenv('OKX_SECRET_KEY')
            passphrase = os.getenv('OKX_PASSPHRASE')
            
            if api_key and secret and passphrase:
                self.exchange = ccxt.okx({
                    'apiKey': api_key,
                    'secret': secret,
                    'password': passphrase,
                    'sandbox': False,
                    'enableRateLimit': True,
                })
                logger.info("TradingView dashboard connected to OKX")
            else:
                logger.warning("OKX credentials not configured")
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
    
    def setup_database(self):
        """Setup enhanced database"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS chart_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timeframe, timestamp)
                    )
                ''')
                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS technical_indicators (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp INTEGER NOT NULL,
                        rsi REAL,
                        macd REAL,
                        macd_signal REAL,
                        bb_upper REAL,
                        bb_lower REAL,
                        ema_20 REAL,
                        ema_50 REAL,
                        volume_sma REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timeframe, timestamp)
                    )
                ''')
                
                conn.commit()
                logger.info("TradingView database initialized")
                
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def fetch_and_store_data(self, symbol, timeframe='1h', limit=500):
        """Fetch and store real market data"""
        if not self.exchange:
            return []
        
        try:
            # Fetch OHLCV data from OKX
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return []
            
            # Store raw data
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                for candle in ohlcv:
                    timestamp, open_price, high, low, close, volume = candle
                    
                    cursor.execute('''
                        INSERT OR REPLACE INTO chart_data 
                        (symbol, timeframe, timestamp, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (symbol, timeframe, timestamp, open_price, high, low, close, volume))
                
                conn.commit()
            
            # Calculate and store technical indicators
            self.calculate_and_store_indicators(symbol, timeframe)
            
            return ohlcv
            
        except Exception as e:
            logger.error(f"Data fetch failed for {symbol} {timeframe}: {e}")
            return []
    
    def calculate_and_store_indicators(self, symbol, timeframe):
        """Calculate and store technical indicators"""
        try:
            # Get stored chart data
            with sqlite3.connect('enhanced_trading.db') as conn:
                df = pd.read_sql_query('''
                    SELECT timestamp, open, high, low, close, volume 
                    FROM chart_data 
                    WHERE symbol = ? AND timeframe = ? 
                    ORDER BY timestamp
                ''', conn, params=(symbol, timeframe))
            
            if len(df) < 50:
                return
            
            # Calculate technical indicators
            df['rsi'] = ta.rsi(df['close'], length=14)
            
            # MACD
            macd_data = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if not macd_data.empty and len(macd_data.columns) >= 2:
                df['macd'] = macd_data.iloc[:, 0]
                df['macd_signal'] = macd_data.iloc[:, 1]
            else:
                df['macd'] = 0
                df['macd_signal'] = 0
            
            # Bollinger Bands
            bb_data = ta.bbands(df['close'], length=20, std=2)
            if not bb_data.empty and len(bb_data.columns) >= 3:
                df['bb_upper'] = bb_data.iloc[:, 0]
                df['bb_lower'] = bb_data.iloc[:, 2]
            else:
                df['bb_upper'] = df['close']
                df['bb_lower'] = df['close']
            
            # EMAs
            df['ema_20'] = ta.ema(df['close'], length=20)
            df['ema_50'] = ta.ema(df['close'], length=50)
            
            # Volume SMA
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            
            # Store indicators
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                for _, row in df.iterrows():
                    cursor.execute('''
                        INSERT OR REPLACE INTO technical_indicators 
                        (symbol, timeframe, timestamp, rsi, macd, macd_signal, 
                         bb_upper, bb_lower, ema_20, ema_50, volume_sma)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol, timeframe, row['timestamp'],
                        row['rsi'] if not pd.isna(row['rsi']) else None,
                        row['macd'] if not pd.isna(row['macd']) else None,
                        row['macd_signal'] if not pd.isna(row['macd_signal']) else None,
                        row['bb_upper'] if not pd.isna(row['bb_upper']) else None,
                        row['bb_lower'] if not pd.isna(row['bb_lower']) else None,
                        row['ema_20'] if not pd.isna(row['ema_20']) else None,
                        row['ema_50'] if not pd.isna(row['ema_50']) else None,
                        row['volume_sma'] if not pd.isna(row['volume_sma']) else None
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Indicator calculation failed for {symbol} {timeframe}: {e}")
    
    def get_chart_data(self, symbol, timeframe='1h', limit=200):
        """Get chart data with indicators"""
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                # Get chart data with indicators
                query = '''
                    SELECT 
                        c.timestamp, c.open, c.high, c.low, c.close, c.volume,
                        i.rsi, i.macd, i.macd_signal, i.bb_upper, i.bb_lower,
                        i.ema_20, i.ema_50, i.volume_sma
                    FROM chart_data c
                    LEFT JOIN technical_indicators i ON 
                        c.symbol = i.symbol AND 
                        c.timeframe = i.timeframe AND 
                        c.timestamp = i.timestamp
                    WHERE c.symbol = ? AND c.timeframe = ?
                    ORDER BY c.timestamp DESC
                    LIMIT ?
                '''
                
                df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
                
                if df.empty:
                    # Fetch fresh data if none available
                    self.fetch_and_store_data(symbol, timeframe)
                    df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
                
                # Convert to chart format
                chart_data = []
                for _, row in df.iterrows():
                    chart_data.append({
                        'timestamp': int(row['timestamp']),
                        'open': float(row['open']),
                        'high': float(row['high']),
                        'low': float(row['low']),
                        'close': float(row['close']),
                        'volume': float(row['volume']),
                        'rsi': float(row['rsi']) if row['rsi'] else None,
                        'macd': float(row['macd']) if row['macd'] else None,
                        'macd_signal': float(row['macd_signal']) if row['macd_signal'] else None,
                        'bb_upper': float(row['bb_upper']) if row['bb_upper'] else None,
                        'bb_lower': float(row['bb_lower']) if row['bb_lower'] else None,
                        'ema_20': float(row['ema_20']) if row['ema_20'] else None,
                        'ema_50': float(row['ema_50']) if row['ema_50'] else None
                    })
                
                return list(reversed(chart_data))  # Chronological order
                
        except Exception as e:
            logger.error(f"Chart data retrieval failed: {e}")
            return []
    
    def get_market_overview(self):
        """Get market overview with key metrics"""
        try:
            overview = []
            
            for symbol in self.symbols:
                try:
                    # Get latest data
                    ticker = self.exchange.fetch_ticker(symbol)
                    
                    # Get chart data for analysis
                    chart_data = self.get_chart_data(symbol, '1h', 50)
                    
                    if chart_data:
                        latest = chart_data[-1]
                        
                        # Calculate 24h change
                        price_24h_ago = chart_data[-24]['close'] if len(chart_data) >= 24 else latest['close']
                        change_24h = ((latest['close'] - price_24h_ago) / price_24h_ago * 100)
                        
                        # Determine trend
                        if latest['ema_20'] and latest['ema_50']:
                            trend = 'BULLISH' if latest['ema_20'] > latest['ema_50'] else 'BEARISH'
                        else:
                            trend = 'NEUTRAL'
                        
                        # RSI signal
                        rsi_signal = 'OVERSOLD' if latest['rsi'] and latest['rsi'] < 30 else \
                                   'OVERBOUGHT' if latest['rsi'] and latest['rsi'] > 70 else 'NEUTRAL'
                        
                        overview.append({
                            'symbol': symbol,
                            'price': latest['close'],
                            'change_24h': round(change_24h, 2),
                            'volume': ticker.get('quoteVolume', 0),
                            'trend': trend,
                            'rsi': round(latest['rsi'], 2) if latest['rsi'] else None,
                            'rsi_signal': rsi_signal
                        })
                
                except Exception as e:
                    logger.error(f"Overview failed for {symbol}: {e}")
                    continue
            
            return overview
            
        except Exception as e:
            logger.error(f"Market overview failed: {e}")
            return []
    
    def analyze_trading_signals(self):
        """Generate trading signals based on technical analysis"""
        signals = []
        
        for symbol in self.symbols:
            try:
                chart_data = self.get_chart_data(symbol, '1h', 100)
                
                if len(chart_data) < 50:
                    continue
                
                latest = chart_data[-1]
                
                # Signal calculation
                signal_strength = 50  # Base
                action = 'HOLD'
                
                # RSI signals
                if latest['rsi']:
                    if latest['rsi'] < 30:
                        signal_strength += 25
                        action = 'BUY'
                    elif latest['rsi'] > 70:
                        signal_strength += 25
                        action = 'SELL'
                    elif latest['rsi'] < 40:
                        signal_strength += 10
                        action = 'BUY'
                    elif latest['rsi'] > 60:
                        signal_strength += 10
                        action = 'SELL'
                
                # MACD signals
                if latest['macd'] and latest['macd_signal']:
                    if latest['macd'] > latest['macd_signal']:
                        signal_strength += 15
                        if action != 'SELL':
                            action = 'BUY'
                    else:
                        signal_strength += 10
                        if action != 'BUY':
                            action = 'SELL'
                
                # EMA trend
                if latest['ema_20'] and latest['ema_50']:
                    if latest['ema_20'] > latest['ema_50']:
                        signal_strength += 10
                        if action == 'HOLD':
                            action = 'BUY'
                    else:
                        signal_strength += 5
                        if action == 'HOLD':
                            action = 'SELL'
                
                # Only include signals above threshold
                if signal_strength >= 65:
                    signals.append({
                        'symbol': symbol,
                        'action': action,
                        'confidence': min(95, signal_strength),
                        'price': latest['close'],
                        'rsi': latest['rsi'],
                        'reasoning': f"Technical analysis: RSI {latest['rsi']:.1f}, MACD trend, EMA positioning"
                    })
            
            except Exception as e:
                logger.error(f"Signal analysis failed for {symbol}: {e}")
                continue
        
        return signals

# Initialize dashboard
dashboard = TradingViewDashboard()

# Flask routes
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('tradingview_dashboard.html')

@app.route('/api/chart/<symbol>')
def get_chart(symbol):
    """Get chart data for symbol"""
    timeframe = request.args.get('timeframe', '1h')
    limit = int(request.args.get('limit', 200))
    
    # Ensure fresh data
    dashboard.fetch_and_store_data(symbol, timeframe, limit)
    
    data = dashboard.get_chart_data(symbol, timeframe, limit)
    return jsonify(data)

@app.route('/api/market-overview')
def market_overview():
    """Get market overview"""
    overview = dashboard.get_market_overview()
    return jsonify(overview)

@app.route('/api/signals')
def trading_signals():
    """Get trading signals"""
    signals = dashboard.analyze_trading_signals()
    return jsonify(signals)

@app.route('/api/indicators/<symbol>')
def get_indicators(symbol):
    """Get technical indicators for symbol"""
    timeframe = request.args.get('timeframe', '1h')
    
    try:
        with sqlite3.connect('enhanced_trading.db') as conn:
            df = pd.read_sql_query('''
                SELECT timestamp, rsi, macd, macd_signal, bb_upper, bb_lower, ema_20, ema_50
                FROM technical_indicators 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC 
                LIMIT 100
            ''', conn, params=(symbol, timeframe))
            
            indicators = df.to_dict('records')
            return jsonify(indicators)
            
    except Exception as e:
        logger.error(f"Indicators fetch failed: {e}")
        return jsonify([])

@app.route('/api/refresh-data')
def refresh_data():
    """Refresh all market data"""
    try:
        updated_symbols = []
        
        for symbol in dashboard.symbols:
            for timeframe in ['1h', '4h']:
                data = dashboard.fetch_and_store_data(symbol, timeframe)
                if data:
                    updated_symbols.append(f"{symbol}_{timeframe}")
        
        return jsonify({
            'status': 'success',
            'updated': updated_symbols,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# HTML Template
@app.route('/templates/tradingview_dashboard.html')
def dashboard_template():
    """Return TradingView dashboard template"""
    template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TradingView Professional Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #1e1e1e; color: #fff; }
        
        .header { background: #2d2d30; padding: 15px 20px; border-bottom: 1px solid #404040; }
        .header h1 { color: #00d4aa; font-size: 24px; display: inline-block; }
        .header .refresh-btn { float: right; background: #00d4aa; color: #000; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; }
        
        .dashboard { display: grid; grid-template-columns: 1fr 300px; height: calc(100vh - 80px); }
        
        .main-content { padding: 20px; overflow-y: auto; }
        .sidebar { background: #252526; border-left: 1px solid #404040; padding: 20px; overflow-y: auto; }
        
        .chart-container { background: #2d2d30; border-radius: 8px; padding: 20px; margin-bottom: 20px; height: 500px; }
        .chart-title { font-size: 18px; margin-bottom: 15px; color: #00d4aa; }
        
        .market-overview { margin-bottom: 30px; }
        .overview-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .overview-card { background: #2d2d30; padding: 15px; border-radius: 8px; border-left: 4px solid #00d4aa; }
        .overview-card h3 { font-size: 16px; margin-bottom: 8px; }
        .overview-card .price { font-size: 20px; font-weight: bold; color: #00d4aa; }
        .overview-card .change { font-size: 14px; margin-top: 5px; }
        .positive { color: #4caf50; }
        .negative { color: #f44336; }
        
        .signals-panel h3 { margin-bottom: 15px; color: #00d4aa; }
        .signal-item { background: #2d2d30; padding: 12px; border-radius: 6px; margin-bottom: 10px; border-left: 4px solid #00d4aa; }
        .signal-action { font-weight: bold; }
        .signal-confidence { font-size: 12px; color: #888; }
        
        .indicators-section { margin-top: 20px; }
        .indicator-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #404040; }
        
        #chart { width: 100%; height: 400px; background: #1e1e1e; border-radius: 4px; }
        
        .loading { text-align: center; padding: 50px; color: #888; }
    </style>
</head>
<body>
    <div class="header">
        <h1>TradingView Professional Dashboard</h1>
        <button class="refresh-btn" onclick="refreshData()">Refresh Data</button>
    </div>
    
    <div class="dashboard">
        <div class="main-content">
            <div class="market-overview">
                <h2>Market Overview</h2>
                <div id="market-overview" class="overview-grid">
                    <div class="loading">Loading market data...</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">BTC/USDT Chart</div>
                <div id="chart">
                    <div class="loading">Loading chart...</div>
                </div>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="signals-panel">
                <h3>Trading Signals</h3>
                <div id="signals">
                    <div class="loading">Loading signals...</div>
                </div>
            </div>
            
            <div class="indicators-section">
                <h3>Technical Indicators</h3>
                <div id="indicators">
                    <div class="loading">Loading indicators...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Auto-refresh data every 30 seconds
        setInterval(loadAllData, 30000);
        
        // Load initial data
        loadAllData();
        
        function loadAllData() {
            loadMarketOverview();
            loadSignals();
            loadChart('BTC/USDT');
        }
        
        async function loadMarketOverview() {
            try {
                const response = await fetch('/api/market-overview');
                const data = await response.json();
                
                const container = document.getElementById('market-overview');
                container.innerHTML = data.map(item => `
                    <div class="overview-card">
                        <h3>${item.symbol}</h3>
                        <div class="price">$${item.price.toFixed(4)}</div>
                        <div class="change ${item.change_24h >= 0 ? 'positive' : 'negative'}">
                            ${item.change_24h >= 0 ? '+' : ''}${item.change_24h.toFixed(2)}%
                        </div>
                        <div style="font-size: 12px; margin-top: 5px;">
                            RSI: ${item.rsi ? item.rsi.toFixed(1) : 'N/A'} | ${item.trend}
                        </div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Failed to load market overview:', error);
            }
        }
        
        async function loadSignals() {
            try {
                const response = await fetch('/api/signals');
                const data = await response.json();
                
                const container = document.getElementById('signals');
                
                if (data.length === 0) {
                    container.innerHTML = '<div class="loading">No signals available</div>';
                    return;
                }
                
                container.innerHTML = data.map(signal => `
                    <div class="signal-item">
                        <div class="signal-action">${signal.symbol} - ${signal.action}</div>
                        <div class="signal-confidence">Confidence: ${signal.confidence.toFixed(1)}%</div>
                        <div style="font-size: 12px; margin-top: 5px;">${signal.reasoning}</div>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Failed to load signals:', error);
            }
        }
        
        async function loadChart(symbol) {
            try {
                const response = await fetch(`/api/chart/${symbol}?timeframe=1h&limit=100`);
                const data = await response.json();
                
                // Simple chart representation (would normally use TradingView widget or Chart.js)
                const container = document.getElementById('chart');
                
                if (data.length === 0) {
                    container.innerHTML = '<div class="loading">No chart data available</div>';
                    return;
                }
                
                const latest = data[data.length - 1];
                container.innerHTML = `
                    <div style="padding: 20px;">
                        <h3>${symbol}</h3>
                        <div style="font-size: 24px; color: #00d4aa;">$${latest.close.toFixed(4)}</div>
                        <div style="margin: 10px 0;">
                            <div>Open: $${latest.open.toFixed(4)}</div>
                            <div>High: $${latest.high.toFixed(4)}</div>
                            <div>Low: $${latest.low.toFixed(4)}</div>
                            <div>Volume: ${latest.volume.toFixed(0)}</div>
                        </div>
                        <div style="margin-top: 20px;">
                            <div>RSI: ${latest.rsi ? latest.rsi.toFixed(1) : 'N/A'}</div>
                            <div>MACD: ${latest.macd ? latest.macd.toFixed(4) : 'N/A'}</div>
                            <div>EMA 20: ${latest.ema_20 ? latest.ema_20.toFixed(4) : 'N/A'}</div>
                            <div>EMA 50: ${latest.ema_50 ? latest.ema_50.toFixed(4) : 'N/A'}</div>
                        </div>
                    </div>
                `;
                
                // Load indicators
                loadIndicators(symbol);
            } catch (error) {
                console.error('Failed to load chart:', error);
            }
        }
        
        async function loadIndicators(symbol) {
            try {
                const response = await fetch(`/api/indicators/${symbol}?timeframe=1h`);
                const data = await response.json();
                
                const container = document.getElementById('indicators');
                
                if (data.length === 0) {
                    container.innerHTML = '<div class="loading">No indicators available</div>';
                    return;
                }
                
                const latest = data[0];
                container.innerHTML = `
                    <div class="indicator-row">
                        <span>RSI (14)</span>
                        <span>${latest.rsi ? latest.rsi.toFixed(1) : 'N/A'}</span>
                    </div>
                    <div class="indicator-row">
                        <span>MACD</span>
                        <span>${latest.macd ? latest.macd.toFixed(4) : 'N/A'}</span>
                    </div>
                    <div class="indicator-row">
                        <span>EMA 20</span>
                        <span>${latest.ema_20 ? latest.ema_20.toFixed(4) : 'N/A'}</span>
                    </div>
                    <div class="indicator-row">
                        <span>EMA 50</span>
                        <span>${latest.ema_50 ? latest.ema_50.toFixed(4) : 'N/A'}</span>
                    </div>
                    <div class="indicator-row">
                        <span>BB Upper</span>
                        <span>${latest.bb_upper ? latest.bb_upper.toFixed(4) : 'N/A'}</span>
                    </div>
                    <div class="indicator-row">
                        <span>BB Lower</span>
                        <span>${latest.bb_lower ? latest.bb_lower.toFixed(4) : 'N/A'}</span>
                    </div>
                `;
            } catch (error) {
                console.error('Failed to load indicators:', error);
            }
        }
        
        async function refreshData() {
            const btn = document.querySelector('.refresh-btn');
            btn.textContent = 'Refreshing...';
            btn.disabled = true;
            
            try {
                await fetch('/api/refresh-data');
                await loadAllData();
            } catch (error) {
                console.error('Refresh failed:', error);
            } finally {
                btn.textContent = 'Refresh Data';
                btn.disabled = false;
            }
        }
    </script>
</body>
</html>
    '''
    return template

def main():
    """Main function to run TradingView dashboard"""
    if not dashboard.exchange:
        print("OKX connection required for TradingView dashboard")
        return
    
    # Initialize with fresh data
    logger.info("Initializing TradingView dashboard with fresh market data...")
    
    for symbol in dashboard.symbols:
        for timeframe in ['1h', '4h']:
            dashboard.fetch_and_store_data(symbol, timeframe)
    
    logger.info("TradingView dashboard ready")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5001, debug=False)

if __name__ == "__main__":
    main()