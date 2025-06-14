#!/usr/bin/env python3
"""
Dynamic Trading System with Adaptive Take Profit and SELL Signal Detection
Implements signal strength-based take profit (5-15%) and full BUY/SELL signal capability
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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import Flask, render_template_string, jsonify
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trading.log')
    ]
)
logger = logging.getLogger(__name__)

class DynamicTradingSystem:
    """Dynamic trading system with adaptive take profit and dual signal detection"""
    
    def __init__(self):
        self.exchange = None
        self.is_running = False
        
        # Dynamic trading parameters
        self.min_confidence = 70.0
        self.position_size_pct = 0.08  # 8% for $47+ trades with $597 balance
        self.stop_loss_pct = 12.0
        self.base_take_profit_pct = 8.0  # Base 8%, adjusts 5-15% based on signal strength
        self.min_trade_usd = 10.0
        self.max_daily_trades = 30
        
        # Symbol tracking
        self.symbols = []
        self.symbol_prices = {}
        self.executed_signals = set()
        
        # Initialize system
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize OKX connection and database"""
        try:
            # Connect to OKX
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            # Load markets and get symbols under $200
            self.exchange.load_markets()
            self.symbols = self.fetch_symbols_under_200()
            
            # Setup database
            self.setup_database()
            
            logger.info(f"Dynamic trading system initialized with {len(self.symbols)} symbols")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            raise
    
    def fetch_symbols_under_200(self) -> List[str]:
        """Fetch live symbols under $200 from OKX"""
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
    
    def setup_database(self):
        """Setup dynamic trading database"""
        try:
            conn = sqlite3.connect('dynamic_trading.db')
            cursor = conn.cursor()
            
            # Signals table with dynamic take profit
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dynamic_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    signal_type TEXT,
                    confidence REAL,
                    price REAL,
                    target_price REAL,
                    stop_loss REAL,
                    take_profit_pct REAL,
                    buy_score REAL,
                    sell_score REAL,
                    volume_score REAL,
                    momentum_score REAL,
                    signals TEXT
                )
            ''')
            
            # Trades table with enhanced tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dynamic_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    side TEXT,
                    amount REAL,
                    price REAL,
                    order_id TEXT,
                    confidence REAL,
                    take_profit_pct REAL,
                    status TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Dynamic trading database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def get_market_data(self, symbol: str, timeframe: str = '1h', limit: int = 200) -> Optional[pd.DataFrame]:
        """Get market data with enhanced indicators"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = self.calculate_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Market data fetch failed for {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        # RSI
        df['rsi'] = ta.rsi(df['close'], length=14)
        
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        
        # EMAs
        df['ema_9'] = ta.ema(df['close'], length=9)
        df['ema_21'] = ta.ema(df['close'], length=21)
        df['ema_50'] = ta.ema(df['close'], length=50)
        
        # Bollinger Bands
        bb = ta.bbands(df['close'], length=20, std=2)
        df = pd.concat([df, bb], axis=1)
        
        # Volume indicators
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum
        df['mom'] = ta.mom(df['close'], length=10)
        df['roc'] = ta.roc(df['close'], length=10)
        
        return df
    
    def analyze_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Analyze market data to generate BUY/SELL signals with dynamic take profit"""
        if len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Separate scoring for BUY and SELL signals
        buy_score = 0
        sell_score = 0
        volume_score = 0
        momentum_score = 0
        signals_list = []
        
        # RSI analysis (weight: 25)
        rsi = latest['rsi']
        if pd.notna(rsi):
            if rsi < 30:
                buy_score += 25
                signals_list.append("RSI oversold")
            elif rsi < 40:
                buy_score += 15
                signals_list.append("RSI bullish")
            elif rsi > 70:
                sell_score += 25
                signals_list.append("RSI overbought")
            elif rsi > 60:
                sell_score += 10
                signals_list.append("RSI bearish")
        
        # MACD analysis (weight: 20)
        if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
            macd = latest['MACD_12_26_9']
            macd_signal = latest['MACDs_12_26_9']
            if pd.notna(macd) and pd.notna(macd_signal):
                if macd > macd_signal and prev['MACD_12_26_9'] <= prev['MACDs_12_26_9']:
                    buy_score += 20
                    signals_list.append("MACD bullish crossover")
                elif macd > macd_signal:
                    buy_score += 10
                    signals_list.append("MACD bullish")
                elif macd < macd_signal and prev['MACD_12_26_9'] >= prev['MACDs_12_26_9']:
                    sell_score += 20
                    signals_list.append("MACD bearish crossover")
                elif macd < macd_signal:
                    sell_score += 10
                    signals_list.append("MACD bearish")
        
        # EMA trend analysis (weight: 20)
        if all(col in df.columns for col in ['ema_9', 'ema_21', 'ema_50']):
            if latest['close'] > latest['ema_9'] > latest['ema_21'] > latest['ema_50']:
                buy_score += 20
                signals_list.append("Strong uptrend")
            elif latest['close'] > latest['ema_9'] > latest['ema_21']:
                buy_score += 10
                signals_list.append("Uptrend")
            elif latest['close'] < latest['ema_9'] < latest['ema_21'] < latest['ema_50']:
                sell_score += 20
                signals_list.append("Strong downtrend")
            elif latest['close'] < latest['ema_9'] < latest['ema_21']:
                sell_score += 10
                signals_list.append("Downtrend")
        
        # Bollinger Bands analysis (weight: 15)
        if 'BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
            bb_lower = latest['BBL_20_2.0']
            bb_upper = latest['BBU_20_2.0']
            if pd.notna(bb_lower) and pd.notna(bb_upper):
                if latest['close'] < bb_lower:
                    buy_score += 15
                    signals_list.append("Below BB lower band")
                elif latest['close'] < bb_lower * 1.02:
                    buy_score += 10
                    signals_list.append("Near BB lower band")
                elif latest['close'] > bb_upper:
                    sell_score += 15
                    signals_list.append("Above BB upper band")
                elif latest['close'] > bb_upper * 0.98:
                    sell_score += 10
                    signals_list.append("Near BB upper band")
        
        # Volume analysis (weight: 15)
        if pd.notna(latest['volume_ratio']):
            volume_ratio = latest['volume_ratio']
            if volume_ratio > 2:
                volume_score += 15
                signals_list.append("High volume surge")
            elif volume_ratio > 1.5:
                volume_score += 10
                signals_list.append("Above average volume")
        
        # Momentum analysis (weight: 5)
        if pd.notna(latest['mom']) and latest['mom'] > 0:
            momentum_score += 5
            signals_list.append("Positive momentum")
        elif pd.notna(latest['mom']) and latest['mom'] < 0:
            momentum_score -= 5
            signals_list.append("Negative momentum")
        
        # Determine signal type and calculate dynamic take profit
        if buy_score > sell_score and buy_score >= 50:
            signal_type = 'BUY'
            confidence = min(max(buy_score + volume_score + momentum_score, 0), 100)
            # Dynamic take profit: 8% base + boost for high confidence
            take_profit_pct = self.base_take_profit_pct + (confidence - 70) * 0.2
            take_profit_pct = max(5, min(take_profit_pct, 15))  # Cap 5-15%
            
        elif sell_score > buy_score and sell_score >= 50:
            signal_type = 'SELL'
            confidence = min(max(sell_score + volume_score + momentum_score, 0), 100)
            take_profit_pct = self.base_take_profit_pct + (confidence - 70) * 0.2
            take_profit_pct = max(5, min(take_profit_pct, 15))
        else:
            return None  # No clear signal
        
        # Only generate signal if confidence meets threshold
        if confidence >= self.min_confidence:
            if signal_type == 'BUY':
                target_price = float(latest['close'] * (1 + take_profit_pct / 100))
                stop_loss = float(latest['close'] * (1 - self.stop_loss_pct / 100))
            else:  # SELL
                target_price = float(latest['close'] * (1 - take_profit_pct / 100))
                stop_loss = float(latest['close'] * (1 + self.stop_loss_pct / 100))
            
            signal = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'signal_type': signal_type,
                'confidence': confidence,
                'price': float(latest['close']),
                'target_price': target_price,
                'stop_loss': stop_loss,
                'take_profit_pct': take_profit_pct,
                'volume_24h': float(latest['volume']),
                'rsi': float(rsi) if pd.notna(rsi) else None,
                'buy_score': buy_score,
                'sell_score': sell_score,
                'volume_score': volume_score,
                'momentum_score': momentum_score,
                'signals': signals_list
            }
            
            return signal
        
        return None
    
    def execute_signal(self, signal: Dict) -> bool:
        """Execute BUY/SELL signal with proper position sizing"""
        if not self.exchange:
            return False
        
        try:
            symbol = signal['symbol']
            current_price = signal['price']
            signal_key = f"{symbol}_{signal['signal_type']}_{int(signal['confidence'])}"
            
            # Check if already executed
            if signal_key in self.executed_signals:
                return False
            
            # Get balance and calculate position size
            balance = self.exchange.fetch_balance()
            usdt_balance = float(balance.get('USDT', {}).get('free', 0))
            
            if usdt_balance < self.min_trade_usd:
                logger.warning(f"Insufficient USDT balance: ${usdt_balance:.2f}")
                return False
            
            # Calculate trade amount
            if signal['signal_type'] == 'BUY':
                trade_value = usdt_balance * self.position_size_pct
                quantity = trade_value / current_price
                
                # Execute buy order
                quantity = self.exchange.amount_to_precision(symbol, quantity)
                order = self.exchange.create_market_buy_order(symbol, float(quantity))
                side = 'BUY'
                
            else:  # SELL signal
                base_currency = symbol.split('/')[0]
                available_amount = float(balance.get(base_currency, {}).get('free', 0))
                min_amount = self.exchange.markets[symbol]['limits']['amount']['min']
                
                if available_amount < min_amount:
                    logger.warning(f"Insufficient {base_currency} balance for SELL: {available_amount}")
                    return False
                
                # Calculate sell quantity (use 80% of available to account for fees)
                sell_quantity = available_amount * 0.8
                sell_quantity = self.exchange.amount_to_precision(symbol, sell_quantity)
                order = self.exchange.create_market_sell_order(symbol, float(sell_quantity))
                quantity = sell_quantity
                side = 'SELL'
            
            # Record execution
            self.executed_signals.add(signal_key)
            
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': side,
                'amount': float(quantity),
                'price': current_price,
                'order_id': order.get('id', ''),
                'confidence': signal['confidence'],
                'take_profit_pct': signal['take_profit_pct'],
                'status': 'EXECUTED'
            }
            
            self.save_trade(trade_record)
            
            logger.info(f"✅ TRADE EXECUTED: {side} {quantity} {symbol} @ ${current_price:.6f} "
                       f"({signal['confidence']:.1f}%, TP: {signal['take_profit_pct']:.1f}%)")
            return True
            
        except Exception as e:
            logger.error(f"Trade execution failed for {signal['symbol']}: {e}")
            return False
    
    def save_signal(self, signal: Dict):
        """Save signal to database"""
        try:
            conn = sqlite3.connect('dynamic_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO dynamic_signals (
                    timestamp, symbol, signal_type, confidence, price, target_price, 
                    stop_loss, take_profit_pct, buy_score, sell_score, 
                    volume_score, momentum_score, signals
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['timestamp'], signal['symbol'], signal['signal_type'],
                signal['confidence'], signal['price'], signal['target_price'],
                signal['stop_loss'], signal['take_profit_pct'], signal['buy_score'],
                signal['sell_score'], signal['volume_score'], signal['momentum_score'],
                ', '.join(signal['signals'])
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")
    
    def save_trade(self, trade: Dict):
        """Save trade to database"""
        try:
            conn = sqlite3.connect('dynamic_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO dynamic_trades (
                    timestamp, symbol, side, amount, price, order_id, 
                    confidence, take_profit_pct, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade['timestamp'], trade['symbol'], trade['side'],
                trade['amount'], trade['price'], trade['order_id'],
                trade['confidence'], trade['take_profit_pct'], trade['status']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")
    
    def trading_loop(self):
        """Main dynamic trading loop"""
        logger.info("🚀 Starting Dynamic Trading Loop")
        logger.info(f"Configuration: Min Confidence: {self.min_confidence}%, "
                   f"Base Take Profit: {self.base_take_profit_pct}% (Dynamic 5-15%)")
        
        while self.is_running:
            try:
                signals_generated = 0
                trades_executed = 0
                
                logger.info(f"🔄 Scanning {len(self.symbols)} symbols for BUY/SELL signals...")
                
                for symbol in self.symbols:
                    try:
                        df = self.get_market_data(symbol)
                        if df is not None:
                            signal = self.analyze_signal(symbol, df)
                            if signal:
                                signals_generated += 1
                                self.save_signal(signal)
                                
                                # Execute signal immediately
                                if self.execute_signal(signal):
                                    trades_executed += 1
                                    
                                logger.info(f"📊 {symbol}: {signal['signal_type']} "
                                          f"(Confidence: {signal['confidence']:.1f}%, "
                                          f"TP: {signal['take_profit_pct']:.1f}%)")
                    
                    except Exception as e:
                        logger.error(f"Signal analysis failed for {symbol}: {e}")
                        continue
                
                logger.info(f"✅ Scan complete: {signals_generated} signals, {trades_executed} trades executed")
                
                # Dynamic scan interval (faster during high volatility)
                scan_interval = max(120, 300 - (signals_generated * 10))  # 2-5 minutes
                logger.info(f"⏰ Next scan in {scan_interval} seconds...")
                
                for _ in range(scan_interval):
                    if not self.is_running:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(60)
    
    def start_trading(self):
        """Start the dynamic trading system"""
        if self.is_running:
            return
        
        self.is_running = True
        trading_thread = threading.Thread(target=self.trading_loop, daemon=True)
        trading_thread.start()
    
    def stop_trading(self):
        """Stop the dynamic trading system"""
        self.is_running = False

# Flask web interface
app = Flask(__name__)
trading_system = DynamicTradingSystem()

@app.route('/')
def dashboard():
    """Dynamic trading dashboard"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dynamic Trading System - Adaptive Take Profit & BUY/SELL</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { font-family: 'Segoe UI', Arial, sans-serif; background: #0a0e27; color: #fff; }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; margin-bottom: 30px; }
            .header h1 { color: #4CAF50; font-size: 2.5em; margin-bottom: 10px; }
            .header p { color: #999; font-size: 1.1em; }
            .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }
            .stat-card { background: linear-gradient(135deg, #1e3c72, #2a5298); padding: 20px; border-radius: 10px; text-align: center; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
            .stat-value { font-size: 2em; font-weight: bold; color: #4CAF50; }
            .stat-label { color: #ccc; margin-top: 5px; }
            .content-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .signals-section, .trades-section { background: #1a1a2e; border-radius: 10px; padding: 20px; }
            .section-title { color: #4CAF50; font-size: 1.5em; margin-bottom: 15px; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
            .signal-item, .trade-item { background: #252545; margin: 10px 0; padding: 15px; border-radius: 8px; border-left: 4px solid #4CAF50; }
            .buy-signal { border-left-color: #4CAF50; }
            .sell-signal { border-left-color: #ff6b6b; }
            .signal-header { display: flex; justify-content: between; align-items: center; margin-bottom: 8px; }
            .signal-symbol { font-weight: bold; color: #fff; }
            .signal-type { padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
            .buy-type { background: #4CAF50; color: #fff; }
            .sell-type { background: #ff6b6b; color: #fff; }
            .signal-confidence { color: #4CAF50; font-weight: bold; }
            .signal-details { font-size: 0.9em; color: #ccc; }
            .take-profit { color: #ffa726; font-weight: bold; }
            @media (max-width: 768px) { .content-grid { grid-template-columns: 1fr; } }
        </style>
        <script>
            function updateData() {
                Promise.all([
                    fetch('/api/status').then(r => r.json()),
                    fetch('/api/signals').then(r => r.json()),
                    fetch('/api/trades').then(r => r.json())
                ]).then(([status, signals, trades]) => {
                    document.getElementById('system-status').textContent = status.status;
                    document.getElementById('confidence-threshold').textContent = status.min_confidence + '%';
                    document.getElementById('signals-count').textContent = signals.length;
                    document.getElementById('trades-count').textContent = trades.length;
                    
                    // Update signals
                    const signalsHtml = signals.slice(0, 10).map(signal => `
                        <div class="signal-item ${signal.signal_type.toLowerCase()}-signal">
                            <div class="signal-header">
                                <span class="signal-symbol">${signal.symbol}</span>
                                <span class="signal-type ${signal.signal_type.toLowerCase()}-type">${signal.signal_type}</span>
                            </div>
                            <div class="signal-details">
                                Confidence: <span class="signal-confidence">${signal.confidence.toFixed(1)}%</span> | 
                                Price: $${signal.price.toFixed(6)} | 
                                <span class="take-profit">Take Profit: ${signal.take_profit_pct.toFixed(1)}%</span>
                            </div>
                            <div class="signal-details">${signal.signals.join(', ')}</div>
                        </div>
                    `).join('');
                    document.getElementById('signals-list').innerHTML = signalsHtml;
                    
                    // Update trades
                    const tradesHtml = trades.slice(0, 10).map(trade => `
                        <div class="trade-item">
                            <div class="signal-header">
                                <span class="signal-symbol">${trade.symbol}</span>
                                <span class="signal-type ${trade.side.toLowerCase()}-type">${trade.side}</span>
                            </div>
                            <div class="signal-details">
                                Amount: ${trade.amount.toFixed(6)} | Price: $${trade.price.toFixed(6)} | 
                                <span class="take-profit">TP: ${trade.take_profit_pct.toFixed(1)}%</span>
                            </div>
                        </div>
                    `).join('');
                    document.getElementById('trades-list').innerHTML = tradesHtml;
                });
            }
            setInterval(updateData, 3000);
            updateData();
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔄 Dynamic Trading System</h1>
                <p>Adaptive Take Profit (5-15%) • BUY/SELL Signal Detection • Real-time OKX Data</p>
            </div>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="system-status">Running</div>
                    <div class="stat-label">System Status</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="confidence-threshold">70%</div>
                    <div class="stat-label">Min Confidence</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="signals-count">0</div>
                    <div class="stat-label">Signals Generated</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="trades-count">0</div>
                    <div class="stat-label">Trades Executed</div>
                </div>
            </div>
            
            <div class="content-grid">
                <div class="signals-section">
                    <div class="section-title">🎯 Recent Signals</div>
                    <div id="signals-list">Loading signals...</div>
                </div>
                
                <div class="trades-section">
                    <div class="section-title">💰 Recent Trades</div>
                    <div id="trades-list">Loading trades...</div>
                </div>
            </div>
        </div>
    </body>
    </html>
    ''')

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'status': 'Running' if trading_system.is_running else 'Stopped',
        'symbols_count': len(trading_system.symbols),
        'min_confidence': trading_system.min_confidence,
        'base_take_profit': trading_system.base_take_profit_pct
    })

@app.route('/api/signals')
def get_signals():
    """Get recent signals"""
    try:
        conn = sqlite3.connect('dynamic_trading.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM dynamic_signals 
            ORDER BY timestamp DESC LIMIT 50
        ''')
        
        signals = []
        for row in cursor.fetchall():
            signals.append({
                'id': row[0], 'timestamp': row[1], 'symbol': row[2],
                'signal_type': row[3], 'confidence': row[4], 'price': row[5],
                'target_price': row[6], 'stop_loss': row[7], 'take_profit_pct': row[8],
                'buy_score': row[9], 'sell_score': row[10], 'volume_score': row[11],
                'momentum_score': row[12], 'signals': row[13].split(', ') if row[13] else []
            })
        
        conn.close()
        return jsonify(signals)
        
    except Exception as e:
        logger.error(f"Failed to get signals: {e}")
        return jsonify([])

@app.route('/api/trades')
def get_trades():
    """Get recent trades"""
    try:
        conn = sqlite3.connect('dynamic_trading.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM dynamic_trades 
            ORDER BY timestamp DESC LIMIT 50
        ''')
        
        trades = []
        for row in cursor.fetchall():
            trades.append({
                'id': row[0], 'timestamp': row[1], 'symbol': row[2],
                'side': row[3], 'amount': row[4], 'price': row[5],
                'order_id': row[6], 'confidence': row[7], 'take_profit_pct': row[8],
                'status': row[9]
            })
        
        conn.close()
        return jsonify(trades)
        
    except Exception as e:
        logger.error(f"Failed to get trades: {e}")
        return jsonify([])

def main():
    """Main function"""
    try:
        # Start trading system
        trading_system.start_trading()
        
        # Start web interface
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except KeyboardInterrupt:
        logger.info("Shutting down dynamic trading system...")
        trading_system.stop_trading()
    except Exception as e:
        logger.error(f"System error: {e}")

if __name__ == "__main__":
    main()