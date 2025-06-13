#!/usr/bin/env python3
"""
Signal Execution Bridge - Converts AI signals into live trades
Bridges the gap between signal generation and trade execution
"""

import os
import time
import sqlite3
import logging
import ccxt
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalExecutionBridge:
    """Bridge that monitors signals and executes trades automatically"""
    
    def __init__(self):
        self.db_path = 'trading_platform.db'
        self.live_db_path = 'live_trading.db'
        self.exchange = None
        self.is_running = False
        self.last_execution_time = {}
        self.rate_limit_delay = 0.2  # 200ms between API calls (5 req/sec max)
        self.execution_threshold = 75.0  # 75% confidence minimum (CRITICAL FIX)
        self.max_position_size_pct = 0.035  # 3.5% per trade (optimized)
        self.min_trade_amount = 5   # Minimum $5 USDT to enable execution
        
        self.initialize_exchange()
        
    def initialize_exchange(self):
        """Initialize OKX exchange for live trading"""
        try:
            api_key = os.environ.get('OKX_API_KEY')
            secret_key = os.environ.get('OKX_SECRET_KEY')
            passphrase = os.environ.get('OKX_PASSPHRASE')
            
            if api_key and secret_key and passphrase:
                self.exchange = ccxt.okx({
                    'apiKey': api_key,
                    'secret': secret_key,
                    'password': passphrase,
                    'sandbox': False,
                    'rateLimit': 1200,
                    'enableRateLimit': True,
                })
                logger.info("Signal execution bridge connected to OKX")
            else:
                logger.error("OKX credentials not found - bridge disabled")
                
        except Exception as e:
            logger.error(f"Failed to initialize OKX for signal bridge: {e}")
            
    def get_fresh_signals(self) -> List[Dict]:
        """Get unprocessed signals from the last 60 seconds"""
        signals = []
        
        # Check multiple databases for signals
        databases = [
            ('pure_local_trading.db', 'local_signals'),
            ('enhanced_trading.db', 'ai_signals'),
            ('enhanced_trading.db', 'unified_signals'),
            ('autonomous_trading.db', 'signals')
        ]
        
        cutoff_time = (datetime.now() - timedelta(seconds=300)).isoformat()  # 5 minutes window
        
        for db_path, table_name in databases:
            if not os.path.exists(db_path):
                continue
                
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Check if table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
                if not cursor.fetchone():
                    conn.close()
                    continue
                
                # Get column names
                cursor.execute(f"PRAGMA table_info({table_name})")
                columns = [row[1] for row in cursor.fetchall()]
                
                # Build query based on available columns
                signal_col = 'signal' if 'signal' in columns else 'action' if 'action' in columns else 'side'
                confidence_col = 'confidence' if 'confidence' in columns else 'score'
                
                if signal_col not in columns or confidence_col not in columns:
                    conn.close()
                    continue
                
                cursor.execute(f'''
                    SELECT symbol, {signal_col}, {confidence_col}, timestamp
                    FROM {table_name} 
                    WHERE timestamp > ? 
                    AND {confidence_col} >= ?
                    AND ({signal_col} = 'BUY' OR {signal_col} = 'buy')
                    ORDER BY timestamp DESC LIMIT 20
                ''', (cutoff_time, self.execution_threshold))
                
                for row in cursor.fetchall():
                    signal = {
                        'symbol': row[0],
                        'action': row[1],
                        'confidence': float(row[2])/100 if float(row[2]) > 1 else float(row[2]),
                        'timestamp': row[3],
                        'reasoning': f'From {table_name} in {db_path}'
                    }
                    
                    # Check if we haven't processed this symbol recently
                    symbol_key = f"{signal['symbol']}_{signal['action']}"
                    if symbol_key not in self.last_execution_time:
                        signals.append(signal)
                    else:
                        last_time = datetime.fromisoformat(self.last_execution_time[symbol_key])
                        if (datetime.now() - last_time).seconds > 300:  # 5 minute cooldown
                            signals.append(signal)
                
                conn.close()
                
            except Exception as e:
                logger.error(f"Error processing {db_path}/{table_name}: {e}")
                continue
        

    
    def calculate_position_size(self, symbol: str, confidence: float = 0.6) -> float:
        """Calculate position size based on 1% risk and confidence"""
        try:
            time.sleep(self.rate_limit_delay)  # Rate limiting
            balance = self.exchange.fetch_balance()
            usdt_balance = float(balance['USDT']['free'])
            
            # Convert confidence from percentage to decimal if needed
            if confidence > 1:
                confidence = confidence / 100
            
            # Use 1% of available USDT balance, adjusted by confidence
            base_position_value = usdt_balance * self.max_position_size_pct
            position_value = base_position_value * confidence
            
            # Ensure position is meaningful
            if position_value < self.min_trade_amount:
                position_value = self.min_trade_amount
            
            logger.info(f"Position sizing: USDT=${usdt_balance:.2f}, Base=${base_position_value:.2f}, Confidence={confidence:.1%}, Final=${position_value:.2f}")
            return position_value
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return self.min_trade_amount
    
    def execute_signal_trade(self, signal: Dict) -> Optional[Dict]:
        """Execute trade based on signal"""
        try:
            symbol = signal['symbol']
            # Format symbol for OKX trading (add /USDT if not present)
            if not symbol.endswith('/USDT'):
                symbol = f"{symbol}/USDT"
            action = signal['action'].lower()
            confidence = signal['confidence']
            
            logger.info(f"EXECUTING SIGNAL: {action.upper()} {symbol} (Confidence: {confidence:.1%})")
            
            # Get current price with rate limiting
            time.sleep(self.rate_limit_delay)
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate position size
            position_value = self.calculate_position_size(symbol, confidence)
            
            # Calculate amount
            if action == 'buy':
                amount = float(position_value) / float(current_price)
                amount = float(self.exchange.amount_to_precision(symbol, amount))
                
                trade_value = float(amount) * float(current_price)
                logger.info(f"Executing trade: {amount:.6f} {symbol} @ ${current_price:.4f} = ${trade_value:.2f}")
                
                # Execute buy order with rate limiting
                time.sleep(self.rate_limit_delay)
                order = self.exchange.create_market_buy_order(symbol, amount)
                
            elif action == 'sell':
                # Check available balance for selling
                time.sleep(self.rate_limit_delay)
                balance = self.exchange.fetch_balance()
                base_currency = symbol.split('/')[0]
                available = float(balance[base_currency]['free'])
                
                if available == 0:
                    logger.warning(f"No {base_currency} available to sell")
                    return None
                
                # Use available amount or calculated amount (whichever is smaller)
                calculated_amount = float(position_value) / float(current_price)
                amount = min(available, calculated_amount)
                amount = float(self.exchange.amount_to_precision(symbol, amount))
                
                # Execute sell order with rate limiting
                time.sleep(self.rate_limit_delay)
                order = self.exchange.create_market_sell_order(symbol, amount)
            
            else:
                logger.warning(f"Unknown action: {action}")
                return None
            
            # Record successful execution
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': action.upper(),
                'amount': amount,
                'price': current_price,
                'order_id': order['id'],
                'strategy': 'AI_Signal_Bridge',
                'ai_confidence': confidence,
                'status': 'EXECUTED',
                'pnl': 0.0
            }
            
            # Log to live trading database
            self.log_live_trade(trade_record)
            
            # Update execution tracking
            symbol_key = f"{symbol}_{action.upper()}"
            self.last_execution_time[symbol_key] = datetime.now().isoformat()
            
            logger.info(f"✅ TRADE EXECUTED: {action.upper()} {amount} {symbol} at ${current_price:.4f}")
            logger.info(f"Order ID: {order['id']}, Value: ${amount * current_price:.2f}")
            
            return trade_record
            
        except Exception as e:
            logger.error(f"Failed to execute signal trade: {e}")
            return None
    
    def log_live_trade(self, trade: Dict):
        """Log trade to live trading database"""
        try:
            conn = sqlite3.connect(self.live_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO live_trades 
                (timestamp, symbol, side, amount, price, order_id, strategy, ai_confidence, status, pnl)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade['timestamp'], trade['symbol'], trade['side'],
                trade['amount'], trade['price'], trade['order_id'],
                trade['strategy'], trade['ai_confidence'], trade['status'], trade['pnl']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging live trade: {e}")
    
    def monitor_and_execute(self):
        """Main monitoring loop for signal execution"""
        logger.info("🔄 SIGNAL EXECUTION BRIDGE ACTIVATED")
        logger.info(f"Monitoring for AI signals with ≥{self.execution_threshold}% confidence...")
        
        while self.is_running:
            try:
                # Get fresh signals
                signals = self.get_fresh_signals()
                
                if signals:
                    logger.info(f"Found {len(signals)} executable signals")
                    
                    for signal in signals:
                        # Execute each signal with proper rate limiting
                        result = self.execute_signal_trade(signal)
                        
                        if result:
                            # Add delay between trades
                            time.sleep(2)
                
                # Sleep before next check
                time.sleep(30)  # Check every 30 seconds
                
            except KeyboardInterrupt:
                logger.info("Signal bridge interrupted")
                break
            except Exception as e:
                logger.error(f"Signal bridge error: {e}")
                time.sleep(10)  # Wait before retry
    
    def start(self):
        """Start the signal execution bridge"""
        if not self.exchange:
            logger.error("Cannot start bridge - OKX exchange not initialized")
            return
        
        if self.is_running:
            logger.warning("Signal bridge already running")
            return
        
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self.monitor_and_execute, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Signal execution bridge started")
    
    def stop(self):
        """Stop the signal execution bridge"""
        self.is_running = False
        logger.info("Signal execution bridge stopped")

def main():
    """Main entry point for signal execution bridge"""
    bridge = SignalExecutionBridge()
    
    try:
        bridge.start()
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down signal execution bridge...")
        bridge.stop()
    except Exception as e:
        logger.error(f"Bridge error: {e}")

if __name__ == '__main__':
    main()