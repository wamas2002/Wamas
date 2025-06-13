"""
Direct Execution Fix - Enable automatic trading based on Pure Local signals
"""
import sqlite3
import ccxt
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class DirectAutoTrader:
    """Direct auto-trader that executes Pure Local signals immediately"""
    
    def __init__(self):
        self.exchange = None
        self.min_confidence = 75.0
        self.position_size_pct = 0.03  # 3% per trade
        self.min_trade_usd = 5.0
        self.executed_signals = set()
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.initialize_exchange()
        self.setup_database()
    
    def initialize_exchange(self):
        """Initialize OKX exchange for live trading"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.environ.get('OKX_API_KEY'),
                'secret': os.environ.get('OKX_SECRET_KEY'),
                'password': os.environ.get('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            # Test connection
            balance = self.exchange.fetch_balance()
            self.logger.info(f"Direct trader connected to OKX")
            
        except Exception as e:
            self.logger.error(f"Exchange connection failed: {e}")
            self.exchange = None
    
    def setup_database(self):
        """Setup execution tracking database"""
        conn = sqlite3.connect('direct_executions.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS executed_trades (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                symbol TEXT,
                side TEXT,
                amount REAL,
                price REAL,
                order_id TEXT,
                confidence REAL,
                signal_timestamp TEXT,
                status TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_fresh_signals(self) -> List[Dict]:
        """Get unprocessed high confidence signals from Pure Local Engine"""
        if not os.path.exists('pure_local_trading.db'):
            return []
        
        try:
            conn = sqlite3.connect('pure_local_trading.db')
            cursor = conn.cursor()
            
            # Get signals from last 10 minutes with high confidence
            cutoff_time = (datetime.now() - timedelta(minutes=10)).isoformat()
            
            cursor.execute("""
                SELECT symbol, signal_type, confidence, timestamp, price, target_price
                FROM local_signals 
                WHERE signal_type = 'BUY' 
                AND confidence >= ?
                AND timestamp > ?
                ORDER BY confidence DESC
            """, (self.min_confidence, cutoff_time))
            
            signals = []
            for row in cursor.fetchall():
                signal_key = f"{row[0]}_{row[3]}"  # symbol_timestamp
                if signal_key not in self.executed_signals:
                    signals.append({
                        'symbol': row[0],
                        'action': row[1],
                        'confidence': row[2],
                        'timestamp': row[3],
                        'price': row[4],
                        'target_price': row[5]
                    })
            
            conn.close()
            return signals
            
        except Exception as e:
            self.logger.error(f"Error getting signals: {e}")
            return []
    
    def calculate_position_size(self, symbol: str) -> float:
        """Calculate position size based on portfolio"""
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = float(balance.get('USDT', {}).get('free', 0))
            
            # Use 3% of USDT balance
            position_value = usdt_balance * self.position_size_pct
            
            if position_value < self.min_trade_usd:
                return 0
            
            return position_value
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {e}")
            return 0
    
    def execute_trade(self, signal: Dict) -> Optional[Dict]:
        """Execute trade based on signal"""
        if not self.exchange:
            return None
        
        try:
            symbol = signal['symbol']
            position_value = self.calculate_position_size(symbol)
            
            if position_value == 0:
                self.logger.warning(f"Insufficient balance for {symbol}")
                return None
            
            # Get current price
            ticker = self.exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            # Calculate quantity
            quantity = position_value / current_price
            quantity = self.exchange.amount_to_precision(symbol, quantity)
            
            if float(quantity) * current_price < self.min_trade_usd:
                return None
            
            # Execute market buy order
            order = self.exchange.create_market_buy_order(symbol, quantity)
            
            self.logger.info(f"✅ EXECUTED: BUY {quantity} {symbol} @ ${current_price}")
            
            # Record execution
            signal_key = f"{symbol}_{signal['timestamp']}"
            self.executed_signals.add(signal_key)
            
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'side': 'BUY',
                'amount': float(quantity),
                'price': current_price,
                'order_id': order.get('id', ''),
                'confidence': signal['confidence'],
                'signal_timestamp': signal['timestamp'],
                'status': 'EXECUTED'
            }
            
            self.log_execution(trade_record)
            return trade_record
            
        except Exception as e:
            self.logger.error(f"Trade execution failed for {signal['symbol']}: {e}")
            return None
    
    def log_execution(self, trade: Dict):
        """Log executed trade to database"""
        try:
            conn = sqlite3.connect('direct_executions.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO executed_trades 
                (timestamp, symbol, side, amount, price, order_id, confidence, signal_timestamp, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade['timestamp'], trade['symbol'], trade['side'],
                trade['amount'], trade['price'], trade['order_id'],
                trade['confidence'], trade['signal_timestamp'], trade['status']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Error logging execution: {e}")
    
    def run_auto_execution(self):
        """Main auto-execution loop"""
        self.logger.info("🚀 DIRECT AUTO-TRADER ACTIVATED")
        self.logger.info(f"Executing signals ≥{self.min_confidence}% confidence automatically")
        
        while True:
            try:
                # Get fresh signals
                signals = self.get_fresh_signals()
                
                if signals:
                    self.logger.info(f"🎯 Processing {len(signals)} high confidence signals")
                    
                    for signal in signals:
                        symbol = signal['symbol']
                        confidence = signal['confidence']
                        
                        self.logger.info(f"🚀 Executing: {symbol} ({confidence:.1f}% confidence)")
                        
                        # Execute trade immediately
                        result = self.execute_trade(signal)
                        
                        if result:
                            amount = result['amount']
                            price = result['price']
                            self.logger.info(f"✅ SUCCESS: {amount} {symbol} @ ${price:.4f}")
                        else:
                            self.logger.warning(f"❌ FAILED: {symbol} execution failed")
                        
                        # Rate limiting
                        time.sleep(0.5)
                
                # Check for new signals every 5 seconds
                time.sleep(5)
                
            except KeyboardInterrupt:
                self.logger.info("Auto-trader stopped by user")
                break
            except Exception as e:
                self.logger.error(f"Auto-execution error: {e}")
                time.sleep(10)

def main():
    """Run direct auto-execution"""
    trader = DirectAutoTrader()
    trader.run_auto_execution()

if __name__ == "__main__":
    main()