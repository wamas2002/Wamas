#!/usr/bin/env python3
"""
Stop Loss Engine
Automated stop-loss and profit-taking with real-time position monitoring
"""

import os
import ccxt
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class StopLossEngine:
    def __init__(self):
        self.exchange = self.connect_okx()
        self.stop_loss_percentage = 2.0  # 2% stop loss
        self.profit_targets = {
            'quick': 1.5,   # 1.5% quick profit
            'medium': 3.0,  # 3.0% medium profit
            'high': 5.0     # 5.0% high profit
        }
        self.monitoring_active = True
        
    def connect_okx(self):
        """Connect to OKX exchange"""
        try:
            return ccxt.okx({
                'apiKey': os.environ.get('OKX_API_KEY'),
                'secret': os.environ.get('OKX_SECRET_KEY'),
                'password': os.environ.get('OKX_PASSPHRASE'),
                'sandbox': False,
                'rateLimit': 2000,
                'enableRateLimit': True,
            })
        except Exception as e:
            print(f"OKX connection error: {e}")
            return None
    
    def get_current_positions_with_pnl(self) -> Dict:
        """Get current positions with profit/loss calculations"""
        if not self.exchange:
            return {}
        
        try:
            balance = self.exchange.fetch_balance()
            positions = {}
            
            for currency in balance:
                if currency != 'USDT' and balance[currency]['free'] > 0:
                    amount = float(balance[currency]['free'])
                    if amount > 0:
                        try:
                            symbol = f"{currency}/USDT"
                            ticker = self.exchange.fetch_ticker(symbol)
                            current_price = float(ticker['last'])
                            
                            # Get entry price from recent trades
                            entry_price = self.get_entry_price(symbol, amount)
                            
                            if entry_price:
                                profit_pct = ((current_price - entry_price) / entry_price) * 100
                                profit_usdt = (current_price - entry_price) * amount
                                
                                positions[symbol] = {
                                    'currency': currency,
                                    'amount': amount,
                                    'entry_price': entry_price,
                                    'current_price': current_price,
                                    'profit_pct': profit_pct,
                                    'profit_usdt': profit_usdt,
                                    'position_value': current_price * amount,
                                    'stop_loss_price': entry_price * (1 - self.stop_loss_percentage / 100),
                                    'profit_target_quick': entry_price * (1 + self.profit_targets['quick'] / 100),
                                    'profit_target_medium': entry_price * (1 + self.profit_targets['medium'] / 100),
                                    'profit_target_high': entry_price * (1 + self.profit_targets['high'] / 100)
                                }
                        except Exception as e:
                            print(f"Error processing {currency}: {e}")
                            continue
            
            return positions
            
        except Exception as e:
            print(f"Error getting positions: {e}")
            return {}
    
    def get_entry_price(self, symbol: str, amount: float) -> Optional[float]:
        """Get average entry price from trade history"""
        try:
            if not os.path.exists('live_trading.db'):
                return None
            
            conn = sqlite3.connect('live_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT amount, price FROM live_trades 
                WHERE symbol = ? AND side = 'buy' 
                ORDER BY timestamp DESC LIMIT 10
            ''', (symbol,))
            
            trades = cursor.fetchall()
            conn.close()
            
            if not trades:
                return None
            
            # Calculate weighted average entry price
            total_amount = 0
            total_cost = 0
            
            for trade_amount, trade_price in trades:
                trade_amount = float(trade_amount)
                trade_price = float(trade_price)
                
                if total_amount < amount:
                    remaining_needed = amount - total_amount
                    amount_to_use = min(trade_amount, remaining_needed)
                    
                    total_amount += amount_to_use
                    total_cost += amount_to_use * trade_price
                    
                    if total_amount >= amount:
                        break
            
            if total_amount > 0:
                return total_cost / total_amount
            
            return None
            
        except Exception as e:
            print(f"Error calculating entry price: {e}")
            return None
    
    def check_stop_loss_conditions(self, positions: Dict) -> List[Dict]:
        """Check for positions that need stop-loss execution"""
        stop_loss_actions = []
        
        for symbol, position in positions.items():
            current_price = position['current_price']
            stop_loss_price = position['stop_loss_price']
            profit_pct = position['profit_pct']
            
            # Check if stop-loss should be triggered
            if current_price <= stop_loss_price:
                stop_loss_actions.append({
                    'symbol': symbol,
                    'action': 'STOP_LOSS',
                    'reason': f'Price ${current_price:.4f} hit stop-loss at ${stop_loss_price:.4f}',
                    'sell_percentage': 100,  # Sell entire position
                    'priority': 'critical',
                    'position_data': position,
                    'loss_pct': profit_pct
                })
            
            # Check for trailing stop-loss (if position is profitable)
            elif profit_pct > 3.0:  # Only use trailing stop if > 3% profit
                trailing_stop_price = current_price * 0.98  # 2% trailing stop
                if trailing_stop_price > position['entry_price']:
                    stop_loss_actions.append({
                        'symbol': symbol,
                        'action': 'TRAILING_STOP',
                        'reason': f'Trailing stop at ${trailing_stop_price:.4f} (2% below current)',
                        'sell_percentage': 50,  # Sell half position
                        'priority': 'medium',
                        'position_data': position,
                        'profit_pct': profit_pct
                    })
        
        return stop_loss_actions
    
    def check_profit_taking_conditions(self, positions: Dict) -> List[Dict]:
        """Check for positions that meet profit-taking criteria"""
        profit_actions = []
        
        for symbol, position in positions.items():
            current_price = position['current_price']
            profit_pct = position['profit_pct']
            
            # High profit target (5%)
            if current_price >= position['profit_target_high']:
                profit_actions.append({
                    'symbol': symbol,
                    'action': 'PROFIT_HIGH',
                    'reason': f'High profit target reached: {profit_pct:.2f}%',
                    'sell_percentage': 75,  # Sell 75% of position
                    'priority': 'high',
                    'position_data': position
                })
            
            # Medium profit target (3%)
            elif current_price >= position['profit_target_medium']:
                profit_actions.append({
                    'symbol': symbol,
                    'action': 'PROFIT_MEDIUM',
                    'reason': f'Medium profit target reached: {profit_pct:.2f}%',
                    'sell_percentage': 50,  # Sell 50% of position
                    'priority': 'medium',
                    'position_data': position
                })
            
            # Quick profit target (1.5%)
            elif current_price >= position['profit_target_quick']:
                profit_actions.append({
                    'symbol': symbol,
                    'action': 'PROFIT_QUICK',
                    'reason': f'Quick profit target reached: {profit_pct:.2f}%',
                    'sell_percentage': 25,  # Sell 25% of position
                    'priority': 'low',
                    'position_data': position
                })
        
        return profit_actions
    
    def execute_sell_order(self, action: Dict) -> Dict:
        """Execute SELL order for stop-loss or profit-taking"""
        if not self.exchange:
            return {'success': False, 'error': 'No exchange connection'}
        
        try:
            symbol = action['symbol']
            position = action['position_data']
            sell_percentage = action['sell_percentage']
            
            # Calculate sell amount
            total_amount = position['amount']
            sell_amount = total_amount * (sell_percentage / 100)
            current_price = position['current_price']
            
            # Use precise amount formatting
            sell_amount = float(self.exchange.amount_to_precision(symbol, sell_amount))
            
            if sell_amount > 0:
                # Execute market sell order
                order = self.exchange.create_market_sell_order(symbol, sell_amount)
                
                # Calculate profit/loss
                entry_price = position['entry_price']
                pnl_per_unit = current_price - entry_price
                total_pnl = pnl_per_unit * sell_amount
                
                # Log the trade
                self.log_sell_trade(action, order, total_pnl)
                
                return {
                    'success': True,
                    'order_id': order['id'],
                    'symbol': symbol,
                    'amount_sold': sell_amount,
                    'sell_price': current_price,
                    'pnl': total_pnl,
                    'action_type': action['action']
                }
            else:
                return {'success': False, 'error': 'Sell amount too small'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def log_sell_trade(self, action: Dict, order: Dict, pnl: float):
        """Log sell trade to database"""
        try:
            conn = sqlite3.connect('live_trading.db')
            cursor = conn.cursor()
            
            # Create sell trades table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sell_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    amount REAL,
                    sell_price REAL,
                    entry_price REAL,
                    pnl_usdt REAL,
                    pnl_pct REAL,
                    order_id TEXT,
                    action_type TEXT,
                    reason TEXT
                )
            ''')
            
            position = action['position_data']
            
            cursor.execute('''
                INSERT INTO sell_trades 
                (timestamp, symbol, amount, sell_price, entry_price, pnl_usdt, pnl_pct, order_id, action_type, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                action['symbol'],
                order.get('amount', 0),
                position['current_price'],
                position['entry_price'],
                pnl,
                position['profit_pct'],
                order.get('id', ''),
                action['action'],
                action['reason']
            ))
            
            conn.commit()
            conn.close()
            
            print(f"SELL ORDER EXECUTED: {action['symbol']} - {action['action']} ${pnl:+.2f}")
            
        except Exception as e:
            print(f"Error logging sell trade: {e}")
    
    def run_monitoring_cycle(self):
        """Execute one monitoring cycle for stop-loss and profit-taking"""
        positions = self.get_current_positions_with_pnl()
        
        if not positions:
            print("No positions to monitor")
            return
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring {len(positions)} positions...")
        
        # Check stop-loss conditions (highest priority)
        stop_loss_actions = self.check_stop_loss_conditions(positions)
        profit_actions = self.check_profit_taking_conditions(positions)
        
        # Combine and prioritize actions
        all_actions = stop_loss_actions + profit_actions
        all_actions.sort(key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x['priority']])
        
        executed_trades = 0
        
        for action in all_actions:
            if executed_trades >= 3:  # Limit trades per cycle
                break
                
            result = self.execute_sell_order(action)
            
            if result['success']:
                action_icon = "🛑" if 'STOP' in action['action'] else "💰"
                print(f"{action_icon} {action['action']}: {result['symbol']} ${result['pnl']:+.2f}")
                executed_trades += 1
                time.sleep(2)  # Rate limiting
            else:
                print(f"FAILED {action['action']} for {action['symbol']}: {result['error']}")
        
        # Display position status
        if positions:
            print("\nCurrent Position Status:")
            for symbol, pos in positions.items():
                status_icon = "🟢" if pos['profit_pct'] > 0 else "🔴"
                stop_distance = ((pos['current_price'] - pos['stop_loss_price']) / pos['current_price']) * 100
                print(f"  {status_icon} {symbol}: {pos['profit_pct']:+.2f}% | Stop: {stop_distance:.1f}% away")

def main():
    """Test stop-loss engine"""
    engine = StopLossEngine()
    
    print("STOP LOSS & PROFIT TAKING ENGINE")
    print("=" * 40)
    
    # Run one monitoring cycle
    engine.run_monitoring_cycle()
    
    print("=" * 40)

if __name__ == '__main__':
    main()