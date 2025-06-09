#!/usr/bin/env python3
"""
Automated Profit Taking Engine
Smart exit strategies and profit optimization for live trading positions
"""

import sqlite3
import os
import ccxt
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading

class AutomatedProfitTaking:
    def __init__(self):
        self.exchange = self.connect_okx()
        self.monitoring_active = True
        self.profit_targets = {
            'quick_profit': 1.5,    # 1.5% quick profit
            'medium_profit': 3.0,   # 3% medium profit
            'high_profit': 5.0,     # 5% high profit
            'trailing_stop': 0.8    # 0.8% trailing stop
        }
        
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
            print(f"Profit taking OKX connection error: {e}")
            return None
    
    def get_current_positions(self) -> Dict:
        """Get current portfolio positions with profit calculations"""
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
                                    'amount': amount,
                                    'entry_price': entry_price,
                                    'current_price': current_price,
                                    'profit_pct': profit_pct,
                                    'profit_usdt': profit_usdt,
                                    'position_value': current_price * amount
                                }
                        except Exception as e:
                            print(f"Error processing {currency}: {e}")
                            continue
            
            return positions
            
        except Exception as e:
            print(f"Error getting positions: {e}")
            return {}
    
    def get_entry_price(self, symbol: str, amount: float) -> Optional[float]:
        """Get average entry price for a position from trade history"""
        try:
            if not os.path.exists('live_trading.db'):
                return None
            
            conn = sqlite3.connect('live_trading.db')
            cursor = conn.cursor()
            
            # Get recent buy trades for this symbol
            cursor.execute('''
                SELECT amount, price FROM live_trades 
                WHERE symbol = ? AND side = 'buy' 
                ORDER BY timestamp DESC LIMIT 5
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
                
                # Only include relevant amount
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
            print(f"Error calculating entry price for {symbol}: {e}")
            return None
    
    def check_profit_targets(self, positions: Dict) -> List[Dict]:
        """Check which positions meet profit-taking criteria"""
        profit_opportunities = []
        
        for symbol, position in positions.items():
            profit_pct = position['profit_pct']
            
            # Check different profit levels
            if profit_pct >= self.profit_targets['high_profit']:
                profit_opportunities.append({
                    'symbol': symbol,
                    'action': 'SELL_HIGH',
                    'reason': f'High profit target reached: {profit_pct:.2f}%',
                    'sell_percentage': 75,  # Sell 75% of position
                    'priority': 'high',
                    'position_data': position
                })
            elif profit_pct >= self.profit_targets['medium_profit']:
                profit_opportunities.append({
                    'symbol': symbol,
                    'action': 'SELL_MEDIUM',
                    'reason': f'Medium profit target reached: {profit_pct:.2f}%',
                    'sell_percentage': 50,  # Sell 50% of position
                    'priority': 'medium',
                    'position_data': position
                })
            elif profit_pct >= self.profit_targets['quick_profit']:
                profit_opportunities.append({
                    'symbol': symbol,
                    'action': 'SELL_QUICK',
                    'reason': f'Quick profit target reached: {profit_pct:.2f}%',
                    'sell_percentage': 25,  # Sell 25% of position
                    'priority': 'low',
                    'position_data': position
                })
        
        return profit_opportunities
    
    def execute_profit_taking(self, opportunity: Dict) -> Dict:
        """Execute profit-taking trade"""
        if not self.exchange:
            return {'success': False, 'error': 'No exchange connection'}
        
        try:
            symbol = opportunity['symbol']
            position = opportunity['position_data']
            sell_percentage = opportunity['sell_percentage']
            
            # Calculate sell amount
            total_amount = position['amount']
            sell_amount = total_amount * (sell_percentage / 100)
            
            # Execute sell order
            current_price = position['current_price']
            
            # Use precise amount formatting
            sell_amount = float(self.exchange.amount_to_precision(symbol, sell_amount))
            
            if sell_amount > 0:
                order = self.exchange.create_market_sell_order(symbol, sell_amount)
                
                # Calculate profit taken
                entry_price = position['entry_price']
                profit_per_unit = current_price - entry_price
                total_profit = profit_per_unit * sell_amount
                
                # Log the trade
                self.log_profit_taking_trade(opportunity, order, total_profit)
                
                return {
                    'success': True,
                    'order_id': order['id'],
                    'symbol': symbol,
                    'amount_sold': sell_amount,
                    'sell_price': current_price,
                    'profit_taken': total_profit,
                    'profit_pct': opportunity['position_data']['profit_pct']
                }
            else:
                return {'success': False, 'error': 'Sell amount too small'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def log_profit_taking_trade(self, opportunity: Dict, order: Dict, profit: float):
        """Log profit-taking trade to database"""
        try:
            conn = sqlite3.connect('live_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS profit_taking_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    amount REAL,
                    sell_price REAL,
                    entry_price REAL,
                    profit_usdt REAL,
                    profit_pct REAL,
                    order_id TEXT,
                    strategy TEXT
                )
            ''')
            
            position = opportunity['position_data']
            
            cursor.execute('''
                INSERT INTO profit_taking_trades 
                (timestamp, symbol, amount, sell_price, entry_price, profit_usdt, profit_pct, order_id, strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                opportunity['symbol'],
                order.get('amount', 0),
                position['current_price'],
                position['entry_price'],
                profit,
                position['profit_pct'],
                order.get('id', ''),
                opportunity['action']
            ))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Profit-taking logged: {opportunity['symbol']} +${profit:.2f}")
            
        except Exception as e:
            print(f"Error logging profit-taking trade: {e}")
    
    def check_stop_losses(self, positions: Dict) -> List[Dict]:
        """Check for positions that need stop-loss protection"""
        stop_loss_alerts = []
        
        for symbol, position in positions.items():
            profit_pct = position['profit_pct']
            
            # Check for significant losses
            if profit_pct <= -2.0:  # 2% loss threshold
                stop_loss_alerts.append({
                    'symbol': symbol,
                    'action': 'STOP_LOSS',
                    'reason': f'Stop loss triggered: {profit_pct:.2f}%',
                    'sell_percentage': 100,  # Sell entire position
                    'priority': 'critical',
                    'position_data': position
                })
        
        return stop_loss_alerts
    
    def run_profit_monitoring_cycle(self):
        """Execute one cycle of profit monitoring"""
        positions = self.get_current_positions()
        
        if not positions:
            return
        
        # Check profit opportunities
        profit_opportunities = self.check_profit_targets(positions)
        stop_losses = self.check_stop_losses(positions)
        
        # Execute high-priority actions first
        all_actions = profit_opportunities + stop_losses
        all_actions.sort(key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x['priority']])
        
        executed_trades = 0
        
        for action in all_actions:
            if executed_trades >= 3:  # Limit to 3 trades per cycle
                break
                
            result = self.execute_profit_taking(action)
            
            if result['success']:
                print(f"🎯 {action['action']}: {result['symbol']} +${result['profit_taken']:.2f}")
                executed_trades += 1
                time.sleep(2)  # Rate limiting
            else:
                print(f"❌ {action['action']} failed for {action['symbol']}: {result['error']}")
        
        # Display current positions
        if positions:
            print(f"\n📊 Current Positions:")
            for symbol, pos in positions.items():
                status = "🟢" if pos['profit_pct'] > 0 else "🔴"
                print(f"   {status} {symbol}: {pos['profit_pct']:+.2f}% (${pos['profit_usdt']:+.2f})")
    
    def start_profit_monitoring(self, check_interval_minutes: int = 5):
        """Start continuous profit monitoring"""
        def profit_monitor():
            while self.monitoring_active:
                try:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Checking profit opportunities...")
                    self.run_profit_monitoring_cycle()
                    time.sleep(check_interval_minutes * 60)
                    
                except Exception as e:
                    print(f"Profit monitoring error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=profit_monitor, daemon=True)
        monitor_thread.start()
        print(f"🎯 Profit monitoring started (interval: {check_interval_minutes} minutes)")
        
        return monitor_thread

def main():
    """Test automated profit taking"""
    profit_engine = AutomatedProfitTaking()
    
    print("AUTOMATED PROFIT TAKING ENGINE")
    print("=" * 40)
    
    # Run one monitoring cycle
    profit_engine.run_profit_monitoring_cycle()
    
    print("=" * 40)

if __name__ == '__main__':
    main()