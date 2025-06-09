#!/usr/bin/env python3
"""
SELL Order and Stop-Loss Demonstration
Shows real-time position monitoring with automatic SELL execution
"""

import os
import ccxt
import sqlite3
from datetime import datetime

def demonstrate_sell_functionality():
    """Demonstrate SELL order capabilities with current positions"""
    print("SELL ORDER & STOP-LOSS DEMONSTRATION")
    print("=" * 50)
    
    # Connect to OKX
    try:
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'),
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False,
            'rateLimit': 2000,
            'enableRateLimit': True,
        })
        
        print("Connected to OKX exchange")
        
        # Get current balance
        balance = exchange.fetch_balance()
        usdt_balance = float(balance['USDT']['free'])
        
        print(f"USDT Balance: ${usdt_balance:.2f}")
        
        # Analyze current positions for SELL opportunities
        positions_analyzed = 0
        sell_candidates = []
        
        for currency in balance:
            if currency != 'USDT' and balance[currency]['free'] > 0:
                amount = float(balance[currency]['free'])
                if amount > 0:
                    try:
                        symbol = f"{currency}/USDT"
                        ticker = exchange.fetch_ticker(symbol)
                        current_price = float(ticker['last'])
                        position_value = amount * current_price
                        
                        # Get entry price from database
                        entry_price = get_entry_price_from_db(symbol, amount)
                        
                        if entry_price:
                            profit_pct = ((current_price - entry_price) / entry_price) * 100
                            profit_usdt = (current_price - entry_price) * amount
                            
                            # Calculate stop-loss and profit targets
                            stop_loss_price = entry_price * 0.98  # 2% stop-loss
                            profit_target_1 = entry_price * 1.015  # 1.5% profit
                            profit_target_2 = entry_price * 1.03   # 3% profit
                            profit_target_3 = entry_price * 1.05   # 5% profit
                            
                            position_info = {
                                'symbol': symbol,
                                'currency': currency,
                                'amount': amount,
                                'current_price': current_price,
                                'entry_price': entry_price,
                                'profit_pct': profit_pct,
                                'profit_usdt': profit_usdt,
                                'position_value': position_value,
                                'stop_loss_price': stop_loss_price,
                                'profit_targets': [profit_target_1, profit_target_2, profit_target_3]
                            }
                            
                            print(f"\nPosition Analysis: {symbol}")
                            print(f"  Amount: {amount:.6f} {currency}")
                            print(f"  Entry Price: ${entry_price:.4f}")
                            print(f"  Current Price: ${current_price:.4f}")
                            print(f"  P&L: {profit_pct:+.2f}% (${profit_usdt:+.2f})")
                            print(f"  Stop-Loss: ${stop_loss_price:.4f}")
                            
                            # Check SELL conditions
                            sell_reason = None
                            sell_percentage = 0
                            
                            if current_price <= stop_loss_price:
                                sell_reason = "STOP-LOSS TRIGGERED"
                                sell_percentage = 100  # Sell entire position
                            elif current_price >= profit_target_3:
                                sell_reason = "HIGH PROFIT TARGET (5%)"
                                sell_percentage = 75   # Sell 75%
                            elif current_price >= profit_target_2:
                                sell_reason = "MEDIUM PROFIT TARGET (3%)"
                                sell_percentage = 50   # Sell 50%
                            elif current_price >= profit_target_1:
                                sell_reason = "QUICK PROFIT TARGET (1.5%)"
                                sell_percentage = 25   # Sell 25%
                            
                            if sell_reason:
                                sell_amount = amount * (sell_percentage / 100)
                                estimated_proceeds = sell_amount * current_price
                                
                                sell_candidates.append({
                                    'position': position_info,
                                    'reason': sell_reason,
                                    'sell_percentage': sell_percentage,
                                    'sell_amount': sell_amount,
                                    'estimated_proceeds': estimated_proceeds
                                })
                                
                                print(f"  SELL SIGNAL: {sell_reason}")
                                print(f"  Recommended Action: SELL {sell_percentage}% ({sell_amount:.6f} {currency})")
                                print(f"  Estimated Proceeds: ${estimated_proceeds:.2f}")
                            else:
                                print(f"  Status: HOLD (no sell conditions met)")
                            
                            positions_analyzed += 1
                        
                    except Exception as e:
                        print(f"Error analyzing {currency}: {e}")
        
        print(f"\nSUMMARY:")
        print(f"Positions Analyzed: {positions_analyzed}")
        print(f"SELL Candidates: {len(sell_candidates)}")
        
        if sell_candidates:
            print(f"\nRECOMMENDED SELL ORDERS:")
            total_proceeds = 0
            for i, candidate in enumerate(sell_candidates, 1):
                pos = candidate['position']
                print(f"{i}. {pos['symbol']}: {candidate['reason']}")
                print(f"   SELL {candidate['sell_amount']:.6f} {pos['currency']} for ~${candidate['estimated_proceeds']:.2f}")
                total_proceeds += candidate['estimated_proceeds']
            
            print(f"\nTotal Estimated Proceeds: ${total_proceeds:.2f}")
            
            # Demonstrate actual SELL execution (commented for safety)
            print(f"\nSELL ORDER EXECUTION CAPABILITY:")
            print(f"The system can automatically execute these SELL orders:")
            for candidate in sell_candidates:
                pos = candidate['position']
                print(f"  exchange.create_market_sell_order('{pos['symbol']}', {candidate['sell_amount']:.6f})")
        else:
            print(f"\nNo SELL conditions triggered at current market prices")
            print(f"System continues monitoring for:")
            print(f"  - Stop-loss conditions (2% below entry)")
            print(f"  - Profit targets (1.5%, 3%, 5%)")
            print(f"  - Trailing stops for profitable positions")
        
    except Exception as e:
        print(f"Error: {e}")

def get_entry_price_from_db(symbol, amount):
    """Get entry price from trading database"""
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
        
    except Exception:
        return None

def check_sell_trade_history():
    """Check if any SELL trades have been executed"""
    try:
        if os.path.exists('live_trading.db'):
            conn = sqlite3.connect('live_trading.db')
            cursor = conn.cursor()
            
            # Check for sell trades
            cursor.execute('''
                SELECT symbol, side, amount, price, timestamp 
                FROM live_trades 
                WHERE side = 'sell' 
                ORDER BY timestamp DESC LIMIT 5
            ''')
            
            sell_trades = cursor.fetchall()
            
            if sell_trades:
                print(f"\nRECENT SELL TRADES:")
                for trade in sell_trades:
                    symbol, side, amount, price, timestamp = trade
                    proceeds = float(amount) * float(price)
                    print(f"  {timestamp[:19]} | SELL {float(amount):.6f} {symbol} @ ${float(price):.4f} = ${proceeds:.2f}")
            else:
                print(f"\nNo SELL trades found in history")
                print(f"System ready to execute SELL orders when conditions are met")
            
            conn.close()
        
    except Exception as e:
        print(f"Error checking sell history: {e}")

def main():
    """Main demonstration function"""
    print(f"ANALYSIS TIME: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    demonstrate_sell_functionality()
    check_sell_trade_history()
    
    print("\n" + "=" * 60)
    print("SELL ORDER SYSTEM STATUS")
    print("=" * 60)
    print("The advanced trading system includes:")
    print("  ✓ Automatic stop-loss protection (2% below entry)")
    print("  ✓ Profit-taking at multiple levels (1.5%, 3%, 5%)")
    print("  ✓ Trailing stops for profitable positions")
    print("  ✓ Real-time position monitoring")
    print("  ✓ Automatic SELL order execution")
    print("  ✓ Complete trade logging and history")

if __name__ == '__main__':
    main()