#!/usr/bin/env python3
import os
import ccxt
import sqlite3

def quick_position_analysis():
    """Quick analysis of current positions for SELL opportunities"""
    try:
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'), 
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False
        })
        
        print("SELL ORDER ANALYSIS - Current Positions")
        print("=" * 45)
        
        balance = exchange.fetch_balance()
        usdt_balance = balance['USDT']['free']
        print(f"USDT Balance: ${usdt_balance:.2f}")
        
        sell_candidates = []
        
        for currency in ['BTC', 'ETH', 'SOL', 'DOGE', 'ADA']:
            if currency in balance and balance[currency]['free'] > 0:
                amount = balance[currency]['free']
                symbol = f"{currency}/USDT"
                
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                position_value = amount * current_price
                
                if position_value > 1:  # Only analyze significant positions
                    # Get entry price from database
                    entry_price = get_entry_price(symbol, amount)
                    
                    if entry_price:
                        profit_pct = ((current_price - entry_price) / entry_price) * 100
                        
                        print(f"\n{symbol}:")
                        print(f"  Amount: {amount:.6f} {currency}")
                        print(f"  Entry: ${entry_price:.4f}")
                        print(f"  Current: ${current_price:.4f}")
                        print(f"  Value: ${position_value:.2f}")
                        print(f"  P&L: {profit_pct:+.2f}%")
                        
                        # SELL conditions
                        stop_loss = entry_price * 0.98  # 2% stop-loss
                        profit_1_5 = entry_price * 1.015  # 1.5% profit
                        profit_3 = entry_price * 1.03    # 3% profit
                        profit_5 = entry_price * 1.05    # 5% profit
                        
                        if current_price <= stop_loss:
                            action = "STOP-LOSS (100%)"
                            reason = "Price hit 2% stop-loss"
                            sell_candidates.append((symbol, action, reason, 100, amount))
                        elif current_price >= profit_5:
                            action = "PROFIT-HIGH (75%)"
                            reason = "5% profit target reached"
                            sell_candidates.append((symbol, action, reason, 75, amount * 0.75))
                        elif current_price >= profit_3:
                            action = "PROFIT-MED (50%)"
                            reason = "3% profit target reached"
                            sell_candidates.append((symbol, action, reason, 50, amount * 0.50))
                        elif current_price >= profit_1_5:
                            action = "PROFIT-QUICK (25%)"
                            reason = "1.5% profit target reached"
                            sell_candidates.append((symbol, action, reason, 25, amount * 0.25))
                        else:
                            print(f"  Status: HOLD (no sell conditions)")
                            continue
                            
                        print(f"  SELL SIGNAL: {action}")
                        print(f"  Reason: {reason}")
        
        if sell_candidates:
            print(f"\nSELL ORDER EXECUTION PLAN:")
            print("=" * 30)
            for symbol, action, reason, pct, sell_amount in sell_candidates:
                proceeds = sell_amount * exchange.fetch_ticker(symbol)['last']
                print(f"{symbol}: {action}")
                print(f"  Sell: {sell_amount:.6f}")
                print(f"  Proceeds: ${proceeds:.2f}")
                print(f"  Command: exchange.create_market_sell_order('{symbol}', {sell_amount:.6f})")
                print()
        else:
            print(f"\nNo SELL conditions triggered")
            print(f"System monitoring for stop-loss and profit targets")
        
        # Check for recent SELL trades
        print(f"\nRECENT SELL ACTIVITY:")
        check_sell_history()
        
    except Exception as e:
        print(f"Error: {e}")

def get_entry_price(symbol, amount):
    """Get weighted entry price from trade history"""
    try:
        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT amount, price FROM live_trades 
            WHERE symbol = ? AND side = 'buy' 
            ORDER BY timestamp DESC LIMIT 5
        ''', (symbol,))
        
        trades = cursor.fetchall()
        conn.close()
        
        if trades:
            total_cost = sum(float(amt) * float(price) for amt, price in trades)
            total_amount = sum(float(amt) for amt, price in trades)
            return total_cost / total_amount if total_amount > 0 else None
        return None
    except:
        return None

def check_sell_history():
    """Check recent SELL trades"""
    try:
        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, amount, price, timestamp 
            FROM live_trades 
            WHERE side = 'sell' 
            ORDER BY timestamp DESC LIMIT 5
        ''')
        
        sells = cursor.fetchall()
        
        if sells:
            for symbol, amount, price, timestamp in sells:
                proceeds = float(amount) * float(price)
                print(f"  {timestamp[:19]} | SELL {amount} {symbol} @ ${price} = ${proceeds:.2f}")
        else:
            print(f"  No SELL orders executed yet")
            print(f"  System ready to execute when conditions are met")
        
        conn.close()
    except:
        print(f"  No trading history available")

if __name__ == '__main__':
    quick_position_analysis()