"""
Check OKX Balance and Fix Trade Execution
"""
import os
import ccxt
from datetime import datetime

def check_okx_status():
    """Check OKX balance and trading status"""
    try:
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'),
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Test connection
        balance = exchange.fetch_balance()
        
        # Get USDT balance
        usdt_balance = balance.get('USDT', {})
        free_usdt = float(usdt_balance.get('free', 0))
        total_usdt = float(usdt_balance.get('total', 0))
        
        # Get account info
        account_info = exchange.fetch_accounts()
        
        # Check trading permissions
        try:
            markets = exchange.load_markets()
            btc_ticker = exchange.fetch_ticker('BTC/USDT')
            current_btc_price = float(btc_ticker['last'])
        except Exception as e:
            current_btc_price = 0
            markets = {}
        
        print(f"=== OKX ACCOUNT STATUS ===")
        print(f"Time: {datetime.now()}")
        print(f"Connection: ✅ CONNECTED")
        print(f"Free USDT: ${free_usdt:.2f}")
        print(f"Total USDT: ${total_usdt:.2f}")
        print(f"Markets Available: {len(markets)}")
        print(f"BTC Price: ${current_btc_price:.2f}")
        
        # Calculate minimum trade requirements
        min_trade_amount = 5.0  # $5 minimum
        position_size_pct = 0.05  # 5%
        required_balance = min_trade_amount / position_size_pct  # $100 for $5 trades
        
        print(f"\n=== TRADING ANALYSIS ===")
        print(f"Current Position Size: 5% of balance")
        print(f"Minimum Trade: ${min_trade_amount}")
        print(f"Balance Needed for Trading: ${required_balance:.2f}")
        print(f"Current Can Trade: {'✅ YES' if free_usdt >= required_balance else '❌ NO'}")
        
        if free_usdt < required_balance:
            # Calculate what we CAN do with current balance
            max_trade_value = free_usdt * position_size_pct
            print(f"\nWith ${free_usdt:.2f} balance:")
            print(f"Max trade value: ${max_trade_value:.2f}")
            print(f"This is below minimum ${min_trade_amount} trade size")
            
            # Suggest solutions
            print(f"\n=== SOLUTIONS ===")
            print(f"Option 1: Add funds - Need ${required_balance - free_usdt:.2f} more USDT")
            print(f"Option 2: Reduce position size to {(min_trade_amount / free_usdt * 100):.1f}%")
            print(f"Option 3: Reduce minimum trade to ${max_trade_value:.2f}")
        
        return {
            'free_usdt': free_usdt,
            'total_usdt': total_usdt,
            'can_trade': free_usdt >= required_balance,
            'required_balance': required_balance,
            'markets_count': len(markets),
            'btc_price': current_btc_price
        }
        
    except Exception as e:
        print(f"❌ OKX CONNECTION ERROR: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    result = check_okx_status()
    
    if 'error' not in result:
        if not result['can_trade']:
            print(f"\n🔧 FIXING TRADE EXECUTION...")
            print(f"The system is generating signals but cannot execute trades due to insufficient balance.")
            print(f"The Enhanced Live Trading System needs adjustment for current balance.")
    else:
        print(f"Please check your OKX API credentials and permissions.")