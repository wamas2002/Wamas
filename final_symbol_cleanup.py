"""
Final Symbol Cleanup
Removes all invalid symbols from futures trading engine
"""
import re
import os

def fix_futures_engine_symbols():
    """Fix futures engine to only use verified symbols"""
    
    verified_futures_symbols = [
        'BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'XRP/USDT:USDT', 'ADA/USDT:USDT',
        'SOL/USDT:USDT', 'DOGE/USDT:USDT', 'LINK/USDT:USDT', 'LTC/USDT:USDT', 'DOT/USDT:USDT',
        'AVAX/USDT:USDT', 'UNI/USDT:USDT', 'ATOM/USDT:USDT', 'NEAR/USDT:USDT', 'TRX/USDT:USDT',
        'ICP/USDT:USDT', 'ALGO/USDT:USDT', 'HBAR/USDT:USDT', 'XLM/USDT:USDT', 'SAND/USDT:USDT',
        'MANA/USDT:USDT', 'THETA/USDT:USDT', 'AXS/USDT:USDT', 'FIL/USDT:USDT', 'ETC/USDT:USDT',
        'EGLD/USDT:USDT', 'FLOW/USDT:USDT', 'ENJ/USDT:USDT', 'CHZ/USDT:USDT', 'CRV/USDT:USDT'
    ]
    
    filename = 'advanced_futures_trading_engine.py'
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            content = f.read()
        
        # Replace the symbol list with verified symbols only
        symbol_pattern = r'self\.symbols\s*=\s*\[.*?\]'
        new_symbols = f"self.symbols = {verified_futures_symbols}"
        
        content = re.sub(symbol_pattern, new_symbols, content, flags=re.DOTALL)
        
        with open(filename, 'w') as f:
            f.write(content)
        
        print(f"Updated {filename} with verified symbols only")

def main():
    fix_futures_engine_symbols()
    print("✅ Final symbol cleanup complete")
    print("🎯 All invalid symbols removed from futures engine")

if __name__ == "__main__":
    main()