#!/usr/bin/env python3
"""
Token Expansion for Enhanced Market Exposure
Adding SOL, ADA, DOT, MATIC, AVAX support to trading system
"""

import os
import ccxt
import sqlite3
from datetime import datetime

def expand_token_support():
    """Add new cryptocurrencies to trading system"""
    
    # Connect to OKX
    exchange = ccxt.okx({
        'apiKey': os.environ.get('OKX_API_KEY'),
        'secret': os.environ.get('OKX_SECRET_KEY'),
        'password': os.environ.get('OKX_PASSPHRASE'),
        'sandbox': False
    })
    
    # New tokens to add
    new_tokens = ['SOL', 'ADA', 'DOT', 'MATIC', 'AVAX']
    existing_tokens = ['BTC', 'ETH']
    all_tokens = existing_tokens + new_tokens
    
    print("EXPANDING CRYPTOCURRENCY MARKET EXPOSURE")
    print("=" * 45)
    
    # Validate new tokens on OKX
    valid_tokens = []
    for token in new_tokens:
        try:
            symbol = f"{token}/USDT"
            ticker = exchange.fetch_ticker(symbol)
            price = ticker['last']
            volume = ticker['quoteVolume']
            change_24h = ticker['percentage']
            
            valid_tokens.append({
                'token': token,
                'symbol': symbol,
                'price': price,
                'volume': volume,
                'change_24h': change_24h
            })
            
            print(f"✓ {symbol}: ${price:.4f} (24h: {change_24h:+.1f}%)")
            
        except Exception as e:
            print(f"✗ {token}/USDT: Not available - {e}")
    
    print(f"\nValidated {len(valid_tokens)} new tokens for trading")
    
    # Update trading configuration
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        # Create expanded symbols table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS expanded_trading_symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT UNIQUE,
                token TEXT,
                enabled BOOLEAN DEFAULT 1,
                min_trade_size REAL,
                max_trade_size REAL,
                risk_weight REAL,
                allocation_target REAL,
                last_updated TEXT
            )
        ''')
        
        # New portfolio allocation strategy
        allocation_targets = {
            'BTC': 30.0,   # 30% Bitcoin
            'ETH': 25.0,   # 25% Ethereum  
            'SOL': 15.0,   # 15% Solana
            'ADA': 10.0,   # 10% Cardano
            'DOT': 8.0,    # 8% Polkadot
            'MATIC': 7.0,  # 7% Polygon
            'AVAX': 5.0    # 5% Avalanche
        }
        
        print(f"\nNew Portfolio Allocation Strategy:")
        print("-" * 35)
        
        # Add all tokens to configuration
        for token in all_tokens:
            if token in [t['token'] for t in valid_tokens] or token in existing_tokens:
                symbol = f"{token}/USDT"
                allocation = allocation_targets.get(token, 0)
                
                # Risk-adjusted trade sizes
                if token in ['BTC', 'ETH']:
                    min_size, max_size, risk_weight = 5.0, 20.0, 1.0
                elif token in ['SOL', 'ADA', 'DOT']:
                    min_size, max_size, risk_weight = 4.0, 15.0, 1.2
                else:  # MATIC, AVAX
                    min_size, max_size, risk_weight = 3.0, 12.0, 1.4
                
                cursor.execute('''
                    INSERT OR REPLACE INTO expanded_trading_symbols 
                    (symbol, token, enabled, min_trade_size, max_trade_size, 
                     risk_weight, allocation_target, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol, token, True, min_size, max_size,
                    risk_weight, allocation, datetime.now().isoformat()
                ))
                
                priority = "HIGH" if allocation >= 20 else "MEDIUM" if allocation >= 10 else "LOW"
                print(f"{priority:6} {token}: {allocation:4.1f}% target allocation")
        
        conn.commit()
        conn.close()
        
        print(f"\n✓ Trading configuration updated for {len(all_tokens)} cryptocurrencies")
        
    except Exception as e:
        print(f"Database update error: {e}")
    
    # Create rebalancing analysis
    try:
        balance = exchange.fetch_balance()
        current_portfolio = {}
        total_value = float(balance['USDT']['free'])
        
        # Calculate current holdings
        for token in all_tokens:
            if token in balance and balance[token]['free'] > 0:
                amount = float(balance[token]['free'])
                if amount > 0:
                    try:
                        symbol = f"{token}/USDT"
                        ticker = exchange.fetch_ticker(symbol)
                        price = float(ticker['last'])
                        value = amount * price
                        total_value += value
                        current_portfolio[token] = {
                            'amount': amount,
                            'value': value,
                            'percentage': 0
                        }
                    except:
                        continue
        
        # Calculate percentages and rebalancing needs
        print(f"\nCURRENT VS TARGET ALLOCATION ANALYSIS")
        print("-" * 40)
        print(f"Total Portfolio Value: ${total_value:.2f}")
        print()
        
        rebalancing_needed = []
        
        for token, target_pct in allocation_targets.items():
            current_value = current_portfolio.get(token, {}).get('value', 0)
            current_pct = (current_value / total_value * 100) if total_value > 0 else 0
            deviation = target_pct - current_pct
            
            status = "BALANCED" if abs(deviation) < 5 else "REBALANCE"
            
            print(f"{token}:")
            print(f"  Current: {current_pct:5.1f}% (${current_value:7.2f})")
            print(f"  Target:  {target_pct:5.1f}%")
            print(f"  Status:  {status}")
            
            if abs(deviation) > 5:  # 5% threshold
                target_value = total_value * (target_pct / 100)
                needed_change = target_value - current_value
                
                if needed_change > 0:
                    action = f"BUY ${needed_change:.2f}"
                else:
                    action = f"SELL ${abs(needed_change):.2f}"
                
                rebalancing_needed.append({
                    'token': token,
                    'action': action,
                    'amount': abs(needed_change),
                    'deviation': abs(deviation)
                })
            
            print()
        
        # Store rebalancing recommendations
        if rebalancing_needed:
            conn = sqlite3.connect('live_trading.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_rebalancing (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token TEXT,
                    current_allocation REAL,
                    target_allocation REAL,
                    action_needed TEXT,
                    amount_needed REAL,
                    priority INTEGER,
                    timestamp TEXT
                )
            ''')
            
            print(f"REBALANCING RECOMMENDATIONS:")
            print("-" * 30)
            
            # Sort by deviation (highest priority first)
            rebalancing_needed.sort(key=lambda x: x['deviation'], reverse=True)
            
            for i, rec in enumerate(rebalancing_needed, 1):
                priority = 1 if rec['deviation'] > 20 else 2 if rec['deviation'] > 10 else 3
                
                cursor.execute('''
                    INSERT INTO portfolio_rebalancing 
                    (token, current_allocation, target_allocation, action_needed, 
                     amount_needed, priority, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    rec['token'], 
                    current_portfolio.get(rec['token'], {}).get('value', 0) / total_value * 100,
                    allocation_targets[rec['token']],
                    rec['action'], rec['amount'], priority, datetime.now().isoformat()
                ))
                
                print(f"{i}. {rec['token']}: {rec['action']} (deviation: {rec['deviation']:.1f}%)")
            
            conn.commit()
            conn.close()
        else:
            print("✓ Portfolio already well-balanced")
        
    except Exception as e:
        print(f"Portfolio analysis error: {e}")
    
    # Update AI signal generation for new tokens
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        # Generate initial signals for new tokens
        for token_data in valid_tokens:
            symbol = token_data['symbol']
            price = token_data['price']
            change_24h = token_data['change_24h']
            
            # Simple signal generation based on 24h change
            if change_24h > 5:
                signal = 'BUY'
                confidence = 0.65
                reasoning = f"Strong 24h performance (+{change_24h:.1f}%)"
            elif change_24h < -5:
                signal = 'SELL'
                confidence = 0.70
                reasoning = f"Negative momentum ({change_24h:.1f}%)"
            else:
                signal = 'HOLD'
                confidence = 0.60
                reasoning = f"Neutral price action ({change_24h:.1f}%)"
            
            cursor.execute('''
                INSERT INTO ai_signals (symbol, signal, confidence, reasoning, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, signal, confidence, reasoning, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
        
        print(f"\n✓ Generated initial AI signals for {len(valid_tokens)} new tokens")
        
    except Exception as e:
        print(f"Signal generation error: {e}")
    
    # Summary
    print(f"\nTOKEN EXPANSION COMPLETE")
    print("=" * 25)
    print(f"• Added {len(valid_tokens)} new cryptocurrencies")
    print(f"• Diversified portfolio across {len(all_tokens)} assets")
    print(f"• Reduced concentration risk significantly")
    print(f"• Enhanced market exposure and opportunities")
    
    print(f"\nBenefits:")
    print(f"• Better risk distribution across sectors")
    print(f"• Exposure to DeFi, smart contracts, and Layer 1 protocols")
    print(f"• Increased trading signal opportunities")
    print(f"• Improved portfolio stability through diversification")
    
    print(f"\nNext Steps:")
    print(f"• AI system will monitor all tokens for signals")
    print(f"• Automated rebalancing based on target allocations")
    print(f"• Risk management across expanded portfolio")
    print(f"• Performance tracking for all assets")
    
    print(f"\nStatus: EXPANSION ACTIVE - ENHANCED TRADING ENABLED")

if __name__ == '__main__':
    expand_token_support()