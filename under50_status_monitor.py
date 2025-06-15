"""
Under $50 Futures Trading Status Monitor
Quick status check for the trading engine
"""

import sqlite3
import ccxt
import os
from datetime import datetime

def check_engine_status():
    print("🔍 Under $50 Futures Trading Engine Status Check")
    print("=" * 50)
    
    try:
        # Check OKX connection
        exchange = ccxt.okx({
            'apiKey': os.getenv('OKX_API_KEY'),
            'secret': os.getenv('OKX_SECRET_KEY'),
            'password': os.getenv('OKX_PASSPHRASE'),
            'sandbox': False,
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'}
        })
        
        balance = exchange.fetch_balance()
        print("✅ OKX Connection: ACTIVE")
        
        # Check some under $50 tokens
        test_symbols = ['ADA/USDT:USDT', 'DOGE/USDT:USDT', 'CHZ/USDT:USDT']
        under_50_count = 0
        
        for symbol in test_symbols:
            try:
                ticker = exchange.fetch_ticker(symbol)
                price = ticker['last']
                if price < 50:
                    under_50_count += 1
                    print(f"✅ {symbol}: ${price:.4f} - UNDER $50")
                else:
                    print(f"❌ {symbol}: ${price:.4f} - OVER $50")
            except Exception as e:
                print(f"⚠️ {symbol}: Error - {e}")
        
        print(f"\n📊 Tokens under $50: {under_50_count}/{len(test_symbols)}")
        
        # Check database
        try:
            conn = sqlite3.connect('under50_futures_trading.db')
            cursor = conn.cursor()
            
            # Check signals
            cursor.execute("SELECT COUNT(*) FROM under50_futures_signals")
            signal_count = cursor.fetchone()[0]
            
            # Check trades
            cursor.execute("SELECT COUNT(*) FROM under50_futures_trades")
            trade_count = cursor.fetchone()[0]
            
            print(f"📈 Database Signals: {signal_count}")
            print(f"💰 Database Trades: {trade_count}")
            
            # Get latest signals if any
            if signal_count > 0:
                cursor.execute("SELECT symbol, signal, confidence, price, price_tier FROM under50_futures_signals ORDER BY timestamp DESC LIMIT 3")
                recent_signals = cursor.fetchall()
                print("\n🎯 Recent Signals:")
                for signal in recent_signals:
                    print(f"   {signal[0]} {signal[1]} (Conf: {signal[2]}%, Price: ${signal[3]:.6f}, Tier: {signal[4]})")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Database Error: {e}")
        
        # Generate a test signal for CHZ (very low price)
        print("\n🧪 Testing Signal Generation for CHZ...")
        try:
            chz_ticker = exchange.fetch_ticker('CHZ/USDT:USDT')
            chz_price = chz_ticker['last']
            
            # Get OHLCV data
            ohlcv = exchange.fetch_ohlcv('CHZ/USDT:USDT', '15m', limit=50)
            if len(ohlcv) >= 20:
                prices = [candle[4] for candle in ohlcv]  # Close prices
                
                # Simple RSI calculation
                gains = []
                losses = []
                for i in range(1, len(prices)):
                    change = prices[i] - prices[i-1]
                    if change > 0:
                        gains.append(change)
                        losses.append(0)
                    else:
                        gains.append(0)
                        losses.append(abs(change))
                
                if len(gains) >= 14:
                    avg_gain = sum(gains[-14:]) / 14
                    avg_loss = sum(losses[-14:]) / 14
                    
                    if avg_loss > 0:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
                        
                        print(f"   CHZ Price: ${chz_price:.6f}")
                        print(f"   CHZ RSI: {rsi:.1f}")
                        
                        if rsi < 35:
                            print(f"   🟢 POTENTIAL BUY SIGNAL (Oversold)")
                        elif rsi > 65:
                            print(f"   🔴 POTENTIAL SELL SIGNAL (Overbought)")
                        else:
                            print(f"   🟡 NEUTRAL ZONE")
                    else:
                        print(f"   ⚠️ No price movement detected")
                else:
                    print(f"   ⚠️ Insufficient data for RSI calculation")
            else:
                print(f"   ❌ Insufficient OHLCV data")
                
        except Exception as e:
            print(f"   ❌ CHZ Test Failed: {e}")
        
        print("\n" + "=" * 50)
        print(f"Status Check Complete - {datetime.now().strftime('%H:%M:%S')}")
        
    except Exception as e:
        print(f"❌ Status Check Failed: {e}")

if __name__ == "__main__":
    check_engine_status()