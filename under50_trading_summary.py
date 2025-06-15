"""
Under $50 Futures Trading Engine - Performance Summary
Real-time analysis of trading signals and performance
"""

import sqlite3
import pandas as pd
from datetime import datetime

def generate_trading_summary():
    print("🎯 Under $50 Futures Trading Engine - Live Performance Summary")
    print("=" * 70)
    
    try:
        conn = sqlite3.connect('under50_futures_trading.db')
        
        # Get signal statistics
        signals_df = pd.read_sql_query("SELECT * FROM under50_futures_signals ORDER BY timestamp DESC", conn)
        trades_df = pd.read_sql_query("SELECT * FROM under50_futures_trades ORDER BY entry_time DESC", conn)
        
        if len(signals_df) > 0:
            print(f"📊 Total Signals Generated: {len(signals_df)}")
            print(f"💰 Total Trades Executed: {len(trades_df)}")
            print()
            
            # Signal distribution by type
            signal_dist = signals_df['signal'].value_counts()
            print("📈 Signal Distribution:")
            for signal_type, count in signal_dist.items():
                print(f"   {signal_type}: {count} signals")
            print()
            
            # Price tier analysis
            tier_dist = signals_df['price_tier'].value_counts()
            print("🏷️ Price Tier Distribution:")
            for tier, count in tier_dist.items():
                avg_conf = signals_df[signals_df['price_tier'] == tier]['confidence'].mean()
                print(f"   {tier}: {count} signals (Avg Confidence: {avg_conf:.1f}%)")
            print()
            
            # Top performing signals
            top_signals = signals_df.nlargest(5, 'confidence')[['symbol', 'signal', 'confidence', 'price', 'price_tier', 'leverage']]
            print("🌟 Top 5 High-Confidence Signals:")
            for _, signal in top_signals.iterrows():
                print(f"   {signal['symbol']}: {signal['signal']} (Conf: {signal['confidence']:.1f}%, "
                      f"Price: ${signal['price']:.6f}, Tier: {signal['price_tier']}, Leverage: {signal['leverage']}x)")
            print()
            
            # Leverage distribution
            leverage_dist = signals_df['leverage'].value_counts().sort_index()
            print("⚖️ Leverage Distribution:")
            for leverage, count in leverage_dist.items():
                print(f"   {leverage}x: {count} signals")
            print()
            
            # Recent signals
            recent_signals = signals_df.head(10)[['symbol', 'signal', 'confidence', 'price', 'price_tier']]
            print("🕐 Most Recent Signals:")
            for _, signal in recent_signals.iterrows():
                print(f"   {signal['symbol']}: {signal['signal']} (Conf: {signal['confidence']:.1f}%, "
                      f"${signal['price']:.6f}, {signal['price_tier']})")
            
            # Performance metrics
            if len(trades_df) > 0:
                print()
                print("📊 Trading Performance Metrics:")
                avg_confidence = trades_df['confidence'].mean()
                high_conf_trades = len(trades_df[trades_df['confidence'] >= 80])
                
                print(f"   Average Confidence: {avg_confidence:.1f}%")
                print(f"   High Confidence Trades (≥80%): {high_conf_trades}")
                
                # Potential returns analysis
                if 'pnl_percentage' in trades_df.columns:
                    avg_potential_return = trades_df['pnl_percentage'].mean()
                    max_potential_return = trades_df['pnl_percentage'].max()
                    print(f"   Average Potential Return: {avg_potential_return:.1f}%")
                    print(f"   Maximum Potential Return: {max_potential_return:.1f}%")
        
        else:
            print("No signals generated yet - engine still initializing")
        
        conn.close()
        
        print()
        print("=" * 70)
        print(f"Summary Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"Error generating summary: {e}")

if __name__ == "__main__":
    generate_trading_summary()