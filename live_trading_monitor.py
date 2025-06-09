#!/usr/bin/env python3
"""
Live Trading Performance Monitor
Real-time monitoring and adjustment of autonomous trading parameters
"""

import sqlite3
import time
import os
import ccxt
from datetime import datetime, timedelta
from typing import Dict, List

class LiveTradingMonitor:
    def __init__(self):
        self.exchange = self.connect_okx()
        self.db_path = 'trading_platform.db'
        self.live_db_path = 'live_trading.db'
        
    def connect_okx(self):
        """Connect to OKX exchange"""
        try:
            exchange = ccxt.okx({
                'apiKey': os.environ.get('OKX_API_KEY'),
                'secret': os.environ.get('OKX_SECRET_KEY'),
                'password': os.environ.get('OKX_PASSPHRASE'),
                'sandbox': False,
                'rateLimit': 2000,
                'enableRateLimit': True,
            })
            return exchange
        except Exception as e:
            print(f"OKX connection error: {e}")
            return None
    
    def check_signal_execution_status(self) -> Dict:
        """Check if signals are being executed properly"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent high-confidence signals
            cursor.execute('''
                SELECT symbol, signal, confidence, timestamp, id
                FROM ai_signals 
                WHERE confidence >= 60 
                AND datetime(timestamp) >= datetime('now', '-1 hour')
                ORDER BY id DESC LIMIT 10
            ''')
            signals = cursor.fetchall()
            
            # Check if live trades database exists
            try:
                live_conn = sqlite3.connect(self.live_db_path)
                live_cursor = live_conn.cursor()
                
                live_cursor.execute('''
                    SELECT COUNT(*) FROM live_trades 
                    WHERE datetime(timestamp) >= datetime('now', '-1 hour')
                ''')
                recent_trades = live_cursor.fetchone()[0]
                live_conn.close()
            except:
                recent_trades = 0
            
            conn.close()
            
            return {
                'high_confidence_signals': len(signals),
                'recent_trades_executed': recent_trades,
                'execution_rate': recent_trades / max(len(signals), 1) * 100,
                'signals': signals[:5]  # Last 5 signals
            }
            
        except Exception as e:
            print(f"Error checking execution status: {e}")
            return {}
    
    def check_portfolio_performance(self) -> Dict:
        """Monitor real-time portfolio performance"""
        try:
            if not self.exchange:
                return {}
                
            balance = self.exchange.fetch_balance()
            usdt_balance = float(balance['USDT']['free'])
            
            # Calculate total portfolio value
            total_value = usdt_balance
            positions = []
            
            for currency in balance:
                if currency != 'USDT' and balance[currency]['free'] > 0:
                    amount = float(balance[currency]['free'])
                    if amount > 0:
                        try:
                            symbol = f"{currency}/USDT"
                            ticker = self.exchange.fetch_ticker(symbol)
                            price = float(ticker['last'])
                            value = amount * price
                            total_value += value
                            
                            positions.append({
                                'symbol': symbol,
                                'amount': amount,
                                'price': price,
                                'value': value
                            })
                        except:
                            continue
            
            return {
                'total_value': total_value,
                'usdt_balance': usdt_balance,
                'positions': positions,
                'position_count': len(positions)
            }
            
        except Exception as e:
            print(f"Error checking portfolio: {e}")
            return {}
    
    def adjust_trading_parameters(self, performance_data: Dict) -> Dict:
        """Suggest trading parameter adjustments based on performance"""
        suggestions = []
        
        if performance_data.get('execution_rate', 0) < 50:
            suggestions.append({
                'parameter': 'confidence_threshold',
                'current': '60%',
                'suggested': '55%',
                'reason': 'Low execution rate - consider lowering confidence threshold'
            })
        
        if performance_data.get('position_count', 0) == 0:
            suggestions.append({
                'parameter': 'position_sizing',
                'current': '1%',
                'suggested': '1.5%',
                'reason': 'No active positions - consider increasing position size'
            })
        
        if len(performance_data.get('positions', [])) > 5:
            suggestions.append({
                'parameter': 'max_positions',
                'current': 'unlimited',
                'suggested': '5',
                'reason': 'High position count - consider limiting concurrent positions'
            })
        
        return {
            'suggestions': suggestions,
            'auto_adjust': False,  # Manual approval required
            'timestamp': datetime.now().isoformat()
        }
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        execution_status = self.check_signal_execution_status()
        portfolio_data = self.check_portfolio_performance()
        adjustments = self.adjust_trading_parameters({**execution_status, **portfolio_data})
        
        return {
            'timestamp': datetime.now().isoformat(),
            'signal_execution': execution_status,
            'portfolio': portfolio_data,
            'suggested_adjustments': adjustments,
            'system_health': 'operational' if execution_status.get('high_confidence_signals', 0) > 0 else 'monitoring'
        }
    
    def print_live_status(self):
        """Print real-time trading status"""
        report = self.generate_performance_report()
        
        print("\n🚀 LIVE TRADING SYSTEM STATUS")
        print("=" * 50)
        
        # Signal execution
        execution = report['signal_execution']
        print(f"📊 Signals (1h): {execution.get('high_confidence_signals', 0)} generated")
        print(f"⚡ Trades Executed: {execution.get('recent_trades_executed', 0)}")
        print(f"🎯 Execution Rate: {execution.get('execution_rate', 0):.1f}%")
        
        # Portfolio status
        portfolio = report['portfolio']
        if portfolio:
            print(f"💰 Portfolio Value: ${portfolio.get('total_value', 0):.2f}")
            print(f"💵 USDT Available: ${portfolio.get('usdt_balance', 0):.2f}")
            print(f"📈 Active Positions: {portfolio.get('position_count', 0)}")
        
        # Recent signals
        if execution.get('signals'):
            print("\n📋 Recent High-Confidence Signals:")
            for signal in execution['signals']:
                timestamp = signal[3][:16] if len(signal[3]) > 16 else signal[3]
                print(f"   {signal[0]} {signal[1]} @ {signal[2]:.0f}% - {timestamp}")
        
        # Suggestions
        suggestions = report['suggested_adjustments']['suggestions']
        if suggestions:
            print("\n💡 Performance Optimization Suggestions:")
            for suggestion in suggestions:
                print(f"   • {suggestion['parameter']}: {suggestion['current']} → {suggestion['suggested']}")
                print(f"     Reason: {suggestion['reason']}")
        
        print(f"\n🔄 System Health: {report['system_health'].upper()}")
        print("=" * 50)

def main():
    """Monitor live trading performance"""
    monitor = LiveTradingMonitor()
    monitor.print_live_status()

if __name__ == '__main__':
    main()