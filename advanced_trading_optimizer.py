#!/usr/bin/env python3
"""
Advanced Trading System Optimizer
Dynamic parameter adjustment and performance enhancement engine
"""

import sqlite3
import os
import ccxt
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading

class AdvancedTradingOptimizer:
    def __init__(self):
        self.exchange = self.connect_okx()
        self.optimization_active = True
        self.current_params = {
            'confidence_threshold': 60.0,
            'position_size_pct': 1.0,
            'risk_limit_pct': 1.0,
            'max_concurrent_trades': 5,
            'profit_target_pct': 3.0,
            'stop_loss_pct': 1.5
        }
        
    def connect_okx(self):
        """Connect to OKX exchange with error handling"""
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
    
    def analyze_market_volatility(self) -> Dict:
        """Real-time market volatility analysis"""
        if not self.exchange:
            return {'volatility': 'unknown', 'trend': 'neutral'}
        
        try:
            symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
            volatility_data = []
            
            for symbol in symbols:
                ticker = self.exchange.fetch_ticker(symbol)
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', limit=24)
                
                high_24h = max([candle[2] for candle in ohlcv])
                low_24h = min([candle[3] for candle in ohlcv])
                current_price = float(ticker['last'])
                
                volatility = ((high_24h - low_24h) / current_price) * 100
                volatility_data.append(volatility)
            
            avg_volatility = sum(volatility_data) / len(volatility_data)
            
            if avg_volatility > 6:
                vol_level = 'extreme'
            elif avg_volatility > 4:
                vol_level = 'high'
            elif avg_volatility > 2:
                vol_level = 'medium'
            else:
                vol_level = 'low'
            
            return {
                'volatility': vol_level,
                'volatility_pct': avg_volatility,
                'individual_volatilities': dict(zip(symbols, volatility_data))
            }
            
        except Exception as e:
            print(f"Market analysis error: {e}")
            return {'volatility': 'medium', 'trend': 'neutral'}
    
    def get_portfolio_performance(self) -> Dict:
        """Calculate real-time portfolio performance metrics"""
        try:
            if not self.exchange:
                return {}
            
            balance = self.exchange.fetch_balance()
            usdt_balance = float(balance['USDT']['free'])
            
            # Calculate total portfolio value and positions
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
                                'symbol': currency,
                                'amount': amount,
                                'current_price': price,
                                'value': value,
                                'allocation_pct': (value / total_value) * 100
                            })
                        except:
                            continue
            
            return {
                'total_value': total_value,
                'usdt_balance': usdt_balance,
                'positions': positions,
                'position_count': len(positions),
                'diversification_score': self.calculate_diversification_score(positions)
            }
            
        except Exception as e:
            print(f"Portfolio analysis error: {e}")
            return {}
    
    def calculate_diversification_score(self, positions: List[Dict]) -> float:
        """Calculate portfolio diversification score (0-100)"""
        if len(positions) == 0:
            return 0
        
        # Calculate allocation concentration
        allocations = [pos['allocation_pct'] for pos in positions]
        max_allocation = max(allocations) if allocations else 0
        
        # Score based on number of positions and concentration
        position_score = min(len(positions) * 20, 60)  # Max 60 for positions
        concentration_score = max(0, 40 - max_allocation)  # Penalize concentration
        
        return position_score + concentration_score
    
    def optimize_confidence_threshold(self, market_data: Dict, portfolio_data: Dict) -> float:
        """Dynamic confidence threshold optimization"""
        base_threshold = 60.0
        
        # Adjust for market volatility
        volatility = market_data.get('volatility', 'medium')
        if volatility == 'extreme':
            base_threshold += 15
        elif volatility == 'high':
            base_threshold += 8
        elif volatility == 'low':
            base_threshold -= 5
        
        # Adjust for portfolio diversification
        diversification = portfolio_data.get('diversification_score', 50)
        if diversification < 30:  # Low diversification - be more selective
            base_threshold += 5
        elif diversification > 70:  # Well diversified - can be more aggressive
            base_threshold -= 3
        
        # Adjust for USDT balance
        usdt_balance = portfolio_data.get('usdt_balance', 0)
        if usdt_balance < 50:  # Low cash - be conservative
            base_threshold += 10
        elif usdt_balance > 200:  # High cash - can be aggressive
            base_threshold -= 5
        
        return max(50.0, min(80.0, base_threshold))
    
    def optimize_position_sizing(self, market_data: Dict, portfolio_data: Dict) -> float:
        """Dynamic position sizing optimization"""
        base_size = 1.0  # 1% base position size
        
        volatility = market_data.get('volatility', 'medium')
        if volatility == 'extreme':
            base_size *= 0.5
        elif volatility == 'high':
            base_size *= 0.7
        elif volatility == 'low':
            base_size *= 1.2
        
        # Adjust for portfolio concentration
        position_count = portfolio_data.get('position_count', 0)
        if position_count > 8:  # Too many positions
            base_size *= 0.8
        elif position_count < 3:  # Few positions, can size up
            base_size *= 1.1
        
        return max(0.3, min(2.0, base_size))
    
    def check_recent_trades(self) -> Dict:
        """Analyze recent trading activity for optimization"""
        try:
            if not os.path.exists('live_trading.db'):
                return {'trade_count': 0, 'success_rate': 0}
            
            conn = sqlite3.connect('live_trading.db')
            cursor = conn.cursor()
            
            # Get trades from last 2 hours
            cursor.execute('''
                SELECT symbol, side, amount, price, status, timestamp
                FROM live_trades 
                WHERE datetime(timestamp) >= datetime('now', '-2 hours')
                ORDER BY timestamp DESC
            ''')
            
            trades = cursor.fetchall()
            conn.close()
            
            if not trades:
                return {'trade_count': 0, 'success_rate': 0}
            
            successful_trades = len([t for t in trades if t[4] == 'filled'])
            success_rate = (successful_trades / len(trades)) * 100
            
            return {
                'trade_count': len(trades),
                'success_rate': success_rate,
                'recent_trades': trades
            }
            
        except Exception as e:
            return {'trade_count': 0, 'success_rate': 0}
    
    def apply_optimizations(self, optimized_params: Dict) -> bool:
        """Apply optimized parameters to trading system"""
        try:
            # Log optimization changes
            conn = sqlite3.connect('trading_platform.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    old_confidence REAL,
                    new_confidence REAL,
                    old_position_size REAL,
                    new_position_size REAL,
                    reason TEXT
                )
            ''')
            
            cursor.execute('''
                INSERT INTO optimization_log 
                (timestamp, old_confidence, new_confidence, old_position_size, new_position_size, reason)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                self.current_params['confidence_threshold'],
                optimized_params['confidence_threshold'],
                self.current_params['position_size_pct'],
                optimized_params['position_size_pct'],
                'Dynamic optimization based on market conditions'
            ))
            
            conn.commit()
            conn.close()
            
            # Update current parameters
            self.current_params.update(optimized_params)
            
            return True
            
        except Exception as e:
            print(f"Optimization application error: {e}")
            return False
    
    def run_optimization_cycle(self):
        """Execute complete optimization cycle"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running optimization cycle...")
        
        # Gather data
        market_data = self.analyze_market_volatility()
        portfolio_data = self.get_portfolio_performance()
        trade_data = self.check_recent_trades()
        
        # Calculate optimal parameters
        optimal_confidence = self.optimize_confidence_threshold(market_data, portfolio_data)
        optimal_position_size = self.optimize_position_sizing(market_data, portfolio_data)
        
        # Check if changes are significant enough to apply
        confidence_change = abs(optimal_confidence - self.current_params['confidence_threshold'])
        size_change = abs(optimal_position_size - self.current_params['position_size_pct'])
        
        if confidence_change >= 3.0 or size_change >= 0.2:
            optimized_params = {
                'confidence_threshold': optimal_confidence,
                'position_size_pct': optimal_position_size
            }
            
            if self.apply_optimizations(optimized_params):
                print(f"✅ Applied optimizations:")
                print(f"   Confidence: {self.current_params['confidence_threshold']:.1f}%")
                print(f"   Position size: {self.current_params['position_size_pct']:.2f}%")
                print(f"   Reason: Market volatility={market_data.get('volatility', 'unknown')}")
            else:
                print("❌ Failed to apply optimizations")
        else:
            print("✅ Parameters already optimal")
        
        # Display current status
        print(f"📊 Market: {market_data.get('volatility', 'unknown')} volatility")
        print(f"💰 Portfolio: ${portfolio_data.get('total_value', 0):.2f}")
        print(f"🔄 Recent trades: {trade_data.get('trade_count', 0)}")
    
    def start_optimization_loop(self, interval_minutes: int = 10):
        """Start continuous optimization loop"""
        def optimization_worker():
            while self.optimization_active:
                try:
                    self.run_optimization_cycle()
                    time.sleep(interval_minutes * 60)
                except Exception as e:
                    print(f"Optimization loop error: {e}")
                    time.sleep(60)  # Wait 1 minute before retry
        
        optimization_thread = threading.Thread(target=optimization_worker, daemon=True)
        optimization_thread.start()
        print(f"🚀 Optimization loop started (interval: {interval_minutes} minutes)")
        
        return optimization_thread

def main():
    """Main optimization function"""
    optimizer = AdvancedTradingOptimizer()
    
    print("ADVANCED TRADING SYSTEM OPTIMIZER")
    print("=" * 50)
    
    # Run immediate optimization
    optimizer.run_optimization_cycle()
    
    print("\n" + "=" * 50)
    print("Optimization complete. System parameters updated.")

if __name__ == '__main__':
    main()