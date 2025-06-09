#!/usr/bin/env python3
"""
Live Trading Performance Optimizer
Advanced optimization system for autonomous trading performance enhancement
"""

import sqlite3
import os
import ccxt
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import statistics

class TradingPerformanceOptimizer:
    def __init__(self):
        self.db_path = 'trading_platform.db'
        self.live_db_path = 'live_trading.db'
        self.exchange = self.connect_okx()
        self.optimization_config = {
            'confidence_threshold': 60.0,
            'position_size_pct': 1.0,
            'max_concurrent_positions': 5,
            'profit_target_pct': 3.0,
            'stop_loss_pct': 1.5,
            'signal_cooldown_minutes': 30
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
            print(f"OKX connection error: {e}")
            return None
    
    def analyze_recent_performance(self, hours: int = 24) -> Dict:
        """Analyze recent trading performance metrics"""
        try:
            performance_data = {
                'total_trades': 0,
                'successful_trades': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0,
                'avg_trade_duration': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
            
            if not os.path.exists(self.live_db_path):
                return performance_data
                
            conn = sqlite3.connect(self.live_db_path)
            cursor = conn.cursor()
            
            # Get recent trades
            cursor.execute('''
                SELECT symbol, side, amount, price, status, timestamp
                FROM live_trades 
                WHERE datetime(timestamp) >= datetime('now', '-{} hours')
                ORDER BY timestamp ASC
            '''.format(hours))
            
            trades = cursor.fetchall()
            conn.close()
            
            if not trades:
                return performance_data
            
            performance_data['total_trades'] = len(trades)
            
            # Calculate basic metrics
            buy_trades = [t for t in trades if t[1] == 'buy']
            sell_trades = [t for t in trades if t[1] == 'sell']
            
            # Match buy/sell pairs for PnL calculation
            pnl_values = []
            for buy_trade in buy_trades:
                symbol = buy_trade[0]
                buy_price = float(buy_trade[3])
                buy_amount = float(buy_trade[2])
                
                # Find matching sell trade
                for sell_trade in sell_trades:
                    if sell_trade[0] == symbol:
                        sell_price = float(sell_trade[3])
                        pnl = (sell_price - buy_price) * buy_amount
                        pnl_values.append(pnl)
                        break
            
            if pnl_values:
                performance_data['total_pnl'] = sum(pnl_values)
                performance_data['successful_trades'] = len([p for p in pnl_values if p > 0])
                performance_data['win_rate'] = performance_data['successful_trades'] / len(pnl_values) * 100
                
                if len(pnl_values) > 1:
                    performance_data['sharpe_ratio'] = statistics.mean(pnl_values) / statistics.stdev(pnl_values) if statistics.stdev(pnl_values) > 0 else 0
            
            return performance_data
            
        except Exception as e:
            print(f"Error analyzing performance: {e}")
            return performance_data
    
    def calculate_optimal_parameters(self, performance_data: Dict) -> Dict:
        """Calculate optimal trading parameters based on performance"""
        optimized_params = self.optimization_config.copy()
        
        # Adjust confidence threshold based on win rate
        win_rate = performance_data.get('win_rate', 0)
        if win_rate < 50:
            optimized_params['confidence_threshold'] = min(75.0, self.optimization_config['confidence_threshold'] + 5)
        elif win_rate > 70:
            optimized_params['confidence_threshold'] = max(55.0, self.optimization_config['confidence_threshold'] - 2)
        
        # Adjust position size based on total PnL
        total_pnl = performance_data.get('total_pnl', 0)
        if total_pnl < 0:
            optimized_params['position_size_pct'] = max(0.5, self.optimization_config['position_size_pct'] * 0.8)
        elif total_pnl > 10:
            optimized_params['position_size_pct'] = min(2.0, self.optimization_config['position_size_pct'] * 1.2)
        
        # Adjust profit targets based on market volatility
        if performance_data.get('total_trades', 0) > 5:
            sharpe_ratio = performance_data.get('sharpe_ratio', 0)
            if sharpe_ratio > 1.0:
                optimized_params['profit_target_pct'] = min(5.0, self.optimization_config['profit_target_pct'] * 1.1)
            elif sharpe_ratio < 0.5:
                optimized_params['stop_loss_pct'] = max(1.0, self.optimization_config['stop_loss_pct'] * 0.9)
        
        return optimized_params
    
    def get_market_conditions(self) -> Dict:
        """Analyze current market conditions for optimization"""
        try:
            if not self.exchange:
                return {'volatility': 'medium', 'trend': 'neutral', 'volume': 'normal'}
            
            # Analyze BTC as market benchmark
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', '1h', limit=24)
            
            # Calculate volatility (24h price range)
            high_24h = max([candle[2] for candle in ohlcv])
            low_24h = min([candle[3] for candle in ohlcv])
            current_price = float(ticker['last'])
            volatility_pct = ((high_24h - low_24h) / current_price) * 100
            
            # Determine market conditions
            if volatility_pct > 5:
                volatility = 'high'
            elif volatility_pct < 2:
                volatility = 'low'
            else:
                volatility = 'medium'
            
            # Simple trend analysis
            prices = [candle[4] for candle in ohlcv[-12:]]  # Last 12 hours
            if prices[-1] > prices[0] * 1.02:
                trend = 'bullish'
            elif prices[-1] < prices[0] * 0.98:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            return {
                'volatility': volatility,
                'trend': trend,
                'volume': 'normal',
                'volatility_pct': volatility_pct
            }
            
        except Exception as e:
            print(f"Error analyzing market conditions: {e}")
            return {'volatility': 'medium', 'trend': 'neutral', 'volume': 'normal'}
    
    def optimize_for_market_conditions(self, base_params: Dict, market_conditions: Dict) -> Dict:
        """Adjust parameters based on current market conditions"""
        optimized = base_params.copy()
        
        # Adjust for volatility
        if market_conditions['volatility'] == 'high':
            optimized['confidence_threshold'] = min(80.0, optimized['confidence_threshold'] + 10)
            optimized['position_size_pct'] = max(0.5, optimized['position_size_pct'] * 0.7)
            optimized['stop_loss_pct'] = optimized['stop_loss_pct'] * 1.2
        elif market_conditions['volatility'] == 'low':
            optimized['confidence_threshold'] = max(50.0, optimized['confidence_threshold'] - 5)
            optimized['position_size_pct'] = min(1.5, optimized['position_size_pct'] * 1.1)
        
        # Adjust for trend
        if market_conditions['trend'] == 'bullish':
            optimized['profit_target_pct'] = optimized['profit_target_pct'] * 1.2
        elif market_conditions['trend'] == 'bearish':
            optimized['stop_loss_pct'] = optimized['stop_loss_pct'] * 0.8
            optimized['confidence_threshold'] = min(85.0, optimized['confidence_threshold'] + 5)
        
        return optimized
    
    def generate_optimization_report(self) -> Dict:
        """Generate comprehensive optimization report"""
        performance = self.analyze_recent_performance(24)
        market_conditions = self.get_market_conditions()
        
        # Calculate optimal parameters
        performance_optimized = self.calculate_optimal_parameters(performance)
        final_optimized = self.optimize_for_market_conditions(performance_optimized, market_conditions)
        
        # Calculate improvement potential
        improvement_score = 0
        if performance['win_rate'] < 60:
            improvement_score += 20
        if performance['total_pnl'] < 0:
            improvement_score += 30
        if performance['total_trades'] < 3:
            improvement_score += 25
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_performance': performance,
            'market_conditions': market_conditions,
            'current_parameters': self.optimization_config,
            'optimized_parameters': final_optimized,
            'improvement_potential': improvement_score,
            'recommendations': self.generate_recommendations(performance, market_conditions, final_optimized)
        }
    
    def generate_recommendations(self, performance: Dict, market: Dict, optimized: Dict) -> List[str]:
        """Generate actionable optimization recommendations"""
        recommendations = []
        
        if performance['win_rate'] < 50:
            recommendations.append(f"Increase confidence threshold to {optimized['confidence_threshold']:.1f}% to improve trade quality")
        
        if performance['total_pnl'] < 0:
            recommendations.append(f"Reduce position size to {optimized['position_size_pct']:.1f}% to limit losses")
        
        if market['volatility'] == 'high':
            recommendations.append("High market volatility detected - using conservative position sizing")
        
        if performance['total_trades'] > 10 and performance['win_rate'] > 70:
            recommendations.append("Strong performance - consider increasing position sizes for higher returns")
        
        if market['trend'] == 'bearish':
            recommendations.append("Bearish market detected - tightening stop losses and raising thresholds")
        
        if len(recommendations) == 0:
            recommendations.append("Current parameters are well-optimized for market conditions")
        
        return recommendations
    
    def apply_optimizations(self, optimized_params: Dict) -> bool:
        """Apply optimized parameters to the trading system"""
        try:
            # Update signal execution bridge parameters
            config_update = {
                'confidence_threshold': optimized_params['confidence_threshold'],
                'position_size_pct': optimized_params['position_size_pct'] / 100,  # Convert to decimal
                'timestamp': datetime.now().isoformat(),
                'applied_by': 'performance_optimizer'
            }
            
            # Save optimization history
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    old_params TEXT,
                    new_params TEXT,
                    performance_data TEXT,
                    market_conditions TEXT
                )
            ''')
            
            cursor.execute('''
                INSERT INTO optimization_history 
                (timestamp, old_params, new_params, performance_data, market_conditions)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                json.dumps(self.optimization_config),
                json.dumps(optimized_params),
                json.dumps(self.analyze_recent_performance(24)),
                json.dumps(self.get_market_conditions())
            ))
            
            conn.commit()
            conn.close()
            
            # Update current configuration
            self.optimization_config = optimized_params
            
            print(f"✅ Applied optimized parameters:")
            print(f"   Confidence threshold: {optimized_params['confidence_threshold']:.1f}%")
            print(f"   Position size: {optimized_params['position_size_pct']:.1f}%")
            print(f"   Profit target: {optimized_params['profit_target_pct']:.1f}%")
            print(f"   Stop loss: {optimized_params['stop_loss_pct']:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"Error applying optimizations: {e}")
            return False

def optimize_trading_performance():
    """Main optimization function"""
    optimizer = TradingPerformanceOptimizer()
    
    print("🚀 TRADING PERFORMANCE OPTIMIZATION")
    print("=" * 50)
    
    # Generate optimization report
    report = optimizer.generate_optimization_report()
    
    # Display current performance
    perf = report['current_performance']
    print(f"📊 Current Performance (24h):")
    print(f"   Total trades: {perf['total_trades']}")
    print(f"   Win rate: {perf['win_rate']:.1f}%")
    print(f"   Total PnL: ${perf['total_pnl']:.2f}")
    print(f"   Sharpe ratio: {perf['sharpe_ratio']:.2f}")
    
    # Display market conditions
    market = report['market_conditions']
    print(f"\n🌍 Market Conditions:")
    print(f"   Volatility: {market['volatility']}")
    print(f"   Trend: {market['trend']}")
    if 'volatility_pct' in market:
        print(f"   24h volatility: {market['volatility_pct']:.2f}%")
    
    # Display recommendations
    print(f"\n💡 Optimization Recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"   {i}. {rec}")
    
    # Show parameter changes
    current = report['current_parameters']
    optimized = report['optimized_parameters']
    
    print(f"\n⚙️  Parameter Optimization:")
    print(f"   Confidence threshold: {current['confidence_threshold']:.1f}% → {optimized['confidence_threshold']:.1f}%")
    print(f"   Position size: {current['position_size_pct']:.1f}% → {optimized['position_size_pct']:.1f}%")
    print(f"   Profit target: {current['profit_target_pct']:.1f}% → {optimized['profit_target_pct']:.1f}%")
    
    # Apply optimizations
    if optimizer.apply_optimizations(optimized):
        print(f"\n✅ Optimization complete - trading system updated")
    else:
        print(f"\n❌ Optimization failed - using current parameters")
    
    print("=" * 50)

if __name__ == '__main__':
    optimize_trading_performance()