#!/usr/bin/env python3
"""
Dynamic Trading Optimization Dashboard
Real-time recommendations and actionable insights for trading system optimization
"""

import sqlite3
import ccxt
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import time
from flask import Flask, jsonify, render_template_string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicOptimizationEngine:
    def __init__(self):
        self.db_path = 'trading_platform.db'
        self.live_db_path = 'live_trading.db'
        self.exchange = None
        self.initialize_exchange()
        
    def initialize_exchange(self):
        """Initialize OKX connection"""
        try:
            api_key = os.environ.get('OKX_API_KEY')
            secret_key = os.environ.get('OKX_SECRET_KEY')
            passphrase = os.environ.get('OKX_PASSPHRASE')
            
            if api_key and secret_key and passphrase:
                self.exchange = ccxt.okx({
                    'apiKey': api_key,
                    'secret': secret_key,
                    'password': passphrase,
                    'sandbox': False,
                    'rateLimit': 800,
                    'enableRateLimit': True,
                })
        except Exception as e:
            logger.error(f"Exchange initialization failed: {e}")

    def analyze_current_market_conditions(self):
        """Analyze real-time market conditions for optimization"""
        try:
            # Get live market data
            btc_ticker = self.exchange.fetch_ticker('BTC/USDT')
            eth_ticker = self.exchange.fetch_ticker('ETH/USDT')
            
            btc_ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', '1h', limit=48)
            eth_ohlcv = self.exchange.fetch_ohlcv('ETH/USDT', '1h', limit=48)
            
            # Calculate market metrics
            btc_prices = [candle[4] for candle in btc_ohlcv]  # Close prices
            eth_prices = [candle[4] for candle in eth_ohlcv]
            
            # Volatility analysis
            btc_returns = [(btc_prices[i]/btc_prices[i-1] - 1) for i in range(1, len(btc_prices))]
            eth_returns = [(eth_prices[i]/eth_prices[i-1] - 1) for i in range(1, len(eth_prices))]
            
            btc_volatility = np.std(btc_returns[-24:]) * np.sqrt(24)  # 24h volatility
            eth_volatility = np.std(eth_returns[-24:]) * np.sqrt(24)
            
            market_volatility = (btc_volatility + eth_volatility) / 2
            
            # Trend analysis
            btc_trend = (btc_prices[-1] / btc_prices[-24] - 1)  # 24h trend
            eth_trend = (eth_prices[-1] / eth_prices[-24] - 1)
            
            market_trend_strength = abs((btc_trend + eth_trend) / 2)
            
            # Volume analysis
            btc_volume_recent = sum([candle[5] for candle in btc_ohlcv[-6:]])  # Last 6h
            btc_volume_previous = sum([candle[5] for candle in btc_ohlcv[-12:-6]])  # Previous 6h
            volume_ratio = btc_volume_recent / btc_volume_previous if btc_volume_previous > 0 else 1
            
            return {
                'volatility': market_volatility,
                'trend_strength': market_trend_strength,
                'volume_ratio': volume_ratio,
                'btc_price': btc_ticker['last'],
                'eth_price': eth_ticker['last'],
                'btc_24h_change': btc_ticker['percentage'],
                'eth_24h_change': eth_ticker['percentage']
            }
            
        except Exception as e:
            logger.error(f"Market analysis error: {e}")
            return None

    def get_portfolio_risk_metrics(self):
        """Calculate real-time portfolio risk metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT symbol, quantity, current_price
                FROM portfolio
                WHERE symbol != 'USDT' AND quantity > 0
            ''')
            
            positions = cursor.fetchall()
            conn.close()
            
            if not positions:
                return None
            
            # Calculate portfolio metrics
            total_value = sum(float(qty) * float(price) for _, qty, price in positions)
            position_weights = [(float(qty) * float(price))/total_value for _, qty, price in positions]
            
            # Concentration risk
            max_weight = max(position_weights)
            concentration_risk = "HIGH" if max_weight > 0.4 else "MEDIUM" if max_weight > 0.25 else "LOW"
            
            # Diversification score
            diversification_score = 1 - sum(w**2 for w in position_weights)
            
            return {
                'total_value': total_value,
                'position_count': len(positions),
                'max_concentration': max_weight,
                'concentration_risk': concentration_risk,
                'diversification_score': diversification_score,
                'positions': positions
            }
            
        except Exception as e:
            logger.error(f"Portfolio risk analysis error: {e}")
            return None

    def analyze_recent_trading_performance(self):
        """Analyze recent trading performance for optimization insights"""
        try:
            conn = sqlite3.connect(self.live_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT symbol, side, amount, price, pnl, ai_confidence, timestamp
                FROM live_trades
                WHERE timestamp > datetime('now', '-24 hours')
                ORDER BY timestamp DESC
            ''')
            
            trades = cursor.fetchall()
            conn.close()
            
            if not trades:
                return None
            
            # Performance metrics
            total_trades = len(trades)
            profitable_trades = [t for t in trades if t[4] and float(t[4]) > 0]
            losing_trades = [t for t in trades if t[4] and float(t[4]) < 0]
            
            win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
            
            # PnL analysis
            total_pnl = sum(float(t[4]) for t in trades if t[4])
            avg_win = np.mean([float(t[4]) for t in profitable_trades]) if profitable_trades else 0
            avg_loss = np.mean([float(t[4]) for t in losing_trades]) if losing_trades else 0
            
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            
            # Confidence analysis
            confidence_performance = {}
            for trade in trades:
                if trade[5]:  # ai_confidence
                    conf_bucket = int(float(trade[5]) / 10) * 10
                    if conf_bucket not in confidence_performance:
                        confidence_performance[conf_bucket] = []
                    if trade[4]:  # pnl
                        confidence_performance[conf_bucket].append(float(trade[4]))
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'profit_factor': profit_factor,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'confidence_performance': confidence_performance,
                'recent_trades': trades[:5]
            }
            
        except Exception as e:
            logger.error(f"Performance analysis error: {e}")
            return None

    def generate_optimization_recommendations(self):
        """Generate real-time optimization recommendations"""
        recommendations = []
        
        # Market condition analysis
        market_conditions = self.analyze_current_market_conditions()
        if market_conditions:
            volatility = market_conditions['volatility']
            trend_strength = market_conditions['trend_strength']
            volume_ratio = market_conditions['volume_ratio']
            
            # Volatility-based recommendations
            if volatility > 0.05:  # High volatility
                recommendations.append({
                    'type': 'RISK_MANAGEMENT',
                    'priority': 'CRITICAL',
                    'action': f'Reduce position sizes by 35% - Volatility at {volatility:.1%}',
                    'reasoning': 'High market volatility detected, risk reduction required',
                    'confidence': 90,
                    'timeframe': 'Immediate',
                    'impact': 'Protect capital during volatile conditions'
                })
            elif volatility < 0.015:  # Low volatility
                recommendations.append({
                    'type': 'STRATEGY_ADJUSTMENT',
                    'priority': 'HIGH',
                    'action': 'Increase position sizes by 20% - Low volatility environment',
                    'reasoning': f'Market volatility low at {volatility:.1%}, safer to increase exposure',
                    'confidence': 80,
                    'timeframe': 'Next 4 hours',
                    'impact': 'Capitalize on stable market conditions'
                })
            
            # Trend-based recommendations
            if trend_strength > 0.08:  # Strong trend
                recommendations.append({
                    'type': 'MOMENTUM_STRATEGY',
                    'priority': 'HIGH',
                    'action': 'Focus on momentum signals - Strong trend detected',
                    'reasoning': f'Market showing {trend_strength:.1%} trend strength',
                    'confidence': 85,
                    'timeframe': 'Next 6 hours',
                    'impact': 'Align with market direction'
                })
            
            # Volume-based recommendations
            if volume_ratio > 1.5:  # High volume
                recommendations.append({
                    'type': 'EXECUTION_TIMING',
                    'priority': 'MEDIUM',
                    'action': 'Execute pending signals now - High volume detected',
                    'reasoning': f'Volume {volume_ratio:.1f}x higher than previous period',
                    'confidence': 75,
                    'timeframe': 'Next 30 minutes',
                    'impact': 'Better execution in liquid market'
                })
        
        # Portfolio risk recommendations
        portfolio_metrics = self.get_portfolio_risk_metrics()
        if portfolio_metrics:
            if portfolio_metrics['concentration_risk'] == 'HIGH':
                recommendations.append({
                    'type': 'PORTFOLIO_REBALANCE',
                    'priority': 'HIGH',
                    'action': f'Reduce largest position - {portfolio_metrics["max_concentration"]:.1%} concentration',
                    'reasoning': 'Portfolio concentration exceeds safe limits',
                    'confidence': 85,
                    'timeframe': 'Within 2 hours',
                    'impact': 'Reduce concentration risk'
                })
            
            if portfolio_metrics['diversification_score'] < 0.6:
                recommendations.append({
                    'type': 'DIVERSIFICATION',
                    'priority': 'MEDIUM',
                    'action': 'Add positions in uncorrelated assets',
                    'reasoning': f'Diversification score low at {portfolio_metrics["diversification_score"]:.2f}',
                    'confidence': 70,
                    'timeframe': 'Next trading cycle',
                    'impact': 'Improve risk-adjusted returns'
                })
        
        # Performance-based recommendations
        performance_data = self.analyze_recent_trading_performance()
        if performance_data:
            win_rate = performance_data['win_rate']
            profit_factor = performance_data['profit_factor']
            
            if win_rate < 0.45:  # Low win rate
                recommendations.append({
                    'type': 'SIGNAL_QUALITY',
                    'priority': 'CRITICAL',
                    'action': f'Increase confidence threshold - Win rate only {win_rate:.1%}',
                    'reasoning': 'Recent performance below acceptable levels',
                    'confidence': 95,
                    'timeframe': 'Next signal generation',
                    'impact': 'Improve signal quality and win rate'
                })
            
            if profit_factor < 1.5:  # Poor profit factor
                recommendations.append({
                    'type': 'RISK_REWARD',
                    'priority': 'HIGH',
                    'action': 'Adjust stop losses and take profits for better R/R ratio',
                    'reasoning': f'Profit factor at {profit_factor:.2f}, below optimal',
                    'confidence': 80,
                    'timeframe': 'Apply to next trades',
                    'impact': 'Improve risk-reward profile'
                })
        
        # Market timing recommendations
        current_hour = datetime.now().hour
        if 14 <= current_hour <= 16:  # US market open overlap
            recommendations.append({
                'type': 'MARKET_TIMING',
                'priority': 'MEDIUM',
                'action': 'Optimal trading window - Increase activity',
                'reasoning': 'High liquidity during US-Asia overlap',
                'confidence': 75,
                'timeframe': 'Next 2 hours',
                'impact': 'Better execution and reduced slippage'
            })
        
        return sorted(recommendations, key=lambda x: {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[x['priority']], reverse=True)

    def get_system_health_metrics(self):
        """Get overall system health and performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Signal generation rate
            cursor.execute('''
                SELECT COUNT(*) FROM ai_signals 
                WHERE timestamp > datetime('now', '-1 hour')
            ''')
            signals_last_hour = cursor.fetchone()[0]
            
            # ML model performance
            cursor.execute('''
                SELECT AVG(accuracy) FROM ml_model_performance 
                WHERE timestamp > datetime('now', '-24 hours')
            ''')
            avg_ml_accuracy = cursor.fetchone()[0] or 0
            
            # Market regime confidence
            cursor.execute('''
                SELECT regime_type, confidence FROM market_regime_classification
                ORDER BY timestamp DESC LIMIT 1
            ''')
            regime_data = cursor.fetchone()
            
            conn.close()
            
            # Trading system uptime
            performance_data = self.analyze_recent_trading_performance()
            portfolio_data = self.get_portfolio_risk_metrics()
            
            # Calculate overall health score
            health_score = 0
            
            # Signal generation health (25%)
            signal_health = min(100, (signals_last_hour / 3) * 100)  # Target: 3 signals/hour
            health_score += signal_health * 0.25
            
            # ML accuracy health (25%)
            ml_health = avg_ml_accuracy * 100 if avg_ml_accuracy else 50
            health_score += ml_health * 0.25
            
            # Trading performance health (25%)
            if performance_data:
                perf_health = performance_data['win_rate'] * 100
                health_score += perf_health * 0.25
            else:
                health_score += 50 * 0.25  # Neutral if no recent trades
            
            # Portfolio health (25%)
            if portfolio_data:
                portfolio_health = 100 - (portfolio_data['max_concentration'] * 100)  # Lower concentration = better health
                health_score += portfolio_health * 0.25
            else:
                health_score += 75 * 0.25  # Good if no positions
            
            return {
                'overall_health': min(100, max(0, health_score)),
                'signals_per_hour': signals_last_hour,
                'ml_accuracy': avg_ml_accuracy,
                'current_regime': regime_data[0] if regime_data else 'UNKNOWN',
                'regime_confidence': regime_data[1] if regime_data else 50,
                'components': {
                    'signal_generation': signal_health,
                    'ml_performance': ml_health,
                    'trading_performance': performance_data['win_rate'] * 100 if performance_data else 50,
                    'portfolio_health': portfolio_health if portfolio_data else 75
                }
            }
            
        except Exception as e:
            logger.error(f"System health metrics error: {e}")
            return {
                'overall_health': 60,
                'signals_per_hour': 0,
                'ml_accuracy': 0,
                'current_regime': 'ANALYZING',
                'regime_confidence': 60
            }

    def run_optimization_analysis(self):
        """Run complete optimization analysis"""
        logger.info("Running Dynamic Optimization Analysis")
        
        try:
            # Get all optimization data
            recommendations = self.generate_optimization_recommendations()
            health_metrics = self.get_system_health_metrics()
            market_conditions = self.analyze_current_market_conditions()
            performance_data = self.analyze_recent_trading_performance()
            portfolio_data = self.get_portfolio_risk_metrics()
            
            optimization_report = {
                'timestamp': datetime.now().isoformat(),
                'system_health': health_metrics,
                'market_conditions': market_conditions,
                'recommendations': recommendations[:8],  # Top 8
                'performance_summary': performance_data,
                'portfolio_status': portfolio_data,
                'next_analysis': (datetime.now() + timedelta(minutes=10)).isoformat()
            }
            
            logger.info(f"Generated {len(recommendations)} optimization recommendations")
            logger.info(f"System health: {health_metrics['overall_health']:.1f}%")
            
            return optimization_report
            
        except Exception as e:
            logger.error(f"Optimization analysis error: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'system_health': {'overall_health': 50},
                'recommendations': [{
                    'type': 'SYSTEM_CHECK',
                    'priority': 'HIGH',
                    'action': 'System analysis in progress',
                    'confidence': 60
                }]
            }

def main():
    """Main optimization function"""
    optimizer = DynamicOptimizationEngine()
    result = optimizer.run_optimization_analysis()
    
    print("\n=== DYNAMIC OPTIMIZATION REPORT ===")
    print(f"System Health: {result['system_health']['overall_health']:.1f}%")
    print(f"Active Recommendations: {len(result.get('recommendations', []))}")
    
    if 'recommendations' in result:
        print("\nTOP RECOMMENDATIONS:")
        for i, rec in enumerate(result['recommendations'][:5], 1):
            print(f"{i}. [{rec['priority']}] {rec['action']}")
    
    return result

if __name__ == "__main__":
    main()