#!/usr/bin/env python3
"""
Comprehensive Trading System Performance Audit
Analyzes all aspects of the AI trading system performance and provides actionable recommendations
"""

import sqlite3
import json
import os
import ccxt
import time
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingSystemAuditor:
    def __init__(self):
        self.audit_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': 0,
            'critical_issues': [],
            'recommendations': [],
            'performance_metrics': {},
            'system_health': {},
            'data_quality': {},
            'trading_effectiveness': {},
            'risk_assessment': {}
        }
        
        # Initialize OKX connection for authentic data
        self.exchange = None
        self.initialize_exchange()
        
    def initialize_exchange(self):
        """Initialize OKX exchange connection"""
        try:
            api_key = os.getenv('OKX_API_KEY')
            secret = os.getenv('OKX_SECRET_KEY')
            passphrase = os.getenv('OKX_PASSPHRASE')
            
            if api_key and secret and passphrase:
                self.exchange = ccxt.okx({
                    'apiKey': api_key,
                    'secret': secret,
                    'password': passphrase,
                    'sandbox': False,
                    'enableRateLimit': True,
                })
                logger.info("✓ OKX exchange connection established")
            else:
                logger.warning("⚠ OKX credentials not configured - using limited analysis")
        except Exception as e:
            logger.error(f"Exchange connection failed: {e}")
    
    def audit_data_authenticity(self) -> Dict:
        """Audit data sources for authenticity and reliability"""
        logger.info("Auditing data authenticity...")
        
        data_audit = {
            'score': 0,
            'issues': [],
            'sources_verified': [],
            'mock_data_detected': []
        }
        
        try:
            # Test OKX API connectivity
            if self.exchange:
                try:
                    balance = self.exchange.fetch_balance()
                    data_audit['sources_verified'].append('OKX Balance API')
                    
                    # Test market data
                    ticker = self.exchange.fetch_ticker('BTC/USDT')
                    data_audit['sources_verified'].append('OKX Market Data API')
                    
                    # Test trading history
                    trades = self.exchange.fetch_my_trades('BTC/USDT', limit=5)
                    data_audit['sources_verified'].append('OKX Trading History API')
                    
                    data_audit['score'] += 30
                    
                except Exception as e:
                    data_audit['issues'].append(f"OKX API connection issues: {e}")
                    data_audit['score'] -= 20
            else:
                data_audit['issues'].append("No OKX API credentials configured")
                data_audit['score'] -= 30
            
            # Check database authenticity
            try:
                with sqlite3.connect('enhanced_trading.db') as conn:
                    cursor = conn.cursor()
                    
                    # Check for real vs mock data patterns
                    cursor.execute("SELECT COUNT(*) FROM unified_signals WHERE timestamp > datetime('now', '-1 hour')")
                    recent_signals = cursor.fetchone()[0]
                    
                    if recent_signals > 0:
                        data_audit['sources_verified'].append('Recent AI Signals Database')
                        data_audit['score'] += 20
                    else:
                        data_audit['issues'].append("No recent signals in database")
                    
                    # Check portfolio data
                    cursor.execute("SELECT COUNT(*) FROM portfolio_data WHERE timestamp > datetime('now', '-1 hour')")
                    recent_portfolio = cursor.fetchone()[0]
                    
                    if recent_portfolio > 0:
                        data_audit['sources_verified'].append('Recent Portfolio Data')
                        data_audit['score'] += 15
                    
            except Exception as e:
                data_audit['issues'].append(f"Database access error: {e}")
                data_audit['score'] -= 15
        
        except Exception as e:
            data_audit['issues'].append(f"Data audit error: {e}")
        
        data_audit['score'] = max(0, min(100, data_audit['score']))
        return data_audit
    
    def audit_signal_performance(self) -> Dict:
        """Audit AI signal generation performance"""
        logger.info("Auditing signal performance...")
        
        signal_audit = {
            'score': 0,
            'total_signals': 0,
            'high_confidence_signals': 0,
            'signal_accuracy': 0,
            'avg_confidence': 0,
            'issues': [],
            'strengths': []
        }
        
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                # Analyze recent signals
                cursor.execute("""
                    SELECT confidence, action, symbol, timestamp 
                    FROM unified_signals 
                    WHERE timestamp > datetime('now', '-24 hours')
                    ORDER BY timestamp DESC
                """)
                
                signals = cursor.fetchall()
                signal_audit['total_signals'] = len(signals)
                
                if signals:
                    confidences = [float(s[0]) for s in signals]
                    signal_audit['avg_confidence'] = sum(confidences) / len(confidences)
                    signal_audit['high_confidence_signals'] = len([c for c in confidences if c >= 75])
                    
                    # Score based on signal quality
                    if signal_audit['avg_confidence'] >= 70:
                        signal_audit['score'] += 25
                        signal_audit['strengths'].append(f"High average confidence: {signal_audit['avg_confidence']:.1f}%")
                    elif signal_audit['avg_confidence'] >= 60:
                        signal_audit['score'] += 15
                    else:
                        signal_audit['issues'].append(f"Low average confidence: {signal_audit['avg_confidence']:.1f}%")
                    
                    # Signal frequency check
                    if signal_audit['total_signals'] >= 10:
                        signal_audit['score'] += 20
                        signal_audit['strengths'].append(f"Good signal frequency: {signal_audit['total_signals']} signals/day")
                    elif signal_audit['total_signals'] >= 5:
                        signal_audit['score'] += 10
                    else:
                        signal_audit['issues'].append(f"Low signal frequency: {signal_audit['total_signals']} signals/day")
                    
                    # High confidence signal ratio
                    hc_ratio = signal_audit['high_confidence_signals'] / signal_audit['total_signals']
                    if hc_ratio >= 0.3:
                        signal_audit['score'] += 15
                        signal_audit['strengths'].append(f"Good high-confidence ratio: {hc_ratio:.1%}")
                    else:
                        signal_audit['issues'].append(f"Low high-confidence ratio: {hc_ratio:.1%}")
                        
                else:
                    signal_audit['issues'].append("No signals generated in last 24 hours")
                    signal_audit['score'] -= 30
        
        except Exception as e:
            signal_audit['issues'].append(f"Signal analysis error: {e}")
        
        signal_audit['score'] = max(0, min(100, signal_audit['score']))
        return signal_audit
    
    def audit_trading_performance(self) -> Dict:
        """Audit actual trading performance using OKX data"""
        logger.info("Auditing trading performance...")
        
        trading_audit = {
            'score': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'total_trades': 0,
            'total_pnl': 0,
            'sharpe_ratio': 0,
            'issues': [],
            'strengths': []
        }
        
        try:
            if self.exchange:
                # Get authentic trading performance
                symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
                all_trades = []
                
                for symbol in symbols:
                    try:
                        trades = self.exchange.fetch_my_trades(symbol, limit=100)
                        all_trades.extend(trades)
                    except Exception as e:
                        trading_audit['issues'].append(f"Failed to fetch {symbol} trades: {e}")
                
                if all_trades:
                    # Calculate performance metrics
                    symbol_positions = {}
                    
                    for trade in all_trades:
                        symbol = trade['symbol']
                        if symbol not in symbol_positions:
                            symbol_positions[symbol] = {'buys': [], 'sells': [], 'total_fees': 0}
                        
                        fee = trade.get('fee', {}).get('cost', 0) if trade.get('fee') else 0
                        symbol_positions[symbol]['total_fees'] += fee
                        
                        if trade['side'] == 'buy':
                            symbol_positions[symbol]['buys'].append({
                                'amount': trade['amount'], 
                                'price': trade['price']
                            })
                        else:
                            symbol_positions[symbol]['sells'].append({
                                'amount': trade['amount'], 
                                'price': trade['price']
                            })
                    
                    # Calculate win rate and P&L
                    winning_trades = 0
                    total_profit = 0
                    total_loss = 0
                    completed_trades = 0
                    
                    for symbol, positions in symbol_positions.items():
                        if positions['buys'] and positions['sells']:
                            avg_buy = sum(b['price'] * b['amount'] for b in positions['buys']) / sum(b['amount'] for b in positions['buys'])
                            avg_sell = sum(s['price'] * s['amount'] for s in positions['sells']) / sum(s['amount'] for s in positions['sells'])
                            
                            min_amount = min(
                                sum(b['amount'] for b in positions['buys']),
                                sum(s['amount'] for s in positions['sells'])
                            )
                            
                            if min_amount > 0:
                                pnl = (avg_sell - avg_buy) * min_amount - positions['total_fees']
                                completed_trades += 1
                                
                                if pnl > 0:
                                    winning_trades += 1
                                    total_profit += pnl
                                else:
                                    total_loss += abs(pnl)
                    
                    if completed_trades > 0:
                        trading_audit['total_trades'] = completed_trades
                        trading_audit['win_rate'] = (winning_trades / completed_trades) * 100
                        trading_audit['profit_factor'] = total_profit / total_loss if total_loss > 0 else 1.0
                        trading_audit['total_pnl'] = total_profit - total_loss
                        
                        # Score based on performance
                        if trading_audit['win_rate'] >= 60:
                            trading_audit['score'] += 30
                            trading_audit['strengths'].append(f"Excellent win rate: {trading_audit['win_rate']:.1f}%")
                        elif trading_audit['win_rate'] >= 45:
                            trading_audit['score'] += 20
                            trading_audit['strengths'].append(f"Good win rate: {trading_audit['win_rate']:.1f}%")
                        else:
                            trading_audit['issues'].append(f"Low win rate: {trading_audit['win_rate']:.1f}%")
                        
                        if trading_audit['profit_factor'] >= 2.0:
                            trading_audit['score'] += 25
                            trading_audit['strengths'].append(f"Excellent profit factor: {trading_audit['profit_factor']:.2f}")
                        elif trading_audit['profit_factor'] >= 1.5:
                            trading_audit['score'] += 15
                        else:
                            trading_audit['issues'].append(f"Low profit factor: {trading_audit['profit_factor']:.2f}")
                        
                        if trading_audit['total_pnl'] > 0:
                            trading_audit['score'] += 20
                            trading_audit['strengths'].append(f"Profitable: ${trading_audit['total_pnl']:.2f}")
                        else:
                            trading_audit['issues'].append(f"Unprofitable: ${trading_audit['total_pnl']:.2f}")
                    else:
                        trading_audit['issues'].append("No completed round-trip trades found")
                else:
                    trading_audit['issues'].append("No trading history available")
            else:
                trading_audit['issues'].append("OKX connection required for trading performance audit")
        
        except Exception as e:
            trading_audit['issues'].append(f"Trading performance audit error: {e}")
        
        trading_audit['score'] = max(0, min(100, trading_audit['score']))
        return trading_audit
    
    def audit_risk_management(self) -> Dict:
        """Audit risk management effectiveness"""
        logger.info("Auditing risk management...")
        
        risk_audit = {
            'score': 0,
            'max_drawdown': 0,
            'position_sizing': 'Unknown',
            'diversification': 0,
            'stop_loss_usage': 0,
            'issues': [],
            'strengths': []
        }
        
        try:
            # Check portfolio diversification
            if self.exchange:
                try:
                    balance = self.exchange.fetch_balance()
                    non_zero_balances = {k: v for k, v in balance['total'].items() if v > 0 and k != 'USDT'}
                    
                    risk_audit['diversification'] = len(non_zero_balances)
                    
                    if risk_audit['diversification'] >= 5:
                        risk_audit['score'] += 25
                        risk_audit['strengths'].append(f"Good diversification: {risk_audit['diversification']} assets")
                    elif risk_audit['diversification'] >= 3:
                        risk_audit['score'] += 15
                    else:
                        risk_audit['issues'].append(f"Low diversification: {risk_audit['diversification']} assets")
                    
                    # Check position concentration
                    if non_zero_balances:
                        total_value = sum(balance['total'].values())
                        largest_position = max(balance['total'].values()) / total_value if total_value > 0 else 0
                        
                        if largest_position < 0.4:
                            risk_audit['score'] += 20
                            risk_audit['strengths'].append(f"Good position sizing: Max {largest_position:.1%}")
                        else:
                            risk_audit['issues'].append(f"High concentration risk: {largest_position:.1%}")
                            
                except Exception as e:
                    risk_audit['issues'].append(f"Portfolio analysis error: {e}")
            
            # Check database for risk management features
            try:
                with sqlite3.connect('enhanced_trading.db') as conn:
                    cursor = conn.cursor()
                    
                    # Check for stop loss implementation
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='stop_losses'")
                    if cursor.fetchone():
                        cursor.execute("SELECT COUNT(*) FROM stop_losses WHERE active = 1")
                        active_stops = cursor.fetchone()[0]
                        
                        if active_stops > 0:
                            risk_audit['score'] += 15
                            risk_audit['strengths'].append(f"Active stop losses: {active_stops}")
                        else:
                            risk_audit['issues'].append("No active stop losses detected")
                    else:
                        risk_audit['issues'].append("Stop loss system not implemented")
                    
            except Exception as e:
                risk_audit['issues'].append(f"Risk database check error: {e}")
        
        except Exception as e:
            risk_audit['issues'].append(f"Risk management audit error: {e}")
        
        risk_audit['score'] = max(0, min(100, risk_audit['score']))
        return risk_audit
    
    def audit_system_health(self) -> Dict:
        """Audit overall system health and infrastructure"""
        logger.info("Auditing system health...")
        
        health_audit = {
            'score': 0,
            'uptime': 0,
            'error_rate': 0,
            'response_time': 0,
            'database_health': 0,
            'issues': [],
            'strengths': []
        }
        
        try:
            # Test API response times
            start_time = time.time()
            try:
                import requests
                response = requests.get('http://localhost:5000/api/unified/health', timeout=5)
                response_time = time.time() - start_time
                
                if response.status_code == 200:
                    health_audit['score'] += 20
                    health_audit['strengths'].append("API endpoints responding")
                    
                    if response_time < 1.0:
                        health_audit['score'] += 15
                        health_audit['strengths'].append(f"Fast response time: {response_time:.2f}s")
                    else:
                        health_audit['issues'].append(f"Slow response time: {response_time:.2f}s")
                else:
                    health_audit['issues'].append(f"API error: {response.status_code}")
            except Exception as e:
                health_audit['issues'].append(f"API connectivity issue: {e}")
            
            # Check database health
            try:
                with sqlite3.connect('enhanced_trading.db') as conn:
                    cursor = conn.cursor()
                    
                    # Check table integrity
                    tables = ['unified_signals', 'portfolio_data', 'trading_performance']
                    healthy_tables = 0
                    
                    for table in tables:
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {table}")
                            count = cursor.fetchone()[0]
                            if count >= 0:
                                healthy_tables += 1
                        except:
                            health_audit['issues'].append(f"Table {table} corrupted or missing")
                    
                    health_audit['database_health'] = (healthy_tables / len(tables)) * 100
                    
                    if health_audit['database_health'] == 100:
                        health_audit['score'] += 25
                        health_audit['strengths'].append("All database tables healthy")
                    elif health_audit['database_health'] >= 80:
                        health_audit['score'] += 15
                    else:
                        health_audit['issues'].append(f"Database health: {health_audit['database_health']:.0f}%")
                        
            except Exception as e:
                health_audit['issues'].append(f"Database health check failed: {e}")
        
        except Exception as e:
            health_audit['issues'].append(f"System health audit error: {e}")
        
        health_audit['score'] = max(0, min(100, health_audit['score']))
        return health_audit
    
    def generate_recommendations(self) -> List[Dict]:
        """Generate actionable recommendations based on audit results"""
        recommendations = []
        
        # Critical recommendations based on audit scores
        if self.audit_results['data_quality']['score'] < 70:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'Data Quality',
                'title': 'Improve Data Source Reliability',
                'description': 'Configure proper OKX API credentials and fix data connectivity issues',
                'impact': 'HIGH',
                'effort': 'MEDIUM'
            })
        
        if self.audit_results['trading_effectiveness']['score'] < 60:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Trading Performance',
                'title': 'Optimize Trading Strategy',
                'description': 'Review signal thresholds and implement better entry/exit rules',
                'impact': 'HIGH',
                'effort': 'HIGH'
            })
        
        if self.audit_results['risk_assessment']['score'] < 50:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'Risk Management',
                'title': 'Implement Comprehensive Risk Controls',
                'description': 'Add stop losses, position sizing limits, and portfolio diversification rules',
                'impact': 'CRITICAL',
                'effort': 'MEDIUM'
            })
        
        # Performance optimization recommendations
        if self.audit_results['performance_metrics'].get('avg_confidence', 0) < 70:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'Signal Quality',
                'title': 'Improve Signal Confidence',
                'description': 'Retrain ML models with more recent data and optimize feature selection',
                'impact': 'MEDIUM',
                'effort': 'HIGH'
            })
        
        if self.audit_results['system_health']['score'] < 80:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'System Infrastructure',
                'title': 'Optimize System Performance',
                'description': 'Improve API response times and database query optimization',
                'impact': 'MEDIUM',
                'effort': 'MEDIUM'
            })
        
        # Always include these best practice recommendations
        recommendations.extend([
            {
                'priority': 'LOW',
                'category': 'Monitoring',
                'title': 'Enhanced Performance Tracking',
                'description': 'Implement real-time performance dashboards and automated alerts',
                'impact': 'MEDIUM',
                'effort': 'LOW'
            },
            {
                'priority': 'LOW',
                'category': 'Compliance',
                'title': 'Add Trading Journal',
                'description': 'Implement detailed trade logging for performance analysis and compliance',
                'impact': 'LOW',
                'effort': 'LOW'
            }
        ])
        
        return recommendations
    
    def run_comprehensive_audit(self) -> Dict:
        """Run complete system audit"""
        logger.info("Starting comprehensive trading system audit...")
        
        # Run all audit components
        self.audit_results['data_quality'] = self.audit_data_authenticity()
        self.audit_results['performance_metrics'] = self.audit_signal_performance()
        self.audit_results['trading_effectiveness'] = self.audit_trading_performance()
        self.audit_results['risk_assessment'] = self.audit_risk_management()
        self.audit_results['system_health'] = self.audit_system_health()
        
        # Calculate overall score
        scores = [
            self.audit_results['data_quality']['score'],
            self.audit_results['performance_metrics']['score'],
            self.audit_results['trading_effectiveness']['score'],
            self.audit_results['risk_assessment']['score'],
            self.audit_results['system_health']['score']
        ]
        
        self.audit_results['overall_score'] = sum(scores) / len(scores)
        
        # Generate recommendations
        self.audit_results['recommendations'] = self.generate_recommendations()
        
        # Identify critical issues
        for category, results in self.audit_results.items():
            if isinstance(results, dict) and 'issues' in results:
                for issue in results['issues']:
                    if results['score'] < 50:
                        self.audit_results['critical_issues'].append(f"{category}: {issue}")
        
        return self.audit_results
    
    def save_audit_report(self):
        """Save audit results to file"""
        report_file = f"trading_system_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.audit_results, f, indent=2, default=str)
        
        logger.info(f"Audit report saved to {report_file}")
        return report_file
    
    def print_audit_summary(self):
        """Print executive summary of audit results"""
        print("\n" + "="*60)
        print("COMPREHENSIVE TRADING SYSTEM AUDIT REPORT")
        print("="*60)
        print(f"Audit Date: {self.audit_results['timestamp']}")
        print(f"Overall System Score: {self.audit_results['overall_score']:.1f}/100")
        
        if self.audit_results['overall_score'] >= 80:
            status = "EXCELLENT"
        elif self.audit_results['overall_score'] >= 70:
            status = "GOOD"
        elif self.audit_results['overall_score'] >= 60:
            status = "FAIR"
        else:
            status = "NEEDS IMPROVEMENT"
        
        print(f"System Status: {status}")
        print("\n" + "-"*60)
        print("COMPONENT SCORES:")
        print("-"*60)
        
        components = [
            ('Data Quality', self.audit_results['data_quality']['score']),
            ('Signal Performance', self.audit_results['performance_metrics']['score']),
            ('Trading Effectiveness', self.audit_results['trading_effectiveness']['score']),
            ('Risk Management', self.audit_results['risk_assessment']['score']),
            ('System Health', self.audit_results['system_health']['score'])
        ]
        
        for name, score in components:
            print(f"{name:<25} {score:>6.1f}/100")
        
        print("\n" + "-"*60)
        print("CRITICAL ISSUES:")
        print("-"*60)
        
        if self.audit_results['critical_issues']:
            for i, issue in enumerate(self.audit_results['critical_issues'][:5], 1):
                print(f"{i}. {issue}")
        else:
            print("No critical issues detected")
        
        print("\n" + "-"*60)
        print("TOP RECOMMENDATIONS:")
        print("-"*60)
        
        priority_order = {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}
        sorted_recs = sorted(self.audit_results['recommendations'], 
                           key=lambda x: priority_order.get(x['priority'], 4))
        
        for i, rec in enumerate(sorted_recs[:5], 1):
            print(f"{i}. [{rec['priority']}] {rec['title']}")
            print(f"   {rec['description']}")
            print(f"   Impact: {rec['impact']} | Effort: {rec['effort']}")
            print()

def main():
    """Run comprehensive audit"""
    auditor = TradingSystemAuditor()
    
    # Run the audit
    results = auditor.run_comprehensive_audit()
    
    # Print summary
    auditor.print_audit_summary()
    
    # Save detailed report
    report_file = auditor.save_audit_report()
    
    print(f"\nDetailed audit report saved to: {report_file}")
    print("\nAudit completed successfully!")
    
    return results

if __name__ == "__main__":
    main()