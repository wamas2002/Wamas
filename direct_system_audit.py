#!/usr/bin/env python3
"""
Direct System Performance Audit - Database Analysis
Comprehensive performance analysis using direct database access
"""

import sqlite3
import json
import os
import ccxt
from datetime import datetime, timedelta
import statistics

def analyze_trading_system():
    """Comprehensive system analysis"""
    
    print("="*70)
    print("COMPREHENSIVE TRADING SYSTEM PERFORMANCE AUDIT")
    print("="*70)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    audit_results = {
        'overall_score': 0,
        'component_scores': {},
        'critical_findings': [],
        'recommendations': [],
        'performance_metrics': {}
    }
    
    # 1. SIGNAL GENERATION ANALYSIS
    print("\n1. AI SIGNAL GENERATION PERFORMANCE")
    print("-" * 50)
    
    signal_metrics = analyze_signal_performance()
    audit_results['component_scores']['signal_generation'] = signal_metrics['score']
    audit_results['performance_metrics']['signals'] = signal_metrics
    
    for finding in signal_metrics.get('findings', []):
        print(f"   {finding}")
    
    # 2. TRADING PERFORMANCE ANALYSIS
    print("\n2. TRADING EXECUTION PERFORMANCE")
    print("-" * 50)
    
    trading_metrics = analyze_trading_performance()
    audit_results['component_scores']['trading_execution'] = trading_metrics['score']
    audit_results['performance_metrics']['trading'] = trading_metrics
    
    for finding in trading_metrics.get('findings', []):
        print(f"   {finding}")
    
    # 3. DATA QUALITY ANALYSIS
    print("\n3. DATA QUALITY & AUTHENTICITY")
    print("-" * 50)
    
    data_metrics = analyze_data_quality()
    audit_results['component_scores']['data_quality'] = data_metrics['score']
    audit_results['performance_metrics']['data'] = data_metrics
    
    for finding in data_metrics.get('findings', []):
        print(f"   {finding}")
    
    # 4. SYSTEM HEALTH ANALYSIS
    print("\n4. SYSTEM INFRASTRUCTURE HEALTH")
    print("-" * 50)
    
    system_metrics = analyze_system_health()
    audit_results['component_scores']['system_health'] = system_metrics['score']
    audit_results['performance_metrics']['system'] = system_metrics
    
    for finding in system_metrics.get('findings', []):
        print(f"   {finding}")
    
    # 5. RISK MANAGEMENT ANALYSIS
    print("\n5. RISK MANAGEMENT ASSESSMENT")
    print("-" * 50)
    
    risk_metrics = analyze_risk_management()
    audit_results['component_scores']['risk_management'] = risk_metrics['score']
    audit_results['performance_metrics']['risk'] = risk_metrics
    
    for finding in risk_metrics.get('findings', []):
        print(f"   {finding}")
    
    # Calculate overall score
    scores = list(audit_results['component_scores'].values())
    audit_results['overall_score'] = sum(scores) / len(scores) if scores else 0
    
    # Generate comprehensive recommendations
    recommendations = generate_recommendations(audit_results)
    audit_results['recommendations'] = recommendations
    
    # Print summary
    print_audit_summary(audit_results)
    
    # Save detailed report
    save_audit_report(audit_results)
    
    return audit_results

def analyze_signal_performance():
    """Analyze AI signal generation performance"""
    metrics = {
        'score': 0,
        'total_signals_24h': 0,
        'avg_confidence': 0,
        'high_confidence_count': 0,
        'signal_distribution': {},
        'findings': []
    }
    
    try:
        with sqlite3.connect('enhanced_trading.db') as conn:
            cursor = conn.cursor()
            
            # Recent signals analysis
            cursor.execute("""
                SELECT confidence, action, symbol, timestamp 
                FROM unified_signals 
                WHERE timestamp > datetime('now', '-24 hours')
                ORDER BY timestamp DESC
            """)
            signals = cursor.fetchall()
            
            if signals:
                metrics['total_signals_24h'] = len(signals)
                confidences = [float(s[0]) for s in signals]
                metrics['avg_confidence'] = statistics.mean(confidences)
                metrics['high_confidence_count'] = len([c for c in confidences if c >= 75])
                
                # Signal distribution by action
                actions = [s[1] for s in signals]
                metrics['signal_distribution'] = {
                    'BUY': actions.count('BUY'),
                    'SELL': actions.count('SELL'),
                    'HOLD': actions.count('HOLD')
                }
                
                # Scoring
                if metrics['avg_confidence'] >= 70:
                    metrics['score'] += 30
                    metrics['findings'].append(f"✓ Excellent signal confidence: {metrics['avg_confidence']:.1f}%")
                elif metrics['avg_confidence'] >= 60:
                    metrics['score'] += 20
                    metrics['findings'].append(f"✓ Good signal confidence: {metrics['avg_confidence']:.1f}%")
                else:
                    metrics['findings'].append(f"⚠ Low signal confidence: {metrics['avg_confidence']:.1f}%")
                
                if metrics['total_signals_24h'] >= 10:
                    metrics['score'] += 25
                    metrics['findings'].append(f"✓ Active signal generation: {metrics['total_signals_24h']} signals/day")
                elif metrics['total_signals_24h'] >= 5:
                    metrics['score'] += 15
                else:
                    metrics['findings'].append(f"⚠ Low signal frequency: {metrics['total_signals_24h']} signals/day")
                
                hc_ratio = metrics['high_confidence_count'] / metrics['total_signals_24h']
                if hc_ratio >= 0.3:
                    metrics['score'] += 20
                    metrics['findings'].append(f"✓ Good high-confidence ratio: {hc_ratio:.1%}")
                else:
                    metrics['findings'].append(f"⚠ Low high-confidence signals: {hc_ratio:.1%}")
                
                # Signal balance
                buy_ratio = metrics['signal_distribution'].get('BUY', 0) / metrics['total_signals_24h']
                if 0.3 <= buy_ratio <= 0.7:
                    metrics['score'] += 15
                    metrics['findings'].append(f"✓ Balanced signal distribution")
                else:
                    metrics['findings'].append(f"⚠ Unbalanced signals: {buy_ratio:.1%} BUY")
            else:
                metrics['findings'].append("✗ No signals generated in last 24 hours")
                
    except Exception as e:
        metrics['findings'].append(f"✗ Signal analysis error: {e}")
    
    return metrics

def analyze_trading_performance():
    """Analyze actual trading performance"""
    metrics = {
        'score': 0,
        'total_trades': 0,
        'win_rate': 0,
        'profit_factor': 0,
        'avg_trade_duration': 0,
        'findings': []
    }
    
    try:
        # Check if OKX is properly configured
        if not (os.getenv('OKX_API_KEY') and os.getenv('OKX_SECRET_KEY')):
            metrics['findings'].append("⚠ OKX API credentials not configured")
            return metrics
        
        # Initialize OKX connection
        exchange = ccxt.okx({
            'apiKey': os.getenv('OKX_API_KEY'),
            'secret': os.getenv('OKX_SECRET_KEY'),
            'password': os.getenv('OKX_PASSPHRASE'),
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Analyze recent trading performance
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        all_trades = []
        
        for symbol in symbols:
            try:
                trades = exchange.fetch_my_trades(symbol, limit=50)
                all_trades.extend(trades)
            except Exception as e:
                metrics['findings'].append(f"⚠ Could not fetch {symbol} trades: {str(e)[:50]}")
        
        if all_trades:
            # Group trades for P&L calculation
            symbol_positions = {}
            
            for trade in all_trades:
                symbol = trade['symbol']
                if symbol not in symbol_positions:
                    symbol_positions[symbol] = {'buys': [], 'sells': [], 'fees': 0}
                
                fee = trade.get('fee', {}).get('cost', 0) if trade.get('fee') else 0
                symbol_positions[symbol]['fees'] += fee
                
                if trade['side'] == 'buy':
                    symbol_positions[symbol]['buys'].append({
                        'amount': trade['amount'], 
                        'price': trade['price'],
                        'timestamp': trade['timestamp']
                    })
                else:
                    symbol_positions[symbol]['sells'].append({
                        'amount': trade['amount'], 
                        'price': trade['price'],
                        'timestamp': trade['timestamp']
                    })
            
            # Calculate performance metrics
            winning_trades = 0
            total_profit = 0
            total_loss = 0
            completed_trades = 0
            trade_durations = []
            
            for symbol, positions in symbol_positions.items():
                if positions['buys'] and positions['sells']:
                    # Calculate weighted average prices
                    total_buy_value = sum(b['price'] * b['amount'] for b in positions['buys'])
                    total_buy_amount = sum(b['amount'] for b in positions['buys'])
                    avg_buy_price = total_buy_value / total_buy_amount
                    
                    total_sell_value = sum(s['price'] * s['amount'] for s in positions['sells'])
                    total_sell_amount = sum(s['amount'] for s in positions['sells'])
                    avg_sell_price = total_sell_value / total_sell_amount
                    
                    # Calculate P&L for matched amount
                    matched_amount = min(total_buy_amount, total_sell_amount)
                    
                    if matched_amount > 0:
                        pnl = (avg_sell_price - avg_buy_price) * matched_amount - positions['fees']
                        completed_trades += 1
                        
                        if pnl > 0:
                            winning_trades += 1
                            total_profit += pnl
                        else:
                            total_loss += abs(pnl)
                        
                        # Calculate trade duration (simplified)
                        if positions['buys'] and positions['sells']:
                            avg_buy_time = statistics.mean([b['timestamp'] for b in positions['buys']])
                            avg_sell_time = statistics.mean([s['timestamp'] for s in positions['sells']])
                            duration_hours = abs(avg_sell_time - avg_buy_time) / (1000 * 3600)
                            trade_durations.append(duration_hours)
            
            metrics['total_trades'] = completed_trades
            
            if completed_trades > 0:
                metrics['win_rate'] = (winning_trades / completed_trades) * 100
                metrics['profit_factor'] = total_profit / total_loss if total_loss > 0 else 1.0
                
                if trade_durations:
                    metrics['avg_trade_duration'] = statistics.mean(trade_durations)
                
                # Scoring
                if metrics['win_rate'] >= 60:
                    metrics['score'] += 35
                    metrics['findings'].append(f"✓ Excellent win rate: {metrics['win_rate']:.1f}%")
                elif metrics['win_rate'] >= 45:
                    metrics['score'] += 25
                    metrics['findings'].append(f"✓ Good win rate: {metrics['win_rate']:.1f}%")
                else:
                    metrics['findings'].append(f"⚠ Low win rate: {metrics['win_rate']:.1f}%")
                
                if metrics['profit_factor'] >= 2.0:
                    metrics['score'] += 30
                    metrics['findings'].append(f"✓ Excellent profit factor: {metrics['profit_factor']:.2f}")
                elif metrics['profit_factor'] >= 1.5:
                    metrics['score'] += 20
                else:
                    metrics['findings'].append(f"⚠ Low profit factor: {metrics['profit_factor']:.2f}")
                
                if completed_trades >= 10:
                    metrics['score'] += 20
                    metrics['findings'].append(f"✓ Active trading: {completed_trades} completed trades")
                elif completed_trades >= 5:
                    metrics['score'] += 10
                else:
                    metrics['findings'].append(f"⚠ Limited trading activity: {completed_trades} trades")
                
                total_pnl = total_profit - total_loss
                if total_pnl > 0:
                    metrics['score'] += 15
                    metrics['findings'].append(f"✓ Profitable: ${total_pnl:.2f}")
                else:
                    metrics['findings'].append(f"⚠ Currently unprofitable: ${total_pnl:.2f}")
            else:
                metrics['findings'].append("⚠ No completed round-trip trades found")
        else:
            metrics['findings'].append("⚠ No trading history available")
            
    except Exception as e:
        metrics['findings'].append(f"✗ Trading analysis error: {str(e)[:100]}")
    
    return metrics

def analyze_data_quality():
    """Analyze data quality and authenticity"""
    metrics = {
        'score': 0,
        'okx_connection': False,
        'database_health': 0,
        'data_freshness': 0,
        'findings': []
    }
    
    # Check OKX API configuration
    if os.getenv('OKX_API_KEY') and os.getenv('OKX_SECRET_KEY') and os.getenv('OKX_PASSPHRASE'):
        metrics['okx_connection'] = True
        metrics['score'] += 25
        metrics['findings'].append("✓ OKX API credentials properly configured")
        
        # Test actual OKX connection
        try:
            exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            # Test connection with a simple call
            ticker = exchange.fetch_ticker('BTC/USDT')
            if ticker and 'last' in ticker:
                metrics['score'] += 20
                metrics['findings'].append(f"✓ Live OKX data: BTC/USDT at ${ticker['last']:,.2f}")
        except Exception as e:
            metrics['findings'].append(f"⚠ OKX connection test failed: {str(e)[:50]}")
    else:
        metrics['findings'].append("⚠ OKX API credentials missing or incomplete")
    
    # Database health check
    try:
        with sqlite3.connect('enhanced_trading.db') as conn:
            cursor = conn.cursor()
            
            required_tables = ['unified_signals', 'portfolio_data', 'trading_performance']
            healthy_tables = 0
            total_records = 0
            
            for table in required_tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    healthy_tables += 1
                    total_records += count
                    
                    # Check data freshness
                    cursor.execute(f"SELECT timestamp FROM {table} ORDER BY timestamp DESC LIMIT 1")
                    latest = cursor.fetchone()
                    if latest:
                        latest_time = datetime.fromisoformat(latest[0].replace('Z', '+00:00'))
                        age_hours = (datetime.now() - latest_time.replace(tzinfo=None)).total_seconds() / 3600
                        
                        if age_hours < 1:
                            metrics['data_freshness'] += 1
                        
                except sqlite3.OperationalError:
                    metrics['findings'].append(f"⚠ Table {table} missing or corrupted")
            
            metrics['database_health'] = (healthy_tables / len(required_tables)) * 100
            
            if metrics['database_health'] == 100:
                metrics['score'] += 25
                metrics['findings'].append(f"✓ All database tables healthy ({total_records} total records)")
            elif metrics['database_health'] >= 80:
                metrics['score'] += 15
            else:
                metrics['findings'].append(f"⚠ Database health: {metrics['database_health']:.0f}%")
            
            if metrics['data_freshness'] >= 2:
                metrics['score'] += 15
                metrics['findings'].append("✓ Fresh data in multiple tables")
            elif metrics['data_freshness'] >= 1:
                metrics['score'] += 10
            else:
                metrics['findings'].append("⚠ Stale data detected")
                
    except Exception as e:
        metrics['findings'].append(f"✗ Database analysis failed: {e}")
    
    return metrics

def analyze_system_health():
    """Analyze system infrastructure health"""
    metrics = {
        'score': 0,
        'workflow_status': {},
        'file_integrity': 0,
        'findings': []
    }
    
    # Check core system files
    critical_files = [
        'unified_trading_platform.py',
        'enhanced_trading_system.py',
        'advanced_ml_optimizer.py',
        'signal_execution_bridge.py'
    ]
    
    existing_files = 0
    for file in critical_files:
        if os.path.exists(file):
            existing_files += 1
            file_size = os.path.getsize(file)
            if file_size > 1000:  # Reasonable size check
                metrics['score'] += 5
    
    metrics['file_integrity'] = (existing_files / len(critical_files)) * 100
    
    if metrics['file_integrity'] == 100:
        metrics['findings'].append("✓ All critical system files present")
    else:
        metrics['findings'].append(f"⚠ Missing system files: {metrics['file_integrity']:.0f}% complete")
    
    # Check database file integrity
    if os.path.exists('enhanced_trading.db'):
        db_size = os.path.getsize('enhanced_trading.db')
        if db_size > 10000:  # Reasonable database size
            metrics['score'] += 15
            metrics['findings'].append(f"✓ Database file healthy ({db_size:,} bytes)")
        else:
            metrics['findings'].append("⚠ Database file too small")
    else:
        metrics['findings'].append("✗ Database file missing")
    
    # Check for recent activity (log files, temp files)
    recent_activity = False
    for file in os.listdir('.'):
        if file.endswith('.log') or 'audit' in file or 'report' in file:
            mtime = os.path.getmtime(file)
            if (datetime.now().timestamp() - mtime) < 3600:  # Modified in last hour
                recent_activity = True
                break
    
    if recent_activity:
        metrics['score'] += 10
        metrics['findings'].append("✓ Recent system activity detected")
    
    return metrics

def analyze_risk_management():
    """Analyze risk management implementation"""
    metrics = {
        'score': 0,
        'diversification': 0,
        'position_sizing': 'Unknown',
        'stop_loss_coverage': 0,
        'findings': []
    }
    
    try:
        # Check portfolio diversification from OKX
        if os.getenv('OKX_API_KEY'):
            try:
                exchange = ccxt.okx({
                    'apiKey': os.getenv('OKX_API_KEY'),
                    'secret': os.getenv('OKX_SECRET_KEY'),
                    'password': os.getenv('OKX_PASSPHRASE'),
                    'sandbox': False,
                })
                
                balance = exchange.fetch_balance()
                non_zero_assets = {k: v for k, v in balance['total'].items() if v > 0 and k != 'USDT'}
                
                metrics['diversification'] = len(non_zero_assets)
                
                if metrics['diversification'] >= 5:
                    metrics['score'] += 25
                    metrics['findings'].append(f"✓ Good diversification: {metrics['diversification']} assets")
                elif metrics['diversification'] >= 3:
                    metrics['score'] += 15
                    metrics['findings'].append(f"✓ Moderate diversification: {metrics['diversification']} assets")
                else:
                    metrics['findings'].append(f"⚠ Low diversification: {metrics['diversification']} assets")
                
                # Position sizing analysis
                if non_zero_assets:
                    total_value = sum(balance['total'].values())
                    largest_position = max(balance['total'].values()) / total_value if total_value > 0 else 0
                    
                    if largest_position < 0.3:
                        metrics['score'] += 20
                        metrics['findings'].append(f"✓ Good position sizing: max {largest_position:.1%}")
                    elif largest_position < 0.5:
                        metrics['score'] += 10
                        metrics['findings'].append(f"✓ Acceptable position sizing: max {largest_position:.1%}")
                    else:
                        metrics['findings'].append(f"⚠ High concentration risk: {largest_position:.1%}")
                        
            except Exception as e:
                metrics['findings'].append(f"⚠ Portfolio analysis limited: {str(e)[:50]}")
    
        # Check for risk management features in database
        try:
            with sqlite3.connect('enhanced_trading.db') as conn:
                cursor = conn.cursor()
                
                # Check for stop loss implementation
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%stop%'")
                stop_tables = cursor.fetchall()
                
                if stop_tables:
                    metrics['score'] += 15
                    metrics['findings'].append("✓ Stop loss tables detected")
                else:
                    metrics['findings'].append("⚠ No stop loss system detected")
                
                # Check for risk metrics tracking
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%risk%'")
                risk_tables = cursor.fetchall()
                
                if risk_tables:
                    metrics['score'] += 10
                    metrics['findings'].append("✓ Risk tracking system present")
                    
        except Exception as e:
            metrics['findings'].append(f"⚠ Risk system check failed: {e}")
    
    except Exception as e:
        metrics['findings'].append(f"✗ Risk analysis error: {e}")
    
    return metrics

def generate_recommendations(audit_results):
    """Generate prioritized recommendations"""
    recommendations = []
    
    # Critical recommendations
    if audit_results['component_scores'].get('data_quality', 0) < 50:
        recommendations.append({
            'priority': 'CRITICAL',
            'category': 'Data Infrastructure',
            'title': 'Fix Data Connection Issues',
            'description': 'Ensure OKX API credentials are properly configured and test connectivity',
            'impact': 'System cannot function without reliable data sources'
        })
    
    if audit_results['component_scores'].get('trading_execution', 0) < 40:
        recommendations.append({
            'priority': 'CRITICAL',
            'category': 'Trading Performance',
            'title': 'Optimize Trading Strategy',
            'description': 'Review and improve signal thresholds, entry/exit rules, and risk parameters',
            'impact': 'Current strategy may be generating losses'
        })
    
    # High priority recommendations
    if audit_results['performance_metrics'].get('signals', {}).get('avg_confidence', 0) < 65:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Signal Quality',
            'title': 'Improve Signal Confidence',
            'description': 'Retrain ML models with recent market data and optimize feature selection',
            'impact': 'Better signals lead to more profitable trades'
        })
    
    if audit_results['component_scores'].get('risk_management', 0) < 60:
        recommendations.append({
            'priority': 'HIGH',
            'category': 'Risk Management',
            'title': 'Implement Comprehensive Risk Controls',
            'description': 'Add automated stop losses, position sizing limits, and portfolio rebalancing',
            'impact': 'Protect capital from significant losses'
        })
    
    # Medium priority recommendations
    if audit_results['performance_metrics'].get('trading', {}).get('total_trades', 0) < 5:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'Trading Activity',
            'title': 'Increase Trading Frequency',
            'description': 'Lower signal confidence thresholds or expand to more trading pairs',
            'impact': 'More opportunities for profit generation'
        })
    
    if audit_results['component_scores'].get('system_health', 0) < 80:
        recommendations.append({
            'priority': 'MEDIUM',
            'category': 'System Optimization',
            'title': 'Improve System Performance',
            'description': 'Optimize database queries, implement caching, and add monitoring',
            'impact': 'Better system reliability and faster execution'
        })
    
    # Always include these
    recommendations.extend([
        {
            'priority': 'LOW',
            'category': 'Monitoring',
            'title': 'Enhanced Performance Tracking',
            'description': 'Implement real-time dashboards and automated performance alerts',
            'impact': 'Better visibility into system performance'
        },
        {
            'priority': 'LOW',
            'category': 'Documentation',
            'title': 'Trading Journal Implementation',
            'description': 'Detailed logging of all trades with reasoning and outcomes',
            'impact': 'Improved strategy development and compliance'
        }
    ])
    
    return recommendations

def print_audit_summary(results):
    """Print comprehensive audit summary"""
    
    print("\n" + "="*70)
    print("AUDIT SUMMARY")
    print("="*70)
    
    overall_score = results['overall_score']
    
    if overall_score >= 80:
        status = "EXCELLENT"
        recommendation = "System performing well, continue monitoring"
    elif overall_score >= 65:
        status = "GOOD"
        recommendation = "Minor optimizations recommended"
    elif overall_score >= 50:
        status = "FAIR"
        recommendation = "Several improvements needed"
    else:
        status = "POOR"
        recommendation = "Immediate attention required"
    
    print(f"Overall Performance Score: {overall_score:.1f}/100")
    print(f"System Status: {status}")
    print(f"Recommendation: {recommendation}")
    
    print(f"\nComponent Breakdown:")
    for component, score in results['component_scores'].items():
        component_name = component.replace('_', ' ').title()
        print(f"  {component_name:<25} {score:>6.1f}/100")
    
    print(f"\nKey Performance Indicators:")
    
    # Signal metrics
    signal_metrics = results['performance_metrics'].get('signals', {})
    if signal_metrics:
        print(f"  Signal Generation:")
        print(f"    - Daily signals: {signal_metrics.get('total_signals_24h', 0)}")
        print(f"    - Avg confidence: {signal_metrics.get('avg_confidence', 0):.1f}%")
        print(f"    - High confidence: {signal_metrics.get('high_confidence_count', 0)}")
    
    # Trading metrics
    trading_metrics = results['performance_metrics'].get('trading', {})
    if trading_metrics:
        print(f"  Trading Performance:")
        print(f"    - Win rate: {trading_metrics.get('win_rate', 0):.1f}%")
        print(f"    - Profit factor: {trading_metrics.get('profit_factor', 0):.2f}")
        print(f"    - Total trades: {trading_metrics.get('total_trades', 0)}")
    
    print(f"\nTop Priority Recommendations:")
    priority_order = {'CRITICAL': 1, 'HIGH': 2, 'MEDIUM': 3, 'LOW': 4}
    sorted_recs = sorted(results['recommendations'], key=lambda x: priority_order.get(x['priority'], 5))
    
    for i, rec in enumerate(sorted_recs[:5], 1):
        print(f"  {i}. [{rec['priority']}] {rec['title']}")
        print(f"     {rec['description']}")

def save_audit_report(results):
    """Save detailed audit report"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'system_audit_report_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed report saved: {filename}")
    return filename

if __name__ == "__main__":
    analyze_trading_system()