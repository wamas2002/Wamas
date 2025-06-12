#!/usr/bin/env python3
"""
Quick Trading System Performance Audit
Fast analysis with immediate recommendations
"""

import sqlite3
import json
import os
import requests
from datetime import datetime, timedelta

def audit_system_performance():
    """Run quick comprehensive audit"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'overall_score': 0,
        'components': {},
        'critical_issues': [],
        'recommendations': []
    }
    
    print("="*60)
    print("TRADING SYSTEM PERFORMANCE AUDIT")
    print("="*60)
    
    # 1. API Health Check
    print("\n1. CHECKING API HEALTH...")
    api_score = 0
    try:
        response = requests.get('http://localhost:5000/api/unified/health', timeout=3)
        if response.status_code == 200:
            health_data = response.json()
            api_score = 25
            print(f"✓ API responding - System Health: {health_data.get('overall_health', 0)}%")
        else:
            print(f"✗ API error: {response.status_code}")
    except Exception as e:
        print(f"✗ API connection failed: {e}")
    
    results['components']['api_health'] = api_score
    
    # 2. Signal Performance
    print("\n2. ANALYZING SIGNAL PERFORMANCE...")
    signal_score = 0
    try:
        response = requests.get('http://localhost:5000/api/unified/signals', timeout=3)
        if response.status_code == 200:
            signals = response.json()
            if signals and len(signals) > 0:
                avg_confidence = sum(s.get('confidence', 0) for s in signals) / len(signals)
                high_conf_signals = len([s for s in signals if s.get('confidence', 0) >= 75])
                
                print(f"✓ Generated {len(signals)} signals")
                print(f"✓ Average confidence: {avg_confidence:.1f}%")
                print(f"✓ High confidence signals: {high_conf_signals}")
                
                if avg_confidence >= 70:
                    signal_score += 20
                if high_conf_signals >= 2:
                    signal_score += 15
                if len(signals) >= 5:
                    signal_score += 10
            else:
                print("✗ No signals generated")
                results['critical_issues'].append("No AI signals being generated")
        else:
            print(f"✗ Signal API error: {response.status_code}")
    except Exception as e:
        print(f"✗ Signal check failed: {e}")
    
    results['components']['signal_performance'] = signal_score
    
    # 3. Trading Performance
    print("\n3. CHECKING TRADING PERFORMANCE...")
    trading_score = 0
    try:
        response = requests.get('http://localhost:5000/api/unified/performance', timeout=3)
        if response.status_code == 200:
            perf = response.json()
            win_rate = perf.get('win_rate', 0)
            profit_factor = perf.get('profit_factor', 0)
            total_trades = perf.get('total_trades', 0)
            
            print(f"✓ Win Rate: {win_rate}%")
            print(f"✓ Profit Factor: {profit_factor}")
            print(f"✓ Total Trades: {total_trades}")
            
            if win_rate >= 50:
                trading_score += 20
            elif win_rate >= 40:
                trading_score += 10
            else:
                results['critical_issues'].append(f"Low win rate: {win_rate}%")
            
            if profit_factor >= 2.0:
                trading_score += 15
            elif profit_factor >= 1.5:
                trading_score += 10
            
            if total_trades >= 5:
                trading_score += 10
            elif total_trades == 0:
                results['critical_issues'].append("No completed trades")
        else:
            print(f"✗ Performance API error: {response.status_code}")
    except Exception as e:
        print(f"✗ Performance check failed: {e}")
    
    results['components']['trading_performance'] = trading_score
    
    # 4. Data Authenticity
    print("\n4. VERIFYING DATA AUTHENTICITY...")
    data_score = 0
    try:
        # Check if OKX credentials are configured
        if os.getenv('OKX_API_KEY') and os.getenv('OKX_SECRET_KEY'):
            data_score += 20
            print("✓ OKX API credentials configured")
        else:
            print("✗ OKX API credentials missing")
            results['critical_issues'].append("OKX API credentials not configured")
        
        # Check portfolio data
        response = requests.get('http://localhost:5000/api/unified/portfolio', timeout=3)
        if response.status_code == 200:
            portfolio = response.json()
            if portfolio and len(portfolio) > 0:
                data_score += 15
                print(f"✓ Portfolio data available: {len(portfolio)} assets")
            else:
                print("✗ No portfolio data")
        
    except Exception as e:
        print(f"✗ Data check failed: {e}")
    
    results['components']['data_authenticity'] = data_score
    
    # 5. Database Health
    print("\n5. CHECKING DATABASE HEALTH...")
    db_score = 0
    try:
        with sqlite3.connect('enhanced_trading.db') as conn:
            cursor = conn.cursor()
            
            # Check key tables
            tables = ['unified_signals', 'portfolio_data', 'trading_performance']
            healthy_tables = 0
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    healthy_tables += 1
                    print(f"✓ Table {table}: {count} records")
                except:
                    print(f"✗ Table {table}: missing or corrupted")
            
            db_score = (healthy_tables / len(tables)) * 25
            
    except Exception as e:
        print(f"✗ Database check failed: {e}")
        results['critical_issues'].append("Database connectivity issues")
    
    results['components']['database_health'] = db_score
    
    # Calculate overall score
    total_score = sum(results['components'].values())
    results['overall_score'] = total_score
    
    # Generate status
    if total_score >= 80:
        status = "EXCELLENT"
        color = "🟢"
    elif total_score >= 60:
        status = "GOOD"
        color = "🟡"
    elif total_score >= 40:
        status = "FAIR"
        color = "🟠"
    else:
        status = "POOR"
        color = "🔴"
    
    print("\n" + "="*60)
    print("AUDIT RESULTS")
    print("="*60)
    print(f"Overall Score: {total_score:.0f}/100")
    print(f"System Status: {status}")
    
    # Component breakdown
    print("\nComponent Scores:")
    for component, score in results['components'].items():
        print(f"  {component.replace('_', ' ').title()}: {score:.0f}")
    
    # Critical Issues
    if results['critical_issues']:
        print(f"\nCritical Issues ({len(results['critical_issues'])}):")
        for i, issue in enumerate(results['critical_issues'], 1):
            print(f"  {i}. {issue}")
    
    # Generate Recommendations
    recommendations = []
    
    if results['components']['data_authenticity'] < 20:
        recommendations.append({
            'priority': 'CRITICAL',
            'title': 'Configure OKX API Credentials',
            'description': 'Set up OKX_API_KEY, OKX_SECRET_KEY, and OKX_PASSPHRASE environment variables'
        })
    
    if results['components']['trading_performance'] < 15:
        recommendations.append({
            'priority': 'HIGH',
            'title': 'Improve Trading Strategy',
            'description': 'Optimize signal thresholds and implement better risk management'
        })
    
    if results['components']['signal_performance'] < 20:
        recommendations.append({
            'priority': 'HIGH',
            'title': 'Enhance Signal Generation',
            'description': 'Retrain ML models and improve signal confidence levels'
        })
    
    if total_score < 60:
        recommendations.append({
            'priority': 'MEDIUM',
            'title': 'System Health Monitoring',
            'description': 'Implement automated health checks and performance monitoring'
        })
    
    recommendations.append({
        'priority': 'LOW',
        'title': 'Add Stop Loss Protection',
        'description': 'Implement automatic stop losses for risk management'
    })
    
    recommendations.append({
        'priority': 'LOW',
        'title': 'Portfolio Diversification',
        'description': 'Spread investments across more cryptocurrency pairs'
    })
    
    results['recommendations'] = recommendations
    
    print(f"\nTop Recommendations:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"  {i}. [{rec['priority']}] {rec['title']}")
        print(f"     {rec['description']}")
    
    # Save results
    with open('system_audit_report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed report saved to: system_audit_report.json")
    print("Audit completed!")
    
    return results

if __name__ == "__main__":
    audit_system_performance()