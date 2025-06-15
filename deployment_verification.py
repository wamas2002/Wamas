"""
Comprehensive Deployment Verification System
Ensures all components are operational with authentic data before deployment
"""

import os
import sqlite3
import requests
import time
from datetime import datetime
import ccxt

class DeploymentVerifier:
    def __init__(self):
        self.results = {}
        self.issues = []
        
    def verify_okx_connection(self):
        """Verify OKX API connection and balance retrieval"""
        try:
            api_key = os.getenv('OKX_API_KEY')
            secret_key = os.getenv('OKX_SECRET_KEY')
            passphrase = os.getenv('OKX_PASSPHRASE')
            
            if not all([api_key, secret_key, passphrase]):
                self.issues.append("OKX API credentials missing")
                return False
                
            exchange = ccxt.okx({
                'apiKey': api_key,
                'secret': secret_key,
                'password': passphrase,
                'sandbox': False,
                'enableRateLimit': True,
            })
            
            balance = exchange.fetch_balance()
            usdt_balance = balance.get('USDT', {}).get('total', 0)
            
            self.results['okx_connection'] = {
                'status': 'PASS',
                'balance': float(usdt_balance) if usdt_balance else 0,
                'timestamp': datetime.now().isoformat()
            }
            return True
            
        except Exception as e:
            self.issues.append(f"OKX connection failed: {e}")
            self.results['okx_connection'] = {'status': 'FAIL', 'error': str(e)}
            return False
            
    def verify_trading_engines(self):
        """Verify all trading engines are operational"""
        engines = [
            ('Dynamic Trading System', 'http://localhost:5000/api/status'),
            ('Elite Trading Dashboard', 'http://localhost:3000/api/dashboard-data'),
            ('Enhanced AI Trading Dashboard', 'http://localhost:5002/api/enhanced/status'),
            ('Professional Trading Optimizer', 'http://localhost:5001/api/professional/status')
        ]
        
        engine_status = {}
        
        for name, url in engines:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    engine_status[name] = 'OPERATIONAL'
                else:
                    engine_status[name] = f'HTTP {response.status_code}'
                    self.issues.append(f"{name} returned HTTP {response.status_code}")
            except Exception as e:
                engine_status[name] = 'OFFLINE'
                self.issues.append(f"{name} connection failed: {e}")
                
        self.results['trading_engines'] = engine_status
        
    def verify_databases(self):
        """Verify database integrity and signal generation"""
        databases = [
            'dynamic_trading.db',
            'enhanced_trading.db', 
            'professional_trading.db',
            'pure_local_trading.db',
            'elite_dashboard.db'
        ]
        
        db_status = {}
        
        for db_name in databases:
            try:
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()
                
                # Check database file exists and is accessible
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                # Look for recent signals
                signal_count = 0
                for table_row in tables:
                    table_name = table_row[0]
                    if 'signal' in table_name.lower():
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE timestamp > datetime('now', '-1 hour')")
                            count = cursor.fetchone()[0]
                            signal_count += count
                        except:
                            continue
                            
                db_status[db_name] = {
                    'tables': len(tables),
                    'recent_signals': signal_count,
                    'status': 'HEALTHY'
                }
                conn.close()
                
            except Exception as e:
                db_status[db_name] = {'status': 'ERROR', 'error': str(e)}
                self.issues.append(f"Database {db_name} error: {e}")
                
        self.results['databases'] = db_status
        
    def verify_signal_generation(self):
        """Verify active signal generation from trading engines"""
        try:
            # Check Pure Local Trading Engine logs for recent signals
            signal_count = 0
            confidence_sum = 0
            
            # Read from databases
            try:
                conn = sqlite3.connect('dynamic_trading.db')
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trading_signals'")
                if cursor.fetchone():
                    cursor.execute("SELECT confidence FROM trading_signals WHERE timestamp > datetime('now', '-2 hours')")
                    results = cursor.fetchall()
                    for result in results:
                        signal_count += 1
                        confidence_sum += float(result[0])
                conn.close()
            except:
                pass
                
            avg_confidence = confidence_sum / signal_count if signal_count > 0 else 0
            
            self.results['signal_generation'] = {
                'active_signals': signal_count,
                'avg_confidence': round(avg_confidence, 2),
                'status': 'ACTIVE' if signal_count > 0 else 'INACTIVE'
            }
            
            if signal_count == 0:
                self.issues.append("No recent signals detected from trading engines")
                
        except Exception as e:
            self.issues.append(f"Signal verification failed: {e}")
            self.results['signal_generation'] = {'status': 'ERROR', 'error': str(e)}
            
    def verify_portfolio_calculation(self):
        """Verify portfolio balance and P&L calculations"""
        try:
            # Test Elite Dashboard API
            response = requests.get('http://localhost:3000/api/dashboard-data', timeout=10)
            if response.status_code == 200:
                data = response.json()
                portfolio = data.get('portfolio', {})
                pnl = data.get('profit_loss', {})
                
                self.results['portfolio_calculation'] = {
                    'balance': portfolio.get('balance', 0),
                    'total_value': portfolio.get('total_value', 0),
                    'current_profit': pnl.get('current_profit', 0),
                    'status': 'CALCULATED'
                }
            else:
                self.issues.append(f"Portfolio API returned HTTP {response.status_code}")
                
        except Exception as e:
            self.issues.append(f"Portfolio verification failed: {e}")
            self.results['portfolio_calculation'] = {'status': 'ERROR', 'error': str(e)}
            
    def generate_deployment_report(self):
        """Generate comprehensive deployment readiness report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'READY' if len(self.issues) == 0 else 'ISSUES_DETECTED',
            'verification_results': self.results,
            'issues_found': self.issues,
            'recommendations': []
        }
        
        # Add recommendations based on issues
        if self.issues:
            report['recommendations'].append("Resolve all detected issues before deployment")
            
        if report['overall_status'] == 'READY':
            report['recommendations'].append("System is deployment-ready with authentic data integration")
            
        return report
        
    def run_full_verification(self):
        """Run complete deployment verification"""
        print("🔍 Starting Comprehensive Deployment Verification...")
        
        print("📊 Verifying OKX connection...")
        self.verify_okx_connection()
        
        print("🚀 Verifying trading engines...")
        self.verify_trading_engines()
        
        print("💾 Verifying databases...")
        self.verify_databases()
        
        print("📈 Verifying signal generation...")
        self.verify_signal_generation()
        
        print("💰 Verifying portfolio calculations...")
        self.verify_portfolio_calculation()
        
        print("📋 Generating deployment report...")
        report = self.generate_deployment_report()
        
        # Save report
        with open('deployment_verification_report.json', 'w') as f:
            import json
            json.dump(report, f, indent=2)
            
        print(f"✅ Verification complete. Status: {report['overall_status']}")
        print(f"📄 Report saved to deployment_verification_report.json")
        
        return report

if __name__ == '__main__':
    verifier = DeploymentVerifier()
    report = verifier.run_full_verification()
    
    # Print summary
    print("\n" + "="*60)
    print("DEPLOYMENT VERIFICATION SUMMARY")
    print("="*60)
    print(f"Overall Status: {report['overall_status']}")
    print(f"Issues Found: {len(report['issues_found'])}")
    
    if report['issues_found']:
        print("\nIssues to resolve:")
        for issue in report['issues_found']:
            print(f"  ❌ {issue}")
    else:
        print("\n✅ All systems verified and ready for deployment")
        print("✅ Authentic data integration confirmed")
        print("✅ Trading engines operational")
        print("✅ Elite dashboard functional")