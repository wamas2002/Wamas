#!/usr/bin/env python3
"""
Comprehensive System Audit - June 11, 2025
Complete diagnostic analysis of the AI trading platform
"""

import sqlite3
import requests
import json
import time
import os
from datetime import datetime, timedelta
import pandas as pd
import sys

class ComprehensiveSystemAuditor:
    def __init__(self):
        self.audit_timestamp = datetime.now().isoformat()
        self.audit_results = {
            'timestamp': self.audit_timestamp,
            'overall_health': 'UNKNOWN',
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'workflow_status': {},
            'database_health': {},
            'api_health': {},
            'port_analysis': {},
            'performance_metrics': {}
        }
        
    def check_port_conflicts(self):
        """Check for port conflicts and multi-port issues"""
        print("🔍 Analyzing Port Configuration...")
        
        ports_to_check = [5000, 5001, 5002, 5003]
        port_status = {}
        
        for port in ports_to_check:
            try:
                response = requests.get(f"http://localhost:{port}", timeout=5)
                port_status[port] = {
                    'status': 'ACTIVE',
                    'response_code': response.status_code,
                    'accessible': True
                }
                print(f"✅ Port {port}: ACTIVE (HTTP {response.status_code})")
            except requests.exceptions.ConnectionError:
                port_status[port] = {
                    'status': 'INACTIVE',
                    'accessible': False
                }
                print(f"❌ Port {port}: INACTIVE")
            except Exception as e:
                port_status[port] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'accessible': False
                }
                print(f"⚠️ Port {port}: ERROR - {e}")
        
        # Check for multiple services on same port
        active_ports = [p for p, s in port_status.items() if s.get('accessible')]
        
        if len(active_ports) > 2:
            self.audit_results['warnings'].append(
                f"Multiple services running simultaneously on {len(active_ports)} ports: {active_ports}"
            )
        
        self.audit_results['port_analysis'] = port_status
        return port_status
    
    def check_database_integrity(self):
        """Check database health and schema integrity"""
        print("🔍 Analyzing Database Integrity...")
        
        db_files = ['live_trading.db', 'trading.db']
        db_health = {}
        
        for db_file in db_files:
            if os.path.exists(db_file):
                try:
                    conn = sqlite3.connect(db_file, timeout=10.0)
                    cursor = conn.cursor()
                    
                    # Get table list
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    table_info = {}
                    for table in tables:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        table_info[table] = {'row_count': count}
                        
                        # Check for recent data
                        try:
                            cursor.execute(f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT 1")
                            recent = cursor.fetchone()
                            if recent:
                                table_info[table]['has_recent_data'] = True
                            else:
                                table_info[table]['has_recent_data'] = False
                        except:
                            table_info[table]['has_recent_data'] = 'UNKNOWN'
                    
                    conn.close()
                    
                    db_health[db_file] = {
                        'status': 'HEALTHY',
                        'tables': table_info,
                        'total_tables': len(tables)
                    }
                    print(f"✅ {db_file}: HEALTHY ({len(tables)} tables)")
                    
                except Exception as e:
                    db_health[db_file] = {
                        'status': 'ERROR',
                        'error': str(e)
                    }
                    print(f"❌ {db_file}: ERROR - {e}")
                    self.audit_results['critical_issues'].append(f"Database {db_file} error: {e}")
            else:
                db_health[db_file] = {
                    'status': 'MISSING'
                }
                print(f"⚠️ {db_file}: MISSING")
                self.audit_results['warnings'].append(f"Database file {db_file} not found")
        
        self.audit_results['database_health'] = db_health
        return db_health
    
    def check_api_endpoints(self):
        """Test critical API endpoints"""
        print("🔍 Testing API Endpoints...")
        
        critical_endpoints = [
            '/api/portfolio',
            '/api/signals',
            '/api/trading/active-positions',
            '/api/ai/model-insights',
            '/api/screener/scan',
            '/api/system-health'
        ]
        
        api_health = {}
        base_url = "http://localhost:5000"
        
        for endpoint in critical_endpoints:
            try:
                response = requests.get(f"{base_url}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        api_health[endpoint] = {
                            'status': 'HEALTHY',
                            'response_time': response.elapsed.total_seconds(),
                            'data_structure': type(data).__name__
                        }
                        print(f"✅ {endpoint}: HEALTHY ({response.elapsed.total_seconds():.2f}s)")
                    except:
                        api_health[endpoint] = {
                            'status': 'INVALID_JSON',
                            'response_time': response.elapsed.total_seconds()
                        }
                        print(f"⚠️ {endpoint}: Invalid JSON response")
                else:
                    api_health[endpoint] = {
                        'status': 'HTTP_ERROR',
                        'status_code': response.status_code
                    }
                    print(f"❌ {endpoint}: HTTP {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                api_health[endpoint] = {
                    'status': 'CONNECTION_ERROR'
                }
                print(f"❌ {endpoint}: Connection failed")
                self.audit_results['critical_issues'].append(f"API endpoint {endpoint} unreachable")
            except Exception as e:
                api_health[endpoint] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                print(f"❌ {endpoint}: {e}")
        
        self.audit_results['api_health'] = api_health
        return api_health
    
    def analyze_system_performance(self):
        """Analyze system performance metrics"""
        print("🔍 Analyzing System Performance...")
        
        try:
            # Get AI model insights
            response = requests.get("http://localhost:5000/api/ai/model-insights", timeout=10)
            if response.status_code == 200:
                insights = response.json()
                
                performance = {
                    'system_health': insights.get('confidence', 0),
                    'win_rate': insights.get('performance_insights', {}).get('win_rate', 0),
                    'total_trades': insights.get('performance_insights', {}).get('total_trades', 0),
                    'profit_factor': insights.get('performance_insights', {}).get('profit_factor', 0),
                    'signals_per_hour': insights.get('system_status', {}).get('signals_per_hour', 0)
                }
                
                # Analyze performance issues
                if performance['win_rate'] == 0 and performance['total_trades'] > 0:
                    self.audit_results['critical_issues'].append(
                        f"CRITICAL: 0% win rate with {performance['total_trades']} trades"
                    )
                
                if performance['system_health'] < 70:
                    self.audit_results['warnings'].append(
                        f"Low system health: {performance['system_health']:.1f}%"
                    )
                
                if performance['signals_per_hour'] > 50:
                    self.audit_results['warnings'].append(
                        f"High signal frequency: {performance['signals_per_hour']} signals/hour"
                    )
                
                self.audit_results['performance_metrics'] = performance
                print(f"📊 System Health: {performance['system_health']:.1f}%")
                print(f"📊 Win Rate: {performance['win_rate']:.1f}%")
                print(f"📊 Total Trades: {performance['total_trades']}")
                
        except Exception as e:
            print(f"❌ Performance analysis failed: {e}")
            self.audit_results['critical_issues'].append(f"Performance analysis failed: {e}")
    
    def check_okx_connectivity(self):
        """Test OKX API connectivity"""
        print("🔍 Testing OKX API Connectivity...")
        
        try:
            # Test basic market data endpoint
            response = requests.get("http://localhost:5000/api/market-data/BTC", timeout=10)
            if response.status_code == 200:
                print("✅ OKX API: CONNECTED")
                return True
            else:
                print(f"⚠️ OKX API: HTTP {response.status_code}")
                self.audit_results['warnings'].append("OKX API connectivity issues")
                return False
        except Exception as e:
            print(f"❌ OKX API: {e}")
            self.audit_results['critical_issues'].append(f"OKX API error: {e}")
            return False
    
    def generate_recommendations(self):
        """Generate system optimization recommendations"""
        print("🔍 Generating Recommendations...")
        
        recommendations = []
        
        # Performance-based recommendations
        if self.audit_results['performance_metrics'].get('win_rate', 0) == 0:
            recommendations.append({
                'priority': 'CRITICAL',
                'category': 'PERFORMANCE',
                'action': 'Increase signal confidence threshold to improve win rate',
                'impact': 'HIGH'
            })
        
        # Port management recommendations
        active_ports = len([p for p, s in self.audit_results['port_analysis'].items() 
                           if s.get('accessible')])
        if active_ports > 2:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'RESOURCE_MANAGEMENT',
                'action': f'Consolidate services - {active_ports} ports active simultaneously',
                'impact': 'MEDIUM'
            })
        
        # Database optimization
        db_issues = len([db for db, status in self.audit_results['database_health'].items() 
                        if status.get('status') != 'HEALTHY'])
        if db_issues > 0:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'DATABASE',
                'action': 'Resolve database connectivity and schema issues',
                'impact': 'HIGH'
            })
        
        self.audit_results['recommendations'] = recommendations
        
        for rec in recommendations:
            print(f"💡 {rec['priority']}: {rec['action']}")
    
    def calculate_overall_health(self):
        """Calculate overall system health score"""
        health_score = 100
        
        # Deduct for critical issues
        health_score -= len(self.audit_results['critical_issues']) * 25
        
        # Deduct for warnings
        health_score -= len(self.audit_results['warnings']) * 10
        
        # Performance factor
        perf_health = self.audit_results['performance_metrics'].get('system_health', 50)
        health_score = (health_score + perf_health) / 2
        
        health_score = max(0, min(100, health_score))
        
        if health_score >= 90:
            self.audit_results['overall_health'] = 'EXCELLENT'
        elif health_score >= 70:
            self.audit_results['overall_health'] = 'GOOD'
        elif health_score >= 50:
            self.audit_results['overall_health'] = 'FAIR'
        else:
            self.audit_results['overall_health'] = 'CRITICAL'
        
        self.audit_results['health_score'] = health_score
        return health_score
    
    def run_full_audit(self):
        """Execute comprehensive system audit"""
        print("🚀 Starting Comprehensive System Audit")
        print("=" * 60)
        
        # Run all diagnostic checks
        self.check_port_conflicts()
        print()
        
        self.check_database_integrity()
        print()
        
        self.check_api_endpoints()
        print()
        
        self.analyze_system_performance()
        print()
        
        self.check_okx_connectivity()
        print()
        
        self.generate_recommendations()
        print()
        
        # Calculate overall health
        health_score = self.calculate_overall_health()
        
        print("=" * 60)
        print("📋 AUDIT SUMMARY")
        print("=" * 60)
        print(f"Overall Health: {self.audit_results['overall_health']} ({health_score:.1f}%)")
        print(f"Critical Issues: {len(self.audit_results['critical_issues'])}")
        print(f"Warnings: {len(self.audit_results['warnings'])}")
        print(f"Recommendations: {len(self.audit_results['recommendations'])}")
        print()
        
        if self.audit_results['critical_issues']:
            print("🚨 CRITICAL ISSUES:")
            for issue in self.audit_results['critical_issues']:
                print(f"  • {issue}")
            print()
        
        if self.audit_results['warnings']:
            print("⚠️ WARNINGS:")
            for warning in self.audit_results['warnings']:
                print(f"  • {warning}")
            print()
        
        # Save audit results
        audit_filename = f"system_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(audit_filename, 'w') as f:
            json.dump(self.audit_results, f, indent=2, default=str)
        
        print(f"📄 Detailed audit saved to: {audit_filename}")
        
        return self.audit_results

def main():
    """Main audit execution"""
    auditor = ComprehensiveSystemAuditor()
    results = auditor.run_full_audit()
    
    # Return exit code based on health
    if results['overall_health'] in ['EXCELLENT', 'GOOD']:
        sys.exit(0)
    elif results['overall_health'] == 'FAIR':
        sys.exit(1)
    else:
        sys.exit(2)

if __name__ == "__main__":
    main()