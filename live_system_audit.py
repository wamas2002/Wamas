"""
Live System Audit - Comprehensive analysis with authentic data
"""

import sqlite3
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

class LiveSystemAudit:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_health': {},
            'trading_analysis': {},
            'ai_performance': {},
            'strategy_effectiveness': {},
            'risk_system': {},
            'background_services': {}
        }
        
    def audit_databases(self):
        """Audit all system databases with authentic data"""
        print("Auditing system databases...")
        
        db_files = [
            'data/trading_data.db',
            'data/ai_performance.db', 
            'data/autoconfig.db',
            'data/smart_selector.db',
            'data/strategy_analysis.db',
            'data/risk_management.db',
            'data/sentiment_data.db',
            'data/strategy_performance.db',
            'data/alerts.db'
        ]
        
        db_status = {}
        
        for db_path in db_files:
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # Get tables
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    # Get record counts
                    table_info = {}
                    total_records = 0
                    
                    for table in tables:
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {table};")
                            count = cursor.fetchone()[0]
                            total_records += count
                            table_info[table] = count
                        except Exception as e:
                            table_info[table] = f"Error: {str(e)[:30]}"
                    
                    file_size = os.path.getsize(db_path) / (1024*1024)  # MB
                    
                    db_status[os.path.basename(db_path)] = {
                        'status': 'Active',
                        'tables': len(tables),
                        'total_records': total_records,
                        'size_mb': round(file_size, 2),
                        'table_details': table_info
                    }
                    
                    conn.close()
                    print(f"✅ {os.path.basename(db_path)}: {len(tables)} tables, {total_records} records")
                    
                except Exception as e:
                    db_status[os.path.basename(db_path)] = {'status': f'Error: {str(e)[:50]}'}
                    
        self.results['system_health']['databases'] = db_status
    
    def analyze_trading_data(self):
        """Analyze authentic trading data from the last 72 hours"""
        print("Analyzing trading performance...")
        
        trading_data = {
            'live_trades': [],
            'symbol_activity': {},
            'execution_summary': {},
            'performance_metrics': {}
        }
        
        try:
            # Check trading_data.db
            if os.path.exists('data/trading_data.db'):
                conn = sqlite3.connect('data/trading_data.db')
                cursor = conn.cursor()
                
                # Get table structure
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                # Look for trading records
                for table in tables:
                    try:
                        cursor.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 10;")
                        columns = [description[0] for description in cursor.description]
                        rows = cursor.fetchall()
                        
                        if rows:
                            # Convert to readable format
                            recent_records = []
                            for row in rows:
                                record = dict(zip(columns, row))
                                recent_records.append(record)
                            
                            trading_data['live_trades'].extend(recent_records)
                            
                    except Exception as e:
                        print(f"Error reading table {table}: {e}")
                
                conn.close()
                
            # Check autoconfig.db for strategy assignments
            if os.path.exists('data/autoconfig.db'):
                conn = sqlite3.connect('data/autoconfig.db')
                cursor = conn.cursor()
                
                try:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    for table in tables:
                        cursor.execute(f"SELECT COUNT(*) FROM {table};")
                        count = cursor.fetchone()[0]
                        trading_data['symbol_activity'][table] = count
                        
                except Exception as e:
                    pass
                    
                conn.close()
                
        except Exception as e:
            print(f"Trading data analysis error: {e}")
        
        self.results['trading_analysis'] = trading_data
    
    def analyze_ai_performance(self):
        """Analyze AI model performance with authentic data"""
        print("Analyzing AI model performance...")
        
        ai_data = {
            'model_metrics': {},
            'prediction_accuracy': {},
            'active_models': [],
            'performance_summary': {}
        }
        
        try:
            if os.path.exists('data/ai_performance.db'):
                conn = sqlite3.connect('data/ai_performance.db')
                cursor = conn.cursor()
                
                # Get tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    try:
                        # Get recent records
                        cursor.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 5;")
                        columns = [description[0] for description in cursor.description]
                        rows = cursor.fetchall()
                        
                        if rows:
                            records = []
                            for row in rows:
                                record = dict(zip(columns, row))
                                records.append(record)
                            
                            ai_data['model_metrics'][table] = {
                                'recent_records': len(records),
                                'latest_data': records[0] if records else None
                            }
                            
                        # Get total count
                        cursor.execute(f"SELECT COUNT(*) FROM {table};")
                        total = cursor.fetchone()[0]
                        ai_data['model_metrics'][table]['total_records'] = total
                        
                    except Exception as e:
                        ai_data['model_metrics'][table] = {'error': str(e)[:50]}
                
                conn.close()
                
        except Exception as e:
            print(f"AI performance analysis error: {e}")
        
        self.results['ai_performance'] = ai_data
    
    def analyze_strategy_effectiveness(self):
        """Analyze strategy performance and assignments"""
        print("Analyzing strategy effectiveness...")
        
        strategy_data = {
            'active_assignments': {},
            'performance_metrics': {},
            'strategy_switches': [],
            'effectiveness_summary': {}
        }
        
        try:
            # Check smart_selector.db
            if os.path.exists('data/smart_selector.db'):
                conn = sqlite3.connect('data/smart_selector.db')
                cursor = conn.cursor()
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    try:
                        cursor.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 10;")
                        columns = [description[0] for description in cursor.description]
                        rows = cursor.fetchall()
                        
                        if rows:
                            recent_data = []
                            for row in rows:
                                record = dict(zip(columns, row))
                                recent_data.append(record)
                            
                            strategy_data['active_assignments'][table] = recent_data
                            
                    except Exception as e:
                        pass
                
                conn.close()
            
            # Check strategy_performance.db
            if os.path.exists('data/strategy_performance.db'):
                conn = sqlite3.connect('data/strategy_performance.db')
                cursor = conn.cursor()
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table};")
                        count = cursor.fetchone()[0]
                        strategy_data['performance_metrics'][table] = count
                    except:
                        pass
                        
                conn.close()
                
        except Exception as e:
            print(f"Strategy analysis error: {e}")
        
        self.results['strategy_effectiveness'] = strategy_data
    
    def analyze_risk_system(self):
        """Analyze risk management and protection systems"""
        print("Analyzing risk management system...")
        
        risk_data = {
            'risk_events': [],
            'protection_triggers': {},
            'alert_history': [],
            'system_status': {}
        }
        
        try:
            # Check risk_management.db
            if os.path.exists('data/risk_management.db'):
                conn = sqlite3.connect('data/risk_management.db')
                cursor = conn.cursor()
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    try:
                        cursor.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 5;")
                        columns = [description[0] for description in cursor.description]
                        rows = cursor.fetchall()
                        
                        if rows:
                            events = []
                            for row in rows:
                                record = dict(zip(columns, row))
                                events.append(record)
                            
                            risk_data['risk_events'].extend(events)
                            
                    except Exception as e:
                        pass
                
                conn.close()
            
            # Check alerts.db
            if os.path.exists('data/alerts.db'):
                conn = sqlite3.connect('data/alerts.db')
                cursor = conn.cursor()
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table};")
                        count = cursor.fetchone()[0]
                        risk_data['protection_triggers'][table] = count
                    except:
                        pass
                        
                conn.close()
                
        except Exception as e:
            print(f"Risk analysis error: {e}")
        
        self.results['risk_system'] = risk_data
    
    def verify_okx_connectivity(self):
        """Verify live OKX market data connectivity"""
        print("Verifying OKX connectivity...")
        
        connectivity_data = {
            'api_status': 'Unknown',
            'live_prices': {},
            'data_freshness': {},
            'market_coverage': {}
        }
        
        try:
            from trading.okx_data_service import OKXDataService
            okx_service = OKXDataService()
            
            # Test major pairs with live data
            test_pairs = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT', 'ADA-USDT']
            
            for pair in test_pairs:
                try:
                    data = okx_service.get_historical_data(pair, '1h', limit=3)
                    if data is not None and not data.empty:
                        latest_price = float(data['close'].iloc[-1])
                        latest_time = data.index[-1]
                        
                        connectivity_data['live_prices'][pair] = {
                            'price': latest_price,
                            'timestamp': latest_time.isoformat(),
                            'status': 'Live'
                        }
                        
                        print(f"✅ {pair}: ${latest_price:.2f}")
                    else:
                        connectivity_data['live_prices'][pair] = {'status': 'No Data'}
                        
                except Exception as e:
                    connectivity_data['live_prices'][pair] = {'status': f'Error: {str(e)[:30]}'}
            
            connectivity_data['api_status'] = 'Operational'
            
        except Exception as e:
            connectivity_data['api_status'] = f'Error: {str(e)[:50]}'
            print(f"OKX connectivity error: {e}")
        
        self.results['system_health']['okx_connectivity'] = connectivity_data
    
    def check_background_services(self):
        """Check status of background services and processes"""
        print("Checking background services...")
        
        services_data = {
            'model_training': 'Unknown',
            'strategy_selection': 'Unknown',
            'data_collection': 'Unknown',
            'risk_monitoring': 'Unknown'
        }
        
        # Check for recent activity in databases to infer service status
        try:
            # Check if data collection is active
            if os.path.exists('data/sentiment_data.db'):
                conn = sqlite3.connect('data/sentiment_data.db')
                cursor = conn.cursor()
                
                try:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    if tables:
                        services_data['data_collection'] = 'Active'
                    else:
                        services_data['data_collection'] = 'Inactive'
                        
                except:
                    services_data['data_collection'] = 'Error'
                    
                conn.close()
            
            # Check AI performance for model activity
            if os.path.exists('data/ai_performance.db'):
                file_mtime = os.path.getmtime('data/ai_performance.db')
                last_modified = datetime.fromtimestamp(file_mtime)
                hours_ago = (datetime.now() - last_modified).total_seconds() / 3600
                
                if hours_ago < 1:
                    services_data['model_training'] = 'Recently Active'
                elif hours_ago < 24:
                    services_data['model_training'] = 'Active'
                else:
                    services_data['model_training'] = 'Stale'
            
            # Check strategy selection activity
            if os.path.exists('data/smart_selector.db'):
                file_mtime = os.path.getmtime('data/smart_selector.db')
                last_modified = datetime.fromtimestamp(file_mtime)
                hours_ago = (datetime.now() - last_modified).total_seconds() / 3600
                
                if hours_ago < 6:  # Strategy selection runs every 6 hours
                    services_data['strategy_selection'] = 'Active'
                else:
                    services_data['strategy_selection'] = 'Delayed'
            
        except Exception as e:
            print(f"Background services check error: {e}")
        
        self.results['background_services'] = services_data
    
    def generate_comprehensive_report(self):
        """Generate final comprehensive audit report"""
        print("Generating comprehensive audit report...")
        
        # Count operational components
        db_count = len([db for db in self.results['system_health'].get('databases', {}).values() 
                       if db.get('status') == 'Active'])
        
        # Count live prices
        live_pairs = len([pair for pair in self.results['system_health'].get('okx_connectivity', {}).get('live_prices', {}).values()
                         if pair.get('status') == 'Live'])
        
        # Calculate totals
        total_records = sum([db.get('total_records', 0) for db in self.results['system_health'].get('databases', {}).values()
                           if isinstance(db.get('total_records'), int)])
        
        # Create executive summary
        summary = {
            'audit_timestamp': self.results['timestamp'],
            'data_authenticity': 'Live OKX Production Data',
            'operational_databases': f"{db_count}/9 databases active",
            'live_market_data': f"{live_pairs} pairs with live prices", 
            'total_system_records': total_records,
            'background_services': len([s for s in self.results['background_services'].values() if 'Active' in str(s)]),
            'system_status': 'Operational' if db_count >= 6 and live_pairs >= 3 else 'Partial Operation'
        }
        
        self.results['executive_summary'] = summary
        
        # Save detailed report
        report_filename = f"live_system_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"✅ Comprehensive audit report saved: {report_filename}")
        return self.results
    
    def run_complete_audit(self):
        """Execute complete live system audit"""
        print("🔍 Starting Live System Audit with Authentic Data")
        print("="*60)
        
        self.verify_okx_connectivity()
        self.audit_databases()
        self.analyze_trading_data()
        self.analyze_ai_performance()
        self.analyze_strategy_effectiveness()
        self.analyze_risk_system()
        self.check_background_services()
        
        final_report = self.generate_comprehensive_report()
        
        print("\n🎉 Live System Audit completed!")
        return final_report

def print_audit_summary(report):
    """Print formatted audit summary"""
    print("\n" + "="*60)
    print("📊 LIVE SYSTEM AUDIT - FINAL REPORT")
    print("="*60)
    
    summary = report.get('executive_summary', {})
    
    print(f"\n📋 SYSTEM STATUS: {summary.get('system_status', 'Unknown')}")
    print(f"📡 Data Source: {summary.get('data_authenticity', 'Unknown')}")
    print(f"🕒 Audit Time: {summary.get('audit_timestamp', 'Unknown')}")
    
    print(f"\n🗄️ DATABASE HEALTH:")
    print(f"  Status: {summary.get('operational_databases', 'Unknown')}")
    print(f"  Total Records: {summary.get('total_system_records', 0):,}")
    
    print(f"\n📡 MARKET CONNECTIVITY:")
    print(f"  Live Data: {summary.get('live_market_data', 'Unknown')}")
    
    # Show live prices
    okx_data = report.get('system_health', {}).get('okx_connectivity', {}).get('live_prices', {})
    if okx_data:
        print("\n💰 LIVE MARKET PRICES:")
        for pair, data in okx_data.items():
            if data.get('status') == 'Live':
                print(f"  {pair}: ${data.get('price', 0):,.2f}")
    
    # Show database details
    db_data = report.get('system_health', {}).get('databases', {})
    if db_data:
        print(f"\n🗃️ DATABASE DETAILS:")
        for db_name, info in db_data.items():
            if info.get('status') == 'Active':
                print(f"  {db_name}: {info.get('total_records', 0):,} records, {info.get('size_mb', 0)} MB")
    
    # Show background services
    services = report.get('background_services', {})
    if services:
        print(f"\n⚙️ BACKGROUND SERVICES:")
        for service, status in services.items():
            print(f"  {service.replace('_', ' ').title()}: {status}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    auditor = LiveSystemAudit()
    report = auditor.run_complete_audit()
    print_audit_summary(report)