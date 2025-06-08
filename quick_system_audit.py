"""
Quick System Audit with Live OKX Data
Streamlined comprehensive analysis for immediate results
"""

import sqlite3
import pandas as pd
import os
import json
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def audit_okx_integration():
    """Test OKX API integration"""
    logger.info("Testing OKX integration...")
    
    try:
        from trading.okx_data_service import OKXDataService
        okx_service = OKXDataService()
        
        # Test key pairs
        test_pairs = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        results = {}
        
        for pair in test_pairs:
            try:
                price = okx_service.get_current_price(pair)
                data = okx_service.get_historical_data(pair, '1h', 5)
                
                results[pair] = {
                    'live_price': price,
                    'data_points': len(data) if not data.empty else 0,
                    'status': 'OPERATIONAL' if price > 0 and not data.empty else 'ISSUES'
                }
            except Exception as e:
                results[pair] = {'status': 'ERROR', 'error': str(e)[:50]}
        
        operational_count = sum(1 for r in results.values() if r.get('status') == 'OPERATIONAL')
        
        return {
            'pairs_tested': results,
            'operational': f"{operational_count}/{len(test_pairs)}",
            'overall_status': 'HEALTHY' if operational_count >= 2 else 'DEGRADED'
        }
        
    except Exception as e:
        return {'status': 'CRITICAL_ERROR', 'error': str(e)}

def audit_databases():
    """Check database health"""
    logger.info("Checking database health...")
    
    databases = [
        'data/ai_performance.db',
        'data/trading_data.db',
        'data/autoconfig.db',
        'data/smart_selector.db',
        'data/sentiment_data.db'
    ]
    
    db_status = {}
    
    for db_path in databases:
        db_name = os.path.basename(db_path)
        
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                total_records = 0
                for table in tables:
                    if table != 'sqlite_sequence':
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {table};")
                            count = cursor.fetchone()[0]
                            total_records += count
                        except:
                            pass
                
                db_status[db_name] = {
                    'tables': len(tables),
                    'records': total_records,
                    'size_mb': round(os.path.getsize(db_path) / (1024*1024), 2),
                    'status': 'HEALTHY'
                }
                
                conn.close()
                
            except Exception as e:
                db_status[db_name] = {'status': 'ERROR', 'error': str(e)[:50]}
        else:
            db_status[db_name] = {'status': 'NOT_FOUND'}
    
    healthy_dbs = sum(1 for db in db_status.values() if db.get('status') == 'HEALTHY')
    
    return {
        'databases': db_status,
        'healthy_count': f"{healthy_dbs}/{len(databases)}",
        'total_records': sum(db.get('records', 0) for db in db_status.values())
    }

def audit_ai_models():
    """Check AI model status"""
    logger.info("Checking AI model status...")
    
    model_status = {
        'performance_data': {},
        'model_files': 0,
        'recent_activity': False
    }
    
    # Check AI performance database
    if os.path.exists('data/ai_performance.db'):
        try:
            conn = sqlite3.connect('data/ai_performance.db')
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables[:3]:  # Check first 3 tables
                if table != 'sqlite_sequence':
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table};")
                        count = cursor.fetchone()[0]
                        model_status['performance_data'][table] = count
                        
                        if count > 0:
                            model_status['recent_activity'] = True
                    except:
                        pass
            
            conn.close()
            
        except Exception as e:
            model_status['error'] = str(e)[:50]
    
    # Check model files
    model_dirs = ['models', 'ai']
    for directory in model_dirs:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.endswith(('.pkl', '.joblib', '.h5', '.pt')):
                    model_status['model_files'] += 1
    
    return model_status

def audit_trading_performance():
    """Check trading performance"""
    logger.info("Checking trading performance...")
    
    performance = {
        'trades_72h': 0,
        'active_strategies': {},
        'risk_events': 0
    }
    
    # Check trading data
    if os.path.exists('data/trading_data.db'):
        try:
            conn = sqlite3.connect('data/trading_data.db')
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            cutoff_time = (datetime.now() - timedelta(hours=72)).isoformat()
            
            for table in tables:
                try:
                    cursor.execute(f"PRAGMA table_info({table});")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    if 'timestamp' in columns:
                        cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE timestamp > ?", (cutoff_time,))
                        count = cursor.fetchone()[0]
                        performance['trades_72h'] += count
                except:
                    pass
            
            conn.close()
            
        except Exception as e:
            performance['trading_data_error'] = str(e)[:50]
    
    # Check strategy assignments
    if os.path.exists('data/autoconfig.db'):
        try:
            conn = sqlite3.connect('data/autoconfig.db')
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table};")
                    count = cursor.fetchone()[0]
                    if count > 0:
                        performance['active_strategies'][table] = count
                except:
                    pass
            
            conn.close()
            
        except Exception as e:
            performance['strategy_error'] = str(e)[:50]
    
    return performance

def audit_ui_accessibility():
    """Check UI accessibility"""
    logger.info("Checking UI accessibility...")
    
    ui_status = {
        'streamlit_app': False,
        'main_files': {}
    }
    
    # Check if main files exist
    key_files = ['intellectia_app.py', 'app.py']
    for file in key_files:
        if os.path.exists(file):
            ui_status['main_files'][file] = 'EXISTS'
            
            try:
                with open(file, 'r') as f:
                    content = f.read()
                    if 'streamlit' in content and 'def main' in content:
                        ui_status['main_files'][file] = 'VALID'
            except:
                ui_status['main_files'][file] = 'READ_ERROR'
    
    # Test localhost accessibility
    try:
        import requests
        response = requests.get('http://localhost:5000', timeout=3)
        ui_status['streamlit_app'] = response.status_code == 200
    except:
        ui_status['streamlit_app'] = False
    
    return ui_status

def generate_diagnostic_report():
    """Generate comprehensive diagnostic report"""
    logger.info("Generating diagnostic report...")
    
    # Run all audits
    okx_audit = audit_okx_integration()
    db_audit = audit_databases()
    ai_audit = audit_ai_models()
    trading_audit = audit_trading_performance()
    ui_audit = audit_ui_accessibility()
    
    # Calculate overall health
    health_scores = []
    
    # OKX Integration (25%)
    if okx_audit.get('overall_status') == 'HEALTHY':
        health_scores.append(25)
    elif okx_audit.get('overall_status') == 'DEGRADED':
        health_scores.append(15)
    else:
        health_scores.append(5)
    
    # Database Health (25%)
    db_healthy = db_audit.get('healthy_count', '0/5').split('/')[0]
    db_score = (int(db_healthy) / 5) * 25
    health_scores.append(db_score)
    
    # AI Models (20%)
    if ai_audit.get('recent_activity') and ai_audit.get('model_files', 0) > 0:
        health_scores.append(20)
    elif ai_audit.get('model_files', 0) > 0:
        health_scores.append(10)
    else:
        health_scores.append(5)
    
    # Trading Performance (15%)
    if trading_audit.get('active_strategies'):
        health_scores.append(15)
    else:
        health_scores.append(5)
    
    # UI Accessibility (15%)
    if ui_audit.get('streamlit_app'):
        health_scores.append(15)
    elif ui_audit.get('main_files'):
        health_scores.append(8)
    else:
        health_scores.append(3)
    
    overall_score = sum(health_scores)
    
    if overall_score >= 90:
        status = 'EXCELLENT'
    elif overall_score >= 80:
        status = 'GOOD'
    elif overall_score >= 70:
        status = 'ACCEPTABLE'
    else:
        status = 'NEEDS_ATTENTION'
    
    # Compile final report
    report = {
        'timestamp': datetime.now().isoformat(),
        'overall_health': {
            'status': status,
            'score': round(overall_score, 1)
        },
        'okx_integration': okx_audit,
        'database_health': db_audit,
        'ai_models': ai_audit,
        'trading_performance': trading_audit,
        'ui_accessibility': ui_audit
    }
    
    return report

def print_summary_report(report):
    """Print formatted summary report"""
    print("\n" + "="*80)
    print("COMPREHENSIVE SYSTEM AUDIT - DIAGNOSTIC REPORT")
    print("="*80)
    
    overall = report['overall_health']
    print(f"\nOVERALL SYSTEM HEALTH: {overall['status']} ({overall['score']}/100)")
    
    # OKX Integration
    okx = report['okx_integration']
    print(f"\n✅ OKX MARKET DATA INTEGRATION:")
    print(f"  Status: {okx.get('overall_status', 'UNKNOWN')}")
    print(f"  Operational Pairs: {okx.get('operational', '0/0')}")
    
    if 'pairs_tested' in okx:
        for pair, data in okx['pairs_tested'].items():
            if data.get('status') == 'OPERATIONAL':
                print(f"  {pair}: ${data.get('live_price', 0):,.2f} (Live)")
    
    # Database Health
    db = report['database_health']
    print(f"\n📁 DATABASE INTEGRITY:")
    print(f"  Healthy Databases: {db.get('healthy_count', '0/0')}")
    print(f"  Total Records: {db.get('total_records', 0):,}")
    
    for db_name, info in db.get('databases', {}).items():
        if info.get('status') == 'HEALTHY':
            print(f"  {db_name}: {info.get('records', 0):,} records, {info.get('size_mb', 0)} MB")
    
    # AI Models
    ai = report['ai_models']
    print(f"\n🧠 AI MODELS & PERFORMANCE:")
    print(f"  Model Files: {ai.get('model_files', 0)}")
    print(f"  Recent Activity: {'Yes' if ai.get('recent_activity') else 'No'}")
    
    for table, count in ai.get('performance_data', {}).items():
        print(f"  {table}: {count} records")
    
    # Trading Performance
    trading = report['trading_performance']
    print(f"\n📊 TRADING PERFORMANCE (72h):")
    print(f"  Total Trades: {trading.get('trades_72h', 0)}")
    
    strategies = trading.get('active_strategies', {})
    if strategies:
        print(f"  Active Strategies: {len(strategies)} tables")
        for strategy, count in strategies.items():
            print(f"    {strategy}: {count} assignments")
    
    # UI Accessibility
    ui = report['ui_accessibility']
    print(f"\n🖥️ UI ACCESSIBILITY:")
    print(f"  Streamlit App: {'Accessible' if ui.get('streamlit_app') else 'Not Accessible'}")
    
    for file, status in ui.get('main_files', {}).items():
        print(f"  {file}: {status}")
    
    print(f"\n🕒 Audit completed: {report['timestamp']}")
    print("="*80)

if __name__ == "__main__":
    try:
        report = generate_diagnostic_report()
        print_summary_report(report)
        
        # Save report
        filename = f"system_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\nDetailed report saved: {filename}")
        
    except Exception as e:
        print(f"Audit error: {e}")
        logger.error(f"System audit failed: {e}")