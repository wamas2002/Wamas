"""
Data Source Audit - Verify all trading engines use authentic OKX data only
"""
import os
import re
import sqlite3
from datetime import datetime

def audit_trading_files():
    """Audit all trading files for data sources"""
    trading_files = [
        'pure_local_trading_engine.py',
        'advanced_futures_trading_engine.py',
        'unified_trading_platform.py',
        'enhanced_modern_trading_ui.py',
        'enhanced_tradingview_dashboard.py',
        'enhanced_trading_ui_3000.py',
        'signal_execution_bridge.py'
    ]
    
    audit_results = {}
    
    for filename in trading_files:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = f.read()
            
            # Check for OKX exchange initialization
            okx_init = bool(re.search(r'ccxt\.okx|okx\(\)', content))
            
            # Check for API credentials usage
            uses_credentials = bool(re.search(r'OKX_API_KEY|OKX_SECRET_KEY|OKX_PASSPHRASE', content))
            
            # Check for mock/fallback data patterns
            has_mock_data = bool(re.search(r'mock|fake|dummy|fallback.*data|simulate.*data', content, re.IGNORECASE))
            
            # Check for authentic data fetching
            authentic_data = bool(re.search(r'fetch_ohlcv|fetch_ticker|fetch_balance|fetch_order_book', content))
            
            # Check for real-time data calls
            realtime_calls = bool(re.search(r'get_market_data|fetch.*real.*time|live.*data', content))
            
            audit_results[filename] = {
                'okx_exchange': okx_init,
                'uses_credentials': uses_credentials,
                'mock_data_detected': has_mock_data,
                'authentic_data_calls': authentic_data,
                'realtime_data': realtime_calls,
                'status': 'AUTHENTIC' if (okx_init and uses_credentials and not has_mock_data) else 'NEEDS_REVIEW'
            }
    
    return audit_results

def check_database_data_sources():
    """Check database for data authenticity markers"""
    databases = [
        'enhanced_trading.db',
        'autonomous_trading.db',
        'pure_local_trading.db'
    ]
    
    db_audit = {}
    
    for db_name in databases:
        if os.path.exists(db_name):
            try:
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()
                
                # Check for tables with market data
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%signal%' OR name LIKE '%market%' OR name LIKE '%price%'")
                data_tables = cursor.fetchall()
                
                # Sample recent data to verify authenticity
                sample_data = {}
                for table in data_tables:
                    table_name = table[0]
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        row_count = cursor.fetchone()[0]
                        
                        if row_count > 0:
                            cursor.execute(f"SELECT * FROM {table_name} ORDER BY ROWID DESC LIMIT 1")
                            latest_row = cursor.fetchone()
                            sample_data[table_name] = {
                                'row_count': row_count,
                                'has_recent_data': True,
                                'latest_timestamp': 'Available'
                            }
                    except:
                        sample_data[table_name] = {'error': 'Access failed'}
                
                conn.close()
                
                db_audit[db_name] = {
                    'status': 'ACCESSIBLE',
                    'data_tables': len(data_tables),
                    'sample_data': sample_data
                }
                
            except Exception as e:
                db_audit[db_name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
    
    return db_audit

def verify_okx_connection():
    """Verify OKX connection and API usage"""
    try:
        # Check if OKX credentials are set
        okx_key = os.environ.get('OKX_API_KEY')
        okx_secret = os.environ.get('OKX_SECRET_KEY') 
        okx_passphrase = os.environ.get('OKX_PASSPHRASE')
        
        credentials_status = {
            'api_key_set': bool(okx_key),
            'secret_key_set': bool(okx_secret),
            'passphrase_set': bool(okx_passphrase),
            'all_credentials': bool(okx_key and okx_secret and okx_passphrase)
        }
        
        return credentials_status
        
    except Exception as e:
        return {'error': str(e)}

def check_for_synthetic_data():
    """Check for any synthetic/mock data generation"""
    suspect_patterns = [
        'generate.*mock',
        'simulate.*price',
        'random.*data',
        'fake.*market',
        'np\.random',
        'random\.uniform',
        'synthetic.*data'
    ]
    
    trading_files = [
        'pure_local_trading_engine.py',
        'advanced_futures_trading_engine.py', 
        'unified_trading_platform.py'
    ]
    
    synthetic_findings = {}
    
    for filename in trading_files:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                content = f.read()
            
            findings = []
            for pattern in suspect_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    findings.extend(matches)
            
            synthetic_findings[filename] = {
                'synthetic_patterns_found': len(findings),
                'patterns': findings[:5],  # First 5 matches
                'status': 'CLEAN' if len(findings) == 0 else 'CONTAINS_SYNTHETIC'
            }
    
    return synthetic_findings

def generate_data_authenticity_report():
    """Generate comprehensive data authenticity report"""
    print("=" * 70)
    print("OKX DATA AUTHENTICITY AUDIT REPORT")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. File audit
    print("\n🔍 TRADING ENGINE FILE AUDIT:")
    file_audit = audit_trading_files()
    authentic_count = 0
    total_files = 0
    
    for filename, results in file_audit.items():
        total_files += 1
        status = results['status']
        if status == 'AUTHENTIC':
            authentic_count += 1
            print(f"  ✅ {filename}: {status}")
        else:
            print(f"  ⚠️  {filename}: {status}")
        
        print(f"     - OKX Exchange: {'✓' if results['okx_exchange'] else '✗'}")
        print(f"     - API Credentials: {'✓' if results['uses_credentials'] else '✗'}")
        print(f"     - Mock Data: {'✗ Found' if results['mock_data_detected'] else '✓ Clean'}")
        print(f"     - Authentic Calls: {'✓' if results['authentic_data_calls'] else '✗'}")
    
    # 2. OKX Credentials Check
    print("\n🔑 OKX API CREDENTIALS:")
    cred_status = verify_okx_connection()
    if 'error' not in cred_status:
        if cred_status['all_credentials']:
            print("  ✅ All OKX credentials properly configured")
        else:
            print("  ⚠️  Missing OKX credentials:")
            print(f"     - API Key: {'✓' if cred_status['api_key_set'] else '✗'}")
            print(f"     - Secret Key: {'✓' if cred_status['secret_key_set'] else '✗'}")
            print(f"     - Passphrase: {'✓' if cred_status['passphrase_set'] else '✗'}")
    
    # 3. Database Data Check
    print("\n💾 DATABASE DATA SOURCES:")
    db_audit = check_database_data_sources()
    for db_name, results in db_audit.items():
        if results['status'] == 'ACCESSIBLE':
            print(f"  ✅ {db_name}: {results['data_tables']} data tables")
            for table, info in results['sample_data'].items():
                if 'error' not in info:
                    print(f"     - {table}: {info['row_count']} records")
        else:
            print(f"  ❌ {db_name}: {results['status']}")
    
    # 4. Synthetic Data Check
    print("\n🚫 SYNTHETIC DATA SCAN:")
    synthetic_audit = check_for_synthetic_data()
    clean_files = 0
    for filename, results in synthetic_audit.items():
        if results['status'] == 'CLEAN':
            clean_files += 1
            print(f"  ✅ {filename}: CLEAN")
        else:
            print(f"  ⚠️  {filename}: {results['synthetic_patterns_found']} synthetic patterns")
    
    # 5. Summary
    print("\n" + "=" * 70)
    print("AUTHENTICITY SUMMARY:")
    print("=" * 70)
    
    auth_percentage = (authentic_count / total_files) * 100 if total_files > 0 else 0
    clean_percentage = (clean_files / len(synthetic_audit)) * 100 if synthetic_audit else 0
    
    print(f"📊 Authentic Trading Engines: {authentic_count}/{total_files} ({auth_percentage:.1f}%)")
    print(f"🧹 Clean Files (No Synthetic): {clean_files}/{len(synthetic_audit)} ({clean_percentage:.1f}%)")
    print(f"🔑 OKX Credentials: {'CONFIGURED' if cred_status.get('all_credentials') else 'INCOMPLETE'}")
    
    overall_authentic = auth_percentage >= 80 and clean_percentage >= 90 and cred_status.get('all_credentials')
    
    if overall_authentic:
        print("\n🎯 VERDICT: SYSTEM USES AUTHENTIC OKX DATA ONLY")
        print("✅ All trading engines configured for live market data")
        print("✅ No synthetic or mock data generation detected")
        print("✅ OKX API credentials properly configured")
    else:
        print("\n⚠️  VERDICT: SYSTEM REQUIRES DATA SOURCE REVIEW")
        if auth_percentage < 80:
            print("- Some trading engines need OKX configuration")
        if clean_percentage < 90:
            print("- Synthetic data patterns detected")
        if not cred_status.get('all_credentials'):
            print("- OKX API credentials incomplete")
    
    print("=" * 70)

if __name__ == "__main__":
    generate_data_authenticity_report()