
"""
System Startup Validation
Validates all system components on startup
"""
import os
import sqlite3
from datetime import datetime

def validate_system_startup():
    """Comprehensive system validation"""
    validation_results = {
        'timestamp': datetime.now().isoformat(),
        'overall_status': 'UNKNOWN',
        'checks': {},
        'warnings': [],
        'errors': []
    }
    
    # Check 1: Database accessibility
    databases = ['enhanced_trading.db', 'autonomous_trading.db', 'enhanced_ui.db']
    accessible_dbs = 0
    
    for db in databases:
        try:
            if os.path.exists(db):
                conn = sqlite3.connect(db)
                conn.execute('SELECT 1')
                conn.close()
                accessible_dbs += 1
                validation_results['checks'][f'db_{db}'] = 'PASS'
            else:
                validation_results['checks'][f'db_{db}'] = 'MISSING'
                validation_results['warnings'].append(f'Database {db} not found')
        except Exception as e:
            validation_results['checks'][f'db_{db}'] = 'FAIL'
            validation_results['errors'].append(f'Database {db} error: {e}')
    
    # Check 2: Required files
    required_files = [
        'fixed_indicators.py',
        'fallback_indicators.py',
        'verified_symbols.txt',
        'gpt_disabled.flag'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            validation_results['checks'][f'file_{file}'] = 'PASS'
        else:
            validation_results['checks'][f'file_{file}'] = 'MISSING'
            validation_results['warnings'].append(f'Required file {file} not found')
    
    # Check 3: System flags
    try:
        if os.path.exists('gpt_disabled.flag'):
            validation_results['checks']['gpt_disabled'] = 'PASS'
        else:
            validation_results['warnings'].append('GPT analysis not properly disabled')
    except:
        validation_results['errors'].append('Cannot check GPT status')
    
    # Determine overall status
    if len(validation_results['errors']) == 0:
        if len(validation_results['warnings']) == 0:
            validation_results['overall_status'] = 'HEALTHY'
        else:
            validation_results['overall_status'] = 'DEGRADED'
    else:
        validation_results['overall_status'] = 'CRITICAL'
    
    return validation_results

def print_validation_report():
    """Print validation report"""
    results = validate_system_startup()
    
    print("\n" + "="*50)
    print("SYSTEM STARTUP VALIDATION")
    print("="*50)
    print(f"Overall Status: {results['overall_status']}")
    print(f"Timestamp: {results['timestamp']}")
    
    print("\nCOMPONENT CHECKS:")
    for check, status in results['checks'].items():
        status_symbol = "✓" if status == "PASS" else "✗" if status == "FAIL" else "⚠"
        print(f"  {status_symbol} {check}: {status}")
    
    if results['warnings']:
        print("\nWARNINGS:")
        for warning in results['warnings']:
            print(f"  ⚠ {warning}")
    
    if results['errors']:
        print("\nERRORS:")
        for error in results['errors']:
            print(f"  ✗ {error}")
    
    print("="*50)
    return results['overall_status']

if __name__ == "__main__":
    status = print_validation_report()
    exit(0 if status in ['HEALTHY', 'DEGRADED'] else 1)
