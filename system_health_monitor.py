
"""
System Health Monitor
Monitors and reports system status
"""
import sqlite3
import os
from datetime import datetime

def check_system_health():
    health_report = {
        'timestamp': datetime.now().isoformat(),
        'status': 'HEALTHY',
        'issues': [],
        'recommendations': []
    }
    
    # Check for GPT disable flag
    if os.path.exists('gpt_disabled.flag'):
        health_report['issues'].append('GPT analysis disabled due to quota')
        health_report['recommendations'].append('Consider local analysis boost')
    
    # Check database accessibility
    databases = ['enhanced_trading.db', 'autonomous_trading.db', 'enhanced_ui.db']
    accessible_dbs = 0
    
    for db in databases:
        if os.path.exists(db):
            try:
                conn = sqlite3.connect(db)
                conn.close()
                accessible_dbs += 1
            except:
                health_report['issues'].append(f'Database {db} inaccessible')
    
    if accessible_dbs == len(databases):
        health_report['database_status'] = 'ALL_ACCESSIBLE'
    else:
        health_report['database_status'] = f'{accessible_dbs}/{len(databases)}_ACCESSIBLE'
    
    # Overall status
    if len(health_report['issues']) == 0:
        health_report['status'] = 'OPTIMAL'
    elif len(health_report['issues']) <= 2:
        health_report['status'] = 'DEGRADED'
    else:
        health_report['status'] = 'CRITICAL'
    
    return health_report

if __name__ == "__main__":
    report = check_system_health()
    print(f"System Status: {report['status']}")
    if report['issues']:
        print("Issues:")
        for issue in report['issues']:
            print(f"  - {issue}")
