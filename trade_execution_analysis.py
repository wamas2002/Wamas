"""
Trade Execution Analysis
Investigate why high confidence signals are not executing trades
"""
import sqlite3
import os
import ccxt
from datetime import datetime, timedelta

def check_signal_execution_bridge():
    """Check the signal execution bridge configuration"""
    try:
        # Check if signal execution bridge is configured
        with open('signal_execution_bridge.py', 'r') as f:
            content = f.read()
        
        # Check for execution threshold
        execution_threshold = None
        if 'MIN_CONFIDENCE_FOR_EXECUTION' in content:
            import re
            match = re.search(r'MIN_CONFIDENCE_FOR_EXECUTION\s*=\s*(\d+\.?\d*)', content)
            if match:
                execution_threshold = float(match.group(1))
        
        # Check for auto-execution enabled
        auto_execution = 'auto_execute' in content and 'True' in content
        
        return {
            'bridge_exists': True,
            'execution_threshold': execution_threshold,
            'auto_execution_enabled': auto_execution,
            'status': 'CONFIGURED'
        }
    except FileNotFoundError:
        return {'bridge_exists': False, 'status': 'MISSING'}
    except Exception as e:
        return {'bridge_exists': True, 'error': str(e), 'status': 'ERROR'}

def analyze_recent_signals():
    """Analyze recent high confidence signals"""
    databases = ['enhanced_trading.db', 'pure_local_trading.db', 'autonomous_trading.db']
    
    high_confidence_signals = []
    
    for db_name in databases:
        if os.path.exists(db_name):
            try:
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()
                
                # Get table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%signal%'")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    try:
                        # Get recent high confidence signals
                        cursor.execute(f"""
                            SELECT * FROM {table} 
                            WHERE confidence >= 75.0 
                            ORDER BY ROWID DESC LIMIT 10
                        """)
                        
                        columns = [description[0] for description in cursor.description]
                        rows = cursor.fetchall()
                        
                        for row in rows:
                            signal_data = dict(zip(columns, row))
                            signal_data['database'] = db_name
                            signal_data['table'] = table
                            high_confidence_signals.append(signal_data)
                            
                    except sqlite3.OperationalError:
                        continue
                
                conn.close()
            except Exception as e:
                continue
    
    return high_confidence_signals

def check_okx_trading_permissions():
    """Check OKX account trading permissions"""
    try:
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'),
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False,
            'enableRateLimit': True
        })
        
        # Check account info
        account_info = exchange.fetch_balance()
        
        # Check if trading is enabled
        permissions = {
            'account_accessible': True,
            'has_balances': len(account_info['total']) > 0,
            'trading_enabled': True,  # Will be False if trading is restricted
            'total_balance': sum([float(v) for v in account_info['total'].values() if v])
        }
        
        return permissions
        
    except ccxt.AuthenticationError:
        return {'error': 'Authentication failed', 'trading_enabled': False}
    except ccxt.PermissionDenied:
        return {'error': 'Trading permissions denied', 'trading_enabled': False}
    except Exception as e:
        return {'error': str(e), 'trading_enabled': False}

def check_execution_logs():
    """Check for trade execution attempts in logs"""
    execution_attempts = []
    
    # Check database for execution records
    databases = ['enhanced_trading.db', 'autonomous_trading.db']
    
    for db_name in databases:
        if os.path.exists(db_name):
            try:
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()
                
                # Look for execution-related tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND (name LIKE '%trade%' OR name LIKE '%execution%' OR name LIKE '%order%')")
                execution_tables = [row[0] for row in cursor.fetchall()]
                
                for table in execution_tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        execution_attempts.append({
                            'database': db_name,
                            'table': table,
                            'records': count
                        })
                    except:
                        continue
                
                conn.close()
            except:
                continue
    
    return execution_attempts

def identify_execution_barriers():
    """Identify barriers preventing trade execution"""
    barriers = []
    
    # Check signal execution bridge
    bridge_status = check_signal_execution_bridge()
    if not bridge_status['bridge_exists']:
        barriers.append("Signal execution bridge not found")
    elif bridge_status.get('execution_threshold', 0) > 85:
        barriers.append(f"Execution threshold too high: {bridge_status['execution_threshold']}%")
    elif not bridge_status.get('auto_execution_enabled', False):
        barriers.append("Auto-execution disabled in bridge configuration")
    
    # Check OKX permissions
    okx_permissions = check_okx_trading_permissions()
    if not okx_permissions.get('trading_enabled', False):
        barriers.append(f"OKX trading permissions issue: {okx_permissions.get('error', 'Unknown')}")
    elif okx_permissions.get('total_balance', 0) < 10:
        barriers.append(f"Insufficient balance for trading: ${okx_permissions.get('total_balance', 0):.2f}")
    
    # Check for manual approval requirements
    if len(barriers) == 0:
        barriers.append("Manual approval may be required for trade execution")
    
    return barriers

def generate_execution_analysis_report():
    """Generate comprehensive execution analysis"""
    print("=" * 70)
    print("TRADE EXECUTION ANALYSIS REPORT")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. High Confidence Signals
    print("\n📊 HIGH CONFIDENCE SIGNALS ANALYSIS:")
    signals = analyze_recent_signals()
    
    if signals:
        print(f"  Found {len(signals)} high confidence signals (≥75%)")
        
        # Group by confidence ranges
        confidence_ranges = {
            '95-100%': [s for s in signals if s.get('confidence', 0) >= 95],
            '85-94%': [s for s in signals if 85 <= s.get('confidence', 0) < 95],
            '75-84%': [s for s in signals if 75 <= s.get('confidence', 0) < 85]
        }
        
        for range_name, range_signals in confidence_ranges.items():
            if range_signals:
                print(f"  {range_name}: {len(range_signals)} signals")
                for signal in range_signals[:3]:  # Show first 3
                    symbol = signal.get('symbol', 'Unknown')
                    confidence = signal.get('confidence', 0)
                    action = signal.get('action', signal.get('signal_type', 'Unknown'))
                    print(f"    - {symbol}: {action} ({confidence:.1f}%)")
    else:
        print("  No high confidence signals found in recent data")
    
    # 2. Signal Execution Bridge Status
    print("\n🔗 SIGNAL EXECUTION BRIDGE:")
    bridge_status = check_signal_execution_bridge()
    
    if bridge_status['bridge_exists']:
        print("  ✅ Signal execution bridge exists")
        threshold = bridge_status.get('execution_threshold')
        if threshold:
            print(f"  📏 Execution threshold: {threshold}%")
        else:
            print("  ⚠️  Execution threshold not configured")
        
        auto_exec = bridge_status.get('auto_execution_enabled', False)
        print(f"  🤖 Auto-execution: {'ENABLED' if auto_exec else 'DISABLED'}")
    else:
        print("  ❌ Signal execution bridge not found")
    
    # 3. OKX Trading Permissions
    print("\n🔑 OKX TRADING PERMISSIONS:")
    permissions = check_okx_trading_permissions()
    
    if 'error' in permissions:
        print(f"  ❌ {permissions['error']}")
    else:
        print(f"  ✅ Account accessible: {permissions.get('account_accessible', False)}")
        print(f"  💰 Has balances: {permissions.get('has_balances', False)}")
        print(f"  📈 Trading enabled: {permissions.get('trading_enabled', False)}")
        balance = permissions.get('total_balance', 0)
        print(f"  💵 Total balance: ${balance:.2f}")
    
    # 4. Execution Attempts
    print("\n📋 EXECUTION HISTORY:")
    execution_logs = check_execution_logs()
    
    if execution_logs:
        total_records = sum(log['records'] for log in execution_logs)
        print(f"  📊 Total execution records: {total_records}")
        for log in execution_logs:
            print(f"    - {log['table']}: {log['records']} records")
    else:
        print("  ⚠️  No execution history found")
    
    # 5. Identified Barriers
    print("\n🚧 EXECUTION BARRIERS:")
    barriers = identify_execution_barriers()
    
    for i, barrier in enumerate(barriers, 1):
        print(f"  {i}. {barrier}")
    
    # 6. Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    if any('threshold too high' in barrier for barrier in barriers):
        print("  • Lower execution confidence threshold to 70-75%")
    
    if any('Auto-execution disabled' in barrier for barrier in barriers):
        print("  • Enable auto-execution in signal execution bridge")
    
    if any('permissions' in barrier.lower() for barrier in barriers):
        print("  • Verify OKX API key has trading permissions")
        print("  • Check if account is in paper trading mode")
    
    if any('balance' in barrier.lower() for barrier in barriers):
        print("  • Fund OKX account with minimum trading balance")
    
    if not barriers or all('manual approval' in barrier.lower() for barrier in barriers):
        print("  • System may be configured for manual trade approval")
        print("  • Consider enabling automatic execution for high confidence signals")
    
    # 7. Solution Summary
    print("\n" + "=" * 70)
    print("EXECUTION STATUS SUMMARY")
    print("=" * 70)
    
    high_conf_count = len([s for s in signals if s.get('confidence', 0) >= 80])
    
    if high_conf_count > 0:
        print(f"🎯 {high_conf_count} signals above 80% confidence ready for execution")
        
        if len(barriers) > 0:
            print("⚠️  Execution blocked by configuration barriers")
            print("🔧 Fix required: Update execution bridge settings")
        else:
            print("✅ No technical barriers detected")
            print("🤔 Manual approval system may be active")
    else:
        print("📊 Monitoring for higher confidence signals")
    
    print("=" * 70)

if __name__ == "__main__":
    generate_execution_analysis_report()