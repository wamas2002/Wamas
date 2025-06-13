"""
Verify Signal Execution - Check if bridge detects high confidence signals
"""
import sqlite3
import os
from datetime import datetime, timedelta

def check_pure_local_signals():
    """Check current high confidence signals from Pure Local Trading Engine"""
    if not os.path.exists('pure_local_trading.db'):
        return 0, []
    
    conn = sqlite3.connect('pure_local_trading.db')
    cursor = conn.cursor()
    
    # Get recent high confidence BUY signals
    cutoff_time = (datetime.now() - timedelta(minutes=30)).isoformat()
    
    cursor.execute("""
        SELECT symbol, signal_type, confidence, timestamp
        FROM local_signals 
        WHERE signal_type = 'BUY' 
        AND confidence >= 75.0 
        AND timestamp > ?
        ORDER BY confidence DESC LIMIT 10
    """, (cutoff_time,))
    
    signals = cursor.fetchall()
    conn.close()
    
    return len(signals), signals

def test_bridge_detection():
    """Test if execution bridge can detect signals"""
    # Simulate bridge signal detection
    from signal_execution_bridge import SignalExecutionBridge
    
    try:
        bridge = SignalExecutionBridge()
        detected_signals = bridge.get_fresh_signals()
        return len(detected_signals), detected_signals
    except Exception as e:
        return 0, str(e)

def main():
    """Verify signal execution connection"""
    print("SIGNAL EXECUTION VERIFICATION")
    print("=" * 40)
    
    # Check Pure Local signals
    pure_count, pure_signals = check_pure_local_signals()
    print(f"Pure Local Engine signals: {pure_count}")
    
    if pure_signals:
        print("Top signals:")
        for symbol, signal_type, confidence, timestamp in pure_signals[:5]:
            print(f"  {symbol}: {signal_type} ({confidence:.1f}%)")
    
    # Test bridge detection
    bridge_count, bridge_result = test_bridge_detection()
    print(f"\nExecution Bridge detected: {bridge_count}")
    
    if isinstance(bridge_result, str):
        print(f"Bridge error: {bridge_result}")
    elif bridge_result:
        print("Detected signals:")
        for signal in bridge_result[:3]:
            symbol = signal.get('symbol', 'Unknown')
            action = signal.get('action', 'Unknown')
            confidence = signal.get('confidence', 0) * 100
            print(f"  {symbol}: {action} ({confidence:.1f}%)")
    
    print("\n" + "=" * 40)
    if pure_count > 0 and bridge_count > 0:
        print("EXECUTION READY - Bridge detecting signals")
    elif pure_count > 0 and bridge_count == 0:
        print("BRIDGE ISSUE - Signals available but not detected")
    else:
        print("WAITING - No high confidence signals available")

if __name__ == "__main__":
    main()