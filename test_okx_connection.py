#!/usr/bin/env python3
"""
Test OKX Data Connection
"""

import sys
import os
sys.path.append('/home/runner/workspace')

from okx_data_validator import OKXDataValidator
import json

def test_okx_connection():
    """Test OKX data validator connection"""
    print("Testing OKX Data Connection...")
    
    validator = OKXDataValidator()
    
    # Test 1: Basic connection
    if validator.validate_connection():
        print("✅ OKX connection validated")
    else:
        print("❌ OKX connection failed")
        return False
    
    # Test 2: Portfolio data
    try:
        portfolio = validator.get_authentic_portfolio()
        print(f"✅ Portfolio: ${portfolio['balance']:.2f}")
        print(f"✅ Positions: {portfolio['position_count']}")
        print(f"✅ P&L: ${portfolio['total_unrealized_pnl']:.2f}")
        
        # Test dashboard format
        dashboard_portfolio = validator.get_portfolio_data()
        print(f"✅ Dashboard format: ${dashboard_portfolio['total_balance']:.2f}")
        
    except Exception as e:
        print(f"❌ Portfolio error: {e}")
        return False
    
    # Test 3: Trading signals
    try:
        signals = validator.get_authentic_signals()
        print(f"✅ Generated {len(signals)} trading signals")
        
        # Test dashboard format
        dashboard_signals = validator.get_trading_signals()
        print(f"✅ Dashboard signals: {len(dashboard_signals['signals'])}")
        
    except Exception as e:
        print(f"❌ Signals error: {e}")
        return False
    
    # Test 4: Performance metrics
    try:
        performance = validator.get_performance_metrics()
        print(f"✅ Performance metrics: {performance['win_rate']}% win rate")
        
    except Exception as e:
        print(f"❌ Performance error: {e}")
        return False
    
    print("\n🎯 All OKX data connections working properly!")
    return True

if __name__ == "__main__":
    test_okx_connection()