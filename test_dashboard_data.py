#!/usr/bin/env python3
"""
Test Dashboard Data Loading - Diagnose OKX Connection Issues
"""
import sys
import traceback
from okx_data_validator import OKXDataValidator
from elite_dashboard_fixed import CleanEliteDashboard

def test_okx_validator():
    """Test OKX validator directly"""
    print("Testing OKX Data Validator...")
    try:
        validator = OKXDataValidator()
        print("✅ OKX Validator initialized")
        
        portfolio = validator.get_authentic_portfolio()
        print(f"✅ Portfolio data: Balance ${portfolio['balance']:.2f}")
        
        signals = validator.get_authentic_signals()
        print(f"✅ Signals data: {len(signals)} signals")
        
        return True
    except Exception as e:
        print(f"❌ OKX Validator error: {e}")
        traceback.print_exc()
        return False

def test_dashboard_methods():
    """Test dashboard methods individually"""
    print("\nTesting Dashboard Methods...")
    try:
        dashboard = CleanEliteDashboard()
        print("✅ Dashboard initialized")
        
        # Test portfolio data
        portfolio = dashboard.get_portfolio_data()
        print(f"✅ Portfolio: ${portfolio['total_balance']:.2f}")
        
        # Test signals
        signals = dashboard.get_trading_signals()
        print(f"✅ Signals: {len(signals)} signals")
        
        # Test performance
        performance = dashboard.get_performance_metrics()
        print(f"✅ Performance: {performance['total_trades']} trades")
        
        return True
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔍 Diagnosing Dashboard Data Loading...")
    
    # Test OKX validator
    validator_ok = test_okx_validator()
    
    # Test dashboard methods
    dashboard_ok = test_dashboard_methods()
    
    if validator_ok and dashboard_ok:
        print("\n✅ All components working - Issue may be in Flask routing")
    else:
        print("\n❌ Component failures detected")
        sys.exit(1)