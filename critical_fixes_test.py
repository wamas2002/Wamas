#!/usr/bin/env python3
"""
Critical Fixes Verification Test
Tests AutoConfig Engine strategy assignment and Risk Manager datetime serialization
"""
import sys
import traceback
from datetime import datetime
import pandas as pd

def test_autoconfig_engine():
    """Test AutoConfig Engine strategy assignment"""
    print("Testing AutoConfig Engine...")
    try:
        from strategies.autoconfig_engine import AutoConfigEngine
        
        engine = AutoConfigEngine()
        
        # Test strategy assignment for BTCUSDT
        strategy = engine.get_strategy_for_symbol("BTCUSDT")
        print(f"✅ AutoConfig Engine: BTCUSDT strategy = {strategy}")
        
        # Test multiple symbols
        test_symbols = ["ETHUSDT", "ADAUSDT", "BNBUSDT"]
        for symbol in test_symbols:
            strategy = engine.get_strategy_for_symbol(symbol)
            print(f"✅ AutoConfig Engine: {symbol} strategy = {strategy}")
        
        return True
        
    except Exception as e:
        print(f"❌ AutoConfig Engine failed: {e}")
        traceback.print_exc()
        return False

def test_risk_manager():
    """Test Risk Manager datetime serialization"""
    print("\nTesting Risk Manager...")
    try:
        from trading.advanced_risk_manager import AdvancedRiskManager
        
        risk_manager = AdvancedRiskManager()
        
        # Test creating a position with correct parameters
        # This should not raise JSON serialization errors
        position = risk_manager.create_position_risk(
            symbol='BTCUSDT',
            entry_price=50000.0,
            position_size=0.1,
            tp_levels=[0.04, 0.08, 0.12],  # 4%, 8%, 12% profit levels
            sl_percentage=0.02,  # 2% stop loss
            use_trailing_stop=True
        )
        print("✅ Risk Manager: Position created without datetime serialization errors")
        
        # Test risk metrics calculation (would involve datetime serialization)
        summary = risk_manager.get_portfolio_risk_summary()
        print(f"✅ Risk Manager: Portfolio summary generated - {len(summary)} positions")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk Manager failed: {e}")
        traceback.print_exc()
        return False

def test_okx_data_integration():
    """Test OKX data integration with components"""
    print("\nTesting OKX Data Integration...")
    try:
        from trading.okx_data_service import OKXDataService
        
        okx_service = OKXDataService()
        
        # Test live data retrieval
        data = okx_service.get_historical_data("BTCUSDT", "1h", limit=10)
        
        if not data.empty:
            print(f"✅ OKX Integration: Retrieved {len(data)} records for BTCUSDT")
            print(f"✅ OKX Integration: Latest price = ${data['close'].iloc[-1]:,.2f}")
            return True
        else:
            print("❌ OKX Integration: No data returned")
            return False
            
    except Exception as e:
        print(f"❌ OKX Integration failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all critical fixes tests"""
    print("🔧 CRITICAL FIXES VERIFICATION TEST")
    print("=" * 50)
    
    tests = [
        test_autoconfig_engine,
        test_risk_manager, 
        test_okx_data_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"RESULTS: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL CRITICAL FIXES VERIFIED - SYSTEM READY")
        return True
    else:
        print("⚠️  SOME ISSUES REMAIN - NEEDS ATTENTION")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)