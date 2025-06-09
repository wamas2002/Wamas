"""
Minimal test to reproduce the 'float object has no attribute price' error
"""

import traceback
import sys
from flask import Flask
from real_data_service import RealDataService

def test_data_service():
    """Test each method that could cause the price attribute error"""
    try:
        service = RealDataService()
        
        # Test portfolio data
        print("Testing get_real_portfolio_data...")
        portfolio = service.get_real_portfolio_data()
        print(f"Portfolio positions: {len(portfolio.get('positions', []))}")
        
        # Test AI performance
        print("Testing get_real_ai_performance...")
        ai_perf = service.get_real_ai_performance()
        print(f"AI models: {ai_perf.get('active_models', 0)}")
        
        # Test technical signals
        print("Testing get_real_technical_signals...")
        tech_signals = service.get_real_technical_signals()
        print(f"Technical signals: {list(tech_signals.keys())}")
        
        # Test risk metrics
        print("Testing get_real_risk_metrics...")
        risk = service.get_real_risk_metrics()
        print(f"Risk score: {risk.get('risk_score', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"Error in data service: {e}")
        if "'float' object has no attribute 'price'" in str(e):
            print("FOUND THE ERROR SOURCE!")
            traceback.print_exc()
            return False
        raise e

def test_flask_integration():
    """Test Flask integration specifically"""
    try:
        from modern_trading_app import trading_interface
        
        print("Testing get_dashboard_data...")
        data = trading_interface.get_dashboard_data()
        
        if 'error' in data:
            print(f"Dashboard error: {data['error']}")
            return False
        
        print("Dashboard data keys:", list(data.keys()))
        return True
        
    except Exception as e:
        print(f"Error in Flask integration: {e}")
        if "'float' object has no attribute 'price'" in str(e):
            print("FOUND THE ERROR SOURCE IN FLASK!")
            traceback.print_exc()
            return False
        raise e

if __name__ == "__main__":
    print("Starting minimal error reproduction test...")
    
    try:
        # Test data service
        ds_success = test_data_service()
        print(f"Data service test: {'PASS' if ds_success else 'FAIL'}")
        
        # Test Flask integration
        flask_success = test_flask_integration()
        print(f"Flask integration test: {'PASS' if flask_success else 'FAIL'}")
        
        if ds_success and flask_success:
            print("All tests passed - error may be in template rendering")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()