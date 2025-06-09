"""
Diagnostic Script to Identify 'float object has no attribute price' Error
"""

import traceback
import sys
import logging
from flask import Flask
from real_data_service import RealDataService

def diagnose_error():
    """Diagnose the exact location of the price attribute error"""
    
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        print("Testing RealDataService methods...")
        
        # Initialize service
        service = RealDataService()
        
        # Test each method that might access price attributes
        methods_to_test = [
            ('get_real_portfolio_data', service.get_real_portfolio_data),
            ('get_real_ai_performance', service.get_real_ai_performance),
            ('get_real_technical_signals', service.get_real_technical_signals),
            ('get_real_risk_metrics', service.get_real_risk_metrics)
        ]
        
        for method_name, method in methods_to_test:
            try:
                print(f"\nTesting {method_name}...")
                result = method()
                print(f"✓ {method_name} successful")
                
                # Check for any float objects that might have price attributes
                if isinstance(result, dict):
                    for key, value in result.items():
                        if isinstance(value, float):
                            if hasattr(value, 'price'):
                                print(f"WARNING: Float object {key} has price attribute")
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, float) and hasattr(item, 'price'):
                                    print(f"WARNING: Float in list {key} has price attribute")
                
            except Exception as e:
                print(f"✗ {method_name} failed: {e}")
                traceback.print_exc()
                
                # Check if this is the price attribute error
                if "'float' object has no attribute 'price'" in str(e):
                    print(f"FOUND ERROR SOURCE: {method_name}")
                    return method_name
        
        # Test Flask app initialization
        print("\nTesting Flask app...")
        try:
            from modern_trading_app import app, trading_interface
            with app.test_client() as client:
                response = client.get('/api/dashboard-data')
                if response.status_code == 200:
                    print("✓ Flask dashboard API successful")
                else:
                    print(f"✗ Flask dashboard API failed: {response.status_code}")
                    print(response.get_data(as_text=True))
        except Exception as e:
            print(f"✗ Flask app test failed: {e}")
            traceback.print_exc()
            
            if "'float' object has no attribute 'price'" in str(e):
                print("FOUND ERROR SOURCE: Flask app initialization")
                return "Flask app"
        
        print("\nNo price attribute errors detected in direct testing")
        return None
        
    except Exception as e:
        print(f"Diagnostic script error: {e}")
        traceback.print_exc()
        return "diagnostic_error"

if __name__ == "__main__":
    error_source = diagnose_error()
    if error_source:
        print(f"\nERROR SOURCE IDENTIFIED: {error_source}")
    else:
        print("\nNo errors found in diagnostic testing")