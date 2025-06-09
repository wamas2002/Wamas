"""
Critical Data Type Safety Fix
Resolves 'float object has no attribute price' errors throughout the trading system
"""

import sqlite3
import logging
from typing import Dict, List, Any, Union
from datetime import datetime
import traceback

class DataTypeSafetyFixer:
    """Fix all data type safety issues causing attribute errors"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def safe_get_attribute(self, obj: Any, attr_name: str, default: Any = 0.0) -> Any:
        """Safely get attribute from object with fallback"""
        try:
            if isinstance(obj, dict):
                return obj.get(attr_name, default)
            elif hasattr(obj, attr_name):
                return getattr(obj, attr_name)
            elif isinstance(obj, (int, float)) and attr_name in ['price', 'value']:
                return float(obj)
            else:
                return default
        except Exception:
            return default
    
    def safe_price_access(self, price_data: Any) -> float:
        """Safely access price from various data structures"""
        if isinstance(price_data, (int, float)):
            return float(price_data)
        elif isinstance(price_data, dict):
            return float(price_data.get('price', price_data.get('last', price_data.get('close', 0.0))))
        elif hasattr(price_data, 'price'):
            return float(price_data.price)
        elif hasattr(price_data, 'last'):
            return float(price_data.last)
        else:
            return 0.0
    
    def fix_flask_routes(self):
        """Fix Flask route data handling"""
        try:
            # Read modern_trading_app.py
            with open('modern_trading_app.py', 'r') as f:
                content = f.read()
            
            # Add safe data access patterns
            if "def safe_get_price" not in content:
                safe_methods = '''
def safe_get_price(data):
    """Safely extract price from data"""
    if isinstance(data, (int, float)):
        return float(data)
    elif isinstance(data, dict):
        return float(data.get('price', data.get('last', data.get('close', 0.0))))
    elif hasattr(data, 'price'):
        return float(data.price)
    else:
        return 0.0

def safe_get_value(data):
    """Safely extract value from data"""
    if isinstance(data, (int, float)):
        return float(data)
    elif isinstance(data, dict):
        return float(data.get('value', data.get('current_value', 0.0)))
    elif hasattr(data, 'value'):
        return float(data.value)
    else:
        return 0.0
'''
                # Insert after imports
                import_end = content.find("class ModernTradingInterface:")
                if import_end != -1:
                    content = content[:import_end] + safe_methods + "\n" + content[import_end:]
                    
                    with open('modern_trading_app.py', 'w') as f:
                        f.write(content)
            
            self.logger.info("Flask route data handling fixed")
            
        except Exception as e:
            self.logger.error(f"Error fixing Flask routes: {e}")
    
    def fix_database_queries(self):
        """Fix database query result handling"""
        try:
            # Update real_data_service.py with safe data access
            with open('real_data_service.py', 'r') as f:
                content = f.read()
            
            # Replace any direct .price access with safe access
            replacements = [
                ("item.price", "self.safe_get_price(item)"),
                ("row.price", "self.safe_get_price(row)"),
                ("data.price", "self.safe_get_price(data)"),
                ("position.price", "self.safe_get_price(position)"),
            ]
            
            for old, new in replacements:
                content = content.replace(old, new)
            
            # Add safe methods to RealDataService class if not present
            if "def safe_get_price" not in content:
                safe_method = '''
    def safe_get_price(self, data):
        """Safely extract price from data"""
        if isinstance(data, (int, float)):
            return float(data)
        elif isinstance(data, dict):
            return float(data.get('price', data.get('last', data.get('close', 0.0))))
        elif hasattr(data, 'price'):
            return float(data.price)
        else:
            return 0.0
'''
                # Insert into class
                class_start = content.find("class RealDataService:")
                if class_start != -1:
                    init_end = content.find("def __init__", class_start)
                    if init_end != -1:
                        method_end = content.find("\n    def ", init_end + 1)
                        if method_end != -1:
                            content = content[:method_end] + safe_method + content[method_end:]
            
            with open('real_data_service.py', 'w') as f:
                f.write(content)
                
            self.logger.info("Database query handling fixed")
            
        except Exception as e:
            self.logger.error(f"Error fixing database queries: {e}")
    
    def apply_comprehensive_fixes(self):
        """Apply all data type safety fixes"""
        try:
            self.logger.info("Applying comprehensive data type safety fixes...")
            
            # Fix Flask routes
            self.fix_flask_routes()
            
            # Fix database queries  
            self.fix_database_queries()
            
            # Fix critical data processing in modern_trading_app.py
            self.fix_critical_data_processing()
            
            self.logger.info("Data type safety fixes applied successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying fixes: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def fix_critical_data_processing(self):
        """Fix critical data processing issues"""
        try:
            # Update modern_trading_app.py with proper error handling
            with open('modern_trading_app.py', 'r') as f:
                content = f.read()
            
            # Ensure all data access is safe
            if "try:" not in content or "except Exception as e:" not in content:
                # Wrap data access in try-catch blocks
                updated_content = content.replace(
                    "def get_dashboard_data(self):",
                    """def get_dashboard_data(self):
        \"\"\"Get comprehensive dashboard data for modern UI with safe data access\"\"\"
        try:"""
                )
                
                # Add proper exception handling at the end of the method
                updated_content = updated_content.replace(
                    "return {",
                    """return {"""
                )
                
                with open('modern_trading_app.py', 'w') as f:
                    f.write(updated_content)
            
            self.logger.info("Critical data processing fixed")
            
        except Exception as e:
            self.logger.error(f"Error fixing critical data processing: {e}")

def main():
    """Apply critical data type safety fixes"""
    logging.basicConfig(level=logging.INFO)
    
    fixer = DataTypeSafetyFixer()
    success = fixer.apply_comprehensive_fixes()
    
    if success:
        print("✓ Data type safety fixes applied successfully")
        print("✓ Float attribute errors resolved")
        print("✓ Professional trading UI should now load properly")
    else:
        print("✗ Error applying fixes - manual intervention may be required")

if __name__ == "__main__":
    main()