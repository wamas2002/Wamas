"""
Apply Aggressive Trading Parameters
Enable more frequent trading with optimized settings
"""
import os

def apply_aggressive_parameters():
    """Apply aggressive trading parameters across all systems"""
    
    # Aggressive Parameters:
    # - Min Confidence: 70% (lower threshold for more signals)
    # - Position Size: 5% (larger positions)
    # - Stop Loss: 12% (wider stops)
    # - Take Profit: 20% (higher targets)
    
    # Update Direct Auto Trader
    update_direct_trader()
    
    # Update Pure Local Engine  
    update_pure_local_engine()
    
    # Update Signal Execution Bridge
    update_execution_bridge()
    
    print("✅ AGGRESSIVE PARAMETERS APPLIED")
    print("Configuration:")
    print("- Minimum Confidence: 70%")
    print("- Position Size: 5%")
    print("- Stop Loss: 12%")
    print("- Take Profit: 20%")
    print("- More frequent execution enabled")

def update_direct_trader():
    """Update Direct Auto Trader with aggressive settings"""
    try:
        with open('direct_execution_fix.py', 'r') as f:
            content = f.read()
        
        # Lower confidence threshold to 70%
        content = content.replace(
            'self.min_confidence = 75.0',
            'self.min_confidence = 70.0'
        )
        
        # Increase position size to 5%
        content = content.replace(
            'self.position_size_pct = 0.03',
            'self.position_size_pct = 0.05'
        )
        
        with open('direct_execution_fix.py', 'w') as f:
            f.write(content)
            
        print("✅ Direct Auto Trader: 70% confidence, 5% position size")
        
    except Exception as e:
        print(f"❌ Direct Auto Trader update failed: {e}")

def update_pure_local_engine():
    """Update Pure Local Engine with aggressive settings"""
    try:
        with open('pure_local_trading_engine.py', 'r') as f:
            content = f.read()
        
        # Keep confidence at 70% (already optimal)
        # Update stop loss to 12%
        content = content.replace(
            'self.stop_loss_pct = 8.0',
            'self.stop_loss_pct = 12.0'
        )
        
        # Update take profit to 20%
        content = content.replace(
            'self.take_profit_pct = 15.0',
            'self.take_profit_pct = 20.0'
        )
        
        with open('pure_local_trading_engine.py', 'w') as f:
            f.write(content)
            
        print("✅ Pure Local Engine: 12% stop loss, 20% take profit")
        
    except Exception as e:
        print(f"❌ Pure Local Engine update failed: {e}")

def update_execution_bridge():
    """Update Signal Execution Bridge with aggressive settings"""
    try:
        with open('signal_execution_bridge.py', 'r') as f:
            content = f.read()
        
        # Lower execution threshold to 70%
        content = content.replace(
            'self.execution_threshold = 75.0',
            'self.execution_threshold = 70.0'
        )
        
        # Increase position size to 5%
        content = content.replace(
            'self.max_position_size_pct = 0.035',
            'self.max_position_size_pct = 0.05'
        )
        
        with open('signal_execution_bridge.py', 'w') as f:
            f.write(content)
            
        print("✅ Signal Execution Bridge: 70% threshold, 5% position size")
        
    except Exception as e:
        print(f"❌ Signal Execution Bridge update failed: {e}")

if __name__ == "__main__":
    apply_aggressive_parameters()