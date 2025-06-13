"""
Trading Parameters Configuration
Adjust automatic trading settings for optimal performance
"""
import sqlite3
import os
from datetime import datetime

class TradingParametersConfig:
    """Configure and adjust trading parameters"""
    
    def __init__(self):
        self.current_params = self.get_current_parameters()
        self.recommended_params = self.get_recommended_parameters()
    
    def get_current_parameters(self):
        """Get current trading parameters from all systems"""
        return {
            'pure_local_engine': {
                'min_confidence': 70.0,
                'max_position_pct': 25.0,
                'stop_loss_pct': 8.0,
                'take_profit_pct': 15.0,
                'scan_interval': 300  # 5 minutes
            },
            'direct_auto_trader': {
                'min_confidence': 75.0,
                'position_size_pct': 3.0,
                'min_trade_usd': 5.0,
                'check_interval': 5  # seconds
            },
            'futures_engine': {
                'min_confidence': 72.0,
                'max_leverage': 5,
                'max_position_pct': 15.0,
                'scan_interval': 300
            },
            'signal_execution_bridge': {
                'execution_threshold': 75.0,
                'max_position_size_pct': 3.5,
                'min_trade_amount': 5,
                'rate_limit_delay': 0.2
            }
        }
    
    def get_recommended_parameters(self):
        """Get recommended parameter adjustments based on market conditions"""
        return {
            'conservative': {
                'min_confidence': 80.0,
                'position_size_pct': 2.0,
                'stop_loss_pct': 5.0,
                'take_profit_pct': 10.0
            },
            'balanced': {
                'min_confidence': 75.0,
                'position_size_pct': 3.0,
                'stop_loss_pct': 8.0,
                'take_profit_pct': 15.0
            },
            'aggressive': {
                'min_confidence': 70.0,
                'position_size_pct': 5.0,
                'stop_loss_pct': 12.0,
                'take_profit_pct': 20.0
            },
            'high_frequency': {
                'min_confidence': 65.0,
                'position_size_pct': 2.5,
                'scan_interval': 60,  # 1 minute
                'check_interval': 2   # 2 seconds
            }
        }
    
    def apply_parameter_set(self, strategy: str):
        """Apply a predefined parameter set"""
        if strategy not in self.recommended_params:
            return False
        
        params = self.recommended_params[strategy]
        
        # Update Direct Auto Trader
        self.update_direct_trader_params(params)
        
        # Update Pure Local Engine
        self.update_pure_local_params(params)
        
        # Update Signal Execution Bridge
        self.update_execution_bridge_params(params)
        
        return True
    
    def update_direct_trader_params(self, params):
        """Update direct auto trader parameters"""
        try:
            with open('direct_execution_fix.py', 'r') as f:
                content = f.read()
            
            # Update confidence threshold
            if 'min_confidence' in params:
                content = content.replace(
                    'self.min_confidence = 75.0',
                    f'self.min_confidence = {params["min_confidence"]}'
                )
            
            # Update position size
            if 'position_size_pct' in params:
                content = content.replace(
                    'self.position_size_pct = 0.03',
                    f'self.position_size_pct = {params["position_size_pct"]/100}'
                )
            
            with open('direct_execution_fix.py', 'w') as f:
                f.write(content)
                
            print(f"✅ Direct Auto Trader parameters updated")
            
        except Exception as e:
            print(f"❌ Error updating Direct Auto Trader: {e}")
    
    def update_pure_local_params(self, params):
        """Update Pure Local Engine parameters"""
        try:
            with open('pure_local_trading_engine.py', 'r') as f:
                content = f.read()
            
            # Update confidence threshold
            if 'min_confidence' in params:
                content = content.replace(
                    'self.min_confidence = 70.0',
                    f'self.min_confidence = {params["min_confidence"]}'
                )
            
            # Update stop loss
            if 'stop_loss_pct' in params:
                content = content.replace(
                    'self.stop_loss_pct = 8.0',
                    f'self.stop_loss_pct = {params["stop_loss_pct"]}'
                )
            
            # Update take profit
            if 'take_profit_pct' in params:
                content = content.replace(
                    'self.take_profit_pct = 15.0',
                    f'self.take_profit_pct = {params["take_profit_pct"]}'
                )
            
            with open('pure_local_trading_engine.py', 'w') as f:
                f.write(content)
                
            print(f"✅ Pure Local Engine parameters updated")
            
        except Exception as e:
            print(f"❌ Error updating Pure Local Engine: {e}")
    
    def update_execution_bridge_params(self, params):
        """Update Signal Execution Bridge parameters"""
        try:
            with open('signal_execution_bridge.py', 'r') as f:
                content = f.read()
            
            # Update execution threshold
            if 'min_confidence' in params:
                content = content.replace(
                    'self.execution_threshold = 75.0',
                    f'self.execution_threshold = {params["min_confidence"]}'
                )
            
            # Update position size
            if 'position_size_pct' in params:
                content = content.replace(
                    'self.max_position_size_pct = 0.035',
                    f'self.max_position_size_pct = {params["position_size_pct"]/100}'
                )
            
            with open('signal_execution_bridge.py', 'w') as f:
                f.write(content)
                
            print(f"✅ Signal Execution Bridge parameters updated")
            
        except Exception as e:
            print(f"❌ Error updating Signal Execution Bridge: {e}")
    
    def create_custom_parameters(self, min_confidence=75.0, position_size_pct=3.0, 
                                stop_loss_pct=8.0, take_profit_pct=15.0):
        """Create custom parameter set"""
        custom_params = {
            'min_confidence': min_confidence,
            'position_size_pct': position_size_pct,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }
        
        self.update_direct_trader_params(custom_params)
        self.update_pure_local_params(custom_params)
        self.update_execution_bridge_params(custom_params)
        
        return custom_params
    
    def display_current_parameters(self):
        """Display current parameter settings"""
        print("CURRENT TRADING PARAMETERS")
        print("=" * 50)
        
        for system, params in self.current_params.items():
            print(f"\n{system.upper().replace('_', ' ')}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
        
        print("\nRECOMMENDED PARAMETER SETS")
        print("=" * 50)
        
        for strategy, params in self.recommended_params.items():
            print(f"\n{strategy.upper()}:")
            for param, value in params.items():
                print(f"  {param}: {value}")
    
    def get_optimization_suggestions(self):
        """Get parameter optimization suggestions based on recent performance"""
        suggestions = []
        
        # Check if any trades were executed recently
        if os.path.exists('direct_executions.db'):
            conn = sqlite3.connect('direct_executions.db')
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM executed_trades")
            trade_count = cursor.fetchone()[0]
            
            if trade_count == 0:
                suggestions.append("No trades executed yet - consider lowering confidence threshold")
                suggestions.append("Current minimum confidence: 75% - try 70% for more signals")
            else:
                cursor.execute("""
                    SELECT AVG(confidence), COUNT(*) FROM executed_trades 
                    WHERE timestamp > datetime('now', '-1 hour')
                """)
                recent_data = cursor.fetchone()
                
                if recent_data[1] > 5:
                    suggestions.append("High trading frequency - consider raising confidence threshold")
                elif recent_data[1] < 2:
                    suggestions.append("Low trading frequency - consider lowering confidence threshold")
            
            conn.close()
        
        return suggestions

def main():
    """Interactive parameter configuration"""
    config = TradingParametersConfig()
    
    print("TRADING PARAMETERS CONFIGURATION")
    print("=" * 50)
    
    config.display_current_parameters()
    
    print("\nOPTIMIZATION SUGGESTIONS:")
    suggestions = config.get_optimization_suggestions()
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
    
    print("\nAVAILABLE ACTIONS:")
    print("1. Apply CONSERVATIVE parameters (80% confidence, 2% position)")
    print("2. Apply BALANCED parameters (75% confidence, 3% position)")
    print("3. Apply AGGRESSIVE parameters (70% confidence, 5% position)")
    print("4. Apply HIGH FREQUENCY parameters (65% confidence, fast scanning)")
    print("5. Create CUSTOM parameters")
    
    return config

if __name__ == "__main__":
    main()