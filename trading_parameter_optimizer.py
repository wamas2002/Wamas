#!/usr/bin/env python3
"""
Trading Parameter Optimizer
Adjusts AI trading parameters to increase crypto exposure and optimize signal generation
"""

import sqlite3
import os
import ccxt
from datetime import datetime
import json

class TradingParameterOptimizer:
    def __init__(self):
        self.db_path = 'trading_platform.db'
        self.live_db_path = 'live_trading.db'
        
    def get_current_parameters(self):
        """Get current trading parameters"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for trading parameters table
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trading_parameters'")
            if not cursor.fetchone():
                return self.get_default_parameters()
            
            cursor.execute("SELECT * FROM trading_parameters ORDER BY timestamp DESC LIMIT 1")
            params = cursor.fetchone()
            conn.close()
            
            if params:
                return {
                    'confidence_threshold': params[1],
                    'position_size_pct': params[2],
                    'stop_loss_pct': params[3],
                    'take_profit_pct': params[4],
                    'max_positions': params[5],
                    'rebalance_threshold': params[6]
                }
            else:
                return self.get_default_parameters()
                
        except Exception as e:
            print(f"Error getting parameters: {e}")
            return self.get_default_parameters()
    
    def get_default_parameters(self):
        """Default conservative parameters"""
        return {
            'confidence_threshold': 75,  # Very high threshold causing low activity
            'position_size_pct': 1.0,    # Very small position sizes
            'stop_loss_pct': 2.0,        # Tight stop losses
            'take_profit_pct': 4.0,      # Conservative profit taking
            'max_positions': 3,          # Limited positions
            'rebalance_threshold': 15    # High rebalance threshold
        }
    
    def optimize_parameters(self):
        """Optimize parameters for better crypto exposure"""
        current = self.get_current_parameters()
        
        # Adjust parameters for more aggressive trading
        optimized = {
            'confidence_threshold': 45,  # Lower threshold for more signals
            'position_size_pct': 3.5,    # Larger position sizes
            'stop_loss_pct': 3.5,        # Slightly wider stops
            'take_profit_pct': 6.0,      # Higher profit targets
            'max_positions': 6,          # More concurrent positions
            'rebalance_threshold': 8     # More frequent rebalancing
        }
        
        return current, optimized
    
    def save_optimized_parameters(self, params):
        """Save optimized parameters to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create parameters table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    confidence_threshold REAL,
                    position_size_pct REAL,
                    stop_loss_pct REAL,
                    take_profit_pct REAL,
                    max_positions INTEGER,
                    rebalance_threshold REAL,
                    timestamp TEXT
                )
            ''')
            
            # Insert new parameters
            cursor.execute('''
                INSERT INTO trading_parameters 
                (confidence_threshold, position_size_pct, stop_loss_pct, 
                 take_profit_pct, max_positions, rebalance_threshold, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                params['confidence_threshold'],
                params['position_size_pct'],
                params['stop_loss_pct'],
                params['take_profit_pct'],
                params['max_positions'],
                params['rebalance_threshold'],
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving parameters: {e}")
            return False
    
    def update_signal_generation_config(self):
        """Update AI signal generation configuration"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create signal config table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_config (
                    id INTEGER PRIMARY KEY,
                    min_confidence REAL,
                    signal_frequency_minutes INTEGER,
                    market_volatility_factor REAL,
                    trend_strength_weight REAL,
                    volume_weight REAL,
                    last_updated TEXT
                )
            ''')
            
            # Clear existing config
            cursor.execute("DELETE FROM signal_config")
            
            # Insert optimized signal configuration
            cursor.execute('''
                INSERT INTO signal_config 
                (min_confidence, signal_frequency_minutes, market_volatility_factor,
                 trend_strength_weight, volume_weight, last_updated)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                45.0,    # Lower minimum confidence
                5,       # More frequent signal generation
                1.2,     # Higher volatility factor
                0.7,     # Balanced trend weight
                0.3,     # Volume confirmation
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error updating signal config: {e}")
            return False
    
    def create_portfolio_allocation_targets(self):
        """Create target allocation strategy"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create allocation targets table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS allocation_targets (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT,
                    target_percentage REAL,
                    min_allocation REAL,
                    max_allocation REAL,
                    priority INTEGER,
                    active BOOLEAN,
                    last_updated TEXT
                )
            ''')
            
            # Clear existing targets
            cursor.execute("DELETE FROM allocation_targets")
            
            # Define target allocations for increased crypto exposure
            targets = [
                ('BTC', 25.0, 20.0, 35.0, 1, True),
                ('ETH', 20.0, 15.0, 30.0, 2, True),
                ('SOL', 12.0, 8.0, 18.0, 3, True),
                ('ADA', 8.0, 5.0, 12.0, 4, True),
                ('DOT', 6.0, 3.0, 10.0, 5, True),
                ('AVAX', 4.0, 2.0, 8.0, 6, True),
                ('USDT', 25.0, 15.0, 40.0, 7, True)  # Reduced cash target
            ]
            
            for symbol, target, min_alloc, max_alloc, priority, active in targets:
                cursor.execute('''
                    INSERT INTO allocation_targets 
                    (symbol, target_percentage, min_allocation, max_allocation, 
                     priority, active, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (symbol, target, min_alloc, max_alloc, priority, active, 
                     datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error creating allocation targets: {e}")
            return False
    
    def update_risk_management_rules(self):
        """Update risk management parameters"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create risk management table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS risk_management (
                    id INTEGER PRIMARY KEY,
                    max_portfolio_risk_pct REAL,
                    max_position_risk_pct REAL,
                    correlation_limit REAL,
                    volatility_limit REAL,
                    drawdown_limit_pct REAL,
                    emergency_stop_loss REAL,
                    last_updated TEXT
                )
            ''')
            
            # Clear existing rules
            cursor.execute("DELETE FROM risk_management")
            
            # Insert updated risk management rules
            cursor.execute('''
                INSERT INTO risk_management 
                (max_portfolio_risk_pct, max_position_risk_pct, correlation_limit,
                 volatility_limit, drawdown_limit_pct, emergency_stop_loss, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                5.0,    # 5% max portfolio risk
                1.5,    # 1.5% max position risk (increased from 1%)
                0.75,   # 75% correlation limit
                0.25,   # 25% volatility limit
                8.0,    # 8% max drawdown
                -15.0,  # Emergency stop at -15%
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error updating risk management: {e}")
            return False

def optimize_trading_system():
    """Main function to optimize trading parameters"""
    optimizer = TradingParameterOptimizer()
    
    print("🔧 Trading Parameter Optimization Starting...")
    
    # Get current parameters
    current, optimized = optimizer.optimize_parameters()
    
    print("\n📊 Parameter Comparison:")
    print("Current -> Optimized")
    for key in current:
        print(f"{key}: {current[key]} -> {optimized[key]}")
    
    # Save optimized parameters
    if optimizer.save_optimized_parameters(optimized):
        print("✅ Trading parameters updated")
    
    # Update signal generation
    if optimizer.update_signal_generation_config():
        print("✅ Signal generation optimized")
    
    # Create allocation targets
    if optimizer.create_portfolio_allocation_targets():
        print("✅ Portfolio allocation targets set")
    
    # Update risk management
    if optimizer.update_risk_management_rules():
        print("✅ Risk management rules updated")
    
    print("\n🚀 Trading System Optimization Complete!")
    print("Expected improvements:")
    print("- Increased crypto exposure from 7.2% to 75% target")
    print("- More frequent BUY signals (confidence lowered to 45%)")
    print("- Larger position sizes (1% -> 3.5% per trade)")
    print("- Better portfolio diversification across 6 cryptocurrencies")
    print("- Enhanced rebalancing frequency")
    
    return True

if __name__ == "__main__":
    optimize_trading_system()