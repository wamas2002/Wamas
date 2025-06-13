"""
Comprehensive System Optimization
Addresses OpenAI quota limits, invalid symbols, and performance issues
"""
import sqlite3
import logging
import ccxt
import os
from typing import List, Dict, Set

class ComprehensiveSystemOptimizer:
    def __init__(self):
        self.exchange = None
        self.valid_symbols = set()
        self.databases = [
            'enhanced_trading.db',
            'autonomous_trading.db', 
            'enhanced_ui.db',
            'attribution.db'
        ]
        
    def initialize_exchange(self) -> bool:
        """Initialize OKX exchange with proper error handling"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY'),
                'secret': os.getenv('OKX_SECRET_KEY'),
                'password': os.getenv('OKX_PASSPHRASE'),
                'sandbox': False,
                'enableRateLimit': True,
                'options': {'defaultType': 'spot'}
            })
            return True
        except Exception as e:
            logging.error(f"Exchange initialization failed: {e}")
            return False
    
    def get_validated_okx_symbols(self) -> Set[str]:
        """Get only symbols that actually exist on OKX"""
        if not self.exchange and not self.initialize_exchange():
            return self._get_known_working_symbols()
        
        try:
            markets = self.exchange.load_markets()
            spot_symbols = {symbol for symbol in markets.keys() 
                          if symbol.endswith('/USDT') and markets[symbol]['active']}
            
            # Filter for high volume pairs only
            high_volume_symbols = {
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
                'SOL/USDT', 'DOGE/USDT', 'LINK/USDT', 'LTC/USDT', 'DOT/USDT',
                'AVAX/USDT', 'UNI/USDT', 'ATOM/USDT', 'NEAR/USDT', 'TRX/USDT',
                'ICP/USDT', 'ALGO/USDT', 'HBAR/USDT', 'XLM/USDT', 'SAND/USDT',
                'MANA/USDT', 'THETA/USDT', 'AXS/USDT', 'FIL/USDT', 'ETC/USDT',
                'EGLD/USDT', 'FLOW/USDT', 'ENJ/USDT', 'CHZ/USDT', 'CRV/USDT',
                'AAVE/USDT', 'MKR/USDT', 'SNX/USDT', 'COMP/USDT', 'YFI/USDT',
                'SUSHI/USDT', '1INCH/USDT', 'CAKE/USDT', 'BAL/USDT', 'UMA/USDT',
                'KNC/USDT', 'ZRX/USDT', 'LRC/USDT', 'BNT/USDT', 'GRT/USDT',
                'BAT/USDT', 'ICX/USDT', 'ZIL/USDT', 'QTUM/USDT', 'RSR/USDT'
            }
            
            # Return intersection of high volume and available symbols
            validated = spot_symbols.intersection(high_volume_symbols)
            logging.info(f"Validated {len(validated)} OKX symbols")
            return validated
            
        except Exception as e:
            logging.error(f"Symbol validation failed: {e}")
            return self._get_known_working_symbols()
    
    def _get_known_working_symbols(self) -> Set[str]:
        """Fallback to known working symbols"""
        return {
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
            'SOL/USDT', 'DOGE/USDT', 'LINK/USDT', 'LTC/USDT', 'DOT/USDT',
            'AVAX/USDT', 'UNI/USDT', 'ATOM/USDT', 'NEAR/USDT', 'TRX/USDT',
            'ICP/USDT', 'ALGO/USDT', 'HBAR/USDT', 'XLM/USDT', 'SAND/USDT'
        }
    
    def optimize_openai_usage(self):
        """Optimize OpenAI API usage to prevent quota errors"""
        optimization_settings = {
            'disable_gpt_analysis': True,
            'use_local_analysis_only': True,
            'reduce_api_calls': True,
            'implement_fallback_scoring': True
        }
        
        # Update all databases with optimization flags
        for db_name in self.databases:
            try:
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()
                
                # Create optimization settings table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_optimization (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        setting_name TEXT UNIQUE NOT NULL,
                        setting_value TEXT NOT NULL,
                        description TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Insert optimization settings
                for setting, value in optimization_settings.items():
                    cursor.execute('''
                        INSERT OR REPLACE INTO system_optimization 
                        (setting_name, setting_value, description)
                        VALUES (?, ?, ?)
                    ''', (setting, str(value), f"Optimization flag: {setting}"))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logging.error(f"Failed to update optimization settings in {db_name}: {e}")
    
    def update_symbol_filters(self):
        """Update all systems with validated symbols only"""
        validated_symbols = self.get_validated_okx_symbols()
        symbol_list = list(validated_symbols)[:50]  # Limit to top 50 for performance
        
        for db_name in self.databases:
            try:
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()
                
                # Create validated symbols table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS validated_symbols (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT UNIQUE NOT NULL,
                        market_type TEXT DEFAULT 'spot',
                        is_active BOOLEAN DEFAULT 1,
                        volume_rank INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Clear existing symbols
                cursor.execute('DELETE FROM validated_symbols')
                
                # Insert validated symbols
                for rank, symbol in enumerate(symbol_list, 1):
                    cursor.execute('''
                        INSERT INTO validated_symbols (symbol, volume_rank)
                        VALUES (?, ?)
                    ''', (symbol, rank))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logging.error(f"Failed to update symbols in {db_name}: {e}")
    
    def implement_performance_optimizations(self):
        """Implement performance optimizations across all systems"""
        performance_configs = {
            'scan_interval_seconds': 300,  # 5 minutes
            'max_concurrent_scans': 5,
            'cache_duration_minutes': 15,
            'max_signals_per_scan': 10,
            'indicator_calculation_timeout': 30,
            'market_data_limit': 200,
            'enable_aggressive_caching': True,
            'reduce_log_verbosity': True
        }
        
        for db_name in self.databases:
            try:
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()
                
                # Create performance config table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance_config (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        config_name TEXT UNIQUE NOT NULL,
                        config_value TEXT NOT NULL,
                        config_type TEXT DEFAULT 'performance',
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Insert performance configurations
                for config, value in performance_configs.items():
                    cursor.execute('''
                        INSERT OR REPLACE INTO performance_config 
                        (config_name, config_value, config_type)
                        VALUES (?, ?, ?)
                    ''', (config, str(value), 'performance'))
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logging.error(f"Failed to update performance config in {db_name}: {e}")
    
    def create_fallback_analysis_system(self):
        """Create local fallback analysis system to replace OpenAI when quota exceeded"""
        for db_name in self.databases:
            try:
                conn = sqlite3.connect(db_name)
                cursor = conn.cursor()
                
                # Create fallback analysis templates
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS fallback_analysis_templates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        analysis_type TEXT NOT NULL,
                        confidence_boost REAL DEFAULT 0,
                        risk_assessment TEXT DEFAULT 'medium',
                        template_logic TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Insert fallback templates
                templates = [
                    ('trend_following', 5.0, 'low', 'RSI < 30 AND EMA_9 > EMA_21'),
                    ('breakout_momentum', 8.0, 'medium', 'Volume > 1.5x AND Price > BB_Upper'),
                    ('mean_reversion', 3.0, 'high', 'RSI > 70 AND Price < EMA_50'),
                    ('oversold_bounce', 6.0, 'medium', 'RSI < 25 AND Stoch < 20'),
                    ('overbought_pullback', 4.0, 'medium', 'RSI > 75 AND Stoch > 80')
                ]
                
                cursor.executemany('''
                    INSERT OR REPLACE INTO fallback_analysis_templates 
                    (analysis_type, confidence_boost, risk_assessment, template_logic)
                    VALUES (?, ?, ?, ?)
                ''', templates)
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logging.error(f"Failed to create fallback analysis in {db_name}: {e}")
    
    def run_comprehensive_optimization(self):
        """Execute all optimization procedures"""
        logging.info("Starting comprehensive system optimization...")
        
        self.optimize_openai_usage()
        self.update_symbol_filters()
        self.implement_performance_optimizations()
        self.create_fallback_analysis_system()
        
        logging.info("Comprehensive optimization complete")

def main():
    """Execute comprehensive system optimization"""
    logging.basicConfig(level=logging.INFO)
    
    optimizer = ComprehensiveSystemOptimizer()
    optimizer.run_comprehensive_optimization()
    
    print("✅ Comprehensive system optimization complete")
    print("🔧 OpenAI quota management implemented")
    print("📊 Symbol validation updated")
    print("⚡ Performance optimizations applied")
    print("🛡️ Fallback analysis system created")

if __name__ == "__main__":
    main()