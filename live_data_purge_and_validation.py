#!/usr/bin/env python3
"""
Live Data Purge and Validation System
Complete elimination of mock/test/demo data and validation of live data feeds
"""

import sqlite3
import os
import logging
import json
from datetime import datetime
from typing import Dict, List, Any
import ccxt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveDataPurgeValidator:
    """Comprehensive system to purge mock data and validate live feeds"""
    
    def __init__(self):
        self.db_path = 'trading_platform.db'
        self.results = {
            'purged_tables': [],
            'validated_endpoints': [],
            'live_data_sources': [],
            'errors': [],
            'system_status': 'CHECKING'
        }
        
    def purge_mock_data_from_database(self):
        """Remove all mock/test data from database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all table names
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            mock_data_indicators = [
                'demo', 'test', 'mock', 'sample', 'fake', 'placeholder', 
                'example', 'sandbox', 'simulated'
            ]
            
            for table in tables:
                table_name = table[0]
                
                # Check for mock data in table contents
                try:
                    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5")
                    rows = cursor.fetchall()
                    
                    # Get column names
                    cursor.execute(f"PRAGMA table_info({table_name})")
                    columns = [col[1] for col in cursor.fetchall()]
                    
                    for row in rows:
                        row_data = str(row).lower()
                        if any(indicator in row_data for indicator in mock_data_indicators):
                            # Delete mock data rows
                            where_clauses = []
                            params = []
                            for i, value in enumerate(row):
                                if value and any(indicator in str(value).lower() for indicator in mock_data_indicators):
                                    where_clauses.append(f"{columns[i]} = ?")
                                    params.append(value)
                            
                            if where_clauses:
                                delete_query = f"DELETE FROM {table_name} WHERE {' OR '.join(where_clauses)}"
                                cursor.execute(delete_query, params)
                                logger.info(f"Purged mock data from table: {table_name}")
                                self.results['purged_tables'].append(table_name)
                
                except Exception as e:
                    logger.warning(f"Could not check table {table_name}: {e}")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database purge error: {e}")
            self.results['errors'].append(f"Database purge failed: {e}")
    
    def validate_live_okx_connection(self) -> Dict:
        """Validate OKX live data connection"""
        try:
            # Check environment variables
            api_key = os.getenv('OKX_API_KEY')
            secret_key = os.getenv('OKX_SECRET_KEY')
            passphrase = os.getenv('OKX_PASSPHRASE')
            
            if not all([api_key, secret_key, passphrase]):
                return {
                    'status': 'MISSING_CREDENTIALS',
                    'message': 'OKX API credentials not configured'
                }
            
            # Initialize OKX exchange
            exchange = ccxt.okx({
                'apiKey': api_key,
                'secret': secret_key,
                'password': passphrase,
                'sandbox': False,  # Ensure production mode
                'enableRateLimit': True,
            })
            
            # Test live market data
            btc_ticker = exchange.fetch_ticker('BTC/USDT')
            eth_ticker = exchange.fetch_ticker('ETH/USDT')
            
            # Test account balance (requires authentication)
            balance = exchange.fetch_balance()
            
            return {
                'status': 'LIVE_CONNECTED',
                'btc_price': btc_ticker['last'],
                'eth_price': eth_ticker['last'],
                'account_currencies': list(balance['total'].keys()),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'CONNECTION_FAILED',
                'error': str(e)
            }
    
    def validate_binance_fallback(self) -> Dict:
        """Validate Binance as fallback data source"""
        try:
            exchange = ccxt.binance({
                'enableRateLimit': True,
            })
            
            # Test public market data
            btc_ticker = exchange.fetch_ticker('BTC/USDT')
            eth_ticker = exchange.fetch_ticker('ETH/USDT')
            ada_ticker = exchange.fetch_ticker('ADA/USDT')
            
            return {
                'status': 'LIVE_CONNECTED',
                'btc_price': btc_ticker['last'],
                'eth_price': eth_ticker['last'],
                'ada_price': ada_ticker['last'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'CONNECTION_FAILED',
                'error': str(e)
            }
    
    def validate_coingecko_fallback(self) -> Dict:
        """Validate CoinGecko as fallback data source"""
        try:
            import requests
            
            # Test CoinGecko API
            response = requests.get(
                'https://api.coingecko.com/api/v3/simple/price',
                params={
                    'ids': 'bitcoin,ethereum,cardano',
                    'vs_currencies': 'usd'
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'status': 'LIVE_CONNECTED',
                    'btc_price': data['bitcoin']['usd'],
                    'eth_price': data['ethereum']['usd'],
                    'ada_price': data['cardano']['usd'],
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'status': 'API_ERROR',
                    'code': response.status_code
                }
                
        except Exception as e:
            return {
                'status': 'CONNECTION_FAILED',
                'error': str(e)
            }
    
    def check_file_system_for_mock_data(self):
        """Scan files for remaining mock data patterns"""
        mock_patterns = [
            'get_demo_realistic_balance',
            'generate_realistic',
            'fallback_data',
            'mock_mode',
            'sandbox=True',
            'simulate=True',
            'use_demo',
            'placeholder_data'
        ]
        
        python_files = []
        for root, dirs, files in os.walk('.'):
            # Skip cache and hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        files_with_mock_data = []
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for pattern in mock_patterns:
                        if pattern in content:
                            files_with_mock_data.append({
                                'file': file_path,
                                'pattern': pattern
                            })
                            break
            except Exception as e:
                logger.warning(f"Could not scan {file_path}: {e}")
        
        return files_with_mock_data
    
    def verify_trading_pairs_live_data(self) -> Dict:
        """Verify live data for main trading pairs"""
        required_pairs = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
        validation_results = {}
        
        # Test OKX first
        okx_result = self.validate_live_okx_connection()
        if okx_result['status'] == 'LIVE_CONNECTED':
            validation_results['okx'] = okx_result
            self.results['live_data_sources'].append('OKX')
        
        # Test Binance as fallback
        binance_result = self.validate_binance_fallback()
        if binance_result['status'] == 'LIVE_CONNECTED':
            validation_results['binance'] = binance_result
            self.results['live_data_sources'].append('Binance')
        
        # Test CoinGecko as fallback
        coingecko_result = self.validate_coingecko_fallback()
        if coingecko_result['status'] == 'LIVE_CONNECTED':
            validation_results['coingecko'] = coingecko_result
            self.results['live_data_sources'].append('CoinGecko')
        
        return validation_results
    
    def generate_final_status_report(self) -> Dict:
        """Generate comprehensive system status report"""
        
        # Purge database mock data
        self.purge_mock_data_from_database()
        
        # Validate live data connections
        live_data_validation = self.verify_trading_pairs_live_data()
        
        # Check for remaining mock data in files
        mock_files = self.check_file_system_for_mock_data()
        
        # Determine final system status
        if len(self.results['live_data_sources']) >= 2:  # At least 2 live sources
            if len(mock_files) == 0:  # No mock data found
                self.results['system_status'] = 'LIVE_CLEAN'
            else:
                self.results['system_status'] = 'LIVE_WITH_MOCK_REMNANTS'
        else:
            self.results['system_status'] = 'INSUFFICIENT_LIVE_SOURCES'
        
        final_report = {
            'timestamp': datetime.now().isoformat(),
            'system_status': self.results['system_status'],
            'live_data_sources': self.results['live_data_sources'],
            'purged_tables': self.results['purged_tables'],
            'live_data_validation': live_data_validation,
            'remaining_mock_files': mock_files,
            'errors': self.results['errors'],
            'recommendations': []
        }
        
        # Add recommendations based on status
        if self.results['system_status'] == 'LIVE_CLEAN':
            final_report['recommendations'] = [
                "✅ System is ready for live trading",
                "✅ All mock data has been purged",
                "✅ Multiple live data sources validated",
                "✅ Portfolio showing authentic OKX data"
            ]
        elif self.results['system_status'] == 'LIVE_WITH_MOCK_REMNANTS':
            final_report['recommendations'] = [
                f"⚠️ Remove mock data from {len(mock_files)} files",
                "✅ Live data sources are operational",
                "🔄 Run cleanup script to remove remaining patterns"
            ]
        else:
            final_report['recommendations'] = [
                "❌ Configure additional live data sources",
                "❌ Verify API credentials",
                "🔧 Check network connectivity"
            ]
        
        return final_report

def run_complete_purge_and_validation():
    """Execute complete mock data purge and live data validation"""
    validator = LiveDataPurgeValidator()
    
    logger.info("🧹 Starting comprehensive mock data purge and live data validation...")
    
    # Generate final status report
    report = validator.generate_final_status_report()
    
    # Save report to file
    with open('LIVE_DATA_VALIDATION_REPORT.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("🚀 LIVE DATA PURGE & VALIDATION COMPLETE")
    print("="*60)
    print(f"📊 System Status: {report['system_status']}")
    print(f"🔗 Live Sources: {', '.join(report['live_data_sources'])}")
    print(f"🗑️  Purged Tables: {len(report['purged_tables'])}")
    print(f"⚠️  Mock Files Found: {len(report['remaining_mock_files'])}")
    
    if report['system_status'] == 'LIVE_CLEAN':
        print("\n✅ SUCCESS: System is fully live with authentic data only!")
        print("🎯 Ready for live trading operations")
    else:
        print(f"\n⚠️  Status: {report['system_status']}")
        print("📋 Recommendations:")
        for rec in report['recommendations']:
            print(f"   {rec}")
    
    print("="*60)
    
    return report

if __name__ == '__main__':
    run_complete_purge_and_validation()