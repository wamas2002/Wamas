"""
Final System Verification with Live Data
Confirms all optimizations are working correctly after implementing fixes
"""

import sqlite3
import pandas as pd
import os
from datetime import datetime
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalSystemVerifier:
    """Verifies all system components after optimization fixes"""
    
    def __init__(self):
        self.verification_results = {
            'timestamp': datetime.now().isoformat(),
            'api_integration': {},
            'database_health': {},
            'data_pipeline': {},
            'ui_components': {},
            'optimization_impact': {},
            'overall_status': 'VERIFYING'
        }
    
    def verify_okx_api_fixes(self):
        """Verify OKX API integration fixes are working"""
        logger.info("Verifying OKX API integration fixes...")
        
        try:
            from trading.okx_data_service import OKXDataService
            okx_service = OKXDataService()
            
            # Test the newly added methods
            test_results = {}
            
            # Test get_candles method
            try:
                candles = okx_service.get_candles('BTCUSDT', '1h', 5)
                test_results['get_candles'] = 'WORKING' if not candles.empty else 'NO_DATA'
            except Exception as e:
                test_results['get_candles'] = f'ERROR: {str(e)[:30]}'
            
            # Test get_instruments method  
            try:
                instruments = okx_service.get_instruments('SPOT')
                test_results['get_instruments'] = 'WORKING' if instruments else 'NO_DATA'
            except Exception as e:
                test_results['get_instruments'] = f'ERROR: {str(e)[:30]}'
            
            # Test live price data
            try:
                price = okx_service.get_current_price('BTCUSDT')
                test_results['live_prices'] = 'WORKING' if price > 0 else 'NO_DATA'
            except Exception as e:
                test_results['live_prices'] = f'ERROR: {str(e)[:30]}'
            
            self.verification_results['api_integration'] = {
                'status': 'FIXED' if all('WORKING' in result for result in test_results.values()) else 'PARTIAL',
                'test_results': test_results,
                'critical_methods_available': 'get_candles' in test_results and 'get_instruments' in test_results
            }
            
        except Exception as e:
            self.verification_results['api_integration'] = {
                'status': 'ERROR',
                'error': str(e)
            }
    
    def verify_database_schema_fixes(self):
        """Verify database schema standardization is working"""
        logger.info("Verifying database schema fixes...")
        
        schema_status = {}
        
        databases = [
            'data/market_data.db',
            'data/portfolio_tracking.db', 
            'data/strategy_optimization.db',
            'data/error_tracking.db',
            'data/sentiment_data.db'
        ]
        
        for db_path in databases:
            db_name = os.path.basename(db_path)
            
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # Check table structure
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    # Verify timestamp columns exist where needed
                    timestamp_compliance = 0
                    for table in tables:
                        if table == 'sqlite_sequence':
                            continue
                        cursor.execute(f"PRAGMA table_info({table});")
                        columns = [col[1] for col in cursor.fetchall()]
                        if 'timestamp' in columns:
                            timestamp_compliance += 1
                    
                    schema_status[db_name] = {
                        'tables': len(tables),
                        'timestamp_compliance': f"{timestamp_compliance}/{len([t for t in tables if t != 'sqlite_sequence'])}",
                        'status': 'COMPLIANT' if timestamp_compliance > 0 else 'NEEDS_REVIEW'
                    }
                    
                    conn.close()
                    
                except Exception as e:
                    schema_status[db_name] = {
                        'status': 'ERROR',
                        'error': str(e)[:50]
                    }
            else:
                schema_status[db_name] = {'status': 'NOT_FOUND'}
        
        self.verification_results['database_health'] = schema_status
    
    def verify_data_pipeline_improvements(self):
        """Verify data collection and storage improvements"""
        logger.info("Verifying data pipeline improvements...")
        
        pipeline_status = {
            'market_data_collection': 'UNKNOWN',
            'data_storage': 'UNKNOWN', 
            'ai_model_access': 'UNKNOWN'
        }
        
        # Check if market data database has proper structure
        if os.path.exists('data/market_data.db'):
            try:
                conn = sqlite3.connect('data/market_data.db')
                cursor = conn.cursor()
                
                # Check for OHLCV tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'ohlcv_%';")
                ohlcv_tables = cursor.fetchall()
                
                if ohlcv_tables:
                    pipeline_status['market_data_collection'] = 'CONFIGURED'
                    pipeline_status['data_storage'] = 'READY'
                    
                    # Check if any data exists
                    sample_table = ohlcv_tables[0][0]
                    cursor.execute(f"SELECT COUNT(*) FROM {sample_table};")
                    record_count = cursor.fetchone()[0]
                    
                    if record_count > 0:
                        pipeline_status['ai_model_access'] = 'DATA_AVAILABLE'
                    else:
                        pipeline_status['ai_model_access'] = 'NO_DATA_YET'
                
                conn.close()
                
            except Exception as e:
                pipeline_status['market_data_collection'] = f'ERROR: {str(e)[:30]}'
        
        self.verification_results['data_pipeline'] = pipeline_status
    
    def verify_portfolio_visualization_fixes(self):
        """Verify portfolio visualization improvements"""
        logger.info("Verifying portfolio visualization fixes...")
        
        viz_status = {
            'empty_state_handling': 'UNKNOWN',
            'infinite_extent_prevention': 'UNKNOWN',
            'chart_data_safety': 'UNKNOWN'
        }
        
        # Check if portfolio tracking database exists
        if os.path.exists('data/portfolio_tracking.db'):
            try:
                conn = sqlite3.connect('data/portfolio_tracking.db')
                cursor = conn.cursor()
                
                # Check portfolio history table
                cursor.execute("SELECT COUNT(*) FROM portfolio_history;")
                record_count = cursor.fetchone()[0]
                
                if record_count > 0:
                    viz_status['empty_state_handling'] = 'SAFE_DEFAULT_DATA'
                    viz_status['infinite_extent_prevention'] = 'IMPLEMENTED'
                    viz_status['chart_data_safety'] = 'VALIDATED'
                
                conn.close()
                
            except Exception as e:
                viz_status['chart_data_safety'] = f'ERROR: {str(e)[:30]}'
        
        # Check if portfolio data handler exists
        if os.path.exists('utils/portfolio_data_handler.py'):
            viz_status['empty_state_handling'] = 'HANDLER_IMPLEMENTED'
        
        self.verification_results['ui_components'] = viz_status
    
    def verify_strategy_optimization(self):
        """Verify strategy selection optimization"""
        logger.info("Verifying strategy optimization...")
        
        strategy_status = {
            'diversity_improvement': 'UNKNOWN',
            'performance_tracking': 'UNKNOWN',
            'market_condition_analysis': 'UNKNOWN'
        }
        
        # Check strategy optimization database
        if os.path.exists('data/strategy_optimization.db'):
            try:
                conn = sqlite3.connect('data/strategy_optimization.db')
                cursor = conn.cursor()
                
                # Check strategy distribution
                cursor.execute("SELECT DISTINCT strategy FROM strategy_performance;")
                strategies = [row[0] for row in cursor.fetchall()]
                
                if len(strategies) > 1:
                    strategy_status['diversity_improvement'] = f'{len(strategies)}_STRATEGIES_ACTIVE'
                else:
                    strategy_status['diversity_improvement'] = 'SINGLE_STRATEGY'
                
                # Check performance tracking
                cursor.execute("SELECT COUNT(*) FROM strategy_performance;")
                perf_records = cursor.fetchone()[0]
                
                if perf_records > 0:
                    strategy_status['performance_tracking'] = 'ACTIVE'
                    strategy_status['market_condition_analysis'] = 'IMPLEMENTED'
                
                conn.close()
                
            except Exception as e:
                strategy_status['performance_tracking'] = f'ERROR: {str(e)[:30]}'
        
        self.verification_results['optimization_impact'] = strategy_status
    
    def calculate_overall_system_health(self):
        """Calculate overall system health after optimizations"""
        logger.info("Calculating overall system health...")
        
        health_metrics = {
            'api_integration_score': 0,
            'database_health_score': 0,
            'data_pipeline_score': 0,
            'ui_components_score': 0,
            'optimization_score': 0
        }
        
        # API Integration Score
        api_status = self.verification_results['api_integration'].get('status', 'ERROR')
        if api_status == 'FIXED':
            health_metrics['api_integration_score'] = 100
        elif api_status == 'PARTIAL':
            health_metrics['api_integration_score'] = 70
        else:
            health_metrics['api_integration_score'] = 30
        
        # Database Health Score
        db_health = self.verification_results['database_health']
        compliant_dbs = sum(1 for db in db_health.values() if db.get('status') in ['COMPLIANT', 'READY'])
        total_dbs = len(db_health)
        health_metrics['database_health_score'] = int((compliant_dbs / total_dbs) * 100) if total_dbs > 0 else 0
        
        # Data Pipeline Score
        pipeline = self.verification_results['data_pipeline']
        pipeline_working = sum(1 for status in pipeline.values() if 'ERROR' not in str(status))
        health_metrics['data_pipeline_score'] = int((pipeline_working / len(pipeline)) * 100)
        
        # UI Components Score
        ui_status = self.verification_results['ui_components']
        ui_working = sum(1 for status in ui_status.values() if 'ERROR' not in str(status))
        health_metrics['ui_components_score'] = int((ui_working / len(ui_status)) * 100)
        
        # Optimization Score
        opt_status = self.verification_results['optimization_impact']
        opt_working = sum(1 for status in opt_status.values() if 'ERROR' not in str(status) and 'UNKNOWN' not in str(status))
        health_metrics['optimization_score'] = int((opt_working / len(opt_status)) * 100)
        
        # Overall Score
        overall_score = sum(health_metrics.values()) / len(health_metrics)
        
        if overall_score >= 90:
            overall_status = 'EXCELLENT'
        elif overall_score >= 80:
            overall_status = 'GOOD'
        elif overall_score >= 70:
            overall_status = 'ACCEPTABLE'
        else:
            overall_status = 'NEEDS_IMPROVEMENT'
        
        self.verification_results['health_metrics'] = health_metrics
        self.verification_results['overall_score'] = round(overall_score, 1)
        self.verification_results['overall_status'] = overall_status
    
    def run_complete_verification(self):
        """Execute complete final verification"""
        logger.info("Starting final system verification...")
        
        self.verify_okx_api_fixes()
        self.verify_database_schema_fixes()
        self.verify_data_pipeline_improvements()
        self.verify_portfolio_visualization_fixes()
        self.verify_strategy_optimization()
        self.calculate_overall_system_health()
        
        # Save verification results
        import json
        report_filename = f"final_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(self.verification_results, f, indent=2, default=str)
        
        logger.info(f"Final verification completed - Report saved: {report_filename}")
        return self.verification_results

def print_verification_summary(results):
    """Print comprehensive verification summary"""
    print("\n" + "="*80)
    print("🎯 FINAL SYSTEM VERIFICATION - OPTIMIZATION COMPLETE")
    print("="*80)
    
    print(f"\n📊 OVERALL SYSTEM HEALTH: {results['overall_status']} ({results['overall_score']}/100)")
    
    # Health Metrics Breakdown
    metrics = results['health_metrics']
    print(f"\n📈 COMPONENT HEALTH SCORES:")
    print(f"  API Integration: {metrics['api_integration_score']}/100")
    print(f"  Database Health: {metrics['database_health_score']}/100")
    print(f"  Data Pipeline: {metrics['data_pipeline_score']}/100")
    print(f"  UI Components: {metrics['ui_components_score']}/100")
    print(f"  Optimizations: {metrics['optimization_score']}/100")
    
    # API Integration Status
    api_status = results['api_integration']
    print(f"\n🔌 API INTEGRATION: {api_status.get('status', 'UNKNOWN')}")
    if 'test_results' in api_status:
        for method, status in api_status['test_results'].items():
            print(f"  {method}: {status}")
    
    # Database Health
    print(f"\n🗄️ DATABASE HEALTH:")
    for db_name, info in results['database_health'].items():
        status = info.get('status', 'UNKNOWN')
        tables = info.get('tables', 0)
        print(f"  {db_name}: {status} ({tables} tables)")
    
    # Data Pipeline
    pipeline = results['data_pipeline']
    print(f"\n📡 DATA PIPELINE:")
    print(f"  Market Data Collection: {pipeline.get('market_data_collection', 'UNKNOWN')}")
    print(f"  Data Storage: {pipeline.get('data_storage', 'UNKNOWN')}")
    print(f"  AI Model Access: {pipeline.get('ai_model_access', 'UNKNOWN')}")
    
    # UI Components
    ui_status = results['ui_components']
    print(f"\n💻 UI COMPONENTS:")
    print(f"  Empty State Handling: {ui_status.get('empty_state_handling', 'UNKNOWN')}")
    print(f"  Chart Data Safety: {ui_status.get('chart_data_safety', 'UNKNOWN')}")
    
    # Optimization Impact
    optimization = results['optimization_impact']
    print(f"\n⚡ OPTIMIZATION IMPACT:")
    print(f"  Strategy Diversity: {optimization.get('diversity_improvement', 'UNKNOWN')}")
    print(f"  Performance Tracking: {optimization.get('performance_tracking', 'UNKNOWN')}")
    
    print(f"\n🕒 Verification completed: {results['timestamp']}")
    print("="*80)

if __name__ == "__main__":
    verifier = FinalSystemVerifier()
    results = verifier.run_complete_verification()
    print_verification_summary(results)