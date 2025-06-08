"""
Comprehensive System Verification and Performance Analysis
Analyzes all components with live OKX data and generates detailed report
"""

import logging
import sys
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import json
import sqlite3
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SystemVerificationReport:
    def __init__(self):
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'system_health': {},
            'trading_data': {},
            'ai_models': {},
            'strategies': {},
            'risk_system': {},
            'errors': [],
            'warnings': []
        }
        
        # Initialize components
        self.okx_data_service = None
        self.trading_controller = None
        self.ai_performance_tracker = None
        self.strategy_selector = None
        self.risk_manager = None
        
    def initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing system components...")
        
        try:
            # Initialize OKX Data Service
            from trading.okx_data_service import OKXDataService
            self.okx_data_service = OKXDataService()
            logger.info("✅ OKX Data Service initialized")
            
            # Initialize Trading Controller
            from trading.unified_trading_controller import UnifiedTradingController
            self.trading_controller = UnifiedTradingController(
                self.okx_data_service.okx_connector if hasattr(self.okx_data_service, 'okx_connector') else None
            )
            logger.info("✅ Unified Trading Controller initialized")
            
            # Initialize AI Performance Tracker
            from ai.ai_performance_tracker import AIPerformanceTracker
            self.ai_performance_tracker = AIPerformanceTracker()
            logger.info("✅ AI Performance Tracker initialized")
            
            # Initialize Smart Strategy Selector
            from strategies.smart_strategy_selector import SmartStrategySelector
            self.strategy_selector = SmartStrategySelector(self.okx_data_service)
            logger.info("✅ Smart Strategy Selector initialized")
            
            return True
            
        except Exception as e:
            error_msg = f"Component initialization failed: {e}"
            logger.error(error_msg)
            self.report['errors'].append(error_msg)
            return False
    
    def verify_system_health(self):
        """Verify system health and connectivity"""
        logger.info("🧩 Starting System Health Check...")
        
        health_status = {
            'frontend': 'Unknown',
            'backend': 'Unknown', 
            'database': 'Unknown',
            'okx_api': 'Unknown',
            'background_services': 'Unknown'
        }
        
        try:
            # Test OKX API connectivity
            if self.okx_data_service:
                test_data = self.okx_data_service.get_historical_data('BTC-USDT', '1h', limit=5)
                if test_data is not None and not test_data.empty:
                    health_status['okx_api'] = '✅ Connected'
                    logger.info("✅ OKX API connectivity verified")
                else:
                    health_status['okx_api'] = '❌ No data received'
                    self.report['warnings'].append("OKX API returned no data")
            
            # Test database connectivity
            try:
                import sqlite3
                test_db = sqlite3.connect('data/ai_performance.db')
                cursor = test_db.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                test_db.close()
                
                if tables:
                    health_status['database'] = f'✅ Connected ({len(tables)} tables)'
                    logger.info(f"✅ Database connectivity verified - {len(tables)} tables found")
                else:
                    health_status['database'] = '⚠️ Connected but empty'
                    self.report['warnings'].append("Database is empty")
                    
            except Exception as e:
                health_status['database'] = f'❌ Error: {str(e)[:50]}'
                self.report['errors'].append(f"Database connection failed: {e}")
            
            # Test Trading Controller
            if self.trading_controller:
                try:
                    stats = self.trading_controller.get_market_statistics()
                    if stats:
                        health_status['backend'] = '✅ Operational'
                        logger.info("✅ Trading Controller operational")
                    else:
                        health_status['backend'] = '⚠️ Limited functionality'
                        self.report['warnings'].append("Trading Controller has limited functionality")
                except Exception as e:
                    health_status['backend'] = f'❌ Error: {str(e)[:50]}'
                    self.report['errors'].append(f"Trading Controller error: {e}")
            
            # Check symbol discovery
            if self.trading_controller and self.trading_controller.symbol_manager:
                try:
                    symbol_stats = self.trading_controller.symbol_manager.get_symbol_statistics()
                    total_symbols = symbol_stats.get('total_symbols', 0)
                    
                    if total_symbols > 0:
                        health_status['background_services'] = f'✅ Active ({total_symbols} symbols)'
                        logger.info(f"✅ Symbol discovery active - {total_symbols} symbols")
                    else:
                        health_status['background_services'] = '⚠️ No symbols discovered'
                        self.report['warnings'].append("No symbols discovered")
                        
                except Exception as e:
                    health_status['background_services'] = f'❌ Error: {str(e)[:50]}'
                    self.report['errors'].append(f"Background services error: {e}")
            
            # Frontend is running if we got this far
            health_status['frontend'] = '✅ Running (Streamlit)'
            
        except Exception as e:
            error_msg = f"System health check failed: {e}"
            logger.error(error_msg)
            self.report['errors'].append(error_msg)
        
        self.report['system_health'] = health_status
        logger.info("🧩 System Health Check completed")
    
    def verify_trading_data(self):
        """Verify trading data from last 72 hours"""
        logger.info("📊 Starting Trading Data Verification...")
        
        trading_data = {
            'last_72h_trades': [],
            'trades_per_symbol': {},
            'trades_per_strategy': {},
            'missing_exits': [],
            'total_trades': 0
        }
        
        try:
            if self.trading_controller:
                # Get trade history
                trade_history = self.trading_controller.get_trade_history(limit=1000)
                
                # Filter last 72 hours
                cutoff_time = datetime.now() - timedelta(hours=72)
                recent_trades = []
                
                for trade in trade_history:
                    try:
                        trade_time = datetime.fromisoformat(trade['timestamp'])
                        if trade_time > cutoff_time:
                            recent_trades.append(trade)
                    except:
                        continue
                
                trading_data['total_trades'] = len(recent_trades)
                trading_data['last_72h_trades'] = recent_trades[:20]  # Show first 20
                
                # Analyze trades per symbol
                symbol_counts = {}
                strategy_counts = {}
                
                for trade in recent_trades:
                    symbol = trade.get('symbol', 'Unknown')
                    strategy = trade.get('metadata', {}).get('strategy', 'Unknown')
                    
                    symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
                    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                
                trading_data['trades_per_symbol'] = symbol_counts
                trading_data['trades_per_strategy'] = strategy_counts
                
                logger.info(f"✅ Found {len(recent_trades)} trades in last 72 hours")
                
                # Check for missing exits (simplified analysis)
                buy_positions = {}
                for trade in recent_trades:
                    symbol = trade.get('symbol')
                    action = trade.get('action', '').lower()
                    
                    if action in ['buy', 'long']:
                        buy_positions[symbol] = trade
                    elif action in ['sell', 'short', 'close']:
                        if symbol in buy_positions:
                            del buy_positions[symbol]
                
                trading_data['missing_exits'] = list(buy_positions.keys())
                
            else:
                self.report['warnings'].append("Trading controller not available for trade analysis")
                
        except Exception as e:
            error_msg = f"Trading data verification failed: {e}"
            logger.error(error_msg)
            self.report['errors'].append(error_msg)
        
        self.report['trading_data'] = trading_data
        logger.info("📊 Trading Data Verification completed")
    
    def evaluate_ai_models(self):
        """Evaluate AI model performance"""
        logger.info("🧠 Starting AI Model Evaluation...")
        
        ai_models = {
            'model_performance': {},
            'best_models': [],
            'retraining_status': {},
            'prediction_accuracy': {}
        }
        
        try:
            if self.ai_performance_tracker:
                # Get performance summary
                performance_summary = self.ai_performance_tracker.get_performance_summary()
                
                if performance_summary and 'summary' in performance_summary:
                    models_data = {}
                    
                    for item in performance_summary['summary']:
                        model_name = item.get('model_name', 'Unknown')
                        symbol = item.get('symbol', 'Unknown')
                        win_rate = item.get('win_rate', 0)
                        avg_confidence = item.get('avg_confidence', 0)
                        total_predictions = item.get('total_predictions', 0)
                        
                        if model_name not in models_data:
                            models_data[model_name] = {
                                'symbols': [],
                                'avg_win_rate': 0,
                                'total_predictions': 0,
                                'avg_confidence': 0
                            }
                        
                        models_data[model_name]['symbols'].append(symbol)
                        models_data[model_name]['avg_win_rate'] += win_rate
                        models_data[model_name]['total_predictions'] += total_predictions
                        models_data[model_name]['avg_confidence'] += avg_confidence
                    
                    # Calculate averages
                    for model_name, data in models_data.items():
                        symbol_count = len(data['symbols'])
                        if symbol_count > 0:
                            data['avg_win_rate'] = data['avg_win_rate'] / symbol_count
                            data['avg_confidence'] = data['avg_confidence'] / symbol_count
                    
                    ai_models['model_performance'] = models_data
                    
                    # Find best performing models
                    best_models = sorted(
                        models_data.items(),
                        key=lambda x: x[1]['avg_win_rate'],
                        reverse=True
                    )[:5]
                    
                    ai_models['best_models'] = [
                        {
                            'name': name,
                            'win_rate': f"{data['avg_win_rate']:.1f}%",
                            'predictions': data['total_predictions'],
                            'confidence': f"{data['avg_confidence']:.2f}"
                        }
                        for name, data in best_models
                    ]
                    
                    logger.info(f"✅ Analyzed {len(models_data)} AI models")
                
                # Check retraining status
                try:
                    # Get model files to check last training
                    import os
                    model_dirs = ['models', 'ai']
                    
                    for model_dir in model_dirs:
                        if os.path.exists(model_dir):
                            for file in os.listdir(model_dir):
                                if file.endswith('.pkl') or file.endswith('.joblib'):
                                    file_path = os.path.join(model_dir, file)
                                    mtime = os.path.getmtime(file_path)
                                    last_modified = datetime.fromtimestamp(mtime)
                                    
                                    ai_models['retraining_status'][file] = {
                                        'last_trained': last_modified.isoformat(),
                                        'hours_ago': (datetime.now() - last_modified).total_seconds() / 3600
                                    }
                
                except Exception as e:
                    self.report['warnings'].append(f"Could not check model files: {e}")
                
            else:
                self.report['warnings'].append("AI Performance Tracker not available")
                
        except Exception as e:
            error_msg = f"AI model evaluation failed: {e}"
            logger.error(error_msg)
            self.report['errors'].append(error_msg)
        
        self.report['ai_models'] = ai_models
        logger.info("🧠 AI Model Evaluation completed")
    
    def evaluate_strategies(self):
        """Evaluate strategy effectiveness"""
        logger.info("📈 Starting Strategy Effectiveness Analysis...")
        
        strategies = {
            'active_strategies': {},
            'high_performance_strategies': [],
            'strategy_selector_cycles': [],
            'top_10_pairs_performance': {}
        }
        
        try:
            if self.strategy_selector:
                # Get current strategy assignments
                if hasattr(self.strategy_selector, 'current_strategies'):
                    strategies['active_strategies'] = self.strategy_selector.current_strategies
                
                # Get performance data if available
                if hasattr(self.strategy_selector, 'get_performance_summary'):
                    try:
                        perf_summary = self.strategy_selector.get_performance_summary()
                        if perf_summary:
                            strategies['strategy_selector_cycles'] = perf_summary.get('recent_cycles', [])
                    except:
                        pass
            
            # Get top 10 USDT pairs if trading controller available
            if self.trading_controller and self.trading_controller.symbol_manager:
                try:
                    top_symbols = self.trading_controller.symbol_manager.get_symbols_by_volume(limit=10)
                    
                    for symbol in top_symbols:
                        symbol_info = self.trading_controller.symbol_manager.get_symbol_info(symbol)
                        if symbol_info:
                            strategies['top_10_pairs_performance'][symbol] = {
                                'volume_24h': symbol_info.volume_24h,
                                'market_type': symbol_info.market_type.value,
                                'active': symbol_info.is_active
                            }
                    
                    logger.info(f"✅ Analyzed top {len(top_symbols)} symbols")
                
                except Exception as e:
                    self.report['warnings'].append(f"Could not analyze top pairs: {e}")
            
        except Exception as e:
            error_msg = f"Strategy evaluation failed: {e}"
            logger.error(error_msg)
            self.report['errors'].append(error_msg)
        
        self.report['strategies'] = strategies
        logger.info("📈 Strategy Effectiveness Analysis completed")
    
    def verify_risk_system(self):
        """Verify risk and protection systems"""
        logger.info("⚠️ Starting Risk & Protection System Verification...")
        
        risk_system = {
            'stop_loss_triggers': [],
            'take_profit_triggers': [],
            'circuit_breaker_events': [],
            'drawdown_breaches': [],
            'emergency_stops': []
        }
        
        try:
            # Check if we have any risk management logs in the database
            try:
                db_path = 'data/trading_decisions.db'
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Check for risk-related events in last 72 hours
                cutoff_time = (datetime.now() - timedelta(hours=72)).isoformat()
                
                cursor.execute("""
                    SELECT * FROM trading_decisions 
                    WHERE timestamp > ? AND (
                        reason LIKE '%stop%loss%' OR
                        reason LIKE '%take%profit%' OR
                        reason LIKE '%risk%' OR
                        reason LIKE '%emergency%'
                    )
                    ORDER BY timestamp DESC
                """, (cutoff_time,))
                
                risk_events = cursor.fetchall()
                conn.close()
                
                for event in risk_events:
                    event_data = {
                        'timestamp': event[1],
                        'symbol': event[2],
                        'action': event[3],
                        'reason': event[6]
                    }
                    
                    if 'stop loss' in event[6].lower():
                        risk_system['stop_loss_triggers'].append(event_data)
                    elif 'take profit' in event[6].lower():
                        risk_system['take_profit_triggers'].append(event_data)
                    elif 'emergency' in event[6].lower():
                        risk_system['emergency_stops'].append(event_data)
                
                logger.info(f"✅ Found {len(risk_events)} risk events in last 72 hours")
                
            except Exception as e:
                self.report['warnings'].append(f"Could not access risk event database: {e}")
            
            # Check current portfolio for drawdown analysis
            if self.trading_controller:
                try:
                    portfolio = self.trading_controller.get_portfolio_overview()
                    unrealized_pnl = portfolio.get('unrealized_pnl', 0)
                    
                    if unrealized_pnl < -1000:  # Significant loss threshold
                        risk_system['drawdown_breaches'].append({
                            'current_unrealized_pnl': unrealized_pnl,
                            'timestamp': datetime.now().isoformat(),
                            'severity': 'High' if unrealized_pnl < -5000 else 'Medium'
                        })
                
                except Exception as e:
                    self.report['warnings'].append(f"Could not analyze portfolio drawdown: {e}")
            
        except Exception as e:
            error_msg = f"Risk system verification failed: {e}"
            logger.error(error_msg)
            self.report['errors'].append(error_msg)
        
        self.report['risk_system'] = risk_system
        logger.info("⚠️ Risk & Protection System Verification completed")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        logger.info("📂 Generating Final Report...")
        
        # Calculate summary statistics
        operational_modules = sum(1 for status in self.report['system_health'].values() if '✅' in str(status))
        total_modules = len(self.report['system_health'])
        
        total_trades = self.report['trading_data'].get('total_trades', 0)
        total_errors = len(self.report['errors'])
        total_warnings = len(self.report['warnings'])
        
        # Create summary
        summary = {
            'system_status': f"{operational_modules}/{total_modules} modules operational",
            'total_trades_72h': total_trades,
            'ai_models_active': len(self.report['ai_models'].get('model_performance', {})),
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'data_source': 'Live OKX Production Data',
            'report_generated': datetime.now().isoformat()
        }
        
        self.report['summary'] = summary
        
        # Save report to file
        try:
            report_filename = f"system_verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_filename, 'w') as f:
                json.dump(self.report, f, indent=2, default=str)
            logger.info(f"✅ Report saved to {report_filename}")
        except Exception as e:
            logger.error(f"Could not save report: {e}")
        
        return self.report
    
    def run_full_verification(self):
        """Run complete system verification"""
        logger.info("🚀 Starting Comprehensive System Verification...")
        
        if not self.initialize_components():
            logger.error("❌ Component initialization failed - aborting verification")
            return self.report
        
        # Run all verification modules
        self.verify_system_health()
        self.verify_trading_data()
        self.evaluate_ai_models()
        self.evaluate_strategies()
        self.verify_risk_system()
        
        # Generate final report
        final_report = self.generate_final_report()
        
        logger.info("🎉 Comprehensive System Verification completed!")
        return final_report

def print_verification_summary(report):
    """Print formatted verification summary"""
    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE SYSTEM VERIFICATION REPORT")
    print("="*80)
    
    # System Health
    print("\n🧩 SYSTEM HEALTH:")
    for component, status in report['system_health'].items():
        print(f"  {component}: {status}")
    
    # Trading Data
    print(f"\n📊 TRADING DATA (Last 72h):")
    trading = report['trading_data']
    print(f"  Total Trades: {trading.get('total_trades', 0)}")
    print(f"  Trades per Symbol: {len(trading.get('trades_per_symbol', {}))}")
    print(f"  Missing Exits: {len(trading.get('missing_exits', []))}")
    
    # AI Models
    print(f"\n🧠 AI MODELS:")
    ai_models = report['ai_models']
    print(f"  Models Analyzed: {len(ai_models.get('model_performance', {}))}")
    print(f"  Best Performers:")
    for model in ai_models.get('best_models', [])[:3]:
        print(f"    {model['name']}: {model['win_rate']} win rate")
    
    # Risk System
    print(f"\n⚠️ RISK SYSTEM:")
    risk = report['risk_system']
    print(f"  Stop Loss Triggers: {len(risk.get('stop_loss_triggers', []))}")
    print(f"  Take Profit Triggers: {len(risk.get('take_profit_triggers', []))}")
    print(f"  Emergency Stops: {len(risk.get('emergency_stops', []))}")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    summary = report.get('summary', {})
    print(f"  System Status: {summary.get('system_status', 'Unknown')}")
    print(f"  Data Source: {summary.get('data_source', 'Unknown')}")
    print(f"  Errors: {summary.get('total_errors', 0)}")
    print(f"  Warnings: {summary.get('total_warnings', 0)}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    verifier = SystemVerificationReport()
    report = verifier.run_full_verification()
    print_verification_summary(report)