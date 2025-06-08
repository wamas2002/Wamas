"""
Production System Analysis with Live OKX Data
Comprehensive verification and performance analysis
"""

import logging
import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionAnalysis:
    def __init__(self):
        self.report = {
            'timestamp': datetime.now().isoformat(),
            'system_health': {},
            'trading_performance': {},
            'ai_models': {},
            'strategies': {},
            'risk_analysis': {},
            'errors': [],
            'warnings': []
        }
        
    def check_okx_connectivity(self):
        """Verify OKX API connectivity with live data"""
        logger.info("Testing OKX API connectivity...")
        
        try:
            from trading.okx_data_service import OKXDataService
            okx_service = OKXDataService()
            
            # Test major pairs
            test_symbols = ['BTC-USDT', 'ETH-USDT', 'BNB-USDT']
            connectivity_results = {}
            
            for symbol in test_symbols:
                try:
                    data = okx_service.get_historical_data(symbol, '1h', limit=5)
                    if data is not None and not data.empty:
                        latest_price = data['close'].iloc[-1]
                        connectivity_results[symbol] = {
                            'status': 'Connected',
                            'latest_price': float(latest_price),
                            'data_points': len(data),
                            'last_update': data.index[-1].isoformat()
                        }
                        logger.info(f"✅ {symbol}: ${latest_price:.2f}")
                    else:
                        connectivity_results[symbol] = {'status': 'No Data'}
                        self.report['warnings'].append(f"No data received for {symbol}")
                        
                except Exception as e:
                    connectivity_results[symbol] = {'status': f'Error: {str(e)[:50]}'}
                    self.report['errors'].append(f"OKX API error for {symbol}: {e}")
            
            self.report['system_health']['okx_connectivity'] = connectivity_results
            
            # Test symbol discovery
            try:
                if hasattr(okx_service, 'okx_connector'):
                    # Check if we can get tickers
                    test_ticker = okx_service.get_ticker('BTC-USDT')
                    if test_ticker:
                        self.report['system_health']['market_data'] = {
                            'status': 'Operational',
                            'last_price': test_ticker.get('last', 'N/A'),
                            'volume_24h': test_ticker.get('vol24h', 'N/A')
                        }
                    else:
                        self.report['warnings'].append("Ticker data not available")
                        
            except Exception as e:
                self.report['warnings'].append(f"Market data test failed: {e}")
                
        except Exception as e:
            self.report['errors'].append(f"OKX service initialization failed: {e}")
            logger.error(f"OKX connectivity test failed: {e}")
    
    def analyze_database_health(self):
        """Analyze database connectivity and data integrity"""
        logger.info("Analyzing database health...")
        
        db_status = {}
        
        # Check main databases
        databases = [
            'data/ai_performance.db',
            'data/trading_decisions.db',
            'data/market_data.db',
            'data/news_sentiment.db'
        ]
        
        for db_path in databases:
            db_name = os.path.basename(db_path)
            
            try:
                if os.path.exists(db_path):
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # Get table count
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    
                    # Get total record count
                    total_records = 0
                    for table in tables:
                        try:
                            cursor.execute(f"SELECT COUNT(*) FROM {table[0]};")
                            count = cursor.fetchone()[0]
                            total_records += count
                        except:
                            pass
                    
                    # Get file size
                    file_size = os.path.getsize(db_path)
                    
                    db_status[db_name] = {
                        'status': 'Connected',
                        'tables': len(tables),
                        'total_records': total_records,
                        'size_mb': round(file_size / (1024*1024), 2)
                    }
                    
                    conn.close()
                    logger.info(f"✅ {db_name}: {len(tables)} tables, {total_records} records")
                    
                else:
                    db_status[db_name] = {'status': 'Not Found'}
                    self.report['warnings'].append(f"Database not found: {db_name}")
                    
            except Exception as e:
                db_status[db_name] = {'status': f'Error: {str(e)[:50]}'}
                self.report['errors'].append(f"Database error {db_name}: {e}")
        
        self.report['system_health']['databases'] = db_status
    
    def analyze_trading_performance(self):
        """Analyze trading performance from last 72 hours"""
        logger.info("Analyzing trading performance...")
        
        performance_data = {
            'trades_72h': [],
            'symbol_performance': {},
            'strategy_performance': {},
            'pnl_analysis': {},
            'execution_analysis': {}
        }
        
        try:
            # Check trading decisions database
            db_path = 'data/trading_decisions.db'
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                
                # Get trades from last 72 hours
                cutoff_time = (datetime.now() - timedelta(hours=72)).isoformat()
                
                query = """
                SELECT * FROM trading_decisions 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
                """
                
                trades_df = pd.read_sql_query(query, conn, params=(cutoff_time,))
                conn.close()
                
                if not trades_df.empty:
                    # Basic statistics
                    performance_data['total_trades'] = len(trades_df)
                    performance_data['trades_72h'] = trades_df.head(20).to_dict('records')
                    
                    # Symbol analysis
                    symbol_counts = trades_df['symbol'].value_counts().to_dict()
                    performance_data['symbol_performance'] = symbol_counts
                    
                    # Strategy analysis
                    if 'strategy' in trades_df.columns:
                        strategy_counts = trades_df['strategy'].value_counts().to_dict()
                        performance_data['strategy_performance'] = strategy_counts
                    
                    # Action analysis
                    action_counts = trades_df['action'].value_counts().to_dict()
                    performance_data['execution_analysis'] = {
                        'actions': action_counts,
                        'unique_symbols': trades_df['symbol'].nunique(),
                        'avg_confidence': trades_df['confidence'].mean() if 'confidence' in trades_df.columns else 'N/A'
                    }
                    
                    logger.info(f"✅ Found {len(trades_df)} trades in last 72 hours")
                    
                    # Analyze potential missing exits
                    buy_actions = trades_df[trades_df['action'].isin(['buy', 'long'])]
                    sell_actions = trades_df[trades_df['action'].isin(['sell', 'short', 'close'])]
                    
                    open_positions = []
                    for symbol in buy_actions['symbol'].unique():
                        symbol_buys = len(buy_actions[buy_actions['symbol'] == symbol])
                        symbol_sells = len(sell_actions[sell_actions['symbol'] == symbol])
                        
                        if symbol_buys > symbol_sells:
                            open_positions.append({
                                'symbol': symbol,
                                'potential_open': symbol_buys - symbol_sells
                            })
                    
                    performance_data['potential_open_positions'] = open_positions
                    
                else:
                    performance_data['total_trades'] = 0
                    self.report['warnings'].append("No trades found in last 72 hours")
                    
            else:
                self.report['warnings'].append("Trading decisions database not found")
                
        except Exception as e:
            self.report['errors'].append(f"Trading performance analysis failed: {e}")
            logger.error(f"Trading analysis error: {e}")
        
        self.report['trading_performance'] = performance_data
    
    def analyze_ai_models(self):
        """Analyze AI model performance and status"""
        logger.info("Analyzing AI model performance...")
        
        ai_analysis = {
            'model_performance': {},
            'prediction_accuracy': {},
            'model_files': {},
            'retraining_status': {}
        }
        
        try:
            # Check AI performance database
            db_path = 'data/ai_performance.db'
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # Get table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    try:
                        # Get recent performance data
                        query = f"SELECT * FROM {table} ORDER BY timestamp DESC LIMIT 100"
                        df = pd.read_sql_query(query, conn)
                        
                        if not df.empty:
                            # Calculate performance metrics
                            if 'win_rate' in df.columns:
                                avg_win_rate = df['win_rate'].mean()
                                total_predictions = df['total_predictions'].sum() if 'total_predictions' in df.columns else len(df)
                                
                                ai_analysis['model_performance'][table] = {
                                    'avg_win_rate': round(avg_win_rate, 2),
                                    'total_predictions': int(total_predictions),
                                    'recent_records': len(df),
                                    'last_update': df['timestamp'].max() if 'timestamp' in df.columns else 'Unknown'
                                }
                                
                    except Exception as e:
                        self.report['warnings'].append(f"Could not analyze table {table}: {e}")
                
                conn.close()
                logger.info(f"✅ Analyzed {len(ai_analysis['model_performance'])} AI model tables")
                
            else:
                self.report['warnings'].append("AI performance database not found")
            
            # Check model files
            model_dirs = ['models', 'ai', 'datasets']
            for model_dir in model_dirs:
                if os.path.exists(model_dir):
                    for file in os.listdir(model_dir):
                        if file.endswith(('.pkl', '.joblib', '.h5', '.pt')):
                            file_path = os.path.join(model_dir, file)
                            mtime = os.path.getmtime(file_path)
                            last_modified = datetime.fromtimestamp(mtime)
                            
                            ai_analysis['model_files'][file] = {
                                'last_modified': last_modified.isoformat(),
                                'hours_ago': round((datetime.now() - last_modified).total_seconds() / 3600, 1),
                                'size_mb': round(os.path.getsize(file_path) / (1024*1024), 2)
                            }
            
        except Exception as e:
            self.report['errors'].append(f"AI model analysis failed: {e}")
            logger.error(f"AI analysis error: {e}")
        
        self.report['ai_models'] = ai_analysis
    
    def analyze_strategy_effectiveness(self):
        """Analyze strategy assignments and effectiveness"""
        logger.info("Analyzing strategy effectiveness...")
        
        strategy_analysis = {
            'active_strategies': {},
            'strategy_switches': [],
            'performance_by_strategy': {}
        }
        
        try:
            # Check for strategy assignment logs in trading decisions
            db_path = 'data/trading_decisions.db'
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                
                # Look for strategy-related decisions
                query = """
                SELECT symbol, action, reason, timestamp, strategy, confidence
                FROM trading_decisions 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
                """
                
                cutoff_time = (datetime.now() - timedelta(hours=72)).isoformat()
                df = pd.read_sql_query(query, conn, params=(cutoff_time,))
                conn.close()
                
                if not df.empty and 'strategy' in df.columns:
                    # Current strategy assignments
                    latest_strategies = df.groupby('symbol')['strategy'].last().to_dict()
                    strategy_analysis['active_strategies'] = latest_strategies
                    
                    # Strategy performance
                    strategy_performance = {}
                    for strategy in df['strategy'].unique():
                        if pd.notna(strategy):
                            strategy_trades = df[df['strategy'] == strategy]
                            strategy_performance[strategy] = {
                                'total_trades': len(strategy_trades),
                                'symbols_used': strategy_trades['symbol'].nunique(),
                                'avg_confidence': strategy_trades['confidence'].mean() if 'confidence' in strategy_trades.columns else 'N/A'
                            }
                    
                    strategy_analysis['performance_by_strategy'] = strategy_performance
                    logger.info(f"✅ Analyzed {len(latest_strategies)} strategy assignments")
                    
                else:
                    self.report['warnings'].append("No strategy data found in trading decisions")
                    
        except Exception as e:
            self.report['errors'].append(f"Strategy analysis failed: {e}")
            logger.error(f"Strategy analysis error: {e}")
        
        self.report['strategies'] = strategy_analysis
    
    def analyze_risk_system(self):
        """Analyze risk management and protection systems"""
        logger.info("Analyzing risk management system...")
        
        risk_analysis = {
            'stop_loss_events': [],
            'take_profit_events': [],
            'risk_alerts': [],
            'drawdown_analysis': {},
            'protection_status': {}
        }
        
        try:
            # Check for risk events in trading decisions
            db_path = 'data/trading_decisions.db'
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                
                # Look for risk-related events
                cutoff_time = (datetime.now() - timedelta(hours=72)).isoformat()
                
                risk_query = """
                SELECT * FROM trading_decisions 
                WHERE timestamp > ? AND (
                    reason LIKE '%stop%loss%' OR
                    reason LIKE '%take%profit%' OR
                    reason LIKE '%risk%' OR
                    reason LIKE '%emergency%' OR
                    reason LIKE '%protection%'
                )
                ORDER BY timestamp DESC
                """
                
                risk_events = pd.read_sql_query(risk_query, conn, params=(cutoff_time,))
                conn.close()
                
                if not risk_events.empty:
                    for _, event in risk_events.iterrows():
                        event_data = {
                            'timestamp': event['timestamp'],
                            'symbol': event['symbol'],
                            'action': event['action'],
                            'reason': event['reason']
                        }
                        
                        if 'stop' in event['reason'].lower() and 'loss' in event['reason'].lower():
                            risk_analysis['stop_loss_events'].append(event_data)
                        elif 'take' in event['reason'].lower() and 'profit' in event['reason'].lower():
                            risk_analysis['take_profit_events'].append(event_data)
                        else:
                            risk_analysis['risk_alerts'].append(event_data)
                    
                    logger.info(f"✅ Found {len(risk_events)} risk events in last 72 hours")
                else:
                    logger.info("No risk events found in last 72 hours")
                    
            # Check protection system status
            risk_analysis['protection_status'] = {
                'stop_loss_active': len(risk_analysis['stop_loss_events']) > 0,
                'take_profit_active': len(risk_analysis['take_profit_events']) > 0,
                'total_risk_events': len(risk_analysis['stop_loss_events']) + len(risk_analysis['take_profit_events']) + len(risk_analysis['risk_alerts'])
            }
            
        except Exception as e:
            self.report['errors'].append(f"Risk analysis failed: {e}")
            logger.error(f"Risk analysis error: {e}")
        
        self.report['risk_analysis'] = risk_analysis
    
    def generate_comprehensive_report(self):
        """Generate final comprehensive report"""
        logger.info("Generating comprehensive report...")
        
        # Calculate summary metrics
        total_errors = len(self.report['errors'])
        total_warnings = len(self.report['warnings'])
        
        # System health summary
        okx_status = self.report['system_health'].get('okx_connectivity', {})
        connected_pairs = sum(1 for status in okx_status.values() if status.get('status') == 'Connected')
        
        # Trading summary
        trading_perf = self.report['trading_performance']
        total_trades = trading_perf.get('total_trades', 0)
        
        # AI models summary
        ai_models = self.report['ai_models'].get('model_performance', {})
        active_models = len(ai_models)
        
        # Create executive summary
        summary = {
            'report_timestamp': self.report['timestamp'],
            'data_source': 'Live OKX Production Data',
            'okx_connectivity': f"{connected_pairs} pairs connected",
            'total_trades_72h': total_trades,
            'active_ai_models': active_models,
            'risk_events_72h': len(self.report['risk_analysis'].get('stop_loss_events', [])) + len(self.report['risk_analysis'].get('take_profit_events', [])),
            'system_errors': total_errors,
            'system_warnings': total_warnings,
            'overall_status': 'Operational' if total_errors == 0 else 'Issues Detected'
        }
        
        self.report['executive_summary'] = summary
        
        # Save detailed report
        report_filename = f"production_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_filename, 'w') as f:
                json.dump(self.report, f, indent=2, default=str)
            logger.info(f"✅ Detailed report saved to {report_filename}")
        except Exception as e:
            logger.error(f"Could not save report: {e}")
        
        return self.report
    
    def run_full_analysis(self):
        """Execute complete production system analysis"""
        logger.info("🚀 Starting Production System Analysis...")
        
        self.check_okx_connectivity()
        self.analyze_database_health()
        self.analyze_trading_performance()
        self.analyze_ai_models()
        self.analyze_strategy_effectiveness()
        self.analyze_risk_system()
        
        final_report = self.generate_comprehensive_report()
        
        logger.info("🎉 Production System Analysis completed!")
        return final_report

def print_executive_summary(report):
    """Print formatted executive summary"""
    print("\n" + "="*80)
    print("📊 PRODUCTION SYSTEM ANALYSIS - EXECUTIVE SUMMARY")
    print("="*80)
    
    summary = report.get('executive_summary', {})
    
    print(f"\n📋 SYSTEM STATUS: {summary.get('overall_status', 'Unknown')}")
    print(f"📡 Data Source: {summary.get('data_source', 'Unknown')}")
    print(f"🕒 Report Time: {summary.get('report_timestamp', 'Unknown')}")
    
    print(f"\n🧩 CONNECTIVITY:")
    print(f"  OKX API: {summary.get('okx_connectivity', 'Unknown')}")
    
    print(f"\n📊 TRADING PERFORMANCE (72h):")
    print(f"  Total Trades: {summary.get('total_trades_72h', 0)}")
    
    print(f"\n🧠 AI SYSTEM:")
    print(f"  Active Models: {summary.get('active_ai_models', 0)}")
    
    print(f"\n⚠️ RISK MANAGEMENT:")
    print(f"  Risk Events: {summary.get('risk_events_72h', 0)}")
    
    print(f"\n🚨 SYSTEM HEALTH:")
    print(f"  Errors: {summary.get('system_errors', 0)}")
    print(f"  Warnings: {summary.get('system_warnings', 0)}")
    
    # Detailed sections
    if report.get('system_health', {}).get('okx_connectivity'):
        print(f"\n📡 OKX MARKET DATA:")
        for symbol, data in report['system_health']['okx_connectivity'].items():
            if data.get('status') == 'Connected':
                print(f"  {symbol}: ${data.get('latest_price', 'N/A')}")
    
    if report.get('trading_performance', {}).get('symbol_performance'):
        print(f"\n📈 TOP TRADING SYMBOLS (72h):")
        symbol_perf = report['trading_performance']['symbol_performance']
        for symbol, count in list(symbol_perf.items())[:5]:
            print(f"  {symbol}: {count} trades")
    
    if report.get('ai_models', {}).get('model_performance'):
        print(f"\n🎯 AI MODEL PERFORMANCE:")
        for model, data in report['ai_models']['model_performance'].items():
            win_rate = data.get('avg_win_rate', 0)
            print(f"  {model}: {win_rate}% win rate")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    analyzer = ProductionAnalysis()
    report = analyzer.run_full_analysis()
    print_executive_summary(report)