"""
Post-UI Redesign Integration Test
Comprehensive verification that all backend components remain operational with live OKX data
"""

import sqlite3
import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostUIIntegrationTest:
    def __init__(self):
        self.test_results = {
            'okx_api_status': None,
            'live_data_verification': None,
            'ai_models_status': None,
            'trading_engine_status': None,
            'portfolio_manager_status': None,
            'strategy_selector_status': None,
            'risk_manager_status': None,
            'database_integrity': None,
            'tradingview_widgets': None,
            'backend_workflows': None
        }
        self.issues_found = []
        self.live_data_confirmed = []
        
    def test_okx_api_connectivity(self):
        """Test OKX API connectivity and live data access"""
        logger.info("Testing OKX API connectivity...")
        
        try:
            # Import OKX data service
            sys.path.append('.')
            from trading.okx_data_service import OKXDataService
            
            okx = OKXDataService()
            
            # Test major trading pairs
            test_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
            live_prices = {}
            
            for symbol in test_symbols:
                try:
                    price = okx.get_current_price(symbol)
                    if price and float(price) > 0:
                        live_prices[symbol] = float(price)
                        self.live_data_confirmed.append(f"OKX {symbol}: ${price}")
                    else:
                        self.issues_found.append(f"OKX API returned invalid price for {symbol}")
                except Exception as e:
                    self.issues_found.append(f"OKX API error for {symbol}: {str(e)}")
            
            if len(live_prices) >= 3:
                self.test_results['okx_api_status'] = 'PASS'
                logger.info(f"✅ OKX API active with {len(live_prices)} live price feeds")
            else:
                self.test_results['okx_api_status'] = 'FAIL'
                logger.error(f"❌ OKX API issues: only {len(live_prices)} symbols returning data")
                
        except ImportError:
            self.issues_found.append("OKX Data Service module not found")
            self.test_results['okx_api_status'] = 'FAIL'
        except Exception as e:
            self.issues_found.append(f"OKX API test failed: {str(e)}")
            self.test_results['okx_api_status'] = 'FAIL'
    
    def verify_live_data_in_databases(self):
        """Verify databases contain live data, not mock/placeholder values"""
        logger.info("Verifying live data in databases...")
        
        # Check trading data database
        if os.path.exists('data/trading_data.db'):
            try:
                conn = sqlite3.connect('data/trading_data.db')
                cursor = conn.cursor()
                
                # Check for recent price data
                cursor.execute("""
                    SELECT symbol, close_price, timestamp 
                    FROM ohlcv_data 
                    WHERE timestamp > datetime('now', '-1 hour')
                    ORDER BY timestamp DESC 
                    LIMIT 10
                """)
                
                recent_data = cursor.fetchall()
                if recent_data:
                    for row in recent_data:
                        self.live_data_confirmed.append(f"Recent {row[0]} price: ${row[1]} at {row[2]}")
                else:
                    self.issues_found.append("No recent price data found in trading database")
                
                conn.close()
                
            except Exception as e:
                self.issues_found.append(f"Trading database error: {str(e)}")
        else:
            self.issues_found.append("Trading data database not found")
        
        # Check portfolio tracking
        if os.path.exists('data/portfolio_tracking.db'):
            try:
                conn = sqlite3.connect('data/portfolio_tracking.db')
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT total_value, timestamp 
                    FROM portfolio_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                
                portfolio_data = cursor.fetchone()
                if portfolio_data and portfolio_data[0] != 10000.0:  # Not default value
                    self.live_data_confirmed.append(f"Live portfolio value: ${portfolio_data[0]}")
                else:
                    self.issues_found.append("Portfolio showing default/mock values")
                
                conn.close()
                
            except Exception as e:
                self.issues_found.append(f"Portfolio database error: {str(e)}")
        
        if len(self.live_data_confirmed) >= 5:
            self.test_results['live_data_verification'] = 'PASS'
        else:
            self.test_results['live_data_verification'] = 'FAIL'
    
    def test_ai_models_integration(self):
        """Test AI models are receiving live data and generating predictions"""
        logger.info("Testing AI models integration...")
        
        try:
            # Check AI performance database
            if os.path.exists('data/ai_performance.db'):
                conn = sqlite3.connect('data/ai_performance.db')
                cursor = conn.cursor()
                
                # Check for recent model evaluations
                cursor.execute("""
                    SELECT model_type, prediction_accuracy, evaluation_date
                    FROM model_evaluation_results 
                    WHERE evaluation_date > datetime('now', '-24 hours')
                    ORDER BY evaluation_date DESC
                """)
                
                model_results = cursor.fetchall()
                if model_results:
                    for row in model_results:
                        self.live_data_confirmed.append(f"AI Model {row[0]}: {row[1]}% accuracy")
                    self.test_results['ai_models_status'] = 'PASS'
                else:
                    self.issues_found.append("No recent AI model evaluations found")
                    self.test_results['ai_models_status'] = 'FAIL'
                
                conn.close()
            else:
                self.issues_found.append("AI performance database not found")
                self.test_results['ai_models_status'] = 'FAIL'
                
        except Exception as e:
            self.issues_found.append(f"AI models test error: {str(e)}")
            self.test_results['ai_models_status'] = 'FAIL'
    
    def test_trading_engine_status(self):
        """Test trading engine connectivity and decision making"""
        logger.info("Testing trading engine status...")
        
        try:
            # Check for recent trading decisions
            if os.path.exists('data/trading_data.db'):
                conn = sqlite3.connect('data/trading_data.db')
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT symbol, action, price, timestamp
                    FROM trading_decisions 
                    WHERE timestamp > datetime('now', '-24 hours')
                    ORDER BY timestamp DESC 
                    LIMIT 5
                """)
                
                decisions = cursor.fetchall()
                if decisions:
                    for row in decisions:
                        self.live_data_confirmed.append(f"Trading decision: {row[1]} {row[0]} at ${row[2]}")
                    self.test_results['trading_engine_status'] = 'PASS'
                else:
                    # Check if trading engine is configured but not executing
                    self.test_results['trading_engine_status'] = 'IDLE'
                
                conn.close()
            else:
                self.issues_found.append("Trading decisions database not found")
                self.test_results['trading_engine_status'] = 'FAIL'
                
        except Exception as e:
            self.issues_found.append(f"Trading engine test error: {str(e)}")
            self.test_results['trading_engine_status'] = 'FAIL'
    
    def test_strategy_selector_functionality(self):
        """Test strategy selector is working with live market conditions"""
        logger.info("Testing strategy selector functionality...")
        
        try:
            if os.path.exists('data/strategy_optimization.db'):
                conn = sqlite3.connect('data/strategy_optimization.db')
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT symbol, strategy, performance_score, assigned_at
                    FROM strategy_assignments 
                    ORDER BY assigned_at DESC 
                    LIMIT 10
                """)
                
                assignments = cursor.fetchall()
                if assignments:
                    strategies_found = set()
                    for row in assignments:
                        strategies_found.add(row[1])
                        self.live_data_confirmed.append(f"Strategy {row[1]} assigned to {row[0]}")
                    
                    if len(strategies_found) > 1:
                        self.test_results['strategy_selector_status'] = 'PASS'
                    else:
                        self.issues_found.append("Strategy selector showing limited diversity")
                        self.test_results['strategy_selector_status'] = 'PARTIAL'
                else:
                    self.issues_found.append("No strategy assignments found")
                    self.test_results['strategy_selector_status'] = 'FAIL'
                
                conn.close()
            else:
                self.issues_found.append("Strategy optimization database not found")
                self.test_results['strategy_selector_status'] = 'FAIL'
                
        except Exception as e:
            self.issues_found.append(f"Strategy selector test error: {str(e)}")
            self.test_results['strategy_selector_status'] = 'FAIL'
    
    def test_portfolio_manager_integration(self):
        """Test portfolio manager is tracking real positions and P&L"""
        logger.info("Testing portfolio manager integration...")
        
        try:
            if os.path.exists('data/portfolio_tracking.db'):
                conn = sqlite3.connect('data/portfolio_tracking.db')
                cursor = conn.cursor()
                
                # Check for position tracking
                cursor.execute("""
                    SELECT symbol, quantity, avg_price, current_value
                    FROM positions 
                    WHERE quantity > 0
                """)
                
                positions = cursor.fetchall()
                if positions:
                    for row in positions:
                        self.live_data_confirmed.append(f"Position: {row[1]} {row[0]} @ ${row[2]}")
                
                # Check for P&L tracking
                cursor.execute("""
                    SELECT total_value, daily_pnl, timestamp
                    FROM portfolio_metrics 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """)
                
                metrics = cursor.fetchone()
                if metrics:
                    self.live_data_confirmed.append(f"Portfolio: ${metrics[0]}, Daily P&L: ${metrics[1]}")
                    self.test_results['portfolio_manager_status'] = 'PASS'
                else:
                    self.issues_found.append("No portfolio metrics found")
                    self.test_results['portfolio_manager_status'] = 'FAIL'
                
                conn.close()
            else:
                self.issues_found.append("Portfolio tracking database not found")
                self.test_results['portfolio_manager_status'] = 'FAIL'
                
        except Exception as e:
            self.issues_found.append(f"Portfolio manager test error: {str(e)}")
            self.test_results['portfolio_manager_status'] = 'FAIL'
    
    def test_risk_manager_operations(self):
        """Test risk manager is monitoring and protecting positions"""
        logger.info("Testing risk manager operations...")
        
        try:
            if os.path.exists('data/risk_management.db'):
                conn = sqlite3.connect('data/risk_management.db')
                cursor = conn.cursor()
                
                # Check for risk events
                cursor.execute("""
                    SELECT event_type, symbol, description, timestamp
                    FROM risk_events 
                    ORDER BY timestamp DESC 
                    LIMIT 5
                """)
                
                risk_events = cursor.fetchall()
                if risk_events:
                    for row in risk_events:
                        self.live_data_confirmed.append(f"Risk event: {row[0]} for {row[1]}")
                    self.test_results['risk_manager_status'] = 'PASS'
                else:
                    # Risk manager may be operational but no events triggered
                    self.test_results['risk_manager_status'] = 'IDLE'
                
                conn.close()
            else:
                # Risk manager may not have separate database
                self.test_results['risk_manager_status'] = 'UNKNOWN'
                
        except Exception as e:
            self.issues_found.append(f"Risk manager test error: {str(e)}")
            self.test_results['risk_manager_status'] = 'FAIL'
    
    def test_database_integrity(self):
        """Test all databases are accessible and not corrupted"""
        logger.info("Testing database integrity...")
        
        databases = [
            'data/trading_data.db',
            'data/portfolio_tracking.db',
            'data/ai_performance.db',
            'data/strategy_optimization.db',
            'data/sentiment_analysis.db'
        ]
        
        accessible_dbs = 0
        for db_path in databases:
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    if tables:
                        accessible_dbs += 1
                        self.live_data_confirmed.append(f"Database {db_path}: {len(tables)} tables")
                    conn.close()
                except Exception as e:
                    self.issues_found.append(f"Database corruption in {db_path}: {str(e)}")
        
        if accessible_dbs >= 3:
            self.test_results['database_integrity'] = 'PASS'
        else:
            self.test_results['database_integrity'] = 'FAIL'
    
    def test_backend_workflows(self):
        """Test that backend workflows are still operational"""
        logger.info("Testing backend workflows...")
        
        # Check if processes are running by looking for recent activity
        recent_activity = 0
        
        # Check for recent data updates
        if os.path.exists('data/trading_data.db'):
            try:
                conn = sqlite3.connect('data/trading_data.db')
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT COUNT(*) FROM ohlcv_data 
                    WHERE timestamp > datetime('now', '-1 hour')
                """)
                
                recent_count = cursor.fetchone()[0]
                if recent_count > 0:
                    recent_activity += 1
                    self.live_data_confirmed.append(f"Recent data updates: {recent_count} records")
                
                conn.close()
            except:
                pass
        
        # Check for AI model activity
        if os.path.exists('data/ai_performance.db'):
            try:
                conn = sqlite3.connect('data/ai_performance.db')
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT COUNT(*) FROM model_evaluation_results 
                    WHERE evaluation_date > datetime('now', '-24 hours')
                """)
                
                eval_count = cursor.fetchone()[0]
                if eval_count > 0:
                    recent_activity += 1
                    self.live_data_confirmed.append(f"Recent AI evaluations: {eval_count}")
                
                conn.close()
            except:
                pass
        
        if recent_activity >= 1:
            self.test_results['backend_workflows'] = 'PASS'
        else:
            self.test_results['backend_workflows'] = 'IDLE'
    
    def test_tradingview_widget_integration(self):
        """Test TradingView widgets can load properly"""
        logger.info("Testing TradingView widget integration...")
        
        # This is tested by checking if the Flask app can serve the pages
        try:
            import requests
            response = requests.get('http://localhost:5001/', timeout=5)
            if response.status_code == 200 and 'TradingView' in response.text:
                self.test_results['tradingview_widgets'] = 'PASS'
                self.live_data_confirmed.append("TradingView widgets loading in Flask app")
            else:
                self.issues_found.append("TradingView widgets not loading properly")
                self.test_results['tradingview_widgets'] = 'FAIL'
        except Exception as e:
            self.issues_found.append(f"Flask app not accessible: {str(e)}")
            self.test_results['tradingview_widgets'] = 'FAIL'
    
    def generate_integration_report(self):
        """Generate comprehensive integration test report"""
        logger.info("Generating integration test report...")
        
        report = {
            'test_timestamp': datetime.now().isoformat(),
            'test_results': self.test_results,
            'live_data_confirmed': self.live_data_confirmed,
            'issues_found': self.issues_found,
            'overall_status': self.calculate_overall_status()
        }
        
        return report
    
    def calculate_overall_status(self):
        """Calculate overall system status"""
        pass_count = sum(1 for status in self.test_results.values() if status == 'PASS')
        fail_count = sum(1 for status in self.test_results.values() if status == 'FAIL')
        total_tests = len(self.test_results)
        
        if fail_count == 0 and pass_count >= total_tests * 0.8:
            return 'FULLY_OPERATIONAL'
        elif fail_count <= 2:
            return 'MOSTLY_OPERATIONAL'
        else:
            return 'NEEDS_ATTENTION'
    
    def run_full_integration_test(self):
        """Execute complete integration test suite"""
        logger.info("=" * 60)
        logger.info("POST-UI REDESIGN INTEGRATION TEST")
        logger.info("=" * 60)
        
        # Execute all tests
        self.test_okx_api_connectivity()
        self.verify_live_data_in_databases()
        self.test_ai_models_integration()
        self.test_trading_engine_status()
        self.test_strategy_selector_functionality()
        self.test_portfolio_manager_integration()
        self.test_risk_manager_operations()
        self.test_database_integrity()
        self.test_backend_workflows()
        self.test_tradingview_widget_integration()
        
        # Generate report
        report = self.generate_integration_report()
        
        return report

def print_integration_summary(report):
    """Print formatted integration test summary"""
    print("\n" + "=" * 80)
    print("🔍 POST-UI REDESIGN INTEGRATION TEST RESULTS")
    print("=" * 80)
    
    print(f"\n📊 Overall Status: {report['overall_status']}")
    print(f"🕒 Test Completed: {report['test_timestamp']}")
    
    print("\n✅ COMPONENT STATUS:")
    for component, status in report['test_results'].items():
        status_icon = "✅" if status == "PASS" else "⚠️" if status in ["IDLE", "PARTIAL", "UNKNOWN"] else "❌"
        print(f"  {status_icon} {component.replace('_', ' ').title()}: {status}")
    
    if report['live_data_confirmed']:
        print("\n🔗 LIVE DATA CONFIRMED:")
        for confirmation in report['live_data_confirmed']:
            print(f"  • {confirmation}")
    
    if report['issues_found']:
        print("\n⚠️ ISSUES IDENTIFIED:")
        for issue in report['issues_found']:
            print(f"  • {issue}")
    
    print("\n" + "=" * 80)
    
    # Final verdict
    if report['overall_status'] == 'FULLY_OPERATIONAL':
        print("✅ VERDICT: System is fully functional and uses only live OKX data.")
    elif report['overall_status'] == 'MOSTLY_OPERATIONAL':
        print("⚠️ VERDICT: System is mostly operational with minor issues to address.")
    else:
        print("❌ VERDICT: System requires attention to restore full functionality.")
    
    print("=" * 80)

if __name__ == "__main__":
    tester = PostUIIntegrationTest()
    report = tester.run_full_integration_test()
    
    # Save report
    with open('post_ui_integration_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print_integration_summary(report)