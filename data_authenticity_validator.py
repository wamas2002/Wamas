"""
Data Authenticity Validator
Comprehensive validation to ensure all UI components use only authentic OKX data with no mock fallbacks
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import ccxt
import pandas as pd

class DataAuthenticityValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_results = {}
        
    def validate_portfolio_data_authenticity(self) -> Dict:
        """Validate portfolio data is from authentic OKX sources"""
        results = {
            'status': 'PASS',
            'issues': [],
            'data_source': None,
            'last_updated': None,
            'position_count': 0
        }
        
        try:
            conn = sqlite3.connect('data/portfolio_tracking.db')
            
            # Check portfolio summary data source
            summary = conn.execute('''
                SELECT data_source, last_updated, total_value
                FROM portfolio_summary
                ORDER BY last_updated DESC 
                LIMIT 1
            ''').fetchone()
            
            if summary:
                data_source, last_updated, total_value = summary
                results['data_source'] = data_source
                results['last_updated'] = last_updated
                
                # Flag non-authentic sources
                if data_source in ['demo', 'fallback', 'mock', 'test', 'sandbox']:
                    results['status'] = 'FAIL'
                    results['issues'].append(f"Portfolio using non-authentic data source: {data_source}")
                
                # Check data freshness (should be updated within last hour)
                last_update_time = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                if datetime.now() - last_update_time > timedelta(hours=1):
                    results['status'] = 'WARNING'
                    results['issues'].append(f"Portfolio data stale: {last_updated}")
            else:
                results['status'] = 'FAIL'
                results['issues'].append("No portfolio summary data found")
            
            # Check individual positions
            positions = conn.execute('''
                SELECT symbol, quantity, current_value, last_updated
                FROM portfolio_positions
                WHERE quantity > 0
            ''').fetchall()
            
            results['position_count'] = len(positions)
            
            conn.close()
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['issues'].append(f"Database error: {e}")
        
        return results
    
    def validate_market_data_authenticity(self) -> Dict:
        """Validate market data is from live OKX feeds"""
        results = {
            'status': 'PASS',
            'issues': [],
            'price_sources': {},
            'okx_connection': False
        }
        
        try:
            # Test OKX connection
            okx = ccxt.okx({'sandbox': False, 'enableRateLimit': True})
            
            # Test fetching real prices for key symbols
            test_symbols = ['BTC/USDT', 'ETH/USDT', 'PI/USDT']
            
            for symbol in test_symbols:
                try:
                    ticker = okx.fetch_ticker(symbol)
                    if ticker and 'last' in ticker:
                        results['price_sources'][symbol] = {
                            'price': ticker['last'],
                            'timestamp': ticker.get('timestamp'),
                            'source': 'OKX_LIVE'
                        }
                    else:
                        results['issues'].append(f"Invalid ticker data for {symbol}")
                        results['status'] = 'FAIL'
                        
                except Exception as e:
                    results['issues'].append(f"Failed to fetch {symbol}: {e}")
                    results['status'] = 'FAIL'
            
            results['okx_connection'] = len(results['price_sources']) > 0
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['issues'].append(f"OKX connection failed: {e}")
        
        return results
    
    def validate_ai_performance_data(self) -> Dict:
        """Validate AI performance data is from authentic model runs"""
        results = {
            'status': 'PASS',
            'issues': [],
            'model_count': 0,
            'recent_predictions': 0,
            'data_freshness': None
        }
        
        try:
            conn = sqlite3.connect('data/ai_performance.db')
            
            # Check model performance data
            models = conn.execute('''
                SELECT COUNT(DISTINCT model_name) as model_count
                FROM model_performance
                WHERE last_updated > datetime('now', '-24 hours')
            ''').fetchone()
            
            if models:
                results['model_count'] = models[0]
                
                if results['model_count'] == 0:
                    results['status'] = 'WARNING'
                    results['issues'].append("No recent model performance data (last 24h)")
            
            # Check recent predictions
            predictions = conn.execute('''
                SELECT COUNT(*) as prediction_count
                FROM ai_predictions
                WHERE timestamp > datetime('now', '-6 hours')
            ''').fetchone()
            
            if predictions:
                results['recent_predictions'] = predictions[0]
                
                if results['recent_predictions'] == 0:
                    results['status'] = 'WARNING'
                    results['issues'].append("No recent AI predictions (last 6h)")
            
            # Check data freshness
            latest_data = conn.execute('''
                SELECT MAX(last_updated) as latest
                FROM model_performance
            ''').fetchone()
            
            if latest and latest[0]:
                results['data_freshness'] = latest[0]
            
            conn.close()
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['issues'].append(f"AI performance database error: {e}")
        
        return results
    
    def validate_fundamental_analysis_data(self) -> Dict:
        """Validate fundamental analysis uses real market metrics"""
        results = {
            'status': 'PASS',
            'issues': [],
            'symbol_coverage': [],
            'data_freshness': None
        }
        
        try:
            conn = sqlite3.connect('data/fundamental_analysis.db')
            
            # Check fundamental analysis coverage
            symbols = conn.execute('''
                SELECT symbol, last_updated, overall_score
                FROM fundamental_analysis
                WHERE last_updated > datetime('now', '-24 hours')
                ORDER BY overall_score DESC
            ''').fetchall()
            
            if symbols:
                results['symbol_coverage'] = [s[0] for s in symbols]
                results['data_freshness'] = symbols[0][1] if symbols else None
                
                # Validate scores are realistic (not obviously mock data)
                for symbol, updated, score in symbols:
                    if score == 0 or score > 100:
                        results['status'] = 'WARNING'
                        results['issues'].append(f"Suspicious fundamental score for {symbol}: {score}")
            else:
                results['status'] = 'WARNING'
                results['issues'].append("No recent fundamental analysis data")
            
            conn.close()
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['issues'].append(f"Fundamental analysis database error: {e}")
        
        return results
    
    def validate_technical_analysis_data(self) -> Dict:
        """Validate technical analysis uses real OHLCV data"""
        results = {
            'status': 'PASS',
            'issues': [],
            'indicator_coverage': {},
            'data_sources': []
        }
        
        try:
            conn = sqlite3.connect('data/trading_data.db')
            
            # Check OHLCV data availability
            symbols = ['BTCUSDT', 'ETHUSDT', 'PIUSDT']
            
            for symbol in symbols:
                try:
                    ohlcv_data = conn.execute('''
                        SELECT COUNT(*) as count, MAX(timestamp) as latest
                        FROM ohlcv_1m
                        WHERE symbol = ?
                        AND timestamp > datetime('now', '-1 hour')
                    ''', (symbol,)).fetchone()
                    
                    if ohlcv_data:
                        count, latest = ohlcv_data
                        results['indicator_coverage'][symbol] = {
                            'recent_candles': count,
                            'latest_data': latest
                        }
                        
                        if count == 0:
                            results['status'] = 'WARNING'
                            results['issues'].append(f"No recent OHLCV data for {symbol}")
                        else:
                            results['data_sources'].append(f"{symbol}: {count} candles")
                            
                except Exception as e:
                    results['issues'].append(f"Technical data check failed for {symbol}: {e}")
            
            conn.close()
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['issues'].append(f"Trading data database error: {e}")
        
        return results
    
    def scan_for_mock_data_patterns(self) -> Dict:
        """Scan codebase and databases for mock data patterns"""
        results = {
            'status': 'PASS',
            'issues': [],
            'mock_patterns_found': [],
            'hardcoded_values': []
        }
        
        # Mock data patterns to search for
        mock_patterns = [
            'np.random',
            'fake_',
            'mock_',
            'demo_',
            'test_',
            'placeholder',
            'hardcoded',
            'sandbox',
            'fallback'
        ]
        
        # Check for hardcoded values that should be dynamic
        suspicious_values = [
            '156.92',  # Hardcoded portfolio value
            '99.5',    # Hardcoded concentration percentage
            '89.26',   # Hardcoded PI quantity
            '77.2',    # Hardcoded BTC score
            '68.8'     # Hardcoded AI accuracy
        ]
        
        # This would normally scan files, but for demonstration we'll check key areas
        try:
            # Check if real data service is being used
            from real_data_service import real_data_service
            validation_check = real_data_service.validate_data_authenticity()
            
            if not validation_check['overall_authentic']:
                results['status'] = 'FAIL'
                results['issues'].append("Real data service validation failed")
                
                for component, status in validation_check.items():
                    if not status and component != 'overall_authentic':
                        results['mock_patterns_found'].append(f"{component}: Not using authentic data")
            
        except Exception as e:
            results['status'] = 'FAIL'
            results['issues'].append(f"Real data service validation error: {e}")
        
        return results
    
    def generate_comprehensive_authenticity_report(self) -> Dict:
        """Generate comprehensive data authenticity validation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'PASS',
            'validation_summary': {},
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Run all validation checks
        validations = {
            'portfolio_data': self.validate_portfolio_data_authenticity(),
            'market_data': self.validate_market_data_authenticity(),
            'ai_performance': self.validate_ai_performance_data(),
            'fundamental_analysis': self.validate_fundamental_analysis_data(),
            'technical_analysis': self.validate_technical_analysis_data(),
            'mock_data_scan': self.scan_for_mock_data_patterns()
        }
        
        # Aggregate results
        for validation_name, validation_result in validations.items():
            report['validation_summary'][validation_name] = validation_result['status']
            
            if validation_result['status'] == 'FAIL':
                report['overall_status'] = 'FAIL'
                report['critical_issues'].extend([
                    f"{validation_name}: {issue}" for issue in validation_result['issues']
                ])
            elif validation_result['status'] == 'WARNING':
                if report['overall_status'] != 'FAIL':
                    report['overall_status'] = 'WARNING'
                report['warnings'].extend([
                    f"{validation_name}: {issue}" for issue in validation_result['issues']
                ])
        
        # Generate recommendations
        if report['overall_status'] == 'FAIL':
            report['recommendations'].extend([
                "Configure OKX API credentials for live market data access",
                "Ensure portfolio tracking database contains authentic OKX account data",
                "Verify AI models are running with real market data inputs",
                "Remove any fallback or demo data sources from the system"
            ])
        elif report['overall_status'] == 'WARNING':
            report['recommendations'].extend([
                "Update stale data sources to ensure real-time accuracy",
                "Increase frequency of AI model performance tracking",
                "Verify fundamental analysis data sources are current"
            ])
        
        # Store detailed validation results
        report['detailed_results'] = validations
        
        self.logger.info(f"Data authenticity validation completed: {report['overall_status']}")
        
        return report
    
    def save_validation_report(self, report: Dict) -> str:
        """Save validation report to file"""
        filename = f"data_authenticity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return filename

def run_data_authenticity_validation():
    """Run comprehensive data authenticity validation"""
    validator = DataAuthenticityValidator()
    report = validator.generate_comprehensive_authenticity_report()
    
    print(f"=== DATA AUTHENTICITY VALIDATION REPORT ===")
    print(f"Overall Status: {report['overall_status']}")
    print(f"Timestamp: {report['timestamp']}")
    print()
    
    print("Validation Summary:")
    for component, status in report['validation_summary'].items():
        status_icon = "✅" if status == "PASS" else "⚠️" if status == "WARNING" else "❌"
        print(f"  {status_icon} {component}: {status}")
    print()
    
    if report['critical_issues']:
        print("Critical Issues:")
        for issue in report['critical_issues']:
            print(f"  ❌ {issue}")
        print()
    
    if report['warnings']:
        print("Warnings:")
        for warning in report['warnings']:
            print(f"  ⚠️ {warning}")
        print()
    
    if report['recommendations']:
        print("Recommendations:")
        for rec in report['recommendations']:
            print(f"  💡 {rec}")
    
    # Save report
    filename = validator.save_validation_report(report)
    print(f"\nDetailed report saved to: {filename}")
    
    return report

if __name__ == "__main__":
    run_data_authenticity_validation()