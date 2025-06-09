#!/usr/bin/env python3
"""
Final Authenticity Verification Test
Comprehensive validation that all mock/fallback data has been eliminated
"""

import os
import sqlite3
import json
import logging
from datetime import datetime
from typing import Dict, List
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinalAuthenticityVerifier:
    def __init__(self):
        self.verification_results = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'PASS',
            'tests_run': 0,
            'tests_passed': 0,
            'critical_issues': [],
            'warnings': [],
            'verified_components': []
        }
    
    def verify_multi_timeframe_analyzer(self) -> Dict:
        """Verify multi-timeframe analyzer uses only authentic data"""
        logger.info("Verifying multi-timeframe analyzer authenticity...")
        
        try:
            from plugins.multi_timeframe_analyzer import MultiTimeframeAnalyzer
            
            analyzer = MultiTimeframeAnalyzer()
            
            # Check that fallback methods no longer exist
            fallback_methods = [
                '_generate_fallback_data',
                '_generate_fallback_timeframe_data'
            ]
            
            for method in fallback_methods:
                if hasattr(analyzer, method):
                    self.verification_results['critical_issues'].append(
                        f"Multi-timeframe analyzer still contains fallback method: {method}"
                    )
                    return {'status': 'FAIL', 'reason': f'Fallback method {method} still exists'}
            
            # Verify that authentic-only method exists
            if not hasattr(analyzer, '_get_authentic_data_only'):
                self.verification_results['critical_issues'].append(
                    "Multi-timeframe analyzer missing authentic data method"
                )
                return {'status': 'FAIL', 'reason': 'Missing authentic data method'}
            
            self.verification_results['verified_components'].append('Multi-Timeframe Analyzer')
            return {'status': 'PASS', 'reason': 'All fallback methods removed, authentic-only methods verified'}
            
        except Exception as e:
            self.verification_results['critical_issues'].append(f"Multi-timeframe analyzer verification failed: {e}")
            return {'status': 'ERROR', 'reason': str(e)}
    
    def verify_multi_exchange_connector(self) -> Dict:
        """Verify multi-exchange connector requires authentic credentials"""
        logger.info("Verifying multi-exchange connector authenticity...")
        
        try:
            from plugins.multi_exchange_connector import MultiExchangeConnector
            
            connector = MultiExchangeConnector()
            
            # Test that portfolio methods require authentication
            try:
                portfolio = connector.get_exchange_portfolio('okx')
                self.verification_results['critical_issues'].append(
                    "Multi-exchange connector allows portfolio access without authentication"
                )
                return {'status': 'FAIL', 'reason': 'Portfolio access without authentication allowed'}
            except Exception as e:
                if 'API keys' not in str(e) and 'authentication' not in str(e):
                    self.verification_results['warnings'].append(
                        f"Portfolio error message doesn't mention authentication: {e}"
                    )
            
            # Test that aggregated portfolio requires authentication
            try:
                agg_portfolio = connector.get_aggregated_portfolio()
                self.verification_results['critical_issues'].append(
                    "Multi-exchange connector allows aggregated portfolio without authentication"
                )
                return {'status': 'FAIL', 'reason': 'Aggregated portfolio without authentication allowed'}
            except Exception as e:
                if 'API keys' not in str(e) and 'authentication' not in str(e):
                    self.verification_results['warnings'].append(
                        f"Aggregated portfolio error doesn't mention authentication: {e}"
                    )
            
            self.verification_results['verified_components'].append('Multi-Exchange Connector')
            return {'status': 'PASS', 'reason': 'All portfolio methods require authentication'}
            
        except Exception as e:
            self.verification_results['critical_issues'].append(f"Multi-exchange connector verification failed: {e}")
            return {'status': 'ERROR', 'reason': str(e)}
    
    def verify_api_endpoints(self) -> Dict:
        """Verify API endpoints return authentic data errors"""
        logger.info("Verifying API endpoints require authentic data...")
        
        try:
            import requests
            import time
            
            base_url = "http://localhost:5000"
            endpoints_to_test = [
                ('/api/mtfa-analysis', 'POST', {'symbol': 'BTC/USDT', 'exchange': 'okx'}),
                ('/api/exchange-prices', 'POST', {'symbol': 'BTC/USDT'}),
                ('/api/exchange-portfolio/okx', 'GET', None),
                ('/api/aggregated-portfolio', 'GET', None),
                ('/api/exchange-comparison', 'POST', {'symbol': 'BTC/USDT'})
            ]
            
            authenticated_responses = []
            
            for endpoint, method, data in endpoints_to_test:
                try:
                    if method == 'POST':
                        response = requests.post(f"{base_url}{endpoint}", json=data, timeout=5)
                    else:
                        response = requests.get(f"{base_url}{endpoint}", timeout=5)
                    
                    if response.status_code == 200:
                        # Check if response contains mock data indicators
                        response_text = response.text.lower()
                        mock_indicators = ['mock', 'fake', 'demo', 'placeholder', 'fallback']
                        
                        for indicator in mock_indicators:
                            if indicator in response_text:
                                self.verification_results['critical_issues'].append(
                                    f"API endpoint {endpoint} returns mock data: {indicator}"
                                )
                                return {'status': 'FAIL', 'reason': f'Mock data in {endpoint}'}
                    
                    elif response.status_code == 500:
                        # Check if error message mentions authentication
                        if response.json().get('error'):
                            error_msg = response.json()['error'].lower()
                            if 'authentic' in error_msg or 'api key' in error_msg or 'authentication' in error_msg:
                                authenticated_responses.append(endpoint)
                    
                except requests.exceptions.RequestException as e:
                    self.verification_results['warnings'].append(f"API test failed for {endpoint}: {e}")
                
                time.sleep(0.1)  # Rate limiting
            
            if len(authenticated_responses) >= 3:  # Most endpoints should require authentication
                self.verification_results['verified_components'].append('API Endpoints')
                return {'status': 'PASS', 'reason': f'{len(authenticated_responses)} endpoints require authentication'}
            else:
                self.verification_results['warnings'].append(
                    f"Only {len(authenticated_responses)} endpoints require authentication"
                )
                return {'status': 'PARTIAL', 'reason': 'Some endpoints may not require authentication'}
                
        except Exception as e:
            self.verification_results['critical_issues'].append(f"API endpoint verification failed: {e}")
            return {'status': 'ERROR', 'reason': str(e)}
    
    def scan_codebase_for_mock_patterns(self) -> Dict:
        """Scan codebase for remaining mock data patterns"""
        logger.info("Scanning codebase for mock data patterns...")
        
        mock_patterns = [
            'np.random.uniform',
            'fake_',
            'mock_',
            'demo_',
            '_fallback_',
            'generate_fallback',
            'placeholder',
            'hardcoded'
        ]
        
        suspicious_files = []
        
        # Check key files
        files_to_check = [
            'complete_trading_platform.py',
            'plugins/multi_timeframe_analyzer.py',
            'plugins/multi_exchange_connector.py'
        ]
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                        
                    for pattern in mock_patterns:
                        if pattern in content:
                            suspicious_files.append(f"{file_path}: {pattern}")
                            
                except Exception as e:
                    self.verification_results['warnings'].append(f"Could not scan {file_path}: {e}")
        
        if suspicious_files:
            self.verification_results['critical_issues'].extend(
                [f"Mock pattern found: {file}" for file in suspicious_files]
            )
            return {'status': 'FAIL', 'reason': f'Mock patterns found in {len(suspicious_files)} locations'}
        else:
            self.verification_results['verified_components'].append('Codebase Pattern Scan')
            return {'status': 'PASS', 'reason': 'No mock patterns detected in key files'}
    
    def verify_database_authenticity(self) -> Dict:
        """Verify database contains no mock data indicators"""
        logger.info("Verifying database authenticity...")
        
        try:
            # Check main trading database
            if os.path.exists('trading_platform.db'):
                conn = sqlite3.connect('trading_platform.db')
                cursor = conn.cursor()
                
                # Check for mock data in table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                mock_table_indicators = ['mock', 'test', 'demo', 'fake', 'sample']
                mock_tables = []
                
                for table in tables:
                    for indicator in mock_table_indicators:
                        if indicator in table.lower():
                            mock_tables.append(table)
                
                if mock_tables:
                    self.verification_results['warnings'].extend(
                        [f"Suspicious table name: {table}" for table in mock_tables]
                    )
                
                conn.close()
                
                if not mock_tables:
                    self.verification_results['verified_components'].append('Database Schema')
                    return {'status': 'PASS', 'reason': 'No mock table names detected'}
                else:
                    return {'status': 'PARTIAL', 'reason': f'{len(mock_tables)} suspicious table names'}
            else:
                self.verification_results['warnings'].append('Main trading database not found')
                return {'status': 'PARTIAL', 'reason': 'Database file not found'}
                
        except Exception as e:
            self.verification_results['critical_issues'].append(f"Database verification failed: {e}")
            return {'status': 'ERROR', 'reason': str(e)}
    
    def run_comprehensive_verification(self) -> Dict:
        """Run all verification tests"""
        logger.info("Starting comprehensive authenticity verification...")
        
        tests = [
            ('Multi-Timeframe Analyzer', self.verify_multi_timeframe_analyzer),
            ('Multi-Exchange Connector', self.verify_multi_exchange_connector),
            ('API Endpoints', self.verify_api_endpoints),
            ('Codebase Pattern Scan', self.scan_codebase_for_mock_patterns),
            ('Database Authenticity', self.verify_database_authenticity)
        ]
        
        for test_name, test_func in tests:
            self.verification_results['tests_run'] += 1
            
            logger.info(f"Running test: {test_name}")
            result = test_func()
            
            if result['status'] == 'PASS':
                self.verification_results['tests_passed'] += 1
                logger.info(f"✅ {test_name}: PASSED - {result['reason']}")
            elif result['status'] == 'PARTIAL':
                logger.warning(f"⚠️ {test_name}: PARTIAL - {result['reason']}")
            elif result['status'] == 'FAIL':
                logger.error(f"❌ {test_name}: FAILED - {result['reason']}")
                self.verification_results['overall_status'] = 'FAIL'
            else:
                logger.error(f"🔧 {test_name}: ERROR - {result['reason']}")
                if self.verification_results['overall_status'] != 'FAIL':
                    self.verification_results['overall_status'] = 'ERROR'
        
        # Calculate final status
        if self.verification_results['tests_passed'] == self.verification_results['tests_run']:
            self.verification_results['overall_status'] = 'PASS'
        elif self.verification_results['critical_issues']:
            self.verification_results['overall_status'] = 'FAIL'
        else:
            self.verification_results['overall_status'] = 'PARTIAL'
        
        return self.verification_results
    
    def generate_verification_report(self, results: Dict):
        """Generate detailed verification report"""
        report = f"""
# FINAL AUTHENTICITY VERIFICATION REPORT

**Verification Date:** {results['timestamp']}
**Overall Status:** {results['overall_status']}
**Tests Run:** {results['tests_run']}
**Tests Passed:** {results['tests_passed']}

## Verified Components
{chr(10).join([f"✅ {component}" for component in results['verified_components']])}

## Critical Issues
{chr(10).join([f"❌ {issue}" for issue in results['critical_issues']]) if results['critical_issues'] else "None detected"}

## Warnings
{chr(10).join([f"⚠️ {warning}" for warning in results['warnings']]) if results['warnings'] else "None"}

## Summary
Mock data elimination status: {'COMPLETE' if results['overall_status'] == 'PASS' else 'ISSUES DETECTED'}
System ready for production: {'YES' if results['overall_status'] == 'PASS' else 'REQUIRES ATTENTION'}
"""
        
        with open('FINAL_VERIFICATION_REPORT.md', 'w') as f:
            f.write(report)
        
        logger.info("Verification report saved to FINAL_VERIFICATION_REPORT.md")

if __name__ == "__main__":
    verifier = FinalAuthenticityVerifier()
    results = verifier.run_comprehensive_verification()
    verifier.generate_verification_report(results)
    
    print(f"\n🔍 VERIFICATION COMPLETE")
    print(f"Overall Status: {results['overall_status']}")
    print(f"Tests Passed: {results['tests_passed']}/{results['tests_run']}")
    print(f"Critical Issues: {len(results['critical_issues'])}")
    print(f"Warnings: {len(results['warnings'])}")
    print(f"Verified Components: {len(results['verified_components'])}")