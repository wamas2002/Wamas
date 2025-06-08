"""
Production System Optimization Analysis
Identifies weak points and provides targeted improvements based on live system data
"""

import sqlite3
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionOptimizationAnalyzer:
    def __init__(self):
        self.analysis = {
            'timestamp': datetime.now().isoformat(),
            'weak_components': {},
            'performance_issues': {},
            'optimization_recommendations': {},
            'priority_fixes': []
        }
        
    def analyze_api_integration_issues(self):
        """Analyze API integration and method compatibility issues"""
        logger.info("Analyzing API integration issues...")
        
        # From the logs, we can see specific API method issues
        api_issues = {
            'okx_method_errors': [
                "'OKXDataService' object has no attribute 'get_candles'",
                "'OKXDataService' object has no attribute 'get_instruments'",
                "OKX API error for None: argument of type 'NoneType' is not iterable"
            ],
            'data_access_problems': [
                "No OHLCV data found for symbols in 1m timeframe",
                "Database schema inconsistencies in timestamp columns",
                "Infinite extent warnings in portfolio visualization"
            ]
        }
        
        self.analysis['weak_components']['api_integration'] = {
            'severity': 'High',
            'issues': api_issues,
            'impact': 'Prevents AI model evaluation and data collection',
            'affected_modules': ['adaptive_model_selector', 'data_collection', 'strategy_evaluation']
        }
        
    def analyze_ai_model_performance(self):
        """Analyze AI model performance and accuracy issues"""
        logger.info("Analyzing AI model performance...")
        
        # Check AI performance database for model accuracy
        model_issues = {}
        
        try:
            if os.path.exists('data/ai_performance.db'):
                conn = sqlite3.connect('data/ai_performance.db')
                cursor = conn.cursor()
                
                # Get table names
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = [row[0] for row in cursor.fetchall()]
                
                for table in tables:
                    try:
                        # Check for performance data
                        cursor.execute(f"SELECT * FROM {table} ORDER BY rowid DESC LIMIT 10;")
                        columns = [description[0] for description in cursor.description]
                        rows = cursor.fetchall()
                        
                        if rows and 'win_rate' in columns:
                            # Analyze win rates
                            for row in rows:
                                record = dict(zip(columns, row))
                                win_rate = record.get('win_rate', 0)
                                
                                if win_rate < 60:  # Below target threshold
                                    symbol = record.get('symbol', table)
                                    model_issues[symbol] = {
                                        'win_rate': win_rate,
                                        'issue': 'Below 60% accuracy threshold',
                                        'severity': 'Medium' if win_rate > 45 else 'High'
                                    }
                        
                    except Exception as e:
                        model_issues[table] = {'error': str(e)[:50]}
                
                conn.close()
                
        except Exception as e:
            logger.error(f"AI model analysis error: {e}")
        
        self.analysis['weak_components']['ai_models'] = {
            'performance_issues': model_issues,
            'training_data_shortage': 'Insufficient trading history for accurate model training',
            'model_evaluation_blocked': 'Cannot evaluate models due to API method issues'
        }
        
    def analyze_strategy_effectiveness(self):
        """Analyze strategy performance and execution issues"""
        logger.info("Analyzing strategy effectiveness...")
        
        strategy_issues = {
            'execution_problems': {
                'no_live_trades': 'System in monitoring mode - no trades executed in 72h',
                'strategy_evaluation_nan': 'Average scores showing NaN values',
                'limited_strategy_diversity': 'All symbols assigned same grid strategy'
            },
            'performance_metrics': {
                'missing_win_rates': 'Cannot calculate strategy win rates without trade history',
                'no_pnl_tracking': 'No profit/loss data available for strategy optimization',
                'switch_frequency': 'Initial strategy assignment only, no dynamic switching observed'
            }
        }
        
        self.analysis['weak_components']['strategy_system'] = strategy_issues
        
    def analyze_database_schema_issues(self):
        """Analyze database schema and data integrity problems"""
        logger.info("Analyzing database schema issues...")
        
        schema_issues = {}
        
        db_files = [
            'data/ai_performance.db',
            'data/autoconfig.db', 
            'data/smart_selector.db',
            'data/sentiment_data.db'
        ]
        
        for db_path in db_files:
            if os.path.exists(db_path):
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [row[0] for row in cursor.fetchall()]
                    
                    db_issues = []
                    
                    for table in tables:
                        try:
                            # Try to query with timestamp
                            cursor.execute(f"PRAGMA table_info({table});")
                            columns = [col[1] for col in cursor.fetchall()]
                            
                            if 'timestamp' not in columns and table not in ['sqlite_sequence']:
                                db_issues.append(f"Missing timestamp column in {table}")
                                
                        except Exception as e:
                            db_issues.append(f"Schema error in {table}: {str(e)[:30]}")
                    
                    if db_issues:
                        schema_issues[os.path.basename(db_path)] = db_issues
                    
                    conn.close()
                    
                except Exception as e:
                    schema_issues[os.path.basename(db_path)] = [f"Access error: {str(e)[:50]}"]
        
        self.analysis['weak_components']['database_schema'] = schema_issues
        
    def analyze_ui_performance_issues(self):
        """Analyze UI and visualization performance issues"""
        logger.info("Analyzing UI performance issues...")
        
        ui_issues = {
            'visualization_errors': [
                "Infinite extent warnings in portfolio charts",
                "Empty portfolio data causing chart rendering issues",
                "WebSocket connection errors affecting real-time updates"
            ],
            'data_display_problems': [
                "Portfolio value showing Infinity/-Infinity",
                "Missing trade history for performance visualization",
                "Strategy performance metrics unavailable"
            ],
            'responsiveness_concerns': [
                "Chart rendering with insufficient data",
                "Real-time updates may be delayed due to data collection issues"
            ]
        }
        
        self.analysis['weak_components']['ui_performance'] = ui_issues
        
    def generate_priority_fixes(self):
        """Generate prioritized list of fixes and improvements"""
        logger.info("Generating priority fixes...")
        
        priority_fixes = [
            {
                'priority': 'CRITICAL',
                'component': 'OKX API Integration',
                'issue': 'Missing get_candles and get_instruments methods',
                'fix': 'Implement missing API methods in OKXDataService class',
                'impact': 'Blocks AI model evaluation and data collection',
                'effort': 'High'
            },
            {
                'priority': 'HIGH',
                'component': 'Database Schema',
                'issue': 'Missing timestamp columns and schema inconsistencies',
                'fix': 'Standardize database schemas with proper timestamp fields',
                'impact': 'Prevents proper data querying and analysis',
                'effort': 'Medium'
            },
            {
                'priority': 'HIGH',
                'component': 'Data Collection Pipeline',
                'issue': 'No 1-minute OHLCV data for model training',
                'fix': 'Implement proper timeframe mapping and data collection',
                'impact': 'AI models cannot train on fresh data',
                'effort': 'Medium'
            },
            {
                'priority': 'MEDIUM',
                'component': 'Portfolio Visualization',
                'issue': 'Infinite extent errors in charts',
                'fix': 'Add null/empty data handling in chart components',
                'impact': 'Poor user experience and visualization errors',
                'effort': 'Low'
            },
            {
                'priority': 'MEDIUM',
                'component': 'Strategy Diversification',
                'issue': 'All symbols using same grid strategy',
                'fix': 'Enhance strategy selection logic with market condition analysis',
                'impact': 'Limited strategy optimization and performance',
                'effort': 'Medium'
            },
            {
                'priority': 'LOW',
                'component': 'Error Handling',
                'issue': 'None type iteration errors',
                'fix': 'Add proper null checks and error handling throughout codebase',
                'impact': 'System stability and error resilience',
                'effort': 'Low'
            }
        ]
        
        self.analysis['priority_fixes'] = priority_fixes
        
    def generate_optimization_recommendations(self):
        """Generate specific optimization recommendations"""
        logger.info("Generating optimization recommendations...")
        
        recommendations = {
            'api_improvements': [
                "Implement missing OKX API methods: get_candles, get_instruments",
                "Add proper error handling for None values in API responses",
                "Implement retry logic for failed API calls",
                "Add API rate limiting to prevent connection issues"
            ],
            'ai_model_enhancements': [
                "Implement data collection for 1-minute timeframes",
                "Add ensemble model voting for better accuracy",
                "Implement online learning for real-time model updates",
                "Add model performance monitoring and auto-switching"
            ],
            'strategy_optimizations': [
                "Implement dynamic strategy selection based on market volatility",
                "Add confirmation indicators for strategy switches",
                "Implement strategy performance scoring and ranking",
                "Add market regime detection for strategy adaptation"
            ],
            'database_improvements': [
                "Standardize timestamp columns across all tables",
                "Implement proper database migration system",
                "Add data validation and constraint checking",
                "Optimize database queries for performance"
            ],
            'ui_enhancements': [
                "Add empty state handling for charts",
                "Implement progressive loading for large datasets",
                "Add real-time data refresh indicators",
                "Improve error messaging and user feedback"
            ],
            'performance_optimizations': [
                "Implement async processing for data collection",
                "Add caching for frequently accessed data",
                "Optimize database connection pooling",
                "Implement background task queuing"
            ]
        }
        
        self.analysis['optimization_recommendations'] = recommendations
        
    def run_complete_analysis(self):
        """Execute complete optimization analysis"""
        logger.info("🔍 Starting Production Optimization Analysis...")
        
        self.analyze_api_integration_issues()
        self.analyze_ai_model_performance()
        self.analyze_strategy_effectiveness()
        self.analyze_database_schema_issues()
        self.analyze_ui_performance_issues()
        self.generate_priority_fixes()
        self.generate_optimization_recommendations()
        
        # Save detailed analysis
        report_filename = f"optimization_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(self.analysis, f, indent=2, default=str)
        
        logger.info(f"✅ Optimization analysis saved: {report_filename}")
        return self.analysis

def print_optimization_summary(analysis):
    """Print formatted optimization summary"""
    print("\n" + "="*80)
    print("🔧 PRODUCTION OPTIMIZATION ANALYSIS")
    print("="*80)
    
    print(f"\n📊 ANALYSIS TIMESTAMP: {analysis.get('timestamp', 'Unknown')}")
    
    # Priority Fixes
    print(f"\n🚨 PRIORITY FIXES:")
    fixes = analysis.get('priority_fixes', [])
    for i, fix in enumerate(fixes[:6], 1):
        priority = fix.get('priority', 'UNKNOWN')
        component = fix.get('component', 'Unknown')
        issue = fix.get('issue', 'Unknown')
        solution = fix.get('fix', 'Unknown')
        
        priority_icon = "🔥" if priority == "CRITICAL" else "⚠️" if priority == "HIGH" else "💡"
        
        print(f"  {i}. {priority_icon} [{priority}] {component}")
        print(f"     Issue: {issue}")
        print(f"     Fix: {solution}")
        print()
    
    # Weak Components Summary
    print(f"\n🚫 WEAK COMPONENTS IDENTIFIED:")
    weak_components = analysis.get('weak_components', {})
    
    for component, details in weak_components.items():
        component_name = component.replace('_', ' ').title()
        print(f"  • {component_name}")
        
        if isinstance(details, dict):
            if 'severity' in details:
                print(f"    Severity: {details['severity']}")
            if 'impact' in details:
                print(f"    Impact: {details['impact']}")
    
    # Top Recommendations
    print(f"\n💡 TOP OPTIMIZATION RECOMMENDATIONS:")
    recommendations = analysis.get('optimization_recommendations', {})
    
    rec_count = 1
    for category, rec_list in recommendations.items():
        if rec_count > 8:  # Limit to top 8 recommendations
            break
            
        category_name = category.replace('_', ' ').title()
        print(f"  {rec_count}. {category_name}:")
        
        if isinstance(rec_list, list) and rec_list:
            print(f"     → {rec_list[0]}")
            rec_count += 1
    
    print(f"\n" + "="*80)

if __name__ == "__main__":
    analyzer = ProductionOptimizationAnalyzer()
    analysis = analyzer.run_complete_analysis()
    print_optimization_summary(analysis)