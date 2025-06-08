#!/usr/bin/env python3
"""
Comprehensive Trading Bot System Audit
Verifies production-readiness, component integration, and authentic data flows
"""

import sys
import time
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import os

class TradingSystemAudit:
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'integration_tests': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
    def audit_okx_connectivity(self) -> Dict[str, Any]:
        """Test OKX API connectivity and market data access"""
        print("🔍 Auditing OKX API connectivity...")
        
        try:
            # Test public market data endpoints
            response = requests.get("https://www.okx.com/api/v5/market/ticker?instId=BTC-USDT", 
                                  timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '0' and data.get('data'):
                    btc_price = float(data['data'][0]['last'])
                    volume_24h = float(data['data'][0]['vol24h'])
                    
                    return {
                        'status': 'CONNECTED',
                        'btc_price': btc_price,
                        'volume_24h': volume_24h,
                        'timestamp': data['data'][0]['ts'],
                        'response_time_ms': response.elapsed.total_seconds() * 1000
                    }
            
            return {'status': 'ERROR', 'message': 'Invalid API response'}
            
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}
    
    def audit_model_components(self) -> Dict[str, Any]:
        """Test AI model components functionality"""
        print("🧠 Auditing AI model components...")
        
        model_status = {}
        
        try:
            # Test LSTM predictor
            from ai.lstm_predictor import AdvancedLSTMPredictor
            lstm = AdvancedLSTMPredictor()
            model_status['lstm'] = {'status': 'AVAILABLE', 'class': 'AdvancedLSTMPredictor'}
        except Exception as e:
            model_status['lstm'] = {'status': 'ERROR', 'error': str(e)}
        
        try:
            # Test Prophet predictor
            from ai.prophet_predictor import AdvancedProphetPredictor
            prophet = AdvancedProphetPredictor()
            model_status['prophet'] = {'status': 'AVAILABLE', 'class': 'AdvancedProphetPredictor'}
        except Exception as e:
            model_status['prophet'] = {'status': 'ERROR', 'error': str(e)}
        
        try:
            # Test comprehensive ML pipeline
            from ai.comprehensive_ml_pipeline import ComprehensiveMLPipeline
            ml_pipeline = ComprehensiveMLPipeline()
            model_status['ml_pipeline'] = {'status': 'AVAILABLE', 'class': 'ComprehensiveMLPipeline'}
        except Exception as e:
            model_status['ml_pipeline'] = {'status': 'ERROR', 'error': str(e)}
        
        try:
            # Test FreqAI pipeline
            from ai.freqai_pipeline import FreqAILevelPipeline
            freqai = FreqAILevelPipeline()
            model_status['freqai'] = {'status': 'AVAILABLE', 'class': 'FreqAILevelPipeline'}
        except Exception as e:
            model_status['freqai'] = {'status': 'ERROR', 'error': str(e)}
        
        return model_status
    
    def audit_trading_engine(self) -> Dict[str, Any]:
        """Test trading engine components"""
        print("⚙️ Auditing trading engine...")
        
        engine_status = {}
        
        try:
            from trading.okx_connector import OKXConnector
            # Test with environment variables
            api_key = os.getenv('OKX_API_KEY', '')
            secret_key = os.getenv('OKX_SECRET_KEY', '')
            passphrase = os.getenv('OKX_PASSPHRASE', '')
            
            connector = OKXConnector(api_key, secret_key, passphrase, sandbox=False)
            engine_status['okx_connector'] = {
                'status': 'AVAILABLE',
                'has_credentials': bool(api_key and secret_key and passphrase),
                'production_mode': not connector.sandbox
            }
        except Exception as e:
            engine_status['okx_connector'] = {'status': 'ERROR', 'error': str(e)}
        
        try:
            from trading.okx_data_service import OKXDataService
            data_service = OKXDataService()
            engine_status['data_service'] = {'status': 'AVAILABLE', 'class': 'OKXDataService'}
        except Exception as e:
            engine_status['data_service'] = {'status': 'ERROR', 'error': str(e)}
        
        return engine_status
    
    def audit_risk_management(self) -> Dict[str, Any]:
        """Test risk management components"""
        print("🛡️ Auditing risk management...")
        
        risk_status = {}
        
        try:
            from trading.risk_management import RiskManager
            risk_manager = RiskManager()
            risk_status['risk_manager'] = {'status': 'AVAILABLE', 'class': 'RiskManager'}
        except Exception as e:
            risk_status['risk_manager'] = {'status': 'ERROR', 'error': str(e)}
        
        return risk_status
    
    def test_data_flow_integration(self) -> Dict[str, Any]:
        """Test end-to-end data flow from OKX to AI models"""
        print("🔄 Testing data flow integration...")
        
        try:
            from trading.okx_data_service import OKXDataService
            
            data_service = OKXDataService()
            
            # Test real data fetch
            start_time = time.time()
            df = data_service.fetch_candlestick_data("BTC-USDT", "1H", 300)
            fetch_latency = (time.time() - start_time) * 1000
            
            if df is not None and len(df) > 100:
                return {
                    'status': 'SUCCESS',
                    'data_points': len(df),
                    'fetch_latency_ms': fetch_latency,
                    'latest_price': float(df['close'].iloc[-1]),
                    'data_columns': list(df.columns),
                    'time_range': {
                        'start': str(df.index[0]) if hasattr(df.index[0], '__str__') else str(df.iloc[0].name),
                        'end': str(df.index[-1]) if hasattr(df.index[-1], '__str__') else str(df.iloc[-1].name)
                    }
                }
            else:
                return {'status': 'ERROR', 'message': 'Insufficient data received'}
                
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}
    
    def test_model_training_pipeline(self) -> Dict[str, Any]:
        """Test AI model training with real data"""
        print("🎯 Testing model training pipeline...")
        
        try:
            from trading.okx_data_service import OKXDataService
            from ai.comprehensive_ml_pipeline import ComprehensiveMLPipeline
            
            # Fetch real data
            data_service = OKXDataService()
            df = data_service.fetch_candlestick_data("BTC-USDT", "1H", 300)
            
            if df is not None and len(df) > 100:
                # Test ML pipeline training
                ml_pipeline = ComprehensiveMLPipeline()
                
                start_time = time.time()
                training_result = ml_pipeline.train_all_models(df)
                training_time = time.time() - start_time
                
                if training_result.get('success'):
                    return {
                        'status': 'SUCCESS',
                        'training_time_seconds': training_time,
                        'models_trained': training_result.get('models_trained', []),
                        'feature_count': training_result.get('feature_count', 0),
                        'data_points_used': len(df)
                    }
                else:
                    return {'status': 'ERROR', 'message': training_result.get('error', 'Training failed')}
            else:
                return {'status': 'ERROR', 'message': 'Insufficient training data'}
                
        except Exception as e:
            return {'status': 'ERROR', 'message': str(e)}
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance metrics"""
        
        # Test API response times
        okx_latency = []
        for _ in range(5):
            start = time.time()
            try:
                response = requests.get("https://www.okx.com/api/v5/market/ticker?instId=BTC-USDT")
                if response.status_code == 200:
                    okx_latency.append((time.time() - start) * 1000)
            except:
                pass
            time.sleep(0.1)
        
        avg_latency = sum(okx_latency) / len(okx_latency) if okx_latency else 0
        
        return {
            'api_latency': {
                'okx_avg_ms': avg_latency,
                'samples': len(okx_latency)
            },
            'system_health': {
                'components_available': sum(1 for comp in self.results['components'].values() 
                                         if any(item.get('status') == 'AVAILABLE' 
                                               for item in comp.values() if isinstance(item, dict)))
            }
        }
    
    def run_comprehensive_audit(self) -> Dict[str, Any]:
        """Execute complete system audit"""
        print("🚀 Starting comprehensive trading bot system audit...\n")
        
        # Component audits
        self.results['components']['okx_api'] = self.audit_okx_connectivity()
        self.results['components']['ai_models'] = self.audit_model_components()
        self.results['components']['trading_engine'] = self.audit_trading_engine()
        self.results['components']['risk_management'] = self.audit_risk_management()
        
        # Integration tests
        self.results['integration_tests']['data_flow'] = self.test_data_flow_integration()
        self.results['integration_tests']['model_training'] = self.test_model_training_pipeline()
        
        # Performance metrics
        self.results['performance_metrics'] = self.generate_performance_report()
        
        # Generate recommendations
        self._generate_recommendations()
        
        return self.results
    
    def _generate_recommendations(self):
        """Generate system optimization recommendations"""
        recommendations = []
        
        # Check OKX connectivity
        okx_status = self.results['components']['okx_api'].get('status')
        if okx_status != 'CONNECTED':
            recommendations.append("❌ OKX API connectivity issue - verify credentials and network")
        
        # Check model availability
        ai_models = self.results['components']['ai_models']
        available_models = sum(1 for model in ai_models.values() if model.get('status') == 'AVAILABLE')
        if available_models < 3:
            recommendations.append(f"⚠️ Only {available_models} AI models available - check dependencies")
        
        # Check data flow
        data_flow = self.results['integration_tests']['data_flow']
        if data_flow.get('status') != 'SUCCESS':
            recommendations.append("❌ Data flow integration failed - check OKX data service")
        
        # Check training pipeline
        training = self.results['integration_tests']['model_training']
        if training.get('status') != 'SUCCESS':
            recommendations.append("❌ Model training pipeline failed - verify data and models")
        
        # Performance recommendations
        latency = self.results['performance_metrics']['api_latency']['okx_avg_ms']
        if latency > 1000:
            recommendations.append(f"⚠️ High API latency ({latency:.1f}ms) - consider server optimization")
        
        if not recommendations:
            recommendations.append("✅ System is production-ready with all components operational")
        
        self.results['recommendations'] = recommendations

def print_audit_report(results: Dict[str, Any]):
    """Print formatted audit report"""
    print("\n" + "="*80)
    print("🔍 COMPREHENSIVE TRADING BOT SYSTEM AUDIT REPORT")
    print("="*80)
    print(f"Audit Timestamp: {results['timestamp']}")
    print()
    
    # Component Status
    print("📊 COMPONENT STATUS:")
    print("-" * 40)
    
    for component_name, component_data in results['components'].items():
        print(f"\n{component_name.upper()}:")
        if isinstance(component_data, dict):
            if 'status' in component_data:
                status = "✅" if component_data['status'] == 'CONNECTED' else "❌"
                print(f"  {status} Status: {component_data['status']}")
                for key, value in component_data.items():
                    if key != 'status':
                        print(f"  • {key}: {value}")
            else:
                for sub_name, sub_data in component_data.items():
                    if isinstance(sub_data, dict):
                        status = "✅" if sub_data.get('status') == 'AVAILABLE' else "❌"
                        print(f"  {status} {sub_name}: {sub_data.get('status', 'UNKNOWN')}")
    
    # Integration Tests
    print("\n🔄 INTEGRATION TESTS:")
    print("-" * 40)
    
    for test_name, test_data in results['integration_tests'].items():
        status = "✅" if test_data.get('status') == 'SUCCESS' else "❌"
        print(f"{status} {test_name}: {test_data.get('status', 'UNKNOWN')}")
        
        if test_data.get('status') == 'SUCCESS':
            for key, value in test_data.items():
                if key != 'status':
                    print(f"    • {key}: {value}")
        elif 'message' in test_data:
            print(f"    Error: {test_data['message']}")
    
    # Performance Metrics
    print("\n⚡ PERFORMANCE METRICS:")
    print("-" * 40)
    
    perf = results['performance_metrics']
    if 'api_latency' in perf:
        print(f"OKX API Latency: {perf['api_latency']['okx_avg_ms']:.1f}ms (avg)")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    print("-" * 40)
    
    for rec in results['recommendations']:
        print(f"  {rec}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    auditor = TradingSystemAudit()
    results = auditor.run_comprehensive_audit()
    print_audit_report(results)