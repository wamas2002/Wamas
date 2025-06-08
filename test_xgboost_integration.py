#!/usr/bin/env python3
"""
Test XGBoost Integration with Live OKX Market Data
Verifies advanced ML models are working with authentic data feeds
"""

import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def test_xgboost_import():
    """Test XGBoost import and basic functionality"""
    try:
        import xgboost as xgb
        print(f"✅ XGBoost successfully imported - Version: {xgb.__version__}")
        return True
    except ImportError as e:
        print(f"❌ XGBoost import failed: {e}")
        return False

def test_xgboost_with_live_data():
    """Test XGBoost training with authentic OKX market data"""
    try:
        import xgboost as xgb
        from trading.okx_data_service import OKXDataService
        from ai.comprehensive_ml_pipeline import ComprehensiveMLPipeline
        
        print("📡 Fetching live OKX market data...")
        data_service = OKXDataService()
        
        # Get authentic BTC data
        df = data_service.get_historical_data('BTC-USDT', '1H', 200)
        
        if df is None or len(df) < 100:
            print("❌ Failed to fetch sufficient live market data")
            return False
            
        print(f"✅ Retrieved {len(df)} authentic market data points")
        print(f"📊 Latest BTC price: ${df['close'].iloc[-1]:.2f}")
        
        # Test comprehensive ML pipeline with XGBoost
        print("🧠 Testing comprehensive ML pipeline...")
        ml_pipeline = ComprehensiveMLPipeline()
        
        # Generate features
        features_df = ml_pipeline.generate_advanced_features(df)
        print(f"🔧 Generated {len(features_df.columns)} features from authentic data")
        
        # Prepare training data
        X, y, feature_names = ml_pipeline.prepare_training_data(features_df)
        print(f"📈 Training data prepared: {X.shape}")
        
        # Test XGBoost specifically
        print("🚀 Testing XGBoost model training...")
        
        # Create XGBoost model
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        
        # Train on authentic data
        xgb_model.fit(X, y)
        
        # Make predictions
        predictions = xgb_model.predict(X[-10:])  # Last 10 samples
        
        print(f"✅ XGBoost training successful")
        print(f"📊 Sample predictions: {predictions[:3]}")
        print(f"🎯 Feature importance shape: {len(xgb_model.feature_importances_)}")
        
        # Test full pipeline
        print("🔄 Testing full ML pipeline with XGBoost...")
        results = ml_pipeline.train_all_models(df)
        
        if results.get('success'):
            print("✅ Complete ML pipeline successful with authentic OKX data")
            print(f"📊 Models trained: {list(results.get('models', {}).keys())}")
            return True
        else:
            print("❌ ML pipeline training failed")
            return False
            
    except Exception as e:
        print(f"❌ XGBoost testing failed: {e}")
        return False

def test_enhanced_gradient_boosting():
    """Test enhanced gradient boosting pipeline with XGBoost"""
    try:
        from ai.enhanced_gradient_boosting import EnhancedGradientBoostingPipeline
        from trading.okx_data_service import OKXDataService
        
        print("⚡ Testing enhanced gradient boosting with XGBoost...")
        
        data_service = OKXDataService()
        df = data_service.get_historical_data('ETH-USDT', '1H', 150)
        
        if df is None:
            print("❌ Failed to fetch ETH data")
            return False
            
        print(f"📊 Latest ETH price: ${df['close'].iloc[-1]:.2f}")
        
        # Test enhanced pipeline
        enhanced_pipeline = EnhancedGradientBoostingPipeline()
        results = enhanced_pipeline.train_all_models(df)
        
        if results.get('success'):
            models = results.get('models', {})
            print(f"✅ Enhanced gradient boosting successful")
            print(f"🧠 Trained models: {list(models.keys())}")
            
            # Test predictions
            predictions = enhanced_pipeline.predict_ensemble(df)
            if predictions.get('success'):
                print("✅ Ensemble predictions successful")
                print(f"📈 Prediction confidence: {predictions.get('confidence', 0):.2f}")
                return True
                
        print("❌ Enhanced gradient boosting failed")
        return False
        
    except Exception as e:
        print(f"❌ Enhanced gradient boosting test failed: {e}")
        return False

def verify_all_ml_models():
    """Verify all ML models are working with authentic data"""
    try:
        from ai.comprehensive_ml_pipeline import ComprehensiveMLPipeline
        from ai.freqai_pipeline import FreqAILevelPipeline
        from ai.lstm_predictor import AdvancedLSTMPredictor
        from ai.prophet_predictor import AdvancedProphetPredictor
        from trading.okx_data_service import OKXDataService
        
        print("🔬 Comprehensive ML models verification...")
        
        data_service = OKXDataService()
        df = data_service.get_historical_data('ADA-USDT', '1H', 200)
        
        if df is None:
            print("❌ Failed to fetch ADA data")
            return False
            
        print(f"📊 Latest ADA price: ${df['close'].iloc[-1]:.4f}")
        
        models_status = {}
        
        # Test Comprehensive ML
        try:
            comp_ml = ComprehensiveMLPipeline()
            result = comp_ml.train_all_models(df)
            models_status['comprehensive_ml'] = result.get('success', False)
            print(f"{'✅' if models_status['comprehensive_ml'] else '❌'} Comprehensive ML Pipeline")
        except Exception as e:
            models_status['comprehensive_ml'] = False
            print(f"❌ Comprehensive ML failed: {e}")
        
        # Test FreqAI
        try:
            freqai = FreqAILevelPipeline()
            result = freqai.train_all_models(df)
            models_status['freqai'] = result.get('success', False)
            print(f"{'✅' if models_status['freqai'] else '❌'} FreqAI Pipeline")
        except Exception as e:
            models_status['freqai'] = False
            print(f"❌ FreqAI failed: {e}")
        
        # Test LSTM
        try:
            lstm = AdvancedLSTMPredictor()
            result = lstm.train(df)
            models_status['lstm'] = result.get('success', False)
            print(f"{'✅' if models_status['lstm'] else '❌'} LSTM Predictor")
        except Exception as e:
            models_status['lstm'] = False
            print(f"❌ LSTM failed: {e}")
        
        # Test Prophet
        try:
            prophet = AdvancedProphetPredictor()
            result = prophet.train(df)
            models_status['prophet'] = result.get('success', False)
            print(f"{'✅' if models_status['prophet'] else '❌'} Prophet Predictor")
        except Exception as e:
            models_status['prophet'] = False
            print(f"❌ Prophet failed: {e}")
        
        success_count = sum(models_status.values())
        total_models = len(models_status)
        
        print(f"\n📊 Model Status Summary:")
        print(f"✅ Working models: {success_count}/{total_models}")
        print(f"📈 Success rate: {success_count/total_models*100:.1f}%")
        
        return success_count >= 3  # At least 3 models should work
        
    except Exception as e:
        print(f"❌ ML models verification failed: {e}")
        return False

def main():
    """Run comprehensive XGBoost and ML model tests"""
    print("🚀 XGBoost Integration and ML Models Test")
    print("=" * 50)
    
    test_results = {}
    
    # Test 1: XGBoost Import
    print("\n1️⃣ Testing XGBoost Import...")
    test_results['xgboost_import'] = test_xgboost_import()
    
    # Test 2: XGBoost with Live Data
    print("\n2️⃣ Testing XGBoost with Live OKX Data...")
    test_results['xgboost_live_data'] = test_xgboost_with_live_data()
    
    # Test 3: Enhanced Gradient Boosting
    print("\n3️⃣ Testing Enhanced Gradient Boosting...")
    test_results['enhanced_gb'] = test_enhanced_gradient_boosting()
    
    # Test 4: All ML Models Verification
    print("\n4️⃣ Verifying All ML Models...")
    test_results['all_models'] = verify_all_ml_models()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    success_rate = sum(test_results.values()) / len(test_results) * 100
    print(f"\n🎯 Overall Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 75:
        print("🚀 XGBoost and ML models ready for production!")
        return True
    else:
        print("⚠️  Some issues detected - review failed tests")
        return False

if __name__ == "__main__":
    main()