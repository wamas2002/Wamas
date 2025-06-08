import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, Any, List

class AdvancedMLInterface:
    """Advanced ML interface for gradient boosting and transformer models"""
    
    def __init__(self):
        self.available_models = []
        self.training_status = {}
        
    def render_ml_dashboard(self):
        """Render the main ML dashboard"""
        st.title("🧠 Advanced Machine Learning Laboratory")
        st.markdown("**State-of-the-art ML models for cryptocurrency price prediction**")
        
        # Model availability status
        self._render_model_status()
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏗️ Model Training", 
            "🔮 Predictions", 
            "📊 Performance Analysis", 
            "🎯 Feature Engineering", 
            "🧪 Model Insights"
        ])
        
        with tab1:
            self._render_training_interface()
        
        with tab2:
            self._render_prediction_interface()
        
        with tab3:
            self._render_performance_analysis()
        
        with tab4:
            self._render_feature_engineering()
        
        with tab5:
            self._render_model_insights()
    
    def _render_model_status(self):
        """Render model availability and status"""
        st.subheader("🔧 Available Models")
        
        col1, col2, col3, col4 = st.columns(4)
        
        # Check model availability
        models_status = self._check_model_availability()
        
        with col1:
            status = "✅ Available" if models_status.get('random_forest', False) else "❌ Unavailable"
            st.metric("Random Forest", status)
        
        with col2:
            status = "✅ Available" if models_status.get('gradient_boosting', False) else "❌ Unavailable"
            st.metric("Gradient Boosting", status)
        
        with col3:
            status = "✅ Available" if models_status.get('xgboost', False) else "❌ Unavailable"
            st.metric("XGBoost", status)
        
        with col4:
            status = "✅ Available" if models_status.get('transformer', False) else "❌ Unavailable"
            st.metric("Transformer", status)
        
        # Training status
        if hasattr(st.session_state, 'advanced_ml_pipeline') and st.session_state.advanced_ml_pipeline.is_trained:
            st.success("🎯 Models are trained and ready for predictions!")
        else:
            st.info("📚 Models need training before making predictions")
    
    def _check_model_availability(self) -> Dict[str, bool]:
        """Check which models are available"""
        status = {
            'random_forest': True,
            'gradient_boosting': True,
            'xgboost': False,
            'catboost': False,
            'transformer': True
        }
        
        try:
            if hasattr(st.session_state, 'advanced_ml_pipeline'):
                available_models = list(st.session_state.advanced_ml_pipeline.model_configs.keys())
                status['xgboost'] = 'xgboost' in available_models
                status['catboost'] = 'catboost' in available_models
        except:
            pass
        
        return status
    
    def _render_training_interface(self):
        """Render model training interface"""
        st.subheader("🏗️ Model Training Laboratory")
        
        # Symbol selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbol = st.selectbox(
                "Select Cryptocurrency",
                ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "DOTUSDT", "LINKUSDT"],
                key="ml_training_symbol"
            )
        
        with col2:
            data_points = st.selectbox(
                "Training Data Size",
                [500, 1000, 2000, 5000],
                index=1,
                key="ml_training_size"
            )
        
        with col3:
            prediction_horizon = st.selectbox(
                "Prediction Horizon",
                [1, 3, 6, 12, 24],
                index=0,
                key="ml_prediction_horizon"
            )
        
        st.markdown("---")
        
        # Training options
        st.subheader("Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Model Selection**")
            train_ml_pipeline = st.checkbox("Train Gradient Boosting Models", value=True)
            train_transformer = st.checkbox("Train Transformer Ensemble", value=True)
            train_freqai = st.checkbox("Train Complete FreqAI Pipeline", value=False)
        
        with col2:
            st.markdown("**Advanced Options**")
            auto_feature_engineering = st.checkbox("Automatic Feature Engineering", value=True)
            cross_validation = st.checkbox("Cross-Validation Analysis", value=True)
            hyperparameter_tuning = st.checkbox("Hyperparameter Optimization", value=False)
        
        # Training execution
        st.markdown("---")
        
        if st.button("🚀 Start Training", type="primary", key="start_ml_training"):
            self._execute_training(
                symbol, data_points, prediction_horizon,
                train_ml_pipeline, train_transformer, train_freqai,
                auto_feature_engineering, cross_validation, hyperparameter_tuning
            )
    
    def _execute_training(self, symbol: str, data_points: int, horizon: int,
                         train_ml: bool, train_transformer: bool, train_freqai: bool,
                         auto_features: bool, cross_val: bool, hyper_opt: bool):
        """Execute model training"""
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Get training data
            status_text.text("Fetching training data from OKX...")
            progress_bar.progress(10)
            
            data = st.session_state.okx_data_service.get_historical_data(
                symbol, '1h', data_points
            )
            
            if data.empty:
                st.error("Failed to fetch training data")
                return
            
            st.success(f"✅ Fetched {len(data)} data points for {symbol}")
            progress_bar.progress(20)
            
            results = {}
            
            # Train ML Pipeline
            if train_ml and hasattr(st.session_state, 'advanced_ml_pipeline'):
                status_text.text("Training gradient boosting models...")
                progress_bar.progress(40)
                
                ml_results = st.session_state.advanced_ml_pipeline.train_all_models(data)
                results['ml_pipeline'] = ml_results
                
                if ml_results.get('success'):
                    st.success("✅ Gradient boosting models trained successfully!")
                else:
                    st.error(f"❌ ML training failed: {ml_results.get('error', 'Unknown error')}")
            
            # Train Transformer
            if train_transformer and hasattr(st.session_state, 'transformer_ensemble'):
                status_text.text("Training transformer ensemble...")
                progress_bar.progress(60)
                
                transformer_results = st.session_state.transformer_ensemble.train(data, epochs=30)
                results['transformer'] = transformer_results
                
                if transformer_results.get('success'):
                    st.success("✅ Transformer ensemble trained successfully!")
                else:
                    st.error(f"❌ Transformer training failed: {transformer_results.get('error', 'Unknown error')}")
            
            # Train FreqAI Pipeline
            if train_freqai and hasattr(st.session_state, 'freqai_pipeline'):
                status_text.text("Training complete FreqAI pipeline...")
                progress_bar.progress(80)
                
                freqai_results = st.session_state.freqai_pipeline.train_all_models(data)
                results['freqai'] = freqai_results
                
                if freqai_results.get('success'):
                    st.success("✅ FreqAI pipeline trained successfully!")
                else:
                    st.error(f"❌ FreqAI training failed: {freqai_results.get('error', 'Unknown error')}")
            
            progress_bar.progress(100)
            status_text.text("Training completed!")
            
            # Store results
            st.session_state.ml_training_results = results
            
            # Display training summary
            self._display_training_summary(results)
            
        except Exception as e:
            st.error(f"Training error: {e}")
    
    def _display_training_summary(self, results: Dict[str, Any]):
        """Display training results summary"""
        st.subheader("📈 Training Results Summary")
        
        for model_type, result in results.items():
            if result.get('success'):
                with st.expander(f"📊 {model_type.replace('_', ' ').title()} Results"):
                    
                    if model_type == 'ml_pipeline':
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Features Generated", result.get('feature_count', 0))
                            st.metric("Training Samples", result.get('sample_count', 0))
                        
                        with col2:
                            available_models = result.get('available_models', [])
                            st.write("**Trained Models:**")
                            for model in available_models:
                                st.write(f"• {model.replace('_', ' ').title()}")
                    
                    elif model_type == 'transformer':
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("R² Score", f"{result.get('r2_score', 0):.4f}")
                            st.metric("Final MSE", f"{result.get('final_mse', 0):.6f}")
                        
                        with col2:
                            st.metric("Training Sequences", result.get('num_sequences', 0))
                            st.metric("Feature Count", result.get('feature_count', 0))
                    
                    elif model_type == 'freqai':
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Features", result.get('feature_count', 0))
                            st.metric("Training Samples", result.get('sample_count', 0))
                        
                        with col2:
                            available_models = result.get('available_models', [])
                            st.write("**Ensemble Models:**")
                            st.write("• ML Pipeline + Transformer + LSTM")
    
    def _render_prediction_interface(self):
        """Render prediction interface"""
        st.subheader("🔮 AI Price Predictions")
        
        # Check if models are trained
        models_trained = (
            hasattr(st.session_state, 'advanced_ml_pipeline') and st.session_state.advanced_ml_pipeline.is_trained
        ) or (
            hasattr(st.session_state, 'transformer_ensemble') and st.session_state.transformer_ensemble.is_trained
        ) or (
            hasattr(st.session_state, 'freqai_pipeline') and st.session_state.freqai_pipeline.is_trained
        )
        
        if not models_trained:
            st.warning("⚠️ Please train models first in the Training tab")
            return
        
        # Prediction interface
        col1, col2, col3 = st.columns(3)
        
        with col1:
            pred_symbol = st.selectbox(
                "Select Cryptocurrency",
                ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "DOTUSDT", "LINKUSDT"],
                key="ml_pred_symbol"
            )
        
        with col2:
            prediction_type = st.selectbox(
                "Prediction Type",
                ["Single Model", "Ensemble Average", "FreqAI Pipeline"],
                key="ml_pred_type"
            )
        
        with col3:
            time_horizon = st.selectbox(
                "Time Horizon",
                ["1 Hour", "3 Hours", "6 Hours", "12 Hours", "24 Hours"],
                key="ml_time_horizon"
            )
        
        if st.button("🎯 Generate Prediction", type="primary", key="generate_ml_prediction"):
            self._generate_predictions(pred_symbol, prediction_type, time_horizon)
    
    def _generate_predictions(self, symbol: str, pred_type: str, time_horizon: str):
        """Generate and display predictions"""
        try:
            # Get recent data
            data = st.session_state.okx_data_service.get_historical_data(symbol, '1h', 200)
            
            if data.empty:
                st.error("Failed to fetch prediction data")
                return
            
            current_price = data['close'].iloc[-1]
            predictions = {}
            
            # Generate predictions based on type
            if pred_type == "Single Model" and hasattr(st.session_state, 'advanced_ml_pipeline'):
                result = st.session_state.advanced_ml_pipeline.predict_ensemble(data)
                if result.get('success'):
                    predictions['ML Pipeline'] = result
            
            elif pred_type == "Ensemble Average":
                # Get predictions from multiple models
                if hasattr(st.session_state, 'advanced_ml_pipeline'):
                    ml_result = st.session_state.advanced_ml_pipeline.predict_ensemble(data)
                    if ml_result.get('success'):
                        predictions['ML Pipeline'] = ml_result
                
                if hasattr(st.session_state, 'transformer_ensemble'):
                    transformer_result = st.session_state.transformer_ensemble.predict(data)
                    if transformer_result.get('success'):
                        predictions['Transformer'] = transformer_result
            
            elif pred_type == "FreqAI Pipeline" and hasattr(st.session_state, 'freqai_pipeline'):
                result = st.session_state.freqai_pipeline.predict_ensemble(data)
                if result.get('success'):
                    predictions['FreqAI'] = result
            
            # Display predictions
            self._display_predictions(symbol, current_price, predictions, time_horizon)
            
        except Exception as e:
            st.error(f"Prediction error: {e}")
    
    def _display_predictions(self, symbol: str, current_price: float, 
                           predictions: Dict[str, Any], time_horizon: str):
        """Display prediction results"""
        if not predictions:
            st.warning("No predictions generated")
            return
        
        st.subheader(f"🎯 Predictions for {symbol}")
        
        # Current price
        st.metric("Current Price", f"${current_price:.4f}")
        
        # Prediction results
        for model_name, pred_result in predictions.items():
            with st.expander(f"📊 {model_name} Prediction"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    predicted_price = pred_result.get('predicted_price', current_price)
                    price_change = predicted_price - current_price
                    price_change_pct = (price_change / current_price) * 100
                    
                    st.metric(
                        f"Predicted Price ({time_horizon})",
                        f"${predicted_price:.4f}",
                        f"{price_change_pct:+.2f}%"
                    )
                
                with col2:
                    confidence = pred_result.get('confidence', 0) * 100
                    st.metric("Confidence", f"{confidence:.1f}%")
                
                with col3:
                    ensemble_pred = pred_result.get('ensemble_prediction', 0) * 100
                    st.metric("Return Prediction", f"{ensemble_pred:+.2f}%")
                
                # Individual model predictions if available
                individual_preds = pred_result.get('individual_predictions', {})
                if individual_preds:
                    st.write("**Individual Model Predictions:**")
                    for model, pred_value in individual_preds.items():
                        pred_pct = pred_value * 100
                        st.write(f"• {model.replace('_', ' ').title()}: {pred_pct:+.2f}%")
    
    def _render_performance_analysis(self):
        """Render model performance analysis"""
        st.subheader("📊 Model Performance Analysis")
        
        # Check for training results
        if not hasattr(st.session_state, 'ml_training_results'):
            st.info("📚 Train models first to see performance analysis")
            return
        
        results = st.session_state.ml_training_results
        
        # Performance metrics
        if 'ml_pipeline' in results and results['ml_pipeline'].get('success'):
            self._render_ml_performance()
        
        if 'transformer' in results and results['transformer'].get('success'):
            self._render_transformer_performance()
        
        if 'freqai' in results and results['freqai'].get('success'):
            self._render_freqai_performance()
    
    def _render_ml_performance(self):
        """Render ML pipeline performance"""
        st.subheader("🎯 Gradient Boosting Models Performance")
        
        try:
            if hasattr(st.session_state, 'advanced_ml_pipeline'):
                performance = st.session_state.advanced_ml_pipeline.get_model_performance()
                
                if performance.get('success'):
                    model_perf = performance['model_performance']
                    
                    # Performance comparison chart
                    models = list(model_perf.keys())
                    cv_scores = [model_perf[model]['cross_validation_score'] for model in models]
                    
                    fig = px.bar(
                        x=models,
                        y=cv_scores,
                        title="Model Cross-Validation Scores (Lower is Better)",
                        labels={'x': 'Models', 'y': 'CV Score (MSE)'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed metrics
                    for model, metrics in model_perf.items():
                        with st.expander(f"📈 {model.replace('_', ' ').title()} Metrics"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("CV Score", f"{metrics['cross_validation_score']:.6f}")
                            
                            with col2:
                                st.metric("CV Std", f"{metrics['cv_std']:.6f}")
                            
                            with col3:
                                st.metric("Stability", f"{metrics['stability']:.3f}")
                
        except Exception as e:
            st.error(f"Error displaying ML performance: {e}")
    
    def _render_transformer_performance(self):
        """Render transformer performance"""
        st.subheader("🧠 Transformer Ensemble Performance")
        
        try:
            if hasattr(st.session_state, 'transformer_ensemble'):
                # Get training results
                training_results = st.session_state.ml_training_results.get('transformer', {})
                
                if training_results.get('success'):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("R² Score", f"{training_results.get('r2_score', 0):.4f}")
                        st.metric("Training Sequences", training_results.get('num_sequences', 0))
                    
                    with col2:
                        st.metric("Final MSE", f"{training_results.get('final_mse', 0):.6f}")
                        st.metric("Feature Count", training_results.get('feature_count', 0))
                    
                    # Training loss plot
                    losses = training_results.get('training_losses', [])
                    if losses:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=losses,
                            mode='lines',
                            name='Training Loss',
                            line=dict(color='blue')
                        ))
                        fig.update_layout(
                            title="Transformer Training Loss",
                            xaxis_title="Epoch",
                            yaxis_title="Loss",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error displaying transformer performance: {e}")
    
    def _render_freqai_performance(self):
        """Render FreqAI pipeline performance"""
        st.subheader("🚀 FreqAI Pipeline Performance")
        
        try:
            if hasattr(st.session_state, 'freqai_pipeline'):
                training_results = st.session_state.ml_training_results.get('freqai', {})
                
                if training_results.get('success'):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Features", training_results.get('feature_count', 0))
                        st.metric("Training Samples", training_results.get('sample_count', 0))
                    
                    with col2:
                        ensemble_perf = training_results.get('results', {}).get('ensemble_performance', {})
                        if ensemble_perf and 'mse' in ensemble_perf:
                            st.metric("Ensemble MSE", f"{ensemble_perf['mse']:.6f}")
                            st.metric("Direction Accuracy", f"{ensemble_perf.get('direction_accuracy', 0)*100:.1f}%")
        
        except Exception as e:
            st.error(f"Error displaying FreqAI performance: {e}")
    
    def _render_feature_engineering(self):
        """Render feature engineering analysis"""
        st.subheader("🎯 Feature Engineering Analysis")
        
        if not hasattr(st.session_state, 'advanced_ml_pipeline') or not st.session_state.advanced_ml_pipeline.is_trained:
            st.info("📚 Train models first to see feature analysis")
            return
        
        try:
            # Get feature importance
            importance_result = st.session_state.advanced_ml_pipeline.get_feature_importance_summary()
            
            if importance_result.get('success'):
                top_features = importance_result['top_features']
                
                # Feature importance chart
                if top_features:
                    feature_names = [f[0] for f in top_features]
                    importances = [f[1]['mean'] for f in top_features]
                    
                    fig = px.bar(
                        x=importances,
                        y=feature_names,
                        orientation='h',
                        title="Top 20 Most Important Features",
                        labels={'x': 'Importance Score', 'y': 'Features'}
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature categories analysis
                st.subheader("📊 Feature Categories")
                self._analyze_feature_categories(top_features)
        
        except Exception as e:
            st.error(f"Error displaying feature analysis: {e}")
    
    def _analyze_feature_categories(self, top_features: List):
        """Analyze and categorize features"""
        categories = {
            'Price Features': [],
            'Technical Indicators': [],
            'Volume Features': [],
            'Statistical Features': [],
            'Time Features': []
        }
        
        for feature_name, importance in top_features:
            name = feature_name.lower()
            
            if any(keyword in name for keyword in ['price', 'close', 'open', 'high', 'low', 'sma', 'ema']):
                categories['Price Features'].append((feature_name, importance['mean']))
            elif any(keyword in name for keyword in ['rsi', 'macd', 'bb_', 'momentum', 'roc']):
                categories['Technical Indicators'].append((feature_name, importance['mean']))
            elif any(keyword in name for keyword in ['volume', 'vol_']):
                categories['Volume Features'].append((feature_name, importance['mean']))
            elif any(keyword in name for keyword in ['skewness', 'kurtosis', 'std', 'quantile']):
                categories['Statistical Features'].append((feature_name, importance['mean']))
            elif any(keyword in name for keyword in ['hour', 'day', 'sin', 'cos']):
                categories['Time Features'].append((feature_name, importance['mean']))
        
        # Display categories
        for category, features in categories.items():
            if features:
                with st.expander(f"📈 {category} ({len(features)} features)"):
                    for feature, importance in features[:10]:  # Show top 10 per category
                        st.write(f"• **{feature}**: {importance:.4f}")
    
    def _render_model_insights(self):
        """Render advanced model insights"""
        st.subheader("🧪 Model Insights & Interpretability")
        
        # Attention analysis for transformer
        if hasattr(st.session_state, 'transformer_ensemble') and st.session_state.transformer_ensemble.is_trained:
            st.subheader("🧠 Transformer Attention Analysis")
            
            symbol = st.selectbox(
                "Select Symbol for Attention Analysis",
                ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
                key="attention_symbol"
            )
            
            if st.button("🔍 Analyze Attention Patterns", key="analyze_attention"):
                try:
                    data = st.session_state.okx_data_service.get_historical_data(symbol, '1h', 100)
                    attention_result = st.session_state.transformer_ensemble.get_attention_analysis(data)
                    
                    if attention_result.get('success'):
                        analysis = attention_result['analysis']
                        
                        st.write("**Attention Pattern Summary:**")
                        for layer_info in analysis['attention_summary']:
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric(f"Layer {layer_info['layer']} - Max Position", 
                                        layer_info['max_attention_position'])
                            
                            with col2:
                                st.metric(f"Layer {layer_info['layer']} - Attention Value", 
                                        f"{layer_info['max_attention_value']:.4f}")
                            
                            with col3:
                                st.metric(f"Layer {layer_info['layer']} - Focus Score", 
                                        f"{layer_info['attention_focus']:.2f}")
                
                except Exception as e:
                    st.error(f"Attention analysis error: {e}")
        
        # Model comparison
        if hasattr(st.session_state, 'freqai_pipeline') and st.session_state.freqai_pipeline.is_trained:
            st.subheader("⚖️ Model Comparison")
            
            if st.button("📊 Generate Model Insights", key="model_insights"):
                try:
                    data = st.session_state.okx_data_service.get_historical_data("BTCUSDT", '1h', 200)
                    insights = st.session_state.freqai_pipeline.get_model_insights(data)
                    
                    if insights.get('success'):
                        insights_data = insights['insights']
                        
                        # Display ensemble configuration
                        if 'ensemble_configuration' in insights_data:
                            config = insights_data['ensemble_configuration']
                            
                            st.write("**Ensemble Weights:**")
                            weights = config.get('weights', {})
                            for model, weight in weights.items():
                                st.write(f"• {model.replace('_', ' ').title()}: {weight:.1%}")
                
                except Exception as e:
                    st.error(f"Model insights error: {e}")