import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any
import time
from .charts import ChartManager
from .components import UIComponents
from config import Config

class TradingDashboard:
    """Main trading dashboard interface"""
    
    def __init__(self):
        self.chart_manager = ChartManager()
        self.ui_components = UIComponents()
        
    def render_live_trading_tab(self, symbol: str, timeframe: str, strategy: str):
        """Render the live trading tab"""
        try:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(f"📊 {symbol} Live Chart")
                
                # Get market data from OKX API
                if 'okx_data_service' in st.session_state:
                    okx_service = st.session_state.okx_data_service
                    data = okx_service.get_historical_data(symbol, timeframe, 500)
                    
                    if not data.empty:
                        
                        # Create candlestick chart
                        chart = self.chart_manager.create_candlestick_chart(
                            data, symbol, timeframe
                        )
                        st.plotly_chart(chart, use_container_width=True)
                        
                        # Technical indicators chart
                        st.subheader("Technical Indicators")
                        indicators_chart = self.chart_manager.create_indicators_chart(data)
                        st.plotly_chart(indicators_chart, use_container_width=True)
                        
                    else:
                        st.warning(f"No market data available for {symbol}")
                        st.info("The system is loading market data. Please wait...")
                else:
                    st.error("Trading engine not initialized")
            
            with col2:
                st.subheader("🎯 Trading Controls")
                
                # Current price display
                self._render_price_display(symbol)
                
                # Position summary
                self._render_position_summary(symbol)
                
                # Recent signals
                self._render_recent_signals()
                
                # Manual trading controls
                st.subheader("Manual Trading")
                self._render_manual_trading_controls(symbol)
                
        except Exception as e:
            st.error(f"Error rendering live trading tab: {e}")
    
    def render_ai_insights_tab(self, symbol: str, timeframe: str):
        """Render the AI insights tab"""
        try:
            st.subheader("🧠 AI Model Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Model Predictions")
                
                if 'ai_predictor' in st.session_state:
                    predictor = st.session_state.ai_predictor
                    
                    if predictor.is_trained:
                        # Get recent predictions
                        if (hasattr(st.session_state, 'trading_engine') and 
                            st.session_state.trading_engine and
                            hasattr(st.session_state.trading_engine, 'market_data') and
                            symbol in st.session_state.trading_engine.market_data):
                            
                            data = st.session_state.trading_engine.market_data[symbol]
                            predictions = predictor.predict(data)
                            
                            if isinstance(predictions, dict) and 'error' not in predictions:
                                self._render_ai_predictions(predictions)
                            else:
                                error_msg = predictions.get('error', 'Unknown error') if isinstance(predictions, dict) else 'Prediction failed'
                                st.warning(f"AI prediction error: {error_msg}")
                        else:
                            st.info("No market data available for predictions")
                    else:
                        st.info("AI models are not yet trained")
                        if st.button("Train AI Models"):
                            self._train_ai_models(symbol)
                else:
                    st.error("AI predictor not available")
            
            with col2:
                st.subheader("Model Performance")
                self._render_model_performance()
                
                st.subheader("Feature Importance")
                self._render_feature_importance()
            
            # Ensemble strategy insights
            st.subheader("Ensemble Strategy Analysis")
            self._render_ensemble_insights(symbol)
            
            # Market regime detection
            st.subheader("Market Regime Detection")
            self._render_market_regime(symbol)
            
        except Exception as e:
            st.error(f"Error rendering AI insights tab: {e}")
    
    def render_backtesting_tab(self, symbol: str, timeframe: str, strategy: str):
        """Render the backtesting tab"""
        try:
            st.subheader("📈 Strategy Backtesting")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Backtest Parameters")
                
                # Date range selection
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                selected_start = st.date_input("Start Date", start_date)
                selected_end = st.date_input("End Date", end_date)
                
                # Strategy selection
                selected_strategy = st.selectbox(
                    "Strategy",
                    options=["ensemble", "ml"],
                    index=0
                )
                
                # Backtest parameters
                initial_capital = st.number_input("Initial Capital ($)", value=10000, min_value=1000)
                
                if st.button("Run Backtest", type="primary"):
                    self._run_backtest(symbol, selected_strategy, selected_start, selected_end)
            
            with col2:
                st.subheader("Backtest Results")
                
                if 'backtest_results' in st.session_state:
                    results = st.session_state.backtest_results
                    
                    if 'error' not in results:
                        self._render_backtest_results(results)
                    else:
                        st.error(f"Backtest error: {results['error']}")
                else:
                    st.info("Run a backtest to see results")
            
            # Performance comparison
            st.subheader("Strategy Comparison")
            self._render_strategy_comparison()
            
        except Exception as e:
            st.error(f"Error rendering backtesting tab: {e}")
    
    def render_futures_tab(self, symbol: str, timeframe: str):
        """Render the futures trading tab"""
        try:
            st.subheader("⚡ Futures Trading Mode")
            
            # Futures disclaimer
            st.warning("⚠️ Futures trading involves leverage and higher risk. Using authentic OKX market data.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Leverage Settings")
                
                leverage = st.slider("Leverage", min_value=1, max_value=100, value=1)
                st.info(f"Using {leverage}x leverage")
                
                # Position calculator
                st.subheader("Position Calculator")
                margin = st.number_input("Margin ($)", value=1000, min_value=100)
                
                if 'okx_data_service' in st.session_state:
                    current_price = st.session_state.okx_data_service.get_current_price(symbol)
                    if current_price > 0:
                        position_size = (margin * leverage) / current_price
                        
                        st.metric("Position Size", f"{position_size:.6f} {symbol.replace('USDT', '')}")
                        st.metric("Notional Value", f"${margin * leverage:,.2f}")
                    else:
                        st.warning("Unable to fetch current price")
                
                # Risk warning
                st.error(f"⚠️ Risk: With {leverage}x leverage, a {100/leverage:.1f}% move against you will liquidate your position")
            
            with col2:
                st.subheader("Futures Analytics")
                
                # Real funding rate from OKX
                if 'okx_data_service' in st.session_state:
                    funding_rate = st.session_state.okx_data_service.get_funding_rate(symbol)
                    if funding_rate is not None:
                        st.metric("Funding Rate", f"{funding_rate:.4f}%")
                    else:
                        st.metric("Funding Rate", "Loading...")
                else:
                    st.metric("Funding Rate", "API Required")
                
                # Liquidation calculator
                if symbol in st.session_state.get('trading_engine', {}).get('latest_prices', {}):
                    current_price = st.session_state.trading_engine.latest_prices[symbol]
                    liquidation_long = current_price * (1 - 0.8/leverage)
                    liquidation_short = current_price * (1 + 0.8/leverage)
                    
                    st.metric("Liquidation Price (Long)", f"${liquidation_long:.2f}")
                    st.metric("Liquidation Price (Short)", f"${liquidation_short:.2f}")
            
            # Futures-specific charts
            st.subheader("Futures Market Analysis")
            self._render_futures_analysis(symbol)
            
        except Exception as e:
            st.error(f"Error rendering futures tab: {e}")
    
    def render_performance_tab(self):
        """Render the performance tracking tab"""
        try:
            st.subheader("📋 Portfolio Performance")
            
            if 'trading_engine' in st.session_state:
                engine = st.session_state.trading_engine
                
                # Portfolio summary
                portfolio_summary = engine.get_portfolio_summary()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Portfolio Value",
                        f"${portfolio_summary.get('portfolio_value', 0):,.2f}",
                        delta=f"${portfolio_summary.get('portfolio_value', 10000) - 10000:,.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Cash Balance",
                        f"${portfolio_summary.get('cash_balance', 0):,.2f}"
                    )
                
                with col3:
                    st.metric(
                        "Positions Value",
                        f"${portfolio_summary.get('positions_value', 0):,.2f}"
                    )
                
                with col4:
                    st.metric(
                        "Total Trades",
                        portfolio_summary.get('total_trades', 0)
                    )
                
                # Performance chart
                if engine.portfolio_history:
                    st.subheader("Portfolio Value Over Time")
                    portfolio_chart = self._create_portfolio_chart(engine.portfolio_history)
                    st.plotly_chart(portfolio_chart, use_container_width=True)
                
                # Performance metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Performance Metrics")
                    metrics = engine.get_performance_metrics()
                    
                    if 'error' not in metrics:
                        self._render_performance_metrics(metrics)
                    else:
                        st.info("Insufficient data for performance metrics")
                
                with col2:
                    st.subheader("Recent Trades")
                    recent_trades = engine.get_recent_trades(10)
                    
                    if recent_trades:
                        trades_df = pd.DataFrame(recent_trades)
                        st.dataframe(trades_df, use_container_width=True)
                    else:
                        st.info("No trades executed yet")
                
                # Risk metrics
                st.subheader("Risk Analysis")
                self._render_risk_analysis()
                
            else:
                st.error("Trading engine not available")
                
        except Exception as e:
            st.error(f"Error rendering performance tab: {e}")
    
    def _render_price_display(self, symbol: str):
        """Render current price display"""
        try:
            if 'okx_data_service' in st.session_state:
                okx_service = st.session_state.okx_data_service
                current_price = okx_service.get_current_price(symbol)
                
                if current_price > 0:
                    # Get 24hr ticker data
                    ticker_data = okx_service.get_24hr_ticker(symbol)
                    change_24h = ticker_data.get('price_change_percent', 0)
                    
                    st.metric(
                        f"{symbol} Price",
                        f"${current_price:.4f}",
                        delta=f"{change_24h:.2f}%"
                    )
                    
                    # Last update time
                    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
                else:
                    st.warning(f"Unable to fetch price for {symbol}")
            else:
                st.warning(f"Data service not available")
                
        except Exception as e:
            st.error(f"Error displaying price: {e}")
    
    def _render_position_summary(self, symbol: str):
        """Render current position summary"""
        try:
            if 'trading_engine' in st.session_state:
                engine = st.session_state.trading_engine
                position = engine.current_positions.get(symbol, 0.0)
                
                if position > 0:
                    if 'okx_data_service' in st.session_state:
                        current_price = st.session_state.okx_data_service.get_current_price(symbol)
                        if current_price > 0:
                            current_value = position * current_price
                            st.success(f"Position: {position:.6f} {symbol.replace('USDT', '')}")
                            st.info(f"Value: ${current_value:.2f}")
                        else:
                            st.success(f"Position: {position:.6f} {symbol.replace('USDT', '')}")
                    else:
                        st.success(f"Position: {position:.6f} {symbol.replace('USDT', '')}")
                else:
                    st.info("No position")
                    
        except Exception as e:
            st.error(f"Error displaying position: {e}")
    
    def _render_recent_signals(self):
        """Render recent trading signals"""
        try:
            st.subheader("Recent Signals")
            
            if 'trading_engine' in st.session_state:
                recent_signals = st.session_state.trading_engine.get_recent_signals(5)
                
                if recent_signals:
                    for signal in reversed(recent_signals):
                        timestamp = signal.get('timestamp', datetime.now())
                        signal_data = signal.get('signal', {})
                        signal_type = signal_data.get('signal', 'HOLD')
                        confidence = signal_data.get('confidence', 0)
                        
                        # Color coding for signals
                        if signal_type == 'BUY':
                            st.success(f"🟢 {signal_type} - {confidence:.1%} confidence")
                        elif signal_type == 'SELL':
                            st.error(f"🔴 {signal_type} - {confidence:.1%} confidence")
                        else:
                            st.info(f"🟡 {signal_type} - {confidence:.1%} confidence")
                        
                        st.caption(f"{timestamp.strftime('%H:%M:%S')}")
                else:
                    st.info("No signals generated yet")
            else:
                st.warning("Trading engine not available")
                
        except Exception as e:
            st.error(f"Error displaying signals: {e}")
    
    def _render_manual_trading_controls(self, symbol: str):
        """Render manual trading controls"""
        try:
            # Manual buy/sell buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🟢 Manual Buy", type="primary"):
                    st.success("Manual buy order placed (simulation)")
            
            with col2:
                if st.button("🔴 Manual Sell", type="secondary"):
                    st.success("Manual sell order placed (simulation)")
            
            # Quick actions
            st.subheader("Quick Actions")
            if st.button("Close All Positions"):
                st.info("All positions closed (simulation)")
            
            if st.button("Emergency Stop"):
                st.warning("Emergency stop activated (simulation)")
                
        except Exception as e:
            st.error(f"Error rendering manual controls: {e}")
    
    def _render_ai_predictions(self, predictions: Dict[str, Any]):
        """Render AI model predictions"""
        try:
            # Individual model predictions
            st.subheader("Individual Model Predictions")
            
            models = ['lstm', 'prophet', 'transformer']
            cols = st.columns(len(models))
            
            for i, model in enumerate(models):
                with cols[i]:
                    if model in predictions:
                        prediction = predictions[model]
                        st.metric(
                            f"{model.upper()}",
                            f"{prediction:.2%}",
                            delta=None
                        )
                    else:
                        st.metric(f"{model.upper()}", "N/A")
            
            # Ensemble prediction
            if 'ensemble' in predictions:
                st.subheader("Ensemble Prediction")
                ensemble_pred = predictions['ensemble']
                confidence = predictions.get('confidence', 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Ensemble Prediction", f"{ensemble_pred:.2%}")
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # Prediction visualization
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = ensemble_pred * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Prediction (%)"},
                    delta = {'reference': 0},
                    gauge = {
                        'axis': {'range': [-10, 10]},
                        'bar': {'color': "lightgreen" if ensemble_pred > 0 else "lightcoral"},
                        'steps': [
                            {'range': [-10, -2], 'color': "red"},
                            {'range': [-2, 2], 'color': "yellow"},
                            {'range': [2, 10], 'color': "green"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 0
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error rendering AI predictions: {e}")
    
    def _render_model_performance(self):
        """Render model performance metrics"""
        try:
            if 'ai_predictor' in st.session_state:
                predictor = st.session_state.ai_predictor
                performance = predictor.get_model_performance()
                
                if performance:
                    for model_name, metrics in performance.items():
                        st.subheader(f"{model_name.upper()} Performance")
                        
                        if isinstance(metrics, dict):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Mean Score", f"{metrics.get('mean_score', 0):.3f}")
                            with col2:
                                st.metric("Std Score", f"{metrics.get('std_score', 0):.3f}")
                else:
                    st.info("No performance data available")
            else:
                st.warning("AI predictor not available")
                
        except Exception as e:
            st.error(f"Error rendering model performance: {e}")
    
    def _render_feature_importance(self):
        """Render feature importance"""
        try:
            if (hasattr(st.session_state, 'trading_engine') and 
                st.session_state.trading_engine and
                hasattr(st.session_state.trading_engine, 'strategies') and
                'ml' in st.session_state.trading_engine.strategies):
                
                ml_strategy = st.session_state.trading_engine.strategies['ml']
                importance = ml_strategy.get_feature_importance()
                
                if importance and isinstance(importance, dict):
                    # Show feature importance for first model
                    model_name = list(importance.keys())[0]
                    features = importance[model_name]
                    
                    if isinstance(features, dict) and features:
                        # Create bar chart
                        sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:10]
                        
                        fig = px.bar(
                            x=[f[1] for f in sorted_features],
                            y=[f[0] for f in sorted_features],
                            orientation='h',
                            title="Top 10 Important Features"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No feature importance data available")
                else:
                    st.info("No feature importance data available")
            else:
                st.info("ML strategy not available")
                
        except Exception as e:
            st.error(f"Error rendering feature importance: {e}")
    
    def _render_ensemble_insights(self, symbol: str):
        """Render ensemble strategy insights"""
        try:
            if (hasattr(st.session_state, 'trading_engine') and 
                st.session_state.trading_engine and
                hasattr(st.session_state.trading_engine, 'strategies') and
                'ensemble' in st.session_state.trading_engine.strategies):
                
                ensemble = st.session_state.trading_engine.strategies['ensemble']
                summary = ensemble.get_strategy_summary()
                
                if isinstance(summary, dict) and 'error' not in summary:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Signals", summary.get('total_signals', 0))
                        st.metric("Avg Confidence", f"{summary.get('avg_confidence', 0):.1%}")
                    
                    with col2:
                        st.metric("Avg Strength", f"{summary.get('avg_strength', 0):.2f}")
                    
                    # Signal distribution
                    signal_dist = summary.get('signal_distribution', {})
                    if signal_dist and isinstance(signal_dist, dict):
                        fig = px.pie(
                            values=list(signal_dist.values()),
                            names=list(signal_dist.keys()),
                            title="Signal Distribution"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No ensemble data available")
            else:
                st.info("Ensemble strategy not available")
                
        except Exception as e:
            st.error(f"Error rendering ensemble insights: {e}")
    
    def _render_market_regime(self, symbol: str):
        """Render market regime detection"""
        try:
            if 'regime_detector' in st.session_state and 'okx_data_service' in st.session_state:
                okx_service = st.session_state.okx_data_service
                data = okx_service.get_historical_data(symbol, '1h', 500)
                
                if not data.empty:
                    regime_detector = st.session_state.regime_detector
                    regime_result = regime_detector.detect_regime(data)
                    
                    st.metric("Current Market Regime", regime_result['regime'].upper())
                    st.metric("Confidence", f"{regime_result['confidence']:.1%}")
                    st.info(regime_result['description'])
                else:
                    st.warning("Unable to fetch market data for regime detection")
            else:
                st.warning("Regime detection not available")
                
        except Exception as e:
            st.error(f"Error rendering market regime: {e}")
    
    def _train_ai_models(self, symbol: str):
        """Train AI models"""
        try:
            if 'okx_data_service' in st.session_state and 'ai_predictor' in st.session_state:
                okx_service = st.session_state.okx_data_service
                data = okx_service.get_historical_data(symbol, '1h', 1000)
                
                if not data.empty:
                    with st.spinner("Training AI models..."):
                        predictor = st.session_state.ai_predictor
                        results = predictor.train_models(data)
                        
                        if 'error' not in results:
                            st.success("AI models trained successfully!")
                        else:
                            st.error(f"Training failed: {results['error']}")
                else:
                    st.error("Unable to fetch market data for training")
            else:
                st.error("AI components not available")
                
        except Exception as e:
            st.error(f"Error training AI models: {e}")
    
    def _run_backtest(self, symbol: str, strategy: str, start_date, end_date):
        """Run backtest"""
        try:
            if 'trading_engine' in st.session_state:
                with st.spinner("Running backtest..."):
                    engine = st.session_state.trading_engine
                    results = engine.backtest_strategy(
                        symbol, strategy, start_date.isoformat(), end_date.isoformat()
                    )
                    
                    st.session_state.backtest_results = results
                    
                    if 'error' not in results:
                        st.success("Backtest completed successfully!")
                    else:
                        st.error(f"Backtest failed: {results['error']}")
            else:
                st.error("Trading engine not available")
                
        except Exception as e:
            st.error(f"Error running backtest: {e}")
    
    def _render_backtest_results(self, results: Dict[str, Any]):
        """Render backtest results"""
        try:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Initial Value",
                    f"${results.get('initial_value', 0):,.2f}"
                )
            
            with col2:
                st.metric(
                    "Final Value",
                    f"${results.get('final_value', 0):,.2f}"
                )
            
            with col3:
                total_return = results.get('total_return', 0)
                st.metric(
                    "Total Return",
                    f"{total_return:.1%}",
                    delta=f"${results.get('final_value', 0) - results.get('initial_value', 0):,.2f}"
                )
            
            # Trade statistics
            st.subheader("Trade Statistics")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Trades", results.get('total_trades', 0))
            
            with col2:
                test_period = results.get('test_period', {})
                st.metric("Test Period", f"{test_period.get('start', 'N/A')} to {test_period.get('end', 'N/A')}")
                
        except Exception as e:
            st.error(f"Error rendering backtest results: {e}")
    
    def _render_strategy_comparison(self):
        """Render strategy comparison"""
        try:
            st.info("Strategy comparison feature coming soon...")
            
        except Exception as e:
            st.error(f"Error rendering strategy comparison: {e}")
    
    def _render_futures_analysis(self, symbol: str):
        """Render futures-specific analysis"""
        try:
            st.info("Futures analysis charts coming soon...")
            
        except Exception as e:
            st.error(f"Error rendering futures analysis: {e}")
    
    def _create_portfolio_chart(self, portfolio_history: List[Dict[str, Any]]) -> go.Figure:
        """Create portfolio performance chart"""
        try:
            df = pd.DataFrame(portfolio_history)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00ff88', width=2)
            ))
            
            fig.update_layout(
                title="Portfolio Value Over Time",
                xaxis_title="Time",
                yaxis_title="Value ($)",
                height=400
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating portfolio chart: {e}")
            return go.Figure()
    
    def _render_performance_metrics(self, metrics: Dict[str, Any]):
        """Render performance metrics"""
        try:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Return", f"{metrics.get('total_return', 0):.1%}")
                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                st.metric("Max Drawdown", f"{metrics.get('max_drawdown', 0):.1%}")
            
            with col2:
                st.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")
                st.metric("Average Return", f"{metrics.get('avg_return', 0):.2%}")
                st.metric("Volatility", f"{metrics.get('volatility', 0):.2%}")
                
        except Exception as e:
            st.error(f"Error rendering performance metrics: {e}")
    
    def _render_risk_analysis(self):
        """Render risk analysis"""
        try:
            if 'trading_engine' in st.session_state:
                engine = st.session_state.trading_engine
                risk_metrics = engine.risk_manager.get_risk_metrics()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Portfolio Risk",
                        f"{risk_metrics.get('portfolio_risk_pct', 0):.1f}%"
                    )
                    st.metric(
                        "Max Position Size",
                        f"{risk_metrics.get('max_position_size_pct', 0):.1f}%"
                    )
                
                with col2:
                    st.metric(
                        "Stop Loss",
                        f"{risk_metrics.get('stop_loss_pct', 0):.1f}%"
                    )
                    st.metric(
                        "Take Profit",
                        f"{risk_metrics.get('take_profit_pct', 0):.1f}%"
                    )
            else:
                st.info("Risk analysis not available")
                
        except Exception as e:
            st.error(f"Error rendering risk analysis: {e}")
