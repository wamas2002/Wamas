import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
from frontend.dashboard import TradingDashboard
from trading.engine import TradingEngine
from ai.predictor import AIPredictor
from ai.lstm_predictor import LSTMPredictor
from ai.prophet_predictor import ProphetPredictor
from ai.reinforcement_advanced import AdvancedQLearningAgent
from trading.okx_connector import OKXConnector
from trading.risk_manager_advanced import AdvancedRiskManager
from trading.backtesting_engine import BacktestingEngine, WalkForwardAnalyzer
from utils.logger import TradingLogger
from config import Config
from database.services import DatabaseService
from database.models import DatabaseManager
import warnings
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="AI Crypto Trading System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'trading_engine' not in st.session_state:
        st.session_state.trading_engine = TradingEngine()
    
    if 'ai_predictor' not in st.session_state:
        st.session_state.ai_predictor = AIPredictor()
    
    if 'lstm_predictor' not in st.session_state:
        st.session_state.lstm_predictor = LSTMPredictor()
    
    if 'prophet_predictor' not in st.session_state:
        st.session_state.prophet_predictor = ProphetPredictor()
    
    if 'rl_agent' not in st.session_state:
        st.session_state.rl_agent = AdvancedQLearningAgent()
    
    if 'okx_connector' not in st.session_state:
        st.session_state.okx_connector = OKXConnector()
    
    if 'risk_manager' not in st.session_state:
        st.session_state.risk_manager = AdvancedRiskManager()
    
    if 'backtesting_engine' not in st.session_state:
        st.session_state.backtesting_engine = BacktestingEngine()
    
    if 'walk_forward_analyzer' not in st.session_state:
        st.session_state.walk_forward_analyzer = WalkForwardAnalyzer()
    
    if 'db_service' not in st.session_state:
        try:
            st.session_state.db_service = DatabaseService()
        except Exception as e:
            st.session_state.db_service = None
            st.session_state.db_error = str(e)
    
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = TradingDashboard()
    
    if 'logger' not in st.session_state:
        st.session_state.logger = TradingLogger()
    
    if 'trading_active' not in st.session_state:
        st.session_state.trading_active = False
    
    if 'market_data' not in st.session_state:
        st.session_state.market_data = {}
    
    if 'predictions' not in st.session_state:
        st.session_state.predictions = {}
    
    if 'portfolio_history' not in st.session_state:
        st.session_state.portfolio_history = []
    
    # API Configuration
    if 'okx_api_configured' not in st.session_state:
        st.session_state.okx_api_configured = False
    
    if 'use_real_trading' not in st.session_state:
        st.session_state.use_real_trading = False

def main():
    """Main application function"""
    initialize_session_state()
    
    # Main title
    st.title("🚀 AI-Powered Cryptocurrency Trading System")
    st.markdown("---")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🔧 Trading Configuration")
        
        # Symbol selection
        selected_symbol = st.selectbox(
            "Select Trading Pair",
            options=Config.SUPPORTED_SYMBOLS,
            index=0
        )
        
        # Timeframe selection
        selected_timeframe = st.selectbox(
            "Select Timeframe",
            options=["1m", "5m", "15m", "1h", "4h", "1d"],
            index=2
        )
        
        # Strategy selection
        strategy_type = st.selectbox(
            "Select Strategy",
            options=["Ensemble", "ML", "Reinforcement Learning"],
            index=0
        )
        
        # Risk parameters
        st.subheader("Risk Management")
        risk_per_trade = st.slider("Risk per Trade (%)", 1, 10, 2)
        max_positions = st.slider("Max Positions", 1, 5, 3)
        stop_loss_pct = st.slider("Stop Loss (%)", 1, 10, 5)
        take_profit_pct = st.slider("Take Profit (%)", 2, 20, 10)
        
        # Trading controls
        st.subheader("Trading Controls")
        if st.button("🚀 Start Trading", type="primary"):
            st.session_state.trading_active = True
            st.success("Trading started!")
        
        if st.button("⏹️ Stop Trading"):
            st.session_state.trading_active = False
            st.warning("Trading stopped!")
        
        # Paper trading toggle
        paper_trading = st.checkbox("Paper Trading Mode", value=True)
        
        # Auto refresh
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=True)
        
        # OKX API Configuration
        st.subheader("🔑 OKX API Configuration")
        with st.expander("API Settings", expanded=not st.session_state.okx_api_configured):
            okx_api_key = st.text_input("API Key", type="password", help="Your OKX API Key")
            okx_secret_key = st.text_input("Secret Key", type="password", help="Your OKX Secret Key") 
            okx_passphrase = st.text_input("Passphrase", type="password", help="Your OKX Passphrase")
            sandbox_mode = st.checkbox("Sandbox Mode", value=True, help="Use sandbox for testing")
            
            if st.button("Test Connection"):
                if okx_api_key and okx_secret_key and okx_passphrase:
                    # Update OKX connector with credentials
                    st.session_state.okx_connector = OKXConnector(
                        api_key=okx_api_key,
                        secret_key=okx_secret_key, 
                        passphrase=okx_passphrase,
                        sandbox=sandbox_mode
                    )
                    
                    # Test connection
                    test_result = st.session_state.okx_connector.test_connection()
                    if test_result.get('success'):
                        st.success("OKX API connected successfully!")
                        st.session_state.okx_api_configured = True
                        st.session_state.use_real_trading = not paper_trading
                    else:
                        st.error(f"Connection failed: {test_result.get('error', 'Unknown error')}")
                else:
                    st.warning("Please enter all API credentials")
        
        # Real trading toggle (only if API configured)
        if st.session_state.okx_api_configured:
            real_trading = st.checkbox("Enable Real Trading", value=False, 
                                     help="Switch from paper trading to real trading")
            if real_trading and not paper_trading:
                st.session_state.use_real_trading = True
                st.warning("⚠️ Real trading enabled! Use caution.")
        
        # Leverage settings for futures
        st.subheader("⚡ Futures Trading")
        leverage = st.slider("Leverage", 1, 100, 1, help="Trading leverage (1x-100x)")
        if leverage > 1:
            st.warning(f"Using {leverage}x leverage increases risk significantly")
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Live Trading", 
        "🧠 AI Insights", 
        "📈 Backtesting", 
        "⚡ Futures Mode",
        "📋 Performance",
        "🔬 Advanced ML",
        "⚖️ Risk Analysis"
    ])
    
    with tab1:
        st.session_state.dashboard.render_live_trading_tab(
            selected_symbol, selected_timeframe, strategy_type
        )
    
    with tab2:
        st.session_state.dashboard.render_ai_insights_tab(
            selected_symbol, selected_timeframe
        )
    
    with tab3:
        st.session_state.dashboard.render_backtesting_tab(
            selected_symbol, selected_timeframe, strategy_type
        )
    
    with tab4:
        st.session_state.dashboard.render_futures_tab(
            selected_symbol, selected_timeframe
        )
    
    with tab5:
        st.session_state.dashboard.render_performance_tab()
    
    with tab6:
        # Advanced ML Models Tab
        st.header("🔬 Advanced Machine Learning Models")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("LSTM Neural Network")
            
            # Get market data for training
            try:
                market_data = st.session_state.trading_engine.get_market_data(selected_symbol, selected_timeframe)
                
                if len(market_data) > 100:
                    # Train LSTM model
                    if st.button("Train LSTM Model"):
                        with st.spinner("Training LSTM model..."):
                            lstm_result = st.session_state.lstm_predictor.train(market_data)
                            
                        if lstm_result.get('success'):
                            st.success(f"LSTM trained successfully!")
                            st.write(f"Training samples: {lstm_result.get('training_samples', 0)}")
                            st.write(f"Validation MSE: {lstm_result.get('val_mse', 0):.6f}")
                        else:
                            st.error(f"Training failed: {lstm_result.get('error', 'Unknown error')}")
                    
                    # Generate LSTM predictions
                    if st.button("Generate LSTM Prediction"):
                        with st.spinner("Generating prediction..."):
                            prediction = st.session_state.lstm_predictor.predict(market_data)
                            
                        if prediction.get('success'):
                            current_price = prediction.get('current_price', 0)
                            predicted_price = prediction.get('prediction', 0)
                            change_pct = prediction.get('predicted_change', 0) * 100
                            confidence = prediction.get('confidence', 0) * 100
                            
                            st.metric("Current Price", f"${current_price:.4f}")
                            st.metric("Predicted Price", f"${predicted_price:.4f}", 
                                    f"{change_pct:+.2f}%")
                            st.metric("Confidence", f"{confidence:.1f}%")
                        else:
                            st.error(f"Prediction failed: {prediction.get('error', 'Unknown error')}")
                else:
                    st.warning("Insufficient data for LSTM training (need >100 samples)")
                    
            except Exception as e:
                st.error(f"Error accessing market data: {e}")
        
        with col2:
            st.subheader("Prophet Time Series Model")
            
            try:
                if len(market_data) > 100:
                    # Train Prophet model
                    if st.button("Train Prophet Model"):
                        with st.spinner("Training Prophet model..."):
                            prophet_result = st.session_state.prophet_predictor.train(market_data)
                            
                        if prophet_result.get('success'):
                            st.success("Prophet trained successfully!")
                            st.write(f"Training samples: {prophet_result.get('training_samples', 0)}")
                            st.write(f"Trend changepoints: {prophet_result.get('changepoints', 0)}")
                            st.write(f"Seasonal components: {prophet_result.get('seasonal_components', 0)}")
                        else:
                            st.error(f"Training failed: {prophet_result.get('error', 'Unknown error')}")
                    
                    # Generate Prophet predictions
                    if st.button("Generate Prophet Forecast"):
                        forecast_periods = st.slider("Forecast Periods (hours)", 1, 48, 24)
                        
                        with st.spinner("Generating forecast..."):
                            forecast = st.session_state.prophet_predictor.predict(market_data, periods=forecast_periods)
                            
                        if forecast.get('success'):
                            next_price = forecast.get('next_price', 0)
                            price_change = forecast.get('price_change', 0) * 100
                            confidence = forecast.get('confidence', 0) * 100
                            
                            st.metric("Next Price", f"${next_price:.4f}", f"{price_change:+.2f}%")
                            st.metric("Forecast Confidence", f"{confidence:.1f}%")
                            
                            # Show forecast chart
                            predictions = forecast.get('predictions', [])
                            if predictions:
                                forecast_df = pd.DataFrame({
                                    'Hour': range(1, len(predictions) + 1),
                                    'Predicted_Price': predictions
                                })
                                st.line_chart(forecast_df.set_index('Hour'))
                        else:
                            st.error(f"Forecast failed: {forecast.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error with Prophet model: {e}")
        
        st.subheader("Reinforcement Learning Agent")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("**Q-Learning Performance**")
            rl_stats = st.session_state.rl_agent.get_statistics()
            
            if rl_stats:
                st.metric("States Explored", rl_stats.get('total_states_explored', 0))
                st.metric("Total Experiences", rl_stats.get('total_experiences', 0))
                st.metric("Current Epsilon", f"{rl_stats.get('current_epsilon', 0):.3f}")
                st.metric("Recent Avg Reward", f"{rl_stats.get('recent_avg_reward', 0):.4f}")
        
        with col4:
            st.write("**Action Distribution**")
            action_dist = rl_stats.get('action_distribution', {})
            
            if action_dist:
                actions = list(action_dist.keys())
                percentages = [action_dist[action]['percentage'] for action in actions]
                
                fig = go.Figure(data=[go.Pie(labels=actions, values=percentages)])
                fig.update_layout(title="RL Agent Action Distribution", height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Walk-Forward Analysis
        st.subheader("Walk-Forward Analysis")
        
        if st.button("Run Walk-Forward Analysis"):
            with st.spinner("Running comprehensive walk-forward analysis..."):
                try:
                    # Define parameter ranges for optimization
                    param_ranges = {
                        'rsi_period': [10, 14, 18],
                        'macd_fast': [8, 12, 16],
                        'bb_period': [15, 20, 25]
                    }
                    
                    # Create a dummy strategy function for testing
                    def test_strategy(data, **params):
                        return {
                            'total_return': np.random.normal(0.05, 0.15),
                            'sharpe_ratio': np.random.normal(0.8, 0.3),
                            'max_drawdown': np.random.uniform(0.05, 0.25),
                            'total_trades': np.random.randint(10, 50)
                        }
                    
                    wf_result = st.session_state.walk_forward_analyzer.run_walk_forward(
                        market_data, test_strategy, param_ranges
                    )
                    
                    if 'error' not in wf_result:
                        st.success(f"Walk-forward analysis completed!")
                        st.write(f"Total periods analyzed: {wf_result.get('total_periods', 0)}")
                        
                        # Show aggregate performance
                        agg_perf = wf_result.get('aggregate_performance', {})
                        if agg_perf:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Avg Return", f"{agg_perf.get('avg_return', 0):.2%}")
                            with col2:
                                st.metric("Avg Sharpe", f"{agg_perf.get('avg_sharpe', 0):.2f}")
                            with col3:
                                st.metric("Win Rate", f"{agg_perf.get('win_rate', 0):.1%}")
                    else:
                        st.error(f"Analysis failed: {wf_result['error']}")
                        
                except Exception as e:
                    st.error(f"Walk-forward analysis error: {e}")
    
    with tab7:
        # Risk Analysis Tab
        st.header("⚖️ Advanced Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Portfolio Risk Metrics")
            
            # Get risk summary
            risk_summary = st.session_state.risk_manager.get_risk_summary()
            
            if 'error' not in risk_summary:
                st.metric("Portfolio Value", f"${risk_summary.get('portfolio_value', 0):,.2f}")
                st.metric("Cash Available", f"${risk_summary.get('cash_available', 0):,.2f}")
                
                risk_metrics = risk_summary.get('risk_metrics', {})
                if risk_metrics:
                    st.metric("Max Drawdown", f"{risk_metrics.get('max_drawdown', 0):.2%}")
                    st.metric("Volatility", f"{risk_metrics.get('volatility', 0):.2%}")
                    st.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.2f}")
                    st.metric("VaR (95%)", f"{risk_metrics.get('var_95', 0):.4f}")
                
                # Risk status
                risk_status = risk_summary.get('risk_status', 'OK')
                if risk_status == 'VIOLATION':
                    st.error("Risk limit violations detected!")
                    violations = risk_summary.get('violations', [])
                    for violation in violations:
                        st.error(f"• {violation}")
                elif risk_status == 'WARNING':
                    st.warning("Risk warnings:")
                    warnings = risk_summary.get('warnings', [])
                    for warning in warnings:
                        st.warning(f"• {warning}")
                else:
                    st.success("All risk limits within acceptable ranges")
            else:
                st.error(f"Error getting risk data: {risk_summary['error']}")
        
        with col2:
            st.subheader("Position Sizing Calculator")
            
            # Position sizing inputs
            signal_strength = st.slider("Signal Strength", 0.0, 1.0, 0.7, 0.1)
            current_price = st.number_input("Current Price", value=50000.0, min_value=0.01)
            volatility = st.slider("Asset Volatility", 0.01, 0.10, 0.03, 0.01)
            
            if st.button("Calculate Optimal Position Size"):
                position_calc = st.session_state.risk_manager.calculate_position_size(
                    selected_symbol, signal_strength, current_price, volatility
                )
                
                if 'error' not in position_calc:
                    st.metric("Recommended Size", f"{position_calc.get('recommended_size', 0):.6f}")
                    st.metric("Position Value", f"${position_calc.get('position_value', 0):,.2f}")
                    st.metric("Risk Score", f"{position_calc.get('risk_score', 0):.3f}")
                    st.metric("Max Loss", f"${position_calc.get('max_loss', 0):,.2f}")
                    
                    # Show sizing breakdown
                    st.write("**Sizing Methods:**")
                    st.write(f"Kelly Criterion: {position_calc.get('kelly_size', 0):.4f}")
                    st.write(f"Volatility Adjusted: {position_calc.get('vol_adjusted_size', 0):.4f}")
                    st.write(f"Risk Parity: {position_calc.get('risk_parity_size', 0):.4f}")
                else:
                    st.error(f"Calculation failed: {position_calc['error']}")
        
        st.subheader("Advanced Backtesting Results")
        
        if st.button("Run Comprehensive Backtest"):
            with st.spinner("Running advanced backtesting analysis..."):
                try:
                    # Run backtest using the backtesting engine
                    from strategies.ensemble_strategy import EnsembleStrategy
                    strategy = EnsembleStrategy()
                    
                    backtest_result = st.session_state.backtesting_engine.run_backtest(
                        market_data, strategy
                    )
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Return", f"{backtest_result.total_return:.2%}")
                        st.metric("Annual Return", f"{backtest_result.annual_return:.2%}")
                    
                    with col2:
                        st.metric("Sharpe Ratio", f"{backtest_result.sharpe_ratio:.2f}")
                        st.metric("Sortino Ratio", f"{backtest_result.sortino_ratio:.2f}")
                    
                    with col3:
                        st.metric("Max Drawdown", f"{backtest_result.max_drawdown:.2%}")
                        st.metric("Calmar Ratio", f"{backtest_result.calmar_ratio:.2f}")
                    
                    with col4:
                        st.metric("Win Rate", f"{backtest_result.win_rate:.1%}")
                        st.metric("Total Trades", f"{backtest_result.total_trades}")
                    
                    # Get detailed analysis
                    detailed_analysis = st.session_state.backtesting_engine.get_detailed_analysis()
                    
                    if 'error' not in detailed_analysis:
                        st.subheader("Risk Analysis")
                        
                        risk_analysis = detailed_analysis.get('risk_analysis', {})
                        if risk_analysis:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Value at Risk**")
                                st.write(f"VaR 95%: {risk_analysis.get('var_95_daily', 0):.2f}%")
                                st.write(f"VaR 99%: {risk_analysis.get('var_99_daily', 0):.2f}%")
                                st.write(f"Expected Shortfall 95%: {risk_analysis.get('expected_shortfall_95', 0):.2f}%")
                            
                            with col2:
                                st.write("**Distribution Analysis**")
                                st.write(f"Skewness: {risk_analysis.get('skewness', 0):.3f}")
                                st.write(f"Kurtosis: {risk_analysis.get('kurtosis', 0):.3f}")
                                st.write(f"Tail Ratio: {risk_analysis.get('tail_ratio', 0):.3f}")
                
                except Exception as e:
                    st.error(f"Backtesting error: {e}")
        
        # Monte Carlo Simulation
        st.subheader("Monte Carlo Simulation")
        
        if st.button("Run Monte Carlo Analysis"):
            num_simulations = st.slider("Number of Simulations", 100, 2000, 1000, 100)
            
            with st.spinner(f"Running {num_simulations} Monte Carlo simulations..."):
                try:
                    from strategies.ensemble_strategy import EnsembleStrategy
                    strategy = EnsembleStrategy()
                    
                    mc_result = st.session_state.backtesting_engine.run_monte_carlo_simulation(
                        strategy, market_data, num_simulations
                    )
                    
                    if 'error' not in mc_result:
                        st.success(f"Monte Carlo analysis completed!")
                        
                        # Return statistics
                        return_stats = mc_result.get('return_statistics', {})
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Mean Return", f"{return_stats.get('mean', 0):.2%}")
                            st.metric("Return Volatility", f"{return_stats.get('std', 0):.2%}")
                        
                        with col2:
                            st.metric("5th Percentile", f"{return_stats.get('percentile_5', 0):.2%}")
                            st.metric("95th Percentile", f"{return_stats.get('percentile_95', 0):.2%}")
                        
                        with col3:
                            st.metric("Positive Returns", f"{return_stats.get('positive_returns', 0):.1%}")
                            
                        # Drawdown statistics
                        dd_stats = mc_result.get('drawdown_statistics', {})
                        if dd_stats:
                            st.write("**Drawdown Analysis**")
                            st.write(f"Mean Max Drawdown: {dd_stats.get('mean', 0):.2%}")
                            st.write(f"95% Confidence Max Drawdown: {dd_stats.get('percentile_95', 0):.2%}")
                            st.write(f"99% Confidence Max Drawdown: {dd_stats.get('percentile_99', 0):.2%}")
                    else:
                        st.error(f"Monte Carlo failed: {mc_result['error']}")
                        
                except Exception as e:
                    st.error(f"Monte Carlo simulation error: {e}")
    
    # Auto refresh functionality
    if auto_refresh and st.session_state.trading_active:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
