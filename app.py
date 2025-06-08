import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import plotly.graph_objects as go
import plotly.express as px
from frontend.dashboard import TradingDashboard
from frontend.tradingview_charts import TradingViewCharts
from trading.engine import TradingEngine
from ai.predictor import AIPredictor
from ai.lstm_predictor import LSTMPredictor
from ai.prophet_predictor import ProphetPredictor
from ai.reinforcement_advanced import AdvancedQLearningAgent
from ai.market_regime_detector import MarketRegimeDetector
from ai.portfolio_optimizer import PortfolioOptimizer
from trading.okx_connector import OKXConnector
from trading.okx_data_service import OKXDataService
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
    
    if 'regime_detector' not in st.session_state:
        st.session_state.regime_detector = MarketRegimeDetector()
    
    if 'portfolio_optimizer' not in st.session_state:
        st.session_state.portfolio_optimizer = PortfolioOptimizer()
    
    if 'okx_connector' not in st.session_state:
        st.session_state.okx_connector = OKXConnector()
    
    if 'okx_data_service' not in st.session_state:
        st.session_state.okx_data_service = OKXDataService()
    
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
    
    if 'tradingview_charts' not in st.session_state:
        st.session_state.tradingview_charts = TradingViewCharts()
    
    if 'advanced_ml_pipeline' not in st.session_state:
        from ai.advanced_ml_pipeline import AdvancedMLPipeline
        st.session_state.advanced_ml_pipeline = AdvancedMLPipeline()
    
    if 'transformer_ensemble' not in st.session_state:
        from ai.transformer_ensemble import TransformerEnsemble
        st.session_state.transformer_ensemble = TransformerEnsemble()
    
    if 'freqai_pipeline' not in st.session_state:
        from ai.freqai_pipeline import FreqAILevelPipeline
        st.session_state.freqai_pipeline = FreqAILevelPipeline()
    
    if 'advanced_ml_interface' not in st.session_state:
        from frontend.advanced_ml_interface import AdvancedMLInterface
        st.session_state.advanced_ml_interface = AdvancedMLInterface()
    
    if 'logger' not in st.session_state:
        st.session_state.logger = TradingLogger()
    
    if 'trading_active' not in st.session_state:
        st.session_state.trading_active = False
    
    # Clear any cached market data to force live API calls
    st.session_state.market_data = {}
    st.session_state.predictions = {}
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "📊 Live Trading", 
        "🧠 AI Insights", 
        "📈 Backtesting", 
        "⚡ Futures Mode",
        "📋 Performance",
        "🔬 Advanced ML",
        "⚖️ Risk Analysis",
        "🗄️ Database",
        "📈 Live Charts"
    ])
    
    with tab1:
        # Live Trading with OKX Futures Integration
        st.header("📊 Live Futures Trading")
        
        if not st.session_state.okx_api_configured:
            st.warning("⚠️ Please configure OKX API credentials in the sidebar to enable live trading")
            st.info("📝 You need API Key, Secret Key, and Passphrase from OKX")
        else:
            # Trading interface
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🎯 Place Order")
                
                # Order form
                order_col1, order_col2 = st.columns(2)
                
                with order_col1:
                    order_side = st.selectbox("Order Side", ["buy", "sell"])
                    order_type = st.selectbox("Order Type", ["market", "limit"])
                    order_size = st.number_input("Size (USDT)", min_value=1.0, value=100.0, step=1.0)
                
                with order_col2:
                    order_leverage = st.slider("Leverage", 1, 100, leverage, help="Trading leverage for futures")
                    
                    if order_type == "limit":
                        limit_price = st.number_input("Limit Price", min_value=0.01, value=50000.0, step=0.01)
                    else:
                        limit_price = None
                    
                    reduce_only = st.checkbox("Reduce Only", help="Close existing position only")
                
                # Risk management controls
                st.subheader("🛡️ Risk Management")
                risk_col1, risk_col2 = st.columns(2)
                
                with risk_col1:
                    use_stop_loss = st.checkbox("Use Stop Loss")
                    if use_stop_loss:
                        stop_loss_price = st.number_input("Stop Loss Price", min_value=0.01, value=45000.0, step=0.01)
                    else:
                        stop_loss_price = None
                
                with risk_col2:
                    use_take_profit = st.checkbox("Use Take Profit")
                    if use_take_profit:
                        take_profit_price = st.number_input("Take Profit Price", min_value=0.01, value=55000.0, step=0.01)
                    else:
                        take_profit_price = None
                
                # Place order button
                if st.button("🚀 Place Order", type="primary"):
                    if st.session_state.use_real_trading:
                        # Real trading
                        order_result = st.session_state.okx_connector.place_futures_order(
                            symbol=selected_symbol,
                            side=order_side,
                            size=order_size,
                            order_type=order_type,
                            price=limit_price,
                            leverage=order_leverage,
                            reduce_only=reduce_only,
                            stop_loss=stop_loss_price,
                            take_profit=take_profit_price
                        )
                        
                        if order_result.get('success'):
                            st.success(f"✅ Order placed successfully! Order ID: {order_result.get('order_id')}")
                            if order_result.get('stop_loss_order_id'):
                                st.info(f"🛡️ Stop loss order: {order_result.get('stop_loss_order_id')}")
                            if order_result.get('take_profit_order_id'):
                                st.info(f"🎯 Take profit order: {order_result.get('take_profit_order_id')}")
                        else:
                            st.error(f"❌ Order failed: {order_result.get('error')}")
                    else:
                        st.info("📝 Paper trading mode - order simulation only")
            
            with col2:
                st.subheader("💰 Account Info")
                
                # Get account balance
                balance_result = st.session_state.okx_connector.get_account_balance()
                if balance_result.get('success'):
                    balances = balance_result.get('balances', {})
                    for currency, balance_info in balances.items():
                        if currency in ['USDT', 'USD', 'BTC', 'ETH']:
                            st.metric(
                                f"{currency} Balance",
                                f"{balance_info['available']:.2f}",
                                f"Total: {balance_info['total']:.2f}"
                            )
                else:
                    st.error("Failed to fetch balance")
                
                # Funding rate info
                funding_result = st.session_state.okx_connector.get_funding_rate(selected_symbol)
                if funding_result.get('success'):
                    funding_rate = funding_result.get('funding_rate', 0) * 100
                    st.metric("Funding Rate", f"{funding_rate:.4f}%")
                
                # Maximum tradable size
                max_size_result = st.session_state.okx_connector.get_maximum_tradable_size(
                    selected_symbol, "buy", order_leverage
                )
                if max_size_result.get('success'):
                    st.metric("Max Buy Size", f"{max_size_result.get('max_buy_size', 0):.2f}")
            
            # Current positions
            st.subheader("📋 Open Positions")
            positions_result = st.session_state.okx_connector.get_positions()
            
            if positions_result.get('success'):
                positions = positions_result.get('positions', [])
                
                if positions:
                    positions_df = pd.DataFrame(positions)
                    
                    # Display positions table
                    st.dataframe(
                        positions_df[['symbol', 'side', 'size', 'entry_price', 'mark_price', 
                                    'unrealized_pnl', 'leverage', 'margin']],
                        use_container_width=True
                    )
                    
                    # Quick close buttons
                    st.subheader("⚡ Quick Actions")
                    for idx, position in enumerate(positions):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.text(f"{position['symbol']} - {position['side']} - Size: {position['size']}")
                        
                        with col2:
                            if st.button(f"Close 50%", key=f"close_50_{idx}"):
                                close_result = st.session_state.okx_connector.close_position(
                                    position['symbol'], position['size'] * 0.5
                                )
                                if close_result.get('success'):
                                    st.success("50% position closed")
                                else:
                                    st.error(f"Failed: {close_result.get('error')}")
                        
                        with col3:
                            if st.button(f"Close All", key=f"close_all_{idx}"):
                                close_result = st.session_state.okx_connector.close_position(position['symbol'])
                                if close_result.get('success'):
                                    st.success("Position closed")
                                else:
                                    st.error(f"Failed: {close_result.get('error')}")
                else:
                    st.info("No open positions")
            else:
                st.error("Failed to fetch positions")
            
            # Recent orders
            st.subheader("📊 Recent Orders")
            orders_result = st.session_state.okx_connector.get_order_history(selected_symbol, 10)
            
            if orders_result.get('success'):
                orders = orders_result.get('orders', [])
                
                if orders:
                    orders_df = pd.DataFrame(orders)
                    orders_df['timestamp'] = pd.to_datetime(orders_df['timestamp'], unit='ms')
                    
                    st.dataframe(
                        orders_df[['timestamp', 'symbol', 'side', 'order_type', 'size', 
                                 'filled_size', 'avg_price', 'status']],
                        use_container_width=True
                    )
                else:
                    st.info("No recent orders")
            else:
                st.error("Failed to fetch order history")
            
            # Trading instruments info
            st.subheader("📋 Available Instruments")
            if st.button("🔄 Refresh Instruments"):
                instruments_result = st.session_state.okx_connector.get_trading_instruments("SWAP")
                
                if instruments_result.get('success'):
                    instruments = instruments_result.get('instruments', [])
                    instruments_df = pd.DataFrame(instruments)
                    
                    # Filter for popular pairs
                    popular_pairs = instruments_df[
                        instruments_df['symbol'].str.contains('BTC|ETH|ADA|BNB|DOT|LINK|LTC|XRP')
                    ].head(20)
                    
                    st.dataframe(
                        popular_pairs[['symbol', 'max_leverage', 'min_size', 'tick_size', 'status']],
                        use_container_width=True
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
        
        # Market Regime Detection Section
        st.subheader("📊 Market Regime Detection")
        
        try:
            market_data = st.session_state.okx_data_service.get_historical_data(selected_symbol, selected_timeframe, 500)
            
            if not market_data.empty and len(market_data) > 100:
                # Detect current market regime
                if st.button("Analyze Market Regime", key="analyze_regime"):
                    with st.spinner("Analyzing market regime..."):
                        regime_result = st.session_state.regime_detector.detect_regime(market_data)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Regime", regime_result['regime'].title())
                            st.metric("Confidence", f"{regime_result['confidence']:.1%}")
                        
                        with col2:
                            st.write("**Regime Probabilities**")
                            for regime, prob in regime_result['probabilities'].items():
                                st.write(f"{regime.title()}: {prob:.1%}")
                        
                        with col3:
                            st.write("**Description**")
                            st.write(regime_result['description'])
                
                # Portfolio Optimization Section
                st.markdown("---")
                st.subheader("💼 Portfolio Optimization")
                
                # Multi-asset selection for portfolio
                available_symbols = Config.SUPPORTED_SYMBOLS
                selected_assets = st.multiselect(
                    "Select Assets for Portfolio",
                    available_symbols,
                    default=available_symbols[:4],
                    key="portfolio_assets"
                )
                
                if len(selected_assets) >= 2:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        optimization_method = st.selectbox(
                            "Optimization Method",
                            ["max_sharpe", "min_variance", "risk_parity", "max_diversification", "hierarchical_risk_parity"],
                            key="opt_method"
                        )
                        
                        max_weight = st.slider("Maximum Asset Weight", 0.1, 1.0, 0.4, 0.05)
                        
                    with col2:
                        lookback_days = st.slider("Lookback Period (days)", 30, 252, 90)
                        
                        if st.button("Optimize Portfolio", key="optimize_portfolio"):
                            with st.spinner("Optimizing portfolio..."):
                                # Collect price data for selected assets
                                price_data = {}
                                for symbol in selected_assets:
                                    asset_data = st.session_state.okx_data_service.get_historical_data(symbol, selected_timeframe, lookback_days)
                                    if not asset_data.empty:
                                        price_data[symbol] = asset_data.tail(lookback_days)
                                
                                if len(price_data) >= 2:
                                    # Calculate returns
                                    returns_df = st.session_state.portfolio_optimizer.calculate_returns(price_data)
                                    
                                    if not returns_df.empty:
                                        # Optimize portfolio
                                        constraints = {
                                            'max_weight': max_weight,
                                            'min_weight': 0.0,
                                            'long_only': True
                                        }
                                        
                                        optimization_result = st.session_state.portfolio_optimizer.optimize_portfolio(
                                            returns_df, optimization_method, constraints
                                        )
                                        
                                        if 'error' not in optimization_result:
                                            # Display optimization results
                                            st.success("Portfolio optimization completed!")
                                            
                                            col1, col2 = st.columns(2)
                                            
                                            with col1:
                                                st.write("**Optimal Weights**")
                                                weights_df = pd.DataFrame(
                                                    list(optimization_result['weights'].items()),
                                                    columns=['Asset', 'Weight']
                                                )
                                                weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.1%}")
                                                st.dataframe(weights_df, use_container_width=True)
                                                
                                                # Portfolio metrics
                                                metrics = optimization_result['metrics']
                                                st.metric("Expected Return", f"{metrics.get('expected_return', 0):.1%}")
                                                st.metric("Volatility", f"{metrics.get('volatility', 0):.1%}")
                                                st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                                            
                                            with col2:
                                                # Portfolio allocation pie chart
                                                import plotly.express as px
                                                
                                                weights_data = optimization_result['weights']
                                                fig = px.pie(
                                                    values=list(weights_data.values()),
                                                    names=list(weights_data.keys()),
                                                    title="Portfolio Allocation"
                                                )
                                                st.plotly_chart(fig, use_container_width=True)
                                        else:
                                            st.error(f"Optimization failed: {optimization_result['error']}")
                                    else:
                                        st.warning("Unable to calculate returns for selected assets")
                                else:
                                    st.warning("Insufficient price data for optimization")
                
                # Advanced ML Models Section
                st.markdown("---")
                st.subheader("🧠 AI Model Training & Predictions")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**LSTM Neural Network**")
                    
                    if len(market_data) > 100:
                        if st.button("Train LSTM Model", key="train_lstm"):
                            with st.spinner("Training LSTM model..."):
                                lstm_result = st.session_state.lstm_predictor.train(market_data)
                                
                                if lstm_result['success']:
                                    st.success("LSTM model trained successfully!")
                                    st.metric("Training Accuracy", f"{lstm_result.get('training_accuracy', 0):.2%}")
                                    st.metric("Validation Accuracy", f"{lstm_result.get('validation_accuracy', 0):.2%}")
                                    
                                    # Store model state in database
                                    try:
                                        model_id = st.session_state.db_service.store_ai_model_state(
                                            model_type="LSTM",
                                            symbol=selected_symbol,
                                            model_state=lstm_result,
                                            performance_metrics=lstm_result.get('metrics', {})
                                        )
                                        st.info(f"Model saved to database (ID: {model_id})")
                                    except Exception as e:
                                        st.warning(f"Could not save model to database: {e}")
                                else:
                                    st.error("LSTM training failed")
                        
                        if st.button("Generate LSTM Prediction", key="predict_lstm"):
                            with st.spinner("Generating LSTM prediction..."):
                                lstm_prediction = st.session_state.lstm_predictor.predict(market_data)
                                
                                if lstm_prediction.get('success'):
                                    pred_price = lstm_prediction.get('predicted_price', 0)
                                    confidence = lstm_prediction.get('confidence', 0)
                                    current_price = market_data['close'].iloc[-1]
                                    
                                    st.metric("Predicted Price", f"${pred_price:.2f}")
                                    st.metric("Confidence", f"{confidence:.1%}")
                                    st.metric("Price Change", f"{((pred_price - current_price) / current_price):.2%}")
                                    
                                    # Store prediction in database
                                    try:
                                        active_model = st.session_state.db_service.get_active_ai_model("LSTM", selected_symbol)
                                        if active_model:
                                            pred_id = st.session_state.db_service.store_prediction(
                                                model_id=active_model['id'],
                                                symbol=selected_symbol,
                                                predicted_price=pred_price,
                                                prediction_horizon=24,
                                                confidence=confidence
                                            )
                                            st.info(f"Prediction saved (ID: {pred_id})")
                                    except Exception as e:
                                        st.warning(f"Could not save prediction: {e}")
                                else:
                                    st.error("LSTM prediction failed")
                    else:
                        st.warning("Need at least 100 data points for LSTM training")
                
                with col2:
                    st.write("**Prophet Time Series Model**")
                    
                    if len(market_data) > 60:
                        if st.button("Train Prophet Model", key="train_prophet"):
                            with st.spinner("Training Prophet model..."):
                                prophet_result = st.session_state.prophet_predictor.train(market_data)
                                
                                if prophet_result.get('success'):
                                    st.success("Prophet model trained successfully!")
                                    st.metric("Training Samples", prophet_result.get('training_samples', 0))
                                    st.metric("Model Score", f"{prophet_result.get('performance_score', 0):.2%}")
                                    
                                    # Store model state in database
                                    try:
                                        model_id = st.session_state.db_service.store_ai_model_state(
                                            model_type="PROPHET",
                                            symbol=selected_symbol,
                                            model_state=prophet_result,
                                            performance_metrics=prophet_result.get('metrics', {})
                                        )
                                        st.info(f"Model saved to database (ID: {model_id})")
                                    except Exception as e:
                                        st.warning(f"Could not save model to database: {e}")
                                else:
                                    st.error("Prophet training failed")
                        
                        if st.button("Generate Prophet Forecast", key="predict_prophet"):
                            with st.spinner("Generating Prophet forecast..."):
                                prophet_prediction = st.session_state.prophet_predictor.predict(market_data, periods=24)
                                
                                if prophet_prediction.get('success'):
                                    st.success("Prophet forecast generated!")
                                    
                                    forecast_data = prophet_prediction.get('predictions', [])
                                    if forecast_data:
                                        next_price = forecast_data[0] if isinstance(forecast_data, list) else forecast_data
                                        confidence = prophet_prediction.get('confidence', 0)
                                        current_price = market_data['close'].iloc[-1]
                                        
                                        st.metric("Forecasted Price", f"${next_price:.2f}")
                                        st.metric("Confidence", f"{confidence:.1%}")
                                        st.metric("Price Change", f"{((next_price - current_price) / current_price):.2%}")
                                        
                                        # Store prediction in database
                                        try:
                                            active_model = st.session_state.db_service.get_active_ai_model("PROPHET", selected_symbol)
                                            if active_model:
                                                pred_id = st.session_state.db_service.store_prediction(
                                                    model_id=active_model['id'],
                                                    symbol=selected_symbol,
                                                    predicted_price=next_price,
                                                    prediction_horizon=24,
                                                    confidence=confidence
                                                )
                                                st.info(f"Forecast saved (ID: {pred_id})")
                                        except Exception as e:
                                            st.warning(f"Could not save forecast: {e}")
                                    else:
                                        st.warning("No forecast data generated")
                                else:
                                    st.error("Prophet forecast failed")
                    else:
                        st.warning("Need at least 60 data points for Prophet training")
            else:
                st.warning("Insufficient market data for AI analysis")
        
        except Exception as e:
            st.error(f"Error in Advanced ML tab: {e}")
    
    with tab7:
        st.header("Portfolio Analytics")
        
        if st.session_state.market_data is not None and len(st.session_state.market_data) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Portfolio Optimization")
                
                optimization_method = st.selectbox(
                    "Optimization Method",
                    ["max_sharpe", "min_variance", "risk_parity", "equal_weight"],
                    help="Choose portfolio optimization strategy"
                )
                
                if st.button("Optimize Portfolio", key="optimize_portfolio"):
                    with st.spinner("Optimizing portfolio..."):
                        try:
                            # Create sample multi-asset data for demonstration
                            symbols = ["BTC", "ETH", "ADA", "DOT"]
                            price_data = {}
                            
                            base_data = st.session_state.market_data.copy()
                            for i, symbol in enumerate(symbols):
                                variation = base_data.copy()
                                # Create realistic price variations
                                noise = np.random.normal(1, 0.1, len(base_data))
                                variation['close'] = variation['close'] * (0.5 + i * 0.3) * noise
                                price_data[symbol] = variation
                            
                            result = st.session_state.portfolio_optimizer.optimize_portfolio(
                                price_data, optimization_method=optimization_method
                            )
                            
                            if result.get('success'):
                                weights = result.get('weights', {})
                                metrics = result.get('metrics', {})
                                
                                st.success("Portfolio optimized successfully!")
                                
                                # Display optimal weights
                                st.write("**Optimal Allocation:**")
                                for symbol, weight in weights.items():
                                    st.metric(symbol, f"{weight:.1%}")
                                
                                # Portfolio metrics
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Expected Return", f"{metrics.get('expected_return', 0):.2%}")
                                with col_b:
                                    st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
                                
                            else:
                                st.error(f"Optimization failed: {result.get('error', 'Unknown error')}")
                                
                        except Exception as e:
                            st.error(f"Portfolio optimization error: {e}")
            
            with col2:
                st.subheader("Risk Analytics")
                
                # Calculate basic portfolio metrics
                returns = st.session_state.market_data['close'].pct_change().dropna()
                
                if len(returns) > 0:
                    volatility = returns.std() * np.sqrt(24 * 365)  # Annualized volatility
                    sharpe = (returns.mean() * 24 * 365) / volatility if volatility > 0 else 0
                    max_drawdown = (returns.cumsum().cummax() - returns.cumsum()).max()
                    
                    st.metric("Annualized Volatility", f"{volatility:.2%}")
                    st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    st.metric("Max Drawdown", f"{max_drawdown:.2%}")
                    
                    # Risk distribution chart
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=returns, nbinsx=30, name="Return Distribution"))
                    fig.update_layout(
                        title="Return Distribution",
                        xaxis_title="Returns",
                        yaxis_title="Frequency",
                        height=300
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Insufficient data for risk analytics")
        else:
            st.warning("Load market data to access portfolio analytics")
        
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
                    
                    # Use actual trading strategy with real data
                    def test_strategy(data, **params):
                        if data.empty:
                            return {'error': 'No data available'}
                        
                        # Simple momentum strategy using real market data
                        returns = data['close'].pct_change()
                        momentum_signal = returns.rolling(window=params.get('momentum_window', 10)).mean()
                        
                        # Calculate basic performance metrics
                        strategy_returns = momentum_signal.fillna(0) * returns.shift(-1)
                        total_return = (1 + strategy_returns).prod() - 1
                        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
                        max_drawdown = (strategy_returns.cumsum() - strategy_returns.cumsum().expanding().max()).min()
                        
                        return {
                            'total_return': float(total_return),
                            'sharpe_ratio': float(sharpe_ratio),
                            'max_drawdown': abs(float(max_drawdown)),
                            'total_trades': len(strategy_returns[strategy_returns != 0])
                        }
                    
                    wf_result = st.session_state.walk_forward_analyzer.run_walk_forward(
                        st.session_state.market_data, test_strategy, param_ranges
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
    
    with tab8:
        st.header("Advanced Risk Analysis")
        
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
    
    with tab8:
        # Database Dashboard Tab
        st.header("🗄️ Database Analytics & Management")
        
        # Check database connection
        if st.session_state.db_service is None:
            st.error("Database connection failed. Please check DATABASE_URL environment variable.")
            if hasattr(st.session_state, 'db_error'):
                st.error(f"Error: {st.session_state.db_error}")
            return
        
        # Database overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                trading_stats = st.session_state.db_service.get_trading_statistics(30)
                st.metric("Total Trades (30d)", trading_stats.get('total_trades', 0))
            except Exception as e:
                st.metric("Total Trades (30d)", "Error")
        
        with col2:
            try:
                recent_signals = st.session_state.db_service.get_recent_signals(limit=10)
                st.metric("Recent Signals", len(recent_signals))
            except Exception as e:
                st.metric("Recent Signals", "Error")
        
        with col3:
            try:
                portfolio_data = st.session_state.db_service.get_portfolio_history(7)
                st.metric("Portfolio Records", len(portfolio_data))
            except Exception as e:
                st.metric("Portfolio Records", "Error")
        
        with col4:
            try:
                backtest_results = st.session_state.db_service.get_backtest_results(limit=10)
                st.metric("Backtest Results", len(backtest_results))
            except Exception as e:
                st.metric("Backtest Results", "Error")
        
        st.markdown("---")
        
        # Database tabs for different data types
        db_tab1, db_tab2, db_tab3, db_tab4, db_tab5 = st.tabs([
            "📈 Trading History", 
            "🔮 AI Models & Predictions", 
            "📊 Portfolio Analytics", 
            "⚡ Signals & Alerts",
            "🔧 Database Management"
        ])
        
        with db_tab1:
            st.subheader("Trading History & Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Recent Trades**")
                try:
                    trades = st.session_state.db_service.get_trades(limit=20)
                    if trades:
                        trades_df = pd.DataFrame(trades)
                        st.dataframe(trades_df[['symbol', 'trade_type', 'quantity', 'entry_price', 'pnl', 'entry_time']])
                        
                        # Trading performance chart
                        if not trades_df.empty and 'pnl' in trades_df.columns:
                            cumulative_pnl = trades_df['pnl'].fillna(0).cumsum()
                            fig = px.line(x=range(len(cumulative_pnl)), y=cumulative_pnl, 
                                        title="Cumulative P&L")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No trading history available")
                except Exception as e:
                    st.error(f"Error loading trades: {e}")
            
            with col2:
                st.write("**Trading Statistics**")
                try:
                    stats = st.session_state.db_service.get_trading_statistics(30)
                    if stats:
                        st.metric("Win Rate", f"{stats.get('win_rate', 0):.1%}")
                        st.metric("Total P&L", f"${stats.get('total_pnl', 0):.2f}")
                        st.metric("Average P&L", f"${stats.get('average_pnl', 0):.2f}")
                        st.metric("Net P&L", f"${stats.get('net_pnl', 0):.2f}")
                        
                        # P&L distribution
                        if trades:
                            pnl_values = [t['pnl'] for t in trades if t['pnl'] is not None]
                            if pnl_values:
                                fig = px.histogram(x=pnl_values, title="P&L Distribution", 
                                                 nbins=20, color_discrete_sequence=['lightblue'])
                                st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading statistics: {e}")
        
        with db_tab2:
            st.subheader("AI Models & Predictions Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Model Performance Tracking**")
                
                # Model type selector
                model_type = st.selectbox("Select Model Type", 
                                        ["LSTM", "PROPHET", "QLEARNING"], 
                                        key="db_model_type")
                
                try:
                    # Get active model for selected symbol
                    active_model = st.session_state.db_service.get_active_ai_model(
                        model_type, selected_symbol
                    )
                    
                    if active_model:
                        st.success(f"Active {model_type} model found")
                        st.json(active_model['performance_metrics'])
                        
                        # Store prediction button
                        if st.button(f"Store Test Prediction ({model_type})"):
                            try:
                                # Get current price for prediction
                                if selected_symbol in st.session_state.trading_engine.market_data:
                                    current_data = st.session_state.trading_engine.market_data[selected_symbol]
                                    if not current_data.empty:
                                        current_price = st.session_state.okx_data_service.get_current_price(selected_symbol)
                                        
                                        # Generate real AI prediction using trained model
                                        prediction_result = st.session_state.lstm_predictor.predict(current_data)
                                        predicted_price = prediction_result.get('predicted_price', current_price)
                                        confidence = prediction_result.get('confidence', 0.0)
                                        
                                        prediction_id = st.session_state.db_service.store_prediction(
                                            model_id=active_model['id'],
                                            symbol=selected_symbol,
                                            predicted_price=predicted_price,
                                            prediction_horizon=24,  # 24 hours
                                            confidence=confidence
                                        )
                                        st.success(f"Prediction stored with ID: {prediction_id}")
                            except Exception as e:
                                st.error(f"Error storing prediction: {e}")
                    else:
                        st.info(f"No active {model_type} model for {selected_symbol}")
                        
                        # Store sample model button
                        if st.button(f"Store Sample {model_type} Model"):
                            try:
                                model_id = st.session_state.db_service.store_ai_model(
                                    model_name=f"{model_type}_{selected_symbol}",
                                    model_type=model_type,
                                    symbol=selected_symbol,
                                    model_state="sample_model_state",
                                    training_data_hash="sample_hash",
                                    performance_metrics={
                                        "accuracy": 0.0,
                                        "mse": 0.0,
                                        "training_time": 0.0,
                                        "status": "placeholder_model"
                                    }
                                )
                                st.success(f"Sample model stored with ID: {model_id}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error storing model: {e}")
                
                except Exception as e:
                    st.error(f"Error with model operations: {e}")
            
            with col2:
                st.write("**Prediction Accuracy Analysis**")
                
                # Note: In a real implementation, you would have actual predictions to analyze
                st.info("Prediction accuracy tracking will be populated as models make predictions and actual prices are recorded.")
                
                # Simulate some prediction accuracy data for demonstration
                if st.button("Generate Sample Prediction Analytics"):
                    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
                    accuracy_scores = np.random.uniform(0.4, 0.9, 30)
                    
                    fig = px.line(x=dates, y=accuracy_scores, 
                                title="Model Prediction Accuracy Over Time",
                                labels={'x': 'Date', 'y': 'Accuracy Score'})
                    st.plotly_chart(fig, use_container_width=True)
        
        with db_tab3:
            st.subheader("Portfolio Analytics Dashboard")
            
            try:
                portfolio_history = st.session_state.db_service.get_portfolio_history(30)
                
                if not portfolio_history.empty:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Portfolio value over time
                        fig = px.line(portfolio_history, x=portfolio_history.index, 
                                    y='total_value', title="Portfolio Value Over Time")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Returns distribution
                        if 'daily_return' in portfolio_history.columns:
                            returns = portfolio_history['daily_return'].dropna()
                            if not returns.empty:
                                fig = px.histogram(x=returns, title="Daily Returns Distribution",
                                                 nbins=20, color_discrete_sequence=['lightgreen'])
                                st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Portfolio composition
                        latest_portfolio = portfolio_history.iloc[-1]
                        
                        labels = ['Cash', 'Positions']
                        values = [latest_portfolio.get('cash_balance', 0), 
                                latest_portfolio.get('positions_value', 0)]
                        
                        fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
                        fig.update_layout(title="Portfolio Composition")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Risk metrics
                        st.write("**Risk Metrics**")
                        if 'sharpe_ratio' in portfolio_history.columns:
                            latest_sharpe = portfolio_history['sharpe_ratio'].iloc[-1]
                            st.metric("Sharpe Ratio", f"{latest_sharpe:.2f}" if pd.notna(latest_sharpe) else "N/A")
                        
                        if 'volatility' in portfolio_history.columns:
                            latest_vol = portfolio_history['volatility'].iloc[-1]
                            st.metric("Volatility", f"{latest_vol:.2%}" if pd.notna(latest_vol) else "N/A")
                        
                        if 'drawdown' in portfolio_history.columns:
                            latest_dd = portfolio_history['drawdown'].iloc[-1]
                            st.metric("Current Drawdown", f"{latest_dd:.2%}" if pd.notna(latest_dd) else "N/A")
                
                else:
                    st.info("No portfolio history available. Start trading to see analytics.")
                    
                    # Fetch live portfolio data from OKX
                    if st.button("Load Live Portfolio Data"):
                        try:
                            # Get real account balance from OKX
                            balance_result = st.session_state.okx_connector.get_account_balance()
                            if balance_result.get('success'):
                                balances = balance_result.get('balances', {})
                                total_value = sum(float(b.get('total', 0)) for b in balances.values())
                                cash_balance = float(balances.get('USDT', {}).get('available', 0))
                                
                                st.session_state.db_service.store_portfolio_snapshot(
                                    total_value=total_value,
                                    cash_balance=cash_balance,
                                    positions_value=total_value - cash_balance,
                                    daily_return=0.0,
                                total_return=0.0,
                                sharpe_ratio=0.0,
                                volatility=0.0
                            )
                                st.success("Live portfolio data loaded from OKX!")
                                st.rerun()
                            else:
                                st.error("Failed to fetch live portfolio data from OKX")
                        except Exception as e:
                            st.error(f"Error loading live data: {e}")
                            
            except Exception as e:
                st.error(f"Error loading portfolio data: {e}")
        
        with db_tab4:
            st.subheader("Trading Signals & Alert Management")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Recent Trading Signals**")
                try:
                    signals = st.session_state.db_service.get_recent_signals(selected_symbol, 20)
                    
                    if signals:
                        signals_df = pd.DataFrame(signals)
                        st.dataframe(signals_df[['timestamp', 'signal_type', 'strength', 'confidence', 'strategy_name']])
                        
                        # Signal distribution
                        signal_counts = signals_df['signal_type'].value_counts()
                        fig = px.pie(values=signal_counts.values, names=signal_counts.index, 
                                   title="Signal Type Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No recent signals available")
                        
                        # Store sample signal button
                        if st.button("Store Sample Signal"):
                            try:
                                signal_id = st.session_state.db_service.store_trading_signal(
                                    symbol=selected_symbol,
                                    signal_type=np.random.choice(['BUY', 'SELL', 'HOLD']),
                                    strength=np.random.uniform(0.5, 1.0),
                                    confidence=np.random.uniform(0.6, 0.95),
                                    strategy_name="Sample Strategy",
                                    price=50000.0,
                                    market_regime="trending"
                                )
                                st.success(f"Sample signal stored with ID: {signal_id}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error storing signal: {e}")
                
                except Exception as e:
                    st.error(f"Error loading signals: {e}")
            
            with col2:
                st.write("**Signal Performance Analysis**")
                
                try:
                    if signals:
                        # Signal strength over time
                        signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
                        fig = px.scatter(signals_df, x='timestamp', y='strength', 
                                       color='signal_type', title="Signal Strength Over Time")
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Confidence analysis
                        avg_confidence = signals_df.groupby('signal_type')['confidence'].mean()
                        fig = px.bar(x=avg_confidence.index, y=avg_confidence.values,
                                   title="Average Confidence by Signal Type")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Generate signals to see performance analysis")
                
                except Exception as e:
                    st.error(f"Error analyzing signals: {e}")
        
        with db_tab5:
            st.subheader("Database Management & Maintenance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Database Operations**")
                
                # Data cleanup
                if st.button("Clean Old Data (>1 year)"):
                    try:
                        cleanup_result = st.session_state.db_service.cleanup_old_data(365)
                        st.success("Data cleanup completed")
                        st.json(cleanup_result)
                    except Exception as e:
                        st.error(f"Cleanup failed: {e}")
                
                # Export data
                if st.button("Export Trading Data"):
                    try:
                        trades = st.session_state.db_service.get_trades(limit=1000)
                        if trades:
                            trades_df = pd.DataFrame(trades)
                            csv = trades_df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"trading_data_{datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No data to export")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
            
            with col2:
                st.write("**System Health**")
                
                # Database connection test
                if st.button("Test Database Connection"):
                    try:
                        test_stats = st.session_state.db_service.get_trading_statistics(1)
                        st.success("Database connection successful")
                    except Exception as e:
                        st.error(f"Database connection failed: {e}")
                
                # System logs
                st.write("**Recent System Activity**")
                try:
                    # Log current access
                    st.session_state.db_service.log_system_event(
                        log_level="INFO",
                        component="DATABASE_DASHBOARD",
                        message="Database dashboard accessed",
                        details={"user_action": "view_dashboard", "symbol": selected_symbol}
                    )
                    st.info("System activity logged successfully")
                except Exception as e:
                    st.warning(f"Logging error: {e}")
    
    with tab9:
        # Live Charts Tab with TradingView Integration
        st.header("📈 Live Charts - Real-time Price Visualization")
        
        # Chart configuration
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.subheader("Chart Settings")
            
            # Symbol selection for charts
            chart_symbol = st.selectbox(
                "Chart Symbol",
                options=list(st.session_state.tradingview_charts.get_supported_symbols().keys()),
                index=0,
                key="chart_symbol"
            )
            
            # Time interval selection
            chart_interval = st.selectbox(
                "Time Interval",
                options=st.session_state.tradingview_charts.get_available_intervals(),
                index=5,  # Default to 1H
                key="chart_interval"
            )
            
            # Chart theme
            chart_theme = st.selectbox(
                "Theme",
                options=["dark", "light"],
                index=0,
                key="chart_theme"
            )
            
            # Technical indicators
            st.subheader("Technical Indicators")
            available_indicators = st.session_state.tradingview_charts.get_available_indicators()
            
            selected_indicators = st.multiselect(
                "Select Indicators",
                options=list(available_indicators.keys()),
                default=["RSI", "MACD", "Bollinger Bands", "Volume"],
                key="selected_indicators"
            )
            
            # Convert to TradingView format
            tv_indicators = [available_indicators[ind] for ind in selected_indicators]
            
            # Chart options
            st.subheader("Display Options")
            hide_toolbar = st.checkbox("Hide Side Toolbar", value=False)
            allow_symbol_change = st.checkbox("Allow Symbol Change", value=True)
            chart_height = st.slider("Chart Height", 400, 800, 600)
        
        with col1:
            st.subheader(f"Advanced Chart - {chart_symbol}")
            
            # Render main TradingView chart
            st.session_state.tradingview_charts.render_advanced_chart(
                symbol=chart_symbol,
                theme=chart_theme,
                interval=chart_interval,
                height=chart_height,
                indicators=tv_indicators,
                hide_side_toolbar=hide_toolbar,
                allow_symbol_change=allow_symbol_change
            )
        
        # Additional chart widgets
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Technical Analysis Summary")
            st.session_state.tradingview_charts.render_technical_analysis_widget(
                symbol=chart_symbol,
                theme=chart_theme,
                height=400
            )
        
        with col2:
            st.subheader("Market Overview")
            market_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT']
            st.session_state.tradingview_charts.render_market_overview(
                theme=chart_theme,
                height=400,
                symbols=market_symbols
            )
        
        with col3:
            st.subheader("Mini Chart")
            st.session_state.tradingview_charts.render_mini_chart(
                symbol=chart_symbol,
                theme=chart_theme,
                height=400
            )
        
        # Cryptocurrency screener
        st.markdown("---")
        st.subheader("Cryptocurrency Market Screener")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.session_state.tradingview_charts.render_crypto_screener(
                theme=chart_theme,
                height=500
            )
        
        with col2:
            st.subheader("Economic Calendar")
            st.session_state.tradingview_charts.render_economic_calendar(
                theme=chart_theme,
                height=500
            )
        
        # Chart integration with AI analysis
        st.markdown("---")
        st.subheader("AI Chart Analysis Integration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Analyze Current Chart Pattern", key="analyze_chart"):
                with st.spinner("Analyzing chart pattern..."):
                    try:
                        # Get market data for the selected symbol
                        market_data = st.session_state.trading_engine.get_market_data(
                            chart_symbol, chart_interval
                        )
                        
                        if not market_data.empty:
                            # Market regime analysis
                            regime_result = st.session_state.regime_detector.detect_regime(market_data)
                            
                            if 'error' not in regime_result:
                                current_regime = regime_result.get('current_regime', 'Unknown')
                                confidence = regime_result.get('confidence', 0) * 100
                                
                                st.success(f"Market Regime: **{current_regime}**")
                                st.metric("Regime Confidence", f"{confidence:.1f}%")
                                
                                # Additional regime details
                                regime_features = regime_result.get('regime_features', {})
                                if regime_features:
                                    st.write("**Regime Characteristics:**")
                                    for feature, value in regime_features.items():
                                        if isinstance(value, float):
                                            st.write(f"• {feature}: {value:.3f}")
                                        else:
                                            st.write(f"• {feature}: {value}")
                            else:
                                st.error(f"Regime analysis failed: {regime_result['error']}")
                        else:
                            st.warning("No market data available for analysis")
                            
                    except Exception as e:
                        st.error(f"Chart analysis error: {e}")
        
        with col2:
            if st.button("Generate AI Price Prediction", key="predict_chart"):
                with st.spinner("Generating AI prediction..."):
                    try:
                        # Get market data for prediction
                        market_data = st.session_state.trading_engine.get_market_data(
                            chart_symbol, chart_interval
                        )
                        
                        if not market_data.empty and len(market_data) > 60:
                            # LSTM prediction
                            lstm_result = st.session_state.lstm_predictor.predict(market_data)
                            
                            if lstm_result.get('success'):
                                next_price = lstm_result.get('next_price', 0)
                                current_price = market_data['close'].iloc[-1]
                                price_change = ((next_price - current_price) / current_price) * 100
                                confidence = lstm_result.get('confidence', 0) * 100
                                
                                st.success("AI Prediction Generated!")
                                st.metric("Predicted Next Price", f"${next_price:.4f}", f"{price_change:+.2f}%")
                                st.metric("Prediction Confidence", f"{confidence:.1f}%")
                                
                                # Prediction details
                                prediction_horizon = lstm_result.get('prediction_horizon', 1)
                                st.info(f"Prediction horizon: {prediction_horizon} period(s)")
                                
                            else:
                                st.error("LSTM prediction failed")
                        else:
                            st.warning("Insufficient data for AI prediction (need >60 data points)")
                            
                    except Exception as e:
                        st.error(f"Prediction error: {e}")
        
        # Real-time trading signals
        st.markdown("---")
        st.subheader("Real-time Trading Signals")
        
        if st.button("Generate Trading Signal for Current Chart", key="generate_signal"):
            with st.spinner("Analyzing market conditions..."):
                try:
                    # Get current market data
                    market_data = st.session_state.trading_engine.get_market_data(
                        chart_symbol, chart_interval
                    )
                    
                    if not market_data.empty:
                        # Generate trading signal using ensemble strategy
                        signal_result = st.session_state.trading_engine.generate_signal(
                            chart_symbol, market_data
                        )
                        
                        if signal_result and 'signal' in signal_result:
                            signal_type = signal_result['signal']
                            signal_strength = signal_result.get('strength', 0)
                            confidence = signal_result.get('confidence', 0) * 100
                            
                            # Display signal with appropriate color
                            if signal_type == 'BUY':
                                st.success(f"**{signal_type}** Signal Generated")
                            elif signal_type == 'SELL':
                                st.error(f"**{signal_type}** Signal Generated")
                            else:
                                st.info(f"**{signal_type}** Signal Generated")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Signal Strength", f"{signal_strength:.2f}")
                            with col2:
                                st.metric("Confidence", f"{confidence:.1f}%")
                            with col3:
                                current_price = market_data['close'].iloc[-1]
                                st.metric("Current Price", f"${current_price:.4f}")
                            
                            # Signal details
                            indicators = signal_result.get('indicators', {})
                            if indicators:
                                st.write("**Technical Indicators:**")
                                for indicator, value in indicators.items():
                                    if isinstance(value, float):
                                        st.write(f"• {indicator}: {value:.4f}")
                                    else:
                                        st.write(f"• {indicator}: {value}")
                        else:
                            st.warning("No clear trading signal detected")
                    else:
                        st.warning("No market data available for signal generation")
                        
                except Exception as e:
                    st.error(f"Signal generation error: {e}")
        
        # Chart features info
        st.markdown("---")
        st.info("""
        **TradingView Chart Features:**
        • Real-time price data from Binance
        • Professional technical analysis tools
        • Customizable indicators and overlays
        • Multiple timeframe analysis
        • Interactive chart controls
        • Economic calendar integration
        • Market screener functionality
        
        **AI Integration:**
        • Market regime detection
        • LSTM price predictions
        • Automated signal generation
        • Pattern recognition analysis
        """)
    
    # Auto refresh functionality
    if auto_refresh and st.session_state.trading_active:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
