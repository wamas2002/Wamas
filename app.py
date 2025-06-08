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
from frontend.advanced_ml_interface import AdvancedMLInterface
from trading.engine import TradingEngine
from ai.predictor import AIPredictor
from ai.lstm_predictor import AdvancedLSTMPredictor
from ai.prophet_predictor import AdvancedProphetPredictor
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
        from ai.simple_ml_pipeline import SimpleMLPipeline
        st.session_state.advanced_ml_pipeline = SimpleMLPipeline()
    
    if 'transformer_ensemble' not in st.session_state:
        from ai.transformer_ensemble import TransformerEnsemble
        st.session_state.transformer_ensemble = TransformerEnsemble()
    
    if 'freqai_pipeline' not in st.session_state:
        from ai.freqai_pipeline import FreqAILevelPipeline
        st.session_state.freqai_pipeline = FreqAILevelPipeline()
    
    if 'advanced_ml_interface' not in st.session_state:
        from frontend.advanced_ml_interface import AdvancedMLInterface
        st.session_state.advanced_ml_interface = AdvancedMLInterface()
    
    if 'market_sentiment_analyzer' not in st.session_state:
        from ai.market_sentiment_analyzer import MarketSentimentAnalyzer
        st.session_state.market_sentiment_analyzer = MarketSentimentAnalyzer()
    
    if 'comprehensive_ml_pipeline' not in st.session_state:
        from ai.comprehensive_ml_pipeline import ComprehensiveMLPipeline
        st.session_state.comprehensive_ml_pipeline = ComprehensiveMLPipeline()
    
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
        # Advanced ML Laboratory Interface
        st.session_state.advanced_ml_interface.render_ml_dashboard()
    
    # Optional: Additional ML Laboratory Features (if needed)
    # Can be extended with custom model configurations here
    
if __name__ == "__main__":
    main()
