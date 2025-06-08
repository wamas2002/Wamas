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
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Live Trading", 
        "🧠 AI Insights", 
        "📈 Backtesting", 
        "⚡ Futures Mode",
        "📋 Performance"
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
    
    # Auto refresh functionality
    if auto_refresh and st.session_state.trading_active:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
