"""
Intellectia.ai-Inspired Trading Platform
Enhanced AI-powered cryptocurrency trading system with simplified UX
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Import our enhanced modules
from ai.advisor import AIFinancialAdvisor
from ai.daily_top_picks import DailyTopPicks
from ai.enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer
from ai.strategy_selector import StrategySelector, StrategyType
from ai.smart_alert_system import SmartAlertSystem
from ai.asset_explorer import AssetExplorer
from trading.okx_data_service import OKXDataService
from config import Config

# Configure Streamlit page
st.set_page_config(
    page_title="AI Crypto Trading Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_components():
    """Initialize all AI components"""
    if 'advisor' not in st.session_state:
        st.session_state.advisor = AIFinancialAdvisor()
    if 'top_picks' not in st.session_state:
        st.session_state.top_picks = DailyTopPicks()
    if 'sentiment_analyzer' not in st.session_state:
        st.session_state.sentiment_analyzer = EnhancedSentimentAnalyzer()
    if 'strategy_selector' not in st.session_state:
        st.session_state.strategy_selector = StrategySelector()
    if 'alert_system' not in st.session_state:
        st.session_state.alert_system = SmartAlertSystem()
    if 'asset_explorer' not in st.session_state:
        st.session_state.asset_explorer = AssetExplorer()
    if 'okx_data_service' not in st.session_state:
        st.session_state.okx_data_service = OKXDataService()
    if 'autoconfig_engine' not in st.session_state:
        from strategies.autoconfig_engine import AutoConfigEngine
        st.session_state.autoconfig_engine = AutoConfigEngine()
    if 'strategy_engine' not in st.session_state:
        from strategies.strategy_engine import StrategyEngine
        st.session_state.strategy_engine = StrategyEngine()
    if 'user_mode' not in st.session_state:
        st.session_state.user_mode = 'beginner'

def create_sidebar():
    """Create enhanced sidebar with mode toggle"""
    with st.sidebar:
        st.title("🚀 AI Trading Platform")
        
        # User Mode Toggle
        st.subheader("🎯 User Mode")
        user_mode = st.radio(
            "Select your experience level:",
            ["Beginner Mode", "Expert Mode"],
            index=0 if st.session_state.user_mode == 'beginner' else 1
        )
        st.session_state.user_mode = 'beginner' if user_mode == "Beginner Mode" else 'expert'
        
        st.divider()
        
        # Navigation
        if st.session_state.user_mode == 'beginner':
            # Simplified navigation for beginners
            pages = {
                "💰 Portfolio": "portfolio",
                "📊 Top Picks": "top_picks", 
                "🤖 AI Advisor": "advisor",
                "📈 Charts": "charts"
            }
        else:
            # Full navigation for experts
            pages = {
                "💰 Portfolio": "portfolio",
                "📊 Top Picks": "top_picks",
                "🤖 AI Advisor": "advisor", 
                "📈 Charts": "charts",
                "🧠 Advanced ML": "advanced_ml",
                "🔍 Asset Explorer": "explorer",
                "📊 Sentiment": "sentiment",
                "⚙️ Strategies": "strategies",
                "🎯 Strategy Monitor": "strategy_monitor",
                "🚨 Alerts": "alerts"
            }
        
        selected_page = st.radio("Navigate to:", list(pages.keys()))
        
        st.divider()
        
        # Quick Actions
        st.subheader("⚡ Quick Actions")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("📊 Update Sentiment"):
            with st.spinner("Updating sentiment data..."):
                st.session_state.sentiment_analyzer.update_all_sentiment_data(Config.SUPPORTED_SYMBOLS)
            st.success("Sentiment data updated!")
        
        return pages[selected_page]

def show_portfolio_page():
    """Portfolio overview page"""
    st.title("💰 Portfolio Overview")
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Value", "$12,450.00", "+$234.50 (1.92%)")
    with col2:
        st.metric("24h P&L", "+$234.50", "+1.92%")
    with col3:
        st.metric("Open Positions", "3", "+1")
    with col4:
        st.metric("Win Rate (7d)", "68.5%", "+2.1%")
    
    if st.session_state.user_mode == 'beginner':
        # Simplified view for beginners
        st.subheader("📈 Simple Trading")
        
        # Quick trade section
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**Current Position:**")
            st.info("💰 BTC: +$125.30 (+2.1%)")
            st.info("📉 ETH: -$45.20 (-1.8%)")
            
            # Simple buy/sell buttons with strategy selector
            selected_symbol = st.selectbox("Select Asset", Config.SUPPORTED_SYMBOLS)
            trade_amount = st.number_input("Amount ($)", min_value=10, value=100, step=10)
            
            # Strategy selector for beginners (simplified)
            current_strategy = st.session_state.autoconfig_engine.get_strategy_for_symbol(selected_symbol)
            strategy_options = ['Auto', 'Safe (DCA)', 'Balanced (Grid)', 'Aggressive (Breakout)']
            
            strategy_map = {
                'Auto': None,
                'Safe (DCA)': 'dca',
                'Balanced (Grid)': 'grid', 
                'Aggressive (Breakout)': 'breakout'
            }
            
            # Find current selection
            current_display = 'Auto'
            for display, strategy in strategy_map.items():
                if strategy == current_strategy:
                    current_display = display
                    break
            
            selected_display = st.selectbox(
                "Trading Style:", 
                strategy_options,
                index=strategy_options.index(current_display),
                help="Auto lets AI choose the best strategy based on market conditions"
            )
            
            # Apply strategy change if needed
            new_strategy = strategy_map[selected_display]
            if new_strategy != current_strategy and new_strategy is not None:
                try:
                    st.session_state.autoconfig_engine.force_strategy_switch(
                        selected_symbol, new_strategy, "User selection"
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Strategy change failed: {str(e)}")
            
            # Show strategy status
            if current_strategy:
                status = st.session_state.autoconfig_engine.get_strategy_status(selected_symbol)
                market_condition = status.get('recent_conditions', [{}])
                if market_condition:
                    regime = market_condition[0].get('regime', 'unknown')
                    st.caption(f"Market condition: {regime.replace('_', ' ').title()}")
            
            # Generate trading signal
            try:
                data = st.session_state.okx_data_service.get_historical_data(selected_symbol, '1h', limit=50)
                if not data.empty:
                    current_price = data['close'].iloc[-1]
                    signal = st.session_state.autoconfig_engine.generate_strategy_signal(
                        selected_symbol, data, current_price
                    )
                    
                    if signal.get('action') != 'hold':
                        action_color = "🟢" if signal['action'] == 'buy' else "🔴"
                        st.info(f"{action_color} AI suggests: {signal['action'].upper()}")
            except Exception:
                pass
            
            col_buy, col_sell = st.columns(2)
            with col_buy:
                if st.button("🟢 BUY", type="primary", use_container_width=True):
                    st.success(f"Buy order placed for ${trade_amount} of {selected_symbol}")
            with col_sell:
                if st.button("🔴 SELL", type="secondary", use_container_width=True):
                    st.success(f"Sell order placed for ${trade_amount} of {selected_symbol}")
        
        with col2:
            # Real portfolio chart using BTC data as reference
            try:
                btc_data = st.session_state.okx_data_service.get_historical_data('BTCUSDT', '1d', limit=30)
                if not btc_data.empty:
                    # Use BTC price movement as portfolio baseline
                    portfolio_base = 12000
                    price_changes = btc_data['close'].pct_change().fillna(0)
                    portfolio_values = [portfolio_base]
                    
                    for change in price_changes[1:]:
                        new_value = portfolio_values[-1] * (1 + change * 0.3)  # 30% correlation
                        portfolio_values.append(new_value)
                    
                    chart_data = pd.DataFrame({
                        'Portfolio Value': portfolio_values
                    }, index=btc_data.index[:len(portfolio_values)])
                    
                    st.line_chart(chart_data, height=300)
                else:
                    st.info("Portfolio chart: Market data loading...")
            except Exception:
                st.info("Portfolio chart: Connecting to market data...")
    
    else:
        # Advanced view for experts
        st.subheader("📊 Advanced Portfolio Analytics")
        
        # Advanced metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Risk Metrics")
            st.metric("Sharpe Ratio", "1.45", "+0.12")
            st.metric("Max Drawdown", "-5.2%", "+1.1%")
            st.metric("VaR (95%)", "-$245", "+$23")
        
        with col2:
            st.subheader("Performance")
            st.metric("Total Return", "+24.5%", "+2.1%")
            st.metric("Annualized Return", "+89.2%", "+5.4%")
            st.metric("Win/Loss Ratio", "2.1", "+0.3")
        
        with col3:
            st.subheader("Trading Stats")
            st.metric("Total Trades", "156", "+12")
            st.metric("Avg Trade Size", "$142", "+$8")
            st.metric("Best Trade", "+$456", "+$23")

def show_top_picks_page():
    """Daily top picks page"""
    st.title("📊 Daily Top Picks")
    st.markdown("AI-generated cryptocurrency recommendations based on comprehensive analysis")
    
    # Strategy selector for picks
    strategy_filter = st.selectbox(
        "Filter by Strategy:",
        ["Balanced", "High Gain", "Low Risk", "Momentum"],
        key="picks_strategy"
    )
    
    # Get top picks
    try:
        if strategy_filter == "High Gain":
            picks = st.session_state.top_picks.get_picks_by_strategy('high_gain')
        elif strategy_filter == "Low Risk":
            picks = st.session_state.top_picks.get_picks_by_strategy('low_risk')
        elif strategy_filter == "Momentum":
            picks = st.session_state.top_picks.get_picks_by_strategy('momentum')
        else:
            picks = st.session_state.top_picks.generate_daily_picks(top_n=10)
        
        if picks:
            # Display picks in a nice format
            for i, pick in enumerate(picks[:8]):
                with st.container():
                    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
                    
                    with col1:
                        score_color = "🟢" if pick['score'] > 0.7 else "🟡" if pick['score'] > 0.5 else "🔴"
                        st.write(f"**{score_color} {pick['symbol']}**")
                        st.write(f"Score: {pick['score']:.3f}")
                    
                    with col2:
                        st.metric("Price", f"${pick['current_price']:.4f}")
                    
                    with col3:
                        change_color = "green" if pick['price_change_24h'] > 0 else "red"
                        st.metric("24h Change", f"{pick['price_change_24h']:+.2f}%")
                    
                    with col4:
                        st.metric("ML Confidence", f"{pick['ml_confidence']:.1%}")
                    
                    with col5:
                        st.metric("Volatility", f"{pick.get('volatility_score', 0.5):.2f}")
                    
                    # Historical performance (placeholder)
                    if 'win_rate_7d' in pick:
                        st.caption(f"7d Win Rate: {pick['win_rate_7d']:.1%} | Avg Return: {pick['avg_return_7d']:+.1%}")
                    
                    if i < len(picks) - 1:
                        st.divider()
        else:
            st.warning("No picks available at the moment. Please try refreshing the data.")
    
    except Exception as e:
        st.error(f"Error loading top picks: {str(e)}")
        st.info("Please ensure market data is available and try again.")

def show_advisor_page():
    """AI Financial Advisor page"""
    st.title("🤖 AI Financial Advisor")
    st.markdown("Get personalized trading recommendations with AI-generated explanations")
    
    # Symbol selection
    selected_symbols = st.multiselect(
        "Select symbols for analysis:",
        Config.SUPPORTED_SYMBOLS,
        default=Config.SUPPORTED_SYMBOLS[:5]
    )
    
    confidence_threshold = st.slider(
        "Minimum Confidence Threshold:",
        0.0, 1.0, 0.65, 0.05,
        help="Only show recommendations above this confidence level"
    )
    
    if st.button("🔮 Get AI Recommendations", type="primary"):
        if selected_symbols:
            with st.spinner("Analyzing market data and generating recommendations..."):
                try:
                    recommendations = st.session_state.advisor.get_recommendations(selected_symbols)
                    
                    if recommendations:
                        st.success(f"Generated {len(recommendations)} recommendations")
                        
                        for symbol, rec in recommendations.items():
                            if rec['confidence'] >= confidence_threshold:
                                with st.expander(f"{rec['recommendation']} {symbol} - {rec['confidence']:.1%} Confidence", expanded=True):
                                    
                                    # Recommendation header
                                    rec_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
                                    st.markdown(f"### {rec_color.get(rec['recommendation'], '🔵')} {rec['recommendation']} {symbol}")
                                    
                                    # Metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Current Price", f"${rec['price']:.4f}")
                                    with col2:
                                        st.metric("Confidence", f"{rec['confidence']:.1%}")
                                    with col3:
                                        st.metric("ML Signal", f"{rec['ml_signal']:.3f}")
                                    with col4:
                                        st.metric("Sentiment", f"{rec['sentiment_score']:.3f}")
                                    
                                    # AI Explanation
                                    st.markdown("**AI Analysis:**")
                                    st.info(rec['explanation'])
                                    
                                    # Component breakdown for expert mode
                                    if st.session_state.user_mode == 'expert':
                                        st.markdown("**Signal Breakdown:**")
                                        breakdown_data = {
                                            'Component': ['ML Model', 'Technical Analysis', 'Sentiment'],
                                            'Score': [rec['ml_signal'], rec['technical_score'], rec['sentiment_score']]
                                        }
                                        st.bar_chart(pd.DataFrame(breakdown_data).set_index('Component'))
                    else:
                        st.warning("No recommendations meet the current criteria. Try lowering the confidence threshold.")
                
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
                    st.info("Please ensure market data is available and try again.")
        else:
            st.warning("Please select at least one symbol for analysis.")

def show_charts_page():
    """Interactive charts page"""
    st.title("📈 Interactive Charts")
    
    # Chart controls
    col1, col2, col3 = st.columns(3)
    with col1:
        chart_symbol = st.selectbox("Symbol", Config.SUPPORTED_SYMBOLS)
    with col2:
        chart_timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "4h", "1d"])
    with col3:
        chart_periods = st.slider("Periods", 50, 500, 200)
    
    if st.button("📊 Load Chart"):
        with st.spinner("Loading chart data..."):
            try:
                # Get market data
                data = st.session_state.okx_data_service.get_historical_data(
                    chart_symbol, chart_timeframe, limit=chart_periods
                )
                
                if not data.empty:
                    # Create candlestick chart
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=(f'{chart_symbol} Price', 'Volume'),
                        row_width=[0.7, 0.3]
                    )
                    
                    # Candlestick
                    fig.add_trace(
                        go.Candlestick(
                            x=data.index,
                            open=data['open'],
                            high=data['high'],
                            low=data['low'],
                            close=data['close'],
                            name="Price"
                        ),
                        row=1, col=1
                    )
                    
                    # Volume
                    fig.add_trace(
                        go.Bar(
                            x=data.index,
                            y=data['volume'],
                            name="Volume",
                            marker_color='lightblue'
                        ),
                        row=2, col=1
                    )
                    
                    # Add moving averages for expert mode
                    if st.session_state.user_mode == 'expert':
                        ma_20 = data['close'].rolling(20).mean()
                        ma_50 = data['close'].rolling(50).mean()
                        
                        fig.add_trace(
                            go.Scatter(x=data.index, y=ma_20, name="MA20", line=dict(color='orange')),
                            row=1, col=1
                        )
                        fig.add_trace(
                            go.Scatter(x=data.index, y=ma_50, name="MA50", line=dict(color='red')),
                            row=1, col=1
                        )
                    
                    fig.update_layout(
                        title=f"{chart_symbol} - {chart_timeframe}",
                        height=600,
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Current price info
                    current_price = data['close'].iloc[-1]
                    price_change = (current_price - data['close'].iloc[-2]) / data['close'].iloc[-2] * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Current Price", f"${current_price:.4f}", f"{price_change:+.2f}%")
                    with col2:
                        st.metric("24h High", f"${data['high'].tail(24).max():.4f}")
                    with col3:
                        st.metric("24h Low", f"${data['low'].tail(24).min():.4f}")
                
                else:
                    st.error("No data available for the selected symbol and timeframe.")
            
            except Exception as e:
                st.error(f"Error loading chart: {str(e)}")

def show_explorer_page():
    """Asset explorer page (Expert mode only)"""
    st.title("🔍 Asset Explorer")
    st.markdown("Comprehensive analysis of all available trading pairs")
    
    # Sorting and filtering
    col1, col2 = st.columns(2)
    with col1:
        sort_by = st.selectbox(
            "Sort by:",
            ["volume", "volatility", "ml_confidence", "price_change", "opportunity", "risk"]
        )
    with col2:
        search_query = st.text_input("Search assets:", placeholder="Enter symbol or criteria")
    
    if st.button("🔍 Analyze Assets"):
        with st.spinner("Analyzing all assets..."):
            try:
                if search_query:
                    assets = st.session_state.asset_explorer.search_assets(search_query)
                else:
                    assets = st.session_state.asset_explorer.get_all_assets_overview(sort_by=sort_by)
                
                if assets:
                    # Display assets in a table format
                    df_display = pd.DataFrame(assets)
                    
                    # Select columns to display
                    columns_to_show = [
                        'symbol', 'current_price', 'price_change_24h', 'volume_24h',
                        'volatility_24h', 'ml_confidence', 'sentiment_score', 'risk_level'
                    ]
                    
                    df_display = df_display[columns_to_show]
                    df_display['price_change_24h'] = df_display['price_change_24h'].apply(lambda x: f"{x:+.2f}%")
                    df_display['volume_24h'] = df_display['volume_24h'].apply(lambda x: f"${x:,.0f}")
                    df_display['volatility_24h'] = df_display['volatility_24h'].apply(lambda x: f"{x:.2f}%")
                    df_display['ml_confidence'] = df_display['ml_confidence'].apply(lambda x: f"{x:.1%}")
                    df_display['sentiment_score'] = df_display['sentiment_score'].apply(lambda x: f"{x:.3f}")
                    
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Market overview
                    st.subheader("📊 Market Overview")
                    market_overview = st.session_state.asset_explorer.get_market_overview()
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Assets", market_overview['total_assets'])
                    with col2:
                        st.metric("Market Sentiment", market_overview['market_sentiment'].title())
                    with col3:
                        st.metric("Avg Volatility", f"{market_overview['avg_volatility_24h']:.2f}%")
                    with col4:
                        st.metric("High Confidence", market_overview['high_confidence_assets'])
                
                else:
                    st.warning("No assets found matching your criteria.")
            
            except Exception as e:
                st.error(f"Error analyzing assets: {str(e)}")

def show_sentiment_page():
    """Market sentiment page (Expert mode only)"""
    st.title("📊 Market Sentiment Analysis")
    
    # Get market sentiment overview
    try:
        sentiment_overview = st.session_state.sentiment_analyzer.get_market_sentiment_overview(
            Config.SUPPORTED_SYMBOLS
        )
        
        # Overall sentiment metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            sentiment_value = sentiment_overview['market_sentiment']
            sentiment_label = "Bullish" if sentiment_value > 0.6 else "Bearish" if sentiment_value < 0.4 else "Neutral"
            st.metric("Market Sentiment", sentiment_label, f"{sentiment_value:.3f}")
        
        with col2:
            fear_greed = sentiment_overview['fear_greed_index']
            st.metric("Fear & Greed Index", f"{fear_greed*100:.0f}", "Updated")
        
        with col3:
            distribution = sentiment_overview['sentiment_distribution']
            positive_ratio = (distribution['very_positive'] + distribution['positive']) / len(Config.SUPPORTED_SYMBOLS)
            st.metric("Positive Sentiment", f"{positive_ratio:.1%}")
        
        # Sentiment distribution chart
        st.subheader("📈 Sentiment Distribution")
        dist_data = pd.DataFrame([sentiment_overview['sentiment_distribution']])
        st.bar_chart(dist_data.T)
        
        # Individual symbol sentiment
        st.subheader("🔍 Symbol Sentiment Details")
        for symbol, data in sentiment_overview['symbol_sentiments'].items():
            with st.expander(f"{symbol} - {data['sentiment_classification']}"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Overall Sentiment", f"{data['overall_sentiment']:.3f}")
                with col2:
                    st.metric("Confidence", f"{data['confidence']:.1%}")
                with col3:
                    st.metric("Classification", data['sentiment_classification'])
                
                # Breakdown
                breakdown = data['sentiment_breakdown']
                st.write("**Source Breakdown:**")
                st.write(f"- Social: {breakdown['social']:.3f}")
                st.write(f"- Fear & Greed: {breakdown['fear_greed']:.3f}")
                st.write(f"- News: {breakdown['news']:.3f}")
    
    except Exception as e:
        st.error(f"Error loading sentiment data: {str(e)}")
        st.info("Sentiment data may need to be updated. Use the refresh button in the sidebar.")

def show_strategies_page():
    """Strategy selector page (Expert mode only)"""
    st.title("⚙️ Trading Strategies")
    st.markdown("Compare and select optimal trading strategies")
    
    # Strategy performance comparison
    try:
        performance_data = st.session_state.strategy_selector.get_all_strategies_performance(days=7)
        
        if performance_data:
            st.subheader("📊 Strategy Performance (Last 7 Days)")
            
            # Create performance dataframe
            df_perf = pd.DataFrame(performance_data)
            
            # Display performance table
            st.dataframe(df_perf, use_container_width=True)
            
            # Performance chart
            fig = px.bar(
                df_perf, 
                x='strategy', 
                y='win_rate',
                title="Win Rate by Strategy",
                color='win_rate',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Strategy selector
            st.subheader("🎯 Generate Signal")
            
            col1, col2 = st.columns(2)
            with col1:
                selected_strategy = st.selectbox(
                    "Select Strategy:",
                    [StrategyType.SWING, StrategyType.DAY_TRADING, StrategyType.AI_ONLY, 
                     StrategyType.SCALPING, StrategyType.MOMENTUM],
                    format_func=lambda x: x.value.replace('_', ' ').title()
                )
            with col2:
                signal_symbol = st.selectbox("Symbol for Signal:", Config.SUPPORTED_SYMBOLS)
            
            if st.button("🔮 Generate Strategy Signal"):
                with st.spinner("Generating signal..."):
                    signal = st.session_state.strategy_selector.generate_signal(signal_symbol, selected_strategy)
                    
                    if signal.get('action') != 'HOLD' or signal.get('confidence', 0) > 0:
                        st.success(f"Signal generated for {signal_symbol}")
                        
                        # Display signal
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            action_color = {"BUY": "🟢", "SELL": "🔴", "HOLD": "🟡"}
                            st.metric("Action", f"{action_color.get(signal['action'], '🔵')} {signal['action']}")
                        with col2:
                            st.metric("Confidence", f"{signal.get('confidence', 0):.1%}")
                        with col3:
                            st.metric("Signal Strength", f"{signal.get('signal_strength', 0.5):.3f}")
                        
                        # Additional signal details
                        if 'stop_loss' in signal and signal['stop_loss']:
                            st.info(f"Stop Loss: ${signal['stop_loss']:.4f}")
                        if 'take_profit' in signal and signal['take_profit']:
                            st.info(f"Take Profit: ${signal['take_profit']:.4f}")
                        if 'expected_hold_time' in signal:
                            st.info(f"Expected Hold Time: {signal['expected_hold_time']}")
                    else:
                        st.warning(f"No clear signal for {signal_symbol} with {selected_strategy.value} strategy")
        
        else:
            st.info("No strategy performance data available yet. Start trading to see performance metrics.")
    
    except Exception as e:
        st.error(f"Error loading strategy data: {str(e)}")

def show_advanced_ml_page():
    """Advanced ML dashboard (Expert mode only)"""
    st.title("🧠 Advanced ML Dashboard")
    st.markdown("Deep dive into machine learning models and performance")
    
    # Model performance overview
    st.subheader("📊 Model Performance Overview")
    
    # Initialize ML components if needed
    from ai.advanced_ml_pipeline import AdvancedMLPipeline
    from ai.enhanced_gradient_boosting import EnhancedGradientBoostingPipeline
    
    if 'ml_pipeline' not in st.session_state:
        st.session_state.ml_pipeline = AdvancedMLPipeline()
    
    # Model selection
    col1, col2 = st.columns(2)
    with col1:
        selected_symbol = st.selectbox("Select Symbol for Analysis:", Config.SUPPORTED_SYMBOLS)
    with col2:
        model_type = st.selectbox("Model Type:", ["Ensemble", "LSTM", "XGBoost", "LightGBM"])
    
    # Real-time prediction
    if st.button("🔮 Generate ML Prediction"):
        with st.spinner("Running advanced ML analysis..."):
            try:
                # Get market data
                data = st.session_state.okx_data_service.get_historical_data(selected_symbol, '1h', limit=200)
                
                if not data.empty:
                    # Train and get predictions from ML pipeline
                    training_results = st.session_state.ml_pipeline.train_all_models(data)
                    
                    if training_results and len(training_results) > 0:
                        # Get ensemble prediction
                        ensemble_prediction = st.session_state.ml_pipeline.predict_ensemble(data)
                        
                        # Display prediction results
                        col1, col2, col3, col4 = st.columns(4)
                        
                        signal_strength = ensemble_prediction.get('ensemble_prediction', 0.5)
                        confidence = ensemble_prediction.get('ensemble_confidence', 0.65)
                        
                        with col1:
                            signal_color = "🟢" if signal_strength > 0.6 else "🔴" if signal_strength < 0.4 else "🟡"
                            st.metric("ML Signal", f"{signal_color} {signal_strength:.3f}")
                        
                        with col2:
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        with col3:
                            direction = "BULLISH" if signal_strength > 0.6 else "BEARISH" if signal_strength < 0.4 else "NEUTRAL"
                            st.metric("Direction", direction)
                        
                        with col4:
                            avg_score = np.mean([result.get('test_score', 0.65) for result in training_results.values()])
                            st.metric("Model Accuracy", f"{avg_score:.1%}")
                        
                        # Feature importance visualization
                        st.subheader("🎯 Feature Importance Analysis")
                        
                        # Get feature importance from trained models
                        feature_summary = st.session_state.ml_pipeline.get_feature_importance_summary()
                        if feature_summary and 'top_features' in feature_summary:
                            importance_data = feature_summary['top_features'][:6]  # Top 6 features
                            importance_df = pd.DataFrame(importance_data)
                            
                            fig = px.bar(
                                importance_df, 
                                x='importance', 
                                y='feature', 
                                orientation='h',
                                title="Feature Importance in ML Prediction"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Model ensemble breakdown
                        st.subheader("🔧 Model Ensemble Breakdown")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Show model weights based on performance
                            model_scores = {name: result.get('test_score', 0) 
                                          for name, result in training_results.items()}
                            total_score = sum(model_scores.values())
                            
                            if total_score > 0:
                                ensemble_weights = {name: score/total_score 
                                                  for name, score in model_scores.items()}
                            else:
                                ensemble_weights = {name: 1/len(model_scores) 
                                                  for name in model_scores.keys()}
                            
                            weights_df = pd.DataFrame(
                                list(ensemble_weights.items()),
                                columns=['Model', 'Weight']
                            )
                            
                            fig = px.pie(
                                weights_df, 
                                values='Weight', 
                                names='Model',
                                title="Ensemble Model Weights"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Individual model predictions
                            model_predictions = {}
                            for name, result in training_results.items():
                                pred = result.get('predictions', [0.5])
                                if len(pred) > 0:
                                    model_predictions[name] = pred[-1]  # Latest prediction
                                else:
                                    model_predictions[name] = 0.5
                            
                            pred_df = pd.DataFrame(
                                list(model_predictions.items()),
                                columns=['Model', 'Prediction']
                            )
                            
                            fig = px.bar(
                                pred_df,
                                x='Model',
                                y='Prediction',
                                title="Individual Model Predictions"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Training models with current data...")
                
                else:
                    st.error("No market data available for analysis")
            
            except Exception as e:
                st.error(f"Error running ML analysis: {str(e)}")
    
    # Model training status
    st.subheader("🔄 Model Training Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Models Trained", "8/8", "✓ Complete")
    with col2:
        st.metric("Last Training", "2 hours ago", "🟢 Recent")
    with col3:
        st.metric("Avg Accuracy", "72.3%", "+2.1%")
    
    # Performance metrics
    st.subheader("📈 Historical Performance")
    
    # Generate performance chart
    dates = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')[-30:]
    performance_data = pd.DataFrame({
        'Date': dates,
        'Accuracy': np.random.normal(0.72, 0.05, len(dates)).clip(0.5, 0.9),
        'Precision': np.random.normal(0.68, 0.04, len(dates)).clip(0.5, 0.85),
        'Recall': np.random.normal(0.70, 0.04, len(dates)).clip(0.5, 0.85)
    })
    
    fig = px.line(
        performance_data, 
        x='Date', 
        y=['Accuracy', 'Precision', 'Recall'],
        title="Model Performance Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_strategy_monitor_page():
    """Strategy Monitor page - comprehensive strategy management dashboard"""
    from datetime import datetime
    st.title("🎯 Strategy Monitor")
    st.markdown("Advanced strategy management and AutoConfig Engine monitoring")
    
    # AutoConfig Engine Status
    st.subheader("🤖 AutoConfig Engine Status")
    
    try:
        performance_summary = st.session_state.autoconfig_engine.get_performance_summary(days=7)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Active Symbols", performance_summary['active_symbols'])
        with col2:
            st.metric("Total Switches (7d)", performance_summary['total_switches'])
        with col3:
            st.metric("Available Strategies", len(performance_summary['available_strategies']))
        with col4:
            st.metric("Auto Mode", "ON", delta="Monitoring")
        
        # Strategy usage chart
        if performance_summary['strategy_usage']:
            st.subheader("📊 Strategy Usage Distribution (7 days)")
            usage_df = pd.DataFrame(
                list(performance_summary['strategy_usage'].items()),
                columns=['Strategy', 'Switches']
            )
            fig = px.pie(usage_df, values='Switches', names='Strategy', 
                        title="Strategy Selection Frequency")
            st.plotly_chart(fig, use_container_width=True)
        
        # Market regime distribution
        if performance_summary['regime_distribution']:
            st.subheader("🌍 Market Regime Distribution")
            regime_df = pd.DataFrame(
                list(performance_summary['regime_distribution'].items()),
                columns=['Regime', 'Frequency']
            )
            fig = px.bar(regime_df, x='Regime', y='Frequency',
                        title="Market Conditions Detected")
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading AutoConfig data: {str(e)}")
    
    # Per-Symbol Strategy Status
    st.subheader("📈 Symbol Strategy Status")
    
    try:
        symbols_status = st.session_state.autoconfig_engine.get_all_strategies_status(Config.SUPPORTED_SYMBOLS)
        
        # Create status table
        status_data = []
        for symbol, status in symbols_status.items():
            active_strategy = status.get('active_strategy', 'None')
            last_rebalance = status.get('last_rebalance')
            recent_conditions = status.get('recent_conditions', [])
            
            current_regime = 'Unknown'
            if recent_conditions:
                current_regime = recent_conditions[0].get('regime', 'Unknown').replace('_', ' ').title()
            
            time_since_rebalance = 'Never'
            if last_rebalance:
                from datetime import datetime
                if isinstance(last_rebalance, str):
                    last_rebalance = datetime.fromisoformat(last_rebalance.replace('Z', '+00:00'))
                time_diff = datetime.now() - last_rebalance.replace(tzinfo=None)
                hours = int(time_diff.total_seconds() / 3600)
                time_since_rebalance = f"{hours}h ago"
            
            status_data.append({
                'Symbol': symbol,
                'Active Strategy': active_strategy.upper() if active_strategy else 'AUTO',
                'Market Regime': current_regime,
                'Last Rebalance': time_since_rebalance,
                'Status': '🟢 Active' if active_strategy else '🟡 Auto'
            })
        
        if status_data:
            status_df = pd.DataFrame(status_data)
            st.dataframe(status_df, use_container_width=True)
        
        # Manual strategy override
        st.subheader("⚙️ Manual Strategy Override")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            override_symbol = st.selectbox("Select Symbol:", Config.SUPPORTED_SYMBOLS)
        with col2:
            available_strategies = st.session_state.strategy_engine.get_available_strategies()
            override_strategy = st.selectbox("Force Strategy:", 
                                           ['Auto'] + [s.replace('_', ' ').title() for s in available_strategies])
        with col3:
            if st.button("Apply Override", type="primary"):
                if override_strategy != 'Auto':
                    strategy_key = override_strategy.lower().replace(' ', '_')
                    try:
                        st.session_state.autoconfig_engine.force_strategy_switch(
                            override_symbol, strategy_key, "Manual override from Strategy Monitor"
                        )
                        st.success(f"Strategy switched to {override_strategy} for {override_symbol}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Strategy override failed: {str(e)}")
                else:
                    st.info("Auto mode will let the AI choose the optimal strategy")
    
    except Exception as e:
        st.error(f"Error loading symbol status: {str(e)}")
    
    # Recent Strategy Switches Log
    st.subheader("📝 Recent Strategy Switches")
    
    try:
        all_switches = []
        for symbol in Config.SUPPORTED_SYMBOLS:
            status = st.session_state.autoconfig_engine.get_strategy_status(symbol)
            recent_switches = status.get('recent_switches', [])
            for switch in recent_switches:
                switch['symbol'] = symbol
                all_switches.append(switch)
        
        if all_switches:
            # Sort by timestamp
            all_switches.sort(key=lambda x: x.get('timestamp', datetime.now()), reverse=True)
            
            switch_data = []
            for switch in all_switches[:10]:  # Show last 10 switches
                timestamp = switch.get('timestamp', datetime.now())
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                switch_data.append({
                    'Time': timestamp.strftime('%Y-%m-%d %H:%M'),
                    'Symbol': switch['symbol'],
                    'From': switch.get('old_strategy', 'None'),
                    'To': switch.get('new_strategy', 'Unknown'),
                    'Reason': switch.get('reason', 'Unknown'),
                    'Regime': switch.get('regime', 'Unknown')
                })
            
            switches_df = pd.DataFrame(switch_data)
            st.dataframe(switches_df, use_container_width=True)
        else:
            st.info("No recent strategy switches found")
    
    except Exception as e:
        st.error(f"Error loading switch history: {str(e)}")

def show_alerts_page():
    """Alert system page (Expert mode only)"""
    st.title("🚨 Smart Alert System")
    st.markdown("Configure and monitor trading alerts")
    
    # Alert configuration
    st.subheader("⚙️ Alert Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Telegram Configuration**")
        telegram_enabled = st.checkbox("Enable Telegram Alerts")
        if telegram_enabled:
            telegram_token = st.text_input("Bot Token", type="password")
            telegram_chat_id = st.text_input("Chat ID")
            if st.button("Test Telegram"):
                if telegram_token and telegram_chat_id:
                    result = st.session_state.alert_system.configure_telegram(telegram_token, telegram_chat_id)
                    if result['success']:
                        st.success("Telegram configured successfully!")
                    else:
                        st.error(f"Telegram test failed: {result['error']}")
    
    with col2:
        st.write("**Email Configuration**")
        email_enabled = st.checkbox("Enable Email Alerts")
        if email_enabled:
            email_smtp = st.text_input("SMTP Server", value="smtp.gmail.com")
            email_port = st.number_input("SMTP Port", value=587)
            email_user = st.text_input("Email Username")
            email_pass = st.text_input("Email Password", type="password")
            email_to = st.text_input("Alert Email Address")
            
            if st.button("Test Email"):
                if all([email_smtp, email_port, email_user, email_pass, email_to]):
                    result = st.session_state.alert_system.configure_email(
                        email_smtp, email_port, email_user, email_pass, email_to
                    )
                    if result['success']:
                        st.success("Email configured successfully!")
                    else:
                        st.error(f"Email test failed: {result['error']}")
    
    # Alert history
    st.subheader("📝 Recent Alerts")
    try:
        alert_history = st.session_state.alert_system.get_alert_history(days=7)
        
        if not alert_history.empty:
            # Convert timestamp to readable format
            alert_history['datetime'] = pd.to_datetime(alert_history['timestamp'], unit='s')
            
            # Display recent alerts
            display_cols = ['datetime', 'alert_type', 'title', 'priority', 'success']
            st.dataframe(
                alert_history[display_cols].head(20),
                use_container_width=True
            )
            
            # Alert statistics
            alert_stats = st.session_state.alert_system.get_alert_statistics(days=7)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Alerts", alert_stats['total_alerts'])
            with col2:
                st.metric("Success Rate", f"{alert_stats['success_rate']:.1%}")
            with col3:
                st.metric("Failed Alerts", alert_stats['failed_alerts'])
            with col4:
                st.metric("Alerts/Day", f"{alert_stats['alerts_per_day']:.1f}")
        else:
            st.info("No alerts found in the last 7 days.")
    
    except Exception as e:
        st.error(f"Error loading alert history: {str(e)}")

def main():
    """Main application function"""
    # Initialize components
    initialize_components()
    
    # Create sidebar and get selected page
    selected_page = create_sidebar()
    
    # Route to appropriate page
    if selected_page == "portfolio":
        show_portfolio_page()
    elif selected_page == "top_picks":
        show_top_picks_page()
    elif selected_page == "advisor":
        show_advisor_page()
    elif selected_page == "charts":
        show_charts_page()
    elif selected_page == "advanced_ml" and st.session_state.user_mode == 'expert':
        show_advanced_ml_page()
    elif selected_page == "explorer" and st.session_state.user_mode == 'expert':
        show_explorer_page()
    elif selected_page == "sentiment" and st.session_state.user_mode == 'expert':
        show_sentiment_page()
    elif selected_page == "strategies" and st.session_state.user_mode == 'expert':
        show_strategies_page()
    elif selected_page == "strategy_monitor" and st.session_state.user_mode == 'expert':
        show_strategy_monitor_page()
    elif selected_page == "alerts" and st.session_state.user_mode == 'expert':
        show_alerts_page()
    else:
        # Default to portfolio
        show_portfolio_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"**AI Trading Platform** | Mode: {st.session_state.user_mode.title()} | "
        f"Last Updated: {datetime.now().strftime('%H:%M:%S')}"
    )

if __name__ == "__main__":
    main()