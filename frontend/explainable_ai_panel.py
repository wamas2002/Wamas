"""
Explainable AI Panel - Frontend component for displaying trade reasoning
Shows AI decision explanations with visual elements and confidence indicators
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
from utils.confidence_ui import (
    display_confidence_badge, display_decision_card,
    display_prediction_highlight, display_feature_importance,
    display_model_performance_summary, get_confidence_color,
    get_confidence_emoji
)

def show_explainable_ai_panel():
    """Main explainable AI dashboard page"""
    st.title("🧠 Explainable AI - Trade Reasoning")
    st.markdown("**Understand why AI made specific trading decisions**")
    
    # Real-time market analysis and decision generation
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.subheader("🎯 Live AI Decision Center")
    
    with col2:
        if st.button("🔄 Generate Live Decisions", type="primary"):
            with st.spinner("Analyzing markets and generating decisions..."):
                if 'live_decision_generator' in st.session_state:
                    decisions = st.session_state.live_decision_generator.generate_decisions_for_all_symbols()
                    if decisions:
                        st.success(f"Generated {len(decisions)} new trading decisions")
                        st.rerun()
                    else:
                        st.warning("Unable to generate decisions - checking market data")
    
    with col3:
        if st.button("📊 Market Regime"):
            if 'live_decision_generator' in st.session_state:
                regime = st.session_state.live_decision_generator.get_market_regime_analysis()
                display_confidence_badge(regime['confidence'], f"Market: {regime['regime']}")
                st.write(regime['description'])
    
    # Initialize trade reason logger if not already done
    if 'trade_reason_logger' not in st.session_state:
        from ai.trade_reason_logger import TradeReasonLogger
        st.session_state.trade_reason_logger = TradeReasonLogger()
    
    logger = st.session_state.trade_reason_logger
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Recent Decisions", "🎯 By Symbol", "📊 Model Performance", "🔍 Search History"])
    
    with tab1:
        show_recent_decisions(logger)
    
    with tab2:
        show_decisions_by_symbol(logger)
    
    with tab3:
        show_model_performance(logger)
    
    with tab4:
        show_decision_search(logger)

def show_recent_decisions(logger):
    """Display recent trading decisions with explanations"""
    st.subheader("🕒 Recent AI Trading Decisions")
    
    # Get recent explanations
    recent_explanations = logger.get_recent_explanations(limit=10)
    
    if not recent_explanations:
        st.info("🤖 No trade decisions recorded yet. AI explanations will appear here when trades are executed.")
        # Generate sample data for demonstration
        _show_sample_explanations()
        return
    
    # Display each explanation
    for i, explanation in enumerate(recent_explanations):
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                # Symbol and decision
                decision_color = _get_decision_color(explanation['decision'])
                st.markdown(f"**{explanation['symbol']}** - {_get_decision_emoji(explanation['decision'])} **{explanation['decision']}**")
                
                # Model and timestamp
                st.caption(f"Model: {explanation['model']} | {_format_timestamp(explanation['timestamp'])}")
            
            with col2:
                # Confidence meter
                confidence = explanation['confidence']
                st.metric("Confidence", f"{confidence}%", delta=None)
                
                # Signal strength badge
                strength = explanation['signal_strength']
                strength_color = _get_strength_color(strength)
                st.markdown(f"<span style='background-color: {strength_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px'>{strength} Signal</span>", unsafe_allow_html=True)
            
            with col3:
                # Expand button
                if st.button(f"Details", key=f"expand_{i}"):
                    st.session_state[f"show_details_{i}"] = not st.session_state.get(f"show_details_{i}", False)
            
            # Detailed explanation (expandable)
            if st.session_state.get(f"show_details_{i}", False):
                st.markdown("---")
                
                # Top features
                st.markdown("**🎯 Key Factors:**")
                for j, feature in enumerate(explanation['top_features'][:3]):
                    st.markdown(f"• {feature}")
                
                # Market conditions and risks
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown(f"**📊 Market:** {explanation['market_conditions']}")
                with col_right:
                    if explanation['risk_factors']:
                        st.markdown(f"**⚠️ Risks:** {', '.join(explanation['risk_factors'][:2])}")
            
            st.markdown("---")

def show_decisions_by_symbol(logger):
    """Show decisions filtered by trading symbol"""
    st.subheader("🎯 Decisions by Trading Pair")
    
    # Symbol selector
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "XRPUSDT"]
    selected_symbol = st.selectbox("Select Trading Pair:", symbols)
    
    # Get explanations for selected symbol
    symbol_explanations = logger.get_recent_explanations(symbol=selected_symbol, limit=20)
    
    if not symbol_explanations:
        st.info(f"No AI decisions recorded for {selected_symbol} yet.")
        return
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_decisions = len(symbol_explanations)
        st.metric("Total Decisions", total_decisions)
    
    with col2:
        avg_confidence = sum(exp['confidence'] for exp in symbol_explanations) / len(symbol_explanations)
        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
    
    with col3:
        buy_decisions = sum(1 for exp in symbol_explanations if exp['decision'] == 'BUY')
        st.metric("Buy Signals", buy_decisions)
    
    with col4:
        models_used = len(set(exp['model'] for exp in symbol_explanations))
        st.metric("Models Used", models_used)
    
    # Decision timeline chart
    if len(symbol_explanations) > 1:
        st.markdown("**📈 Decision Timeline**")
        _create_decision_timeline(symbol_explanations)
    
    # Recent decisions table
    st.markdown("**📋 Recent Decisions**")
    _create_decisions_table(symbol_explanations[:10])

def show_model_performance(logger):
    """Display model performance analytics"""
    st.subheader("📊 AI Model Performance Analysis")
    
    # Get model performance data
    model_stats = logger.get_model_performance_summary()
    
    if not model_stats:
        st.info("No model performance data available yet.")
        return
    
    # Model comparison metrics
    st.markdown("**🏆 Model Comparison**")
    
    model_df = []
    for model, stats in model_stats.items():
        model_df.append({
            'Model': model,
            'Decisions': stats['count'],
            'Avg Confidence': f"{stats['avg_confidence']:.1f}%",
            'Buy Signals': stats['decision_distribution']['BUY'],
            'Sell Signals': stats['decision_distribution']['SELL'],
            'Hold Signals': stats['decision_distribution']['HOLD']
        })
    
    if model_df:
        st.dataframe(pd.DataFrame(model_df), use_container_width=True)
        
        # Confidence distribution chart
        st.markdown("**📊 Confidence Distribution by Model**")
        _create_confidence_chart(model_stats)
        
        # Decision distribution pie chart
        st.markdown("**🥧 Decision Distribution**")
        _create_decision_pie_chart(model_stats)

def show_decision_search(logger):
    """Search and filter trade decisions"""
    st.subheader("🔍 Search Trade History")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Date range filter
        date_range = st.date_input(
            "Date Range:",
            value=(datetime.now() - timedelta(days=7), datetime.now()),
            max_value=datetime.now()
        )
    
    with col2:
        # Decision type filter
        decision_filter = st.multiselect(
            "Decision Types:",
            ["BUY", "SELL", "HOLD"],
            default=["BUY", "SELL", "HOLD"]
        )
    
    # Confidence range filter
    confidence_range = st.slider("Confidence Range:", 0, 100, (0, 100))
    
    # Get all explanations and filter
    all_explanations = logger.get_recent_explanations(limit=100)
    
    # Apply filters
    filtered_explanations = []
    for exp in all_explanations:
        exp_date = datetime.fromisoformat(exp['timestamp'].replace('Z', '+00:00')).date()
        
        # Date filter
        if len(date_range) == 2 and not (date_range[0] <= exp_date <= date_range[1]):
            continue
            
        # Decision filter
        if exp['decision'] not in decision_filter:
            continue
            
        # Confidence filter
        if not (confidence_range[0] <= exp['confidence'] <= confidence_range[1]):
            continue
        
        filtered_explanations.append(exp)
    
    st.markdown(f"**Found {len(filtered_explanations)} matching decisions**")
    
    if filtered_explanations:
        _create_decisions_table(filtered_explanations[:20])

def _show_sample_explanations():
    """Show sample explanations for demonstration"""
    st.markdown("**📚 Sample AI Decision Explanations:**")
    
    sample_explanations = [
        {
            'symbol': 'BTCUSDT',
            'decision': 'BUY',
            'confidence': 84.2,
            'model': 'Advanced Gradient Boosting',
            'signal_strength': 'Strong',
            'top_features': ['RSI = 28.4 (oversold)', 'EMA crossover signal', 'Volume spike detected'],
            'market_conditions': 'High volatility, bullish trend',
            'timestamp': datetime.now().isoformat()
        },
        {
            'symbol': 'ETHUSDT',
            'decision': 'HOLD',
            'confidence': 67.1,
            'model': 'LSTM Neural Network',
            'signal_strength': 'Moderate',
            'top_features': ['MACD neutral signal', 'Sideways price movement', 'Normal volume activity'],
            'market_conditions': 'Moderate volatility, sideways trend',
            'timestamp': (datetime.now() - timedelta(minutes=15)).isoformat()
        },
        {
            'symbol': 'ADAUSDT',
            'decision': 'SELL',
            'confidence': 91.7,
            'model': 'Transformer Model',
            'signal_strength': 'Strong',
            'top_features': ['RSI = 76.2 (overbought)', 'Resistance level test', 'High volume surge'],
            'market_conditions': 'Low volatility, bearish trend',
            'timestamp': (datetime.now() - timedelta(minutes=30)).isoformat()
        }
    ]
    
    for i, exp in enumerate(sample_explanations):
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                decision_emoji = _get_decision_emoji(exp['decision'])
                st.markdown(f"**{exp['symbol']}** - {decision_emoji} **{exp['decision']}**")
                st.caption(f"Model: {exp['model']} | {_format_timestamp(exp['timestamp'])}")
            
            with col2:
                st.metric("Confidence", f"{exp['confidence']}%")
                strength_color = _get_strength_color(exp['signal_strength'])
                st.markdown(f"<span style='background-color: {strength_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px'>{exp['signal_strength']} Signal</span>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("*Sample*")
            
            # Show features
            st.markdown("**🎯 Key Factors:**")
            for feature in exp['top_features']:
                st.markdown(f"• {feature}")
            
            st.markdown(f"**📊 Market:** {exp['market_conditions']}")
            st.markdown("---")

def _get_decision_color(decision: str) -> str:
    """Get color for decision type"""
    colors = {
        'BUY': '#00C851',
        'SELL': '#FF4444', 
        'HOLD': '#FFA500'
    }
    return colors.get(decision, '#666666')

def _get_decision_emoji(decision: str) -> str:
    """Get emoji for decision type"""
    emojis = {
        'BUY': '🟢',
        'SELL': '🔴',
        'HOLD': '🟡'
    }
    return emojis.get(decision, '⚪')

def _get_strength_color(strength: str) -> str:
    """Get color for signal strength"""
    colors = {
        'Strong': '#00C851',
        'Moderate': '#FFA500',
        'Weak': '#FF4444'
    }
    return colors.get(strength, '#666666')

def _format_timestamp(timestamp: str) -> str:
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now(dt.tzinfo)
        diff = now - dt
        
        if diff.total_seconds() < 60:
            return "Just now"
        elif diff.total_seconds() < 3600:
            return f"{int(diff.total_seconds() / 60)}m ago"
        elif diff.total_seconds() < 86400:
            return f"{int(diff.total_seconds() / 3600)}h ago"
        else:
            return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return timestamp

def _create_decision_timeline(explanations: List[Dict]):
    """Create timeline chart of decisions"""
    df_data = []
    for exp in explanations:
        df_data.append({
            'timestamp': exp['timestamp'],
            'decision': exp['decision'],
            'confidence': exp['confidence'],
            'model': exp['model']
        })
    
    df = pd.DataFrame(df_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = px.scatter(df, x='timestamp', y='confidence', 
                    color='decision', symbol='model',
                    color_discrete_map={'BUY': '#00C851', 'SELL': '#FF4444', 'HOLD': '#FFA500'},
                    title="Decision Timeline")
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def _create_decisions_table(explanations: List[Dict]):
    """Create table of decisions"""
    if not explanations:
        return
    
    table_data = []
    for exp in explanations:
        table_data.append({
            'Time': _format_timestamp(exp['timestamp']),
            'Symbol': exp['symbol'],
            'Decision': f"{_get_decision_emoji(exp['decision'])} {exp['decision']}",
            'Confidence': f"{exp['confidence']}%",
            'Model': exp['model'],
            'Signal': exp['signal_strength'],
            'Key Factor': exp['top_features'][0] if exp['top_features'] else 'N/A'
        })
    
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

def _create_confidence_chart(model_stats: Dict):
    """Create confidence distribution chart"""
    models = list(model_stats.keys())
    confidences = [stats['avg_confidence'] for stats in model_stats.values()]
    
    fig = go.Figure(data=[
        go.Bar(x=models, y=confidences, 
               marker_color=['#00C851' if c >= 80 else '#FFA500' if c >= 60 else '#FF4444' for c in confidences])
    ])
    
    fig.update_layout(
        title="Average Confidence by Model",
        xaxis_title="Model",
        yaxis_title="Average Confidence (%)",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def _create_decision_pie_chart(model_stats: Dict):
    """Create decision distribution pie chart"""
    all_decisions = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    
    for stats in model_stats.values():
        for decision, count in stats['decision_distribution'].items():
            all_decisions[decision] += count
    
    fig = go.Figure(data=[
        go.Pie(labels=list(all_decisions.keys()), 
               values=list(all_decisions.values()),
               marker_colors=['#00C851', '#FF4444', '#FFA500'])
    ])
    
    fig.update_layout(title="Overall Decision Distribution", height=400)
    st.plotly_chart(fig, use_container_width=True)