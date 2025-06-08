"""
AI Performance Dashboard - Advanced model optimization monitoring
Displays adaptive model selection, performance tracking, and hybrid signals
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from utils.confidence_ui import display_confidence_badge, get_confidence_color

def show_ai_performance_dashboard():
    """AI Performance Summary dashboard with adaptive model optimization"""
    st.title("🤖 AI Performance Dashboard")
    st.markdown("**Dynamic Model Optimization & Performance Analytics**")
    
    # Performance overview cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Active Models", "5", delta="Adaptive")
    
    with col2:
        st.metric("Avg Win Rate", "67.3%", delta="+4.2%")
    
    with col3:
        st.metric("Hybrid Signals", "23", delta="+8")
    
    with col4:
        st.metric("Model Switches", "12", delta="+3")
    
    st.markdown("---")
    
    # Main dashboard sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Active Models", "📊 Performance Analytics", 
        "🔄 Hybrid Signals", "⚙️ Retraining Status"
    ])
    
    with tab1:
        show_active_models_panel()
    
    with tab2:
        show_performance_analytics_panel()
    
    with tab3:
        show_hybrid_signals_panel()
    
    with tab4:
        show_retraining_status_panel()

def show_active_models_panel():
    """Show currently active models per symbol with performance metrics"""
    st.subheader("🎯 Active Model Selection")
    
    if 'adaptive_model_selector' not in st.session_state:
        st.warning("Adaptive model selector not initialized")
        return
    
    # Get performance summary
    performance_summary = st.session_state.adaptive_model_selector.get_performance_summary()
    
    symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "XRPUSDT"]
    
    # Model performance table
    model_data = []
    for symbol in symbols:
        active_model = st.session_state.adaptive_model_selector.get_active_model(symbol)
        
        # Get model ranking
        rankings = st.session_state.adaptive_model_selector.get_model_ranking(symbol)
        top_3_models = rankings[:3] if rankings else []
        
        # Get performance data
        perf_data = performance_summary.get('performance_by_symbol', {}).get(symbol, {})
        
        model_data.append({
            'Symbol': symbol,
            'Active Model': active_model,
            'Win Rate': f"{perf_data.get('win_rate', 55.0):.1f}%",
            'Confidence': f"{perf_data.get('confidence', 65.0):.1f}%",
            'Trades': perf_data.get('total_trades', 0),
            'Score': f"{perf_data.get('score', 50.0):.1f}",
            'Top Models': ' → '.join([f"{m[0]} ({m[1]:.0f})" for m in top_3_models[:2]])
        })
    
    # Display as DataFrame
    df = pd.DataFrame(model_data)
    st.dataframe(df, use_container_width=True)
    
    # Model switching controls
    st.subheader("🔄 Manual Model Control")
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        selected_symbol = st.selectbox("Select Symbol", symbols)
    
    with col2:
        available_models = ['LSTM', 'Prophet', 'GradientBoost', 'Technical', 'Ensemble']
        selected_model = st.selectbox("Select Model", available_models)
    
    with col3:
        if st.button("Switch Model", type="primary"):
            success = st.session_state.adaptive_model_selector.set_active_model(selected_symbol, selected_model)
            if success:
                st.success(f"Switched {selected_symbol} to {selected_model}")
                st.rerun()
            else:
                st.error("Failed to switch model")
    
    # Force evaluation button
    if st.button("🔄 Force Model Evaluation"):
        with st.spinner("Evaluating all models..."):
            st.session_state.adaptive_model_selector.force_model_evaluation()
            st.success("Model evaluation completed")
            st.rerun()

def show_performance_analytics_panel():
    """Show detailed performance analytics and trends"""
    st.subheader("📊 Performance Analytics")
    
    if 'ai_performance_tracker' not in st.session_state:
        st.warning("AI performance tracker not initialized")
        return
    
    # Performance summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📈 Model Performance Comparison**")
        
        # Get performance data for comparison
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT"]
        models = ['LSTM', 'Prophet', 'GradientBoost', 'Technical', 'Ensemble']
        
        # Create performance comparison chart
        comparison_data = []
        for symbol in symbols:
            for model in models:
                perf_data = st.session_state.ai_performance_tracker.get_model_performance(symbol, model, 7)
                if perf_data:
                    comparison_data.append({
                        'Symbol': symbol,
                        'Model': model,
                        'Win Rate': perf_data.get('win_rate', 50),
                        'Confidence': perf_data.get('avg_confidence', 60),
                        'Total Trades': perf_data.get('total_trades', 0)
                    })
        
        if comparison_data:
            df_comp = pd.DataFrame(comparison_data)
            
            # Win rate comparison chart
            fig_win_rate = px.bar(
                df_comp, 
                x='Model', 
                y='Win Rate', 
                color='Symbol',
                title="Win Rate by Model & Symbol",
                height=300
            )
            st.plotly_chart(fig_win_rate, use_container_width=True)
    
    with col2:
        st.markdown("**🎯 Performance Metrics**")
        
        # Performance metrics table
        if comparison_data:
            # Calculate averages per model
            avg_metrics = df_comp.groupby('Model').agg({
                'Win Rate': 'mean',
                'Confidence': 'mean',
                'Total Trades': 'sum'
            }).round(1)
            
            # Add performance tags
            avg_metrics['Performance'] = avg_metrics['Win Rate'].apply(
                lambda x: "🟢 Excellent" if x > 65 
                else "🟡 Good" if x > 55 
                else "🔴 Needs Improvement"
            )
            
            st.dataframe(avg_metrics, use_container_width=True)
    
    # Detailed performance trends
    st.subheader("📈 Performance Trends")
    
    selected_symbol = st.selectbox("Select Symbol for Detailed Analysis", 
                                 ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT"])
    
    # Get detailed performance summary
    detailed_summary = st.session_state.ai_performance_tracker.get_performance_summary(selected_symbol)
    
    if detailed_summary and detailed_summary.get('summary'):
        summary_data = detailed_summary['summary']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance metrics
            for model_data in summary_data:
                model_name = model_data['model_name']
                win_rate = model_data['win_rate']
                confidence = model_data['avg_confidence']
                
                with st.container():
                    st.markdown(f"**{model_name}**")
                    display_confidence_badge(win_rate, f"Win Rate: {win_rate}%")
                    st.caption(f"Confidence: {confidence}% | Trades: {model_data['total_decisions']}")
        
        with col2:
            # PnL impact chart
            if summary_data:
                pnl_data = [(m['model_name'], m['total_pnl']) for m in summary_data]
                pnl_df = pd.DataFrame(pnl_data, columns=['Model', 'PnL Impact'])
                
                fig_pnl = px.bar(
                    pnl_df, 
                    x='Model', 
                    y='PnL Impact',
                    title=f"PnL Impact - {selected_symbol}",
                    color='PnL Impact',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_pnl, use_container_width=True)

def show_hybrid_signals_panel():
    """Show hybrid signal analysis and consensus data"""
    st.subheader("🔄 Hybrid Signal Engine")
    
    if 'hybrid_signal_engine' not in st.session_state:
        st.warning("Hybrid signal engine not initialized")
        return
    
    # Hybrid signal controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_symbol = st.selectbox("Select Symbol", 
                                     ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT"], 
                                     key="hybrid_symbol")
    
    with col2:
        if st.button("🎯 Generate Hybrid Signal", type="primary"):
            with st.spinner("Generating hybrid signal..."):
                hybrid_signal = st.session_state.hybrid_signal_engine.simulate_hybrid_signal(selected_symbol)
                st.session_state[f'hybrid_signal_{selected_symbol}'] = hybrid_signal
                st.success("Hybrid signal generated")
                st.rerun()
    
    with col3:
        if st.button("📊 Consensus Analysis"):
            consensus_data = st.session_state.hybrid_signal_engine.get_consensus_analysis(selected_symbol)
            st.session_state[f'consensus_{selected_symbol}'] = consensus_data
            st.rerun()
    
    # Display hybrid signal if available
    signal_key = f'hybrid_signal_{selected_symbol}'
    if signal_key in st.session_state:
        signal = st.session_state[signal_key]
        
        st.markdown("---")
        st.subheader(f"🎯 Latest Hybrid Signal - {selected_symbol}")
        
        # Signal overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            decision_color = get_confidence_color(signal.combined_confidence)
            st.markdown(f"**Decision:** :{decision_color}[{signal.final_decision}]")
        
        with col2:
            display_confidence_badge(signal.combined_confidence, f"Confidence: {signal.combined_confidence:.1f}%")
        
        with col3:
            st.metric("Consensus", f"{signal.consensus_strength:.1%}")
        
        with col4:
            st.metric("Models", len(signal.participating_models))
        
        # Model votes breakdown
        if signal.model_votes:
            st.markdown("**Model Contributions:**")
            
            votes_data = []
            for model_name, model_signal in signal.model_votes.items():
                votes_data.append({
                    'Model': model_name,
                    'Decision': model_signal.decision,
                    'Confidence': f"{model_signal.confidence:.1f}%",
                    'Predicted Return': f"{model_signal.predicted_return:.3f}",
                    'Latency': f"{model_signal.execution_latency:.3f}s"
                })
            
            votes_df = pd.DataFrame(votes_data)
            st.dataframe(votes_df, use_container_width=True)
        
        # Reasoning analysis
        if signal.reasoning:
            st.markdown("**Decision Analysis:**")
            reasoning = signal.reasoning
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'decision_factors' in reasoning:
                    st.markdown("**Key Factors:**")
                    for factor in reasoning['decision_factors']:
                        st.markdown(f"• {factor}")
            
            with col2:
                if 'voting_breakdown' in reasoning:
                    st.markdown("**Voting Breakdown:**")
                    voting = reasoning['voting_breakdown']
                    for decision, percentage in voting.items():
                        st.markdown(f"• {decision}: {percentage}%")
    
    # Consensus analysis display
    consensus_key = f'consensus_{selected_symbol}'
    if consensus_key in st.session_state:
        consensus = st.session_state[consensus_key]
        
        if 'error' not in consensus:
            st.markdown("---")
            st.subheader(f"📊 Consensus Analysis - {selected_symbol}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Recent Signals", consensus['recent_signals'])
                st.metric("Hybrid Signals", consensus['hybrid_signals'])
            
            with col2:
                st.metric("Avg Consensus", f"{consensus['avg_consensus_strength']}%")
                st.metric("Avg Confidence", f"{consensus['avg_confidence']:.1f}%")
            
            with col3:
                decisions = consensus['decision_distribution']
                st.markdown("**Decision Distribution:**")
                for decision, count in decisions.items():
                    st.markdown(f"• {decision}: {count}")

def show_retraining_status_panel():
    """Show retraining optimizer status and controls"""
    st.subheader("⚙️ Automated Retraining System")
    
    if 'retraining_optimizer' not in st.session_state:
        st.warning("Retraining optimizer not initialized")
        return
    
    # Get retraining status
    status = st.session_state.retraining_optimizer.get_retraining_status()
    
    # System status overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        monitoring_status = "🟢 Active" if status['monitoring_active'] else "🔴 Inactive"
        st.metric("Monitoring", monitoring_status)
    
    with col2:
        st.metric("Active Retraining", len(status['active_retraining']))
    
    with col3:
        st.metric("Queued Tasks", status['queued_tasks'])
    
    with col4:
        st.metric("Total Symbols", len(status.get('last_retraining_times', {})))
    
    st.markdown("---")
    
    # Retraining status table
    st.subheader("📊 Symbol Retraining Status")
    
    if status.get('last_retraining_times'):
        retraining_data = []
        
        for symbol, time_data in status['last_retraining_times'].items():
            next_check = status.get('next_scheduled_checks', {}).get(symbol, {})
            perf_status = status.get('performance_status', {}).get(symbol, {})
            
            # Determine status
            hours_ago = time_data['hours_ago']
            if symbol in status['active_retraining']:
                training_status = "🔄 Retraining"
            elif hours_ago > 168:  # 7 days
                training_status = "🔴 Overdue"
            elif hours_ago > 120:  # 5 days
                training_status = "🟡 Due Soon"
            else:
                training_status = "🟢 Current"
            
            retraining_data.append({
                'Symbol': symbol,
                'Status': training_status,
                'Last Retrained': f"{hours_ago:.1f}h ago",
                'Active Model': perf_status.get('active_model', 'N/A'),
                'Win Rate': f"{perf_status.get('current_win_rate', 0):.1f}%",
                'Performance Δ': f"{perf_status.get('performance_change', 0):+.1f}%",
                'Next Forced': f"{next_check.get('hours_until_forced', 0):.0f}h"
            })
        
        retraining_df = pd.DataFrame(retraining_data)
        st.dataframe(retraining_df, use_container_width=True)
    
    # Queue details
    if status['queued_tasks'] > 0:
        st.subheader("📋 Retraining Queue")
        
        queue_details = status.get('queue_details', [])
        for task in queue_details:
            with st.container():
                col1, col2, col3 = st.columns([2, 3, 1])
                
                with col1:
                    st.markdown(f"**{task['symbol']}**")
                
                with col2:
                    st.caption(f"{task['reason']} ({task['queued_minutes_ago']}m ago)")
                
                with col3:
                    priority_color = "🔴" if task['priority'] >= 3 else "🟡" if task['priority'] == 2 else "🟢"
                    st.caption(f"{priority_color} P{task['priority']}")
    
    # Manual controls
    st.markdown("---")
    st.subheader("🎛️ Manual Controls")
    
    col1, col2, col3 = st.columns([2, 3, 1])
    
    with col1:
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "XRPUSDT"]
        manual_symbol = st.selectbox("Select Symbol", symbols, key="manual_retrain")
    
    with col2:
        manual_reason = st.text_input("Reason for retraining", value="Manual trigger")
    
    with col3:
        if st.button("🚀 Trigger Retraining", type="primary"):
            success = st.session_state.retraining_optimizer.trigger_manual_retraining(
                manual_symbol, manual_reason
            )
            if success:
                st.success(f"Retraining queued for {manual_symbol}")
                st.rerun()
            else:
                st.error("Failed to queue retraining")
    
    # Optimization recommendations
    recommendations = st.session_state.retraining_optimizer.get_optimization_recommendations()
    
    if recommendations:
        st.markdown("---")
        st.subheader("💡 Optimization Recommendations")
        
        for rec in recommendations[:5]:  # Show top 5
            priority_color = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
            
            with st.container():
                col1, col2 = st.columns([1, 4])
                
                with col1:
                    st.markdown(f"**{rec['symbol']}**")
                    st.caption(f"{priority_color} {rec['priority'].title()}")
                
                with col2:
                    st.markdown(f"**{rec['type'].replace('_', ' ').title()}**")
                    st.caption(rec['message'])