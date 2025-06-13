"""
Autonomous Trading Monitor
Real-time monitoring dashboard for the autonomous trading engine
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json

st.set_page_config(
    page_title="Autonomous Trading Monitor",
    page_icon="🤖",
    layout="wide"
)

def get_autonomous_trades():
    """Get autonomous trading data"""
    try:
        with sqlite3.connect('enhanced_trading.db') as conn:
            df = pd.read_sql_query('''
                SELECT * FROM autonomous_trades 
                ORDER BY timestamp DESC 
                LIMIT 100
            ''', conn)
            return df
    except Exception as e:
        st.error(f"Error loading trades: {e}")
        return pd.DataFrame()

def get_active_positions():
    """Get active positions"""
    try:
        with sqlite3.connect('enhanced_trading.db') as conn:
            df = pd.read_sql_query('''
                SELECT * FROM active_positions 
                ORDER BY entry_timestamp DESC
            ''', conn)
            return df
    except Exception as e:
        st.error(f"Error loading positions: {e}")
        return pd.DataFrame()

def get_performance_metrics(trades_df):
    """Calculate performance metrics"""
    if trades_df.empty:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_pnl': 0,
            'total_pnl': 0,
            'best_trade': 0,
            'worst_trade': 0
        }
    
    closed_trades = trades_df[trades_df['status'] == 'CLOSED']
    
    if closed_trades.empty:
        return {
            'total_trades': len(trades_df),
            'win_rate': 0,
            'avg_pnl': 0,
            'total_pnl': 0,
            'best_trade': 0,
            'worst_trade': 0
        }
    
    winning_trades = closed_trades[closed_trades['pnl'] > 0]
    win_rate = len(winning_trades) / len(closed_trades) * 100 if len(closed_trades) > 0 else 0
    
    return {
        'total_trades': len(trades_df),
        'closed_trades': len(closed_trades),
        'win_rate': win_rate,
        'avg_pnl': closed_trades['pnl'].mean(),
        'total_pnl': closed_trades['pnl'].sum(),
        'best_trade': closed_trades['pnl'].max(),
        'worst_trade': closed_trades['pnl'].min()
    }

def main():
    st.title("🤖 Autonomous Trading Engine Monitor")
    
    # Auto-refresh
    placeholder = st.empty()
    
    while True:
        with placeholder.container():
            # Get data
            trades_df = get_autonomous_trades()
            positions_df = get_active_positions()
            metrics = get_performance_metrics(trades_df)
            
            # Header metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Trades", metrics['total_trades'])
            
            with col2:
                st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
            
            with col3:
                st.metric("Total P&L", f"{metrics['total_pnl']:.2f}%")
            
            with col4:
                st.metric("Active Positions", len(positions_df))
            
            with col5:
                st.metric("Avg P&L", f"{metrics['avg_pnl']:.2f}%")
            
            # Active Positions
            if not positions_df.empty:
                st.subheader("🎯 Active Positions")
                
                for _, pos in positions_df.iterrows():
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    
                    with col1:
                        st.write(f"**{pos['symbol']}** ({pos['side']})")
                    
                    with col2:
                        st.write(f"Entry: ${pos['entry_price']:.4f}")
                    
                    with col3:
                        pnl_color = "green" if pos['pnl'] > 0 else "red"
                        st.markdown(f"<span style='color: {pnl_color}'>P&L: {pos['pnl']:+.2f}%</span>", 
                                   unsafe_allow_html=True)
                    
                    with col4:
                        st.write(f"Confidence: {pos['confidence']:.1f}%")
            
            # Recent Trades
            if not trades_df.empty:
                st.subheader("📊 Recent Trades")
                
                # Trade history table
                display_trades = trades_df[['symbol', 'side', 'amount', 'price', 'confidence', 
                                          'ai_score', 'technical_score', 'pnl', 'status', 'timestamp']].head(10)
                
                # Color code P&L
                def color_pnl(val):
                    if pd.isna(val) or val == 0:
                        return ''
                    color = 'green' if val > 0 else 'red'
                    return f'color: {color}'
                
                styled_trades = display_trades.style.applymap(color_pnl, subset=['pnl'])
                st.dataframe(styled_trades, use_container_width=True)
                
                # P&L Chart
                if len(trades_df[trades_df['status'] == 'CLOSED']) > 0:
                    st.subheader("📈 P&L Performance")
                    
                    closed_trades = trades_df[trades_df['status'] == 'CLOSED'].copy()
                    closed_trades['timestamp'] = pd.to_datetime(closed_trades['timestamp'])
                    closed_trades = closed_trades.sort_values('timestamp')
                    closed_trades['cumulative_pnl'] = closed_trades['pnl'].cumsum()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=closed_trades['timestamp'],
                        y=closed_trades['cumulative_pnl'],
                        mode='lines+markers',
                        name='Cumulative P&L',
                        line=dict(color='green' if closed_trades['cumulative_pnl'].iloc[-1] > 0 else 'red')
                    ))
                    
                    fig.update_layout(
                        title="Cumulative P&L Over Time",
                        xaxis_title="Time",
                        yaxis_title="Cumulative P&L (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Signal Analysis
                st.subheader("🔍 Signal Analysis")
                
                if len(trades_df) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Confidence distribution
                        fig = px.histogram(trades_df, x='confidence', bins=10, 
                                         title="Signal Confidence Distribution")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # AI vs Technical scores
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=trades_df['technical_score'],
                            y=trades_df['ai_score'],
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=trades_df['confidence'],
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Confidence")
                            ),
                            text=trades_df['symbol'],
                            name='Signals'
                        ))
                        
                        fig.update_layout(
                            title="AI vs Technical Scores",
                            xaxis_title="Technical Score",
                            yaxis_title="AI Score"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
            
            # System Status
            st.subheader("⚙️ System Status")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success("🟢 Autonomous Engine: RUNNING")
            
            with col2:
                st.info("🔄 Market Scanner: ACTIVE")
            
            with col3:
                st.warning("⚡ Signal Threshold: 70%")
            
            # Configuration
            with st.expander("🛠️ Configuration"):
                st.write("**Trading Parameters:**")
                st.write("- Minimum Confidence: 70%")
                st.write("- Maximum Position Size: 25%")
                st.write("- Stop Loss: 8%")
                st.write("- Take Profit: 15%")
                st.write("- Scan Interval: 5 minutes")
                
                st.write("**Monitored Pairs:**")
                pairs = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT',
                        'SOL/USDT', 'DOGE/USDT', 'LINK/USDT', 'LTC/USDT', 'DOT/USDT',
                        'AVAX/USDT', 'UNI/USDT', 'ATOM/USDT', 'NEAR/USDT', 'FTM/USDT']
                st.write(", ".join(pairs))
            
            # Auto-refresh counter
            st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
            st.caption("Auto-refreshing every 30 seconds...")
        
        # Wait 30 seconds before refresh
        time.sleep(30)

if __name__ == "__main__":
    main()