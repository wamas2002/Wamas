#!/usr/bin/env python3
"""
Trading Dashboard - Streamlit App
Real-time portfolio monitoring and trading system status
"""

import streamlit as st
import pandas as pd
import sqlite3
import time
import os
import ccxt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Advanced Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TradingDashboard:
    def __init__(self):
        self.exchange = None
        self.initialize_exchange()
    
    def initialize_exchange(self):
        """Initialize OKX exchange connection"""
        try:
            self.exchange = ccxt.okx({
                'apiKey': os.getenv('OKX_API_KEY', ''),
                'secret': os.getenv('OKX_SECRET_KEY', ''),
                'password': os.getenv('OKX_PASSPHRASE', ''),
                'sandbox': False,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
        except Exception as e:
            st.error(f"Exchange connection failed: {e}")
    
    def get_portfolio_data(self):
        """Get current portfolio data"""
        try:
            if not self.exchange:
                return None
            
            balance = self.exchange.fetch_balance()
            positions = self.exchange.fetch_positions()
            
            total_balance = balance.get('USDT', {}).get('total', 0)
            available_balance = balance.get('USDT', {}).get('free', 0)
            
            active_positions = [p for p in positions if float(p.get('size', 0)) > 0]
            total_pnl = sum(float(p.get('unrealizedPnl', 0)) for p in active_positions)
            
            return {
                'total_balance': total_balance,
                'available_balance': available_balance,
                'active_positions': len(active_positions),
                'total_pnl': total_pnl,
                'positions': active_positions
            }
        except Exception as e:
            st.error(f"Portfolio data error: {e}")
            return None
    
    def get_system_status(self):
        """Get trading system status"""
        systems = [
            'Advanced Position Manager',
            'Advanced Signal Executor', 
            'Live Position Monitor',
            'Intelligent Profit Optimizer',
            'Live Under $50 Futures Trading',
            'Comprehensive System Monitor'
        ]
        
        # Check database files to estimate system activity
        active_systems = 0
        for db_file in ['live_under50_futures.db', 'advanced_positions.db', 'live_position_monitor.db']:
            if os.path.exists(db_file):
                try:
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    if cursor.fetchall():
                        active_systems += 1
                    conn.close()
                except:
                    pass
        
        return {
            'total_systems': len(systems),
            'active_systems': active_systems,
            'efficiency': (active_systems / len(systems)) * 100,
            'systems': systems
        }

# Initialize dashboard
dashboard = TradingDashboard()

# Main dashboard
st.title("🚀 Advanced Trading Dashboard")
st.markdown("---")

# Auto-refresh every 30 seconds
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()

if st.button("🔄 Refresh Data") or (time.time() - st.session_state.last_refresh) > 30:
    st.session_state.last_refresh = time.time()
    st.rerun()

# Portfolio Overview
col1, col2, col3, col4 = st.columns(4)

portfolio_data = dashboard.get_portfolio_data()
system_status = dashboard.get_system_status()

if portfolio_data:
    with col1:
        st.metric(
            "Total Balance", 
            f"${portfolio_data['total_balance']:.2f}",
            delta=f"${portfolio_data['total_pnl']:.2f}"
        )
    
    with col2:
        st.metric(
            "Available Balance", 
            f"${portfolio_data['available_balance']:.2f}"
        )
    
    with col3:
        st.metric(
            "Active Positions", 
            portfolio_data['active_positions']
        )
    
    with col4:
        pnl_color = "normal" if portfolio_data['total_pnl'] >= 0 else "inverse"
        st.metric(
            "Total P&L", 
            f"${portfolio_data['total_pnl']:.2f}",
            delta=f"{(portfolio_data['total_pnl']/portfolio_data['total_balance']*100):.2f}%"
        )

st.markdown("---")

# System Status
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Active Positions")
    
    if portfolio_data and portfolio_data['positions']:
        positions_df = []
        for pos in portfolio_data['positions']:
            if float(pos.get('size', 0)) > 0:
                positions_df.append({
                    'Symbol': pos.get('symbol', 'N/A'),
                    'Side': pos.get('side', 'N/A'),
                    'Size': float(pos.get('size', 0)),
                    'Entry Price': float(pos.get('entryPrice', 0)),
                    'Mark Price': float(pos.get('markPrice', 0)),
                    'P&L (USD)': float(pos.get('unrealizedPnl', 0)),
                    'P&L (%)': float(pos.get('percentage', 0))
                })
        
        if positions_df:
            df = pd.DataFrame(positions_df)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No active positions")
    else:
        st.info("No position data available")

with col2:
    st.subheader("⚡ System Status")
    
    if system_status:
        st.metric(
            "System Efficiency", 
            f"{system_status['efficiency']:.1f}%",
            delta=f"{system_status['active_systems']}/{system_status['total_systems']} Active"
        )
        
        st.markdown("**Active Systems:**")
        for i, system in enumerate(system_status['systems']):
            status = "🟢" if i < system_status['active_systems'] else "🔴"
            st.markdown(f"{status} {system}")

st.markdown("---")

# Trading Performance Chart
st.subheader("📈 Portfolio Performance")

# Get recent portfolio history from database
try:
    conn = sqlite3.connect('unified_dashboard.db')
    query = """
        SELECT timestamp, total_balance, total_pnl, active_positions 
        FROM portfolio_snapshots 
        ORDER BY timestamp DESC 
        LIMIT 100
    """
    df_history = pd.read_sql_query(query, conn)
    conn.close()
    
    if not df_history.empty:
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Portfolio Balance', 'P&L'),
            vertical_spacing=0.1
        )
        
        # Portfolio balance chart
        fig.add_trace(
            go.Scatter(
                x=df_history['timestamp'], 
                y=df_history['total_balance'],
                name='Balance',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # P&L chart
        colors = ['green' if pnl >= 0 else 'red' for pnl in df_history['total_pnl']]
        fig.add_trace(
            go.Scatter(
                x=df_history['timestamp'], 
                y=df_history['total_pnl'],
                name='P&L',
                line=dict(color='orange'),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No historical data available yet")
        
except Exception as e:
    st.warning("Historical data not available")

# Trading Activity Log
st.subheader("📋 Recent Trading Activity")

try:
    # Check multiple databases for recent activity
    activity_data = []
    
    # Check live positions database
    if os.path.exists('live_position_monitor.db'):
        conn = sqlite3.connect('live_position_monitor.db')
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, action, symbol, details 
                FROM position_updates 
                ORDER BY timestamp DESC 
                LIMIT 10
            """)
            results = cursor.fetchall()
            for row in results:
                activity_data.append({
                    'Time': row[0],
                    'Action': row[1], 
                    'Symbol': row[2],
                    'Details': row[3]
                })
        except:
            pass
        finally:
            conn.close()
    
    if activity_data:
        activity_df = pd.DataFrame(activity_data)
        st.dataframe(activity_df, use_container_width=True)
    else:
        st.info("No recent trading activity")
        
except Exception as e:
    st.info("Trading activity log not available")

# Footer
st.markdown("---")
st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.markdown("**Status:** All trading systems operational")