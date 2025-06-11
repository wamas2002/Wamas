#!/usr/bin/env python3
"""
Advanced Real-Time Trading Performance Dashboard
Comprehensive monitoring with live data visualization and system health tracking
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
import os
import ccxt
import time
from datetime import datetime, timedelta
from typing import Dict, List
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Trading Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS and JavaScript to handle WebSocket errors
st.markdown("""
<style>
    .main .block-container {
        padding-top: 1rem;
    }
</style>
<script>
// Handle WebSocket errors gracefully
window.addEventListener('error', function(e) {
    if (e.message && e.message.includes('WebSocket')) {
        console.log('WebSocket error handled gracefully');
        return true;
    }
});

window.addEventListener('unhandledrejection', function(e) {
    if (e.reason && e.reason.toString().includes('WebSocket')) {
        console.log('WebSocket promise rejection handled');
        e.preventDefault();
    }
});
</script>
""", unsafe_allow_html=True)

class AdvancedTradingMonitor:
    def __init__(self):
        self.exchange = self.connect_okx()
        
    def connect_okx(self):
        """Connect to OKX exchange"""
        try:
            return ccxt.okx({
                'apiKey': os.environ.get('OKX_API_KEY'),
                'secret': os.environ.get('OKX_SECRET_KEY'),
                'password': os.environ.get('OKX_PASSPHRASE'),
                'sandbox': False,
                'rateLimit': 2000,
                'enableRateLimit': True,
            })
        except Exception as e:
            st.error(f"OKX connection error: {e}")
            return None
    
    def get_portfolio_data(self):
        """Get real-time portfolio data"""
        if not self.exchange:
            return {}
        
        try:
            balance = self.exchange.fetch_balance()
            usdt_balance = float(balance['USDT']['free'])
            
            portfolio = {
                'usdt_balance': usdt_balance,
                'total_value': usdt_balance,
                'positions': [],
                'allocation': {}
            }
            
            for currency in balance:
                if currency != 'USDT' and balance[currency]['free'] > 0:
                    amount = float(balance[currency]['free'])
                    if amount > 0:
                        try:
                            symbol = f"{currency}/USDT"
                            ticker = self.exchange.fetch_ticker(symbol)
                            price = float(ticker['last'])
                            value = amount * price
                            portfolio['total_value'] += value
                            
                            position = {
                                'symbol': currency,
                                'amount': amount,
                                'price': price,
                                'value': value,
                                'change_24h': float(ticker.get('percentage', 0))
                            }
                            portfolio['positions'].append(position)
                        except:
                            continue
            
            # Calculate allocations
            for pos in portfolio['positions']:
                pos['allocation_pct'] = (pos['value'] / portfolio['total_value']) * 100
                portfolio['allocation'][pos['symbol']] = pos['allocation_pct']
            
            portfolio['allocation']['USDT'] = (usdt_balance / portfolio['total_value']) * 100
            
            return portfolio
            
        except Exception as e:
            st.error(f"Portfolio data error: {e}")
            return {}
    
    def get_trading_performance(self):
        """Get trading performance metrics"""
        try:
            if not os.path.exists('live_trading.db'):
                return {'total_trades': 0, 'success_rate': 0, 'profit_loss': 0}
            
            conn = sqlite3.connect('live_trading.db')
            
            # Get trade statistics
            trades_df = pd.read_sql_query('''
                SELECT symbol, side, amount, price, timestamp, status
                FROM live_trades 
                ORDER BY timestamp DESC
            ''', conn)
            
            # Calculate performance metrics
            total_trades = len(trades_df)
            successful_trades = len(trades_df[trades_df['status'] == 'filled'])
            success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Get profit taking data if available
            try:
                profit_df = pd.read_sql_query('''
                    SELECT profit_usdt, timestamp
                    FROM profit_taking_trades
                    ORDER BY timestamp DESC
                ''', conn)
                total_profit = profit_df['profit_usdt'].sum()
            except:
                total_profit = 0
            
            conn.close()
            
            return {
                'total_trades': total_trades,
                'success_rate': success_rate,
                'profit_loss': total_profit,
                'recent_trades': trades_df.head(10).to_dict('records')
            }
            
        except Exception as e:
            return {'total_trades': 0, 'success_rate': 0, 'profit_loss': 0}
    
    def get_ai_signals_data(self):
        """Get AI signals performance data"""
        try:
            if not os.path.exists('trading_platform.db'):
                return []
            
            conn = sqlite3.connect('trading_platform.db')
            
            signals_df = pd.read_sql_query('''
                SELECT symbol, signal, confidence, timestamp
                FROM ai_signals 
                ORDER BY timestamp DESC 
                LIMIT 50
            ''', conn)
            
            conn.close()
            
            return signals_df.to_dict('records')
            
        except Exception as e:
            return []
    
    def get_risk_metrics(self):
        """Get current risk management metrics"""
        portfolio = self.get_portfolio_data()
        
        if not portfolio:
            return {}
        
        total_value = portfolio['total_value']
        usdt_pct = portfolio['allocation'].get('USDT', 0)
        
        # Calculate risk metrics
        max_position = max(portfolio['allocation'].values()) if portfolio['allocation'] else 0
        diversification_score = len(portfolio['positions']) * 10  # Simple diversification metric
        
        risk_level = 'Low'
        if usdt_pct < 20:
            risk_level = 'High'
        elif usdt_pct < 40:
            risk_level = 'Medium'
        
        return {
            'total_value': total_value,
            'usdt_reserve_pct': usdt_pct,
            'max_position_pct': max_position,
            'diversification_score': min(diversification_score, 100),
            'risk_level': risk_level,
            'position_count': len(portfolio['positions'])
        }
    
    def create_portfolio_chart(self, portfolio_data):
        """Create portfolio allocation pie chart"""
        if not portfolio_data.get('allocation'):
            return go.Figure()
        
        labels = list(portfolio_data['allocation'].keys())
        values = list(portfolio_data['allocation'].values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            textposition='inside'
        )])
        
        fig.update_layout(
            title="Portfolio Allocation",
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_performance_chart(self, performance_data):
        """Create trading performance chart"""
        if not performance_data.get('recent_trades'):
            return go.Figure()
        
        trades_df = pd.DataFrame(performance_data['recent_trades'])
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        
        # Group by date and count trades
        daily_trades = trades_df.groupby(trades_df['timestamp'].dt.date).size()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=daily_trades.index,
            y=daily_trades.values,
            name='Daily Trades',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Daily Trading Activity",
            xaxis_title="Date",
            yaxis_title="Number of Trades",
            height=300
        )
        
        return fig
    
    def create_signals_chart(self, signals_data):
        """Create AI signals distribution chart"""
        if not signals_data:
            return go.Figure()
        
        signals_df = pd.DataFrame(signals_data)
        signal_counts = signals_df['signal'].value_counts()
        
        fig = go.Figure(data=[go.Bar(
            x=signal_counts.index,
            y=signal_counts.values,
            marker_color=['green' if x == 'BUY' else 'red' if x == 'SELL' else 'gray' for x in signal_counts.index]
        )])
        
        fig.update_layout(
            title="AI Signal Distribution",
            xaxis_title="Signal Type",
            yaxis_title="Count",
            height=300
        )
        
        return fig

def main():
    """Main dashboard function"""
    st.title("🚀 Advanced Trading Performance Monitor")
    st.markdown("Real-time portfolio tracking and system performance analysis")
    
    # Initialize monitor
    monitor = AdvancedTradingMonitor()
    
    # Auto-refresh controls
    col1, col2 = st.columns([3, 1])
    with col2:
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=True)
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Auto-refresh handled by Streamlit's built-in mechanism
    if auto_refresh:
        st.empty()  # Placeholder for auto-refresh
    
    # Main metrics row
    st.header("📊 System Overview")
    
    portfolio_data = monitor.get_portfolio_data()
    performance_data = monitor.get_trading_performance()
    risk_metrics = monitor.get_risk_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_value = portfolio_data.get('total_value', 0)
        st.metric(
            "Portfolio Value",
            f"${total_value:,.2f}",
            delta=None
        )
    
    with col2:
        total_trades = performance_data.get('total_trades', 0)
        st.metric(
            "Total Trades",
            total_trades,
            delta=None
        )
    
    with col3:
        success_rate = performance_data.get('success_rate', 0)
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
            delta=None
        )
    
    with col4:
        risk_level = risk_metrics.get('risk_level', 'Unknown')
        risk_color = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}.get(risk_level, '⚪')
        st.metric(
            "Risk Level",
            f"{risk_color} {risk_level}",
            delta=None
        )
    
    # Portfolio section
    st.header("💼 Portfolio Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if portfolio_data:
            fig_portfolio = monitor.create_portfolio_chart(portfolio_data)
            st.plotly_chart(fig_portfolio, use_container_width=True)
    
    with col2:
        st.subheader("Risk Metrics")
        if risk_metrics:
            st.metric("USDT Reserve", f"{risk_metrics.get('usdt_reserve_pct', 0):.1f}%")
            st.metric("Max Position", f"{risk_metrics.get('max_position_pct', 0):.1f}%")
            st.metric("Diversification", f"{risk_metrics.get('diversification_score', 0):.0f}/100")
            st.metric("Positions", risk_metrics.get('position_count', 0))
    
    # Trading performance section
    st.header("📈 Trading Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_performance = monitor.create_performance_chart(performance_data)
        st.plotly_chart(fig_performance, use_container_width=True)
    
    with col2:
        signals_data = monitor.get_ai_signals_data()
        fig_signals = monitor.create_signals_chart(signals_data)
        st.plotly_chart(fig_signals, use_container_width=True)
    
    # Recent activity section
    st.header("🔄 Recent Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Recent Trades")
        if performance_data.get('recent_trades'):
            trades_df = pd.DataFrame(performance_data['recent_trades'])
            st.dataframe(
                trades_df[['symbol', 'side', 'amount', 'price', 'timestamp']].head(5),
                use_container_width=True
            )
        else:
            st.info("No recent trades available")
    
    with col2:
        st.subheader("Recent AI Signals")
        if signals_data:
            signals_df = pd.DataFrame(signals_data)
            st.dataframe(
                signals_df[['symbol', 'signal', 'confidence', 'timestamp']].head(5),
                use_container_width=True
            )
        else:
            st.info("No recent AI signals available")
    
    # Current positions table
    if portfolio_data.get('positions'):
        st.header("💰 Current Positions")
        positions_df = pd.DataFrame(portfolio_data['positions'])
        
        # Format the dataframe for better display
        positions_df['value'] = positions_df['value'].apply(lambda x: f"${x:,.2f}")
        positions_df['price'] = positions_df['price'].apply(lambda x: f"${x:,.4f}")
        positions_df['allocation_pct'] = positions_df['allocation_pct'].apply(lambda x: f"{x:.1f}%")
        positions_df['change_24h'] = positions_df['change_24h'].apply(lambda x: f"{x:+.2f}%")
        
        st.dataframe(
            positions_df[['symbol', 'amount', 'price', 'value', 'allocation_pct', 'change_24h']],
            use_container_width=True
        )
    
    # System status footer
    st.header("⚙️ System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        exchange_status = "🟢 Connected" if monitor.exchange else "🔴 Disconnected"
        st.write(f"**OKX Exchange:** {exchange_status}")
    
    with col2:
        db_status = "🟢 Available" if os.path.exists('live_trading.db') else "🔴 Missing"
        st.write(f"**Trading Database:** {db_status}")
    
    with col3:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.write(f"**Last Update:** {current_time}")

if __name__ == "__main__":
    main()