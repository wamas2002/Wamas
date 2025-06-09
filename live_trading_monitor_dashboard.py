#!/usr/bin/env python3
"""
Live Trading Monitor Dashboard
Real-time monitoring and control interface for autonomous trading system
"""

import streamlit as st
import sqlite3
import os
import ccxt
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Live Trading Monitor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=30)
def get_okx_connection():
    """Get OKX exchange connection"""
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
        st.error(f"OKX connection failed: {e}")
        return None

@st.cache_data(ttl=10)
def get_portfolio_status():
    """Get current portfolio status"""
    exchange = get_okx_connection()
    if not exchange:
        return None
    
    try:
        balance = exchange.fetch_balance()
        usdt_balance = float(balance['USDT']['free'])
        
        portfolio_value = usdt_balance
        positions = []
        
        for currency in balance:
            if currency != 'USDT' and balance[currency]['free'] > 0:
                amount = float(balance[currency]['free'])
                if amount > 0:
                    try:
                        symbol = f"{currency}/USDT"
                        ticker = exchange.fetch_ticker(symbol)
                        price = float(ticker['last'])
                        value = amount * price
                        portfolio_value += value
                        positions.append({
                            'symbol': currency,
                            'amount': amount,
                            'price': price,
                            'value': value
                        })
                    except:
                        continue
        
        return {
            'usdt_balance': usdt_balance,
            'total_value': portfolio_value,
            'positions': positions
        }
    except Exception as e:
        st.error(f"Portfolio fetch error: {e}")
        return None

@st.cache_data(ttl=20)
def get_recent_trades():
    """Get recent trade history"""
    if not os.path.exists('live_trading.db'):
        return []
    
    try:
        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, side, amount, price, status, timestamp
            FROM live_trades 
            ORDER BY timestamp DESC LIMIT 20
        ''')
        
        trades = cursor.fetchall()
        conn.close()
        
        return [
            {
                'symbol': trade[0],
                'side': trade[1],
                'amount': float(trade[2]),
                'price': float(trade[3]),
                'value': float(trade[2]) * float(trade[3]),
                'timestamp': trade[5]
            }
            for trade in trades
        ]
    except Exception as e:
        st.error(f"Trade history error: {e}")
        return []

@st.cache_data(ttl=15)
def get_ai_signals():
    """Get recent AI signals"""
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, signal, confidence, timestamp 
            FROM ai_signals 
            ORDER BY id DESC LIMIT 10
        ''')
        
        signals = cursor.fetchall()
        conn.close()
        
        return [
            {
                'symbol': signal[0],
                'signal': signal[1],
                'confidence': float(signal[2]),
                'timestamp': signal[3]
            }
            for signal in signals
        ]
    except Exception as e:
        st.error(f"AI signals error: {e}")
        return []

@st.cache_data(ttl=60)
def get_market_data():
    """Get current market data for major cryptocurrencies"""
    exchange = get_okx_connection()
    if not exchange:
        return []
    
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT']
    market_data = []
    
    for symbol in symbols:
        try:
            ticker = exchange.fetch_ticker(symbol)
            market_data.append({
                'symbol': symbol.replace('/USDT', ''),
                'price': float(ticker['last']),
                'change_24h': float(ticker['percentage']) if ticker['percentage'] else 0,
                'volume_24h': float(ticker['quoteVolume']) if ticker['quoteVolume'] else 0
            })
        except:
            continue
    
    return market_data

def main():
    """Main dashboard function"""
    st.title("🚀 Live Trading Monitor Dashboard")
    st.markdown("Real-time monitoring of autonomous cryptocurrency trading system")
    
    # Sidebar controls
    st.sidebar.header("Trading Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=True)
    if auto_refresh:
        time.sleep(1)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()
    
    # Emergency stop (placeholder)
    st.sidebar.markdown("---")
    if st.sidebar.button("🛑 Emergency Stop", type="primary"):
        st.sidebar.warning("Emergency stop would halt trading system")
    
    # Main dashboard layout
    col1, col2, col3 = st.columns(3)
    
    # Portfolio Overview
    with col1:
        st.subheader("💰 Portfolio Status")
        portfolio = get_portfolio_status()
        
        if portfolio:
            st.metric(
                "Total Value",
                f"${portfolio['total_value']:.2f}",
                delta=None
            )
            st.metric(
                "USDT Balance",
                f"${portfolio['usdt_balance']:.2f}",
                delta=None
            )
            st.metric(
                "Active Positions",
                len(portfolio['positions']),
                delta=None
            )
        else:
            st.error("Portfolio data unavailable")
    
    # System Status
    with col2:
        st.subheader("⚡ System Status")
        
        # Check if live trading system is running
        recent_trades = get_recent_trades()
        recent_signals = get_ai_signals()
        
        if recent_trades:
            last_trade_time = datetime.fromisoformat(recent_trades[0]['timestamp'].replace('Z', ''))
            time_since_trade = datetime.now() - last_trade_time
            
            if time_since_trade.total_seconds() < 3600:  # Less than 1 hour
                st.success("🟢 Active Trading")
            else:
                st.warning("🟡 Monitoring")
        else:
            st.info("🔵 Initializing")
        
        st.metric(
            "Trades (24h)",
            len([t for t in recent_trades if 
                 datetime.fromisoformat(t['timestamp'].replace('Z', '')) > 
                 datetime.now() - timedelta(days=1)]),
            delta=None
        )
        
        st.metric(
            "AI Signals (1h)",
            len([s for s in recent_signals if 
                 datetime.fromisoformat(s['timestamp']) > 
                 datetime.now() - timedelta(hours=1)]),
            delta=None
        )
    
    # Market Overview
    with col3:
        st.subheader("📊 Market Overview")
        market_data = get_market_data()
        
        if market_data:
            for coin in market_data[:3]:
                delta_color = "normal" if coin['change_24h'] >= 0 else "inverse"
                st.metric(
                    coin['symbol'],
                    f"${coin['price']:,.2f}",
                    delta=f"{coin['change_24h']:+.2f}%",
                    delta_color=delta_color
                )
        else:
            st.error("Market data unavailable")
    
    # Recent Activity Section
    st.markdown("---")
    
    tab1, tab2, tab3 = st.tabs(["Recent Trades", "AI Signals", "Performance"])
    
    with tab1:
        st.subheader("📈 Recent Trade Executions")
        if recent_trades:
            trades_df = pd.DataFrame(recent_trades)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            trades_df = trades_df.sort_values('timestamp', ascending=False)
            
            # Color code by trade side
            def color_side(val):
                color = 'green' if val == 'buy' else 'red'
                return f'color: {color}'
            
            styled_df = trades_df.style.applymap(color_side, subset=['side'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No recent trades found")
    
    with tab2:
        st.subheader("🤖 AI Trading Signals")
        if recent_signals:
            signals_df = pd.DataFrame(recent_signals)
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            signals_df = signals_df.sort_values('timestamp', ascending=False)
            
            # Highlight high-confidence signals
            def highlight_confidence(val):
                if val >= 70:
                    return 'background-color: lightgreen'
                elif val >= 60:
                    return 'background-color: lightyellow'
                else:
                    return ''
            
            styled_signals = signals_df.style.applymap(
                highlight_confidence, subset=['confidence']
            )
            st.dataframe(styled_signals, use_container_width=True)
        else:
            st.info("No recent signals found")
    
    with tab3:
        st.subheader("📊 Performance Analytics")
        
        if recent_trades:
            # Calculate basic performance metrics
            total_trades = len(recent_trades)
            buy_trades = len([t for t in recent_trades if t['side'] == 'buy'])
            total_volume = sum([t['value'] for t in recent_trades])
            
            perf_col1, perf_col2, perf_col3 = st.columns(3)
            
            with perf_col1:
                st.metric("Total Trades", total_trades)
            with perf_col2:
                st.metric("Buy Ratio", f"{(buy_trades/total_trades)*100:.1f}%")
            with perf_col3:
                st.metric("Total Volume", f"${total_volume:.2f}")
            
            # Trade volume over time chart
            if len(recent_trades) > 1:
                trades_df = pd.DataFrame(recent_trades)
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                trades_df = trades_df.sort_values('timestamp')
                
                fig = px.line(
                    trades_df, 
                    x='timestamp', 
                    y='value',
                    title='Trade Volume Over Time',
                    labels={'value': 'Trade Value ($)', 'timestamp': 'Time'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available yet")
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
        f"**System:** Live Trading Active | "
        f"**Mode:** Autonomous"
    )

if __name__ == "__main__":
    main()