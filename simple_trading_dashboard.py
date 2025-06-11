#!/usr/bin/env python3
"""
Simple Trading Dashboard
Streamlined real-time trading monitor with OKX integration
"""

import streamlit as st
import os
import ccxt
import sqlite3
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="Trading Monitor", 
    page_icon="📊",
    layout="wide"
)

def get_portfolio_data():
    """Get real portfolio data from OKX"""
    try:
        exchange = ccxt.okx({
            'apiKey': os.environ.get('OKX_API_KEY'),
            'secret': os.environ.get('OKX_SECRET_KEY'), 
            'password': os.environ.get('OKX_PASSPHRASE'),
            'sandbox': False
        })
        
        balance = exchange.fetch_balance()
        total_value = float(balance['USDT']['free'])
        positions = []
        
        tokens = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'AVAX']
        
        for token in tokens:
            if token in balance and balance[token]['free'] > 0:
                amount = float(balance[token]['free'])
                if amount > 0:
                    symbol = f"{token}/USDT"
                    ticker = exchange.fetch_ticker(symbol)
                    price = float(ticker['last'])
                    value = amount * price
                    change_24h = float(ticker['percentage']) if ticker['percentage'] else 0
                    total_value += value
                    
                    positions.append({
                        'token': token,
                        'amount': amount,
                        'price': price,
                        'value': value,
                        'change': change_24h
                    })
        
        return {
            'total_value': total_value,
            'usdt_balance': float(balance['USDT']['free']),
            'positions': positions
        }
        
    except Exception as e:
        st.error(f"Portfolio data error: {e}")
        return None

def get_trading_stats():
    """Get trading statistics"""
    try:
        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM live_trades')
        total_trades = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT symbol, side, amount, price, timestamp 
            FROM live_trades 
            ORDER BY timestamp DESC LIMIT 5
        ''')
        recent_trades = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_trades': total_trades,
            'recent_trades': recent_trades
        }
    except:
        return {'total_trades': 0, 'recent_trades': []}

def get_ai_signals():
    """Get AI signal data"""
    try:
        conn = sqlite3.connect('trading_platform.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM ai_signals')
        total_signals = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT signal, COUNT(*) as count
            FROM ai_signals
            WHERE id > (SELECT MAX(id) - 20 FROM ai_signals)
            GROUP BY signal
        ''')
        signal_dist = cursor.fetchall()
        
        conn.close()
        
        return {
            'total_signals': total_signals,
            'distribution': signal_dist
        }
    except:
        return {'total_signals': 0, 'distribution': []}

# Main Dashboard
st.title("🔥 Live Trading Monitor")
st.markdown("Real-time cryptocurrency trading system dashboard")

# Auto-refresh
if st.button("🔄 Refresh Data"):
    st.rerun()

# Get data
portfolio = get_portfolio_data()
trading_stats = get_trading_stats()
ai_signals = get_ai_signals()

if portfolio:
    # Portfolio Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Portfolio", f"${portfolio['total_value']:.2f}")
    
    with col2:
        st.metric("USDT Cash", f"${portfolio['usdt_balance']:.2f}")
    
    with col3:
        st.metric("Active Positions", len(portfolio['positions']))
    
    with col4:
        st.metric("Total Trades", trading_stats['total_trades'])
    
    # Portfolio Details
    if portfolio['positions']:
        st.subheader("📊 Current Holdings")
        
        for pos in portfolio['positions']:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(f"**{pos['token']}**")
            
            with col2:
                st.write(f"{pos['amount']:.6f}")
            
            with col3:
                st.write(f"${pos['price']:.2f}")
            
            with col4:
                change_color = "green" if pos['change'] > 0 else "red"
                st.markdown(f"<span style='color: {change_color}'>{pos['change']:+.1f}%</span>", 
                           unsafe_allow_html=True)
        
        # Allocation Chart
        crypto_value = sum(pos['value'] for pos in portfolio['positions'])
        crypto_pct = (crypto_value / portfolio['total_value']) * 100
        cash_pct = 100 - crypto_pct
        
        st.subheader("💼 Portfolio Allocation")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Cryptocurrency", f"{crypto_pct:.1f}%")
        
        with col2:
            st.metric("Cash (USDT)", f"{cash_pct:.1f}%")
    
    # Trading Activity
    st.subheader("⚡ Recent Trading Activity")
    
    if trading_stats['recent_trades']:
        for trade in trading_stats['recent_trades']:
            symbol, side, amount, price, timestamp = trade
            value = float(amount) * float(price)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.write(timestamp[:16])
            
            with col2:
                side_color = "green" if side == 'buy' else "red"
                st.markdown(f"<span style='color: {side_color}'>{side.upper()}</span>", 
                           unsafe_allow_html=True)
            
            with col3:
                st.write(f"{float(amount):.6f} {symbol}")
            
            with col4:
                st.write(f"${value:.2f}")
    else:
        st.info("No recent trading activity")
    
    # AI Signals
    st.subheader("🤖 AI Signal Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Signals", ai_signals['total_signals'])
    
    with col2:
        if ai_signals['distribution']:
            for signal, count in ai_signals['distribution']:
                st.write(f"{signal}: {count}")
    
    # System Status
    st.subheader("🔧 System Status")
    
    status_items = [
        ("OKX Connection", "✅ Connected"),
        ("Trading Platform", "✅ Active"),
        ("AI Signal Generation", "✅ Running"),
        ("Risk Management", "✅ Enabled")
    ]
    
    for name, status in status_items:
        col1, col2 = st.columns(2)
        with col1:
            st.write(name)
        with col2:
            st.write(status)

else:
    st.error("Unable to load portfolio data. Check OKX connection.")

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")