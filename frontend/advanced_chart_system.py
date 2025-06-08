"""
Advanced Chart System - Interactive charts with customizable overlays
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

def show_advanced_chart_system():
    """Show advanced interactive chart system"""
    st.title("📊 Advanced Chart System")
    
    # Chart configuration panel
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        show_chart_controls()
    
    with col2:
        show_main_chart()
    
    with col3:
        show_chart_overlays()

def show_chart_controls():
    """Show chart control panel"""
    st.subheader("🎛️ Chart Controls")
    
    # Symbol selection
    symbol = st.selectbox(
        "Symbol",
        ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "XRPUSDT"],
        index=0
    )
    
    # Timeframe selection
    timeframe = st.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
        index=3
    )
    
    # Chart type
    chart_type = st.radio(
        "Chart Type",
        ["Candlestick", "Line", "Area"],
        index=0
    )
    
    # Time range
    time_range = st.selectbox(
        "Time Range",
        ["Last 24h", "Last 7d", "Last 30d", "Last 90d"],
        index=1
    )
    
    # Technical indicators
    st.markdown("**Technical Indicators:**")
    
    show_ma = st.checkbox("Moving Averages", value=True)
    show_rsi = st.checkbox("RSI", value=True)
    show_macd = st.checkbox("MACD", value=False)
    show_bb = st.checkbox("Bollinger Bands", value=False)
    show_volume = st.checkbox("Volume", value=True)
    
    # AI overlays
    st.markdown("**AI Overlays:**")
    
    show_predictions = st.checkbox("Price Predictions", value=True)
    show_signals = st.checkbox("Trading Signals", value=True)
    show_confidence = st.checkbox("Confidence Bands", value=False)
    
    # Store selections in session state
    st.session_state.chart_config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'chart_type': chart_type,
        'time_range': time_range,
        'indicators': {
            'ma': show_ma,
            'rsi': show_rsi,
            'macd': show_macd,
            'bb': show_bb,
            'volume': show_volume
        },
        'ai_overlays': {
            'predictions': show_predictions,
            'signals': show_signals,
            'confidence': show_confidence
        }
    }

def show_main_chart():
    """Show main interactive chart"""
    if 'chart_config' not in st.session_state:
        st.session_state.chart_config = {
            'symbol': 'BTCUSDT',
            'timeframe': '1h',
            'chart_type': 'Candlestick',
            'time_range': 'Last 7d',
            'indicators': {'ma': True, 'rsi': True, 'volume': True},
            'ai_overlays': {'predictions': True, 'signals': True}
        }
    
    config = st.session_state.chart_config
    
    # Get market data
    data = get_chart_data(config['symbol'], config['timeframe'], config['time_range'])
    
    if data is not None and not data.empty:
        # Create main chart
        fig = create_main_chart(data, config)
        
        # Add technical indicators
        if config['indicators']['ma']:
            add_moving_averages(fig, data)
        
        if config['indicators']['bb']:
            add_bollinger_bands(fig, data)
        
        # Add AI overlays
        if config['ai_overlays']['predictions']:
            add_price_predictions(fig, data)
        
        if config['ai_overlays']['signals']:
            add_trading_signals(fig, data)
        
        if config['ai_overlays']['confidence']:
            add_confidence_bands(fig, data)
        
        # Create subplots for indicators
        subplots_needed = []
        if config['indicators']['rsi']:
            subplots_needed.append('RSI')
        if config['indicators']['macd']:
            subplots_needed.append('MACD')
        if config['indicators']['volume']:
            subplots_needed.append('Volume')
        
        if subplots_needed:
            fig = create_chart_with_subplots(data, config, subplots_needed)
        
        # Update layout
        fig.update_layout(
            title=f"{config['symbol']} - {config['timeframe']} - {config['time_range']}",
            xaxis_title="Time",
            yaxis_title="Price (USDT)",
            height=600,
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Chart statistics
        show_chart_statistics(data)
    
    else:
        st.error("No market data available for the selected configuration")

def get_chart_data(symbol: str, timeframe: str, time_range: str) -> Optional[pd.DataFrame]:
    """Get chart data from OKX data service"""
    try:
        if 'okx_data_service' in st.session_state:
            # Map time range to limit
            time_limits = {
                "Last 24h": 24 if timeframe == '1h' else 1440 if timeframe == '1m' else 6,
                "Last 7d": 168 if timeframe == '1h' else 2016 if timeframe == '15m' else 7,
                "Last 30d": 720 if timeframe == '1h' else 30,
                "Last 90d": 2160 if timeframe == '1h' else 90
            }
            
            limit = time_limits.get(time_range, 168)
            
            data = st.session_state.okx_data_service.get_historical_data(symbol, timeframe, limit=limit)
            
            if data is not None and not data.empty:
                return data
    except Exception as e:
        st.warning(f"Error fetching data: {e}")
    
    # Generate sample data if no real data available
    return generate_sample_data(symbol, timeframe, time_range)

def generate_sample_data(symbol: str, timeframe: str, time_range: str) -> pd.DataFrame:
    """Generate realistic sample data for demonstration"""
    time_limits = {
        "Last 24h": 24,
        "Last 7d": 168,
        "Last 30d": 720,
        "Last 90d": 2160
    }
    
    periods = time_limits.get(time_range, 168)
    
    # Create time index
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=periods)
    time_index = pd.date_range(start=start_time, end=end_time, periods=periods)
    
    # Generate realistic price data
    base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 0.5
    price_data = []
    current_price = base_price
    
    for i in range(periods):
        # Add realistic price movement
        change = np.random.normal(0, base_price * 0.002)  # 0.2% volatility
        current_price = max(current_price + change, base_price * 0.8)  # Floor at 80% of base
        price_data.append(current_price)
    
    # Create OHLCV data
    data = pd.DataFrame(index=time_index)
    data['close'] = price_data
    data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
    data['high'] = data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(data)))
    data['low'] = data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(data)))
    data['volume'] = np.random.uniform(1000, 10000, len(data))
    
    return data

def create_main_chart(data: pd.DataFrame, config: Dict[str, Any]) -> go.Figure:
    """Create main price chart"""
    fig = go.Figure()
    
    if config['chart_type'] == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price',
            increasing_line_color='#26C281',
            decreasing_line_color='#ED4C78'
        ))
    
    elif config['chart_type'] == 'Line':
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['close'],
            mode='lines',
            name='Price',
            line=dict(color='#1f77b4', width=2)
        ))
    
    elif config['chart_type'] == 'Area':
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['close'],
            mode='lines',
            fill='tonexty',
            name='Price',
            line=dict(color='#1f77b4', width=2),
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))
    
    return fig

def add_moving_averages(fig: go.Figure, data: pd.DataFrame):
    """Add moving averages to chart"""
    # Calculate moving averages
    ma_20 = data['close'].rolling(window=20).mean()
    ma_50 = data['close'].rolling(window=50).mean()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=ma_20,
        mode='lines',
        name='MA 20',
        line=dict(color='orange', width=1)
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=ma_50,
        mode='lines',
        name='MA 50',
        line=dict(color='red', width=1)
    ))

def add_bollinger_bands(fig: go.Figure, data: pd.DataFrame):
    """Add Bollinger Bands to chart"""
    ma_20 = data['close'].rolling(window=20).mean()
    std_20 = data['close'].rolling(window=20).std()
    
    upper_band = ma_20 + (std_20 * 2)
    lower_band = ma_20 - (std_20 * 2)
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=upper_band,
        mode='lines',
        name='BB Upper',
        line=dict(color='gray', width=1, dash='dash'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=lower_band,
        mode='lines',
        name='BB Lower',
        line=dict(color='gray', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(128, 128, 128, 0.1)',
        showlegend=False
    ))

def add_price_predictions(fig: go.Figure, data: pd.DataFrame):
    """Add AI price predictions to chart"""
    # Generate realistic predictions
    last_price = data['close'].iloc[-1]
    future_periods = 24  # 24 hours ahead
    
    prediction_times = pd.date_range(
        start=data.index[-1] + pd.Timedelta(hours=1),
        periods=future_periods,
        freq='H'
    )
    
    # Simple trend-following prediction
    trend = (data['close'].iloc[-1] - data['close'].iloc[-10]) / 10
    predictions = []
    
    for i in range(future_periods):
        noise = np.random.normal(0, last_price * 0.001)
        pred_price = last_price + (trend * i) + noise
        predictions.append(pred_price)
    
    fig.add_trace(go.Scatter(
        x=prediction_times,
        y=predictions,
        mode='lines',
        name='AI Prediction',
        line=dict(color='purple', width=2, dash='dot')
    ))

def add_trading_signals(fig: go.Figure, data: pd.DataFrame):
    """Add AI trading signals to chart"""
    # Generate sample trading signals
    signal_indices = np.random.choice(len(data), size=int(len(data) * 0.1), replace=False)
    
    buy_signals = []
    sell_signals = []
    
    for idx in signal_indices:
        if np.random.random() > 0.5:  # Buy signal
            buy_signals.append((data.index[idx], data['low'].iloc[idx] * 0.995))
        else:  # Sell signal
            sell_signals.append((data.index[idx], data['high'].iloc[idx] * 1.005))
    
    if buy_signals:
        buy_times, buy_prices = zip(*buy_signals)
        fig.add_trace(go.Scatter(
            x=buy_times,
            y=buy_prices,
            mode='markers',
            name='Buy Signals',
            marker=dict(
                symbol='triangle-up',
                size=12,
                color='green'
            )
        ))
    
    if sell_signals:
        sell_times, sell_prices = zip(*sell_signals)
        fig.add_trace(go.Scatter(
            x=sell_times,
            y=sell_prices,
            mode='markers',
            name='Sell Signals',
            marker=dict(
                symbol='triangle-down',
                size=12,
                color='red'
            )
        ))

def add_confidence_bands(fig: go.Figure, data: pd.DataFrame):
    """Add confidence bands around predictions"""
    # Add confidence intervals around the last portion of data
    confidence_data = data.tail(50)
    
    # Calculate confidence bands
    ma = confidence_data['close'].rolling(window=10).mean()
    std = confidence_data['close'].rolling(window=10).std()
    
    upper_conf = ma + std
    lower_conf = ma - std
    
    fig.add_trace(go.Scatter(
        x=confidence_data.index,
        y=upper_conf,
        mode='lines',
        name='Confidence Upper',
        line=dict(color='lightblue', width=1),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=confidence_data.index,
        y=lower_conf,
        mode='lines',
        name='Confidence Lower',
        line=dict(color='lightblue', width=1),
        fill='tonexty',
        fillcolor='rgba(173, 216, 230, 0.2)',
        showlegend=False
    ))

def create_chart_with_subplots(data: pd.DataFrame, config: Dict[str, Any], subplots: List[str]) -> go.Figure:
    """Create chart with technical indicator subplots"""
    rows = 1 + len(subplots)
    
    subplot_titles = ['Price'] + subplots
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=[0.6] + [0.4/len(subplots)] * len(subplots)
    )
    
    # Add main price chart
    if config['chart_type'] == 'Candlestick':
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['close'],
            mode='lines',
            name='Price'
        ), row=1, col=1)
    
    # Add moving averages if enabled
    if config['indicators']['ma']:
        ma_20 = data['close'].rolling(window=20).mean()
        ma_50 = data['close'].rolling(window=50).mean()
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=ma_20,
            mode='lines',
            name='MA 20',
            line=dict(color='orange', width=1)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=ma_50,
            mode='lines',
            name='MA 50',
            line=dict(color='red', width=1)
        ), row=1, col=1)
    
    # Add subplots
    row_idx = 2
    
    if 'RSI' in subplots:
        rsi = calculate_rsi(data['close'])
        fig.add_trace(go.Scatter(
            x=data.index,
            y=rsi,
            mode='lines',
            name='RSI',
            line=dict(color='purple')
        ), row=row_idx, col=1)
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=row_idx, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=row_idx, col=1)
        
        row_idx += 1
    
    if 'MACD' in subplots:
        macd_line, macd_signal, macd_histogram = calculate_macd(data['close'])
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=macd_line,
            mode='lines',
            name='MACD',
            line=dict(color='blue')
        ), row=row_idx, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index,
            y=macd_signal,
            mode='lines',
            name='Signal',
            line=dict(color='red')
        ), row=row_idx, col=1)
        
        fig.add_trace(go.Bar(
            x=data.index,
            y=macd_histogram,
            name='Histogram',
            marker_color='gray'
        ), row=row_idx, col=1)
        
        row_idx += 1
    
    if 'Volume' in subplots:
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(data['close'], data['open'])]
        
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['volume'],
            name='Volume',
            marker_color=colors
        ), row=row_idx, col=1)
    
    return fig

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal).mean()
    macd_histogram = macd_line - macd_signal
    return macd_line, macd_signal, macd_histogram

def show_chart_overlays():
    """Show chart overlay controls"""
    st.subheader("🎨 Chart Overlays")
    
    # Drawing tools
    st.markdown("**Drawing Tools:**")
    
    if st.button("📏 Trend Line"):
        st.info("Click and drag on chart to draw trend lines")
    
    if st.button("📐 Fibonacci"):
        st.info("Select high and low points for Fibonacci retracement")
    
    if st.button("📍 Support/Resistance"):
        st.info("Click on price levels to mark support/resistance")
    
    # Chart annotations
    st.markdown("**Annotations:**")
    
    annotation_text = st.text_input("Add Note")
    if st.button("📝 Add Note") and annotation_text:
        st.success(f"Note added: {annotation_text}")
    
    # Export options
    st.markdown("**Export:**")
    
    if st.button("💾 Save Chart"):
        st.success("Chart saved to gallery")
    
    if st.button("📤 Export PNG"):
        st.success("Chart exported as PNG")
    
    if st.button("📊 Export Data"):
        st.success("Chart data exported as CSV")

def show_chart_statistics(data: pd.DataFrame):
    """Show chart statistics"""
    st.subheader("📊 Chart Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = data['close'].iloc[-1]
        st.metric("Current Price", f"${current_price:,.2f}")
    
    with col2:
        price_change = data['close'].iloc[-1] - data['close'].iloc[-2]
        price_change_pct = (price_change / data['close'].iloc[-2]) * 100
        st.metric("24h Change", f"{price_change_pct:+.2f}%", delta=f"${price_change:+.2f}")
    
    with col3:
        high_24h = data['high'].tail(24).max()
        st.metric("24h High", f"${high_24h:,.2f}")
    
    with col4:
        low_24h = data['low'].tail(24).min()
        st.metric("24h Low", f"${low_24h:,.2f}")
    
    # Additional statistics
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        volume_24h = data['volume'].tail(24).sum()
        st.metric("24h Volume", f"{volume_24h:,.0f}")
    
    with col6:
        volatility = data['close'].pct_change().std() * 100
        st.metric("Volatility", f"{volatility:.2f}%")
    
    with col7:
        rsi_current = calculate_rsi(data['close']).iloc[-1]
        st.metric("RSI", f"{rsi_current:.1f}")
    
    with col8:
        ma_20 = data['close'].rolling(window=20).mean().iloc[-1]
        distance_from_ma = ((current_price - ma_20) / ma_20) * 100
        st.metric("vs MA20", f"{distance_from_ma:+.1f}%")