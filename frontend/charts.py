import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import pandas_ta as ta

class ChartManager:
    """Manages chart creation and visualization"""
    
    def __init__(self):
        self.color_scheme = {
            'background': '#0e1117',
            'paper': '#262730',
            'text': '#fafafa',
            'grid': '#3d3d3d',
            'up': '#00ff88',
            'down': '#ff4b4b',
            'volume': '#1f77b4'
        }
    
    def create_candlestick_chart(self, data: pd.DataFrame, symbol: str, timeframe: str) -> go.Figure:
        """Create candlestick chart with volume"""
        try:
            if data.empty:
                return self._create_empty_chart("No data available")
            
            # Create subplots with secondary y-axis for volume
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(f'{symbol} - {timeframe}', 'Volume'),
                row_heights=[0.7, 0.3]
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='Price',
                    increasing_line_color=self.color_scheme['up'],
                    decreasing_line_color=self.color_scheme['down']
                ),
                row=1, col=1
            )
            
            # Add moving averages if enough data
            if len(data) >= 20:
                sma_20 = data['close'].rolling(20).mean()
                sma_50 = data['close'].rolling(50).mean()
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=sma_20,
                        mode='lines',
                        name='SMA 20',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
                
                if len(data) >= 50:
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=sma_50,
                            mode='lines',
                            name='SMA 50',
                            line=dict(color='purple', width=1)
                        ),
                        row=1, col=1
                    )
            
            # Volume chart
            colors = ['red' if data['close'].iloc[i] < data['open'].iloc[i] else 'green' 
                     for i in range(len(data))]
            
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['volume'],
                    name='Volume',
                    marker_color=colors,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} Price Chart",
                xaxis_rangeslider_visible=False,
                height=600,
                template='plotly_dark',
                showlegend=True,
                font=dict(color=self.color_scheme['text'])
            )
            
            # Update axes
            fig.update_xaxes(gridcolor=self.color_scheme['grid'])
            fig.update_yaxes(gridcolor=self.color_scheme['grid'])
            
            return fig
            
        except Exception as e:
            print(f"Error creating candlestick chart: {e}")
            return self._create_empty_chart(f"Error: {e}")
    
    def create_indicators_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create technical indicators chart"""
        try:
            if data.empty or len(data) < 20:
                return self._create_empty_chart("Insufficient data for indicators")
            
            # Create subplots for different indicators
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=('RSI', 'MACD', 'Bollinger Bands'),
                row_heights=[0.3, 0.3, 0.4]
            )
            
            # RSI
            try:
                rsi = ta.rsi(data['close'], length=14)
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=rsi,
                        mode='lines',
                        name='RSI',
                        line=dict(color='yellow')
                    ),
                    row=1, col=1
                )
                
                # RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="white", opacity=0.5, row=1, col=1)
                
            except Exception as e:
                print(f"Error calculating RSI: {e}")
            
            # MACD
            try:
                macd_data = ta.macd(data['close'], fast=12, slow=26, signal=9)
                if macd_data is not None and not macd_data.empty:
                    macd_col = macd_data.columns[0]
                    signal_col = macd_data.columns[2]
                    histogram_col = macd_data.columns[1]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=macd_data[macd_col],
                            mode='lines',
                            name='MACD',
                            line=dict(color='blue')
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=macd_data[signal_col],
                            mode='lines',
                            name='Signal',
                            line=dict(color='red')
                        ),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Bar(
                            x=data.index,
                            y=macd_data[histogram_col],
                            name='Histogram',
                            marker_color='gray',
                            opacity=0.6
                        ),
                        row=2, col=1
                    )
                    
            except Exception as e:
                print(f"Error calculating MACD: {e}")
            
            # Bollinger Bands
            try:
                bb_data = ta.bbands(data['close'], length=20, std=2)
                if bb_data is not None and not bb_data.empty:
                    upper_col = [col for col in bb_data.columns if 'BBU' in col][0]
                    middle_col = [col for col in bb_data.columns if 'BBM' in col][0]
                    lower_col = [col for col in bb_data.columns if 'BBL' in col][0]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=bb_data[upper_col],
                            mode='lines',
                            name='BB Upper',
                            line=dict(color='red', width=1)
                        ),
                        row=3, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=bb_data[middle_col],
                            mode='lines',
                            name='BB Middle',
                            line=dict(color='yellow', width=1)
                        ),
                        row=3, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=bb_data[lower_col],
                            mode='lines',
                            name='BB Lower',
                            line=dict(color='green', width=1),
                            fill='tonexty',
                            fillcolor='rgba(0,255,136,0.1)'
                        ),
                        row=3, col=1
                    )
                    
                    # Add price line
                    fig.add_trace(
                        go.Scatter(
                            x=data.index,
                            y=data['close'],
                            mode='lines',
                            name='Close Price',
                            line=dict(color='white', width=2)
                        ),
                        row=3, col=1
                    )
                    
            except Exception as e:
                print(f"Error calculating Bollinger Bands: {e}")
            
            # Update layout
            fig.update_layout(
                title="Technical Indicators",
                height=800,
                template='plotly_dark',
                showlegend=True,
                font=dict(color=self.color_scheme['text'])
            )
            
            # Update y-axis ranges
            fig.update_yaxes(range=[0, 100], row=1, col=1)  # RSI range
            
            return fig
            
        except Exception as e:
            print(f"Error creating indicators chart: {e}")
            return self._create_empty_chart(f"Error: {e}")
    
    def create_portfolio_performance_chart(self, portfolio_history: List[Dict[str, Any]]) -> go.Figure:
        """Create portfolio performance chart"""
        try:
            if not portfolio_history:
                return self._create_empty_chart("No portfolio history available")
            
            df = pd.DataFrame(portfolio_history)
            
            fig = go.Figure()
            
            # Portfolio value line
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['portfolio_value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color=self.color_scheme['up'], width=3)
                )
            )
            
            # Cash balance line
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['cash_balance'],
                    mode='lines',
                    name='Cash Balance',
                    line=dict(color='orange', width=2, dash='dash')
                )
            )
            
            # Calculate portfolio returns
            if len(df) > 1:
                initial_value = df['portfolio_value'].iloc[0]
                returns = ((df['portfolio_value'] - initial_value) / initial_value * 100)
                
                # Add benchmark line (assuming 0% return)
                fig.add_hline(
                    y=initial_value,
                    line_dash="dot",
                    line_color="white",
                    opacity=0.5,
                    annotation_text="Break-even"
                )
            
            fig.update_layout(
                title="Portfolio Performance Over Time",
                xaxis_title="Time",
                yaxis_title="Value ($)",
                height=400,
                template='plotly_dark',
                font=dict(color=self.color_scheme['text'])
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating portfolio chart: {e}")
            return self._create_empty_chart(f"Error: {e}")
    
    def create_trade_analysis_chart(self, trades: List[Dict[str, Any]]) -> go.Figure:
        """Create trade analysis chart"""
        try:
            if not trades:
                return self._create_empty_chart("No trades available")
            
            df = pd.DataFrame(trades)
            
            # Calculate P&L for each trade
            df['pnl'] = 0.0
            buy_prices = {}
            
            for i, trade in df.iterrows():
                symbol = trade['symbol']
                action = trade['action']
                price = trade['price']
                quantity = trade['quantity']
                
                if action == 'BUY':
                    buy_prices[symbol] = price
                elif action == 'SELL' and symbol in buy_prices:
                    pnl = (price - buy_prices[symbol]) * quantity
                    df.at[i, 'pnl'] = pnl
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Trade P&L', 'Cumulative P&L')
            )
            
            # Individual trade P&L
            colors = ['green' if pnl > 0 else 'red' for pnl in df['pnl']]
            
            fig.add_trace(
                go.Bar(
                    x=df['timestamp'],
                    y=df['pnl'],
                    name='Trade P&L',
                    marker_color=colors
                ),
                row=1, col=1
            )
            
            # Cumulative P&L
            cumulative_pnl = df['pnl'].cumsum()
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=cumulative_pnl,
                    mode='lines',
                    name='Cumulative P&L',
                    line=dict(color=self.color_scheme['up'], width=2)
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title="Trade Analysis",
                height=600,
                template='plotly_dark',
                font=dict(color=self.color_scheme['text'])
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating trade analysis chart: {e}")
            return self._create_empty_chart(f"Error: {e}")
    
    def create_signal_strength_chart(self, signals: List[Dict[str, Any]]) -> go.Figure:
        """Create signal strength visualization"""
        try:
            if not signals:
                return self._create_empty_chart("No signals available")
            
            df = pd.DataFrame([
                {
                    'timestamp': s['timestamp'],
                    'signal': s['signal']['signal'],
                    'strength': s['signal']['strength'],
                    'confidence': s['signal']['confidence']
                }
                for s in signals if 'signal' in s
            ])
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=('Signal Strength', 'Signal Confidence')
            )
            
            # Signal strength
            colors = {
                'BUY': 'green',
                'SELL': 'red',
                'HOLD': 'gray'
            }
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['strength'],
                    mode='markers',
                    name='Signal Strength',
                    marker=dict(
                        color=[colors.get(signal, 'gray') for signal in df['signal']],
                        size=10
                    )
                ),
                row=1, col=1
            )
            
            # Signal confidence
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['confidence'],
                    mode='lines+markers',
                    name='Confidence',
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title="Signal Analysis",
                height=500,
                template='plotly_dark',
                font=dict(color=self.color_scheme['text'])
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating signal chart: {e}")
            return self._create_empty_chart(f"Error: {e}")
    
    def create_correlation_heatmap(self, data: Dict[str, pd.DataFrame]) -> go.Figure:
        """Create correlation heatmap for multiple symbols"""
        try:
            if not data:
                return self._create_empty_chart("No data for correlation analysis")
            
            # Extract close prices for each symbol
            prices = {}
            for symbol, df in data.items():
                if not df.empty:
                    prices[symbol] = df['close']
            
            if len(prices) < 2:
                return self._create_empty_chart("Need at least 2 symbols for correlation")
            
            # Create DataFrame with aligned timestamps
            price_df = pd.DataFrame(prices)
            
            # Calculate correlation matrix
            correlation_matrix = price_df.corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=correlation_matrix.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 12},
                hoverongaps=False
            ))
            
            fig.update_layout(
                title="Symbol Correlation Matrix",
                height=500,
                template='plotly_dark',
                font=dict(color=self.color_scheme['text'])
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
            return self._create_empty_chart(f"Error: {e}")
    
    def create_risk_metrics_chart(self, risk_metrics: Dict[str, Any]) -> go.Figure:
        """Create risk metrics visualization"""
        try:
            if not risk_metrics:
                return self._create_empty_chart("No risk metrics available")
            
            # Create gauge chart for portfolio risk
            portfolio_risk_pct = risk_metrics.get('portfolio_risk_pct', 0)
            max_risk_pct = risk_metrics.get('max_portfolio_risk_pct', 20)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=portfolio_risk_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Portfolio Risk (%)"},
                delta={'reference': max_risk_pct},
                gauge={
                    'axis': {'range': [None, max_risk_pct * 1.5]},
                    'bar': {'color': "lightgreen"},
                    'steps': [
                        {'range': [0, max_risk_pct * 0.5], 'color': "lightgray"},
                        {'range': [max_risk_pct * 0.5, max_risk_pct], 'color': "yellow"},
                        {'range': [max_risk_pct, max_risk_pct * 1.5], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': max_risk_pct
                    }
                }
            ))
            
            fig.update_layout(
                height=400,
                template='plotly_dark',
                font=dict(color=self.color_scheme['text'])
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating risk metrics chart: {e}")
            return self._create_empty_chart(f"Error: {e}")
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color=self.color_scheme['text'])
        )
        
        fig.update_layout(
            template='plotly_dark',
            height=400,
            font=dict(color=self.color_scheme['text'])
        )
        
        return fig
