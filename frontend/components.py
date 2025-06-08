import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

class UIComponents:
    """Reusable UI components for the trading dashboard"""
    
    def __init__(self):
        self.color_scheme = {
            'success': '#00ff88',
            'danger': '#ff4b4b',
            'warning': '#ffa500',
            'info': '#1f77b4',
            'neutral': '#808080'
        }
    
    def render_metric_card(self, title: str, value: str, delta: Optional[str] = None, 
                          delta_color: str = 'normal') -> None:
        """Render a metric card"""
        try:
            st.metric(
                label=title,
                value=value,
                delta=delta,
                delta_color=delta_color
            )
        except Exception as e:
            st.error(f"Error rendering metric card: {e}")
    
    def render_status_indicator(self, status: str, message: str) -> None:
        """Render status indicator with color coding"""
        try:
            if status.lower() == 'success':
                st.success(f"✅ {message}")
            elif status.lower() == 'warning':
                st.warning(f"⚠️ {message}")
            elif status.lower() == 'error':
                st.error(f"❌ {message}")
            elif status.lower() == 'info':
                st.info(f"ℹ️ {message}")
            else:
                st.write(f"🔘 {message}")
        except Exception as e:
            st.error(f"Error rendering status indicator: {e}")
    
    def render_progress_bar(self, current: float, maximum: float, 
                           label: str = "", format_func=None) -> None:
        """Render progress bar with optional formatting"""
        try:
            progress = min(current / maximum, 1.0) if maximum > 0 else 0.0
            
            if format_func:
                display_text = format_func(current, maximum)
            else:
                display_text = f"{label}: {current:.2f} / {maximum:.2f}"
            
            st.progress(progress)
            st.caption(display_text)
            
        except Exception as e:
            st.error(f"Error rendering progress bar: {e}")
    
    def render_signal_badge(self, signal: str, confidence: float = 0.0) -> None:
        """Render trading signal badge with color coding"""
        try:
            if signal.upper() == 'BUY':
                st.success(f"🟢 BUY ({confidence:.1%})")
            elif signal.upper() == 'SELL':
                st.error(f"🔴 SELL ({confidence:.1%})")
            elif signal.upper() == 'HOLD':
                st.info(f"🟡 HOLD ({confidence:.1%})")
            else:
                st.write(f"🔘 {signal} ({confidence:.1%})")
        except Exception as e:
            st.error(f"Error rendering signal badge: {e}")
    
    def render_data_table(self, data: pd.DataFrame, title: str = "", 
                         height: Optional[int] = None) -> None:
        """Render data table with optional title and height"""
        try:
            if title:
                st.subheader(title)
            
            if data.empty:
                st.info("No data available")
                return
            
            st.dataframe(data, height=height, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering data table: {e}")
    
    def render_key_value_pairs(self, data: Dict[str, Any], title: str = "") -> None:
        """Render key-value pairs in a formatted way"""
        try:
            if title:
                st.subheader(title)
            
            for key, value in data.items():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write(f"**{key}:**")
                with col2:
                    if isinstance(value, float):
                        st.write(f"{value:.4f}")
                    elif isinstance(value, (int, str)):
                        st.write(str(value))
                    else:
                        st.write(str(value))
                        
        except Exception as e:
            st.error(f"Error rendering key-value pairs: {e}")
    
    def render_trade_history_table(self, trades: List[Dict[str, Any]]) -> None:
        """Render trade history in a formatted table"""
        try:
            if not trades:
                st.info("No trades executed yet")
                return
            
            # Convert to DataFrame for better display
            df = pd.DataFrame(trades)
            
            # Format columns
            if 'timestamp' in df.columns:
                df['Time'] = pd.to_datetime(df['timestamp']).dt.strftime('%H:%M:%S')
            
            if 'price' in df.columns:
                df['Price'] = df['price'].apply(lambda x: f"${x:.4f}")
            
            if 'quantity' in df.columns:
                df['Quantity'] = df['quantity'].apply(lambda x: f"{x:.6f}")
            
            if 'value' in df.columns:
                df['Value'] = df['value'].apply(lambda x: f"${x:.2f}")
            
            # Select and rename columns for display
            display_columns = {
                'Time': 'Time',
                'symbol': 'Symbol',
                'action': 'Action',
                'Quantity': 'Quantity',
                'Price': 'Price',
                'Value': 'Value'
            }
            
            available_columns = {k: v for k, v in display_columns.items() if k in df.columns}
            
            if available_columns:
                display_df = df[list(available_columns.keys())].rename(columns=available_columns)
                st.dataframe(display_df, use_container_width=True)
            else:
                st.dataframe(df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error rendering trade history: {e}")
    
    def render_portfolio_summary(self, portfolio_data: Dict[str, Any]) -> None:
        """Render portfolio summary cards"""
        try:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                portfolio_value = portfolio_data.get('portfolio_value', 0)
                self.render_metric_card(
                    "Portfolio Value",
                    f"${portfolio_value:,.2f}",
                    f"${portfolio_value - 10000:,.2f}"
                )
            
            with col2:
                cash_balance = portfolio_data.get('cash_balance', 0)
                self.render_metric_card(
                    "Cash Balance",
                    f"${cash_balance:,.2f}"
                )
            
            with col3:
                positions_value = portfolio_data.get('positions_value', 0)
                self.render_metric_card(
                    "Positions Value",
                    f"${positions_value:,.2f}"
                )
            
            with col4:
                total_trades = portfolio_data.get('total_trades', 0)
                self.render_metric_card(
                    "Total Trades",
                    str(total_trades)
                )
                
        except Exception as e:
            st.error(f"Error rendering portfolio summary: {e}")
    
    def render_risk_gauge(self, current_risk: float, max_risk: float, 
                         title: str = "Risk Level") -> None:
        """Render risk level gauge"""
        try:
            risk_percentage = (current_risk / max_risk * 100) if max_risk > 0 else 0
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = risk_percentage,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': title},
                delta = {'reference': 100},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "lightgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300, template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error rendering risk gauge: {e}")
    
    def render_signal_history(self, signals: List[Dict[str, Any]], limit: int = 10) -> None:
        """Render recent signals history"""
        try:
            st.subheader("Recent Signals")
            
            if not signals:
                st.info("No signals generated yet")
                return
            
            recent_signals = signals[-limit:] if len(signals) > limit else signals
            
            for signal_data in reversed(recent_signals):
                timestamp = signal_data.get('timestamp', datetime.now())
                signal = signal_data.get('signal', {})
                
                signal_type = signal.get('signal', 'HOLD')
                confidence = signal.get('confidence', 0)
                strength = signal.get('strength', 0)
                
                # Create signal container
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        self.render_signal_badge(signal_type, confidence)
                    
                    with col2:
                        st.write(f"Strength: {strength:.2f}")
                    
                    with col3:
                        st.caption(timestamp.strftime('%H:%M:%S'))
                    
                    st.divider()
                    
        except Exception as e:
            st.error(f"Error rendering signal history: {e}")
    
    def render_loading_spinner(self, message: str = "Loading...") -> None:
        """Render loading spinner with message"""
        try:
            with st.spinner(message):
                st.empty()
        except Exception as e:
            st.error(f"Error rendering loading spinner: {e}")
    
    def render_alert_box(self, alert_type: str, title: str, message: str, 
                        dismissible: bool = False) -> None:
        """Render alert box with different types"""
        try:
            alert_content = f"**{title}**\n\n{message}"
            
            if alert_type.lower() == 'success':
                st.success(alert_content)
            elif alert_type.lower() == 'warning':
                st.warning(alert_content)
            elif alert_type.lower() == 'error':
                st.error(alert_content)
            elif alert_type.lower() == 'info':
                st.info(alert_content)
            else:
                st.write(alert_content)
                
        except Exception as e:
            st.error(f"Error rendering alert box: {e}")
    
    def render_performance_summary(self, metrics: Dict[str, Any]) -> None:
        """Render performance metrics summary"""
        try:
            if not metrics:
                st.info("No performance metrics available")
                return
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_return = metrics.get('total_return', 0)
                self.render_metric_card(
                    "Total Return",
                    f"{total_return:.1%}",
                    delta_color='normal' if total_return >= 0 else 'inverse'
                )
                
                win_rate = metrics.get('win_rate', 0)
                self.render_metric_card(
                    "Win Rate",
                    f"{win_rate:.1%}"
                )
            
            with col2:
                sharpe_ratio = metrics.get('sharpe_ratio', 0)
                self.render_metric_card(
                    "Sharpe Ratio",
                    f"{sharpe_ratio:.2f}"
                )
                
                volatility = metrics.get('volatility', 0)
                self.render_metric_card(
                    "Volatility",
                    f"{volatility:.1%}"
                )
            
            with col3:
                max_drawdown = metrics.get('max_drawdown', 0)
                self.render_metric_card(
                    "Max Drawdown",
                    f"{max_drawdown:.1%}",
                    delta_color='inverse'
                )
                
                avg_return = metrics.get('avg_return', 0)
                self.render_metric_card(
                    "Avg Return",
                    f"{avg_return:.2%}"
                )
                
        except Exception as e:
            st.error(f"Error rendering performance summary: {e}")
    
    def render_position_details(self, positions: Dict[str, Dict[str, Any]]) -> None:
        """Render detailed position information"""
        try:
            if not positions:
                st.info("No active positions")
                return
            
            for symbol, position in positions.items():
                with st.expander(f"{symbol} Position"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Quantity:** {position.get('quantity', 0):.6f}")
                        st.write(f"**Entry Price:** ${position.get('price', 0):.4f}")
                    
                    with col2:
                        st.write(f"**Current Value:** ${position.get('value', 0):.2f}")
                        
                        # Calculate P&L if current price available
                        entry_price = position.get('price', 0)
                        current_price = position.get('current_price', entry_price)
                        if entry_price > 0:
                            pnl_pct = (current_price - entry_price) / entry_price * 100
                            st.write(f"**P&L:** {pnl_pct:+.2f}%")
                            
        except Exception as e:
            st.error(f"Error rendering position details: {e}")
    
    def render_quick_stats(self, stats: Dict[str, Any]) -> None:
        """Render quick statistics in a compact format"""
        try:
            with st.container():
                cols = st.columns(len(stats))
                
                for i, (key, value) in enumerate(stats.items()):
                    with cols[i]:
                        if isinstance(value, (int, float)):
                            if isinstance(value, float):
                                display_value = f"{value:.2f}"
                            else:
                                display_value = str(value)
                        else:
                            display_value = str(value)
                        
                        st.metric(key, display_value)
                        
        except Exception as e:
            st.error(f"Error rendering quick stats: {e}")
    
    def render_trend_indicator(self, current_value: float, previous_value: float, 
                              label: str = "Trend") -> None:
        """Render trend indicator with arrow"""
        try:
            if previous_value == 0:
                trend = "Neutral"
                arrow = "➡️"
            elif current_value > previous_value:
                trend = "Up"
                arrow = "⬆️"
            else:
                trend = "Down"
                arrow = "⬇️"
            
            change_pct = ((current_value - previous_value) / previous_value * 100) if previous_value != 0 else 0
            
            st.write(f"{arrow} **{label}:** {trend} ({change_pct:+.1f}%)")
            
        except Exception as e:
            st.error(f"Error rendering trend indicator: {e}")
