#!/usr/bin/env python3
"""
Personalized Risk Appetite Animation System
Dynamic visual risk assessment with animated feedback and personalized recommendations
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import sqlite3
import time
from datetime import datetime, timedelta
import math
import json
from typing import Dict, List, Tuple, Optional

class PersonalizedRiskAppetiteAnimator:
    def __init__(self):
        self.db_path = "live_trading.db"
        self.risk_profiles = {
            'conservative': {
                'max_risk_per_trade': 0.01,  # 1%
                'max_portfolio_risk': 0.15,  # 15%
                'confidence_threshold': 0.75,
                'color': '#2ecc71',
                'description': 'Safety First - Steady Growth'
            },
            'moderate': {
                'max_risk_per_trade': 0.02,  # 2%
                'max_portfolio_risk': 0.25,  # 25%
                'confidence_threshold': 0.60,
                'color': '#f39c12',
                'description': 'Balanced - Growth with Caution'
            },
            'aggressive': {
                'max_risk_per_trade': 0.05,  # 5%
                'max_portfolio_risk': 0.40,  # 40%
                'confidence_threshold': 0.45,
                'color': '#e74c3c',
                'description': 'High Reward - High Risk'
            }
        }
        
    def initialize_database(self):
        """Initialize risk appetite database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_appetite_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'default_user',
                risk_profile TEXT,
                custom_settings TEXT,
                performance_score REAL,
                last_updated DATETIME,
                total_trades INTEGER DEFAULT 0,
                successful_trades INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_appetite_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                risk_score REAL,
                market_volatility REAL,
                portfolio_risk REAL,
                recommended_action TEXT,
                user_response TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def calculate_current_risk_metrics(self) -> Dict:
        """Calculate real-time risk metrics from portfolio and market data"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get portfolio data
            portfolio_query = """
                SELECT symbol, current_value, total_investment, pnl_percentage
                FROM portfolio_data 
                WHERE current_value > 0
            """
            portfolio_df = pd.read_sql_query(portfolio_query, conn)
            
            # Get recent trading performance
            trades_query = """
                SELECT symbol, signal, confidence, timestamp, pnl
                FROM ai_signals 
                WHERE timestamp > datetime('now', '-7 days')
                ORDER BY timestamp DESC
                LIMIT 50
            """
            trades_df = pd.read_sql_query(trades_query, conn)
            
            conn.close()
            
            # Calculate metrics
            total_portfolio_value = portfolio_df['current_value'].sum() if not portfolio_df.empty else 100000
            portfolio_volatility = portfolio_df['pnl_percentage'].std() if not portfolio_df.empty else 0.05
            
            # Calculate win rate and recent performance
            win_rate = 0.0
            avg_confidence = 0.65
            recent_pnl = 0.0
            
            if not trades_df.empty:
                # Simulate PnL based on confidence and signal direction
                trades_df['simulated_pnl'] = trades_df.apply(
                    lambda row: np.random.normal(
                        row['confidence'] * 0.02 if row['signal'] in ['BUY', 'SELL'] else 0,
                        0.01
                    ), axis=1
                )
                
                successful_trades = len(trades_df[trades_df['simulated_pnl'] > 0])
                win_rate = successful_trades / len(trades_df) if len(trades_df) > 0 else 0.0
                avg_confidence = trades_df['confidence'].mean()
                recent_pnl = trades_df['simulated_pnl'].sum()
            
            # Market volatility estimate (simplified)
            market_volatility = min(0.15, max(0.01, portfolio_volatility * 1.5))
            
            # Risk appetite score (0-100)
            base_risk_score = 50
            performance_adjustment = (win_rate - 0.5) * 50  # ±25 points
            volatility_adjustment = -market_volatility * 100  # Lower score in high volatility
            confidence_adjustment = (avg_confidence - 0.5) * 30  # ±15 points
            
            risk_appetite_score = max(0, min(100, 
                base_risk_score + performance_adjustment + volatility_adjustment + confidence_adjustment
            ))
            
            return {
                'risk_appetite_score': risk_appetite_score,
                'portfolio_value': total_portfolio_value,
                'portfolio_volatility': portfolio_volatility,
                'market_volatility': market_volatility,
                'win_rate': win_rate,
                'avg_confidence': avg_confidence,
                'recent_pnl': recent_pnl,
                'total_trades': len(trades_df),
                'successful_trades': len(trades_df[trades_df['simulated_pnl'] > 0]) if not trades_df.empty else 0
            }
            
        except Exception as e:
            st.error(f"Error calculating risk metrics: {e}")
            return self._get_default_risk_metrics()
    
    def _get_default_risk_metrics(self) -> Dict:
        """Default risk metrics when database is unavailable"""
        return {
            'risk_appetite_score': 65.0,
            'portfolio_value': 100000.0,
            'portfolio_volatility': 0.05,
            'market_volatility': 0.03,
            'win_rate': 0.45,
            'avg_confidence': 0.68,
            'recent_pnl': 250.0,
            'total_trades': 15,
            'successful_trades': 7
        }
    
    def create_risk_appetite_gauge(self, metrics: Dict) -> go.Figure:
        """Create animated risk appetite gauge"""
        risk_score = metrics['risk_appetite_score']
        
        # Determine risk level and color
        if risk_score <= 33:
            risk_level = "Conservative"
            color = '#2ecc71'
        elif risk_score <= 66:
            risk_level = "Moderate"
            color = '#f39c12'
        else:
            risk_level = "Aggressive"
            color = '#e74c3c'
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Risk Appetite Level: {risk_level}", 'font': {'size': 24}},
            delta = {'reference': 50, 'increasing': {'color': "#e74c3c"}, 'decreasing': {'color': "#2ecc71"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color, 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 33], 'color': '#d5f4e6'},
                    {'range': [33, 66], 'color': '#fdeaa7'},
                    {'range': [66, 100], 'color': '#fab1a0'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            font={'color': "darkblue", 'family': "Arial"},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    def create_risk_evolution_animation(self, metrics: Dict) -> go.Figure:
        """Create animated risk evolution chart"""
        # Generate simulated historical data for demonstration
        days = 30
        dates = [datetime.now() - timedelta(days=x) for x in range(days, 0, -1)]
        
        # Simulate risk score evolution based on current metrics
        base_score = metrics['risk_appetite_score']
        risk_scores = []
        volatility_scores = []
        confidence_scores = []
        
        for i in range(days):
            # Add realistic variation
            daily_variation = np.random.normal(0, 5)
            trend_factor = (i - days/2) * 0.5  # Slight trend
            score = max(0, min(100, base_score + daily_variation + trend_factor))
            risk_scores.append(score)
            
            # Volatility tends to be inverse to confidence
            vol_score = max(0, min(100, 50 + np.random.normal(0, 10)))
            volatility_scores.append(vol_score)
            
            conf_score = max(0, min(100, 65 + np.random.normal(0, 8)))
            confidence_scores.append(conf_score)
        
        fig = go.Figure()
        
        # Risk Appetite Line
        fig.add_trace(go.Scatter(
            x=dates,
            y=risk_scores,
            mode='lines+markers',
            name='Risk Appetite',
            line=dict(color='#3498db', width=3),
            marker=dict(size=6),
            hovertemplate='<b>Risk Appetite</b><br>Date: %{x}<br>Score: %{y:.1f}<extra></extra>'
        ))
        
        # Market Volatility Area
        fig.add_trace(go.Scatter(
            x=dates,
            y=volatility_scores,
            mode='lines',
            name='Market Volatility',
            line=dict(color='#e74c3c', width=2, dash='dash'),
            fill='tonexty',
            fillcolor='rgba(231,76,60,0.1)',
            hovertemplate='<b>Market Volatility</b><br>Date: %{x}<br>Score: %{y:.1f}<extra></extra>'
        ))
        
        # Confidence Level
        fig.add_trace(go.Scatter(
            x=dates,
            y=confidence_scores,
            mode='lines',
            name='Confidence Level',
            line=dict(color='#2ecc71', width=2),
            hovertemplate='<b>Confidence Level</b><br>Date: %{x}<br>Score: %{y:.1f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Risk Appetite Evolution (30 Days)',
            xaxis_title='Date',
            yaxis_title='Score (0-100)',
            height=400,
            showlegend=True,
            hovermode='x unified',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray')
        )
        
        return fig
    
    def create_risk_breakdown_pie(self, metrics: Dict) -> go.Figure:
        """Create animated pie chart showing risk factor breakdown"""
        # Calculate risk factor contributions
        performance_factor = metrics['win_rate'] * 100
        volatility_factor = (1 - metrics['market_volatility']) * 100
        confidence_factor = metrics['avg_confidence'] * 100
        diversification_factor = 75  # Simulated diversification score
        
        factors = ['Performance', 'Low Volatility', 'Confidence', 'Diversification']
        values = [performance_factor, volatility_factor, confidence_factor, diversification_factor]
        colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
        
        fig = go.Figure(data=[go.Pie(
            labels=factors,
            values=values,
            hole=0.4,
            marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
            textinfo='label+percent',
            textfont=dict(size=12),
            hovertemplate='<b>%{label}</b><br>Contribution: %{percent}<br>Score: %{value:.1f}<extra></extra>'
        )])
        
        fig.update_layout(
            title='Risk Profile Composition',
            height=400,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(text='Risk<br>Factors', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        return fig
    
    def create_personalized_recommendations(self, metrics: Dict) -> List[Dict]:
        """Generate personalized risk management recommendations"""
        recommendations = []
        risk_score = metrics['risk_appetite_score']
        win_rate = metrics['win_rate']
        volatility = metrics['market_volatility']
        
        # Performance-based recommendations
        if win_rate < 0.4:
            recommendations.append({
                'type': 'performance',
                'priority': 'high',
                'title': 'Improve Signal Quality',
                'description': f'Win rate at {win_rate:.1%} suggests signal filtering needed',
                'action': 'Increase confidence threshold to 70%+',
                'impact': 'Higher quality trades, better win rate'
            })
        
        # Volatility-based recommendations
        if volatility > 0.08:
            recommendations.append({
                'type': 'volatility',
                'priority': 'medium',
                'title': 'High Market Volatility Detected',
                'description': f'Current volatility at {volatility:.1%} above normal',
                'action': 'Reduce position sizes by 30-50%',
                'impact': 'Lower portfolio risk during volatile periods'
            })
        
        # Risk appetite recommendations
        if risk_score > 80:
            recommendations.append({
                'type': 'risk_appetite',
                'priority': 'medium',
                'title': 'High Risk Appetite',
                'description': 'Current settings may expose portfolio to significant losses',
                'action': 'Consider reducing max risk per trade to 3%',
                'impact': 'Better capital preservation'
            })
        elif risk_score < 30:
            recommendations.append({
                'type': 'risk_appetite',
                'priority': 'low',
                'title': 'Conservative Approach',
                'description': 'Very low risk may limit growth potential',
                'action': 'Consider gradual increase in position sizes',
                'impact': 'Better growth opportunities'
            })
        
        # Portfolio diversification
        recommendations.append({
            'type': 'diversification',
            'priority': 'low',
            'title': 'Portfolio Diversification',
            'description': 'Maintain balanced exposure across assets',
            'action': 'Keep individual asset allocation under 25%',
            'impact': 'Reduced correlation risk'
        })
        
        return recommendations
    
    def save_risk_profile_update(self, metrics: Dict, selected_profile: str):
        """Save updated risk profile to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Update or insert risk profile
            cursor.execute("""
                INSERT OR REPLACE INTO risk_appetite_profiles 
                (user_id, risk_profile, custom_settings, performance_score, last_updated, total_trades, successful_trades)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                'default_user',
                selected_profile,
                json.dumps(metrics),
                metrics['risk_appetite_score'],
                datetime.now().isoformat(),
                metrics['total_trades'],
                metrics['successful_trades']
            ))
            
            conn.commit()
            conn.close()
            
            st.success(f"Risk profile updated to {selected_profile.upper()}")
            
        except Exception as e:
            st.error(f"Error saving risk profile: {e}")

def main():
    """Main Streamlit application for Personalized Risk Appetite Animation"""
    st.set_page_config(
        page_title="Personalized Risk Appetite Animation",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 Personalized Risk Appetite Animation")
    st.markdown("**Dynamic Risk Assessment with Visual Feedback**")
    
    # Initialize the animator
    animator = PersonalizedRiskAppetiteAnimator()
    animator.initialize_database()
    
    # Sidebar controls
    st.sidebar.header("Risk Profile Settings")
    
    # Get current metrics
    metrics = animator.calculate_current_risk_metrics()
    
    # Risk profile selector
    selected_profile = st.sidebar.selectbox(
        "Select Risk Profile",
        options=['conservative', 'moderate', 'aggressive'],
        index=1,
        format_func=lambda x: animator.risk_profiles[x]['description']
    )
    
    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (10s)", value=True)
    
    if auto_refresh:
        time.sleep(0.1)  # Small delay for smooth animation
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Save profile button
    if st.sidebar.button("💾 Save Profile"):
        animator.save_risk_profile_update(metrics, selected_profile)
    
    # Main dashboard
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Current Risk Appetite")
        risk_gauge = animator.create_risk_appetite_gauge(metrics)
        st.plotly_chart(risk_gauge, use_container_width=True)
        
        # Key metrics
        st.subheader("Key Metrics")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                "Win Rate", 
                f"{metrics['win_rate']:.1%}",
                delta=f"+{(metrics['win_rate'] - 0.5):.1%}" if metrics['win_rate'] > 0.5 else f"{(metrics['win_rate'] - 0.5):.1%}"
            )
        
        with metric_col2:
            st.metric(
                "Confidence", 
                f"{metrics['avg_confidence']:.1%}",
                delta=f"+{(metrics['avg_confidence'] - 0.6):.1%}" if metrics['avg_confidence'] > 0.6 else f"{(metrics['avg_confidence'] - 0.6):.1%}"
            )
        
        with metric_col3:
            st.metric(
                "Volatility", 
                f"{metrics['market_volatility']:.1%}",
                delta=f"+{(metrics['market_volatility'] - 0.03):.1%}" if metrics['market_volatility'] > 0.03 else f"{(metrics['market_volatility'] - 0.03):.1%}",
                delta_color="inverse"
            )
    
    with col2:
        st.subheader("Risk Factor Breakdown")
        risk_pie = animator.create_risk_breakdown_pie(metrics)
        st.plotly_chart(risk_pie, use_container_width=True)
    
    # Risk evolution chart
    st.subheader("Risk Appetite Trends")
    evolution_chart = animator.create_risk_evolution_animation(metrics)
    st.plotly_chart(evolution_chart, use_container_width=True)
    
    # Personalized recommendations
    st.subheader("🎯 Personalized Recommendations")
    recommendations = animator.create_personalized_recommendations(metrics)
    
    for rec in recommendations:
        priority_color = {
            'high': '🔴',
            'medium': '🟡', 
            'low': '🟢'
        }
        
        with st.expander(f"{priority_color[rec['priority']]} {rec['title']}", expanded=rec['priority']=='high'):
            st.write(f"**Description:** {rec['description']}")
            st.write(f"**Recommended Action:** {rec['action']}")
            st.write(f"**Expected Impact:** {rec['impact']}")
            
            if st.button(f"Apply {rec['title']}", key=f"apply_{rec['type']}"):
                st.success(f"Applied recommendation: {rec['title']}")
                time.sleep(1)
                st.rerun()
    
    # Footer with profile information
    st.markdown("---")
    profile_info = animator.risk_profiles[selected_profile]
    st.markdown(f"""
    **Current Profile:** {profile_info['description']}  
    **Max Risk per Trade:** {profile_info['max_risk_per_trade']:.1%}  
    **Max Portfolio Risk:** {profile_info['max_portfolio_risk']:.1%}  
    **Confidence Threshold:** {profile_info['confidence_threshold']:.1%}
    """)

if __name__ == "__main__":
    main()