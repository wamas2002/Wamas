"""
Confidence-based UI utilities for highlighting decisions and predictions
Provides consistent confidence-based styling across all interface components
"""

import streamlit as st
from typing import Optional, Dict, Any


def get_confidence_color(confidence: float) -> str:
    """Get color based on confidence level"""
    if confidence >= 80:
        return "#00C851"  # Green for high confidence
    elif confidence >= 60:
        return "#ffbb33"  # Amber for moderate confidence
    else:
        return "#ff4444"  # Red for low confidence


def get_confidence_emoji(confidence: float) -> str:
    """Get emoji based on confidence level"""
    if confidence >= 80:
        return "🟢"  # High confidence
    elif confidence >= 60:
        return "🟡"  # Moderate confidence
    else:
        return "🔴"  # Low confidence


def display_confidence_badge(confidence: float, label: str = "Confidence") -> None:
    """Display a confidence badge with color coding"""
    color = get_confidence_color(confidence)
    emoji = get_confidence_emoji(confidence)
    
    st.markdown(
        f"""
        <div style="
            background-color: {color}15;
            border: 1px solid {color};
            border-radius: 5px;
            padding: 5px 10px;
            margin: 5px 0;
            display: inline-block;
        ">
            {emoji} <strong>{label}:</strong> {confidence:.1f}%
        </div>
        """,
        unsafe_allow_html=True
    )


def display_decision_card(
    symbol: str, 
    decision: str, 
    confidence: float, 
    reasoning: Optional[str] = None,
    model: Optional[str] = None
) -> None:
    """Display a decision card with confidence highlighting"""
    
    decision_colors = {
        "BUY": "#00C851",
        "SELL": "#ff4444", 
        "HOLD": "#17a2b8"
    }
    
    decision_emojis = {
        "BUY": "📈",
        "SELL": "📉",
        "HOLD": "⏸️"
    }
    
    decision_color = decision_colors.get(decision, "#6c757d")
    decision_emoji = decision_emojis.get(decision, "❓")
    confidence_color = get_confidence_color(confidence)
    confidence_emoji = get_confidence_emoji(confidence)
    
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {decision_color}15, {confidence_color}10);
            border-left: 5px solid {decision_color};
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: {decision_color};">
                        {decision_emoji} {symbol} - {decision}
                    </h4>
                    {f"<small style='color: #666;'>{model}</small>" if model else ""}
                </div>
                <div style="text-align: right;">
                    <div style="
                        background-color: {confidence_color};
                        color: white;
                        padding: 5px 10px;
                        border-radius: 15px;
                        font-weight: bold;
                        font-size: 0.9em;
                    ">
                        {confidence_emoji} {confidence:.1f}%
                    </div>
                </div>
            </div>
            {f"<p style='margin: 10px 0 0 0; color: #555;'>{reasoning}</p>" if reasoning else ""}
        </div>
        """,
        unsafe_allow_html=True
    )


def display_prediction_highlight(
    value: float, 
    confidence: float, 
    label: str = "Prediction",
    unit: str = ""
) -> None:
    """Display a prediction with confidence-based highlighting"""
    
    confidence_color = get_confidence_color(confidence)
    confidence_emoji = get_confidence_emoji(confidence)
    
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, {confidence_color}20, transparent);
            border: 1px solid {confidence_color};
            border-radius: 6px;
            padding: 10px;
            margin: 5px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong>{label}:</strong> {value:.4f}{unit}
                </div>
                <small style="color: {confidence_color};">
                    {confidence_emoji} {confidence:.1f}%
                </small>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def display_feature_importance(features: Dict[str, float], confidence: float) -> None:
    """Display feature importance with confidence highlighting"""
    
    confidence_color = get_confidence_color(confidence)
    
    st.markdown(f"**Key Features** (Confidence: {confidence:.1f}%)")
    
    for feature, importance in features.items():
        importance_pct = importance * 100
        bar_color = confidence_color if importance > 0.5 else "#dee2e6"
        
        st.markdown(
            f"""
            <div style="margin: 5px 0;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.9em;">{feature}</span>
                    <span style="font-size: 0.8em; color: {confidence_color};">
                        {importance_pct:.1f}%
                    </span>
                </div>
                <div style="
                    background-color: #f8f9fa;
                    border-radius: 3px;
                    height: 6px;
                    margin-top: 2px;
                ">
                    <div style="
                        background-color: {bar_color};
                        height: 100%;
                        width: {importance_pct}%;
                        border-radius: 3px;
                        transition: width 0.3s ease;
                    "></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def display_model_performance_summary(model_stats: Dict[str, Any]) -> None:
    """Display model performance summary with confidence highlighting"""
    
    st.subheader("🤖 Model Performance Overview")
    
    for model_name, stats in model_stats.items():
        avg_confidence = stats.get('avg_confidence', 0)
        total_decisions = stats.get('total_decisions', 0)
        accuracy = stats.get('accuracy', 0) * 100
        
        confidence_color = get_confidence_color(avg_confidence)
        confidence_emoji = get_confidence_emoji(avg_confidence)
        
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, {confidence_color}15, transparent);
                border: 1px solid {confidence_color}30;
                border-radius: 8px;
                padding: 15px;
                margin: 10px 0;
            ">
                <h4 style="margin: 0 0 10px 0; color: {confidence_color};">
                    {confidence_emoji} {model_name}
                </h4>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 15px;">
                    <div>
                        <small style="color: #666;">Avg Confidence</small><br>
                        <strong style="color: {confidence_color};">{avg_confidence:.1f}%</strong>
                    </div>
                    <div>
                        <small style="color: #666;">Total Decisions</small><br>
                        <strong>{total_decisions}</strong>
                    </div>
                    <div>
                        <small style="color: #666;">Accuracy</small><br>
                        <strong>{accuracy:.1f}%</strong>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def display_risk_assessment(risk_level: str, confidence: float, factors: list) -> None:
    """Display risk assessment with confidence highlighting"""
    
    risk_colors = {
        "LOW": "#00C851",
        "MODERATE": "#ffbb33",
        "HIGH": "#ff4444",
        "CRITICAL": "#cc0000"
    }
    
    risk_emojis = {
        "LOW": "🟢",
        "MODERATE": "🟡", 
        "HIGH": "🔴",
        "CRITICAL": "🚨"
    }
    
    risk_color = risk_colors.get(risk_level, "#6c757d")
    risk_emoji = risk_emojis.get(risk_level, "❓")
    confidence_color = get_confidence_color(confidence)
    confidence_emoji = get_confidence_emoji(confidence)
    
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {risk_color}15, {confidence_color}10);
            border-left: 5px solid {risk_color};
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h4 style="margin: 0; color: {risk_color};">
                    {risk_emoji} Risk Level: {risk_level}
                </h4>
                <span style="
                    background-color: {confidence_color};
                    color: white;
                    padding: 3px 8px;
                    border-radius: 10px;
                    font-size: 0.8em;
                ">
                    {confidence_emoji} {confidence:.1f}%
                </span>
            </div>
            {f"<ul style='margin: 10px 0 0 20px; color: #555;'>{''.join([f'<li>{factor}</li>' for factor in factors])}</ul>" if factors else ""}
        </div>
        """,
        unsafe_allow_html=True
    )