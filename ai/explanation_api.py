"""
AI Explanation API - REST endpoints for trade reasoning
Provides structured JSON responses for AI decision explanations
"""

from datetime import datetime
from typing import Dict, Any, Optional
import json
import random

def get_explanation_json(symbol: str, trade_reason_logger) -> Optional[Dict[str, Any]]:
    """Get explanation in JSON format for API integration"""
    try:
        explanation = trade_reason_logger.get_explanation_by_symbol(symbol.upper())
        
        if not explanation:
            return None
        
        return {
            "symbol": explanation['symbol'],
            "model": explanation['model'],
            "decision": explanation['decision'],
            "confidence": explanation['confidence'],
            "top_features": explanation['top_features'][:3],
            "timestamp": explanation['timestamp']
        }
        
    except Exception as e:
        print(f"Error getting explanation JSON: {e}")
        return None

def get_all_explanations_json(trade_reason_logger, limit: int = 10, symbol: str = None, decision: str = None) -> Dict[str, Any]:
    """Get all explanations in JSON format with filters"""
    try:
        explanations = trade_reason_logger.get_recent_explanations(symbol=symbol, limit=limit)
        
        if decision:
            explanations = [exp for exp in explanations if exp['decision'] == decision.upper()]
        
        formatted_explanations = []
        for exp in explanations:
            formatted_explanations.append({
                "symbol": exp['symbol'],
                "model": exp['model'],
                "decision": exp['decision'],
                "confidence": exp['confidence'],
                "top_features": exp['top_features'][:3],
                "timestamp": exp['timestamp'],
                "signal_strength": exp['signal_strength']
            })
        
        return {
            "explanations": formatted_explanations,
            "count": len(formatted_explanations),
            "filters": {
                "symbol": symbol,
                "decision": decision,
                "limit": limit
            }
        }
        
    except Exception as e:
        return {
            "error": "Internal server error",
            "message": str(e)
        }

def simulate_ai_decision(symbol: str, model: str, trade_reason_logger) -> Dict[str, Any]:
    """Simulate an AI decision for demonstration purposes"""
    
    # Sample decisions for demonstration
    decisions = ['BUY', 'SELL', 'HOLD']
    
    decision = random.choice(decisions)
    confidence = random.uniform(60, 95)
    
    # Generate realistic features based on decision
    if decision == 'BUY':
        features = [
            f"RSI = {random.uniform(25, 35):.1f} (oversold)",
            "EMA crossover signal detected", 
            "Volume spike above average"
        ]
    elif decision == 'SELL':
        features = [
            f"RSI = {random.uniform(70, 80):.1f} (overbought)",
            "Resistance level rejection",
            "Bearish divergence pattern"
        ]
    else:  # HOLD
        features = [
            f"RSI = {random.uniform(45, 55):.1f} (neutral)",
            "Sideways price movement",
            "Low volatility environment"
        ]
    
    # Log the simulated decision
    simulated_explanation = trade_reason_logger.log_decision(
        symbol=symbol,
        model_name=model,
        decision=decision,
        confidence=confidence,
        features_data={
            'feature_importance': {
                'rsi': random.uniform(0.2, 0.4),
                'ema_crossover': random.uniform(0.1, 0.3),
                'volume': random.uniform(0.1, 0.2)
            }
        }
    )
    
    return {
        "symbol": simulated_explanation.symbol,
        "model": simulated_explanation.model,
        "decision": simulated_explanation.decision,
        "confidence": simulated_explanation.confidence,
        "top_features": simulated_explanation.top_features,
        "timestamp": simulated_explanation.timestamp
    }