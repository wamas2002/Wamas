"""
Feedback Logger - Performance Self-Awareness System
Logs every executed trade with comprehensive details for learning analysis
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)

class FeedbackLogger:
    """Log trade executions and signal performance for analysis"""
    
    def __init__(self, db_path: str = 'feedback_learning.db'):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Initialize feedback logging database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Trade execution logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    original_confidence REAL,
                    adjusted_confidence REAL,
                    
                    -- Market context
                    market_regime TEXT,
                    regime_confidence REAL,
                    volatility REAL,
                    volume_ratio REAL,
                    
                    -- Execution details
                    entry_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    position_size_pct REAL,
                    
                    -- Targets and stops
                    target_price REAL,
                    stop_loss_price REAL,
                    
                    -- Technical indicators at entry
                    rsi REAL,
                    macd REAL,
                    ema20 REAL,
                    ema50 REAL,
                    
                    -- Trade outcome (updated later)
                    exit_price REAL,
                    exit_timestamp DATETIME,
                    exit_reason TEXT,
                    pnl_pct REAL,
                    win_loss TEXT,
                    holding_period_hours REAL,
                    
                    -- Analysis fields
                    signal_accuracy_score REAL,
                    notes TEXT
                )
            ''')
            
            # Signal performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signal_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    market_regime TEXT,
                    
                    -- Signal components
                    rsi_signal BOOLEAN,
                    macd_signal BOOLEAN,
                    ema_signal BOOLEAN,
                    volume_signal BOOLEAN,
                    
                    -- Context
                    volatility_level TEXT,
                    trend_strength TEXT,
                    
                    -- Outcome tracking
                    executed BOOLEAN DEFAULT FALSE,
                    execution_delay_minutes REAL,
                    outcome_24h TEXT,
                    price_movement_24h REAL,
                    
                    -- Learning metrics
                    prediction_accuracy REAL,
                    confidence_calibration REAL
                )
            ''')
            
            # Pattern recognition logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    pattern_type TEXT NOT NULL,
                    pattern_details TEXT,
                    market_conditions TEXT,
                    success_rate REAL,
                    sample_size INTEGER,
                    confidence_interval TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Feedback logging database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def log_trade_execution(self, signal: Dict, execution_details: Dict) -> str:
        """Log a trade execution with full context"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Generate unique signal ID
            signal_id = f"{signal['symbol']}_{signal['signal_type']}_{int(datetime.now().timestamp())}"
            
            cursor.execute('''
                INSERT INTO trade_executions (
                    signal_id, symbol, signal_type, confidence, original_confidence,
                    adjusted_confidence, market_regime, regime_confidence, volatility,
                    volume_ratio, entry_price, quantity, position_size_pct,
                    target_price, stop_loss_price, rsi, macd, ema20, ema50
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal_id,
                signal['symbol'],
                signal['signal_type'],
                signal['confidence'],
                signal.get('original_confidence', signal['confidence']),
                signal.get('adjusted_confidence', signal['confidence']),
                signal.get('market_regime', 'unknown'),
                signal.get('regime_confidence', 0.0),
                signal.get('volatility', 0.0),
                signal.get('volume_ratio', 1.0),
                execution_details['price'],
                execution_details['quantity'],
                execution_details.get('position_size_pct', 0.0),
                signal.get('target_price'),
                signal.get('stop_loss'),
                signal.get('rsi'),
                signal.get('macd'),
                signal.get('ema20'),
                signal.get('ema50')
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Trade execution logged: {signal_id}")
            return signal_id
            
        except Exception as e:
            logger.error(f"Trade execution logging failed: {e}")
            return ""
    
    def get_learning_insights(self) -> List[str]:
        """Generate human-readable learning insights"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Check if we have enough data
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM trade_executions WHERE exit_price IS NOT NULL")
            completed_trades = cursor.fetchone()[0]
            
            if completed_trades < 5:
                return ["Collecting trade data for analysis...", 
                       f"Completed trades: {completed_trades}/5 minimum for insights"]
            
            insights = []
            
            # Overall performance
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN win_loss = 'win' THEN 1 ELSE 0 END) as wins,
                    AVG(pnl_pct) as avg_pnl
                FROM trade_executions 
                WHERE exit_price IS NOT NULL
                AND timestamp > datetime('now', '-7 days')
            ''')
            
            result = cursor.fetchone()
            if result and result[0] > 0:
                total, wins, avg_pnl = result
                win_rate = (wins / total) * 100
                insights.append(f"7-day performance: {win_rate:.1f}% win rate ({wins}/{total} trades)")
                insights.append(f"Average PnL: {avg_pnl:.2f}% per trade")
            
            # Regime analysis
            cursor.execute('''
                SELECT market_regime, COUNT(*) as count,
                       SUM(CASE WHEN win_loss = 'win' THEN 1 ELSE 0 END) as wins
                FROM trade_executions 
                WHERE exit_price IS NOT NULL 
                GROUP BY market_regime
                HAVING count >= 2
            ''')
            
            regime_results = cursor.fetchall()
            for regime, count, wins in regime_results:
                win_rate = (wins / count) * 100
                insights.append(f"{regime.title()} market: {win_rate:.1f}% win rate ({count} trades)")
            
            conn.close()
            return insights if insights else ["Analyzing performance patterns..."]
            
        except Exception as e:
            logger.error(f"Learning insights failed: {e}")
            return ["Performance analysis temporarily unavailable"]