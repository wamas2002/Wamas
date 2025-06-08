"""
AI Performance Tracker - Logs and stores model performance statistics
Tracks accuracy, latency, PnL impact and exposes API endpoints
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import json
import sqlite3
from dataclasses import dataclass, asdict
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceRecord:
    """Individual performance record for a model decision"""
    timestamp: datetime
    symbol: str
    model_name: str
    decision: str
    confidence: float
    entry_price: float
    exit_price: Optional[float]
    actual_return: Optional[float]
    predicted_return: float
    execution_latency: float
    is_correct: Optional[bool]
    pnl_impact: float

class AIPerformanceTracker:
    """
    Tracks and stores AI model performance statistics in lightweight database
    Provides API endpoints for performance retrieval
    """
    
    def __init__(self, db_path: str = "data/ai_performance.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._ensure_database_exists()
        self._create_tables()
        
        # In-memory cache for recent performance
        self.performance_cache = {}  # symbol -> recent records
        self.cache_size = 1000  # Keep last 1000 records per symbol
        
        logger.info(f"AI Performance Tracker initialized with database: {db_path}")
    
    def _ensure_database_exists(self):
        """Ensure the database directory and file exist"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_tables(self):
        """Create database tables for performance tracking"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    actual_return REAL,
                    predicted_return REAL NOT NULL,
                    execution_latency REAL NOT NULL,
                    is_correct INTEGER,
                    pnl_impact REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_aggregates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    total_decisions INTEGER DEFAULT 0,
                    correct_decisions INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    avg_confidence REAL DEFAULT 0.0,
                    avg_latency REAL DEFAULT 0.0,
                    total_pnl REAL DEFAULT 0.0,
                    sharpe_ratio REAL DEFAULT 0.0,
                    max_drawdown REAL DEFAULT 0.0,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, model_name, date)
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_symbol_model ON performance_records(symbol, model_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_timestamp ON performance_records(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agg_symbol_model ON model_aggregates(symbol, model_name)")
            
            conn.commit()
    
    def log_decision(self, symbol: str, model_name: str, decision: str, confidence: float,
                    entry_price: float, predicted_return: float, execution_latency: float) -> str:
        """Log a new AI decision for performance tracking"""
        try:
            record = PerformanceRecord(
                timestamp=datetime.now(),
                symbol=symbol,
                model_name=model_name,
                decision=decision,
                confidence=confidence,
                entry_price=entry_price,
                exit_price=None,
                actual_return=None,
                predicted_return=predicted_return,
                execution_latency=execution_latency,
                is_correct=None,
                pnl_impact=0.0
            )
            
            record_id = self._store_record(record)
            self._update_cache(symbol, record)
            
            logger.debug(f"Logged decision: {symbol} {model_name} {decision} (ID: {record_id})")
            return record_id
            
        except Exception as e:
            logger.error(f"Error logging decision: {e}")
            return None
    
    def update_decision_outcome(self, record_id: str, exit_price: float, actual_return: float):
        """Update decision outcome when position is closed"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get the original record
                cursor = conn.execute("""
                    SELECT * FROM performance_records WHERE id = ?
                """, (record_id,))
                
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"Record {record_id} not found for outcome update")
                    return
                
                # Calculate if prediction was correct
                predicted_return = row[9]  # predicted_return column
                is_correct = (predicted_return > 0 and actual_return > 0) or \
                           (predicted_return < 0 and actual_return < 0) or \
                           (abs(predicted_return) < 0.001 and abs(actual_return) < 0.001)
                
                # Calculate PnL impact (simplified)
                pnl_impact = actual_return * 0.01  # Assume 1% position size
                
                # Update the record
                conn.execute("""
                    UPDATE performance_records 
                    SET exit_price = ?, actual_return = ?, is_correct = ?, pnl_impact = ?
                    WHERE id = ?
                """, (exit_price, actual_return, int(is_correct), pnl_impact, record_id))
                
                conn.commit()
                
                # Update daily aggregates
                self._update_daily_aggregates(row[2], row[3])  # symbol, model_name
                
                logger.debug(f"Updated decision outcome: {record_id} correct={is_correct}")
                
        except Exception as e:
            logger.error(f"Error updating decision outcome: {e}")
    
    def _store_record(self, record: PerformanceRecord) -> str:
        """Store performance record in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO performance_records 
                (timestamp, symbol, model_name, decision, confidence, entry_price, 
                 exit_price, actual_return, predicted_return, execution_latency, 
                 is_correct, pnl_impact)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.timestamp.isoformat(),
                record.symbol,
                record.model_name,
                record.decision,
                record.confidence,
                record.entry_price,
                record.exit_price,
                record.actual_return,
                record.predicted_return,
                record.execution_latency,
                record.is_correct,
                record.pnl_impact
            ))
            
            conn.commit()
            return str(cursor.lastrowid)
    
    def _update_cache(self, symbol: str, record: PerformanceRecord):
        """Update in-memory cache with new record"""
        if symbol not in self.performance_cache:
            self.performance_cache[symbol] = []
        
        self.performance_cache[symbol].append(record)
        
        # Keep cache size manageable
        if len(self.performance_cache[symbol]) > self.cache_size:
            self.performance_cache[symbol] = self.performance_cache[symbol][-self.cache_size:]
    
    def _update_daily_aggregates(self, symbol: str, model_name: str):
        """Update daily performance aggregates for a model"""
        try:
            today = datetime.now().date().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Calculate daily statistics
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_decisions,
                        SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as correct_decisions,
                        AVG(confidence) as avg_confidence,
                        AVG(execution_latency) as avg_latency,
                        SUM(pnl_impact) as total_pnl,
                        AVG(pnl_impact) as avg_pnl
                    FROM performance_records 
                    WHERE symbol = ? AND model_name = ? 
                    AND DATE(timestamp) = ? 
                    AND is_correct IS NOT NULL
                """, (symbol, model_name, today))
                
                row = cursor.fetchone()
                if row and row[0] > 0:  # total_decisions > 0
                    total_decisions = row[0]
                    correct_decisions = row[1] or 0
                    win_rate = (correct_decisions / total_decisions) * 100 if total_decisions > 0 else 0
                    avg_confidence = row[2] or 0
                    avg_latency = row[3] or 0
                    total_pnl = row[4] or 0
                    
                    # Calculate Sharpe ratio (simplified)
                    returns_cursor = conn.execute("""
                        SELECT actual_return FROM performance_records 
                        WHERE symbol = ? AND model_name = ? 
                        AND DATE(timestamp) = ? 
                        AND actual_return IS NOT NULL
                    """, (symbol, model_name, today))
                    
                    returns = [r[0] for r in returns_cursor.fetchall()]
                    sharpe_ratio = 0.0
                    if len(returns) > 1:
                        returns_std = np.std(returns)
                        if returns_std > 0:
                            sharpe_ratio = (np.mean(returns) / returns_std) * np.sqrt(252)  # Annualized
                    
                    # Calculate max drawdown (simplified)
                    max_drawdown = 0.0
                    if len(returns) > 0:
                        cumulative_returns = np.cumsum(returns)
                        running_max = np.maximum.accumulate(cumulative_returns)
                        drawdowns = (cumulative_returns - running_max) / (running_max + 1e-10)
                        max_drawdown = abs(np.min(drawdowns)) * 100
                    
                    # Insert or update aggregate
                    conn.execute("""
                        INSERT OR REPLACE INTO model_aggregates 
                        (symbol, model_name, date, total_decisions, correct_decisions, 
                         win_rate, avg_confidence, avg_latency, total_pnl, sharpe_ratio, max_drawdown)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (symbol, model_name, today, total_decisions, correct_decisions,
                          win_rate, avg_confidence, avg_latency, total_pnl, sharpe_ratio, max_drawdown))
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error updating daily aggregates: {e}")
    
    def get_model_performance(self, symbol: str, model_name: str = None, 
                            days_back: int = 7) -> Dict[str, Any]:
        """Get model performance statistics"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            with sqlite3.connect(self.db_path) as conn:
                if model_name:
                    # Specific model performance
                    cursor = conn.execute("""
                        SELECT 
                            COUNT(*) as total_trades,
                            SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as wins,
                            AVG(confidence) as avg_confidence,
                            AVG(execution_latency) as avg_latency,
                            SUM(pnl_impact) as total_pnl,
                            MAX(timestamp) as last_trade
                        FROM performance_records 
                        WHERE symbol = ? AND model_name = ?
                        AND timestamp >= ? AND is_correct IS NOT NULL
                    """, (symbol, model_name, start_date.isoformat()))
                    
                    row = cursor.fetchone()
                    if row and row[0] > 0:
                        total_trades = row[0]
                        wins = row[1] or 0
                        return {
                            "symbol": symbol,
                            "model_name": model_name,
                            "total_trades": total_trades,
                            "win_rate": round((wins / total_trades) * 100, 1) if total_trades > 0 else 0,
                            "avg_confidence": round(row[2] or 0, 1),
                            "avg_latency": round(row[3] or 0, 3),
                            "total_pnl": round(row[4] or 0, 4),
                            "last_trade": row[5],
                            "period_days": days_back
                        }
                else:
                    # All models for symbol
                    cursor = conn.execute("""
                        SELECT 
                            model_name,
                            COUNT(*) as total_trades,
                            SUM(CASE WHEN is_correct = 1 THEN 1 ELSE 0 END) as wins,
                            AVG(confidence) as avg_confidence,
                            AVG(execution_latency) as avg_latency,
                            SUM(pnl_impact) as total_pnl
                        FROM performance_records 
                        WHERE symbol = ? AND timestamp >= ? AND is_correct IS NOT NULL
                        GROUP BY model_name
                    """, (symbol, start_date.isoformat()))
                    
                    results = {}
                    for row in cursor.fetchall():
                        model = row[0]
                        total_trades = row[1]
                        wins = row[2] or 0
                        results[model] = {
                            "total_trades": total_trades,
                            "win_rate": round((wins / total_trades) * 100, 1) if total_trades > 0 else 0,
                            "avg_confidence": round(row[3] or 0, 1),
                            "avg_latency": round(row[4] or 0, 3),
                            "total_pnl": round(row[5] or 0, 4)
                        }
                    
                    return {
                        "symbol": symbol,
                        "models": results,
                        "period_days": days_back
                    }
            
            return {}
            
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return {}
    
    def get_performance_summary(self, symbol: str = None) -> Dict[str, Any]:
        """Get performance summary for all models and symbols"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if symbol:
                    # Symbol-specific summary
                    cursor = conn.execute("""
                        SELECT 
                            model_name,
                            AVG(win_rate) as avg_win_rate,
                            AVG(avg_confidence) as avg_confidence,
                            SUM(total_decisions) as total_decisions,
                            SUM(total_pnl) as total_pnl,
                            AVG(avg_latency) as avg_latency
                        FROM model_aggregates 
                        WHERE symbol = ?
                        AND date >= date('now', '-7 days')
                        GROUP BY model_name
                        ORDER BY avg_win_rate DESC
                    """, (symbol,))
                else:
                    # Overall summary
                    cursor = conn.execute("""
                        SELECT 
                            symbol,
                            model_name,
                            AVG(win_rate) as avg_win_rate,
                            AVG(avg_confidence) as avg_confidence,
                            SUM(total_decisions) as total_decisions,
                            SUM(total_pnl) as total_pnl
                        FROM model_aggregates 
                        WHERE date >= date('now', '-7 days')
                        GROUP BY symbol, model_name
                        ORDER BY symbol, avg_win_rate DESC
                    """)
                
                results = []
                for row in cursor.fetchall():
                    if symbol:
                        results.append({
                            "model_name": row[0],
                            "win_rate": round(row[1] or 0, 1),
                            "avg_confidence": round(row[2] or 0, 1),
                            "total_decisions": row[3] or 0,
                            "total_pnl": round(row[4] or 0, 4),
                            "avg_latency": round(row[5] or 0, 3)
                        })
                    else:
                        results.append({
                            "symbol": row[0],
                            "model_name": row[1],
                            "win_rate": round(row[2] or 0, 1),
                            "avg_confidence": round(row[3] or 0, 1),
                            "total_decisions": row[4] or 0,
                            "total_pnl": round(row[5] or 0, 4)
                        })
                
                return {
                    "summary": results,
                    "timestamp": datetime.now().isoformat(),
                    "period": "7_days"
                }
                
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {"summary": [], "error": str(e)}
    
    def cleanup_old_records(self, days_to_keep: int = 30):
        """Clean up old performance records to manage database size"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM performance_records 
                    WHERE timestamp < ?
                """, (cutoff_date.isoformat(),))
                
                deleted_count = cursor.rowcount
                
                # Also cleanup old aggregates
                conn.execute("""
                    DELETE FROM model_aggregates 
                    WHERE date < ?
                """, (cutoff_date.date().isoformat(),))
                
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old performance records")
                
        except Exception as e:
            logger.error(f"Error cleaning up old records: {e}")
    
    def simulate_historical_performance(self, symbol: str, days: int = 7):
        """Simulate some historical performance data for demonstration"""
        try:
            models = ['LSTM', 'Prophet', 'GradientBoost', 'Technical', 'Ensemble']
            
            for day in range(days):
                date = datetime.now() - timedelta(days=day)
                
                for model in models:
                    # Simulate some trades for each model
                    trades_count = np.random.randint(3, 12)
                    
                    for _ in range(trades_count):
                        # Generate realistic performance data
                        confidence = np.random.normal(70, 15)
                        confidence = max(40, min(95, confidence))
                        
                        # Model-specific performance characteristics
                        if model == 'LSTM':
                            base_accuracy = 0.58
                        elif model == 'GradientBoost':
                            base_accuracy = 0.62
                        elif model == 'Ensemble':
                            base_accuracy = 0.65
                        else:
                            base_accuracy = 0.55
                        
                        is_correct = np.random.random() < base_accuracy
                        actual_return = np.random.normal(0.001 if is_correct else -0.001, 0.005)
                        predicted_return = np.random.normal(0.002 if is_correct else -0.002, 0.003)
                        
                        record = PerformanceRecord(
                            timestamp=date - timedelta(seconds=np.random.randint(0, 86400)),
                            symbol=symbol,
                            model_name=model,
                            decision='BUY' if predicted_return > 0 else 'SELL',
                            confidence=confidence,
                            entry_price=np.random.uniform(45000, 55000),
                            exit_price=np.random.uniform(45000, 55000),
                            actual_return=actual_return,
                            predicted_return=predicted_return,
                            execution_latency=np.random.uniform(0.05, 0.3),
                            is_correct=is_correct,
                            pnl_impact=actual_return * 0.01
                        )
                        
                        self._store_record(record)
            
            # Update aggregates
            for model in models:
                for day in range(days):
                    date = (datetime.now() - timedelta(days=day)).date().isoformat()
                    self._update_daily_aggregates(symbol, model)
            
            logger.info(f"Simulated {days} days of performance data for {symbol}")
            
        except Exception as e:
            logger.error(f"Error simulating historical performance: {e}")