"""
Anomaly Detector - Real-time System Health and Market Anomaly Detection
Detects BUY/SELL ratio imbalances, drawdown streaks, and abnormal market conditions
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import threading
import time

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Detect trading system anomalies and market irregularities"""
    
    def __init__(self):
        self.is_monitoring = False
        self.anomaly_thresholds = {
            'buy_sell_ratio_extreme': 0.9,  # >90% BUYs or <10% BUYs
            'max_consecutive_losses': 5,
            'max_drawdown_pct': 8.0,
            'volume_spike_multiplier': 5.0,
            'volatility_spike_multiplier': 3.0,
            'confidence_drop_threshold': 15.0  # 15% drop in avg confidence
        }
        
        self.alert_history = []
        self.system_health_score = 100.0
        
        self.setup_database()
    
    def setup_database(self):
        """Initialize anomaly detection database"""
        try:
            conn = sqlite3.connect('anomaly_detection.db')
            cursor = conn.cursor()
            
            # Anomaly alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS anomaly_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    symbol TEXT,
                    description TEXT,
                    metrics TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_time DATETIME
                )
            ''')
            
            # System health metrics
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    health_score REAL,
                    buy_sell_ratio REAL,
                    avg_confidence REAL,
                    consecutive_losses INTEGER,
                    current_drawdown_pct REAL,
                    active_anomalies INTEGER,
                    notes TEXT
                )
            ''')
            
            # Market condition anomalies
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    severity TEXT,
                    current_value REAL,
                    baseline_value REAL,
                    deviation_factor REAL,
                    duration_minutes INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Anomaly detection database initialized")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
    
    def analyze_signal_distribution(self, signals: List[Dict]) -> Dict:
        """Analyze BUY/SELL signal distribution for anomalies"""
        try:
            if not signals:
                return {'anomaly_detected': False, 'buy_ratio': 0.5}
            
            buy_signals = len([s for s in signals if s.get('signal_type') == 'BUY'])
            sell_signals = len([s for s in signals if s.get('signal_type') == 'SELL'])
            total_signals = buy_signals + sell_signals
            
            if total_signals == 0:
                return {'anomaly_detected': False, 'buy_ratio': 0.5}
            
            buy_ratio = buy_signals / total_signals
            
            # Check for extreme ratios
            anomaly_detected = False
            severity = 'low'
            
            if buy_ratio > self.anomaly_thresholds['buy_sell_ratio_extreme']:
                anomaly_detected = True
                severity = 'high' if buy_ratio > 0.95 else 'medium'
                message = f"Extreme BUY bias: {buy_ratio*100:.1f}% BUY signals"
            elif buy_ratio < (1 - self.anomaly_thresholds['buy_sell_ratio_extreme']):
                anomaly_detected = True
                severity = 'high' if buy_ratio < 0.05 else 'medium'
                message = f"Extreme SELL bias: {(1-buy_ratio)*100:.1f}% SELL signals"
            else:
                message = f"Balanced signal distribution: {buy_ratio*100:.1f}% BUY"
            
            result = {
                'anomaly_detected': anomaly_detected,
                'severity': severity,
                'buy_ratio': buy_ratio,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'total_signals': total_signals,
                'message': message
            }
            
            if anomaly_detected:
                self.log_anomaly('signal_distribution', severity, None, message, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Signal distribution analysis failed: {e}")
            return {'anomaly_detected': False, 'buy_ratio': 0.5}
    
    def analyze_trading_performance(self, trade_history: List[Dict]) -> Dict:
        """Analyze trading performance for anomalies"""
        try:
            if not trade_history:
                return {'anomaly_detected': False, 'consecutive_losses': 0}
            
            # Analyze consecutive losses
            consecutive_losses = 0
            current_streak = 0
            
            for trade in reversed(trade_history[-20:]):  # Last 20 trades
                if trade.get('win_loss') == 'loss':
                    current_streak += 1
                    consecutive_losses = max(consecutive_losses, current_streak)
                else:
                    current_streak = 0
            
            # Calculate drawdown
            portfolio_values = []
            running_balance = 1000  # Starting value
            
            for trade in trade_history[-50:]:  # Last 50 trades
                pnl_pct = trade.get('pnl_pct', 0)
                running_balance *= (1 + pnl_pct / 100)
                portfolio_values.append(running_balance)
            
            if portfolio_values:
                peak = max(portfolio_values)
                current = portfolio_values[-1]
                drawdown_pct = ((peak - current) / peak) * 100
            else:
                drawdown_pct = 0
            
            # Check for anomalies
            anomalies = []
            
            if consecutive_losses >= self.anomaly_thresholds['max_consecutive_losses']:
                severity = 'high' if consecutive_losses >= 8 else 'medium'
                anomalies.append({
                    'type': 'consecutive_losses',
                    'severity': severity,
                    'value': consecutive_losses,
                    'message': f"{consecutive_losses} consecutive losses detected"
                })
            
            if drawdown_pct >= self.anomaly_thresholds['max_drawdown_pct']:
                severity = 'high' if drawdown_pct >= 15 else 'medium'
                anomalies.append({
                    'type': 'drawdown',
                    'severity': severity,
                    'value': drawdown_pct,
                    'message': f"High drawdown: {drawdown_pct:.1f}%"
                })
            
            # Log anomalies
            for anomaly in anomalies:
                self.log_anomaly(anomaly['type'], anomaly['severity'], None, 
                               anomaly['message'], {'value': anomaly['value']})
            
            return {
                'anomaly_detected': len(anomalies) > 0,
                'anomalies': anomalies,
                'consecutive_losses': consecutive_losses,
                'current_drawdown_pct': drawdown_pct,
                'total_trades': len(trade_history)
            }
            
        except Exception as e:
            logger.error(f"Trading performance analysis failed: {e}")
            return {'anomaly_detected': False, 'consecutive_losses': 0}
    
    def analyze_market_conditions(self, symbol: str, market_data: pd.DataFrame) -> Dict:
        """Analyze market conditions for anomalies"""
        try:
            if len(market_data) < 20:
                return {'anomaly_detected': False}
            
            anomalies = []
            
            # Volume spike detection
            if 'volume' in market_data.columns:
                recent_volume = market_data['volume'].iloc[-1]
                avg_volume = market_data['volume'].rolling(20).mean().iloc[-1]
                
                if avg_volume > 0:
                    volume_ratio = recent_volume / avg_volume
                    
                    if volume_ratio >= self.anomaly_thresholds['volume_spike_multiplier']:
                        severity = 'high' if volume_ratio >= 10 else 'medium'
                        anomalies.append({
                            'type': 'volume_spike',
                            'severity': severity,
                            'current_value': recent_volume,
                            'baseline_value': avg_volume,
                            'ratio': volume_ratio,
                            'message': f"Volume spike: {volume_ratio:.1f}x normal"
                        })
            
            # Volatility spike detection
            returns = market_data['close'].pct_change()
            recent_volatility = returns.rolling(5).std().iloc[-1]
            avg_volatility = returns.rolling(20).std().mean()
            
            if avg_volatility > 0:
                volatility_ratio = recent_volatility / avg_volatility
                
                if volatility_ratio >= self.anomaly_thresholds['volatility_spike_multiplier']:
                    severity = 'high' if volatility_ratio >= 5 else 'medium'
                    anomalies.append({
                        'type': 'volatility_spike',
                        'severity': severity,
                        'current_value': recent_volatility,
                        'baseline_value': avg_volatility,
                        'ratio': volatility_ratio,
                        'message': f"Volatility spike: {volatility_ratio:.1f}x normal"
                    })
            
            # Price gap detection
            if len(market_data) >= 2:
                prev_close = market_data['close'].iloc[-2]
                current_open = market_data['open'].iloc[-1]
                gap_pct = abs(current_open - prev_close) / prev_close * 100
                
                if gap_pct >= 5.0:  # 5% gap
                    severity = 'high' if gap_pct >= 10 else 'medium'
                    anomalies.append({
                        'type': 'price_gap',
                        'severity': severity,
                        'current_value': gap_pct,
                        'message': f"Price gap: {gap_pct:.1f}%"
                    })
            
            # Log market anomalies
            for anomaly in anomalies:
                self.log_market_anomaly(symbol, anomaly)
            
            return {
                'anomaly_detected': len(anomalies) > 0,
                'anomalies': anomalies,
                'symbol': symbol,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Market condition analysis failed for {symbol}: {e}")
            return {'anomaly_detected': False}
    
    def analyze_confidence_trends(self, recent_signals: List[Dict]) -> Dict:
        """Analyze signal confidence trends for anomalies"""
        try:
            if len(recent_signals) < 10:
                return {'anomaly_detected': False}
            
            # Calculate confidence trends
            recent_confidences = [s.get('confidence', 0) for s in recent_signals[-10:]]
            older_confidences = [s.get('confidence', 0) for s in recent_signals[-20:-10]] if len(recent_signals) >= 20 else recent_confidences
            
            recent_avg = np.mean(recent_confidences)
            older_avg = np.mean(older_confidences)
            
            confidence_drop = older_avg - recent_avg
            
            if confidence_drop >= self.anomaly_thresholds['confidence_drop_threshold']:
                severity = 'high' if confidence_drop >= 25 else 'medium'
                message = f"Confidence drop: {confidence_drop:.1f}% ({older_avg:.1f}% → {recent_avg:.1f}%)"
                
                self.log_anomaly('confidence_drop', severity, None, message, {
                    'recent_avg': recent_avg,
                    'older_avg': older_avg,
                    'drop': confidence_drop
                })
                
                return {
                    'anomaly_detected': True,
                    'severity': severity,
                    'confidence_drop': confidence_drop,
                    'recent_avg': recent_avg,
                    'older_avg': older_avg,
                    'message': message
                }
            
            return {'anomaly_detected': False, 'recent_avg': recent_avg}
            
        except Exception as e:
            logger.error(f"Confidence trend analysis failed: {e}")
            return {'anomaly_detected': False}
    
    def log_anomaly(self, anomaly_type: str, severity: str, symbol: Optional[str], 
                   description: str, metrics: Dict):
        """Log anomaly to database"""
        try:
            conn = sqlite3.connect('anomaly_detection.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO anomaly_alerts (
                    anomaly_type, severity, symbol, description, metrics
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                anomaly_type, severity, symbol, description, json.dumps(metrics)
            ))
            
            conn.commit()
            conn.close()
            
            # Update system health score
            self.update_system_health_score(anomaly_type, severity)
            
            logger.warning(f"Anomaly detected: {anomaly_type} ({severity}) - {description}")
            
        except Exception as e:
            logger.error(f"Failed to log anomaly: {e}")
    
    def log_market_anomaly(self, symbol: str, anomaly: Dict):
        """Log market-specific anomaly"""
        try:
            conn = sqlite3.connect('anomaly_detection.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO market_anomalies (
                    symbol, anomaly_type, severity, current_value, 
                    baseline_value, deviation_factor
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                anomaly['type'],
                anomaly['severity'],
                anomaly.get('current_value', 0),
                anomaly.get('baseline_value', 0),
                anomaly.get('ratio', 1)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to log market anomaly: {e}")
    
    def update_system_health_score(self, anomaly_type: str, severity: str):
        """Update overall system health score"""
        try:
            # Health score penalties
            penalties = {
                'low': 2,
                'medium': 5,
                'high': 10
            }
            
            penalty = penalties.get(severity, 5)
            self.system_health_score = max(0, self.system_health_score - penalty)
            
            # Recovery over time (1 point per hour if no new anomalies)
            # This would be called periodically in a real implementation
            
        except Exception as e:
            logger.error(f"Failed to update system health score: {e}")
    
    def get_active_anomalies(self) -> List[Dict]:
        """Get currently active anomalies"""
        try:
            conn = sqlite3.connect('anomaly_detection.db')
            
            active_anomalies = pd.read_sql_query('''
                SELECT * FROM anomaly_alerts 
                WHERE resolved = FALSE 
                AND timestamp > datetime('now', '-4 hours')
                ORDER BY timestamp DESC
            ''', conn)
            
            conn.close()
            
            return active_anomalies.to_dict('records') if not active_anomalies.empty else []
            
        except Exception as e:
            logger.error(f"Failed to get active anomalies: {e}")
            return []
    
    def get_anomaly_summary(self) -> Dict:
        """Get summary of anomaly detection status"""
        try:
            active_anomalies = self.get_active_anomalies()
            
            severity_counts = {
                'high': len([a for a in active_anomalies if a['severity'] == 'high']),
                'medium': len([a for a in active_anomalies if a['severity'] == 'medium']),
                'low': len([a for a in active_anomalies if a['severity'] == 'low'])
            }
            
            # Calculate health status
            if self.system_health_score >= 90:
                health_status = 'excellent'
            elif self.system_health_score >= 75:
                health_status = 'good'
            elif self.system_health_score >= 50:
                health_status = 'fair'
            else:
                health_status = 'poor'
            
            return {
                'system_health_score': self.system_health_score,
                'health_status': health_status,
                'active_anomalies': len(active_anomalies),
                'severity_breakdown': severity_counts,
                'monitoring_active': self.is_monitoring,
                'last_check': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Anomaly summary failed: {e}")
            return {
                'system_health_score': 50,
                'health_status': 'unknown',
                'active_anomalies': 0
            }
    
    def resolve_anomaly(self, anomaly_id: int):
        """Mark anomaly as resolved"""
        try:
            conn = sqlite3.connect('anomaly_detection.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE anomaly_alerts 
                SET resolved = TRUE, resolution_time = ?
                WHERE id = ?
            ''', (datetime.now(), anomaly_id))
            
            conn.commit()
            conn.close()
            
            # Improve health score when anomalies are resolved
            self.system_health_score = min(100, self.system_health_score + 5)
            
        except Exception as e:
            logger.error(f"Failed to resolve anomaly: {e}")
    
    def get_anomaly_insights(self) -> List[str]:
        """Get human-readable anomaly insights"""
        try:
            insights = []
            summary = self.get_anomaly_summary()
            
            # Health status insight
            health_status = summary['health_status']
            health_score = summary['system_health_score']
            insights.append(f"System health: {health_status.title()} ({health_score:.1f}/100)")
            
            # Active anomalies
            active_count = summary['active_anomalies']
            if active_count > 0:
                severity_counts = summary['severity_breakdown']
                high_count = severity_counts['high']
                medium_count = severity_counts['medium']
                
                if high_count > 0:
                    insights.append(f"⚠️ {high_count} high-severity anomalies detected")
                if medium_count > 0:
                    insights.append(f"🔸 {medium_count} medium-severity anomalies detected")
                    
                insights.append(f"Total active anomalies: {active_count}")
            else:
                insights.append("✅ No active anomalies detected")
            
            return insights
            
        except Exception as e:
            logger.error(f"Anomaly insights failed: {e}")
            return ["Anomaly monitoring temporarily unavailable"]
    
    def comprehensive_anomaly_scan(self, signals: List[Dict], trades: List[Dict], 
                                 market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Perform comprehensive anomaly detection across all systems"""
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'anomalies_detected': False,
                'total_anomalies': 0,
                'details': {}
            }
            
            # Signal distribution analysis
            signal_analysis = self.analyze_signal_distribution(signals)
            results['details']['signal_distribution'] = signal_analysis
            
            # Trading performance analysis
            performance_analysis = self.analyze_trading_performance(trades)
            results['details']['trading_performance'] = performance_analysis
            
            # Confidence trend analysis
            confidence_analysis = self.analyze_confidence_trends(signals)
            results['details']['confidence_trends'] = confidence_analysis
            
            # Market condition analysis for key symbols
            market_anomalies = []
            for symbol, data in list(market_data.items())[:5]:  # Limit to 5 symbols
                market_analysis = self.analyze_market_conditions(symbol, data)
                if market_analysis['anomaly_detected']:
                    market_anomalies.append(market_analysis)
            
            results['details']['market_conditions'] = market_anomalies
            
            # Count total anomalies
            total_anomalies = 0
            if signal_analysis['anomaly_detected']:
                total_anomalies += 1
            if performance_analysis['anomaly_detected']:
                total_anomalies += len(performance_analysis.get('anomalies', []))
            if confidence_analysis['anomaly_detected']:
                total_anomalies += 1
            total_anomalies += len(market_anomalies)
            
            results['anomalies_detected'] = total_anomalies > 0
            results['total_anomalies'] = total_anomalies
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive anomaly scan failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'anomalies_detected': False,
                'total_anomalies': 0,
                'error': str(e)
            }