import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import os
from pathlib import Path

class TradingLogger:
    """Comprehensive logging system for trading operations"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup different loggers
        self.setup_loggers()
        
        # In-memory storage for recent logs
        self.trade_logs = []
        self.signal_logs = []
        self.error_logs = []
        self.performance_logs = []
        
        # Log retention settings
        self.max_memory_logs = 1000
        self.max_log_files = 30  # days
        
    def setup_loggers(self):
        """Setup different loggers for different types of events"""
        try:
            # Trade logger
            self.trade_logger = logging.getLogger('trading.trades')
            self.trade_logger.setLevel(logging.INFO)
            
            trade_handler = logging.FileHandler(
                self.log_dir / f'trades_{datetime.now().strftime("%Y%m%d")}.log'
            )
            trade_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            trade_handler.setFormatter(trade_formatter)
            self.trade_logger.addHandler(trade_handler)
            
            # Signal logger
            self.signal_logger = logging.getLogger('trading.signals')
            self.signal_logger.setLevel(logging.INFO)
            
            signal_handler = logging.FileHandler(
                self.log_dir / f'signals_{datetime.now().strftime("%Y%m%d")}.log'
            )
            signal_handler.setFormatter(trade_formatter)
            self.signal_logger.addHandler(signal_handler)
            
            # Error logger
            self.error_logger = logging.getLogger('trading.errors')
            self.error_logger.setLevel(logging.ERROR)
            
            error_handler = logging.FileHandler(
                self.log_dir / f'errors_{datetime.now().strftime("%Y%m%d")}.log'
            )
            error_handler.setFormatter(trade_formatter)
            self.error_logger.addHandler(error_handler)
            
            # Performance logger
            self.performance_logger = logging.getLogger('trading.performance')
            self.performance_logger.setLevel(logging.INFO)
            
            perf_handler = logging.FileHandler(
                self.log_dir / f'performance_{datetime.now().strftime("%Y%m%d")}.log'
            )
            perf_handler.setFormatter(trade_formatter)
            self.performance_logger.addHandler(perf_handler)
            
            # Console logger for important events
            self.console_logger = logging.getLogger('trading.console')
            self.console_logger.setLevel(logging.INFO)
            
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.console_logger.addHandler(console_handler)
            
        except Exception as e:
            print(f"Error setting up loggers: {e}")
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """Log trade execution"""
        try:
            timestamp = trade_data.get('timestamp', datetime.now())
            symbol = trade_data.get('symbol', 'UNKNOWN')
            action = trade_data.get('action', 'UNKNOWN')
            quantity = trade_data.get('quantity', 0)
            price = trade_data.get('price', 0)
            value = trade_data.get('value', 0)
            
            # Create log entry
            log_entry = {
                'timestamp': timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp),
                'type': 'TRADE',
                'symbol': symbol,
                'action': action,
                'quantity': quantity,
                'price': price,
                'value': value,
                'signal_strength': trade_data.get('signal_strength', 0),
                'portfolio_value': trade_data.get('portfolio_value', 0)
            }
            
            # Log to file
            log_message = (
                f"TRADE - {symbol} {action} {quantity:.6f} @ ${price:.4f} "
                f"(Value: ${value:.2f}, Strength: {trade_data.get('signal_strength', 0):.2f})"
            )
            self.trade_logger.info(log_message)
            
            # Store in memory
            self.trade_logs.append(log_entry)
            self._cleanup_memory_logs('trade')
            
            # Console output for important trades
            if value > 100:  # Only log significant trades to console
                self.console_logger.info(f"Trade executed: {log_message}")
                
        except Exception as e:
            self.log_error(f"Error logging trade: {e}")
    
    def log_signal(self, symbol: str, signal_data: Dict[str, Any]):
        """Log trading signal generation"""
        try:
            timestamp = datetime.now()
            
            log_entry = {
                'timestamp': timestamp.isoformat(),
                'type': 'SIGNAL',
                'symbol': symbol,
                'signal': signal_data.get('signal', 'HOLD'),
                'strength': signal_data.get('strength', 0),
                'confidence': signal_data.get('confidence', 0),
                'market_regime': signal_data.get('market_regime', 'unknown'),
                'combined_score': signal_data.get('combined_score', 0)
            }
            
            # Log to file
            log_message = (
                f"SIGNAL - {symbol} {signal_data.get('signal', 'HOLD')} "
                f"(Strength: {signal_data.get('strength', 0):.2f}, "
                f"Confidence: {signal_data.get('confidence', 0):.2f})"
            )
            self.signal_logger.info(log_message)
            
            # Store in memory
            self.signal_logs.append(log_entry)
            self._cleanup_memory_logs('signal')
            
        except Exception as e:
            self.log_error(f"Error logging signal: {e}")
    
    def log_error(self, error_message: str, context: Dict[str, Any] = None):
        """Log error events"""
        try:
            timestamp = datetime.now()
            
            log_entry = {
                'timestamp': timestamp.isoformat(),
                'type': 'ERROR',
                'message': str(error_message),
                'context': context or {}
            }
            
            # Log to file and console
            self.error_logger.error(error_message)
            self.console_logger.error(f"ERROR: {error_message}")
            
            # Store in memory
            self.error_logs.append(log_entry)
            self._cleanup_memory_logs('error')
            
        except Exception as e:
            print(f"Critical error in logger: {e}")
    
    def log_performance(self, performance_data: Dict[str, Any]):
        """Log performance metrics"""
        try:
            timestamp = datetime.now()
            
            log_entry = {
                'timestamp': timestamp.isoformat(),
                'type': 'PERFORMANCE',
                'portfolio_value': performance_data.get('portfolio_value', 0),
                'total_return': performance_data.get('total_return', 0),
                'daily_return': performance_data.get('daily_return', 0),
                'sharpe_ratio': performance_data.get('sharpe_ratio', 0),
                'max_drawdown': performance_data.get('max_drawdown', 0),
                'win_rate': performance_data.get('win_rate', 0),
                'total_trades': performance_data.get('total_trades', 0)
            }
            
            # Log to file
            log_message = (
                f"PERFORMANCE - Portfolio: ${performance_data.get('portfolio_value', 0):,.2f}, "
                f"Return: {performance_data.get('total_return', 0):.2%}, "
                f"Sharpe: {performance_data.get('sharpe_ratio', 0):.2f}"
            )
            self.performance_logger.info(log_message)
            
            # Store in memory
            self.performance_logs.append(log_entry)
            self._cleanup_memory_logs('performance')
            
        except Exception as e:
            self.log_error(f"Error logging performance: {e}")
    
    def log_system_event(self, event_type: str, message: str, data: Dict[str, Any] = None):
        """Log general system events"""
        try:
            timestamp = datetime.now()
            
            log_entry = {
                'timestamp': timestamp.isoformat(),
                'type': 'SYSTEM',
                'event_type': event_type,
                'message': message,
                'data': data or {}
            }
            
            # Log to console and appropriate file logger
            self.console_logger.info(f"{event_type}: {message}")
            
            if event_type.upper() in ['START', 'STOP', 'INIT']:
                self.trade_logger.info(f"SYSTEM - {event_type}: {message}")
            
        except Exception as e:
            self.log_error(f"Error logging system event: {e}")
    
    def _cleanup_memory_logs(self, log_type: str):
        """Clean up in-memory logs to prevent memory issues"""
        try:
            if log_type == 'trade' and len(self.trade_logs) > self.max_memory_logs:
                self.trade_logs = self.trade_logs[-self.max_memory_logs:]
            elif log_type == 'signal' and len(self.signal_logs) > self.max_memory_logs:
                self.signal_logs = self.signal_logs[-self.max_memory_logs:]
            elif log_type == 'error' and len(self.error_logs) > self.max_memory_logs:
                self.error_logs = self.error_logs[-self.max_memory_logs:]
            elif log_type == 'performance' and len(self.performance_logs) > self.max_memory_logs:
                self.performance_logs = self.performance_logs[-self.max_memory_logs:]
                
        except Exception as e:
            print(f"Error cleaning up memory logs: {e}")
    
    def get_recent_logs(self, log_type: str = 'all', limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent logs from memory"""
        try:
            if log_type == 'trade':
                return self.trade_logs[-limit:]
            elif log_type == 'signal':
                return self.signal_logs[-limit:]
            elif log_type == 'error':
                return self.error_logs[-limit:]
            elif log_type == 'performance':
                return self.performance_logs[-limit:]
            else:
                # Return all logs mixed and sorted by timestamp
                all_logs = (self.trade_logs + self.signal_logs + 
                           self.error_logs + self.performance_logs)
                
                # Sort by timestamp
                sorted_logs = sorted(
                    all_logs, 
                    key=lambda x: x.get('timestamp', ''),
                    reverse=True
                )
                
                return sorted_logs[:limit]
                
        except Exception as e:
            self.log_error(f"Error getting recent logs: {e}")
            return []
    
    def get_log_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of logs for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            cutoff_str = cutoff_time.isoformat()
            
            # Filter logs by time period
            recent_trades = [
                log for log in self.trade_logs 
                if log.get('timestamp', '') >= cutoff_str
            ]
            
            recent_signals = [
                log for log in self.signal_logs 
                if log.get('timestamp', '') >= cutoff_str
            ]
            
            recent_errors = [
                log for log in self.error_logs 
                if log.get('timestamp', '') >= cutoff_str
            ]
            
            # Calculate summary statistics
            summary = {
                'time_period': f"Last {hours} hours",
                'total_trades': len(recent_trades),
                'total_signals': len(recent_signals),
                'total_errors': len(recent_errors),
                'trade_breakdown': {},
                'signal_breakdown': {},
                'error_rate': len(recent_errors) / max(1, len(recent_trades) + len(recent_signals))
            }
            
            # Trade breakdown
            if recent_trades:
                buy_trades = len([t for t in recent_trades if t.get('action') == 'BUY'])
                sell_trades = len([t for t in recent_trades if t.get('action') == 'SELL'])
                total_volume = sum(t.get('value', 0) for t in recent_trades)
                
                summary['trade_breakdown'] = {
                    'buy_trades': buy_trades,
                    'sell_trades': sell_trades,
                    'total_volume': total_volume,
                    'avg_trade_size': total_volume / len(recent_trades)
                }
            
            # Signal breakdown
            if recent_signals:
                buy_signals = len([s for s in recent_signals if s.get('signal') == 'BUY'])
                sell_signals = len([s for s in recent_signals if s.get('signal') == 'SELL'])
                hold_signals = len([s for s in recent_signals if s.get('signal') == 'HOLD'])
                avg_confidence = np.mean([s.get('confidence', 0) for s in recent_signals])
                
                summary['signal_breakdown'] = {
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'hold_signals': hold_signals,
                    'avg_confidence': avg_confidence
                }
            
            return summary
            
        except Exception as e:
            self.log_error(f"Error getting log summary: {e}")
            return {}
    
    def export_logs_to_csv(self, log_type: str = 'trade', 
                          start_date: str = None, end_date: str = None) -> str:
        """Export logs to CSV file"""
        try:
            # Get appropriate logs
            if log_type == 'trade':
                logs = self.trade_logs
            elif log_type == 'signal':
                logs = self.signal_logs
            elif log_type == 'error':
                logs = self.error_logs
            elif log_type == 'performance':
                logs = self.performance_logs
            else:
                logs = []
            
            if not logs:
                return ""
            
            # Filter by date range if specified
            if start_date or end_date:
                filtered_logs = []
                for log in logs:
                    log_date = log.get('timestamp', '')
                    if start_date and log_date < start_date:
                        continue
                    if end_date and log_date > end_date:
                        continue
                    filtered_logs.append(log)
                logs = filtered_logs
            
            # Convert to DataFrame
            df = pd.DataFrame(logs)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f"{log_type}_logs_{timestamp}.csv"
            
            # Export to CSV
            df.to_csv(filename, index=False)
            
            self.log_system_event('EXPORT', f"Exported {len(logs)} {log_type} logs to {filename}")
            
            return str(filename)
            
        except Exception as e:
            self.log_error(f"Error exporting logs to CSV: {e}")
            return ""
    
    def cleanup_old_logs(self):
        """Clean up old log files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.max_log_files)
            cutoff_str = cutoff_date.strftime("%Y%m%d")
            
            deleted_count = 0
            
            for log_file in self.log_dir.glob("*.log"):
                # Extract date from filename
                try:
                    filename = log_file.stem
                    if '_' in filename:
                        date_part = filename.split('_')[-1]
                        if len(date_part) == 8 and date_part.isdigit():
                            if date_part < cutoff_str:
                                log_file.unlink()
                                deleted_count += 1
                except:
                    continue
            
            if deleted_count > 0:
                self.log_system_event(
                    'CLEANUP', 
                    f"Deleted {deleted_count} old log files"
                )
                
        except Exception as e:
            self.log_error(f"Error cleaning up old logs: {e}")
    
    def get_log_file_info(self) -> Dict[str, Any]:
        """Get information about log files"""
        try:
            log_files = []
            total_size = 0
            
            for log_file in self.log_dir.glob("*.log"):
                file_size = log_file.stat().st_size
                total_size += file_size
                
                log_files.append({
                    'filename': log_file.name,
                    'size_bytes': file_size,
                    'size_mb': file_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                })
            
            return {
                'log_directory': str(self.log_dir),
                'total_files': len(log_files),
                'total_size_mb': total_size / (1024 * 1024),
                'files': sorted(log_files, key=lambda x: x['modified'], reverse=True)
            }
            
        except Exception as e:
            self.log_error(f"Error getting log file info: {e}")
            return {}
