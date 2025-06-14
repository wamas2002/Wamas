"""
Feedback Analyzer - Intelligent Signal Pattern Recognition
Analyzes signal performance to derive actionable trading rules
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class FeedbackAnalyzer:
    """Analyze signal performance patterns and generate actionable insights"""
    
    def __init__(self, db_path: str = 'feedback_learning.db'):
        self.db_path = db_path
        self.analysis_rules = []
    
    def analyze_signal_patterns(self, min_samples: int = 10) -> List[Dict]:
        """Analyze signal patterns to identify successful/failing conditions"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get completed trades for analysis
            df = pd.read_sql_query('''
                SELECT * FROM trade_executions 
                WHERE exit_price IS NOT NULL 
                AND timestamp > datetime('now', '-30 days')
            ''', conn)
            
            conn.close()
            
            if len(df) < min_samples:
                return [{'rule': 'Insufficient data', 'confidence': 0, 'sample_size': len(df)}]
            
            patterns = []
            
            # Analyze confidence vs success rate
            confidence_pattern = self.analyze_confidence_correlation(df)
            if confidence_pattern:
                patterns.append(confidence_pattern)
            
            # Analyze market regime performance
            regime_patterns = self.analyze_regime_performance(df)
            patterns.extend(regime_patterns)
            
            # Analyze volume conditions
            volume_pattern = self.analyze_volume_impact(df)
            if volume_pattern:
                patterns.append(volume_pattern)
            
            # Analyze volatility impact
            volatility_pattern = self.analyze_volatility_impact(df)
            if volatility_pattern:
                patterns.append(volatility_pattern)
            
            # Analyze holding period patterns
            timing_pattern = self.analyze_timing_patterns(df)
            if timing_pattern:
                patterns.append(timing_pattern)
            
            return patterns if patterns else [{'rule': 'No significant patterns detected', 'confidence': 0}]
            
        except Exception as e:
            logger.error(f"Signal pattern analysis failed: {e}")
            return [{'rule': 'Analysis error', 'confidence': 0, 'error': str(e)}]
    
    def analyze_confidence_correlation(self, df: pd.DataFrame) -> Dict:
        """Analyze correlation between signal confidence and success"""
        try:
            if len(df) < 10:
                return None
            
            # Group by confidence ranges
            df['confidence_range'] = pd.cut(df['confidence'], 
                                          bins=[0, 70, 80, 90, 100], 
                                          labels=['60-70%', '70-80%', '80-90%', '90-100%'])
            
            confidence_stats = df.groupby('confidence_range').agg({
                'win_loss': lambda x: (x == 'win').mean(),
                'pnl_pct': 'mean',
                'confidence': 'count'
            }).round(3)
            
            confidence_stats.columns = ['win_rate', 'avg_pnl', 'sample_size']
            
            # Find best performing confidence range
            best_range = confidence_stats['win_rate'].idxmax()
            best_win_rate = confidence_stats.loc[best_range, 'win_rate']
            
            if best_win_rate > 0.7 and confidence_stats.loc[best_range, 'sample_size'] >= 5:
                return {
                    'rule': f'Signals with {best_range} confidence show {best_win_rate*100:.1f}% win rate',
                    'confidence': min(0.9, best_win_rate * 1.2),
                    'sample_size': int(confidence_stats.loc[best_range, 'sample_size']),
                    'category': 'confidence_optimization',
                    'actionable': f'Prioritize signals with confidence in {best_range} range'
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Confidence correlation analysis failed: {e}")
            return None
    
    def analyze_regime_performance(self, df: pd.DataFrame) -> List[Dict]:
        """Analyze performance by market regime"""
        try:
            patterns = []
            
            regime_stats = df.groupby(['market_regime', 'signal_type']).agg({
                'win_loss': lambda x: (x == 'win').mean(),
                'pnl_pct': 'mean',
                'confidence': 'count'
            }).round(3)
            
            regime_stats.columns = ['win_rate', 'avg_pnl', 'sample_size']
            
            for (regime, signal_type), stats in regime_stats.iterrows():
                if stats['sample_size'] >= 5:  # Minimum sample size
                    if stats['win_rate'] > 0.75:  # High success rate
                        patterns.append({
                            'rule': f'{signal_type} signals in {regime} market: {stats["win_rate"]*100:.1f}% win rate',
                            'confidence': min(0.9, stats['win_rate'] * 1.1),
                            'sample_size': int(stats['sample_size']),
                            'category': 'regime_optimization',
                            'actionable': f'Favor {signal_type} signals during {regime} market conditions'
                        })
                    elif stats['win_rate'] < 0.4:  # Poor performance
                        patterns.append({
                            'rule': f'Avoid {signal_type} signals in {regime} market: {stats["win_rate"]*100:.1f}% win rate',
                            'confidence': min(0.8, (1 - stats['win_rate']) * 1.1),
                            'sample_size': int(stats['sample_size']),
                            'category': 'regime_avoidance',
                            'actionable': f'Reduce or avoid {signal_type} signals during {regime} conditions'
                        })
            
            return patterns
            
        except Exception as e:
            logger.warning(f"Regime performance analysis failed: {e}")
            return []
    
    def analyze_volume_impact(self, df: pd.DataFrame) -> Dict:
        """Analyze impact of volume conditions on signal success"""
        try:
            if 'volume_ratio' not in df.columns or len(df) < 10:
                return None
            
            # Group by volume conditions
            df['volume_category'] = pd.cut(df['volume_ratio'], 
                                         bins=[0, 0.8, 1.5, 3.0, float('inf')], 
                                         labels=['Low', 'Normal', 'High', 'Very High'])
            
            volume_stats = df.groupby('volume_category').agg({
                'win_loss': lambda x: (x == 'win').mean(),
                'pnl_pct': 'mean',
                'confidence': 'count'
            }).round(3)
            
            volume_stats.columns = ['win_rate', 'avg_pnl', 'sample_size']
            
            # Find patterns
            for category, stats in volume_stats.iterrows():
                if stats['sample_size'] >= 5:
                    if stats['win_rate'] > 0.7:
                        return {
                            'rule': f'{category} volume conditions: {stats["win_rate"]*100:.1f}% win rate',
                            'confidence': min(0.8, stats['win_rate'] * 1.1),
                            'sample_size': int(stats['sample_size']),
                            'category': 'volume_optimization',
                            'actionable': f'Favor signals during {category.lower()} volume periods'
                        }
                    elif stats['win_rate'] < 0.4:
                        return {
                            'rule': f'Avoid trading during {category.lower()} volume: {stats["win_rate"]*100:.1f}% win rate',
                            'confidence': min(0.8, (1 - stats['win_rate']) * 1.1),
                            'sample_size': int(stats['sample_size']),
                            'category': 'volume_avoidance',
                            'actionable': f'Reduce signal confidence during {category.lower()} volume'
                        }
            
            return None
            
        except Exception as e:
            logger.warning(f"Volume impact analysis failed: {e}")
            return None
    
    def analyze_volatility_impact(self, df: pd.DataFrame) -> Dict:
        """Analyze impact of market volatility on performance"""
        try:
            if 'volatility' not in df.columns or len(df) < 10:
                return None
            
            # Group by volatility levels
            df['volatility_level'] = pd.cut(df['volatility'], 
                                          bins=[0, 0.02, 0.04, 0.08, float('inf')], 
                                          labels=['Low', 'Normal', 'High', 'Extreme'])
            
            volatility_stats = df.groupby('volatility_level').agg({
                'win_loss': lambda x: (x == 'win').mean(),
                'pnl_pct': 'mean',
                'confidence': 'count'
            }).round(3)
            
            volatility_stats.columns = ['win_rate', 'avg_pnl', 'sample_size']
            
            # Look for extreme volatility impact
            if 'Extreme' in volatility_stats.index:
                extreme_stats = volatility_stats.loc['Extreme']
                if extreme_stats['sample_size'] >= 3 and extreme_stats['win_rate'] < 0.4:
                    return {
                        'rule': f'Extreme volatility reduces win rate to {extreme_stats["win_rate"]*100:.1f}%',
                        'confidence': 0.75,
                        'sample_size': int(extreme_stats['sample_size']),
                        'category': 'volatility_warning',
                        'actionable': 'Avoid or reduce position size during extreme volatility'
                    }
            
            return None
            
        except Exception as e:
            logger.warning(f"Volatility impact analysis failed: {e}")
            return None
    
    def analyze_timing_patterns(self, df: pd.DataFrame) -> Dict:
        """Analyze holding period and timing patterns"""
        try:
            if 'holding_period_hours' not in df.columns or len(df) < 10:
                return None
            
            # Group by holding periods
            df['holding_category'] = pd.cut(df['holding_period_hours'], 
                                          bins=[0, 2, 8, 24, 72, float('inf')], 
                                          labels=['<2h', '2-8h', '8-24h', '1-3d', '>3d'])
            
            timing_stats = df.groupby('holding_category').agg({
                'win_loss': lambda x: (x == 'win').mean(),
                'pnl_pct': 'mean',
                'confidence': 'count'
            }).round(3)
            
            timing_stats.columns = ['win_rate', 'avg_pnl', 'sample_size']
            
            # Find optimal holding period
            best_period = timing_stats['win_rate'].idxmax()
            best_stats = timing_stats.loc[best_period]
            
            if best_stats['sample_size'] >= 5 and best_stats['win_rate'] > 0.65:
                return {
                    'rule': f'Optimal holding period {best_period}: {best_stats["win_rate"]*100:.1f}% win rate',
                    'confidence': min(0.8, best_stats['win_rate'] * 1.1),
                    'sample_size': int(best_stats['sample_size']),
                    'category': 'timing_optimization',
                    'actionable': f'Target {best_period} holding periods for better performance'
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Timing pattern analysis failed: {e}")
            return None
    
    def generate_actionable_rules(self) -> List[str]:
        """Generate human-readable actionable trading rules"""
        try:
            patterns = self.analyze_signal_patterns()
            rules = []
            
            for pattern in patterns:
                if pattern.get('confidence', 0) > 0.6 and pattern.get('sample_size', 0) >= 5:
                    actionable = pattern.get('actionable')
                    if actionable:
                        confidence_pct = pattern['confidence'] * 100
                        sample_size = pattern['sample_size']
                        rules.append(f"{actionable} (Confidence: {confidence_pct:.0f}%, Sample: {sample_size})")
            
            return rules if rules else ["Continue collecting performance data for rule generation"]
            
        except Exception as e:
            logger.error(f"Rule generation failed: {e}")
            return ["Rule generation temporarily unavailable"]
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Overall metrics
            overall_query = '''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN win_loss = 'win' THEN 1 ELSE 0 END) as wins,
                    AVG(pnl_pct) as avg_pnl,
                    AVG(confidence) as avg_confidence,
                    MIN(timestamp) as first_trade,
                    MAX(timestamp) as last_trade
                FROM trade_executions 
                WHERE exit_price IS NOT NULL
            '''
            
            overall_stats = pd.read_sql_query(overall_query, conn).iloc[0].to_dict()
            
            # Calculate win rate
            if overall_stats['total_trades'] > 0:
                overall_stats['win_rate'] = overall_stats['wins'] / overall_stats['total_trades']
            else:
                overall_stats['win_rate'] = 0
            
            # Recent performance (last 7 days)
            recent_query = '''
                SELECT 
                    COUNT(*) as recent_trades,
                    SUM(CASE WHEN win_loss = 'win' THEN 1 ELSE 0 END) as recent_wins,
                    AVG(pnl_pct) as recent_avg_pnl
                FROM trade_executions 
                WHERE exit_price IS NOT NULL 
                AND timestamp > datetime('now', '-7 days')
            '''
            
            recent_stats = pd.read_sql_query(recent_query, conn).iloc[0].to_dict()
            
            if recent_stats['recent_trades'] > 0:
                recent_stats['recent_win_rate'] = recent_stats['recent_wins'] / recent_stats['recent_trades']
            else:
                recent_stats['recent_win_rate'] = 0
            
            conn.close()
            
            return {
                'overall': overall_stats,
                'recent': recent_stats,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Performance summary failed: {e}")
            return {'error': str(e)}