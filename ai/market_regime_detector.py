import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """Advanced market regime detection using multiple methodologies"""
    
    def __init__(self, lookback_period: int = 252, n_regimes: int = 3):
        self.lookback_period = lookback_period
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        self.hmm_model = None
        self.regime_features = None
        self.current_regime = 'unknown'
        self.regime_probabilities = {}
        self.regime_history = []
        
        # Regime characteristics
        self.regime_profiles = {
            'trending': {'volatility': 'low', 'momentum': 'high', 'mean_reversion': 'low'},
            'ranging': {'volatility': 'medium', 'momentum': 'low', 'mean_reversion': 'high'},
            'volatile': {'volatility': 'high', 'momentum': 'medium', 'mean_reversion': 'medium'},
            'crisis': {'volatility': 'very_high', 'momentum': 'high', 'mean_reversion': 'low'}
        }
    
    def extract_regime_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Extract features for regime detection"""
        try:
            features = pd.DataFrame(index=price_data.index)
            
            # Price-based features
            returns = price_data['close'].pct_change()
            features['returns'] = returns
            features['log_returns'] = np.log(price_data['close'] / price_data['close'].shift(1))
            
            # Volatility measures
            features['volatility_5'] = returns.rolling(5).std()
            features['volatility_20'] = returns.rolling(20).std()
            features['volatility_60'] = returns.rolling(60).std()
            features['volatility_ratio'] = features['volatility_5'] / features['volatility_20']
            
            # Momentum indicators
            features['momentum_5'] = price_data['close'] / price_data['close'].shift(5) - 1
            features['momentum_20'] = price_data['close'] / price_data['close'].shift(20) - 1
            features['momentum_60'] = price_data['close'] / price_data['close'].shift(60) - 1
            
            # Trend strength
            features['trend_strength'] = self._calculate_trend_strength(price_data['close'])
            
            # Mean reversion indicators
            features['mean_reversion'] = self._calculate_mean_reversion_strength(returns)
            features['bollinger_position'] = self._calculate_bollinger_position(price_data)
            
            # Volume-based features (if available)
            if 'volume' in price_data.columns:
                features['volume_trend'] = self._calculate_volume_trend(price_data['volume'])
                features['price_volume_trend'] = features['returns'] * features['volume_trend']
            
            # Market microstructure
            features['bid_ask_spread'] = self._estimate_bid_ask_spread(price_data)
            features['price_efficiency'] = self._calculate_price_efficiency(returns)
            
            # Cross-market features
            features['volatility_clustering'] = self._detect_volatility_clustering(returns)
            features['jump_intensity'] = self._detect_price_jumps(returns)
            
            # Regime persistence
            features['regime_persistence'] = self._calculate_regime_persistence(features)
            
            return features.fillna(method='ffill').fillna(0)
            
        except Exception as e:
            print(f"Error extracting regime features: {e}")
            return pd.DataFrame()
    
    def _calculate_trend_strength(self, prices: pd.Series) -> pd.Series:
        """Calculate trend strength using multiple timeframes"""
        try:
            # ADX-like calculation
            high_low = prices.rolling(2).apply(lambda x: x.max() - x.min())
            close_prev = abs(prices - prices.shift(1))
            
            tr = pd.concat([high_low, close_prev], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            
            # Directional movement
            dm_plus = (prices - prices.shift(1)).where(prices > prices.shift(1), 0)
            dm_minus = (prices.shift(1) - prices).where(prices < prices.shift(1), 0)
            
            di_plus = 100 * dm_plus.rolling(14).mean() / atr
            di_minus = 100 * dm_minus.rolling(14).mean() / atr
            
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(14).mean()
            
            return adx / 100  # Normalize to 0-1
            
        except Exception:
            return pd.Series(index=prices.index).fillna(0.5)
    
    def _calculate_mean_reversion_strength(self, returns: pd.Series) -> pd.Series:
        """Calculate mean reversion tendency"""
        try:
            # Hurst exponent estimation
            def hurst_exponent(ts, max_lag=20):
                lags = range(2, max_lag)
                tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0]
            
            # Rolling Hurst exponent
            window = 60
            hurst_values = []
            
            for i in range(len(returns)):
                if i < window:
                    hurst_values.append(0.5)
                else:
                    window_data = returns.iloc[i-window:i]
                    try:
                        h = hurst_exponent(window_data.values)
                        hurst_values.append(h)
                    except:
                        hurst_values.append(0.5)
            
            # Convert to mean reversion strength (inverse of trending)
            hurst_series = pd.Series(hurst_values, index=returns.index)
            mean_reversion = 1 - hurst_series  # Higher value = more mean reverting
            
            return mean_reversion.fillna(0.5)
            
        except Exception:
            return pd.Series(index=returns.index).fillna(0.5)
    
    def _calculate_bollinger_position(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate position within Bollinger Bands"""
        try:
            close = price_data['close']
            sma = close.rolling(20).mean()
            std = close.rolling(20).std()
            
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            # Position within bands (0 = lower band, 1 = upper band)
            bb_position = (close - lower_band) / (upper_band - lower_band)
            
            return bb_position.fillna(0.5)
            
        except Exception:
            return pd.Series(index=price_data.index).fillna(0.5)
    
    def _calculate_volume_trend(self, volume: pd.Series) -> pd.Series:
        """Calculate volume trend strength"""
        try:
            volume_sma = volume.rolling(20).mean()
            volume_ratio = volume / volume_sma
            
            # Smooth the ratio
            volume_trend = volume_ratio.rolling(5).mean()
            
            return (volume_trend - 1).fillna(0)  # Deviation from average
            
        except Exception:
            return pd.Series(index=volume.index).fillna(0)
    
    def _estimate_bid_ask_spread(self, price_data: pd.DataFrame) -> pd.Series:
        """Estimate bid-ask spread from price data"""
        try:
            # Use high-low range as proxy for spread
            spread_proxy = (price_data['high'] - price_data['low']) / price_data['close']
            
            return spread_proxy.rolling(5).mean().fillna(0)
            
        except Exception:
            return pd.Series(index=price_data.index).fillna(0)
    
    def _calculate_price_efficiency(self, returns: pd.Series) -> pd.Series:
        """Calculate market efficiency using autocorrelation"""
        try:
            # Rolling autocorrelation
            window = 20
            autocorr_values = []
            
            for i in range(len(returns)):
                if i < window:
                    autocorr_values.append(0)
                else:
                    window_returns = returns.iloc[i-window:i]
                    try:
                        autocorr = window_returns.autocorr(lag=1)
                        autocorr_values.append(abs(autocorr) if not np.isnan(autocorr) else 0)
                    except:
                        autocorr_values.append(0)
            
            return pd.Series(autocorr_values, index=returns.index)
            
        except Exception:
            return pd.Series(index=returns.index).fillna(0)
    
    def _detect_volatility_clustering(self, returns: pd.Series) -> pd.Series:
        """Detect volatility clustering (GARCH effects)"""
        try:
            squared_returns = returns ** 2
            
            # ARCH test - correlation of squared returns
            volatility_clustering = squared_returns.rolling(20).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
            )
            
            return volatility_clustering.fillna(0)
            
        except Exception:
            return pd.Series(index=returns.index).fillna(0)
    
    def _detect_price_jumps(self, returns: pd.Series) -> pd.Series:
        """Detect price jumps and their intensity"""
        try:
            # Rolling volatility
            vol = returns.rolling(20).std()
            
            # Jump detection threshold (3 standard deviations)
            jump_threshold = 3 * vol
            
            # Jump intensity
            jump_intensity = (abs(returns) / jump_threshold).where(
                abs(returns) > jump_threshold, 0
            )
            
            return jump_intensity.rolling(5).sum().fillna(0)
            
        except Exception:
            return pd.Series(index=returns.index).fillna(0)
    
    def _calculate_regime_persistence(self, features: pd.DataFrame) -> pd.Series:
        """Calculate how persistent current regime characteristics are"""
        try:
            # Use volatility as proxy for regime changes
            vol_changes = features['volatility_20'].pct_change().abs()
            regime_persistence = 1 / (1 + vol_changes.rolling(10).mean())
            
            return regime_persistence.fillna(0.5)
            
        except Exception:
            return pd.Series(index=features.index).fillna(0.5)
    
    def detect_regime(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect current market regime using multiple methods"""
        try:
            if len(price_data) < self.lookback_period:
                return {
                    'regime': 'insufficient_data',
                    'confidence': 0.0,
                    'probabilities': {},
                    'features': {},
                    'description': 'Not enough data for regime detection'
                }
            
            # Extract features
            features = self.extract_regime_features(price_data)
            
            if features.empty:
                return {
                    'regime': 'unknown',
                    'confidence': 0.0,
                    'probabilities': {},
                    'features': {},
                    'description': 'Feature extraction failed'
                }
            
            # Get recent features for classification
            recent_features = features.tail(20)  # Last 20 periods
            feature_vector = recent_features.mean()  # Average recent features
            
            # Method 1: Rule-based classification
            rule_based_regime = self._classify_regime_rules(feature_vector)
            
            # Method 2: Statistical clustering (if enough data)
            if len(features) > 100:
                cluster_regime = self._classify_regime_clustering(features, recent_features)
            else:
                cluster_regime = rule_based_regime
            
            # Method 3: Volatility-based regime
            volatility_regime = self._classify_volatility_regime(feature_vector)
            
            # Ensemble classification
            regime_votes = [rule_based_regime, cluster_regime, volatility_regime]
            regime_counts = pd.Series(regime_votes).value_counts()
            
            final_regime = regime_counts.index[0]
            confidence = regime_counts.iloc[0] / len(regime_votes)
            
            # Calculate regime probabilities
            probabilities = self._calculate_regime_probabilities(feature_vector)
            
            # Update history
            self.current_regime = final_regime
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': final_regime,
                'confidence': confidence,
                'features': feature_vector.to_dict()
            })
            
            # Keep only recent history
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]
            
            return {
                'regime': final_regime,
                'confidence': float(confidence),
                'probabilities': probabilities,
                'features': feature_vector.to_dict(),
                'description': self._get_regime_description(final_regime),
                'methods': {
                    'rule_based': rule_based_regime,
                    'clustering': cluster_regime,
                    'volatility': volatility_regime
                }
            }
            
        except Exception as e:
            print(f"Error in regime detection: {e}")
            return {
                'regime': 'error',
                'confidence': 0.0,
                'probabilities': {},
                'features': {},
                'description': f'Error in regime detection: {e}'
            }
    
    def _classify_regime_rules(self, features: pd.Series) -> str:
        """Rule-based regime classification"""
        try:
            vol = features.get('volatility_20', 0)
            momentum = features.get('momentum_20', 0)
            trend_strength = features.get('trend_strength', 0)
            mean_reversion = features.get('mean_reversion', 0)
            
            # Thresholds
            high_vol = vol > 0.03  # 3% daily volatility
            very_high_vol = vol > 0.05  # 5% daily volatility
            strong_trend = trend_strength > 0.6
            strong_momentum = abs(momentum) > 0.1
            high_mean_reversion = mean_reversion > 0.6
            
            if very_high_vol:
                return 'crisis'
            elif high_vol and strong_momentum:
                return 'volatile'
            elif strong_trend and not high_vol:
                return 'trending'
            elif high_mean_reversion and not strong_trend:
                return 'ranging'
            else:
                return 'mixed'
                
        except Exception:
            return 'unknown'
    
    def _classify_regime_clustering(self, features: pd.DataFrame, recent_features: pd.DataFrame) -> str:
        """Clustering-based regime classification"""
        try:
            # Select key features for clustering
            key_features = ['volatility_20', 'momentum_20', 'trend_strength', 'mean_reversion']
            available_features = [f for f in key_features if f in features.columns]
            
            if len(available_features) < 2:
                return 'unknown'
            
            # Prepare data
            cluster_data = features[available_features].dropna()
            
            if len(cluster_data) < 50:
                return 'unknown'
            
            # Normalize features
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(cluster_data)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(normalized_data)
            
            # Classify recent data
            recent_data = recent_features[available_features].dropna()
            if recent_data.empty:
                return 'unknown'
            
            recent_normalized = scaler.transform(recent_data)
            recent_cluster = kmeans.predict(recent_normalized.mean().reshape(1, -1))[0]
            
            # Map cluster to regime name
            cluster_centers = kmeans.cluster_centers_
            regime_mapping = self._map_clusters_to_regimes(cluster_centers, available_features)
            
            return regime_mapping.get(recent_cluster, 'unknown')
            
        except Exception:
            return 'unknown'
    
    def _classify_volatility_regime(self, features: pd.Series) -> str:
        """Simple volatility-based regime classification"""
        try:
            vol = features.get('volatility_20', 0)
            
            if vol > 0.05:
                return 'crisis'
            elif vol > 0.03:
                return 'volatile'
            elif vol > 0.015:
                return 'trending'
            else:
                return 'ranging'
                
        except Exception:
            return 'unknown'
    
    def _map_clusters_to_regimes(self, cluster_centers: np.ndarray, feature_names: List[str]) -> Dict[int, str]:
        """Map cluster centers to regime names"""
        try:
            regime_mapping = {}
            
            for i, center in enumerate(cluster_centers):
                # Create feature dict
                center_features = dict(zip(feature_names, center))
                
                vol_idx = feature_names.index('volatility_20') if 'volatility_20' in feature_names else 0
                trend_idx = feature_names.index('trend_strength') if 'trend_strength' in feature_names else 0
                
                vol_level = center[vol_idx]
                trend_level = center[trend_idx] if trend_idx else 0
                
                # Simple mapping based on volatility and trend
                if vol_level > 1:  # High volatility (normalized)
                    regime_mapping[i] = 'volatile'
                elif vol_level < -0.5:  # Low volatility
                    if trend_level > 0.5:
                        regime_mapping[i] = 'trending'
                    else:
                        regime_mapping[i] = 'ranging'
                else:
                    regime_mapping[i] = 'mixed'
            
            return regime_mapping
            
        except Exception:
            return {i: 'unknown' for i in range(len(cluster_centers))}
    
    def _calculate_regime_probabilities(self, features: pd.Series) -> Dict[str, float]:
        """Calculate probabilities for each regime"""
        try:
            # Simple probability calculation based on feature distances
            regimes = ['trending', 'ranging', 'volatile', 'crisis']
            probabilities = {}
            
            vol = features.get('volatility_20', 0)
            trend = features.get('trend_strength', 0)
            momentum = abs(features.get('momentum_20', 0))
            
            # Distance-based probabilities
            probabilities['trending'] = max(0, trend - vol * 2)
            probabilities['ranging'] = max(0, (1 - trend) * (1 - momentum))
            probabilities['volatile'] = min(vol * 10, 1.0)
            probabilities['crisis'] = max(0, vol * 20 - 1) if vol > 0.05 else 0
            
            # Normalize probabilities
            total = sum(probabilities.values())
            if total > 0:
                probabilities = {k: v / total for k, v in probabilities.items()}
            else:
                probabilities = {k: 0.25 for k in regimes}
            
            return probabilities
            
        except Exception:
            return {'trending': 0.25, 'ranging': 0.25, 'volatile': 0.25, 'crisis': 0.25}
    
    def _get_regime_description(self, regime: str) -> str:
        """Get description for regime"""
        descriptions = {
            'trending': 'Strong directional movement with low volatility',
            'ranging': 'Sideways movement with mean reversion tendencies',
            'volatile': 'High volatility with mixed directional signals',
            'crisis': 'Extremely high volatility, potential crisis conditions',
            'mixed': 'Mixed signals, transitional regime',
            'unknown': 'Unable to classify regime',
            'insufficient_data': 'Not enough data for regime analysis',
            'error': 'Error in regime detection'
        }
        
        return descriptions.get(regime, 'Unknown regime type')
    
    def get_regime_history(self, lookback_periods: int = 100) -> List[Dict[str, Any]]:
        """Get recent regime history"""
        return self.regime_history[-lookback_periods:]
    
    def get_current_regime_info(self) -> Dict[str, Any]:
        """Get current regime information"""
        if self.regime_history:
            return self.regime_history[-1]
        else:
            return {
                'regime': 'unknown',
                'confidence': 0.0,
                'features': {},
                'timestamp': None
            }
    
    def analyze_regime_transitions(self) -> Dict[str, Any]:
        """Analyze regime transition patterns"""
        try:
            if len(self.regime_history) < 10:
                return {'error': 'Insufficient history for transition analysis'}
            
            # Extract regime sequence
            regimes = [entry['regime'] for entry in self.regime_history]
            
            # Transition matrix
            unique_regimes = list(set(regimes))
            transition_matrix = pd.DataFrame(
                0, index=unique_regimes, columns=unique_regimes
            )
            
            for i in range(len(regimes) - 1):
                current = regimes[i]
                next_regime = regimes[i + 1]
                transition_matrix.loc[current, next_regime] += 1
            
            # Normalize to probabilities
            transition_probs = transition_matrix.div(
                transition_matrix.sum(axis=1), axis=0
            ).fillna(0)
            
            # Regime persistence
            persistence = {
                regime: transition_probs.loc[regime, regime] 
                for regime in unique_regimes
            }
            
            return {
                'transition_matrix': transition_probs.to_dict(),
                'persistence': persistence,
                'regime_counts': pd.Series(regimes).value_counts().to_dict(),
                'analysis_period': len(regimes)
            }
            
        except Exception as e:
            return {'error': f'Transition analysis failed: {e}'}