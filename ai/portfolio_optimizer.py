import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """Advanced portfolio optimization using multiple methodologies"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.returns_data = None
        self.covariance_matrix = None
        self.expected_returns = None
        self.optimization_history = []
        
    def calculate_returns(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate returns for multiple assets"""
        try:
            returns_dict = {}
            
            for symbol, data in price_data.items():
                if 'close' in data.columns and len(data) > 1:
                    returns = data['close'].pct_change().dropna()
                    returns_dict[symbol] = returns
            
            if not returns_dict:
                return pd.DataFrame()
            
            # Align all return series by timestamp
            returns_df = pd.DataFrame(returns_dict)
            returns_df = returns_df.dropna()
            
            self.returns_data = returns_df
            return returns_df
            
        except Exception as e:
            print(f"Error calculating returns: {e}")
            return pd.DataFrame()
    
    def estimate_expected_returns(self, returns_df: pd.DataFrame, method: str = 'mean') -> pd.Series:
        """Estimate expected returns using various methods"""
        try:
            if method == 'mean':
                # Simple historical mean
                expected_returns = returns_df.mean() * 252  # Annualized
                
            elif method == 'ewm':
                # Exponentially weighted mean
                expected_returns = returns_df.ewm(span=60).mean().iloc[-1] * 252
                
            elif method == 'capm':
                # CAPM-based estimation (using first asset as market proxy)
                market_returns = returns_df.iloc[:, 0]  # Use first asset as market
                expected_returns = pd.Series(index=returns_df.columns)
                
                for asset in returns_df.columns:
                    asset_returns = returns_df[asset]
                    beta = np.cov(asset_returns, market_returns)[0, 1] / np.var(market_returns)
                    market_premium = market_returns.mean() * 252 - self.risk_free_rate
                    expected_returns[asset] = self.risk_free_rate + beta * market_premium
                    
            elif method == 'black_litterman':
                # Simplified Black-Litterman
                market_caps = np.ones(len(returns_df.columns))  # Equal market caps assumption
                expected_returns = self._black_litterman_returns(returns_df, market_caps)
                
            else:
                expected_returns = returns_df.mean() * 252
            
            self.expected_returns = expected_returns
            return expected_returns
            
        except Exception as e:
            print(f"Error estimating expected returns: {e}")
            return pd.Series(index=returns_df.columns).fillna(0.1)
    
    def estimate_covariance_matrix(self, returns_df: pd.DataFrame, method: str = 'ledoit_wolf') -> pd.DataFrame:
        """Estimate covariance matrix using various methods"""
        try:
            if method == 'sample':
                # Sample covariance matrix
                cov_matrix = returns_df.cov() * 252  # Annualized
                
            elif method == 'ledoit_wolf':
                # Ledoit-Wolf shrinkage estimator
                lw = LedoitWolf()
                cov_array = lw.fit(returns_df.values).covariance_ * 252
                cov_matrix = pd.DataFrame(cov_array, index=returns_df.columns, columns=returns_df.columns)
                
            elif method == 'ewm':
                # Exponentially weighted covariance
                cov_matrix = returns_df.ewm(span=60).cov().iloc[-len(returns_df.columns):] * 252
                
            elif method == 'robust':
                # Robust covariance estimation
                from sklearn.covariance import MinCovDet
                robust_cov = MinCovDet()
                cov_array = robust_cov.fit(returns_df.values).covariance_ * 252
                cov_matrix = pd.DataFrame(cov_array, index=returns_df.columns, columns=returns_df.columns)
                
            else:
                cov_matrix = returns_df.cov() * 252
            
            self.covariance_matrix = cov_matrix
            return cov_matrix
            
        except Exception as e:
            print(f"Error estimating covariance matrix: {e}")
            # Fallback to simple correlation with assumed volatilities
            correlation = returns_df.corr()
            volatilities = returns_df.std() * np.sqrt(252)
            cov_matrix = correlation.multiply(volatilities, axis=0).multiply(volatilities, axis=1)
            self.covariance_matrix = cov_matrix
            return cov_matrix
    
    def optimize_portfolio(self, returns_df: pd.DataFrame, 
                          optimization_method: str = 'max_sharpe',
                          constraints: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize portfolio weights using various methods"""
        try:
            if returns_df.empty or len(returns_df.columns) < 2:
                return {'error': 'Insufficient data for optimization'}
            
            # Estimate parameters
            expected_returns = self.estimate_expected_returns(returns_df)
            cov_matrix = self.estimate_covariance_matrix(returns_df)
            
            n_assets = len(returns_df.columns)
            
            # Default constraints
            if constraints is None:
                constraints = {
                    'max_weight': 0.4,
                    'min_weight': 0.0,
                    'long_only': True
                }
            
            # Optimization based on method
            if optimization_method == 'max_sharpe':
                weights = self._optimize_max_sharpe(expected_returns, cov_matrix, constraints)
                
            elif optimization_method == 'min_variance':
                weights = self._optimize_min_variance(cov_matrix, constraints)
                
            elif optimization_method == 'risk_parity':
                weights = self._optimize_risk_parity(cov_matrix, constraints)
                
            elif optimization_method == 'max_diversification':
                weights = self._optimize_max_diversification(cov_matrix, constraints)
                
            elif optimization_method == 'black_litterman':
                weights = self._optimize_black_litterman(returns_df, constraints)
                
            elif optimization_method == 'hierarchical_risk_parity':
                weights = self._optimize_hrp(returns_df, constraints)
                
            else:
                weights = self._optimize_equal_weight(n_assets)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(
                weights, expected_returns, cov_matrix
            )
            
            # Store optimization result
            optimization_result = {
                'weights': weights.to_dict() if hasattr(weights, 'to_dict') else dict(zip(returns_df.columns, weights)),
                'method': optimization_method,
                'metrics': portfolio_metrics,
                'timestamp': pd.Timestamp.now(),
                'assets': list(returns_df.columns),
                'constraints': constraints
            }
            
            self.optimization_history.append(optimization_result)
            
            return optimization_result
            
        except Exception as e:
            print(f"Error in portfolio optimization: {e}")
            return {'error': f'Optimization failed: {e}'}
    
    def _optimize_max_sharpe(self, expected_returns: pd.Series, 
                           cov_matrix: pd.DataFrame, 
                           constraints: Dict[str, Any]) -> pd.Series:
        """Optimize for maximum Sharpe ratio"""
        n_assets = len(expected_returns)
        
        # Objective function (negative Sharpe ratio)
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_std = np.sqrt(portfolio_variance)
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
            return -sharpe_ratio  # Negative for minimization
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Bounds
        bounds = [(constraints.get('min_weight', 0), constraints.get('max_weight', 1)) 
                 for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        return pd.Series(result.x, index=expected_returns.index)
    
    def _optimize_min_variance(self, cov_matrix: pd.DataFrame, 
                             constraints: Dict[str, Any]) -> pd.Series:
        """Optimize for minimum variance"""
        n_assets = len(cov_matrix)
        
        # Objective function
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = [(constraints.get('min_weight', 0), constraints.get('max_weight', 1)) 
                 for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        return pd.Series(result.x, index=cov_matrix.index)
    
    def _optimize_risk_parity(self, cov_matrix: pd.DataFrame, 
                            constraints: Dict[str, Any]) -> pd.Series:
        """Optimize for risk parity (equal risk contribution)"""
        n_assets = len(cov_matrix)
        
        def objective(weights):
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            marginal_contrib = np.dot(cov_matrix, weights)
            contrib = weights * marginal_contrib / portfolio_variance
            
            # Target equal risk contribution (1/n for each asset)
            target_contrib = np.array([1/n_assets] * n_assets)
            
            # Sum of squared deviations from target
            return np.sum((contrib - target_contrib) ** 2)
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = [(constraints.get('min_weight', 0.01), constraints.get('max_weight', 1)) 
                 for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        return pd.Series(result.x, index=cov_matrix.index)
    
    def _optimize_max_diversification(self, cov_matrix: pd.DataFrame, 
                                    constraints: Dict[str, Any]) -> pd.Series:
        """Optimize for maximum diversification ratio"""
        n_assets = len(cov_matrix)
        asset_volatilities = np.sqrt(np.diag(cov_matrix))
        
        def objective(weights):
            weighted_avg_vol = np.sum(weights * asset_volatilities)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            diversification_ratio = weighted_avg_vol / portfolio_vol
            return -diversification_ratio  # Negative for minimization
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = [(constraints.get('min_weight', 0), constraints.get('max_weight', 1)) 
                 for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        return pd.Series(result.x, index=cov_matrix.index)
    
    def _optimize_black_litterman(self, returns_df: pd.DataFrame, 
                                constraints: Dict[str, Any]) -> pd.Series:
        """Black-Litterman optimization"""
        try:
            # Simplified Black-Litterman implementation
            n_assets = len(returns_df.columns)
            
            # Market cap weights (equal assumption)
            market_weights = np.array([1/n_assets] * n_assets)
            
            # Risk aversion parameter
            risk_aversion = 3.0
            
            # Prior returns (reverse optimization)
            cov_matrix = self.estimate_covariance_matrix(returns_df)
            prior_returns = risk_aversion * np.dot(cov_matrix, market_weights)
            
            # Views (simplified - no views)
            # In practice, you would incorporate analyst views here
            
            # Black-Litterman expected returns (without views, equals prior)
            bl_returns = pd.Series(prior_returns, index=returns_df.columns)
            
            # Optimize using Black-Litterman returns
            return self._optimize_max_sharpe(bl_returns, cov_matrix, constraints)
            
        except Exception:
            # Fallback to equal weights
            return self._optimize_equal_weight(len(returns_df.columns))
    
    def _optimize_hrp(self, returns_df: pd.DataFrame, 
                     constraints: Dict[str, Any]) -> pd.Series:
        """Hierarchical Risk Parity optimization"""
        try:
            from scipy.cluster.hierarchy import linkage, dendrogram
            from scipy.spatial.distance import squareform
            
            # Calculate correlation matrix
            corr_matrix = returns_df.corr()
            
            # Distance matrix
            distance_matrix = np.sqrt((1 - corr_matrix) / 2)
            
            # Hierarchical clustering
            linkage_matrix = linkage(squareform(distance_matrix), method='ward')
            
            # Get cluster order
            def get_cluster_order(linkage_matrix, num_assets):
                cluster_order = []
                
                def traverse(node_id):
                    if node_id < num_assets:
                        cluster_order.append(node_id)
                    else:
                        left_child = int(linkage_matrix[node_id - num_assets, 0])
                        right_child = int(linkage_matrix[node_id - num_assets, 1])
                        traverse(left_child)
                        traverse(right_child)
                
                traverse(2 * num_assets - 2)
                return cluster_order
            
            # Get cluster weights using recursive bisection
            cluster_order = get_cluster_order(linkage_matrix, len(returns_df.columns))
            weights = self._get_cluster_weights(returns_df.iloc[:, cluster_order])
            
            # Reorder weights to original asset order
            weight_dict = dict(zip([returns_df.columns[i] for i in cluster_order], weights))
            final_weights = pd.Series([weight_dict[col] for col in returns_df.columns], 
                                    index=returns_df.columns)
            
            return final_weights
            
        except Exception:
            # Fallback to equal weights
            return self._optimize_equal_weight(len(returns_df.columns))
    
    def _get_cluster_weights(self, returns_cluster: pd.DataFrame) -> np.ndarray:
        """Calculate weights for clustered assets using recursive bisection"""
        if len(returns_cluster.columns) == 1:
            return np.array([1.0])
        
        # Split into two clusters
        mid = len(returns_cluster.columns) // 2
        left_cluster = returns_cluster.iloc[:, :mid]
        right_cluster = returns_cluster.iloc[:, mid:]
        
        # Calculate cluster variances
        left_var = self._calculate_cluster_variance(left_cluster)
        right_var = self._calculate_cluster_variance(right_cluster)
        
        # Allocate weights inversely proportional to variance
        total_var = left_var + right_var
        left_weight = right_var / total_var
        right_weight = left_var / total_var
        
        # Recursively calculate weights for sub-clusters
        left_weights = self._get_cluster_weights(left_cluster) * left_weight
        right_weights = self._get_cluster_weights(right_cluster) * right_weight
        
        return np.concatenate([left_weights, right_weights])
    
    def _calculate_cluster_variance(self, cluster_returns: pd.DataFrame) -> float:
        """Calculate variance of a cluster"""
        if len(cluster_returns.columns) == 1:
            return cluster_returns.var().iloc[0]
        
        # Equal weights within cluster
        weights = np.array([1/len(cluster_returns.columns)] * len(cluster_returns.columns))
        cluster_cov = cluster_returns.cov()
        cluster_variance = np.dot(weights.T, np.dot(cluster_cov, weights))
        
        return cluster_variance
    
    def _optimize_equal_weight(self, n_assets: int) -> pd.Series:
        """Equal weight portfolio"""
        weights = np.array([1/n_assets] * n_assets)
        return pd.Series(weights)
    
    def _black_litterman_returns(self, returns_df: pd.DataFrame, 
                               market_caps: np.ndarray) -> pd.Series:
        """Calculate Black-Litterman expected returns"""
        try:
            # Market cap weights
            market_weights = market_caps / np.sum(market_caps)
            
            # Covariance matrix
            cov_matrix = returns_df.cov() * 252
            
            # Risk aversion (market implied)
            market_return = np.sum(market_weights * returns_df.mean() * 252)
            market_variance = np.dot(market_weights.T, np.dot(cov_matrix, market_weights))
            risk_aversion = (market_return - self.risk_free_rate) / market_variance
            
            # Implied equilibrium returns
            implied_returns = risk_aversion * np.dot(cov_matrix, market_weights)
            
            return pd.Series(implied_returns, index=returns_df.columns)
            
        except Exception:
            return returns_df.mean() * 252
    
    def _calculate_portfolio_metrics(self, weights: pd.Series, 
                                   expected_returns: pd.Series, 
                                   cov_matrix: pd.DataFrame) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        try:
            # Portfolio return and risk
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Sharpe ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            
            # Diversification ratio
            asset_volatilities = np.sqrt(np.diag(cov_matrix))
            weighted_avg_vol = np.sum(weights * asset_volatilities)
            diversification_ratio = weighted_avg_vol / portfolio_volatility
            
            # Risk contributions
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_variance
            
            # Concentration metrics
            herfindahl_index = np.sum(weights ** 2)
            effective_num_assets = 1 / herfindahl_index
            
            return {
                'expected_return': float(portfolio_return),
                'volatility': float(portfolio_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'diversification_ratio': float(diversification_ratio),
                'herfindahl_index': float(herfindahl_index),
                'effective_num_assets': float(effective_num_assets),
                'max_weight': float(weights.max()),
                'min_weight': float(weights.min()),
                'risk_contributions': risk_contrib.to_dict() if hasattr(risk_contrib, 'to_dict') else {}
            }
            
        except Exception as e:
            print(f"Error calculating portfolio metrics: {e}")
            return {}
    
    def rebalance_portfolio(self, current_weights: Dict[str, float], 
                          target_weights: Dict[str, float],
                          threshold: float = 0.05) -> Dict[str, Any]:
        """Determine rebalancing trades"""
        try:
            trades = {}
            total_drift = 0
            
            for asset in target_weights:
                current_weight = current_weights.get(asset, 0)
                target_weight = target_weights[asset]
                drift = abs(current_weight - target_weight)
                total_drift += drift
                
                if drift > threshold:
                    trades[asset] = {
                        'current_weight': current_weight,
                        'target_weight': target_weight,
                        'trade_amount': target_weight - current_weight,
                        'drift': drift
                    }
            
            return {
                'trades_needed': len(trades) > 0,
                'total_drift': total_drift,
                'trades': trades,
                'rebalancing_cost': self._estimate_rebalancing_cost(trades)
            }
            
        except Exception as e:
            return {'error': f'Rebalancing calculation failed: {e}'}
    
    def _estimate_rebalancing_cost(self, trades: Dict[str, Any]) -> float:
        """Estimate transaction costs for rebalancing"""
        try:
            total_cost = 0
            transaction_cost_rate = 0.001  # 0.1% per trade
            
            for asset, trade_info in trades.items():
                trade_amount = abs(trade_info['trade_amount'])
                cost = trade_amount * transaction_cost_rate
                total_cost += cost
            
            return total_cost
            
        except Exception:
            return 0.0
    
    def get_optimization_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent optimization history"""
        return self.optimization_history[-limit:]
    
    def analyze_portfolio_performance(self, weights: Dict[str, float], 
                                    returns_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze portfolio performance over time"""
        try:
            # Convert weights to series
            weight_series = pd.Series(weights)
            
            # Calculate portfolio returns
            portfolio_returns = (returns_df * weight_series).sum(axis=1)
            
            # Performance metrics
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
            annualized_volatility = portfolio_returns.std() * np.sqrt(252)
            sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
            
            # Drawdown analysis
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Risk metrics
            var_95 = portfolio_returns.quantile(0.05)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
            
            return {
                'total_return': float(total_return),
                'annualized_return': float(annualized_return),
                'annualized_volatility': float(annualized_volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'var_95': float(var_95),
                'cvar_95': float(cvar_95),
                'portfolio_returns': portfolio_returns.to_dict()
            }
            
        except Exception as e:
            return {'error': f'Performance analysis failed: {e}'}