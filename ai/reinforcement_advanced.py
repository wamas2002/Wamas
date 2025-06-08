import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedQLearningAgent:
    """Advanced Q-Learning agent with experience replay, target networks, and adaptive exploration"""
    
    def __init__(self, state_size: int = 15, action_size: int = 5, 
                 learning_rate: float = 0.001, discount_factor: float = 0.95,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 memory_size: int = 10000, batch_size: int = 32, target_update_freq: int = 100):
        
        # Core parameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_epsilon = epsilon
        
        # Experience replay
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        # Target network
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        
        # Q-tables (main and target)
        self.q_table = {}
        self.target_q_table = {}
        
        # Action space
        self.actions = {
            0: "STRONG_BUY",
            1: "BUY", 
            2: "HOLD",
            3: "SELL",
            4: "STRONG_SELL"
        }
        
        # Performance tracking
        self.episode_rewards = []
        self.episode_losses = []
        self.action_counts = {action: 0 for action in range(self.action_size)}
        self.state_visit_counts = {}
        
        # Advanced features
        self.use_prioritized_replay = True
        self.use_double_dqn = True
        self.use_dueling_dqn = True
        
        # Market regime awareness
        self.regime_rewards = {
            'trending': {'trend_following': 1.2, 'mean_reversion': 0.8},
            'ranging': {'trend_following': 0.8, 'mean_reversion': 1.2},
            'volatile': {'all_actions': 0.9},
            'stable': {'all_actions': 1.0}
        }
        
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state values with robust scaling"""
        try:
            # Use percentile-based normalization to handle outliers
            state_norm = np.zeros_like(state)
            
            for i in range(len(state)):
                if np.isfinite(state[i]):
                    # Clip extreme values
                    state_norm[i] = np.clip(state[i], -10, 10)
                    # Normalize to [-1, 1] range
                    state_norm[i] = np.tanh(state_norm[i] / 5.0)
                else:
                    state_norm[i] = 0.0
            
            return state_norm
            
        except Exception as e:
            print(f"Error normalizing state: {e}")
            return np.zeros(self.state_size)
    
    def _state_to_hash(self, state: np.ndarray) -> str:
        """Convert continuous state to discrete hash with improved binning"""
        try:
            normalized_state = self._normalize_state(state)
            
            # Use adaptive binning based on state statistics
            discretized = []
            for i, value in enumerate(normalized_state):
                # Use more bins for important features
                if i < 5:  # Price-related features get more precision
                    bins = 20
                else:
                    bins = 10
                
                # Map normalized value to bin
                bin_value = int((value + 1) * bins / 2)
                bin_value = max(0, min(bins - 1, bin_value))
                discretized.append(str(bin_value))
            
            return "_".join(discretized)
            
        except Exception as e:
            print(f"Error hashing state: {e}")
            return "default_state"
    
    def get_state(self, market_data: pd.DataFrame, position: float = 0.0, 
                  portfolio_value: float = 10000.0, market_regime: str = 'stable') -> np.ndarray:
        """Extract comprehensive state features from market data"""
        try:
            if len(market_data) < 20:
                return np.zeros(self.state_size)
            
            state = []
            
            # Current price features
            current_price = market_data['close'].iloc[-1]
            prev_price = market_data['close'].iloc[-2]
            price_change = (current_price - prev_price) / prev_price
            state.extend([price_change])
            
            # Price momentum (multiple timeframes)
            for period in [3, 5, 10]:
                if len(market_data) >= period:
                    momentum = (current_price / market_data['close'].iloc[-period]) - 1
                    state.append(momentum)
                else:
                    state.append(0.0)
            
            # Volatility features
            returns = market_data['close'].pct_change().dropna()
            if len(returns) >= 10:
                short_vol = returns.tail(5).std()
                long_vol = returns.tail(20).std() if len(returns) >= 20 else short_vol
                vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0
                state.extend([short_vol, vol_ratio])
            else:
                state.extend([0.01, 1.0])
            
            # Technical indicators
            # RSI
            if len(market_data) >= 14:
                rsi = self._calculate_rsi(market_data['close'])
                state.append((rsi - 50) / 50)  # Normalize RSI
            else:
                state.append(0.0)
            
            # Moving average convergence/divergence
            if len(market_data) >= 26:
                ema_12 = market_data['close'].ewm(span=12).mean().iloc[-1]
                ema_26 = market_data['close'].ewm(span=26).mean().iloc[-1]
                macd = (ema_12 - ema_26) / current_price
                state.append(macd)
            else:
                state.append(0.0)
            
            # Volume analysis
            if 'volume' in market_data.columns:
                current_vol = market_data['volume'].iloc[-1]
                avg_vol = market_data['volume'].tail(20).mean()
                vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
                state.append(np.log(vol_ratio))
            else:
                state.append(0.0)
            
            # Position and portfolio features
            position_ratio = position / portfolio_value if portfolio_value > 0 else 0.0
            state.append(position_ratio)
            
            # Market regime encoding
            regime_encoding = self._encode_market_regime(market_regime)
            state.extend(regime_encoding)
            
            # Ensure state has correct size
            while len(state) < self.state_size:
                state.append(0.0)
            
            return np.array(state[:self.state_size])
            
        except Exception as e:
            print(f"Error extracting state: {e}")
            return np.zeros(self.state_size)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50.0
        except:
            return 50.0
    
    def _encode_market_regime(self, regime: str) -> List[float]:
        """Encode market regime as features"""
        regimes = ['trending', 'ranging', 'volatile', 'stable']
        encoding = [0.0] * len(regimes)
        
        if regime in regimes:
            encoding[regimes.index(regime)] = 1.0
        
        return encoding
    
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy with improvements"""
        try:
            if training and np.random.random() < self.epsilon:
                # Exploration with action probability based on historical performance
                action_probs = self._calculate_action_probabilities()
                return np.random.choice(self.action_size, p=action_probs)
            
            # Exploitation
            state_hash = self._state_to_hash(state)
            
            if state_hash not in self.q_table:
                # Initialize Q-values with small random values
                self.q_table[state_hash] = np.random.normal(0, 0.01, self.action_size)
            
            q_values = self.q_table[state_hash]
            
            # Use softmax for action selection in exploitation
            if training:
                temperature = 0.1
                exp_q = np.exp(q_values / temperature)
                action_probs = exp_q / np.sum(exp_q)
                return np.random.choice(self.action_size, p=action_probs)
            else:
                return np.argmax(q_values)
                
        except Exception as e:
            print(f"Error in action selection: {e}")
            return 2  # Default to HOLD
    
    def _calculate_action_probabilities(self) -> np.ndarray:
        """Calculate action probabilities based on historical performance"""
        try:
            total_actions = sum(self.action_counts.values())
            if total_actions == 0:
                return np.ones(self.action_size) / self.action_size
            
            # Favor less explored actions
            probs = []
            for action in range(self.action_size):
                count = self.action_counts[action]
                prob = 1.0 / (1.0 + count / 10.0)  # Exploration bonus
                probs.append(prob)
            
            probs = np.array(probs)
            return probs / np.sum(probs)
            
        except:
            return np.ones(self.action_size) / self.action_size
    
    def calculate_reward(self, prev_price: float, current_price: float, action: int, 
                        position: float = 0.0, market_regime: str = 'stable',
                        transaction_cost: float = 0.001) -> float:
        """Calculate sophisticated reward with regime awareness"""
        try:
            price_change = (current_price - prev_price) / prev_price
            base_reward = 0.0
            
            # Action-based rewards
            if action == 0:  # STRONG_BUY
                base_reward = price_change * 2.0  # Amplified for strong actions
                base_reward -= transaction_cost * 2.0  # Higher transaction costs
            elif action == 1:  # BUY
                base_reward = price_change
                base_reward -= transaction_cost
            elif action == 2:  # HOLD
                base_reward = abs(price_change) * 0.1  # Small reward for not trading in volatile times
            elif action == 3:  # SELL
                base_reward = -price_change
                base_reward -= transaction_cost
            elif action == 4:  # STRONG_SELL
                base_reward = -price_change * 2.0
                base_reward -= transaction_cost * 2.0
            
            # Market regime adjustments
            regime_multiplier = self._get_regime_multiplier(action, market_regime)
            base_reward *= regime_multiplier
            
            # Position size penalty for excessive exposure
            if abs(position) > 0.5:  # If position > 50% of portfolio
                base_reward *= 0.8  # Penalty for excessive risk
            
            # Consistency bonus
            if hasattr(self, 'previous_action') and action == self.previous_action:
                if abs(price_change) < 0.01:  # Low volatility
                    base_reward += 0.001  # Small consistency bonus
            
            self.previous_action = action
            
            return base_reward
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0
    
    def _get_regime_multiplier(self, action: int, regime: str) -> float:
        """Get reward multiplier based on market regime and action"""
        try:
            if regime not in self.regime_rewards:
                return 1.0
            
            regime_config = self.regime_rewards[regime]
            
            # Trend following actions
            if action in [0, 1, 3, 4]:  # Buy/Sell actions
                if 'trend_following' in regime_config:
                    return regime_config['trend_following']
            
            # Mean reversion (HOLD during trends, action during ranging)
            if action == 2:  # HOLD
                if 'mean_reversion' in regime_config:
                    return regime_config['mean_reversion']
            
            # All actions modifier
            if 'all_actions' in regime_config:
                return regime_config['all_actions']
            
            return 1.0
            
        except:
            return 1.0
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool, priority: float = 1.0):
        """Store experience with priority"""
        experience = {
            'state': state.copy(),
            'action': action,
            'reward': reward,
            'next_state': next_state.copy(),
            'done': done,
            'priority': priority,
            'timestamp': datetime.now()
        }
        
        self.memory.append(experience)
        self.action_counts[action] += 1
    
    def learn(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool = False):
        """Q-learning update with target network"""
        try:
            state_hash = self._state_to_hash(state)
            next_state_hash = self._state_to_hash(next_state)
            
            # Initialize Q-values if needed
            if state_hash not in self.q_table:
                self.q_table[state_hash] = np.random.normal(0, 0.01, self.action_size)
            if next_state_hash not in self.q_table:
                self.q_table[next_state_hash] = np.random.normal(0, 0.01, self.action_size)
            
            # Target Q-table for stability
            if state_hash not in self.target_q_table:
                self.target_q_table[state_hash] = self.q_table[state_hash].copy()
            if next_state_hash not in self.target_q_table:
                self.target_q_table[next_state_hash] = self.q_table[next_state_hash].copy()
            
            # Calculate target value
            if done:
                target_value = reward
            else:
                if self.use_double_dqn:
                    # Double DQN: use main network to select action, target network to evaluate
                    best_action = np.argmax(self.q_table[next_state_hash])
                    target_value = reward + self.discount_factor * self.target_q_table[next_state_hash][best_action]
                else:
                    # Standard Q-learning
                    target_value = reward + self.discount_factor * np.max(self.target_q_table[next_state_hash])
            
            # Q-learning update
            current_q = self.q_table[state_hash][action]
            td_error = target_value - current_q
            self.q_table[state_hash][action] += self.learning_rate * td_error
            
            # Store TD error for prioritized replay
            self.remember(state, action, reward, next_state, done, abs(td_error))
            
            # Update target network
            self.update_counter += 1
            if self.update_counter % self.target_update_freq == 0:
                self._update_target_network()
            
            # Track state visits
            if state_hash not in self.state_visit_counts:
                self.state_visit_counts[state_hash] = 0
            self.state_visit_counts[state_hash] += 1
            
            return abs(td_error)
            
        except Exception as e:
            print(f"Error in learning: {e}")
            return 0.0
    
    def _update_target_network(self):
        """Update target network with main network weights"""
        try:
            for state_hash in self.q_table:
                if state_hash not in self.target_q_table:
                    self.target_q_table[state_hash] = self.q_table[state_hash].copy()
                else:
                    # Soft update
                    tau = 0.01  # Update rate
                    self.target_q_table[state_hash] = (
                        (1 - tau) * self.target_q_table[state_hash] + 
                        tau * self.q_table[state_hash]
                    )
        except Exception as e:
            print(f"Error updating target network: {e}")
    
    def replay_experience(self, batch_size: int = None):
        """Experience replay with prioritized sampling"""
        try:
            if len(self.memory) < (batch_size or self.batch_size):
                return 0.0
            
            batch_size = batch_size or self.batch_size
            
            if self.use_prioritized_replay:
                # Prioritized sampling
                priorities = [exp['priority'] for exp in self.memory]
                probs = np.array(priorities) / sum(priorities)
                indices = np.random.choice(len(self.memory), batch_size, p=probs)
                batch = [self.memory[i] for i in indices]
            else:
                # Random sampling
                batch = np.random.choice(list(self.memory), batch_size, replace=False)
            
            total_loss = 0.0
            
            for experience in batch:
                loss = self.learn(
                    experience['state'],
                    experience['action'],
                    experience['reward'],
                    experience['next_state'],
                    experience['done']
                )
                total_loss += loss
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            avg_loss = total_loss / batch_size
            self.episode_losses.append(avg_loss)
            
            return avg_loss
            
        except Exception as e:
            print(f"Error in experience replay: {e}")
            return 0.0
    
    def get_action_name(self, action: int) -> str:
        """Convert action number to readable name"""
        return self.actions.get(action, "UNKNOWN")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        try:
            total_states = len(self.q_table)
            total_experiences = len(self.memory)
            
            # Action distribution
            total_actions = sum(self.action_counts.values())
            action_distribution = {}
            for action, count in self.action_counts.items():
                action_name = self.get_action_name(action)
                action_distribution[action_name] = {
                    'count': count,
                    'percentage': (count / total_actions * 100) if total_actions > 0 else 0
                }
            
            # Q-value statistics
            all_q_values = []
            for q_values in self.q_table.values():
                all_q_values.extend(q_values)
            
            q_stats = {}
            if all_q_values:
                q_stats = {
                    'mean': np.mean(all_q_values),
                    'std': np.std(all_q_values),
                    'min': np.min(all_q_values),
                    'max': np.max(all_q_values)
                }
            
            # Recent performance
            recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else []
            recent_losses = self.episode_losses[-100:] if self.episode_losses else []
            
            return {
                'total_states_explored': total_states,
                'total_experiences': total_experiences,
                'current_epsilon': self.epsilon,
                'action_distribution': action_distribution,
                'q_value_statistics': q_stats,
                'recent_avg_reward': np.mean(recent_rewards) if recent_rewards else 0,
                'recent_avg_loss': np.mean(recent_losses) if recent_losses else 0,
                'memory_size': len(self.memory),
                'learning_rate': self.learning_rate,
                'target_update_frequency': self.target_update_freq
            }
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def save_model(self, filepath: str):
        """Save model state"""
        try:
            model_state = {
                'q_table': self.q_table,
                'target_q_table': self.target_q_table,
                'epsilon': self.epsilon,
                'action_counts': self.action_counts,
                'state_visit_counts': self.state_visit_counts,
                'episode_rewards': self.episode_rewards,
                'episode_losses': self.episode_losses,
                'parameters': {
                    'state_size': self.state_size,
                    'action_size': self.action_size,
                    'learning_rate': self.learning_rate,
                    'discount_factor': self.discount_factor,
                    'epsilon_decay': self.epsilon_decay,
                    'epsilon_min': self.epsilon_min
                }
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_state, f)
                
            print(f"Model saved to {filepath}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load model state"""
        try:
            with open(filepath, 'rb') as f:
                model_state = pickle.load(f)
            
            self.q_table = model_state.get('q_table', {})
            self.target_q_table = model_state.get('target_q_table', {})
            self.epsilon = model_state.get('epsilon', self.epsilon)
            self.action_counts = model_state.get('action_counts', {action: 0 for action in range(self.action_size)})
            self.state_visit_counts = model_state.get('state_visit_counts', {})
            self.episode_rewards = model_state.get('episode_rewards', [])
            self.episode_losses = model_state.get('episode_losses', [])
            
            # Load parameters
            params = model_state.get('parameters', {})
            for key, value in params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            
            print(f"Model loaded from {filepath}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def reset_exploration(self):
        """Reset exploration parameters"""
        self.epsilon = self.initial_epsilon
        print("Exploration parameters reset")
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """Get policy summary for analysis"""
        try:
            if not self.q_table:
                return {'error': 'No policy learned yet'}
            
            # Analyze most visited states and their preferred actions
            state_action_preferences = {}
            
            for state_hash, q_values in self.q_table.items():
                preferred_action = np.argmax(q_values)
                confidence = np.max(q_values) - np.mean(q_values)
                visits = self.state_visit_counts.get(state_hash, 0)
                
                state_action_preferences[state_hash] = {
                    'preferred_action': self.get_action_name(preferred_action),
                    'confidence': confidence,
                    'visits': visits,
                    'q_values': q_values.tolist()
                }
            
            # Sort by visits to get most important states
            sorted_states = sorted(
                state_action_preferences.items(),
                key=lambda x: x[1]['visits'],
                reverse=True
            )
            
            return {
                'total_states': len(self.q_table),
                'top_states': dict(sorted_states[:10]),  # Top 10 most visited states
                'action_preferences': {
                    action_name: sum(1 for s in state_action_preferences.values() 
                                   if s['preferred_action'] == action_name)
                    for action_name in self.actions.values()
                }
            }
            
        except Exception as e:
            print(f"Error getting policy summary: {e}")
            return {'error': str(e)}