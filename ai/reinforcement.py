import numpy as np
import pandas as pd
import random
from collections import deque
from typing import Dict, List, Tuple, Any
from datetime import datetime

class QLearningAgent:
    """Q-Learning agent for cryptocurrency trading"""
    
    def __init__(self, state_size: int = 10, action_size: int = 3, 
                 learning_rate: float = 0.01, discount_factor: float = 0.95,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        self.state_size = state_size
        self.action_size = action_size  # 0: Hold, 1: Buy, 2: Sell
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Q-table (state_hash -> action_values)
        self.q_table = {}
        self.memory = deque(maxlen=10000)
        self.action_history = []
        self.reward_history = []
        
        # State normalization parameters
        self.state_normalizers = {}
        
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize state values to [0, 1] range"""
        try:
            normalized_state = np.zeros_like(state)
            
            for i, value in enumerate(state):
                key = f'feature_{i}'
                
                if key not in self.state_normalizers:
                    self.state_normalizers[key] = {'min': value, 'max': value}
                else:
                    self.state_normalizers[key]['min'] = min(self.state_normalizers[key]['min'], value)
                    self.state_normalizers[key]['max'] = max(self.state_normalizers[key]['max'], value)
                
                # Normalize using min-max scaling
                min_val = self.state_normalizers[key]['min']
                max_val = self.state_normalizers[key]['max']
                
                if max_val > min_val:
                    normalized_state[i] = (value - min_val) / (max_val - min_val)
                else:
                    normalized_state[i] = 0.5  # Default middle value
            
            return normalized_state
            
        except Exception as e:
            print(f"Error normalizing state: {e}")
            return state
    
    def _state_to_hash(self, state: np.ndarray) -> str:
        """Convert continuous state to discrete hash for Q-table"""
        try:
            # Discretize state values to reduce Q-table size
            discretized = np.round(state * 10).astype(int)  # 10 bins per feature
            return str(discretized.tolist())
        except:
            return "default_state"
    
    def get_state(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract state features from market data"""
        try:
            if len(market_data) < 20:
                return np.zeros(self.state_size)
            
            features = []
            
            # Price momentum features
            returns = market_data['close'].pct_change().fillna(0)
            features.append(returns.iloc[-1])  # Current return
            features.append(returns.rolling(5).mean().iloc[-1])  # 5-period avg return
            
            # Volatility features
            features.append(returns.rolling(10).std().iloc[-1])  # Volatility
            
            # Trend features
            sma_short = market_data['close'].rolling(5).mean()
            sma_long = market_data['close'].rolling(20).mean()
            trend = (sma_short.iloc[-1] / sma_long.iloc[-1] - 1) if sma_long.iloc[-1] != 0 else 0
            features.append(trend)
            
            # Volume features
            volume_ratio = (market_data['volume'].iloc[-1] / 
                          market_data['volume'].rolling(20).mean().iloc[-1]) if market_data['volume'].rolling(20).mean().iloc[-1] != 0 else 1
            features.append(volume_ratio)
            
            # Price position in recent range
            high_20 = market_data['high'].rolling(20).max().iloc[-1]
            low_20 = market_data['low'].rolling(20).min().iloc[-1]
            price_position = ((market_data['close'].iloc[-1] - low_20) / 
                            (high_20 - low_20)) if (high_20 - low_20) != 0 else 0.5
            features.append(price_position)
            
            # Add more features to reach state_size
            while len(features) < self.state_size:
                if len(features) < len(returns):
                    features.append(returns.iloc[-len(features)-1])
                else:
                    features.append(0.0)
            
            # Ensure we have exactly state_size features
            features = features[:self.state_size]
            
            # Replace NaN and inf values
            features = np.array(features)
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return self._normalize_state(features)
            
        except Exception as e:
            print(f"Error getting state: {e}")
            return np.zeros(self.state_size)
    
    def act(self, state: np.ndarray) -> int:
        """Choose action using epsilon-greedy policy"""
        try:
            state_hash = self._state_to_hash(state)
            
            # Exploration vs exploitation
            if random.random() < self.epsilon:
                action = random.randrange(self.action_size)
            else:
                # Get Q-values for current state
                if state_hash in self.q_table:
                    q_values = self.q_table[state_hash]
                    action = np.argmax(q_values)
                else:
                    # Initialize new state with random action
                    action = random.randrange(self.action_size)
                    self.q_table[state_hash] = np.zeros(self.action_size)
            
            self.action_history.append(action)
            return action
            
        except Exception as e:
            print(f"Error in act: {e}")
            return 0  # Default to hold
    
    def calculate_reward(self, prev_price: float, current_price: float, 
                        action: int, position: float = 0.0) -> float:
        """Calculate reward based on action and price movement"""
        try:
            if prev_price <= 0 or current_price <= 0:
                return 0.0
            
            price_change = (current_price - prev_price) / prev_price
            
            # Base reward based on price movement and action
            if action == 1:  # Buy
                reward = price_change  # Positive if price goes up
                # Penalty for buying at high prices
                if price_change < -0.01:  # -1% threshold
                    reward -= 0.1
            elif action == 2:  # Sell
                reward = -price_change  # Positive if price goes down
                # Penalty for selling at low prices
                if price_change > 0.01:  # +1% threshold
                    reward -= 0.1
            else:  # Hold
                reward = -abs(price_change) * 0.1  # Small penalty for inaction during volatile moves
            
            # Additional reward for maintaining profitable position
            if position != 0:
                position_return = position * price_change
                reward += position_return * 0.5
            
            # Normalize reward
            reward = max(-1.0, min(1.0, reward))
            
            self.reward_history.append(reward)
            return reward
            
        except Exception as e:
            print(f"Error calculating reward: {e}")
            return 0.0
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def learn(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool = False):
        """Update Q-values using Q-learning algorithm"""
        try:
            state_hash = self._state_to_hash(state)
            next_state_hash = self._state_to_hash(next_state)
            
            # Initialize Q-values if not exists
            if state_hash not in self.q_table:
                self.q_table[state_hash] = np.zeros(self.action_size)
            if next_state_hash not in self.q_table:
                self.q_table[next_state_hash] = np.zeros(self.action_size)
            
            # Q-learning update
            current_q = self.q_table[state_hash][action]
            
            if done:
                target_q = reward
            else:
                max_next_q = np.max(self.q_table[next_state_hash])
                target_q = reward + self.discount_factor * max_next_q
            
            # Update Q-value
            self.q_table[state_hash][action] = (current_q + 
                                              self.learning_rate * (target_q - current_q))
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                
        except Exception as e:
            print(f"Error in learn: {e}")
    
    def replay_experience(self, batch_size: int = 32):
        """Replay experiences from memory for additional learning"""
        try:
            if len(self.memory) < batch_size:
                return
            
            # Sample random batch from memory
            batch = random.sample(self.memory, batch_size)
            
            for state, action, reward, next_state, done in batch:
                self.learn(state, action, reward, next_state, done)
                
        except Exception as e:
            print(f"Error in replay_experience: {e}")
    
    def get_action_name(self, action: int) -> str:
        """Convert action number to readable name"""
        action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
        return action_names.get(action, 'UNKNOWN')
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get learning statistics"""
        try:
            stats = {
                'epsilon': self.epsilon,
                'q_table_size': len(self.q_table),
                'memory_size': len(self.memory),
                'total_actions': len(self.action_history),
                'action_distribution': {},
                'avg_reward': 0.0,
                'total_reward': 0.0
            }
            
            # Action distribution
            if self.action_history:
                for i in range(self.action_size):
                    count = self.action_history.count(i)
                    stats['action_distribution'][self.get_action_name(i)] = count
            
            # Reward statistics
            if self.reward_history:
                stats['avg_reward'] = np.mean(self.reward_history)
                stats['total_reward'] = np.sum(self.reward_history)
                stats['reward_std'] = np.std(self.reward_history)
                stats['max_reward'] = np.max(self.reward_history)
                stats['min_reward'] = np.min(self.reward_history)
            
            return stats
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return {}
    
    def save_model(self, filepath: str):
        """Save Q-table and parameters"""
        try:
            model_data = {
                'q_table': self.q_table,
                'state_normalizers': self.state_normalizers,
                'epsilon': self.epsilon,
                'action_history': self.action_history[-1000:],  # Keep last 1000
                'reward_history': self.reward_history[-1000:]
            }
            
            import json
            with open(filepath, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_data = {}
                for key, value in model_data.items():
                    if key == 'q_table':
                        serializable_data[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                                                for k, v in value.items()}
                    else:
                        serializable_data[key] = value
                
                json.dump(serializable_data, f, indent=2)
                
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load Q-table and parameters"""
        try:
            import json
            with open(filepath, 'r') as f:
                model_data = json.load(f)
            
            # Restore Q-table
            self.q_table = {}
            if 'q_table' in model_data:
                for state_hash, q_values in model_data['q_table'].items():
                    self.q_table[state_hash] = np.array(q_values)
            
            # Restore other parameters
            if 'state_normalizers' in model_data:
                self.state_normalizers = model_data['state_normalizers']
            if 'epsilon' in model_data:
                self.epsilon = model_data['epsilon']
            if 'action_history' in model_data:
                self.action_history = model_data['action_history']
            if 'reward_history' in model_data:
                self.reward_history = model_data['reward_history']
                
        except Exception as e:
            print(f"Error loading model: {e}")
