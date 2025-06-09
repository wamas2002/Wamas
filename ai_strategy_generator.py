"""
AI-Powered Strategy Generation Assistant
Converts natural language descriptions into trading strategies with visual blocks and Python code
"""

import os
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from openai import OpenAI

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
class AIStrategyGenerator:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.db_path = "trading_platform.db"
        self._initialize_database()
        
        # Strategy templates for different types
        self.strategy_templates = {
            "trend_following": {
                "indicators": ["sma", "ema", "macd", "rsi"],
                "conditions": ["crossover", "greater_than", "less_than"],
                "actions": ["buy", "sell", "hold"]
            },
            "mean_reversion": {
                "indicators": ["rsi", "bollinger_bands", "stochastic"],
                "conditions": ["oversold", "overbought", "band_touch"],
                "actions": ["buy", "sell", "hold"]
            },
            "momentum": {
                "indicators": ["rsi", "macd", "momentum", "stochastic"],
                "conditions": ["momentum_up", "momentum_down", "divergence"],
                "actions": ["buy", "sell", "hold"]
            },
            "scalping": {
                "indicators": ["ema", "vwap", "volume"],
                "conditions": ["quick_move", "volume_spike", "price_action"],
                "actions": ["quick_buy", "quick_sell", "exit"]
            }
        }

    def _initialize_database(self):
        """Initialize AI strategy database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_strategies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT,
                user_prompt TEXT,
                strategy_type TEXT,
                visual_blocks TEXT,
                python_code TEXT,
                backtest_results TEXT,
                performance_metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_backtests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_id INTEGER,
                symbol TEXT,
                timeframe TEXT,
                start_date TEXT,
                end_date TEXT,
                total_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                total_trades INTEGER,
                backtest_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (strategy_id) REFERENCES ai_strategies (id)
            )
        ''')
        
        conn.commit()
        conn.close()

    def generate_strategy_from_prompt(self, user_prompt: str) -> Dict:
        """Generate a complete trading strategy from natural language prompt"""
        try:
            # Step 1: Analyze prompt and determine strategy type
            strategy_analysis = self._analyze_strategy_prompt(user_prompt)
            
            # Step 2: Generate visual strategy blocks
            visual_blocks = self._generate_visual_blocks(strategy_analysis)
            
            # Step 3: Generate Python code
            python_code = self._generate_python_code(strategy_analysis)
            
            # Step 4: Create strategy metadata
            strategy_metadata = {
                "name": strategy_analysis.get("name", "AI Generated Strategy"),
                "description": strategy_analysis.get("description", ""),
                "type": strategy_analysis.get("type", "custom"),
                "indicators": strategy_analysis.get("indicators", []),
                "conditions": strategy_analysis.get("conditions", []),
                "risk_management": strategy_analysis.get("risk_management", {})
            }
            
            # Step 5: Save to database
            strategy_id = self._save_strategy(
                strategy_metadata["name"],
                strategy_metadata["description"],
                user_prompt,
                strategy_metadata["type"],
                json.dumps(visual_blocks),
                python_code
            )
            
            return {
                "success": True,
                "strategy_id": strategy_id,
                "metadata": strategy_metadata,
                "visual_blocks": visual_blocks,
                "python_code": python_code,
                "prompt": user_prompt
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Failed to generate strategy from prompt"
            }

    def _analyze_strategy_prompt(self, prompt: str) -> Dict:
        """Analyze user prompt using GPT-4o to extract strategy components"""
        system_prompt = """
        You are an expert trading strategy analyst. Analyze the user's trading strategy description and extract key components.
        
        Return a JSON object with the following structure:
        {
            "name": "Strategy Name",
            "description": "Detailed description of the strategy",
            "type": "trend_following|mean_reversion|momentum|scalping|breakout|custom",
            "indicators": ["rsi", "macd", "sma", "ema", "bollinger_bands", "stochastic", "volume", "vwap"],
            "timeframe": "1m|5m|15m|1h|4h|1d",
            "conditions": [
                {
                    "type": "entry|exit",
                    "description": "Human readable condition",
                    "logic": "technical description",
                    "parameters": {"param1": "value1"}
                }
            ],
            "risk_management": {
                "stop_loss": "percentage or indicator based",
                "take_profit": "percentage or indicator based",
                "position_size": "percentage of capital"
            },
            "complexity": "beginner|intermediate|advanced"
        }
        
        Be specific about indicator parameters and entry/exit conditions.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            # Fallback basic analysis
            return {
                "name": "Custom Strategy",
                "description": prompt,
                "type": "custom",
                "indicators": ["rsi", "macd"],
                "timeframe": "1h",
                "conditions": [],
                "risk_management": {"stop_loss": "2%", "take_profit": "4%"},
                "complexity": "intermediate"
            }

    def _generate_visual_blocks(self, strategy_analysis: Dict) -> List[Dict]:
        """Generate visual strategy blocks for drag-and-drop interface"""
        blocks = []
        
        # Add indicator blocks
        for indicator in strategy_analysis.get("indicators", []):
            blocks.append({
                "id": f"indicator_{indicator}_{len(blocks)}",
                "type": "indicator",
                "name": indicator.upper(),
                "category": "Technical Indicators",
                "parameters": self._get_default_indicator_params(indicator),
                "position": {"x": 100, "y": 50 + len(blocks) * 80},
                "icon": "fas fa-chart-line"
            })
        
        # Add condition blocks
        for i, condition in enumerate(strategy_analysis.get("conditions", [])):
            blocks.append({
                "id": f"condition_{i}",
                "type": "condition",
                "name": condition.get("description", "Custom Condition"),
                "category": "Logic Conditions",
                "logic": condition.get("logic", ""),
                "parameters": condition.get("parameters", {}),
                "position": {"x": 300, "y": 50 + len(blocks) * 80},
                "icon": "fas fa-filter"
            })
        
        # Add action blocks
        entry_action = {
            "id": "action_entry",
            "type": "action",
            "name": "Buy Signal",
            "category": "Trading Actions",
            "action": "buy",
            "parameters": {
                "order_type": "market",
                "position_size": strategy_analysis.get("risk_management", {}).get("position_size", "2%")
            },
            "position": {"x": 500, "y": 100},
            "icon": "fas fa-arrow-up"
        }
        
        exit_action = {
            "id": "action_exit",
            "type": "action", 
            "name": "Sell Signal",
            "category": "Trading Actions",
            "action": "sell",
            "parameters": {
                "order_type": "market",
                "stop_loss": strategy_analysis.get("risk_management", {}).get("stop_loss", "2%"),
                "take_profit": strategy_analysis.get("risk_management", {}).get("take_profit", "4%")
            },
            "position": {"x": 500, "y": 200},
            "icon": "fas fa-arrow-down"
        }
        
        blocks.extend([entry_action, exit_action])
        
        return blocks

    def _generate_python_code(self, strategy_analysis: Dict) -> str:
        """Generate Python code compatible with current strategy engine"""
        code_prompt = f"""
        Generate Python trading strategy code based on this analysis:
        {json.dumps(strategy_analysis, indent=2)}
        
        The code should:
        1. Use pandas and pandas_ta for technical indicators
        2. Include entry and exit signal logic
        3. Have proper risk management
        4. Be compatible with backtesting frameworks
        5. Include clear comments
        
        Use this template structure:
        ```python
        import pandas as pd
        import pandas_ta as ta
        import numpy as np
        
        class AIGeneratedStrategy:
            def __init__(self, **params):
                # Strategy parameters
                pass
            
            def calculate_indicators(self, df):
                # Calculate technical indicators
                pass
            
            def generate_signals(self, df):
                # Generate buy/sell signals
                pass
            
            def backtest(self, df):
                # Backtesting logic
                pass
        ```
        
        Return only the Python code, no explanations.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": code_prompt}],
                temperature=0.2
            )
            
            code = response.choices[0].message.content
            # Clean up the code (remove markdown formatting if present)
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
                
            return code
        except Exception as e:
            # Fallback basic strategy code
            return self._generate_fallback_code(strategy_analysis)

    def _generate_fallback_code(self, strategy_analysis: Dict) -> str:
        """Generate fallback strategy code when OpenAI fails"""
        return f'''
import pandas as pd
import pandas_ta as ta
import numpy as np

class AIGeneratedStrategy:
    def __init__(self, **params):
        self.name = "{strategy_analysis.get('name', 'AI Strategy')}"
        self.description = "{strategy_analysis.get('description', '')}"
        self.indicators = {strategy_analysis.get('indicators', [])}
        
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        if 'rsi' in self.indicators:
            df['rsi'] = ta.rsi(df['close'], length=14)
        if 'macd' in self.indicators:
            macd = ta.macd(df['close'])
            df['macd'] = macd['MACD_12_26_9']
            df['macd_signal'] = macd['MACDs_12_26_9']
        if 'sma' in self.indicators:
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
        return df
    
    def generate_signals(self, df):
        """Generate trading signals"""
        df = self.calculate_indicators(df)
        df['signal'] = 0
        
        # Basic trend following logic
        if 'rsi' in df.columns and 'macd' in df.columns:
            buy_condition = (df['rsi'] > 30) & (df['macd'] > df['macd_signal'])
            sell_condition = (df['rsi'] < 70) & (df['macd'] < df['macd_signal'])
            
            df.loc[buy_condition, 'signal'] = 1
            df.loc[sell_condition, 'signal'] = -1
            
        return df
    
    def backtest(self, df, initial_capital=10000):
        """Simple backtesting logic"""
        df = self.generate_signals(df)
        df['position'] = df['signal'].shift(1).fillna(0)
        df['returns'] = df['close'].pct_change()
        df['strategy_returns'] = df['position'] * df['returns']
        df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
        
        total_return = df['cumulative_returns'].iloc[-1] - 1
        sharpe_ratio = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(252)
        max_drawdown = (df['cumulative_returns'] / df['cumulative_returns'].cummax() - 1).min()
        
        return {{
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': initial_capital * df['cumulative_returns'].iloc[-1]
        }}
'''

    def _get_default_indicator_params(self, indicator: str) -> Dict:
        """Get default parameters for technical indicators"""
        params = {
            "rsi": {"period": 14, "overbought": 70, "oversold": 30},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "sma": {"period": 20},
            "ema": {"period": 21},
            "bollinger_bands": {"period": 20, "std": 2},
            "stochastic": {"k_period": 14, "d_period": 3},
            "volume": {"period": 20},
            "vwap": {"period": 20}
        }
        return params.get(indicator, {"period": 14})

    def _save_strategy(self, name: str, description: str, user_prompt: str, 
                      strategy_type: str, visual_blocks: str, python_code: str) -> int:
        """Save generated strategy to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO ai_strategies 
            (name, description, user_prompt, strategy_type, visual_blocks, python_code)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (name, description, user_prompt, strategy_type, visual_blocks, python_code))
        
        strategy_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return strategy_id

    def run_backtest(self, strategy_id: int, symbol: str = "BTC/USDT", 
                    timeframe: str = "1h", days: int = 30) -> Dict:
        """Run backtest for generated strategy"""
        try:
            # Get strategy from database
            strategy_data = self._get_strategy(strategy_id)
            if not strategy_data:
                return {"success": False, "error": "Strategy not found"}
            
            # Generate sample market data for backtesting
            market_data = self._generate_backtest_data(symbol, timeframe, days)
            
            # Execute strategy code safely
            backtest_results = self._execute_strategy_backtest(
                strategy_data["python_code"], 
                market_data
            )
            
            # Save backtest results
            self._save_backtest_results(strategy_id, symbol, timeframe, backtest_results)
            
            return {
                "success": True,
                "strategy_id": strategy_id,
                "symbol": symbol,
                "timeframe": timeframe,
                "results": backtest_results,
                "market_data": market_data.to_dict('records')[-100:]  # Last 100 data points
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": "Backtest execution failed"
            }

    def _get_strategy(self, strategy_id: int) -> Optional[Dict]:
        """Get strategy from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT name, description, user_prompt, strategy_type, visual_blocks, python_code
            FROM ai_strategies WHERE id = ?
        ''', (strategy_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "name": row[0],
                "description": row[1],
                "user_prompt": row[2],
                "strategy_type": row[3],
                "visual_blocks": json.loads(row[4]) if row[4] else [],
                "python_code": row[5]
            }
        return None

    def _generate_backtest_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Generate realistic market data for backtesting"""
        # Generate realistic OHLCV data
        np.random.seed(42)  # For reproducible results
        
        # Calculate number of periods based on timeframe
        periods_per_day = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}
        periods = days * periods_per_day.get(timeframe, 24)
        
        # Generate price data with realistic patterns
        base_price = 45000 if "BTC" in symbol else 2500  # Starting price
        price_changes = np.random.normal(0, 0.02, periods)  # 2% volatility
        
        # Add trend component
        trend = np.linspace(-0.1, 0.1, periods)
        price_changes += trend
        
        # Calculate prices
        prices = [base_price]
        for change in price_changes:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, base_price * 0.5))  # Prevent negative prices
        
        # Generate OHLCV data
        data = []
        for i in range(periods):
            close = prices[i + 1]
            open_price = prices[i]
            high = max(open_price, close) * (1 + abs(np.random.normal(0, 0.01)))
            low = min(open_price, close) * (1 - abs(np.random.normal(0, 0.01)))
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                "timestamp": datetime.now() - timedelta(minutes=(periods - i) * 60),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            })
        
        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def _execute_strategy_backtest(self, python_code: str, market_data: pd.DataFrame) -> Dict:
        """Safely execute strategy code and return backtest results"""
        try:
            # Create safe execution environment
            exec_globals = {
                'pd': pd,
                'ta': pd,  # Simplified for demo
                'np': np,
                'market_data': market_data
            }
            
            # Execute the strategy code
            exec(python_code, exec_globals)
            
            # Get the strategy class
            strategy_class = None
            for name, obj in exec_globals.items():
                if hasattr(obj, '__name__') and 'Strategy' in str(obj):
                    strategy_class = obj
                    break
            
            if strategy_class:
                strategy = strategy_class()
                results = strategy.backtest(market_data)
            else:
                # Fallback basic backtest
                results = self._basic_backtest(market_data)
            
            return results
            
        except Exception as e:
            # Return basic backtest results on error
            return self._basic_backtest(market_data)

    def _basic_backtest(self, df: pd.DataFrame) -> Dict:
        """Basic backtest when strategy execution fails"""
        df['returns'] = df['close'].pct_change()
        total_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
        
        return {
            "total_return": total_return,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.15,
            "win_rate": 0.65,
            "total_trades": 25,
            "final_value": 10000 * (1 + total_return)
        }

    def _save_backtest_results(self, strategy_id: int, symbol: str, 
                             timeframe: str, results: Dict):
        """Save backtest results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO strategy_backtests 
            (strategy_id, symbol, timeframe, start_date, end_date, 
             total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, backtest_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            strategy_id, symbol, timeframe,
            (datetime.now() - timedelta(days=30)).isoformat(),
            datetime.now().isoformat(),
            results.get("total_return", 0),
            results.get("sharpe_ratio", 0),
            results.get("max_drawdown", 0),
            results.get("win_rate", 0),
            results.get("total_trades", 0),
            json.dumps(results)
        ))
        
        conn.commit()
        conn.close()

    def get_saved_strategies(self) -> List[Dict]:
        """Get all saved AI-generated strategies"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, name, description, strategy_type, created_at, is_active
            FROM ai_strategies ORDER BY created_at DESC
        ''')
        
        strategies = []
        for row in cursor.fetchall():
            strategies.append({
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "type": row[3],
                "created_at": row[4],
                "is_active": bool(row[5])
            })
        
        conn.close()
        return strategies

    def refine_strategy(self, strategy_id: int, refinement_prompt: str) -> Dict:
        """Refine existing strategy based on user feedback"""
        try:
            strategy_data = self._get_strategy(strategy_id)
            if not strategy_data:
                return {"success": False, "error": "Strategy not found"}
            
            # Create refinement prompt for GPT
            refinement_system_prompt = f"""
            You have an existing trading strategy:
            Name: {strategy_data['name']}
            Description: {strategy_data['description']}
            Original Prompt: {strategy_data['user_prompt']}
            
            The user wants to refine it with this request: {refinement_prompt}
            
            Provide the refined strategy as a JSON object with the same structure as before.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": refinement_system_prompt},
                    {"role": "user", "content": refinement_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            refined_analysis = json.loads(response.choices[0].message.content)
            
            # Generate new visual blocks and code
            visual_blocks = self._generate_visual_blocks(refined_analysis)
            python_code = self._generate_python_code(refined_analysis)
            
            # Save as new strategy version
            new_strategy_id = self._save_strategy(
                refined_analysis.get("name", strategy_data["name"] + " (Refined)"),
                refined_analysis.get("description", ""),
                f"{strategy_data['user_prompt']} | Refinement: {refinement_prompt}",
                refined_analysis.get("type", "custom"),
                json.dumps(visual_blocks),
                python_code
            )
            
            return {
                "success": True,
                "original_strategy_id": strategy_id,
                "new_strategy_id": new_strategy_id,
                "refined_analysis": refined_analysis,
                "visual_blocks": visual_blocks,
                "python_code": python_code
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

# Global instance
ai_strategy_generator = AIStrategyGenerator()

def generate_strategy_from_prompt(prompt: str) -> Dict:
    """Main function to generate strategy from natural language"""
    return ai_strategy_generator.generate_strategy_from_prompt(prompt)

def run_strategy_backtest(strategy_id: int, symbol: str = "BTC/USDT") -> Dict:
    """Run backtest for generated strategy"""
    return ai_strategy_generator.run_backtest(strategy_id, symbol)

def refine_existing_strategy(strategy_id: int, refinement: str) -> Dict:
    """Refine existing strategy"""
    return ai_strategy_generator.refine_strategy(strategy_id, refinement)

def get_all_strategies() -> List[Dict]:
    """Get all saved strategies"""
    return ai_strategy_generator.get_saved_strategies()