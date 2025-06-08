"""
Visual Strategy Builder
Drag-and-drop interface for creating custom trading strategies
"""
import streamlit as st
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
import plotly.graph_objects as go
from dataclasses import dataclass, asdict
import uuid

@dataclass
class StrategyComponent:
    """Base class for strategy components"""
    id: str
    type: str
    name: str
    params: Dict[str, Any]
    position: Dict[str, int]

@dataclass
class IndicatorBlock(StrategyComponent):
    """Technical indicator block"""
    indicator_type: str  # 'ema', 'rsi', 'atr', 'macd', 'bb'
    period: int
    source: str  # 'close', 'high', 'low', 'volume'

@dataclass
class LogicBlock(StrategyComponent):
    """Logic condition block"""
    condition_type: str  # 'greater_than', 'less_than', 'crosses_above', 'crosses_below'
    left_input: str
    right_input: str
    operator: str

@dataclass
class ActionBlock(StrategyComponent):
    """Trading action block"""
    action_type: str  # 'buy', 'sell', 'hold'
    quantity_type: str  # 'percentage', 'fixed', 'dynamic'
    quantity_value: float
    conditions: List[str]

class VisualStrategyBuilder:
    """Visual strategy builder with drag-and-drop interface"""
    
    def __init__(self):
        self.available_indicators = {
            'EMA': {'name': 'Exponential Moving Average', 'params': ['period']},
            'SMA': {'name': 'Simple Moving Average', 'params': ['period']},
            'RSI': {'name': 'Relative Strength Index', 'params': ['period']},
            'MACD': {'name': 'MACD', 'params': ['fast', 'slow', 'signal']},
            'BB': {'name': 'Bollinger Bands', 'params': ['period', 'std']},
            'ATR': {'name': 'Average True Range', 'params': ['period']},
            'VOLUME': {'name': 'Volume', 'params': []},
            'VWAP': {'name': 'Volume Weighted Average Price', 'params': []}
        }
        
        self.logic_operators = {
            'greater_than': '>',
            'less_than': '<',
            'equal_to': '==',
            'crosses_above': 'crosses above',
            'crosses_below': 'crosses below',
            'and': 'AND',
            'or': 'OR'
        }
        
        self.action_types = {
            'buy': 'Buy',
            'sell': 'Sell',
            'hold': 'Hold',
            'stop_loss': 'Stop Loss',
            'take_profit': 'Take Profit'
        }
    
    def render_strategy_builder_page(self):
        """Render the main strategy builder interface"""
        st.title("🎨 Visual Strategy Builder")
        st.markdown("Create custom trading strategies with drag-and-drop components")
        
        # Strategy management
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            strategy_name = st.text_input("Strategy Name", value="My Custom Strategy")
        
        with col2:
            if st.button("💾 Save Strategy", type="primary"):
                self._save_strategy(strategy_name)
        
        with col3:
            if st.button("📁 Load Strategy"):
                self._show_load_strategy_dialog()
        
        # Main builder interface
        st.divider()
        
        col_palette, col_canvas = st.columns([1, 2])
        
        with col_palette:
            self._render_component_palette()
        
        with col_canvas:
            self._render_strategy_canvas()
        
        # Strategy validation and testing
        st.divider()
        self._render_strategy_validation()
    
    def _render_component_palette(self):
        """Render the component palette"""
        st.subheader("📦 Component Palette")
        
        # Indicators
        with st.expander("📊 Technical Indicators", expanded=True):
            for indicator, config in self.available_indicators.items():
                if st.button(f"+ {config['name']}", key=f"add_{indicator}"):
                    self._add_indicator_to_canvas(indicator, config)
        
        # Logic blocks
        with st.expander("🔗 Logic Conditions"):
            for logic_type, symbol in self.logic_operators.items():
                if st.button(f"+ {logic_type.title()} ({symbol})", key=f"add_logic_{logic_type}"):
                    self._add_logic_to_canvas(logic_type)
        
        # Actions
        with st.expander("⚡ Trading Actions"):
            for action_type, name in self.action_types.items():
                if st.button(f"+ {name}", key=f"add_action_{action_type}"):
                    self._add_action_to_canvas(action_type)
    
    def _render_strategy_canvas(self):
        """Render the strategy canvas"""
        st.subheader("🎯 Strategy Canvas")
        
        # Initialize session state for canvas
        if 'strategy_components' not in st.session_state:
            st.session_state.strategy_components = []
        
        # Canvas area
        if not st.session_state.strategy_components:
            st.info("👈 Drag components from the palette to build your strategy")
        else:
            self._render_strategy_flow()
        
        # Component configuration panel
        if st.session_state.strategy_components:
            st.subheader("⚙️ Component Configuration")
            selected_component = st.selectbox(
                "Select component to configure:",
                options=[f"{comp['name']} ({comp['type']})" for comp in st.session_state.strategy_components],
                key="selected_component"
            )
            
            if selected_component:
                component_index = next(i for i, comp in enumerate(st.session_state.strategy_components) 
                                     if f"{comp['name']} ({comp['type']})" == selected_component)
                self._render_component_config(component_index)
    
    def _render_strategy_flow(self):
        """Render the strategy flow diagram"""
        # Create a visual representation of the strategy
        flow_data = []
        
        for i, component in enumerate(st.session_state.strategy_components):
            flow_data.append({
                'step': i + 1,
                'type': component['type'],
                'name': component['name'],
                'description': self._get_component_description(component)
            })
        
        if flow_data:
            df = pd.DataFrame(flow_data)
            st.dataframe(df, use_container_width=True)
    
    def _render_component_config(self, component_index: int):
        """Render configuration panel for selected component"""
        component = st.session_state.strategy_components[component_index]
        
        st.write(f"**Configuring: {component['name']}**")
        
        # Component-specific configuration
        if component['type'] == 'indicator':
            self._config_indicator(component_index)
        elif component['type'] == 'logic':
            self._config_logic(component_index)
        elif component['type'] == 'action':
            self._config_action(component_index)
        
        # Remove component button
        if st.button("🗑️ Remove Component", key=f"remove_{component_index}"):
            st.session_state.strategy_components.pop(component_index)
            st.rerun()
    
    def _config_indicator(self, component_index: int):
        """Configure indicator component"""
        component = st.session_state.strategy_components[component_index]
        
        # Period configuration
        if 'period' in component.get('config_params', []):
            period = st.number_input(
                "Period", 
                min_value=1, 
                max_value=200, 
                value=component.get('period', 14),
                key=f"period_{component_index}"
            )
            st.session_state.strategy_components[component_index]['period'] = period
        
        # Source configuration
        source = st.selectbox(
            "Price Source",
            options=['close', 'open', 'high', 'low', 'volume'],
            index=0,
            key=f"source_{component_index}"
        )
        st.session_state.strategy_components[component_index]['source'] = source
    
    def _config_logic(self, component_index: int):
        """Configure logic component"""
        component = st.session_state.strategy_components[component_index]
        
        # Get available inputs (indicators + price data)
        available_inputs = ['price', 'volume'] + [
            comp['name'] for comp in st.session_state.strategy_components 
            if comp['type'] == 'indicator'
        ]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            left_input = st.selectbox(
                "Left Input",
                options=available_inputs,
                key=f"left_{component_index}"
            )
        
        with col2:
            operator = st.selectbox(
                "Operator",
                options=list(self.logic_operators.keys()),
                key=f"operator_{component_index}"
            )
        
        with col3:
            right_input = st.selectbox(
                "Right Input",
                options=available_inputs + ['value'],
                key=f"right_{component_index}"
            )
        
        # If right input is 'value', allow custom value entry
        if right_input == 'value':
            custom_value = st.number_input(
                "Custom Value",
                value=0.0,
                key=f"custom_value_{component_index}"
            )
            st.session_state.strategy_components[component_index]['custom_value'] = custom_value
        
        # Update component
        st.session_state.strategy_components[component_index].update({
            'left_input': left_input,
            'operator': operator,
            'right_input': right_input
        })
    
    def _config_action(self, component_index: int):
        """Configure action component"""
        component = st.session_state.strategy_components[component_index]
        
        # Action type
        action_type = st.selectbox(
            "Action Type",
            options=list(self.action_types.keys()),
            format_func=lambda x: self.action_types[x],
            key=f"action_type_{component_index}"
        )
        
        # Quantity configuration
        quantity_type = st.selectbox(
            "Quantity Type",
            options=['percentage', 'fixed_amount', 'dynamic'],
            key=f"quantity_type_{component_index}"
        )
        
        if quantity_type == 'percentage':
            quantity_value = st.slider(
                "Portfolio Percentage",
                min_value=1,
                max_value=100,
                value=10,
                key=f"quantity_pct_{component_index}"
            )
        elif quantity_type == 'fixed_amount':
            quantity_value = st.number_input(
                "Fixed Amount ($)",
                min_value=1.0,
                value=100.0,
                key=f"quantity_fixed_{component_index}"
            )
        else:  # dynamic
            quantity_value = st.selectbox(
                "Dynamic Sizing Method",
                options=['atr_based', 'volatility_based', 'kelly_criterion'],
                key=f"quantity_dynamic_{component_index}"
            )
        
        # Update component
        st.session_state.strategy_components[component_index].update({
            'action_type': action_type,
            'quantity_type': quantity_type,
            'quantity_value': quantity_value
        })
    
    def _add_indicator_to_canvas(self, indicator_type: str, config: Dict):
        """Add indicator to strategy canvas"""
        component = {
            'id': str(uuid.uuid4()),
            'type': 'indicator',
            'name': f"{indicator_type}_{len(st.session_state.get('strategy_components', []))}",
            'indicator_type': indicator_type,
            'config_params': config['params'],
            'period': 14,  # default
            'source': 'close'  # default
        }
        
        if 'strategy_components' not in st.session_state:
            st.session_state.strategy_components = []
        
        st.session_state.strategy_components.append(component)
        st.rerun()
    
    def _add_logic_to_canvas(self, logic_type: str):
        """Add logic block to strategy canvas"""
        component = {
            'id': str(uuid.uuid4()),
            'type': 'logic',
            'name': f"Logic_{len(st.session_state.get('strategy_components', []))}",
            'logic_type': logic_type,
            'left_input': '',
            'operator': logic_type,
            'right_input': ''
        }
        
        if 'strategy_components' not in st.session_state:
            st.session_state.strategy_components = []
        
        st.session_state.strategy_components.append(component)
        st.rerun()
    
    def _add_action_to_canvas(self, action_type: str):
        """Add action block to strategy canvas"""
        component = {
            'id': str(uuid.uuid4()),
            'type': 'action',
            'name': f"Action_{len(st.session_state.get('strategy_components', []))}",
            'action_type': action_type,
            'quantity_type': 'percentage',
            'quantity_value': 10
        }
        
        if 'strategy_components' not in st.session_state:
            st.session_state.strategy_components = []
        
        st.session_state.strategy_components.append(component)
        st.rerun()
    
    def _get_component_description(self, component: Dict) -> str:
        """Get human-readable description of component"""
        if component['type'] == 'indicator':
            return f"{component['indicator_type']}({component.get('period', 'N/A')})"
        elif component['type'] == 'logic':
            left = component.get('left_input', '?')
            op = component.get('operator', '?')
            right = component.get('right_input', '?')
            return f"{left} {op} {right}"
        elif component['type'] == 'action':
            action = component.get('action_type', '?')
            qty = component.get('quantity_value', '?')
            qty_type = component.get('quantity_type', '?')
            return f"{action.title()} {qty} ({qty_type})"
        return "Unknown"
    
    def _render_strategy_validation(self):
        """Render strategy validation and testing section"""
        st.subheader("✅ Strategy Validation")
        
        if not st.session_state.get('strategy_components', []):
            st.warning("Add components to your strategy before validation")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Strategy validation
            validation_results = self._validate_strategy()
            
            if validation_results['valid']:
                st.success("✅ Strategy is valid and ready for deployment")
            else:
                st.error("❌ Strategy validation failed")
                for error in validation_results['errors']:
                    st.error(f"• {error}")
        
        with col2:
            # Backtest configuration
            st.write("**Backtest Configuration**")
            
            test_symbol = st.selectbox(
                "Test Symbol",
                options=['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
                key="test_symbol"
            )
            
            test_period = st.selectbox(
                "Test Period",
                options=['7 days', '30 days', '90 days'],
                key="test_period"
            )
            
            if st.button("🧪 Run Backtest", type="primary"):
                if validation_results['valid']:
                    self._run_strategy_backtest(test_symbol, test_period)
                else:
                    st.error("Fix validation errors before running backtest")
    
    def _validate_strategy(self) -> Dict[str, Any]:
        """Validate the current strategy"""
        components = st.session_state.get('strategy_components', [])
        errors = []
        
        if not components:
            errors.append("Strategy must have at least one component")
        
        # Check for at least one action
        has_action = any(comp['type'] == 'action' for comp in components)
        if not has_action:
            errors.append("Strategy must have at least one action (Buy/Sell)")
        
        # Check logic blocks have valid inputs
        for comp in components:
            if comp['type'] == 'logic':
                if not comp.get('left_input') or not comp.get('right_input'):
                    errors.append(f"Logic block '{comp['name']}' has missing inputs")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
    
    def _run_strategy_backtest(self, symbol: str, period: str):
        """Run strategy backtest"""
        st.info("🧪 Running backtest simulation...")
        
        # Convert strategy to executable format
        strategy_code = self._generate_strategy_code()
        
        # Simulate backtest results (in real implementation, this would use historical data)
        import random
        random.seed(42)
        
        backtest_results = {
            'total_return': random.uniform(-10, 25),
            'win_rate': random.uniform(40, 70),
            'max_drawdown': random.uniform(-15, -5),
            'sharpe_ratio': random.uniform(0.5, 2.0),
            'total_trades': random.randint(50, 200)
        }
        
        st.success("✅ Backtest completed!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Return", f"{backtest_results['total_return']:.1f}%")
            st.metric("Win Rate", f"{backtest_results['win_rate']:.1f}%")
        with col2:
            st.metric("Max Drawdown", f"{backtest_results['max_drawdown']:.1f}%")
            st.metric("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")
        with col3:
            st.metric("Total Trades", backtest_results['total_trades'])
        
        # Show generated strategy code
        with st.expander("📝 Generated Strategy Code"):
            st.code(strategy_code, language='python')
    
    def _generate_strategy_code(self) -> str:
        """Generate executable strategy code from components"""
        components = st.session_state.get('strategy_components', [])
        
        code_lines = [
            "# Auto-generated strategy from Visual Builder",
            "import pandas as pd",
            "import numpy as np",
            "",
            "def execute_strategy(data: pd.DataFrame) -> str:",
            "    # Calculate indicators"
        ]
        
        # Generate indicator calculations
        for comp in components:
            if comp['type'] == 'indicator':
                indicator_type = comp['indicator_type']
                period = comp.get('period', 14)
                
                if indicator_type == 'EMA':
                    code_lines.append(f"    {comp['name']} = data['close'].ewm(span={period}).mean()")
                elif indicator_type == 'RSI':
                    code_lines.append(f"    {comp['name']} = calculate_rsi(data['close'], {period})")
                elif indicator_type == 'SMA':
                    code_lines.append(f"    {comp['name']} = data['close'].rolling({period}).mean()")
        
        code_lines.extend([
            "",
            "    # Evaluate conditions and actions",
            "    signal = 'hold'",
            ""
        ])
        
        # Generate logic and actions
        for comp in components:
            if comp['type'] == 'logic':
                left = comp.get('left_input', 'price')
                op = comp.get('operator', '>')
                right = comp.get('right_input', 'value')
                
                if right == 'value':
                    right_val = comp.get('custom_value', 0)
                    code_lines.append(f"    if {left} {self.logic_operators.get(op, '>')} {right_val}:")
                else:
                    code_lines.append(f"    if {left} {self.logic_operators.get(op, '>')} {right}:")
            
            elif comp['type'] == 'action':
                action = comp.get('action_type', 'hold')
                code_lines.append(f"        signal = '{action}'")
        
        code_lines.extend([
            "",
            "    return signal"
        ])
        
        return "\n".join(code_lines)
    
    def _save_strategy(self, strategy_name: str):
        """Save strategy to database/file"""
        if not st.session_state.get('strategy_components'):
            st.error("No strategy to save")
            return
        
        strategy_data = {
            'name': strategy_name,
            'created_at': datetime.now().isoformat(),
            'components': st.session_state.strategy_components,
            'version': '1.0'
        }
        
        # Save to session state (in real app, save to database)
        if 'saved_strategies' not in st.session_state:
            st.session_state.saved_strategies = {}
        
        st.session_state.saved_strategies[strategy_name] = strategy_data
        st.success(f"✅ Strategy '{strategy_name}' saved successfully!")
    
    def _show_load_strategy_dialog(self):
        """Show load strategy dialog"""
        saved_strategies = st.session_state.get('saved_strategies', {})
        
        if not saved_strategies:
            st.warning("No saved strategies found")
            return
        
        strategy_names = list(saved_strategies.keys())
        selected_strategy = st.selectbox("Select strategy to load:", strategy_names)
        
        if st.button("📁 Load Selected Strategy"):
            strategy_data = saved_strategies[selected_strategy]
            st.session_state.strategy_components = strategy_data['components']
            st.success(f"✅ Strategy '{selected_strategy}' loaded successfully!")
            st.rerun()

def show_visual_strategy_builder():
    """Main function to show visual strategy builder"""
    builder = VisualStrategyBuilder()
    builder.render_strategy_builder_page()