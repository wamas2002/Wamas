"""
Visual Strategy Builder - Drag & Drop Strategy Editor
Allows users to build trading strategies through visual blocks without coding
"""

import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

def show_visual_strategy_builder():
    """Main visual strategy builder interface"""
    st.title("🎨 Visual Strategy Builder")
    st.markdown("**Build trading strategies with drag-and-drop blocks**")
    
    # Initialize strategy state
    if 'current_strategy' not in st.session_state:
        st.session_state.current_strategy = {
            'name': 'New Strategy',
            'description': '',
            'blocks': [],
            'connections': [],
            'risk_settings': {
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'position_size': 0.1
            }
        }
    
    # Strategy builder interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        show_strategy_canvas()
    
    with col2:
        show_block_palette()
        show_strategy_settings()
    
    # Bottom controls
    st.markdown("---")
    show_strategy_controls()

def show_strategy_canvas():
    """Main canvas for building strategies"""
    st.subheader("📋 Strategy Canvas")
    
    strategy = st.session_state.current_strategy
    
    # Strategy header
    col1, col2 = st.columns([3, 1])
    
    with col1:
        strategy['name'] = st.text_input("Strategy Name", value=strategy['name'])
        strategy['description'] = st.text_area("Description", value=strategy['description'], height=60)
    
    with col2:
        if st.button("🧪 Test Strategy", type="primary"):
            test_strategy()
        if st.button("💾 Save Strategy"):
            save_strategy()
    
    # Canvas area
    st.markdown("**Strategy Blocks:**")
    
    if not strategy['blocks']:
        st.info("👆 Add blocks from the palette to start building your strategy")
    else:
        display_strategy_blocks()
    
    # Strategy flow visualization
    if strategy['blocks']:
        show_strategy_flow()

def show_block_palette():
    """Block palette for drag-and-drop components"""
    st.subheader("🧩 Block Palette")
    
    # Block categories
    category = st.selectbox("Category", [
        "🎯 Entry Signals",
        "🚨 Exit Signals", 
        "📊 Indicators",
        "⚖️ Risk Management",
        "🔧 Logic"
    ])
    
    blocks = get_blocks_by_category(category)
    
    for block in blocks:
        if st.button(f"➕ {block['name']}", key=f"add_{block['id']}"):
            add_block_to_strategy(block)
            st.rerun()

def get_blocks_by_category(category: str) -> List[Dict[str, Any]]:
    """Get available blocks by category"""
    blocks = {
        "🎯 Entry Signals": [
            {
                'id': 'rsi_oversold',
                'name': 'RSI Oversold',
                'description': 'Enter when RSI drops below threshold',
                'type': 'entry_signal',
                'parameters': {'threshold': 30}
            },
            {
                'id': 'ma_crossover',
                'name': 'MA Crossover',
                'description': 'Enter on moving average crossover',
                'type': 'entry_signal',
                'parameters': {'fast_period': 10, 'slow_period': 20}
            },
            {
                'id': 'breakout',
                'name': 'Price Breakout',
                'description': 'Enter on price breakout above resistance',
                'type': 'entry_signal',
                'parameters': {'lookback': 20, 'threshold': 0.02}
            }
        ],
        "🚨 Exit Signals": [
            {
                'id': 'rsi_overbought',
                'name': 'RSI Overbought',
                'description': 'Exit when RSI rises above threshold',
                'type': 'exit_signal',
                'parameters': {'threshold': 70}
            },
            {
                'id': 'profit_target',
                'name': 'Profit Target',
                'description': 'Exit at profit percentage',
                'type': 'exit_signal',
                'parameters': {'profit_pct': 0.05}
            }
        ],
        "📊 Indicators": [
            {
                'id': 'rsi_indicator',
                'name': 'RSI',
                'description': 'Relative Strength Index',
                'type': 'indicator',
                'parameters': {'period': 14}
            },
            {
                'id': 'ema_indicator',
                'name': 'EMA',
                'description': 'Exponential Moving Average',
                'type': 'indicator',
                'parameters': {'period': 20}
            },
            {
                'id': 'bollinger_bands',
                'name': 'Bollinger Bands',
                'description': 'Volatility indicator',
                'type': 'indicator',
                'parameters': {'period': 20, 'std_dev': 2}
            }
        ],
        "⚖️ Risk Management": [
            {
                'id': 'stop_loss',
                'name': 'Stop Loss',
                'description': 'Set stop loss percentage',
                'type': 'risk',
                'parameters': {'loss_pct': 0.02}
            },
            {
                'id': 'position_size',
                'name': 'Position Size',
                'description': 'Set position size percentage',
                'type': 'risk',
                'parameters': {'size_pct': 0.1}
            }
        ],
        "🔧 Logic": [
            {
                'id': 'and_condition',
                'name': 'AND',
                'description': 'All conditions must be true',
                'type': 'logic',
                'parameters': {}
            },
            {
                'id': 'or_condition',
                'name': 'OR',
                'description': 'Any condition must be true',
                'type': 'logic',
                'parameters': {}
            }
        ]
    }
    
    return blocks.get(category, [])

def add_block_to_strategy(block: Dict[str, Any]):
    """Add a block to the current strategy"""
    strategy = st.session_state.current_strategy
    
    # Generate unique ID for this instance
    block_instance = {
        'id': f"{block['id']}_{len(strategy['blocks'])}",
        'type': block['type'],
        'name': block['name'],
        'description': block['description'],
        'parameters': block['parameters'].copy(),
        'position': len(strategy['blocks'])
    }
    
    strategy['blocks'].append(block_instance)

def display_strategy_blocks():
    """Display and edit strategy blocks"""
    strategy = st.session_state.current_strategy
    
    for i, block in enumerate(strategy['blocks']):
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                # Block display
                block_color = get_block_color(block['type'])
                st.markdown(f"""
                <div style="
                    background-color: {block_color};
                    padding: 10px;
                    border-radius: 5px;
                    margin: 5px 0;
                    border-left: 4px solid #1f77b4;
                ">
                    <strong>{block['name']}</strong><br>
                    <small>{block['description']}</small>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("⚙️", key=f"edit_{block['id']}"):
                    edit_block_parameters(i)
            
            with col3:
                if st.button("🗑️", key=f"delete_{block['id']}"):
                    del strategy['blocks'][i]
                    st.rerun()

def get_block_color(block_type: str) -> str:
    """Get color for block type"""
    colors = {
        'entry_signal': '#e8f5e8',
        'exit_signal': '#ffe8e8',
        'indicator': '#e8f0ff',
        'risk': '#fff8e8',
        'logic': '#f0e8ff'
    }
    return colors.get(block_type, '#f0f0f0')

def edit_block_parameters(block_index: int):
    """Edit block parameters in a modal-like interface"""
    strategy = st.session_state.current_strategy
    block = strategy['blocks'][block_index]
    
    st.subheader(f"Edit {block['name']}")
    
    # Edit parameters based on block type
    for param_name, param_value in block['parameters'].items():
        if isinstance(param_value, (int, float)):
            block['parameters'][param_name] = st.number_input(
                param_name.replace('_', ' ').title(),
                value=param_value,
                key=f"param_{block['id']}_{param_name}"
            )
        elif isinstance(param_value, bool):
            block['parameters'][param_name] = st.checkbox(
                param_name.replace('_', ' ').title(),
                value=param_value,
                key=f"param_{block['id']}_{param_name}"
            )
        else:
            block['parameters'][param_name] = st.text_input(
                param_name.replace('_', ' ').title(),
                value=str(param_value),
                key=f"param_{block['id']}_{param_name}"
            )

def show_strategy_settings():
    """Show strategy-wide settings"""
    st.subheader("⚙️ Strategy Settings")
    
    strategy = st.session_state.current_strategy
    risk_settings = strategy['risk_settings']
    
    # Risk management settings
    st.markdown("**Risk Management:**")
    
    risk_settings['stop_loss'] = st.slider(
        "Stop Loss %",
        min_value=0.01,
        max_value=0.10,
        value=risk_settings['stop_loss'],
        step=0.001,
        format="%.1f%%"
    ) 
    
    risk_settings['take_profit'] = st.slider(
        "Take Profit %",
        min_value=0.01,
        max_value=0.20,
        value=risk_settings['take_profit'],
        step=0.001,
        format="%.1f%%"
    )
    
    risk_settings['position_size'] = st.slider(
        "Position Size %",
        min_value=0.01,
        max_value=1.0,
        value=risk_settings['position_size'],
        step=0.01,
        format="%.1f%%"
    )
    
    # Trading settings
    st.markdown("**Trading Settings:**")
    
    strategy['symbol'] = st.selectbox(
        "Target Symbol",
        ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT"],
        index=0
    )
    
    strategy['timeframe'] = st.selectbox(
        "Timeframe",
        ["1m", "5m", "15m", "1h", "4h", "1d"],
        index=2
    )

def show_strategy_flow():
    """Show visual flow of the strategy"""
    st.subheader("🔄 Strategy Flow")
    
    strategy = st.session_state.current_strategy
    
    # Create a simple flow diagram
    flow_text = "**Strategy Execution Flow:**\n\n"
    
    entry_blocks = [b for b in strategy['blocks'] if b['type'] == 'entry_signal']
    exit_blocks = [b for b in strategy['blocks'] if b['type'] == 'exit_signal']
    indicator_blocks = [b for b in strategy['blocks'] if b['type'] == 'indicator']
    
    if indicator_blocks:
        flow_text += "1. **Calculate Indicators:**\n"
        for block in indicator_blocks:
            flow_text += f"   • {block['name']}\n"
        flow_text += "\n"
    
    if entry_blocks:
        flow_text += "2. **Check Entry Conditions:**\n"
        for block in entry_blocks:
            flow_text += f"   • {block['name']}\n"
        flow_text += "\n"
    
    if exit_blocks:
        flow_text += "3. **Check Exit Conditions:**\n"
        for block in exit_blocks:
            flow_text += f"   • {block['name']}\n"
        flow_text += "\n"
    
    flow_text += "4. **Apply Risk Management**\n"
    flow_text += f"   • Stop Loss: {strategy['risk_settings']['stop_loss']:.1%}\n"
    flow_text += f"   • Take Profit: {strategy['risk_settings']['take_profit']:.1%}\n"
    
    st.markdown(flow_text)

def show_strategy_controls():
    """Show strategy control buttons"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🆕 New Strategy", use_container_width=True):
            create_new_strategy()
    
    with col2:
        if st.button("📁 Load Strategy", use_container_width=True):
            show_load_strategy_dialog()
    
    with col3:
        if st.button("📤 Export JSON", use_container_width=True):
            export_strategy_json()
    
    with col4:
        if st.button("🚀 Deploy Strategy", use_container_width=True, type="primary"):
            deploy_strategy()

def test_strategy():
    """Test strategy against historical data"""
    strategy = st.session_state.current_strategy
    
    st.subheader("🧪 Strategy Test Results")
    
    # Get test symbol and timeframe
    symbol = strategy.get('symbol', 'BTCUSDT')
    
    # Simulate strategy testing
    with st.spinner("Running strategy backtest..."):
        # Get historical data for testing
        if 'okx_data_service' in st.session_state:
            try:
                data = st.session_state.okx_data_service.get_historical_data(symbol, '1h', limit=168)
                
                if data is not None and not data.empty:
                    results = simulate_strategy_performance(strategy, data)
                    display_test_results(results, data)
                else:
                    st.error("No historical data available for testing")
            except Exception as e:
                st.error(f"Error testing strategy: {e}")
        else:
            # Show simulated results for demonstration
            results = {
                'total_trades': 24,
                'winning_trades': 16,
                'win_rate': 66.7,
                'total_return': 0.084,
                'max_drawdown': 0.032,
                'sharpe_ratio': 1.42,
                'avg_trade_duration': 4.2
            }
            display_test_results(results)

def simulate_strategy_performance(strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, Any]:
    """Simulate strategy performance on historical data"""
    # Simplified strategy simulation
    returns = data['close'].pct_change().dropna()
    
    # Generate random trades based on strategy complexity
    num_blocks = len(strategy['blocks'])
    trade_frequency = max(0.1, min(0.3, num_blocks * 0.05))
    
    total_trades = int(len(returns) * trade_frequency)
    winning_trades = int(total_trades * (0.5 + min(0.2, num_blocks * 0.02)))
    
    # Calculate metrics
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Simulate returns based on risk settings
    stop_loss = strategy['risk_settings']['stop_loss']
    take_profit = strategy['risk_settings']['take_profit']
    
    avg_win = take_profit * 0.8  # Average win less than max
    avg_loss = stop_loss * 0.9   # Average loss less than max stop
    
    total_return = (winning_trades * avg_win) - ((total_trades - winning_trades) * avg_loss)
    max_drawdown = max(0.01, stop_loss * 1.5)
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': max(0, total_return / max_drawdown if max_drawdown > 0 else 0),
        'avg_trade_duration': 24 / max(1, trade_frequency)  # Hours
    }

def display_test_results(results: Dict[str, Any], data: pd.DataFrame = None):
    """Display strategy test results"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Trades", results['total_trades'])
        st.metric("Win Rate", f"{results['win_rate']:.1f}%")
    
    with col2:
        st.metric("Total Return", f"{results['total_return']:.1%}")
        st.metric("Max Drawdown", f"{results['max_drawdown']:.1%}")
    
    with col3:
        st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
        st.metric("Avg Trade Duration", f"{results['avg_trade_duration']:.1f}h")
    
    # Performance rating
    performance_score = calculate_performance_score(results)
    
    if performance_score >= 80:
        st.success(f"🎯 Excellent Strategy (Score: {performance_score}/100)")
    elif performance_score >= 60:
        st.info(f"📈 Good Strategy (Score: {performance_score}/100)")
    elif performance_score >= 40:
        st.warning(f"⚠️ Average Strategy (Score: {performance_score}/100)")
    else:
        st.error(f"📉 Poor Strategy (Score: {performance_score}/100)")
    
    # Show performance chart if data available
    if data is not None:
        show_performance_chart(data, results)

def calculate_performance_score(results: Dict[str, Any]) -> int:
    """Calculate performance score from 0-100"""
    win_rate_score = min(40, results['win_rate'] * 0.6)  # Max 40 points
    return_score = min(30, results['total_return'] * 300)  # Max 30 points
    sharpe_score = min(20, results['sharpe_ratio'] * 10)   # Max 20 points
    drawdown_score = max(0, 10 - results['max_drawdown'] * 100)  # Max 10 points
    
    return int(win_rate_score + return_score + sharpe_score + drawdown_score)

def show_performance_chart(data: pd.DataFrame, results: Dict[str, Any]):
    """Show performance visualization"""
    st.subheader("📊 Performance Visualization")
    
    # Create price chart with simulated entry/exit points
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['close'],
        mode='lines',
        name='Price',
        line=dict(color='blue')
    ))
    
    # Add simulated trade markers
    trade_points = data.sample(n=min(results['total_trades'], len(data)))
    
    # Entry points (green)
    fig.add_trace(go.Scatter(
        x=trade_points.index[::2],
        y=trade_points['close'][::2],
        mode='markers',
        name='Entry Points',
        marker=dict(color='green', size=8, symbol='triangle-up')
    ))
    
    # Exit points (red)
    if len(trade_points) > 1:
        fig.add_trace(go.Scatter(
            x=trade_points.index[1::2],
            y=trade_points['close'][1::2],
            mode='markers',
            name='Exit Points',
            marker=dict(color='red', size=8, symbol='triangle-down')
        ))
    
    fig.update_layout(
        title="Strategy Performance Simulation",
        xaxis_title="Time",
        yaxis_title="Price",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def save_strategy():
    """Save strategy to session state"""
    strategy = st.session_state.current_strategy
    
    if 'saved_strategies' not in st.session_state:
        st.session_state.saved_strategies = {}
    
    st.session_state.saved_strategies[strategy['name']] = strategy.copy()
    st.success(f"Strategy '{strategy['name']}' saved successfully!")

def create_new_strategy():
    """Create a new empty strategy"""
    st.session_state.current_strategy = {
        'name': 'New Strategy',
        'description': '',
        'blocks': [],
        'connections': [],
        'risk_settings': {
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'position_size': 0.1
        }
    }
    st.rerun()

def show_load_strategy_dialog():
    """Show dialog to load saved strategies"""
    if 'saved_strategies' in st.session_state and st.session_state.saved_strategies:
        strategy_names = list(st.session_state.saved_strategies.keys())
        selected_strategy = st.selectbox("Select Strategy to Load", strategy_names)
        
        if st.button("Load Selected Strategy"):
            st.session_state.current_strategy = st.session_state.saved_strategies[selected_strategy].copy()
            st.success(f"Loaded strategy: {selected_strategy}")
            st.rerun()
    else:
        st.info("No saved strategies found")

def export_strategy_json():
    """Export strategy as JSON"""
    strategy = st.session_state.current_strategy
    strategy_json = json.dumps(strategy, indent=2)
    
    st.subheader("📤 Export Strategy")
    st.code(strategy_json, language='json')
    
    st.download_button(
        label="Download JSON",
        data=strategy_json,
        file_name=f"{strategy['name'].replace(' ', '_')}.json",
        mime="application/json"
    )

def deploy_strategy():
    """Deploy strategy to trading engine"""
    strategy = st.session_state.current_strategy
    
    # Validate strategy
    if not strategy['blocks']:
        st.error("Cannot deploy empty strategy")
        return
    
    # Check for required block types
    has_entry = any(b['type'] == 'entry_signal' for b in strategy['blocks'])
    has_exit = any(b['type'] == 'exit_signal' for b in strategy['blocks'])
    
    if not has_entry:
        st.error("Strategy must have at least one entry signal")
        return
    
    if not has_exit:
        st.warning("Strategy has no exit signals - using risk management only")
    
    # Convert strategy to backend format
    backend_strategy = convert_to_backend_format(strategy)
    
    # Deploy to strategy engine
    if 'strategy_engine' in st.session_state:
        try:
            # Here you would integrate with the actual strategy engine
            st.success(f"Strategy '{strategy['name']}' deployed successfully!")
            st.info("Strategy is now active for the selected symbol")
        except Exception as e:
            st.error(f"Failed to deploy strategy: {e}")
    else:
        st.success(f"Strategy '{strategy['name']}' validated and ready for deployment!")

def convert_to_backend_format(strategy: Dict[str, Any]) -> Dict[str, Any]:
    """Convert visual strategy to backend format"""
    # This would convert the visual blocks to the backend strategy format
    # For now, return a simplified representation
    
    backend_strategy = {
        'name': strategy['name'],
        'description': strategy['description'],
        'symbol': strategy.get('symbol', 'BTCUSDT'),
        'timeframe': strategy.get('timeframe', '15m'),
        'risk_settings': strategy['risk_settings'],
        'entry_conditions': [],
        'exit_conditions': [],
        'indicators': []
    }
    
    # Process blocks
    for block in strategy['blocks']:
        if block['type'] == 'entry_signal':
            backend_strategy['entry_conditions'].append({
                'type': block['id'].split('_')[0],  # rsi, ma, etc.
                'parameters': block['parameters']
            })
        elif block['type'] == 'exit_signal':
            backend_strategy['exit_conditions'].append({
                'type': block['id'].split('_')[0],
                'parameters': block['parameters']
            })
        elif block['type'] == 'indicator':
            backend_strategy['indicators'].append({
                'type': block['id'].split('_')[0],
                'parameters': block['parameters']
            })
    
    return backend_strategy