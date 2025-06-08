"""
Multi-Market Dashboard - Comprehensive view of all USDT trading pairs across Spot and Futures
Dynamic symbol support with real-time market data and analytics
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

def show_multi_market_dashboard():
    """Show comprehensive multi-market trading dashboard"""
    st.title("🌐 Multi-Market Trading Dashboard")
    st.markdown("Real-time view of all USDT trading pairs across Spot and Futures markets")
    
    # Initialize trading controller if not available
    if 'trading_controller' not in st.session_state:
        st.error("Trading controller not initialized. Please restart the application.")
        return
    
    controller = st.session_state.trading_controller
    
    # Market overview tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "💼 Portfolio", "📈 Symbol Explorer", "⚙️ Settings"])
    
    with tab1:
        show_market_overview(controller)
    
    with tab2:
        show_portfolio_view(controller)
    
    with tab3:
        show_symbol_explorer(controller)
    
    with tab4:
        show_market_settings(controller)

def show_market_overview(controller):
    """Show comprehensive market overview"""
    st.subheader("📊 Market Overview")
    
    # Get market statistics
    try:
        market_stats = controller.get_market_statistics()
        portfolio_overview = controller.get_portfolio_overview()
        
        # Top metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Active Symbols", 
                market_stats.get('active_symbols_count', 0),
                help="Total tradeable symbols across all markets"
            )
        
        with col2:
            st.metric(
                "Spot Pairs", 
                market_stats.get('spot_symbols', 0),
                help="Available spot trading pairs"
            )
        
        with col3:
            st.metric(
                "Futures Pairs", 
                market_stats.get('futures_symbols', 0),
                help="Available futures trading pairs"
            )
        
        with col4:
            total_positions = portfolio_overview.get('active_positions', 0)
            st.metric(
                "Active Positions", 
                total_positions,
                help="Current open positions across all markets"
            )
        
        # Market breakdown
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Distribution")
            
            # Create market distribution chart
            labels = ['Spot', 'Futures']
            values = [
                market_stats.get('spot_symbols', 0),
                market_stats.get('futures_symbols', 0)
            ]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values,
                hole=0.3,
                marker_colors=['#26C281', '#ED4C78']
            )])
            
            fig.update_layout(
                title="Symbol Distribution by Market",
                height=300,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Portfolio Value Distribution")
            
            # Portfolio breakdown
            markets = portfolio_overview.get('markets', {})
            portfolio_labels = []
            portfolio_values = []
            
            if 'spot' in markets:
                portfolio_labels.append('Spot Portfolio')
                portfolio_values.append(markets['spot'].get('total_value', 0))
            
            if 'futures' in markets:
                portfolio_labels.append('Futures Exposure')
                portfolio_values.append(markets['futures'].get('total_exposure', 0))
            
            if portfolio_values:
                fig = go.Figure(data=[go.Pie(
                    labels=portfolio_labels, 
                    values=portfolio_values,
                    hole=0.3,
                    marker_colors=['#1f77b4', '#ff7f0e']
                )])
                
                fig.update_layout(
                    title="Portfolio Value by Market",
                    height=300,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No portfolio data available")
        
        # Top symbols by volume
        st.markdown("---")
        st.subheader("🔥 Top Volume Symbols")
        
        show_top_symbols_table(controller)
        
    except Exception as e:
        st.error(f"Error loading market overview: {e}")

def show_top_symbols_table(controller):
    """Show table of top symbols by volume"""
    try:
        symbol_manager = controller.symbol_manager
        
        # Get top symbols by volume for each market
        top_spot = symbol_manager.get_symbols_by_volume(
            market_type=symbol_manager.MarketType.SPOT, limit=10
        )
        top_futures = symbol_manager.get_symbols_by_volume(
            market_type=symbol_manager.MarketType.FUTURES, limit=10
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Spot Symbols**")
            spot_data = []
            for symbol in top_spot:
                info = symbol_manager.get_symbol_info(symbol)
                if info:
                    spot_data.append({
                        'Symbol': symbol,
                        'Base Asset': info.base_asset,
                        '24h Volume': f"{info.volume_24h:,.0f}",
                        'Active': "✅" if info.is_active else "❌"
                    })
            
            if spot_data:
                st.dataframe(pd.DataFrame(spot_data), use_container_width=True)
            else:
                st.info("No spot symbols available")
        
        with col2:
            st.markdown("**Top Futures Symbols**")
            futures_data = []
            for symbol in top_futures:
                info = symbol_manager.get_symbol_info(symbol)
                if info:
                    funding_rate = f"{info.funding_rate:.4f}" if info.funding_rate else "N/A"
                    futures_data.append({
                        'Symbol': symbol,
                        'Base Asset': info.base_asset,
                        '24h Volume': f"{info.volume_24h:,.0f}",
                        'Funding Rate': funding_rate
                    })
            
            if futures_data:
                st.dataframe(pd.DataFrame(futures_data), use_container_width=True)
            else:
                st.info("No futures symbols available")
                
    except Exception as e:
        st.error(f"Error loading symbols table: {e}")

def show_portfolio_view(controller):
    """Show detailed portfolio view across markets"""
    st.subheader("💼 Portfolio Overview")
    
    try:
        portfolio_overview = controller.get_portfolio_overview()
        
        # Portfolio summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_value = portfolio_overview.get('total_value', 0)
            st.metric("Total Portfolio Value", f"${total_value:,.2f}")
        
        with col2:
            unrealized_pnl = portfolio_overview.get('unrealized_pnl', 0)
            pnl_color = "normal" if unrealized_pnl >= 0 else "inverse"
            st.metric("Unrealized P&L", f"${unrealized_pnl:,.2f}", delta=None)
        
        with col3:
            active_positions = portfolio_overview.get('active_positions', 0)
            st.metric("Active Positions", active_positions)
        
        with col4:
            active_symbols = len(portfolio_overview.get('active_symbols', []))
            st.metric("Active Symbols", active_symbols)
        
        # Market-specific views
        st.markdown("---")
        
        markets = portfolio_overview.get('markets', {})
        
        if 'spot' in markets and 'futures' in markets:
            tab1, tab2 = st.tabs(["💰 Spot Portfolio", "📊 Futures Portfolio"])
            
            with tab1:
                show_spot_portfolio(controller, markets['spot'])
            
            with tab2:
                show_futures_portfolio(controller, markets['futures'])
        
        elif 'spot' in markets:
            show_spot_portfolio(controller, markets['spot'])
        
        elif 'futures' in markets:
            show_futures_portfolio(controller, markets['futures'])
        
        else:
            st.info("No active positions in any market")
        
        # Position details
        st.markdown("---")
        st.subheader("📋 All Positions")
        
        all_positions = controller.get_all_positions()
        if all_positions:
            positions_data = []
            for pos_id, position in all_positions.items():
                positions_data.append({
                    'Symbol': position.get('symbol', 'Unknown'),
                    'Market': position.get('market_type', 'Unknown').title(),
                    'Size': f"{position.get('size', 0):.6f}",
                    'Avg Price': f"${position.get('avg_price', 0):.2f}",
                    'Unrealized P&L': f"${position.get('unrealized_pnl', 0):.2f}",
                    'Side': position.get('side', 'Unknown').title() if 'side' in position else 'Long'
                })
            
            st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
        else:
            st.info("No active positions")
            
    except Exception as e:
        st.error(f"Error loading portfolio: {e}")

def show_spot_portfolio(controller, spot_data):
    """Show detailed spot portfolio"""
    st.subheader("💰 Spot Portfolio")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cash_balance = spot_data.get('cash_balance', 0)
        st.metric("Cash Balance", f"${cash_balance:,.2f}")
    
    with col2:
        total_value = spot_data.get('total_value', 0)
        st.metric("Total Value", f"${total_value:,.2f}")
    
    with col3:
        unrealized_pnl = spot_data.get('unrealized_pnl', 0)
        st.metric("Unrealized P&L", f"${unrealized_pnl:,.2f}")
    
    # Spot positions
    spot_positions = controller.get_positions_by_market(controller.symbol_manager.MarketType.SPOT)
    
    if spot_positions:
        st.markdown("**Spot Positions:**")
        spot_pos_data = []
        for symbol, position in spot_positions.items():
            spot_pos_data.append({
                'Symbol': symbol,
                'Size': f"{position.get('size', 0):.6f}",
                'Avg Price': f"${position.get('avg_price', 0):.2f}",
                'Value': f"${position.get('size', 0) * position.get('avg_price', 0):.2f}",
                'P&L': f"${position.get('unrealized_pnl', 0):.2f}"
            })
        
        st.dataframe(pd.DataFrame(spot_pos_data), use_container_width=True)
    else:
        st.info("No spot positions")

def show_futures_portfolio(controller, futures_data):
    """Show detailed futures portfolio"""
    st.subheader("📊 Futures Portfolio")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        margin_balance = futures_data.get('margin_balance', 0)
        st.metric("Margin Balance", f"${margin_balance:,.2f}")
    
    with col2:
        margin_used = futures_data.get('total_margin_used', 0)
        st.metric("Margin Used", f"${margin_used:,.2f}")
    
    with col3:
        total_exposure = futures_data.get('total_exposure', 0)
        st.metric("Total Exposure", f"${total_exposure:,.2f}")
    
    with col4:
        unrealized_pnl = futures_data.get('unrealized_pnl', 0)
        st.metric("Unrealized P&L", f"${unrealized_pnl:,.2f}")
    
    # Futures positions
    futures_positions = controller.get_positions_by_market(controller.symbol_manager.MarketType.FUTURES)
    
    if futures_positions:
        st.markdown("**Futures Positions:**")
        futures_pos_data = []
        for symbol, position in futures_positions.items():
            side = position.get('side', 'unknown')
            leverage = position.get('leverage', 1)
            futures_pos_data.append({
                'Symbol': symbol,
                'Side': side.title(),
                'Size': f"{position.get('size', 0):.6f}",
                'Avg Price': f"${position.get('avg_price', 0):.2f}",
                'Leverage': f"{leverage}x",
                'Margin': f"${position.get('margin', 0):.2f}",
                'P&L': f"${position.get('unrealized_pnl', 0):.2f}"
            })
        
        st.dataframe(pd.DataFrame(futures_pos_data), use_container_width=True)
        
        # Funding rates
        funding_rates = futures_data.get('funding_rates', {})
        if funding_rates:
            st.markdown("**Funding Rates:**")
            funding_data = []
            for symbol, rate in funding_rates.items():
                funding_data.append({
                    'Symbol': symbol,
                    'Funding Rate': f"{rate:.6f}",
                    'Annual Rate': f"{rate * 24 * 365:.2%}"
                })
            
            st.dataframe(pd.DataFrame(funding_data), use_container_width=True)
    else:
        st.info("No futures positions")

def show_symbol_explorer(controller):
    """Show dynamic symbol explorer"""
    st.subheader("📈 Symbol Explorer")
    
    try:
        symbol_manager = controller.symbol_manager
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            market_filter = st.selectbox(
                "Market Type",
                ["All", "Spot", "Futures"],
                index=0
            )
        
        with col2:
            min_volume = st.number_input(
                "Min 24h Volume",
                min_value=0,
                value=100000,
                step=50000,
                format="%d"
            )
        
        with col3:
            limit = st.number_input(
                "Max Results",
                min_value=10,
                max_value=100,
                value=50,
                step=10
            )
        
        # Apply filters
        market_type = None
        if market_filter == "Spot":
            market_type = symbol_manager.MarketType.SPOT
        elif market_filter == "Futures":
            market_type = symbol_manager.MarketType.FUTURES
        
        # Get filtered symbols
        filtered_symbols = symbol_manager.filter_symbols(
            market_type=market_type,
            min_volume=min_volume,
            active_only=True
        )[:limit]
        
        st.markdown(f"**Found {len(filtered_symbols)} symbols matching criteria**")
        
        # Symbol table
        if filtered_symbols:
            symbol_data = []
            for symbol in filtered_symbols:
                info = symbol_manager.get_symbol_info(symbol)
                if info:
                    symbol_data.append({
                        'Symbol': symbol,
                        'Market': info.market_type.value.title(),
                        'Base Asset': info.base_asset,
                        '24h Volume': f"{info.volume_24h:,.0f}",
                        'Min Notional': f"{info.min_notional:.4f}",
                        'Tick Size': f"{info.tick_size:.6f}",
                        'Active': "✅" if info.is_active else "❌"
                    })
            
            df = pd.DataFrame(symbol_data)
            
            # Interactive table with selection
            selected_symbol = st.selectbox(
                "Select Symbol for Details",
                [""] + [row['Symbol'] for row in symbol_data]
            )
            
            st.dataframe(df, use_container_width=True)
            
            # Show detailed info for selected symbol
            if selected_symbol:
                show_symbol_details(controller, selected_symbol)
        
        else:
            st.info("No symbols match the current filters")
            
    except Exception as e:
        st.error(f"Error in symbol explorer: {e}")

def show_symbol_details(controller, symbol):
    """Show detailed information for selected symbol"""
    st.markdown("---")
    st.subheader(f"📊 {symbol} Details")
    
    try:
        symbol_info = controller.get_symbol_info(symbol)
        
        if symbol_info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Symbol Information:**")
                st.markdown(f"- **Market Type:** {symbol_info.market_type.value.title()}")
                st.markdown(f"- **Base Asset:** {symbol_info.base_asset}")
                st.markdown(f"- **Quote Asset:** {symbol_info.quote_asset}")
                st.markdown(f"- **Active:** {'Yes' if symbol_info.is_active else 'No'}")
                st.markdown(f"- **24h Volume:** {symbol_info.volume_24h:,.0f}")
            
            with col2:
                st.markdown("**Trading Specifications:**")
                st.markdown(f"- **Min Notional:** {symbol_info.min_notional:.4f}")
                st.markdown(f"- **Tick Size:** {symbol_info.tick_size:.6f}")
                st.markdown(f"- **Step Size:** {symbol_info.step_size:.6f}")
                st.markdown(f"- **Price Precision:** {symbol_info.price_precision}")
                st.markdown(f"- **Quantity Precision:** {symbol_info.quantity_precision}")
                
                if symbol_info.market_type.value == "futures":
                    st.markdown(f"- **Funding Rate:** {symbol_info.funding_rate:.6f}" if symbol_info.funding_rate else "- **Funding Rate:** N/A")
        
        else:
            st.error("Symbol information not available")
            
    except Exception as e:
        st.error(f"Error loading symbol details: {e}")

def show_market_settings(controller):
    """Show market configuration settings"""
    st.subheader("⚙️ Market Settings")
    
    try:
        # Symbol discovery settings
        st.markdown("**Symbol Discovery**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Update Symbol Discovery"):
                with st.spinner("Updating symbol discovery..."):
                    success = controller.update_symbol_discovery()
                    if success:
                        st.success("Symbol discovery updated successfully!")
                        st.rerun()
                    else:
                        st.error("Failed to update symbol discovery")
        
        with col2:
            market_stats = controller.get_market_statistics()
            last_update = market_stats.get('last_update', 'Never')
            st.info(f"Last Update: {last_update}")
        
        # Symbol statistics
        st.markdown("---")
        st.markdown("**Symbol Statistics**")
        
        symbol_stats = controller.symbol_manager.get_symbol_statistics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Symbols", symbol_stats.get('total_symbols', 0))
        
        with col2:
            st.metric("Spot Symbols", symbol_stats.get('spot_symbols', 0))
        
        with col3:
            st.metric("Futures Symbols", symbol_stats.get('futures_symbols', 0))
        
        # Volume statistics
        col1, col2 = st.columns(2)
        
        with col1:
            total_volume = symbol_stats.get('total_volume_24h', 0)
            st.metric("Total 24h Volume", f"{total_volume:,.0f} USDT")
        
        with col2:
            update_interval = symbol_stats.get('update_interval_hours', 1)
            st.metric("Update Interval", f"{update_interval} hours")
        
        # Export configuration
        st.markdown("---")
        st.markdown("**Export Configuration**")
        
        if st.button("📄 Export Trading Config"):
            try:
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    controller.export_trading_config(f.name)
                    
                    # Read the file content
                    with open(f.name, 'r') as config_file:
                        config_content = config_file.read()
                    
                    st.download_button(
                        label="📥 Download Configuration",
                        data=config_content,
                        file_name=f"trading_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                    # Clean up temp file
                    os.unlink(f.name)
                    
            except Exception as e:
                st.error(f"Error exporting configuration: {e}")
                
    except Exception as e:
        st.error(f"Error loading settings: {e}")