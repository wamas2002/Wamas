"""
UI Mode Manager - Handles beginner/expert mode switching with enhanced UX
"""

import streamlit as st
from typing import Dict, Any

def initialize_ui_mode():
    """Initialize UI mode in session state"""
    if 'user_mode' not in st.session_state:
        st.session_state.user_mode = 'beginner'
    
    if 'ui_preferences' not in st.session_state:
        st.session_state.ui_preferences = {
            'theme': 'light',
            'show_tooltips': True,
            'auto_refresh': True,
            'advanced_charts': False
        }

def show_mode_selector():
    """Show mode selector in sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Interface Mode")
    
    current_mode = st.session_state.user_mode
    
    # Mode selection
    mode = st.sidebar.radio(
        "Select Mode:",
        options=['beginner', 'expert'],
        index=0 if current_mode == 'beginner' else 1,
        format_func=lambda x: "🔰 Beginner" if x == 'beginner' else "⚡ Expert"
    )
    
    if mode != current_mode:
        st.session_state.user_mode = mode
        st.rerun()
    
    # Mode description
    if mode == 'beginner':
        st.sidebar.info("""
        **Beginner Mode Features:**
        • Simplified interface
        • Essential controls only
        • Guided tooltips
        • Auto-optimization
        """)
    else:
        st.sidebar.info("""
        **Expert Mode Features:**
        • Full control access
        • Advanced analytics
        • Custom strategies
        • Manual optimization
        """)
    
    # UI preferences
    show_ui_preferences()

def show_ui_preferences():
    """Show UI customization preferences"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Preferences")
    
    prefs = st.session_state.ui_preferences
    
    # Theme selection
    prefs['theme'] = st.sidebar.selectbox(
        "Theme",
        options=['light', 'dark', 'auto'],
        index=['light', 'dark', 'auto'].index(prefs['theme'])
    )
    
    # Other preferences
    prefs['show_tooltips'] = st.sidebar.checkbox(
        "Show Tooltips",
        value=prefs['show_tooltips']
    )
    
    prefs['auto_refresh'] = st.sidebar.checkbox(
        "Auto Refresh",
        value=prefs['auto_refresh']
    )
    
    if st.session_state.user_mode == 'expert':
        prefs['advanced_charts'] = st.sidebar.checkbox(
            "Advanced Charts",
            value=prefs['advanced_charts']
        )

def get_navigation_items() -> Dict[str, Any]:
    """Get navigation items based on current mode"""
    mode = st.session_state.user_mode
    
    if mode == 'beginner':
        return {
            "portfolio": {
                "label": "💰 Portfolio",
                "description": "View your trading portfolio"
            },
            "top_picks": {
                "label": "⭐ Top Picks",
                "description": "AI-recommended trading opportunities"
            },
            "advisor": {
                "label": "🤖 AI Advisor",
                "description": "Get AI trading advice"
            },
            "charts": {
                "label": "📊 Charts",
                "description": "View market charts"
            }
        }
    else:  # expert mode
        return {
            "portfolio": {
                "label": "💰 Portfolio",
                "description": "Portfolio overview and analytics"
            },
            "top_picks": {
                "label": "⭐ Top Picks",
                "description": "AI-recommended opportunities"
            },
            "advisor": {
                "label": "🤖 AI Advisor",
                "description": "AI financial advisor"
            },
            "charts": {
                "label": "📊 Charts",
                "description": "Interactive market charts"
            },
            "enhanced_dashboard": {
                "label": "📈 Enhanced Dashboard",
                "description": "Modern responsive dashboard"
            },
            "visual_builder": {
                "label": "🎨 Strategy Builder",
                "description": "Visual drag-and-drop strategy editor"
            },
            "ai_performance": {
                "label": "🔬 AI Performance",
                "description": "AI model performance analytics"
            },
            "explainable_ai": {
                "label": "🧠 Explainable AI",
                "description": "AI decision explanations"
            },
            "advanced_ml": {
                "label": "🤖 Advanced ML",
                "description": "Machine learning dashboard"
            },
            "strategy_monitor": {
                "label": "📋 Strategy Monitor",
                "description": "Strategy management dashboard"
            },
            "auto_analyzer": {
                "label": "🔍 Auto Analyzer",
                "description": "Automated market analysis"
            },
            "risk_manager": {
                "label": "🛡️ Risk Manager",
                "description": "Advanced risk management"
            },
            "explorer": {
                "label": "🌐 Explorer",
                "description": "Asset exploration tools"
            },
            "sentiment": {
                "label": "📰 Sentiment",
                "description": "Market sentiment analysis"
            },
            "strategies": {
                "label": "⚙️ Strategies",
                "description": "Strategy configuration"
            },
            "alerts": {
                "label": "🔔 Alerts",
                "description": "Alert management"
            }
        }

def show_enhanced_navigation():
    """Show enhanced navigation with mode-appropriate items"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Navigation")
    
    nav_items = get_navigation_items()
    mode = st.session_state.user_mode
    
    # Show mode indicator
    mode_indicator = "🔰 Beginner Mode" if mode == 'beginner' else "⚡ Expert Mode"
    st.sidebar.markdown(f"**Current Mode:** {mode_indicator}")
    
    # Navigation menu
    selected_page = None
    
    for page_key, page_info in nav_items.items():
        if st.sidebar.button(
            page_info["label"],
            key=f"nav_{page_key}",
            use_container_width=True,
            help=page_info["description"] if st.session_state.ui_preferences['show_tooltips'] else None
        ):
            selected_page = page_key
    
    return selected_page

def show_beginner_help_panel():
    """Show help panel for beginner users"""
    if st.session_state.user_mode == 'beginner':
        with st.expander("❓ Need Help?"):
            st.markdown("""
            **Getting Started:**
            
            1. **Portfolio** - See your current investments and profits
            2. **Top Picks** - Get AI recommendations for trades
            3. **AI Advisor** - Ask questions about trading
            4. **Charts** - View price movements and trends
            
            **Tips:**
            - The AI automatically manages risk and strategy selection
            - Green numbers mean profit, red means loss
            - All trading is done safely with built-in protections
            
            **Need Expert Features?**
            Switch to Expert Mode in the sidebar for advanced controls.
            """)

def apply_mode_specific_styling():
    """Apply CSS styling based on current mode"""
    mode = st.session_state.user_mode
    theme = st.session_state.ui_preferences['theme']
    
    if mode == 'beginner':
        # Simplified, clean styling for beginners
        st.markdown("""
        <style>
        .beginner-mode {
            font-size: 1.1em;
            line-height: 1.6;
        }
        
        .metric-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.75rem;
            border: 2px solid #e9ecef;
            margin: 1rem 0;
            text-align: center;
        }
        
        .help-text {
            background-color: #e7f3ff;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #0066cc;
            margin: 1rem 0;
        }
        
        .simplified-button {
            background-color: #28a745;
            color: white;
            border: none;
            padding: 1rem 2rem;
            border-radius: 0.5rem;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            width: 100%;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    else:  # expert mode
        # Advanced, feature-rich styling for experts
        st.markdown("""
        <style>
        .expert-mode {
            font-size: 0.95em;
        }
        
        .advanced-metric {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        
        .expert-panel {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 0.375rem;
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .performance-indicator {
            display: inline-block;
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
            font-size: 0.875em;
            font-weight: bold;
        }
        
        .performance-good {
            background-color: #d4edda;
            color: #155724;
        }
        
        .performance-poor {
            background-color: #f8d7da;
            color: #721c24;
        }
        </style>
        """, unsafe_allow_html=True)

def show_quick_actions():
    """Show quick action buttons based on mode"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Actions")
    
    mode = st.session_state.user_mode
    
    if mode == 'beginner':
        # Simple actions for beginners
        if st.sidebar.button("🎯 Get AI Recommendation", use_container_width=True):
            st.session_state.quick_action = 'ai_recommendation'
        
        if st.sidebar.button("📊 View Performance", use_container_width=True):
            st.session_state.quick_action = 'view_performance'
        
        if st.sidebar.button("❓ Get Help", use_container_width=True):
            st.session_state.quick_action = 'get_help'
    
    else:  # expert mode
        # Advanced actions for experts
        if st.sidebar.button("🔄 Refresh All Data", use_container_width=True):
            st.session_state.quick_action = 'refresh_data'
        
        if st.sidebar.button("⚙️ Optimize Strategies", use_container_width=True):
            st.session_state.quick_action = 'optimize_strategies'
        
        if st.sidebar.button("📈 Generate Report", use_container_width=True):
            st.session_state.quick_action = 'generate_report'
        
        if st.sidebar.button("🧪 Test Strategy", use_container_width=True):
            st.session_state.quick_action = 'test_strategy'

def handle_quick_actions():
    """Handle quick action executions"""
    if 'quick_action' in st.session_state:
        action = st.session_state.quick_action
        
        if action == 'ai_recommendation':
            st.info("🤖 Getting AI recommendation...")
            # Implementation would integrate with AI advisor
        
        elif action == 'view_performance':
            st.info("📊 Loading performance data...")
            # Implementation would show performance metrics
        
        elif action == 'get_help':
            show_beginner_help_panel()
        
        elif action == 'refresh_data':
            st.info("🔄 Refreshing all data sources...")
            # Implementation would refresh data
        
        elif action == 'optimize_strategies':
            st.info("⚙️ Running strategy optimization...")
            # Implementation would trigger optimization
        
        elif action == 'generate_report':
            st.info("📈 Generating performance report...")
            # Implementation would generate report
        
        elif action == 'test_strategy':
            st.info("🧪 Opening strategy tester...")
            # Implementation would open strategy tester
        
        # Clear the action
        del st.session_state.quick_action

def show_mode_transition_guide():
    """Show guide when switching between modes"""
    if 'show_mode_guide' in st.session_state:
        mode = st.session_state.user_mode
        
        if mode == 'expert':
            st.info("""
            🎉 **Welcome to Expert Mode!**
            
            You now have access to:
            • Advanced AI model controls
            • Visual strategy builder
            • Detailed performance analytics
            • Custom risk management
            • Real-time model optimization
            
            Explore the new menu options in the sidebar.
            """)
        else:
            st.info("""
            🔰 **Welcome to Beginner Mode!**
            
            The interface is now simplified with:
            • Essential trading controls
            • Clear performance displays
            • Guided AI recommendations
            • Automatic optimization
            
            All advanced features run automatically in the background.
            """)
        
        if st.button("Got it!"):
            del st.session_state.show_mode_guide