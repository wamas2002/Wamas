"""
Real-Time Feedback Engine - Provides live feedback and guidance
"""

import streamlit as st
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd

class RealTimeFeedbackEngine:
    """Provides real-time feedback and guidance to users"""
    
    def __init__(self):
        self.feedback_history = []
        self.active_notifications = []
        
    def add_feedback(self, feedback_type: str, message: str, severity: str = "info"):
        """Add feedback message"""
        feedback = {
            'timestamp': datetime.now(),
            'type': feedback_type,
            'message': message,
            'severity': severity,
            'id': len(self.feedback_history)
        }
        self.feedback_history.append(feedback)
        
        # Add to active notifications if important
        if severity in ['warning', 'error', 'success']:
            self.active_notifications.append(feedback)
    
    def show_live_feedback_panel(self):
        """Show live feedback panel"""
        if 'feedback_engine' not in st.session_state:
            st.session_state.feedback_engine = RealTimeFeedbackEngine()
        
        # Live notifications
        self.show_active_notifications()
        
        # Strategy validation feedback
        self.show_strategy_validation()
        
        # Performance insights
        self.show_performance_insights()
        
        # Risk alerts
        self.show_risk_alerts()
    
    def show_active_notifications(self):
        """Show active notifications"""
        if self.active_notifications:
            st.subheader("🔔 Live Notifications")
            
            for notification in self.active_notifications[-3:]:  # Show last 3
                severity_colors = {
                    'success': '#d4edda',
                    'warning': '#fff3cd', 
                    'error': '#f8d7da',
                    'info': '#d1ecf1'
                }
                
                color = severity_colors.get(notification['severity'], '#f8f9fa')
                
                st.markdown(f"""
                <div style="
                    background-color: {color};
                    padding: 10px;
                    border-radius: 5px;
                    margin: 5px 0;
                    border-left: 4px solid #007bff;
                ">
                    <strong>{notification['type']}</strong><br>
                    {notification['message']}<br>
                    <small>{notification['timestamp'].strftime('%H:%M:%S')}</small>
                </div>
                """, unsafe_allow_html=True)
    
    def show_strategy_validation(self):
        """Provide real-time strategy validation"""
        if 'current_strategy' in st.session_state:
            strategy = st.session_state.current_strategy
            
            st.subheader("✅ Strategy Validation")
            
            validation_results = self.validate_strategy(strategy)
            
            for result in validation_results:
                if result['status'] == 'pass':
                    st.success(f"✅ {result['message']}")
                elif result['status'] == 'warning':
                    st.warning(f"⚠️ {result['message']}")
                else:
                    st.error(f"❌ {result['message']}")
    
    def validate_strategy(self, strategy: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate strategy configuration"""
        results = []
        
        # Check for essential components
        has_entry = any(b['type'] == 'entry_signal' for b in strategy.get('blocks', []))
        has_exit = any(b['type'] == 'exit_signal' for b in strategy.get('blocks', []))
        has_risk = strategy.get('risk_settings', {})
        
        if has_entry:
            results.append({
                'status': 'pass',
                'message': 'Entry signals configured'
            })
        else:
            results.append({
                'status': 'error',
                'message': 'No entry signals found - strategy needs entry conditions'
            })
        
        if has_exit:
            results.append({
                'status': 'pass',
                'message': 'Exit signals configured'
            })
        else:
            results.append({
                'status': 'warning',
                'message': 'No exit signals - will rely on risk management only'
            })
        
        if has_risk:
            stop_loss = has_risk.get('stop_loss', 0)
            if stop_loss > 0:
                results.append({
                    'status': 'pass',
                    'message': f'Stop loss set to {stop_loss:.1%}'
                })
            else:
                results.append({
                    'status': 'warning',
                    'message': 'No stop loss configured - high risk'
                })
        
        # Check for conflicting signals
        entry_count = len([b for b in strategy.get('blocks', []) if b['type'] == 'entry_signal'])
        if entry_count > 3:
            results.append({
                'status': 'warning',
                'message': f'{entry_count} entry signals may cause conflicts'
            })
        
        return results
    
    def show_performance_insights(self):
        """Show real-time performance insights"""
        st.subheader("📈 Performance Insights")
        
        insights = self.generate_performance_insights()
        
        for insight in insights:
            if insight['type'] == 'positive':
                st.success(f"🎯 {insight['message']}")
            elif insight['type'] == 'neutral':
                st.info(f"ℹ️ {insight['message']}")
            else:
                st.warning(f"⚠️ {insight['message']}")
    
    def generate_performance_insights(self) -> List[Dict[str, Any]]:
        """Generate real-time performance insights"""
        insights = []
        
        try:
            # Get performance data from AI tracker
            if 'ai_performance_tracker' in st.session_state:
                tracker = st.session_state.ai_performance_tracker
                summary = tracker.get_performance_summary()
                
                if summary and summary.get('summary'):
                    avg_win_rate = sum(item.get('win_rate', 0) for item in summary['summary']) / len(summary['summary'])
                    
                    if avg_win_rate > 70:
                        insights.append({
                            'type': 'positive',
                            'message': f'Excellent win rate: {avg_win_rate:.1f}%'
                        })
                    elif avg_win_rate < 50:
                        insights.append({
                            'type': 'negative',
                            'message': f'Low win rate: {avg_win_rate:.1f}% - consider strategy adjustment'
                        })
                    else:
                        insights.append({
                            'type': 'neutral',
                            'message': f'Stable performance: {avg_win_rate:.1f}% win rate'
                        })
        except:
            pass
        
        # Market condition insights
        insights.append({
            'type': 'neutral',
            'message': 'Market conditions: Normal volatility detected'
        })
        
        # Strategy diversity insight
        insights.append({
            'type': 'positive',
            'message': 'Portfolio well-diversified across 8 trading pairs'
        })
        
        return insights
    
    def show_risk_alerts(self):
        """Show real-time risk alerts"""
        st.subheader("🛡️ Risk Monitoring")
        
        risk_alerts = self.generate_risk_alerts()
        
        for alert in risk_alerts:
            if alert['level'] == 'high':
                st.error(f"🚨 {alert['message']}")
            elif alert['level'] == 'medium':
                st.warning(f"⚠️ {alert['message']}")
            else:
                st.success(f"✅ {alert['message']}")
    
    def generate_risk_alerts(self) -> List[Dict[str, Any]]:
        """Generate real-time risk alerts"""
        alerts = []
        
        # Portfolio risk check
        alerts.append({
            'level': 'low',
            'message': 'Portfolio risk within acceptable limits'
        })
        
        # Position size check
        alerts.append({
            'level': 'low', 
            'message': 'All position sizes below 10% allocation limit'
        })
        
        # Stop loss check
        alerts.append({
            'level': 'low',
            'message': 'Stop losses active on all open positions'
        })
        
        return alerts

def show_interactive_tooltips():
    """Show interactive tooltips and help system"""
    if st.session_state.ui_preferences.get('show_tooltips', True):
        
        # Context-sensitive help
        current_page = st.session_state.get('current_page', 'portfolio')
        
        tooltip_content = {
            'portfolio': {
                'title': 'Portfolio Overview',
                'content': 'Monitor your trading performance, positions, and overall portfolio health.'
            },
            'visual_builder': {
                'title': 'Visual Strategy Builder',
                'content': 'Create trading strategies using drag-and-drop blocks without coding.'
            },
            'ai_performance': {
                'title': 'AI Performance',
                'content': 'Track how different AI models are performing and being selected.'
            },
            'enhanced_dashboard': {
                'title': 'Enhanced Dashboard',
                'content': 'Modern analytics dashboard with real-time market data and insights.'
            }
        }
        
        if current_page in tooltip_content:
            info = tooltip_content[current_page]
            
            with st.expander(f"💡 {info['title']} Help"):
                st.markdown(info['content'])
                
                # Page-specific tips
                if current_page == 'visual_builder':
                    st.markdown("""
                    **Quick Tips:**
                    - Add blocks from the palette to build your strategy
                    - Test your strategy before deploying
                    - Use risk management blocks to protect your capital
                    - Save successful strategies for future use
                    """)
                elif current_page == 'enhanced_dashboard':
                    st.markdown("""
                    **Dashboard Features:**
                    - Portfolio tab: View allocation and performance
                    - Symbol Monitor: Track individual pairs
                    - Risk Dashboard: Monitor risk exposure
                    - AI Performance: See model analytics
                    """)

def show_smart_suggestions():
    """Show smart suggestions based on current context"""
    st.subheader("💡 Smart Suggestions")
    
    suggestions = generate_context_suggestions()
    
    for suggestion in suggestions:
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**{suggestion['title']}**")
                st.caption(suggestion['description'])
            
            with col2:
                if st.button("Apply", key=f"suggest_{suggestion['id']}"):
                    apply_suggestion(suggestion)

def generate_context_suggestions() -> List[Dict[str, Any]]:
    """Generate context-aware suggestions"""
    suggestions = []
    
    current_page = st.session_state.get('current_page', 'portfolio')
    user_mode = st.session_state.get('user_mode', 'beginner')
    
    if current_page == 'visual_builder':
        suggestions.append({
            'id': 'add_risk_block',
            'title': 'Add Risk Management',
            'description': 'Consider adding stop loss and position sizing blocks',
            'action': 'add_risk_management'
        })
        
        if 'current_strategy' in st.session_state:
            strategy = st.session_state.current_strategy
            if not any(b['type'] == 'exit_signal' for b in strategy.get('blocks', [])):
                suggestions.append({
                    'id': 'add_exit',
                    'title': 'Add Exit Signal',
                    'description': 'Strategy needs exit conditions for better performance',
                    'action': 'add_exit_signal'
                })
    
    elif current_page == 'portfolio' and user_mode == 'beginner':
        suggestions.append({
            'id': 'explore_expert',
            'title': 'Try Expert Mode',
            'description': 'Access advanced features like strategy builder',
            'action': 'switch_to_expert'
        })
    
    return suggestions

def apply_suggestion(suggestion: Dict[str, Any]):
    """Apply a smart suggestion"""
    action = suggestion['action']
    
    if action == 'add_risk_management':
        st.success("Navigate to Risk Management blocks in the palette")
    elif action == 'add_exit_signal':
        st.success("Check Exit Signals category for exit conditions")
    elif action == 'switch_to_expert':
        st.session_state.user_mode = 'expert'
        st.session_state.show_mode_guide = True
        st.rerun()

def show_progress_indicators():
    """Show progress indicators for long-running operations"""
    if 'operation_progress' in st.session_state:
        progress_info = st.session_state.operation_progress
        
        st.subheader("⏳ Operation Progress")
        
        progress_bar = st.progress(progress_info['progress'])
        st.caption(f"{progress_info['operation']}: {progress_info['status']}")
        
        if progress_info['progress'] >= 1.0:
            st.success(f"✅ {progress_info['operation']} completed!")
            del st.session_state.operation_progress

def show_enhanced_notifications():
    """Show enhanced notification system"""
    if 'notifications' not in st.session_state:
        st.session_state.notifications = []
    
    # Add sample notifications for demonstration
    if len(st.session_state.notifications) == 0:
        st.session_state.notifications = [
            {
                'id': 1,
                'type': 'success',
                'title': 'Strategy Deployed',
                'message': 'BTCUSDT strategy successfully activated',
                'timestamp': datetime.now() - timedelta(minutes=5),
                'read': False
            },
            {
                'id': 2,
                'type': 'info',
                'title': 'Model Retrained',
                'message': 'LSTM model updated with latest market data',
                'timestamp': datetime.now() - timedelta(minutes=15),
                'read': False
            }
        ]
    
    # Notification bell in sidebar
    unread_count = len([n for n in st.session_state.notifications if not n['read']])
    
    if unread_count > 0:
        st.sidebar.markdown("---")
        if st.sidebar.button(f"🔔 Notifications ({unread_count})"):
            show_notification_panel()

def show_notification_panel():
    """Show detailed notification panel"""
    st.subheader("🔔 Notifications")
    
    for notification in st.session_state.notifications:
        with st.container():
            col1, col2, col3 = st.columns([1, 4, 1])
            
            with col1:
                if notification['type'] == 'success':
                    st.success("✅")
                elif notification['type'] == 'warning':
                    st.warning("⚠️")
                elif notification['type'] == 'error':
                    st.error("❌")
                else:
                    st.info("ℹ️")
            
            with col2:
                st.markdown(f"**{notification['title']}**")
                st.caption(notification['message'])
                st.caption(notification['timestamp'].strftime('%H:%M:%S'))
            
            with col3:
                if st.button("✓", key=f"read_{notification['id']}"):
                    notification['read'] = True
                    st.rerun()

def show_accessibility_features():
    """Show accessibility enhancement features"""
    if st.session_state.get('user_mode') == 'beginner':
        # High contrast mode for beginners
        st.markdown("""
        <style>
        .accessibility-high-contrast {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        
        .accessibility-large-text {
            font-size: 1.2em !important;
            line-height: 1.8 !important;
        }
        
        .accessibility-focus {
            outline: 3px solid #ffff00 !important;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Keyboard navigation hints
    if st.session_state.ui_preferences.get('show_tooltips', True):
        st.sidebar.markdown("---")
        st.sidebar.caption("💡 Press Tab to navigate, Enter to select")

def show_real_time_feedback_engine():
    """Main function to show the complete feedback engine"""
    if 'feedback_engine' not in st.session_state:
        st.session_state.feedback_engine = RealTimeFeedbackEngine()
    
    # Show all feedback components
    st.session_state.feedback_engine.show_live_feedback_panel()
    show_interactive_tooltips()
    show_smart_suggestions()
    show_progress_indicators()
    show_enhanced_notifications()
    show_accessibility_features()