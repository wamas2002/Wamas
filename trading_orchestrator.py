#!/usr/bin/env python3
"""
Advanced Trading System Orchestrator
Coordinates all trading components for optimal autonomous performance
"""

import os
import time
import threading
from datetime import datetime
from typing import Dict, List
from advanced_trading_optimizer import AdvancedTradingOptimizer
from risk_management_engine import RiskManagementEngine
from automated_profit_taking import AutomatedProfitTaking
from ml_signal_predictor import MLSignalPredictor

class TradingOrchestrator:
    def __init__(self):
        self.active = True
        self.components = {}
        self.performance_metrics = {
            'trades_executed': 0,
            'profit_taken': 0.0,
            'optimization_cycles': 0,
            'risk_alerts': 0
        }
        
        # Initialize all components
        self.initialize_components()
        
    def initialize_components(self):
        """Initialize all trading system components"""
        print("Initializing Advanced Trading System Components...")
        
        try:
            self.components['optimizer'] = AdvancedTradingOptimizer()
            print("✅ Trading Optimizer initialized")
        except Exception as e:
            print(f"⚠️ Optimizer initialization failed: {e}")
        
        try:
            self.components['risk_manager'] = RiskManagementEngine()
            print("✅ Risk Management Engine initialized")
        except Exception as e:
            print(f"⚠️ Risk Manager initialization failed: {e}")
        
        try:
            self.components['profit_taker'] = AutomatedProfitTaking()
            print("✅ Automated Profit Taking initialized")
        except Exception as e:
            print(f"⚠️ Profit Taker initialization failed: {e}")
        
        try:
            self.components['ml_predictor'] = MLSignalPredictor()
            print("✅ ML Signal Predictor initialized")
        except Exception as e:
            print(f"⚠️ ML Predictor initialization failed: {e}")
    
    def run_optimization_cycle(self):
        """Execute comprehensive optimization cycle"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running optimization cycle...")
        
        # 1. Risk Assessment
        if 'risk_manager' in self.components:
            try:
                emergency_status = self.components['risk_manager'].emergency_stop_conditions()
                if emergency_status['emergency_stop']:
                    print(f"🚨 EMERGENCY STOP: {emergency_status['triggers']}")
                    self.performance_metrics['risk_alerts'] += 1
                    return False
            except Exception as e:
                print(f"Risk assessment error: {e}")
        
        # 2. ML Signal Generation
        if 'ml_predictor' in self.components:
            try:
                symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
                ml_signals = self.components['ml_predictor'].batch_generate_signals(symbols)
                print(f"📊 Generated ML signals for {len(ml_signals)} symbols")
            except Exception as e:
                print(f"ML signal generation error: {e}")
        
        # 3. Parameter Optimization
        if 'optimizer' in self.components:
            try:
                self.components['optimizer'].run_optimization_cycle()
                self.performance_metrics['optimization_cycles'] += 1
            except Exception as e:
                print(f"Optimization error: {e}")
        
        # 4. Profit Taking
        if 'profit_taker' in self.components:
            try:
                self.components['profit_taker'].run_profit_monitoring_cycle()
            except Exception as e:
                print(f"Profit taking error: {e}")
        
        return True
    
    def start_autonomous_trading(self):
        """Start fully autonomous trading with all optimizations"""
        def orchestrator_loop():
            print("🚀 AUTONOMOUS TRADING ORCHESTRATOR STARTED")
            print("=" * 60)
            
            while self.active:
                try:
                    # Run comprehensive optimization every 15 minutes
                    success = self.run_optimization_cycle()
                    
                    if not success:
                        print("⚠️ Optimization cycle failed - pausing for 5 minutes")
                        time.sleep(300)
                        continue
                    
                    # Display performance summary
                    self.display_performance_summary()
                    
                    # Wait 15 minutes before next cycle
                    time.sleep(900)
                    
                except KeyboardInterrupt:
                    print("\n🛑 Orchestrator stopped by user")
                    break
                except Exception as e:
                    print(f"Orchestrator error: {e}")
                    time.sleep(60)
        
        orchestrator_thread = threading.Thread(target=orchestrator_loop, daemon=True)
        orchestrator_thread.start()
        return orchestrator_thread
    
    def start_background_monitoring(self):
        """Start background monitoring threads"""
        threads = []
        
        # Risk monitoring every 60 seconds
        if 'risk_manager' in self.components:
            risk_thread = self.components['risk_manager'].start_risk_monitoring(60)
            threads.append(risk_thread)
            print("📊 Risk monitoring started")
        
        # Profit monitoring every 5 minutes
        if 'profit_taker' in self.components:
            profit_thread = self.components['profit_taker'].start_profit_monitoring(5)
            threads.append(profit_thread)
            print("💰 Profit monitoring started")
        
        # Parameter optimization every 10 minutes
        if 'optimizer' in self.components:
            opt_thread = self.components['optimizer'].start_optimization_loop(10)
            threads.append(opt_thread)
            print("⚙️ Parameter optimization started")
        
        return threads
    
    def display_performance_summary(self):
        """Display current system performance"""
        print(f"\n📈 SYSTEM PERFORMANCE SUMMARY")
        print(f"   Optimization Cycles: {self.performance_metrics['optimization_cycles']}")
        print(f"   Risk Alerts: {self.performance_metrics['risk_alerts']}")
        print(f"   Status: {'🟢 ACTIVE' if self.active else '🔴 STOPPED'}")
        
        # Get current portfolio status if available
        if 'profit_taker' in self.components:
            try:
                positions = self.components['profit_taker'].get_current_positions()
                if positions:
                    total_profit = sum(pos['profit_usdt'] for pos in positions.values())
                    print(f"   Current Positions: {len(positions)}")
                    print(f"   Unrealized P&L: ${total_profit:+.2f}")
            except:
                pass
    
    def emergency_shutdown(self):
        """Emergency shutdown of all trading activities"""
        print("🚨 EMERGENCY SHUTDOWN INITIATED")
        self.active = False
        
        # Stop all monitoring
        for component_name, component in self.components.items():
            if hasattr(component, 'monitoring_active'):
                component.monitoring_active = False
            if hasattr(component, 'optimization_active'):
                component.optimization_active = False
        
        print("🛑 All trading activities stopped")
    
    def get_system_health(self) -> Dict:
        """Get comprehensive system health status"""
        health = {
            'overall_status': 'healthy',
            'components': {},
            'performance': self.performance_metrics
        }
        
        # Check each component
        for name, component in self.components.items():
            try:
                if hasattr(component, 'exchange') and component.exchange:
                    health['components'][name] = 'operational'
                else:
                    health['components'][name] = 'connection_error'
            except:
                health['components'][name] = 'error'
        
        # Determine overall status
        if any(status == 'error' for status in health['components'].values()):
            health['overall_status'] = 'degraded'
        elif any(status == 'connection_error' for status in health['components'].values()):
            health['overall_status'] = 'limited'
        
        return health

def run_full_autonomous_system():
    """Launch the complete autonomous trading system"""
    orchestrator = TradingOrchestrator()
    
    print("ADVANCED AUTONOMOUS TRADING SYSTEM")
    print("=" * 50)
    
    # Display system health
    health = orchestrator.get_system_health()
    print(f"System Health: {health['overall_status'].upper()}")
    
    for component, status in health['components'].items():
        status_icon = "✅" if status == 'operational' else "⚠️"
        print(f"  {status_icon} {component}: {status}")
    
    print("\n" + "=" * 50)
    
    # Start background monitoring
    monitor_threads = orchestrator.start_background_monitoring()
    
    # Start main orchestrator
    main_thread = orchestrator.start_autonomous_trading()
    
    try:
        # Keep the system running
        while orchestrator.active:
            time.sleep(10)
            
            # Check if main thread is still alive
            if not main_thread.is_alive():
                print("⚠️ Main orchestrator thread stopped - restarting...")
                main_thread = orchestrator.start_autonomous_trading()
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down autonomous trading system...")
        orchestrator.emergency_shutdown()

def main():
    """Main function to start the orchestrated trading system"""
    try:
        run_full_autonomous_system()
    except Exception as e:
        print(f"Critical system error: {e}")
        print("Please check configuration and restart")

if __name__ == '__main__':
    main()