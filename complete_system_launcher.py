#!/usr/bin/env python3
"""
Complete Advanced Trading System Launcher
Launches all system components with full optimization and monitoring
"""

import os
import time
import threading
import subprocess
from datetime import datetime
from typing import Dict, List

class CompleteSystemLauncher:
    def __init__(self):
        self.active_processes = {}
        self.system_status = {
            'main_platform': False,
            'trading_bridge': False,
            'monitoring_dashboard': False,
            'advanced_monitor': False,
            'optimization_active': False
        }
        
    def launch_main_platform(self):
        """Launch the main trading platform"""
        print("Starting main trading platform...")
        try:
            process = subprocess.Popen(['python', 'complete_trading_platform.py'])
            self.active_processes['main_platform'] = process
            self.system_status['main_platform'] = True
            print("✅ Main trading platform started")
            return True
        except Exception as e:
            print(f"❌ Failed to start main platform: {e}")
            return False
    
    def launch_trading_bridge(self):
        """Launch the signal execution bridge"""
        print("Starting signal execution bridge...")
        try:
            process = subprocess.Popen(['python', 'signal_execution_bridge.py'])
            self.active_processes['trading_bridge'] = process
            self.system_status['trading_bridge'] = True
            print("✅ Signal execution bridge started")
            return True
        except Exception as e:
            print(f"❌ Failed to start trading bridge: {e}")
            return False
    
    def launch_monitoring_dashboards(self):
        """Launch both monitoring dashboards"""
        print("Starting monitoring dashboards...")
        
        # Original monitoring dashboard
        try:
            process1 = subprocess.Popen(['streamlit', 'run', 'live_trading_monitor_dashboard.py', '--server.port', '5001'])
            self.active_processes['monitor_dashboard'] = process1
            self.system_status['monitoring_dashboard'] = True
            print("✅ Primary monitoring dashboard started on port 5001")
        except Exception as e:
            print(f"❌ Failed to start primary monitor: {e}")
        
        # Advanced monitoring dashboard
        try:
            process2 = subprocess.Popen(['streamlit', 'run', 'advanced_monitoring_dashboard.py', '--server.port', '5002'])
            self.active_processes['advanced_monitor'] = process2
            self.system_status['advanced_monitor'] = True
            print("✅ Advanced monitoring dashboard started on port 5002")
        except Exception as e:
            print(f"❌ Failed to start advanced monitor: {e}")
    
    def start_optimization_services(self):
        """Start background optimization services"""
        print("Starting optimization services...")
        
        def optimization_worker():
            try:
                from advanced_trading_optimizer import AdvancedTradingOptimizer
                from risk_management_engine import RiskManagementEngine
                from automated_profit_taking import AutomatedProfitTaking
                
                # Initialize optimizers
                optimizer = AdvancedTradingOptimizer()
                risk_manager = RiskManagementEngine()
                profit_taker = AutomatedProfitTaking()
                
                # Start background monitoring
                optimizer.start_optimization_loop(10)  # Every 10 minutes
                risk_manager.start_risk_monitoring(60)  # Every minute
                profit_taker.start_profit_monitoring(5)  # Every 5 minutes
                
                self.system_status['optimization_active'] = True
                print("✅ Background optimization services started")
                
                # Keep services running
                while True:
                    time.sleep(300)  # Check every 5 minutes
                    
            except Exception as e:
                print(f"❌ Optimization services error: {e}")
        
        optimization_thread = threading.Thread(target=optimization_worker, daemon=True)
        optimization_thread.start()
    
    def display_system_status(self):
        """Display current system status"""
        print("\n" + "=" * 60)
        print("ADVANCED TRADING SYSTEM STATUS")
        print("=" * 60)
        
        status_indicators = {
            'main_platform': "Main Trading Platform (Port 5000)",
            'trading_bridge': "Signal Execution Bridge",
            'monitoring_dashboard': "Primary Monitor (Port 5001)",
            'advanced_monitor': "Advanced Monitor (Port 5002)",
            'optimization_active': "Background Optimization"
        }
        
        for component, description in status_indicators.items():
            status = "🟢 ACTIVE" if self.system_status[component] else "🔴 INACTIVE"
            print(f"  {description}: {status}")
        
        print("\n📊 Access Points:")
        if self.system_status['main_platform']:
            print("  • Main Platform: http://localhost:5000")
        if self.system_status['monitoring_dashboard']:
            print("  • Primary Monitor: http://localhost:5001")
        if self.system_status['advanced_monitor']:
            print("  • Advanced Monitor: http://localhost:5002")
        
        print("\n⚙️ Active Features:")
        print("  • Live OKX trading with real market data")
        print("  • AI signal generation and execution")
        print("  • Dynamic parameter optimization")
        print("  • Automated risk management")
        print("  • Smart profit taking")
        print("  • Machine learning predictions")
        print("  • Real-time portfolio monitoring")
        
        print("=" * 60)
    
    def monitor_system_health(self):
        """Monitor system health and restart if needed"""
        def health_monitor():
            while True:
                try:
                    # Check if processes are still running
                    for name, process in self.active_processes.items():
                        if process.poll() is not None:
                            print(f"⚠️ {name} process stopped - attempting restart...")
                            self.system_status[name] = False
                            
                            # Restart based on component
                            if name == 'main_platform':
                                self.launch_main_platform()
                            elif name == 'trading_bridge':
                                self.launch_trading_bridge()
                    
                    time.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    print(f"Health monitor error: {e}")
                    time.sleep(60)
        
        monitor_thread = threading.Thread(target=health_monitor, daemon=True)
        monitor_thread.start()
        print("✅ System health monitoring started")
    
    def launch_complete_system(self):
        """Launch the complete advanced trading system"""
        print("LAUNCHING COMPLETE ADVANCED TRADING SYSTEM")
        print("=" * 60)
        print(f"Launch Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        # Launch core components in sequence
        self.launch_main_platform()
        time.sleep(3)
        
        self.launch_trading_bridge()
        time.sleep(3)
        
        self.launch_monitoring_dashboards()
        time.sleep(5)
        
        # Start optimization services
        self.start_optimization_services()
        time.sleep(2)
        
        # Start health monitoring
        self.monitor_system_health()
        
        # Display final status
        self.display_system_status()
        
        print("\n🚀 SYSTEM LAUNCH COMPLETE")
        print("The advanced trading system is now fully operational!")
        
        return True
    
    def shutdown_system(self):
        """Gracefully shutdown all system components"""
        print("\n🛑 Shutting down system...")
        
        for name, process in self.active_processes.items():
            try:
                process.terminate()
                print(f"✅ {name} terminated")
            except:
                pass
        
        print("System shutdown complete")

def main():
    """Main launcher function"""
    launcher = CompleteSystemLauncher()
    
    try:
        launcher.launch_complete_system()
        
        # Keep system running
        print("\nPress Ctrl+C to shutdown the system...")
        while True:
            time.sleep(10)
            
    except KeyboardInterrupt:
        launcher.shutdown_system()

if __name__ == '__main__':
    main()