#!/bin/bash
# AI Trading System Startup Script
echo "Starting AI Trading System..."

# Check environment
if [ ! -f ".env" ]; then
    echo "Error: .env file not found. Please create from .env.template"
    exit 1
fi

# Start components
echo "Starting Main Dashboard..."
python elite_dashboard_fixed.py &

echo "Starting Position Monitor..."
python live_position_monitor.py &

echo "Starting Signal Executor..."
python advanced_signal_executor.py &

echo "Starting Position Manager..."
python advanced_position_manager.py &

echo "Starting Profit Optimizer..."
python intelligent_profit_optimizer.py &

echo "✓ All components started successfully"
echo "Access Main Dashboard: http://localhost:3005"
echo "Access Portfolio Analytics: http://localhost:5000"
echo "Press Ctrl+C to stop system"

wait