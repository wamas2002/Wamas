# Local Deployment Guide - AI Trading System

## System Requirements

### 1. Python 3.11+
```bash
python --version
```

### 2. Required Libraries
```bash
pip install ccxt pandas numpy scikit-learn lightgbm xgboost flask flask-cors flask-socketio requests psutil schedule streamlit plotly
```

### 3. Environment Variables
Create `.env` file in root directory:
```env
OKX_API_KEY=your_okx_api_key_here
OKX_SECRET_KEY=your_okx_secret_key_here
OKX_PASSPHRASE=your_okx_passphrase_here
```

## Core System Files

### 1. Main Trading Dashboard
```bash
python elite_dashboard_fixed.py
```
- Access: http://localhost:3005
- Professional dashboard with real OKX data

### 2. Live Position Monitor
```bash
python live_position_monitor.py
```
- Real-time position monitoring
- P&L tracking

### 3. Advanced Signal Executor
```bash
python advanced_signal_executor.py
```
- Automatic trade execution based on high-quality signals

### 4. Advanced Position Manager
```bash
python advanced_position_manager.py
```
- Intelligent position management with exit strategies

### 5. Intelligent Profit Optimizer
```bash
python intelligent_profit_optimizer.py
```
- Smart profit-taking with optimal timing

## System Startup

### Method 1: Individual Components
```bash
# Terminal 1 - Main Dashboard
python elite_dashboard_fixed.py

# Terminal 2 - Position Monitor
python live_position_monitor.py

# Terminal 3 - Signal Executor
python advanced_signal_executor.py

# Terminal 4 - Position Manager
python advanced_position_manager.py

# Terminal 5 - Profit Optimizer
python intelligent_profit_optimizer.py
```

### Method 2: Automated Launcher
```bash
python local_system_launcher.py
```

## Risk Management

### 1. Risk Controls
- Maximum risk: 2% of portfolio per trade
- Automatic stop loss: enabled
- Automatic take profit: enabled

### 2. Trading Limits
- Minimum balance: $50 USDT
- Maximum open positions: 3 positions
- Maximum leverage: 10x

## Performance Monitoring

### Access Points:
1. **Main Dashboard**: http://localhost:3005
2. **Portfolio Analytics**: http://localhost:5000
3. **System Monitor**: console logs

### Key Metrics:
- Total portfolio balance
- Number of open positions
- Unrealized P&L
- Signal success rate
- System efficiency

## Troubleshooting

### 1. OKX Connection Issues
```bash
# Test API keys
python test_okx_connection.py
```

### 2. Database Issues
```bash
# Reset databases
python reset_databases.py
```

### 3. Memory Issues
```bash
# Monitor memory usage
python system_monitor.py
```

## Security and Backup

### 1. Database Backup
```bash
python backup_system.py
```

### 2. Key Encryption
- Use environment variables only
- Never store keys directly in code

### 3. Security Monitoring
- Regular connection checks
- Complete operation logging
- Automatic security alerts

## Updates and Maintenance

### System Updates:
```bash
git pull origin main
pip install -r requirements.txt
python update_system.py
```

### Regular Maintenance:
- Clean databases weekly
- Performance checks monthly
- Model updates quarterly

## Technical Support

### Log Files:
- `logs/system.log` - General system log
- `logs/trading.log` - Trading operations
- `logs/errors.log` - Error tracking

### Documentation:
- README.md - General documentation
- examples/ - Usage examples
- tests/ - System tests

---

**Warning**: This is a live trading system using real money. Understand risks before operation.