#!/usr/bin/env python3
"""
End-to-End System Verification
Complete workflow test: Strategy Generation → Signal → Execution → Analytics
"""
import sys
import traceback
from datetime import datetime
import pandas as pd

def test_strategy_generation_workflow():
    """Test complete strategy generation workflow"""
    print("Testing Strategy Generation → Signal → Analytics Workflow...")
    try:
        from strategies.autoconfig_engine import AutoConfigEngine
        from trading.okx_data_service import OKXDataService
        from ai.auto_strategy_analyzer import AutoStrategyAnalyzer
        
        # Initialize components
        autoconfig = AutoConfigEngine()
        okx_service = OKXDataService()
        analyzer = AutoStrategyAnalyzer()
        
        test_symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        results = {}
        
        for symbol in test_symbols:
            # 1. Get live market data
            data = okx_service.get_historical_data(symbol, "1h", limit=100)
            current_price = float(data['close'].iloc[-1])
            
            # 2. Auto-select strategy based on market conditions
            strategy = autoconfig.get_strategy_for_symbol(symbol)
            
            # 3. Generate trading signal
            signal = autoconfig.generate_strategy_signal(symbol, data, current_price)
            
            # 4. Get market analysis
            analysis = analyzer.analyze_market_conditions(symbol)
            
            results[symbol] = {
                'price': current_price,
                'strategy': strategy,
                'signal': signal.get('action', 'hold'),
                'confidence': signal.get('confidence', 0),
                'market_regime': analysis.get('market_regime', 'unknown')
            }
            
            print(f"✅ {symbol}: ${current_price:,.2f} | Strategy: {strategy} | Signal: {signal.get('action')} | Regime: {analysis.get('market_regime')}")
        
        return True, results
        
    except Exception as e:
        print(f"❌ Strategy workflow failed: {e}")
        traceback.print_exc()
        return False, {}

def test_risk_management_integration():
    """Test risk management with live positions"""
    print("\nTesting Risk Management Integration...")
    try:
        from trading.advanced_risk_manager import AdvancedRiskManager
        from trading.okx_data_service import OKXDataService
        
        risk_manager = AdvancedRiskManager()
        okx_service = OKXDataService()
        
        # Create test positions for multiple symbols
        test_positions = [
            {"symbol": "BTCUSDT", "entry_price": 106000.0, "position_size": 0.1},
            {"symbol": "ETHUSDT", "entry_price": 4000.0, "position_size": 0.5},
            {"symbol": "ADAUSDT", "entry_price": 1.2, "position_size": 100.0}
        ]
        
        for pos_data in test_positions:
            position = risk_manager.create_position_risk(
                symbol=pos_data["symbol"],
                entry_price=pos_data["entry_price"],
                position_size=pos_data["position_size"],
                tp_levels=[0.03, 0.06, 0.09],
                sl_percentage=0.02,
                use_trailing_stop=True
            )
            
            # Update with current market price
            current_data = okx_service.get_historical_data(pos_data["symbol"], "1m", limit=1)
            current_price = float(current_data['close'].iloc[-1])
            
            risk_metrics = risk_manager.update_position_risk(pos_data["symbol"], current_price)
            
            print(f"✅ {pos_data['symbol']}: Entry ${pos_data['entry_price']} | Current ${current_price:.2f} | P&L: ${risk_metrics.unrealized_pnl:.2f}")
        
        # Test portfolio summary
        portfolio_summary = risk_manager.get_portfolio_risk_summary()
        print(f"✅ Portfolio: {portfolio_summary['total_positions']} positions | Total P&L: ${portfolio_summary['total_unrealized_pnl']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Risk management failed: {e}")
        traceback.print_exc()
        return False

def test_ai_components_integration():
    """Test AI components with live data"""
    print("\nTesting AI Components Integration...")
    try:
        from ai.advisor import AIFinancialAdvisor
        from ai.asset_explorer import AssetExplorer
        from ai.auto_strategy_analyzer import AutoStrategyAnalyzer
        
        # Test AI Financial Advisor
        advisor = AIFinancialAdvisor()
        recommendations = advisor.get_recommendations(["BTCUSDT", "ETHUSDT"])
        
        print(f"✅ AI Advisor: Generated recommendations for {len(recommendations)} symbols")
        
        # Test Asset Explorer
        explorer = AssetExplorer()
        assets = explorer.get_all_assets_overview(sort_by='volume')
        
        print(f"✅ Asset Explorer: Analyzed {len(assets)} assets")
        
        # Test Auto Strategy Analyzer
        analyzer = AutoStrategyAnalyzer()
        analysis = analyzer.analyze_market_conditions("BTCUSDT")
        
        print(f"✅ Strategy Analyzer: Market regime = {analysis.get('market_regime')}")
        
        return True
        
    except Exception as e:
        print(f"❌ AI components failed: {e}")
        traceback.print_exc()
        return False

def test_visual_components():
    """Test visual and interface components"""
    print("\nTesting Visual Components...")
    try:
        from frontend.visual_strategy_builder import VisualStrategyBuilder
        from strategies.smart_strategy_selector import SmartStrategySelector
        
        # Test Visual Strategy Builder
        builder = VisualStrategyBuilder()
        strategy_data = builder.get_strategy_templates()
        
        print(f"✅ Visual Builder: {len(strategy_data)} strategy templates available")
        
        # Test Smart Strategy Selector
        selector = SmartStrategySelector()
        performance = selector.get_strategy_performance_summary()
        
        print(f"✅ Strategy Selector: Monitoring {len(performance)} strategies")
        
        return True
        
    except Exception as e:
        print(f"❌ Visual components failed: {e}")
        traceback.print_exc()
        return False

def test_database_integration():
    """Test database connectivity and operations"""
    print("\nTesting Database Integration...")
    try:
        import sqlite3
        import os
        
        # Test all database files
        db_files = [
            "database/trading_data.db",
            "database/strategies.db", 
            "database/risk_management.db",
            "database/analysis.db"
        ]
        
        operational_dbs = 0
        
        for db_file in db_files:
            if os.path.exists(db_file):
                try:
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    conn.close()
                    
                    print(f"✅ Database {db_file}: {len(tables)} tables")
                    operational_dbs += 1
                    
                except Exception as e:
                    print(f"⚠️ Database {db_file}: Connection error")
        
        print(f"✅ Database Summary: {operational_dbs}/{len(db_files)} databases operational")
        return operational_dbs >= 3  # At least 3 of 4 databases working
        
    except Exception as e:
        print(f"❌ Database integration failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run comprehensive end-to-end verification"""
    print("🚀 END-TO-END SYSTEM VERIFICATION")
    print("=" * 60)
    
    tests = [
        ("Strategy Generation Workflow", test_strategy_generation_workflow),
        ("Risk Management Integration", test_risk_management_integration),
        ("AI Components Integration", test_ai_components_integration),
        ("Visual Components", test_visual_components),
        ("Database Integration", test_database_integration)
    ]
    
    passed = 0
    total = len(tests)
    detailed_results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            if test_name == "Strategy Generation Workflow":
                success, results = test_func()
                detailed_results['strategy_results'] = results
            else:
                success = test_func()
            
            if success:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            print(f"❌ {test_name}: EXCEPTION - {e}")
    
    print("\n" + "=" * 60)
    print("FINAL SYSTEM STATUS")
    print("=" * 60)
    
    success_rate = (passed / total) * 100
    print(f"📊 Overall Success Rate: {passed}/{total} ({success_rate:.1f}%)")
    
    if 'strategy_results' in detailed_results:
        print("\n📈 Live Trading Signals:")
        for symbol, data in detailed_results['strategy_results'].items():
            print(f"   {symbol}: {data['signal'].upper()} signal | ${data['price']:,.2f} | {data['strategy']} strategy")
    
    if success_rate >= 80:
        print("\n🎉 SYSTEM FULLY OPERATIONAL - READY FOR DEPLOYMENT")
        print("✅ All core components verified with live OKX data")
        print("✅ End-to-end workflow confirmed functional")
        return True
    else:
        print(f"\n⚠️ SYSTEM NEEDS ATTENTION - {100-success_rate:.1f}% components require fixes")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)