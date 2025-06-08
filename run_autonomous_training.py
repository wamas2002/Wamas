"""
Autonomous Training System Runner
Orchestrates data collection, feature engineering, and model training
Runs continuously with 24-hour retraining cycles
"""

import time
import logging
import schedule
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append('.')

from data.binance_data_collector import BinanceDataCollector
from data.news_sentiment_collector import NewsSentimentCollector
from data.feature_engineer import AdvancedFeatureEngineer
from data.dataset_manager import DatasetManager
from ai.autonomous_training_pipeline import AutonomousTrainingPipeline

class AutonomousTrainingOrchestrator:
    """Orchestrates the complete autonomous training workflow"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Major trading pairs to train on
        self.symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT',
            'SOL/USDT', 'XRP/USDT', 'DOT/USDT', 'AVAX/USDT'
        ]
        
        # Initialize components - using OKX instead of Binance due to geographic restrictions
        from trading.okx_data_service import OKXDataService
        self.okx_data_service = OKXDataService()
        self.news_collector = NewsSentimentCollector()
        self.feature_engineer = AdvancedFeatureEngineer()
        self.dataset_manager = DatasetManager()
        self.training_pipeline = AutonomousTrainingPipeline()
        
        self.logger.info("Autonomous Training Orchestrator initialized")
    
    def collect_historical_data(self, days_back: int = 30):
        """Collect comprehensive historical data"""
        
        self.logger.info("Starting historical data collection...")
        
        # Collect OHLCV data from OKX
        self.logger.info("Collecting OHLCV data from OKX...")
        for symbol in self.symbols:
            try:
                # Convert symbol format for OKX (BTC/USDT -> BTCUSDT)
                okx_symbol = symbol.replace('/', '')
                data = self.okx_data_service.get_historical_data(okx_symbol, '1m', limit=days_back * 1440)
                if not data.empty:
                    self.logger.info(f"Successfully collected {len(data)} records for {symbol}")
                else:
                    self.logger.warning(f"No data received for {symbol}")
            except Exception as e:
                self.logger.error(f"Error collecting data for {symbol}: {e}")
        
        # Collect news sentiment data
        self.logger.info("Collecting news sentiment data...")
        news_items = self.news_collector.collect_all_news()
        if news_items:
            self.news_collector.save_news_to_database(news_items)
            
            # Aggregate sentiment for each symbol
            for symbol in self.symbols:
                symbol_clean = symbol.replace('/', '').replace('USDT', '')
                self.news_collector.aggregate_sentiment_by_minute(symbol_clean)
        
        self.logger.info("Historical data collection completed")
    
    def create_training_datasets(self):
        """Create comprehensive training datasets with all features"""
        
        self.logger.info("Creating training datasets...")
        
        try:
            # Create datasets for all symbols
            dataset_results = self.dataset_manager.create_datasets_for_multiple_symbols(
                self.symbols, classification_type='binary'
            )
            
            self.logger.info(f"Created datasets for {len(dataset_results)} symbols")
            
            for symbol, (train_path, val_path) in dataset_results.items():
                self.logger.info(f"{symbol}: Training={train_path}, Validation={val_path}")
            
            return dataset_results
            
        except Exception as e:
            self.logger.error(f"Error creating datasets: {e}")
            return {}
    
    def train_all_models(self):
        """Train all ML models for each symbol"""
        
        self.logger.info("Starting comprehensive model training...")
        
        training_results = {}
        
        for symbol in self.symbols:
            try:
                self.logger.info(f"Training models for {symbol}")
                
                # Train all models for this symbol
                results = self.training_pipeline.train_all_models(symbol)
                
                if results:
                    # Save models to disk
                    self.training_pipeline.save_models(symbol, results)
                    
                    # Store performance metrics in database
                    self.training_pipeline.store_performance_metrics(symbol, results)
                    
                    training_results[symbol] = results
                    
                    # Get best model info
                    best_model_name, best_model = self.training_pipeline.select_best_model(results)
                    if best_model:
                        f1_score = best_model.get('f1_score', 0)
                        self.logger.info(f"{symbol} - Best model: {best_model_name} (F1: {f1_score:.4f})")
                
            except Exception as e:
                self.logger.error(f"Failed to train models for {symbol}: {e}")
                continue
        
        self.logger.info(f"Model training completed for {len(training_results)} symbols")
        return training_results
    
    def run_full_training_cycle(self):
        """Execute complete training cycle"""
        
        start_time = datetime.now()
        self.logger.info("=" * 50)
        self.logger.info("STARTING FULL TRAINING CYCLE")
        self.logger.info("=" * 50)
        
        try:
            # Step 1: Collect fresh data
            self.collect_historical_data(days_back=30)
            
            # Step 2: Create training datasets
            dataset_results = self.create_training_datasets()
            
            if not dataset_results:
                self.logger.error("No datasets created, skipping model training")
                return
            
            # Step 3: Train all models
            training_results = self.train_all_models()
            
            # Step 4: Generate summary report
            self.generate_training_report(training_results)
            
            duration = datetime.now() - start_time
            self.logger.info(f"Full training cycle completed in {duration}")
            
        except Exception as e:
            self.logger.error(f"Error in training cycle: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_training_report(self, training_results: dict):
        """Generate and save training performance report"""
        
        report_lines = [
            "=" * 60,
            "AUTONOMOUS TRADING MODEL TRAINING REPORT",
            "=" * 60,
            f"Training Date: {datetime.now().isoformat()}",
            f"Symbols Trained: {len(training_results)}",
            "",
            "MODEL PERFORMANCE SUMMARY:",
            "-" * 40
        ]
        
        for symbol, results in training_results.items():
            report_lines.append(f"\n{symbol}:")
            
            # Get model performances
            model_performances = []
            for model_name, model_results in results.items():
                if isinstance(model_results, dict) and 'f1_score' in model_results:
                    f1 = model_results['f1_score']
                    precision = model_results['precision']
                    recall = model_results['recall']
                    model_performances.append((model_name, f1, precision, recall))
            
            # Sort by F1 score
            model_performances.sort(key=lambda x: x[1], reverse=True)
            
            for model_name, f1, precision, recall in model_performances:
                report_lines.append(f"  {model_name:15} - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            
            if model_performances:
                best_model = model_performances[0]
                report_lines.append(f"  → BEST: {best_model[0]} (F1: {best_model[1]:.4f})")
        
        report_lines.extend([
            "",
            "=" * 60,
            "Training cycle completed successfully",
            "Models saved to: ./models/",
            "Next training: 24 hours from now",
            "=" * 60
        ])
        
        # Print report
        report_text = "\n".join(report_lines)
        self.logger.info(f"\n{report_text}")
        
        # Save report to file
        report_filename = f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(f"models/{report_filename}", 'w') as f:
            f.write(report_text)
        
        self.logger.info(f"Training report saved to models/{report_filename}")
    
    def run_continuous_news_collection(self):
        """Run continuous news collection every 10 minutes"""
        
        self.logger.info("Starting continuous news collection...")
        
        while True:
            try:
                # Collect latest news
                news_items = self.news_collector.collect_all_news()
                
                if news_items:
                    self.news_collector.save_news_to_database(news_items)
                    
                    # Aggregate sentiment for major symbols
                    for symbol in ['BTC', 'ETH', 'BNB', 'ADA', 'SOL']:
                        self.news_collector.aggregate_sentiment_by_minute(symbol)
                    
                    self.logger.info(f"Collected and processed {len(news_items)} news items")
                
                # Wait 10 minutes
                time.sleep(600)
                
            except KeyboardInterrupt:
                self.logger.info("News collection stopped")
                break
            except Exception as e:
                self.logger.error(f"Error in news collection: {e}")
                time.sleep(60)  # Wait 1 minute before retry
    
    def schedule_training_cycles(self):
        """Schedule automatic training cycles every 24 hours"""
        
        self.logger.info("Scheduling training cycles...")
        
        # Schedule daily training at 2 AM
        schedule.every().day.at("02:00").do(self.run_full_training_cycle)
        
        # Run initial training cycle
        self.logger.info("Running initial training cycle...")
        self.run_full_training_cycle()
        
        self.logger.info("Training scheduled for daily execution at 2:00 AM")
        
        # Keep scheduler running
        while True:
            try:
                schedule.run_pending()
                time.sleep(3600)  # Check every hour
                
            except KeyboardInterrupt:
                self.logger.info("Scheduler stopped")
                break
            except Exception as e:
                self.logger.error(f"Error in scheduler: {e}")
                time.sleep(300)  # Wait 5 minutes before retry
    
    def run_autonomous_system(self):
        """Run complete autonomous training system"""
        
        self.logger.info("Starting Autonomous Trading Model Training System")
        self.logger.info("Features:")
        self.logger.info("- CCXT Binance data collection (1 year history)")
        self.logger.info("- Multi-source news sentiment analysis")
        self.logger.info("- 100+ technical indicators")
        self.logger.info("- 7 ML models: XGBoost, LightGBM, CatBoost, RandomForest, GradientBoosting, LSTM+Attention, Prophet")
        self.logger.info("- Automatic 24-hour retraining cycles")
        self.logger.info("- CSV dataset export")
        self.logger.info("- Performance tracking and model selection")
        
        try:
            # Start with immediate training cycle
            self.schedule_training_cycles()
            
        except KeyboardInterrupt:
            self.logger.info("System stopped by user")
        except Exception as e:
            self.logger.error(f"System error: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main entry point"""
    
    print("🚀 Autonomous AI Trading Model Training System")
    print("=" * 50)
    print("Initializing comprehensive ML pipeline...")
    
    orchestrator = AutonomousTrainingOrchestrator()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'collect':
            print("Running data collection only...")
            orchestrator.collect_historical_data(days_back=365)  # 1 year
            
        elif command == 'train':
            print("Running model training only...")
            orchestrator.run_full_training_cycle()
            
        elif command == 'news':
            print("Running continuous news collection...")
            orchestrator.run_continuous_news_collection()
            
        elif command == 'datasets':
            print("Creating training datasets...")
            orchestrator.create_training_datasets()
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: collect, train, news, datasets")
    else:
        # Run full autonomous system
        orchestrator.run_autonomous_system()

if __name__ == "__main__":
    main()