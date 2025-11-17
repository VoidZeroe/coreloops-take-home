"""
Main pipeline script for ETL, feature engineering, and model training.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.data.aggregator import DailyAggregator
from src.utils.fx import FXConverter
from src.features.engineering import FeatureEngineer
from src.models.trainer import RevenuePredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Run the complete data pipeline and model training."""
    
    logger.info("\n" + "="*80)
    logger.info("STARTING DATA PIPELINE")
    logger.info("\n" + "="*80)
    
    # Create artifacts directory
    artifacts_dir = Path('artifacts')
    artifacts_dir.mkdir(exist_ok=True)
    model_dir = artifacts_dir / 'model'
    model_dir.mkdir(exist_ok=True)
    
    try:
        # ========== Step 1: Data Loading ==========
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA LOADING")
        logger.info("\n" + "="*80)
        
        loader = DataLoader()
        
        # Load FX rates
        fx_rates = loader.load_fx_rates()
        logger.info(f"Loaded FX rates: {len(fx_rates)} records")
        
        # Load transaction files
        
        transactions = loader.load_all_files()
        
        logger.info(f"Loaded transactions: {len(transactions)} records")
        logger.info(f"Date range: {transactions['timestamp'].min()} to {transactions['timestamp'].max()}")
        logger.info(f"Unique customers: {transactions['customer_id'].nunique()}")
        
        # ========== Step 2: Data Cleaning ==========
        logger.info("\n" + "="*80)
        logger.info("STEP 2: DATA CLEANING")
        logger.info("\n" + "="*80)
        
        cleaner = DataCleaner()
        clean_transactions = cleaner.clean(transactions)
        
        # ========== Step 3: FX Conversion ==========
        logger.info("\n" + "="*80)
        logger.info("STEP 3: FX CONVERSION")
        logger.info("\n" + "="*80)
        
        fx_converter = FXConverter(fx_rates)
        clean_transactions = fx_converter.convert_to_gbp(clean_transactions)
        
        # ========== Step 4: Aggregation ==========
        logger.info("\n" + "="*80)
        logger.info("STEP 4: DAILY AGGREGATION")
        logger.info("\n" + "="*80)
        
        aggregator = DailyAggregator()
        daily_metrics = aggregator.aggregate(clean_transactions)
        
        
        # Save aggregated metrics
        metrics_path = artifacts_dir / 'daily_customer_metrics.parquet'
        daily_metrics.to_parquet(metrics_path, index=False)
        logger.info(f"Saved daily metrics to {metrics_path}")
        
        # save as CSV for easy inspection
        daily_metrics.to_csv(artifacts_dir / 'daily_customer_metrics.csv', index=False)
        
        # ========== Step 5: Feature Engineering ==========
        logger.info("\n" + "="*80)
        logger.info("STEP 5: FEATURE ENGINEERING")
        logger.info("\n" + "="*80)
        
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.engineer_features(daily_metrics, target_col='next_day_net_gbp')
        
        # ========== Step 6: Train/Test Split ==========
        logger.info("\n" + "="*80)
        logger.info("STEP 6: TRAIN/TEST SPLIT")
        logger.info("\n" + "="*80)

        X_train, X_test, y_train, y_test = \
            feature_engineer.prepare_train_test_split(
                features_df,
                target_col='next_day_net_gbp',
                test_size=0.2,
                min_train_days=1
            )
        
        # ========== Step 7: Model Training ==========
        logger.info("\n" + "="*80)
        logger.info("STEP 7: MODEL TRAINING")
        logger.info("\n" + "="*80)
        
        predictor = RevenuePredictor()
        
        # Train model
        val_metrics, train_metrics = predictor.train(
                                        X_train, y_train,
                                        X_test, y_test
                                    )
        
        # ========== Step 8: Model Evaluation ==========
        logger.info("\n" + "="*80)
        logger.info("STEP 8: MODEL EVALUATION")
        logger.info("\n" + "="*80)
        
        # Feature importance
        importance_df = predictor.get_feature_importance()
        logger.info("\nTop 10 Most Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.2f}")
        
        # ========== Step 9: Save Model ==========
        logger.info("\n" + "="*80)
        logger.info("STEP 9: SAVING MODEL")
        logger.info("\n" + "="*80)
        
        predictor.save(model_dir)
        
        # ========== Pipeline Summary ==========
        logger.info("\n" + "="*80)
        logger.info("PIPELINE SUMMARY:")

        logger.info(f"✓ Training samples: {len(X_train):,}")
        logger.info(f"✓ Test samples: {len(X_test):,}")

        logger.info(f"✓ Features: {X_train.shape[1]}")
        logger.info(f"✓ Test MAE: £{val_metrics['mae']:.2f}")
        logger.info(f"✓ Test RMSE: £{val_metrics['rmse']:.2f}")
        logger.info(f"✓ Test R²: {val_metrics['r2']:.4f}")

        logger.info(f"✓ Train MAE: £{train_metrics['mae']:.2f}")
        logger.info(f"✓ Train RMSE: £{train_metrics['rmse']:.2f}")
        logger.info(f"✓ Train R²: {train_metrics['r2']:.4f}")

        logger.info(f"\n✓ Model saved to: {model_dir}")
        logger.info(f"✓ Metrics saved to: {metrics_path}")
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("\n" + "="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())