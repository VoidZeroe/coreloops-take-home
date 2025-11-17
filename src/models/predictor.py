"""
Prediction interface for making next-day revenue forecasts.
"""
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import numpy as np

from src.models.trainer import RevenuePredictor
from src.features.engineering import FeatureEngineer

logger = logging.getLogger(__name__)


class RevenueForecast:
    """Interface for making revenue predictions for customers."""
    
    def __init__(
        self,
        model_dir: Path,
        metrics_path: Path
    ):
        """
        Initialize forecasting interface.
        
        Args:
            model_dir: Directory containing trained model
            metrics_path: Path to daily customer metrics parquet file
        """
        self.model_dir = Path(model_dir)
        self.metrics_path = Path(metrics_path)
        
        # Load model
        self.predictor = RevenuePredictor()
        self.predictor.load(model_dir)
        logger.info("Loaded trained model")
        
        # Load metrics data
        self.metrics_df = pd.read_parquet(metrics_path)
        # self.metrics_df['date'] = pd.to_datetime(self.metrics_df['date'])
        logger.info(f"Loaded {len(self.metrics_df)} metric records")
        
        # Initialize feature engineer (needs to match training)
        self.feature_engineer = FeatureEngineer()
        
    def predict_next_day(
        self,
        customer_id: str,
        prediction_date: datetime
    ) -> dict:
        """
        Predict next-day revenue for a specific customer and date.
        
        Args:
            customer_id: Customer identifier
            prediction_date: Date to make prediction for
            
        Returns:
            Dictionary with prediction and supporting information
        """
        # Validate customer exists
        if customer_id not in self.metrics_df['customer_id'].values:
            return {
                'customer_id': customer_id,
                'prediction_date': prediction_date.date(),
                'predicted_net_gbp': 0.0,
                'confidence': 'unknown_customer',
                'message': f"Customer {customer_id} not found in historical data"
            }
        
        # Get historical data up to (but not including) prediction date
        customer_history = self.metrics_df[
            (self.metrics_df['customer_id'] == customer_id) &
            (self.metrics_df['date'] <= prediction_date)
        ].copy()
        
        if len(customer_history) == 0:
            return {
                'customer_id': customer_id,
                'prediction_date': prediction_date.date(),
                'predicted_net_gbp': 0.0,
                'confidence': 'no_history',
                'message': f"No historical data before {prediction_date.date()}"
            }
        
        # Create a dummy row for the prediction date
        # We'll compute features based on historical data
        last_known_date = customer_history['date'].max()
        
        # Generate complete date range for feature computation
        date_range = pd.date_range(
            start=customer_history['date'].min(),
            end=prediction_date,
            freq='D'
        )
        
        # Create complete grid
        complete_grid = pd.DataFrame({
            'date': date_range,
            'customer_id': customer_id
        })
        
        # Merge with actual history
        complete_data = complete_grid.merge(
            customer_history,
            on=['date', 'customer_id'],
            how='left'
        )
        
        # Fill missing numeric columns with 0
        numeric_cols = [
            'orders', 'items', 'gross_gbp', 'returns_gbp', 
            'net_gbp', 'unique_products', 'return_rate', 'avg_order_value'
        ]
        for col in numeric_cols:
            if col in complete_data.columns:
                complete_data[col] = complete_data[col].fillna(0)
        
        # Engineer features
        features_df = self.feature_engineer.engineer_features(complete_data, inference_time=True)
        
        # Get the row for prediction date
        prediction_row = features_df[features_df['date'] == prediction_date]
        
        if len(prediction_row) == 0:
            return {
                'customer_id': customer_id,
                'prediction_date': prediction_date.date(),
                'predicted_net_gbp': 0.0,
                'confidence': 'feature_error',
                'message': "Could not generate features for prediction date"
            }
        
        # Extract features
        try:
            X = prediction_row[self.predictor.feature_names]
            # Make prediction
            prediction = self.predictor.predict(X)[0]
            
            # Ensure non-negative prediction (revenue can't be negative net)
            prediction = max(prediction, 0.0)
            
            # Calculate confidence based on recent activity
            recent_activity = customer_history[
                customer_history['date'] >= prediction_date - timedelta(days=30)
            ]
            
            if len(recent_activity) > 0:
                avg_recent_revenue = recent_activity['net_gbp'].mean()
                std_recent_revenue = recent_activity['net_gbp'].std()
                days_active = (recent_activity['orders'] > 0).sum()
                
                confidence = 'high' if days_active >= 5 else 'medium'
            else:
                confidence = 'low'
                avg_recent_revenue = 0
                std_recent_revenue = 0
                
            return {
                'customer_id': customer_id,
                'prediction_date': prediction_date.date(),
                'predicted_net_gbp': round(prediction, 2),
                'confidence': confidence,
                'recent_avg_revenue': round(avg_recent_revenue, 2),
                'recent_std_revenue': round(std_recent_revenue, 2),
                'last_transaction_date': last_known_date.date(),
                'days_since_last_transaction': (prediction_date - last_known_date).days
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'customer_id': customer_id,
                'prediction_date': prediction_date.date(),
                'predicted_net_gbp': 0.0,
                'confidence': 'error',
                'message': f"Prediction failed: {str(e)}"
            }
    
    
