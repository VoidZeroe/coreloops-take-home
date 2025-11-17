"""
Model training module for next-day revenue prediction.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

logger = logging.getLogger(__name__)


class RevenuePredictor:
    """Trains and manages Gradient Boosting model for revenue prediction."""
    
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize predictor with model parameters.
        
        Args:
            model_params: GradientBoostingRegressor parameters (uses defaults if None)
        """
        self.model_params = model_params or {
            'n_estimators': 300,
            'learning_rate': 0.05,
            'max_depth': 3,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'subsample': 0.8,
            'random_state': 42,
            'verbose': 1
        }

        
        self.model = None
        self.feature_names = None
        self.metadata = {}
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ) -> Dict[str, float]:
        """
        Train Gradient Boosting model with validation.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Training Gradient Boosting model")
        logger.info(f"  - Training samples: {len(X_train)}")
        logger.info(f"  - Validation samples: {len(X_val)}")
        logger.info(f"  - Features: {X_train.shape[1]}")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Create and train model
        self.model = GradientBoostingRegressor(**self.model_params)
        
        logger.info("Fitting model...")
        self.model.fit(X_train, y_train,)
        
        logger.info(f"Training completed. Used {self.model.n_estimators} estimators")
        
        # Evaluate on dataset
        train_metrics = self.evaluate(X_train, y_train, dataset_name='training')
        validation_metrics = self.evaluate(X_val, y_val, dataset_name='validation')
        
        # Store metadata
        self.metadata = {
            'model_type': 'GradientBoostingRegressor',
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'n_estimators': self.model.n_estimators,
            'model_params': self.model_params,
            'val_metrics': [validation_metrics, train_metrics]
        }
        
        return validation_metrics, train_metrics
        
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = 'test'
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True target values
            dataset_name: Name for logging
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        y_pred = self.predict(X)
        
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y - y_pred) / (y + 1e-8))) * 100  # Add small constant to avoid div by zero
        
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape
        }
        
        logger.info(f"Evaluation on {dataset_name} set:")
        logger.info(f"  - MAE: £{mae:.2f}")
        logger.info(f"  - RMSE: £{rmse:.2f}")
        logger.info(f"  - R²: {r2:.4f}")
        logger.info(f"  - MAPE: {mape:.2f}%")
        
        return metrics
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        # Ensure feature order matches training
        X = X[self.feature_names]
        
        predictions = self.model.predict(X)
        
        return predictions
        
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance (only 'gain' supported for sklearn)
            
        Returns:
            DataFrame with feature importances
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        importance = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
        
    def save(self, model_dir: Path):
        """
        Save model and metadata to disk.
        
        Args:
            model_dir: Directory to save model files
        """
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = model_dir / 'model.joblib'
        joblib.dump(self.model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Save metadata
        metadata_path = model_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Save feature importance
        importance_df = self.get_feature_importance()
        importance_path = model_dir / 'feature_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Saved feature importance to {importance_path}")
        
    def load(self, model_dir: Path):
        """
        Load model and metadata from disk.
        
        Args:
            model_dir: Directory containing model files
        """
        model_dir = Path(model_dir)
        
        # Load model
        model_path = model_dir / 'model.joblib'
        self.model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        # Load metadata
        metadata_path = model_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.feature_names = self.metadata['feature_names']
        logger.info(f"Loaded metadata from {metadata_path}")
        
    