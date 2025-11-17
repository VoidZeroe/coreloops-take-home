"""
Feature engineering module for ML model training.
"""
import logging
from typing import List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Generates features for predicting next-day net_gbp."""
    
    def __init__(self, lookback_windows: Optional[List[int]] = None):
        """
        Initialize feature engineer.
        
        Args:
            lookback_windows: List of rolling window sizes in days
        """
        self.lookback_windows = lookback_windows or [7, 14, 30]
        self.feature_columns = []
        
    def engineer_features(
        self, 
        df: pd.DataFrame,
        target_col: str = 'net_gbp',
        inference_time: bool = False
    ) -> pd.DataFrame:
        """
        Generate all features for ML model.
        
        Args:
            df: Daily customer metrics DataFrame
            target_col: Target variable to predict
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features for ML model")
        
        df = df.copy()
        df = df.sort_values(['customer_id', 'date'])
        

        # ----------------------------------------------------
        # 1. CUSTOMER-LIFETIME STATIC FEATURES
        # ----------------------------------------------------
        customer_stats = df.groupby("customer_id")["net_gbp"].agg(
            lifetime_mean="mean",
            lifetime_std="std",
            lifetime_max="max",
            lifetime_sum="sum"
        ).reset_index()

        df = df.merge(customer_stats, on="customer_id", how="left")

        # ----------------------------------------------------
        # 2. 30-DAY ORDER FREQUENCY
        # ----------------------------------------------------
        df["rolling_order_freq_30d"] = (
            df.groupby("customer_id")["orders"]
              .transform(lambda x: x.rolling(30, min_periods=1).sum())
        )

        # ----------------------------------------------------
        # 3. 30-DAY SPEND TREND
        # ----------------------------------------------------
        df["spend_trend_30d"] = (
            df.groupby("customer_id")["net_gbp"]
              .transform(lambda x: x.pct_change(30))
        ).fillna(0)

        # ----------------------------------------------------
        # 4. TEMPORAL FEATURES
        # ----------------------------------------------------
        df = self._add_temporal_features(df)

        # ----------------------------------------------------
        # 5. ROLLING WINDOW FEATURES (UPDATED TO LONGER WINDOWS)
        # ----------------------------------------------------
       
        df = self._add_rolling_features(df)

        # ----------------------------------------------------
        # 6. RECENCY FEATURES
        # ----------------------------------------------------
        df = self._add_recency_features(df)

        # ----------------------------------------------------
        # 7. SMOOTHED TARGET (ONLY DURING TRAINING)
        # ----------------------------------------------------
        if not inference_time:
            df["smoothed_target"] = (
                df.groupby("customer_id")["next_day_net_gbp"]
                  .transform(lambda x: x.shift(-1).rolling(3, min_periods=1).mean())
            )

            # Remove rows without next-day observation
            df = df[df["next_day_net_gbp"].notna()].reset_index(drop=True)

        # ----------------------------------------------------
        # 8. FEATURE COLUMN REGISTRATION
        # ----------------------------------------------------
        excluded = ['date', 'customer_id', target_col]
        if not inference_time:
            excluded.append('smoothed_target')

        self.feature_columns = [
            col for col in df.columns 
            if col not in excluded
        ]

        logger.info(f"Generated {len(self.feature_columns)} features")
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features."""
        df['day_of_week'] = df['date'].dt.dayofweek
        df['day_of_month'] = df['date'].dt.day
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_start'] = (df['date'].dt.day <= 7).astype(int)
        df['is_month_end'] = (df['date'].dt.day >= 24).astype(int)
        
        return df
        
    def _add_rolling_features(
        self, 
        df: pd.DataFrame, 
    ) -> pd.DataFrame:
        """Add rolling window statistics."""
        
        # Columns to compute rolling stats for
        roll_cols = ['net_gbp', 'gross_gbp']
        
        for window in self.lookback_windows:
            for col in roll_cols:
                if col not in df.columns:
                    continue
                    
                # Rolling mean
                df[f'{col}_roll_{window}d_mean'] = (
                    df.groupby('customer_id')[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )
                
                # Rolling std
                df[f'{col}_roll_{window}d_std'] = (
                    df.groupby('customer_id')[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).std())
                )
                
                # Rolling max
                df[f'{col}_roll_{window}d_max'] = (
                    df.groupby('customer_id')[col]
                    .transform(lambda x: x.rolling(window, min_periods=1).max())
                )
                
        # Fill NaN std with 0 (happens when window has <2 values)
        std_cols = [col for col in df.columns if '_std' in col]
        df[std_cols] = df[std_cols].fillna(0)
        
        return df
                
    def _add_recency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add recency-based features (days since last event)."""
        
        # Days since last order
        df['has_order'] = (df['orders'] > 0).astype(int)
        df['days_since_last_order'] = (
            df.groupby('customer_id')['has_order']
            .transform(lambda x: x[::-1].eq(1).cumsum()[::-1])
        )
        
        # Days since last return
        df['has_return'] = (df['returns_gbp'] < 0).astype(int)
        df['days_since_last_return'] = (
            df.groupby('customer_id')['has_return']
            .transform(lambda x: x[::-1].eq(1).cumsum()[::-1])
        )
        
        # Cap at reasonable maximum
        df['days_since_last_order'] = df['days_since_last_order'].clip(upper=90)
        df['days_since_last_return'] = df['days_since_last_return'].clip(upper=180)
        
        # Drop helper columns
        df = df.drop(columns=['has_order', 'has_return'])
        
        return df
        
        
    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        target_col: str = 'net_gbp',
        test_size: float = 0.2,
        min_train_days: int = 2
    ) -> tuple:
        """
        Prepare time-based train/test split.
        
        Args:
            df: DataFrame with features
            target_col: Target variable name
            test_size: Fraction of data for test set
            min_train_days: Minimum days of history required per customer
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, train_dates, test_dates)
        """
        df = df.copy()
        logger.info(f"Train/test size is {len(df)}")

        # Remove records without sufficient history

        df = df[df['customer_id'].map(df['customer_id'].value_counts()) >= min_train_days]

        logger.info(f"Train/test size is {len(df)}")
        df.to_csv("df.csv")

        # Time-based split
        sorted_dates = sorted(df['date'].unique())
        if sorted_dates == []:
            raise ValueError("cleaned data empty, reduce the minimum training days and try again.")
        split_idx = int(len(sorted_dates) * (1 - test_size))
        split_date = sorted_dates[split_idx]
        
        train_df = df[df['date'] < split_date]
        test_df = df[df['date'] >= split_date]
        
        logger.info(f"Train/test split at {split_date}")
        logger.info(f"  - Train: {len(train_df)} records, {train_df['date'].nunique()} days")
        logger.info(f"  - Test: {len(test_df)} records, {test_df['date'].nunique()} days")
        
        # Prepare features and target
        X_train = train_df[self.feature_columns]
        y_train = train_df[target_col]
        X_test = test_df[self.feature_columns]
        y_test = test_df[target_col]
        return X_train, X_test, y_train, y_test
        
