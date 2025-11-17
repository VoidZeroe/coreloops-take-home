"""
Data cleaning module for handling duplicates, missing values, and data quality.
"""
import logging
from typing import Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataCleaner:
    """Handles data cleaning, deduplication, and missing value imputation."""
    
    def __init__(self):
        self.stats = {
            'duplicates_removed': 0,
            'missing_customer_id': 0,
            'missing_unit_price': 0,
            'missing_description': 0,
            'records_dropped': 0
        }
        
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply full cleaning pipeline to transaction data.
        
        Args:
            df: Raw transaction DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting cleaning process with {len(df)} records")
        
        # Reset stats
        self.stats = {k: 0 for k in self.stats}
        
        # Parse dates
        df = self._parse_dates(df)
        
        # Deduplicate
        df = self._deduplicate(df)
        
        # Handle missing values
        df = self._handle_missing_customer_id(df)
        df = self._handle_missing_description(df)
        df = self._handle_missing_unit_price(df)
        
        # Validate data types
        df = self._validate_types(df)
        
        # Log statistics
        self._log_stats(df)
        
        return df
        
    def _parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamp to datetime."""
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Drop records with invalid dates
        invalid_dates = df['timestamp'].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Dropping {invalid_dates} records with invalid dates")
            df = df.dropna(subset=['timestamp'])
            self.stats['records_dropped'] += invalid_dates
            
        return df
        
    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate transaction-line entries.

        Criteria: Rows with identical invoice_id, product_id, quantity,
        unit_price, and timestamp are considered duplicates.
        """
        initial_count = len(df)

        duplicate_cols = ['invoice_id', 'product_id', 'quantity', 'unit_price', 'timestamp']
        df = df.drop_duplicates(subset=duplicate_cols, keep='first')

        duplicates_removed = initial_count - len(df)
        self.stats['duplicates_removed'] = duplicates_removed

        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate records")

        return df

        
    def _handle_missing_customer_id(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing customer_id values.
        
        Strategy: Drop rows without customer_id as we need this for 
        customer-level predictions.
        """
        missing_count = df['customer_id'].isna().sum()
        
        if missing_count > 0:
            logger.info(
                f"Dropping {missing_count} records with missing customer_id "
                f"({missing_count/len(df)*100:.2f}%)"
            )
            # df = df.dropna(subset=['customer_id'])
            df = df[df['customer_id'].notna()]
            self.stats['missing_customer_id'] = missing_count
            self.stats['records_dropped'] += missing_count
            
        return df
        
    def _handle_missing_description(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing description values.
        
        Strategy: Fill with 'UNKNOWN' as description is non-critical.
        """
        missing_count = df['description'].isna().sum()
        
        if missing_count > 0:
            logger.info(f"Filling {missing_count} missing descriptions with 'UNKNOWN'")
            df = df.copy()
            df['description'] = df['description'].fillna('UNKNOWN')
            self.stats['missing_description'] = missing_count
            
        return df
    def _handle_missing_unit_price(self, df: pd.DataFrame, 
                                   trailing_window:int = 7, 
                                   min_trailing_window_size:int=1) -> pd.DataFrame:
        """
        Handle missing unit_price values.

        Strategy:
        1. Impute using median price per (product_id, currency) over trailing 7 days
        2. Fallback to global median for product_id (any currency)
        3. Drop remaining rows if still missing
        """
        missing_mask = df['unit_price'].isna()
        missing_count = missing_mask.sum()

        if missing_count == 0:
            return df

        logger.info(f"Imputing {missing_count} missing unit_price values")

        df = df.copy()
        df = df.sort_values('timestamp')

        for idx in df[missing_mask].index:
            product_id = df.loc[idx, 'product_id']
            currency = df.loc[idx, 'currency']
            ts = df.loc[idx, 'timestamp']

            # Strategy 1: Trailing 7-day median for (product_id, currency)
            window_mask = (
                (df['product_id'] == product_id) &
                (df['currency'] == currency) &
                (df['timestamp'] < ts) &
                (df['timestamp'] >= ts - pd.Timedelta(days=trailing_window)) &
                (df['unit_price'].notna())
            )

            if window_mask.sum() >= min_trailing_window_size:   # require minimum observations
                df.loc[idx, 'unit_price'] = df.loc[window_mask, 'unit_price'].median()
                continue


        # Strategy 2: drop unresolved missing
        still_missing = df['unit_price'].isna().sum()
        if still_missing > 0:
            logger.warning(
                f"Dropping {still_missing} records with unit_price still missing after imputation"
            )
            df = df.dropna(subset=['unit_price'])

            # ensure counter exists
            if 'records_dropped' not in self.stats:
                self.stats['records_dropped'] = 0

            self.stats['records_dropped'] += still_missing

        self.stats['missing_unit_price'] = missing_count

        return df
        
    def _validate_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data types."""
        df = df.copy()
        
        # Ensure numeric types
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
        
        # Drop any records with invalid numeric conversions
        invalid = df[['quantity', 'unit_price']].isna().any(axis=1).sum()
        if invalid > 0:
            logger.warning(f"Dropping {invalid} records with invalid numeric values")
            df = df.dropna(subset=['quantity', 'unit_price'])
            self.stats['records_dropped'] += invalid
            
        return df
        
    def _log_stats(self, df: pd.DataFrame):
        """Log cleaning statistics."""
        logger.info("Cleaning complete:")
        logger.info(f"  - Final record count: {len(df)}")
        logger.info(f"  - Duplicates removed: {self.stats['duplicates_removed']}")
        logger.info(f"  - Missing customer_id: {self.stats['missing_customer_id']}")
        logger.info(f"  - Missing unit_price: {self.stats['missing_unit_price']}")
        logger.info(f"  - Missing description: {self.stats['missing_description']}")
        logger.info(f"  - Total records dropped: {self.stats['records_dropped']}")