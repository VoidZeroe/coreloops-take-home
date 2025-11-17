"""
Unit tests for data cleaning module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cleaner import DataCleaner


class TestDataCleaner:
    """Test suite for DataCleaner class."""

    @pytest.fixture
    def sample_raw_transactions(self):
        """Create realistic dirty transaction data matching the production schema."""
        base_ts = pd.Timestamp("2024-10-01 10:00")

        data = {
            'invoice_id': ['INV001', 'INV001', 'INV002', 'INV003', 'INV004', 'INV005'],
            'customer_id': ['C001', 'C001', None, 'C002', 'C003', 'C004'],
            'country': ['GB', 'GB', 'GB', 'GB', 'GB', 'GB'],
            'currency': ['GBP', 'GBP', 'GBP', 'GBP', 'GBP', 'GBP'],
            'product_id': ['P100', 'P100', 'P200', 'P300', 'P200', 'P999'],
            'product_category': ['A', 'A', 'B', 'C', 'B', 'Z'],
            'description': ['Item A', None, None, 'Item C', 'Item B', 'Item Z'],
            'quantity': [2, 2, 5, -1, 3, 1],
            'unit_price': [10.0, 10.0, None, 5.0, np.nan, 12.0],
            'timestamp': [
                base_ts,
                base_ts,                       # duplicate row
                base_ts + timedelta(hours=4),
                base_ts + timedelta(days=1),
                base_ts + timedelta(days=1, hours=1),
                "bad-timestamp"                # invalid date
            ]
        }

        return pd.DataFrame(data)

    # ------------------------------------------------------------------
    # Basic structure
    # ------------------------------------------------------------------

    def test_clean_returns_dataframe(self, sample_raw_transactions):
        """Ensure clean() returns a DataFrame with expected columns."""
        cleaner = DataCleaner()
        cleaned = cleaner.clean(sample_raw_transactions)

        expected_cols = {
            'invoice_id', 'customer_id', 'country', 'currency',
            'product_id', 'product_category', 'description',
            'quantity', 'unit_price', 'timestamp'
        }

        assert isinstance(cleaned, pd.DataFrame)
        assert expected_cols.issubset(cleaned.columns)

    # ------------------------------------------------------------------
    # Date parsing
    # ------------------------------------------------------------------

    def test_invalid_dates_are_removed(self, sample_raw_transactions):
        """Verify rows with unparseable timestamps are dropped."""
        cleaner = DataCleaner()
        cleaned = cleaner.clean(sample_raw_transactions)

        # One row has "bad-timestamp"
        assert cleaner.stats['records_dropped'] >= 1
        assert not cleaned['timestamp'].isna().any()

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    def test_duplicate_rows_removed(self, sample_raw_transactions):
        """Ensure exact duplicates are removed."""
        cleaner = DataCleaner()
        cleaned = cleaner.clean(sample_raw_transactions)

        # INV001 first two rows are exact duplicates
        assert cleaner.stats['duplicates_removed'] == 1

    # ------------------------------------------------------------------
    # Missing customer ID
    # ------------------------------------------------------------------

    def test_missing_customer_ids_are_dropped(self, sample_raw_transactions):
        """Rows missing customer_id are removed entirely."""
        cleaner = DataCleaner()
        cleaned = cleaner.clean(sample_raw_transactions)

        assert cleaned['customer_id'].isna().sum() == 0
        assert cleaner.stats['missing_customer_id'] == 1
        assert cleaner.stats['records_dropped'] >= 1

    
    # ------------------------------------------------------------------
    # Missing unit price
    # ------------------------------------------------------------------

    def test_unit_price_imputation_or_drop(self, sample_raw_transactions):
        """Missing unit_price values are imputed when possible or dropped otherwise."""
        cleaner = DataCleaner()
        cleaned = cleaner.clean(sample_raw_transactions)

        # All unit_price must now be valid numbers
        assert cleaned['unit_price'].isna().sum() == 0
        assert cleaner.stats['missing_unit_price'] >= 1

    # ------------------------------------------------------------------
    # Numeric type validation
    # ------------------------------------------------------------------

    def test_numeric_type_validation(self):
        """Invalid numeric values trigger row removal."""
        cleaner = DataCleaner()

        df = pd.DataFrame({
            'invoice_id': ['INV1', 'INV2'],
            'customer_id': ['C1', 'C2'],
            'country': ['GB', 'GB'],
            'currency': ['GBP', 'GBP'],
            'product_id': ['P1', 'P2'],
            'product_category': ['A', 'B'],
            'description': ['X', 'Y'],
            'quantity': ['bad', 2],
            'unit_price': [5.0, 'oops'],
            'timestamp': [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
        })

        cleaned = cleaner.clean(df)

        assert len(cleaned) == 0
        assert cleaner.stats['records_dropped'] == 2

    # ------------------------------------------------------------------
    # End-to-end sanity check
    # ------------------------------------------------------------------

    def test_full_cleaning_pipeline(self, sample_raw_transactions):
        """Verify total rows after full cleaning pass."""
        cleaner = DataCleaner()
        cleaned = cleaner.clean(sample_raw_transactions)

        # Expected behavior:
        # - 1 duplicate
        # - 1 missing customer_id
        # - 1 invalid timestamp
        dropped = cleaner.stats['records_dropped']
        assert dropped >= 3
        

        # No missing required fields remain
        assert cleaned['customer_id'].isna().sum() == 0
        assert cleaned['timestamp'].isna().sum() == 0
        assert cleaned['quantity'].isna().sum() == 0
        assert cleaned['unit_price'].isna().sum() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
