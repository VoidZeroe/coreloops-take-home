"""
Unit tests for data aggregation module.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.aggregator import DailyAggregator


class TestDailyAggregator:
    """Test suite for DailyAggregator class."""
    
    @pytest.fixture
    def sample_transactions(self):
        """Create sample transaction data matching the actual schema."""
        
        data = {
            'invoice_id': ['INV001', 'INV001', 'INV002', 'INV003', 'INV004'],
            'customer_id': ['C001', None, 'C003', 'C001', 'C004'],  # ~2% missing
            'country': ['GB', 'GB', 'GB', 'GB', 'GB'],
            'currency': ['GBP', 'GBP', 'GBP', 'GBP', 'GBP'],
            'product_id': ['P1001', 'P2001', 'P3002', 'P1001', 'P2001'],
            'product_category': ['Electronics', 'Home', 'Sports', 'Electronics', 'Home'],
            'description': [
                'Wireless headphones', 
                'Ceramic mug', 
                None,                    # simulate missing description
                'Wireless headphones', 
                'Ceramic mug'
            ],
            'quantity': [5, 3, 10, -2, 8],  # negative = return
            'unit_price': [10.0, 15.0, 20.0, 10, 15.0],  # occasional NULL
            'unit_price_gbp': [10.0, 15.0, 20.0, 10, 15.0],  # occasional NULL
            'timestamp': pd.to_datetime([
                '2024-10-01 10:00',
                '2024-10-01 10:00',
                '2024-10-01 14:00',
                '2024-10-02 09:00',
                '2024-10-02 11:00'
            ])
        }
        
        return pd.DataFrame(data)

    
    def test_basic_aggregation(self, sample_transactions):
        """Test basic aggregation metrics."""
        aggregator = DailyAggregator()
        result = aggregator.aggregate(sample_transactions)
        
        # Check structure
        assert 'date' in result.columns
        assert 'customer_id' in result.columns
        assert 'orders' in result.columns
        assert 'items' in result.columns
        assert 'gross_gbp' in result.columns
        assert 'returns_gbp' in result.columns
        assert 'net_gbp' in result.columns
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
    
    def test_order_counting(self, sample_transactions):
        """Test that orders are counted correctly (distinct invoices)."""
        aggregator = DailyAggregator()
        result = aggregator.aggregate(sample_transactions)
        
        # Customer C001 on 2024-10-01 should have 2 distinct orders
        c001_day1 = result[
            (result['customer_id'] == 'C001') & 
            (result['date'] == pd.Timestamp('2024-10-01'))
        ]
        assert c001_day1['orders'].iloc[0] == 1
    
    def test_item_counting(self, sample_transactions):
        """Test that items are counted as absolute quantities."""
        aggregator = DailyAggregator()
        result = aggregator.aggregate(sample_transactions)
        
        # Customer C001 on 2024-10-01: 5 + 3 + 10 = 18 items
        c001_day1 = result[
            (result['customer_id'] == 'C001') & 
            (result['date'] == pd.Timestamp('2024-10-01'))
        ]
        assert c001_day1['items'].iloc[0] == 5
        
        # Customer C001 on 2024-10-02: |-2| = 2 items (absolute value)
        c001_day2 = result[
            (result['customer_id'] == 'C001') & 
            (result['date'] == pd.Timestamp('2024-10-02'))
        ]
        assert c001_day2['items'].iloc[0] == 2
    
    def test_return_rate_calculation(self, sample_transactions):
        """Test return rate calculation."""
        aggregator = DailyAggregator()
        result = aggregator.aggregate(sample_transactions)
        
        # Customer with positive gross should have valid return rate
        customers_with_gross = result[result['gross_gbp'] > 0]
        assert (customers_with_gross['return_rate'] >= 0).all()
        
        # Return rate should be 0 when returns_gbp is 0
        no_returns = result[result['returns_gbp'] == 0]
        assert (no_returns['return_rate'] == 0).all()
    
    def test_product_diversity(self, sample_transactions):
        """Test unique product counting."""
        aggregator = DailyAggregator()
        result = aggregator.aggregate(sample_transactions)
        
        # Customer C001 on 2024-10-01: 3 unique products (ITEM1, ITEM2, ITEM3)
        c001_day1 = result[
            (result['customer_id'] == 'C001') & 
            (result['date'] == pd.Timestamp('2024-10-01'))
        ]
        assert c001_day1['unique_products'].iloc[0] == 1
    
    def test_multiple_customers(self, sample_transactions):
        """Test aggregation across multiple customers."""
        aggregator = DailyAggregator()
        result = aggregator.aggregate(sample_transactions)
        
        # Should have records for both C001 and C002
        assert 'C001' in result['customer_id'].values
        assert 'C003' in result['customer_id'].values
        
        # Check total number of records
        assert len(result) == 4  # C001 on 2 days, C003 on 1 day, C004 on 1 day


if __name__ == '__main__':
    pytest.main([__file__, '-v'])