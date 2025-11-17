"""
Foreign exchange rate utilities for currency conversion.
"""
import logging
from typing import Dict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FXConverter:
    """Handles conversion of monetary values to GBP using daily FX rates."""
    
    def __init__(self, fx_rates: pd.DataFrame):
        """
        Initialize converter with FX rate data.
        
        Args:
            fx_rates: DataFrame with columns: date, currency, rate_to_gbp
        """
        self.fx_rates = fx_rates.copy()
        self.fx_rates['date'] = pd.to_datetime(self.fx_rates['date']).dt.date
        
        # Create lookup dictionary for fast access
        self._build_rate_lookup()
        
        logger.info(
            f"Initialized FX converter with {len(self.fx_rates)} rate records"
        )
        
    def _build_rate_lookup(self):
        """Build nested dictionary for O(1) rate lookups."""
        self.rate_lookup = {}
        
        for _, row in self.fx_rates.iterrows():
            date = row['date']
            currency = row['currency']
            rate = row['rate_to_gbp']
            
            if date not in self.rate_lookup:
                self.rate_lookup[date] = {}
            self.rate_lookup[date][currency] = rate
            
        # Log available currencies and date range
        currencies = set()
        for date_rates in self.rate_lookup.values():
            currencies.update(date_rates.keys())
            
        dates = sorted(self.rate_lookup.keys())
        logger.info(f"Available currencies: {sorted(currencies)}")
        logger.info(f"FX rate date range: {dates[0]} to {dates[-1]}")
        
    def convert_to_gbp(
        self, 
        df: pd.DataFrame,
        amount_col: str = 'unit_price',
        currency_col: str = 'currency',
        date_col: str = 'timestamp'
    ) -> pd.DataFrame:
        """
        Convert monetary amounts to GBP.
        
        Args:
            df: DataFrame with transactions
            amount_col: Column containing amount to convert
            currency_col: Column containing currency code
            date_col: Column containing transaction date
            
        Returns:
            DataFrame with additional columns: 
            - {amount_col}_gbp: Converted amount
            - fx_rate_applied: Rate used for conversion
        """
        df = df.copy()
        
        # Ensure date is in correct format
        df['_date_key'] = pd.to_datetime(df[date_col]).dt.date
        
        # Apply conversion
        df['fx_rate_applied'] = df.apply(
            lambda row: self._get_rate(row['_date_key'], row[currency_col]),
            axis=1
        )
        
        df[f'{amount_col}_gbp'] = df[amount_col] * df['fx_rate_applied']
        
        # Log conversion statistics
        conversions_by_currency = df.groupby(currency_col).size()
        logger.info("Conversion statistics:")
        for currency, count in conversions_by_currency.items():
            logger.info(f"  - {currency}: {count} records")
            
        # Check for any failed conversions
        failed = df['fx_rate_applied'].isna().sum()
        if failed > 0:
            logger.warning(
                f"Failed to find FX rates for {failed} records "
                f"({failed/len(df)*100:.2f}%)"
            )
            
        df = df.drop(columns=['_date_key'])
        
        return df
        
    def _get_rate(self, date, currency: str) -> float:
        """
        Get FX rate for a specific date and currency.
        
        Args:
            date: Transaction date
            currency: Currency code (e.g., 'USD', 'EUR', 'GBP')
            
        Returns:
            Conversion rate to GBP
        """
        # GBP to GBP is always 1.0
        if currency == 'GBP':
            return 1.0
            
        # Try exact date match
        if date in self.rate_lookup and currency in self.rate_lookup[date]:
            return self.rate_lookup[date][currency]
            
        # Forward fill: Use most recent rate before this date
        available_dates = sorted([d for d in self.rate_lookup.keys() if d <= date])
        
        for past_date in reversed(available_dates):
            if currency in self.rate_lookup[past_date]:
                rate = self.rate_lookup[past_date][currency]
                logger.debug(
                    f"Forward-filled rate for {currency} on {date} from {past_date}"
                )
                return rate
                
        # Backward fill: Use first available rate after this date
        future_dates = sorted([d for d in self.rate_lookup.keys() if d > date])
        
        for future_date in future_dates:
            if currency in self.rate_lookup[future_date]:
                rate = self.rate_lookup[future_date][currency]
                logger.warning(
                    f"Backward-filled rate for {currency} on {date} from {future_date}"
                )
                return rate
                
        # No rate found - this will result in NaN
        logger.error(f"No FX rate found for {currency} on {date}")
        return np.nan
        
    
