# """
# Data aggregation module for computing daily customer metrics.
# """
import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class DailyAggregator:
    """Aggregates transaction data into daily per-customer metrics."""
    
    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Aggregating {len(df)} transactions into daily metrics")

        df = df.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date

        # Line-level GBP total
        df['line_total_gbp'] = df['quantity'] * df['unit_price_gbp']

        # Aggregation (aligned with schema)
        agg = df.groupby(['date', 'customer_id']).agg(
            orders=('invoice_id', 'nunique'),
            items=('quantity', lambda x: np.abs(x).sum()),
            gross_gbp=('line_total_gbp', lambda x: x[x > 0].sum()),
            returns_gbp=('line_total_gbp', lambda x: x[x < 0].sum()),
            net_gbp=('line_total_gbp', 'sum'),
            unique_products=('product_id', 'nunique'),
        ).reset_index()

        # Convert date to datetime64 (keeps day precision)
        agg['date'] = pd.to_datetime(agg['date'])

        # return_rate = negative returns / positive gross revenue
        agg['return_rate'] = np.where(
            agg['gross_gbp'] > 0,
            -agg['returns_gbp'] / agg['gross_gbp'],
            0
        )

        # Average order value (net revenue / number of distinct invoices)
        agg['avg_order_value'] = np.where(
            agg['orders'] > 0,
            agg['net_gbp'] / agg['orders'],
            0
        )

        # ---- NEW: next observed day net_gbp for the same customer ----
        # Sort then shift so each row gets the next row's net_gbp for the same customer.
        agg = agg.sort_values(['customer_id', 'date'])
        agg['next_day_net_gbp'] = agg.groupby('customer_id')['net_gbp'] \
                            .transform(lambda x: x.shift(-1).rolling(2, min_periods=1).mean())

        # agg['next_day_net_gbp'] = agg.groupby('customer_id')['net_gbp'].shift(-1)

        # # Drop rows where there is no next day for that customer (last observed day)
        # agg = agg[agg['next_day_net_gbp'].notna()].reset_index(drop=True)

        self._log_summary(agg)

        return agg
    # ---------------------------------------------------------------------- #

    def _log_summary(self, df: pd.DataFrame):
        """Log summary statistics of aggregated metrics."""

        logger.info("Aggregation summary:")
        logger.info(f"  - Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"  - Unique customers: {df['customer_id'].nunique()}")
        logger.info(f"  - Total daily records: {len(df)}")
        logger.info(f"  - Avg orders per customer-day: {df['orders'].mean():.2f}")
        logger.info(f"  - Avg net_gbp per customer-day: {df['net_gbp'].mean():.2f}")
        logger.info(f"  - Total net revenue: £{df['net_gbp'].sum():,.2f}")

        logger.info("Revenue distribution:")
        logger.info(f"  - Median net_gbp: £{df['net_gbp'].median():.2f}")
        logger.info(f"  - 90th percentile: £{df['net_gbp'].quantile(0.90):.2f}")
        logger.info(f"  - 99th percentile: £{df['net_gbp'].quantile(0.99):.2f}")

        active_days = df[df['gross_gbp'] > 0]
        if len(active_days) > 0:
            logger.info(
                f"  - Avg return rate: {active_days['return_rate'].mean():.2%}"
            )
