"""
Data loading module for fetching transaction files from GCS.
"""
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import requests
from io import StringIO
import os
import re
import tempfile
import logging
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse
import requests
import pandas as pd

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("DATA_BUCKET_URL","https://storage.googleapis.com/tech-test-file-storage")


class DataLoader:
    """Handles loading of daily transaction files from GCS."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url                      
        self.session = requests.Session()
        
    def load_fx_rates(self) -> pd.DataFrame:
        """
        Load FX conversion rates from GCS.
        
        Returns:
            DataFrame with columns: date, currency, rate_to_gbp
        """
        url = f"{self.base_url}/fx_rates.csv"
        logger.info(f"Loading FX rates from {url}")
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text))
            df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"Loaded {len(df)} FX rate records")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Failed to load FX rates: {e}")
            raise
            
    def load_daily_file(self, date: datetime) -> Optional[pd.DataFrame]:
        """
        Load a single daily transaction file.
        
        Args:
            date: Date of the file to load
            
        Returns:
            DataFrame with transaction data, or None if file doesn't exist
        """
        date_str = date.strftime("%Y-%m-%d")
        url = f"{self.base_url}/data/{date_str}.csv"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text))
            df['file_date'] = date
            
            logger.info(f"Loaded {len(df)} records from {date_str}")
            return df
            
        except requests.RequestException as e:
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 404:
                logger.debug(f"No file found for {date_str}")
                return None
            else:
                logger.warning(f"Error loading {date_str}: {e}")
                return None
                
    def load_all_files(
        self,
        *,
        max_files: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Discover all date-like CSV filenames under data/ in the bucket and load them
        using the existing load_daily_file(date) method.

        This replaces the inefficient date-range loop with a discovery step, but
        keeps your load_daily_file() logic intact.
        """
        # -------------------------------------------------------
        # 1) Try to get bucket name from base_url.
        # -------------------------------------------------------
        parsed = urlparse(self.base_url)
        bucket = None
        if parsed.scheme in ("http", "https"):
            parts = self.base_url.strip("/").split("/")
            if parts:
                bucket = parts[-1]
        
        if not bucket:
            raise ValueError(f"Unable to infer GCS bucket name from base_url: {self.base_url}")

        # -------------------------------------------------------
        # 2) List objects via public GCS JSON API (works for public buckets)
        # -------------------------------------------------------
        list_url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
        params = {"prefix": "data/", "fields": "items/name,nextPageToken"}
        object_names = []
        next_page = None

        while True:
            if next_page:
                params["pageToken"] = next_page
            resp = self.session.get(list_url, params=params, timeout=30)
            resp.raise_for_status()
            payload = resp.json()
            items = payload.get("items", [])
            for it in items:
                name = it.get("name")
                if name and name.lower().endswith(".csv"):
                    object_names.append(name)
            next_page = payload.get("nextPageToken")
            if not next_page:
                break

        if not object_names:
            raise ValueError("No CSV transaction files found in GCS under data/")

        # -------------------------------------------------------
        # 3) Extract dates from filenames that match YYYY-MM-DD.csv
        # -------------------------------------------------------
        dated = []
        for obj_name in object_names:
            fname = obj_name.split("/")[-1]  # e.g. "2024-01-01.csv"
            try:
                date_part = fname.replace(".csv", "")
                date_val = datetime.strptime(date_part, "%Y-%m-%d")
                dated.append((date_val, obj_name))
            except ValueError:
                # skip non-date-like names
                logger.debug(f"Skipping non-date filename: {obj_name}")
                continue

        if not dated:
            raise ValueError("No date-like CSV filenames (YYYY-MM-DD.csv) found under data/")

        # sort by date ascending
        dated.sort(key=lambda x: x[0])

        if max_files is not None:
            dated = dated[:max_files]

        # -------------------------------------------------------
        # 4) Load each discovered file using  load_daily_file(date)
        # -------------------------------------------------------
        dfs = []
        loaded_files = 0
        for file_date, _obj_name in dated:
            df = self.load_daily_file(file_date)
            if df is not None:
                dfs.append(df)
                loaded_files += 1
            else:
                logger.debug(f"File for {file_date.date()} exists but failed to load via load_daily_file")

        if not dfs:
            raise ValueError("No transaction files could be loaded after discovery")

        combined = pd.concat(dfs, ignore_index=True)
        # ensure file_date is datetime and sort (your load_daily_file sets file_date already)
        if 'file_date' in combined.columns:
            combined['file_date'] = pd.to_datetime(combined['file_date'])
            combined = combined.sort_values('file_date').reset_index(drop=True)

        logger.info(f"Loaded {loaded_files} files with {len(combined)} total records")
        return combined

        
   