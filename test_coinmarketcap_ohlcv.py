"""
Test CoinMarketCap OHLCV helper
--------------------------------
Verifies we can pull historical OHLCV for BTC/USD via CoinMarketCap.
"""

import unittest
import pandas as pd

from config import get_coinmarketcap_api_key
from data_pipeline.historical_data_pipeline import get_available_pairs
from data_pipeline.coinmarketcap_api import get_ohlcv_historical


class CoinMarketCapOhlcvTest(unittest.TestCase):
    @unittest.skipUnless(get_coinmarketcap_api_key(), "CoinMarketCap API key not configured")
    def test_top3_pairs_hourly_ohlcv(self):
        """Fetch ~700 hours of OHLCV data for the first three Roostoo pairs."""
        pairs = get_available_pairs()[:3]
        self.assertTrue(pairs, "No trading pairs returned by Roostoo exchange info.")

        for roostoo_pair in pairs:
            symbol = roostoo_pair.split('/')[0]
            with self.subTest(pair=roostoo_pair):
                df = get_ohlcv_historical(
                    symbol=symbol,
                    interval='1h',
                    count=600,
                    as_dataframe=True,
                )
                print(df.shape)
                print(df.head())
                print(df.tail())
                self.assertIsNotNone(df, f"{roostoo_pair}: Expected DataFrame, got None")
                self.assertIsInstance(df, pd.DataFrame)
                self.assertFalse(df.empty, f"{roostoo_pair}: DataFrame is empty")

                required_cols = {'datetime', 'open', 'high', 'low', 'close', 'volume'}
                self.assertTrue(required_cols.issubset(df.columns), f"{roostoo_pair}: Missing columns {required_cols - set(df.columns)}")

                min_date = df['datetime'].min()
                max_date = df['datetime'].max()
                self.assertLess(min_date, max_date, f"{roostoo_pair}: Datetime range invalid")
                coverage_hours = (max_date - min_date).total_seconds() / 3600.0
                self.assertGreaterEqual(coverage_hours, 600, f"{roostoo_pair}: Coverage too short: {coverage_hours} hours")

                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    self.assertTrue(pd.api.types.is_numeric_dtype(df[col]), f"{roostoo_pair}: {col} should be numeric")


if __name__ == "__main__":
    unittest.main()

