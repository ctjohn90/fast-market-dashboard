"""Alpha Vantage data fetcher for technical indicators.

Free tier: 25 requests/day. Use sparingly for indicators not available elsewhere.
"""

import logging
import time
from datetime import datetime

import httpx
import pandas as pd

from fast_market_dashboard.data.cache import DataCache
from fast_market_dashboard.config import Settings


logger = logging.getLogger(__name__)

# Rate limit: 5 requests per minute on free tier
REQUEST_DELAY = 12  # seconds between requests


class AlphaVantageFetcher:
    """Fetches data from Alpha Vantage API."""

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.cache = DataCache(self.settings.db_path)
        self._client: httpx.Client | None = None

        if not self.settings.has_alpha_vantage():
            logger.warning("ALPHA_VANTAGE_API_KEY not set")

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=30.0)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "AlphaVantageFetcher":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _request(self, params: dict) -> dict:
        """Make API request with rate limiting."""
        params["apikey"] = self.settings.alpha_vantage_api_key

        response = self.client.get(self.BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()

        # Check for API errors
        if "Error Message" in data:
            raise ValueError(data["Error Message"])
        if "Note" in data:
            # Rate limit warning
            logger.warning(f"API Note: {data['Note']}")

        return data

    def fetch_rsi(self, symbol: str, interval: str = "daily", period: int = 14) -> pd.DataFrame:
        """
        Fetch RSI (Relative Strength Index) for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., "SPY")
            interval: "daily", "weekly", "monthly"
            period: RSI period (default 14)
            
        Returns:
            DataFrame with RSI values
        """
        series_id = f"AV_RSI_{symbol}_{period}"
        logger.info(f"Fetching RSI for {symbol} (period={period})...")

        data = self._request({
            "function": "RSI",
            "symbol": symbol,
            "interval": interval,
            "time_period": period,
            "series_type": "close",
        })

        key = f"Technical Analysis: RSI"
        if key not in data:
            logger.warning(f"No RSI data for {symbol}")
            return pd.DataFrame()

        rsi_data = data[key]
        df = pd.DataFrame.from_dict(rsi_data, orient="index")
        df.index = pd.to_datetime(df.index)
        df.columns = ["value"]
        df["value"] = pd.to_numeric(df["value"])
        df = df.sort_index()

        # Store in cache
        self.cache.store_observations(series_id, df, datetime.now())
        logger.info(f"  Stored {len(df)} RSI observations")

        return df

    def fetch_macd(self, symbol: str, interval: str = "daily") -> pd.DataFrame:
        """
        Fetch MACD for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., "SPY")
            interval: "daily", "weekly", "monthly"
            
        Returns:
            DataFrame with MACD, signal, and histogram
        """
        series_id = f"AV_MACD_{symbol}"
        logger.info(f"Fetching MACD for {symbol}...")

        data = self._request({
            "function": "MACD",
            "symbol": symbol,
            "interval": interval,
            "series_type": "close",
        })

        key = "Technical Analysis: MACD"
        if key not in data:
            logger.warning(f"No MACD data for {symbol}")
            return pd.DataFrame()

        macd_data = data[key]
        records = []
        for date_str, values in macd_data.items():
            records.append({
                "date": pd.to_datetime(date_str),
                "macd": float(values["MACD"]),
                "signal": float(values["MACD_Signal"]),
                "histogram": float(values["MACD_Hist"]),
            })

        df = pd.DataFrame(records).set_index("date").sort_index()

        # Store histogram as the stress signal (negative = bearish)
        hist_df = pd.DataFrame({"value": df["histogram"]})
        self.cache.store_observations(series_id, hist_df, datetime.now())
        logger.info(f"  Stored {len(df)} MACD observations")

        return df

    def fetch_adx(self, symbol: str, interval: str = "daily", period: int = 14) -> pd.DataFrame:
        """
        Fetch ADX (Average Directional Index) for trend strength.
        
        ADX > 25 = strong trend, ADX < 20 = weak/no trend
        
        Args:
            symbol: Stock symbol
            interval: "daily", "weekly", "monthly"
            period: ADX period (default 14)
            
        Returns:
            DataFrame with ADX values
        """
        series_id = f"AV_ADX_{symbol}_{period}"
        logger.info(f"Fetching ADX for {symbol}...")

        data = self._request({
            "function": "ADX",
            "symbol": symbol,
            "interval": interval,
            "time_period": period,
        })

        key = "Technical Analysis: ADX"
        if key not in data:
            logger.warning(f"No ADX data for {symbol}")
            return pd.DataFrame()

        adx_data = data[key]
        df = pd.DataFrame.from_dict(adx_data, orient="index")
        df.index = pd.to_datetime(df.index)
        df.columns = ["value"]
        df["value"] = pd.to_numeric(df["value"])
        df = df.sort_index()

        self.cache.store_observations(series_id, df, datetime.now())
        logger.info(f"  Stored {len(df)} ADX observations")

        return df

    def fetch_all_technicals(self, symbols: list[str] | None = None) -> dict[str, pd.DataFrame]:
        """
        Fetch RSI for key symbols.
        
        Default symbols: SPY, QQQ, IWM
        Uses 3 API calls per symbol, so be mindful of rate limits.
        """
        if symbols is None:
            symbols = ["SPY"]  # Conservative default (1 call)

        results = {}

        for symbol in symbols:
            try:
                # RSI only - most useful for stress detection
                rsi = self.fetch_rsi(symbol)
                if not rsi.empty:
                    results[f"{symbol}_RSI"] = rsi

                time.sleep(REQUEST_DELAY)

            except Exception as e:
                logger.error(f"Error fetching technicals for {symbol}: {e}")

        return results

    def get_status(self) -> dict[str, dict]:
        """Get cache status for Alpha Vantage series."""
        cache_status = self.cache.get_cache_status()

        # Filter for AV_ prefixed series
        return {
            k: v for k, v in cache_status.items()
            if k.startswith("AV_")
        }


def main() -> None:
    """CLI entry point."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fetch Alpha Vantage technical indicators")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status and exit",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="SPY",
        help="Symbol to fetch (default: SPY)",
    )
    parser.add_argument(
        "--indicator",
        type=str,
        choices=["rsi", "macd", "adx", "all"],
        default="rsi",
        help="Indicator to fetch",
    )
    args = parser.parse_args()

    settings = Settings()
    if not settings.has_alpha_vantage():
        print("Error: ALPHA_VANTAGE_API_KEY not set in .env")
        print("Get a free key at: https://www.alphavantage.co/support/#api-key")
        return

    with AlphaVantageFetcher(settings) as fetcher:
        if args.status:
            status = fetcher.get_status()
            print("\nAlpha Vantage Cache Status:")
            print("-" * 60)
            if not status:
                print("No cached Alpha Vantage data")
            for series_id, info in sorted(status.items()):
                count = info["observation_count"]
                last = info["last_date"] or "N/A"
                print(f"{series_id:30} | {count:6} obs | Last: {last}")
            return

        if args.indicator == "rsi":
            fetcher.fetch_rsi(args.symbol)
        elif args.indicator == "macd":
            fetcher.fetch_macd(args.symbol)
        elif args.indicator == "adx":
            fetcher.fetch_adx(args.symbol)
        elif args.indicator == "all":
            fetcher.fetch_rsi(args.symbol)
            time.sleep(REQUEST_DELAY)
            fetcher.fetch_macd(args.symbol)
            time.sleep(REQUEST_DELAY)
            fetcher.fetch_adx(args.symbol)

        print("\nDone.")


if __name__ == "__main__":
    main()
