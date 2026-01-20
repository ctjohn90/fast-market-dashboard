"""FRED API data fetcher with delta updates."""

import logging
from datetime import date, datetime, timedelta

import httpx
import pandas as pd

from fast_market_dashboard.config import Settings, FRED_SERIES, ALL_FRED_SERIES
from fast_market_dashboard.data.cache import DataCache


logger = logging.getLogger(__name__)


class FredFetcher:
    """Fetches data from FRED API with local caching."""

    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.settings.validate()
        self.cache = DataCache(self.settings.db_path)
        self._client: httpx.Client | None = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            self._client = httpx.Client(timeout=30.0)
        return self._client

    def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "FredFetcher":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _fetch_series_info(self, series_id: str) -> dict:
        """Fetch metadata for a series from FRED."""
        response = self.client.get(
            f"{self.BASE_URL}/series",
            params={
                "series_id": series_id,
                "api_key": self.settings.fred_api_key,
                "file_type": "json",
            },
        )
        response.raise_for_status()
        data = response.json()

        if "seriess" not in data or not data["seriess"]:
            raise ValueError(f"Series {series_id} not found")

        return data["seriess"][0]

    def _fetch_observations(
        self, series_id: str, start_date: date | None = None
    ) -> pd.DataFrame:
        """
        Fetch observations from FRED API.
        
        Args:
            series_id: FRED series ID
            start_date: Only fetch data after this date (for delta updates)
            
        Returns:
            DataFrame with date index and value column
        """
        params = {
            "series_id": series_id,
            "api_key": self.settings.fred_api_key,
            "file_type": "json",
        }

        if start_date:
            # Add 1 day to avoid re-fetching the last date we have
            params["observation_start"] = (start_date + timedelta(days=1)).isoformat()

        response = self.client.get(
            f"{self.BASE_URL}/series/observations",
            params=params,
        )
        response.raise_for_status()
        data = response.json()

        observations = data.get("observations", [])
        if not observations:
            return pd.DataFrame(columns=["value"])

        df = pd.DataFrame(observations)
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df[["date", "value"]].dropna()
        df.set_index("date", inplace=True)

        return df

    def fetch_series(self, series_id: str, force_full: bool = False) -> pd.DataFrame:
        """
        Fetch a single series, using delta updates when possible.
        
        Args:
            series_id: FRED series ID
            force_full: If True, fetch entire history regardless of cache
            
        Returns:
            Complete DataFrame including cached + new data
        """
        logger.info(f"Fetching {series_id}...")

        # Determine start date for fetch
        start_date = None
        if not force_full:
            start_date = self.cache.get_latest_date(series_id)
            if start_date:
                logger.info(f"  Delta update from {start_date}")

        # Fetch from API
        new_data = self._fetch_observations(series_id, start_date)
        fetched_at = datetime.now()

        if not new_data.empty:
            rows_stored = self.cache.store_observations(series_id, new_data, fetched_at)
            logger.info(f"  Stored {rows_stored} new observations")

            # Update metadata
            try:
                info = self._fetch_series_info(series_id)
                self.cache.store_metadata(
                    series_id=series_id,
                    title=info.get("title", ""),
                    frequency=info.get("frequency", ""),
                    units=info.get("units", ""),
                    last_updated=datetime.fromisoformat(
                        info.get("last_updated", fetched_at.isoformat()).replace(
                            " ", "T"
                        )
                    ),
                )
            except Exception as e:
                logger.warning(f"  Could not fetch metadata: {e}")
        else:
            logger.info("  No new data")

        # Return full cached series
        return self.cache.get_series(series_id)

    def fetch_all(
        self, force_full: bool = False, include_experimental: bool = False
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch all configured FRED series.
        
        Args:
            force_full: If True, fetch entire history for all series
            include_experimental: If True, also fetch experimental series
            
        Returns:
            Dict mapping series_id to DataFrame
        """
        results = {}
        errors = {}

        series_to_fetch = ALL_FRED_SERIES if include_experimental else FRED_SERIES

        for series_id in series_to_fetch:
            try:
                results[series_id] = self.fetch_series(series_id, force_full)
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error fetching {series_id}: {e.response.status_code}")
                errors[series_id] = str(e)
            except Exception as e:
                logger.error(f"Error fetching {series_id}: {e}")
                errors[series_id] = str(e)

        if errors:
            logger.warning(f"Failed to fetch {len(errors)} series: {list(errors.keys())}")

        return results

    def get_cached_data(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> dict[str, pd.DataFrame]:
        """Get all data from cache without fetching."""
        return self.cache.get_all_series(start_date, end_date)

    def get_status(self) -> dict:
        """Get cache status for all series."""
        status = self.cache.get_cache_status()

        # Add missing series info
        for series_id, title in FRED_SERIES.items():
            if series_id not in status:
                status[series_id] = {
                    "title": title,
                    "observation_count": 0,
                    "first_date": None,
                    "last_date": None,
                    "last_fetched": None,
                }

        return status


def main() -> None:
    """CLI entry point for fetching data."""
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Fetch FRED market data")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Force full refresh instead of delta update",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status and exit",
    )
    parser.add_argument(
        "--series",
        type=str,
        help="Fetch specific series only",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Fetch all series including experimental",
    )
    args = parser.parse_args()

    try:
        with FredFetcher() as fetcher:
            if args.status:
                status = fetcher.get_status()
                print("\nCache Status:")
                print("-" * 70)
                for series_id, info in sorted(status.items()):
                    count = info["observation_count"]
                    last = info["last_date"] or "N/A"
                    title = info.get("title", ALL_FRED_SERIES.get(series_id, ""))
                    print(f"{series_id:20} | {count:6} obs | Last: {last:10} | {title}")
                return

            if args.series:
                if args.series not in ALL_FRED_SERIES:
                    print(f"Unknown series: {args.series}")
                    print(f"Available: {', '.join(ALL_FRED_SERIES.keys())}")
                    sys.exit(1)
                fetcher.fetch_series(args.series, force_full=args.full)
            else:
                fetcher.fetch_all(force_full=args.full, include_experimental=args.all)

            print("\nDone. Cache status:")
            status = fetcher.get_status()
            series_to_show = ALL_FRED_SERIES if args.all else FRED_SERIES
            for series_id in series_to_show:
                info = status.get(series_id, {})
                count = info.get("observation_count", 0)
                last = info.get("last_date", "N/A")
                print(f"  {series_id}: {count} observations, last date: {last}")

    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        print(f"API error: {e.response.status_code} - {e.response.text}")
        sys.exit(1)


if __name__ == "__main__":
    main()
