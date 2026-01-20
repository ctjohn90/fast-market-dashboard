"""Configuration settings for the dashboard."""

from dataclasses import dataclass, field
from pathlib import Path
import os

from dotenv import load_dotenv


load_dotenv()


# FRED series definitions - Core indicators (used in composite)
FRED_SERIES: dict[str, str] = {
    "VIXCLS": "VIX Close",
    "DGS2": "2-Year Treasury Yield",
    "DGS10": "10-Year Treasury Yield",
    "T10Y2Y": "10Y-2Y Spread",
    "BAMLH0A0HYM2": "ICE BofA High Yield Spread",
    "BAMLC0A0CM": "ICE BofA IG Corporate Spread",
    "DTWEXBGS": "Trade Weighted USD Index",
    "SP500": "S&P 500 Index",
}

# Supplementary series - for experimentation
FRED_SERIES_EXPERIMENTAL: dict[str, str] = {
    # Pre-built stress indices (good benchmarks)
    "STLFSI4": "St. Louis Fed Financial Stress Index",
    "NFCI": "Chicago Fed National Financial Conditions",
    "KCFSI": "Kansas City Financial Stress Index",
    # Credit
    "BAMLC0A4CBBB": "BBB Corporate Spread",
    "BAMLC0A1CAAA": "AAA Corporate Spread",
    # Rates & Curve
    "T10Y3M": "10Y-3M Spread (alternative curve)",
    "DFII10": "10Y TIPS Yield (real rates)",
    "DFF": "Federal Funds Rate",
    # Commodities (risk sentiment)
    "DCOILWTICO": "WTI Crude Oil Price",
    # Money markets
    "DPRIME": "Bank Prime Loan Rate",
}

# All series combined for fetching
ALL_FRED_SERIES: dict[str, str] = {**FRED_SERIES, **FRED_SERIES_EXPERIMENTAL}


@dataclass
class Settings:
    """Application settings."""

    fred_api_key: str = field(default_factory=lambda: os.getenv("FRED_API_KEY", ""))
    alpha_vantage_api_key: str = field(
        default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY", "")
    )
    cache_dir: Path = field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "cache"
    )
    db_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "market_data.db"

    def validate(self) -> None:
        """Validate required settings."""
        if not self.fred_api_key:
            raise ValueError(
                "FRED_API_KEY not set. Get one at: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )

    def has_alpha_vantage(self) -> bool:
        """Check if Alpha Vantage API key is configured."""
        return bool(self.alpha_vantage_api_key)
