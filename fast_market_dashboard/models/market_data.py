"""Data models for market data."""

from dataclasses import dataclass
from datetime import date, datetime


@dataclass
class SeriesObservation:
    """Single observation from a FRED series."""

    series_id: str
    date: date
    value: float


@dataclass
class SeriesMetadata:
    """Metadata for a FRED series."""

    series_id: str
    title: str
    frequency: str
    units: str
    last_updated: datetime
