"""SQLite cache for market data."""

import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd


class DataCache:
    """SQLite-based cache for FRED data."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    series_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    value REAL NOT NULL,
                    fetched_at TEXT NOT NULL,
                    PRIMARY KEY (series_id, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS series_metadata (
                    series_id TEXT PRIMARY KEY,
                    title TEXT,
                    frequency TEXT,
                    units TEXT,
                    last_updated TEXT,
                    last_fetched TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_obs_series_date 
                ON observations(series_id, date)
            """)

    def get_latest_date(self, series_id: str) -> date | None:
        """Get the most recent date we have cached for a series."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT MAX(date) as max_date FROM observations WHERE series_id = ?",
                (series_id,),
            ).fetchone()
            if row and row["max_date"]:
                return date.fromisoformat(row["max_date"])
        return None

    def store_observations(
        self, series_id: str, df: pd.DataFrame, fetched_at: datetime
    ) -> int:
        """
        Store observations for a series.
        
        Args:
            series_id: FRED series ID
            df: DataFrame with date index and 'value' column
            fetched_at: When the data was fetched
            
        Returns:
            Number of rows inserted/updated
        """
        if df.empty:
            return 0

        fetched_str = fetched_at.isoformat()
        rows = [
            (series_id, idx.strftime("%Y-%m-%d"), float(val), fetched_str)
            for idx, val in df["value"].items()
            if pd.notna(val)
        ]

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO observations (series_id, date, value, fetched_at)
                VALUES (?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def store_metadata(
        self,
        series_id: str,
        title: str,
        frequency: str,
        units: str,
        last_updated: datetime,
    ) -> None:
        """Store or update series metadata."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO series_metadata 
                (series_id, title, frequency, units, last_updated, last_fetched)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    series_id,
                    title,
                    frequency,
                    units,
                    last_updated.isoformat(),
                    datetime.now().isoformat(),
                ),
            )

    def get_series(
        self, series_id: str, start_date: date | None = None, end_date: date | None = None
    ) -> pd.DataFrame:
        """
        Retrieve cached data for a series.
        
        Returns:
            DataFrame with DatetimeIndex and 'value' column
        """
        query = "SELECT date, value FROM observations WHERE series_id = ?"
        params: list = [series_id]

        if start_date:
            query += " AND date >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY date"

        with self._get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return pd.DataFrame(columns=["value"])

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df

    def get_all_series(
        self, start_date: date | None = None, end_date: date | None = None
    ) -> dict[str, pd.DataFrame]:
        """Retrieve all cached series as a dict of DataFrames."""
        with self._get_connection() as conn:
            series_ids = [
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT series_id FROM observations"
                ).fetchall()
            ]

        return {sid: self.get_series(sid, start_date, end_date) for sid in series_ids}

    def get_cache_status(self) -> dict[str, dict]:
        """Get status of cached data for each series."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT 
                    o.series_id,
                    COUNT(*) as observation_count,
                    MIN(o.date) as first_date,
                    MAX(o.date) as last_date,
                    m.title,
                    m.last_fetched
                FROM observations o
                LEFT JOIN series_metadata m ON o.series_id = m.series_id
                GROUP BY o.series_id
            """).fetchall()

        return {
            row["series_id"]: {
                "observation_count": row["observation_count"],
                "first_date": row["first_date"],
                "last_date": row["last_date"],
                "title": row["title"],
                "last_fetched": row["last_fetched"],
            }
            for row in rows
        }
