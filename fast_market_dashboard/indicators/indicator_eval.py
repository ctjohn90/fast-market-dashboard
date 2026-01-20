"""Evaluate individual indicator effectiveness."""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from fast_market_dashboard.data.cache import DataCache
from fast_market_dashboard.config import Settings, ALL_FRED_SERIES


# Known market stress events with S&P 500 drawdowns
STRESS_EVENTS = [
    {"name": "Financial Crisis", "start": date(2008, 9, 15), "end": date(2009, 3, 9)},
    {"name": "Debt Ceiling 2011", "start": date(2011, 7, 22), "end": date(2011, 10, 3)},
    {"name": "China Deval 2015", "start": date(2015, 8, 18), "end": date(2015, 9, 29)},
    {"name": "Fed Tightening 2018", "start": date(2018, 10, 3), "end": date(2018, 12, 24)},
    {"name": "COVID Crash", "start": date(2020, 2, 19), "end": date(2020, 3, 23)},
    {"name": "Rate Shock 2022", "start": date(2022, 1, 3), "end": date(2022, 10, 12)},
]


@dataclass
class IndicatorEval:
    """Evaluation results for a single indicator."""

    series_id: str
    name: str
    data_start: date
    data_end: date
    obs_count: int
    events_covered: int
    events_detected: int
    detection_rate: float
    avg_signal_during_stress: float
    avg_signal_normal: float
    signal_ratio: float  # stress / normal - higher is better
    forward_return_10d_high: float  # Avg 10-day return when signal > 80
    forward_return_10d_low: float  # Avg 10-day return when signal < 20


class IndicatorEvaluator:
    """Evaluate all available indicators."""

    LOOKBACK = 252
    SIGNAL_THRESHOLD = 70

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.cache = DataCache(self.settings.db_path)

    def _percentile_rank(self, series: pd.Series) -> pd.Series:
        """Calculate rolling percentile rank (0-100)."""
        def calc_pct(s: pd.Series) -> float:
            if len(s) < 20:
                return 50.0
            current = s.iloc[-1]
            return float((s < current).sum() / (len(s) - 1) * 100)

        return series.rolling(window=self.LOOKBACK, min_periods=20).apply(
            calc_pct, raw=False
        )

    def _is_stress_date(self, d) -> bool:
        """Check if date is during a stress event."""
        if hasattr(d, 'date'):
            d = d.date()
        for event in STRESS_EVENTS:
            if event["start"] <= d <= event["end"]:
                return True
        return False

    def _events_in_range(self, start: date, end: date) -> list[dict]:
        """Get events that fall within the data range."""
        return [
            e for e in STRESS_EVENTS
            if e["start"] >= start and e["end"] <= end
        ]

    def evaluate_series(self, series_id: str) -> IndicatorEval | None:
        """Evaluate a single series as a stress indicator."""
        df = self.cache.get_series(series_id)
        if df.empty or len(df) < 300:
            return None

        name = ALL_FRED_SERIES.get(series_id, series_id)
        values = df["value"].dropna()

        # Calculate percentile scores
        scores = self._percentile_rank(values).dropna()
        if len(scores) < 252:
            return None

        # Data range
        data_start = scores.index[0].date()
        data_end = scores.index[-1].date()

        # Get S&P 500 for forward returns
        sp500 = self.cache.get_series("SP500")
        if not sp500.empty:
            sp_aligned = sp500["value"].reindex(scores.index)
            forward_10d = sp_aligned.pct_change(10).shift(-10) * 100
        else:
            forward_10d = pd.Series(index=scores.index, dtype=float)

        # Identify stress periods
        is_stress = pd.Series(
            [self._is_stress_date(d) for d in scores.index],
            index=scores.index
        )

        # Events covered by this data
        events_covered = self._events_in_range(data_start, data_end)

        # Event detection: did the indicator spike during each event?
        events_detected = 0
        for event in events_covered:
            event_scores = scores[
                (scores.index >= pd.Timestamp(event["start"]))
                & (scores.index <= pd.Timestamp(event["end"]))
            ]
            if not event_scores.empty and event_scores.max() >= self.SIGNAL_THRESHOLD:
                events_detected += 1

        # Signal strength comparison
        stress_signals = scores[is_stress]
        normal_signals = scores[~is_stress]

        avg_stress = stress_signals.mean() if len(stress_signals) > 0 else 50
        avg_normal = normal_signals.mean() if len(normal_signals) > 0 else 50
        signal_ratio = avg_stress / avg_normal if avg_normal > 0 else 1.0

        # Forward returns
        high_signal_returns = forward_10d[scores >= 80].mean()
        low_signal_returns = forward_10d[scores <= 20].mean()

        return IndicatorEval(
            series_id=series_id,
            name=name,
            data_start=data_start,
            data_end=data_end,
            obs_count=len(scores),
            events_covered=len(events_covered),
            events_detected=events_detected,
            detection_rate=events_detected / len(events_covered) if events_covered else 0,
            avg_signal_during_stress=round(avg_stress, 1),
            avg_signal_normal=round(avg_normal, 1),
            signal_ratio=round(signal_ratio, 2),
            forward_return_10d_high=round(high_signal_returns, 2) if pd.notna(high_signal_returns) else 0,
            forward_return_10d_low=round(low_signal_returns, 2) if pd.notna(low_signal_returns) else 0,
        )

    def evaluate_all(self) -> list[IndicatorEval]:
        """Evaluate all cached series."""
        results = []
        status = self.cache.get_cache_status()

        for series_id in status.keys():
            eval_result = self.evaluate_series(series_id)
            if eval_result:
                results.append(eval_result)

        # Sort by signal ratio (best discriminators first)
        results.sort(key=lambda x: x.signal_ratio, reverse=True)
        return results

    def print_report(self) -> None:
        """Print evaluation report."""
        results = self.evaluate_all()

        if not results:
            print("No data available for evaluation.")
            return

        print("\n" + "=" * 100)
        print("INDICATOR EVALUATION REPORT")
        print("=" * 100)

        print("\n--- Ranking by Stress Detection Power (Signal Ratio) ---\n")
        print(f"{'Series':<18} {'Name':<35} {'Events':<8} {'Detect%':<8} {'Ratio':<7} {'Stress':<8} {'Normal':<8}")
        print("-" * 100)

        for r in results:
            events = f"{r.events_detected}/{r.events_covered}" if r.events_covered > 0 else "N/A"
            detect = f"{r.detection_rate*100:.0f}%" if r.events_covered > 0 else "N/A"
            print(
                f"{r.series_id:<18} {r.name[:34]:<35} {events:<8} {detect:<8} "
                f"{r.signal_ratio:<7.2f} {r.avg_signal_during_stress:<8.1f} {r.avg_signal_normal:<8.1f}"
            )

        print("\n--- Forward Return Predictability ---\n")
        print(f"{'Series':<18} {'Name':<35} {'High Signal':<12} {'Low Signal':<12} {'Spread':<10}")
        print("-" * 100)

        # Sort by return spread
        by_returns = sorted(results, key=lambda x: x.forward_return_10d_low - x.forward_return_10d_high, reverse=True)

        for r in by_returns:
            spread = r.forward_return_10d_low - r.forward_return_10d_high
            high_str = f"{r.forward_return_10d_high:+.2f}%" if r.forward_return_10d_high != 0 else "N/A"
            low_str = f"{r.forward_return_10d_low:+.2f}%" if r.forward_return_10d_low != 0 else "N/A"
            spread_str = f"{spread:+.2f}%" if spread != 0 else "N/A"
            print(f"{r.series_id:<18} {r.name[:34]:<35} {high_str:<12} {low_str:<12} {spread_str:<10}")

        print("\n" + "=" * 100)

        # Summary
        print("\nKEY FINDINGS:")
        print("-" * 50)

        # Best discriminators
        best_ratio = results[0] if results else None
        if best_ratio:
            print(f"\nBest stress discriminator: {best_ratio.series_id}")
            print(f"  - Avg signal during stress: {best_ratio.avg_signal_during_stress}")
            print(f"  - Avg signal during normal: {best_ratio.avg_signal_normal}")
            print(f"  - Ratio: {best_ratio.signal_ratio}x higher during stress")

        # Best return predictor
        best_return = by_returns[0] if by_returns else None
        if best_return:
            spread = best_return.forward_return_10d_low - best_return.forward_return_10d_high
            if spread > 0:
                print(f"\nBest return predictor: {best_return.series_id}")
                print(f"  - 10-day return after high signal (>80): {best_return.forward_return_10d_high:+.2f}%")
                print(f"  - 10-day return after low signal (<20): {best_return.forward_return_10d_low:+.2f}%")
                print(f"  - Spread: {spread:+.2f}%")

        # Detection rate leaders
        full_detect = [r for r in results if r.events_covered >= 3 and r.detection_rate == 1.0]
        if full_detect:
            print(f"\nPerfect detection (100% of 3+ events): {', '.join(r.series_id for r in full_detect)}")


def main() -> None:
    """CLI entry point."""
    evaluator = IndicatorEvaluator()
    evaluator.print_report()


if __name__ == "__main__":
    main()
