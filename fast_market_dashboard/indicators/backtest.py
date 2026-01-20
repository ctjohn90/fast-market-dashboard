"""Backtest indicators against historical stress events."""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from fast_market_dashboard.data.cache import DataCache
from fast_market_dashboard.config import Settings
from fast_market_dashboard.indicators.calculator import IndicatorCalculator


# Known market stress events
STRESS_EVENTS = [
    {
        "name": "Financial Crisis",
        "start": date(2008, 9, 15),
        "end": date(2009, 3, 9),
        "peak_drawdown": 56.8,
    },
    {
        "name": "Debt Ceiling Crisis",
        "start": date(2011, 7, 22),
        "end": date(2011, 10, 3),
        "peak_drawdown": 19.4,
    },
    {
        "name": "China Devaluation",
        "start": date(2015, 8, 18),
        "end": date(2015, 9, 29),
        "peak_drawdown": 12.4,
    },
    {
        "name": "Fed Tightening",
        "start": date(2018, 10, 3),
        "end": date(2018, 12, 24),
        "peak_drawdown": 19.8,
    },
    {
        "name": "COVID Crash",
        "start": date(2020, 2, 19),
        "end": date(2020, 3, 23),
        "peak_drawdown": 33.9,
    },
    {
        "name": "2022 Rate Shock",
        "start": date(2022, 1, 3),
        "end": date(2022, 10, 12),
        "peak_drawdown": 25.4,
    },
    {
        "name": "Tariff Shock 2025",
        "start": date(2025, 4, 2),
        "end": date(2025, 4, 11),
        "peak_drawdown": 18.9,
    },
]


@dataclass
class IndicatorPerformance:
    """Performance metrics for a single indicator."""

    name: str
    events_detected: int  # How many stress events it signaled
    total_events: int
    avg_lead_days: float  # Average days before event it signaled
    false_positive_rate: float  # % of signals outside events
    forward_return_spread: float  # Return diff: high signal vs low signal


@dataclass
class BacktestResult:
    """Complete backtest results."""

    indicators: dict[str, IndicatorPerformance]
    composite_accuracy: float
    composite_precision: float
    composite_recall: float
    event_detection: list[dict]


class IndicatorBacktest:
    """Backtest indicator effectiveness."""

    SIGNAL_THRESHOLD = 70  # Percentile to consider "elevated"
    LOOKBACK_WINDOW = 10  # Days before event to check for lead signal

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.cache = DataCache(self.settings.db_path)
        self.calculator = IndicatorCalculator(settings)

    def _is_during_event(self, d: date) -> bool:
        """Check if date falls within any stress event."""
        for event in STRESS_EVENTS:
            if event["start"] <= d <= event["end"]:
                return True
        return False

    def _get_event_for_date(self, d: date) -> dict | None:
        """Get event info if date is within an event."""
        for event in STRESS_EVENTS:
            if event["start"] <= d <= event["end"]:
                return event
        return None

    def run(self) -> BacktestResult | None:
        """
        Run full backtest on cached data.
        
        Returns None if insufficient data.
        """
        # Get composite history
        history = self.calculator.get_history(days=5000)  # Get all available
        if history.empty or len(history) < 252:
            return None

        # Get S&P 500 for forward returns
        sp500 = self.cache.get_series("SP500")
        if sp500.empty:
            return None

        # Align data
        history = history[history.index.isin(sp500.index)]
        sp500_aligned = sp500.loc[history.index]

        # Calculate forward returns
        forward_5d = sp500_aligned["value"].pct_change(5).shift(-5) * 100
        forward_10d = sp500_aligned["value"].pct_change(10).shift(-10) * 100

        # Analyze composite score
        composite = history["composite"]

        # Event detection analysis
        event_detection = []
        for event in STRESS_EVENTS:
            # Check if we have data for this event
            event_dates = composite[
                (composite.index >= pd.Timestamp(event["start"]))
                & (composite.index <= pd.Timestamp(event["end"]))
            ]

            if event_dates.empty:
                continue

            # Check pre-event signal (LOOKBACK_WINDOW days before)
            pre_event = composite[
                (composite.index >= pd.Timestamp(event["start"]) - pd.Timedelta(days=self.LOOKBACK_WINDOW))
                & (composite.index < pd.Timestamp(event["start"]))
            ]

            lead_signal = pre_event[pre_event >= self.SIGNAL_THRESHOLD]
            lead_days = len(lead_signal) if not lead_signal.empty else 0

            # Peak signal during event
            peak_signal = event_dates.max()
            detected = peak_signal >= self.SIGNAL_THRESHOLD

            event_detection.append({
                "name": event["name"],
                "start": event["start"],
                "detected": detected,
                "peak_signal": round(peak_signal, 1),
                "lead_days": lead_days,
            })

        # Calculate precision/recall
        # True positive: composite > threshold AND during stress event
        # False positive: composite > threshold AND NOT during stress event
        high_signals = composite >= self.SIGNAL_THRESHOLD
        during_stress = pd.Series(
            [self._is_during_event(d.date()) for d in composite.index],
            index=composite.index,
        )

        true_positives = (high_signals & during_stress).sum()
        false_positives = (high_signals & ~during_stress).sum()
        false_negatives = (~high_signals & during_stress).sum()

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        accuracy = ((high_signals == during_stress).sum()) / len(composite)

        # Forward return analysis
        high_signal_returns = forward_10d[composite >= 80].mean()
        low_signal_returns = forward_10d[composite <= 20].mean()
        return_spread = low_signal_returns - high_signal_returns if pd.notna(high_signal_returns) and pd.notna(low_signal_returns) else 0

        return BacktestResult(
            indicators={},  # Individual indicator analysis could be added
            composite_accuracy=round(accuracy * 100, 1),
            composite_precision=round(precision * 100, 1),
            composite_recall=round(recall * 100, 1),
            event_detection=event_detection,
        )

    def print_report(self) -> None:
        """Print formatted backtest report."""
        result = self.run()

        if result is None:
            print("Insufficient data for backtest.")
            return

        print("\n" + "=" * 70)
        print("INDICATOR BACKTEST REPORT")
        print("=" * 70)

        print("\n--- Event Detection ---\n")
        print(f"{'Event':<25} {'Detected':<10} {'Peak Score':<12} {'Lead Days':<10}")
        print("-" * 60)

        detected_count = 0
        for event in result.event_detection:
            status = "Yes" if event["detected"] else "No"
            if event["detected"]:
                detected_count += 1
            print(f"{event['name']:<25} {status:<10} {event['peak_signal']:<12} {event['lead_days']:<10}")

        total_events = len(result.event_detection)
        print(f"\nDetection rate: {detected_count}/{total_events} events ({detected_count/total_events*100:.0f}%)")

        print("\n--- Classification Metrics ---\n")
        print(f"Accuracy:  {result.composite_accuracy:.1f}%")
        print(f"Precision: {result.composite_precision:.1f}% (of stress signals, % that were real)")
        print(f"Recall:    {result.composite_recall:.1f}% (of real stress, % we caught)")

        print("\n" + "=" * 70)


def main() -> None:
    """CLI entry point."""
    backtest = IndicatorBacktest()
    backtest.print_report()


if __name__ == "__main__":
    main()
