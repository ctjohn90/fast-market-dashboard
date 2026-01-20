"""Comparative backtest framework for linear vs non-linear models.

Tests each model variant against historical stress events and provides
metrics for comparison: detection rate, lead time, precision, false positives.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd
import numpy as np

from fast_market_dashboard.indicators.nonlinear import NonlinearCalculator, ModelType
from fast_market_dashboard.indicators.calculator import IndicatorCalculator
from fast_market_dashboard.config import Settings


# Historical stress events with metadata
STRESS_EVENTS = [
    {
        "name": "Financial Crisis",
        "start": date(2008, 9, 15),
        "end": date(2009, 3, 9),
        "peak_drawdown": 56.8,
        "severity": "extreme",
    },
    {
        "name": "Debt Ceiling 2011",
        "start": date(2011, 7, 22),
        "end": date(2011, 10, 3),
        "peak_drawdown": 19.4,
        "severity": "moderate",
    },
    {
        "name": "China Deval 2015",
        "start": date(2015, 8, 18),
        "end": date(2015, 9, 29),
        "peak_drawdown": 12.4,
        "severity": "mild",
    },
    {
        "name": "Fed Tightening 2018",
        "start": date(2018, 10, 3),
        "end": date(2018, 12, 24),
        "peak_drawdown": 19.8,
        "severity": "moderate",
    },
    {
        "name": "COVID Crash",
        "start": date(2020, 2, 19),
        "end": date(2020, 3, 23),
        "peak_drawdown": 33.9,
        "severity": "severe",
    },
    {
        "name": "Rate Shock 2022",
        "start": date(2022, 1, 3),
        "end": date(2022, 10, 12),
        "peak_drawdown": 25.4,
        "severity": "moderate",
    },
]


@dataclass
class EventResult:
    """Results for a single model on a single event."""
    
    event_name: str
    detected: bool  # Did score exceed threshold during event?
    peak_score: float  # Maximum score during event
    lead_days: int  # Days before event start that score exceeded threshold
    time_to_peak: int  # Days from event start to peak score


@dataclass 
class ModelResult:
    """Aggregate results for a single model across all events."""
    
    model_name: str
    events: list[EventResult]
    detection_rate: float  # % of events detected
    avg_lead_days: float  # Average early warning days
    avg_peak_score: float  # Average peak during events
    precision: float  # % of high-signal days that were actual stress
    false_positive_rate: float  # % of calm days with high signal
    signal_ratio: float  # Avg signal during stress / avg during calm
    score_series: pd.Series  # Full historical scores


class ModelBacktest:
    """Compare model performance across historical stress events."""

    SIGNAL_THRESHOLD = 70  # Score above this = elevated
    LEAD_WINDOW = 10  # Days before event to check for early warning
    
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.calculator = IndicatorCalculator(self.settings)
        self.nonlinear = NonlinearCalculator(self.calculator)
        self._scores_df: pd.DataFrame | None = None

    def _get_scores(self) -> pd.DataFrame:
        """Generate historical scores for all models."""
        if self._scores_df is not None:
            return self._scores_df
        
        print("Generating historical scores for all models...")
        self._scores_df = self.nonlinear.generate_historical_scores(days=6000)
        print(f"Generated {len(self._scores_df)} days of scores")
        return self._scores_df

    def _is_stress_date(self, d) -> bool:
        """Check if date is during a stress event."""
        if hasattr(d, "date"):
            d = d.date()
        for event in STRESS_EVENTS:
            if event["start"] <= d <= event["end"]:
                return True
        return False

    def _events_in_range(self, start: date, end: date) -> list[dict]:
        """Get events within data range."""
        return [
            e for e in STRESS_EVENTS
            if e["start"] >= start and e["end"] <= end
        ]

    def _evaluate_event(
        self, 
        scores: pd.Series, 
        event: dict,
    ) -> EventResult:
        """Evaluate model performance for a single event."""
        event_start = pd.Timestamp(event["start"])
        event_end = pd.Timestamp(event["end"])
        
        # Get scores during event
        event_scores = scores[
            (scores.index >= event_start) & 
            (scores.index <= event_end)
        ]
        
        if event_scores.empty:
            return EventResult(
                event_name=event["name"],
                detected=False,
                peak_score=0,
                lead_days=0,
                time_to_peak=0,
            )
        
        peak_score = event_scores.max()
        detected = peak_score >= self.SIGNAL_THRESHOLD
        
        # Time to peak (days from event start)
        peak_date = event_scores.idxmax()
        time_to_peak = (peak_date - event_start).days
        
        # Lead days: how many days before event did signal exceed threshold?
        pre_event = scores[
            (scores.index >= event_start - pd.Timedelta(days=self.LEAD_WINDOW)) &
            (scores.index < event_start)
        ]
        lead_signals = pre_event[pre_event >= self.SIGNAL_THRESHOLD]
        lead_days = len(lead_signals)
        
        return EventResult(
            event_name=event["name"],
            detected=detected,
            peak_score=round(peak_score, 1),
            lead_days=lead_days,
            time_to_peak=time_to_peak,
        )

    def _evaluate_model(self, model_name: str, scores: pd.Series) -> ModelResult:
        """Evaluate a single model across all events."""
        if scores.empty:
            return ModelResult(
                model_name=model_name,
                events=[],
                detection_rate=0,
                avg_lead_days=0,
                avg_peak_score=0,
                precision=0,
                false_positive_rate=0,
                signal_ratio=1.0,
                score_series=scores,
            )
        
        # Get data range
        data_start = scores.index[0].date()
        data_end = scores.index[-1].date()
        events_in_range = self._events_in_range(data_start, data_end)
        
        if not events_in_range:
            return ModelResult(
                model_name=model_name,
                events=[],
                detection_rate=0,
                avg_lead_days=0,
                avg_peak_score=0,
                precision=0,
                false_positive_rate=0,
                signal_ratio=1.0,
                score_series=scores,
            )
        
        # Evaluate each event
        event_results = [
            self._evaluate_event(scores, event) 
            for event in events_in_range
        ]
        
        detected_count = sum(1 for e in event_results if e.detected)
        detection_rate = detected_count / len(event_results) if event_results else 0
        
        avg_lead = np.mean([e.lead_days for e in event_results]) if event_results else 0
        avg_peak = np.mean([e.peak_score for e in event_results]) if event_results else 0
        
        # Precision: of all high-signal days, what % were actual stress?
        high_signal_days = scores >= self.SIGNAL_THRESHOLD
        is_stress = pd.Series(
            [self._is_stress_date(d) for d in scores.index],
            index=scores.index,
        )
        
        true_positives = (high_signal_days & is_stress).sum()
        false_positives = (high_signal_days & ~is_stress).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        # False positive rate: of calm days, what % had high signal?
        calm_days = ~is_stress
        fp_rate = false_positives / calm_days.sum() if calm_days.sum() > 0 else 0
        
        # Signal ratio: average during stress / average during calm
        stress_avg = scores[is_stress].mean() if is_stress.sum() > 0 else 50
        calm_avg = scores[~is_stress].mean() if (~is_stress).sum() > 0 else 50
        signal_ratio = stress_avg / calm_avg if calm_avg > 0 else 1.0
        
        return ModelResult(
            model_name=model_name,
            events=event_results,
            detection_rate=round(detection_rate * 100, 1),
            avg_lead_days=round(avg_lead, 1),
            avg_peak_score=round(avg_peak, 1),
            precision=round(precision * 100, 1),
            false_positive_rate=round(fp_rate * 100, 2),
            signal_ratio=round(signal_ratio, 2),
            score_series=scores,
        )

    def run(self) -> dict[str, ModelResult]:
        """Run backtest for all models."""
        scores_df = self._get_scores()
        
        if scores_df.empty:
            print("No historical scores available.")
            return {}
        
        results = {}
        for model_type in ModelType:
            model_name = model_type.value
            if model_name in scores_df.columns:
                scores = scores_df[model_name]
                results[model_name] = self._evaluate_model(model_name, scores)
        
        return results

    def print_report(self) -> None:
        """Print formatted comparison report."""
        results = self.run()
        
        if not results:
            print("No results available.")
            return
        
        print("\n" + "=" * 90)
        print("MODEL COMPARISON BACKTEST REPORT")
        print("=" * 90)
        
        # Summary table
        print("\n--- Model Performance Summary ---\n")
        print(f"{'Model':<15} {'Detect%':<10} {'Lead Days':<12} {'Precision':<12} {'FP Rate':<10} {'Signal Ratio':<12}")
        print("-" * 75)
        
        for model_name, result in sorted(results.items(), key=lambda x: x[1].signal_ratio, reverse=True):
            print(
                f"{model_name:<15} "
                f"{result.detection_rate:>6.1f}%   "
                f"{result.avg_lead_days:>8.1f}     "
                f"{result.precision:>8.1f}%    "
                f"{result.false_positive_rate:>6.2f}%   "
                f"{result.signal_ratio:>8.2f}x"
            )
        
        # Event-by-event breakdown
        print("\n--- Event Detection by Model ---\n")
        
        # Get first result to get event names
        first_result = list(results.values())[0]
        if not first_result.events:
            print("No events in data range.")
            return
        
        # Header
        header = f"{'Event':<22}"
        for model_name in results.keys():
            header += f" {model_name[:10]:<12}"
        print(header)
        print("-" * (22 + 13 * len(results)))
        
        # Each event
        for i, event_result in enumerate(first_result.events):
            row = f"{event_result.event_name:<22}"
            for model_name, model_result in results.items():
                if i < len(model_result.events):
                    er = model_result.events[i]
                    status = f"{er.peak_score:.0f}" if er.detected else "MISS"
                    lead = f"(+{er.lead_days}d)" if er.lead_days > 0 else ""
                    row += f" {status:>6}{lead:<6}"
                else:
                    row += " N/A         "
            print(row)
        
        # Key findings
        print("\n--- Key Findings ---\n")
        
        # Best detector
        best_detect = max(results.values(), key=lambda x: x.detection_rate)
        print(f"Best Detection Rate: {best_detect.model_name} ({best_detect.detection_rate}%)")
        
        # Best signal ratio
        best_ratio = max(results.values(), key=lambda x: x.signal_ratio)
        print(f"Best Signal Ratio: {best_ratio.model_name} ({best_ratio.signal_ratio}x)")
        
        # Best precision
        best_precision = max(results.values(), key=lambda x: x.precision)
        print(f"Best Precision: {best_precision.model_name} ({best_precision.precision}%)")
        
        # Most early warning
        best_lead = max(results.values(), key=lambda x: x.avg_lead_days)
        print(f"Most Lead Time: {best_lead.model_name} ({best_lead.avg_lead_days} days avg)")
        
        # Improvement over linear
        linear_result = results.get("linear")
        hybrid_result = results.get("hybrid")
        
        if linear_result and hybrid_result:
            print("\n--- Hybrid vs Linear Improvement ---\n")
            
            ratio_diff = hybrid_result.signal_ratio - linear_result.signal_ratio
            precision_diff = hybrid_result.precision - linear_result.precision
            lead_diff = hybrid_result.avg_lead_days - linear_result.avg_lead_days
            
            print(f"Signal Ratio: {ratio_diff:+.2f}x ({linear_result.signal_ratio:.2f} -> {hybrid_result.signal_ratio:.2f})")
            print(f"Precision: {precision_diff:+.1f}% ({linear_result.precision:.1f}% -> {hybrid_result.precision:.1f}%)")
            print(f"Lead Days: {lead_diff:+.1f} ({linear_result.avg_lead_days:.1f} -> {hybrid_result.avg_lead_days:.1f})")
        
        print("\n" + "=" * 90)

    def get_comparison_dataframe(self) -> pd.DataFrame:
        """Get results as DataFrame for dashboard display."""
        results = self.run()
        
        data = []
        for model_name, result in results.items():
            data.append({
                "Model": model_name.title(),
                "Detection Rate": f"{result.detection_rate:.0f}%",
                "Avg Lead Days": f"{result.avg_lead_days:.1f}",
                "Precision": f"{result.precision:.0f}%",
                "False Positive Rate": f"{result.false_positive_rate:.2f}%",
                "Signal Ratio": f"{result.signal_ratio:.2f}x",
            })
        
        df = pd.DataFrame(data)
        # Sort by signal ratio descending
        df = df.sort_values("Signal Ratio", ascending=False)
        return df


def main() -> None:
    """CLI entry point."""
    backtest = ModelBacktest()
    backtest.print_report()


if __name__ == "__main__":
    main()
