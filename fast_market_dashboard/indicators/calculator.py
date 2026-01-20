"""Calculate risk indicators from raw market data."""

from dataclasses import dataclass
from datetime import date

import pandas as pd

from fast_market_dashboard.data.cache import DataCache
from fast_market_dashboard.config import Settings


@dataclass
class IndicatorResult:
    """Result for a single indicator."""

    name: str
    series_id: str
    current_value: float
    percentile: float  # 0-100
    history: pd.Series  # Last 30 days of percentile scores


@dataclass
class CompositeResult:
    """Complete dashboard result."""

    as_of_date: date
    composite_score: float  # 0-100
    stress_level: str  # Calm, Normal, Elevated, Fast Market
    indicators: dict[str, IndicatorResult]


# Indicator weights for composite score
# Weights based on backtest signal ratios and detection rates
WEIGHTS = {
    # Tier 1: Credit spreads (best discriminators, 2.1-2.2x ratio)
    "hy_credit": 0.20,           # 2.19x ratio, 100% detection
    "bbb_credit": 0.15,          # 2.18x ratio, 100% detection
    # Tier 2: Volatility & Flight (1.7-1.9x ratio)
    "volatility": 0.15,          # 1.89x ratio, 100% detection
    "defensive_rotation": 0.10,  # 1.89x ratio, 100% detection (NEW)
    "usd_flight": 0.10,          # 1.73x ratio, 100% detection
    # Tier 3: Market structure (1.6-1.8x ratio)
    "sector_correlation": 0.10,  # 1.75x ratio, unique signal
    "vix_term_structure": 0.10,  # 1.63x ratio, panic indicator
    # Tier 4: Supporting indicators
    "safe_haven": 0.05,          # 1.38x ratio
    "equity_drawdown": 0.05,     # Direct measure
}

# Indicator metadata for dashboard descriptions
INDICATOR_INFO = {
    "hy_credit": {
        "name": "High Yield Credit Spread",
        "source": "FRED (BAMLH0A0HYM2)",
        "description": "The spread between high yield (junk) bonds and treasuries. Widens sharply during stress as investors demand more compensation for risk.",
        "interpretation": "Higher = more stress. During crises, spreads can exceed 10%. Normal range is 3-5%.",
        "signal_ratio": "2.19x",
        "detection_rate": "100% (6/6 events)",
        "weight_rationale": "Highest signal ratio. Credit markets often lead equities in signaling stress.",
    },
    "bbb_credit": {
        "name": "BBB Credit Spread",
        "source": "FRED (BAMLC0A4CBBB)",
        "description": "Spread for BBB-rated corporate bonds (lowest investment grade). Sensitive to downgrade fears.",
        "interpretation": "Higher = more stress. BBB is the largest segment of corporate debt.",
        "signal_ratio": "2.18x",
        "detection_rate": "100% (6/6 events)",
        "weight_rationale": "Second-highest signal ratio. Captures IG downgrade risk during stress.",
    },
    "volatility": {
        "name": "VIX (Volatility Index)",
        "source": "FRED (VIXCLS)",
        "description": "The 'fear gauge' - measures expected 30-day S&P 500 volatility implied by options prices.",
        "interpretation": "Higher = more fear. Above 30 is elevated, above 40 is panic.",
        "signal_ratio": "1.89x",
        "detection_rate": "100% (6/6 events)",
        "weight_rationale": "Industry-standard fear gauge. Fast-reacting and universally recognized.",
    },
    "defensive_rotation": {
        "name": "Defensive Rotation (XLU/XLY)",
        "source": "Yahoo Finance (derived)",
        "description": "Ratio of Utilities (defensive) to Consumer Discretionary (cyclical). Rising ratio indicates rotation to safety.",
        "interpretation": "Higher = more defensive positioning. Spikes during risk-off periods.",
        "signal_ratio": "1.89x",
        "detection_rate": "100%",
        "weight_rationale": "Captures sector rotation behavior distinct from VIX or credit.",
    },
    "usd_flight": {
        "name": "USD Index",
        "source": "FRED (DTWEXBGS)",
        "description": "Trade-weighted US dollar index. Dollar strengthens as global reserve currency during crises.",
        "interpretation": "Higher = flight to safety. Strong dollar can also create EM stress.",
        "signal_ratio": "1.73x",
        "detection_rate": "100% (6/6 events)",
        "weight_rationale": "Independent macro signal. Flight to USD is a global risk-off indicator.",
    },
    "sector_correlation": {
        "name": "Sector Correlation",
        "source": "Yahoo Finance (derived)",
        "description": "Average pairwise correlation of S&P sector ETFs (60-day rolling). High correlation means diversification breaks down.",
        "interpretation": "Higher = more stress. When all sectors move together, it signals panic selling.",
        "signal_ratio": "1.75x",
        "detection_rate": "100%",
        "weight_rationale": "Unique signal: diversification breakdown during panic.",
    },
    "vix_term_structure": {
        "name": "VIX Term Structure",
        "source": "Yahoo Finance (VIX/VIX3M)",
        "description": "Ratio of spot VIX to 3-month VIX. Normally in contango (>1). Backwardation (<1) signals panic.",
        "interpretation": "Below 1.0 = backwardation = panic. Below 0.9 = severe stress.",
        "signal_ratio": "1.63x",
        "detection_rate": "100%",
        "weight_rationale": "Complements VIX level. Backwardation signals extreme near-term fear.",
    },
    "safe_haven": {
        "name": "Safe Haven Demand (GLD/SPY)",
        "source": "Yahoo Finance (derived)",
        "description": "Ratio of gold to S&P 500. Rising ratio indicates flight to gold as safe haven.",
        "interpretation": "Higher = more fear. Gold outperforms equities during stress.",
        "signal_ratio": "1.38x",
        "detection_rate": "100%",
        "weight_rationale": "Lower weight due to weaker signal ratio. Gold not always a stress indicator.",
    },
    "equity_drawdown": {
        "name": "S&P 500 Drawdown",
        "source": "FRED (SP500)",
        "description": "Current drawdown from 52-week high. Direct measure of equity market stress.",
        "interpretation": "Higher = deeper drawdown. 10%+ is correction, 20%+ is bear market.",
        "signal_ratio": "N/A",
        "detection_rate": "100%",
        "weight_rationale": "Direct measure but lagging. Confirms stress rather than predicting it.",
    },
}


class IndicatorCalculator:
    """Transforms raw market data into risk indicators."""

    LOOKBACK_DAYS = 252  # 1 year for percentile calculation
    HISTORY_DAYS = 30  # Days of history for sparklines

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.cache = DataCache(self.settings.db_path)

    def _percentile_rank(self, series: pd.Series, window: int = LOOKBACK_DAYS) -> pd.Series:
        """
        Calculate rolling percentile rank.
        
        Returns 0-100 score: where does each value sit in its trailing window?
        """
        def calc_pct(s: pd.Series) -> float:
            if len(s) < 2:
                return 50.0
            current = s.iloc[-1]
            return float((s < current).sum() / (len(s) - 1) * 100)

        return series.rolling(window=window, min_periods=20).apply(calc_pct, raw=False)

    def _calculate_drawdown(self, prices: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate percentage drawdown from rolling high.
        
        Returns 0-100 score: 0 = at high, 100 = max historical drawdown
        """
        rolling_max = prices.rolling(window=window, min_periods=1).max()
        drawdown_pct = (rolling_max - prices) / rolling_max * 100
        
        # Normalize: use historical max drawdown as reference
        # S&P 500 typical max drawdown ~50%, so scale accordingly
        normalized = (drawdown_pct / 50 * 100).clip(0, 100)
        return normalized

    def _calculate_curve_stress(self, spread: pd.Series) -> pd.Series:
        """
        Convert yield curve spread to stress score.
        
        T10Y2Y: positive = normal, negative = inverted (stress)
        Returns 0-100: 0 = steep curve, 100 = deeply inverted
        """
        # Inversion stress: more negative = higher stress
        # Typical range: -1% (deep inversion) to +2.5% (steep curve)
        # Map to 0-100 where 0 = +2.5%, 100 = -1%
        stress = ((2.5 - spread) / 3.5 * 100).clip(0, 100)
        return stress

    def _get_indicator(
        self, series_id: str, name: str, transform: str = "percentile"
    ) -> IndicatorResult | None:
        """
        Calculate a single indicator from cached data.
        
        Args:
            series_id: FRED or derived series ID
            name: Human-readable indicator name
            transform: "percentile", "percentile_inverse", "drawdown", "curve", 
                       "vix_term", "correlation"
        """
        df = self.cache.get_series(series_id)
        if df.empty or len(df) < 20:
            return None

        values = df["value"]

        if transform == "percentile":
            scores = self._percentile_rank(values)
        elif transform == "percentile_inverse":
            # Low values = high stress (e.g., risk appetite, VIX term structure)
            scores = 100 - self._percentile_rank(values)
        elif transform == "drawdown":
            scores = self._calculate_drawdown(values)
        elif transform == "curve":
            scores = self._calculate_curve_stress(values)
        elif transform == "vix_term":
            # VIX term structure: < 1 = backwardation = stress
            # Map 0.7-1.3 range to 100-0 stress score
            stress = ((1.3 - values) / 0.6 * 100).clip(0, 100)
            scores = stress
        elif transform == "correlation":
            # Sector correlation: high = stress (everything moving together)
            # Map 0-0.8 range to 0-100
            scores = (values / 0.8 * 100).clip(0, 100)
        else:
            raise ValueError(f"Unknown transform: {transform}")

        if isinstance(scores, pd.Series):
            scores = scores.dropna()
        if scores.empty if isinstance(scores, pd.Series) else len(scores) == 0:
            return None

        return IndicatorResult(
            name=name,
            series_id=series_id,
            current_value=float(values.iloc[-1]),
            percentile=float(scores.iloc[-1]) if isinstance(scores, pd.Series) else float(scores[-1]),
            history=scores.tail(self.HISTORY_DAYS) if isinstance(scores, pd.Series) else scores[-self.HISTORY_DAYS:],
        )

    def calculate(self) -> CompositeResult | None:
        """
        Calculate all indicators and composite score.
        
        Returns None if insufficient data.
        """
        # Calculate individual indicators (ordered by reliability from backtest)
        indicators = {}

        # ===== FRED-based indicators (primary) =====
        hy = self._get_indicator("BAMLH0A0HYM2", "HY Credit Spread", "percentile")
        if hy:
            indicators["hy_credit"] = hy

        vix = self._get_indicator("VIXCLS", "VIX", "percentile")
        if vix:
            indicators["volatility"] = vix

        bbb = self._get_indicator("BAMLC0A4CBBB", "BBB Credit Spread", "percentile")
        if bbb:
            indicators["bbb_credit"] = bbb

        usd = self._get_indicator("DTWEXBGS", "USD Index", "percentile")
        if usd:
            indicators["usd_flight"] = usd

        # ===== Yahoo Finance derived indicators =====
        defensive = self._get_indicator("DEFENSIVE_ROTATION", "Defensive Rotation", "percentile")
        if defensive:
            indicators["defensive_rotation"] = defensive

        sector_corr = self._get_indicator("SECTOR_CORRELATION", "Sector Correlation", "correlation")
        if sector_corr:
            indicators["sector_correlation"] = sector_corr

        vix_ts = self._get_indicator("VIX_TERM_STRUCTURE", "VIX Term Structure", "vix_term")
        if vix_ts:
            indicators["vix_term_structure"] = vix_ts

        safe_haven = self._get_indicator("SAFE_HAVEN", "Safe Haven (GLD/SPY)", "percentile")
        if safe_haven:
            indicators["safe_haven"] = safe_haven

        # ===== Secondary indicators =====
        equity = self._get_indicator("SP500", "S&P 500 Drawdown", "drawdown")
        if equity:
            indicators["equity_drawdown"] = equity

        if not indicators:
            return None

        # Calculate composite score
        composite = 0.0
        total_weight = 0.0

        for key, weight in WEIGHTS.items():
            if key in indicators:
                composite += indicators[key].percentile * weight
                total_weight += weight

        if total_weight > 0:
            composite = composite / total_weight * (sum(WEIGHTS.values()))

        # Determine stress level
        if composite < 30:
            stress_level = "Calm"
        elif composite < 50:
            stress_level = "Normal"
        elif composite < 70:
            stress_level = "Elevated"
        else:
            stress_level = "Fast Market"

        # Determine as-of date from data
        dates = [
            ind.history.index[-1]
            for ind in indicators.values()
            if not ind.history.empty
        ]
        as_of = max(dates).date() if dates else date.today()

        return CompositeResult(
            as_of_date=as_of,
            composite_score=round(composite, 1),
            stress_level=stress_level,
            indicators=indicators,
        )

    def get_history(self, days: int = 90) -> pd.DataFrame:
        """
        Get composite score history for charting.
        
        Returns DataFrame with composite score over time.
        """
        # Get all series
        data = self.cache.get_all_series()
        if not data:
            return pd.DataFrame()

        # Build aligned DataFrame
        combined = pd.DataFrame()
        for series_id, df in data.items():
            if not df.empty:
                combined[series_id] = df["value"]

        if combined.empty:
            return pd.DataFrame()

        # Calculate daily composite scores
        results = []

        for i in range(self.LOOKBACK_DAYS, len(combined)):
            window = combined.iloc[max(0, i - self.LOOKBACK_DAYS):i + 1]
            row_date = combined.index[i]

            scores = {}

            # VIX percentile
            if "VIXCLS" in window.columns:
                vix = window["VIXCLS"].dropna()
                if len(vix) >= 20:
                    current = vix.iloc[-1]
                    scores["volatility"] = (vix < current).sum() / (len(vix) - 1) * 100

            # HY Credit percentile
            if "BAMLH0A0HYM2" in window.columns:
                hy = window["BAMLH0A0HYM2"].dropna()
                if len(hy) >= 20:
                    current = hy.iloc[-1]
                    scores["hy_credit"] = (hy < current).sum() / (len(hy) - 1) * 100

            # IG Credit percentile
            if "BAMLC0A0CM" in window.columns:
                ig = window["BAMLC0A0CM"].dropna()
                if len(ig) >= 20:
                    current = ig.iloc[-1]
                    scores["ig_credit"] = (ig < current).sum() / (len(ig) - 1) * 100

            # Curve stress
            if "T10Y2Y" in window.columns:
                spread = window["T10Y2Y"].dropna()
                if len(spread) >= 1:
                    current = spread.iloc[-1]
                    scores["curve_stress"] = ((2.5 - current) / 3.5 * 100)

            # Equity drawdown
            if "SP500" in window.columns:
                sp = window["SP500"].dropna()
                if len(sp) >= 1:
                    high = sp.max()
                    current = sp.iloc[-1]
                    dd = (high - current) / high * 100
                    scores["equity_drawdown"] = min(dd / 50 * 100, 100)

            # USD flight
            if "DTWEXBGS" in window.columns:
                usd = window["DTWEXBGS"].dropna()
                if len(usd) >= 20:
                    current = usd.iloc[-1]
                    scores["usd_flight"] = (usd < current).sum() / (len(usd) - 1) * 100

            # Composite
            if scores:
                composite = 0.0
                total_weight = 0.0
                for key, weight in WEIGHTS.items():
                    if key in scores:
                        composite += scores[key] * weight
                        total_weight += weight
                if total_weight > 0:
                    composite = composite / total_weight * sum(WEIGHTS.values())
                results.append({"date": row_date, "composite": composite})

        if not results:
            return pd.DataFrame()

        history = pd.DataFrame(results).set_index("date").tail(days)
        return history
