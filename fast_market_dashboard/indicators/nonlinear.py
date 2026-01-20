"""Non-linear composite score models for fast market detection.

Implements multiple non-linear approaches based on market microstructure intuition:
1. Convexity: High readings have disproportionate impact
2. Convergence: Multiple elevated indicators compound
3. Regime-switching: Different behavior in calm vs. stress
4. Lead-lag: Credit leads, VIX confirms, drawdown lags
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum

import pandas as pd
import numpy as np

from fast_market_dashboard.indicators.calculator import (
    IndicatorCalculator,
    CompositeResult,
    WEIGHTS,
)


class ModelType(Enum):
    """Available composite score models."""
    LINEAR = "linear"  # Current baseline
    CONVEX = "convex"  # Power transformation
    CONVERGENCE = "convergence"  # Elevated count multiplier
    REGIME = "regime"  # Different weights by regime
    HYBRID = "hybrid"  # Convex + convergence + lead-lag


@dataclass
class ModelComparison:
    """Results from comparing models."""
    
    model_name: str
    composite_series: pd.Series
    events_detected: int
    avg_lead_days: float  # Days signal exceeded 70 before event start
    precision: float  # % of high signals that were real stress
    false_positive_days: int  # Days > 70 outside stress events
    avg_stress_signal: float  # Average score during stress events
    avg_calm_signal: float  # Average score during calm periods
    signal_ratio: float  # stress / calm


class NonlinearCalculator:
    """Calculate non-linear composite scores."""

    # Thresholds for regime detection
    ELEVATED_THRESHOLD = 70
    HIGH_THRESHOLD = 85
    CALM_CEILING = 40
    
    # Power for convex transformation
    CONVEX_POWER = 1.4
    
    # Convergence multiplier parameters
    CONVERGENCE_BASE = 2  # Start multiplying after this many elevated
    CONVERGENCE_FACTOR = 0.08  # Per additional elevated indicator

    def __init__(self, calculator: IndicatorCalculator | None = None):
        self.calculator = calculator or IndicatorCalculator()
        self._indicator_history: pd.DataFrame | None = None

    def _get_indicator_history(self, days: int = 5000) -> pd.DataFrame:
        """Get historical percentiles for all indicators."""
        if self._indicator_history is not None:
            return self._indicator_history
        
        # Get composite history which includes individual indicators
        history = self.calculator.get_history(days=days)
        self._indicator_history = history
        return history

    def _convex_transform(self, pct: float, power: float = None) -> float:
        """
        Apply convex (power) transformation to percentile.
        
        Intuition: At extremes, each additional percentile point matters more.
        A move from 85→95 is more significant than 45→55.
        
        power > 1: amplifies high values, dampens low values
        power = 1.4: 50 → 43, 70 → 62, 85 → 80, 95 → 93
        """
        if power is None:
            power = self.CONVEX_POWER
        # Handle edge cases (negative or NaN values)
        if pct is None or pct < 0:
            return 50.0
        pct = min(100, max(0, pct))  # Clamp to 0-100
        return (pct / 100) ** power * 100

    def _convergence_multiplier(self, indicator_pcts: dict[str, float]) -> float:
        """
        Calculate multiplier based on how many indicators are elevated.
        
        Intuition: When multiple risk signals fire together, it's not additive.
        VIX + credit + USD all spiking = systemic stress, not coincidence.
        Diversification fails, correlations spike, forced selling begins.
        
        Returns multiplier (1.0 = no adjustment, >1.0 = amplified)
        """
        elevated_count = sum(
            1 for pct in indicator_pcts.values() 
            if pct >= self.ELEVATED_THRESHOLD
        )
        high_count = sum(
            1 for pct in indicator_pcts.values() 
            if pct >= self.HIGH_THRESHOLD
        )
        
        if elevated_count <= self.CONVERGENCE_BASE:
            return 1.0
        
        # Each additional elevated indicator adds to multiplier
        # High signals add extra
        extra_elevated = elevated_count - self.CONVERGENCE_BASE
        multiplier = 1.0 + (extra_elevated * self.CONVERGENCE_FACTOR)
        multiplier += high_count * 0.03  # Bonus for very high readings
        
        return min(multiplier, 1.5)  # Cap at 1.5x

    def _regime_weights(self, base_score: float) -> dict[str, float]:
        """
        Return different weights based on current regime.
        
        Intuition: In calm markets, emphasize leading indicators (credit).
        In stress, weight confirming indicators (VIX, correlation) more.
        
        CFA logic:
        - Credit markets are more informed (institutional, less retail noise)
        - VIX is reactive but reliable once stress begins
        - Drawdown is lagging (confirms but doesn't predict)
        """
        if base_score < 35:
            # Calm regime: emphasize early warning signals
            return {
                "hy_credit": 0.25,       # Credit leads
                "bbb_credit": 0.20,      
                "vix_term_structure": 0.15,  # Term structure flips early
                "volatility": 0.10,      # VIX less useful in calm
                "defensive_rotation": 0.10,
                "usd_flight": 0.08,
                "sector_correlation": 0.07,
                "safe_haven": 0.03,
                "equity_drawdown": 0.02,  # Lagging, low weight
            }
        elif base_score < 60:
            # Transition regime: balanced approach
            return WEIGHTS  # Use default weights
        else:
            # Stress regime: confirming indicators matter more
            return {
                "hy_credit": 0.18,
                "bbb_credit": 0.12,
                "volatility": 0.18,      # VIX reliable in stress
                "sector_correlation": 0.15,  # Correlation breakdown key
                "vix_term_structure": 0.12,
                "defensive_rotation": 0.08,
                "usd_flight": 0.08,
                "safe_haven": 0.05,
                "equity_drawdown": 0.04,
            }

    def _lead_lag_adjustment(
        self, 
        current_pcts: dict[str, float],
        history: pd.DataFrame,
        lookback: int = 5,
    ) -> float:
        """
        Adjust score based on lead-lag relationships.
        
        Intuition: Credit spike followed by VIX spike = confirmed stress.
        If credit spiked 3-5 days ago and VIX is spiking now, add premium.
        
        This captures the institutional → retail contagion pattern.
        """
        if len(history) < lookback + 1:
            return 0.0
        
        adjustment = 0.0
        
        # Check if credit led (was elevated 3-5 days ago)
        credit_cols = [c for c in history.columns if "credit" in c.lower()]
        vix_col = "volatility" if "volatility" in history.columns else None
        
        if not credit_cols or vix_col is None:
            return 0.0
        
        # Average credit percentile 3-5 days ago
        past_credit = history[credit_cols].iloc[-lookback-1:-2].mean().mean()
        current_vix = current_pcts.get("volatility", 50)
        
        # If credit was elevated recently and VIX is spiking now: confirming pattern
        if past_credit >= 65 and current_vix >= 70:
            adjustment += 3.0  # Small boost for confirmed pattern
        
        # If VIX spiked first without credit: potentially false alarm
        past_vix = history[vix_col].iloc[-lookback-1:-2].mean() if vix_col in history.columns else 50
        current_credit = sum(current_pcts.get(c.replace("_pct", ""), 50) for c in credit_cols) / len(credit_cols)
        
        if past_vix >= 70 and current_credit < 60:
            adjustment -= 2.0  # Slight discount: VIX spike without credit confirmation
        
        return adjustment

    def calculate_linear(self, indicator_pcts: dict[str, float]) -> float:
        """Current baseline: simple weighted average."""
        score = sum(
            indicator_pcts.get(key, 50) * weight 
            for key, weight in WEIGHTS.items()
        )
        return min(100, max(0, score))

    def calculate_convex(self, indicator_pcts: dict[str, float]) -> float:
        """Apply convex transformation to each indicator before averaging."""
        transformed = {
            key: self._convex_transform(pct) 
            for key, pct in indicator_pcts.items()
        }
        score = sum(
            transformed.get(key, 50) * weight 
            for key, weight in WEIGHTS.items()
        )
        return min(100, max(0, score))

    def calculate_convergence(self, indicator_pcts: dict[str, float]) -> float:
        """Linear base with convergence multiplier."""
        base = self.calculate_linear(indicator_pcts)
        multiplier = self._convergence_multiplier(indicator_pcts)
        return min(100, base * multiplier)

    def calculate_regime(self, indicator_pcts: dict[str, float]) -> float:
        """Use different weights based on current regime."""
        # First pass: get rough regime estimate with default weights
        rough_score = self.calculate_linear(indicator_pcts)
        
        # Get regime-appropriate weights
        weights = self._regime_weights(rough_score)
        
        # Recalculate with regime weights
        score = sum(
            indicator_pcts.get(key, 50) * weight 
            for key, weight in weights.items()
        )
        return min(100, max(0, score))

    def calculate_hybrid(
        self, 
        indicator_pcts: dict[str, float],
        history: pd.DataFrame | None = None,
    ) -> float:
        """
        Full hybrid model combining all non-linear effects.
        
        1. Convex transformation (amplify extremes)
        2. Regime-based weights (context-aware)
        3. Convergence multiplier (compound risk)
        4. Lead-lag adjustment (confirmation bonus/penalty)
        """
        # Step 1: Convex transform each indicator
        transformed = {
            key: self._convex_transform(pct) 
            for key, pct in indicator_pcts.items()
        }
        
        # Step 2: Get rough score to determine regime
        rough_score = sum(
            transformed.get(key, 50) * weight 
            for key, weight in WEIGHTS.items()
        )
        
        # Step 3: Apply regime-specific weights
        weights = self._regime_weights(rough_score)
        base_score = sum(
            transformed.get(key, 50) * weight 
            for key, weight in weights.items()
        )
        
        # Step 4: Convergence multiplier (based on original percentiles, not transformed)
        multiplier = self._convergence_multiplier(indicator_pcts)
        score = base_score * multiplier
        
        # Step 5: Lead-lag adjustment if history available
        if history is not None and len(history) > 5:
            adjustment = self._lead_lag_adjustment(indicator_pcts, history)
            score += adjustment
        
        return min(100, max(0, score))

    def calculate_all_models(
        self, 
        indicator_pcts: dict[str, float],
        history: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Calculate composite score using all model variants."""
        return {
            ModelType.LINEAR.value: self.calculate_linear(indicator_pcts),
            ModelType.CONVEX.value: self.calculate_convex(indicator_pcts),
            ModelType.CONVERGENCE.value: self.calculate_convergence(indicator_pcts),
            ModelType.REGIME.value: self.calculate_regime(indicator_pcts),
            ModelType.HYBRID.value: self.calculate_hybrid(indicator_pcts, history),
        }

    def generate_historical_scores(self, days: int = 5000) -> pd.DataFrame:
        """
        Generate historical composite scores for all models.
        
        Returns DataFrame with date index and columns for each model.
        """
        history = self._get_indicator_history(days)
        if history.empty:
            return pd.DataFrame()
        
        # Identify indicator columns (exclude 'composite')
        indicator_cols = [c for c in history.columns if c != "composite"]
        
        results = []
        for i in range(len(history)):
            row = history.iloc[i]
            indicator_pcts = {col: row[col] for col in indicator_cols if pd.notna(row[col])}
            
            # For hybrid, pass history up to this point
            history_to_date = history.iloc[:i+1] if i > 5 else None
            
            scores = {
                "date": history.index[i],
                ModelType.LINEAR.value: self.calculate_linear(indicator_pcts),
                ModelType.CONVEX.value: self.calculate_convex(indicator_pcts),
                ModelType.CONVERGENCE.value: self.calculate_convergence(indicator_pcts),
                ModelType.REGIME.value: self.calculate_regime(indicator_pcts),
                ModelType.HYBRID.value: self.calculate_hybrid(indicator_pcts, history_to_date),
            }
            results.append(scores)
        
        df = pd.DataFrame(results)
        df.set_index("date", inplace=True)
        return df
