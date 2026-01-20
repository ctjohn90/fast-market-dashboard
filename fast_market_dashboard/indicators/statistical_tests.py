"""Statistical validation tests for the composite model."""

from datetime import date
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from fast_market_dashboard.data.cache import DataCache
from fast_market_dashboard.config import Settings
from fast_market_dashboard.indicators.calculator import WEIGHTS


# Known stress events for testing
STRESS_EVENTS = [
    {"name": "Financial Crisis", "start": date(2008, 9, 15), "end": date(2009, 3, 9)},
    {"name": "Debt Ceiling 2011", "start": date(2011, 7, 22), "end": date(2011, 10, 3)},
    {"name": "China Deval 2015", "start": date(2015, 8, 18), "end": date(2015, 9, 29)},
    {"name": "Fed Tightening 2018", "start": date(2018, 10, 3), "end": date(2018, 12, 24)},
    {"name": "COVID Crash 2020", "start": date(2020, 2, 19), "end": date(2020, 3, 23)},
    {"name": "Rate Shock 2022", "start": date(2022, 1, 3), "end": date(2022, 10, 12)},
    {"name": "Tariff Shock 2025", "start": date(2025, 4, 2), "end": date(2025, 4, 11)},
]


@dataclass
class VIFResult:
    """Variance Inflation Factor result for one indicator."""
    indicator: str
    vif: float
    interpretation: str


@dataclass 
class OutOfSampleResult:
    """Out-of-sample test result."""
    train_period: str
    test_period: str
    train_detection_rate: float
    test_detection_rate: float
    interpretation: str


class StatisticalValidator:
    """Run statistical validation tests on the composite model."""
    
    LOOKBACK = 252
    
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.cache = DataCache(self.settings.db_path)
    
    def _get_indicator_data(self) -> pd.DataFrame:
        """Get all indicator series aligned by date."""
        series_map = {
            "hy_credit": "BAMLH0A0HYM2",
            "ig_credit": "BAMLC0A0CM",
            # BBB removed - VIF > 100 showed redundancy with IG
            "volatility": "VIXCLS",
            "usd_flight": "DTWEXBGS",
            "equity_drawdown": "SP500",
        }
        
        data = {}
        for name, series_id in series_map.items():
            df = self.cache.get_series(series_id)
            if not df.empty:
                data[name] = df["value"]
        
        if not data:
            return pd.DataFrame()
        
        combined = pd.DataFrame(data).dropna()
        return combined
    
    def _calculate_percentiles(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert raw values to rolling percentile scores."""
        def pct_rank(series: pd.Series) -> pd.Series:
            def calc(s):
                if len(s) < 20:
                    return 50.0
                current = s.iloc[-1]
                return (s < current).sum() / (len(s) - 1) * 100
            return series.rolling(self.LOOKBACK, min_periods=20).apply(calc, raw=False)
        
        percentiles = pd.DataFrame()
        for col in data.columns:
            percentiles[col] = pct_rank(data[col])
        
        return percentiles.dropna()
    
    def _is_stress_period(self, dt, events: list[dict]) -> bool:
        """Check if date falls within any stress event."""
        if hasattr(dt, 'date'):
            dt = dt.date()
        for event in events:
            if event["start"] <= dt <= event["end"]:
                return True
        return False
    
    # =========================================================================
    # TEST 1: Variance Inflation Factor (Multicollinearity)
    # =========================================================================
    
    def run_vif_analysis(self) -> list[VIFResult]:
        """
        Calculate Variance Inflation Factor for each indicator.
        
        VIF measures how much an indicator is explained by other indicators.
        - VIF = 1: No correlation with others (good)
        - VIF = 1-5: Moderate correlation (acceptable)
        - VIF > 5: High correlation (concerning)
        - VIF > 10: Severe multicollinearity (redundant indicator)
        
        Plain English: If VIF is high, the indicator is telling us the same
        thing as other indicators - we might be double-counting.
        """
        data = self._get_indicator_data()
        if data.empty or len(data.columns) < 2:
            return []
        
        percentiles = self._calculate_percentiles(data)
        
        results = []
        for i, col in enumerate(percentiles.columns):
            # VIF = 1 / (1 - R²) where R² is from regressing col on all others
            y = percentiles[col]
            X = percentiles.drop(columns=[col])
            
            # Add constant for regression
            X_with_const = np.column_stack([np.ones(len(X)), X.values])
            
            try:
                # OLS regression
                coeffs, residuals, rank, s = np.linalg.lstsq(X_with_const, y.values, rcond=None)
                
                # Calculate R²
                y_pred = X_with_const @ coeffs
                ss_res = np.sum((y.values - y_pred) ** 2)
                ss_tot = np.sum((y.values - y.mean()) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # VIF
                vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
                
                # Interpretation
                if vif < 2:
                    interp = "Low correlation - unique signal"
                elif vif < 5:
                    interp = "Moderate correlation - acceptable"
                elif vif < 10:
                    interp = "High correlation - consider reducing weight"
                else:
                    interp = "SEVERE - likely redundant with other indicators"
                
                results.append(VIFResult(indicator=col, vif=round(vif, 2), interpretation=interp))
                
            except Exception as e:
                results.append(VIFResult(indicator=col, vif=0, interpretation=f"Error: {e}"))
        
        # Sort by VIF descending
        results.sort(key=lambda x: x.vif, reverse=True)
        return results
    
    # =========================================================================
    # TEST 2: Out-of-Sample Validation
    # =========================================================================
    
    def run_out_of_sample_test(self) -> OutOfSampleResult:
        """
        Test if the model works on data it wasn't trained on.
        
        Approach:
        - Training period: 2008-2020 (5 stress events)
        - Test period: 2022-2025 (2 stress events)
        
        Plain English: We pretend we built the model in 2020 and see if it
        would have caught the 2022 Rate Shock and 2025 Tariff Shock.
        """
        train_events = [e for e in STRESS_EVENTS if e["end"].year <= 2020]
        test_events = [e for e in STRESS_EVENTS if e["start"].year >= 2022]
        
        data = self._get_indicator_data()
        if data.empty:
            return OutOfSampleResult(
                train_period="N/A", test_period="N/A",
                train_detection_rate=0, test_detection_rate=0,
                interpretation="Insufficient data"
            )
        
        percentiles = self._calculate_percentiles(data)
        
        # Calculate composite score using current weights
        def calc_composite(row):
            score = 0
            total_weight = 0
            for ind, weight in WEIGHTS.items():
                if ind in row.index and not pd.isna(row[ind]):
                    score += row[ind] * weight
                    total_weight += weight
            return score / total_weight * sum(WEIGHTS.values()) if total_weight > 0 else 50
        
        percentiles["composite"] = percentiles.apply(calc_composite, axis=1)
        
        # Check detection for each event
        def check_detection(events, threshold=70):
            detected = 0
            for event in events:
                event_data = percentiles[
                    (percentiles.index >= pd.Timestamp(event["start"])) &
                    (percentiles.index <= pd.Timestamp(event["end"]))
                ]
                if not event_data.empty and event_data["composite"].max() >= threshold:
                    detected += 1
            return detected / len(events) if events else 0
        
        train_rate = check_detection(train_events)
        test_rate = check_detection(test_events)
        
        # Interpretation
        if test_rate >= 0.8:
            interp = "STRONG - Model generalizes well to unseen crises"
        elif test_rate >= 0.5:
            interp = "MODERATE - Model partially generalizes"
        else:
            interp = "WEAK - Model may be overfit to training data"
        
        return OutOfSampleResult(
            train_period="2008-2020 (5 events)",
            test_period="2022-2025 (2 events)",
            train_detection_rate=round(train_rate * 100, 1),
            test_detection_rate=round(test_rate * 100, 1),
            interpretation=interp
        )
    
    # =========================================================================
    # TEST 3: Granger Causality (Do indicators lead returns?)
    # =========================================================================
    
    def run_granger_test(self, max_lag: int = 5) -> dict:
        """
        Test if indicators 'Granger-cause' market returns.
        
        Granger causality: X Granger-causes Y if past values of X help predict Y
        beyond what past values of Y alone would predict.
        
        Plain English: Do credit spreads widening TODAY predict stock drops
        in the NEXT few days? If yes, the indicator has predictive value.
        """
        data = self._get_indicator_data()
        if data.empty or "equity_drawdown" not in data.columns:
            return {"error": "Insufficient data"}
        
        # Get S&P 500 returns
        sp500 = self.cache.get_series("SP500")
        if sp500.empty:
            return {"error": "No S&P 500 data"}
        
        returns = sp500["value"].pct_change().dropna()
        
        results = {}
        indicators_to_test = ["hy_credit", "ig_credit", "volatility", "usd_flight"]
        
        for ind in indicators_to_test:
            if ind not in data.columns:
                continue
            
            # Align data
            aligned = pd.concat([returns, data[ind]], axis=1, keys=["returns", ind]).dropna()
            if len(aligned) < 500:
                continue
            
            # Simple Granger test using lagged regression
            # H0: Lagged indicator does NOT help predict returns
            # H1: Lagged indicator DOES help predict returns
            
            y = aligned["returns"].iloc[max_lag:]
            X_base = np.column_stack([aligned["returns"].shift(i).iloc[max_lag:] for i in range(1, max_lag + 1)])
            X_full = np.column_stack([X_base] + [aligned[ind].shift(i).iloc[max_lag:] for i in range(1, max_lag + 1)])
            
            # Remove any remaining NaN
            valid = ~(np.isnan(X_full).any(axis=1) | np.isnan(y.values))
            y_clean = y.values[valid]
            X_base_clean = X_base[valid]
            X_full_clean = X_full[valid]
            
            if len(y_clean) < 100:
                continue
            
            # Fit restricted model (returns only)
            X_base_const = np.column_stack([np.ones(len(X_base_clean)), X_base_clean])
            try:
                coeffs_r, _, _, _ = np.linalg.lstsq(X_base_const, y_clean, rcond=None)
                ss_res_r = np.sum((y_clean - X_base_const @ coeffs_r) ** 2)
            except:
                continue
            
            # Fit unrestricted model (returns + indicator)
            X_full_const = np.column_stack([np.ones(len(X_full_clean)), X_full_clean])
            try:
                coeffs_u, _, _, _ = np.linalg.lstsq(X_full_const, y_clean, rcond=None)
                ss_res_u = np.sum((y_clean - X_full_const @ coeffs_u) ** 2)
            except:
                continue
            
            # F-test
            n = len(y_clean)
            k_r = X_base_const.shape[1]
            k_u = X_full_const.shape[1]
            
            if ss_res_u > 0 and k_u > k_r:
                f_stat = ((ss_res_r - ss_res_u) / (k_u - k_r)) / (ss_res_u / (n - k_u))
                p_value = 1 - stats.f.cdf(f_stat, k_u - k_r, n - k_u)
                
                if p_value < 0.01:
                    interp = "STRONG evidence of predictive power"
                elif p_value < 0.05:
                    interp = "Moderate evidence of predictive power"
                elif p_value < 0.10:
                    interp = "Weak evidence of predictive power"
                else:
                    interp = "No significant predictive power"
                
                results[ind] = {
                    "f_statistic": round(f_stat, 2),
                    "p_value": round(p_value, 4),
                    "interpretation": interp
                }
        
        return results
    
    # =========================================================================
    # TEST 4: Composite vs VIX-Only Comparison
    # =========================================================================
    
    def compare_composite_vs_vix(self) -> dict:
        """
        Does our composite add value beyond just using VIX?
        
        Compare detection rates:
        - Composite > 70
        - VIX percentile > 70
        
        Plain English: Is all this complexity worth it, or could we just
        watch VIX and get the same result?
        """
        data = self._get_indicator_data()
        if data.empty:
            return {"error": "Insufficient data"}
        
        percentiles = self._calculate_percentiles(data)
        
        # Calculate composite
        def calc_composite(row):
            score = 0
            total_weight = 0
            for ind, weight in WEIGHTS.items():
                if ind in row.index and not pd.isna(row[ind]):
                    score += row[ind] * weight
                    total_weight += weight
            return score / total_weight * sum(WEIGHTS.values()) if total_weight > 0 else 50
        
        percentiles["composite"] = percentiles.apply(calc_composite, axis=1)
        
        # Check both methods against all events
        composite_detected = 0
        vix_detected = 0
        
        for event in STRESS_EVENTS:
            event_data = percentiles[
                (percentiles.index >= pd.Timestamp(event["start"])) &
                (percentiles.index <= pd.Timestamp(event["end"]))
            ]
            
            if event_data.empty:
                continue
            
            if event_data["composite"].max() >= 70:
                composite_detected += 1
            
            if "volatility" in event_data.columns and event_data["volatility"].max() >= 70:
                vix_detected += 1
        
        total_events = len(STRESS_EVENTS)
        
        composite_rate = composite_detected / total_events * 100
        vix_rate = vix_detected / total_events * 100
        
        if composite_rate > vix_rate:
            interp = f"Composite BEATS VIX-only by {composite_rate - vix_rate:.0f} percentage points"
        elif composite_rate == vix_rate:
            interp = "Composite and VIX-only perform equally"
        else:
            interp = f"VIX-only BEATS composite by {vix_rate - composite_rate:.0f} percentage points"
        
        return {
            "composite_detection_rate": f"{composite_rate:.0f}%",
            "vix_only_detection_rate": f"{vix_rate:.0f}%",
            "interpretation": interp
        }
    
    # =========================================================================
    # RUN ALL TESTS
    # =========================================================================
    
    def run_all_tests(self) -> None:
        """Run all statistical tests and print results."""
        
        print("\n" + "=" * 70)
        print("STATISTICAL VALIDATION REPORT")
        print("=" * 70)
        
        # Test 1: VIF
        print("\n" + "-" * 70)
        print("TEST 1: MULTICOLLINEARITY (VIF)")
        print("-" * 70)
        print("Question: Are any indicators redundant (measuring the same thing)?")
        print("Rule: VIF > 5 = concerning, VIF > 10 = likely redundant\n")
        
        vif_results = self.run_vif_analysis()
        if vif_results:
            print(f"{'Indicator':<20} {'VIF':>8}  {'Assessment'}")
            print("-" * 60)
            for r in vif_results:
                print(f"{r.indicator:<20} {r.vif:>8.2f}  {r.interpretation}")
        else:
            print("Could not calculate VIF - insufficient data")
        
        # Test 2: Out-of-sample
        print("\n" + "-" * 70)
        print("TEST 2: OUT-OF-SAMPLE VALIDATION")
        print("-" * 70)
        print("Question: Does the model work on crises it wasn't trained on?")
        print("Method: Train on 2008-2020, test on 2022-2025\n")
        
        oos_result = self.run_out_of_sample_test()
        print(f"Training period: {oos_result.train_period}")
        print(f"Training detection rate: {oos_result.train_detection_rate}%")
        print(f"\nTest period: {oos_result.test_period}")
        print(f"Test detection rate: {oos_result.test_detection_rate}%")
        print(f"\nVerdict: {oos_result.interpretation}")
        
        # Test 3: Granger causality
        print("\n" + "-" * 70)
        print("TEST 3: GRANGER CAUSALITY")
        print("-" * 70)
        print("Question: Do indicators predict future returns (lead the market)?")
        print("Rule: p-value < 0.05 = statistically significant\n")
        
        granger_results = self.run_granger_test()
        if "error" not in granger_results:
            print(f"{'Indicator':<15} {'F-stat':>8} {'p-value':>10}  {'Assessment'}")
            print("-" * 60)
            for ind, res in granger_results.items():
                print(f"{ind:<15} {res['f_statistic']:>8.2f} {res['p_value']:>10.4f}  {res['interpretation']}")
        else:
            print(f"Could not run test: {granger_results['error']}")
        
        # Test 4: Composite vs VIX
        print("\n" + "-" * 70)
        print("TEST 4: COMPOSITE VS VIX-ONLY")
        print("-" * 70)
        print("Question: Is the composite better than just watching VIX?\n")
        
        comparison = self.compare_composite_vs_vix()
        if "error" not in comparison:
            print(f"Composite detection rate: {comparison['composite_detection_rate']}")
            print(f"VIX-only detection rate:  {comparison['vix_only_detection_rate']}")
            print(f"\nVerdict: {comparison['interpretation']}")
        else:
            print(f"Could not run comparison: {comparison['error']}")
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY & RECOMMENDATIONS")
        print("=" * 70)


def main():
    """Run statistical validation."""
    validator = StatisticalValidator()
    validator.run_all_tests()


if __name__ == "__main__":
    main()
