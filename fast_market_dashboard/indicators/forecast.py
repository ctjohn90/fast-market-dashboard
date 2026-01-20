"""
Neural Network Forecasting Model for S&P 500 Large Moves (>3%)

Uses t-1 indicator data to predict whether S&P 500 will move more than 3%
(either direction) at t=0.

This is a binary classification problem:
- Target: 1 if |return| > 3%, 0 otherwise
- Features: Lagged (t-1) values of all stress indicators
"""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from fast_market_dashboard.data.cache import DataCache
from fast_market_dashboard.config import Settings


@dataclass
class ForecastResult:
    """Result of a single forecast."""
    date: date
    probability: float  # Probability of >3% move
    prediction: int     # 1 = predict large move, 0 = no large move
    actual: Optional[int] = None  # Actual outcome (for backtesting)
    actual_return: Optional[float] = None  # Actual S&P return


@dataclass
class BacktestMetrics:
    """Backtesting performance metrics."""
    accuracy: float
    precision: float      # Of predicted large moves, how many were correct?
    recall: float         # Of actual large moves, how many did we catch?
    f1_score: float
    auc_roc: float
    total_predictions: int
    large_moves_caught: int
    large_moves_total: int
    false_alarms: int
    
    # Confusion matrix
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int


class LargeMoveForecaster:
    """
    Neural network model to predict S&P 500 moves > 3%.
    
    Uses a Multi-Layer Perceptron (MLP) classifier with:
    - Input: Lagged indicator values (t-1)
    - Output: Probability of >3% move at t=0
    
    Key design choices for rare event prediction:
    - Class weights to handle severe imbalance (~1% positive rate)
    - Lower prediction threshold (0.3 instead of 0.5)
    - Oversampling of rare events during training
    """
    
    THRESHOLD = 0.03  # 3% move threshold
    PREDICTION_THRESHOLD = 0.15  # Lower threshold for rare events (default 0.5)
    
    # Feature definitions: series_id -> (name, transform)
    # Transform: 'raw' = use raw value, 'pct' = percentile rank
    FEATURES = {
        # Core stress indicators (from FRED)
        "BAMLH0A0HYM2": ("hy_spread", "raw"),
        "BAMLC0A0CM": ("ig_spread", "raw"),
        "VIXCLS": ("vix", "raw"),
        "DTWEXBGS": ("usd", "raw"),
        "SP500": ("sp500_level", "raw"),
        # Derived indicators (from Yahoo)
        "VIX_TERM_STRUCTURE": ("vix_term", "raw"),
        "SECTOR_CORRELATION": ("sector_corr", "raw"),
        "DEFENSIVE_ROTATION": ("defensive", "raw"),
        "SAFE_HAVEN": ("safe_haven", "raw"),
        # NEW: Enhanced forecast features
        "VVIX_LEVEL": ("vvix", "raw"),                    # Volatility of VIX
        "BOND_VOLATILITY": ("bond_vol", "raw"),           # TLT vol (MOVE proxy)
        "SPY_REL_VOLUME": ("rel_volume", "raw"),          # Relative volume
        "CREDIT_APPETITE_MOM": ("credit_mom", "raw"),     # HYG/LQD momentum
        "SMALLCAP_APPETITE_MOM": ("smallcap_mom", "raw"), # IWM/SPY momentum
        "YEN_CARRY_SIGNAL": ("yen_carry", "raw"),         # Carry trade unwinding
        "OIL_STRESS": ("oil_stress", "raw"),              # Oil 5-day return
        "VIX_MOMENTUM": ("vix_mom", "raw"),               # VIX rate of change
        "SPREAD_ACCELERATION": ("spread_accel", "raw"),   # Credit spread 2nd derivative
    }
    
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.cache = DataCache(self.settings.db_path)
        
        # Model components
        self.model: Optional[MLPClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: list[str] = []
        self.is_trained = False
        
        # Training data
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
    
    def _oversample(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        target_ratio: float = 0.15
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Oversample minority class to handle extreme imbalance.
        
        Simple random oversampling with noise to avoid pure duplication.
        """
        X_arr = X.values
        y_arr = y.values
        
        pos_idx = np.where(y_arr == 1)[0]
        neg_idx = np.where(y_arr == 0)[0]
        
        if len(pos_idx) == 0:
            return X_arr, y_arr
        
        # Calculate how many positive samples we need
        n_neg = len(neg_idx)
        target_n_pos = int(n_neg * target_ratio / (1 - target_ratio))
        n_to_add = target_n_pos - len(pos_idx)
        
        if n_to_add <= 0:
            return X_arr, y_arr
        
        # Oversample with slight noise to avoid exact duplicates
        oversample_idx = np.random.choice(pos_idx, size=n_to_add, replace=True)
        X_oversample = X_arr[oversample_idx] + np.random.normal(0, 0.01, X_arr[oversample_idx].shape)
        y_oversample = y_arr[oversample_idx]
        
        X_balanced = np.vstack([X_arr, X_oversample])
        y_balanced = np.concatenate([y_arr, y_oversample])
        
        # Shuffle
        shuffle_idx = np.random.permutation(len(y_balanced))
        return X_balanced[shuffle_idx], y_balanced[shuffle_idx]
    
    def _prepare_features(self) -> pd.DataFrame:
        """
        Prepare feature matrix with lagged values.
        
        Returns DataFrame with:
        - Index: dates
        - Columns: lagged indicator values (t-1)
        - Also includes sp500_return for target creation
        """
        data = {}
        
        # Load all indicator series
        for series_id, (name, transform) in self.FEATURES.items():
            df = self.cache.get_series(series_id)
            if not df.empty:
                data[name] = df["value"]
        
        if not data:
            return pd.DataFrame()
        
        # Combine into single DataFrame
        combined = pd.DataFrame(data)
        combined = combined.dropna()
        
        if combined.empty or len(combined) < 252:
            return pd.DataFrame()
        
        # Calculate S&P 500 returns
        if "sp500_level" in combined.columns:
            combined["sp500_return"] = combined["sp500_level"].pct_change()
        else:
            return pd.DataFrame()
        
        # Create lagged features (t-1 values)
        feature_cols = [c for c in combined.columns if c != "sp500_return"]
        
        lagged = pd.DataFrame(index=combined.index)
        for col in feature_cols:
            lagged[f"{col}_t1"] = combined[col].shift(1)
        
        # Add rolling statistics as additional features
        lagged["vix_5d_ma"] = combined["vix"].rolling(5).mean().shift(1) if "vix" in combined.columns else np.nan
        lagged["vix_5d_std"] = combined["vix"].rolling(5).std().shift(1) if "vix" in combined.columns else np.nan
        lagged["hy_spread_5d_change"] = combined["hy_spread"].pct_change(5).shift(1) if "hy_spread" in combined.columns else np.nan
        lagged["sp500_5d_return"] = combined["sp500_level"].pct_change(5).shift(1) if "sp500_level" in combined.columns else np.nan
        lagged["sp500_20d_vol"] = combined["sp500_return"].rolling(20).std().shift(1) * np.sqrt(252)  # Annualized vol
        
        # NEW: Additional rolling statistics for enhanced features
        if "vvix" in combined.columns:
            lagged["vvix_5d_change"] = combined["vvix"].pct_change(5).shift(1)
            lagged["vvix_vs_vix"] = (combined["vvix"] / combined["vix"]).shift(1) if "vix" in combined.columns else np.nan
        
        if "bond_vol" in combined.columns:
            lagged["bond_vol_5d_change"] = combined["bond_vol"].pct_change(5).shift(1)
        
        if "rel_volume" in combined.columns:
            lagged["volume_surge"] = (combined["rel_volume"] > 2.0).astype(int).shift(1)  # Binary: volume > 2x average
        
        # Cross-asset stress indicator: count of elevated signals
        stress_cols = ["vix", "hy_spread", "vvix", "bond_vol"]
        available_stress = [c for c in stress_cols if c in combined.columns]
        if len(available_stress) >= 2:
            # Count how many stress indicators are above their 75th percentile
            stress_count = pd.DataFrame()
            for col in available_stress:
                pct75 = combined[col].rolling(252).quantile(0.75)
                stress_count[col] = (combined[col] > pct75).astype(int)
            lagged["stress_count"] = stress_count.sum(axis=1).shift(1)
        
        # Keep target (not lagged - this is what we're predicting)
        lagged["sp500_return"] = combined["sp500_return"]
        
        # Drop rows with NaN
        lagged = lagged.dropna()
        
        return lagged
    
    def _create_target(self, returns: pd.Series) -> pd.Series:
        """
        Create binary target: 1 if |return| > 3%, 0 otherwise.
        """
        return (returns.abs() > self.THRESHOLD).astype(int)
    
    def train(self, test_size: float = 0.2) -> dict:
        """
        Train the neural network model.
        
        Args:
            test_size: Fraction of data to hold out for testing
            
        Returns:
            Dict with training metrics
        """
        # Prepare data
        data = self._prepare_features()
        if data.empty:
            return {"error": "Insufficient data for training"}
        
        # Split features and target
        self.feature_names = [c for c in data.columns if c != "sp500_return"]
        X = data[self.feature_names]
        y = self._create_target(data["sp500_return"])
        
        # Time-series split (don't use future data!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Handle class imbalance: compute class weights
        # Large moves are ~1% of data, so weight them heavily
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        
        if n_pos > 0:
            # Weight positive class proportionally to its rarity
            class_weight_ratio = n_neg / n_pos
            # Cap it to avoid extreme weights
            class_weight_ratio = min(class_weight_ratio, 20)
        else:
            class_weight_ratio = 10
        
        # Oversample positive class for training
        X_train_balanced, y_train_balanced = self._oversample(
            pd.DataFrame(X_train_scaled, columns=self.feature_names),
            y_train.reset_index(drop=True),
            target_ratio=0.15  # Aim for 15% positive class
        )
        
        # Train neural network
        # Architecture tuned for small dataset with binary classification
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),  # 3 hidden layers
            activation='relu',
            solver='adam',
            alpha=0.01,  # Stronger L2 regularization for imbalanced data
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=42,
        )
        
        self.model.fit(X_train_balanced, y_train_balanced)
        self.is_trained = True
        
        # Store training data for reference
        self.X_train = X_train
        self.y_train = y_train
        
        # Evaluate on test set using lower threshold
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob >= self.PREDICTION_THRESHOLD).astype(int)
        
        metrics = {
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "large_moves_train": int(y_train.sum()),
            "large_moves_test": int(y_test.sum()),
            "prediction_threshold": self.PREDICTION_THRESHOLD,
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "auc_roc": round(roc_auc_score(y_test, y_prob), 4) if y_test.sum() > 0 else 0,
        }
        
        return metrics
    
    def predict(self, as_of_date: Optional[date] = None) -> Optional[ForecastResult]:
        """
        Generate forecast for a specific date.
        
        Args:
            as_of_date: Date to forecast (default: latest available)
            
        Returns:
            ForecastResult with probability and prediction
        """
        if not self.is_trained:
            return None
        
        data = self._prepare_features()
        if data.empty:
            return None
        
        # Get the row for the specified date (or latest)
        if as_of_date:
            target_date = pd.Timestamp(as_of_date)
            if target_date not in data.index:
                # Find nearest date
                nearest_idx = data.index.get_indexer([target_date], method='nearest')[0]
                target_date = data.index[nearest_idx]
        else:
            target_date = data.index[-1]
        
        # Get features for this date
        X = data.loc[[target_date], self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        # Predict using lower threshold for rare events
        prob = self.model.predict_proba(X_scaled)[0, 1]
        pred = int(prob >= self.PREDICTION_THRESHOLD)
        
        # Get actual return if available
        actual_return = data.loc[target_date, "sp500_return"]
        actual = int(abs(actual_return) > self.THRESHOLD) if pd.notna(actual_return) else None
        
        return ForecastResult(
            date=target_date.date() if hasattr(target_date, 'date') else target_date,
            probability=round(float(prob), 4),
            prediction=pred,
            actual=actual,
            actual_return=round(float(actual_return) * 100, 2) if pd.notna(actual_return) else None,
        )
    
    def backtest(self, start_year: int = 2020) -> BacktestMetrics:
        """
        Run walk-forward backtest.
        
        Uses expanding window: train on all data up to t-1, predict t.
        
        Args:
            start_year: Start backtesting from this year
            
        Returns:
            BacktestMetrics with performance statistics
        """
        data = self._prepare_features()
        if data.empty:
            return None
        
        # Filter to backtest period
        backtest_start = pd.Timestamp(f"{start_year}-01-01")
        backtest_data = data[data.index >= backtest_start]
        
        if len(backtest_data) < 50:
            return None
        
        predictions = []
        actuals = []
        probabilities = []
        
        # Walk-forward: train on expanding window, predict next day
        min_train_samples = 252  # At least 1 year of training data
        
        for i in range(min_train_samples, len(data)):
            train_data = data.iloc[:i]
            test_row = data.iloc[[i]]
            
            if test_row.index[0] < backtest_start:
                continue
            
            # Train on historical data
            X_train = train_data[self.feature_names]
            y_train = self._create_target(train_data["sp500_return"])
            
            # Skip if no positive examples in training
            if y_train.sum() < 5:
                continue
            
            # Scale and fit
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            
            # Oversample for balanced training
            X_balanced, y_balanced = self._oversample(
                pd.DataFrame(X_train_scaled, columns=self.feature_names),
                y_train.reset_index(drop=True),
                target_ratio=0.15
            )
            
            model = MLPClassifier(
                hidden_layer_sizes=(64, 32, 16),
                activation='relu',
                solver='adam',
                alpha=0.01,
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
            )
            
            try:
                model.fit(X_balanced, y_balanced)
            except Exception:
                continue
            
            # Predict
            X_test = test_row[self.feature_names]
            X_test_scaled = scaler.transform(X_test)
            
            prob = model.predict_proba(X_test_scaled)[0, 1]
            pred = int(prob >= self.PREDICTION_THRESHOLD)
            actual = int(abs(test_row["sp500_return"].iloc[0]) > self.THRESHOLD)
            
            predictions.append(pred)
            actuals.append(actual)
            probabilities.append(prob)
        
        if not predictions:
            return None
        
        # Calculate metrics
        y_true = np.array(actuals)
        y_pred = np.array(predictions)
        y_prob = np.array(probabilities)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        return BacktestMetrics(
            accuracy=round(accuracy_score(y_true, y_pred), 4),
            precision=round(precision_score(y_true, y_pred, zero_division=0), 4),
            recall=round(recall_score(y_true, y_pred, zero_division=0), 4),
            f1_score=round(f1_score(y_true, y_pred, zero_division=0), 4),
            auc_roc=round(roc_auc_score(y_true, y_prob), 4) if y_true.sum() > 0 else 0,
            total_predictions=len(predictions),
            large_moves_caught=int(tp),
            large_moves_total=int(y_true.sum()),
            false_alarms=int(fp),
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
        )
    
    def get_feature_importance(self) -> dict[str, float]:
        """
        Get approximate feature importance based on weight magnitudes.
        
        Note: For MLP, this is a rough approximation using first layer weights.
        """
        if not self.is_trained or self.model is None:
            return {}
        
        # Get first layer weights
        weights = np.abs(self.model.coefs_[0])
        importance = weights.sum(axis=1)
        importance = importance / importance.sum()  # Normalize
        
        return {
            name: round(float(imp), 4)
            for name, imp in zip(self.feature_names, importance)
        }
    
    def get_forecast_history(self, days: int = 90) -> pd.DataFrame:
        """
        Get historical forecasts for charting.
        
        Returns DataFrame with dates, probabilities, predictions, and actuals.
        """
        if not self.is_trained:
            return pd.DataFrame()
        
        data = self._prepare_features()
        if data.empty:
            return pd.DataFrame()
        
        # Get last N days
        recent = data.tail(days)
        
        results = []
        for idx in recent.index:
            X = recent.loc[[idx], self.feature_names]
            X_scaled = self.scaler.transform(X)
            
            prob = self.model.predict_proba(X_scaled)[0, 1]
            actual_return = recent.loc[idx, "sp500_return"]
            actual = int(abs(actual_return) > self.THRESHOLD)
            
            results.append({
                "date": idx,
                "probability": prob,
                "prediction": int(prob >= self.PREDICTION_THRESHOLD),
                "actual": actual,
                "return_pct": actual_return * 100,
            })
        
        return pd.DataFrame(results).set_index("date")


def main():
    """CLI entry point for testing."""
    print("\n" + "=" * 70)
    print("LARGE MOVE FORECASTER - NEURAL NETWORK MODEL")
    print("=" * 70)
    print("\nTarget: Predict S&P 500 moves > 3% (either direction)")
    print("Method: MLP Neural Network with t-1 lagged features\n")
    
    forecaster = LargeMoveForecaster()
    
    # Train model
    print("Training model...")
    train_metrics = forecaster.train()
    
    if "error" in train_metrics:
        print(f"Error: {train_metrics['error']}")
        return
    
    print("\n--- Training Results ---")
    print(f"Training samples: {train_metrics['train_samples']}")
    print(f"Test samples: {train_metrics['test_samples']}")
    print(f"Large moves in training: {train_metrics['large_moves_train']} ({train_metrics['large_moves_train']/train_metrics['train_samples']:.1%})")
    print(f"Large moves in test: {train_metrics['large_moves_test']} ({train_metrics['large_moves_test']/train_metrics['test_samples']:.1%})")
    print(f"Prediction threshold: {train_metrics['prediction_threshold']} (lowered from 0.5 for rare events)")
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {train_metrics['accuracy']:.1%}")
    print(f"  Precision: {train_metrics['precision']:.1%}")
    print(f"  Recall:    {train_metrics['recall']:.1%}")
    print(f"  F1 Score:  {train_metrics['f1']:.4f}")
    print(f"  AUC-ROC:   {train_metrics['auc_roc']:.4f}")
    
    # Current forecast
    print("\n--- Current Forecast ---")
    forecast = forecaster.predict()
    if forecast:
        print(f"Date: {forecast.date}")
        print(f"Probability of >3% move: {forecast.probability:.1%}")
        print(f"Prediction: {'LARGE MOVE EXPECTED' if forecast.prediction else 'Normal day expected'}")
        if forecast.actual is not None:
            print(f"Actual: {'LARGE MOVE' if forecast.actual else 'Normal'} ({forecast.actual_return:+.2f}%)")
    
    # Feature importance
    print("\n--- Feature Importance (approximate) ---")
    importance = forecaster.get_feature_importance()
    for name, imp in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name:<25} {imp:.1%}")
    
    # Walk-forward backtest
    print("\n--- Walk-Forward Backtest (2020-present) ---")
    backtest = forecaster.backtest(start_year=2020)
    if backtest:
        print(f"Total predictions: {backtest.total_predictions}")
        print(f"Large moves in period: {backtest.large_moves_total}")
        print(f"\nPerformance:")
        print(f"  Accuracy:  {backtest.accuracy:.1%}")
        print(f"  Precision: {backtest.precision:.1%} (of predicted moves, {backtest.precision:.0%} were real)")
        print(f"  Recall:    {backtest.recall:.1%} (caught {backtest.large_moves_caught}/{backtest.large_moves_total} large moves)")
        print(f"  F1 Score:  {backtest.f1_score:.4f}")
        print(f"  AUC-ROC:   {backtest.auc_roc:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives:  {backtest.true_positives} (correctly predicted large moves)")
        print(f"  True Negatives:  {backtest.true_negatives} (correctly predicted normal days)")
        print(f"  False Positives: {backtest.false_positives} (false alarms)")
        print(f"  False Negatives: {backtest.false_negatives} (missed large moves)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
