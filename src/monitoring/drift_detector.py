"""
Drift Detection Module
Monitors data and model drift in production.
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from dataclasses import dataclass


@dataclass
class DriftResult:
    """Result of drift detection analysis."""
    feature_name: str
    drift_score: float
    p_value: float
    is_drifted: bool
    method: str


class DriftDetector:
    """Detect data drift between reference and production distributions."""

    def __init__(self, threshold: float = 0.05, psi_threshold: float = 0.2):
        self.threshold = threshold
        self.psi_threshold = psi_threshold
        self.reference_stats: dict = {}

    def fit(self, reference_data: pd.DataFrame) -> "DriftDetector":
        """Fit reference distribution statistics."""
        self.reference_data = reference_data
        for col in reference_data.select_dtypes(include=[np.number]).columns:
            self.reference_stats[col] = {
                "mean": reference_data[col].mean(),
                "std": reference_data[col].std(),
                "min": reference_data[col].min(),
                "max": reference_data[col].max(),
                "quantiles": reference_data[col].quantile([0.25, 0.5, 0.75]).to_dict(),
            }
        return self

    def detect(self, current_data: pd.DataFrame) -> list[DriftResult]:
        """Detect drift across all features."""
        results = []
        numeric_cols = current_data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in self.reference_data.columns:
                continue

            # KS Test
            ks_result = self._ks_test(col, current_data[col])
            results.append(ks_result)

            # PSI
            psi_result = self._calculate_psi(col, current_data[col])
            results.append(psi_result)

        return results

    def _ks_test(self, feature_name: str, current_values: pd.Series) -> DriftResult:
        """Kolmogorov-Smirnov test for drift detection."""
        reference_values = self.reference_data[feature_name].dropna()
        current_clean = current_values.dropna()

        stat, p_value = ks_2samp(reference_values, current_clean)

        return DriftResult(
            feature_name=feature_name,
            drift_score=stat,
            p_value=p_value,
            is_drifted=p_value < self.threshold,
            method="ks_test",
        )

    def _calculate_psi(self, feature_name: str, current_values: pd.Series) -> DriftResult:
        """Population Stability Index for drift magnitude."""
        reference_values = self.reference_data[feature_name].dropna()
        current_clean = current_values.dropna()

        # Create bins from reference distribution
        n_bins = 10
        bins = np.linspace(
            min(reference_values.min(), current_clean.min()),
            max(reference_values.max(), current_clean.max()),
            n_bins + 1,
        )

        # Calculate proportions
        ref_counts = np.histogram(reference_values, bins=bins)[0]
        cur_counts = np.histogram(current_clean, bins=bins)[0]

        # Add smoothing
        ref_proportions = (ref_counts + 1) / (len(reference_values) + n_bins)
        cur_proportions = (cur_counts + 1) / (len(current_clean) + n_bins)

        # PSI formula
        psi = np.sum(
            (cur_proportions - ref_proportions)
            * np.log(cur_proportions / ref_proportions)
        )

        return DriftResult(
            feature_name=feature_name,
            drift_score=psi,
            p_value=0.0,  # PSI doesn't have p-value
            is_drifted=psi > self.psi_threshold,
            method="psi",
        )

    def get_summary(self, results: list[DriftResult]) -> dict:
        """Summarize drift detection results."""
        drifted_features = [r for r in results if r.is_drifted]
        return {
            "total_features_checked": len(results),
            "features_drifted": len(drifted_features),
            "drift_percentage": len(drifted_features) / max(len(results), 1) * 100,
            "max_drift_score": max(r.drift_score for r in results) if results else 0,
            "drifted_features": [r.feature_name for r in drifted_features],
            "should_retrain": len(drifted_features) > len(results) * 0.3,
        }
