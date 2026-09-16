# securemesh_sce/evaluation/statistics.py
"""Statistical evaluation, multi-seed aggregation, and significance testing.

Implements:
- 95% Confidence intervals (CI)
- Mann-Whitney U test (non-parametric comparison)
- Cohen's d effect size
- SCENE hypothesis PASS/FAIL verification
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class MetricDistribution:
    mean: float
    std: float
    median: float
    ci_lower: float
    ci_upper: float
    raw_values: List[float]


@dataclass
class HypothesisResult:
    hypothesis_text: str
    metric: str
    threshold: float
    direction: str  # "above" or "below"
    observed_mean: float
    ci_95: Tuple[float, float]
    passed: bool
    verdict: str  # "PASS" or "FAIL"
    p_value: Optional[float] = None


class StatisticalAnalyzer:
    """Computes statistical summaries and hypothesis outcomes across seeds and episodes."""

    @staticmethod
    def summarize_metric(values: List[float]) -> MetricDistribution:
        arr = np.array(values, dtype=np.float64)
        if len(arr) == 0:
            return MetricDistribution(0.0, 0.0, 0.0, 0.0, 0.0, [])

        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        median = float(np.median(arr))

        if len(arr) > 1 and std > 1e-8:
            sem = std / np.sqrt(len(arr))
            ci = stats.t.interval(0.95, df=len(arr) - 1, loc=mean, scale=sem)
            ci_lower, ci_upper = float(ci[0]), float(ci[1])
        else:
            ci_lower, ci_upper = mean, mean

        return MetricDistribution(
            mean=round(mean, 4),
            std=round(std, 4),
            median=round(median, 4),
            ci_lower=round(ci_lower, 4),
            ci_upper=round(ci_upper, 4),
            raw_values=[round(float(v), 4) for v in values],
        )

    @staticmethod
    def compare_groups(
        group_a: List[float], group_b: List[float]
    ) -> Dict[str, Any]:
        """Perform Mann-Whitney U test and compute Cohen's d effect size."""
        a = np.array(group_a, dtype=np.float64)
        b = np.array(group_b, dtype=np.float64)

        if len(a) < 2 or len(b) < 2:
            return {"u_stat": 0.0, "p_value": 1.0, "cohens_d": 0.0}

        try:
            u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            u_stat, p_val = 0.0, 1.0

        # Cohen's d
        n1, n2 = len(a), len(b)
        s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        d = float((np.mean(a) - np.mean(b)) / pooled_std) if pooled_std > 1e-8 else 0.0

        return {
            "u_statistic": float(u_stat),
            "p_value": float(p_val),
            "significant_at_05": p_val < 0.05,
            "cohens_d": round(d, 4),
        }

    @staticmethod
    def evaluate_hypothesis(
        hypothesis_config: Dict[str, Any], observed_values: List[float]
    ) -> HypothesisResult:
        text = hypothesis_config.get("text", "").strip()
        metric = hypothesis_config.get("metric", "")
        threshold = float(hypothesis_config.get("threshold", 0.5))
        direction = hypothesis_config.get("direction", "above")

        dist = StatisticalAnalyzer.summarize_metric(observed_values)
        obs_mean = dist.mean

        if direction == "above":
            passed = obs_mean >= threshold
        else:
            passed = obs_mean <= threshold

        verdict = "PASS" if passed else "FAIL"

        return HypothesisResult(
            hypothesis_text=text,
            metric=metric,
            threshold=threshold,
            direction=direction,
            observed_mean=obs_mean,
            ci_95=(dist.ci_lower, dist.ci_upper),
            passed=passed,
            verdict=verdict,
        )
