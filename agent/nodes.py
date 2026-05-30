"""
agent/nodes.py — LangGraph pipeline nodes.

Each function receives QualityState and returns an updated QualityState.
Nodes never overwrite state keys set by other nodes — they only fill
their own designated keys and append to alerts.
"""

from scipy import stats
import numpy as np
import pandas as pd
from typing import Any


DIMENSIONS = ["od_mm", "wt_mm", "id_mm"]


def _descriptive_for_series(series: pd.Series) -> dict[str, Any]:
    """
    Compute descriptive statistics for a single numeric series.

    Covers content: position (mean, median, mode,
    percentiles, quartiles), dispersion (std, variance, CV, amplitude,
    IQR), and shape (Fisher skewness g1, Fisher excess kurtosis g2).

    Parameters
    ----------
    series : pd.Series
        Numeric measurements for one dimension.

    Returns
    -------
    dict
        Keys: n, mean, median, mode, std, variance, cv_pct,
              amplitude, p5, p25, p50, p75, p95, iqr,
              skewness, kurtosis, min, max.

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> r = _descriptive_for_series(s)
    >>> r["mean"]
    3.0
    >>> r["median"]
    3.0
    """
    n = len(series)
    mean = series.mean()
    median = series.median()

    mode_result = stats.mode(series, keepdims=True)
    mode_val = float(mode_result.mode[0])

    std = series.std(ddof=1)
    variance = series.var(ddof=1)
    cv_pct = (std / mean) * 100 if mean != 0 else np.nan

    amplitude = series.max() - series.min()
    p5  = np.percentile(series, 5)
    p25 = np.percentile(series, 25)  # Q1
    p50 = np.percentile(series, 50)  # Q2 = median
    p75 = np.percentile(series, 75)  # Q3
    p95 = np.percentile(series, 95)
    iqr = p75 - p25

    # bias=False → Fisher corrected (g1, g2) — consistent with Excel SKEW/KURT
    skewness = float(stats.skew(series, bias=False))
    kurtosis = float(stats.kurtosis(series, bias=False))  # excess: g2=0 means normal

    return {
        "n":         n,
        "mean":      round(float(mean),     4),
        "median":    round(float(median),   4),
        "mode":      round(mode_val,        4),
        "std":       round(float(std),      4),
        "variance":  round(float(variance), 6),
        "cv_pct":    round(float(cv_pct),   4),
        "amplitude": round(float(amplitude),4),
        "p5":        round(float(p5),       4),
        "p25":       round(float(p25),      4),
        "p50":       round(float(p50),      4),
        "p75":       round(float(p75),      4),
        "p95":       round(float(p95),      4),
        "iqr":       round(float(iqr),      4),
        "skewness":  round(skewness,        4),
        "kurtosis":  round(kurtosis,        4),
        "min":       round(float(series.min()), 4),
        "max":       round(float(series.max()), 4),
    }


def node_descriptive(state: dict) -> dict:
    """
    LangGraph node: compute descriptive statistics for all dimensions.

    Reads  : state["inspection_data"] (pd.DataFrame)
    Writes : state["descriptive_stats"]
    Appends: state["alerts"] if skewness or kurtosis indicate
             non-symmetric or heavy-tailed distribution.

    The per_lot breakdown is stored alongside overall stats so that
    downstream nodes (Levene, t-test) can consume it directly without
    re-grouping the DataFrame.

    Parameters
    ----------
    state : dict
        QualityState with at least inspection_data and alerts.

    Returns
    -------
    dict
        Updated state with descriptive_stats populated.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     "lot_id": ["A", "A", "B", "B"],
    ...     "od_mm":  [73.0, 73.1, 73.2, 73.0],
    ...     "wt_mm":  [5.5,  5.6,  5.4,  5.5],
    ...     "id_mm":  [62.0, 61.9, 62.4, 62.0],
    ... })
    >>> s = node_descriptive({"inspection_data": df, "alerts": []})
    >>> "od_mm" in s["descriptive_stats"]
    True
    >>> s["descriptive_stats"]["od_mm"]["overall"]["n"]
    4
    """
    df: pd.DataFrame = state["inspection_data"]
    alerts: list     = state.get("alerts", [])
    descriptive_stats: dict = {}

    for dim in DIMENSIONS:
        series = df[dim].dropna()
        overall = _descriptive_for_series(series)

        # Per-lot breakdown — consumed by Levene and t-test nodes
        per_lot: dict = {}
        for lot, group in df.groupby("lot_id"):
            per_lot[str(lot)] = _descriptive_for_series(group[dim].dropna())

        descriptive_stats[dim] = {
            "overall":  overall,
            "per_lot":  per_lot,
        }

    # Shape alerts for OD and WT only (ID has no spec limits)
    for dim in ["od_mm", "wt_mm"]:
        g1 = descriptive_stats[dim]["overall"]["skewness"]
        g2 = descriptive_stats[dim]["overall"]["kurtosis"]
        if abs(g1) > 1.0:
            alerts.append(
                f"[DESCRIPTIVE] {dim.upper()}: high skewness (g1={g1:.3f}) — "
                "distribution may not be symmetric."
            )
        if abs(g2) > 1.0:
            alerts.append(
                f"[DESCRIPTIVE] {dim.upper()}: high excess kurtosis (g2={g2:.3f}) — "
                "distribution deviates from normal shape."
            )

    return {
        **state,
        "descriptive_stats": descriptive_stats,
        "alerts":            alerts,
    }


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)