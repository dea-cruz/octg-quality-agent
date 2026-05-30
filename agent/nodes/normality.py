"""
agent/nodes/normality.py — Shapiro-Wilk normality test node.
"""

from scipy import stats
import pandas as pd


NORMALITY_ALPHA = 0.05


def _shapiro_for_series(series: pd.Series) -> dict:
    """
    Run Shapiro-Wilk normality test on a numeric series.

    Parameters
    ----------
    series : pd.Series

    Returns
    -------
    dict with statistic, p_value, normal (bool).

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> s = pd.Series(np.random.normal(0, 1, 50))
    >>> r = _shapiro_for_series(s)
    >>> "statistic" in r and "p_value" in r and "normal" in r
    True
    """
    stat, p_value = stats.shapiro(series)
    return {
        "statistic": round(float(stat),    4),
        "p_value":   round(float(p_value), 4),
        "normal":    bool(p_value >= NORMALITY_ALPHA),
    }


def node_normality(state: dict) -> dict:
    """
    LangGraph node: run Shapiro-Wilk normality test for OD and WT.

    Note: Shapiro-Wilk is not covered in MBA USP Esalq coursework.
    Used here as industry standard practice for process capability
    analysis. Documented in README as beyond course content.

    Reads  : state["inspection_data"]
    Writes : state["normality_results"]
    Appends: state["alerts"] if normality is rejected (p < 0.05)

    Parameters
    ----------
    state : dict
        QualityState with at least inspection_data and alerts.

    Returns
    -------
    dict
        Updated state with normality_results populated.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     "lot_id": ["A"] * 25 + ["B"] * 25,
    ...     "od_mm":  np.random.normal(73.02, 0.20, 50),
    ...     "wt_mm":  np.random.normal(5.51,  0.10, 50),
    ...     "id_mm":  np.random.normal(62.00, 0.15, 50),
    ... })
    >>> s = node_normality({"inspection_data": df, "alerts": []})
    >>> "od_mm" in s["normality_results"]
    True
    >>> "per_lot" in s["normality_results"]["od_mm"]
    True
    """
    df: pd.DataFrame = state["inspection_data"]
    alerts: list     = state.get("alerts", [])
    normality_results: dict = {}

    for dim in ["od_mm", "wt_mm"]:
        overall = _shapiro_for_series(df[dim].dropna())
        per_lot = {
            lot: _shapiro_for_series(group[dim].dropna())
            for lot, group in df.groupby("lot_id")
        }
        normality_results[dim] = {"overall": overall, "per_lot": per_lot}
        if not overall["normal"]:
            alerts.append(
                f"[NORMALITY] {dim.upper()}: normality rejected "
                f"(p={overall['p_value']:.4f}) — "
                "Shapiro-Wilk test failed; consider non-parametric methods."
            )

    return {
        **state,
        "normality_results": normality_results,
        "alerts":            alerts,
    }
