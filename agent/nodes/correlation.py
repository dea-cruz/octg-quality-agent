"""
agent/nodes/correlation.py — Pearson correlation node.
"""

from scipy import stats
import numpy as np
import pandas as pd


CORR_ALPHA = 0.05


def _pearson_pair(x: pd.Series, y: pd.Series) -> dict:
    """
    Compute covariance, Pearson r, t-statistic and p-value for two series.

    Uses the t-test for Pearson correlation significance per MBA USP Esalq
    content (Fávero e Belfiore, 2024, Cap. 8):
        t = r / sqrt((1 - r²) / (n - 2))
    with n-2 degrees of freedom.

    Parameters
    ----------
    x : pd.Series
    y : pd.Series

    Returns
    -------
    dict with covariance, r, t_statistic, p_value, significant.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(0)
    >>> x = pd.Series(np.random.normal(0, 1, 50))
    >>> y = x * 2 + pd.Series(np.random.normal(0, 0.1, 50))
    >>> r = _pearson_pair(x, y)
    >>> r["r"] > 0.99
    True
    >>> r["significant"]
    True
    """
    n = len(x)
    cov = float(x.cov(y))
    r, p_value = stats.pearsonr(x, y)
    t_stat = r / np.sqrt((1 - r**2) / (n - 2))

    return {
        "covariance":  round(cov,           6),
        "r":           round(float(r),      4),
        "t_statistic": round(float(t_stat), 4),
        "p_value":     round(float(p_value),4),
        "significant": bool(p_value < CORR_ALPHA),
    }


def node_correlation(state: dict) -> dict:
    """
    LangGraph node: compute Pearson correlation for all dimension pairs.

    Pairs: OD×WT, OD×ID, WT×ID.

    Note: ID = OD - 2*WT by definition (API 5CT), so correlations
    involving ID are mathematically determined — results are reported
    but flagged as derived in the output.

    Aligns with MBA USP Esalq content: Pearson correlation and t-test
    for correlation significance (Fávero e Belfiore, 2024, Cap. 8).

    Reads  : state["inspection_data"]
    Writes : state["correlation_result"]
    Appends: state["alerts"] for strong or moderate significant correlations

    Parameters
    ----------
    state : dict
        QualityState with at least inspection_data and alerts.

    Returns
    -------
    dict
        Updated state with correlation_result populated.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     "lot_id": ["A"] * 50 + ["B"] * 50,
    ...     "od_mm":  np.random.normal(73.02, 0.20, 100),
    ...     "wt_mm":  np.random.normal(5.51,  0.10, 100),
    ...     "id_mm":  np.random.normal(62.00, 0.15, 100),
    ... })
    >>> s = node_correlation({"inspection_data": df, "alerts": []})
    >>> "od_x_wt" in s["correlation_result"]
    True
    >>> "od_x_id" in s["correlation_result"]
    True
    >>> "wt_x_id" in s["correlation_result"]
    True
    """
    df: pd.DataFrame = state["inspection_data"]
    alerts: list     = state.get("alerts", [])
    correlation_result: dict = {}

    pairs = [
        ("od_x_wt", "od_mm", "wt_mm", False),
        ("od_x_id", "od_mm", "id_mm", True),
        ("wt_x_id", "wt_mm", "id_mm", True),
    ]

    for key, col_x, col_y, derived in pairs:
        result = _pearson_pair(df[col_x].dropna(), df[col_y].dropna())
        result["id_derived"] = derived
        correlation_result[key] = result

        if result["significant"]:
            abs_r = abs(result["r"])
            label = col_x.replace("_mm","").upper() + "×" + col_y.replace("_mm","").upper()

            if abs_r > 0.7:
                alerts.append(
                    f"[CORRELATION] {label}: strong significant correlation "
                    f"(r={result['r']:.4f}, p={result['p_value']:.4f})."
                    + (" [ID is derived — expected]" if derived else "")
                )
            elif abs_r > 0.4:
                alerts.append(
                    f"[CORRELATION] {label}: moderate significant correlation "
                    f"(r={result['r']:.4f}, p={result['p_value']:.4f})."
                    + (" [ID is derived — expected]" if derived else "")
                )

    return {
        **state,
        "correlation_result": correlation_result,
        "alerts":             alerts,
    }
