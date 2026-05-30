"""
agent/nodes/levene.py — Levene variance equality test node.
"""

from scipy import stats
import pandas as pd


LEVENE_ALPHA = 0.05


def node_levene(state: dict) -> dict:
    """
    LangGraph node: run Levene test for equality of variances between lots.

    Tests OD and WT separately. If any dimension rejects variance equality,
    variances_equal is set to False — routing the graph to Welch t-test
    (Decision 2). If both pass, variances_equal is True — Student t-test.

    Reads  : state["inspection_data"]
    Writes : state["levene_result"], state["variances_equal"]
    Appends: state["alerts"] if variances are unequal (p < 0.05)

    Parameters
    ----------
    state : dict
        QualityState with at least inspection_data and alerts.

    Returns
    -------
    dict
        Updated state with levene_result and variances_equal populated.

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
    >>> s = node_levene({"inspection_data": df, "alerts": []})
    >>> "od_mm" in s["levene_result"]
    True
    >>> isinstance(s["variances_equal"], bool)
    True
    """
    df: pd.DataFrame = state["inspection_data"]
    alerts: list     = state.get("alerts", [])
    levene_result: dict = {}
    variances_equal = True  # assume equal until any dimension rejects

    lots = sorted(df["lot_id"].unique())

    for dim in ["od_mm", "wt_mm"]:
        groups = [df[df["lot_id"] == lot][dim].dropna().values for lot in lots]

        stat, p_value = stats.levene(*groups, center="median")
        equal = bool(p_value >= LEVENE_ALPHA)

        levene_result[dim] = {
            "statistic":       round(float(stat),    4),
            "p_value":         round(float(p_value), 4),
            "variances_equal": equal,
        }

        if not equal:
            variances_equal = False
            alerts.append(
                f"[LEVENE] {dim.upper()}: unequal variances between lots "
                f"(F={stat:.4f}, p={p_value:.4f}). "
                "Welch t-test will be applied for drift detection."
            )

    return {
        **state,
        "levene_result":   levene_result,
        "variances_equal": variances_equal,
        "alerts":          alerts,
    }
