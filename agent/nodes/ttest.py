"""
agent/nodes/ttest.py — Student / Welch t-test drift detection node.
"""

from scipy import stats
import pandas as pd


TTEST_ALPHA = 0.05


def node_ttest(state: dict) -> dict:
    """
    LangGraph node: compare lot means with Student or Welch t-test.

    Routes based on state["variances_equal"] (Decision 2 output):
    - True  → Student t-test (equal_var=True)
    - False → Welch t-test   (equal_var=False)

    If any dimension rejects mean equality, drift_detected is set to
    True (Decision 3), triggering a drift alert before continuing.

    Reads  : state["inspection_data"], state["variances_equal"]
    Writes : state["ttest_result"], state["drift_detected"]
    Appends: state["alerts"] if drift is detected (p < 0.05)

    Parameters
    ----------
    state : dict
        QualityState with at least inspection_data, variances_equal,
        and alerts.

    Returns
    -------
    dict
        Updated state with ttest_result and drift_detected populated.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> df = pd.DataFrame({
    ...     "lot_id": ["A"] * 25 + ["B"] * 25,
    ...     "od_mm":  np.concatenate([
    ...         np.random.normal(73.02, 0.20, 25),
    ...         np.random.normal(73.02, 0.20, 25),
    ...     ]),
    ...     "wt_mm":  np.concatenate([
    ...         np.random.normal(5.51, 0.10, 25),
    ...         np.random.normal(5.51, 0.10, 25),
    ...     ]),
    ...     "id_mm":  np.random.normal(62.00, 0.15, 50),
    ... })
    >>> s = node_ttest({
    ...     "inspection_data": df,
    ...     "variances_equal": True,
    ...     "alerts": [],
    ... })
    >>> "od_mm" in s["ttest_result"]
    True
    >>> s["ttest_result"]["method"]
    'student'
    >>> isinstance(s["drift_detected"], bool)
    True
    """
    df: pd.DataFrame  = state["inspection_data"]
    variances_equal   = state.get("variances_equal", True)
    alerts: list      = state.get("alerts", [])
    ttest_result: dict = {}
    drift_detected = False

    method = "student" if variances_equal else "welch"
    ttest_result["method"] = method

    lots = sorted(df["lot_id"].unique())
    lot_a, lot_b = lots[0], lots[1]

    for dim in ["od_mm", "wt_mm"]:
        group_a = df[df["lot_id"] == lot_a][dim].dropna().values
        group_b = df[df["lot_id"] == lot_b][dim].dropna().values

        stat, p_value = stats.ttest_ind(
            group_a, group_b,
            equal_var=variances_equal,
        )

        mean_a = round(float(group_a.mean()), 4)
        mean_b = round(float(group_b.mean()), 4)
        delta  = round(mean_b - mean_a, 4)
        drift  = bool(p_value < TTEST_ALPHA)

        ttest_result[dim] = {
            "statistic":  round(float(stat),    4),
            "p_value":    round(float(p_value), 4),
            "drift":      drift,
            "mean_lot_a": mean_a,
            "mean_lot_b": mean_b,
            "delta":      delta,
        }

        if drift:
            drift_detected = True
            direction = "+" if delta > 0 else ""
            alerts.append(
                f"[TTEST] {dim.upper()}: mean drift detected between lots "
                f"({method.capitalize()}, t={stat:.4f}, p={p_value:.4f}, "
                f"delta={direction}{delta:.4f}mm). "
                "Investigate process shift between lots."
            )

    return {
        **state,
        "ttest_result":   ttest_result,
        "drift_detected": drift_detected,
        "alerts":         alerts,
    }
