"""
agent/nodes/chisquare.py — chi-square goodness of fit node.
"""

from scipy import stats
import numpy as np
import pandas as pd


CHISQ_ALPHA = 0.05


def node_chisquare(state: dict) -> dict:
    """
    LangGraph node: chi-square goodness of fit test for conformance distribution.

    Tests whether the observed distribution of parts across spec categories
    (below LSL, within spec, above USL) differs significantly from a uniform
    distribution. A significant result indicates non-conformances are not
    randomly distributed — the process is systematically producing defects.

    OD — 3 categories: below_lsl, within_spec, above_usl
    WT — 2 categories: below_lsl, within_spec (unilateral — no USL)

    Aligns with MBA USP Esalq content: chi-square goodness of fit test
    (Fávero e Belfiore, 2024, Cap. 8).

    Reads  : state["inspection_data"], state["spec_limits"]
    Writes : state["chisquare_result"]
    Appends: state["alerts"] if distribution rejects uniformity (p < 0.05)

    Parameters
    ----------
    state : dict
        QualityState with at least inspection_data, spec_limits, and alerts.

    Returns
    -------
    dict
        Updated state with chisquare_result populated.

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
    >>> spec = {
    ...     "od_mm": {"lsl": 72.23, "usl": 73.81},
    ...     "wt_mm": {"lsl": 4.82,  "usl": None},
    ... }
    >>> s = node_chisquare({
    ...     "inspection_data": df,
    ...     "spec_limits": spec,
    ...     "alerts": [],
    ... })
    >>> "od_mm" in s["chisquare_result"]
    True
    >>> "wt_mm" in s["chisquare_result"]
    True
    """
    df: pd.DataFrame = state["inspection_data"]
    specs            = state["spec_limits"]
    alerts: list     = state.get("alerts", [])
    chisquare_result: dict = {}

    # --- OD: 3 categories ---
    od     = df["od_mm"].dropna()
    od_lsl = specs["od_mm"]["lsl"]
    od_usl = specs["od_mm"]["usl"]

    od_observed = np.array([
        (od < od_lsl).sum(),
        ((od >= od_lsl) & (od <= od_usl)).sum(),
        (od > od_usl).sum(),
    ], dtype=float)

    od_expected = np.full(3, od_observed.sum() / 3)
    od_stat, od_p = stats.chisquare(od_observed, f_exp=od_expected)
    od_reject = bool(od_p < CHISQ_ALPHA)

    chisquare_result["od_mm"] = {
        "statistic":  round(float(od_stat), 4),
        "p_value":    round(float(od_p),    4),
        "reject":     od_reject,
        "categories": ["below_lsl", "within_spec", "above_usl"],
        "observed":   od_observed.tolist(),
        "expected":   [round(e, 2) for e in od_expected.tolist()],
    }

    if od_reject:
        alerts.append(
            f"[CHISQUARE] OD: non-uniform conformance distribution "
            f"(χ²={od_stat:.4f}, p={od_p:.4f}). "
            "Non-conformances are not randomly distributed across spec categories."
        )

    # --- WT: 2 categories (unilateral) ---
    wt     = df["wt_mm"].dropna()
    wt_lsl = specs["wt_mm"]["lsl"]

    wt_observed = np.array([
        (wt < wt_lsl).sum(),
        (wt >= wt_lsl).sum(),
    ], dtype=float)

    wt_expected = np.full(2, wt_observed.sum() / 2)
    wt_stat, wt_p = stats.chisquare(wt_observed, f_exp=wt_expected)
    wt_reject = bool(wt_p < CHISQ_ALPHA)

    chisquare_result["wt_mm"] = {
        "statistic":  round(float(wt_stat), 4),
        "p_value":    round(float(wt_p),    4),
        "reject":     wt_reject,
        "categories": ["below_lsl", "within_spec"],
        "observed":   wt_observed.tolist(),
        "expected":   [round(e, 2) for e in wt_expected.tolist()],
    }

    if wt_reject:
        alerts.append(
            f"[CHISQUARE] WT: non-uniform conformance distribution "
            f"(χ²={wt_stat:.4f}, p={wt_p:.4f}). "
            "Non-conformances are not randomly distributed across spec categories."
        )

    return {
        **state,
        "chisquare_result": chisquare_result,
        "alerts":           alerts,
    }
