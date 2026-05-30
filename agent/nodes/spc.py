"""
agent/nodes/spc.py — individuals control chart (3-sigma rule) node.
"""

import pandas as pd


def node_spc(state: dict) -> dict:
    """
    LangGraph node: individuals control chart with 3-sigma rule (Rule 1).

    Computes control limits (CL, UCL, LCL) from the data and identifies
    points beyond ±3σ. Control limits are statistical — distinct from
    API 5CT spec limits (LSL/USL).

    Note: SPC control charts are not covered in MBA USP Esalq coursework.
    Used here as industry standard practice for process monitoring.
    Documented in README as beyond course content.

    Reads  : state["inspection_data"]
    Writes : state["spc_results"]
    Appends: state["alerts"] if any violations are detected

    Parameters
    ----------
    state : dict
        QualityState with at least inspection_data and alerts.

    Returns
    -------
    dict
        Updated state with spc_results populated.

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
    >>> s = node_spc({"inspection_data": df, "alerts": []})
    >>> "od_mm" in s["spc_results"]
    True
    >>> "ucl" in s["spc_results"]["od_mm"]
    True
    >>> "violations" in s["spc_results"]["od_mm"]
    True
    """
    df: pd.DataFrame = state["inspection_data"]
    alerts: list     = state.get("alerts", [])
    spc_results: dict = {}

    for dim in ["od_mm", "wt_mm"]:
        series = df[dim].dropna().reset_index(drop=True)

        cl  = float(series.mean())
        std = float(series.std(ddof=1))
        ucl = cl + 3 * std
        lcl = cl - 3 * std

        violations = []
        for i, val in series.items():
            if val > ucl:
                violations.append({
                    "index": int(i),
                    "value": round(float(val), 4),
                    "side":  "above_ucl",
                })
            elif val < lcl:
                violations.append({
                    "index": int(i),
                    "value": round(float(val), 4),
                    "side":  "below_lcl",
                })

        spc_results[dim] = {
            "cl":           round(cl,  4),
            "ucl":          round(ucl, 4),
            "lcl":          round(lcl, 4),
            "std":          round(std, 4),
            "n_violations": len(violations),
            "violations":   violations,
        }

        if violations:
            alerts.append(
                f"[SPC] {dim.upper()}: {len(violations)} point(s) beyond 3σ control limits "
                f"(UCL={ucl:.4f}, LCL={lcl:.4f}). "
                "Investigate assignable causes."
            )

    return {
        **state,
        "spc_results": spc_results,
        "alerts":      alerts,
    }
