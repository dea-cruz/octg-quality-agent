"""
agent/nodes/capability.py — process capability node.
"""

from typing import Any


def _cpk_bilateral(mean: float, std: float, lsl: float, usl: float) -> dict[str, Any]:
    """
    Compute Cp and Cpk for a bilateral specification (LSL and USL defined).

    Parameters
    ----------
    mean : float
    std  : float
    lsl  : float
    usl  : float

    Returns
    -------
    dict with cp, cpk, cpu, cpl.

    Examples
    --------
    >>> r = _cpk_bilateral(73.02, 0.20, 72.23, 73.81)
    >>> r["cp"] > 1.0
    True
    >>> r["cpk"] > 1.0
    True
    """
    cp  = (usl - lsl) / (6 * std)
    cpu = (usl - mean) / (3 * std)
    cpl = (mean - lsl) / (3 * std)
    cpk = min(cpu, cpl)

    return {
        "cp":  round(cp,  4),
        "cpk": round(cpk, 4),
        "cpu": round(cpu, 4),
        "cpl": round(cpl, 4),
    }


def _cpk_unilateral_lower(mean: float, std: float, lsl: float) -> dict[str, Any]:
    """
    Compute Cpk for a unilateral lower specification (LSL only, no USL).

    Used for WT per API 5CT Section 7.11.2 — only minimum wall is defined.

    Parameters
    ----------
    mean : float
    std  : float
    lsl  : float

    Returns
    -------
    dict with cpk_lower. cp is None (not applicable without USL).

    Examples
    --------
    >>> r = _cpk_unilateral_lower(5.51, 0.10, 4.82)
    >>> r["cpk_lower"] > 1.0
    True
    """
    cpk_lower = (mean - lsl) / (3 * std)
    return {
        "cpk_lower": round(cpk_lower, 4),
        "cp":        None,
    }


def _append_capability_alerts(dim: str, cpk: float, alerts: list) -> None:
    """Append capability alert based on Cpk threshold."""
    if cpk < 1.00:
        alerts.append(
            f"[CAPABILITY] {dim}: Cpk={cpk:.4f} — process INCAPABLE. "
            "Immediate investigation required."
        )
    elif cpk < 1.33:
        alerts.append(
            f"[CAPABILITY] {dim}: Cpk={cpk:.4f} — process capable but without margin. "
            "Monitor closely."
        )


def node_capability(state: dict) -> dict:
    """
    LangGraph node: compute process capability indices (Cp, Cpk).

    OD  — bilateral:        Cp and Cpk (LSL and USL from API 5CT Table 15)
    WT  — unilateral lower: Cpk lower only (LSL from API 5CT Section 7.11.2)
    ID  — not applicable:   no spec limits in API 5CT

    Sets cpk_critical = True if any Cpk < 1.00 (Decision 1 in the graph).

    Reads  : state["descriptive_stats"], state["spec_limits"]
    Writes : state["capability_results"], state["cpk_critical"]
    Appends: state["alerts"]

    Examples
    --------
    >>> spec = {
    ...     "od_mm": {"lsl": 72.23, "usl": 73.81},
    ...     "wt_mm": {"lsl": 4.82,  "usl": None},
    ... }
    >>> desc = {
    ...     "od_mm": {"overall": {"mean": 73.02, "std": 0.20}},
    ...     "wt_mm": {"overall": {"mean": 5.51,  "std": 0.10}},
    ... }
    >>> s = node_capability({
    ...     "descriptive_stats": desc,
    ...     "spec_limits": spec,
    ...     "alerts": [],
    ... })
    >>> s["capability_results"]["od_mm"]["cp"] > 1.0
    True
    >>> s["cpk_critical"]
    False
    """
    desc  = state["descriptive_stats"]
    specs = state["spec_limits"]
    alerts: list = state.get("alerts", [])
    capability_results: dict = {}
    cpk_critical = False

    # --- OD: bilateral ---
    od_mean = desc["od_mm"]["overall"]["mean"]
    od_std  = desc["od_mm"]["overall"]["std"]
    od_lsl  = specs["od_mm"]["lsl"]
    od_usl  = specs["od_mm"]["usl"]

    od_cap = _cpk_bilateral(od_mean, od_std, od_lsl, od_usl)
    capability_results["od_mm"] = od_cap
    _append_capability_alerts("OD", od_cap["cpk"], alerts)
    if od_cap["cpk"] < 1.00:
        cpk_critical = True

    # --- WT: unilateral lower ---
    wt_mean = desc["wt_mm"]["overall"]["mean"]
    wt_std  = desc["wt_mm"]["overall"]["std"]
    wt_lsl  = specs["wt_mm"]["lsl"]

    wt_cap = _cpk_unilateral_lower(wt_mean, wt_std, wt_lsl)
    capability_results["wt_mm"] = wt_cap
    _append_capability_alerts("WT", wt_cap["cpk_lower"], alerts)
    if wt_cap["cpk_lower"] < 1.00:
        cpk_critical = True

    # --- ID: no spec limits ---
    capability_results["id_mm"] = {
        "cp":   None,
        "cpk":  None,
        "note": "No spec limits in API 5CT",
    }

    return {
        **state,
        "capability_results": capability_results,
        "cpk_critical":       cpk_critical,
        "alerts":             alerts,
    }
