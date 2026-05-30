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

# ---------------------------------------------------------------------------
# node_capability — helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# node_capability
# ---------------------------------------------------------------------------

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
# ---------------------------------------------------------------------------
# node_normality — helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# node_normality
# ---------------------------------------------------------------------------

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
# ---------------------------------------------------------------------------
# node_levene
# ---------------------------------------------------------------------------

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
# ---------------------------------------------------------------------------
# node_ttest
# ---------------------------------------------------------------------------

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
# ---------------------------------------------------------------------------
# node_chisquare
# ---------------------------------------------------------------------------

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
# ---------------------------------------------------------------------------
# node_correlation — helper
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# node_correlation
# ---------------------------------------------------------------------------

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
# ---------------------------------------------------------------------------
# node_spc
# ---------------------------------------------------------------------------

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