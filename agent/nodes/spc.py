"""
agent/nodes/spc.py — individuals control chart (3-sigma rule) node.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted", font_scale=0.95)

_OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"

_DIM_LABELS = {
    "od_mm": "Outside Diameter (mm)",
    "wt_mm": "Wall Thickness (mm)",
}


def _plot_control_chart(
    series: pd.Series,
    dim: str,
    cl: float,
    ucl: float,
    lcl: float,
    violations: list[dict],
) -> None:
    """
    Save an individuals control chart for *dim* to outputs/.

    Parameters
    ----------
    series : pd.Series
        Measurement values in inspection order.
    dim : str
        Column name — used for file naming and axis label.
    cl : float
        Center line (process mean).
    ucl : float
        Upper control limit (mean + 3σ).
    lcl : float
        Lower control limit (mean − 3σ).
    violations : list[dict]
        Dicts with keys 'index' and 'value' for points beyond ±3σ.
    """
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    violation_idx = {v["index"] for v in violations}
    normal_idx = [i for i in series.index if i not in violation_idx]
    violation_idx_list = [i for i in series.index if i in violation_idx]

    fig, ax = plt.subplots(figsize=(12, 4))

    # — individual points
    ax.plot(series.index, series.values, color="#4C72B0", linewidth=0.8,
            zorder=2)
    ax.scatter(normal_idx, series[normal_idx], color="#4C72B0",
               s=18, zorder=3, label="Observation")
    if violation_idx_list:
        ax.scatter(violation_idx_list, series[violation_idx_list],
                   color="#C44E52", s=45, zorder=4, label="Violation (>3σ)")

    # — control lines
    ax.axhline(cl,  color="#2ca02c", linewidth=1.2, linestyle="-",
               label=f"CL = {cl:.4f}")
    ax.axhline(ucl, color="#d62728", linewidth=1.0, linestyle="--",
               label=f"UCL = {ucl:.4f}")
    ax.axhline(lcl, color="#d62728", linewidth=1.0, linestyle="--",
               label=f"LCL = {lcl:.4f}")

    ax.set_title(
        f"Individuals Control Chart — {_DIM_LABELS.get(dim, dim)}",
        fontsize=11, pad=10,
    )
    ax.set_xlabel("Observation", fontsize=9)
    ax.set_ylabel(_DIM_LABELS.get(dim, dim), fontsize=9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out_path = _OUTPUTS_DIR / f"spc_{dim}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def node_spc(state: dict) -> dict:
    """
    LangGraph node: individuals control chart with 3-sigma rule (Rule 1).

    Computes control limits (CL, UCL, LCL) from the data and identifies
    points beyond ±3σ. Control limits are statistical — distinct from
    API 5CT spec limits (LSL/USL).

    Saves one PNG per dimension to outputs/:
        outputs/spc_od_mm.png
        outputs/spc_wt_mm.png

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

        _plot_control_chart(series, dim, cl, ucl, lcl, violations)

    return {
        **state,
        "spc_results": spc_results,
        "alerts":      alerts,
    }