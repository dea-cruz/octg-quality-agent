"""
LangGraph state definition for the OCTG Quality Agent.

This module defines the single shared state object that flows through
every node in the pipeline. Each node reads from and writes to this state.
No statistical logic lives here — only the data structure.
"""

from typing import Optional
from typing_extensions import TypedDict
import pandas as pd


class QualityState(TypedDict):
    """
    Shared state for the OCTG dimensional quality control pipeline.

    Input fields are populated before the pipeline starts.
    Result fields are populated by each node as the pipeline executes.
    Flags drive conditional edges between nodes.
    Alerts accumulate across all nodes — never overwritten.
    """

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    # Raw inspection measurements (OD, WT, ID columns + lot identifier).
    # Loaded from CSV into memory. Known limitation: pandas DataFrames are
    # not natively serializable by LangGraph checkpointing. Planned
    # evolution: replace with PostgreSQL source.
    inspection_data: pd.DataFrame

    # Specification limits derived from API 5CT, 11th Edition (2018).
    # Structure: {"OD": {"LSL": 72.23, "USL": 73.81, "nominal": 73.02},
    #             "WT": {"LSL": 4.82,  "USL": None,  "nominal": 5.51},
    #             "ID": {"LSL": None,  "USL": None,  "nominal": 62.00}}
    # USL for WT is None — unilateral specification per Section 7.11.2.
    # Limits for ID are None — no direct tolerance in API 5CT.
    # Values are injected by specs.py and support parametrization across
    # different pipe sizes, weights, and grades.
    spec_limits: dict

    # ------------------------------------------------------------------
    # Statistical results — one dict per node, each node fills its own
    # ------------------------------------------------------------------

    # Descriptive statistics: mean, median, mode, std, CV, skewness,
    # kurtosis, percentiles, quartiles, IQR — per dimension.
    descriptive_stats: Optional[dict]

    # Process capability: Cp and Cpk per dimension.
    # OD: bilateral (LSL and USL defined).
    # WT: unilateral lower only (no USL per API 5CT Section 7.11.2).
    # ID: not calculated (no direct tolerance in API 5CT).
    capability_results: Optional[dict]

    # Shapiro-Wilk normality test results per dimension.
    # Not covered in MBA course material — applied as industry practice.
    normality_results: Optional[dict]

    # Levene test for equality of variances between lots.
    # Result routes Decision 2: equal variances → Student / unequal → Welch.
    levene_result: Optional[dict]

    # t-test result for drift detection between lots.
    # Test type (Welch or Student) is determined by Decision 2.
    # Population std is unknown in industrial processes — Test Z not used.
    ttest_result: Optional[dict]

    # Chi-square goodness of fit test.
    chisquare_result: Optional[dict]

    # Pearson correlation between OD, WT, and ID.
    correlation_result: Optional[dict]

    # SPC control charts and 3-sigma rule violations per dimension.
    # Not covered in MBA course material — applied as industry practice.
    spc_results: Optional[dict]

    # ------------------------------------------------------------------
    # Conditional decision flags — drive edges between nodes
    # ------------------------------------------------------------------

    # Decision 1 (after capability node):
    # True if Cpk < 1.0 in any dimension → triggers critical alert.
    cpk_critical: Optional[bool]

    # Decision 2 (after Levene test):
    # True if variances are statistically equal between lots.
    # True  → Student t-test (pooled variance).
    # False → Welch t-test (separate variances).
    variances_equal: Optional[bool]

    # Decision 3 (after t-test):
    # True if significant drift is detected between lots.
    drift_detected: Optional[bool]

    # ------------------------------------------------------------------
    # Accumulated alerts
    # ------------------------------------------------------------------

    # Each node appends to this list — never overwrites.
    # The final report consolidates all alerts from the full pipeline.
    alerts: list[str]

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    # Structured report assembled by the last node.
    # Contains consolidated alerts, key indicators, and chart references.
    report: Optional[dict]
