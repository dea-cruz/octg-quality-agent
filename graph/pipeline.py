"""
graph/pipeline.py — LangGraph pipeline for OCTG dimensional quality control.

Defines the stateful graph with 3 conditional decision points:
  Decision 1 — after node_capability: cpk_critical flag
  Decision 2 — after node_levene: variances_equal routes to Student or Welch
  Decision 3 — after node_ttest: drift_detected flag

All decisions append alerts but never skip nodes — the pipeline always
runs to completion so the final report contains the full statistical picture.
"""

from langgraph.graph import StateGraph, END
from agent.state import QualityState
from agent.nodes import (
    node_descriptive,
    node_capability,
    node_normality,
    node_levene,
    node_ttest,
    node_chisquare,
    node_correlation,
    node_spc,
)


# ---------------------------------------------------------------------------
# Conditional routing functions
# ---------------------------------------------------------------------------

def route_cpk(state: QualityState) -> str:
    """
    Decision 1: after node_capability.

    cpk_critical=True means at least one dimension has Cpk < 1.00.
    Alert is already appended by node_capability.
    Both paths continue to node_normality — the flag drives alerting,
    not a different execution branch.
    """
    if state.get("cpk_critical"):
        return "cpk_critical"
    return "cpk_ok"


def route_levene(state: QualityState) -> str:
    """
    Decision 2: after node_levene.

    variances_equal=True → node_ttest will apply Student's pooled t-test.
    variances_equal=False → node_ttest will apply Welch's t-test.
    Both paths go to node_ttest; the flag is read internally by that node.
    """
    if state.get("variances_equal"):
        return "equal_variances"
    return "unequal_variances"


def route_ttest(state: QualityState) -> str:
    """
    Decision 3: after node_ttest.

    drift_detected=True means significant mean shift between lots.
    Alert already appended by node_ttest.
    Both paths continue to node_chisquare.
    """
    if state.get("drift_detected"):
        return "drift_detected"
    return "no_drift"


# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

def build_pipeline() -> StateGraph:
    """
    Build and compile the LangGraph quality control pipeline.

    Returns
    -------
    Compiled LangGraph app ready for invocation.
    """
    graph = StateGraph(QualityState)

    # Register nodes
    graph.add_node("node_descriptive",  node_descriptive)
    graph.add_node("node_capability",   node_capability)
    graph.add_node("node_normality",    node_normality)
    graph.add_node("node_levene",       node_levene)
    graph.add_node("node_ttest",        node_ttest)
    graph.add_node("node_chisquare",    node_chisquare)
    graph.add_node("node_correlation",  node_correlation)
    graph.add_node("node_spc",          node_spc)

    # Entry point
    graph.set_entry_point("node_descriptive")

    # Linear edge: descriptive → capability
    graph.add_edge("node_descriptive", "node_capability")

    # Decision 1 — cpk_critical
    # Both paths lead to node_normality; flag drives alert, not branching
    graph.add_conditional_edges(
        "node_capability",
        route_cpk,
        {
            "cpk_critical": "node_normality",
            "cpk_ok":        "node_normality",
        },
    )

    # Linear edge: normality → levene
    graph.add_edge("node_normality", "node_levene")

    # Decision 2 — variances_equal
    # Both paths lead to node_ttest; flag read internally by that node
    graph.add_conditional_edges(
        "node_levene",
        route_levene,
        {
            "equal_variances":   "node_ttest",
            "unequal_variances": "node_ttest",
        },
    )

    # Decision 3 — drift_detected
    # Both paths lead to node_chisquare; flag drives alert, not branching
    graph.add_conditional_edges(
        "node_ttest",
        route_ttest,
        {
            "drift_detected": "node_chisquare",
            "no_drift":       "node_chisquare",
        },
    )

    # Remaining linear edges
    graph.add_edge("node_chisquare",   "node_correlation")
    graph.add_edge("node_correlation", "node_spc")
    graph.add_edge("node_spc",         END)

    return graph.compile()


# Module-level compiled pipeline — imported by main.py
pipeline = build_pipeline()
