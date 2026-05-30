"""
graph/pipeline.py — LangGraph pipeline for OCTG dimensional quality control.

Defines the stateful graph with 3 conditional decision points:
  Decision 1 — after node_capability: cpk_critical flag set
  Decision 2 — after node_levene: variances_equal routes to Student or Welch
  Decision 3 — after node_ttest: drift_detected flag set

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
    Alert is already appended by node_capability — routing always
    continues to node_normality regardless.
    """
    return "node_normality"


def route_levene(state: QualityState) -> str:
    """
    Decision 2: after node_levene.

    variances_equal controls which t-test variant node_ttest applies
    internally. Routing always goes to node_ttest.
    """
    return "node_ttest"


def route_ttest(state: QualityState) -> str:
    """
    Decision 3: after node_ttest.

    drift_detected=True means at least one dimension shows significant
    mean shift between lots. Alert already appended by node_ttest —
    routing always continues to node_chisquare.
    """
    return "node_chisquare"


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

    # Linear edges
    graph.add_edge("node_descriptive", "node_capability")

    # Decision 1 — cpk_critical
    graph.add_conditional_edges(
        "node_capability",
        route_cpk,
        {"node_normality": "node_normality"},
    )

    # Decision 2 — variances_equal
    graph.add_edge("node_normality", "node_levene")
    graph.add_conditional_edges(
        "node_levene",
        route_levene,
        {"node_ttest": "node_ttest"},
    )

    # Decision 3 — drift_detected
    graph.add_conditional_edges(
        "node_ttest",
        route_ttest,
        {"node_chisquare": "node_chisquare"},
    )

    # Remaining linear edges
    graph.add_edge("node_chisquare",   "node_correlation")
    graph.add_edge("node_correlation", "node_spc")
    graph.add_edge("node_spc",         END)

    return graph.compile()


# Module-level compiled pipeline — imported by main.py
pipeline = build_pipeline()