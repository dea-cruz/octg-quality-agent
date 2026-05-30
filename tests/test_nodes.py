import numpy as np
import pandas as pd
import pytest

from agent.nodes import (
    node_chisquare,
    node_capability,
    node_correlation,
    node_descriptive,
    node_levene,
    node_normality,
    node_spc,
    node_ttest,
)

N_PER_LOT = 50
RNG_SEED = 42


@pytest.fixture
def df():
    rng = np.random.default_rng(RNG_SEED)
    n = N_PER_LOT * 2
    return pd.DataFrame({
        "lot_id": ["A"] * N_PER_LOT + ["B"] * N_PER_LOT,
        "od_mm":  rng.normal(73.02, 0.20, n),
        "wt_mm":  rng.normal(5.51,  0.10, n),
        "id_mm":  rng.normal(62.00, 0.15, n),
    })


@pytest.fixture
def spec_limits():
    return {
        "od_mm": {"lsl": 72.23, "usl": 73.81},
        "wt_mm": {"lsl": 4.82,  "usl": None},
    }


# ---------------------------------------------------------------------------
# node_descriptive
# ---------------------------------------------------------------------------

def test_node_descriptive_dimension_keys(df):
    state = node_descriptive({"inspection_data": df, "alerts": []})
    for dim in ("od_mm", "wt_mm", "id_mm"):
        assert dim in state["descriptive_stats"]
        assert "overall" in state["descriptive_stats"][dim]
        assert "per_lot" in state["descriptive_stats"][dim]


def test_node_descriptive_n_equals_dataframe_length(df):
    state = node_descriptive({"inspection_data": df, "alerts": []})
    assert state["descriptive_stats"]["od_mm"]["overall"]["n"] == len(df)


def test_node_descriptive_per_lot_has_both_lots(df):
    state = node_descriptive({"inspection_data": df, "alerts": []})
    assert set(state["descriptive_stats"]["od_mm"]["per_lot"].keys()) == {"A", "B"}


def test_node_descriptive_overall_stat_fields(df):
    state = node_descriptive({"inspection_data": df, "alerts": []})
    overall = state["descriptive_stats"]["od_mm"]["overall"]
    expected = {"n", "mean", "median", "mode", "std", "variance", "cv_pct",
                "amplitude", "p5", "p25", "p50", "p75", "p95", "iqr",
                "skewness", "kurtosis", "min", "max"}
    assert expected.issubset(overall.keys())


# ---------------------------------------------------------------------------
# node_capability
# ---------------------------------------------------------------------------

def test_node_capability_dimension_keys(df, spec_limits):
    desc = node_descriptive({"inspection_data": df, "alerts": []})
    state = node_capability({**desc, "spec_limits": spec_limits})
    for dim in ("od_mm", "wt_mm", "id_mm"):
        assert dim in state["capability_results"]


def test_node_capability_od_indices_positive(df, spec_limits):
    desc = node_descriptive({"inspection_data": df, "alerts": []})
    state = node_capability({**desc, "spec_limits": spec_limits})
    od = state["capability_results"]["od_mm"]
    assert od["cp"] > 1.0
    assert od["cpk"] > 1.0


def test_node_capability_cpk_critical_false_for_capable_process(df, spec_limits):
    desc = node_descriptive({"inspection_data": df, "alerts": []})
    state = node_capability({**desc, "spec_limits": spec_limits})
    assert state["cpk_critical"] is False


def test_node_capability_id_has_no_spec_limits(df, spec_limits):
    desc = node_descriptive({"inspection_data": df, "alerts": []})
    state = node_capability({**desc, "spec_limits": spec_limits})
    id_cap = state["capability_results"]["id_mm"]
    assert id_cap["cp"] is None
    assert id_cap["cpk"] is None


def test_node_capability_cpk_critical_true_when_incapable(spec_limits):
    rng = np.random.default_rng(0)
    n = 100
    # Process centered well above USL → incapable
    df_bad = pd.DataFrame({
        "lot_id": ["A"] * 50 + ["B"] * 50,
        "od_mm":  rng.normal(74.50, 0.05, n),
        "wt_mm":  rng.normal(5.51,  0.10, n),
        "id_mm":  rng.normal(62.00, 0.15, n),
    })
    desc = node_descriptive({"inspection_data": df_bad, "alerts": []})
    state = node_capability({**desc, "spec_limits": spec_limits})
    assert state["cpk_critical"] is True


# ---------------------------------------------------------------------------
# node_normality
# ---------------------------------------------------------------------------

def test_node_normality_dimension_keys(df):
    state = node_normality({"inspection_data": df, "alerts": []})
    for dim in ("od_mm", "wt_mm"):
        assert dim in state["normality_results"]
        assert "overall" in state["normality_results"][dim]
        assert "per_lot" in state["normality_results"][dim]


def test_node_normality_result_fields(df):
    state = node_normality({"inspection_data": df, "alerts": []})
    overall = state["normality_results"]["od_mm"]["overall"]
    assert {"statistic", "p_value", "normal"} == overall.keys()
    assert isinstance(overall["normal"], bool)


def test_node_normality_normal_data_passes(df):
    state = node_normality({"inspection_data": df, "alerts": []})
    # Synthetic normal data should pass Shapiro-Wilk at n=100
    assert state["normality_results"]["od_mm"]["overall"]["normal"] is True


# ---------------------------------------------------------------------------
# node_levene
# ---------------------------------------------------------------------------

def test_node_levene_dimension_keys(df):
    state = node_levene({"inspection_data": df, "alerts": []})
    assert "od_mm" in state["levene_result"]
    assert "wt_mm" in state["levene_result"]


def test_node_levene_result_fields(df):
    state = node_levene({"inspection_data": df, "alerts": []})
    result = state["levene_result"]["od_mm"]
    assert {"statistic", "p_value", "variances_equal"} == result.keys()


def test_node_levene_variances_equal_is_bool(df):
    state = node_levene({"inspection_data": df, "alerts": []})
    assert isinstance(state["variances_equal"], bool)


def test_node_levene_equal_variance_data_passes(df):
    state = node_levene({"inspection_data": df, "alerts": []})
    # Lots drawn from same distribution should not reject variance equality
    assert state["levene_result"]["od_mm"]["variances_equal"] is True


# ---------------------------------------------------------------------------
# node_ttest
# ---------------------------------------------------------------------------

def test_node_ttest_student_method(df):
    state = node_ttest({"inspection_data": df, "variances_equal": True, "alerts": []})
    assert state["ttest_result"]["method"] == "student"


def test_node_ttest_welch_method(df):
    state = node_ttest({"inspection_data": df, "variances_equal": False, "alerts": []})
    assert state["ttest_result"]["method"] == "welch"


def test_node_ttest_result_fields(df):
    state = node_ttest({"inspection_data": df, "variances_equal": True, "alerts": []})
    for dim in ("od_mm", "wt_mm"):
        result = state["ttest_result"][dim]
        assert {"statistic", "p_value", "drift", "mean_lot_a", "mean_lot_b", "delta"} == result.keys()


def test_node_ttest_drift_detected_is_bool(df):
    state = node_ttest({"inspection_data": df, "variances_equal": True, "alerts": []})
    assert isinstance(state["drift_detected"], bool)


def test_node_ttest_detects_drift_between_shifted_lots():
    rng = np.random.default_rng(0)
    df_drift = pd.DataFrame({
        "lot_id": ["A"] * 30 + ["B"] * 30,
        "od_mm":  np.concatenate([rng.normal(73.02, 0.05, 30), rng.normal(73.50, 0.05, 30)]),
        "wt_mm":  np.concatenate([rng.normal(5.51,  0.05, 30), rng.normal(5.80,  0.05, 30)]),
        "id_mm":  rng.normal(62.00, 0.10, 60),
    })
    state = node_ttest({"inspection_data": df_drift, "variances_equal": True, "alerts": []})
    assert state["drift_detected"] is True
    assert len(state["alerts"]) >= 1


# ---------------------------------------------------------------------------
# node_chisquare
# ---------------------------------------------------------------------------

def test_node_chisquare_dimension_keys(df, spec_limits):
    state = node_chisquare({"inspection_data": df, "spec_limits": spec_limits, "alerts": []})
    assert "od_mm" in state["chisquare_result"]
    assert "wt_mm" in state["chisquare_result"]


def test_node_chisquare_result_fields(df, spec_limits):
    state = node_chisquare({"inspection_data": df, "spec_limits": spec_limits, "alerts": []})
    od = state["chisquare_result"]["od_mm"]
    assert {"statistic", "p_value", "reject", "categories", "observed", "expected"} == od.keys()


def test_node_chisquare_od_three_categories(df, spec_limits):
    state = node_chisquare({"inspection_data": df, "spec_limits": spec_limits, "alerts": []})
    assert state["chisquare_result"]["od_mm"]["categories"] == ["below_lsl", "within_spec", "above_usl"]
    assert len(state["chisquare_result"]["od_mm"]["observed"]) == 3


def test_node_chisquare_wt_two_categories(df, spec_limits):
    state = node_chisquare({"inspection_data": df, "spec_limits": spec_limits, "alerts": []})
    assert state["chisquare_result"]["wt_mm"]["categories"] == ["below_lsl", "within_spec"]
    assert len(state["chisquare_result"]["wt_mm"]["observed"]) == 2


# ---------------------------------------------------------------------------
# node_correlation
# ---------------------------------------------------------------------------

def test_node_correlation_pair_keys(df):
    state = node_correlation({"inspection_data": df, "alerts": []})
    for key in ("od_x_wt", "od_x_id", "wt_x_id"):
        assert key in state["correlation_result"]


def test_node_correlation_result_fields(df):
    state = node_correlation({"inspection_data": df, "alerts": []})
    result = state["correlation_result"]["od_x_wt"]
    assert {"covariance", "r", "t_statistic", "p_value", "significant", "id_derived"} == result.keys()


def test_node_correlation_id_derived_flag(df):
    state = node_correlation({"inspection_data": df, "alerts": []})
    assert state["correlation_result"]["od_x_id"]["id_derived"] is True
    assert state["correlation_result"]["wt_x_id"]["id_derived"] is True
    assert state["correlation_result"]["od_x_wt"]["id_derived"] is False


def test_node_correlation_strong_correlation_detected():
    rng = np.random.default_rng(0)
    x = rng.normal(73.02, 0.20, 100)
    df_corr = pd.DataFrame({
        "lot_id": ["A"] * 50 + ["B"] * 50,
        "od_mm":  x,
        "wt_mm":  x * 0.075 + rng.normal(0, 0.001, 100),  # nearly perfect linear relation
        "id_mm":  rng.normal(62.00, 0.15, 100),
    })
    state = node_correlation({"inspection_data": df_corr, "alerts": []})
    assert abs(state["correlation_result"]["od_x_wt"]["r"]) > 0.99
    assert state["correlation_result"]["od_x_wt"]["significant"] is True


# ---------------------------------------------------------------------------
# node_spc
# ---------------------------------------------------------------------------

def test_node_spc_dimension_keys(df):
    state = node_spc({"inspection_data": df, "alerts": []})
    for dim in ("od_mm", "wt_mm"):
        assert dim in state["spc_results"]


def test_node_spc_result_fields(df):
    state = node_spc({"inspection_data": df, "alerts": []})
    result = state["spc_results"]["od_mm"]
    assert {"cl", "ucl", "lcl", "std", "n_violations", "violations"} == result.keys()


def test_node_spc_control_limits_order(df):
    state = node_spc({"inspection_data": df, "alerts": []})
    for dim in ("od_mm", "wt_mm"):
        r = state["spc_results"][dim]
        assert r["lcl"] < r["cl"] < r["ucl"]


def test_node_spc_detects_outlier_violation():
    rng = np.random.default_rng(0)
    values = rng.normal(73.02, 0.05, 99).tolist() + [80.0]  # clear outlier
    df_spike = pd.DataFrame({
        "lot_id": ["A"] * 50 + ["B"] * 50,
        "od_mm":  values,
        "wt_mm":  rng.normal(5.51, 0.05, 100),
        "id_mm":  rng.normal(62.00, 0.10, 100),
    })
    state = node_spc({"inspection_data": df_spike, "alerts": []})
    assert state["spc_results"]["od_mm"]["n_violations"] >= 1
    assert any("[SPC]" in a for a in state["alerts"])
