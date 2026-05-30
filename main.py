"""
main.py — Entry point for the OCTG Quality Agent.

Loads inspection data and API 5CT spec limits, runs the full
LangGraph statistical pipeline, and prints a structured report
with decision flags and accumulated alerts.

Usage
-----
    python main.py
    python main.py --data path/to/data.csv
"""

import argparse
import sys
import pandas as pd
from graph.pipeline import pipeline
from agent.specs import get_spec_limits

sys.stdout.reconfigure(encoding="utf-8")


def load_data(path: str) -> pd.DataFrame:
    """Load inspection CSV and validate required columns."""
    df = pd.read_csv(path)
    required = {"lot_id", "od_mm", "wt_mm", "id_mm"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df


def print_report(state: dict) -> None:
    """Print structured report from final pipeline state."""
    print("\n" + "=" * 60)
    print("OCTG QUALITY AGENT — INSPECTION REPORT")
    print("=" * 60)

    print("\n[ DECISION FLAGS ]")
    print(f"  cpk_critical:     {state.get('cpk_critical')}")
    print(f"  variances_equal:  {state.get('variances_equal')}")
    print(f"  drift_detected:   {state.get('drift_detected')}")
    print(f"  t-test method:    {state.get('ttest_result', {}).get('method', 'n/a')}")

    print("\n[ CAPABILITY ]")
    cap = state.get("capability_results", {})
    od = cap.get("od_mm", {})
    wt = cap.get("wt_mm", {})
    print(f"  OD — Cp={od.get('cp')}, Cpk={od.get('cpk')}")
    print(f"  WT — Cpk_lower={wt.get('cpk_lower')}")

    print("\n[ SPC VIOLATIONS ]")
    spc = state.get("spc_results", {})
    for dim, r in spc.items():
        print(f"  {dim}: {r['n_violations']} violation(s) "
              f"(UCL={r['ucl']}, LCL={r['lcl']})")

    print("\n[ ALERTS ]")
    alerts = state.get("alerts", [])
    if alerts:
        for a in alerts:
            print(f"  • {a}")
    else:
        print("  No alerts generated.")

    print("\n" + "=" * 60)
    print(f"Total alerts: {len(alerts)}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="OCTG Quality Agent")
    parser.add_argument(
        "--data",
        default="data/inspection_sample.csv",
        help="Path to inspection CSV (default: data/inspection_sample.csv)",
    )
    parser.add_argument(
        "--size",
        default="2-7/8",
        help="Nominal tubing size, e.g. '2-7/8' (default: 2-7/8)",
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=6.40,
        help="Linear weight in lb/ft, e.g. 6.40 (default: 6.40)",
    )
    args = parser.parse_args()

    print(f"Loading data from: {args.data}")
    df = load_data(args.data)
    print(f"Loaded {len(df)} inspections — lots: {sorted(df['lot_id'].unique())}")

    spec_limits = get_spec_limits(args.size, args.weight)
    print(f"Spec limits loaded: OD [{spec_limits['od_mm']['lsl']}, "
          f"{spec_limits['od_mm']['usl']}] | "
          f"WT LSL={spec_limits['wt_mm']['lsl']}")

    initial_state = {
        "inspection_data": df,
        "spec_limits":     spec_limits,
        "alerts":          [],
    }

    print("\nRunning pipeline...")
    final_state = pipeline.invoke(initial_state)
    print("Pipeline complete.")

    print_report(final_state)


if __name__ == "__main__":
    main()