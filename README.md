
# OCTG Quality Agent

Dimensional inspection of OCTG tubulars generates hundreds of measurements
per production lot. Manual review cannot reliably detect process drift,
capability loss, or variance shifts between lots — patterns that precede
nonconformity and can compromise well integrity long before any part formally
fails specification.

## How the Agent Solves It

The agent receives a CSV of OD, WT, and ID measurements and runs a fully
automated statistical pipeline with three embedded decision points:

1. **Capability check (Decision 1):** computes Cp and Cpk against API 5CT
   spec limits. If any dimension has Cpk < 1.00, a critical alert is raised
   and `cpk_critical` is set to `True`.
2. **Variance equality check (Decision 2):** runs Levene's test between lots.
   The result routes the next step — equal variances → Student t-test;
   unequal variances → Welch t-test.
3. **Drift detection (Decision 3):** the t-test compares lot means. If a
   significant shift is found, `drift_detected` is set to `True` and an alert
   is appended.

All three decisions accumulate alerts but never skip nodes — the pipeline
always runs to completion and delivers the full statistical picture in a
single structured report.

## Sample Output

SPC individuals control charts generated automatically for each controlled dimension:

**Outside Diameter (OD)**
![SPC - Outside Diameter](https://claude.ai/chat/docs/images/spc_od_mm.png)

**Wall Thickness (WT)**
![SPC - Wall Thickness](https://claude.ai/chat/docs/images/spc_wt_mm.png)

Red points indicate observations beyond ±3σ control limits. Control limits
(CL, UCL, LCL) are statistical — distinct from API 5CT spec limits (LSL/USL).

## How to Run

```bash
git clone <repo-url>
cd octg-quality-agent
pip install -r requirements.txt
python main.py --size "2-7/8" --weight 6.40
```

Use `--data` to point to a different inspection CSV:

```bash
python main.py --data path/to/data.csv --size "2-7/8" --weight 6.40
```

The input CSV must contain the columns: `lot_id`, `od_mm`, `wt_mm`, `id_mm`.

## Project Structure

```
octg-quality-agent/
├── agent/
│   ├── nodes/
│   │   ├── __init__.py       # Re-exports all 8 node functions
│   │   ├── descriptive.py    # Descriptive stats (mean, std, CV, skewness, kurtosis…)
│   │   ├── capability.py     # Cp, Cpk — bilateral OD; unilateral lower WT
│   │   ├── normality.py      # Shapiro-Wilk per dimension and per lot
│   │   ├── levene.py         # Levene test for variance equality between lots
│   │   ├── ttest.py          # Student / Welch t-test for drift detection
│   │   ├── chisquare.py      # Chi-square goodness of fit for conformance distribution
│   │   ├── correlation.py    # Pearson correlation for all dimension pairs
│   │   └── spc.py            # Individuals control chart — 3-sigma rule
│   ├── state.py              # QualityState TypedDict — shared pipeline state
│   └── specs.py              # API 5CT spec limits catalog (size × weight)
├── graph/
│   └── pipeline.py           # Compiled pipeline with 3 conditional edges
├── tests/
│   └── test_nodes.py         # 33 pytest tests with shared DataFrame fixture
├── data/
│   └── inspection_sample.csv # Sample data — 2-7/8" J55, 100 parts
├── docs/
│   └── images/               # Sample output charts
├── main.py                   # CLI entry point
└── requirements.txt
```

## Statistical Methods

| Statistical Methods (USP Esalq)                                             | Project Application                                                                                        |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Measures of position (mean, median, mode) and dispersion (std, CV, IQR)      | `node_descriptive`— overall and per-lot breakdown for OD, WT, ID                                        |
| Shape indicators: Fisher skewness (g1) and excess kurtosis (g2)              | `node_descriptive`— alerts when                                                                         |
| Process capability indices Cp and Cpk                                        | `node_capability`— bilateral for OD; unilateral lower for WT (API 5CT §7.11.2)                         |
| Student t-test for comparison of two independent means                       | `node_ttest`— lot-to-lot drift detection when variances are equal                                       |
| Chi-square goodness of fit (Fávero e Belfiore, 2024, Cap. 8)                | `node_chisquare`— tests whether non-conformances are uniformly distributed across spec categories       |
| Pearson correlation and significance test (Fávero e Belfiore, 2024, Cap. 8) | `node_correlation`— OD×WT, OD×ID, WT×ID pairs                                                        |
| *(industry practice)*                                                      | `node_normality`— Shapiro-Wilk;`node_levene`— variance equality;`node_spc`— 3-sigma control chart |

## Design Decisions

### (a) Why Levene runs before the t-test

The Student t-test assumes equal variances between groups. Running Levene's
test first is the correct parametric protocol: if variances are equal
(p ≥ 0.05), Student's pooled-variance t-test is applied; if unequal
(p < 0.05), the pipeline automatically switches to Welch's t-test, which does
not require equal variances. This keeps the drift detection result
statistically valid regardless of the lot's dispersion profile.

### (b) Why Test Z is not used for drift detection

Test Z assumes the population standard deviation (σ) is known. In real
manufacturing processes σ is never known — it must be estimated from sample
data. Using a sample standard deviation with a Z-test is statistically
incorrect; the t-test (Student or Welch) is the appropriate choice whenever
σ is unknown, which is always the case in industrial inspection.

### (c) Why ID has no Cpk

API 5CT specifies no direct dimensional tolerance for internal diameter. ID is
a derived dimension: `ID = OD − 2 × WT`. Its variation is mathematically
determined by OD and WT tolerances. Computing Cpk for ID would require
inventing a specification limit not present in the standard, producing a
meaningless result. The pipeline reports ID descriptive statistics and
correlation but omits capability analysis for this dimension.

## Known Limitations and Future Improvements

* **Data source:** the pipeline currently reads from a local CSV file. The
  planned evolution is to replace this with a PostgreSQL source, enabling the
  agent to run against live inspection databases without manual file exports.
* **Natural language report:** the final node terminates with a structured
  dict. A planned LLM node will consume this state and generate a natural
  language inspection report via the Claude API or AWS Bedrock, making results
  accessible to non-technical stakeholders.
* **Two-lot assumption:** the t-test and Levene nodes assume exactly two lots.
  Multi-lot support (ANOVA, Bartlett) is not yet implemented.
* **LangGraph checkpointing:** `pd.DataFrame` is not natively serializable by
  LangGraph's checkpointer. Replacing the CSV input with PostgreSQL will also
  resolve this limitation.

## References

* API 5CT, 11th Edition (2018) — Tables C.25 and 15, Sections 7.11.1 and 7.11.2
* Fávero, L. P.; Belfiore, P.  *Manual de Análise de Dados* . 2024, Cap. 8.
