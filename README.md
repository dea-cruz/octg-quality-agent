

# OCTG Quality Agent

Statistical quality control agent for OCTG tubular inspection — API 5CT 2-7/8" J55

---

## Problem

Dimensional inspection of OCTG tubulars generates hundreds of measurements per lot. Manual review cannot reliably detect process drift, capability loss, or variance shifts between lots — patterns that precede nonconformity and compromise well integrity.

## Solution

LangGraph pipeline that receives OD, WT, and ID inspection data and automatically runs descriptive statistics, process capability (Cp/Cpk), normality testing, variance comparison, drift detection, chi-square, Pearson correlation, and SPC control charts — delivering a structured report with alerts grounded in statistical evidence.

## Future Improvements

* Data source replacing CSV input.
* LLM node for natural language report generation via Claude API or AWS Bedrock.

## References and Context

Specification limits sourced directly from API 5CT, 11th Edition (2018) — Table C.2, Table 15, Section 7.11.1, Section 7.11.2. Statistical pipeline applies content from the MBA in Data Science & AI at USP Esalq.
