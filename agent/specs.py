"""
API 5CT (11th Edition, 2018) specification limits for OCTG tubing.

OD tolerance  : ±0.79 mm for all tubing sizes up to 4-1/2" (Table C.25).
WT tolerance  : -12.5 % of nominal, no upper limit (Section 7.11.2).
ID tolerance  : not specified directly in API 5CT — derived dimension only.

cpk_type values:
  "bilateral"        — both LSL and USL defined (OD).
  "unilateral_lower" — only LSL defined (WT).
  "none"             — no tolerance; Cpk not calculated (ID).
"""

_OD_TOL = 0.79  # mm, ±

# Keys: (size_str, weight_lb_ft)
# Values: (OD_nominal_mm, WT_nominal_mm, ID_nominal_mm)
_CATALOG: dict[tuple[str, float], tuple[float, float, float]] = {
    ("2-3/8", 4.60): (60.325, 4.039, 52.247),
    ("2-3/8", 5.80): (60.325, 5.156, 50.013),
    ("2-7/8", 6.40): (73.02, 5.51, 62.00),
    ("2-7/8", 8.60): (73.025, 7.620, 57.785),
    ("3-1/2", 7.70): (88.900, 5.486, 77.928),
    ("3-1/2", 9.20): (88.900, 6.553, 75.794),
    ("3-1/2", 10.20): (88.900, 7.366, 74.168),
    ("4",     9.50): (101.600, 5.740, 90.120),
    ("4",     11.00): (101.600, 6.650, 88.300),
    ("4-1/2", 9.50): (114.300, 5.212, 103.876),
    ("4-1/2", 11.60): (114.300, 6.350, 101.600),
    ("4-1/2", 13.50): (114.300, 7.366, 99.568),
    ("4-1/2", 15.10): (114.300, 8.560, 97.180),
}


def get_specs(size: str, weight: float) -> dict:
    """
    Return API 5CT specification limits for the given tubing size and weight.

    Parameters
    ----------
    size   : nominal size string, e.g. '2-7/8'
    weight : linear weight in lb/ft, e.g. 6.40

    Returns
    -------
    dict with keys 'OD', 'WT', 'ID', each containing:
        LSL      : lower spec limit in mm (None if not specified)
        USL      : upper spec limit in mm (None if not specified)
        nominal  : nominal value in mm
        cpk_type : 'bilateral' | 'unilateral_lower' | 'none'

    Raises
    ------
    KeyError if (size, weight) combination is not in the catalog.
    """
    key = (size, float(weight))
    if key not in _CATALOG:
        available = sorted(_CATALOG.keys())
        raise KeyError(
            f"No specs for size={size!r}, weight={weight}. "
            f"Available combinations: {available}"
        )

    od_nom, wt_nom, id_nom = _CATALOG[key]
    wt_lsl = round(wt_nom * 0.875, 2)

    return {
        "OD": {
            "LSL": round(od_nom - _OD_TOL, 3),
            "USL": round(od_nom + _OD_TOL, 3),
            "nominal": od_nom,
            "cpk_type": "bilateral",
        },
        "WT": {
            "LSL": wt_lsl,
            "USL": None,
            "nominal": wt_nom,
            "cpk_type": "unilateral_lower",
        },
        "ID": {
            "LSL": None,
            "USL": None,
            "nominal": id_nom,
            "cpk_type": "none",
        },
    }


def get_spec_limits(size: str = "2-7/8", weight: float = 6.40) -> dict:
    """
    Return spec limits in the lowercase-key format consumed by pipeline nodes.

    Wraps get_specs and maps OD→od_mm, WT→wt_mm, LSL→lsl, USL→usl.

    Parameters
    ----------
    size   : nominal size string (default '2-7/8')
    weight : linear weight in lb/ft (default 6.40)

    Returns
    -------
    dict with keys 'od_mm' and 'wt_mm', each containing 'lsl' and 'usl'.
    """
    raw = get_specs(size, float(weight))
    return {
        "od_mm": {"lsl": raw["OD"]["LSL"], "usl": raw["OD"]["USL"]},
        "wt_mm": {"lsl": raw["WT"]["LSL"], "usl": raw["WT"]["USL"]},
    }
