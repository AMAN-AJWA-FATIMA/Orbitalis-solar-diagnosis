"""Documented-condition labelling of panel observations.

Implements the two anomaly definitions of SRC-1 (Jara Cespedes et al., 2022):

* Type 1, Eq.(1):  V_i(t) < 1200 mV
* Type 2, Eq.(2):  2000 mV <= V_i(t-1), V_i(t), V_i(t+1) <= 3000 mV  AND  I_i(t) < 50 mA

All numbers are read from ``knowledge_base/faults.json`` (``detection_parameters``);
nothing is hard-coded here. The labels are *rule-derived*: they follow the
paper's labelling definition, they are not independent ground truth from the
satellite operators.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

GROUP_KEYS = [config.SATELLITE_COL, config.SESSION_COL, "Panel"]


def label_documented_conditions(long_df: pd.DataFrame, detection_parameters: dict) -> pd.DataFrame:
    """Add boolean columns ``Type1_Condition``, ``Type2_Condition``,
    ``Type2_Unverifiable`` plus ``Documented_Label`` (0/1) and ``Documented_Type``."""
    p1 = detection_parameters["type1_solar_panel_failure"]
    p2 = detection_parameters["type2_solar_cell_failure"]
    df = long_df.sort_values(GROUP_KEYS + ["Elapsed_s"]).reset_index(drop=True).copy()

    v = df["Voltage_mV"]
    i = df["Current_mA"]
    df["Type1_Condition"] = (v < p1["voltage_below_mv"]).fillna(False).to_numpy(dtype=bool)

    v_in = v.between(p2["voltage_min_mv"], p2["voltage_max_mv"]).to_numpy(dtype=bool)
    i_low = (i < p2["current_below_ma"]).fillna(False).to_numpy(dtype=bool)
    df["_v_in"] = v_in
    grouped = df.groupby(GROUP_KEYS, sort=False)
    prev_in = grouped["_v_in"].shift(1, fill_value=False).to_numpy(dtype=bool)
    next_in = grouped["_v_in"].shift(-1, fill_value=False).to_numpy(dtype=bool)
    prev_nan = grouped["Voltage_mV"].shift(1).isna().to_numpy()
    next_nan = grouped["Voltage_mV"].shift(-1).isna().to_numpy()
    prev_c = df["Prev_contiguous"].to_numpy(dtype=bool)
    next_c = df["Next_contiguous"].to_numpy(dtype=bool)

    candidate = v_in & i_low
    type2 = candidate & prev_c & next_c & prev_in & next_in
    # a neighbour is "unknown" (cannot disprove) when it is absent, not contiguous or NaN
    prev_unknown = (~prev_c) | prev_nan
    next_unknown = (~next_c) | next_nan
    prev_ok_or_unknown = prev_unknown | prev_in
    next_ok_or_unknown = next_unknown | next_in
    unverifiable = candidate & ~type2 & prev_ok_or_unknown & next_ok_or_unknown

    df["Type2_Condition"] = type2
    df["Type2_Unverifiable"] = unverifiable
    df = df.drop(columns="_v_in")
    df["Documented_Label"] = (df["Type1_Condition"] | df["Type2_Condition"]).astype(int)
    df["Documented_Type"] = np.select(
        [df["Type1_Condition"], df["Type2_Condition"]], ["Type 1", "Type 2"], default="None"
    )
    return df
