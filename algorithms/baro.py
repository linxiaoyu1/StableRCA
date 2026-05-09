# ./algorithms/baro.py
"""BARO RobustScorer-style RCA wrapper.

This file keeps the essential scoring path from BARO's `robust_scorer`
(DataFrame mode) and returns per-node numeric scores directly.
"""

from typing import Dict

import pandas as pd
from sklearn.preprocessing import RobustScaler


def _drop_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ("time", "Time", "timestamp"):
        if c in df:
            df = df.drop(columns=[c])
    return df


def _drop_constant(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[:, (df != df.iloc[0]).any()]


def _convert_mem_mb(df: pd.DataFrame) -> pd.DataFrame:
    def update_mem(col: pd.Series) -> pd.Series:
        if col.name.endswith("_mem"):
            return col / 1e6
        return col

    return df.apply(update_mem)


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Matches BARO preprocess essentials for robust_scorer DataFrame path.
    return _convert_mem_mb(_drop_constant(_drop_time(df)))


def apply_baro(normal_data: pd.DataFrame, anomaly_data: pd.DataFrame) -> Dict[str, float]:
    """Run BARO RobustScorer-style scoring and return metric -> score."""
    normal_df = _preprocess(normal_data)
    anomal_df = _preprocess(anomaly_data)

    intersects = [c for c in normal_df.columns if c in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    scores: Dict[str, float] = {}

    for col in normal_df.columns:
        a = pd.to_numeric(normal_df[col], errors="coerce").dropna().to_numpy(dtype=float)
        b = pd.to_numeric(anomal_df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(a) == 0 or len(b) == 0:
            continue

        scaler = RobustScaler().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        scores[col] = float(max(zscores))

    return scores
