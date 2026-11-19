"""Détection et traitement des valeurs aberrantes (outliers)."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class OutlierInfo:
    column: str
    method: str
    lower_bound: float
    upper_bound: float
    outlier_count: int
    outlier_pct: float
    can_log_transform: bool
    recommended_action: str


OUTLIER_ACTIONS = {
    "none": "Aucune modification (conserver)",
    "clip": "Plafonnement / Winsorisation (borner aux seuils)",
    "log1p": "Transformation logarithmique log(1 + x)",
    "drop": "Supprimer les lignes contenant des valeurs aberrantes",
}


def detect_column_outliers_iqr(series: pd.Series, k: float = 1.5) -> OutlierInfo:
    """Détecte les valeurs aberrantes par l'écart interquartile (IQR)."""
    clean = series.dropna()
    col = series.name or "colonne"
    if len(clean) < 5:
        return OutlierInfo(col, "IQR", 0.0, 0.0, 0, 0.0, False, "none")

    q25 = float(np.percentile(clean, 25))
    q75 = float(np.percentile(clean, 75))
    iqr = q75 - q25
    lower = q25 - k * iqr
    upper = q75 + k * iqr

    outliers = clean[(clean < lower) | (clean > upper)]
    count = int(len(outliers))
    pct = round((count / len(series)) * 100, 2)
    can_log = bool((clean >= 0).all())

    rec = "clip" if count > 0 else "none"
    return OutlierInfo(col, "IQR", round(lower, 4), round(upper, 4), count, pct, can_log, rec)


def detect_column_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> OutlierInfo:
    """Détecte les valeurs aberrantes par le Z-Score standardisé."""
    clean = series.dropna()
    col = series.name or "colonne"
    if len(clean) < 5 or clean.std() == 0:
        return OutlierInfo(col, "Z-Score", 0.0, 0.0, 0, 0.0, False, "none")

    mean = float(clean.mean())
    std = float(clean.std())
    lower = mean - threshold * std
    upper = mean + threshold * std

    outliers = clean[(clean < lower) | (clean > upper)]
    count = int(len(outliers))
    pct = round((count / len(series)) * 100, 2)
    can_log = bool((clean >= 0).all())

    rec = "clip" if count > 0 else "none"
    return OutlierInfo(col, "Z-Score", round(lower, 4), round(upper, 4), count, pct, can_log, rec)


def analyze_outliers(
    df: pd.DataFrame,
    numeric_cols: list[str],
    method: str = "IQR",
) -> dict[str, OutlierInfo]:
    """Analyse les outliers pour l'ensemble des colonnes numériques sélectionnées."""
    results: dict[str, OutlierInfo] = {}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if method.upper() == "Z-SCORE":
            results[col] = detect_column_outliers_zscore(s)
        else:
            results[col] = detect_column_outliers_iqr(s)
    return results
