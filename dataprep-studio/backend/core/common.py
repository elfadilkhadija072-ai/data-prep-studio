"""Constantes partagées, détection des types et sérialisation JSON propre."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

SEVERITY_ORDER = {"info": 0, "warn": 1, "crit": 2}

# ─── Options de transformation par « slot » (étape de la chaîne de nettoyage) ───
SLOT_LABELS = {
    "clean": "Nettoyage du texte",
    "impute": "Imputation",
    "outliers": "Valeurs aberrantes",
    "transform": "Transformation",
    "scale": "Mise à l'échelle",
    "encode": "Encodage",
}

def _o(*pairs):
    return [{"value": v, "label": l} for v, l in pairs]

NONE = ("none", "Ne rien appliquer")
OPTIONS = {
    "numeric": {
        "impute": _o(NONE, ("mean", "Moyenne"), ("median", "Médiane"), ("mode", "Mode"), ("constant", "Constante (0)")),
        "outliers": _o(NONE, ("clip_iqr", "Winsorisation IQR (1,5×)"), ("clip_zscore", "Winsorisation Z-score (±3σ)"),
                       ("clip_pct", "Winsorisation percentiles (1 % – 99 %)"), ("remove_rows", "Supprimer les lignes (train, IQR)")),
        "transform": _o(NONE, ("log1p", "log(1 + x)"), ("sqrt", "Racine carrée"), ("yeojohnson", "Yeo-Johnson")),
        "scale": _o(NONE, ("standard", "Standardisation (Z-score)"), ("minmax", "Min-Max [0, 1]"), ("robust", "Robuste (médiane / IQR)")),
    },
    "categorical": {
        "clean": _o(NONE, ("strip_lower", "Espaces + minuscules")),
        "impute": _o(NONE, ("mode", "Mode"), ("constant", "Catégorie « Missing »")),
        "encode": _o(NONE, ("onehot", "One-Hot"), ("ordinal", "Ordinal (0…k-1)"), ("frequency", "Fréquence")),
    },
    "datetime": {
        "encode": _o(NONE, ("parts", "Extraire année / mois / jour / jour-semaine")),
    },
    "target_classification": {"encode": _o(NONE, ("label", "Label Encoding"))},
    "target_regression": {"transform": _o(NONE, ("log1p", "log(1 + y)"), ("sqrt", "Racine carrée")),
                          "scale": _o(NONE, ("standard", "Standardisation"))},
}
OPTIONS["boolean"] = {"impute": OPTIONS["categorical"]["impute"], "encode": _o(NONE, ("ordinal", "Ordinal (0 / 1)"))}

SLOT_ORDER = {
    "numeric": ["impute", "outliers", "transform", "scale"],
    "categorical": ["clean", "impute", "encode"],
    "boolean": ["impute", "encode"],
    "datetime": ["encode"],
    "target_classification": ["encode"],
    "target_regression": ["transform", "scale"],
}

def slots_key(kind: str, role: str, problem: str | None) -> str:
    if role == "target":
        return "target_classification" if problem == "classification" else "target_regression"
    return kind

def empty_plan() -> dict:
    return {"drop": False, "clean": "none", "impute": "none", "outliers": "none",
            "transform": "none", "scale": "none", "encode": "none"}

# ─── Types ───
def infer_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "categorical"

def detect_problem(y: pd.Series) -> str:
    """classification vs régression à partir de la cible."""
    yy = y.dropna()
    if not pd.api.types.is_numeric_dtype(yy) or pd.api.types.is_bool_dtype(yy):
        return "classification"
    nun = yy.nunique()
    integer_like = bool(len(yy)) and bool(np.all(np.isclose(yy.to_numpy(float) % 1, 0)))
    if nun <= 10 or (integer_like and nun <= 20 and nun / max(len(yy), 1) < 0.05):
        return "classification"
    return "regression"

# ─── Sérialisation ───
def sanitize(o):
    """Convertit récursivement numpy/pandas → types JSON ; NaN/inf → None."""
    if isinstance(o, dict):
        return {str(k): sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [sanitize(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 6)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(o))
    if o is pd.NA or o is pd.NaT:
        return None
    if isinstance(o, np.ndarray):
        return sanitize(o.tolist())
    return o
