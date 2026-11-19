"""Détection et configuration de la variable cible et du problème de Machine Learning supervisé."""
from __future__ import annotations

import re
from dataclasses import dataclass
import pandas as pd


TARGET_KEYWORDS = [
    "target", "cible", "label", "y", "churn", "survived", "achat", "buy",
    "fraud", "default", "price", "prix", "outcome", "class", "classe",
    "status", "salaire", "salary", "revenue", "ventes", "sales", "score"
]


@dataclass
class TargetProposal:
    column: str
    problem_type: str  # 'classification_binaire', 'classification_multiclasse', 'regression'
    reason: str
    missing_count: int
    missing_pct: float
    classes_distribution: dict[str, int] | None = None
    is_imbalanced: bool = False
    imbalance_warning: str | None = None


def detect_problem_type(series: pd.Series) -> str:
    """Détermine si la cible relève d'une régression ou d'une classification."""
    clean = series.dropna()
    n_unique = clean.nunique()
    
    # Si booléen ou objet ou chaîne
    if pd.api.types.is_bool_dtype(series) or clean.dtype == object or str(clean.dtype).startswith("string"):
        return "classification_binaire" if n_unique == 2 else "classification_multiclasse"
    
    # Si numérique
    if pd.api.types.is_numeric_dtype(series):
        # Peu de valeurs distinctes et entières -> classification
        if n_unique <= 10 and (clean % 1 == 0).all():
            return "classification_binaire" if n_unique == 2 else "classification_multiclasse"
        # Flottant continu ou beaucoup de valeurs -> régression
        return "regression"
    
    return "classification_binaire" if n_unique == 2 else "classification_multiclasse"


def score_target_candidate(col: str, series: pd.Series, idx: int, total_cols: int) -> tuple[float, list[str]]:
    """Attribue un score d'admissibilité à une colonne comme cible supervisée."""
    score = 0.0
    reasons = []
    col_clean = re.sub(r"[^a-zA-Z0-9]", "", col.lower())
    
    # 1. Correspondance de nom
    for kw in TARGET_KEYWORDS:
        if col_clean == kw:
            score += 50.0
            reasons.append(f"Nom exact « {col} »")
            break
        elif kw in col_clean:
            score += 25.0
            reasons.append(f"Nom évocateur « {col} »")
            break

    # 2. Position en fin de tableau (convention ML très fréquente)
    if idx == total_cols - 1:
        score += 30.0
        reasons.append("Position en dernière colonne")
    elif idx == total_cols - 2 and total_cols > 3:
        score += 10.0

    # 3. Cardinalité et type
    clean = series.dropna()
    n_unique = clean.nunique()
    n_total = len(series)
    
    # Pénalité sévère pour les identifiants ou colonnes constantes
    if n_unique <= 1:
        return -100.0, ["Colonne constante"]
    if n_total > 20 and n_unique / n_total > 0.95 and not pd.api.types.is_float_dtype(series):
        return -50.0, ["Cardinalité trop élevée (identifiant)"]

    if n_unique == 2:
        score += 20.0
        reasons.append("Distribution binaire adaptée")
    elif 3 <= n_unique <= 20:
        score += 10.0
        reasons.append(f"Nombre de classes modéré ({n_unique})")
    elif pd.api.types.is_float_dtype(series):
        score += 15.0
        reasons.append("Variable numérique continue")

    return score, reasons


def suggest_target(df: pd.DataFrame) -> TargetProposal | None:
    """Propose la meilleure colonne cible avec ses explications et son type de problème."""
    if df.empty or df.shape[1] == 0:
        return None

    best_col = None
    best_score = -999.0
    best_reasons = []

    total_cols = len(df.columns)
    for idx, col in enumerate(df.columns):
        score, reasons = score_target_candidate(col, df[col], idx, total_cols)
        if score > best_score:
            best_score = score
            best_col = col
            best_reasons = reasons

    if best_col is None:
        best_col = df.columns[-1]
        best_reasons = ["Dernière colonne par défaut"]

    series = df[best_col]
    prob_type = detect_problem_type(series)
    
    n_miss = int(series.isna().sum())
    pct_miss = round(n_miss / len(df) * 100, 2) if len(df) > 0 else 0.0

    # Vérification du déséquilibre si classification
    classes_dist = None
    is_imbal = False
    imbal_warn = None
    
    if "classification" in prob_type:
        clean = series.dropna()
        counts = clean.value_counts()
        classes_dist = {str(k): int(v) for k, v in counts.items()}
        if len(counts) >= 2:
            min_share = counts.min() / counts.sum()
            if min_share < 0.20:
                is_imbal = True
                pct_min = round(min_share * 100, 1)
                imbal_warn = f"Classe minoritaire à {pct_min}% (déséquilibre détecté)."

    reason_str = ", ".join(best_reasons) if best_reasons else "Sélectionnée par défaut"

    return TargetProposal(
        column=best_col,
        problem_type=prob_type,
        reason=reason_str,
        missing_count=n_miss,
        missing_pct=pct_miss,
        classes_distribution=classes_dist,
        is_imbalanced=is_imbal,
        imbalance_warning=imbal_warn,
    )


def filter_missing_target(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, int]:
    """Exclut les observations dont la cible est manquante pour l'apprentissage supervisé."""
    if target_col not in df.columns:
        return df, 0
    missing_mask = df[target_col].isna()
    n_removed = int(missing_mask.sum())
    if n_removed > 0:
        df_filtered = df[~missing_mask].copy()
        return df_filtered, n_removed
    return df, 0
