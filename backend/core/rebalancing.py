"""Module de rééquilibrage des classes pour les données déséquilibrées.

Applique la stratégie de rééquilibrage UNIQUEMENT sur l'ensemble train (Anti-Data-Leakage).
Supporte : Aucune / Undersampling (RandomUnderSampler) / SMOTE (Synthetic Minority Oversampling Technique).

Formule SMOTE :
    nouveau = A + λ·(B − A), λ ∈ [0,1]
    où A est un point de la classe minoritaire et B un de ses k plus proches voisins.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd


def apply_rebalancing(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str = "none",
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """Applique la stratégie de rééquilibrage UNIQUEMENT sur l'ensemble train.

    Args:
        X_train: Features du train (déjà encodées et mises à l'échelle, 100% numériques).
        y_train: Labels du train.
        strategy: 'none' | 'undersample' | 'smote'.
        params: Paramètres supplémentaires : k_neighbors (défaut 5),
                sampling_strategy ('auto', 0.5, 1.0), random_state (défaut 42).

    Returns:
        (X_rebal, y_rebal, rebal_info) où rebal_info contient les comptages avant/après.
    """
    if params is None:
        params = {}

    random_state = int(params.get("random_state", 42))
    sampling_strategy = params.get("sampling_strategy", "auto")

    # Conversion en float si sampling_strategy est passé comme string numérique
    if isinstance(sampling_strategy, str) and sampling_strategy not in ("auto",):
        try:
            sampling_strategy = float(sampling_strategy)
        except ValueError:
            sampling_strategy = "auto"

    # Comptages avant rééquilibrage
    before_counts = {str(k): int(v) for k, v in y_train.value_counts().items()}
    n_before = len(y_train)

    if strategy == "none":
        return X_train, y_train, {
            "strategy": "none",
            "strategy_label": "Aucune",
            "before": before_counts,
            "after": before_counts,
            "n_before": n_before,
            "n_after": n_before,
            "k_neighbors_used": None,
            "note": "Aucune transformation appliquée. Le déséquilibre original est conservé.",
        }

    if strategy == "smote":
        # Vérification de l'absence de NaN avant SMOTE
        nan_count = int(X_train.isnull().sum().sum())
        if nan_count > 0:
            raise ValueError(
                f"SMOTE impossible : {nan_count} valeurs NaN détectées dans X_train. "
                "Assurez-vous que l'imputation est appliquée avant le rééquilibrage."
            )

        # Calcul du nombre de samples de la classe minoritaire dans le train
        min_class_count = int(y_train.value_counts().min())
        k_requested = int(params.get("k_neighbors", 5))

        # Auto-ajustement de k_neighbors : k doit être < min_class_count
        k_actual = min(k_requested, min_class_count - 1)
        k_actual = max(1, k_actual)  # Au moins 1 voisin

        if k_actual < k_requested:
            note = (
                f"k_neighbors automatiquement réduit de {k_requested} à {k_actual} "
                f"car la classe minoritaire ne contient que {min_class_count} exemples dans le train."
            )
        else:
            note = (
                f"SMOTE appliqué avec k_neighbors={k_actual}. "
                f"Formule : nouveau = A + λ·(B−A), λ ∈ [0,1]. "
                "Attention : sensible aux outliers et au chevauchement de classes."
            )

        try:
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(
                k_neighbors=k_actual,
                sampling_strategy=sampling_strategy,
                random_state=random_state,
            )
        except ImportError:
            raise ImportError(
                "Le package 'imbalanced-learn' est requis pour SMOTE. "
                "Installez-le avec : pip install imbalanced-learn"
            )

        X_rebal, y_rebal = sampler.fit_resample(X_train, y_train)
        X_rebal = pd.DataFrame(X_rebal, columns=X_train.columns)
        y_rebal = pd.Series(y_rebal, name=y_train.name)

        after_counts = {str(k): int(v) for k, v in y_rebal.value_counts().items()}
        n_after = len(y_rebal)

        return X_rebal, y_rebal, {
            "strategy": "smote",
            "strategy_label": "SMOTE",
            "before": before_counts,
            "after": after_counts,
            "n_before": n_before,
            "n_after": n_after,
            "k_neighbors_used": k_actual,
            "k_neighbors_requested": k_requested,
            "note": note,
        }

    elif strategy == "undersample":
        try:
            from imblearn.under_sampling import RandomUnderSampler
            sampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
            )
        except ImportError:
            raise ImportError(
                "Le package 'imbalanced-learn' est requis pour RandomUnderSampler. "
                "Installez-le avec : pip install imbalanced-learn"
            )

        X_rebal, y_rebal = sampler.fit_resample(X_train, y_train)
        X_rebal = pd.DataFrame(X_rebal, columns=X_train.columns)
        y_rebal = pd.Series(y_rebal, name=y_train.name)

        after_counts = {str(k): int(v) for k, v in y_rebal.value_counts().items()}
        n_after = len(y_rebal)

        return X_rebal, y_rebal, {
            "strategy": "undersample",
            "strategy_label": "Undersampling (RandomUnderSampler)",
            "before": before_counts,
            "after": after_counts,
            "n_before": n_before,
            "n_after": n_after,
            "k_neighbors_used": None,
            "note": (
                "Undersampling : les exemples de la classe majoritaire sont aléatoirement supprimés. "
                "Avantage : rapide. Inconvénient : perte d'information potentiellement importante."
            ),
        }

    else:
        raise ValueError(f"Stratégie de rééquilibrage inconnue : '{strategy}'. Utilisez 'none', 'smote' ou 'undersample'.")
