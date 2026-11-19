"""Analyse et configuration du traitement des valeurs manquantes."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class ColumnMissingInfo:
    column: str
    missing_count: int
    missing_pct: float
    recommended_strategy: str
    recommended_action: str  # 'imputer', 'supprimer_colonne', 'conserver'
    warning: str | None = None


NUMERIC_STRATEGIES = {
    "mediane": "Médiane (robuste aux valeurs extrêmes)",
    "moyenne": "Moyenne arithmétique",
    "constante": "Valeur constante (ex: 0)",
    "supprimer_colonne": "Supprimer la colonne entière",
    "aucune": "Conserver telles quelles (ignorer)",
}

CATEGORICAL_STRATEGIES = {
    "mode": "Modalité la plus fréquente (Mode)",
    "valeur_manquante": "Créer une catégorie 'Missing'",
    "constante": "Valeur constante personnalisée",
    "supprimer_colonne": "Supprimer la colonne entière",
    "aucune": "Conserver telles quelles (ignorer)",
}

DATE_STRATEGIES = {
    "mode": "Date la plus fréquente",
    "supprimer_colonne": "Supprimer la colonne",
    "aucune": "Conserver telles quelles",
}


def analyze_missing(
    df: pd.DataFrame,
    col_types: dict[str, str],
    threshold_drop_pct: float = 50.0,
) -> list[ColumnMissingInfo]:
    """Analyse les valeurs manquantes et génère des recommandations adaptées par colonne."""
    results: list[ColumnMissingInfo] = []
    n = len(df)
    if n == 0:
        return results

    for col in df.columns:
        s = df[col]
        miss_count = int(s.isna().sum())
        miss_pct = round((miss_count / n) * 100, 2)
        v_type = col_types.get(col, "categorielle_nominale")

        rec_strategy = "aucune"
        rec_action = "conserver"
        warning = None

        if miss_count > 0:
            if miss_pct >= threshold_drop_pct:
                rec_strategy = "supprimer_colonne"
                rec_action = "supprimer_colonne"
                warning = f"Taux très élevé ({miss_pct}%). Suppression recommandée."
            elif v_type in ("numerique_continue", "numerique_discrete"):
                rec_strategy = "mediane"
                rec_action = "imputer"
            elif v_type in ("categorielle_nominale", "categorielle_ordinale", "booleenne", "texte_libre"):
                rec_strategy = "valeur_manquante" if v_type == "categorielle_nominale" else "mode"
                rec_action = "imputer"
            elif v_type == "date_heure":
                rec_strategy = "mode"
                rec_action = "imputer"

        results.append(
            ColumnMissingInfo(
                column=col,
                missing_count=miss_count,
                missing_pct=miss_pct,
                recommended_strategy=rec_strategy,
                recommended_action=rec_action,
                warning=warning,
            )
        )

    return results
