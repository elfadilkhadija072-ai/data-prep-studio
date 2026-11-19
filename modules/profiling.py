"""Analyse et classification automatique des variables d'un dataset."""
from __future__ import annotations

import re
from typing import Any
import numpy as np
import pandas as pd


TYPE_LABELS = {
    "numerique_continue": "Numérique continue",
    "numerique_discrete": "Numérique discrète",
    "categorielle_nominale": "Catégorielle nominale",
    "categorielle_ordinale": "Catégorielle ordinale",
    "booleenne": "Booléenne",
    "date_heure": "Date et heure",
    "texte_libre": "Texte libre",
    "identifiant": "Identifiant / Clé unique",
    "constante": "Constante / Quasi constante",
}

BOOLEAN_VALS = {
    "true", "false", "0", "1", "0.0", "1.0", "oui", "non", "yes", "no", "vrai", "faux", "t", "f", "y", "n"
}

ID_KEYWORDS = {"id", "uuid", "guid", "identifiant", "code", "matricule", "reference", "ref", "index", "key"}


def is_date_series(series: pd.Series, sample_size: int = 100) -> bool:
    """Détecte si une série texte représente des dates."""
    if series.dtype == "datetime64[ns]" or isinstance(series.dtype, pd.DatetimeTZDtype):
        return True
    
    clean = series.dropna()
    if clean.empty or clean.dtype not in (object, "string"):
        return False
    
    sample = clean.astype(str).str.strip().head(sample_size)
    # Vérifier s'il y a des séparateurs typiques de dates
    has_date_symbols = sample.str.contains(r"[/\-:]", regex=True).mean()
    if has_date_symbols < 0.7:
        return False

    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return (parsed.notna().sum() / len(sample)) >= 0.8
    except Exception:
        return False


def is_boolean_series(series: pd.Series) -> bool:
    """Détecte si une série est binaire / booléenne."""
    if series.dtype == bool or str(series.dtype).startswith("boolean"):
        return True
    
    clean = series.dropna()
    if clean.empty:
        return False
    
    unique_vals = set(clean.unique())
    if len(unique_vals) <= 2:
        str_vals = {str(v).strip().lower() for v in unique_vals}
        if str_vals.issubset(BOOLEAN_VALS):
            return True
        if {0, 1}.issuperset(unique_vals):
            return True
    return False


def is_identifier_series(series: pd.Series, col_name: str) -> bool:
    """Détecte les identifiants uniques ou quasi uniques."""
    name_clean = re.sub(r"[^a-zA-Z0-9]", "_", col_name.lower()).strip("_")
    name_tokens = set(name_clean.split("_"))
    has_id_name = bool(name_tokens & ID_KEYWORDS) or name_clean.endswith("id") or name_clean.startswith("id_")
    
    clean = series.dropna()
    n = len(clean)
    if n == 0:
        return False
    
    ratio = clean.nunique() / n
    # Si le nom évoque un ID et cardinalité très forte (>80%)
    if has_id_name and ratio >= 0.7:
        return True
    # Si cardinalité presque parfaite et dataset significatif
    if n >= 20 and ratio >= 0.98 and not pd.api.types.is_float_dtype(series):
        return True
    return False


def is_free_text_series(series: pd.Series, sample_size: int = 100) -> bool:
    """Détecte les colonnes de texte libre (descriptions, commentaires)."""
    clean = series.dropna()
    if clean.empty or clean.dtype not in (object, "string"):
        return False
    
    sample = clean.astype(str).head(sample_size)
    mean_length = sample.str.len().mean()
    mean_words = sample.str.split().str.len().mean()
    
    # Texte long ou contenant plusieurs phrases / mots
    if mean_words >= 4 or mean_length >= 40:
        return True
    return False


def detect_variable_type(series: pd.Series, col_name: str) -> str:
    """Attribue la catégorie la plus probable à une série pandas."""
    clean = series.dropna()
    n_total = len(series)
    n_clean = len(clean)

    # 1. Constante ou quasi constante
    if n_clean == 0:
        return "constante"
    if clean.nunique() <= 1:
        return "constante"
    most_freq_share = clean.value_counts(normalize=True).iloc[0]
    if most_freq_share >= 0.995 and n_clean >= 50:
        return "constante"

    # 2. Booléenne
    if is_boolean_series(series):
        return "booleenne"

    # 3. Date et heure (testé avant identifiant pour ne pas confondre dates uniques et ID)
    if is_date_series(series):
        return "date_heure"

    # 4. Identifiant
    if is_identifier_series(series, col_name):
        return "identifiant"

    # 5. Numérique
    if pd.api.types.is_numeric_dtype(series):
        # Entier ou flottant avec peu de modalités -> discrète
        n_unique = clean.nunique()
        if n_unique <= 15 and (clean % 1 == 0).all():
            return "numerique_discrete"
        return "numerique_continue"

    # 6. Texte libre
    if is_free_text_series(series):
        return "texte_libre"

    # 7. Par défaut pour objet / string : catégorielle
    return "categorielle_nominale"


def classify_dataframe(df: pd.DataFrame) -> dict[str, str]:
    """Classifie automatiquement toutes les colonnes d'un DataFrame."""
    return {col: detect_variable_type(df[col], col) for col in df.columns}


def get_column_stats(df: pd.DataFrame, classifications: dict[str, str]) -> pd.DataFrame:
    """Génère un tableau de synthèse pour l'interface de vérification des types."""
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        n_missing = int(s.isna().sum())
        pct_missing = round((n_missing / n * 100), 2) if n > 0 else 0.0
        n_unique = int(s.nunique(dropna=True))
        inferred = classifications.get(col, "categorielle_nominale")
        
        sample_vals = ", ".join(map(str, s.dropna().unique()[:3]))
        if len(sample_vals) > 50:
            sample_vals = sample_vals[:47] + "…"

        rows.append({
            "colonne": col,
            "type_detecte": inferred,
            "manquants": n_missing,
            "% manquants": pct_missing,
            "valeurs_uniques": n_unique,
            "exemples": sample_vals,
        })
    return pd.DataFrame(rows)
