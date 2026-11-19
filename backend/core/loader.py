"""Module de chargement et conversion automatique de données CSV / Excel."""
from __future__ import annotations

import csv
import io
import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from backend.core.common import NA_TOKENS

CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]
CANDIDATE_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


def detect_csv_separator(text_sample: str) -> str:
    """Détecte le séparateur CSV à partir d'un échantillon de texte."""
    try:
        return csv.Sniffer().sniff(text_sample, delimiters="".join(CANDIDATE_DELIMITERS)).delimiter
    except Exception:
        lines = [l for l in text_sample.splitlines() if l.strip()][:30]
        if not lines:
            return ","
        scores = {d: sum(l.count(d) for l in lines) for d in CANDIDATE_DELIMITERS}
        return max(scores, key=scores.get)


def detect_encoding(raw_bytes: bytes) -> str:
    """Devine l'encodage du fichier parmi les encodages usuels."""
    for enc in CANDIDATE_ENCODINGS:
        try:
            raw_bytes[:100_000].decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def clean_and_coerce_column(series: pd.Series) -> pd.Series:
    """Convertit automatiquement les valeurs NA, les nombres en texte et les dates."""
    # Remplacer les jetons NA textuels par pd.NA
    s = series.replace(NA_TOKENS, pd.NA)
    non_na = s.dropna()
    if non_na.empty:
        return s

    # 1. Vérification si déjà numérique ou booléen
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return s

    str_sample = non_na.astype(str).str.strip()

    # 2. Tentative de conversion numérique (ex: " 1 200,50 € ", "3.14", "15%")
    # Remplacer espaces insécables et symboles monétaires fréquents
    num_cand = (
        str_sample.str.replace("\u00a0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    # Gérer la virgule décimale française vs point
    num_cand = num_cand.str.replace(",", ".", regex=False)
    converted_num = pd.to_numeric(num_cand, errors="coerce")
    if converted_num.notna().sum() / len(non_na) >= 0.90:
        # Convertir l'ensemble de la série
        full_cand = (
            s.astype("string")
            .str.strip()
            .str.replace("\u00a0", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace("€", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
        )
        return pd.to_numeric(full_cand, errors="coerce")

    # 3. Tentative de conversion date
    if str_sample.str.contains(r"[/\-:]", regex=True).mean() >= 0.70:
        try:
            parsed_dates = pd.to_datetime(str_sample.head(100), errors="coerce", format="mixed")
            if parsed_dates.notna().sum() / len(str_sample.head(100)) >= 0.85:
                # La colonne est une date
                return pd.to_datetime(s, errors="coerce", format="mixed")
        except Exception:
            pass

    return s


def infer_column_type(series: pd.Series, col_name: str) -> str:
    """Détermine le type sémantique d'une colonne :
    'numeric', 'categorical', 'boolean', 'date', 'id', 'text'.
    """
    clean = series.dropna()
    n_clean = len(clean)
    if n_clean == 0:
        return "categorical"

    # Booléen
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if clean.nunique() <= 2:
        str_vals = {str(v).strip().lower() for v in clean.unique()}
        if str_vals.issubset({"0", "1", "0.0", "1.0", "true", "false", "oui", "non", "yes", "no"}):
            return "boolean"

    # Date
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    # Identifiant (ex: transaction_id, client_id, ou 100% unique string)
    name_lower = col_name.lower()
    has_id_kw = any(kw in name_lower for kw in ["id", "uuid", "guid", "matricule", "code", "index", "ref"])
    if n_clean >= 20 and clean.nunique() / n_clean >= 0.95 and not pd.api.types.is_float_dtype(series):
        return "id"
    if has_id_kw and clean.nunique() / n_clean >= 0.60:
        return "id"

    # Numérique
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Texte libre (mots multiples)
    str_sample = clean.astype(str).head(100)
    avg_words = str_sample.str.split().str.len().mean()
    if avg_words >= 3.5:
        return "text"

    return "categorical"


def suggest_target_and_problem(df: pd.DataFrame) -> Tuple[Optional[str], str]:
    """Identifie automatiquement la colonne cible la plus probable et le type de tâche ML."""
    if df.empty or len(df.columns) == 0:
        return None, "unsupervised"

    target_keywords = ["target", "cible", "label", "is_fraud", "fraud", "churn", "achat", "survived", "prix", "price", "ventes", "score"]
    best_col = None
    best_score = -999.0

    total_cols = len(df.columns)
    for idx, col in enumerate(df.columns):
        score = 0.0
        clean_name = col.lower().strip()
        series = df[col].dropna()
        if series.empty:
            continue

        for kw in target_keywords:
            if clean_name == kw:
                score += 50.0
                break
            elif kw in clean_name:
                score += 25.0
                break

        # Position en dernière colonne (convention classique)
        if idx == total_cols - 1:
            score += 30.0
        elif idx == total_cols - 2:
            score += 10.0

        # Éviter les identifiants
        if series.nunique() / len(series) > 0.90 and not pd.api.types.is_numeric_dtype(series):
            score -= 50.0

        if score > best_score:
            best_score = score
            best_col = col

    best_col = best_col or df.columns[-1]
    series = df[best_col].dropna()

    # Détection classification vs régression
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 10 and (series % 1 == 0).all():
            prob_type = "classification"
        else:
            prob_type = "regression"
    else:
        prob_type = "classification"

    return best_col, prob_type


def load_dataset_from_bytes(file_bytes: bytes, filename: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Lit un fichier CSV ou Excel et applique les conversions automatiques de qualité."""
    fname_lower = filename.lower()
    
    if fname_lower.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes), na_values=NA_TOKENS)
        load_info = {"format": "Excel", "encoding": "N/A", "separator": "N/A"}
    else:
        encoding = detect_encoding(file_bytes)
        text_sample = file_bytes[:100_000].decode(encoding, errors="replace")
        separator = detect_csv_separator(text_sample)
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=separator,
            encoding=encoding,
            na_values=NA_TOKENS,
            skipinitialspace=True,
            on_bad_lines="skip",
        )
        load_info = {"format": "CSV", "encoding": encoding, "separator": separator}

    # Nettoyage des noms de colonnes
    df.columns = [str(c).strip() or f"col_{i}" for i, c in enumerate(df.columns)]
    # Dédoublonnage des noms de colonnes
    seen = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols

    # Conversion automatique des colonnes
    for col in df.columns:
        df[col] = clean_and_coerce_column(df[col])

    # Détection des types
    types = {col: infer_column_type(df[col], col) for col in df.columns}
    target_sugg, prob_sugg = suggest_target_and_problem(df)

    load_info.update({
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "types": types,
        "suggested_target": target_sugg,
        "suggested_problem": prob_sugg,
    })

    return df, load_info
