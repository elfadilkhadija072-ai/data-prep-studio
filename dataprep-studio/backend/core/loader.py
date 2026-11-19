"""Lecture CSV / Excel + normalisation automatique des types (texte→numérique, texte→date)."""
from __future__ import annotations
import csv, io, re
import numpy as np
import pandas as pd

NA_TOKENS = {"", "na", "n/a", "nan", "null", "?", "#n/a", "#na"}
DATE_RE = re.compile(r"^\s*\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?\s*$")


def read_any(filename: str, content: bytes) -> pd.DataFrame:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext in ("xlsx", "xlsm", "xls"):
        return pd.read_excel(io.BytesIO(content))
    if ext not in ("csv", "txt", "tsv", ""):
        raise ValueError("Format non supporté : utilisez .csv, .xlsx ou .xls")
    sample = content[:50_000].decode("utf-8", errors="replace")
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        sep = ","
    last = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(content), sep=sep, encoding=enc, low_memory=False)
        except UnicodeDecodeError as e:
            last = e
    raise ValueError(f"Encodage illisible : {last}")


def normalize(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Nettoie les noms de colonnes et convertit les colonnes texte quand c'est évident.
    Retourne (df, notes) où notes[col] explique chaque conversion automatique."""
    notes: dict[str, str] = {}
    df = df.copy()
    cols, seen = [], {}
    for c in df.columns:
        c = str(c).strip() or "colonne"
        seen[c] = seen.get(c, 0) + 1
        cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    df.columns = cols

    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
            continue
        st = s.astype("string").str.strip()
        st = st.mask(st.str.lower().isin(NA_TOKENS))
        nn = st.dropna()
        if nn.empty:
            df[c] = pd.Series(np.nan, index=df.index, dtype=float)
            continue
        num = pd.to_numeric(nn, errors="coerce")
        if num.notna().mean() >= 0.95:
            df[c] = pd.to_numeric(st, errors="coerce")
            notes[c] = "Colonne numérique stockée en texte → convertie en nombre"
            continue
        probe = nn.iloc[: min(len(nn), 500)]
        if probe.str.match(DATE_RE).mean() > 0.9:
            parsed = pd.to_datetime(st, errors="coerce", format="mixed")
            if parsed.notna().sum() / len(nn) > 0.9:
                df[c] = parsed
                notes[c] = "Format date détecté → converti en datetime"
                continue
        low = nn.str.lower().unique()
        if len(low) <= 2 and set(low) <= {"true", "false", "yes", "no", "oui", "non", "y", "n"}:
            pass  # reste catégoriel binaire (encodage ordinal 0/1 recommandé)
        df[c] = st.astype(object).where(st.notna(), np.nan)
    return df, notes
