"""Lecture robuste d'un fichier CSV : encodage, séparateur, conversion des types."""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field

import pandas as pd

try:
    import chardet
except ImportError:  # chardet est optionnel
    chardet = None

CANDIDATE_SEPARATORS = [",", ";", "\t", "|"]
FALLBACK_ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]


class CSVLoadError(Exception):
    """Erreur de lecture avec un message compréhensible pour l'utilisateur."""


@dataclass
class LoadInfo:
    encoding: str
    separator: str
    decimal: str
    header: bool
    converted_columns: list[str] = field(default_factory=list)


def detect_encoding(raw: bytes) -> str:
    """Devine l'encodage à partir des premiers octets."""
    sample = raw[:200_000]
    if chardet is not None:
        guess = chardet.detect(sample)
        if guess.get("encoding") and (guess.get("confidence") or 0) > 0.6:
            enc = guess["encoding"].lower()
            return "utf-8-sig" if enc in ("utf-8", "ascii") else enc
    for enc in FALLBACK_ENCODINGS:
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"


def detect_separator(text_sample: str) -> str:
    """Devine le séparateur avec csv.Sniffer, puis par comptage."""
    try:
        return csv.Sniffer().sniff(text_sample, delimiters="".join(CANDIDATE_SEPARATORS)).delimiter
    except csv.Error:
        lines = [l for l in text_sample.splitlines() if l.strip()][:20]
        if not lines:
            return ","
        scores = {s: sum(l.count(s) for l in lines) for s in CANDIDATE_SEPARATORS}
        return max(scores, key=scores.get)


def _coerce_numeric_text(df: pd.DataFrame, decimal: str, threshold: float = 0.95) -> tuple[pd.DataFrame, list[str]]:
    """Convertit les colonnes texte qui contiennent en fait des nombres."""
    converted = []
    for col in df.columns:
        if df[col].dtype != object:
            continue
        s = df[col].astype("string").str.strip().str.replace("\u00a0", "", regex=False).str.replace(" ", "", regex=False)
        if decimal == ",":
            s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        else:
            s = s.str.replace(",", "", regex=False)
        s = s.replace({"": pd.NA})
        non_null = s.notna().sum()
        if non_null == 0:
            continue
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().sum() / non_null >= threshold:
            df[col] = num
            converted.append(col)
    return df, converted


def _make_unique_columns(columns) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for i, c in enumerate(columns):
        name = str(c).strip() or f"colonne_{i}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def load_csv(
    raw: bytes,
    encoding: str | None = None,
    separator: str | None = None,
    decimal: str = ".",
    header: bool = True,
    coerce_numeric: bool = True,
) -> tuple[pd.DataFrame, LoadInfo]:
    """Charge un CSV depuis des octets. Les paramètres à None sont détectés."""
    if not raw or not raw.strip():
        raise CSVLoadError("Le fichier est vide.")

    encoding = encoding or detect_encoding(raw)
    try:
        text = raw.decode(encoding)
    except (UnicodeDecodeError, LookupError) as exc:
        raise CSVLoadError(
            f"Impossible de décoder le fichier avec l'encodage « {encoding} ». "
            "Essayez utf-8, latin-1 ou cp1252 dans les paramètres avancés."
        ) from exc

    separator = separator or detect_separator(text[:50_000])

    try:
        df = pd.read_csv(
            io.StringIO(text),
            sep=separator,
            header=0 if header else None,
            engine="python" if len(separator) > 1 else "c",
            on_bad_lines="warn",
            skip_blank_lines=True,
        )
    except pd.errors.EmptyDataError as exc:
        raise CSVLoadError("Aucune donnée exploitable n'a été trouvée dans le fichier.") from exc
    except pd.errors.ParserError as exc:
        raise CSVLoadError(
            f"Le fichier est mal formé avec le séparateur « {separator} » : {exc}. "
            "Vérifiez le séparateur dans les paramètres avancés."
        ) from exc
    except Exception as exc:  # garde-fou
        raise CSVLoadError(f"Erreur inattendue à la lecture : {exc}") from exc

    if df.empty or df.shape[1] == 0:
        raise CSVLoadError("Le fichier ne contient aucune ligne ou aucune colonne.")
    if df.shape[1] == 1 and separator != "\t":
        # Un seul champ : le séparateur est probablement faux
        raise CSVLoadError(
            "Une seule colonne a été détectée : le séparateur est probablement incorrect. "
            "Choisissez-en un autre dans les paramètres avancés."
        )

    if not header:
        df.columns = [f"colonne_{i}" for i in range(df.shape[1])]
    df.columns = _make_unique_columns(df.columns)

    converted: list[str] = []
    if coerce_numeric:
        df, converted = _coerce_numeric_text(df, decimal)

    return df, LoadInfo(encoding, separator, decimal, header, converted)


def dataset_summary(df: pd.DataFrame) -> dict:
    """Dimensions, doublons, mémoire et valeurs manquantes globales."""
    return {
        "lignes": int(df.shape[0]),
        "colonnes": int(df.shape[1]),
        "doublons": int(df.duplicated().sum()),
        "valeurs_manquantes": int(df.isna().sum().sum()),
        "memoire_Mo": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }


def dtypes_table(df: pd.DataFrame) -> pd.DataFrame:
    """Tableau type / manquants / valeurs uniques par colonne."""
    return pd.DataFrame(
        {
            "colonne": df.columns,
            "type": [str(t) for t in df.dtypes],
            "manquants": df.isna().sum().values,
            "% manquants": (df.isna().mean() * 100).round(2).values,
            "valeurs_uniques": df.nunique(dropna=True).values,
        }
    )