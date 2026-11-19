"""Analyse exploratoire : statistiques, distributions, manquants, doublons, outliers, corrélations."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .common import infer_kind

SAMPLE = 200_000  # au-delà, les calculs lourds (corrélations, associations) sont faits sur un échantillon


def _hist(v: np.ndarray, bins: int = 30):
    if len(v) == 0:
        return []
    counts, edges = np.histogram(v, bins=bins)
    return [{"x0": float(edges[i]), "x1": float(edges[i + 1]),
             "label": f"{edges[i]:.3g}", "count": int(counts[i])} for i in range(len(counts))]


def numeric_stats(s: pd.Series) -> dict:
    v = s.dropna().to_numpy(float)
    if len(v) == 0:
        return {}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out = int(((v < lo_f) | (v > hi_f)).sum()) if iqr > 0 else 0
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    z_out = int((np.abs((v - v.mean()) / sd) > 3).sum()) if sd > 0 else 0
    inside = v[(v >= lo_f) & (v <= hi_f)] if iqr > 0 else v
    whisk_lo = float(inside.min()) if len(inside) else float(v.min())
    whisk_hi = float(inside.max()) if len(inside) else float(v.max())
    p1, p99 = np.percentile(v, [1, 99])
    zoom = v[(v >= p1) & (v <= p99)]
    uniq = np.unique(v)
    out = {
        "count": int(len(v)), "mean": float(v.mean()), "std": sd, "min": float(v.min()),
        "q1": float(q1), "median": float(med), "q3": float(q3), "max": float(v.max()),
        "skew": float(s.skew()) if len(v) > 2 else 0.0, "kurtosis": float(s.kurt()) if len(v) > 3 else 0.0,
        "zeros": int((v == 0).sum()), "negatives": int((v < 0).sum()),
        "lower_fence": float(lo_f), "upper_fence": float(hi_f),
        "outliers_iqr": n_out, "outliers_iqr_pct": n_out / len(v) * 100,
        "outliers_z": z_out, "p1": float(p1), "p99": float(p99),
        "whisker_lo": whisk_lo, "whisker_hi": whisk_hi,
        "integer_like": bool(np.all(np.isclose(v % 1, 0))),
        "is_binary": bool(len(uniq) == 2),
        "top_share": float(pd.Series(v).value_counts(normalize=True).iloc[0]),
    }
    if len(uniq) <= 25:
        vc = pd.Series(v).value_counts().sort_index()
        out["discrete"] = [{"label": f"{k:g}", "count": int(c)} for k, c in vc.items()]
    out["hist"] = _hist(v)
    out["hist_zoom"] = _hist(zoom) if len(zoom) > 5 and p99 > p1 else _hist(v)
    return out


def categorical_stats(s: pd.Series) -> dict:
    nn = s.dropna().astype(str)
    if nn.empty:
        return {}
    vc = nn.value_counts()
    n = len(nn)
    norm = nn.str.strip().str.lower().nunique()
    top = [{"value": str(k), "count": int(c), "pct": c / n * 100} for k, c in vc.head(20).items()]
    return {
        "top": top, "top_share": float(vc.iloc[0] / n),
        "rare_count": int((vc / n < 0.01).sum()),
        "normalized_unique": int(norm),
        "case_issue": bool(norm < len(vc)),
        "avg_len": float(nn.str.len().mean()),
    }


def association(x: pd.Series, kind: str, y: pd.Series, problem: str) -> float | None:
    """Force de lien feature ↔ cible, dans [0, 1] (|Pearson|, rapport de corrélation η, Cramér V)."""
    try:
        d = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(d) < 10:
            return None
        if len(d) > SAMPLE:
            d = d.sample(SAMPLE, random_state=0)
        if kind == "datetime":
            return None
        if problem == "regression":
            if kind == "numeric":
                return abs(float(d["x"].astype(float).corr(d["y"].astype(float))))
            return _eta(d["y"].astype(float), d["x"].astype(str))
        if kind == "numeric":
            return _eta(d["x"].astype(float), d["y"].astype(str))
        ct = pd.crosstab(d["x"].astype(str), d["y"].astype(str))
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return 0.0
        from scipy.stats import chi2_contingency
        chi2 = chi2_contingency(ct, correction=False)[0]
        return float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))))
    except Exception:
        return None


def _eta(values: pd.Series, groups: pd.Series) -> float:
    tot = ((values - values.mean()) ** 2).sum()
    if tot == 0:
        return 0.0
    g = values.groupby(groups.values)
    between = (g.count() * (g.mean() - values.mean()) ** 2).sum()
    return float(np.sqrt(between / tot))


def correlations(df: pd.DataFrame, kinds: dict, method: str = "pearson", max_cols: int = 30) -> dict:
    cols = [c for c in df.columns if kinds[c] == "numeric" and df[c].nunique() > 1]
    cols = sorted(cols, key=lambda c: df[c].isna().mean())[:max_cols]
    if len(cols) < 2:
        return {"columns": cols, "matrix": [], "pairs": []}
    d = df[cols]
    if len(d) > SAMPLE:
        d = d.sample(SAMPLE, random_state=0)
    m = d.corr(method=method)
    pairs = []
    for i, a in enumerate(cols):
        for j in range(i + 1, len(cols)):
            v = m.iloc[i, j]
            if pd.notna(v):
                pairs.append({"a": a, "b": cols[j], "r": float(v)})
    pairs.sort(key=lambda p: -abs(p["r"]))
    return {"columns": cols, "matrix": m.round(3).fillna(0).values.tolist(), "pairs": pairs[:25], "method": method,
            "sampled": len(df) > SAMPLE}


def compute_eda(df: pd.DataFrame, target: str | None = None, problem: str | None = None) -> dict:
    n = len(df)
    kinds = {c: infer_kind(df[c]) for c in df.columns}
    feats = []
    for c in df.columns:
        s, k = df[c], kinds[c]
        nn = int(s.notna().sum())
        f = {
            "name": c, "kind": k, "dtype": str(s.dtype), "count": nn,
            "missing": n - nn, "missing_pct": (n - nn) / n * 100 if n else 0,
            "unique": int(s.nunique()), "unique_pct": s.nunique() / nn * 100 if nn else 0,
            "is_target": c == target,
        }
        if k == "numeric":
            f["stats"] = numeric_stats(s)
        elif k == "datetime":
            nn_s = s.dropna()
            f["stats"] = {"min": str(nn_s.min()), "max": str(nn_s.max()),
                          "span_days": float((nn_s.max() - nn_s.min()).days) if nn else 0}
        else:
            f["stats"] = categorical_stats(s)
        if target and c != target and problem:
            f["target_assoc"] = association(s, k, df[target], problem)
        feats.append(f)

    dup_mask = df.duplicated(keep="first")
    dups = int(dup_mask.sum())
    miss_rows = df.isna().any(axis=1)
    constants = [f["name"] for f in feats if f["unique"] <= 1]
    overview = {
        "rows": n, "cols": len(df.columns), "memory_mb": float(df.memory_usage(deep=True).sum() / 1e6),
        "kinds": {k: sum(1 for v in kinds.values() if v == k) for k in ("numeric", "categorical", "datetime", "boolean")},
        "missing_cells": int(df.isna().sum().sum()),
        "missing_cells_pct": float(df.isna().sum().sum() / max(df.size, 1) * 100),
        "rows_with_missing": int(miss_rows.sum()),
        "rows_with_missing_pct": float(miss_rows.mean() * 100) if n else 0,
        "duplicates": dups, "duplicates_pct": dups / n * 100 if n else 0,
        "constant_columns": constants,
    }
    dup_sample = df[df.duplicated(keep=False)].head(8).astype(str).to_dict("records") if dups else []
    out = {"overview": overview, "features": feats, "corr": correlations(df, kinds),
           "duplicates_sample": dup_sample, "target": None}
    if target:
        y = df[target]
        t = {"name": target, "problem": problem}
        if problem == "classification":
            vc = y.value_counts(dropna=True)
            tot = vc.sum()
            t["classes"] = [{"value": str(k), "count": int(c), "pct": c / tot * 100} for k, c in vc.items()][:30]
            t["imbalance_ratio"] = float(vc.max() / vc.min()) if len(vc) > 1 and vc.min() > 0 else None
        out["target"] = t
    return out
