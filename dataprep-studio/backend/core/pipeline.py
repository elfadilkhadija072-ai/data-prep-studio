"""Exécution des transformations SANS data leakage.

Principe : split train/test d'abord, puis TOUTES les statistiques (moyenne, médiane, bornes d'outliers,
scalers, encodeurs…) sont apprises sur le train uniquement et simplement appliquées au test.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (MinMaxScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer,
                                   RobustScaler, StandardScaler)
from .common import empty_plan, infer_kind

SEED = 42


def make_split(df: pd.DataFrame, target: str | None, problem: str | None, test_size: float):
    """Retourne (index_train, index_test|None). Stratifié pour la classification."""
    if not target or test_size <= 0:
        return df.index, None
    strat = None
    if problem == "classification":
        vc = df[target].value_counts()
        if vc.min() >= 2:
            strat = df[target]
    tr, te = train_test_split(df.index, test_size=test_size, random_state=SEED, shuffle=True, stratify=strat)
    return df.index[df.index.isin(tr)], df.index[df.index.isin(te)]


# ───────────────────────── numérique ─────────────────────────
def _numeric(name, plan, tr, te, info):
    x = pd.to_numeric(tr, errors="coerce").astype(float)
    z = pd.to_numeric(te, errors="coerce").astype(float) if te is not None else None
    P = info["params"]

    imp = plan["impute"]
    if imp != "none":
        v = {"mean": x.mean, "median": x.median}.get(imp)
        if imp == "mode":
            m = x.mode(); val = m.iloc[0] if len(m) else 0.0
        elif imp == "constant":
            val = 0.0
        else:
            val = v()
        val = 0.0 if pd.isna(val) else float(val)
        info["imputed"] = int(x.isna().sum() + (z.isna().sum() if z is not None else 0))
        x = x.fillna(val); z = z.fillna(val) if z is not None else None
        P["imputation"] = {"méthode": imp, "valeur_apprise_sur_train": val}
        info["steps"].append(f"Imputation {imp} (valeur={val:.6g})")

    remove = None
    out = plan["outliers"]
    if out != "none":
        c = x.dropna()
        if len(c):
            if out in ("clip_iqr", "remove_rows"):
                q1, q3 = c.quantile([.25, .75]); iqr = q3 - q1
                lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            elif out == "clip_zscore":
                lo, hi = c.mean() - 3 * c.std(), c.mean() + 3 * c.std()
            else:
                lo, hi = c.quantile([.01, .99])
            if out == "remove_rows":
                remove = (x < lo) | (x > hi)
                info["removed_rows"] = int(remove.sum())
                info["steps"].append(f"Suppression de {info['removed_rows']} lignes outliers (train uniquement)")
            else:
                info["clipped"] = int(((x < lo) | (x > hi)).sum() + (((z < lo) | (z > hi)).sum() if z is not None else 0))
                x = x.clip(lo, hi); z = z.clip(lo, hi) if z is not None else None
                info["steps"].append(f"Winsorisation [{lo:.4g} ; {hi:.4g}] ({info['clipped']} valeurs)")
            P["bornes_outliers"] = {"min": float(lo), "max": float(hi), "méthode": out}
    fit = ~remove if remove is not None else pd.Series(True, index=x.index)

    t = plan["transform"]
    if t in ("log1p", "sqrt") and x[fit].min() < 0:
        info["warnings"].append(f"{t} impossible (valeurs négatives) → Yeo-Johnson utilisé à la place")
        t = "yeojohnson"
    if t == "log1p":
        x = np.log1p(x); z = np.log1p(z.clip(lower=0)) if z is not None else None
        info["steps"].append("log(1+x)")
    elif t == "sqrt":
        x = np.sqrt(x); z = np.sqrt(z.clip(lower=0)) if z is not None else None
        info["steps"].append("racine carrée")
    elif t == "yeojohnson":
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pt.fit(x[fit].dropna().to_numpy().reshape(-1, 1))
            x = pd.Series(pt.transform(x.to_numpy().reshape(-1, 1)).ravel(), index=x.index)
            if z is not None:
                z = pd.Series(pt.transform(z.to_numpy().reshape(-1, 1)).ravel(), index=z.index)
        P["yeojohnson_lambda"] = float(pt.lambdas_[0])
        info["steps"].append(f"Yeo-Johnson (λ={pt.lambdas_[0]:.3f})")

    s = plan["scale"]
    if s != "none":
        sc = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}[s]()
        fv = x[fit].dropna()
        if len(fv):
            sc.fit(fv.to_numpy().reshape(-1, 1))
            x = pd.Series(sc.transform(x.to_numpy().reshape(-1, 1)).ravel(), index=x.index)
            if z is not None:
                z = pd.Series(sc.transform(z.to_numpy().reshape(-1, 1)).ravel(), index=z.index)
            P["scaler"] = {"type": s, **({"mean": float(sc.mean_[0]), "std": float(sc.scale_[0])} if s == "standard" else
                           {"min": float(sc.data_min_[0]), "max": float(sc.data_max_[0])} if s == "minmax" else
                           {"median": float(sc.center_[0]), "iqr": float(sc.scale_[0])})}
            info["steps"].append({"standard": "Standardisation", "minmax": "Min-Max", "robust": "Échelle robuste"}[s])
    otr = pd.DataFrame({name: x.values}, index=tr.index)
    ote = pd.DataFrame({name: z.values}, index=te.index) if z is not None else None
    return otr, ote, remove


# ───────────────────────── catégoriel ─────────────────────────
def _to_obj(s: pd.Series, clean: str) -> pd.Series:
    st = s.astype("string")
    if clean == "strip_lower":
        st = st.str.strip().str.lower()
    return st.astype(object).where(st.notna(), np.nan)


def _categorical(name, plan, tr, te, info):
    x = _to_obj(tr, plan["clean"]); z = _to_obj(te, plan["clean"]) if te is not None else None
    P = info["params"]
    if plan["clean"] != "none":
        info["steps"].append("Nettoyage texte (strip + minuscules)")
    imp = plan["impute"]
    if imp != "none":
        m = x.mode()
        val = (m.iloc[0] if len(m) else "Missing") if imp == "mode" else "Missing"
        info["imputed"] = int(x.isna().sum() + (z.isna().sum() if z is not None else 0))
        x = x.fillna(val); z = z.fillna(val) if z is not None else None
        P["imputation"] = {"méthode": imp, "valeur_apprise_sur_train": str(val)}
        info["steps"].append(f"Imputation {imp} (« {val} »)")
    e = plan["encode"]
    if e == "onehot":
        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, sparse_output=False)
        ohe.fit(x.to_frame())
        cols = [f"{name}__{c[len(name) + 1:]}" for c in ohe.get_feature_names_out([name])]
        otr = pd.DataFrame(ohe.transform(x.to_frame()), index=tr.index, columns=cols)
        ote = pd.DataFrame(ohe.transform(z.to_frame()), index=te.index, columns=cols) if z is not None else None
        P["one_hot"] = {"modalités": [str(c) for c in ohe.categories_[0]], "colonnes_créées": len(cols)}
        info["steps"].append(f"One-Hot ({len(cols)} colonnes)")
        return otr, ote, None
    if e == "ordinal":
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
        oe.fit(x.to_frame())
        otr = pd.DataFrame({name: oe.transform(x.to_frame()).ravel()}, index=tr.index)
        ote = pd.DataFrame({name: oe.transform(z.to_frame()).ravel()}, index=te.index) if z is not None else None
        P["ordinal"] = {str(c): i for i, c in enumerate(oe.categories_[0])}
        info["steps"].append("Encodage ordinal")
        return otr, ote, None
    if e == "frequency":
        fr = x.value_counts(normalize=True)
        otr = pd.DataFrame({name: x.map(fr).astype(float).values}, index=tr.index)
        ote = pd.DataFrame({name: z.map(fr).fillna(0.0).astype(float).values}, index=te.index) if z is not None else None
        P["frequency"] = {"modalités_apprises": int(len(fr))}
        info["steps"].append("Encodage par fréquence (appris sur train)")
        return otr, ote, None
    return x.to_frame(name), (z.to_frame(name) if z is not None else None), None


# ───────────────────────── date ─────────────────────────
def _datetime(name, plan, tr, te, info):
    d = pd.to_datetime(tr, errors="coerce"); e = pd.to_datetime(te, errors="coerce") if te is not None else None
    if plan["encode"] != "parts":
        return d.to_frame(name), (e.to_frame(name) if e is not None else None), None
    use_hour = bool((d.dt.hour.fillna(0) != 0).any())
    parts = {"year": lambda s: s.dt.year, "month": lambda s: s.dt.month, "day": lambda s: s.dt.day,
             "dayofweek": lambda s: s.dt.dayofweek}
    if use_hour:
        parts["hour"] = lambda s: s.dt.hour
    otr, ote = pd.DataFrame(index=tr.index), (pd.DataFrame(index=te.index) if e is not None else None)
    for p, fn in parts.items():
        a = fn(d).astype(float)
        m = a.mode(); fill = float(m.iloc[0]) if len(m) else 0.0   # mode appris sur train
        otr[f"{name}_{p}"] = a.fillna(fill)
        if ote is not None:
            ote[f"{name}_{p}"] = fn(e).astype(float).fillna(fill)
    info["steps"].append(f"Extraction des composantes date ({', '.join(parts)})")
    return otr, ote, None


# ───────────────────────── cible ─────────────────────────
def _target(name, plan, tr, te, info, problem):
    if problem == "classification":
        if plan["encode"] == "label":
            codes, uniques = pd.factorize(tr.astype(str), sort=True)
            mp = {u: i for i, u in enumerate(uniques)}
            otr = pd.DataFrame({name: codes}, index=tr.index)
            ote = pd.DataFrame({name: te.astype(str).map(mp).fillna(-1).astype(int).values}, index=te.index) if te is not None else None
            info["params"]["label_encoding"] = mp
            info["steps"].append(f"Label Encoding ({len(mp)} classes)")
            return otr, ote, None
        return tr.to_frame(name), (te.to_frame(name) if te is not None else None), None
    x = pd.to_numeric(tr, errors="coerce"); z = pd.to_numeric(te, errors="coerce") if te is not None else None
    if plan["transform"] == "log1p":
        x = np.log1p(x.clip(lower=0)); z = np.log1p(z.clip(lower=0)) if z is not None else None
        info["steps"].append("log(1+y) – inverser avec expm1 après prédiction")
    elif plan["transform"] == "sqrt":
        x = np.sqrt(x.clip(lower=0)); z = np.sqrt(z.clip(lower=0)) if z is not None else None
        info["steps"].append("racine carrée de y")
    if plan["scale"] == "standard":
        mu, sd = x.mean(), x.std() or 1.0
        x = (x - mu) / sd; z = (z - mu) / sd if z is not None else None
        info["params"]["target_scaler"] = {"mean": float(mu), "std": float(sd)}
        info["steps"].append("Standardisation de y")
    return x.to_frame(name), (z.to_frame(name) if z is not None else None), None


# ───────────────────────── orchestrateur ─────────────────────────
def merged_plan(rec_plan: dict, choice: dict | None) -> dict:
    p = empty_plan(); p.update(rec_plan)
    if choice:
        p.update({k: v for k, v in choice.items() if k in p})
    return p


def process_feature(name, kind, plan, tr, te, role, problem):
    info = {"feature": name, "kind": kind, "steps": [], "params": {}, "new_columns": [], "warnings": [],
            "imputed": 0, "clipped": 0, "removed_rows": 0, "dropped": False}
    if plan.get("drop") and role != "target":
        info["dropped"] = True
        info["steps"].append("Colonne supprimée")
        return None, None, info, None
    if role == "target":
        otr, ote, rm = _target(name, plan, tr, te, info, problem)
    elif kind == "numeric":
        otr, ote, rm = _numeric(name, plan, tr, te, info)
    elif kind == "datetime":
        otr, ote, rm = _datetime(name, plan, tr, te, info)
    else:
        if kind == "boolean":
            tr = tr.map({True: "True", False: "False"}); te = te.map({True: "True", False: "False"}) if te is not None else None
        otr, ote, rm = _categorical(name, plan, tr, te, info)
    info["new_columns"] = list(otr.columns)
    return otr, ote, info, rm


def summarize_columns(frame: pd.DataFrame) -> dict:
    out = {}
    for c in frame.columns:
        s = frame[c]
        d = {"missing": int(s.isna().sum()), "unique": int(s.nunique())}
        if pd.api.types.is_numeric_dtype(s):
            d.update(mean=float(s.mean()), std=float(s.std()), min=float(s.min()), max=float(s.max()),
                     skew=float(s.skew()) if s.notna().sum() > 2 else None)
        out[c] = d
    return out


def run_pipeline(df, kinds, recs, cfg, choices):
    """cfg: {task, target, problem, test_size, drop_duplicates}. Retourne dict de résultats."""
    target, problem, supervised = cfg.get("target"), cfg.get("problem"), cfg["task"] == "supervised"
    glog = []
    work = df
    if cfg.get("drop_duplicates"):
        before = len(work); work = work.drop_duplicates()
        glog.append(f"{before - len(work)} lignes dupliquées supprimées AVANT le split (évite les copies train/test)")
    if supervised and target:
        before = len(work); work = work[work[target].notna()]
        if before != len(work):
            glog.append(f"{before - len(work)} lignes sans cible supprimées")
    tr_idx, te_idx = make_split(work, target if supervised else None, problem, cfg.get("test_size", 0.2) if supervised else 0)
    train = work.loc[tr_idx]; test = work.loc[te_idx] if te_idx is not None else None
    glog.append(f"Split {'stratifié ' if supervised and problem == 'classification' else ''}train={len(train)} / "
                f"test={len(test) if test is not None else 0} (seed={SEED}) effectué AVANT toute transformation")
    glog.append("Toutes les statistiques sont apprises sur le train uniquement (fit) puis appliquées au test (transform)")

    infos, otrs, otes, remove_mask = {}, [], [], pd.Series(False, index=train.index)
    for c in work.columns:
        role = "target" if (supervised and c == target) else "feature"
        rec = recs["features"][c]
        plan = merged_plan(rec["plan"], (choices or {}).get(c))
        otr, ote, info, rm = process_feature(c, kinds[c], plan, train[c], test[c] if test is not None else None, role, problem)
        info["plan"] = plan
        infos[c] = info
        if otr is None:
            continue
        otrs.append(otr)
        if ote is not None:
            otes.append(ote)
        if rm is not None:
            remove_mask |= rm.reindex(train.index, fill_value=False)
    train_out = pd.concat(otrs, axis=1)
    test_out = pd.concat(otes, axis=1) if otes else None
    n_removed = int(remove_mask.sum())
    if n_removed:
        train_out = train_out.loc[~remove_mask]
        glog.append(f"{n_removed} lignes outliers retirées du TRAIN uniquement (le test reste intact)")
    after = summarize_columns(train_out)
    return {"train": train_out, "test": test_out, "infos": infos, "global_log": glog,
            "rows_before": len(df), "cols_before": df.shape[1], "rows_train": len(train_out),
            "rows_test": 0 if test_out is None else len(test_out), "cols_after": train_out.shape[1],
            "after_stats": after, "removed_rows": n_removed,
            "remaining_missing": int(train_out.isna().sum().sum() + (test_out.isna().sum().sum() if test_out is not None else 0))}


def preview_feature(df, kinds, recs, cfg, name, plan):
    """Aperçu avant/après d'UNE feature (fit sur le train du split courant, transform sur tout)."""
    target, problem, supervised = cfg.get("target"), cfg.get("problem"), cfg["task"] == "supervised"
    tr_idx, _ = make_split(df, target if supervised else None, problem, cfg.get("test_size", 0.2) if supervised else 0)
    tr = df.loc[tr_idx, name]
    role = "target" if (supervised and name == target) else "feature"
    p = merged_plan(recs["features"][name]["plan"], plan)
    otr, _, info, rm = process_feature(name, kinds[name], p, tr, None, role, problem)
    return tr, otr, info, rm
