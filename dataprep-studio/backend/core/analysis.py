"""Moteur de recommandations : anomalies détectées + plan de transformation justifié statistiquement."""
from __future__ import annotations
import re
from .common import OPTIONS, SLOT_ORDER, SEVERITY_ORDER, empty_plan, slots_key

ID_NAME = re.compile(r"(^|[^a-z])(id|uuid|guid|index|idx|key)([^a-z]|$)|^unnamed", re.I)


def _a(code, sev, title, detail):
    return {"code": code, "severity": sev, "title": title, "detail": detail}


def build_recommendations(df, eda: dict, target: str | None, problem: str | None) -> dict:
    n = eda["overview"]["rows"]
    feats = {f["name"]: f for f in eda["features"]}
    pairs = eda["corr"]["pairs"]

    # colonnes redondantes : |r| ≥ 0.95 → on garde la première, on propose de supprimer la seconde
    redundant: dict[str, tuple[str, float]] = {}
    for p in pairs:
        if abs(p["r"]) >= 0.95:
            a, b = p["a"], p["b"]
            if target in (a, b) or a in redundant or b in redundant:
                continue
            redundant[b] = (a, p["r"])

    recs = {}
    for name, f in feats.items():
        role = "target" if name == target else "feature"
        kind = f["kind"]
        st = f.get("stats", {})
        plan, why, an = empty_plan(), {}, []
        mp = f["missing_pct"]

        # ───────── Cible ─────────
        if role == "target":
            if f["missing"]:
                an.append(_a("target_missing", "crit", "Cible manquante",
                             f"{f['missing']} lignes ({mp:.2f} %) sans cible : elles seront retirées avant le split (inutilisables en supervisé)."))
            if problem == "classification":
                t = eda["target"]
                ratio = t.get("imbalance_ratio")
                minpct = min(c["pct"] for c in t["classes"])
                if ratio and ratio >= 3:
                    sev = "crit" if minpct < 5 else "warn"
                    an.append(_a("imbalance", sev, "Classes déséquilibrées",
                                 f"Classe minoritaire = {minpct:.2f} % (ratio majoritaire/minoritaire = {ratio:.1f}). "
                                 "Split stratifié appliqué ; prévoir class_weight / SMOTE / métriques PR-AUC hors du preprocessing."))
                if f["dtype"] in ("object", "str", "string") or kind == "categorical":
                    plan["encode"] = "label"
                    why["encode"] = f"Cible textuelle ({f['unique']} classes) → Label Encoding, requis par scikit-learn."
                else:
                    why["encode"] = "Cible déjà numérique : aucun encodage nécessaire."
            else:
                sk = st.get("skew", 0)
                if abs(sk) > 2 and st.get("min", 0) >= 0:
                    plan["transform"] = "log1p"
                    why["transform"] = (f"Asymétrie de la cible = {sk:.2f} (|skew| > 2) → log(1+y) stabilise la variance ; "
                                        "pensez à inverser (expm1) les prédictions.")
                    an.append(_a("skew", "warn", "Cible très asymétrique", f"skew = {sk:.2f}"))
                else:
                    why["transform"] = f"Asymétrie de la cible = {sk:.2f} : transformation non nécessaire."
                why["scale"] = "Mise à l'échelle de la cible rarement utile : laissée telle quelle."
            recs[name] = _pack(name, kind, role, problem, an, plan, why)
            continue

        # ───────── Anomalies communes ─────────
        if f["unique"] <= 1:
            an.append(_a("constant", "crit", "Colonne constante",
                         f"{f['unique']} valeur distincte : variance nulle, aucune information utile."))
            plan["drop"] = True
            why["drop"] = "Variance nulle (écart-type = 0) → aucun pouvoir discriminant."
        elif mp > 60:
            an.append(_a("missing_high", "crit", "Trop de valeurs manquantes",
                         f"{mp:.1f} % manquantes (> 60 %) : imputer fabriquerait l'essentiel de la colonne."))
            plan["drop"] = True
            why["drop"] = f"{mp:.1f} % de manquants > seuil de 60 % → suppression."
        elif mp > 0:
            sev = "warn" if mp >= 5 else "info"
            an.append(_a("missing", sev, "Valeurs manquantes", f"{f['missing']} manquantes ({mp:.2f} %)."))

        if name in redundant and not plan["drop"]:
            other, r = redundant[name]
            an.append(_a("redundant", "warn", "Colonne redondante",
                         f"Corrélation de {r:+.3f} avec « {other} » (|r| ≥ 0.95) : multicolinéarité."))
            plan["drop"] = True
            why["drop"] = f"|r| = {abs(r):.3f} avec « {other} » : information quasi identique, risque de multicolinéarité."

        ta = f.get("target_assoc")
        if ta is not None and ta >= 0.95:
            an.append(_a("leakage", "crit", "Fuite de cible suspectée",
                         f"Association avec la cible = {ta:.3f} : vérifiez que cette variable n'est pas connue APRÈS l'événement prédit."))

        if kind == "numeric":
            _numeric(name, f, st, n, plan, why, an, target, problem)
        elif kind in ("categorical", "boolean"):
            _categorical(name, f, st, n, kind, plan, why, an)
        elif kind == "datetime":
            if st:
                an.append(_a("datetime", "info", "Colonne date/heure",
                             f"Période {st['min'][:10]} → {st['max'][:10]} ({st['span_days']:.0f} jours)."))
            plan["encode"] = "parts"
            why["encode"] = "Les modèles ne lisent pas les dates brutes → extraction année / mois / jour / jour-semaine."

        if plan["drop"]:
            for k in ("clean", "impute", "outliers", "transform", "scale", "encode"):
                plan[k] = "none"
        recs[name] = _pack(name, kind, role, problem, an, plan, why)

    # ───────── Recommandations globales ─────────
    ov = eda["overview"]
    glob = []
    if ov["duplicates"]:
        glob.append(_a("duplicates", "warn", "Lignes dupliquées",
                       f"{ov['duplicates']} doublons ({ov['duplicates_pct']:.2f} %). Les supprimer AVANT le split évite qu'une même ligne "
                       "se retrouve en train et en test (fuite de données)."))
    return {"features": recs, "global": glob,
            "global_plan": {"drop_duplicates": bool(ov["duplicates"])}}


def _numeric(name, f, st, n, plan, why, an, target, problem):
    if not st:
        return
    mp, sk = f["missing_pct"], st["skew"]
    binary = st["is_binary"]
    if INDEX_LIKE(name, f, st, n):
        an.append(_a("id_like", "crit", "Identifiant probable",
                     f"{f['unique']} valeurs uniques sur {f['count']} lignes (≈ 100 %), entiers : colonne d'identification sans pouvoir prédictif."))
        plan["drop"] = True
        why["drop"] = "Identifiant → aucune généralisation possible, risque de sur-apprentissage."
        return
    if st["top_share"] >= 0.99 and f["unique"] > 1 and not binary:
        an.append(_a("quasi_constant", "warn", "Quasi-constante",
                     f"{st['top_share']*100:.2f} % des lignes ont la même valeur."))
    if 0 < st["negatives"] <= 0.05 * f["count"] and st["min"] < 0 and st["q1"] >= 0:
        an.append(_a("negatives", "warn", "Valeurs négatives suspectes",
                     f"{st['negatives']} valeurs < 0 (min = {st['min']:.4g}) alors que Q1 ≥ 0 : code sentinelle ou erreur de saisie ?"))
    if binary:
        an.append(_a("binary", "info", "Variable binaire", "Deux valeurs distinctes : ni outliers ni mise à l'échelle nécessaires."))
    # imputation
    if mp > 0 and not plan["drop"]:
        if abs(sk) > 1 or st["outliers_iqr_pct"] > 5:
            plan["impute"] = "median"
            why["impute"] = (f"{mp:.2f} % manquantes ; |asymétrie| = {abs(sk):.2f} / outliers = {st['outliers_iqr_pct']:.1f} % → "
                             "médiane, robuste aux valeurs extrêmes.")
        else:
            plan["impute"] = "mean"
            why["impute"] = f"{mp:.2f} % manquantes ; distribution quasi symétrique (skew = {sk:.2f}) → moyenne."
    if binary:
        return
    # outliers
    pct = st["outliers_iqr_pct"]
    if st["outliers_iqr"] > 0:
        sev = "warn" if pct >= 1 else "info"
        an.append(_a("outliers", sev, "Valeurs aberrantes (IQR)",
                     f"{st['outliers_iqr']} valeurs ({pct:.2f} %) hors [{st['lower_fence']:.4g} ; {st['upper_fence']:.4g}] ; "
                     f"|Z| > 3 : {st['outliers_z']}."))
        if pct <= 5:
            plan["outliers"] = "clip_iqr"
            extra = " En détection de fraude/anomalies les extrêmes peuvent être informatifs : on écrête (winsorise) au lieu de supprimer." if problem == "classification" else ""
            why["outliers"] = f"{pct:.2f} % hors bornes de Tukey (Q1−1.5·IQR, Q3+1.5·IQR) → winsorisation, sans perdre de lignes.{extra}"
        else:
            why["outliers"] = (f"{pct:.1f} % hors bornes (> 5 %) : queue lourde plutôt que valeurs isolées → on traite par transformation/échelle robuste, "
                               "pas par écrêtage.")
    # transformation
    if abs(sk) > 1:
        an.append(_a("skew", "warn" if abs(sk) > 2 else "info", "Distribution asymétrique",
                     f"skew = {sk:+.2f}, kurtosis = {st['kurtosis']:.2f}."))
        if sk > 1 and st["min"] >= 0:
            plan["transform"] = "log1p"
            why["transform"] = f"Asymétrie positive (skew = {sk:.2f} > 1) et valeurs ≥ 0 → log(1+x) rapproche d'une loi normale."
        else:
            plan["transform"] = "yeojohnson"
            why["transform"] = f"|skew| = {abs(sk):.2f} > 1 avec valeurs négatives/asymétrie négative → Yeo-Johnson (défini sur ℝ)."
    # échelle
    sd, rng = st["std"], st["max"] - st["min"]
    if sd > 0:
        if plan["outliers"] == "none" and pct > 1:
            plan["scale"] = "robust"
            why["scale"] = f"{pct:.1f} % d'outliers conservés → échelle robuste (médiane / IQR) moins sensible aux extrêmes."
        else:
            plan["scale"] = "standard"
            why["scale"] = (f"σ = {sd:.4g}, étendue = {rng:.4g} → Z-score pour mettre les variables sur une échelle comparable "
                            "(SVM, KNN, régressions, PCA, k-means).")
    if st["integer_like"] and f["unique"] <= 10 and not binary:
        an.append(_a("low_card", "info", "Faible cardinalité", f"{f['unique']} valeurs entières distinctes : peut se comporter comme une catégorie ordinale."))


def INDEX_LIKE(name, f, st, n):
    near_unique = f["count"] > 30 and f["unique"] >= 0.99 * f["count"]
    return (st["integer_like"] and near_unique
            and (ID_NAME.search(name) is not None or (st["std"] > 0 and st["max"] - st["min"] + 1 <= f["count"] * 1.05)))


def _categorical(name, f, st, n, kind, plan, why, an):
    if not st:
        return
    mp, uniq = f["missing_pct"], f["unique"]
    if uniq / max(f["count"], 1) > 0.95 and f["count"] > 30 and kind == "categorical":
        an.append(_a("id_like", "crit", "Identifiant / texte libre",
                     f"{uniq} modalités pour {f['count']} lignes ({f['unique_pct']:.0f} % uniques) : inexploitable par encodage."))
        plan["drop"] = True
        why["drop"] = "Cardinalité ≈ nombre de lignes → identifiant ou texte libre, suppression."
        return
    if st["case_issue"]:
        an.append(_a("case", "warn", "Casse / espaces incohérents",
                     f"{uniq} modalités brutes → {st['normalized_unique']} après normalisation (ex. « Yes » / « yes »)."))
        plan["clean"] = "strip_lower"
        why["clean"] = f"strip + minuscules fusionne {uniq - st['normalized_unique']} doublons de modalités."
    eff = st["normalized_unique"] if plan["clean"] != "none" else uniq
    if st["top_share"] >= 0.99:
        an.append(_a("quasi_constant", "warn", "Quasi-constante", f"{st['top_share']*100:.2f} % sur une seule modalité."))
    if st["rare_count"] and eff > 2:
        an.append(_a("rare", "info", "Modalités rares", f"{st['rare_count']} modalités < 1 % des lignes (regroupées par le One-Hot)."))
    if mp > 0:
        if mp <= 5:
            plan["impute"] = "mode"
            why["impute"] = f"{mp:.2f} % manquantes (≤ 5 %) → mode (modalité la plus fréquente)."
        else:
            plan["impute"] = "constant"
            why["impute"] = f"{mp:.2f} % manquantes (> 5 %) : l'absence peut être informative → catégorie « Missing »."
    if eff <= 2:
        plan["encode"] = "ordinal"
        why["encode"] = f"{eff} modalité(s) → encodage binaire 0/1."
    elif eff <= 15:
        plan["encode"] = "onehot"
        why["encode"] = f"{eff} modalités (≤ 15), variable nominale → One-Hot sans ordre artificiel."
    else:
        plan["encode"] = "frequency"
        why["encode"] = f"{eff} modalités (> 15) : One-Hot créerait trop de colonnes → encodage par fréquence."
        an.append(_a("high_card", "warn", "Cardinalité élevée", f"{eff} modalités distinctes."))


def _pack(name, kind, role, problem, an, plan, why):
    key = slots_key(kind, role, problem)
    an.sort(key=lambda a: -SEVERITY_ORDER[a["severity"]])
    sev = an[0]["severity"] if an else "ok"
    return {"feature": name, "kind": kind, "role": role, "severity": sev, "anomalies": an,
            "plan": plan, "why": why, "slots": SLOT_ORDER[key], "options": OPTIONS[key],
            "can_drop": role != "target"}
