"""Module d'analyse d'anomalies et moteur de recommandations de préparation."""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from backend.core.common import FeatureStepConfig


SUSPECT_POSITIVE_KEYWORDS = [
    "montant", "prix", "price", "amount", "surface", "superficie",
    "age", "duree", "duration", "score", "revenu", "salaire", "nombre", "nb_"
]


def detect_column_anomalies(
    df: pd.DataFrame,
    col: str,
    col_type: str,
    target_col: Optional[str] = None,
    corr_matrix: Optional[pd.DataFrame] = None,
) -> List[Dict[str, Any]]:
    """Détecte l'ensemble des anomalies d'une colonne avec sévérité et explication."""
    s = df[col]
    n_rows = len(df)
    clean_s = s.dropna()
    n_clean = len(clean_s)
    n_unique = clean_s.nunique() if n_clean > 0 else 0

    anomalies: List[Dict[str, Any]] = []

    # 1. Valeurs manquantes
    miss_count = int(s.isna().sum())
    if miss_count > 0:
        miss_pct = round(miss_count / n_rows * 100, 2)
        if miss_pct >= 60.0:
            anomalies.append({
                "type": "manquants",
                "severity": "critique",
                "title": f"Valeurs manquantes critiques ({miss_pct}%)",
                "detail": f"{miss_count} lignes manquantes sur {n_rows}. Un seuil > 60% justifie généralement la suppression.",
            })
        elif miss_pct >= 5.0:
            anomalies.append({
                "type": "manquants",
                "severity": "attention",
                "title": f"Valeurs manquantes modérées ({miss_pct}%)",
                "detail": f"{miss_count} lignes manquantes nécessitant une stratégie d'imputation robuste.",
            })
        else:
            anomalies.append({
                "type": "manquants",
                "severity": "info",
                "title": f"Faible taux de manquants ({miss_pct}%)",
                "detail": f"{miss_count} lignes manquantes faciles à imputer.",
            })

    # 2. Constante / Quasi-constante
    if n_clean > 0:
        if n_unique <= 1:
            anomalies.append({
                "type": "constante",
                "severity": "critique",
                "title": "Colonne constante (variance nulle)",
                "detail": "Une seule valeur distincte présente. N'apporte aucun pouvoir prédictif.",
            })
        else:
            top_freq_pct = round(clean_s.value_counts(normalize=True).iloc[0] * 100, 2)
            if top_freq_pct >= 99.0 and n_clean >= 50:
                anomalies.append({
                    "type": "quasi_constante",
                    "severity": "attention",
                    "title": f"Quasi-constante ({top_freq_pct}%)",
                    "detail": f"La modalité prédominante occupe {top_freq_pct}% des observations.",
                })

    # 3. Identifiant / Clé primaire
    if col_type == "id" or (n_clean >= 30 and n_unique / n_clean >= 0.98 and col_type != "numeric"):
        anomalies.append({
            "type": "identifiant",
            "severity": "critique",
            "title": "Identifiant / Cardinalité quasi-unique",
            "detail": f"{n_unique} valeurs uniques ({round(n_unique/n_clean*100, 1)}%). Risque fort de surapprentissage si conservé.",
        })

    # 4. Fuite de cible (Target leakage)
    if corr_matrix is not None and target_col and target_col in corr_matrix.columns and col != target_col:
        if col in corr_matrix.index:
            r_target = abs(float(corr_matrix.loc[col, target_col]))
            if r_target >= 0.98:
                anomalies.append({
                    "type": "fuite_cible",
                    "severity": "critique",
                    "title": f"Suspicion de fuite de cible (|r| = {round(r_target, 3)})",
                    "detail": "Corrélation quasi-parfaite avec la cible. Vérifier si cette variable est connue au moment de la prédiction.",
                })

    # 5. Collinéarité / Redondance (|r| >= 0.95)
    if corr_matrix is not None and col in corr_matrix.index:
        for other_col in corr_matrix.columns:
            if other_col != col and other_col != target_col:
                r_val = abs(float(corr_matrix.loc[col, other_col]))
                if r_val >= 0.95:
                    anomalies.append({
                        "type": "collinearite",
                        "severity": "attention",
                        "title": f"Redondance avec '{other_col}' (|r| = {round(r_val, 2)})",
                        "detail": f"Forte collinéarité susceptible d'instabiliser les modèles linéaires.",
                    })
                    break

    # 6. Outliers & Asymétrie pour numériques
    if col_type in ("numeric", "boolean") and n_clean > 10:
        num_s = pd.to_numeric(clean_s, errors="coerce").dropna()
        if len(num_s) > 10:
            q25 = float(num_s.quantile(0.25))
            q75 = float(num_s.quantile(0.75))
            iqr = q75 - q25
            outliers = num_s[(num_s < q25 - 1.5 * iqr) | (num_s > q75 + 1.5 * iqr)]
            outlier_pct = round(len(outliers) / len(num_s) * 100, 2)
            if outlier_pct >= 1.0:
                anomalies.append({
                    "type": "outliers",
                    "severity": "attention" if outlier_pct >= 4.0 else "info",
                    "title": f"Valeurs aberrantes détectées ({outlier_pct}%)",
                    "detail": f"{len(outliers)} observations hors bornes [{round(q25 - 1.5*iqr, 1)} ; {round(q75 + 1.5*iqr, 1)}].",
                })

            skew_val = float(num_s.skew()) if len(num_s) > 2 else 0.0
            if abs(skew_val) >= 1.5:
                direction = "droite (positive)" if skew_val > 0 else "gauche (négative)"
                anomalies.append({
                    "type": "asymetrie",
                    "severity": "attention",
                    "title": f"Forte asymétrie ({round(skew_val, 2)})",
                    "detail": f"Distribution étirée vers la {direction}. Une transformation logarithmique ou racine carrée est recommandée.",
                })

            # Valeurs négatives suspectes
            col_lower = col.lower()
            if any(pos_kw in col_lower for pos_kw in SUSPECT_POSITIVE_KEYWORDS):
                neg_count = int((num_s < 0).sum())
                if neg_count > 0:
                    anomalies.append({
                        "type": "valeurs_negatives_suspectes",
                        "severity": "attention",
                        "title": f"Valeurs négatives suspectes ({neg_count})",
                        "detail": f"Grandeur censée être positive (ex: montant/surface) comportant {neg_count} valeurs < 0.",
                    })

    # 7. Forte cardinalité & Casse hétérogène pour catégoriels
    if col_type in ("categorical", "text") and n_clean > 0:
        if n_unique > 50 and col_type != "id":
            anomalies.append({
                "type": "forte_cardinalite",
                "severity": "attention",
                "title": f"Forte cardinalité ({n_unique} modalités)",
                "detail": "Le One-Hot Encoding produira un espace de grande dimension. Encodage par fréquence ou groupement rare recommandé.",
            })

        # Test casse hétérogène
        str_vals = clean_s.astype(str)
        lower_set = set(str_vals.str.lower())
        if len(lower_set) < n_unique:
            anomalies.append({
                "type": "casse_incoherente",
                "severity": "info",
                "title": "Casse textuelle hétérogène",
                "detail": f"{n_unique} modalités actuelles se réduisent à {len(lower_set)} après passage en minuscules.",
            })

    return anomalies


def build_recommended_plan(
    df: pd.DataFrame,
    col: str,
    col_type: str,
    anomalies: List[Dict[str, Any]],
    target_col: Optional[str] = None,
) -> Tuple[FeatureStepConfig, List[str]]:
    """Génère le plan de transformation optimal avec justification chiffrée pour chaque étape."""
    cfg = FeatureStepConfig()
    justifications: List[str] = []

    anomaly_types = {a["type"]: a for a in anomalies}
    s = df[col]
    clean_s = s.dropna()
    n_rows = len(df)
    miss_pct = round(s.isna().sum() / n_rows * 100, 2) if n_rows > 0 else 0.0

    # 1. Étape Suppression
    if "constante" in anomaly_types:
        cfg.action_drop = True
        justifications.append("Suppression recommandée : colonne constante sans variance.")
        return cfg, justifications
    if "identifiant" in anomaly_types and col != target_col:
        cfg.action_drop = True
        justifications.append("Suppression recommandée : identifiant unique risquant de provoquer du surapprentissage.")
        return cfg, justifications
    if "fuite_cible" in anomaly_types and col != target_col:
        cfg.action_drop = True
        justifications.append("Suppression recommandée : risque critique de fuite d'information sur la cible.")
        return cfg, justifications
    if miss_pct >= 65.0:
        cfg.action_drop = True
        justifications.append(f"Suppression recommandée : plus de {miss_pct}% de valeurs manquantes.")
        return cfg, justifications

    # 2. Nettoyage de texte
    if col_type in ("categorical", "text"):
        cfg.text_clean = True
        if "casse_incoherente" in anomaly_types:
            justifications.append("Nettoyage texte : normalisation de la casse pour unifier les doublons de modalités.")
        else:
            justifications.append("Nettoyage texte : suppression des espaces superflus et standardisation.")

    # 3. Imputation
    if miss_pct > 0:
        if col_type in ("numeric", "boolean"):
            num_s = pd.to_numeric(clean_s, errors="coerce").dropna()
            skew_val = abs(float(num_s.skew())) if len(num_s) > 2 else 0.0
            has_outliers = "outliers" in anomaly_types
            if skew_val > 1.0 or has_outliers:
                cfg.impute_strategy = "median"
                justifications.append(f"Imputation par médiane : recommandée face à l'asymétrie ({round(skew_val, 2)}) et aux outliers.")
            else:
                cfg.impute_strategy = "mean"
                justifications.append("Imputation par moyenne : distribution symétrique sans valeurs extrêmes perturbatrices.")
        elif col_type == "date":
            cfg.impute_strategy = "mode"
            justifications.append("Imputation par date la plus fréquente (mode).")
        else:
            if miss_pct >= 10.0:
                cfg.impute_strategy = "constant"
                cfg.impute_value = "Missing"
                justifications.append(f"Imputation catégorie 'Missing' : taux de {miss_pct}% suffisant pour constituer une classe informative.")
            else:
                cfg.impute_strategy = "mode"
                justifications.append("Imputation par mode : modalité la plus fréquente pour faible taux de manquants.")
    else:
        cfg.impute_strategy = "none"

    # 4. Outliers
    if col_type in ("numeric", "boolean") and "outliers" in anomaly_types:
        outlier_info = anomaly_types["outliers"]
        cfg.outlier_strategy = "winsorize_iqr"
        cfg.outlier_iqr_factor = 1.5
        justifications.append(f"Winsorisation IQR (1.5) : plafonne {outlier_info['title']} aux bornes interquartiles sans perte de données.")
    else:
        cfg.outlier_strategy = "none"

    # 5. Transformation de puissance
    if col_type in ("numeric", "boolean") and "asymetrie" in anomaly_types:
        num_s = pd.to_numeric(clean_s, errors="coerce").dropna()
        if (num_s >= 0).all():
            cfg.transform_strategy = "log1p"
            justifications.append("Transformation log(1+x) : stabilise la variance et réduit l'asymétrie positive.")
        else:
            cfg.transform_strategy = "yeojohnson"
            justifications.append("Transformation Yeo-Johnson : normalise la distribution en supportant les valeurs négatives.")
    else:
        cfg.transform_strategy = "none"

    # 6. Mise à l'échelle (Scaling)
    if col_type in ("numeric", "boolean") and col != target_col:
        if "outliers" in anomaly_types:
            cfg.scaling_strategy = "robust"
            justifications.append("RobustScaler : mise à l'échelle centrée sur la médiane, insensible aux valeurs résiduelles extrêmes.")
        elif cfg.transform_strategy != "none":
            cfg.scaling_strategy = "standard"
            justifications.append("StandardScaler : centrage et réduction adaptés suite à la normalisation de la distribution.")
        else:
            cfg.scaling_strategy = "standard"
            justifications.append("StandardScaler : standardisation standard recommandée pour les algorithmes sensibles aux échelles.")
    else:
        cfg.scaling_strategy = "none"

    # 7. Encodage
    if col != target_col:
        if col_type == "categorical":
            n_cat = clean_s.nunique()
            if n_cat <= 12:
                cfg.encoding_strategy = "onehot"
                cfg.rare_category_min_freq = 0.03
                justifications.append(f"One-Hot Encoding ({n_cat} modalités) avec regroupement rare à 3% pour éviter la sparsité.")
            elif n_cat <= 50:
                cfg.encoding_strategy = "onehot"
                cfg.rare_category_min_freq = 0.05
                justifications.append("One-Hot Encoding avec seuil rare à 5% face au nombre moyen de modalités.")
            else:
                cfg.encoding_strategy = "frequency"
                justifications.append(f"Frequency Encoding : préserve la compacité de l'espace pour {n_cat} modalités.")
        elif col_type == "date":
            cfg.encoding_strategy = "date_decompose"
            justifications.append("Décomposition temporelle : extraction de l'année, du mois, du jour et du jour de la semaine.")
        elif col_type == "boolean":
            cfg.encoding_strategy = "ordinal"
            justifications.append("Encodage binaire 0 / 1 standardisé.")
        else:
            cfg.encoding_strategy = "none"
    else:
        cfg.encoding_strategy = "none"

    return cfg, justifications


def generate_full_analysis_plan(
    df: pd.DataFrame,
    column_types: Dict[str, str],
    target_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Génère l'ensemble des anomalies et recommandations pour toutes les colonnes du dataset."""
    num_df = df.select_dtypes(include="number")
    corr_mat = num_df.corr().round(3) if len(num_df.columns) >= 2 else None

    features_plan = {}
    total_reco = 0
    total_drop = 0

    for col in df.columns:
        c_type = column_types.get(col, "categorical")
        anomalies = detect_column_anomalies(df, col, c_type, target_col, corr_mat)
        reco_cfg, justifications = build_recommended_plan(df, col, c_type, anomalies, target_col)
        
        if reco_cfg.action_drop:
            total_drop += 1
        else:
            total_reco += 1

        features_plan[col] = {
            "name": col,
            "type": c_type,
            "is_target": (col == target_col),
            "anomalies": anomalies,
            "recommended_config": reco_cfg.model_dump(),
            "user_config": reco_cfg.model_dump(),
            "status": "recommended",  # 'recommended', 'customized', 'passthrough'
            "justifications": justifications,
        }

    return {
        "features": features_plan,
        "summary": {
            "total_features": len(df.columns),
            "recommended_count": total_reco,
            "drop_count": total_drop,
            "customized_count": 0,
        }
    }


def preview_feature_transformation(
    series: pd.Series,
    col_type: str,
    cfg: FeatureStepConfig,
) -> Dict[str, Any]:
    """Génère un aperçu en direct Avant / Après pour une feature donnée."""
    s_before = series.copy()
    
    # 1. Stats Avant
    n_total = len(s_before)
    miss_before = int(s_before.isna().sum())
    stats_before = {
        "count": n_total,
        "missing": miss_before,
        "missing_pct": round(miss_before / n_total * 100, 2) if n_total > 0 else 0,
        "unique": int(s_before.nunique(dropna=True)),
        "mean": None,
        "std": None,
        "min": None,
        "median": None,
        "max": None,
    }

    if col_type in ("numeric", "boolean"):
        num_b = pd.to_numeric(s_before, errors="coerce").dropna()
        if len(num_b) > 0:
            stats_before.update({
                "mean": round(float(num_b.mean()), 2),
                "std": round(float(num_b.std()), 2) if len(num_b) > 1 else 0.0,
                "min": round(float(num_b.min()), 2),
                "median": round(float(num_b.median()), 2),
                "max": round(float(num_b.max()), 2),
            })

    if cfg.action_drop:
        return {
            "is_dropped": True,
            "stats_before": stats_before,
            "stats_after": None,
            "hist_before": [],
            "hist_after": [],
        }

    # Application séquentielle simulée
    s_after = s_before.copy()

    # Nettoyage texte
    if col_type in ("categorical", "text") and cfg.text_clean:
        s_after = s_after.astype("string").str.strip().str.lower()

    # Imputation
    if cfg.impute_strategy != "none" and s_after.isna().any():
        if col_type in ("numeric", "boolean"):
            num_vals = pd.to_numeric(s_after, errors="coerce")
            imp_val = float(num_vals.median()) if cfg.impute_strategy == "median" else float(num_vals.mean())
            s_after = num_vals.fillna(imp_val)
        else:
            if cfg.impute_strategy == "constant":
                s_after = s_after.fillna(cfg.impute_value or "Missing")
            else:
                mode_val = s_after.mode()
                s_after = s_after.fillna(mode_val.iloc[0] if len(mode_val) else "Inconnu")

    # Outliers
    if col_type in ("numeric", "boolean") and cfg.outlier_strategy != "none":
        num_vals = pd.to_numeric(s_after, errors="coerce").copy()
        if cfg.outlier_strategy == "winsorize_iqr":
            q25, q75 = num_vals.quantile(0.25), num_vals.quantile(0.75)
            iqr = q75 - q25
            s_after = num_vals.clip(lower=q25 - cfg.outlier_iqr_factor * iqr, upper=q75 + cfg.outlier_iqr_factor * iqr)
        elif cfg.outlier_strategy == "winsorize_percentiles":
            p1, p99 = num_vals.quantile(0.01), num_vals.quantile(0.99)
            s_after = num_vals.clip(lower=p1, upper=p99)

    # Transformation de puissance
    if col_type in ("numeric", "boolean") and cfg.transform_strategy != "none":
        num_vals = pd.to_numeric(s_after, errors="coerce").copy()
        if cfg.transform_strategy == "log1p":
            s_after = np.log1p(np.clip(num_vals, 0, None))
        elif cfg.transform_strategy == "sqrt":
            s_after = np.sqrt(np.clip(num_vals, 0, None))
        elif cfg.transform_strategy == "yeojohnson":
            try:
                from sklearn.preprocessing import PowerTransformer
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                arr = num_vals.fillna(0.0).values.reshape(-1, 1)
                s_after = pd.Series(pt.fit_transform(arr).flatten(), index=s_after.index)
            except Exception:
                s_after = np.log1p(np.clip(num_vals, 0, None))

    # Scaling
    if col_type in ("numeric", "boolean") and cfg.scaling_strategy != "none":
        num_vals = pd.to_numeric(s_after, errors="coerce").copy()
        if cfg.scaling_strategy == "standard":
            std = float(num_vals.std())
            mean = float(num_vals.mean())
            s_after = (num_vals - mean) / (std if std != 0 else 1.0)
        elif cfg.scaling_strategy == "minmax":
            mi, ma = float(num_vals.min()), float(num_vals.max())
            s_after = (num_vals - mi) / (ma - mi if ma != mi else 1.0)
        elif cfg.scaling_strategy == "robust":
            med = float(num_vals.median())
            iqr = float(num_vals.quantile(0.75) - num_vals.quantile(0.25))
            s_after = (num_vals - med) / (iqr if iqr != 0 else 1.0)

    # Stats Après
    miss_after = int(s_after.isna().sum())
    stats_after = {
        "count": len(s_after),
        "missing": miss_after,
        "missing_pct": round(miss_after / len(s_after) * 100, 2) if len(s_after) > 0 else 0,
        "unique": int(s_after.nunique(dropna=True)),
        "mean": None,
        "std": None,
        "min": None,
        "median": None,
        "max": None,
    }

    hist_before = []
    hist_after = []

    if col_type in ("numeric", "boolean"):
        num_a = pd.to_numeric(s_after, errors="coerce").dropna()
        if len(num_a) > 0:
            stats_after.update({
                "mean": round(float(num_a.mean()), 2),
                "std": round(float(num_a.std()), 2) if len(num_a) > 1 else 0.0,
                "min": round(float(num_a.min()), 2),
                "median": round(float(num_a.median()), 2),
                "max": round(float(num_a.max()), 2),
            })
            # Histogrammes comparatifs
            num_b = pd.to_numeric(s_before, errors="coerce").dropna()
            if len(num_b) > 5:
                cnt_b, edg_b = np.histogram(num_b, bins=15)
                for i in range(len(cnt_b)):
                    hist_before.append({"label": f"{round(float(edg_b[i]), 1)}", "count": int(cnt_b[i])})
            if len(num_a) > 5:
                cnt_a, edg_a = np.histogram(num_a, bins=15)
                for i in range(len(cnt_a)):
                    hist_after.append({"label": f"{round(float(edg_a[i]), 1)}", "count": int(cnt_a[i])})
    else:
        # Catégories
        cnt_b = s_before.astype(str).value_counts().head(8)
        hist_before = [{"label": str(k), "count": int(v)} for k, v in cnt_b.items()]
        cnt_a = s_after.astype(str).value_counts().head(8)
        hist_after = [{"label": str(k), "count": int(v)} for k, v in cnt_a.items()]

    return {
        "is_dropped": False,
        "stats_before": stats_before,
        "stats_after": stats_after,
        "hist_before": hist_before,
        "hist_after": hist_after,
    }
