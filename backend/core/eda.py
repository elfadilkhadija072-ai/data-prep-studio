"""Module d'Analyse Exploratoire des Données (EDA).
Statistiques descriptives, métriques de qualité, corrélations et calcul des distributions.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats


def compute_kpis(df: pd.DataFrame, target_col: Optional[str] = None, problem_type: str = "classification") -> Dict[str, Any]:
    """Calcule les indicateurs clés de performance (KPI) globaux du dataset."""
    total_cells = df.size
    total_missing = int(df.isna().sum().sum())
    missing_pct = round((total_missing / total_cells * 100), 2) if total_cells > 0 else 0.0
    duplicates_count = int(df.duplicated().sum())
    memory_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)

    target_info = None
    if target_col and target_col in df.columns:
        raw_s = df[target_col]
        missing_target = int(raw_s.isna().sum())
        missing_target_pct = round((missing_target / len(df) * 100), 2) if len(df) > 0 else 0.0
        s = raw_s.dropna()

        if problem_type == "classification":
            counts = s.value_counts()
            total_labeled = len(s)
            dist = {str(k): int(v) for k, v in counts.items()}
            pcts = {str(k): round(int(v) / total_labeled * 100, 2) for k, v in counts.items()} if total_labeled > 0 else {}
            
            is_imbalanced = False
            imbalance_ratio = 1.0
            imbalance_ratio_inv = 1.0
            minority_class = None
            minority_pct = 0.0
            majority_class = None

            if len(counts) >= 2:
                min_c = counts.min()
                max_c = counts.max()
                minority_class = str(counts.idxmin())
                majority_class = str(counts.idxmax())
                minority_pct = round(float(min_c / total_labeled * 100), 2)
                # Ratio majoritaire / minoritaire (ex: 20.1:1)
                imbalance_ratio = round(float(max_c / min_c), 2) if min_c > 0 else 999.0
                # Ratio inverse minoritaire / majoritaire
                imbalance_ratio_inv = round(float(min_c / max_c), 3) if max_c > 0 else 0.0
                if minority_pct < 20.0 or imbalance_ratio > 3.0:
                    is_imbalanced = True

            target_info = {
                "type": "classification",
                "classes": dist,
                "class_percentages": pcts,
                "total_labeled": total_labeled,
                "missing_target_count": missing_target,
                "missing_target_pct": missing_target_pct,
                "is_imbalanced": is_imbalanced,
                "imbalance_ratio": imbalance_ratio,
                "imbalance_ratio_inv": imbalance_ratio_inv,
                "minority_class": minority_class,
                "minority_pct": minority_pct,
                "majority_class": majority_class,
            }
        else:
            target_info = {
                "type": "regression",
                "total_labeled": len(s),
                "missing_target_count": missing_target,
                "missing_target_pct": missing_target_pct,
                "mean": round(float(s.mean()), 2) if len(s) else 0.0,
                "std": round(float(s.std()), 2) if len(s) else 0.0,
                "min": round(float(s.min()), 2) if len(s) else 0.0,
                "max": round(float(s.max()), 2) if len(s) else 0.0,
                "skewness": round(float(s.skew()), 2) if len(s) > 2 else 0.0,
            }

    return {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "missing_count": total_missing,
        "missing_pct": missing_pct,
        "duplicates_count": duplicates_count,
        "memory_mb": memory_mb,
        "target_info": target_info,
    }


def compute_correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """Calcule le rapport de corrélation η (eta) entre une variable catégorielle et continue."""
    try:
        df_temp = pd.DataFrame({"cat": categories, "val": pd.to_numeric(measurements, errors="coerce")}).dropna()
        if df_temp.empty or df_temp["cat"].nunique() <= 1:
            return 0.0
        overall_mean = df_temp["val"].mean()
        group_means = df_temp.groupby("cat")["val"].mean()
        group_counts = df_temp.groupby("cat")["val"].count()
        
        ss_between = ((group_means - overall_mean) ** 2 * group_counts).sum()
        ss_total = ((df_temp["val"] - overall_mean) ** 2).sum()
        if ss_total == 0:
            return 0.0
        eta = math.sqrt(max(0.0, ss_between / ss_total))
        return round(float(eta), 3)
    except Exception:
        return 0.0


def compute_cramers_v(cat1: pd.Series, cat2: pd.Series) -> float:
    """Calcule le V de Cramér entre deux variables catégorielles."""
    try:
        conf_matrix = pd.crosstab(cat1, cat2)
        if conf_matrix.empty or conf_matrix.shape[0] <= 1 or conf_matrix.shape[1] <= 1:
            return 0.0
        chi2 = stats.chi2_contingency(conf_matrix)[0]
        n = conf_matrix.sum().sum()
        r, k = conf_matrix.shape
        phi2 = chi2 / n
        phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
        r_corr = r - ((r - 1) ** 2) / (n - 1)
        k_corr = k - ((k - 1) ** 2) / (n - 1)
        min_dim = min(k_corr - 1, r_corr - 1)
        if min_dim <= 0:
            return 0.0
        return round(float(math.sqrt(phi2_corr / min_dim)), 3)
    except Exception:
        return 0.0


def compute_features_table(
    df: pd.DataFrame,
    column_types: Dict[str, str],
    target_col: Optional[str] = None,
    problem_type: str = "classification",
) -> List[Dict[str, Any]]:
    """Construit le tableau descriptif de toutes les features avec statistiques et liens avec la cible."""
    n_rows = len(df)
    results = []

    target_series = df[target_col] if target_col and target_col in df.columns else None

    for col in df.columns:
        s = df[col]
        col_type = column_types.get(col, "categorical")
        miss_count = int(s.isna().sum())
        miss_pct = round((miss_count / n_rows * 100), 2) if n_rows > 0 else 0.0
        unique_count = int(s.nunique(dropna=True))
        
        row_stat: Dict[str, Any] = {
            "name": col,
            "type": col_type,
            "is_target": (col == target_col),
            "missing_count": miss_count,
            "missing_pct": miss_pct,
            "unique_count": unique_count,
            "mean": None,
            "std": None,
            "min": None,
            "q1": None,
            "median": None,
            "q3": None,
            "max": None,
            "skewness": None,
            "kurtosis": None,
            "outliers_count": 0,
            "outliers_pct": 0.0,
            "target_association": None,
            "target_association_metric": None,
        }

        # Stats numériques
        if col_type in ("numeric", "boolean"):
            num_s = pd.to_numeric(s, errors="coerce").dropna()
            if len(num_s) > 0:
                q25 = float(num_s.quantile(0.25))
                med = float(num_s.median())
                q75 = float(num_s.quantile(0.75))
                iqr = q75 - q25
                lower_b = q25 - 1.5 * iqr
                upper_b = q75 + 1.5 * iqr
                outliers = num_s[(num_s < lower_b) | (num_s > upper_b)]
                
                row_stat["mean"] = round(float(num_s.mean()), 2)
                row_stat["std"] = round(float(num_s.std()), 2) if len(num_s) > 1 else 0.0
                row_stat["min"] = round(float(num_s.min()), 2)
                row_stat["q1"] = round(q25, 2)
                row_stat["median"] = round(med, 2)
                row_stat["q3"] = round(q75, 2)
                row_stat["max"] = round(float(num_s.max()), 2)
                row_stat["skewness"] = round(float(num_s.skew()), 2) if len(num_s) > 2 else 0.0
                row_stat["kurtosis"] = round(float(num_s.kurt()), 2) if len(num_s) > 3 else 0.0
                row_stat["outliers_count"] = int(len(outliers))
                row_stat["outliers_pct"] = round(len(outliers) / len(num_s) * 100, 2)

        # Calcul d'association avec la cible
        if target_series is not None and col != target_col:
            tgt_clean = target_series.dropna()
            feat_clean = s.dropna()
            common_idx = tgt_clean.index.intersection(feat_clean.index)
            
            if len(common_idx) > 10:
                t_sub = target_series.loc[common_idx]
                f_sub = s.loc[common_idx]

                if problem_type == "regression":
                    if col_type in ("numeric", "boolean"):
                        # Pearson r
                        r, _ = stats.pearsonr(pd.to_numeric(f_sub, errors="coerce").fillna(0), pd.to_numeric(t_sub, errors="coerce").fillna(0))
                        row_stat["target_association"] = round(abs(float(r)), 3)
                        row_stat["target_association_metric"] = "|Pearson r|"
                    else:
                        # Rapport de corrélation eta
                        eta = compute_correlation_ratio(f_sub, t_sub)
                        row_stat["target_association"] = eta
                        row_stat["target_association_metric"] = "Rapport eta"
                else:  # classification
                    if col_type in ("numeric", "boolean"):
                        # Rapport de corrélation eta (cible catégorielle, feature continue)
                        eta = compute_correlation_ratio(t_sub, f_sub)
                        row_stat["target_association"] = eta
                        row_stat["target_association_metric"] = "Rapport eta"
                    else:
                        # Cramér V
                        v = compute_cramers_v(t_sub, f_sub)
                        row_stat["target_association"] = v
                        row_stat["target_association_metric"] = "Cramér V"

        results.append(row_stat)

    return results


def get_column_distribution(
    df: pd.DataFrame,
    col: str,
    col_type: str,
    zoom_p99: bool = False,
    log_scale: bool = False,
    target_col: Optional[str] = None,
    problem_type: str = "classification",
) -> Dict[str, Any]:
    """Génère les données d'histogramme, boxplot, top modalités et regroupement par classe."""
    s = df[col].dropna()
    if s.empty:
        return {"error": "Colonne vide ou valeurs toutes manquantes"}

    res: Dict[str, Any] = {
        "column": col,
        "type": col_type,
        "histogram": [],
        "boxplot": None,
        "top_categories": [],
        "target_comparison": None,
    }

    if col_type in ("numeric", "boolean"):
        num_s = pd.to_numeric(s, errors="coerce").dropna()
        if not num_s.empty:
            work_s = num_s.copy()
            if zoom_p99 and len(work_s) > 20:
                p1, p99 = float(work_s.quantile(0.01)), float(work_s.quantile(0.99))
                work_s = work_s[(work_s >= p1) & (work_s <= p99)]
            if log_scale:
                # Appliquer log10 (pour valeurs > 0)
                work_s = np.log10(np.clip(work_s, 1e-6, None))

            counts, bin_edges = np.histogram(work_s, bins=25)
            hist_data = []
            for i in range(len(counts)):
                hist_data.append({
                    "bin_start": round(float(bin_edges[i]), 2),
                    "bin_end": round(float(bin_edges[i+1]), 2),
                    "label": f"[{round(float(bin_edges[i]), 1)} ; {round(float(bin_edges[i+1]), 1)}]",
                    "count": int(counts[i]),
                })
            res["histogram"] = hist_data

            # Boxplot
            q25 = float(num_s.quantile(0.25))
            med = float(num_s.median())
            q75 = float(num_s.quantile(0.75))
            iqr = q75 - q25
            low_whisker = float(max(num_s.min(), q25 - 1.5 * iqr))
            high_whisker = float(min(num_s.max(), q75 + 1.5 * iqr))
            outliers_sample = num_s[(num_s < low_whisker) | (num_s > high_whisker)].head(50).tolist()
            
            res["boxplot"] = {
                "min": round(float(num_s.min()), 2),
                "low_whisker": round(low_whisker, 2),
                "q1": round(q25, 2),
                "median": round(med, 2),
                "q3": round(q75, 2),
                "high_whisker": round(high_whisker, 2),
                "max": round(float(num_s.max()), 2),
                "outliers_sample": [round(float(v), 2) for v in outliers_sample],
            }

            # Comparaison avec la cible si classification
            if target_col and target_col in df.columns and col != target_col and problem_type == "classification":
                df_pair = df[[col, target_col]].dropna()
                classes = df_pair[target_col].unique()
                if 2 <= len(classes) <= 6:
                    comp = {}
                    for cls in classes:
                        cls_vals = pd.to_numeric(df_pair[df_pair[target_col] == cls][col], errors="coerce").dropna()
                        comp[str(cls)] = {
                            "mean": round(float(cls_vals.mean()), 2) if len(cls_vals) else 0,
                            "median": round(float(cls_vals.median()), 2) if len(cls_vals) else 0,
                            "count": int(len(cls_vals)),
                        }
                    res["target_comparison"] = comp

    else:
        # Catégoriel / Date / Texte
        val_counts = s.astype(str).value_counts().head(20)
        res["top_categories"] = [
            {"category": str(k), "count": int(v), "percentage": round(v / len(s) * 100, 1)}
            for k, v in val_counts.items()
        ]

    return res


def compute_correlations_matrix(df: pd.DataFrame, method: str = "pearson", max_sample: int = 200_000) -> Dict[str, Any]:
    """Calcule la matrice de corrélation et identifie les paires fortement corrélées."""
    num_df = df.select_dtypes(include="number")
    if num_df.empty or num_df.shape[1] < 2:
        return {"columns": [], "matrix": [], "top_pairs": []}

    # Échantillonnage si très grand dataset (> 200k lignes) pour garantir la réactivité
    if len(num_df) > max_sample:
        num_df = num_df.sample(max_sample, random_state=42)

    corr_mat = num_df.corr(method=method).round(3)
    cols = list(corr_mat.columns)
    
    matrix_rows = []
    for r in cols:
        matrix_rows.append({
            "col": r,
            **{c: float(corr_mat.loc[r, c]) for c in cols}
        })

    # Top paires corrélées
    top_pairs = []
    seen = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c1, c2 = cols[i], cols[j]
            val = float(corr_mat.loc[c1, c2])
            if not math.isnan(val) and abs(val) >= 0.40:
                top_pairs.append({
                    "feature_1": c1,
                    "feature_2": c2,
                    "correlation": val,
                    "abs_correlation": abs(val),
                })

    top_pairs = sorted(top_pairs, key=lambda x: x["abs_correlation"], reverse=True)[:25]

    return {
        "columns": cols,
        "matrix": matrix_rows,
        "top_pairs": top_pairs,
    }


def compute_scatter_data(df: pd.DataFrame, x_col: str, y_col: str, target_col: Optional[str] = None, max_points: int = 500) -> List[Dict[str, Any]]:
    """Génère les points d'un nuage de points échantillonné avec code couleur selon la cible."""
    needed = [x_col, y_col]
    if target_col and target_col in df.columns:
        needed.append(target_col)
    
    sub = df[needed].dropna()
    if len(sub) > max_points:
        sub = sub.sample(max_points, random_state=42)

    points = []
    for _, row in sub.iterrows():
        pt = {
            "x": round(float(row[x_col]), 2),
            "y": round(float(row[y_col]), 2),
        }
        if target_col and target_col in df.columns:
            pt["target"] = str(row[target_col])
        points.append(pt)

    return points


def get_data_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Rapporte le taux de manquants, les valeurs aberrantes et les doublons."""
    n = len(df)
    missing_data = []
    outlier_data = []

    for col in df.columns:
        s = df[col]
        miss = int(s.isna().sum())
        if miss > 0:
            missing_data.append({
                "column": col,
                "missing_count": miss,
                "missing_pct": round(miss / n * 100, 2),
            })
        
        if pd.api.types.is_numeric_dtype(s):
            clean_s = s.dropna()
            if len(clean_s) > 10:
                q25, q75 = clean_s.quantile(0.25), clean_s.quantile(0.75)
                iqr = q75 - q25
                outliers = clean_s[(clean_s < q25 - 1.5 * iqr) | (clean_s > q75 + 1.5 * iqr)]
                if len(outliers) > 0:
                    outlier_data.append({
                        "column": col,
                        "outlier_count": int(len(outliers)),
                        "outlier_pct": round(len(outliers) / len(clean_s) * 100, 2),
                    })

    missing_data = sorted(missing_data, key=lambda x: x["missing_pct"], reverse=True)
    outlier_data = sorted(outlier_data, key=lambda x: x["outlier_pct"], reverse=True)

    # Exemples de doublons
    duplicates_mask = df.duplicated(keep=False)
    dup_examples = []
    if duplicates_mask.any():
        dup_df = df[duplicates_mask].head(6)
        dup_examples = dup_df.fillna("NA").to_dict(orient="records")

    return {
        "missing_by_column": missing_data,
        "outliers_by_column": outlier_data,
        "duplicates_count": int(df.duplicated().sum()),
        "duplicate_rows_sample": dup_examples,
    }
