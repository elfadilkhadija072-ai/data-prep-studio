"""Module d'entraînement, évaluation et audit de modèles Machine Learning pour DataPrep Studio.

Nouveautés : intégration du rééquilibrage des classes (SMOTE / Undersampling),
métriques AUC-PR et AUC-ROC, tableau comparatif des stratégies.
"""
from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier

from backend.core.security import (
    audit_adversarial_robustness,
    audit_membership_leakage,
    audit_poisoning_risk,
    generate_cryptographic_manifest,
)
from backend.core.rebalancing import apply_rebalancing


AVAILABLE_MODELS = {
    "classification": {
        "knn": {
            "name": "K-Plus Proches Voisins (KNN)",
            "description": "Classification non paramétrique basée sur la proximité géométrique des k plus proches voisins.",
            "default_params": {"n_neighbors": 5, "weights": "distance"},
        },
        "logistic_regression": {
            "name": "Régression Logistique",
            "description": "Modèle linéaire probabiliste rapide, interprétable et robuste avec régularisation L2.",
            "default_params": {"C": 1.0, "max_iter": 500},
        },
        "random_forest": {
            "name": "Forêt Aléatoire (Random Forest)",
            "description": "Ensemble d'arbres de décision réduisant la variance et capturant les interactions non linéaires complexes.",
            "default_params": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        },
        "decision_tree": {
            "name": "Arbre de Décision (CART)",
            "description": "Modèle arborescent hautement interprétable par règles de décision successives.",
            "default_params": {"max_depth": 6, "criterion": "gini", "random_state": 42},
        },
    },
    "regression": {
        "knn_regressor": {
            "name": "KNN Régresseur",
            "description": "Estimation continue par moyenne pondérée des k plus proches voisins.",
            "default_params": {"n_neighbors": 5, "weights": "distance"},
        },
        "linear_regression": {
            "name": "Régression Linéaire (MCO)",
            "description": "Ajustement linéaire classique minimisant la somme des carrés des résiduels.",
            "default_params": {},
        },
        "ridge": {
            "name": "Régression Ridge (L2)",
            "description": "Régression linéaire régularisée L2 pour contrer la multicolinéarité et le surapprentissage.",
            "default_params": {"alpha": 1.0},
        },
        "random_forest_regressor": {
            "name": "Forêt Aléatoire Régression",
            "description": "Agrégation d'arbres régresseurs robuste aux valeurs extrêmes et aux non-linéarités.",
            "default_params": {"n_estimators": 100, "max_depth": 10, "random_state": 42},
        },
    },
}


def get_available_models_list(problem_type: str = "classification") -> List[Dict[str, Any]]:
    """Retourne la liste des modèles applicables pour le type de problème."""
    cat = "regression" if problem_type == "regression" else "classification"
    models_dict = AVAILABLE_MODELS.get(cat, AVAILABLE_MODELS["classification"])
    res = []
    for m_id, info in models_dict.items():
        res.append({
            "id": m_id,
            "name": info["name"],
            "description": info["description"],
            "category": cat,
            "default_params": info["default_params"],
        })
    return res


def instantiate_model(model_id: str, problem_type: str, hyperparams: Dict[str, Any]) -> Any:
    """Instancie le modèle scikit-learn avec les hyperparamètres passés."""
    p = hyperparams.copy()

    if problem_type == "regression":
        if model_id in ("knn_regressor", "knn"):
            k = int(p.get("n_neighbors", 5))
            weights = str(p.get("weights", "distance"))
            return KNeighborsRegressor(n_neighbors=max(1, k), weights=weights)
        elif model_id == "linear_regression":
            return LinearRegression()
        elif model_id == "ridge":
            alpha = float(p.get("alpha", 1.0))
            return Ridge(alpha=max(1e-4, alpha))
        elif model_id in ("random_forest_regressor", "random_forest"):
            n_est = int(p.get("n_estimators", 100))
            max_d = int(p.get("max_depth", 10)) if p.get("max_depth") else None
            return RandomForestRegressor(n_estimators=max(10, n_est), max_depth=max_d, random_state=42)
        else:
            return LinearRegression()
    else:
        # Classification
        if model_id == "knn":
            k = int(p.get("n_neighbors", 5))
            weights = str(p.get("weights", "distance"))
            return KNeighborsClassifier(n_neighbors=max(1, k), weights=weights)
        elif model_id == "logistic_regression":
            c_val = float(p.get("C", 1.0))
            max_i = int(p.get("max_iter", 500))
            return LogisticRegression(C=max(1e-4, c_val), max_iter=max(100, max_i), random_state=42)
        elif model_id == "random_forest":
            n_est = int(p.get("n_estimators", 100))
            max_d = int(p.get("max_depth", 10)) if p.get("max_depth") else None
            return RandomForestClassifier(n_estimators=max(10, n_est), max_depth=max_d, random_state=42)
        elif model_id == "decision_tree":
            max_d = int(p.get("max_depth", 6)) if p.get("max_depth") else None
            criterion = str(p.get("criterion", "gini"))
            return DecisionTreeClassifier(max_depth=max_d, criterion=criterion, random_state=42)
        else:
            return LogisticRegression(max_iter=500, random_state=42)


def _prepare_xy(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    problem_type: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """Prépare X_train, y_train, X_test, y_test avec conversion numérique robuste.

    Returns:
        (X_train, y_train, X_test, y_test, feature_cols)
    """
    raw_feature_cols = [c for c in df_train.columns if c != target_col and c != "_split"]
    X_train = df_train[raw_feature_cols].copy()
    y_train = df_train[target_col].copy()
    X_test = df_test[raw_feature_cols].copy()
    y_test = df_test[target_col].copy()

    # Conversion numérique robuste de toutes les features
    valid_cols = []
    for c in raw_feature_cols:
        if pd.api.types.is_numeric_dtype(X_train[c]):
            valid_cols.append(c)
        else:
            converted_tr = pd.to_numeric(X_train[c], errors="coerce")
            converted_te = pd.to_numeric(X_test[c], errors="coerce")
            if converted_tr.notna().sum() > len(X_train) * 0.5:
                X_train[c] = converted_tr.fillna(0)
                X_test[c] = converted_te.fillna(0)
                valid_cols.append(c)
            else:
                try:
                    le = LabelEncoder()
                    X_train[c] = le.fit_transform(X_train[c].astype(str))
                    test_vals = X_test[c].astype(str).map(
                        lambda s: s if s in le.classes_ else le.classes_[0]
                    )
                    X_test[c] = le.transform(test_vals)
                    valid_cols.append(c)
                except Exception:
                    pass

    X_train = X_train[valid_cols].astype(float).fillna(0)
    X_test = X_test[valid_cols].astype(float).fillna(0)

    # Conversion de la cible
    if problem_type == "classification" and not pd.api.types.is_numeric_dtype(y_train):
        le_y = LabelEncoder()
        y_train = pd.Series(le_y.fit_transform(y_train.astype(str)), index=df_train.index)
        y_test = pd.Series(
            le_y.transform(
                y_test.astype(str).map(lambda s: s if s in le_y.classes_ else le_y.classes_[0])
            ),
            index=df_test.index,
        )
    elif problem_type == "regression":
        y_train = pd.to_numeric(y_train, errors="coerce").fillna(0)
        y_test = pd.to_numeric(y_test, errors="coerce").fillna(0)

    return X_train, y_train, X_test, y_test, valid_cols


def _compute_classification_metrics(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[Dict[str, Any], Dict[str, Any], float, float]:
    """Calcule toutes les métriques de classification dont AUC-PR et AUC-ROC.

    Returns:
        (metrics_dict, chart_data_dict, train_score, test_score)
    """
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    acc_train = float(accuracy_score(y_train, y_pred_train))
    acc_test = float(accuracy_score(y_test, y_pred_test))
    prec = float(precision_score(y_test, y_pred_test, average="weighted", zero_division=0))
    rec = float(recall_score(y_test, y_pred_test, average="weighted", zero_division=0))
    f1 = float(f1_score(y_test, y_pred_test, average="weighted", zero_division=0))

    # AUC-ROC et AUC-PR (seulement si proba disponible et classification binaire ou multi)
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    labels = sorted(list(set(y_test.unique()).union(set(y_pred_test))))

    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
            if len(labels) == 2:
                # Binaire : proba de la classe positive (index 1)
                y_prob_pos = y_prob[:, 1]
                auc_roc = round(float(roc_auc_score(y_test, y_prob_pos)), 4)
                auc_pr = round(float(average_precision_score(y_test, y_prob_pos)), 4)
            else:
                # Multi-classes : OVR macro
                auc_roc = round(float(roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")), 4)
                # AUC-PR macro sur chaque classe
                ap_scores = []
                for i, lbl in enumerate(model.classes_):
                    if i < y_prob.shape[1]:
                        ap_scores.append(average_precision_score((y_test == lbl).astype(int), y_prob[:, i]))
                auc_pr = round(float(np.mean(ap_scores)), 4) if ap_scores else None
        except Exception:
            pass

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred_test, labels=labels)
    cm_formatted = []
    for i, row in enumerate(cm):
        total_row = int(sum(row))
        cm_formatted.append({
            "actual": str(labels[i]),
            "predictions": {str(labels[j]): int(val) for j, val in enumerate(row)},
            "total": total_row,
            "accuracy_row_pct": round(row[i] / total_row * 100, 1) if total_row > 0 else 0,
        })

    metrics = {
        "accuracy_test": round(acc_test, 4),
        "accuracy_train": round(acc_train, 4),
        "precision_weighted": round(prec, 4),
        "recall_weighted": round(rec, 4),
        "f1_weighted": round(f1, 4),
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "test_samples": len(y_test),
        "classes_count": len(labels),
        "classes": [str(l) for l in labels],
    }

    chart_data = {
        "confusion_matrix": cm_formatted,
        "classes": [str(l) for l in labels],
    }

    return metrics, chart_data, acc_train, acc_test


def train_and_audit_model(
    session: Any,
    model_id: str,
    hyperparams: Optional[Dict[str, Any]] = None,
    rebalancing_strategy: str = "none",
    rebalancing_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Entraîne le modèle sélectionné avec rééquilibrage optionnel, calcule les métriques et l'audit cyber.

    Anti-Data-Leakage : le rééquilibrage s'applique UNIQUEMENT sur X_train / y_train,
    jamais sur X_test / y_test. Le test set reste strictement intact.

    Args:
        session: Session DataPrep Studio avec df_train / df_test préparés.
        model_id: Identifiant du modèle à entraîner.
        hyperparams: Hyperparamètres du modèle.
        rebalancing_strategy: 'none' | 'undersample' | 'smote'.
        rebalancing_params: Paramètres du rééquilibrage (k_neighbors, sampling_strategy, etc.).
    """
    if session.df_train is None or session.df_test is None:
        raise ValueError("Les données n'ont pas encore été préparées. Veuillez exécuter le pipeline de préparation.")

    params = hyperparams or {}
    target_col = session.target_col
    problem_type = session.problem_type or "classification"

    # Préparation des données X/y avec conversion numérique robuste
    X_train, y_train, X_test, y_test, feature_cols = _prepare_xy(
        session.df_train.copy(),
        session.df_test.copy(),
        target_col,
        problem_type,
    )

    # -------------------------------------------------------------------
    # RÉÉQUILIBRAGE DES CLASSES — UNIQUEMENT SUR LE TRAIN (Anti-Leakage)
    # -------------------------------------------------------------------
    rebal_info: Dict[str, Any] = {"strategy": "none", "strategy_label": "Aucune"}
    df_train_rebalanced: Optional[pd.DataFrame] = None

    if problem_type == "classification" and rebalancing_strategy != "none":
        try:
            X_train_rebal, y_train_rebal, rebal_info = apply_rebalancing(
                X_train, y_train,
                strategy=rebalancing_strategy,
                params=rebalancing_params or {},
            )
            # Reconstruire un DataFrame annoté pour l'export CSV
            df_train_rebalanced = X_train_rebal.copy()
            df_train_rebalanced[target_col] = y_train_rebal.values
            # Remplacer X_train / y_train par la version rééquilibrée
            X_train = X_train_rebal
            y_train = y_train_rebal
        except Exception as exc:
            rebal_info = {
                "strategy": rebalancing_strategy,
                "strategy_label": rebalancing_strategy,
                "error": str(exc),
                "before": {str(k): int(v) for k, v in y_train.value_counts().items()},
                "after": {str(k): int(v) for k, v in y_train.value_counts().items()},
                "n_before": len(y_train),
                "n_after": len(y_train),
            }
    else:
        # Rééquilibrage "none" : calculer quand même les comptages pour l'affichage
        before_counts = {str(k): int(v) for k, v in y_train.value_counts().items()}
        rebal_info = {
            "strategy": "none",
            "strategy_label": "Aucune",
            "before": before_counts,
            "after": before_counts,
            "n_before": len(y_train),
            "n_after": len(y_train),
            "note": "Aucun rééquilibrage appliqué.",
        }

    # Stocker le train rééquilibré dans la session pour l'export
    if df_train_rebalanced is not None:
        session.df_train_rebalanced = df_train_rebalanced
    else:
        session.df_train_rebalanced = None

    # -------------------------------------------------------------------
    # INSTANCIATION ET ENTRAÎNEMENT DU MODÈLE
    # -------------------------------------------------------------------
    model = instantiate_model(model_id, problem_type, params)

    t0 = time.time()
    model.fit(X_train, y_train)
    fit_duration_ms = round((time.time() - t0) * 1000, 2)

    # -------------------------------------------------------------------
    # MÉTRIQUES DE PERFORMANCE
    # -------------------------------------------------------------------
    metrics: Dict[str, Any] = {}
    chart_data: Dict[str, Any] = {}

    if problem_type == "classification":
        metrics, chart_data, train_benchmark_score, test_benchmark_score = _compute_classification_metrics(
            model, X_train, y_train, X_test, y_test
        )
    else:
        # Régression
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        r2_train = float(max(-1.0, r2_score(y_train, y_pred_train)))
        r2_test = float(max(-1.0, r2_score(y_test, y_pred_test)))
        mae = float(mean_absolute_error(y_test, y_pred_test))
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))

        metrics = {
            "r2_test": round(r2_test, 4),
            "r2_train": round(r2_train, 4),
            "mae": round(mae, 4),
            "rmse": round(rmse, 4),
            "test_samples": len(y_test),
        }

        # Nuage Réel vs Prédit
        scatter_sample = []
        sample_indices = np.random.RandomState(42).choice(
            len(y_test), size=min(150, len(y_test)), replace=False
        )
        y_test_arr = np.array(y_test)
        y_pred_arr = np.array(y_pred_test)
        for idx in sample_indices:
            scatter_sample.append({
                "actual": round(float(y_test_arr[idx]), 2),
                "predicted": round(float(y_pred_arr[idx]), 2),
                "residual": round(float(y_pred_arr[idx] - y_test_arr[idx]), 2),
            })
        chart_data = {"actual_vs_predicted": scatter_sample}
        train_benchmark_score = max(0.0, r2_train)
        test_benchmark_score = max(0.0, r2_test)

    # -------------------------------------------------------------------
    # IMPORTANCES DES FEATURES
    # -------------------------------------------------------------------
    feature_importances = []
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        sorted_idx = np.argsort(fi)[::-1][:10]
        for idx in sorted_idx:
            feature_importances.append({
                "feature": feature_cols[idx],
                "importance": round(float(fi[idx]), 4),
                "importance_pct": round(float(fi[idx]) * 100, 1),
            })
    elif hasattr(model, "coef_"):
        coefs = np.abs(model.coef_)
        if coefs.ndim > 1:
            coefs = np.mean(coefs, axis=0)
        sorted_idx = np.argsort(coefs)[::-1][:10]
        max_c = np.max(coefs) + 1e-9
        for idx in sorted_idx:
            feature_importances.append({
                "feature": feature_cols[idx],
                "importance": round(float(coefs[idx]), 4),
                "importance_pct": round(float(coefs[idx] / max_c * 100), 1),
            })

    # -------------------------------------------------------------------
    # AUDIT CYBERSÉCURITÉ COMPLET SUR LE MODÈLE
    # -------------------------------------------------------------------
    cyber_poisoning = audit_poisoning_risk(X_train, y_train)
    cyber_adversarial = audit_adversarial_robustness(model, X_test, y_test, problem_type)
    cyber_membership = audit_membership_leakage(train_benchmark_score, test_benchmark_score)
    manifest_data = generate_cryptographic_manifest(
        model=model,
        model_name=f"{model_id}_{int(time.time())}",
        feature_names=feature_cols,
        metrics=metrics,
    )

    global_security_score = int(
        0.35 * cyber_adversarial["robustness_score"]
        + 0.35 * cyber_poisoning["robustness_score"]
        + 0.30 * cyber_membership["privacy_score"]
    )

    if global_security_score >= 80:
        sec_grade, sec_label, sec_color = "A+", "Modèle Hautement Sécurisé & Robuste", "var(--green-400)"
    elif global_security_score >= 65:
        sec_grade, sec_label, sec_color = "B", "Conforme avec Vigilance Modérée", "var(--teal-400)"
    else:
        sec_grade, sec_label, sec_color = "C", "Attention requise : Sensibilité aux attaques détectée", "var(--amber-400)"

    cyber_audit = {
        "global_score": global_security_score,
        "grade": sec_grade,
        "label": sec_label,
        "color": sec_color,
        "adversarial_robustness": cyber_adversarial,
        "data_poisoning": cyber_poisoning,
        "membership_inference": cyber_membership,
        "cryptographic_manifest": manifest_data["manifest"],
        "sha256": manifest_data["sha256"],
        "hmac": manifest_data["hmac"],
        "model_size_kb": round(manifest_data["model_size_bytes"] / 1024, 2),
    }

    # Sérialisation pour téléchargement sécurisé ultérieur
    buf = io.BytesIO()
    joblib.dump({
        "model": model,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "problem_type": problem_type,
        "hyperparams": params,
        "metrics": metrics,
        "sha256": manifest_data["sha256"],
        "rebalancing_strategy": rebalancing_strategy,
    }, buf)

    session.last_trained_model = {
        "model_id": model_id,
        "model_name": AVAILABLE_MODELS.get(problem_type, {}).get(model_id, {}).get("name", model_id),
        "problem_type": problem_type,
        "fit_duration_ms": fit_duration_ms,
        "metrics": metrics,
        "chart_data": chart_data,
        "feature_importances": feature_importances,
        "cyber_audit": cyber_audit,
        "model_bytes": buf.getvalue(),
        "rebalancing_info": rebal_info,
    }

    # -------------------------------------------------------------------
    # TABLEAU COMPARATIF : stocker les résultats pour comparaison
    # -------------------------------------------------------------------
    if not hasattr(session, "rebalancing_comparison"):
        session.rebalancing_comparison = {}

    session.rebalancing_comparison[rebalancing_strategy] = {
        "strategy": rebalancing_strategy,
        "strategy_label": rebal_info.get("strategy_label", rebalancing_strategy),
        "metrics": metrics,
        "n_train_after": rebal_info.get("n_after", len(y_train)),
    }

    return {
        "status": "success",
        "model_id": model_id,
        "model_name": session.last_trained_model["model_name"],
        "problem_type": problem_type,
        "fit_duration_ms": fit_duration_ms,
        "metrics": metrics,
        "chart_data": chart_data,
        "feature_importances": feature_importances,
        "cyber_audit": cyber_audit,
        "rebalancing_info": rebal_info,
        "rebalancing_comparison": session.rebalancing_comparison,
    }
