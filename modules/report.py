"""Génération des rapports d'audit et préparation des exports (ZIP, CSV, Joblib, JSON)."""
from __future__ import annotations

import io
import json
import zipfile
from typing import Any
import joblib
import pandas as pd
from modules.pipeline import PipelineConfig, PreparationResult


def build_preparation_report(
    config: PipelineConfig,
    column_types: dict[str, str],
    result: PreparationResult,
    filename: str,
) -> dict[str, Any]:
    """Construit un rapport complet et structuré des choix de prétraitement."""
    # Colonnes retenues et exclues
    features_in = [
        c for c in column_types.keys()
        if c != config.target_col
        and c not in config.excluded_columns
        and column_types[c] not in ("constante", "identifiant")
    ]
    excluded_or_dropped = [
        c for c in column_types.keys()
        if c != config.target_col and c not in features_in
    ]

    report = {
        "fichier_source": filename,
        "cible": {
            "nom": config.target_col,
            "type_probleme": config.problem_type,
            "lignes_cible_manquante_supprimees": result.n_dropped_missing_target,
        },
        "separation_donnees": {
            "test_size": config.test_size,
            "random_state": config.random_state,
            "stratification": config.stratify,
            "lignes_train": int(result.X_train.shape[0]),
            "lignes_test": int(result.X_test.shape[0]),
        },
        "dimensions": {
            "features_initiales": len(features_in),
            "features_apres_encodage": int(result.X_train.shape[1]),
            "lignes_outliers_supprimees": result.n_dropped_outliers,
        },
        "colonnes_retenues": features_in,
        "colonnes_exclues_ou_supprimees": excluded_or_dropped,
        "strategies_appliquees": {
            "mise_a_l_echelle": config.numeric_scaler,
            "imputation_numerique": config.numeric_imputer,
            "traitement_outliers": f"{config.outlier_action} ({config.outlier_method.upper()}, facteur {config.outlier_factor})",
            "imputation_categorielle": config.categorical_imputer,
            "encodage_categoriel": config.categorical_encoder,
            "frequence_min_categories_rares": config.rare_categories_min_freq,
            "extraction_dates": config.date_features_enabled,
            "traitement_texte": config.text_strategy,
        },
        "noms_features_finales": result.feature_names,
        "avertissements": result.warnings,
    }
    return report


def create_download_zip(
    result: PreparationResult,
    report_dict: dict[str, Any],
) -> bytes:
    """Regroupe tous les artefacts préparés dans une archive ZIP prête au téléchargement."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Jeux de données CSV
        zf.writestr("X_train.csv", result.X_train.to_csv(index=False).encode("utf-8"))
        zf.writestr("X_test.csv", result.X_test.to_csv(index=False).encode("utf-8"))
        zf.writestr("y_train.csv", pd.DataFrame({report_dict["cible"]["nom"]: result.y_train}).to_csv(index=False).encode("utf-8"))
        zf.writestr("y_test.csv", pd.DataFrame({report_dict["cible"]["nom"]: result.y_test}).to_csv(index=False).encode("utf-8"))

        # Dataset complet préparé concaténé
        full_X = pd.concat([result.X_train, result.X_test], axis=0)
        full_y = pd.concat([result.y_train, result.y_test], axis=0)
        full_df = full_X.copy()
        full_df["__TARGET__" + report_dict["cible"]["nom"]] = full_y.values
        zf.writestr("dataset_complet_prepare.csv", full_df.to_csv(index=False).encode("utf-8"))

        # 2. Pipeline scikit-learn sérialisée avec joblib
        joblib_buffer = io.BytesIO()
        joblib.dump(result.pipeline, joblib_buffer)
        zf.writestr("pipeline_preprocesseur.joblib", joblib_buffer.getvalue())

        # 3. Rapport JSON d'audit
        json_report = json.dumps(report_dict, indent=2, ensure_ascii=False)
        zf.writestr("rapport_preparation.json", json_report.encode("utf-8"))

    return zip_buffer.getvalue()
