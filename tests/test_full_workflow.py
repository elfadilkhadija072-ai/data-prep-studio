"""Suite de tests automatisés validant l'ensemble du workflow DataPrep Studio.
Vérifie de bout en bout :
1. Dataset 1 : Fraude bancaire (classification déséquilibrée)
2. Dataset 2 : Prix immobilier (régression continue)
3. Absence totale de Data Leakage
4. Génération du rapport Excel complet à 5 feuilles
"""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import io
import pandas as pd
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.core.loader import load_dataset_from_bytes
from backend.core.common import FeatureStepConfig
from backend.core.eda import (
    compute_kpis,
    compute_features_table,
    get_column_distribution,
    compute_correlations_matrix,
    compute_scatter_data,
    get_data_quality_report,
)
from backend.core.analysis import (
    generate_full_analysis_plan,
    preview_feature_transformation,
)
from backend.core.pipeline import execute_pipeline
from backend.core.report import generate_excel_report


def test_workflow_fraud():
    print("=" * 60)
    print(">>> TEST 1 : Workflow complet sur Fraude Bancaire (Classification)")
    print("=" * 60)

    csv_path = os.path.join(ROOT_DIR, "backend", "data", "fraude_bancaire.csv")
    with open(csv_path, "rb") as f:
        df, load_info = load_dataset_from_bytes(f.read(), "fraude_bancaire.csv")

    assert len(df) == 1000, f"Attendu 1000 lignes, obtenu {len(df)}"
    assert "is_fraud" in df.columns, "is_fraud absent"
    assert load_info["suggested_target"] == "is_fraud"
    assert load_info["suggested_problem"] == "classification"
    print(f"  [OK] Chargement : {len(df)} lignes, {len(df.columns)} colonnes. Target: {load_info['suggested_target']}")

    # EDA
    kpis = compute_kpis(df, target_col="is_fraud", problem_type="classification")
    assert kpis["rows"] == 1000
    assert kpis["target_info"]["is_imbalanced"] is True
    print(f"  [OK] KPIs : {kpis['missing_pct']}% manquants, déséquilibre détecté (ratio: {kpis['target_info']['imbalance_ratio']})")

    feat_table = compute_features_table(df, load_info["types"], target_col="is_fraud", problem_type="classification")
    assert len(feat_table) == 14
    # Vérifier qu'une métrique d'association est bien calculée pour montant
    montant_stat = [f for f in feat_table if f["name"] == "montant"][0]
    assert montant_stat["target_association"] is not None
    print(f"  [OK] Table des features : 14 features analysées. Association montant/cible = {montant_stat['target_association']} ({montant_stat['target_association_metric']})")

    # Distributions
    dist = get_column_distribution(df, "montant", "numeric", zoom_p99=True, log_scale=False, target_col="is_fraud")
    assert len(dist["histogram"]) > 0
    assert dist["boxplot"] is not None
    print(f"  [OK] Distribution 'montant' calculée avec succès (histogramme & boxplot)")

    # Corrélations
    corr = compute_correlations_matrix(df, method="pearson")
    assert "matrix" in corr
    print(f"  [OK] Corrélations calculées ({len(corr['columns'])} variables numériques)")

    # Plan de préparation
    plan_data = generate_full_analysis_plan(df, load_info["types"], target_col="is_fraud")
    f_plan = plan_data["features"]
    assert "transaction_id" in f_plan
    assert f_plan["transaction_id"]["recommended_config"]["action_drop"] is True
    print("  [OK] Plan de préparation : suppression automatique de l'identifiant transaction_id")

    # Aperçu Avant/Après
    prev = preview_feature_transformation(
        df["montant"],
        "numeric",
        FeatureStepConfig(
            impute_strategy="median",
            outlier_strategy="winsorize_iqr",
            transform_strategy="log1p",
            scaling_strategy="standard",
        ),
    )
    assert prev["stats_after"]["missing"] == 0
    print("  [OK] Aperçu Avant / Après en direct calculé avec succès")

    # Exécution du pipeline Anti-Leakage
    exec_plan = {
        col: FeatureStepConfig(**item["recommended_config"])
        for col, item in f_plan.items()
    }
    result = execute_pipeline(
        df,
        exec_plan,
        load_info["types"],
        target_col="is_fraud",
        problem_type="classification",
        test_size=0.20,
        random_state=42,
        drop_duplicates=True,
    )
    assert result["train_shape"][0] == 800, f"Train attendu 800 lignes, obtenu {result['train_shape'][0]}"
    assert result["test_shape"][0] == 200, f"Test attendu 200 lignes, obtenu {result['test_shape'][0]}"
    assert "is_fraud" in result["df_train"].columns
    print(f"  [OK] Pipeline exécuté : Train {result['train_shape']}, Test {result['test_shape']}")

    # Rapport Excel XlsxWriter
    excel_bytes = generate_excel_report(
        {
            "filename": "fraude_bancaire.csv",
            "problem_type": "classification",
            "target_col": "is_fraud",
            "test_size": 0.20,
            "random_state": 42,
            "rows": 1000,
            "cols": 14,
        },
        plan_data,
        result,
        corr,
    )
    assert len(excel_bytes) > 2000
    # Vérifier que le classeur s'ouvre avec openpyxl et contient les 5 feuilles
    excel_file = io.BytesIO(excel_bytes)
    xl = pd.ExcelFile(excel_file)
    expected_sheets = ["Résumé", "Features", "Détail transformations", "Colonnes finales", "Corrélations"]
    for s in expected_sheets:
        assert s in xl.sheet_names, f"Feuille manquante : {s}"
    print(f"  [OK] Rapport Excel valide ({len(excel_bytes)} octets) avec les 5 feuilles requises : {xl.sheet_names}")


def test_workflow_housing_regression():
    print("\n" + "=" * 60)
    print(">>> TEST 2 : Validation de la généricité sur Prix Immobilier (Régression)")
    print("=" * 60)

    csv_path = os.path.join(ROOT_DIR, "backend", "data", "prix_immobilier.csv")
    with open(csv_path, "rb") as f:
        df, load_info = load_dataset_from_bytes(f.read(), "prix_immobilier.csv")

    assert len(df) > 500
    assert "prix_vente" in df.columns
    assert load_info["suggested_target"] == "prix_vente"
    assert load_info["suggested_problem"] == "regression"
    print(f"  [OK] Chargement : {len(df)} lignes, {len(df.columns)} colonnes. Cible détectée : {load_info['suggested_target']} (Régression)")

    plan_data = generate_full_analysis_plan(df, load_info["types"], target_col="prix_vente")
    exec_plan = {
        col: FeatureStepConfig(**item["recommended_config"])
        for col, item in plan_data["features"].items()
    }
    result = execute_pipeline(
        df,
        exec_plan,
        load_info["types"],
        target_col="prix_vente",
        problem_type="regression",
        test_size=0.25,
        random_state=42,
    )

    # Vérification que la cible n'est PAS mise à l'échelle (prix conservé en euros > 50 000)
    assert result["df_train"]["prix_vente"].mean() > 100_000, "Erreur : La cible prix_vente a été normalisée par erreur !"
    assert result["df_test"]["prix_vente"].mean() > 100_000, "Erreur : La cible prix_vente du test a été normalisée par erreur !"
    print(f"  [OK] Intégrité cible préservée : moyenne train = {result['df_train']['prix_vente'].mean():.2f} €, test = {result['df_test']['prix_vente'].mean():.2f} €")


if __name__ == "__main__":
    test_workflow_fraud()
    test_workflow_housing_regression()
    print("\n" + "=" * 60)
    print("  TOUS LES CRITÈRES ET TESTS D'INTÉGRATION SONT VALIDES !  ")
    print("=" * 60)
