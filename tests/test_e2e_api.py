"""Test d'intégration validant toutes les routes API et le flux complet de bout en bout."""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.main import (
    upload_dataset,
    configure_session,
    get_kpis,
    get_features_table,
    get_distribution,
    get_correlations,
    get_scatter,
    get_quality,
    get_prep_plan,
    preview_feature,
    execute_prep_pipeline,
    get_export_summary,
    download_file,
)
from backend.core.common import ConfigureRequest, ExecutePipelineRequest, FeatureStepConfig, PreviewFeatureRequest


def run_e2e():
    print(">>> 1. Chargement du dataset de démonstration 'fraud'...")
    res = upload_dataset(file=None, demo="fraud")
    session_id = res["session_id"]
    assert session_id is not None
    assert res["load_info"]["rows"] == 1000
    assert "is_fraud" in res["columns"]
    print(f"    [SUCCES] Dataset chargé. Session: {session_id}, Lignes: {res['load_info']['rows']}")

    print(">>> 2. Configuration du problème ML (Classification supervisée)...")
    config_payload = ConfigureRequest(
        session_id=session_id,
        is_supervised=True,
        target_col="is_fraud",
        problem_type="classification",
        test_size=0.20,
        random_state=42,
        drop_duplicates=True,
    )
    conf_res = configure_session(config_payload)
    assert conf_res["status"] == "success"
    print("    [SUCCES] Configuration enregistrée.")

    print(">>> 3. Endpoints EDA...")
    kpis = get_kpis(session_id)
    assert kpis["rows"] == 1000
    assert kpis["target_info"]["is_imbalanced"] is True
    print(f"    [SUCCES] KPIs récupérés : {kpis['rows']} lignes, {kpis['missing_pct']}% manquants.")

    features = get_features_table(session_id)
    assert len(features) == 14
    print(f"    [SUCCES] Tableau des features récupéré ({len(features)} colonnes).")

    dist = get_distribution(session_id, "montant", zoom_p99=True, log_scale=False)
    assert len(dist["histogram"]) > 0
    print("    [SUCCES] Distribution univariée calculée.")

    corr = get_correlations(session_id, method="pearson")
    assert "matrix" in corr
    print("    [SUCCES] Matrice de corrélation générée.")

    scatter = get_scatter(session_id, x="montant", y="score_risque_client")
    assert len(scatter) > 0
    print("    [SUCCES] Nuage de points généré.")

    qual = get_quality(session_id)
    assert "missing_by_column" in qual
    print("    [SUCCES] Rapport qualité calculé.")

    print(">>> 4. Récupération du plan de preprocessing & recommandations...")
    plan_data = get_prep_plan(session_id)
    assert "features" in plan_data
    features_plan = plan_data["features"]
    print(f"    [SUCCES] Plan généré pour {len(features_plan)} features.")

    print(">>> 5. Test de la prévisualisation live d'une feature...")
    prev = preview_feature(req=PreviewFeatureRequest(
        session_id=session_id,
        feature_name="montant",
        config=FeatureStepConfig(
            impute_strategy="median",
            outlier_strategy="winsorize_iqr",
            transform_strategy="log1p",
            scaling_strategy="standard",
            encoding_strategy="none",
        ),
    ))
    assert not prev["is_dropped"]
    assert prev["stats_after"]["missing"] == 0
    print("    [SUCCES] Prévisualisation avant/après validée.")

    print(">>> 6. Exécution du pipeline complet anti-data-leakage...")
    exec_plan = {
        col: FeatureStepConfig(**config["recommended_config"])
        for col, config in features_plan.items()
    }
    exec_res = execute_prep_pipeline(req=ExecutePipelineRequest(
        session_id=session_id,
        plan=exec_plan,
        drop_duplicates=True,
    ))
    assert exec_res["status"] == "success"
    print(f"    [SUCCES] Pipeline exécuté : {exec_res['train_shape']} train, {exec_res['test_shape']} test.")

    print(">>> 7. Vérification du résumé d'export...")
    export_summary = get_export_summary(session_id)
    assert len(export_summary["applied_table"]) > 0
    print("    [SUCCES] Bilan d'export validé.")

    print(">>> 8. Vérification des flux de téléchargement...")
    rep_res = download_file(session_id, "report_xlsx")
    assert len(rep_res.body) > 1000
    print(f"    [SUCCES] Rapport Excel téléchargé ({len(rep_res.body)} octets).")

    train_res = download_file(session_id, "train_csv")
    assert b"is_fraud" in train_res.body
    print(f"    [SUCCES] train.csv téléchargé ({len(train_res.body)} octets).")

    test_res = download_file(session_id, "test_csv")
    assert b"is_fraud" in test_res.body
    print(f"    [SUCCES] test.csv téléchargé ({len(test_res.body)} octets).")

    print("\n=======================================================")
    print("  TOUTES LES ROUTES ET LE FLUX E2E SONT OPERATIONNELS !")
    print("=======================================================")


if __name__ == "__main__":
    run_e2e()
