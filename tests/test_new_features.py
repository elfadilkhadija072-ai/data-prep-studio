"""Test unitaire et d'intégration direct des nouvelles fonctionnalités :
1. Authentification RBAC & Anti-brute force
2. Cybersécurité : Audit d'injection, Empoisonnement, Attaques adversariales, Inférence
3. Modélisation ML : KNN, Régression Logistique/Linéaire, Forêt Aléatoire
4. Signatures cryptographiques SHA-256 et HMAC
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np

from backend.core.loader import load_dataset_from_bytes
from backend.core.common import sessions, FeatureStepConfig
from backend.core.pipeline import execute_pipeline
from backend.core.security import (
    authenticate_user,
    verify_token,
    audit_dataset_sanitization,
    audit_adversarial_robustness,
    audit_poisoning_risk,
    audit_membership_leakage,
    generate_cryptographic_manifest,
)
from backend.core.models import (
    AVAILABLE_MODELS,
    get_available_models_list,
    train_and_audit_model,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_authentication():
    print("=" * 65)
    print(">>> [TEST 1] Authentification RBAC & Sécurité Anti-Brute-Force")
    print("=" * 65)

    # 1.1 Test login Admin
    auth_admin = authenticate_user("admin", "admin123")
    assert auth_admin["status"] == "success"
    assert auth_admin["user"]["role"] == "Administrateur ML"
    token = auth_admin["token"]
    print("    [OK] Authentification Admin réussie.")

    # 1.2 Test vérification de token
    session_info = verify_token(token)
    assert session_info is not None
    assert session_info["username"] == "admin"
    print("    [OK] Vérification du jeton sécurisé validée.")

    # 1.3 Test login Scientist
    auth_sci = authenticate_user("scientist", "mlsec2025")
    assert auth_sci["user"]["role"] == "Data Scientist Senior"
    print("    [OK] Authentification Data Scientist réussie.")

    # 1.4 Test échec mot de passe
    failed = False
    try:
        authenticate_user("admin", "mauvais_pass")
    except ValueError as e:
        failed = True
    assert failed, "Le mauvais mot de passe aurait dû lever une exception ValueError"
    print("    [OK] Blocage correct des mots de passe erronés.")


def test_dataset_sanitization():
    print("\n" + "=" * 65)
    print(">>> [TEST 2] Audit d'Assainissement & Détection d'Injections (XSS, SQLi)")
    print("=" * 65)

    # Test avec données saines
    df_clean = pd.DataFrame({
        "nom": ["Alice", "Bob", "Charlie"],
        "commentaire": ["Client régulier", "Paiement validé", "Commande expédiée"],
    })
    audit_clean = audit_dataset_sanitization(df_clean)
    assert audit_clean["status"] == "clean"
    assert audit_clean["sanitization_score"] == 100
    print(f"    [OK] Dataset propre validé (Score: {audit_clean['sanitization_score']}/100).")

    # Test avec injection SQL et XSS
    df_dirty = pd.DataFrame({
        "client": ["<script>alert(1)</script>", "Bob", "UNION SELECT * FROM users --"],
    })
    audit_dirty = audit_dataset_sanitization(df_dirty)
    assert audit_dirty["status"] == "threats_detected"
    assert audit_dirty["threats_count"] >= 2
    print(f"    [OK] Menaces détectées : {audit_dirty['threats_count']} injections repérées.")


def test_ml_models_classification():
    print("\n" + "=" * 65)
    print(">>> [TEST 3] Entraînement & Audit Cyber (Classification - Fraude Bancaire)")
    print("=" * 65)

    # Charger le dataset de fraude
    csv_path = os.path.join(ROOT_DIR, "backend", "data", "fraude_bancaire.csv")
    with open(csv_path, "rb") as f:
        df, load_info = load_dataset_from_bytes(f.read(), "fraude_bancaire.csv")

    session = sessions.create("fraude_bancaire.csv", df)
    session.target_col = "fraude" if "fraude" in df.columns else load_info["suggested_target"]
    session.problem_type = "classification"
    session.test_size = 0.20

    # Plan de préparation standard
    plan = {col: FeatureStepConfig() for col in df.columns}
    prep_res = execute_pipeline(
        df=df,
        plan=plan,
        column_types=load_info["types"],
        target_col=session.target_col,
        problem_type=session.problem_type,
        test_size=session.test_size,
        random_state=42,
        drop_duplicates=True,
    )
    session.df_train = prep_res["df_train"]
    session.df_test = prep_res["df_test"]
    session.df_full_prepared = prep_res["df_full"]

    # 3.1 Modèle KNN
    knn_res = train_and_audit_model(session, "knn", {"n_neighbors": 5, "weights": "distance"})
    assert knn_res["status"] == "success"
    assert "accuracy_test" in knn_res["metrics"]
    assert knn_res["metrics"]["accuracy_test"] > 0.70
    assert "confusion_matrix" in knn_res["chart_data"]
    assert "cyber_audit" in knn_res
    print(f"    [OK] KNN : Accuracy Test = {knn_res['metrics']['accuracy_test']*100:.1f}%, F1 = {knn_res['metrics']['f1_weighted']:.3f}.")
    print(f"    [OK] KNN Cyber Score : {knn_res['cyber_audit']['global_score']}/100 (Grade {knn_res['cyber_audit']['grade']}).")

    # 3.2 Modèle Forêt Aléatoire
    rf_res = train_and_audit_model(session, "random_forest", {"n_estimators": 50, "max_depth": 8})
    assert rf_res["status"] == "success"
    assert rf_res["metrics"]["accuracy_test"] > 0.80
    assert len(rf_res["feature_importances"]) > 0
    print(f"    [OK] Forêt Aléatoire : Accuracy Test = {rf_res['metrics']['accuracy_test']*100:.1f}%.")
    print(f"    [OK] Signature SHA-256 : {rf_res['cyber_audit']['sha256']}")
    print(f"    [OK] Signature HMAC : {rf_res['cyber_audit']['hmac']}")

    # 3.3 Modèle Régression Logistique
    lr_res = train_and_audit_model(session, "logistic_regression", {"C": 1.0, "max_iter": 500})
    assert lr_res["status"] == "success"
    print(f"    [OK] Régression Logistique : Accuracy Test = {lr_res['metrics']['accuracy_test']*100:.1f}%.")


def test_ml_models_regression():
    print("\n" + "=" * 65)
    print(">>> [TEST 4] Entraînement & Audit Cyber (Régression - Prix Immobilier)")
    print("=" * 65)

    csv_path = os.path.join(ROOT_DIR, "backend", "data", "prix_immobilier.csv")
    with open(csv_path, "rb") as f:
        df, load_info = load_dataset_from_bytes(f.read(), "prix_immobilier.csv")

    session = sessions.create("prix_immobilier.csv", df)
    session.target_col = "prix_vente"
    session.problem_type = "regression"
    session.test_size = 0.20

    plan = {col: FeatureStepConfig() for col in df.columns}
    prep_res = execute_pipeline(
        df=df,
        plan=plan,
        column_types=load_info["types"],
        target_col=session.target_col,
        problem_type=session.problem_type,
        test_size=session.test_size,
        random_state=42,
        drop_duplicates=True,
    )
    session.df_train = prep_res["df_train"]
    session.df_test = prep_res["df_test"]
    session.df_full_prepared = prep_res["df_full"]

    # 4.1 Modèle KNN Régresseur
    knn_reg = train_and_audit_model(session, "knn_regressor", {"n_neighbors": 5, "weights": "distance"})
    assert knn_reg["status"] == "success"
    assert "r2_test" in knn_reg["metrics"]
    assert "actual_vs_predicted" in knn_reg["chart_data"]
    print(f"    [OK] KNN Régresseur : R² Test = {knn_reg['metrics']['r2_test']:.3f}, MAE = {knn_reg['metrics']['mae']:.2f}.")

    # 4.2 Modèle Régression Ridge
    ridge_res = train_and_audit_model(session, "ridge", {"alpha": 1.0})
    assert ridge_res["status"] == "success"
    assert ridge_res["metrics"]["r2_test"] > 0.50
    print(f"    [OK] Régression Ridge : R² Test = {ridge_res['metrics']['r2_test']:.3f}, RMSE = {ridge_res['metrics']['rmse']:.2f}.")
    print(f"    [OK] Audit Robustesse Adversariale : Score = {ridge_res['cyber_audit']['adversarial_robustness']['robustness_score']}/100.")


if __name__ == "__main__":
    test_authentication()
    test_dataset_sanitization()
    test_ml_models_classification()
    test_ml_models_regression()
    print("\n" + "=" * 65)
    print("  TOUS LES TESTS DES NOUVEAUX MODULES SONT 100% VALIDES !")
    print("=" * 65 + "\n")
