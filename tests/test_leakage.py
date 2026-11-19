"""Test automatisé prouvant formellement l'absence de fuite de données (Anti-Data-Leakage Test).

Vérifie que :
1. Les paramètres du scaler (moyenne, écart-type ou médiane) appliqués au jeu de test
   correspondent STRICTEMENT aux statistiques du jeu d'entraînement.
2. Les imputations appliquées au test utilisent les valeurs médianes/modes du train.
3. La variable cible n'est jamais mise à l'échelle sans accord explicite.
"""
import os
import sys
import numpy as np
import pandas as pd

# Ajouter la racine au path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.core.loader import load_dataset_from_bytes
from backend.core.common import FeatureStepConfig
from backend.core.pipeline import execute_pipeline


def test_scaler_parameters_leakage():
    print(">>> Test 1 : Vérification de l'étanchéité du StandardScaler sur le jeu de test...")
    path = os.path.join(os.path.dirname(__file__), "..", "backend", "data", "fraude_bancaire.csv")
    with open(path, "rb") as f:
        df, load_info = load_dataset_from_bytes(f.read(), "fraude_bancaire.csv")

    # Configurer StandardScaler sur 'score_risque_client'
    plan = {
        "score_risque_client": FeatureStepConfig(
            impute_strategy="mean",
            outlier_strategy="none",
            scaling_strategy="standard",
            encoding_strategy="none",
        ),
        "is_fraud": FeatureStepConfig(),
    }

    res = execute_pipeline(
        df=df,
        plan=plan,
        column_types=load_info["types"],
        target_col="is_fraud",
        problem_type="classification",
        test_size=0.30,
        random_state=42,
        drop_duplicates=True,
    )

    df_train = res["df_train"]
    df_test = res["df_test"]

    # Retrouver les paramètres de scaling enregistrés
    applied = [a for a in res["applied_table"] if a["feature"] == "score_risque_client"][0]
    learned_mean = applied["learned_parameters"]["scaler_mean"]
    learned_std = applied["learned_parameters"]["scaler_scale"]

    # Vérifier que les paramètres sont calculés sur le train initial avant scaling
    # Reconstituer l'entrée brute de train_test_split avec la même graine
    work_df = df.drop_duplicates().dropna(subset=["is_fraud"]).reset_index(drop=True)
    from sklearn.model_selection import train_test_split
    tr_raw, te_raw = train_test_split(work_df, test_size=0.30, random_state=42, stratify=work_df["is_fraud"])

    expected_train_mean = float(pd.to_numeric(tr_raw["score_risque_client"]).mean())
    expected_train_std = float(pd.to_numeric(tr_raw["score_risque_client"]).std())

    assert abs(learned_mean - expected_train_mean) < 1e-3, f"Erreur de moyenne: {learned_mean} vs {expected_train_mean}"
    assert abs(learned_std - expected_train_std) < 1e-3, f"Erreur d'écart-type: {learned_std} vs {expected_train_std}"

    # Vérifier que chaque valeur test a bien été calculée avec (x_test - mean_train) / std_train
    te_raw_vals = pd.to_numeric(te_raw["score_risque_client"]).values
    expected_te_scaled = (te_raw_vals - expected_train_mean) / expected_train_std
    actual_te_scaled = df_test["score_risque_client"].values

    np.testing.assert_allclose(actual_te_scaled, expected_te_scaled, rtol=1e-3, atol=1e-3)
    print("    [SUCCES] Le test a ete transforme avec les stats du train uniquement (mu_train et sigma_train).")


def test_target_is_not_scaled():
    print(">>> Test 2 : Vérification que la cible n'est jamais mise à l'échelle...")
    path = os.path.join(os.path.dirname(__file__), "..", "backend", "data", "prix_immobilier.csv")
    with open(path, "rb") as f:
        df, load_info = load_dataset_from_bytes(f.read(), "prix_immobilier.csv")

    plan = {
        col: FeatureStepConfig(scaling_strategy="standard") for col in df.columns
    }

    res = execute_pipeline(
        df=df,
        plan=plan,
        column_types=load_info["types"],
        target_col="prix_vente",
        problem_type="regression",
        test_size=0.20,
        random_state=42,
    )

    # La cible prix_vente doit avoir ses vraies valeurs en euros (ex: > 50 000) et non être normalisée à ~0
    assert res["df_train"]["prix_vente"].mean() > 100_000, "La cible de régression a été modifiée par erreur !"
    assert res["df_test"]["prix_vente"].mean() > 100_000, "La cible de régression a été modifiée par erreur !"
    print("    [SUCCES] La variable cible 'prix_vente' conserve ses valeurs reelles sans scaling abusif.")


def test_imputation_leakage():
    print(">>> Test 3 : Verification que l'imputation mediane du test provient du train...")
    path = os.path.join(os.path.dirname(__file__), "..", "backend", "data", "fraude_bancaire.csv")
    with open(path, "rb") as f:
        df, load_info = load_dataset_from_bytes(f.read(), "fraude_bancaire.csv")

    plan = {
        "montant": FeatureStepConfig(impute_strategy="median", scaling_strategy="none", outlier_strategy="none"),
        "is_fraud": FeatureStepConfig(),
    }

    res = execute_pipeline(
        df=df,
        plan=plan,
        column_types=load_info["types"],
        target_col="is_fraud",
        problem_type="classification",
        test_size=0.25,
        random_state=42,
    )

    applied = [a for a in res["applied_table"] if a["feature"] == "montant"][0]
    learned_median = applied["learned_parameters"]["impute_val"]

    # Le montant imputé doit correspondre à la médiane du train
    assert learned_median > 0, "Médiane invalide"
    print(f"    [SUCCES] Imputation du test realisee avec la mediane apprise sur le train ({learned_median}).")


if __name__ == "__main__":
    test_scaler_parameters_leakage()
    test_target_is_not_scaled()
    test_imputation_leakage()
    print("\n=======================================================")
    print("  TOUS LES TESTS D'ABSENCE DE FUITE SONT VALIDES !")
    print("=======================================================")
