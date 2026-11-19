"""Tests automatisés du module de rééquilibrage et de l'étanchéité anti-data-leakage."""
import numpy as np
import pandas as pd
from backend.core.rebalancing import apply_rebalancing
from backend.core.models import train_and_audit_model
from backend.core.pipeline import execute_pipeline
from backend.core.common import DatasetSession, FeatureStepConfig
from backend.core.eda import compute_kpis


def test_rebalancing_train_only():
    """Vérifie que SMOTE s'applique UNIQUEMENT sur le Train et respecte l'anti-leakage."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "feat1": np.random.randn(n),
        "feat2": np.random.randn(n),
    })
    y = pd.Series([0] * 190 + [1] * 10, name="target")

    # 1. Test SMOTE
    X_smote, y_smote, info_smote = apply_rebalancing(
        X, y, strategy="smote", params={"k_neighbors": 5, "sampling_strategy": "auto"}
    )
    assert len(X_smote) > len(X), "SMOTE doit augmenter le nombre de lignes du train"
    assert y_smote.value_counts()[1] == y_smote.value_counts()[0], "La classe minoritaire doit être équilibrée"
    assert info_smote["strategy"] == "smote"

    # 2. Test Undersampling
    X_under, y_under, info_under = apply_rebalancing(
        X, y, strategy="undersample", params={"sampling_strategy": "auto"}
    )
    assert len(X_under) < len(X), "Undersampling doit réduire le nombre de lignes du train"
    assert y_under.value_counts()[0] == 10, "La classe majoritaire doit être réduite à 10"
    assert info_under["strategy"] == "undersample"


def test_smote_k_neighbors_auto_adjustment():
    """Vérifie que k_neighbors est automatiquement ajusté si la classe minoritaire a < k+1 échantillons."""
    X = pd.DataFrame({"feat1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
    y = pd.Series([0, 0, 0, 0, 1, 1, 1], name="target")  # Classe 1 n'a que 3 échantillons

    X_smote, y_smote, info = apply_rebalancing(
        X, y, strategy="smote", params={"k_neighbors": 5}
    )
    assert info["k_neighbors_used"] == 2, f"Attendu k=2, obtenu {info['k_neighbors_used']}"


def test_smote_nan_detection():
    """Vérifie que SMOTE lève une exception explicite en présence de NaN."""
    X = pd.DataFrame({"feat1": [1.0, np.nan, 3.0, 4.0]})
    y = pd.Series([0, 0, 1, 1])
    raised = False
    try:
        apply_rebalancing(X, y, strategy="smote")
    except ValueError as e:
        assert "NaN détectées" in str(e)
        raised = True
    assert raised, "Une ValueError devait être levée en présence de NaN"


def test_eda_kpi_formula_and_missing_target():
    """Vérifie le calcul correct du ratio de déséquilibre et la détection des cibles manquantes."""
    target_vals = [0] * 905 + [1] * 45 + [np.nan] * 50
    df = pd.DataFrame({"is_fraud": target_vals, "montant": range(1000)})

    kpis = compute_kpis(df, target_col="is_fraud", problem_type="classification")
    t_info = kpis["target_info"]

    assert t_info["total_labeled"] == 950
    assert t_info["minority_pct"] == 4.74
    assert t_info["missing_target_count"] == 50
    assert t_info["imbalance_ratio"] == 20.11


def test_pipeline_drops_missing_target_before_split():
    """Vérifie que les lignes avec cible manquante sont purgées avant le split."""
    target_vals = [0] * 90 + [1] * 10 + [np.nan] * 10
    df = pd.DataFrame({"target": target_vals, "feat": range(110)})
    plan = {"feat": FeatureStepConfig()}
    column_types = {"feat": "numeric", "target": "categorical"}

    res = execute_pipeline(
        df=df,
        plan=plan,
        column_types=column_types,
        target_col="target",
        problem_type="classification",
        test_size=0.2,
        drop_duplicates=False,
    )

    train_n = res["train_shape"][0]
    test_n = res["test_shape"][0]
    assert train_n + test_n == 100, f"Attendu 100 lignes au total, obtenu {train_n + test_n}"


if __name__ == "__main__":
    test_rebalancing_train_only()
    test_smote_k_neighbors_auto_adjustment()
    test_smote_nan_detection()
    test_eda_kpi_formula_and_missing_target()
    test_pipeline_drops_missing_target_before_split()
    print("TOUS LES TESTS ONT RÉUSSI AVEC SUCCÈS !")
