"""Script de test automatisé de l'ensemble du pipeline DataPrep ML."""
import io
import zipfile
import pandas as pd
from modules.loader import CSVLoadError, load_csv
from modules.profiling import classify_dataframe
from modules.target import suggest_target
from modules.pipeline import PipelineConfig, execute_preparation
from modules.report import build_preparation_report, create_download_zip


def test_clean_csv():
    print("--- Test 1 : CSV propre (1_propre.csv) ---")
    with open("tests_csv/1_propre.csv", "rb") as f:
        df, info = load_csv(f.read())
    assert df.shape == (200, 5), f"Shape inattendu: {df.shape}"
    types = classify_dataframe(df)
    target = suggest_target(df)
    assert target.column == "achat", f"Cible inattendue: {target.column}"
    assert target.problem_type == "classification_binaire"

    cfg = PipelineConfig(target_col=target.column, problem_type=target.problem_type)
    res = execute_preparation(df, cfg, types)
    assert res.X_train.shape[0] == 160
    assert res.X_test.shape[0] == 40
    print("Test 1 PASSÉ avec succès !")


def test_latin1_pointvirgule():
    print("\n--- Test 2 : Latin-1 & point-virgule (2_latin1_pointvirgule.csv) ---")
    with open("tests_csv/2_latin1_pointvirgule.csv", "rb") as f:
        df, info = load_csv(f.read())
    assert info.separator == ";", f"Séparateur inattendu: {info.separator}"
    types = classify_dataframe(df)
    target = suggest_target(df)
    cfg = PipelineConfig(target_col=target.column, problem_type=target.problem_type)
    res = execute_preparation(df, cfg, types)
    assert res.X_train.shape[0] == 160
    print("Test 2 PASSÉ avec succès !")


def test_missing_dates_id_text():
    print("\n--- Test 3 : Manquantes, dates, id et texte (3_manquantes_dates_id.csv) ---")
    with open("tests_csv/3_manquantes_dates_id.csv", "rb") as f:
        df, info = load_csv(f.read())
    types = classify_dataframe(df)
    assert types["id"] == "identifiant"
    assert types["date_inscription"] == "date_heure"
    assert types["colonne_vide"] == "constante"

    target = suggest_target(df)
    assert target.column == "cible"

    cfg = PipelineConfig(
        target_col=target.column,
        problem_type=target.problem_type,
        numeric_scaler="RobustScaler",
        outlier_action="clip",
        date_features_enabled=True,
        text_strategy="tfidf",
    )
    res = execute_preparation(df, cfg, types)
    rep = build_preparation_report(cfg, types, res, "3_manquantes_dates_id.csv")
    zip_bytes = create_download_zip(res, rep)

    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    assert "pipeline_preprocesseur.joblib" in zf.namelist()
    assert "rapport_preparation.json" in zf.namelist()
    print("Test 3 PASSÉ avec succès ! Fichiers dans zip:", len(zf.namelist()))


def test_empty_csv():
    print("\n--- Test 4 : Fichier vide (4_vide.csv) ---")
    with open("tests_csv/4_vide.csv", "rb") as f:
        try:
            load_csv(f.read())
            assert False, "Aurait dû lever une CSVLoadError !"
        except CSVLoadError as e:
            print("Test 4 PASSÉ : CSVLoadError bien levée :", e)


if __name__ == "__main__":
    test_clean_csv()
    test_latin1_pointvirgule()
    test_missing_dates_id_text()
    test_empty_csv()
    print("\nTOUS LES TESTS SONT PASSES AVEC SUCCES !")
