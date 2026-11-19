"""Tests des endpoints de téléchargement et de génération de rapports Excel."""
import io
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import openpyxl
import pandas as pd
from backend.main import (
    upload_dataset,
    get_prep_plan,
    execute_prep_pipeline,
    download_file,
    train_model_endpoint,
)
from backend.core.common import ExecutePipelineRequest, FeatureStepConfig, TrainModelRequest
from backend.core.report import generate_excel_report


def test_download_workflow():
    # 1. Chargement du dataset fraude_bancaire.csv
    u_res = upload_dataset(demo="fraud")
    sid = u_res["session_id"]
    assert sid is not None

    # 2. Récupération du plan et exécution
    p_res = get_prep_plan(sid)
    plan = {k: FeatureStepConfig(**v["recommended_config"]) for k, v in p_res["features"].items()}
    exec_req = ExecutePipelineRequest(session_id=sid, plan=plan, drop_duplicates=True)
    exec_res = execute_prep_pipeline(exec_req)
    assert exec_res["train_shape"][0] == 800
    assert exec_res["test_shape"][0] == 200

    # 3. Test du téléchargement du rapport Excel (.xlsx)
    resp_xlsx = download_file(sid, "report_xlsx")
    assert resp_xlsx.headers["Content-Disposition"].endswith(".xlsx")
    assert resp_xlsx.media_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    wb = openpyxl.load_workbook(io.BytesIO(resp_xlsx.body))
    expected_sheets = ["Résumé", "Features", "Détail transformations", "Colonnes finales", "Corrélations"]
    assert wb.sheetnames == expected_sheets
    print(f"  [OK] Rapport Excel valide ({len(resp_xlsx.body)} octets) avec 5 feuilles : {wb.sheetnames}")

    # 4. Test des autres formats d'export
    for ftype, expected_ext, expected_shape in [
        ("full_csv", ".csv", (1000, 45)),
        ("full_xlsx", ".xlsx", (1000, 45)),
        ("train_csv", ".csv", (800, 45)),
        ("test_csv", ".csv", (200, 45)),
    ]:
        resp = download_file(sid, ftype)
        assert resp.headers["Content-Disposition"].endswith(expected_ext)
        if expected_ext == ".csv":
            df = pd.read_csv(io.BytesIO(resp.body))
        else:
            df = pd.read_excel(io.BytesIO(resp.body))
        assert df.shape == expected_shape
        print(f"  [OK] Export {ftype} valide : shape {df.shape}")

    # 5. Test train rééquilibré
    train_req = TrainModelRequest(
        session_id=sid,
        model_id="decision_tree",
        rebalancing_strategy="smote",
        rebalancing_params={"sampling_strategy": "auto"},
    )
    train_model_endpoint(train_req)
    resp_rebal = download_file(sid, "train_rebalanced")
    df_rebal = pd.read_csv(io.BytesIO(resp_rebal.body))
    assert not df_rebal.empty
    print(f"  [OK] Export train_rebalanced valide : shape {df_rebal.shape}")

    # 6. Test robustesse du rapport avec sections vides / None
    b_none = generate_excel_report(None, None, None, None)
    wb_none = openpyxl.load_workbook(io.BytesIO(b_none))
    assert wb_none.sheetnames == expected_sheets
    print("  [OK] Rapport résistant aux entrées vides/None")


if __name__ == "__main__":
    test_download_workflow()
    print("Tous les tests d'export ont réussi !")
