"""Application FastAPI principale exposant l'API REST de DataPrep Studio et servant le frontend."""
from __future__ import annotations

import io
import json
import logging
import os
from typing import Any, Dict, Optional
from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger("dataprep.backend")

from backend.core.common import (
    ConfigureRequest,
    ExecutePipelineRequest,
    FeatureStepConfig,
    LoginRequest,
    PreviewFeatureRequest,
    TrainModelRequest,
    sessions,
)
from backend.core.loader import load_dataset_from_bytes
from backend.core.eda import (
    compute_correlations_matrix,
    compute_features_table,
    compute_kpis,
    compute_scatter_data,
    get_column_distribution,
    get_data_quality_report,
)
from backend.core.analysis import (
    generate_full_analysis_plan,
    preview_feature_transformation,
)
from backend.core.pipeline import execute_pipeline
from backend.core.report import generate_excel_report
from backend.core.security import (
    authenticate_user,
    audit_dataset_sanitization,
    verify_token,
)
from backend.core.models import (
    get_available_models_list,
    train_and_audit_model,
)

app = FastAPI(
    title="DataPrep Studio API",
    description="Backend API de préparation, nettoyage et audit de données pour le Machine Learning",
    version="1.0.0",
)

# Configuration CORS pour le mode développement (Vite sur port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEMO_DATASETS = {
    "fraud": ("fraude_bancaire.csv", os.path.join(os.path.dirname(__file__), "data", "fraude_bancaire.csv")),
    "housing": ("prix_immobilier.csv", os.path.join(os.path.dirname(__file__), "data", "prix_immobilier.csv")),
}


# ---------------------------------------------------------------------------
# Routes 1. Importation et Configuration
# ---------------------------------------------------------------------------

@app.post("/api/upload")
def upload_dataset(
    file: Optional[UploadFile] = File(None),
    demo: Optional[str] = Query(None),
):
    """Charge un dataset par upload ou depuis l'un des jeux de démonstration."""
    if demo and isinstance(demo, str) and demo in DEMO_DATASETS:
        filename, path = DEMO_DATASETS[demo]
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail=f"Fichier de démo introuvable: {path}")
        with open(path, "rb") as f:
            file_bytes = f.read()
    elif file and hasattr(file, "file"):
        file_bytes = file.file.read()
        filename = file.filename or "dataset.csv"
    elif isinstance(demo, str):
        raise HTTPException(status_code=400, detail=f"Dataset de démo inconnu: {demo}")
    else:
        raise HTTPException(status_code=400, detail="Aucun fichier ou démo spécifié.")

    try:
        df, load_info = load_dataset_from_bytes(file_bytes, filename)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur de lecture du fichier : {str(exc)}")

    session = sessions.create(filename, df)
    session.column_types = load_info["types"]
    session.target_col = load_info.get("suggested_target")
    session.problem_type = load_info.get("suggested_problem", "classification")

    preview_rows = df.head(10).fillna("").to_dict(orient="records")

    return {
        "session_id": session.session_id,
        "filename": filename,
        "load_info": load_info,
        "preview_rows": preview_rows,
        "columns": list(df.columns),
        "types": session.column_types,
        "suggested_target": session.target_col,
        "suggested_problem": session.problem_type,
        "row_count": len(df),
        "column_count": df.shape[1],
    }


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable ou expirée.")
    return {
        "session_id": session.session_id,
        "filename": session.filename,
        "rows": len(session.df_raw),
        "cols": session.df_raw.shape[1],
        "columns": list(session.df_raw.columns),
        "types": session.column_types,
        "target_col": session.target_col,
        "problem_type": session.problem_type,
        "is_supervised": session.is_supervised,
        "test_size": session.test_size,
        "drop_duplicates": session.drop_duplicates,
        "has_executed": session.execution_result is not None,
    }


@app.post("/api/configure")
def configure_session(req: ConfigureRequest):
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")

    session.is_supervised = req.is_supervised
    session.target_col = req.target_col if req.is_supervised else None
    if req.problem_type:
        session.problem_type = req.problem_type
    session.test_size = req.test_size
    session.random_state = req.random_state
    session.drop_duplicates = req.drop_duplicates

    # Invalider tout résultat d'exécution précédent
    session.execution_result = None
    session.prep_plan = {}

    return {"status": "success", "session_id": session.session_id}


# ---------------------------------------------------------------------------
# Routes 2. Onglet EDA (Analyse exploratoire)
# ---------------------------------------------------------------------------

@app.get("/api/eda/kpis/{session_id}")
def get_kpis(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")
    return compute_kpis(session.df_raw, session.target_col, session.problem_type)


@app.get("/api/eda/features/{session_id}")
def get_features_table(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")
    return compute_features_table(
        session.df_raw,
        session.column_types,
        target_col=session.target_col,
        problem_type=session.problem_type,
    )


@app.get("/api/eda/distribution/{session_id}/{col}")
def get_distribution(
    session_id: str,
    col: str,
    zoom_p99: bool = Query(False),
    log_scale: bool = Query(False),
):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")
    if col not in session.df_raw.columns:
        raise HTTPException(status_code=400, detail=f"Colonne inconnue : {col}")

    c_type = session.column_types.get(col, "categorical")
    return get_column_distribution(
        session.df_raw,
        col,
        c_type,
        zoom_p99=zoom_p99,
        log_scale=log_scale,
        target_col=session.target_col,
        problem_type=session.problem_type,
    )


@app.get("/api/eda/correlations/{session_id}")
def get_correlations(session_id: str, method: str = Query("pearson")):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")
    return compute_correlations_matrix(session.df_raw, method=method)


@app.get("/api/eda/scatter/{session_id}")
def get_scatter(session_id: str, x: str = Query(...), y: str = Query(...)):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")
    if x not in session.df_raw.columns or y not in session.df_raw.columns:
        raise HTTPException(status_code=400, detail="Colonnes de scatter invalides.")
    return compute_scatter_data(session.df_raw, x, y, session.target_col)


@app.get("/api/eda/quality/{session_id}")
def get_quality(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")
    return get_data_quality_report(session.df_raw)


# ---------------------------------------------------------------------------
# Routes 3. Onglet Preprocessing & Recommandations
# ---------------------------------------------------------------------------

@app.get("/api/prep/plan/{session_id}")
def get_prep_plan(session_id: str):
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")
    plan_data = generate_full_analysis_plan(
        session.df_raw,
        session.column_types,
        session.target_col,
    )
    session.prep_plan = plan_data
    return plan_data


@app.post("/api/prep/preview_feature")
def preview_feature(req: PreviewFeatureRequest):
    """Prévisualisation avant/après d'une transformation sur une feature."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")

    fname = req.feature_name
    if not fname or fname not in session.df_raw.columns:
        raise HTTPException(status_code=400, detail=f"Feature introuvable : {fname}")

    c_type = session.column_types.get(fname, "categorical")
    return preview_feature_transformation(
        session.df_raw[fname],
        c_type,
        req.config,
    )


@app.post("/api/prep/execute")
def execute_prep_pipeline(req: ExecutePipelineRequest):
    """Exécute le pipeline complet de préparation avec anti-data-leakage."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")

    plan_cfg = req.plan
    drop_dups = req.drop_duplicates

    # Exécution stricte anti-leakage
    result = execute_pipeline(
        session.df_raw,
        plan_cfg,
        session.column_types,
        target_col=session.target_col,
        problem_type=session.problem_type,
        test_size=session.test_size,
        random_state=session.random_state,
        drop_duplicates=drop_dups,
    )

    session.execution_result = result
    session.df_train = result["df_train"]
    session.df_test = result["df_test"]
    session.df_full_prepared = result["df_full"]

    # Générer le rapport Excel
    corr_data = compute_correlations_matrix(session.df_full_prepared)
    features_analysis = session.prep_plan or generate_full_analysis_plan(
        session.df_raw, session.column_types, session.target_col
    )
    excel_bytes = generate_excel_report(
        {
            "filename": session.filename,
            "problem_type": session.problem_type,
            "target_col": session.target_col,
            "test_size": session.test_size,
            "random_state": session.random_state,
            "rows": len(session.df_raw),
            "cols": session.df_raw.shape[1],
        },
        features_analysis,
        result,
        corr_data,
    )
    session.excel_report_bytes = excel_bytes

    preview_head = result["df_full"].head(15).fillna("").to_dict(orient="records")

    return {
        "status": "success",
        "train_shape": result["train_shape"],
        "test_shape": result["test_shape"],
        "final_features_count": result["final_features_count"],
        "final_columns": result["final_columns"],
        "applied_table": result["applied_table"],
        "best_practices_checks": result["best_practices_checks"],
        "preview_head": preview_head,
    }


# ---------------------------------------------------------------------------
# Routes 4. Onglet Exportation et Téléchargement
# ---------------------------------------------------------------------------

@app.get("/api/export/summary/{session_id}")
def get_export_summary(session_id: str):
    session = sessions.get(session_id)
    if not session or not session.execution_result:
        raise HTTPException(status_code=400, detail="Aucun pipeline n'a encore été exécuté.")

    res = session.execution_result
    preview_head = res["df_full"].head(15).fillna("").to_dict(orient="records")
    return {
        "train_shape": res["train_shape"],
        "test_shape": res["test_shape"],
        "final_features_count": res["final_features_count"],
        "final_columns": res["final_columns"],
        "applied_table": res["applied_table"],
        "best_practices_checks": res["best_practices_checks"],
        "preview_head": preview_head,
    }


@app.get("/api/export/download/{session_id}/{file_type}")
def download_file(session_id: str, file_type: str):
    session = sessions.get(session_id)
    if not session or not session.execution_result:
        raise HTTPException(status_code=400, detail="Aucun résultat disponible pour téléchargement.")

    base_name = os.path.splitext(session.filename)[0]

    try:
        if file_type == "full_csv":
            if session.df_full_prepared is None:
                raise HTTPException(status_code=404, detail="Dataset préparé indisponible.")
            csv_bytes = session.df_full_prepared.to_csv(index=False).encode("utf-8")
            return Response(
                content=csv_bytes,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={base_name}_complet_prepare.csv"},
            )
        elif file_type == "full_xlsx":
            if session.df_full_prepared is None:
                raise HTTPException(status_code=404, detail="Dataset préparé indisponible.")
            buf = io.BytesIO()
            session.df_full_prepared.to_excel(buf, index=False, engine="openpyxl")
            return Response(
                content=buf.getvalue(),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename={base_name}_complet_prepare.xlsx"},
            )
        elif file_type == "train_csv":
            if session.df_train is None:
                raise HTTPException(status_code=404, detail="Jeu d'entraînement indisponible.")
            csv_bytes = session.df_train.to_csv(index=False).encode("utf-8")
            return Response(
                content=csv_bytes,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={base_name}_train.csv"},
            )
        elif file_type == "test_csv":
            if session.df_test is None:
                raise HTTPException(status_code=404, detail="Jeu de test indisponible.")
            csv_bytes = session.df_test.to_csv(index=False).encode("utf-8")
            return Response(
                content=csv_bytes,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={base_name}_test.csv"},
            )
        elif file_type in ("train_rebalanced_csv", "train_rebalanced"):
            df_rebal = getattr(session, "df_train_rebalanced", None)
            if df_rebal is None:
                raise HTTPException(status_code=404, detail="Jeu d'entraînement rééquilibré indisponible. Entraînez un modèle avec SMOTE ou Undersampling.")
            csv_bytes = df_rebal.to_csv(index=False).encode("utf-8")
            return Response(
                content=csv_bytes,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename={base_name}_train_rebalanced.csv"},
            )
        elif file_type in ("report_xlsx", "report_excel"):
            # Regénérer le rapport pour inclure les infos de rééquilibrage si un modèle a été entraîné
            rebal_info = session.last_trained_model.get("rebalancing_info") if session.last_trained_model else None
            features_analysis = session.prep_plan or generate_full_analysis_plan(
                session.df_raw, session.column_types, session.target_col
            )
            report_bytes = generate_excel_report(
                {
                    "filename": session.filename,
                    "problem_type": session.problem_type,
                    "target_col": session.target_col,
                    "test_size": session.test_size,
                    "random_state": session.random_state,
                    "rows": len(session.df_raw) if session.df_raw is not None else 0,
                    "cols": session.df_raw.shape[1] if session.df_raw is not None else 0,
                    "rebalancing_info": rebal_info,
                },
                features_analysis,
                session.execution_result or {},
                session.df_full_prepared,
            )
            session.excel_report_bytes = report_bytes
            return Response(
                content=report_bytes,
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": f"attachment; filename=Rapport_Preparation_{base_name}.xlsx"},
            )
        else:
            raise HTTPException(status_code=400, detail=f"Type de fichier inconnu : {file_type}")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            f"Erreur lors de la génération du fichier '{file_type}' pour la session {session_id} : {exc}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération du fichier '{file_type}' : {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Routes 5. Authentification & Cybersécurité
# ---------------------------------------------------------------------------

@app.post("/api/auth/login")
def login(req: LoginRequest):
    """Authentifie un utilisateur et retourne un jeton sécurisé."""
    try:
        return authenticate_user(req.username, req.password)
    except PermissionError as p_err:
        raise HTTPException(status_code=429, detail=str(p_err))
    except ValueError as v_err:
        raise HTTPException(status_code=401, detail=str(v_err))


@app.get("/api/auth/verify")
def check_auth(authorization: Optional[str] = Header(None)):
    """Vérifie le jeton transmis via le header Authorization."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Jeton absent.")
    token = authorization.replace("Bearer ", "").strip()
    session_data = verify_token(token)
    if not session_data:
        raise HTTPException(status_code=401, detail="Session expirée ou invalide.")
    return {"status": "authenticated", "user": session_data}


@app.get("/api/security/audit-data/{session_id}")
def audit_data_security(session_id: str):
    """Effectue un scan de cybersécurité et d'assainissement d'injections sur le dataset brut."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")
    return audit_dataset_sanitization(session.df_raw)


@app.get("/api/debugger/diagnostics/{session_id}")
def get_diagnostics(session_id: str):
    """Retourne l'état complet du système pour le débogage (mémoire, nan, intégrité, pipeline)."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")

    mem_raw = round(session.df_raw.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    mem_prep = round(session.df_full_prepared.memory_usage(deep=True).sum() / (1024 * 1024), 2) if session.df_full_prepared is not None else 0.0

    raw_nan_count = int(session.df_raw.isna().sum().sum())
    prep_nan_count = int(session.df_full_prepared.isna().sum().sum()) if session.df_full_prepared is not None else 0

    return {
        "session_id": session.session_id,
        "filename": session.filename,
        "raw_shape": [len(session.df_raw), session.df_raw.shape[1]],
        "raw_memory_mb": mem_raw,
        "raw_nan_count": raw_nan_count,
        "has_executed": session.execution_result is not None,
        "prep_shape": [len(session.df_full_prepared), session.df_full_prepared.shape[1]] if session.df_full_prepared is not None else None,
        "prep_memory_mb": mem_prep,
        "prep_nan_count": prep_nan_count,
        "target_col": session.target_col,
        "problem_type": session.problem_type,
        "is_supervised": session.is_supervised,
        "has_trained_model": session.last_trained_model is not None,
        "server_status": "optimal",
    }


# ---------------------------------------------------------------------------
# Routes 6. Modélisation Machine Learning (KNN, Régression, Arbres, etc.)
# ---------------------------------------------------------------------------

@app.get("/api/models/available/{session_id}")
def list_available_models(session_id: str):
    """Liste les algorithmes de Machine Learning adaptés au type de problème."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")
    problem_type = session.problem_type or "classification"
    return {
        "problem_type": problem_type,
        "target_col": session.target_col,
        "models": get_available_models_list(problem_type),
    }


@app.post("/api/models/train")
def train_model_endpoint(req: TrainModelRequest):
    """Entraîne un modèle, calcule les métriques et exécute l'audit cyber."""
    session = sessions.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")
    try:
        return train_and_audit_model(
            session,
            req.model_id,
            req.hyperparams,
            rebalancing_strategy=req.rebalancing_strategy,
            rebalancing_params=req.rebalancing_params,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erreur d'entraînement : {str(exc)}")


@app.get("/api/models/current/{session_id}")
def get_current_model(session_id: str):
    """Retourne les résultats et l'audit cyber du dernier modèle entraîné."""
    session = sessions.get(session_id)
    if not session or not session.last_trained_model:
        raise HTTPException(status_code=404, detail="Aucun modèle entraîné dans cette session.")
    mod = session.last_trained_model
    result = {
        "status": "success",
        "model_id": mod["model_id"],
        "model_name": mod["model_name"],
        "problem_type": mod["problem_type"],
        "fit_duration_ms": mod["fit_duration_ms"],
        "metrics": mod["metrics"],
        "chart_data": mod["chart_data"],
        "feature_importances": mod["feature_importances"],
        "cyber_audit": mod["cyber_audit"],
    }
    if "rebalancing_info" in mod:
        result["rebalancing_info"] = mod["rebalancing_info"]
    if "rebalancing_comparison" in mod:
        result["rebalancing_comparison"] = mod["rebalancing_comparison"]
    return result


@app.get("/api/models/download/{session_id}")
def download_trained_model(session_id: str):
    """Télécharge l'enveloppe du modèle entraîné avec intégrité SHA-256."""
    session = sessions.get(session_id)
    if not session or not session.last_trained_model:
        raise HTTPException(status_code=404, detail="Aucun modèle entraîné à télécharger.")
    
    model_bytes = session.last_trained_model["model_bytes"]
    model_name = session.last_trained_model["model_id"]
    return Response(
        content=model_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=dataprep_model_{model_name}.joblib"},
    )


@app.get("/api/models/manifest/{session_id}")
def download_model_manifest(session_id: str):
    """Télécharge le manifeste JSON cryptographique du modèle certifié."""
    session = sessions.get(session_id)
    if not session or not session.last_trained_model:
        raise HTTPException(status_code=404, detail="Aucun modèle disponible.")

    manifest_json = json.dumps(
        session.last_trained_model["cyber_audit"]["cryptographic_manifest"], indent=2
    ).encode("utf-8")
    return Response(
        content=manifest_json,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=model_cyber_manifest.json"},
    )


@app.get("/api/models/rebalancing-comparison/{session_id}")
def get_rebalancing_comparison(session_id: str):
    """Retourne le tableau comparatif des métriques pour chaque stratégie de rééquilibrage testée."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")
    comparison = getattr(session, "rebalancing_comparison", {})
    return {"comparison": list(comparison.values())}


@app.get("/api/models/download-rebalanced-train/{session_id}")
def download_rebalanced_train_csv(session_id: str):
    """Télécharge le dataset train rééquilibré en CSV (disponible après entraînement avec SMOTE/Undersampling)."""
    session = sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session introuvable.")
    df_rebal = getattr(session, "df_train_rebalanced", None)
    if df_rebal is None:
        raise HTTPException(status_code=404, detail="Aucun train rééquilibré disponible. Lancez un entraînement avec SMOTE ou Undersampling.")
    csv_bytes = df_rebal.to_csv(index=False).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=train_rebalanced.csv"},
    )


# ---------------------------------------------------------------------------
# Montage du Frontend compilé (Production)
# ---------------------------------------------------------------------------

dist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend", "dist"))

if os.path.exists(dist_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(dist_dir, "assets")), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        file_path = os.path.join(dist_dir, full_path)
        if full_path and os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(
            os.path.join(dist_dir, "index.html"),
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
else:
    @app.get("/")
    def root_placeholder():
        return {
            "message": "DataPrep Studio API active.",
            "status": "Frontend non compilé. Exécutez 'npm run build' dans frontend/ ou lancez 'python run.py'.",
            "docs": "/docs",
        }
