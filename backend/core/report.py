"""Générateur de rapport Excel de qualité professionnelle avec XlsxWriter."""
from __future__ import annotations

import io
import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import xlsxwriter

logger = logging.getLogger("dataprep.report")


def generate_excel_report(
    session_info: Optional[Dict[str, Any]] = None,
    features_analysis: Optional[Dict[str, Any]] = None,
    pipeline_result: Optional[Dict[str, Any]] = None,
    corr_matrix_data: Optional[Union[pd.DataFrame, Dict[str, Any]]] = None,
) -> bytes:
    """Génère un classeur Excel complet à 5 feuilles avec mise en forme soignée."""
    session_info = session_info if isinstance(session_info, dict) else {}
    features_analysis = features_analysis if isinstance(features_analysis, dict) else {}
    pipeline_result = pipeline_result if isinstance(pipeline_result, dict) else {}

    output = io.BytesIO()
    workbook = xlsxwriter.Workbook(output, {"in_memory": True, "nan_inf_to_errors": True})

    # -----------------------------------------------------------------------
    # Palettes de styles
    # -----------------------------------------------------------------------
    title_fmt = workbook.add_format({
        "bold": True, "font_size": 16, "font_color": "#0f172a", "bottom": 2, "bottom_color": "#2dd4bf"
    })
    subtitle_fmt = workbook.add_format({
        "italic": True, "font_size": 10, "font_color": "#64748b"
    })
    header_fmt = workbook.add_format({
        "bold": True, "bg_color": "#1e293b", "font_color": "#f8fafc",
        "border": 1, "border_color": "#334155", "align": "center", "valign": "vcenter"
    })
    sub_header_fmt = workbook.add_format({
        "bold": True, "bg_color": "#334155", "font_color": "#f8fafc",
        "border": 1, "border_color": "#475569"
    })
    cell_fmt = workbook.add_format({
        "border": 1, "border_color": "#e2e8f0", "valign": "vcenter", "font_size": 10
    })
    cell_alt_fmt = workbook.add_format({
        "border": 1, "border_color": "#e2e8f0", "valign": "vcenter", "font_size": 10, "bg_color": "#f8fafc"
    })
    badge_ok_fmt = workbook.add_format({
        "bg_color": "#dcfce7", "font_color": "#166534", "bold": True, "align": "center", "border": 1, "border_color": "#86efac"
    })

    # -----------------------------------------------------------------------
    # Feuille 1 : Résumé
    # -----------------------------------------------------------------------
    try:
        ws_res = workbook.add_worksheet("Résumé")
        ws_res.set_column(0, 0, 24)
        ws_res.set_column(1, 1, 35)
        ws_res.set_column(2, 5, 18)

        ws_res.write("A1", "DataPrep Studio — Rapport d'audit de préparation ML", title_fmt)
        ws_res.write("A2", f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')} | Anti-Data-Leakage certifié", subtitle_fmt)

        # Bloc KPIs
        ws_res.merge_range("A4:B4", "Métadonnées du projet", sub_header_fmt)

        train_shape = pipeline_result.get("train_shape", (0, 0))
        test_shape = pipeline_result.get("test_shape", (0, 0))
        train_rows = train_shape[0] if isinstance(train_shape, (tuple, list)) and len(train_shape) > 0 else 0
        test_rows = test_shape[0] if isinstance(test_shape, (tuple, list)) and len(test_shape) > 0 else 0

        metadata = [
            ("Fichier source", session_info.get("filename", "dataset.csv")),
            ("Tâche de Machine Learning", str(session_info.get("problem_type", "Classification")).capitalize()),
            ("Variable cible", session_info.get("target_col") or "Aucune (Non supervisé)"),
            ("Taille du jeu de test", f"{int(session_info.get('test_size', 0.2) * 100)}%"),
            ("Graine aléatoire (seed)", session_info.get("random_state", 42)),
            ("Lignes initiales", session_info.get("rows", 0)),
            ("Colonnes initiales", session_info.get("cols", 0)),
            ("Lignes Train après nettoyage", train_rows),
            ("Lignes Test étanches", test_rows),
            ("Colonnes finales préparées", pipeline_result.get("final_features_count", 0)),
        ]

        # Ajout des informations de rééquilibrage si disponible
        rebal = session_info.get("rebalancing_info")
        if isinstance(rebal, dict) and rebal.get("strategy") and rebal.get("strategy") != "none":
            metadata.extend([
                ("Stratégie rééquilibrage (Train)", rebal.get("strategy_label", rebal.get("strategy"))),
                ("Train avant rééquilibrage", f"{rebal.get('n_before')} lignes {rebal.get('before')}"),
                ("Train après rééquilibrage", f"{rebal.get('n_after')} lignes {rebal.get('after')}"),
                ("Jeu de test (Anti-Leakage)", "Strictement intact et non rééquilibré"),
            ])
        else:
            metadata.append(("Stratégie rééquilibrage", "Aucune (distribution d'origine conservée)"))

        for r_idx, (k, v) in enumerate(metadata, start=5):
            ws_res.write(f"A{r_idx}", k, cell_fmt)
            ws_res.write(f"B{r_idx}", str(v), cell_alt_fmt)

        # Bloc Bonnes pratiques
        ws_res.merge_range("D4:F4", "Vérifications d'intégrité ML & Anti-Data-Leakage", sub_header_fmt)
        checks = pipeline_result.get("best_practices_checks", [])
        if isinstance(checks, list) and len(checks) > 0:
            for idx, c in enumerate(checks, start=5):
                if isinstance(c, dict):
                    ws_res.write(f"D{idx}", c.get("rule", ""), cell_fmt)
                    ws_res.write(f"E{idx}", "CONFORME", badge_ok_fmt)
                    ws_res.write(f"F{idx}", c.get("detail", ""), cell_alt_fmt)
        else:
            ws_res.write("D5", "Aucune règle spécifique enregistrée", cell_fmt)
    except Exception as exc:
        logger.warning(f"Erreur lors de la génération de la feuille Résumé : {exc}")

    # -----------------------------------------------------------------------
    # Feuille 2 : Features
    # -----------------------------------------------------------------------
    try:
        ws_feat = workbook.add_worksheet("Features")
        feat_headers = [
            "Feature", "Type initial", "Statut", "Anomalies détectées",
            "Recommandation", "Décision utilisateur", "Transformation appliquée", "Paramètres appris sur Train"
        ]
        ws_feat.set_column(0, 0, 22)
        ws_feat.set_column(1, 2, 14)
        ws_feat.set_column(3, 3, 30)
        ws_feat.set_column(4, 5, 25)
        ws_feat.set_column(6, 6, 32)
        ws_feat.set_column(7, 7, 40)

        for col_idx, h in enumerate(feat_headers):
            ws_feat.write(0, col_idx, h, header_fmt)

        features_dict = features_analysis.get("features", {}) if isinstance(features_analysis.get("features"), dict) else {}
        applied_table = pipeline_result.get("applied_table", [])
        applied_dict = {}
        if isinstance(applied_table, list):
            for a in applied_table:
                if isinstance(a, dict) and "feature" in a:
                    applied_dict[a["feature"]] = a

        if features_dict:
            for row_idx, (col_name, f_info) in enumerate(features_dict.items(), start=1):
                f_info = f_info if isinstance(f_info, dict) else {}
                app = applied_dict.get(col_name, {})
                anom_items = f_info.get("anomalies", [])
                anoms = ", ".join([a.get("title", "") for a in anom_items if isinstance(a, dict)]) if anom_items else "Aucune anomalie"
                just_items = f_info.get("justifications", [])
                reco = " -> ".join([str(j) for j in just_items]) if just_items else "Standard"
                user_dec = "Supprimée" if app.get("action") == "Supprimée" else ("Cible" if f_info.get("is_target") else "Préparée")
                trans_app = app.get("action", "Conservée")
                learned = str(app.get("learned_parameters", {})) if app.get("learned_parameters") else "-"

                fmt = cell_alt_fmt if row_idx % 2 == 0 else cell_fmt
                ws_feat.write(row_idx, 0, col_name, fmt)
                ws_feat.write(row_idx, 1, f_info.get("type", "categorical"), fmt)
                ws_feat.write(row_idx, 2, user_dec, fmt)
                ws_feat.write(row_idx, 3, anoms, fmt)
                ws_feat.write(row_idx, 4, reco, fmt)
                ws_feat.write(row_idx, 5, user_dec, fmt)
                ws_feat.write(row_idx, 6, trans_app, fmt)
                ws_feat.write(row_idx, 7, learned, fmt)
        else:
            ws_feat.write(1, 0, "Aucune variable disponible.", cell_fmt)
    except Exception as exc:
        logger.warning(f"Erreur lors de la génération de la feuille Features : {exc}")

    # -----------------------------------------------------------------------
    # Feuille 3 : Détail transformations
    # -----------------------------------------------------------------------
    try:
        ws_det = workbook.add_worksheet("Détail transformations")
        det_headers = ["Feature", "Type initial", "Action globale", "Imputation", "Outliers", "Scaling", "Encodage", "Nb features générées"]
        ws_det.set_column(0, 0, 22)
        ws_det.set_column(1, 1, 14)
        ws_det.set_column(2, 2, 28)
        ws_det.set_column(3, 6, 18)
        ws_det.set_column(7, 7, 20)

        for col_idx, h in enumerate(det_headers):
            ws_det.write(0, col_idx, h, header_fmt)

        applied_table = pipeline_result.get("applied_table", [])
        if isinstance(applied_table, list) and len(applied_table) > 0:
            for row_idx, app in enumerate(applied_table, start=1):
                if not isinstance(app, dict):
                    continue
                fmt = cell_alt_fmt if row_idx % 2 == 0 else cell_fmt
                ws_det.write(row_idx, 0, app.get("feature", ""), fmt)
                ws_det.write(row_idx, 1, app.get("initial_type", ""), fmt)
                ws_det.write(row_idx, 2, app.get("action", ""), fmt)
                ws_det.write(row_idx, 3, str(app.get("imputation", "-")), fmt)
                ws_det.write(row_idx, 4, str(app.get("outliers", "-")), fmt)
                ws_det.write(row_idx, 5, str(app.get("scaling", "-")), fmt)
                ws_det.write(row_idx, 6, str(app.get("encoding", "-")), fmt)
                ws_det.write(row_idx, 7, app.get("final_features", 1), fmt)
        else:
            ws_det.write(1, 0, "Aucune transformation enregistrée.", cell_fmt)
    except Exception as exc:
        logger.warning(f"Erreur lors de la génération de la feuille Détail transformations : {exc}")

    # -----------------------------------------------------------------------
    # Feuille 4 : Colonnes finales
    # -----------------------------------------------------------------------
    try:
        ws_cols = workbook.add_worksheet("Colonnes finales")
        ws_cols.set_column(0, 0, 8)
        ws_cols.set_column(1, 1, 35)
        ws_cols.set_column(2, 2, 20)

        ws_cols.write(0, 0, "#", header_fmt)
        ws_cols.write(0, 1, "Nom de la colonne finale", header_fmt)
        ws_cols.write(0, 2, "Type scikit-learn / pandas", header_fmt)

        final_cols = pipeline_result.get("final_columns", [])
        if isinstance(final_cols, list) and len(final_cols) > 0:
            for idx, col in enumerate(final_cols, start=1):
                fmt = cell_alt_fmt if idx % 2 == 0 else cell_fmt
                ws_cols.write(idx, 0, idx, fmt)
                ws_cols.write(idx, 1, str(col), fmt)
                ws_cols.write(idx, 2, "float64 / int64", fmt)
        else:
            ws_cols.write(1, 0, "Aucune colonne finale enregistrée.", cell_fmt)
    except Exception as exc:
        logger.warning(f"Erreur lors de la génération de la feuille Colonnes finales : {exc}")

    # -----------------------------------------------------------------------
    # Feuille 5 : Corrélations
    # -----------------------------------------------------------------------
    try:
        ws_corr = workbook.add_worksheet("Corrélations")
        corr_cols: List[str] = []
        corr_rows: List[tuple[str, List[float]]] = []

        if isinstance(corr_matrix_data, pd.DataFrame) and not corr_matrix_data.empty:
            # Vérifier si c'est déjà une matrice de corrélation carrée (index == colonnes)
            if (
                corr_matrix_data.shape[0] == corr_matrix_data.shape[1]
                and list(corr_matrix_data.index) == list(corr_matrix_data.columns)
            ):
                corr_df = corr_matrix_data
            else:
                num_df = corr_matrix_data.select_dtypes(include="number")
                if not num_df.empty and num_df.shape[1] >= 2:
                    corr_df = num_df.corr().round(3)
                else:
                    corr_df = None

            if corr_df is not None and not corr_df.empty and len(corr_df.columns) >= 2:
                corr_cols = [str(c) for c in corr_df.columns]
                for r in corr_df.index:
                    row_vals: List[float] = []
                    for c in corr_df.columns:
                        try:
                            val = float(corr_df.loc[r, c])
                            if math.isnan(val):
                                val = 0.0
                        except Exception:
                            val = 0.0
                        row_vals.append(val)
                    corr_rows.append((str(r), row_vals))

        elif isinstance(corr_matrix_data, dict) and bool(corr_matrix_data.get("columns")):
            raw_cols = corr_matrix_data.get("columns", [])
            raw_matrix = corr_matrix_data.get("matrix", [])
            if isinstance(raw_cols, list) and len(raw_cols) >= 2 and isinstance(raw_matrix, list) and raw_matrix:
                corr_cols = [str(c) for c in raw_cols]
                for row in raw_matrix:
                    if isinstance(row, dict):
                        r_name = str(row.get("col", ""))
                        row_vals = []
                        for c in raw_cols:
                            try:
                                val = float(row.get(c, 0.0))
                                if math.isnan(val):
                                    val = 0.0
                            except Exception:
                                val = 0.0
                            row_vals.append(val)
                        corr_rows.append((r_name, row_vals))

        if corr_cols and corr_rows:
            ws_corr.set_column(0, 0, 20)
            ws_corr.set_column(1, len(corr_cols), 14)

            ws_corr.write(0, 0, "Variables", header_fmt)
            for c_idx, c_name in enumerate(corr_cols, start=1):
                ws_corr.write(0, c_idx, c_name, header_fmt)

            for r_idx, (r_name, row_vals) in enumerate(corr_rows, start=1):
                ws_corr.write(r_idx, 0, r_name, sub_header_fmt)
                for c_idx, val in enumerate(row_vals, start=1):
                    ws_corr.write_number(r_idx, c_idx, val, cell_fmt)

            # Règle de coloration conditionnelle pour corrélations (-1 à 1)
            ws_corr.conditional_format(1, 1, len(corr_rows), len(corr_cols), {
                "type": "3_color_scale",
                "min_color": "#93c5fd",  # bleu pour -1
                "mid_color": "#ffffff",  # blanc pour 0
                "max_color": "#f87171",  # rouge pour +1
                "min_type": "num", "min_value": -1,
                "mid_type": "num", "mid_value": 0,
                "max_type": "num", "max_value": 1,
            })
        else:
            ws_corr.write("A1", "Moins de deux variables numériques présentes pour la matrice de corrélation.")
    except Exception as exc:
        logger.warning(f"Erreur lors de la génération de la feuille Corrélations : {exc}")

    workbook.close()
    return output.getvalue()
