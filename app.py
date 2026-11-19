"""Application Streamlit complète de préparation et prétraitement de données pour le Machine Learning supervisé."""
from __future__ import annotations

import io
import json
import joblib
import pandas as pd
import streamlit as st

from modules.loader import CSVLoadError, dataset_summary, dtypes_table, load_csv
from modules.missing import (
    CATEGORICAL_STRATEGIES,
    NUMERIC_STRATEGIES,
    analyze_missing,
)
from modules.outliers import OUTLIER_ACTIONS, analyze_outliers
from modules.pipeline import (
    PipelineConfig,
    execute_preparation,
)
from modules.profiling import (
    TYPE_LABELS,
    classify_dataframe,
    get_column_stats,
)
from modules.report import build_preparation_report, create_download_zip
from modules.target import filter_missing_target, suggest_target

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="DataPrep ML Pro",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

STEPS = [
    "1. Importation",
    "2. Exploration",
    "3. Choix de la cible",
    "4. Vérification des types",
    "5. Manquantes et aberrantes",
    "6. Encodage et mise à l'échelle",
    "7. Prévisualisation",
    "8. Exécution et téléchargement",
]


def init_session_state():
    """Initialise les variables d'état persistantes de l'application."""
    defaults = {
        "step_index": 0,
        "df_raw": None,
        "load_info": None,
        "filename": None,
        "column_types": {},
        "excluded_columns": [],
        "ordinal_orders": {},
        "target_col": None,
        "problem_type": "classification_binaire",
        "drop_missing_target": True,
        "numeric_scaler": "StandardScaler",
        "numeric_imputer": "mediane",
        "categorical_imputer": "valeur_manquante",
        "outlier_method": "IQR",
        "outlier_action": "clip",
        "drop_outliers_confirmed": False,
        "rare_categories_min_freq": 0.03,
        "date_features_enabled": True,
        "text_strategy": "exclude",
        "text_max_features": 10,
        "test_size": 0.20,
        "random_state": 42,
        "stratify": True,
        "prep_result": None,
        "prep_report": None,
        "prep_zip": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


def set_step(new_index: int):
    st.session_state["step_index"] = max(0, min(new_index, len(STEPS) - 1))
    st.rerun()


def nav_buttons(prev_idx: int | None = None, next_idx: int | None = None, next_label: str = "Étape suivante ➔"):
    """Boutons de navigation fluides en bas de chaque page."""
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 4, 1])
    if prev_idx is not None:
        with c1:
            if st.button("⬅ Précédent", key=f"btn_prev_{prev_idx}"):
                set_step(prev_idx)
    if next_idx is not None:
        with c3:
            if st.button(next_label, type="primary", key=f"btn_next_{next_idx}"):
                set_step(next_idx)


# ---------------------------------------------------------------------------
# Étape 1 : Importation
# ---------------------------------------------------------------------------
def page_import():
    st.header("1. Importation et validation du fichier CSV")
    st.caption("Chargez n'importe quel fichier CSV pour débuter la préparation.")

    uploaded = st.file_uploader(
        "Sélectionnez un fichier CSV, TSV ou TXT",
        type=["csv", "tsv", "txt"],
        help="Détection automatique du délimiteur, de l'encodage et des formats numériques.",
    )

    with st.expander("Paramètres de lecture manuels (en cas d'erreur ou d'ajustement spécifique)"):
        c1, c2, c3, c4 = st.columns(4)
        enc = c1.selectbox("Encodage", ["Auto", "utf-8-sig", "utf-8", "cp1252", "latin-1"])
        sep = c2.selectbox("Séparateur", ["Auto", ",", ";", "Tabulation (\\t)", "|"])
        dec = c3.selectbox("Séparateur décimal", [".", ","])
        header = c4.checkbox("La première ligne contient les en-têtes", value=True)
        coerce = st.checkbox("Convertir les nombres formatés en texte (ex: espaces, virgules)", value=True)

    if uploaded is None:
        if st.session_state["df_raw"] is not None:
            st.success(f"Fichier déjà en mémoire : **{st.session_state['filename']}**.")
            nav_buttons(next_idx=1, next_label="Aller à l'exploration ➔")
        else:
            st.info("Veuillez charger un fichier CSV pour continuer.")
        return

    raw_bytes = uploaded.getvalue()
    sep_val = {"Auto": None, "Tabulation (\\t)": "\t"}.get(sep, sep)
    enc_val = None if enc == "Auto" else enc

    try:
        df, info = load_csv(raw_bytes, enc_val, sep_val, dec, header, coerce)
    except CSVLoadError as exc:
        st.error(f"Erreur d'importation : {exc}")
        st.caption("Vérifiez les paramètres de lecture ci-dessus (encodage ou séparateur) puis réessayez.")
        return

    # Mise à jour de l'état
    st.session_state["df_raw"] = df
    st.session_state["load_info"] = info
    st.session_state["filename"] = uploaded.name

    # Réinitialisation de l'analyse automatique sur nouveau fichier
    if "last_loaded_name" not in st.session_state or st.session_state["last_loaded_name"] != uploaded.name:
        st.session_state["last_loaded_name"] = uploaded.name
        st.session_state["column_types"] = classify_dataframe(df)
        tgt = suggest_target(df)
        if tgt:
            st.session_state["target_col"] = tgt.column
            st.session_state["problem_type"] = tgt.problem_type
        # Exclure automatiquement les colonnes constantes ou identifiants
        st.session_state["excluded_columns"] = [
            c for c, t in st.session_state["column_types"].items()
            if t in ("constante", "identifiant")
        ]

    st.success(f"Fichier **{uploaded.name}** lu avec succès (Encodage : `{info.encoding}`, Séparateur : `{repr(info.separator)}`).")
    if info.converted_columns:
        st.info(f"Colonnes texte converties en numérique : {', '.join(info.converted_columns)}")

    summ = dataset_summary(df)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Lignes", f"{summ['lignes']:,}")
    m2.metric("Colonnes", summ["colonnes"])
    m3.metric("Doublons", summ["doublons"])
    m4.metric("Valeurs manquantes", f"{summ['valeurs_manquantes']:,}")
    m5.metric("Mémoire", f"{summ['memoire_Mo']} Mo")

    if summ["doublons"] > 0:
        st.warning(f"{summ['doublons']} ligne(s) dupliquée(s) détectée(s) dans le fichier source.")

    st.subheader("Aperçu des 50 premières lignes")
    st.dataframe(df.head(50), use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Types détectés")
        st.dataframe(dtypes_table(df), use_container_width=True, hide_index=True)
    with c_right:
        st.subheader("Statistiques descriptives")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    nav_buttons(next_idx=1, next_label="Aller à l'exploration ➔")


# ---------------------------------------------------------------------------
# Étape 2 : Exploration
# ---------------------------------------------------------------------------
def page_exploration():
    st.header("2. Exploration des variables")
    df: pd.DataFrame | None = st.session_state.get("df_raw")
    if df is None:
        st.warning("Veuillez d'abord importer un fichier à l'étape 1.")
        nav_buttons(prev_idx=0)
        return

    st.markdown("Visualisez la distribution de vos variables numériques et catégorielles.")
    
    tab_dist, tab_corr, tab_miss = st.tabs(["Distributions", "Corrélations numériques", "Taux de manquants"])

    with tab_dist:
        col_to_plot = st.selectbox("Sélectionnez une variable à analyser", df.columns)
        series = df[col_to_plot]
        
        c1, c2 = st.columns([2, 1])
        with c1:
            if pd.api.types.is_numeric_dtype(series):
                clean_s = series.dropna()
                st.write(f"**Histogramme de `{col_to_plot}`**")
                # Diagramme en barres des quantiles
                counts, bin_edges = pd.cut(clean_s, bins=20, retbins=True)
                chart_df = counts.value_counts().sort_index().reset_index()
                chart_df.columns = ["Intervalle", "Fréquence"]
                chart_df["Intervalle"] = chart_df["Intervalle"].astype(str)
                st.bar_chart(chart_df.set_index("Intervalle"))
            else:
                st.write(f"**Top 20 modalités de `{col_to_plot}`**")
                counts = series.fillna("Valeur manquante").value_counts().head(20)
                st.bar_chart(counts)
        with c2:
            st.write("**Statistiques de la variable**")
            st.write(series.describe())

    with tab_corr:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(num_cols) >= 2:
            st.write("**Matrice de corrélation linéaire de Pearson**")
            corr_df = df[num_cols].corr().round(2)
            st.dataframe(corr_df.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1), use_container_width=True)
        else:
            st.info("Moins de deux variables numériques présentes pour calculer des corrélations.")

    with tab_miss:
        miss_df = pd.DataFrame({
            "Colonne": df.columns,
            "% Manquants": (df.isna().mean() * 100).round(2).values
        }).sort_values(by="% Manquants", ascending=False)
        st.bar_chart(miss_df.set_index("Colonne"))

    nav_buttons(prev_idx=0, next_idx=2, next_label="Étape 3 : Choix de la cible ➔")


# ---------------------------------------------------------------------------
# Étape 3 : Choix de la cible
# ---------------------------------------------------------------------------
def page_target():
    st.header("3. Détection et choix de la variable cible (Supervisé)")
    df: pd.DataFrame | None = st.session_state.get("df_raw")
    if df is None:
        st.warning("Veuillez d'abord importer un fichier à l'étape 1.")
        nav_buttons(prev_idx=0)
        return

    proposal = suggest_target(df)
    
    st.subheader("Proposition automatique")
    if proposal:
        st.info(
            f"🎯 Variable suggérée : **{proposal.column}**  \n"
            f"**Critères retenus :** {proposal.reason}  \n"
            f"**Type de problème détecté :** `{proposal.problem_type}`"
        )
    
    st.subheader("Confirmation ou sélection manuelle")
    col_names = list(df.columns)
    default_idx = col_names.index(st.session_state["target_col"]) if st.session_state["target_col"] in col_names else len(col_names) - 1

    selected_target = st.selectbox(
        "Sélectionnez la variable cible (y)",
        col_names,
        index=default_idx,
        help="Cette variable sera exclue de tous les pipelines appliqués aux variables explicatives.",
    )
    st.session_state["target_col"] = selected_target

    target_series = df[selected_target]

    prob_options = {
        "classification_binaire": "Classification binaire (2 classes)",
        "classification_multiclasse": "Classification multiclasse (> 2 classes)",
        "regression": "Régression (valeur continue)",
    }
    
    # Suggestion de problème adaptée à la colonne choisie
    inferred_prob = proposal.problem_type if proposal and proposal.column == selected_target else ("regression" if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 10 else "classification_binaire")
    default_prob_idx = list(prob_options.keys()).index(st.session_state.get("problem_type", inferred_prob))

    selected_problem = st.selectbox(
        "Type de tâche supervisée",
        list(prob_options.keys()),
        index=default_prob_idx,
        format_func=lambda x: prob_options[x],
    )
    st.session_state["problem_type"] = selected_problem

    # Analyse détaillée de la cible choisie
    n_missing = int(target_series.isna().sum())
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Valeurs manquantes dans la cible", f"{n_missing} ({round(n_missing / len(df) * 100, 2)}%)")
        if n_missing > 0:
            st.warning("En apprentissage supervisé, les lignes dont la cible est manquante ne peuvent pas servir à l'entraînement.")
            st.session_state["drop_missing_target"] = st.checkbox(
                "Supprimer automatiquement les lignes où la cible est manquante",
                value=st.session_state.get("drop_missing_target", True),
            )
        else:
            st.success("Aucune valeur manquante dans la cible.")

    with c2:
        if "classification" in selected_problem:
            clean_tgt = target_series.dropna()
            counts = clean_tgt.value_counts()
            st.write("**Répartition des classes :**")
            st.dataframe(pd.DataFrame({"Effectif": counts, "%": (counts / len(clean_tgt) * 100).round(2)}), use_container_width=True)
            if len(counts) >= 2 and (counts.min() / counts.sum()) < 0.20:
                pct_min = round((counts.min() / counts.sum()) * 100, 1)
                st.warning(f"Déséquilibre de classe détecté : la classe minoritaire ne représente que {pct_min}%. La stratification sera activée.")
        else:
            st.write("**Statistiques de la variable continue :**")
            st.dataframe(target_series.describe(), use_container_width=True)

    nav_buttons(prev_idx=1, next_idx=3, next_label="Étape 4 : Vérification des types ➔")


# ---------------------------------------------------------------------------
# Étape 4 : Vérification des types et rôles
# ---------------------------------------------------------------------------
def page_types():
    st.header("4. Vérification et modification des types de variables")
    df: pd.DataFrame | None = st.session_state.get("df_raw")
    if df is None:
        st.warning("Veuillez d'abord importer un fichier à l'étape 1.")
        nav_buttons(prev_idx=0)
        return

    target_col = st.session_state["target_col"]
    feature_cols = [c for c in df.columns if c != target_col]

    st.markdown(
        "L'application a classé automatiquement chaque colonne explicative. "
        "Vous pouvez corriger le type et décider d'exclure certaines colonnes (identifiants, colonnes inutiles, etc.)."
    )

    col_stats = get_column_stats(df[feature_cols], st.session_state["column_types"])

    type_options = list(TYPE_LABELS.keys())

    # Interface modifiable pour chaque colonne
    new_types = {}
    new_excluded = list(st.session_state.get("excluded_columns", []))
    new_ordinal_orders = dict(st.session_state.get("ordinal_orders", {}))

    for idx, row in col_stats.iterrows():
        col_name = row["colonne"]
        current_type = st.session_state["column_types"].get(col_name, row["type_detecte"])
        is_initially_excluded = col_name in new_excluded or current_type in ("constante", "identifiant")

        with st.expander(f"Colonne : **{col_name}** | Type actuel : `{TYPE_LABELS.get(current_type, current_type)}`", expanded=False):
            c1, c2, c3 = st.columns([2, 2, 2])
            with c1:
                st.caption(f"Manquants : {row['manquants']} ({row['% manquants']}%) | Uniques : {row['valeurs_uniques']}")
                st.caption(f"Exemples : `{row['exemples']}`")
            with c2:
                selected_type = st.selectbox(
                    f"Type pour {col_name}",
                    type_options,
                    index=type_options.index(current_type) if current_type in type_options else 0,
                    format_func=lambda x: TYPE_LABELS[x],
                    key=f"type_select_{col_name}",
                )
                new_types[col_name] = selected_type

                if selected_type == "categorielle_ordinale":
                    existing_order = ", ".join(new_ordinal_orders.get(col_name, []))
                    order_input = st.text_input(
                        "Ordre des modalités (séparées par une virgule)",
                        value=existing_order,
                        help="Ex: faible, moyen, fort",
                        key=f"order_{col_name}",
                    )
                    if order_input.strip():
                        new_ordinal_orders[col_name] = [x.strip() for x in order_input.split(",") if x.strip()]

            with c3:
                exclude_col = st.checkbox(
                    "Exclure des features du modèle",
                    value=is_initially_excluded,
                    key=f"exclude_{col_name}",
                    help="Coché par défaut pour les identifiants ou colonnes constantes.",
                )
                if exclude_col:
                    if col_name not in new_excluded:
                        new_excluded.append(col_name)
                else:
                    if col_name in new_excluded:
                        new_excluded.remove(col_name)

    st.session_state["column_types"] = new_types
    st.session_state["excluded_columns"] = new_excluded
    st.session_state["ordinal_orders"] = new_ordinal_orders

    active_features = [c for c in feature_cols if c not in new_excluded]
    st.info(f"**{len(active_features)}** feature(s) active(s) sélectionnée(s) pour la préparation. **{len(new_excluded)}** exclue(s).")

    nav_buttons(prev_idx=2, next_idx=4, next_label="Étape 5 : Manquantes et aberrantes ➔")


# ---------------------------------------------------------------------------
# Étape 5 : Valeurs manquantes et aberrantes
# ---------------------------------------------------------------------------
def page_missing_outliers():
    st.header("5. Configuration des valeurs manquantes et aberrantes")
    df: pd.DataFrame | None = st.session_state.get("df_raw")
    if df is None:
        st.warning("Veuillez d'abord importer un fichier à l'étape 1.")
        nav_buttons(prev_idx=0)
        return

    target_col = st.session_state["target_col"]
    feature_cols = [c for c in df.columns if c != target_col and c not in st.session_state["excluded_columns"]]
    col_types = st.session_state["column_types"]

    st.subheader("A. Valeurs manquantes")
    missing_analysis = analyze_missing(df[feature_cols], col_types)
    cols_with_missing = [m for m in missing_analysis if m.missing_count > 0]

    if cols_with_missing:
        st.dataframe(
            pd.DataFrame([
                {
                    "Colonne": m.column,
                    "Manquants": m.missing_count,
                    "%": m.missing_pct,
                    "Action recommandée": m.recommended_strategy,
                    "Avertissement": m.warning or "-",
                }
                for m in cols_with_missing
            ]),
            use_container_width=True,
        )
    else:
        st.success("Aucune valeur manquante détectée dans les variables explicatives.")

    c1, c2 = st.columns(2)
    with c1:
        st.session_state["numeric_imputer"] = st.selectbox(
            "Stratégie d'imputation numérique globale",
            list(NUMERIC_STRATEGIES.keys()),
            index=list(NUMERIC_STRATEGIES.keys()).index(st.session_state["numeric_imputer"]),
            format_func=lambda x: NUMERIC_STRATEGIES[x],
        )
    with c2:
        st.session_state["categorical_imputer"] = st.selectbox(
            "Stratégie d'imputation catégorielle globale",
            list(CATEGORICAL_STRATEGIES.keys()),
            index=list(CATEGORICAL_STRATEGIES.keys()).index(st.session_state["categorical_imputer"]),
            format_func=lambda x: CATEGORICAL_STRATEGIES[x],
        )

    st.markdown("---")
    st.subheader("B. Valeurs aberrantes (Outliers)")
    num_features = [c for c in feature_cols if col_types.get(c) in ("numerique_continue", "numerique_discrete")]

    c_meth, c_act = st.columns(2)
    with c_meth:
        outlier_meth = st.selectbox("Méthode de détection", ["IQR", "Z-Score"], index=0)
        st.session_state["outlier_method"] = outlier_meth.lower()
    with c_act:
        outlier_action = st.selectbox(
            "Action sur les valeurs aberrantes",
            list(OUTLIER_ACTIONS.keys()),
            index=list(OUTLIER_ACTIONS.keys()).index(st.session_state["outlier_action"]),
            format_func=lambda x: OUTLIER_ACTIONS[x],
            help="Le plafonnement (winsorisation) borne les valeurs extrêmes sans perdre d'observations.",
        )
        st.session_state["outlier_action"] = outlier_action

    if num_features:
        outlier_results = analyze_outliers(df, num_features, method=outlier_meth)
        outlier_rows = []
        for col, info in outlier_results.items():
            if info.outlier_count > 0:
                outlier_rows.append({
                    "Colonne": col,
                    "Borne basse": info.lower_bound,
                    "Borne haute": info.upper_bound,
                    "Nb Outliers": info.outlier_count,
                    "% Outliers": info.outlier_pct,
                })
        if outlier_rows:
            st.warning(f"Valeurs aberrantes détectées dans {len(outlier_rows)} variable(s) numérique(s).")
            st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)
            # Recommandation automatique de mise à l'échelle robuste
            if st.session_state["numeric_scaler"] == "StandardScaler":
                st.session_state["numeric_scaler"] = "RobustScaler"
        else:
            st.success("Aucune valeur aberrante excessive détectée selon les critères actuels.")

    if outlier_action == "drop":
        st.error("⚠️ Attention : La suppression des lignes avec outliers éliminera définitivement ces observations.")
        st.session_state["drop_outliers_confirmed"] = st.checkbox(
            "Je confirme expressément vouloir supprimer les lignes contenant des valeurs aberrantes",
            value=st.session_state.get("drop_outliers_confirmed", False),
        )
    else:
        st.session_state["drop_outliers_confirmed"] = False

    nav_buttons(prev_idx=3, next_idx=5, next_label="Étape 6 : Encodage et mise à l'échelle ➔")


# ---------------------------------------------------------------------------
# Étape 6 : Encodage et mise à l'échelle
# ---------------------------------------------------------------------------
def page_encoding_scaling():
    st.header("6. Encodage et mise à l'échelle des caractéristiques")
    df: pd.DataFrame | None = st.session_state.get("df_raw")
    if df is None:
        st.warning("Veuillez d'abord importer un fichier à l'étape 1.")
        nav_buttons(prev_idx=0)
        return

    st.subheader("A. Encodage des variables catégorielles et temporelles")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state["rare_categories_min_freq"] = st.slider(
            "Regroupement des modalités rares (fréquence min)",
            min_value=0.0,
            max_value=0.15,
            value=float(st.session_state.get("rare_categories_min_freq", 0.03)),
            step=0.01,
            help="Les modalités représentant moins de ce pourcentage seront regroupées sous une catégorie 'infrequent_sklearn'.",
        )
        st.session_state["date_features_enabled"] = st.checkbox(
            "Extraire les composantes de dates (Année, Mois, Jour, Jour de semaine, Jours écoulés)",
            value=st.session_state.get("date_features_enabled", True),
        )
    with c2:
        st.session_state["text_strategy"] = st.selectbox(
            "Traitement du texte libre",
            ["exclude", "tfidf"],
            index=0 if st.session_state.get("text_strategy", "exclude") == "exclude" else 1,
            format_func=lambda x: "Exclure par défaut (recommandé)" if x == "exclude" else "Vectorisation TF-IDF supervisée",
        )
        if st.session_state["text_strategy"] == "tfidf":
            st.session_state["text_max_features"] = st.slider("Nombre maximal de termes TF-IDF", 3, 30, 10)

    st.markdown("---")
    st.subheader("B. Standardisation et Normalisation numérique")
    
    scaler_options = {
        "RobustScaler": "RobustScaler (Recommandé si des valeurs aberrantes existent)",
        "StandardScaler": "StandardScaler (Moyenne 0, Variance 1)",
        "MinMaxScaler": "MinMaxScaler (Normalisation entre 0 et 1)",
        "none": "Aucune mise à l'échelle (conserver l'échelle d'origine)",
    }
    
    st.session_state["numeric_scaler"] = st.radio(
        "Choix de la méthode de mise à l'échelle",
        list(scaler_options.keys()),
        index=list(scaler_options.keys()).index(st.session_state["numeric_scaler"]),
        format_func=lambda x: scaler_options[x],
    )

    nav_buttons(prev_idx=4, next_idx=6, next_label="Étape 7 : Prévisualisation ➔")


# ---------------------------------------------------------------------------
# Étape 7 : Prévisualisation & Prévention de la fuite de données
# ---------------------------------------------------------------------------
def page_preview():
    st.header("7. Prévention de la fuite de données et récapitulatif")
    df: pd.DataFrame | None = st.session_state.get("df_raw")
    if df is None:
        st.warning("Veuillez d'abord importer un fichier à l'étape 1.")
        nav_buttons(prev_idx=0)
        return

    st.subheader("Paramètres de séparation Train / Test")
    st.caption("La séparation est impérativement effectuée AVANT tout ajustement d'encodeur ou de mise à l'échelle afin d'éviter tout data leakage.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.session_state["test_size"] = st.slider("Taille du jeu de test (test_size)", 0.10, 0.40, float(st.session_state["test_size"]), 0.05)
    with c2:
        st.session_state["random_state"] = st.number_input("Graine aléatoire (random_state)", value=int(st.session_state["random_state"]), step=1)
    with c3:
        if "classification" in st.session_state["problem_type"]:
            st.session_state["stratify"] = st.checkbox("Activer la stratification sur la cible", value=st.session_state["stratify"])
        else:
            st.session_state["stratify"] = False
            st.caption("Stratification non applicable à une régression continue.")

    st.markdown("---")
    st.subheader("Résumé des transformations planifiées")
    target_col = st.session_state["target_col"]
    col_types = st.session_state["column_types"]
    active_features = [c for c in df.columns if c != target_col and c not in st.session_state["excluded_columns"]]
    dropped_cols = [c for c in df.columns if c != target_col and c not in active_features]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Variable Cible", target_col)
    m2.metric("Features actives", len(active_features))
    m3.metric("Colonnes exclues", len(dropped_cols))
    m4.metric("Scaler numérique", st.session_state["numeric_scaler"])

    if dropped_cols:
        st.write(f"Colonnes exclues : `{', '.join(dropped_cols)}`")

    st.markdown("---")
    st.success("Toutes les étapes de configuration sont prêtes. Cliquez sur le bouton ci-dessous pour déclencher l'exécution.")

    c_prev, _, c_exec = st.columns([1, 2, 2])
    with c_prev:
        if st.button("⬅ Étape précédente"):
            set_step(5)
    with c_exec:
        if st.button("🚀 Exécuter la préparation complète ➔", type="primary"):
            set_step(7)


# ---------------------------------------------------------------------------
# Étape 8 : Exécution et téléchargement
# ---------------------------------------------------------------------------
def page_execution():
    st.header("8. Exécution et téléchargement des jeux de données préparés")
    df: pd.DataFrame | None = st.session_state.get("df_raw")
    if df is None:
        st.warning("Veuillez d'abord importer un fichier à l'étape 1.")
        nav_buttons(prev_idx=0)
        return

    config = PipelineConfig(
        target_col=st.session_state["target_col"],
        problem_type=st.session_state["problem_type"],
        test_size=st.session_state["test_size"],
        random_state=st.session_state["random_state"],
        stratify=st.session_state["stratify"],
        numeric_scaler=st.session_state["numeric_scaler"],
        numeric_imputer=st.session_state["numeric_imputer"],
        outlier_action=st.session_state["outlier_action"],
        outlier_method=st.session_state["outlier_method"],
        drop_outliers_confirmed=st.session_state["drop_outliers_confirmed"],
        categorical_imputer=st.session_state["categorical_imputer"],
        rare_categories_min_freq=st.session_state["rare_categories_min_freq"],
        date_features_enabled=st.session_state["date_features_enabled"],
        text_strategy=st.session_state["text_strategy"],
        text_max_features=st.session_state["text_max_features"],
        excluded_columns=st.session_state["excluded_columns"],
        ordinal_orders=st.session_state["ordinal_orders"],
        drop_missing_target=st.session_state["drop_missing_target"],
    )

    with st.spinner("Exécution du pipeline de préparation en cours…"):
        try:
            result = execute_preparation(df, config, st.session_state["column_types"])
            report = build_preparation_report(config, st.session_state["column_types"], result, st.session_state["filename"])
            zip_bytes = create_download_zip(result, report)

            st.session_state["prep_result"] = result
            st.session_state["prep_report"] = report
            st.session_state["prep_zip"] = zip_bytes
        except Exception as exc:
            st.error(f"Erreur durant l'exécution de la préparation : {exc}")
            st.exception(exc)
            nav_buttons(prev_idx=6)
            return

    st.success("✅ Données préparées avec succès et prêtes pour l'entraînement !")

    # Avertissements éventuels
    if result.warnings:
        for w in result.warnings:
            st.warning(w)

    # Métriques
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Dimensions X_train", f"{result.X_train.shape[0]} × {result.X_train.shape[1]}")
    m2.metric("Dimensions X_test", f"{result.X_test.shape[0]} × {result.X_test.shape[1]}")
    m3.metric("Features générées", len(result.feature_names))
    m4.metric("Fuite de données", "Évitée (Fit sur Train)")

    st.subheader("Aperçu de X_train transformé (20 premières lignes)")
    st.dataframe(result.X_train.head(20), use_container_width=True)

    st.markdown("---")
    st.subheader("📦 Zone de Téléchargement")

    # Bouton téléchargement ZIP tout-en-un
    st.download_button(
        label="📥 Télécharger l'Archive Complète (ZIP : CSV, Pipeline .joblib, Rapport JSON)",
        data=zip_bytes,
        file_name="donnees_preparees_ml.zip",
        mime="application/zip",
        type="primary",
        use_container_width=True,
    )

    st.markdown("#### Téléchargements individuels :")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button(
            "Télécharger X_train.csv",
            data=result.X_train.to_csv(index=False).encode("utf-8"),
            file_name="X_train.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Télécharger X_test.csv",
            data=result.X_test.to_csv(index=False).encode("utf-8"),
            file_name="X_test.csv",
            mime="text/csv",
        )
    with c3:
        st.download_button(
            "Télécharger y_train.csv",
            data=pd.DataFrame({config.target_col: result.y_train}).to_csv(index=False).encode("utf-8"),
            file_name="y_train.csv",
            mime="text/csv",
        )
    with c4:
        st.download_button(
            "Télécharger y_test.csv",
            data=pd.DataFrame({config.target_col: result.y_test}).to_csv(index=False).encode("utf-8"),
            file_name="y_test.csv",
            mime="text/csv",
        )

    c5, c6 = st.columns(2)
    with c5:
        joblib_buf = io.BytesIO()
        joblib.dump(result.pipeline, joblib_buf)
        st.download_button(
            "Télécharger le Preprocesseur (.joblib)",
            data=joblib_buf.getvalue(),
            file_name="pipeline_preprocesseur.joblib",
            mime="application/octet-stream",
        )
    with c6:
        st.download_button(
            "Télécharger le Rapport d'Audit (.json)",
            data=json.dumps(report, indent=2, ensure_ascii=False).encode("utf-8"),
            file_name="rapport_preparation.json",
            mime="application/json",
        )

    nav_buttons(prev_idx=6, next_idx=None)


# ---------------------------------------------------------------------------
# Point d'entrée principal et routage
# ---------------------------------------------------------------------------
def main():
    st.sidebar.title("🧪 DataPrep ML Pro")
    st.sidebar.caption("Pipeline de préparation automatisé pour le ML supervisé")
    
    current_idx = st.session_state["step_index"]
    selected_step = st.sidebar.radio(
        "Navigation",
        STEPS,
        index=current_idx,
        key="sidebar_nav",
    )
    selected_idx = STEPS.index(selected_step)
    if selected_idx != current_idx:
        st.session_state["step_index"] = selected_idx
        st.rerun()

    # Routage de la vue
    if current_idx == 0:
        page_import()
    elif current_idx == 1:
        page_exploration()
    elif current_idx == 2:
        page_target()
    elif current_idx == 3:
        page_types()
    elif current_idx == 4:
        page_missing_outliers()
    elif current_idx == 5:
        page_encoding_scaling()
    elif current_idx == 6:
        page_preview()
    elif current_idx == 7:
        page_execution()


if __name__ == "__main__":
    main()
