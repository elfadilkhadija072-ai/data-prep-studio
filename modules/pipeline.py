"""Construction du pipeline scikit-learn et exécution sans fuite de données."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------------------------
# Transformateurs personnalisés compatibles scikit-learn
# ---------------------------------------------------------------------------

class OutlierWinsorizer(BaseEstimator, TransformerMixin):
    """Plafonne (winsorise) les colonnes numériques aux bornes IQR ou Z-score apprises sur le Train."""

    def __init__(self, factor: float = 1.5, method: str = "iqr"):
        self.factor = factor
        self.method = method
        self.bounds_: dict[str, tuple[float, float]] = {}
        self.feature_names_in_: np.ndarray | None = None

    def fit(self, X: Any, y: Any = None) -> OutlierWinsorizer:
        X_df = pd.DataFrame(X)
        self.bounds_ = {}
        for col in X_df.columns:
            s = pd.to_numeric(X_df[col], errors="coerce").dropna()
            if len(s) >= 5:
                if self.method == "zscore":
                    mean = float(s.mean())
                    std = float(s.std())
                    self.bounds_[col] = (mean - self.factor * std, mean + self.factor * std)
                else:
                    q25 = float(np.percentile(s, 25))
                    q75 = float(np.percentile(s, 75))
                    iqr = q75 - q25
                    self.bounds_[col] = (q25 - self.factor * iqr, q75 + self.factor * iqr)
            else:
                self.bounds_[col] = (-np.inf, np.inf)
        self.feature_names_in_ = np.array(X_df.columns, dtype=object)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        for col, (low, high) in self.bounds_.items():
            if col in X_df.columns:
                X_df[col] = pd.to_numeric(X_df[col], errors="coerce").clip(lower=low, upper=high)
        return X_df

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        if input_features is not None:
            return np.array(input_features, dtype=object)
        return self.feature_names_in_ if self.feature_names_in_ is not None else np.array([], dtype=object)


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extrait l'année, mois, jour, jour de semaine et jours écoulés depuis la date min du Train."""

    def __init__(self):
        self.ref_dates_: dict[str, pd.Timestamp] = {}
        self.feature_names_out_: np.ndarray | None = None

    def fit(self, X: Any, y: Any = None) -> DateFeatureExtractor:
        X_df = pd.DataFrame(X)
        self.ref_dates_ = {}
        out_names = []
        for col in X_df.columns:
            dt = pd.to_datetime(X_df[col], errors="coerce", format="mixed")
            min_date = dt.min()
            self.ref_dates_[col] = min_date if pd.notna(min_date) else pd.Timestamp("1970-01-01")
            out_names.extend([
                f"{col}_year",
                f"{col}_month",
                f"{col}_day",
                f"{col}_day_of_week",
                f"{col}_days_elapsed",
            ])
        self.feature_names_out_ = np.array(out_names, dtype=object)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        X_df = pd.DataFrame(X)
        cols_dict = {}
        for col in X_df.columns:
            dt = pd.to_datetime(X_df[col], errors="coerce", format="mixed")
            ref = self.ref_dates_.get(col, pd.Timestamp("1970-01-01"))
            cols_dict[f"{col}_year"] = dt.dt.year.fillna(0).astype(int)
            cols_dict[f"{col}_month"] = dt.dt.month.fillna(0).astype(int)
            cols_dict[f"{col}_day"] = dt.dt.day.fillna(0).astype(int)
            cols_dict[f"{col}_day_of_week"] = dt.dt.day_of_week.fillna(0).astype(int)
            cols_dict[f"{col}_days_elapsed"] = (dt - ref).dt.days.fillna(0).astype(int)
        return pd.DataFrame(cols_dict, index=X_df.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return self.feature_names_out_ if self.feature_names_out_ is not None else np.array([], dtype=object)


class TextTfidfExtractor(BaseEstimator, TransformerMixin):
    """Vectorisation TF-IDF supervisée et robuste pour les champs de texte libre."""

    def __init__(self, max_features: int = 10):
        self.max_features = max_features
        self.vectorizers_: dict[str, TfidfVectorizer] = {}
        self.feature_names_out_: np.ndarray | None = None

    def fit(self, X: Any, y: Any = None) -> TextTfidfExtractor:
        X_df = pd.DataFrame(X)
        self.vectorizers_ = {}
        out_names = []
        for col in X_df.columns:
            vec = TfidfVectorizer(max_features=self.max_features)
            text_series = X_df[col].fillna("").astype(str)
            vec.fit(text_series)
            self.vectorizers_[col] = vec
            terms = vec.get_feature_names_out()
            out_names.extend([f"{col}_tfidf_{term}" for term in terms])
        self.feature_names_out_ = np.array(out_names, dtype=object)
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        X_df = pd.DataFrame(X)
        dfs = []
        for col in X_df.columns:
            vec = self.vectorizers_.get(col)
            if vec is None:
                continue
            text_series = X_df[col].fillna("").astype(str)
            arr = vec.transform(text_series).toarray()
            terms = [f"{col}_tfidf_{term}" for term in vec.get_feature_names_out()]
            dfs.append(pd.DataFrame(arr, columns=terms, index=X_df.index))
        if dfs:
            return pd.concat(dfs, axis=1)
        return pd.DataFrame(index=X_df.index)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        return self.feature_names_out_ if self.feature_names_out_ is not None else np.array([], dtype=object)


# ---------------------------------------------------------------------------
# Configuration et orchestration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    target_col: str
    problem_type: str
    test_size: float = 0.20
    random_state: int = 42
    stratify: bool = True
    numeric_scaler: str = "StandardScaler"  # 'StandardScaler', 'MinMaxScaler', 'RobustScaler', 'none'
    numeric_imputer: str = "mediane"        # 'mediane', 'moyenne', 'constante'
    numeric_imputer_const: float = 0.0
    outlier_action: str = "clip"            # 'none', 'clip', 'drop'
    outlier_method: str = "iqr"             # 'iqr', 'zscore'
    outlier_factor: float = 1.5
    drop_outliers_confirmed: bool = False
    categorical_imputer: str = "valeur_manquante"  # 'mode', 'valeur_manquante', 'constante'
    categorical_encoder: str = "ohe"        # 'ohe', 'ordinal'
    rare_categories_min_freq: float = 0.03
    date_features_enabled: bool = True
    text_strategy: str = "exclude"          # 'exclude', 'tfidf'
    text_max_features: int = 10
    excluded_columns: list[str] = field(default_factory=list)
    ordinal_orders: dict[str, list[str]] = field(default_factory=dict)
    drop_missing_target: bool = True


@dataclass
class PreparationResult:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    pipeline: ColumnTransformer
    feature_names: list[str]
    warnings: list[str]
    n_dropped_missing_target: int
    n_dropped_outliers: int


def build_preprocessor(
    config: PipelineConfig,
    column_types: dict[str, str],
    feature_cols: list[str],
) -> ColumnTransformer:
    """Construit un ColumnTransformer scikit-learn propre et reproductible."""
    transformers = []

    # 1. Numériques
    num_cols = [
        c for c in feature_cols
        if column_types.get(c) in ("numerique_continue", "numerique_discrete")
    ]
    if num_cols:
        steps = []
        if config.numeric_imputer == "moyenne":
            steps.append(("imputer", SimpleImputer(strategy="mean")))
        elif config.numeric_imputer == "constante":
            steps.append(("imputer", SimpleImputer(strategy="constant", fill_value=config.numeric_imputer_const)))
        else:
            steps.append(("imputer", SimpleImputer(strategy="median")))

        if config.outlier_action == "clip":
            steps.append(("winsorizer", OutlierWinsorizer(factor=config.outlier_factor, method=config.outlier_method)))

        if config.numeric_scaler == "StandardScaler":
            steps.append(("scaler", StandardScaler()))
        elif config.numeric_scaler == "MinMaxScaler":
            steps.append(("scaler", MinMaxScaler()))
        elif config.numeric_scaler == "RobustScaler":
            steps.append(("scaler", RobustScaler()))
        # si 'none', aucun scaler ajouté

        transformers.append(("num", Pipeline(steps), num_cols))

    # 2. Booléennes
    bool_cols = [c for c in feature_cols if column_types.get(c) == "booleenne"]
    if bool_cols:
        bool_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append(("bool", bool_pipe, bool_cols))

    # 3. Catégorielles nominales
    nom_cols = [c for c in feature_cols if column_types.get(c) == "categorielle_nominale"]
    if nom_cols:
        fill_val = "Missing" if config.categorical_imputer == "valeur_manquante" else "Inconnu"
        cat_strat = "most_frequent" if config.categorical_imputer == "mode" else "constant"
        cat_imp = SimpleImputer(strategy=cat_strat, fill_value=fill_val)
        
        min_f = config.rare_categories_min_freq if config.rare_categories_min_freq > 0 else None
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=min_f)
        
        transformers.append(("cat_nom", Pipeline([("imp", cat_imp), ("ohe", ohe)]), nom_cols))

    # 4. Catégorielles ordinales
    ord_cols = [c for c in feature_cols if column_types.get(c) == "categorielle_ordinale"]
    for col in ord_cols:
        order = config.ordinal_orders.get(col)
        cats = [order] if order else "auto"
        ord_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(categories=cats, handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers.append((f"cat_ord_{col}", ord_pipe, [col]))

    # 5. Dates
    date_cols = [c for c in feature_cols if column_types.get(c) == "date_heure"]
    if date_cols and config.date_features_enabled:
        transformers.append(("dates", DateFeatureExtractor(), date_cols))

    # 6. Texte libre
    text_cols = [c for c in feature_cols if column_types.get(c) == "texte_libre"]
    if text_cols and config.text_strategy == "tfidf":
        transformers.append(("text", TextTfidfExtractor(max_features=config.text_max_features), text_cols))

    if not transformers:
        raise ValueError("Aucune variable explicative valide n'a été sélectionnée pour le préprocesseur.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    preprocessor.set_output(transform="pandas")
    return preprocessor


def execute_preparation(
    df: pd.DataFrame,
    config: PipelineConfig,
    column_types: dict[str, str],
) -> PreparationResult:
    """Exécute la préparation complète des données en prévenant toute fuite de données."""
    warnings: list[str] = []
    work_df = df.copy()

    # 1. Vérification de la cible
    target_col = config.target_col
    if target_col not in work_df.columns:
        raise ValueError(f"La colonne cible « {target_col} » est introuvable.")

    # 2. Lignes avec cible manquante
    n_dropped_missing_target = 0
    if config.drop_missing_target:
        missing_target_mask = work_df[target_col].isna()
        n_dropped_missing_target = int(missing_target_mask.sum())
        if n_dropped_missing_target > 0:
            work_df = work_df[~missing_target_mask].copy()
            warnings.append(f"{n_dropped_missing_target} ligne(s) sans cible supprimée(s).")

    if work_df.empty:
        raise ValueError("Toutes les lignes ont été supprimées car la cible était manquante.")

    # 3. Suppression explicite des outliers (si demandé et confirmé)
    n_dropped_outliers = 0
    if config.outlier_action == "drop" and config.drop_outliers_confirmed:
        num_cols = [
            c for c in work_df.columns
            if c != target_col and column_types.get(c) in ("numerique_continue", "numerique_discrete")
        ]
        outlier_mask = pd.Series(False, index=work_df.index)
        for c in num_cols:
            s = pd.to_numeric(work_df[c], errors="coerce").dropna()
            if len(s) >= 5:
                q25, q75 = np.percentile(s, 25), np.percentile(s, 75)
                iqr = q75 - q25
                low, high = q25 - config.outlier_factor * iqr, q75 + config.outlier_factor * iqr
                outlier_mask |= (work_df[c] < low) | (work_df[c] > high)
        n_dropped_outliers = int(outlier_mask.sum())
        if n_dropped_outliers > 0:
            work_df = work_df[~outlier_mask].copy()
            warnings.append(f"{n_dropped_outliers} ligne(s) avec valeurs aberrantes supprimée(s).")

    # 4. Sélection des variables explicatives X
    feature_cols = [
        c for c in work_df.columns
        if c != target_col
        and c not in config.excluded_columns
        and column_types.get(c) not in ("constante", "identifiant")
    ]
    if not feature_cols:
        raise ValueError("Aucune colonne explicative disponible après filtrage des identifiants et constantes.")

    X = work_df[feature_cols]
    y = work_df[target_col]

    # 5. Détermination de la stratification
    stratify = None
    if "classification" in config.problem_type and config.stratify:
        y_clean = y.dropna()
        counts = y_clean.value_counts()
        if (counts >= 2).all() and len(counts) >= 2:
            stratify = y
        else:
            warnings.append("Stratification désactivée : certaines classes ont moins de 2 observations.")

    # 6. Séparation Train / Test AVANT tout ajustement
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    # 7. Construction et ajustement strict sur le Train
    preprocessor = build_preprocessor(config, column_types, feature_cols)
    preprocessor.fit(X_train)

    X_train_prep = preprocessor.transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    feature_names = list(X_train_prep.columns)

    return PreparationResult(
        X_train=X_train_prep,
        X_test=X_test_prep,
        y_train=y_train,
        y_test=y_test,
        pipeline=preprocessor,
        feature_names=feature_names,
        warnings=warnings,
        n_dropped_missing_target=n_dropped_missing_target,
        n_dropped_outliers=n_dropped_outliers,
    )
