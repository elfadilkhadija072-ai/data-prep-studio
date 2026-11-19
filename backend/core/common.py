"""Module commun : gestion des sessions mémoire, modèles Pydantic et constantes."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field

# Jetons traités comme des valeurs manquantes
NA_TOKENS = [
    "", "NA", "na", "null", "NULL", "Null", "?", "-", "--",
    "nan", "NaN", "NAN", "None", "none", "#N/A", "N/A", "n/a"
]

# Capacité maximale de sessions LRU en mémoire
MAX_SESSIONS = 10
SESSION_EXPIRY_SECONDS = 7200  # 2 heures


@dataclass
class DatasetSession:
    """Session contenant l'état d'un dataset en cours de traitement."""
    session_id: str
    filename: str
    df_raw: pd.DataFrame
    df_cleaned_raw: pd.DataFrame
    target_col: Optional[str] = None
    problem_type: str = "classification"  # 'classification', 'regression', 'unsupervised'
    is_supervised: bool = True
    test_size: float = 0.20
    random_state: int = 42
    drop_duplicates: bool = True
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    
    # Résultats d'analyse et plan
    column_types: Dict[str, str] = field(default_factory=dict)
    prep_plan: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Résultats de l'exécution
    execution_result: Optional[Dict[str, Any]] = None
    df_train: Optional[pd.DataFrame] = None
    df_test: Optional[pd.DataFrame] = None
    df_full_prepared: Optional[pd.DataFrame] = None
    excel_report_bytes: Optional[bytes] = None

    # Modélisation et Sécurité ML
    last_trained_model: Optional[Dict[str, Any]] = None
    debugger_logs: List[Dict[str, Any]] = field(default_factory=list)


class SessionManager:
    """Gestionnaire de sessions en mémoire avec nettoyage automatique LRU."""
    def __init__(self):
        self._sessions: Dict[str, DatasetSession] = {}

    def create(self, filename: str, df: pd.DataFrame) -> DatasetSession:
        self.cleanup()
        session_id = str(uuid.uuid4())[:8]
        session = DatasetSession(
            session_id=session_id,
            filename=filename,
            df_raw=df.copy(),
            df_cleaned_raw=df.copy(),
        )
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[DatasetSession]:
        session = self._sessions.get(session_id)
        if session:
            session.last_accessed = time.time()
        return session

    def cleanup(self):
        now = time.time()
        # Supprimer les sessions expirées
        expired = [sid for sid, s in self._sessions.items() if now - s.last_accessed > SESSION_EXPIRY_SECONDS]
        for sid in expired:
            del self._sessions[sid]
        # Si trop de sessions, virer les plus anciennes
        if len(self._sessions) >= MAX_SESSIONS:
            sorted_by_access = sorted(self._sessions.items(), key=lambda item: item[1].last_accessed)
            for sid, _ in sorted_by_access[:len(self._sessions) - MAX_SESSIONS + 1]:
                del self._sessions[sid]


# Singleton global pour les sessions
sessions = SessionManager()


# ---------------------------------------------------------------------------
# Modèles Pydantic pour l'API
# ---------------------------------------------------------------------------

class ConfigureRequest(BaseModel):
    session_id: str
    is_supervised: bool = True
    target_col: Optional[str] = None
    problem_type: Optional[str] = None  # 'classification', 'regression', 'unsupervised'
    test_size: float = Field(default=0.20, ge=0.05, le=0.50)
    random_state: int = 42
    drop_duplicates: bool = True


class FeatureStepConfig(BaseModel):
    """Configuration du plan de nettoyage pour une feature."""
    action_drop: bool = False
    text_clean: bool = True
    impute_strategy: str = "auto"  # 'median', 'mean', 'mode', 'constant', 'none'
    impute_value: Optional[Any] = None
    outlier_strategy: str = "auto"  # 'winsorize_iqr', 'winsorize_zscore', 'winsorize_percentiles', 'drop_train', 'none'
    outlier_iqr_factor: float = 1.5
    transform_strategy: str = "none"  # 'log1p', 'sqrt', 'yeojohnson', 'none'
    scaling_strategy: str = "auto"  # 'robust', 'standard', 'minmax', 'none'
    encoding_strategy: str = "auto"  # 'onehot', 'ordinal', 'frequency', 'date_decompose', 'none'
    rare_category_min_freq: float = 0.03


class ExecutePipelineRequest(BaseModel):
    session_id: str
    drop_duplicates: bool = True
    plan: Dict[str, FeatureStepConfig]


class PreviewFeatureRequest(BaseModel):
    session_id: str
    feature_name: str
    config: FeatureStepConfig


class LoginRequest(BaseModel):
    username: str
    password: str


class TrainModelRequest(BaseModel):
    session_id: str
    model_id: str
    hyperparams: Optional[Dict[str, Any]] = None
    rebalancing_strategy: str = "none"  # 'none', 'undersample', 'smote'
    rebalancing_params: Optional[Dict[str, Any]] = None  # k_neighbors, sampling_strategy, random_state
