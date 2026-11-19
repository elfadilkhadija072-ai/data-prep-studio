import React, { useState, useEffect } from 'react';
import {
  Wand2,
  Trash2,
  Settings,
  Eye,
  CheckCheck,
  RotateCcw,
  Sliders,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  Play,
} from 'lucide-react';
import { getPrepPlan, previewFeatureTransformation, executePipeline } from '../api';
import { SeverityBadge, TypeBadge, LoadingSkeleton, Modal } from '../ui';
import { HistogramChart } from '../charts';

export function PrepTab({ sessionId, onExecutionComplete }) {
  const [loading, setLoading] = useState(true);
  const [featuresPlan, setFeaturesPlan] = useState({});
  const [summary, setSummary] = useState({ total_features: 0, recommended_count: 0, drop_count: 0, customized_count: 0 });
  const [expandedCol, setExpandedCol] = useState(null);
  const [dropDuplicates, setDropDuplicates] = useState(true);
  const [executing, setExecuting] = useState(false);
  const [error, setError] = useState(null);

  // Modal de prévisualisation Avant / Après en direct
  const [previewCol, setPreviewCol] = useState(null);
  const [previewData, setPreviewData] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    setLoading(true);
    getPrepPlan(sessionId)
      .then((data) => {
        setFeaturesPlan(data.features || {});
        setSummary(data.summary || {});
        const firstCol = Object.keys(data.features || {})[0];
        if (firstCol) setExpandedCol(firstCol);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [sessionId]);

  const updateFeatureConfig = (colName, partialConfig) => {
    setFeaturesPlan((prev) => {
      const current = prev[colName];
      const newConfig = { ...current.user_config, ...partialConfig };
      return {
        ...prev,
        [colName]: {
          ...current,
          user_config: newConfig,
          status: 'customized',
        },
      };
    });
  };

  const acceptRecommendation = (colName) => {
    setFeaturesPlan((prev) => {
      const current = prev[colName];
      return {
        ...prev,
        [colName]: {
          ...current,
          user_config: { ...current.recommended_config },
          status: 'recommended',
        },
      };
    });
  };

  const setPassthrough = (colName) => {
    setFeaturesPlan((prev) => {
      const current = prev[colName];
      return {
        ...prev,
        [colName]: {
          ...current,
          user_config: {
            action_drop: false,
            text_clean: false,
            impute_strategy: 'none',
            outlier_strategy: 'none',
            transform_strategy: 'none',
            scaling_strategy: 'none',
            encoding_strategy: 'none',
          },
          status: 'passthrough',
        },
      };
    });
  };

  const acceptAllRecommendations = () => {
    setFeaturesPlan((prev) => {
      const updated = {};
      for (const [col, item] of Object.entries(prev)) {
        updated[col] = {
          ...item,
          user_config: { ...item.recommended_config },
          status: 'recommended',
        };
      }
      return updated;
    });
  };

  // Prévisualisation Avant/Après en direct
  const handleOpenPreview = async (colName) => {
    setPreviewCol(colName);
    setPreviewLoading(true);
    const item = featuresPlan[colName];
    try {
      const res = await previewFeatureTransformation(sessionId, colName, item.user_config);
      setPreviewData(res);
    } catch (err) {
      console.error(err);
    } finally {
      setPreviewLoading(false);
    }
  };

  // Exécution du pipeline complet
  const handleExecute = async () => {
    setExecuting(true);
    setError(null);
    try {
      const planPayload = {};
      for (const [col, item] of Object.entries(featuresPlan)) {
        planPayload[col] = item.user_config;
      }
      const res = await executePipeline(sessionId, planPayload, dropDuplicates);
      onExecutionComplete(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setExecuting(false);
    }
  };

  // Décomptes en direct pour la barre d'action
  const totalFeatures = Object.keys(featuresPlan).length;
  const droppedCount = Object.values(featuresPlan).filter((f) => f.user_config?.action_drop).length;
  const customizedCount = Object.values(featuresPlan).filter((f) => f.status === 'customized').length;
  const recommendedCount = Object.values(featuresPlan).filter((f) => f.status === 'recommended').length;

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <LoadingSkeleton height={80} />
        <LoadingSkeleton height={400} />
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* En-tête */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.85rem', marginBottom: '0.4rem' }}>
            3. Preprocessing, Nettoyage & Recommandations
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
            Chaque feature bénéficie d'un plan de transformation sur-mesure justifié par ses propriétés statistiques.
          </p>
        </div>
        <button className="btn btn-secondary" onClick={acceptAllRecommendations}>
          <CheckCheck size={16} color="var(--teal-400)" />
          Tout accepter d'un clic
        </button>
      </div>

      {error && (
        <div style={{ padding: '1rem', background: 'var(--red-bg)', color: 'var(--red-400)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(248,113,113,0.3)' }}>
          {error}
        </div>
      )}

      {/* Liste des features et leurs plans de nettoyage */}
      <div>
        {Object.entries(featuresPlan).map(([colName, item]) => {
          const cfg = item.user_config;
          const isExpanded = expandedCol === colName;
          const isTarget = item.is_target;

          let cardClass = 'feature-card';
          if (cfg.action_drop) cardClass += ' dropped';
          else if (item.status === 'customized') cardClass += ' customized';
          else if (item.status === 'recommended') cardClass += ' recommended';

          return (
            <div key={colName} className={cardClass}>
              <div
                className="feature-header"
                onClick={() => setExpandedCol(isExpanded ? null : colName)}
              >
                <div className="feature-title-group">
                  {isExpanded ? <ChevronDown size={18} color="var(--text-muted)" /> : <ChevronRight size={18} color="var(--text-muted)" />}
                  <span className="feature-name">{colName}</span>
                  <TypeBadge type={item.type} />
                  {isTarget && <span className="brand-badge">CIBLE</span>}
                  {item.anomalies?.map((a, idx) => (
                    <SeverityBadge key={idx} severity={a.severity} title={a.title} />
                  ))}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }} onClick={(e) => e.stopPropagation()}>
                  <button
                    className="btn btn-ghost btn-sm"
                    onClick={() => handleOpenPreview(colName)}
                    title="Prévisualiser avant/après"
                  >
                    <Eye size={14} />
                    Aperçu Direct
                  </button>
                  {!isTarget && (
                    <>
                      <button
                        className={`btn btn-sm ${item.status === 'recommended' ? 'btn-primary' : 'btn-secondary'}`}
                        onClick={() => acceptRecommendation(colName)}
                        title="Accepter la recommandation automatique"
                      >
                        <Wand2 size={13} />
                        Recommandé
                      </button>
                      <button
                        className={`btn btn-sm ${item.status === 'passthrough' ? 'btn-primary' : 'btn-ghost'}`}
                        onClick={() => setPassthrough(colName)}
                        title="Ne rien modifier sur cette colonne"
                      >
                        Ne rien faire
                      </button>
                    </>
                  )}
                </div>
              </div>

              {isExpanded && (
                <div className="feature-body">
                  {/* Justification de la recommandation */}
                  {item.justifications && item.justifications.length > 0 && (
                    <div style={{ background: 'rgba(45, 212, 191, 0.06)', border: '1px solid rgba(45, 212, 191, 0.2)', padding: '0.75rem 1rem', borderRadius: 'var(--radius-sm)', marginBottom: '1rem' }}>
                      <div style={{ fontWeight: 600, fontSize: '0.8rem', color: 'var(--teal-400)', marginBottom: '0.2rem' }}>
                        Justification statistique chiffrée :
                      </div>
                      <ul style={{ paddingLeft: '1.2rem', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                        {item.justifications.map((j, jIdx) => (
                          <li key={jIdx}>{j}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Contrôles de personnalisation */}
                  {!isTarget && (
                    <div className="prep-controls-grid">
                      {/* 1. Action globale : Supprimer ou conserver */}
                      <div className="control-group">
                        <label className="control-label">Action sur la colonne</label>
                        <select
                          className="form-select"
                          value={cfg.action_drop ? 'drop' : 'keep'}
                          onChange={(e) => updateFeatureConfig(colName, { action_drop: e.target.value === 'drop' })}
                        >
                          <option value="keep">Conserver dans le modèle</option>
                          <option value="drop">Supprimer la colonne</option>
                        </select>
                      </div>

                      {/* 2. Imputation */}
                      {!cfg.action_drop && (
                        <div className="control-group">
                          <label className="control-label">Stratégie d'Imputation</label>
                          <select
                            className="form-select"
                            value={cfg.impute_strategy}
                            onChange={(e) => updateFeatureConfig(colName, { impute_strategy: e.target.value })}
                          >
                            <option value="none">Aucune imputation (ignorer)</option>
                            <option value="median">Médiane (robuste)</option>
                            <option value="mean">Moyenne arithmétique</option>
                            <option value="mode">Mode (plus fréquente)</option>
                            <option value="constant">Valeur constante ('Missing' ou 0)</option>
                          </select>
                        </div>
                      )}

                      {/* 3. Outliers si numérique */}
                      {!cfg.action_drop && (item.type === 'numeric' || item.type === 'boolean') && (
                        <div className="control-group">
                          <label className="control-label">Traitement des Outliers</label>
                          <select
                            className="form-select"
                            value={cfg.outlier_strategy}
                            onChange={(e) => updateFeatureConfig(colName, { outlier_strategy: e.target.value })}
                          >
                            <option value="none">Conserver sans modification</option>
                            <option value="winsorize_iqr">Winsorisation IQR (1.5)</option>
                            <option value="winsorize_zscore">Winsorisation Z-Score (3.0)</option>
                            <option value="winsorize_percentiles">Plafonnement P1–P99</option>
                          </select>
                        </div>
                      )}

                      {/* 4. Transformation asymétrie si numérique */}
                      {!cfg.action_drop && item.type === 'numeric' && (
                        <div className="control-group">
                          <label className="control-label">Transformation de variance</label>
                          <select
                            className="form-select"
                            value={cfg.transform_strategy}
                            onChange={(e) => updateFeatureConfig(colName, { transform_strategy: e.target.value })}
                          >
                            <option value="none">Aucune transformation</option>
                            <option value="log1p">Logarithme log(1 + x)</option>
                            <option value="sqrt">Racine carrée √x</option>
                          </select>
                        </div>
                      )}

                      {/* 5. Scaling si numérique */}
                      {!cfg.action_drop && (item.type === 'numeric' || item.type === 'boolean') && (
                        <div className="control-group">
                          <label className="control-label">Mise à l'échelle (Scaler)</label>
                          <select
                            className="form-select"
                            value={cfg.scaling_strategy}
                            onChange={(e) => updateFeatureConfig(colName, { scaling_strategy: e.target.value })}
                          >
                            <option value="none">Aucun scaler (échelle brute)</option>
                            <option value="robust">RobustScaler (centré médiane/IQR)</option>
                            <option value="standard">StandardScaler (Moyenne 0, Écart-type 1)</option>
                            <option value="minmax">MinMaxScaler [0, 1]</option>
                          </select>
                        </div>
                      )}

                      {/* 6. Encodage */}
                      {!cfg.action_drop && item.type === 'categorical' && (
                        <div className="control-group">
                          <label className="control-label">Encodage catégoriel</label>
                          <select
                            className="form-select"
                            value={cfg.encoding_strategy}
                            onChange={(e) => updateFeatureConfig(colName, { encoding_strategy: e.target.value })}
                          >
                            <option value="onehot">One-Hot Encoding (avec seuil rare)</option>
                            <option value="frequency">Frequency Encoding (taux d'apparition)</option>
                            <option value="ordinal">Ordinal Encoding (entiers)</option>
                            <option value="none">Conserver brut</option>
                          </select>
                        </div>
                      )}

                      {!cfg.action_drop && item.type === 'date' && (
                        <div className="control-group">
                          <label className="control-label">Traitement temporel</label>
                          <select
                            className="form-select"
                            value={cfg.encoding_strategy}
                            onChange={(e) => updateFeatureConfig(colName, { encoding_strategy: e.target.value })}
                          >
                            <option value="date_decompose">Extraire Année, Mois, Jour, Jour de semaine, Jours écoulés</option>
                            <option value="none">Conserver format texte</option>
                          </select>
                        </div>
                      )}
                    </div>
                  )}

                  {isTarget && (
                    <p style={{ fontSize: '0.82rem', color: 'var(--violet-400)', fontStyle: 'italic' }}>
                      Cette variable est la cible du problème supervisé. Elle est protégée contre le scaling et les modifications involontaires.
                    </p>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Barre d'action fixe en bas */}
      <div className="sticky-action-bar">
        <div className="action-bar-counts">
          <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', cursor: 'pointer', fontSize: '0.85rem' }}>
            <input
              type="checkbox"
              checked={dropDuplicates}
              onChange={(e) => setDropDuplicates(e.target.checked)}
              style={{ accentColor: 'var(--teal-400)', width: 16, height: 16 }}
            />
            Supprimer les doublons stricts avant séparation
          </label>
          <div className="count-pill">
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--teal-400)' }} />
            Recommandées : <strong>{recommendedCount}</strong>
          </div>
          <div className="count-pill">
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--violet-400)' }} />
            Personnalisées : <strong>{customizedCount}</strong>
          </div>
          <div className="count-pill">
            <span style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--red-400)' }} />
            Supprimées : <strong>{droppedCount}</strong>
          </div>
        </div>

        <button
          className="btn btn-primary"
          onClick={handleExecute}
          disabled={executing}
        >
          <Play size={16} />
          {executing ? 'Exécution du pipeline en cours…' : 'Exécuter la préparation complète (Anti-Leakage) ➔'}
        </button>
      </div>

      {/* Modal de prévisualisation Avant / Après */}
      <Modal
        isOpen={Boolean(previewCol)}
        onClose={() => setPreviewCol(null)}
        title={`Aperçu Avant / Après : ${previewCol}`}
      >
        {previewLoading ? (
          <LoadingSkeleton height={250} />
        ) : previewData ? (
          <div>
            {previewData.is_dropped ? (
              <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--red-400)' }}>
                Cette colonne est configurée pour être <strong>supprimée</strong> du dataset final.
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                {/* Avant */}
                <div style={{ background: 'rgba(0,0,0,0.25)', padding: '1rem', borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
                  <h4 style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '0.75rem', fontWeight: 600 }}>
                    1. AVANT NETTOYAGE
                  </h4>
                  <table className="data-table" style={{ marginBottom: '1rem' }}>
                    <tbody>
                      <tr><td>Manquants</td><td>{previewData.stats_before.missing} ({previewData.stats_before.missing_pct}%)</td></tr>
                      <tr><td>Moyenne / Médiane</td><td>{previewData.stats_before.mean ?? '-'} / {previewData.stats_before.median ?? '-'}</td></tr>
                      <tr><td>Min / Max</td><td>{previewData.stats_before.min ?? '-'} / {previewData.stats_before.max ?? '-'}</td></tr>
                    </tbody>
                  </table>
                  <HistogramChart data={previewData.hist_before} height={180} color="#94a3b8" />
                </div>

                {/* Après */}
                <div style={{ background: 'rgba(45, 212, 191, 0.05)', padding: '1rem', borderRadius: 8, border: '1px solid rgba(45, 212, 191, 0.2)' }}>
                  <h4 style={{ color: 'var(--teal-400)', fontSize: '0.85rem', marginBottom: '0.75rem', fontWeight: 600 }}>
                    2. APRÈS NETTOYAGE & TRANSFORMATION
                  </h4>
                  <table className="data-table" style={{ marginBottom: '1rem' }}>
                    <tbody>
                      <tr><td>Manquants</td><td>{previewData.stats_after.missing} ({previewData.stats_after.missing_pct}%)</td></tr>
                      <tr><td>Moyenne / Médiane</td><td>{previewData.stats_after.mean ?? '-'} / {previewData.stats_after.median ?? '-'}</td></tr>
                      <tr><td>Min / Max</td><td>{previewData.stats_after.min ?? '-'} / {previewData.stats_after.max ?? '-'}</td></tr>
                    </tbody>
                  </table>
                  <HistogramChart data={previewData.hist_after} height={180} color="#2dd4bf" />
                </div>
              </div>
            )}
          </div>
        ) : null}
      </Modal>
    </div>
  );
}
