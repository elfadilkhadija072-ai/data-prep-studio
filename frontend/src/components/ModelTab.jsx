import React, { useState, useEffect } from 'react';
import {
  BrainCircuit,
  ShieldCheck,
  ShieldAlert,
  Play,
  CheckCircle2,
  Zap,
  Lock,
  Download,
  Sliders,
  TrendingUp,
  Cpu,
  Layers,
  Sparkles,
  ArrowRight,
  RotateCcw,
  AlertTriangle,
  KeyRound,
  FileCode,
} from 'lucide-react';
import {
  getAvailableModels,
  trainModel,
  getCurrentModel,
  getModelDownloadUrl,
  getModelManifestUrl,
} from '../api';
import { KpiCard, LoadingSkeleton } from '../ui';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ScatterChart, Scatter } from 'recharts';

export function ModelTab({ sessionId, sessionData, onProceedToExport }) {
  const [loading, setLoading] = useState(true);
  const [modelsList, setModelsList] = useState([]);
  const [problemType, setProblemType] = useState('classification');
  const [targetCol, setTargetCol] = useState('');
  const [selectedModelId, setSelectedModelId] = useState('');
  
  // Hyperparamètres personnalisables
  const [hyperparams, setHyperparams] = useState({});
  const [training, setTraining] = useState(false);
  const [modelResult, setModelResult] = useState(null);
  const [error, setError] = useState(null);

  // Rééquilibrage des classes
  const [rebalStrategy, setRebalStrategy] = useState('none');
  const [rebalKNeighbors, setRebalKNeighbors] = useState(5);
  const [rebalSamplingStrategy, setRebalSamplingStrategy] = useState('auto');

  useEffect(() => {
    if (!sessionId) return;
    setLoading(true);
    getAvailableModels(sessionId)
      .then((data) => {
        setModelsList(data.models || []);
        setProblemType(data.problem_type || 'classification');
        setTargetCol(data.target_col || '');
        if (data.models && data.models.length > 0) {
          const first = data.models[0];
          setSelectedModelId(first.id);
          setHyperparams(first.default_params || {});
        }
        // Vérifier si un modèle a déjà été entraîné
        return getCurrentModel(sessionId);
      })
      .then((curr) => {
        if (curr && curr.status === 'success') {
          setModelResult(curr);
          setSelectedModelId(curr.model_id);
        }
      })
      .catch((err) => {
        console.warn(err);
      })
      .finally(() => setLoading(false));
  }, [sessionId]);

  const handleSelectModel = (mId) => {
    setSelectedModelId(mId);
    const found = modelsList.find((m) => m.id === mId);
    if (found) {
      setHyperparams(found.default_params || {});
    }
  };

  const handleHyperparamChange = (key, value) => {
    setHyperparams((prev) => ({ ...prev, [key]: value }));
  };

  const handleTrain = async () => {
    if (!sessionId || !selectedModelId) return;
    setTraining(true);
    setError(null);
    try {
      const rebalParams = rebalStrategy !== 'none' ? { k_neighbors: rebalKNeighbors, sampling_strategy: rebalSamplingStrategy, random_state: 42 } : null;
      const res = await trainModel(sessionId, selectedModelId, hyperparams, rebalStrategy, rebalParams);
      setModelResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setTraining(false);
    }
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <LoadingSkeleton height={80} />
        <LoadingSkeleton height={400} />
      </div>
    );
  }

  const selectedModelInfo = modelsList.find((m) => m.id === selectedModelId) || modelsList[0];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* En-tête de l'onglet */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.3rem' }}>
            <span className="brand-badge" style={{ background: 'rgba(167, 139, 250, 0.15)', color: 'var(--violet-400)', borderColor: 'rgba(167, 139, 250, 0.3)' }}>
              <BrainCircuit size={13} />
              Machine Learning Studio
            </span>
            <span className="brand-badge" style={{ background: 'rgba(45, 212, 191, 0.15)', color: 'var(--teal-400)', borderColor: 'rgba(45, 212, 191, 0.3)' }}>
              <Lock size={12} />
              Audit Cyber Automatique
            </span>
          </div>
          <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.85rem' }}>
            4. Sélection, Entraînement & Sécurisation du Modèle
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
            Comparez et entraînez des algorithmes de pointe (KNN, Régression, Arbres, Forêts) et auditez leur robustesse face aux cyber-menaces.
          </p>
        </div>

        {modelResult && (
          <button className="btn btn-primary" onClick={onProceedToExport}>
            Accéder à l'Export Final
            <ArrowRight size={16} />
          </button>
        )}
      </div>

      {error && (
        <div style={{ padding: '1rem', background: 'var(--red-bg)', color: 'var(--red-400)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(248,113,113,0.3)', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <AlertTriangle size={18} />
          <span>{error}</span>
        </div>
      )}

      {/* Grille de sélection des modèles */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '1rem' }}>
        {modelsList.map((model) => {
          const isSelected = selectedModelId === model.id;
          return (
            <div
              key={model.id}
              onClick={() => handleSelectModel(model.id)}
              className="glass-card"
              style={{
                cursor: 'pointer',
                borderColor: isSelected ? 'var(--teal-400)' : 'var(--border-subtle)',
                background: isSelected ? 'rgba(45, 212, 191, 0.08)' : 'var(--bg-card)',
                boxShadow: isSelected ? '0 0 20px rgba(45, 212, 191, 0.2)' : 'none',
                transition: 'var(--transition-normal)',
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              {isSelected && (
                <div style={{ position: 'absolute', top: 0, left: 0, width: 4, height: '100%', background: 'var(--teal-400)' }} />
              )}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                <span style={{ fontWeight: 700, fontSize: '1.05rem', color: isSelected ? '#ffffff' : 'var(--text-secondary)' }}>
                  {model.name}
                </span>
                {isSelected && <CheckCircle2 size={18} color="var(--teal-400)" />}
              </div>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', lineHeight: 1.4, marginBottom: '0.75rem' }}>
                {model.description}
              </p>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem', color: 'var(--text-dim)' }}>
                <Cpu size={12} />
                <span>Paramètres par défaut prêts</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Panneau de configuration des hyperparamètres & Lancement */}
      {selectedModelInfo && (
        <div className="glass-card" style={{ borderColor: 'rgba(167, 139, 250, 0.25)', background: 'linear-gradient(135deg, rgba(167, 139, 250, 0.04), rgba(45, 212, 191, 0.04))' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{ width: 36, height: 36, borderRadius: 'var(--radius-sm)', background: 'rgba(167, 139, 250, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Sliders size={20} color="var(--violet-400)" />
              </div>
              <div>
                <h3 style={{ fontSize: '1.15rem', color: '#ffffff', fontWeight: 600 }}>
                  Configuration : {selectedModelInfo.name}
                </h3>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Cible supervisée : <strong style={{ color: 'var(--teal-400)' }}>{targetCol}</strong> ({problemType})
                </span>
              </div>
            </div>

            <button
              className="btn btn-primary"
              onClick={handleTrain}
              disabled={training}
              style={{ padding: '0.65rem 1.4rem', fontSize: '0.92rem' }}
            >
              {training ? (
                <>
                  <RotateCcw size={16} className="spin-animation" />
                  Entraînement & Audit Cyber en cours…
                </>
              ) : (
                <>
                  <Play size={16} />
                  Entraîner & Certifier le Modèle
                </>
              )}
            </button>
          </div>

          {/* ============================================================ */}
          {/* BLOC REEQUILIBRAGE DES CLASSES (Anti-Data-Leakage) */}
          {/* ============================================================ */}
          {problemType === 'classification' && (
            <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid rgba(251,191,36,0.25)', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '1rem' }}>
                <Layers size={18} color="var(--amber-400)" />
                <h4 style={{ fontSize: '1rem', fontWeight: 600, color: '#ffffff' }}>
                  Gestion du Déséquilibre des Classes
                </h4>
                <span style={{ fontSize: '0.72rem', background: 'rgba(45,212,191,0.12)', color: 'var(--teal-400)', border: '1px solid rgba(45,212,191,0.3)', borderRadius: 20, padding: '0.1rem 0.6rem', fontWeight: 600 }}>
                  Anti-Data-Leakage · Train uniquement
                </span>
              </div>

              {/* Explication pédagogique */}
              <div style={{ background: 'rgba(167,139,250,0.08)', border: '1px solid rgba(167,139,250,0.2)', borderRadius: 'var(--radius-sm)', padding: '0.85rem 1rem', marginBottom: '1rem', fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                <strong style={{ color: 'var(--violet-400)' }}>Méthodes disponibles :</strong><br />
                <strong style={{ color: 'var(--teal-400)' }}>• Undersampling</strong> — Réduit aléatoirement la classe majoritaire. Rapide, mais <em>perte d&apos;information</em> potentiellement importante.<br />
                <strong style={{ color: 'var(--green-400)' }}>• SMOTE</strong> — Génère des exemples synthétiques de la classe minoritaire entre un point A et son voisin B :<br />
                <code style={{ background: 'rgba(0,0,0,0.3)', padding: '0.1rem 0.4rem', borderRadius: 4, fontFamily: 'monospace', color: '#fbbf24' }}>
                  nouveau = A + λ·(B − A), λ ∈ [0,1]
                </code>{' '}
                Peut introduire du bruit si les classes se chevauchent ou si des outliers sont présents.
              </div>

              {/* Sélecteur de stratégie */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '0.75rem' }}>
                {[
                  { id: 'none', label: 'Aucune', desc: 'Entraîner avec le déséquilibre original', color: 'var(--text-muted)', icon: '—' },
                  { id: 'undersample', label: 'Undersampling', desc: 'Réduire la classe majoritaire', color: 'var(--amber-400)', icon: '↓' },
                  { id: 'smote', label: 'SMOTE', desc: 'Générer des exemples synthétiques', color: 'var(--green-400)', icon: '+' },
                ].map(opt => (
                  <div
                    key={opt.id}
                    onClick={() => setRebalStrategy(opt.id)}
                    style={{
                      cursor: 'pointer',
                      padding: '0.75rem 1rem',
                      borderRadius: 'var(--radius-sm)',
                      border: `1px solid ${rebalStrategy === opt.id ? opt.color : 'var(--border-subtle)'}`,
                      background: rebalStrategy === opt.id ? `rgba(${opt.id === 'none' ? '100,116,139' : opt.id === 'undersample' ? '251,191,36' : '74,222,128'},0.1)` : 'rgba(0,0,0,0.2)',
                      transition: 'var(--transition-fast)',
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: 700, color: rebalStrategy === opt.id ? opt.color : 'var(--text-secondary)', fontSize: '0.9rem' }}>
                        {opt.label}
                      </span>
                      {rebalStrategy === opt.id && <CheckCircle2 size={15} color={opt.color} />}
                    </div>
                    <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.3rem' }}>{opt.desc}</p>
                  </div>
                ))}
              </div>

              {/* Paramètres SMOTE */}
              {rebalStrategy === 'smote' && (
                <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem' }}>
                  <div className="control-group">
                    <label className="control-label">Voisins k (SMOTE) : {rebalKNeighbors}</label>
                    <input type="range" min="1" max="20" value={rebalKNeighbors}
                      onChange={(e) => setRebalKNeighbors(parseInt(e.target.value))}
                      style={{ accentColor: 'var(--green-400)', marginTop: '0.4rem' }} />
                    <small style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>Auto-ajusté si minorité &lt; k+1</small>
                  </div>
                  <div className="control-group">
                    <label className="control-label">Stratégie d&apos;échantillonnage</label>
                    <select className="form-select" value={rebalSamplingStrategy} onChange={(e) => setRebalSamplingStrategy(e.target.value)}>
                      <option value="auto">Auto (équilibrer la minorité)</option>
                      <option value="0.5">0.5 — 50% de la classe majoritaire</option>
                      <option value="1.0">1.0 — Équilibre total</option>
                    </select>
                  </div>
                </div>
              )}

              {/* Paramètres Undersampling */}
              {rebalStrategy === 'undersample' && (
                <div style={{ marginTop: '1rem', display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.75rem' }}>
                  <div className="control-group">
                    <label className="control-label">Stratégie d&apos;échantillonnage</label>
                    <select className="form-select" value={rebalSamplingStrategy} onChange={(e) => setRebalSamplingStrategy(e.target.value)}>
                      <option value="auto">Auto (équilibrer la majoritaire)</option>
                      <option value="0.5">0.5 — Réduire au double de la minorité</option>
                      <option value="1.0">1.0 — Équilibre total</option>
                    </select>
                  </div>
                </div>
              )}

              {/* Note test inchangé */}
              <div style={{ marginTop: '0.85rem', padding: '0.5rem 0.85rem', background: 'rgba(45,212,191,0.07)', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(45,212,191,0.2)', fontSize: '0.78rem', color: 'var(--teal-400)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <ShieldCheck size={14} />
                <span>Jeu de test : <strong>inchangé</strong>. Le rééquilibrage s&apos;applique exclusivement sur le train (Anti-Data-Leakage).</span>
              </div>
            </div>
          )}
          {/* Hyperparamètres dynamiques */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', background: 'rgba(0,0,0,0.25)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            {/* KNN */}
            {(selectedModelId === 'knn' || selectedModelId === 'knn_regressor') && (
              <>
                <div className="control-group">
                  <label className="control-label">Voisins (k) : {hyperparams.n_neighbors || 5}</label>
                  <input
                    type="range"
                    min="1"
                    max="30"
                    value={hyperparams.n_neighbors || 5}
                    onChange={(e) => handleHyperparamChange('n_neighbors', parseInt(e.target.value))}
                    style={{ accentColor: 'var(--teal-400)', marginTop: '0.4rem' }}
                  />
                </div>
                <div className="control-group">
                  <label className="control-label">Pondération des poids</label>
                  <select
                    className="form-select"
                    value={hyperparams.weights || 'distance'}
                    onChange={(e) => handleHyperparamChange('weights', e.target.value)}
                  >
                    <option value="distance">Distance inverse (1 / d)</option>
                    <option value="uniform">Uniforme (équi-pondéré)</option>
                  </select>
                </div>
              </>
            )}

            {/* Régression Logistique */}
            {selectedModelId === 'logistic_regression' && (
              <>
                <div className="control-group">
                  <label className="control-label">Régularisation Inverse (C) : {hyperparams.C || 1.0}</label>
                  <input
                    type="range"
                    min="0.01"
                    max="10"
                    step="0.1"
                    value={hyperparams.C || 1.0}
                    onChange={(e) => handleHyperparamChange('C', parseFloat(e.target.value))}
                    style={{ accentColor: 'var(--teal-400)', marginTop: '0.4rem' }}
                  />
                </div>
                <div className="control-group">
                  <label className="control-label">Itérations Max</label>
                  <select
                    className="form-select"
                    value={hyperparams.max_iter || 500}
                    onChange={(e) => handleHyperparamChange('max_iter', parseInt(e.target.value))}
                  >
                    <option value="300">300 itérations</option>
                    <option value="500">500 itérations (Recommandé)</option>
                    <option value="1000">1000 itérations</option>
                  </select>
                </div>
              </>
            )}

            {/* Random Forest (Classifier ou Regressor) */}
            {(selectedModelId === 'random_forest' || selectedModelId === 'random_forest_regressor') && (
              <>
                <div className="control-group">
                  <label className="control-label">Nombre d'arbres : {hyperparams.n_estimators || 100}</label>
                  <input
                    type="range"
                    min="20"
                    max="250"
                    step="10"
                    value={hyperparams.n_estimators || 100}
                    onChange={(e) => handleHyperparamChange('n_estimators', parseInt(e.target.value))}
                    style={{ accentColor: 'var(--teal-400)', marginTop: '0.4rem' }}
                  />
                </div>
                <div className="control-group">
                  <label className="control-label">Profondeur Max : {hyperparams.max_depth || 10}</label>
                  <input
                    type="range"
                    min="3"
                    max="25"
                    value={hyperparams.max_depth || 10}
                    onChange={(e) => handleHyperparamChange('max_depth', parseInt(e.target.value))}
                    style={{ accentColor: 'var(--violet-400)', marginTop: '0.4rem' }}
                  />
                </div>
              </>
            )}

            {/* Arbre de décision */}
            {selectedModelId === 'decision_tree' && (
              <>
                <div className="control-group">
                  <label className="control-label">Profondeur Max : {hyperparams.max_depth || 6}</label>
                  <input
                    type="range"
                    min="2"
                    max="15"
                    value={hyperparams.max_depth || 6}
                    onChange={(e) => handleHyperparamChange('max_depth', parseInt(e.target.value))}
                    style={{ accentColor: 'var(--teal-400)', marginTop: '0.4rem' }}
                  />
                </div>
                <div className="control-group">
                  <label className="control-label">Critère de Split</label>
                  <select
                    className="form-select"
                    value={hyperparams.criterion || 'gini'}
                    onChange={(e) => handleHyperparamChange('criterion', e.target.value)}
                  >
                    <option value="gini">Indice de Gini</option>
                    <option value="entropy">Entropie (Gain d'information)</option>
                  </select>
                </div>
              </>
            )}

            {/* Ridge */}
            {selectedModelId === 'ridge' && (
              <div className="control-group">
                <label className="control-label">Coefficient de pénalité L2 (Alpha) : {hyperparams.alpha || 1.0}</label>
                <input
                  type="range"
                  min="0.01"
                  max="10"
                  step="0.1"
                  value={hyperparams.alpha || 1.0}
                  onChange={(e) => handleHyperparamChange('alpha', parseFloat(e.target.value))}
                  style={{ accentColor: 'var(--teal-400)', marginTop: '0.4rem' }}
                />
              </div>
            )}

            {/* OLS */}
            {selectedModelId === 'linear_regression' && (
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', gridColumn: '1 / -1' }}>
                La régression linéaire standard estime directement les coefficients par moindres carrés sans hyperparamètre libre.
              </p>
            )}
          </div>
        </div>
      )}

      {/* RÉSULTATS DE L'ENTRAÎNEMENT & AUDIT CYBER */}
      {modelResult && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
          {/* Bannière de confirmation */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem 1.5rem', background: 'rgba(45, 212, 191, 0.08)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(45, 212, 191, 0.25)', flexWrap: 'wrap', gap: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <div style={{ width: 40, height: 40, borderRadius: '50%', background: 'rgba(45, 212, 191, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <CheckCircle2 size={22} color="var(--teal-400)" />
              </div>
              <div>
                <h4 style={{ fontSize: '1.05rem', color: '#ffffff', fontWeight: 600 }}>
                  Modèle entraîné avec succès : {modelResult.model_name}
                </h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Temps d'ajustement : <strong>{modelResult.fit_duration_ms} ms</strong> • Signature SHA-256 certifiée
                </p>
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <a
                href={getModelDownloadUrl(sessionId)}
                download
                className="btn btn-secondary btn-sm"
                style={{ textDecoration: 'none' }}
              >
                <Download size={14} color="var(--teal-400)" />
                Exporter le Modèle (.joblib)
              </a>
              <a
                href={getModelManifestUrl(sessionId)}
                download
                className="btn btn-secondary btn-sm"
                style={{ textDecoration: 'none' }}
              >
                <FileCode size={14} color="var(--violet-400)" />
                Manifeste Cyber (.json)
              </a>
            </div>
          </div>

          {/* KPI de performances ML */}
          <div>
            <h3 className="section-title" style={{ fontSize: '1.25rem', marginBottom: '0.75rem' }}>
              <TrendingUp size={18} color="var(--teal-400)" />
              Métriques de Performance sur l'Échantillon de Test (Non Vu)
            </h3>
            <div className="kpis-grid">
              {problemType === 'classification' ? (
                <>
                  <KpiCard
                    label="Exactitude (Accuracy Test)"
                    value={`${(modelResult.metrics.accuracy_test * 100).toFixed(1)}%`}
                    sub={`Train : ${(modelResult.metrics.accuracy_train * 100).toFixed(1)}%`}
                    color="var(--teal-400)"
                  />
                  <KpiCard
                    label="Précision Pondérée"
                    value={`${(modelResult.metrics.precision_weighted * 100).toFixed(1)}%`}
                    sub="Taux de faux positifs modéré"
                    color="var(--violet-400)"
                  />
                  <KpiCard
                    label="Rappel (Recall) Pondéré"
                    value={`${(modelResult.metrics.recall_weighted * 100).toFixed(1)}%`}
                    sub="Couverture des classes positives"
                    color="var(--blue-400)"
                  />
                  <KpiCard
                    label="Score F1 Synthétique"
                    value={(modelResult.metrics.f1_weighted).toFixed(3)}
                    sub="Équilibre Précision / Rappel"
                    color="var(--green-400)"
                  />
                  {modelResult.metrics.auc_roc !== null && modelResult.metrics.auc_roc !== undefined && (
                    <KpiCard
                      label="Aire ROC (AUC-ROC)"
                      value={modelResult.metrics.auc_roc.toFixed(3)}
                      sub="Capacité de discrimination globale"
                      color="var(--teal-400)"
                    />
                  )}
                  {modelResult.metrics.auc_pr !== null && modelResult.metrics.auc_pr !== undefined && (
                    <KpiCard
                      label="Aire PR (AUC-PR)"
                      value={modelResult.metrics.auc_pr.toFixed(3)}
                      sub="Crucial en données déséquilibrées"
                      color="var(--amber-400)"
                    />
                  )}
                </>
              ) : (
                <>
                  <KpiCard
                    label="Coefficient R² (Test)"
                    value={(modelResult.metrics.r2_test).toFixed(3)}
                    sub={`Train R² : ${(modelResult.metrics.r2_train).toFixed(3)}`}
                    color="var(--teal-400)"
                  />
                  <KpiCard
                    label="Erreur Absolue Moyenne (MAE)"
                    value={modelResult.metrics.mae.toLocaleString('fr-FR')}
                    sub="Écart moyen réel - prédit"
                    color="var(--violet-400)"
                  />
                  <KpiCard
                    label="Racine Carrée MSE (RMSE)"
                    value={modelResult.metrics.rmse.toLocaleString('fr-FR')}
                    sub="Pénalisation des fortes erreurs"
                    color="var(--amber-400)"
                  />
                  <KpiCard
                    label="Observations Testées"
                    value={modelResult.metrics.test_samples}
                    sub="Échantillon d'évaluation étanche"
                    color="var(--green-400)"
                  />
                </>
              )}
            </div>
          </div>

          {/* Graphique de diagnostic : Matrice de Confusion OU Nuage Réel vs Prédit */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(360px, 1fr))', gap: '1.5rem' }}>
            {problemType === 'classification' && modelResult.chart_data?.confusion_matrix && (
              <div className="glass-card">
                <h4 style={{ fontSize: '1rem', fontWeight: 600, color: '#ffffff', marginBottom: '0.75rem' }}>
                  Matrice de Confusion (Évaluation sur le Test Set)
                </h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                  Lignes = Classes Réelles • Colonnes = Prédictions du Modèle
                </p>
                <div className="table-wrapper">
                  <table className="data-table" style={{ textAlign: 'center' }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: 'left' }}>Réel \ Prédit</th>
                        {modelResult.chart_data.classes.map((c) => (
                          <th key={c} style={{ textAlign: 'center' }}>Classe {c}</th>
                        ))}
                        <th style={{ textAlign: 'center' }}>Précision Ligne</th>
                      </tr>
                    </thead>
                    <tbody>
                      {modelResult.chart_data.confusion_matrix.map((row) => (
                        <tr key={row.actual}>
                          <td style={{ textAlign: 'left', fontWeight: 600 }}>Classe {row.actual}</td>
                          {modelResult.chart_data.classes.map((c) => {
                            const val = row.predictions[c] || 0;
                            const isDiagonal = row.actual === c;
                            return (
                              <td
                                key={c}
                                style={{
                                  background: isDiagonal ? 'rgba(45, 212, 191, 0.15)' : (val > 0 ? 'rgba(248, 113, 113, 0.1)' : 'transparent'),
                                  fontWeight: isDiagonal ? 700 : 400,
                                  color: isDiagonal ? 'var(--teal-400)' : 'var(--text-secondary)',
                                }}
                              >
                                {val}
                              </td>
                            );
                          })}
                          <td>
                            <span style={{ fontWeight: 600, color: 'var(--green-400)' }}>
                              {row.accuracy_row_pct}%
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {problemType === 'regression' && modelResult.chart_data?.actual_vs_predicted && (
              <div className="glass-card">
                <h4 style={{ fontSize: '1rem', fontWeight: 600, color: '#ffffff', marginBottom: '0.75rem' }}>
                  Nuage de Points : Valeurs Réelles vs Prédictions
                </h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                  Plus les points sont alignés sur la diagonale, meilleure est la fidélité du modèle.
                </p>
                <div style={{ width: '100%', height: 260 }}>
                  <ResponsiveContainer>
                    <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis type="number" dataKey="actual" name="Réel" stroke="#64748b" fontSize={11} tickLine={false} />
                      <YAxis type="number" dataKey="predicted" name="Prédit" stroke="#64748b" fontSize={11} tickLine={false} />
                      <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                      <Scatter data={modelResult.chart_data.actual_vs_predicted} fill="#2dd4bf" />
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

            {/* Importance des Features */}
            {modelResult.feature_importances && modelResult.feature_importances.length > 0 && (
              <div className="glass-card">
                <h4 style={{ fontSize: '1rem', fontWeight: 600, color: '#ffffff', marginBottom: '0.75rem' }}>
                  Top 10 Variables Prédictives Majeures
                </h4>
                <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
                  Contribution relative dans les décisions du modèle.
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                  {modelResult.feature_importances.map((item, idx) => (
                    <div key={idx} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.82rem' }}>
                      <span style={{ width: 140, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', color: 'var(--text-secondary)' }} title={item.feature}>
                        {item.feature}
                      </span>
                      <div style={{ flex: 1, height: 8, background: 'rgba(255,255,255,0.06)', borderRadius: 4, overflow: 'hidden' }}>
                        <div
                          style={{
                            width: `${item.importance_pct}%`,
                            height: '100%',
                            background: 'linear-gradient(90deg, var(--teal-400), var(--violet-400))',
                            borderRadius: 4,
                          }}
                        />
                      </div>
                      <span style={{ width: 45, textAlign: 'right', fontWeight: 600, color: 'var(--teal-400)' }}>
                        {item.importance_pct}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* --------------------------------------------------------------- */}
          {/* SECTION RÉÉQUILIBRAGE : IMPACT TRAIN & TABLEAU COMPARATIF */}
          {/* --------------------------------------------------------------- */}
          {problemType === 'classification' && modelResult.rebalancing_info && (
            <div className="glass-card" style={{ borderColor: 'rgba(251, 191, 36, 0.3)', background: 'linear-gradient(135deg, rgba(251, 191, 36, 0.04), rgba(45, 212, 191, 0.04))' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <div style={{ width: 38, height: 38, borderRadius: 'var(--radius-sm)', background: 'rgba(251, 191, 36, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                    <Layers size={20} color="var(--amber-400)" />
                  </div>
                  <div>
                    <h3 style={{ fontSize: '1.2rem', color: '#ffffff', fontWeight: 600 }}>
                      Impact du Rééquilibrage & Comparatif des Stratégies
                    </h3>
                    <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                      Stratégie active : <strong style={{ color: 'var(--amber-400)' }}>{modelResult.rebalancing_info.strategy_label || modelResult.rebalancing_info.strategy}</strong>
                      {modelResult.rebalancing_info.k_neighbors_used ? ` (k=${modelResult.rebalancing_info.k_neighbors_used})` : ''}
                    </p>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', background: 'rgba(45,212,191,0.08)', padding: '0.4rem 0.8rem', borderRadius: 'var(--radius-sm)', border: '1px solid rgba(45,212,191,0.2)', fontSize: '0.78rem', color: 'var(--teal-400)' }}>
                  <ShieldCheck size={15} />
                  <span>Jeu de test : <strong>strictement inchangé</strong> ({modelResult.metrics.test_samples} lignes testées)</span>
                </div>
              </div>

              {/* Distribution Avant / Après sur le Train */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
                {/* Avant */}
                <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
                    <span style={{ fontWeight: 600, color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                      Distribution Train — AVANT Rééquilibrage
                    </span>
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      Total : <strong>{modelResult.rebalancing_info.n_before}</strong> lignes
                    </span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {Object.entries(modelResult.rebalancing_info.before || {}).map(([cls, count]) => {
                      const pct = ((count / (modelResult.rebalancing_info.n_before || 1)) * 100).toFixed(1);
                      return (
                        <div key={cls}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '0.2rem' }}>
                            <span>Classe {cls}</span>
                            <span><strong>{count}</strong> ({pct}%)</span>
                          </div>
                          <div style={{ height: 6, background: 'rgba(255,255,255,0.06)', borderRadius: 3, overflow: 'hidden' }}>
                            <div style={{ width: `${pct}%`, height: '100%', background: cls === '1' ? '#f59e0b' : '#0d9488', borderRadius: 3 }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Après */}
                <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem' }}>
                    <span style={{ fontWeight: 600, color: 'var(--teal-400)', fontSize: '0.85rem' }}>
                      Distribution Train — APRÈS Rééquilibrage
                    </span>
                    <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                      Total : <strong style={{ color: 'var(--teal-400)' }}>{modelResult.rebalancing_info.n_after}</strong> lignes
                    </span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    {Object.entries(modelResult.rebalancing_info.after || {}).map(([cls, count]) => {
                      const pct = ((count / (modelResult.rebalancing_info.n_after || 1)) * 100).toFixed(1);
                      return (
                        <div key={cls}>
                          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', marginBottom: '0.2rem' }}>
                            <span>Classe {cls}</span>
                            <span><strong style={{ color: 'var(--teal-400)' }}>{count}</strong> ({pct}%)</span>
                          </div>
                          <div style={{ height: 6, background: 'rgba(255,255,255,0.06)', borderRadius: 3, overflow: 'hidden' }}>
                            <div style={{ width: `${pct}%`, height: '100%', background: cls === '1' ? 'linear-gradient(90deg, #f59e0b, #4ade80)' : 'linear-gradient(90deg, #0d9488, #2dd4bf)', borderRadius: 3 }} />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Tableau comparatif des stratégies testées */}
              {modelResult.rebalancing_comparison && Object.keys(modelResult.rebalancing_comparison).length > 0 && (
                <div>
                  <h4 style={{ fontSize: '0.92rem', fontWeight: 600, color: '#ffffff', marginBottom: '0.6rem' }}>
                    Tableau Comparatif des Stratégies (Évaluation sur le Test Set)
                  </h4>
                  <div className="table-wrapper">
                    <table className="data-table" style={{ fontSize: '0.82rem' }}>
                      <thead>
                        <tr>
                          <th>Stratégie</th>
                          <th style={{ textAlign: 'center' }}>Lignes Train</th>
                          <th style={{ textAlign: 'center' }}>Accuracy Test</th>
                          <th style={{ textAlign: 'center' }}>Précision</th>
                          <th style={{ textAlign: 'center' }}>Rappel (Recall)</th>
                          <th style={{ textAlign: 'center' }}>Score F1</th>
                          <th style={{ textAlign: 'center' }}>AUC-ROC</th>
                          <th style={{ textAlign: 'center' }}>AUC-PR</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.values(modelResult.rebalancing_comparison).map((row) => {
                          const isActive = row.strategy === (modelResult.rebalancing_info?.strategy || 'none');
                          return (
                            <tr key={row.strategy} style={{ background: isActive ? 'rgba(45,212,191,0.08)' : 'transparent' }}>
                              <td>
                                <strong style={{ color: isActive ? 'var(--teal-400)' : 'var(--text-secondary)' }}>
                                  {row.strategy_label || row.strategy}
                                </strong>
                                {isActive && <span style={{ marginLeft: 6, fontSize: '0.7rem', color: 'var(--teal-400)' }}>(actuel)</span>}
                              </td>
                              <td style={{ textAlign: 'center' }}>{row.n_train_after}</td>
                              <td style={{ textAlign: 'center' }}>{(row.metrics?.accuracy_test * 100)?.toFixed(1)}%</td>
                              <td style={{ textAlign: 'center' }}>{(row.metrics?.precision_weighted * 100)?.toFixed(1)}%</td>
                              <td style={{ textAlign: 'center', color: 'var(--blue-400)', fontWeight: 600 }}>{(row.metrics?.recall_weighted * 100)?.toFixed(1)}%</td>
                              <td style={{ textAlign: 'center', color: 'var(--green-400)', fontWeight: 600 }}>{row.metrics?.f1_weighted?.toFixed(3)}</td>
                              <td style={{ textAlign: 'center' }}>{row.metrics?.auc_roc !== null && row.metrics?.auc_roc !== undefined ? row.metrics.auc_roc.toFixed(3) : '—'}</td>
                              <td style={{ textAlign: 'center', color: 'var(--amber-400)', fontWeight: 600 }}>{row.metrics?.auc_pr !== null && row.metrics?.auc_pr !== undefined ? row.metrics.auc_pr.toFixed(3) : '—'}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* --------------------------------------------------------------- */}
          {/* SECTION CYBERSÉCURITÉ ET ROBUSTESSE DU MODÈLE */}
          {/* --------------------------------------------------------------- */}
          <div className="glass-card" style={{ borderColor: 'rgba(45, 212, 191, 0.3)', background: 'linear-gradient(135deg, rgba(8, 12, 20, 0.9), rgba(15, 23, 42, 0.8))' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
                <div style={{ width: 42, height: 42, borderRadius: 'var(--radius-md)', background: 'rgba(45, 212, 191, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <ShieldCheck size={24} color="var(--teal-400)" />
                </div>
                <div>
                  <h3 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.35rem', color: '#ffffff' }}>
                    Audit Cybersécurité & Robustesse du Modèle
                  </h3>
                  <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    Évaluation contre le data poisoning, les attaques adversariales, l'inversion et la falsification.
                  </p>
                </div>
              </div>

              {/* Score Cyber Badge */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', background: 'rgba(255,255,255,0.04)', padding: '0.5rem 1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>Score de Résilience</div>
                  <div style={{ fontSize: '1.4rem', fontWeight: 800, color: modelResult.cyber_audit.color }}>
                    {modelResult.cyber_audit.global_score} / 100
                  </div>
                </div>
                <div style={{ width: 36, height: 36, borderRadius: '50%', background: 'rgba(45,212,191,0.1)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '1.1rem', fontWeight: 800, color: modelResult.cyber_audit.color }}>
                  {modelResult.cyber_audit.grade}
                </div>
              </div>
            </div>

            {/* Grille des 3 axes de vulnérabilité testés */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', marginBottom: '1.5rem' }}>
              {/* 1. Résistance aux perturbations adversariales */}
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <Zap size={16} color="var(--amber-400)" />
                  <span style={{ fontWeight: 600, fontSize: '0.9rem', color: '#f8fafc' }}>
                    Robustesse Adversariale (Bruit ε)
                  </span>
                </div>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                  {modelResult.cyber_audit.adversarial_robustness.summary}
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                  {modelResult.cyber_audit.adversarial_robustness.perturbation_decay.map((p, pIdx) => (
                    <div key={pIdx} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', padding: '0.25rem 0.5rem', background: 'rgba(255,255,255,0.03)', borderRadius: 4 }}>
                      <span style={{ color: 'var(--text-muted)' }}>{p.epsilon}</span>
                      <span style={{ color: p.performance_drop_pct < 10 ? 'var(--green-400)' : 'var(--amber-400)', fontWeight: 600 }}>
                        Chute : -{p.performance_drop_pct}% ({p.status})
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {/* 2. Empoisonnement de données */}
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <ShieldAlert size={16} color="var(--teal-400)" />
                  <span style={{ fontWeight: 600, fontSize: '0.9rem', color: '#f8fafc' }}>
                    Risque d'Empoisonnement (Poisoning)
                  </span>
                </div>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                  {modelResult.cyber_audit.data_poisoning.detail}
                </p>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', padding: '0.4rem 0.6rem', background: 'rgba(45,212,191,0.08)', borderRadius: 4 }}>
                  <span>Niveau de Risque :</span>
                  <strong style={{ color: 'var(--teal-400)' }}>{modelResult.cyber_audit.data_poisoning.risk_label}</strong>
                </div>
              </div>

              {/* 3. Inférence d'appartenance / Fuite de confidentialité */}
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                  <Lock size={16} color="var(--violet-400)" />
                  <span style={{ fontWeight: 600, fontSize: '0.9rem', color: '#f8fafc' }}>
                    Fuite de Confidentialité (Membership Inference)
                  </span>
                </div>
                <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                  {modelResult.cyber_audit.membership_inference.detail}
                </p>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', padding: '0.4rem 0.6rem', background: 'rgba(167,139,250,0.08)', borderRadius: 4 }}>
                  <span>Écart Généralisation :</span>
                  <strong style={{ color: 'var(--violet-400)' }}>
                    {modelResult.cyber_audit.membership_inference.generalization_gap_pct}%
                  </strong>
                </div>
              </div>
            </div>

            {/* Empreinte cryptographique & Signature HMAC */}
            <div style={{ background: 'rgba(0,0,0,0.4)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.6rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem', fontWeight: 600, color: 'var(--teal-400)' }}>
                  <KeyRound size={15} />
                  <span>Empreinte Cryptographique d'Intégrité (Anti-Tampering)</span>
                </div>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  Taille binaire sérialisée : {modelResult.cyber_audit.model_size_kb} KB
                </span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem', fontSize: '0.78rem', fontFamily: 'monospace' }}>
                <div style={{ display: 'flex', gap: '0.5rem', color: 'var(--text-secondary)' }}>
                  <span style={{ color: 'var(--text-muted)', width: 80 }}>SHA-256 :</span>
                  <span style={{ wordBreak: 'break-all', color: 'var(--teal-400)' }}>{modelResult.cyber_audit.sha256}</span>
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', color: 'var(--text-secondary)' }}>
                  <span style={{ color: 'var(--text-muted)', width: 80 }}>HMAC :</span>
                  <span style={{ wordBreak: 'break-all', color: 'var(--violet-400)' }}>{modelResult.cyber_audit.hmac}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
