import React, { useState } from 'react';
import { UploadCloud, FileSpreadsheet, ShieldAlert, Home, ArrowRight, CheckCircle2 } from 'lucide-react';
import { uploadDataset, configureSession } from '../api';
import { TypeBadge } from '../ui';

export function ImportTab({ sessionData, onDatasetLoaded, onProceedToEda }) {
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Configuration supervisée
  const [isSupervised, setIsSupervised] = useState(true);
  const [targetCol, setTargetCol] = useState('');
  const [problemType, setProblemType] = useState('classification');
  const [testSize, setTestSize] = useState(0.2);
  const [dropDuplicates, setDropDuplicates] = useState(true);

  // Synchroniser avec sessionData au chargement
  React.useEffect(() => {
    if (sessionData) {
      setIsSupervised(sessionData.is_supervised !== false);
      setTargetCol(sessionData.target_col || sessionData.suggested_target || sessionData.columns?.[sessionData.columns.length - 1] || '');
      setProblemType(sessionData.problem_type || sessionData.suggested_problem || 'classification');
      setTestSize(sessionData.test_size || 0.2);
      setDropDuplicates(sessionData.drop_duplicates !== false);
    }
  }, [sessionData]);

  const handleFileUpload = async (file) => {
    setLoading(true);
    setError(null);
    try {
      const data = await uploadDataset(file);
      onDatasetLoaded(data);
      if (data.suggested_target) setTargetCol(data.suggested_target);
      if (data.suggested_problem) setProblemType(data.suggested_problem);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDemoLoad = async (demoKey) => {
    setLoading(true);
    setError(null);
    try {
      const data = await uploadDataset(null, demoKey);
      onDatasetLoaded(data);
      if (data.suggested_target) setTargetCol(data.suggested_target);
      if (data.suggested_problem) setProblemType(data.suggested_problem);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleSaveAndProceed = async () => {
    if (!sessionData) return;
    setLoading(true);
    try {
      await configureSession({
        session_id: sessionData.session_id,
        is_supervised: isSupervised,
        target_col: isSupervised ? targetCol : null,
        problem_type: isSupervised ? problemType : 'unsupervised',
        test_size: parseFloat(testSize),
        random_state: 42,
        drop_duplicates: dropDuplicates,
      });
      onProceedToEda();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* En-tête */}
      <div>
        <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.85rem', marginBottom: '0.4rem' }}>
          1. Importation & Configuration du Dataset
        </h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
          Chargez votre propre fichier CSV / Excel ou explorez un jeu de démonstration pour démarrer l'analyse.
        </p>
      </div>

      {error && (
        <div style={{ padding: '1rem', background: 'var(--red-bg)', color: 'var(--red-400)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(248,113,113,0.3)', fontSize: '0.9rem' }}>
          {error}
        </div>
      )}

      {/* Zone de glisser-déposer */}
      <div
        className={`dropzone-container ${isDragging ? 'drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => document.getElementById('file-input-hidden')?.click()}
      >
        <input
          id="file-input-hidden"
          type="file"
          accept=".csv,.tsv,.txt,.xlsx,.xls"
          style={{ display: 'none' }}
          onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
        />
        <div className="dropzone-icon">
          <UploadCloud size={32} />
        </div>
        <div>
          <p style={{ fontWeight: 600, fontSize: '1.05rem', color: '#f8fafc', marginBottom: '0.3rem' }}>
            Glissez-déposez votre fichier ici, ou cliquez pour parcourir
          </p>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
            Formats supportés : CSV (détection automatique séparateur & encodage) et Excel (.xlsx, .xls)
          </p>
        </div>
        {loading && <p style={{ color: 'var(--teal-400)', fontSize: '0.85rem' }}>Chargement et conversion des données en cours…</p>}
      </div>

      {/* Raccourcis Jeux de Démonstration */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)', fontWeight: 500 }}>
          Ou chargez un jeu de test pré-configuré :
        </span>
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => handleDemoLoad('fraud')}
          disabled={loading}
        >
          <ShieldAlert size={14} color="#f87171" />
          Démo 1 : Fraude Bancaire (Classification)
        </button>
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => handleDemoLoad('housing')}
          disabled={loading}
        >
          <Home size={14} color="#60a5fa" />
          Démo 2 : Prix Immobilier (Régression)
        </button>
      </div>

      {/* Section Aperçu & Configuration supervisée si dataset chargé */}
      {sessionData && (
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
            <div>
              <h3 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.25rem', color: '#ffffff' }}>
                Fichier chargé : <span style={{ color: 'var(--teal-400)' }}>{sessionData.filename}</span>
              </h3>
              <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                {sessionData.load_info?.rows?.toLocaleString('fr-FR')} lignes × {sessionData.load_info?.cols} colonnes | Format : {sessionData.load_info?.format} ({sessionData.load_info?.encoding})
              </p>
            </div>
            <span className="brand-badge" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <CheckCircle2 size={12} />
              Données Validées
            </span>
          </div>

          {/* Formulaire de configuration supervisée */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(230px, 1fr))', gap: '1.25rem', padding: '1.25rem', background: 'rgba(0,0,0,0.25)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
            <div className="control-group">
              <label className="control-label">Type d'apprentissage</label>
              <select
                className="form-select"
                value={isSupervised ? 'supervised' : 'unsupervised'}
                onChange={(e) => setIsSupervised(e.target.value === 'supervised')}
              >
                <option value="supervised">Supervisé (avec variable cible)</option>
                <option value="unsupervised">Non supervisé (clustering / sans cible)</option>
              </select>
            </div>

            {isSupervised && (
              <>
                <div className="control-group">
                  <label className="control-label">Variable cible (y)</label>
                  <select
                    className="form-select"
                    value={targetCol}
                    onChange={(e) => setTargetCol(e.target.value)}
                  >
                    {sessionData.columns?.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>

                <div className="control-group">
                  <label className="control-label">Tâche supervisée</label>
                  <select
                    className="form-select"
                    value={problemType}
                    onChange={(e) => setProblemType(e.target.value)}
                  >
                    <option value="classification">Classification (catégories / binaire)</option>
                    <option value="regression">Régression (valeur continue)</option>
                  </select>
                </div>

                <div className="control-group">
                  <label className="control-label">Taille du jeu de test : {Math.round(testSize * 100)}%</label>
                  <input
                    type="range"
                    min="0.10"
                    max="0.40"
                    step="0.05"
                    value={testSize}
                    onChange={(e) => setTestSize(parseFloat(e.target.value))}
                    style={{ accentColor: 'var(--teal-400)', marginTop: '0.4rem' }}
                  />
                </div>
              </>
            )}

            <div className="control-group" style={{ justifyContent: 'center' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', cursor: 'pointer', fontSize: '0.82rem', marginTop: '1rem' }}>
                <input
                  type="checkbox"
                  checked={dropDuplicates}
                  onChange={(e) => setDropDuplicates(e.target.checked)}
                  style={{ accentColor: 'var(--teal-400)', width: 16, height: 16 }}
                />
                Supprimer les doublons stricts avant le split
              </label>
            </div>
          </div>

          {/* Aperçu des 10 premières lignes */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
              <span style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
                Aperçu des 10 premières observations :
              </span>
            </div>
            <div className="table-wrapper">
              <table className="data-table">
                <thead>
                  <tr>
                    {sessionData.columns?.map((c) => (
                      <th key={c}>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                          <span>{c}</span>
                          <TypeBadge type={sessionData.types?.[c] || 'categorical'} />
                        </div>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sessionData.preview_rows?.map((row, rIdx) => (
                    <tr key={rIdx}>
                      {sessionData.columns?.map((c) => (
                        <td key={c}>
                          {row[c] !== null && row[c] !== undefined && row[c] !== '' ? String(row[c]) : (
                            <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>NA</span>
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Bouton d'action pour passer à l'étape 2 */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', paddingTop: '0.5rem' }}>
            <button
              className="btn btn-primary"
              onClick={handleSaveAndProceed}
              disabled={loading}
            >
              Valider la configuration et passer à l'Analyse Exploratoire (EDA)
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
