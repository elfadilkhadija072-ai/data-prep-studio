import React, { useState, useEffect } from 'react';
import {
  Terminal,
  Activity,
  Cpu,
  Database,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  X,
  ShieldCheck,
  Zap,
  HardDrive,
  Sliders,
} from 'lucide-react';
import { getDiagnostics, getDatasetSecurityAudit } from '../api';

export function DebuggerModal({ isOpen, onClose, sessionId }) {
  const [loading, setLoading] = useState(false);
  const [diag, setDiag] = useState(null);
  const [securityAudit, setSecurityAudit] = useState(null);
  const [activeSubTab, setActiveSubTab] = useState('system'); // 'system' | 'security'

  const refreshData = async () => {
    if (!sessionId) return;
    setLoading(true);
    try {
      const [dData, sData] = await Promise.all([
        getDiagnostics(sessionId).catch(() => null),
        getDatasetSecurityAudit(sessionId).catch(() => null),
      ]);
      setDiag(dData);
      setSecurityAudit(sData);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen && sessionId) {
      refreshData();
    }
  }, [isOpen, sessionId]);

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="glass-card"
        style={{
          width: '90%',
          maxWidth: '750px',
          maxHeight: '85vh',
          overflowY: 'auto',
          background: 'rgba(10, 15, 26, 0.95)',
          borderColor: 'rgba(45, 212, 191, 0.3)',
          boxShadow: '0 20px 50px rgba(0,0,0,0.8), 0 0 30px rgba(45, 212, 191, 0.15)',
          padding: '1.75rem',
        }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* En-tête du Débogueur */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.85rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
            <div style={{ width: 36, height: 36, borderRadius: 'var(--radius-sm)', background: 'rgba(45, 212, 191, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <Terminal size={20} color="var(--teal-400)" />
            </div>
            <div>
              <h3 style={{ fontSize: '1.2rem', color: '#ffffff', fontWeight: 600 }}>
                Console de Débogage & Diagnostic Système
              </h3>
              <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                Surveillance de la mémoire, intégrité NaN, métriques de pipeline et audit d'injection
              </p>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <button
              className="btn btn-secondary btn-sm"
              onClick={refreshData}
              disabled={loading}
              title="Actualiser les métriques"
            >
              <RefreshCw size={13} className={loading ? 'spin-animation' : ''} />
              Actualiser
            </button>
            <button
              className="btn btn-ghost btn-sm"
              onClick={onClose}
              style={{ padding: '0.35rem' }}
            >
              <X size={18} />
            </button>
          </div>
        </div>

        {/* Sous-navigation */}
        <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.25rem' }}>
          <button
            className={`btn btn-sm ${activeSubTab === 'system' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setActiveSubTab('system')}
          >
            <Activity size={14} />
            Diagnostic Système & Pipeline
          </button>
          <button
            className={`btn btn-sm ${activeSubTab === 'security' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => setActiveSubTab('security')}
          >
            <ShieldCheck size={14} />
            Audit Anti-Injection Dataset
          </button>
        </div>

        {/* Contenu : Diagnostic Système */}
        {activeSubTab === 'system' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {diag ? (
              <>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '0.85rem' }}>
                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.85rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.3rem' }}>
                      <HardDrive size={13} color="var(--teal-400)" />
                      Empreinte Mémoire RAM (Brut)
                    </div>
                    <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>
                      {diag.raw_memory_mb} MB
                    </div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>
                      {diag.raw_shape?.[0]?.toLocaleString()} lignes × {diag.raw_shape?.[1]} colonnes
                    </div>
                  </div>

                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.85rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.3rem' }}>
                      <HardDrive size={13} color="var(--violet-400)" />
                      Empreinte Mémoire (Préparé)
                    </div>
                    <div style={{ fontSize: '1.25rem', fontWeight: 700, color: '#f8fafc' }}>
                      {diag.prep_memory_mb || 0} MB
                    </div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>
                      {diag.has_executed ? `${diag.prep_shape?.[0]?.toLocaleString()} lig. × ${diag.prep_shape?.[1]} col.` : 'En attente d’exécution'}
                    </div>
                  </div>

                  <div style={{ background: 'rgba(0,0,0,0.3)', padding: '0.85rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.3rem' }}>
                      <CheckCircle2 size={13} color="var(--green-400)" />
                      Tolérance Valeurs NaN Résiduelles
                    </div>
                    <div style={{ fontSize: '1.25rem', fontWeight: 700, color: diag.prep_nan_count === 0 ? 'var(--green-400)' : 'var(--red-400)' }}>
                      {diag.has_executed ? `${diag.prep_nan_count} NaN résiduels` : `${diag.raw_nan_count} NaN initiaux`}
                    </div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-dim)' }}>
                      {diag.prep_nan_count === 0 ? 'Zero-NaN Conforme' : 'Imputation requise'}
                    </div>
                  </div>
                </div>

                {/* État des modules actifs */}
                <div style={{ background: 'rgba(0,0,0,0.25)', padding: '1rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                  <h4 style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                    Statut des Modules du Pipeline en Mémoire
                  </h4>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.8rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0.6rem', background: 'rgba(255,255,255,0.02)', borderRadius: 4 }}>
                      <span>Session ID Actuelle</span>
                      <strong style={{ fontFamily: 'monospace', color: 'var(--teal-400)' }}>{diag.session_id}</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0.6rem', background: 'rgba(255,255,255,0.02)', borderRadius: 4 }}>
                      <span>Type d'apprentissage</span>
                      <strong style={{ color: 'var(--violet-400)' }}>{diag.problem_type}</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0.6rem', background: 'rgba(255,255,255,0.02)', borderRadius: 4 }}>
                      <span>Variable Cible Définie</span>
                      <strong style={{ color: 'var(--teal-400)' }}>{diag.target_col || 'Aucune (Non supervisé)'}</strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0.6rem', background: 'rgba(255,255,255,0.02)', borderRadius: 4 }}>
                      <span>Pipeline Exécuté & Split Étanche</span>
                      <strong style={{ color: diag.has_executed ? 'var(--green-400)' : 'var(--amber-400)' }}>
                        {diag.has_executed ? 'Validé (Train/Test générés)' : 'Non exécuté'}
                      </strong>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '0.4rem 0.6rem', background: 'rgba(255,255,255,0.02)', borderRadius: 4 }}>
                      <span>Modèle Machine Learning Ajusté</span>
                      <strong style={{ color: diag.has_trained_model ? 'var(--green-400)' : 'var(--text-muted)' }}>
                        {diag.has_trained_model ? 'Oui (Prêt à l’emploi & Certifié)' : 'Non entraîné'}
                      </strong>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '2rem' }}>
                Chargement des diagnostics…
              </p>
            )}
          </div>
        )}

        {/* Contenu : Audit Anti-Injection */}
        {activeSubTab === 'security' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {securityAudit ? (
              <>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.85rem 1rem', background: securityAudit.threats_count === 0 ? 'rgba(74, 222, 128, 0.08)' : 'rgba(248, 113, 113, 0.08)', borderRadius: 'var(--radius-md)', border: `1px solid ${securityAudit.threats_count === 0 ? 'rgba(74, 222, 128, 0.25)' : 'rgba(248, 113, 113, 0.25)'}` }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    {securityAudit.threats_count === 0 ? (
                      <CheckCircle2 size={20} color="var(--green-400)" />
                    ) : (
                      <AlertTriangle size={20} color="var(--red-400)" />
                    )}
                    <div>
                      <div style={{ fontWeight: 600, fontSize: '0.9rem', color: '#f8fafc' }}>
                        {securityAudit.recommendation}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {securityAudit.scanned_columns_count} colonnes textuelles scannées
                      </div>
                    </div>
                  </div>
                  <div style={{ fontSize: '1.2rem', fontWeight: 800, color: securityAudit.threats_count === 0 ? 'var(--green-400)' : 'var(--red-400)' }}>
                    {securityAudit.sanitization_score} / 100
                  </div>
                </div>

                {securityAudit.threats_count > 0 ? (
                  <div className="table-wrapper">
                    <table className="data-table">
                      <thead>
                        <tr>
                          <th>Colonne</th>
                          <th>Menace</th>
                          <th>Extrait Détecté</th>
                          <th>Gravité</th>
                        </tr>
                      </thead>
                      <tbody>
                        {securityAudit.threats.map((t, idx) => (
                          <tr key={idx}>
                            <td>{t.column}</td>
                            <td style={{ color: 'var(--red-400)', fontWeight: 600 }}>{t.threat_type}</td>
                            <td><code>{t.sample_snippet}</code></td>
                            <td><span className="badge-critique">CRITIQUE</span></td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    Aucune menace XSS, SQLi ou SSTI n'a été trouvée dans les colonnes textuelles de ce dataset.
                  </div>
                )}
              </>
            ) : (
              <p style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '2rem' }}>
                Analyse de cybersécurité en cours…
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
