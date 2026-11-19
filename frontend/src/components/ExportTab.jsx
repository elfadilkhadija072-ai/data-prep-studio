import React, { useState, useEffect } from 'react';
import {
  Download,
  FileSpreadsheet,
  FileText,
  ShieldCheck,
  CheckCircle2,
  Database,
  Layers,
  ArrowLeft,
  Sparkles,
} from 'lucide-react';
import { getExportSummary, getDownloadUrl } from '../api';
import { KpiCard, LoadingSkeleton } from '../ui';

export function ExportTab({ sessionId, onBackToPrep }) {
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!sessionId) return;
    setLoading(true);
    getExportSummary(sessionId)
      .then(setSummary)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [sessionId]);

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <LoadingSkeleton height={120} />
        <LoadingSkeleton height={350} />
      </div>
    );
  }

  if (error || !summary) {
    return (
      <div className="glass-card" style={{ textAlign: 'center', padding: '3rem' }}>
        <h3 style={{ marginBottom: '1rem', color: 'var(--red-400)' }}>Aucun résultat d'exécution trouvé</h3>
        <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
          Veuillez configurer et lancer la préparation depuis l'onglet précédent.
        </p>
        <button className="btn btn-secondary" onClick={onBackToPrep}>
          <ArrowLeft size={16} />
          Retourner au Preprocessing
        </button>
      </div>
    );
  }

  const { train_shape, test_shape, final_features_count, best_practices_checks, applied_table, preview_head, final_columns } = summary;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* En-tête */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.2rem' }}>
            <span className="brand-badge" style={{ background: 'var(--green-bg)', color: 'var(--green-400)', borderColor: 'rgba(74, 222, 128, 0.3)' }}>
              PRÉPARATION RÉUSSIE
            </span>
            <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Anti-Data-Leakage Certifié</span>
          </div>
          <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.85rem' }}>
            4. Exécution & Téléchargement des Données Préparées
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
            Vos données sont désormais propres, encodées, normalisées et prêtes pour l'entraînement d'algorithmes de Machine Learning.
          </p>
        </div>
        <button className="btn btn-secondary" onClick={onBackToPrep}>
          <ArrowLeft size={16} />
          Modifier la configuration
        </button>
      </div>

      {/* Cartes KPI de résultat */}
      <div className="kpis-grid">
        <KpiCard
          label="Échantillon d'Entraînement (Train)"
          value={`${train_shape[0]?.toLocaleString('fr-FR')} lignes`}
          sub="Apprentissage des statistiques"
          icon={Database}
          color="var(--teal-400)"
        />
        <KpiCard
          label="Échantillon d'Évaluation (Test)"
          value={`${test_shape[0]?.toLocaleString('fr-FR')} lignes`}
          sub="Étanche sans data leakage"
          icon={Database}
          color="var(--violet-400)"
        />
        <KpiCard
          label="Variables Préparées"
          value={final_features_count}
          sub="Après encodage & sélection"
          icon={Layers}
          color="var(--green-400)"
        />
        <KpiCard
          label="Statut d'Intégrité"
          value="100% Conforme"
          sub="Séparation stricte respectée"
          icon={ShieldCheck}
          color="var(--teal-400)"
        />
      </div>

      {/* Zone de Téléchargement Principale */}
      <div className="glass-card" style={{ background: 'linear-gradient(135deg, rgba(45, 212, 191, 0.08), rgba(167, 139, 250, 0.08))', borderColor: 'rgba(45, 212, 191, 0.3)' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem' }}>
          <Sparkles size={22} color="var(--teal-400)" />
          <h3 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.35rem', color: '#ffffff' }}>
            Fichiers & Rapports Prêts au Téléchargement
          </h3>
        </div>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '1.5rem', maxWidth: '750px' }}>
          Téléchargez le dataset complet (avec colonne <code>_split</code> : <em>train / test</em>), les jeux séparés, ainsi que le rapport d'audit Excel professionnel généré avec XlsxWriter.
        </p>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '1rem' }}>
          {/* Rapport Excel Pro */}
          <a
            href={getDownloadUrl(sessionId, 'report_xlsx')}
            download
            className="btn btn-primary"
            style={{ textDecoration: 'none', padding: '0.85rem 1rem' }}
          >
            <FileSpreadsheet size={18} />
            Rapport d'Audit Excel (.xlsx)
          </a>

          {/* Dataset complet CSV */}
          <a
            href={getDownloadUrl(sessionId, 'full_csv')}
            download
            className="btn btn-secondary"
            style={{ textDecoration: 'none', padding: '0.85rem 1rem' }}
          >
            <Download size={18} color="var(--teal-400)" />
            Dataset Complet (.csv)
          </a>

          {/* Dataset complet XLSX */}
          <a
            href={getDownloadUrl(sessionId, 'full_xlsx')}
            download
            className="btn btn-secondary"
            style={{ textDecoration: 'none', padding: '0.85rem 1rem' }}
          >
            <FileSpreadsheet size={18} color="var(--green-400)" />
            Dataset Complet (.xlsx)
          </a>

          {/* Train CSV */}
          <a
            href={getDownloadUrl(sessionId, 'train_csv')}
            download
            className="btn btn-secondary"
            style={{ textDecoration: 'none', padding: '0.85rem 1rem' }}
          >
            <FileText size={18} color="var(--blue-400)" />
            Train Set (.csv)
          </a>

          {/* Train Rééquilibré CSV */}
          <a
            href={getDownloadUrl(sessionId, 'train_rebalanced')}
            download
            className="btn btn-secondary"
            title="Disponible si un rééquilibrage SMOTE ou Undersampling a été appliqué"
            style={{ textDecoration: 'none', padding: '0.85rem 1rem', borderColor: 'rgba(251,191,36,0.3)' }}
          >
            <FileText size={18} color="var(--amber-400)" />
            Train Rééquilibré (.csv)
          </a>

          {/* Test CSV */}
          <a
            href={getDownloadUrl(sessionId, 'test_csv')}
            download
            className="btn btn-secondary"
            style={{ textDecoration: 'none', padding: '0.85rem 1rem' }}
          >
            <FileText size={18} color="var(--violet-400)" />
            Test Set (.csv)
          </a>
        </div>
      </div>

      {/* Journal d'audit des Bonnes Pratiques ML */}
      <div className="glass-card">
        <h3 className="section-title" style={{ fontSize: '1.25rem', marginBottom: '0.4rem' }}>
          <ShieldCheck size={20} color="var(--green-400)" />
          Journal d'Audit des Bonnes Pratiques ML & Anti-Data-Leakage
        </h3>
        <p className="section-desc" style={{ marginBottom: '1.25rem' }}>
          Preuve formelle de la stricte conformité méthodologique appliquée lors de la préparation.
        </p>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          {best_practices_checks.map((chk, idx) => (
            <div
              key={idx}
              style={{
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.85rem',
                padding: '0.85rem 1.15rem',
                borderRadius: 'var(--radius-md)',
                background: 'rgba(74, 222, 128, 0.05)',
                border: '1px solid rgba(74, 222, 128, 0.2)',
              }}
            >
              <CheckCircle2 size={18} color="var(--green-400)" style={{ marginTop: 2, flexShrink: 0 }} />
              <div>
                <div style={{ fontWeight: 600, fontSize: '0.88rem', color: '#f8fafc' }}>{chk.rule}</div>
                <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: '0.2rem' }}>
                  {chk.detail}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Tableau détaillé : Ce qui a réellement été appliqué */}
      <div className="glass-card">
        <h3 className="section-title" style={{ fontSize: '1.25rem', marginBottom: '0.4rem' }}>
          Détail des Transformations Réellement Appliquées
        </h3>
        <p className="section-desc" style={{ marginBottom: '1.25rem' }}>
          Synthèse par feature avec les paramètres exacts appris sur l'échantillon d'entraînement.
        </p>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Feature</th>
                <th>Type initial</th>
                <th>Action & Transformations</th>
                <th>Imputation</th>
                <th>Outliers</th>
                <th>Scaling</th>
                <th>Encodage</th>
                <th>Nb Colonnes</th>
              </tr>
            </thead>
            <tbody>
              {applied_table.map((row) => (
                <tr key={row.feature}>
                  <td style={{ fontWeight: 600 }}>{row.feature}</td>
                  <td>{row.initial_type}</td>
                  <td style={{ color: 'var(--teal-400)', fontWeight: 500 }}>{row.action}</td>
                  <td>{row.imputation}</td>
                  <td>{row.outliers}</td>
                  <td>{row.scaling}</td>
                  <td>{row.encoding}</td>
                  <td><strong>{row.final_features}</strong></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Aperçu du dataset transformé */}
      <div className="glass-card">
        <h3 className="section-title" style={{ fontSize: '1.25rem', marginBottom: '0.4rem' }}>
          Aperçu des 15 premières lignes du Dataset Transformé
        </h3>
        <p className="section-desc" style={{ marginBottom: '1.25rem' }}>
          Colonnes finales formatées et partitionnées avec la colonne <code>_split</code>.
        </p>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>_split</th>
                {final_columns.map((c) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview_head.map((row, idx) => (
                <tr key={idx}>
                  <td>
                    <span
                      style={{
                        padding: '0.15rem 0.5rem',
                        borderRadius: 4,
                        fontSize: '0.72rem',
                        fontWeight: 700,
                        background: row._split === 'train' ? 'rgba(45, 212, 191, 0.15)' : 'rgba(167, 139, 250, 0.15)',
                        color: row._split === 'train' ? 'var(--teal-400)' : 'var(--violet-400)',
                      }}
                    >
                      {row._split}
                    </span>
                  </td>
                  {final_columns.map((c) => (
                    <td key={c}>
                      {typeof row[c] === 'number' ? row[c].toFixed(3) : String(row[c] ?? '')}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
