import React, { useState, useEffect } from 'react';
import {
  Layers,
  Database,
  AlertCircle,
  Copy,
  Cpu,
  BarChart3,
  TrendingUp,
  ScatterChart as ScatterIcon,
  Search,
  ArrowRight,
  ZoomIn,
} from 'lucide-react';
import {
  getKpis,
  getFeaturesTable,
  getColumnDistribution,
  getCorrelations,
  getScatterData,
  getQualityReport,
} from '../api';
import { KpiCard, TypeBadge, LoadingSkeleton } from '../ui';
import {
  HistogramChart,
  CategoryBarChart,
  BoxplotVisual,
  CorrelationHeatmap,
  ScatterPlotView,
} from '../charts';

export function EdaTab({ sessionId, sessionData, onProceedToPrep }) {
  const [loading, setLoading] = useState(true);
  const [kpis, setKpis] = useState(null);
  const [features, setFeatures] = useState([]);
  const [quality, setQuality] = useState(null);
  const [correlations, setCorrelations] = useState(null);
  const [corrMethod, setCorrMethod] = useState('pearson');

  // Tri du tableau des features
  const [sortField, setSortField] = useState('missing_pct');
  const [sortAsc, setSortAsc] = useState(false);
  const [filterText, setFilterText] = useState('');

  // Feature active pour distribution
  const [selectedCol, setSelectedCol] = useState(null);
  const [distData, setDistData] = useState(null);
  const [zoomP99, setZoomP99] = useState(false);
  const [logScale, setLogScale] = useState(false);

  // Scatter Plot
  const [scatterX, setScatterX] = useState('');
  const [scatterY, setScatterY] = useState('');
  const [scatterPoints, setScatterPoints] = useState([]);

  // Chargement initial des données EDA
  useEffect(() => {
    if (!sessionId) return;
    setLoading(true);
    Promise.all([
      getKpis(sessionId),
      getFeaturesTable(sessionId),
      getQualityReport(sessionId),
      getCorrelations(sessionId, corrMethod),
    ])
      .then(([kpiData, featData, qualData, corrData]) => {
        setKpis(kpiData);
        setFeatures(featData);
        setQuality(qualData);
        setCorrelations(corrData);

        // Sélectionner la première feature numérique ou catégorielle par défaut
        if (featData && featData.length > 0) {
          const firstNum = featData.find((f) => f.type === 'numeric') || featData[0];
          setSelectedCol(firstNum.name);
        }

        // Scatter plot colonnes initiales
        if (corrData?.columns && corrData.columns.length >= 2) {
          setScatterX(corrData.columns[0]);
          setScatterY(corrData.columns[1]);
        }
      })
      .catch((err) => console.error('Erreur EDA:', err))
      .finally(() => setLoading(false));
  }, [sessionId]);

  // Recharger corrélation si méthode changée
  useEffect(() => {
    if (!sessionId) return;
    getCorrelations(sessionId, corrMethod).then(setCorrelations).catch(console.error);
  }, [sessionId, corrMethod]);

  // Recharger distribution si colonne changée
  useEffect(() => {
    if (!sessionId || !selectedCol) return;
    getColumnDistribution(sessionId, selectedCol, zoomP99, logScale)
      .then(setDistData)
      .catch(console.error);
  }, [sessionId, selectedCol, zoomP99, logScale]);

  // Recharger scatter si x ou y changés
  useEffect(() => {
    if (!sessionId || !scatterX || !scatterY) return;
    getScatterData(sessionId, scatterX, scatterY)
      .then(setScatterPoints)
      .catch(console.error);
  }, [sessionId, scatterX, scatterY]);

  // Tri du tableau des features
  const sortedFeatures = [...features]
    .filter((f) => f.name.toLowerCase().includes(filterText.toLowerCase()))
    .sort((a, b) => {
      let va = a[sortField];
      let vb = b[sortField];
      if (va === null || va === undefined) return 1;
      if (vb === null || vb === undefined) return -1;
      if (typeof va === 'string') {
        return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      return sortAsc ? va - vb : vb - va;
    });

  const handleSort = (field) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(false);
    }
  };

  if (loading) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
        <LoadingSkeleton height={100} />
        <LoadingSkeleton height={350} />
        <LoadingSkeleton height={300} />
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
      {/* En-tête */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.85rem', marginBottom: '0.4rem' }}>
            2. Analyse Exploratoire des Données (EDA)
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
            Diagnostic complet des distributions, anomalies statistiques, corrélations et associations avec la cible.
          </p>
        </div>
        <button className="btn btn-primary" onClick={onProceedToPrep}>
          Continuer vers le Preprocessing & Nettoyage
          <ArrowRight size={16} />
        </button>
      </div>

      {/* Cartes KPI */}
      {kpis && (
        <div className="kpis-grid">
          <KpiCard label="Observations" value={kpis.rows?.toLocaleString('fr-FR')} icon={Database} />
          <KpiCard label="Variables (Features)" value={kpis.cols} icon={Layers} />
          <KpiCard
            label="Valeurs Manquantes"
            value={`${kpis.missing_pct}%`}
            sub={`${kpis.missing_count?.toLocaleString('fr-FR')} cellules vides`}
            icon={AlertCircle}
            color={kpis.missing_pct > 10 ? 'var(--red-400)' : 'var(--teal-400)'}
          />
          <KpiCard
            label="Lignes Dupliquées"
            value={kpis.duplicates_count?.toLocaleString('fr-FR')}
            icon={Copy}
            color={kpis.duplicates_count > 0 ? 'var(--amber-400)' : 'var(--teal-400)'}
          />
          <KpiCard label="Empreinte Mémoire" value={`${kpis.memory_mb} Mo`} icon={Cpu} />
          {kpis.target_info?.type === 'classification' && (
            <KpiCard
              label="Équilibre Cible"
              value={kpis.target_info.is_imbalanced ? 'Déséquilibrée' : 'Équilibrée'}
              sub={`Ratio maj/min : ${kpis.target_info.imbalance_ratio}:1 (${kpis.target_info.minority_pct}% min)`}
              color={kpis.target_info.is_imbalanced ? 'var(--amber-400)' : 'var(--green-400)'}
            />
          )}
        </div>
      )}

      {/* Détail Équilibre Cible & Gestion des valeurs manquantes */}
      {kpis?.target_info?.type === 'classification' && (
        <div className="glass-card" style={{ marginBottom: '1.5rem', background: 'rgba(0,0,0,0.25)', border: '1px solid rgba(45,212,191,0.2)' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem', marginBottom: '1rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <BarChart3 size={18} color="var(--teal-400)" />
              <h4 style={{ fontSize: '1rem', fontWeight: 600, color: '#ffffff' }}>
                Répartition des Classes de la Cible ({sessionData?.target_col || 'is_fraud'})
              </h4>
            </div>
            <div style={{ display: 'flex', gap: '0.75rem', fontSize: '0.8rem', flexWrap: 'wrap' }}>
              <span style={{ color: 'var(--text-muted)' }}>
                Ratio de déséquilibre (majoritaire/minoritaire) : <strong style={{ color: 'var(--amber-400)' }}>{kpis.target_info.imbalance_ratio} : 1</strong>
              </span>
              <span style={{ color: 'var(--text-muted)' }}>
                Total étiqueté : <strong style={{ color: '#ffffff' }}>{kpis.target_info.total_labeled}</strong>
              </span>
            </div>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem', alignItems: 'center' }}>
            {/* Barres par classe */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              {Object.entries(kpis.target_info.classes || {}).map(([cls, count]) => {
                const pct = kpis.target_info.class_percentages?.[cls] ?? ((count / (kpis.target_info.total_labeled || 1)) * 100).toFixed(1);
                const isMin = String(cls) === String(kpis.target_info.minority_class);
                return (
                  <div key={cls} style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem' }}>
                      <span style={{ color: isMin ? 'var(--amber-400)' : 'var(--text-secondary)', fontWeight: 600 }}>
                        Classe {cls} {isMin ? '(Minoritaire)' : '(Majoritaire)'}
                      </span>
                      <span>
                        <strong style={{ color: isMin ? 'var(--amber-400)' : 'var(--teal-400)' }}>{count}</strong> ({pct}%)
                      </span>
                    </div>
                    <div style={{ height: 10, background: 'rgba(255,255,255,0.06)', borderRadius: 5, overflow: 'hidden' }}>
                      <div style={{
                        width: `${pct}%`,
                        height: '100%',
                        background: isMin ? 'linear-gradient(90deg, #f59e0b, #fbbf24)' : 'linear-gradient(90deg, #0d9488, #2dd4bf)',
                        borderRadius: 5,
                      }} />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Note calcul & Alerte valeurs manquantes */}
            <div style={{ background: 'rgba(255,255,255,0.03)', padding: '0.85rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)', fontSize: '0.8rem', lineHeight: 1.5 }}>
              <div style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                <strong style={{ color: 'var(--teal-400)' }}>Formule vérifiée :</strong> Pourcentage minoritaire = <code style={{ color: '#fbbf24' }}>count_min / total_étiqueté</code>.
                Ex : 45 / 950 = <strong>4,7%</strong> (et non 5,3% issu d'une division par base inadéquate).
              </div>
              {kpis.target_info.missing_target_count > 0 ? (
                <div style={{ color: 'var(--red-400)', display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.5rem' }}>
                  <AlertCircle size={15} />
                  <span>
                    <strong>{kpis.target_info.missing_target_count} lignes ({kpis.target_info.missing_target_pct}%)</strong> avec cible manquante seront filtrées avant le split train/test.
                  </span>
                </div>
              ) : (
                <div style={{ color: 'var(--green-400)', display: 'flex', alignItems: 'center', gap: '0.4rem', marginTop: '0.5rem' }}>
                  <span>✓ Cible 100% renseignée ({kpis.target_info.total_labeled} observations étiquetées).</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Section 1 : Tableau descriptif de toutes les features */}
      <div className="glass-card">
        <div className="section-header">
          <div>
            <h3 className="section-title">
              <BarChart3 size={20} color="var(--teal-400)" />
              Synthèse Statistique des Variables
            </h3>
            <p className="section-desc">
              Cliquez sur les en-têtes pour trier. Cliquez sur une ligne pour afficher sa distribution détaillée ci-dessous.
            </p>
          </div>
          <div style={{ position: 'relative', width: 240 }}>
            <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }} />
            <input
              type="text"
              placeholder="Filtrer une variable…"
              className="form-input"
              style={{ paddingLeft: '2rem', width: '100%' }}
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
            />
          </div>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th className="sortable" onClick={() => handleSort('name')}>Nom</th>
                <th className="sortable" onClick={() => handleSort('type')}>Type</th>
                <th className="sortable" onClick={() => handleSort('missing_pct')}>% Manquants</th>
                <th className="sortable" onClick={() => handleSort('unique_count')}>Uniques</th>
                <th className="sortable" onClick={() => handleSort('mean')}>Moyenne</th>
                <th className="sortable" onClick={() => handleSort('std')}>Écart-type</th>
                <th className="sortable" onClick={() => handleSort('median')}>Médiane</th>
                <th className="sortable" onClick={() => handleSort('skewness')}>Asymétrie</th>
                <th className="sortable" onClick={() => handleSort('outliers_pct')}>% Outliers (IQR)</th>
                <th className="sortable" onClick={() => handleSort('target_association')}>Lien Cible</th>
              </tr>
            </thead>
            <tbody>
              {sortedFeatures.map((f) => (
                <tr
                  key={f.name}
                  onClick={() => setSelectedCol(f.name)}
                  style={{
                    cursor: 'pointer',
                    background: selectedCol === f.name ? 'rgba(45, 212, 191, 0.08)' : undefined,
                    borderLeft: selectedCol === f.name ? '3px solid var(--teal-400)' : '3px solid transparent',
                  }}
                >
                  <td style={{ fontWeight: 600, color: f.is_target ? 'var(--violet-400)' : undefined }}>
                    {f.name} {f.is_target && <span style={{ fontSize: '0.7rem', color: 'var(--violet-400)' }}>(Cible)</span>}
                  </td>
                  <td><TypeBadge type={f.type} /></td>
                  <td style={{ color: f.missing_pct > 10 ? 'var(--red-400)' : undefined }}>
                    {f.missing_pct > 0 ? `${f.missing_pct}% (${f.missing_count})` : '0%'}
                  </td>
                  <td>{f.unique_count?.toLocaleString('fr-FR')}</td>
                  <td>{f.mean !== null ? f.mean : '-'}</td>
                  <td>{f.std !== null ? f.std : '-'}</td>
                  <td>{f.median !== null ? f.median : '-'}</td>
                  <td style={{ color: Math.abs(f.skewness || 0) > 1.5 ? 'var(--amber-400)' : undefined }}>
                    {f.skewness !== null ? f.skewness : '-'}
                  </td>
                  <td style={{ color: f.outliers_pct > 3 ? 'var(--amber-400)' : undefined }}>
                    {f.outliers_pct > 0 ? `${f.outliers_pct}% (${f.outliers_count})` : '0%'}
                  </td>
                  <td>
                    {f.target_association !== null ? (
                      <span style={{ fontWeight: 600, color: f.target_association > 0.4 ? 'var(--teal-400)' : undefined }}>
                        {f.target_association} <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>({f.target_association_metric})</span>
                      </span>
                    ) : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Section 2 : Distributions interactives */}
      <div className="glass-card">
        <div className="section-header" style={{ flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h3 className="section-title">
              <TrendingUp size={20} color="var(--violet-400)" />
              Distribution Interactive : <span style={{ color: 'var(--teal-400)' }}>{selectedCol}</span>
            </h3>
            <p className="section-desc">
              Visualisation en histogramme, boîte à moustaches et répartition par classe de la cible.
            </p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            {distData?.type === 'numeric' && (
              <>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={zoomP99}
                    onChange={(e) => setZoomP99(e.target.checked)}
                    style={{ accentColor: 'var(--teal-400)' }}
                  />
                  Zoom P1–P99 (Tronquer extrêmes)
                </label>
                <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.8rem', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={logScale}
                    onChange={(e) => setLogScale(e.target.checked)}
                    style={{ accentColor: 'var(--teal-400)' }}
                  />
                  Échelle Log10
                </label>
              </>
            )}
            <select
              className="form-select"
              value={selectedCol || ''}
              onChange={(e) => setSelectedCol(e.target.value)}
            >
              {features.map((f) => (
                <option key={f.name} value={f.name}>{f.name} ({f.type})</option>
              ))}
            </select>
          </div>
        </div>

        {distData && (
          <div style={{ display: 'grid', gridTemplateColumns: distData.target_comparison ? '2fr 1fr' : '1fr', gap: '1.5rem' }}>
            <div>
              {distData.type === 'numeric' || distData.type === 'boolean' ? (
                <>
                  <div style={{ marginBottom: '0.75rem', fontWeight: 600, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    Histogramme des fréquences :
                  </div>
                  <HistogramChart data={distData.histogram} height={240} />
                  <div style={{ marginTop: '1.5rem', fontWeight: 600, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    Boîte à moustaches (Boxplot IQR) :
                  </div>
                  <BoxplotVisual stats={distData.boxplot} />
                </>
              ) : (
                <>
                  <div style={{ marginBottom: '0.75rem', fontWeight: 600, fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    Top modalités les plus fréquentes :
                  </div>
                  <CategoryBarChart data={distData.top_categories} height={300} />
                </>
              )}
            </div>

            {/* Comparaison par classe de la cible si disponible */}
            {distData.target_comparison && (
              <div style={{ background: 'rgba(0,0,0,0.25)', padding: '1.25rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <h4 style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
                  Comparaison par classe de la cible :
                </h4>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Classe</th>
                      <th>Moyenne</th>
                      <th>Médiane</th>
                      <th>Effectif</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(distData.target_comparison).map(([cls, stat]) => (
                      <tr key={cls}>
                        <td style={{ fontWeight: 600, color: 'var(--teal-400)' }}>{cls}</td>
                        <td>{stat.mean}</td>
                        <td>{stat.median}</td>
                        <td>{stat.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Section 3 : Corrélations & Nuage de points */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(480px, 1fr))', gap: '1.5rem' }}>
        {/* Matrice de corrélation */}
        <div className="glass-card">
          <div className="section-header">
            <div>
              <h3 className="section-title" style={{ fontSize: '1.2rem' }}>
                Matrice de Corrélation
              </h3>
              <p className="section-desc">Relations linéaires entre variables continues.</p>
            </div>
            <div style={{ display: 'flex', gap: '0.4rem' }}>
              {['pearson', 'spearman', 'kendall'].map((m) => (
                <button
                  key={m}
                  className={`btn btn-sm ${corrMethod === m ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setCorrMethod(m)}
                >
                  {m.charAt(0).toUpperCase() + m.slice(1)}
                </button>
              ))}
            </div>
          </div>
          <CorrelationHeatmap matrixData={correlations} />
          {correlations?.top_pairs && correlations.top_pairs.length > 0 && (
            <div style={{ marginTop: '1rem' }}>
              <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.4rem' }}>
                Top paires fortement corrélées (|r| ≥ 0.40) :
              </div>
              <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                {correlations.top_pairs.slice(0, 6).map((pair, idx) => (
                  <span
                    key={idx}
                    className="badge-type badge-attention"
                    style={{ cursor: 'pointer' }}
                    onClick={() => { setScatterX(pair.feature_1); setScatterY(pair.feature_2); }}
                  >
                    {pair.feature_1} × {pair.feature_2} (r = {pair.correlation.toFixed(2)})
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Nuage de points interactif */}
        <div className="glass-card">
          <div className="section-header">
            <div>
              <h3 className="section-title" style={{ fontSize: '1.2rem' }}>
                <ScatterIcon size={18} color="var(--teal-400)" />
                Nuage de Points Coloré
              </h3>
              <p className="section-desc">Points colorés selon la variable cible supervisée.</p>
            </div>
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <select className="form-select" value={scatterX} onChange={(e) => setScatterX(e.target.value)}>
                {correlations?.columns?.map((c) => (
                  <option key={c} value={c}>X : {c}</option>
                ))}
              </select>
              <select className="form-select" value={scatterY} onChange={(e) => setScatterY(e.target.value)}>
                {correlations?.columns?.map((c) => (
                  <option key={c} value={c}>Y : {c}</option>
                ))}
              </select>
            </div>
          </div>
          <ScatterPlotView points={scatterPoints} xCol={scatterX} yCol={scatterY} height={280} />
        </div>
      </div>
    </div>
  );
}
