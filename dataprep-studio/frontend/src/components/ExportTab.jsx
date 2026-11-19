// Onglet 4 — Résultat de l'exécution + téléchargement du dataset transformé et du rapport Excel.
import React from 'react'
import { api } from '../api.js'
import { Card, Empty, Kpi, fmt } from '../ui.jsx'

export default function ExportTab({ sid, result, cfg, onBack }) {
  if (!result) return <Empty>Exécutez d'abord les transformations dans l'onglet « Preprocessing ».</Empty>
  const supervised = cfg.task === 'supervised'
  const rows = Object.entries(result.infos)
  return (
    <div className="stack">
      <Card className="success" title="✔ Transformations appliquées" sub="Le dataset est prêt pour l'entraînement.">
        <div className="grid g4">
          <Kpi label="Lignes train" value={fmt(result.rows_train)} hint={`sur ${fmt(result.rows_before)} au départ`} />
          <Kpi label="Lignes test" value={fmt(result.rows_test)} hint={supervised ? 'transformées avec les stats du train' : 'non supervisé : pas de split'} />
          <Kpi label="Colonnes" value={`${result.cols_before} → ${result.cols_after}`} hint="après encodage / suppressions" />
          <Kpi label="Manquants restants" value={fmt(result.remaining_missing)} tone={result.remaining_missing ? 'warn' : 'good'} />
        </div>
      </Card>

      <div className="grid g2">
        <Card title="Bonnes pratiques ML appliquées">
          <div className="flow" style={{ marginBottom: 12 }}><i>Doublons</i><s>→</s><i>Split train/test</i><s>→</s><i>fit sur train</i><s>→</s><i>transform train + test</i></div>
          <ul className="log">{result.global_log.map((l, i) => <li key={i}>{l}</li>)}</ul>
        </Card>
        <Card title="Télécharger">
          <div className="stack" style={{ gap: 10 }}>
            <a className="btn primary lg" href={api.reportUrl(sid)}>📊 Rapport Excel par feature</a>
            <div className="row">
              <a className="btn" href={api.downloadUrl(sid, 'all', 'csv')}>⬇ Dataset complet (CSV)</a>
              <a className="btn" href={api.downloadUrl(sid, 'all', 'xlsx')}>⬇ Dataset (XLSX)</a>
            </div>
            {supervised && <div className="row">
              <a className="btn sm" href={api.downloadUrl(sid, 'train', 'csv')}>train.csv</a>
              <a className="btn sm" href={api.downloadUrl(sid, 'test', 'csv')}>test.csv</a>
              <span className="tip">La colonne <code>_split</code> du fichier complet indique train / test.</span>
            </div>}
            <button className="btn ghost" onClick={onBack}>← Modifier le plan</button>
          </div>
        </Card>
      </div>

      <Card title="Aperçu du dataset transformé (train)">
        <div className="tbl-wrap" style={{ maxHeight: 300 }}>
          <table><thead><tr>{result.columns.map((c) => <th key={c}>{c}</th>)}</tr></thead>
            <tbody>{result.preview.map((r, i) => <tr key={i}>{result.columns.map((c) => <td key={c} className="num">{typeof r[c] === 'number' ? fmt(r[c], 4) : String(r[c] ?? '∅').slice(0, 22)}</td>)}</tr>)}</tbody></table>
        </div>
      </Card>

      <Card title="Ce qui a réellement été appliqué, feature par feature">
        <div className="tbl-wrap">
          <table><thead><tr><th>Feature</th><th>Étapes exécutées</th><th>Colonnes produites</th><th className="num">Imputées</th><th className="num">Écrêtées</th></tr></thead>
            <tbody>{rows.map(([n, i]) => (
              <tr key={n}><td><b>{n}</b></td><td style={{ whiteSpace: 'normal', minWidth: 280 }}>{i.steps.join(' → ') || <span className="muted">aucune</span>}{i.warnings.map((w, k) => <div key={k} style={{ color: 'var(--amber)' }}>⚠ {w}</div>)}</td>
                <td style={{ whiteSpace: 'normal', maxWidth: 260 }} className="mono">{i.dropped ? '—' : i.new_columns.length > 4 ? `${i.new_columns.slice(0, 3).join(', ')} … (+${i.new_columns.length - 3})` : i.new_columns.join(', ')}</td>
                <td className="num">{fmt(i.imputed)}</td><td className="num">{fmt(i.clipped)}</td></tr>))}</tbody></table>
        </div>
      </Card>
    </div>
  )
}
