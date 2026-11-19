// Onglet 1 — Importation du dataset + configuration (supervisé / non supervisé, cible).
import React, { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import { Card, KindBadge, Segmented, Spinner, fmt } from '../ui.jsx'

export default function ImportTab({ session, onLoaded, onAnalyze, busy, setError }) {
  const [samples, setSamples] = useState([])
  const [over, setOver] = useState(false)
  const [loading, setLoading] = useState(false)
  const [task, setTask] = useState('supervised')
  const [target, setTarget] = useState('')
  const [problem, setProblem] = useState('auto')
  const [testSize, setTestSize] = useState(0.2)
  const inp = useRef()

  useEffect(() => { api.samples().then(setSamples).catch(() => {}) }, [])
  useEffect(() => { setTarget(''); setProblem('auto') }, [session?.id])

  const load = async (fn) => {
    setLoading(true); setError(null)
    try { onLoaded(await fn()) } catch (e) { setError(e.message) } finally { setLoading(false) }
  }
  const onDrop = (e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files?.[0]; if (f) load(() => api.upload(f)) }
  const canGo = task === 'unsupervised' || !!target

  return (
    <div className="stack">
      {!session && (
        <div className="hero">
          <h2>Préparez vos données pour le <em>Machine Learning</em></h2>
          <p>Importez un fichier CSV ou Excel : l'application analyse, détecte les anomalies, recommande des transformations justifiées
            statistiquement et exporte un jeu de données propre — sans fuite de données.</p>
        </div>
      )}

      <div className="grid g-21">
        <Card title="1 · Importer un dataset" sub="Formats acceptés : .csv (séparateur auto-détecté), .xlsx, .xls">
          <div className={`drop ${over ? 'over' : ''}`} onClick={() => inp.current.click()}
            onDragOver={(e) => { e.preventDefault(); setOver(true) }} onDragLeave={() => setOver(false)} onDrop={onDrop}>
            <input ref={inp} type="file" hidden accept=".csv,.txt,.xlsx,.xls" onChange={(e) => e.target.files[0] && load(() => api.upload(e.target.files[0]))} />
            {loading ? <Spinner label="Lecture du fichier…" /> : (
              <>
                <div className="big">⇪</div>
                <b>Glissez votre fichier ici</b>
                <p className="muted">ou cliquez pour parcourir</p>
              </>
            )}
          </div>
        </Card>
        <Card title="Jeux de démonstration" sub="Données synthétiques pour tester l'application">
          <div className="stack" style={{ gap: 8 }}>
            {samples.map((s) => (
              <div className="sample" key={s.file}>
                <div><b>{s.label}</b><div className="muted">{s.size_kb} Ko</div></div>
                <button className="btn sm" onClick={() => load(() => api.loadSample(s.file))}>Charger</button>
              </div>
            ))}
            {!samples.length && <p className="muted">Aucun exemple disponible.</p>}
          </div>
        </Card>
      </div>

      {session && (
        <>
          <Card title={`Aperçu · ${session.name}`} sub={`${fmt(session.rows)} lignes × ${session.cols} colonnes`}>
            {Object.keys(session.notes || {}).length > 0 && (
              <div className="notice" style={{ marginBottom: 12 }}>
                <b>Conversions automatiques :</b> {Object.entries(session.notes).map(([k, v]) => `« ${k} » – ${v}`).join(' · ')}
              </div>
            )}
            <div className="tbl-wrap" style={{ maxHeight: 340 }}>
              <table>
                <thead><tr>{session.columns.map((c) => <th key={c.name}>{c.name}<div style={{ marginTop: 4 }}><KindBadge kind={c.kind} /></div></th>)}</tr></thead>
                <tbody>
                  {session.preview.map((r, i) => (
                    <tr key={i}>{session.columns.map((c) => <td key={c.name} className={c.kind === 'numeric' ? 'num' : ''}>{r[c.name] === null || r[c.name] === undefined ? <span className="muted">∅</span> : String(r[c.name]).slice(0, 28)}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <Card title="2 · Configurer l'apprentissage" sub="Le type d'apprentissage conditionne la cible, le split train/test et l'anti-leakage.">
            <div className="grid g2">
              <div className={`choice ${task === 'supervised' ? 'on' : ''}`} onClick={() => setTask('supervised')}>
                <b>🎯 Supervisé</b><span className="muted">Prédire une variable cible (classification ou régression). Split train/test stratifié avant toute transformation.</span>
              </div>
              <div className={`choice ${task === 'unsupervised' ? 'on' : ''}`} onClick={() => setTask('unsupervised')}>
                <b>🧭 Non supervisé</b><span className="muted">Clustering, réduction de dimension… Pas de cible ni de split : les transformations sont apprises sur tout le dataset.</span>
              </div>
            </div>
            {task === 'supervised' && (
              <div className="grid g3" style={{ marginTop: 16 }}>
                <label className="field">Variable cible (target)
                  <select value={target} onChange={(e) => setTarget(e.target.value)}>
                    <option value="">— choisir —</option>
                    {session.columns.map((c) => <option key={c.name} value={c.name}>{c.name}</option>)}
                  </select>
                </label>
                <label className="field">Type de problème
                  <Segmented value={problem} onChange={setProblem} options={[{ value: 'auto', label: 'Auto' }, { value: 'classification', label: 'Classification' }, { value: 'regression', label: 'Régression' }]} />
                </label>
                <label className="field">Part du jeu de test : {Math.round(testSize * 100)} %
                  <input type="range" min="0.1" max="0.4" step="0.05" value={testSize} onChange={(e) => setTestSize(+e.target.value)} />
                </label>
              </div>
            )}
            <div className="row" style={{ marginTop: 20 }}>
              <span className="sp tip">{task === 'supervised' && !target ? 'Sélectionnez la variable cible pour continuer.' : 'Prêt : lancez l\'analyse exploratoire.'}</span>
              <button className="btn primary lg" disabled={!canGo || busy}
                onClick={() => onAnalyze({ task, target: task === 'supervised' ? target : null, problem: problem === 'auto' ? null : problem, test_size: testSize, drop_duplicates: true })}>
                {busy ? 'Analyse en cours…' : 'Lancer l\'analyse →'}
              </button>
            </div>
          </Card>
        </>
      )}
    </div>
  )
}
