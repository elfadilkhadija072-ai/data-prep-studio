// Racine de l'application : état global (session, config, EDA, recommandations, choix, résultat) + navigation par onglets.
import React, { useState } from 'react'
import { api } from './api.js'
import EdaTab from './components/EdaTab.jsx'
import ExportTab from './components/ExportTab.jsx'
import ImportTab from './components/ImportTab.jsx'
import PrepTab from './components/PrepTab.jsx'
import { Spinner, fmt } from './ui.jsx'

const TABS = [
  ['import', '1', 'Import & configuration', 'Dataset · cible'],
  ['eda', '2', 'Analyse exploratoire', 'EDA interactive'],
  ['prep', '3', 'Preprocessing & Cleaning', 'Recommandations'],
  ['export', '4', 'Exécution & export', 'Dataset · rapport'],
]

export default function App() {
  const [tab, setTab] = useState('import')
  const [session, setSession] = useState(null)
  const [cfg, setCfg] = useState(null)
  const [eda, setEda] = useState(null)
  const [recs, setRecs] = useState(null)
  const [choices, setChoices] = useState({})
  const [dropDup, setDropDup] = useState(true)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  const reset = () => { setCfg(null); setEda(null); setRecs(null); setChoices({}); setResult(null) }
  const onLoaded = (s) => { setSession(s); reset(); setTab('import') }

  const onAnalyze = async (c) => {
    setBusy(true); setError(null); reset()
    try {
      const r = await api.config(session.id, c); setCfg(r.cfg)
      const [e, rc] = await Promise.all([api.eda(session.id), api.recs(session.id)])
      setEda(e.eda); setRecs(rc.recs)
      setChoices(Object.fromEntries(Object.entries(rc.recs.features).map(([n, f]) => [n, { ...f.plan }])))
      setDropDup(rc.recs.global_plan.drop_duplicates)
      setTab('eda')
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  const onApply = async () => {
    setBusy(true); setError(null)
    try { setResult(await api.apply(session.id, choices, dropDup)); setTab('export'); window.scrollTo({ top: 0, behavior: 'smooth' }) }
    catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  const enabled = { import: true, eda: !!eda, prep: !!recs, export: !!result }
  const done = { import: !!cfg, eda: !!eda, prep: !!result, export: !!result }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="logo"><svg width="26" height="26" viewBox="0 0 32 32"><path d="M6 23l6-8 5 5 9-12" stroke="#fff" strokeWidth="3" fill="none" strokeLinecap="round" strokeLinejoin="round" /></svg></div>
          <div><h1>DataPrep <em>Studio</em></h1><p>Analyse · nettoyage · préparation de données pour le Machine Learning</p></div>
        </div>
        {session && <div className="ds-chip">📄 <b>{session.name}</b> · {fmt(session.rows)} × {session.cols}{cfg && <> · {cfg.task === 'supervised' ? `🎯 ${cfg.target} (${cfg.problem})` : 'non supervisé'}</>}</div>}
      </header>

      <nav className="tabs">
        {TABS.map(([k, n, t, s]) => (
          <button key={k} className={`tab ${tab === k ? 'on' : ''} ${done[k] && tab !== k ? 'done' : ''}`} disabled={!enabled[k]} onClick={() => setTab(k)}>
            <span className="n">{done[k] && tab !== k ? '✓' : n}</span><span><b>{t}</b><small>{s}</small></span>
          </button>
        ))}
      </nav>

      {error && <div className="error"><span>⚠ {error}</span><button className="btn sm ghost" onClick={() => setError(null)}>✕</button></div>}

      {tab === 'import' && <ImportTab session={session} onLoaded={onLoaded} onAnalyze={onAnalyze} busy={busy} setError={setError} />}
      {tab === 'eda' && <EdaTab sid={session?.id} eda={eda} cfg={cfg} />}
      {tab === 'prep' && recs && <PrepTab sid={session.id} recs={recs} choices={choices} setChoices={setChoices} dropDup={dropDup} setDropDup={setDropDup} onApply={onApply} applying={busy} />}
      {tab === 'export' && <ExportTab sid={session.id} result={result} cfg={cfg} onBack={() => setTab('prep')} />}
      {busy && tab !== 'import' && tab !== 'prep' && <Spinner label="Traitement…" />}
    </div>
  )
}
