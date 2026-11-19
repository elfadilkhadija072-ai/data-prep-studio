// Onglet 3 — Preprocessing & Cleaning : anomalies, recommandations justifiées, choix, avant/après.
import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../api.js'
import { HBars, Hist } from '../charts.jsx'
import { Card, Empty, KIND, KindBadge, SEV, SevDot, Spinner, fmt } from '../ui.jsx'

const SLOT_LABELS = { clean: 'Nettoyage du texte', impute: 'Imputation', outliers: 'Valeurs aberrantes', transform: 'Transformation', scale: 'Mise à l\'échelle', encode: 'Encodage' }

export const statusOf = (rec, plan) => {
  if (!plan) return 'reco'
  const keys = ['drop', ...rec.slots]
  if (keys.every((k) => plan[k] === rec.plan[k])) return 'reco'
  if (plan.drop) return 'drop'
  if (rec.slots.every((k) => plan[k] === 'none')) return 'none'
  return 'custom'
}
const STATUS = { reco: ['pill good', '✔ recommandé'], custom: ['pill custom', '✎ personnalisé'], none: ['pill', '∅ aucun'], drop: ['pill drop', '✕ supprimée'] }

export default function PrepTab({ sid, recs, choices, setChoices, dropDup, setDropDup, onApply, applying }) {
  const names = Object.keys(recs.features)
  const [sel, setSel] = useState(names.find((n) => recs.features[n].severity === 'crit' || recs.features[n].severity === 'warn') || names[0])
  const [q, setQ] = useState(''); const [filter, setFilter] = useState('all')
  const list = names.filter((n) => n.toLowerCase().includes(q.toLowerCase()) && (filter === 'all' || (filter === 'issues' ? recs.features[n].severity !== 'ok' : statusOf(recs.features[n], choices[n]) === filter)))
  const counts = useMemo(() => names.reduce((a, n) => { const s = statusOf(recs.features[n], choices[n]); a[s] = (a[s] || 0) + 1; return a }, {}), [choices])
  const acceptAll = () => setChoices(Object.fromEntries(names.map((n) => [n, { ...recs.features[n].plan }])))
  const noneAll = () => setChoices(Object.fromEntries(names.map((n) => [n, { ...recs.features[n].plan, drop: false, ...Object.fromEntries(recs.features[n].slots.map((s) => [s, 'none'])) }])))
  const rec = recs.features[sel]

  return (
    <div>
      <Card title="Plan de nettoyage" sub="Chaque feature reçoit une recommandation justifiée par des statistiques ; vous gardez le dernier mot." right={
        <div className="row">
          <button className="btn sm" onClick={acceptAll}>✔ Tout accepter</button>
          <button className="btn sm ghost" onClick={noneAll}>∅ Ne rien appliquer</button>
        </div>}>
        <div className="row">
          {recs.global.map((g) => (
            <label key={g.code} className="anom warn toggle" style={{ flex: 1 }}>
              <input type="checkbox" checked={dropDup} onChange={(e) => setDropDup(e.target.checked)} />
              <div><b>Supprimer les {g.title.toLowerCase()}</b><span>{g.detail}</span></div>
            </label>
          ))}
          <div className="row tip">
            {Object.entries(counts).map(([k, v]) => <span key={k} className={STATUS[k][0]}>{STATUS[k][1]} · {v}</span>)}
          </div>
        </div>
      </Card>

      <div className="fgrid" style={{ marginTop: 16 }}>
        <Card title="Features">
          <input type="search" placeholder="Rechercher…" value={q} onChange={(e) => setQ(e.target.value)} style={{ width: '100%', marginBottom: 8 }} />
          <div className="row" style={{ marginBottom: 8, gap: 5 }}>
            {[['all', 'Toutes'], ['issues', 'Anomalies'], ['custom', 'Perso'], ['drop', 'Supprimées']].map(([k, l]) => <button key={k} className={`btn sm ${filter === k ? 'primary' : 'ghost'}`} onClick={() => setFilter(k)}>{l}</button>)}
          </div>
          <div className="flist">
            {list.map((n) => {
              const r = recs.features[n]; const st = statusOf(r, choices[n])
              return (
                <div key={n} className={`fitem ${n === sel ? 'on' : ''}`} onClick={() => setSel(n)}>
                  <SevDot sev={r.severity} /><span className="ic" style={{ color: KIND[r.kind].color }}>{KIND[r.kind].icon}</span>
                  <span className="nm">{n}{r.role === 'target' && ' 🎯'}</span>
                  <span className={STATUS[st][0]} title={STATUS[st][1]}>{STATUS[st][1].split(' ')[0]}</span>
                </div>
              )
            })}
            {!list.length && <Empty>Aucune feature</Empty>}
          </div>
        </Card>
        <FeaturePanel key={sel} sid={sid} rec={rec} plan={choices[sel] || rec.plan} setPlan={(p) => setChoices((c) => ({ ...c, [sel]: p }))} />
      </div>

      <div className="stickybar">
        <div className="sp"><b>{names.length} features</b> <span className="muted">· {counts.reco || 0} recommandées · {counts.custom || 0} personnalisées · {counts.none || 0} sans traitement · {counts.drop || 0} supprimées</span></div>
        <button className="btn primary lg" disabled={applying} onClick={onApply}>{applying ? 'Exécution…' : 'Exécuter les transformations →'}</button>
      </div>
    </div>
  )
}

function FeaturePanel({ sid, rec, plan, setPlan }) {
  const [pv, setPv] = useState(null); const [err, setErr] = useState(null)
  const key = JSON.stringify(plan)
  useEffect(() => {
    setErr(null)
    const t = setTimeout(() => api.preview(sid, rec.feature, plan).then(setPv).catch((e) => setErr(e.message)), 220)
    return () => clearTimeout(t)
  }, [key])
  const set = (slot, v) => setPlan({ ...plan, [slot]: v })
  const status = statusOf(rec, plan)
  return (
    <div className="stack">
      <Card title={rec.feature} sub={rec.role === 'target' ? 'Variable cible — jamais supprimée ni mise à l\'échelle sans votre accord' : undefined}
        right={<div className="row"><KindBadge kind={rec.kind} /><span className={STATUS[status][0]}>{STATUS[status][1]}</span></div>}>
        <h4 style={{ marginBottom: 8 }}>Anomalies détectées</h4>
        <div className="stack" style={{ gap: 8 }}>
          {rec.anomalies.length ? rec.anomalies.map((a, i) => (
            <div key={i} className={`anom ${a.severity}`}><SevDot sev={a.severity} /><div><b>{a.title}</b><span>{a.detail}</span></div></div>
          )) : <div className="anom"><SevDot sev="ok" /><div><b>Aucune anomalie</b><span>Cette feature est propre.</span></div></div>}
        </div>

        <div className="row" style={{ margin: '18px 0 4px' }}>
          <h4 className="sp">Transformations</h4>
          <button className="btn sm" onClick={() => setPlan({ ...rec.plan })}>✔ Accepter la recommandation</button>
          <button className="btn sm ghost" onClick={() => setPlan({ ...rec.plan, drop: false, ...Object.fromEntries(rec.slots.map((s) => [s, 'none'])) })}>∅ Ne rien appliquer</button>
        </div>

        {rec.can_drop && (
          <div className="slot">
            <label className="toggle"><input type="checkbox" checked={plan.drop} onChange={(e) => set('drop', e.target.checked)} /> <b>Supprimer la colonne</b></label>
            <div>{rec.plan.drop && <div className="reco">★ recommandé</div>}</div>
            {rec.why.drop && <div className="why">{rec.why.drop}</div>}
          </div>
        )}
        {!plan.drop && rec.slots.map((s) => (
          <div className="slot" key={s}>
            <div><b>{SLOT_LABELS[s]}</b></div>
            <div>
              <select value={plan[s]} onChange={(e) => set(s, e.target.value)}>
                {rec.options[s].map((o) => <option key={o.value} value={o.value}>{o.label}{o.value === rec.plan[s] && o.value !== 'none' ? '  ★ recommandé' : ''}</option>)}
              </select>
            </div>
            {rec.why[s] && <div className="why">{rec.why[s]}</div>}
          </div>
        ))}
        {plan.drop && <div className="notice" style={{ marginTop: 8 }}>Colonne supprimée : aucune autre transformation ne sera appliquée.</div>}
      </Card>

      <Card title="Avant / Après" sub="Aperçu : les paramètres sont appris sur le train (split courant), comme à l'exécution finale.">
        {err && <div className="error">{err}</div>}
        {!pv ? <Spinner label="Calcul de l'aperçu…" /> : (
          <>
            <div className="ba">
              <div className="before"><h4>Avant</h4><Dist d={pv.before} /></div>
              <div className="after"><h4>Après</h4>{pv.after.dropped ? <Empty>Colonne supprimée</Empty> : <Dist d={pv.after} color="#2dd4bf" />}</div>
            </div>
            {pv.info.steps.length > 0 && <div className="flow" style={{ marginTop: 14 }}>{pv.info.steps.map((s, i) => <React.Fragment key={i}>{i > 0 && <s>→</s>}<i>{s}</i></React.Fragment>)}</div>}
            {pv.info.warnings.map((w, i) => <div key={i} className="notice" style={{ marginTop: 8 }}>⚠ {w}</div>)}
            {pv.info.removed_rows > 0 && <div className="notice" style={{ marginTop: 8 }}>{pv.info.removed_rows} lignes seront retirées du jeu d'entraînement (outliers).</div>}
          </>
        )}
      </Card>
    </div>
  )
}

function Dist({ d, color = '#8b97b3' }) {
  if (d.type === 'multi') return <><p className="muted" style={{ marginBottom: 6 }}>{d.columns.length} colonnes créées — part de 1 (%)</p><HBars data={d.shares.map((s) => ({ label: s.column.split('__').pop(), value: +s.pct.toFixed(2) }))} color="#2dd4bf" unit=" %" /></>
  if (d.type === 'numeric') {
    return (
      <>
        <Hist data={d.hist_zoom?.length ? d.hist_zoom : d.hist} color={color} height={190} />
        <div className="stats" style={{ gridTemplateColumns: 'repeat(2,1fr)' }}>
          {[['Moyenne', d.mean], ['σ', d.std], ['Min', d.min], ['Max', d.max], ['Asymétrie', d.skew], ['Manquants', d.missing]].map(([k, v]) => <div key={k}><span>{k}</span><b>{fmt(v)}</b></div>)}
        </div>
      </>
    )
  }
  if (d.type === 'categorical' && d.top) return <><HBars data={d.top.slice(0, 10).map((t) => ({ label: t.value, value: +t.pct.toFixed(2) }))} color={color} unit=" %" /><p className="muted">Manquants : {fmt(d.missing)}</p></>
  return <Empty>—</Empty>
}
