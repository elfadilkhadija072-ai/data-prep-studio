// Onglet 2 — Analyse exploratoire : vue d'ensemble, features, distributions, qualité, corrélations.
import React, { useEffect, useMemo, useState } from 'react'
import { api } from '../api.js'
import { BoxPlot, COLORS, Heatmap, HBars, Hist, Scatter2D } from '../charts.jsx'
import { Card, Empty, KIND, Kpi, KindBadge, Segmented, Spinner, fmt, pct } from '../ui.jsx'

const SUBS = [['overview', 'Vue d\'ensemble'], ['features', 'Features & statistiques'], ['dist', 'Distributions'], ['quality', 'Manquants · doublons · outliers'], ['corr', 'Corrélations']]

export default function EdaTab({ sid, eda, cfg }) {
  const [sub, setSub] = useState('overview')
  if (!eda) return <Spinner label="Calcul de l'analyse exploratoire…" />
  return (
    <div>
      <div className="sub">{SUBS.map(([k, l]) => <button key={k} className={sub === k ? 'on' : ''} onClick={() => setSub(k)}>{l}</button>)}</div>
      {sub === 'overview' && <Overview eda={eda} cfg={cfg} />}
      {sub === 'features' && <Features eda={eda} />}
      {sub === 'dist' && <Distributions sid={sid} eda={eda} cfg={cfg} />}
      {sub === 'quality' && <Quality eda={eda} />}
      {sub === 'corr' && <Correlations sid={sid} eda={eda} cfg={cfg} />}
    </div>
  )
}

function Overview({ eda, cfg }) {
  const o = eda.overview, t = eda.target
  const kinds = Object.entries(o.kinds).filter(([, v]) => v > 0).map(([k, v]) => ({ label: KIND[k].label, value: v }))
  return (
    <div className="stack">
      <div className="grid g4">
        <Kpi label="Lignes" value={fmt(o.rows)} hint={`${o.memory_mb.toFixed(1)} Mo en mémoire`} />
        <Kpi label="Colonnes" value={o.cols} hint={`${o.kinds.numeric} num · ${o.kinds.categorical} cat · ${o.kinds.datetime} date`} />
        <Kpi label="Cellules manquantes" value={pct(o.missing_cells_pct, 2)} hint={`${fmt(o.rows_with_missing)} lignes touchées (${pct(o.rows_with_missing_pct, 1)})`} tone={o.missing_cells_pct > 5 ? 'warn' : ''} />
        <Kpi label="Doublons" value={fmt(o.duplicates)} hint={pct(o.duplicates_pct, 2)} tone={o.duplicates ? 'warn' : 'good'} />
      </div>
      <div className="grid g2">
        <Card title="Répartition des types"><HBars data={kinds} height={150} /></Card>
        {t ? (
          <Card title={`Cible « ${t.name} »`} sub={t.problem === 'classification' ? 'Classification — distribution des classes' : 'Régression — voir Distributions'}
            right={t.imbalance_ratio ? <span className={`pill ${t.imbalance_ratio >= 3 ? 'custom' : 'good'}`}>ratio {t.imbalance_ratio.toFixed(1)}:1</span> : null}>
            {t.classes ? <HBars data={t.classes.map((c) => ({ label: c.value, value: +c.pct.toFixed(2) }))} unit=" %" color="#f87171" height={Math.max(110, t.classes.length * 30)} />
              : <Empty>Variable continue : consultez l'onglet « Distributions ».</Empty>}
          </Card>
        ) : <Card title="Apprentissage non supervisé"><Empty>Aucune variable cible : l'analyse porte sur toutes les features.</Empty></Card>}
      </div>
      {o.constant_columns.length > 0 && <div className="notice">⚠ Colonnes constantes détectées : <b>{o.constant_columns.join(', ')}</b></div>}
    </div>
  )
}

function Features({ eda }) {
  const [q, setQ] = useState(''); const [sort, setSort] = useState({ k: null, d: 1 })
  const rows = useMemo(() => {
    let r = eda.features.filter((f) => f.name.toLowerCase().includes(q.toLowerCase()))
    if (sort.k) r = [...r].sort((a, b) => ((get(a, sort.k) ?? -Infinity) > (get(b, sort.k) ?? -Infinity) ? 1 : -1) * sort.d)
    return r
  }, [eda, q, sort])
  const get = (f, k) => (k in f ? f[k] : f.stats?.[k])
  const H = ({ k, children, num }) => <th className={`s ${num ? 'num' : ''}`} onClick={() => setSort((s) => ({ k, d: s.k === k ? -s.d : 1 }))}>{children}{sort.k === k ? (sort.d > 0 ? ' ▲' : ' ▼') : ''}</th>
  return (
    <Card title="Toutes les features" sub="Cliquez sur un en-tête pour trier" right={<input type="search" placeholder="Rechercher…" value={q} onChange={(e) => setQ(e.target.value)} />}>
      <div className="tbl-wrap">
        <table>
          <thead><tr><H k="name">Feature</H><th>Type</th><H k="missing_pct" num>Manq. %</H><H k="unique" num>Uniques</H><H k="mean" num>Moyenne</H><H k="std" num>σ</H><H k="min" num>Min</H><H k="median" num>Médiane</H><H k="max" num>Max</H><H k="skew" num>Asym.</H><H k="outliers_iqr_pct" num>Outliers %</H><H k="target_assoc" num>Lien cible</H></tr></thead>
          <tbody>
            {rows.map((f) => {
              const s = f.stats || {}
              return (
                <tr key={f.name}>
                  <td><b>{f.name}</b>{f.is_target && <span className="pill custom" style={{ marginLeft: 8 }}>cible</span>}</td>
                  <td><KindBadge kind={f.kind} /></td>
                  <td className="num" style={{ color: f.missing_pct > 5 ? 'var(--amber)' : undefined }}>{f.missing_pct.toFixed(2)}</td>
                  <td className="num">{fmt(f.unique)}</td>
                  <td className="num">{fmt(s.mean)}</td><td className="num">{fmt(s.std)}</td><td className="num">{fmt(s.min)}</td><td className="num">{fmt(s.median)}</td><td className="num">{fmt(s.max)}</td>
                  <td className="num" style={{ color: Math.abs(s.skew) > 1 ? 'var(--amber)' : undefined }}>{fmt(s.skew, 2)}</td>
                  <td className="num" style={{ color: s.outliers_iqr_pct > 5 ? 'var(--red)' : undefined }}>{s.outliers_iqr_pct === undefined ? '—' : s.outliers_iqr_pct.toFixed(2)}</td>
                  <td className="num">{f.target_assoc == null ? '—' : <div className="bar"><i style={{ width: `${f.target_assoc * 100}%`, background: f.target_assoc > 0.95 ? 'var(--red)' : 'var(--violet)' }} /></div>}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </Card>
  )
}

function Distributions({ sid, eda, cfg }) {
  const [name, setName] = useState(eda.features[0].name)
  const [zoom, setZoom] = useState(true); const [logY, setLogY] = useState(false)
  const [grp, setGrp] = useState(null)
  const f = eda.features.find((x) => x.name === name); const s = f.stats || {}
  useEffect(() => { setGrp(null); if (cfg.problem === 'classification' && !f.is_target) api.groupby(sid, name).then(setGrp).catch(() => {}) }, [sid, name])
  return (
    <div className="fgrid">
      <Card className="flist-card" title="Features">
        <div className="flist">
          {eda.features.map((x) => (
            <div key={x.name} className={`fitem ${x.name === name ? 'on' : ''}`} onClick={() => setName(x.name)}>
              <span className="ic" style={{ color: KIND[x.kind].color }}>{KIND[x.kind].icon}</span><span className="nm">{x.name}</span>
              {x.missing_pct > 0 && <span className="pill">{x.missing_pct.toFixed(0)}%</span>}
            </div>
          ))}
        </div>
      </Card>
      <div className="stack">
        <Card title={f.name} sub={`${f.dtype} · ${fmt(f.count)} valeurs · ${fmt(f.unique)} uniques`} right={<KindBadge kind={f.kind} />}>
          {f.kind === 'numeric' && s.hist && (
            <>
              <div className="row" style={{ marginBottom: 8 }}>
                <label className="toggle"><input type="checkbox" checked={zoom} onChange={(e) => setZoom(e.target.checked)} /> Zoom P1–P99</label>
                <label className="toggle"><input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} /> Échelle log (Y)</label>
              </div>
              {s.discrete ? <Hist data={s.discrete} /> : <Hist data={zoom ? s.hist_zoom : s.hist} logY={logY} />}
              <h4 style={{ margin: '14px 0 4px' }}>Boîte à moustaches</h4>
              <BoxPlot s={s} />
              <div className="stats">
                {[['Moyenne', s.mean], ['Médiane', s.median], ['Écart-type', s.std], ['Min', s.min], ['Max', s.max], ['Q1', s.q1], ['Q3', s.q3], ['Asymétrie', s.skew], ['Kurtosis', s.kurtosis], ['Zéros', s.zeros], ['Négatifs', s.negatives], ['Outliers IQR', `${s.outliers_iqr} (${s.outliers_iqr_pct.toFixed(2)} %)`]].map(([k, v]) => <div key={k}><span>{k}</span><b>{fmt(v)}</b></div>)}
              </div>
            </>
          )}
          {f.kind !== 'numeric' && f.kind !== 'datetime' && s.top && (
            <>
              <HBars data={s.top.map((t) => ({ label: t.value, value: +t.pct.toFixed(2) }))} unit=" %" color="#a78bfa" />
              {s.case_issue && <div className="notice" style={{ marginTop: 10 }}>⚠ Casse/espaces incohérents : {f.unique} modalités brutes → {s.normalized_unique} après normalisation.</div>}
            </>
          )}
          {f.kind === 'datetime' && <div className="stats"><div><span>Début</span><b>{s.min?.slice(0, 16)}</b></div><div><span>Fin</span><b>{s.max?.slice(0, 16)}</b></div><div><span>Durée</span><b>{fmt(s.span_days)} j</b></div></div>}
        </Card>
        {grp?.groups?.length > 0 && (
          <Card title={`« ${name} » selon la classe de « ${cfg.target} »`} sub="Un écart marqué entre classes signale une variable discriminante.">
            {grp.kind === 'numeric'
              ? <HBars data={grp.groups.map((g) => ({ label: `classe ${g.group}`, value: +(g.mean ?? 0).toFixed(3) }))} color="#2dd4bf" unit=" (moyenne)" height={Math.max(100, grp.groups.length * 34)} />
              : <div className="tbl-wrap"><table><thead><tr><th>Classe</th><th>n</th><th>Modalités dominantes</th></tr></thead><tbody>{grp.groups.map((g) => <tr key={g.group}><td>{g.group}</td><td>{fmt(g.n)}</td><td>{g.top.map((t) => `${t.value} (${t.pct.toFixed(0)}%)`).join(' · ')}</td></tr>)}</tbody></table></div>}
          </Card>
        )}
      </div>
    </div>
  )
}

function Quality({ eda }) {
  const miss = eda.features.filter((f) => f.missing > 0).sort((a, b) => b.missing_pct - a.missing_pct).map((f) => ({ label: f.name, value: +f.missing_pct.toFixed(2) }))
  const outl = eda.features.filter((f) => f.stats?.outliers_iqr > 0 && !f.stats.is_binary).sort((a, b) => b.stats.outliers_iqr_pct - a.stats.outliers_iqr_pct).map((f) => ({ label: f.name, value: +f.stats.outliers_iqr_pct.toFixed(2) }))
  const o = eda.overview
  return (
    <div className="stack">
      <div className="grid g2">
        <Card title="Valeurs manquantes par colonne" sub={`${fmt(o.missing_cells)} cellules · ${pct(o.rows_with_missing_pct)} des lignes ont au moins un manquant`}>
          {miss.length ? <HBars data={miss} unit=" %" colorFn={(d) => (d.value > 60 ? '#f87171' : d.value > 5 ? '#fbbf24' : '#2dd4bf')} /> : <Empty>Aucune valeur manquante 🎉</Empty>}
        </Card>
        <Card title="Valeurs aberrantes (règle IQR de Tukey)" sub="Pourcentage de valeurs hors [Q1 − 1,5·IQR ; Q3 + 1,5·IQR]">
          {outl.length ? <HBars data={outl} unit=" %" colorFn={(d) => (d.value > 5 ? '#f87171' : d.value > 1 ? '#fbbf24' : '#60a5fa')} /> : <Empty>Aucun outlier détecté</Empty>}
        </Card>
      </div>
      <Card title="Doublons" sub={`${fmt(o.duplicates)} lignes dupliquées (${pct(o.duplicates_pct, 2)})`}>
        {o.duplicates ? (
          <div className="tbl-wrap" style={{ maxHeight: 260 }}>
            <table><thead><tr>{Object.keys(eda.duplicates_sample[0] || {}).map((k) => <th key={k}>{k}</th>)}</tr></thead>
              <tbody>{eda.duplicates_sample.map((r, i) => <tr key={i}>{Object.values(r).map((v, j) => <td key={j}>{String(v).slice(0, 24)}</td>)}</tr>)}</tbody></table>
          </div>
        ) : <Empty>Aucune ligne dupliquée</Empty>}
      </Card>
    </div>
  )
}

function Correlations({ sid, eda, cfg }) {
  const [method, setMethod] = useState('pearson'); const [corr, setCorr] = useState(eda.corr)
  const [pair, setPair] = useState(null); const [pts, setPts] = useState(null)
  const nums = eda.features.filter((f) => f.kind === 'numeric').map((f) => f.name)
  useEffect(() => { method === 'pearson' ? setCorr(eda.corr) : api.corr(sid, method).then(setCorr) }, [method])
  useEffect(() => {
    if (!pair) return
    const hue = cfg.problem === 'classification' ? cfg.target : null
    api.scatter(sid, pair[0], pair[1], hue && !pair.includes(hue) ? hue : null).then((r) => setPts(r.points))
  }, [pair])
  if (corr.columns.length < 2) return <Empty>Au moins 2 variables numériques sont nécessaires.</Empty>
  return (
    <div className="stack">
      <div className="grid g-21">
        <Card title="Matrice de corrélation" sub={`${corr.columns.length} variables numériques${corr.sampled ? ' · échantillon de 200 000 lignes' : ''} — cliquez une cellule pour le nuage de points`}
          right={<Segmented value={method} onChange={setMethod} options={[{ value: 'pearson', label: 'Pearson' }, { value: 'spearman', label: 'Spearman' }, { value: 'kendall', label: 'Kendall' }]} />}>
          <Heatmap columns={corr.columns} matrix={corr.matrix} onPick={(a, b) => setPair([a, b])} />
          <div className="row tip" style={{ marginTop: 10 }}><span style={{ color: '#60a5fa' }}>■</span> négative <span style={{ color: '#f87171' }}>■</span> positive</div>
        </Card>
        <Card title="Paires les plus corrélées">
          <div className="tbl-wrap" style={{ maxHeight: 420 }}>
            <table><tbody>{corr.pairs.slice(0, 15).map((p) => (
              <tr key={p.a + p.b} className="clk" onClick={() => setPair([p.a, p.b])}>
                <td>{p.a} <span className="muted">×</span> {p.b}</td>
                <td className="num" style={{ color: Math.abs(p.r) >= 0.95 ? 'var(--red)' : Math.abs(p.r) >= 0.7 ? 'var(--amber)' : undefined }}><b>{p.r.toFixed(3)}</b></td>
              </tr>))}</tbody></table>
          </div>
        </Card>
      </div>
      <Card title="Nuage de points" sub="Échantillon de 1 500 points" right={
        <div className="row">
          <select value={pair?.[0] || ''} onChange={(e) => setPair([e.target.value, pair?.[1] || nums[1]])}><option value="">X…</option>{nums.map((n) => <option key={n}>{n}</option>)}</select>
          <select value={pair?.[1] || ''} onChange={(e) => setPair([pair?.[0] || nums[0], e.target.value])}><option value="">Y…</option>{nums.map((n) => <option key={n}>{n}</option>)}</select>
        </div>}>
        {pts ? <Scatter2D points={pts} xl={pair[0]} yl={pair[1]} /> : <Empty>Choisissez deux variables ou cliquez une cellule de la matrice.</Empty>}
      </Card>
    </div>
  )
}
