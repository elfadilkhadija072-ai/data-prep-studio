// Graphiques (Recharts + SVG maison). Palette cohérente avec le thème sombre.
import React from 'react'
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis } from 'recharts'
import { fmt } from './ui.jsx'

export const COLORS = ['#2dd4bf', '#a78bfa', '#fbbf24', '#60a5fa', '#f87171', '#4ade80', '#f472b6', '#fb923c']
const axis = { stroke: '#64708f', fontSize: 11, tickLine: false, axisLine: { stroke: 'rgba(148,163,184,.2)' } }

const compact = new Intl.NumberFormat('fr-FR', { notation: 'compact', maximumFractionDigits: 2 })
const cx = (v) => (typeof v === 'number' ? compact.format(v) : v)

const Tip = ({ active, payload, label, unit }) => active && payload?.length ? (
  <div className="rt"><b>{payload[0].payload.x0 !== undefined ? `${cx(payload[0].payload.x0)} → ${cx(payload[0].payload.x1)}` : (label ?? payload[0].payload.label)}</b><br />{payload.map((p) => `${p.name ?? 'n'} : ${fmt(p.value)}${unit || ''}`).join(' · ')}</div>
) : null

export function Hist({ data, color = '#2dd4bf', height = 220, logY = false }) {
  if (!data?.length) return <div className="empty">Aucune donnée</div>
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 6, right: 8, left: -12, bottom: 0 }} barCategoryGap={1}>
        <CartesianGrid stroke="rgba(148,163,184,.1)" vertical={false} />
        <XAxis dataKey={data[0].x0 !== undefined ? 'x0' : 'label'} {...axis} interval="preserveStartEnd" minTickGap={28} tickFormatter={cx} />
        <YAxis {...axis} scale={logY ? 'symlog' : 'auto'} domain={logY ? [0, 'auto'] : undefined} allowDataOverflow />
        <Tooltip content={<Tip />} cursor={{ fill: 'rgba(148,163,184,.1)' }} />
        <Bar dataKey="count" name="effectif" fill={color} radius={[3, 3, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  )
}

// Barres horizontales : [{label, value}]
export function HBars({ data, color = '#a78bfa', height, unit = '', colorFn }) {
  const h = height || Math.max(120, data.length * 26 + 20)
  return (
    <ResponsiveContainer width="100%" height={h}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 16, left: 4, bottom: 0 }}>
        <CartesianGrid stroke="rgba(148,163,184,.1)" horizontal={false} />
        <XAxis type="number" {...axis} />
        <YAxis type="category" dataKey="label" width={110} {...axis} interval={0} tickFormatter={(v) => (String(v).length > 16 ? `${String(v).slice(0, 15)}…` : v)} />
        <Tooltip content={<Tip unit={unit} />} cursor={{ fill: 'rgba(148,163,184,.1)' }} />
        <Bar dataKey="value" name="valeur" radius={[0, 4, 4, 0]} isAnimationActive={false}>
          {data.map((d, i) => <Cell key={i} fill={colorFn ? colorFn(d) : color} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// Boîte à moustaches horizontale (SVG). L'échelle suit les moustaches : les outliers extrêmes sont
// signalés par une flèche rouge au bord (sinon une seule valeur aberrante écrase toute la boîte).
export function BoxPlot({ s }) {
  if (!s || s.min === undefined) return null
  const pad = (s.whisker_hi - s.whisker_lo) * 0.08 || 1
  const lo = s.whisker_lo - pad, hi = s.whisker_hi + pad, span = hi - lo
  const x = (v) => 14 + Math.min(Math.max((v - lo) / span, 0), 1) * 572
  const outLo = s.min < s.whisker_lo, outHi = s.max > s.whisker_hi
  return (
    <svg viewBox="0 0 600 78" width="100%" role="img" aria-label="boxplot">
      <line x1={x(s.whisker_lo)} x2={x(s.whisker_hi)} y1="34" y2="34" stroke="#8b97b3" strokeWidth="2" />
      <line x1={x(s.whisker_lo)} x2={x(s.whisker_lo)} y1="24" y2="44" stroke="#8b97b3" strokeWidth="2" />
      <line x1={x(s.whisker_hi)} x2={x(s.whisker_hi)} y1="24" y2="44" stroke="#8b97b3" strokeWidth="2" />
      <rect x={x(s.q1)} y="16" width={Math.max(3, x(s.q3) - x(s.q1))} height="36" rx="5" fill="rgba(45,212,191,.28)" stroke="#2dd4bf" strokeWidth="2" />
      <line x1={x(s.median)} x2={x(s.median)} y1="16" y2="52" stroke="#fbbf24" strokeWidth="3" />
      {outLo && <g><path d="M8 34l10-7v14z" fill="#f87171" /><text x="22" y="38" fill="#f87171" fontSize="10">min {cx(s.min)}</text></g>}
      {outHi && <g><path d="M592 34l-10-7v14z" fill="#f87171" /><text x="578" y="38" fill="#f87171" fontSize="10" textAnchor="end">max {cx(s.max)}</text></g>}
      <text x={x(s.q1)} y="70" fill="#8b97b3" fontSize="10" textAnchor="middle">Q1 {cx(s.q1)}</text>
      <text x={x(s.median)} y="11" fill="#fbbf24" fontSize="10" textAnchor="middle">méd. {cx(s.median)}</text>
      <text x={x(s.q3)} y="70" fill="#8b97b3" fontSize="10" textAnchor="middle">Q3 {cx(s.q3)}</text>
    </svg>
  )
}

export function Heatmap({ columns, matrix, onPick }) {
  const n = columns.length
  const color = (v) => {
    const a = Math.min(1, Math.abs(v))
    return v >= 0 ? `rgba(248,113,113,${0.12 + a * 0.88})` : `rgba(96,165,250,${0.12 + a * 0.88})`
  }
  return (
    <div className="heat" style={{ gridTemplateColumns: `130px repeat(${n}, minmax(22px, 1fr))` }}>
      <div />
      {columns.map((c) => <div key={c} className="lab" style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)', height: 110, justifyContent: 'flex-end' }}>{c}</div>)}
      {columns.map((r, i) => (
        <React.Fragment key={r}>
          <div className="lab">{r}</div>
          {matrix[i].map((v, j) => (
            <div key={j} className="c" title={`${r} × ${columns[j]} : ${v.toFixed(3)}`} style={{ background: color(v), cursor: 'pointer' }}
              onClick={() => i !== j && onPick?.(r, columns[j])}>{n <= 14 ? v.toFixed(2).replace('0.', '.').replace('1.00', '1') : ''}</div>
          ))}
        </React.Fragment>
      ))}
    </div>
  )
}

export function Scatter2D({ points, xl, yl }) {
  const groups = [...new Set(points.map((p) => p.g ?? 'all'))].slice(0, 8)
  return (
    <ResponsiveContainer width="100%" height={300}>
      <ScatterChart margin={{ top: 8, right: 12, left: -6, bottom: 4 }}>
        <CartesianGrid stroke="rgba(148,163,184,.1)" />
        <XAxis type="number" dataKey="x" name={xl} {...axis} />
        <YAxis type="number" dataKey="y" name={yl} {...axis} />
        <ZAxis range={[26, 26]} />
        <Tooltip cursor={{ strokeDasharray: '3 3' }} content={<Tip />} />
        {groups.map((g, i) => (
          <Scatter key={g} name={g} data={points.filter((p) => (p.g ?? 'all') === g)} fill={COLORS[i % COLORS.length]} fillOpacity={0.55} isAnimationActive={false} />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  )
}
