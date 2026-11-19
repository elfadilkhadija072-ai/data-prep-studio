// Petits composants UI réutilisables + helpers de formatage.
import React from 'react'

export const fmt = (v, d = 3) => {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  if (typeof v !== 'number') return String(v)
  if (Number.isInteger(v)) return v.toLocaleString('fr-FR')
  const a = Math.abs(v)
  if (a >= 1e6 || (a < 1e-3 && a > 0)) return v.toExponential(2)
  return v.toLocaleString('fr-FR', { maximumFractionDigits: a >= 100 ? 1 : d })
}
export const pct = (v, d = 1) => (v === null || v === undefined ? '—' : `${v.toFixed(d)} %`)

export const KIND = {
  numeric: { label: 'Numérique', color: 'var(--teal)', icon: '#' },
  categorical: { label: 'Catégoriel', color: 'var(--violet)', icon: 'Aa' },
  datetime: { label: 'Date', color: 'var(--amber)', icon: '◷' },
  boolean: { label: 'Booléen', color: 'var(--blue)', icon: '◐' },
}
export const SEV = {
  ok: { label: 'OK', color: 'var(--green)' }, info: { label: 'Info', color: 'var(--blue)' },
  warn: { label: 'Attention', color: 'var(--amber)' }, crit: { label: 'Critique', color: 'var(--red)' },
}

export const KindBadge = ({ kind }) => {
  const k = KIND[kind] || KIND.categorical
  return <span className="badge" style={{ '--c': k.color }}><b>{k.icon}</b> {k.label}</span>
}
export const SevDot = ({ sev }) => <span className="dot" title={SEV[sev]?.label} style={{ background: SEV[sev]?.color }} />

export const Card = ({ title, sub, right, children, className = '' }) => (
  <section className={`card ${className}`}>
    {(title || right) && (
      <header className="card-h">
        <div><h3>{title}</h3>{sub && <p className="muted">{sub}</p>}</div>{right}
      </header>
    )}
    {children}
  </section>
)

export const Kpi = ({ label, value, hint, tone }) => (
  <div className={`kpi ${tone || ''}`}>
    <span className="kpi-l">{label}</span>
    <span className="kpi-v">{value}</span>
    {hint && <span className="kpi-h">{hint}</span>}
  </div>
)

export const Spinner = ({ label = 'Chargement…' }) => (
  <div className="spinner-wrap"><div className="spinner" /><span>{label}</span></div>
)

export const Empty = ({ children }) => <div className="empty">{children}</div>

export const Segmented = ({ value, onChange, options }) => (
  <div className="seg">
    {options.map((o) => (
      <button key={o.value} className={value === o.value ? 'on' : ''} onClick={() => onChange(o.value)}>{o.label}</button>
    ))}
  </div>
)
