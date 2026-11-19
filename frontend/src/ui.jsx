import React from 'react';
import {
  Hash,
  Type,
  ToggleLeft,
  Calendar,
  KeyRound,
  FileText,
  AlertTriangle,
  AlertOctagon,
  Info,
  CheckCircle2,
} from 'lucide-react';

export function TypeBadge({ type }) {
  const map = {
    numeric: { label: 'Numérique', icon: Hash, cls: 'badge-numeric' },
    categorical: { label: 'Catégorielle', icon: Type, cls: 'badge-categorical' },
    boolean: { label: 'Booléenne', icon: ToggleLeft, cls: 'badge-boolean' },
    date: { label: 'Date', icon: Calendar, cls: 'badge-date' },
    id: { label: 'Identifiant', icon: KeyRound, cls: 'badge-id' },
    text: { label: 'Texte', icon: FileText, cls: 'badge-text' },
  };
  const item = map[type] || map.categorical;
  const Icon = item.icon;

  return (
    <span className={`badge-type ${item.cls}`}>
      <Icon size={12} />
      {item.label}
    </span>
  );
}

export function SeverityBadge({ severity, title }) {
  let cls = 'badge-info';
  let Icon = Info;
  if (severity === 'critique') {
    cls = 'badge-critique';
    Icon = AlertOctagon;
  } else if (severity === 'attention') {
    cls = 'badge-attention';
    Icon = AlertTriangle;
  }

  return (
    <span className={`badge-type ${cls}`} title={title || severity}>
      <Icon size={12} />
      {title || severity}
    </span>
  );
}

export function KpiCard({ label, value, sub, icon: Icon, color }) {
  return (
    <div className="kpi-card">
      <div className="kpi-label">
        {Icon && <Icon size={14} style={{ color: color || 'var(--teal-400)' }} />}
        <span>{label}</span>
      </div>
      <div className="kpi-value">{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

export function Modal({ isOpen, onClose, title, children }) {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
          <h3 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.25rem', color: '#f8fafc' }}>{title}</h3>
          <button className="btn btn-ghost btn-sm" onClick={onClose}>Fermer</button>
        </div>
        {children}
      </div>
    </div>
  );
}

export function LoadingSkeleton({ height = 200 }) {
  return <div className="skeleton" style={{ width: '100%', height }} />;
}

export function EmptyState({ title, description, icon: Icon, action }) {
  return (
    <div className="glass-card" style={{ textAlign: 'center', padding: '3.5rem 2rem' }}>
      {Icon && (
        <div style={{ display: 'inline-flex', padding: '1rem', borderRadius: '50%', background: 'rgba(255,255,255,0.04)', marginBottom: '1rem', color: 'var(--text-muted)' }}>
          <Icon size={36} />
        </div>
      )}
      <h3 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.2rem', marginBottom: '0.4rem' }}>{title}</h3>
      <p style={{ color: 'var(--text-muted)', fontSize: '0.88rem', maxWidth: '420px', margin: '0 auto 1.5rem' }}>{description}</p>
      {action}
    </div>
  );
}
