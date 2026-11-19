import React from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ScatterChart,
  Scatter,
  ZAxis,
  Cell,
} from 'recharts';

export function CustomTooltip({ active, payload, label }) {
  if (active && payload && payload.length) {
    return (
      <div className="custom-chart-tooltip">
        <p style={{ fontWeight: 600, marginBottom: '0.2rem' }}>{label || payload[0].name}</p>
        <p style={{ color: 'var(--teal-400)' }}>
          Effectif : <strong>{payload[0].value.toLocaleString('fr-FR')}</strong>
        </p>
      </div>
    );
  }
  return null;
}

export function HistogramChart({ data, height = 260, color = '#2dd4bf' }) {
  if (!data || !data.length) return <div style={{ color: 'var(--text-muted)', textAlign: 'center', padding: '2rem' }}>Aucune donnée</div>;

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 25 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
          <XAxis
            dataKey="label"
            stroke="#64748b"
            fontSize={11}
            tickLine={false}
            angle={-30}
            textAnchor="end"
            interval={Math.ceil(data.length / 10)}
          />
          <YAxis stroke="#64748b" fontSize={11} tickLine={false} />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="count" fill={color} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function CategoryBarChart({ data, height = 260, color = '#a78bfa' }) {
  if (!data || !data.length) return null;

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <BarChart data={data} layout="vertical" margin={{ top: 10, right: 25, left: 40, bottom: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
          <XAxis type="number" stroke="#64748b" fontSize={11} tickLine={false} />
          <YAxis
            dataKey="category"
            type="category"
            stroke="#94a3b8"
            fontSize={11}
            tickLine={false}
            width={85}
          />
          <Tooltip
            formatter={(value, name, props) => [
              `${value} (${props.payload.percentage}%)`,
              'Effectif',
            ]}
            contentStyle={{ background: '#0f172a', borderColor: '#334155', borderRadius: 8, fontSize: 12 }}
          />
          <Bar dataKey="count" fill={color} radius={[0, 4, 4, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function BoxplotVisual({ stats }) {
  if (!stats) return null;

  const { min, low_whisker, q1, median, q3, high_whisker, max, outliers_sample } = stats;
  const range = max - min || 1;
  const toPct = (val) => Math.max(0, Math.min(100, ((val - min) / range) * 100));

  const pLow = toPct(low_whisker);
  const pQ1 = toPct(q1);
  const pMed = toPct(median);
  const pQ3 = toPct(q3);
  const pHigh = toPct(high_whisker);

  return (
    <div style={{ padding: '1rem 0' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
        <span>Min : {min}</span>
        <span>Médiane : {median}</span>
        <span>Max : {max}</span>
      </div>
      <div style={{ position: 'relative', height: 42, background: 'rgba(255,255,255,0.03)', borderRadius: 8, border: '1px solid var(--border-subtle)' }}>
        {/* Ligne médiane / moustache */}
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: `${pLow}%`,
            width: `${pHigh - pLow}%`,
            height: 2,
            background: '#64748b',
            transform: 'translateY(-50%)',
          }}
        />
        {/* Boîte IQR [Q1 - Q3] */}
        <div
          style={{
            position: 'absolute',
            top: '20%',
            bottom: '20%',
            left: `${pQ1}%`,
            width: `${pQ3 - pQ1}%`,
            background: 'rgba(45, 212, 191, 0.25)',
            border: '2px solid #2dd4bf',
            borderRadius: 4,
          }}
        />
        {/* Barre médiane */}
        <div
          style={{
            position: 'absolute',
            top: '15%',
            bottom: '15%',
            left: `${pMed}%`,
            width: 3,
            background: '#ffffff',
            boxShadow: '0 0 6px rgba(255,255,255,0.7)',
          }}
        />
      </div>
      {outliers_sample && outliers_sample.length > 0 && (
        <div style={{ marginTop: '0.6rem', fontSize: '0.75rem', color: 'var(--amber-400)' }}>
          {outliers_sample.length} valeurs extrêmes isolées détectées hors moustaches IQR.
        </div>
      )}
    </div>
  );
}

export function CorrelationHeatmap({ matrixData }) {
  if (!matrixData || !matrixData.columns || !matrixData.columns.length) {
    return <div style={{ color: 'var(--text-muted)', padding: '2rem' }}>Pas assez de variables numériques pour la corrélation.</div>;
  }

  const { columns, matrix } = matrixData;

  const getColor = (val) => {
    if (val === undefined || isNaN(val)) return 'rgba(255,255,255,0.03)';
    if (val > 0) {
      // De blanc/neutre vers rouge (#f87171)
      const alpha = Math.min(1, Math.max(0.05, Math.abs(val)));
      return `rgba(248, 113, 113, ${alpha * 0.85})`;
    } else {
      // De blanc/neutre vers bleu (#60a5fa)
      const alpha = Math.min(1, Math.max(0.05, Math.abs(val)));
      return `rgba(96, 165, 250, ${alpha * 0.85})`;
    }
  };

  return (
    <div style={{ overflowX: 'auto', padding: '0.5rem 0' }}>
      <table style={{ borderCollapse: 'collapse', fontSize: '0.78rem' }}>
        <thead>
          <tr>
            <th style={{ padding: '6px 8px', textAlign: 'left', color: 'var(--text-muted)' }}></th>
            {columns.map((c) => (
              <th key={c} style={{ padding: '6px 8px', color: 'var(--text-secondary)', fontWeight: 600, transform: 'rotate(-35deg)', height: 60, whiteSpace: 'nowrap' }}>
                {c.length > 12 ? c.substring(0, 10) + '…' : c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row) => (
            <tr key={row.col}>
              <td style={{ padding: '6px 10px', color: 'var(--text-primary)', fontWeight: 500, whiteSpace: 'nowrap' }}>
                {row.col}
              </td>
              {columns.map((c) => {
                const val = row[c];
                return (
                  <td
                    key={c}
                    style={{
                      background: getColor(val),
                      padding: '6px 8px',
                      textAlign: 'center',
                      fontFamily: 'monospace',
                      fontWeight: Math.abs(val) > 0.5 ? 700 : 400,
                      color: Math.abs(val) > 0.6 ? '#ffffff' : '#cbd5e1',
                      border: '1px solid rgba(255,255,255,0.04)',
                    }}
                    title={`${row.col} × ${c} : ${val}`}
                  >
                    {val !== undefined ? val.toFixed(2) : '-'}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ScatterPlotView({ points, xCol, yCol, height = 300 }) {
  if (!points || !points.length) return null;

  const targetClasses = Array.from(new Set(points.map((p) => p.target).filter(Boolean)));
  const colors = ['#2dd4bf', '#f87171', '#fbbf24', '#60a5fa', '#a78bfa'];

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
          <XAxis type="number" dataKey="x" name={xCol} stroke="#64748b" fontSize={11} />
          <YAxis type="number" dataKey="y" name={yCol} stroke="#64748b" fontSize={11} />
          <Tooltip
            cursor={{ strokeDasharray: '3 3' }}
            content={({ payload }) => {
              if (payload && payload.length) {
                const d = payload[0].payload;
                return (
                  <div className="custom-chart-tooltip">
                    <p>{xCol} : <strong>{d.x}</strong></p>
                    <p>{yCol} : <strong>{d.y}</strong></p>
                    {d.target !== undefined && <p>Cible : <strong>{d.target}</strong></p>}
                  </div>
                );
              }
              return null;
            }}
          />
          <Scatter data={points} fill="#2dd4bf">
            {points.map((entry, index) => {
              const clsIdx = targetClasses.indexOf(entry.target);
              const fill = clsIdx >= 0 ? colors[clsIdx % colors.length] : '#2dd4bf';
              return <Cell key={`cell-${index}`} fill={fill} />;
            })}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
