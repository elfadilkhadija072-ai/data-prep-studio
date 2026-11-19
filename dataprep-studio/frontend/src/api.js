// Couche d'accès à l'API FastAPI (toutes les routes sont préfixées par /api).
const handle = async (r) => {
  if (!r.ok) {
    let m = r.statusText
    try { const j = await r.json(); m = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail) } catch { /* ignore */ }
    throw new Error(m)
  }
  return r.json()
}
const post = (u, body) => fetch(u, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body ?? {}) }).then(handle)

export const api = {
  upload: (file) => { const f = new FormData(); f.append('file', file); return fetch('/api/upload', { method: 'POST', body: f }).then(handle) },
  samples: () => fetch('/api/samples').then(handle),
  loadSample: (name) => fetch(`/api/samples/${encodeURIComponent(name)}`, { method: 'POST' }).then(handle),
  config: (id, cfg) => post(`/api/session/${id}/config`, cfg),
  eda: (id) => fetch(`/api/session/${id}/eda`).then(handle),
  corr: (id, method) => fetch(`/api/session/${id}/corr?method=${method}`).then(handle),
  scatter: (id, x, y, hue) => fetch(`/api/session/${id}/scatter?x=${encodeURIComponent(x)}&y=${encodeURIComponent(y)}${hue ? `&hue=${encodeURIComponent(hue)}` : ''}`).then(handle),
  groupby: (id, feature) => fetch(`/api/session/${id}/groupby?feature=${encodeURIComponent(feature)}`).then(handle),
  recs: (id) => fetch(`/api/session/${id}/recommendations`).then(handle),
  preview: (id, feature, plan) => post(`/api/session/${id}/preview`, { feature, plan }),
  apply: (id, choices, drop_duplicates) => post(`/api/session/${id}/apply`, { choices, drop_duplicates }),
  downloadUrl: (id, part, fmt) => `/api/session/${id}/download?part=${part}&fmt=${fmt}`,
  reportUrl: (id) => `/api/session/${id}/report`,
}
