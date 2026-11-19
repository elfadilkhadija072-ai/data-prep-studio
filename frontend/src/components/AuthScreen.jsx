import React, { useState } from 'react';
import {
  ShieldCheck,
  Lock,
  User,
  KeyRound,
  ArrowRight,
  Sparkles,
  AlertCircle,
  Eye,
  EyeOff,
  CheckCircle2,
  Cpu,
} from 'lucide-react';
import { loginUser } from '../api';

export function AuthScreen({ onAuthenticated }) {
  const [username, setUsername] = useState('admin');
  const [password, setPassword] = useState('admin123');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const demoAccounts = [
    { label: 'Administrateur', user: 'admin', pwd: 'admin123', role: 'Accès Complet & Config' },
    { label: 'Data Scientist', user: 'scientist', pwd: 'mlsec2025', role: 'Entraînement & Sécurité' },
    { label: 'Auditeur Cyber', user: 'auditeur', pwd: 'audit2025', role: 'Inspection & Rapports' },
  ];

  const handleQuickFill = (acc) => {
    setUsername(acc.user);
    setPassword(acc.pwd);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!username.trim() || !password) {
      setError('Veuillez renseigner un nom d’utilisateur et un mot de passe.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await loginUser(username, password);
      onAuthenticated(data.user);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-overlay">
      <div className="auth-container">
        {/* Glow orb décoratif */}
        <div className="auth-glow-orb" />

        <div className="glass-card auth-card">
          {/* Logo et En-tête */}
          <div style={{ textAlign: 'center', marginBottom: '1.75rem' }}>
            <div className="auth-logo-badge">
              <ShieldCheck size={32} color="#2dd4bf" />
            </div>
            <h2 style={{ fontFamily: 'var(--font-serif)', fontSize: '1.85rem', color: '#ffffff', marginBottom: '0.4rem' }}>
              DataPrep Studio
            </h2>
            <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', padding: '0.2rem 0.75rem', borderRadius: 9999, background: 'rgba(45, 212, 191, 0.1)', border: '1px solid rgba(45, 212, 191, 0.25)', fontSize: '0.78rem', color: 'var(--teal-400)', fontWeight: 600 }}>
              <Lock size={11} />
              Accès Sécurisé • Authentification Pré-Session
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: '0.6rem' }}>
              Connexion requise pour accéder au moteur de préparation et aux modèles ML étanches.
            </p>
          </div>

          {/* Message d'erreur */}
          {error && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', padding: '0.8rem 1rem', background: 'var(--red-bg)', color: 'var(--red-400)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(248,113,113,0.3)', fontSize: '0.82rem', marginBottom: '1.25rem' }}>
              <AlertCircle size={16} style={{ flexShrink: 0 }} />
              <span>{error}</span>
            </div>
          )}

          {/* Formulaire de connexion */}
          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.1rem' }}>
            <div className="control-group">
              <label className="control-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <User size={13} color="var(--teal-400)" />
                Identifiant Utilisateur
              </label>
              <input
                type="text"
                className="form-input"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Ex : admin, scientist, auditeur"
                autoFocus
              />
            </div>

            <div className="control-group">
              <label className="control-label" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <KeyRound size={13} color="var(--teal-400)" />
                Mot de Passe
              </label>
              <div style={{ position: 'relative' }}>
                <input
                  type={showPassword ? 'text' : 'password'}
                  className="form-input"
                  style={{ paddingRight: '2.5rem' }}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  style={{
                    position: 'absolute',
                    right: '0.75rem',
                    top: '50%',
                    transform: 'translateY(-50%)',
                    background: 'none',
                    border: 'none',
                    color: 'var(--text-muted)',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                  }}
                  title={showPassword ? 'Masquer' : 'Afficher'}
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
            </div>

            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading}
              style={{ width: '100%', justifyContent: 'center', padding: '0.75rem', marginTop: '0.5rem', fontSize: '0.95rem' }}
            >
              {loading ? (
                <span>Vérification des accréditations…</span>
              ) : (
                <>
                  <span>Ouvrir l'Espace Sécurisé</span>
                  <ArrowRight size={16} />
                </>
              )}
            </button>
          </form>

          {/* Sélecteur de comptes démo instantanés */}
          <div style={{ marginTop: '1.75rem', paddingTop: '1.25rem', borderTop: '1px solid var(--border-subtle)' }}>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', display: 'block', marginBottom: '0.6rem', textAlign: 'center' }}>
              Accès rapide démo en un clic :
            </span>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.5rem' }}>
              {demoAccounts.map((acc) => (
                <button
                  key={acc.user}
                  type="button"
                  className="btn btn-secondary btn-sm"
                  onClick={() => handleQuickFill(acc)}
                  style={{
                    flexDirection: 'column',
                    padding: '0.5rem 0.25rem',
                    height: 'auto',
                    textAlign: 'center',
                    fontSize: '0.74rem',
                    borderColor: username === acc.user ? 'var(--teal-400)' : 'var(--border-subtle)',
                    background: username === acc.user ? 'rgba(45, 212, 191, 0.1)' : 'transparent',
                  }}
                >
                  <span style={{ fontWeight: 600, color: '#f8fafc' }}>{acc.label}</span>
                  <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{acc.user}</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
