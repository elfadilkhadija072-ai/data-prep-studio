import React, { useState, useEffect } from 'react';
import {
  FileSpreadsheet,
  BarChart3,
  Sparkles,
  Download,
  CheckCircle2,
  ShieldCheck,
  Database,
  RefreshCw,
  Sliders,
  Layers,
  ArrowRight,
  BrainCircuit,
  Lock,
  LogOut,
  User,
  Terminal,
} from 'lucide-react';
import './styles.css';
import { verifyAuthToken, setAuthToken } from './api';
import { ImportTab } from './components/ImportTab';
import { EdaTab } from './components/EdaTab';
import { PrepTab } from './components/PrepTab';
import { ModelTab } from './components/ModelTab';
import { ExportTab } from './components/ExportTab';
import { AuthScreen } from './components/AuthScreen';
import { DebuggerModal } from './components/DebuggerModal';

export default function App() {
  const [currentUser, setCurrentUser] = useState(null);
  const [authChecking, setAuthChecking] = useState(true);
  const [showDebugger, setShowDebugger] = useState(false);

  const [activeTab, setActiveTab] = useState(1);
  const [sessionData, setSessionData] = useState(null);
  const [executionDone, setExecutionDone] = useState(false);

  // Vérification de la session d'authentification au démarrage
  useEffect(() => {
    verifyAuthToken()
      .then((data) => {
        if (data && data.user) {
          setCurrentUser(data.user);
        }
      })
      .catch(() => {
        setCurrentUser(null);
      })
      .finally(() => {
        setAuthChecking(false);
      });
  }, []);

  const handleLogout = () => {
    setAuthToken(null);
    setCurrentUser(null);
    setSessionData(null);
    setExecutionDone(false);
    setActiveTab(1);
  };

  const handleDatasetLoaded = (data) => {
    setSessionData(data);
    setExecutionDone(false);
  };

  const handleProceedToEda = () => {
    setActiveTab(2);
  };

  const handleReset = () => {
    if (window.confirm("Êtes-vous sûr de vouloir charger un nouveau dataset ? Toutes les étapes configurées seront réinitialisées.")) {
      setSessionData(null);
      setExecutionDone(false);
      setActiveTab(1);
    }
  };

  // Stepper à 5 étapes majeures
  const steps = [
    {
      id: 1,
      title: 'Import & Cible',
      subtitle: 'CSV / Excel & Problème ML',
      icon: FileSpreadsheet,
      enabled: true,
      completed: !!sessionData,
    },
    {
      id: 2,
      title: 'Analyse EDA',
      subtitle: 'KPIs, stats, corrélations',
      icon: BarChart3,
      enabled: !!sessionData,
      completed: activeTab > 2 || executionDone,
    },
    {
      id: 3,
      title: 'Preprocessing',
      subtitle: 'Plan & recommandations',
      icon: Sparkles,
      enabled: !!sessionData,
      completed: executionDone,
    },
    {
      id: 4,
      title: 'Modèles ML & Sécurité',
      subtitle: 'KNN, Régression, Audit Cyber',
      icon: BrainCircuit,
      enabled: !!sessionData && executionDone,
      completed: executionDone && activeTab > 4,
    },
    {
      id: 5,
      title: 'Exécution & Export',
      subtitle: 'Datasets étanches, Rapport Excel',
      icon: Download,
      enabled: !!sessionData && executionDone,
      completed: executionDone,
    },
  ];

  // Si pas authentifié, afficher l'écran de pré-authentification
  if (authChecking) {
    return (
      <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-base)' }}>
        <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
          <ShieldCheck size={36} color="var(--teal-400)" className="spin-animation" style={{ margin: '0 auto 1rem' }} />
          <p>Initialisation du protocole de sécurité DataPrep Studio…</p>
        </div>
      </div>
    );
  }

  if (!currentUser) {
    return <AuthScreen onAuthenticated={(user) => setCurrentUser(user)} />;
  }

  return (
    <div className="app-container">
      {/* Header avec Navigation & Branding */}
      <header className="header">
        <div className="header-left">
          <div className="logo-box">
            <Layers size={22} color="#2dd4bf" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <h1 className="header-title">DataPrep Studio</h1>
              <span className="badge-pill">
                <ShieldCheck size={13} color="#2dd4bf" />
                Anti-Leakage & Cyber ML
              </span>
            </div>
            <p className="header-subtitle">
              Nettoyage, modélisation et préparation certifiée sans fuite de données
            </p>
          </div>
        </div>

        {/* Section Utilisateur & Dataset Actif */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          {sessionData && (
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.8rem',
                background: 'rgba(255, 255, 255, 0.04)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                borderRadius: 'var(--radius-md)',
                padding: '0.45rem 0.9rem',
                fontSize: '0.82rem',
              }}
            >
              <Database size={15} color="#38bdf8" />
              <div>
                <span style={{ fontWeight: 600, color: 'var(--text-main)' }}>
                  {sessionData.filename || 'Dataset'}
                </span>
                <span style={{ color: 'var(--text-muted)', marginLeft: '0.4rem' }}>
                  ({sessionData.row_count?.toLocaleString()} lig. × {sessionData.column_count} col.)
                </span>
                {sessionData.target_col && (
                  <span style={{ color: 'var(--accent-teal)', marginLeft: '0.4rem', fontWeight: 500 }}>
                    • Cible: {sessionData.target_col} ({sessionData.problem_type})
                  </span>
                )}
              </div>
              <button
                className="btn btn-secondary"
                style={{ padding: '0.25rem 0.5rem', fontSize: '0.75rem', height: 'auto', marginLeft: '0.4rem' }}
                onClick={handleReset}
                title="Changer de dataset"
              >
                <RefreshCw size={12} />
                Nouveau
              </button>
            </div>
          )}

          {/* Badge Utilisateur connecté */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.6rem',
              background: 'rgba(45, 212, 191, 0.08)',
              border: '1px solid rgba(45, 212, 191, 0.25)',
              borderRadius: 'var(--radius-md)',
              padding: '0.4rem 0.75rem',
              fontSize: '0.8rem',
            }}
          >
            <User size={14} color="var(--teal-400)" />
            <div>
              <div style={{ fontWeight: 600, color: '#f8fafc', lineHeight: 1.2 }}>{currentUser.name}</div>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{currentUser.role}</div>
            </div>
            <button
              onClick={handleLogout}
              className="btn btn-ghost"
              style={{ padding: '0.2rem', marginLeft: '0.4rem' }}
              title="Se déconnecter"
            >
              <LogOut size={13} color="var(--red-400)" />
            </button>
          </div>
        </div>
      </header>

      {/* Stepper Pro à 5 étapes */}
      <nav className="stepper-nav">
        {steps.map((step) => {
          const Icon = step.icon;
          const isActive = activeTab === step.id;
          const isDone = step.completed && !isActive;

          return (
            <button
              key={step.id}
              className={`step-btn ${isActive ? 'active' : ''} ${isDone ? 'completed' : ''}`}
              disabled={!step.enabled}
              onClick={() => step.enabled && setActiveTab(step.id)}
            >
              <div className="step-circle">
                {isDone ? <CheckCircle2 size={16} /> : <Icon size={16} />}
              </div>
              <div className="step-content">
                <span className="step-title">
                  {step.id}. {step.title}
                </span>
                <span className="step-desc">{step.subtitle}</span>
              </div>
            </button>
          );
        })}
      </nav>

      {/* Contenu Principal par Onglet */}
      <main className="tab-viewport">
        {activeTab === 1 && (
          <ImportTab
            sessionData={sessionData}
            onDatasetLoaded={handleDatasetLoaded}
            onProceedToEda={handleProceedToEda}
          />
        )}

        {activeTab === 2 && sessionData && (
          <EdaTab
            sessionId={sessionData.session_id}
            sessionData={sessionData}
            onProceedToPrep={() => setActiveTab(3)}
          />
        )}

        {activeTab === 3 && sessionData && (
          <PrepTab
            sessionId={sessionData.session_id}
            onExecutionComplete={(results) => {
              setExecutionDone(true);
              setActiveTab(4); // Passe directement à l'onglet 4 : Modèles ML & Sécurité
            }}
          />
        )}

        {activeTab === 4 && sessionData && (
          <ModelTab
            sessionId={sessionData.session_id}
            sessionData={sessionData}
            onProceedToExport={() => setActiveTab(5)}
          />
        )}

        {activeTab === 5 && sessionData && (
          <ExportTab
            sessionId={sessionData.session_id}
            onBackToPrep={() => setActiveTab(4)}
          />
        )}
      </main>

      {/* Bouton Flottant Débogueur & Diagnostics */}
      {sessionData && (
        <button
          className="floating-debugger-btn"
          onClick={() => setShowDebugger(true)}
          title="Ouvrir la console de diagnostic et débogage"
        >
          <Terminal size={14} />
          <span>Diagnostic & Débogueur</span>
        </button>
      )}

      {/* Modal du Débogueur Système */}
      <DebuggerModal
        isOpen={showDebugger}
        onClose={() => setShowDebugger(false)}
        sessionId={sessionData?.session_id}
      />

      {/* Footer permanent */}
      <footer
        style={{
          marginTop: '3.5rem',
          padding: '1.25rem 0',
          borderTop: '1px solid rgba(255, 255, 255, 0.06)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          color: 'var(--text-muted)',
          fontSize: '0.8rem',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <ShieldCheck size={15} color="#2dd4bf" />
          <span>DataPrep Studio • Moteur d'isolation anti-data-leakage & certification cyber certifié</span>
        </div>
        <div>
          <span>KNN, Régression, Random Forest • Empreinte SHA-256 HMAC • Export Excel XlsxWriter</span>
        </div>
      </footer>
    </div>
  );
}
