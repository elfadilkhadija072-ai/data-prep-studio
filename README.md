# DataPrep Studio — Plateforme Intelligente de Préparation ML

> **DataPrep Studio** est une application Web complète, moderne et robuste conçue pour automatiser l'analyse exploratoire (EDA), le nettoyage et la préparation de données tabulaires pour le Machine Learning supervisé, avec une **garantie stricte d'absence de fuite de données (Anti-Data-Leakage)**.

---

## 🌟 Points Forts

- **Une seule commande pour tout démarrer** : `python run.py` lance l'API FastAPI et sert l'interface React compilée sur `http://localhost:8501`.
- **Zéro Fuite de Données (Anti-Data-Leakage)** :
  - Déduplication préalable ;
  - Scission Train / Test (*Train-Test Split*) effectuée **avant toute transformation** ;
  - Ajustement (*fit*) de toutes les statistiques (moyennes, médianes, scalers, encodeurs, seuils d'outliers) **sur le train uniquement** ;
  - Application (*transform*) stricte sur le test ;
  - Suppression des observations aberrantes sur le train uniquement ;
  - Protection absolue de la variable cible $y$ (non mise à l'échelle).
- **Système Expert de Recommandation** :
  - Détection automatique des anomalies avec sévérité visuelle (`DANGER`, `WARNING`, `INFO`) ;
  - Plan de transformation recommandé avec **justification statistique chiffrée** pour chaque variable ;
  - Prévisualisation interactive Avant / Après en temps réel avec histogramme comparatif.
- **Rapports et Exports Complets** :
  - Datasets nettoyés : `dataset_prepared.csv`, `dataset_prepared.xlsx` (avec colonne indicatrice `_split`), `train.csv`, `test.csv` ;
  - **Rapport d'audit Microsoft Excel professionnel** généré via XlsxWriter (5 onglets thématiques formatés : Résumé, Features, Détail transformations, Colonnes finales, Corrélations).
- **Interface Utilisateur Moderne** : Thème sombre Glassmorphism sur-mesure (React 18, Vite, Recharts, Lucide Icons, typographie Inter).

---

## 🚀 Démarrage Rapide

### Prérequis
- **Python 3.11+** (Python 3.12 supporté)
- Dépendances Python installées dans l'environnement virtuel.

### 1. Lancement en une seule commande (Production / Démo)
À la racine du projet :
```bash
python run.py
```
Cette commande :
1. Détecte automatiquement l'environnement Node.js / Python ;
2. Compile le frontend si nécessaire ;
3. Démarre le serveur FastAPI sur **http://localhost:8501** ;
4. Ouvre ou sert l'application prête à l'emploi.

### 2. Mode Développement (Optionnel)
Pour développer sur le frontend avec rechargement à chaud (*Hot Module Replacement*) :
```bash
# Terminal 1 : Backend FastAPI
python backend/run.py

# Terminal 2 : Frontend Vite Dev Server
cd frontend
npm run dev
```
L'interface de développement sera accessible sur **http://localhost:5173** avec proxy automatique de `/api` vers le port `8501`.

---

## 🧪 Test Automatisé d'Anti-Data-Leakage

Un test unitaire mathématique dédié valide l'étanchéité absolue du pipeline :
```bash
python tests/test_leakage.py
```
Le test vérifie formellement :
1. Que le `StandardScaler` n'est pas contaminé par les données du test ($\mu_{\text{train}} = 0$, $\mu_{\text{test}} \neq 0$) ;
2. Que la variable cible $y$ n'est jamais mise à l'échelle ;
3. Que l'imputation médiane du jeu de test utilise la médiane calculée sur le jeu d'entraînement uniquement.

---

## 📂 Datasets de Démonstration Inclus

Deux jeux de données réalistes sont intégrés et chargeables en un clic dans l'onglet **Import** :
1. **Fraude bancaire (`fraude_bancaire.csv`)** : 1 005 transactions avec déséquilibre de classe (3.1% de fraudes), montants asymétriques, dates, identifiants, valeurs manquantes et outliers.
2. **Prix immobilier (`prix_immobilier.csv`)** : 854 biens immobiliers pour un problème de régression supervisée avec surfaces, nombres de pièces, dates, types de chauffage et prix cibles.

---

## 🏗️ Structure du Projet

```
projet-app/
├── run.py                          # Lanceur principal en une commande
├── tests/                          # Tests d'étanchéité anti-leakage
├── docs/STRUCTURE.md               # Documentation détaillée de chaque fichier
├── backend/
│   ├── run.py                      # Lanceur API backend
│   ├── main.py                     # Routes FastAPI REST
│   ├── requirements.txt            # Dépendances Python
│   ├── data/                       # Datasets de démo et générateur
│   └── core/
│       ├── common.py               # Cache LRU de session et schémas Pydantic
│       ├── loader.py               # Ingestion CSV/Excel & détection automatique
│       ├── eda.py                  # Moteur d'analyse exploratoire & corrélations
│       ├── analysis.py             # Détecteur d'anomalies & moteur de règles
│       ├── pipeline.py             # Moteur d'exécution strict anti-data-leakage
│       └── report.py               # Générateur de rapport Excel XlsxWriter
└── frontend/
    ├── package.json / vite.config.js
    ├── index.html
    └── src/
        ├── main.jsx / App.jsx      # Montage React & Navigation
        ├── api.js                  # Client API REST
        ├── styles.css              # Thème Glassmorphism moderne
        ├── ui.jsx / charts.jsx     # Composants d'interface et graphiques Recharts
        └── components/
            ├── ImportTab.jsx       # Étape 1 : Ingestion & Cible ML
            ├── EdaTab.jsx          # Étape 2 : Analyse exploratoire & corrélations
            ├── PrepTab.jsx         # Étape 3 : Plan de nettoyage & Avant/Après
            └── ExportTab.jsx       # Étape 4 : Téléchargements & Audit
```

Pour une explication exhaustive de chaque fichier et de l'architecture logicielle, consultez [docs/STRUCTURE.md](file:///c:/Users/TUF/OneDrive/Desktop/projet-app/docs/STRUCTURE.md).

---

## 🛡️ Bonnes Pratiques ML Respectées

- ✅ **Pas de fuite d'information** : aucune statistique globale n'est calculée sur l'ensemble complet avant la scission train/test.
- ✅ **Stratification** : respect des proportions de classes en classification.
- ✅ **Traitement des outliers robuste** : winsorisation ou suppression des lignes uniquement sur l'ensemble d'entraînement.
- ✅ **Reproductibilité** : graine aléatoire fixée (`random_state=42`).
- ✅ **Auditabilité** : traçabilité complète de chaque imputation et seuil dans le rapport Excel téléchargeable.
