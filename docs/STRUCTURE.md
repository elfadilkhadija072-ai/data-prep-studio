# DataPrep Studio — Architecture et Structure du Code

Ce document détaille l'organisation complète du projet **DataPrep Studio**, le rôle précis de chaque composant, ainsi que le flux de données garantissant une étanchéité totale contre les fuites de données (**Anti-Data-Leakage**).

---

## 1. Vue d'Ensemble de l'Architecture

DataPrep Studio est conçu selon une architecture découplée et réactive :
- **Frontend SPA** : Développé en React 18 avec Vite, composants UI modulaires, styles CSS modernes (Glassmorphism sombre, variables CSS, micro-interactions) et visualisations interactives (Recharts).
- **Backend API** : Développé en Python 3.11+ avec FastAPI, Pandas, NumPy, SciPy, Scikit-Learn et XlsxWriter.
- **Serveur Unifié** : Un lanceur unique (`python run.py`) compile ou vérifie le build frontend et sert l'API REST ainsi que l'application React compilée sur le port `8501`.

```
projet-app/
│
├── run.py                          # Lanceur racine en une commande (python run.py)
├── tests/
│   └── test_leakage.py             # Test unitaire d'étanchéité anti-data-leakage
├── docs/
│   └── STRUCTURE.md                # Documentation architecturale (ce fichier)
├── README.md                       # Guide de démarrage rapide et fonctionnalités
│
├── backend/
│   ├── run.py                      # Script de démarrage direct du backend
│   ├── main.py                     # API FastAPI & exposition des routes REST
│   ├── requirements.txt            # Dépendances Python nécessaires
│   ├── data/
│   │   ├── generate_datasets.py    # Générateur reproductible des datasets de démo
│   │   ├── fraude_bancaire.csv     # Dataset démo : classification supervisée
│   │   └── prix_immobilier.csv     # Dataset démo : régression supervisée
│   │
│   └── core/
│       ├── common.py               # Gestionnaire de sessions LRU & modèles Pydantic
│       ├── loader.py               # Ingestion CSV/Excel, parsing de types & cibles
│       ├── eda.py                  # Moteur statistique (KPI, corrélations, boxplots)
│       ├── analysis.py             # Détecteur d'anomalies, règles & justifications
│       ├── pipeline.py             # Moteur de transformation strict anti-fuite
│       └── report.py               # Générateur du classeur Excel stylé (XlsxWriter)
│
└── frontend/
    ├── package.json                # Dépendances React, Lucide et Recharts
    ├── vite.config.js              # Configuration Vite avec proxy /api vers :8501
    ├── index.html                  # Point d'entrée HTML avec polices Inter et Playfair
    ├── src/
    │   ├── main.jsx                # Point de montage ReactDOM
    │   ├── App.jsx                 # Conteneur principal, navigation par onglets
    │   ├── api.js                  # Client API Axios/Fetch vers les routes backend
    │   ├── styles.css              # Thème sombre Glassmorphism sur-mesure
    │   ├── ui.jsx                  # Composants réutilisables (Badges, Modals, Skeleton)
    │   ├── charts.jsx              # Composants graphiques Recharts (Histo, Heatmap, Scatter)
    │   └── components/
    │       ├── ImportTab.jsx       # Étape 1 : Glisser-déposer, config supervisée & aperçu
    │       ├── EdaTab.jsx          # Étape 2 : KPI, tableau des features, distributions
    │       ├── PrepTab.jsx         # Étape 3 : Anomalies, plan recommandé & customisation
    │       └── ExportTab.jsx       # Étape 4 : Téléchargements et synthèse d'exécution
```

---

## 2. Rôle Détaillé des Fichiers Backend

### `backend/main.py`
Point d'entrée de l'API web FastAPI.
- Configure les middlewares CORS pour le développement (`localhost:5173`).
- Enregistre l'ensemble des routes réparties par phase du cycle de vie du projet :
  - `/api/upload` : Réception de fichiers CSV/Excel ou chargement instantané de démos (`fraud`, `housing`).
  - `/api/session` : Récupération de l'état actuel de la session.
  - `/api/configure` : Sauvegarde des métadonnées ML (supervisé ou non, colonne cible, type de problème, test_size).
  - `/api/eda/kpis`, `/api/eda/features`, `/api/eda/correlations`, `/api/eda/distribution`, `/api/eda/scatter`, `/api/eda/quality` : Endpoints statistiques pour l'onglet d'analyse exploratoire.
  - `/api/prep/plan`, `/api/prep/preview-feature`, `/api/prep/execute` : Endpoints pour la recommandation, la prévisualisation avant/après et l'exécution du pipeline de nettoyage.
  - `/api/export/summary`, `/api/export/download/*` : Endpoints fournissant le bilan final et les fichiers téléchargeables (`full_csv`, `full_xlsx`, `train_csv`, `test_csv`, `report_excel`).
- Sert les fichiers statiques de l'application frontend compilée (`frontend/dist`) en fallback SPA.

### `backend/core/common.py`
Structures de données transversales et persistance en mémoire.
- `SessionManager` : Gestionnaire de sessions avec cache LRU garantissant l'isolation des données entre utilisateurs et nettoyant les sessions inactives.
- Modèles Pydantic validant les requêtes entrantes (`ConfigureRequest`, `FeatureStepConfig`, `ExecutePipelineRequest`, `PreviewFeatureRequest`).
- Liste standardisée des jetons de valeurs manquantes (`NA_TOKENS` : `"", "NA", "null", "?", "nan", "NaN", "None"`).

### `backend/core/loader.py`
Module d'importation et d'inférence sémantique.
- **Détection automatique d'encodage** : Analyse UTF-8, Latin-1, CP1252.
- **Détection automatique de séparateur** : `csv.Sniffer` testant virgule, point-virgule, tabulation ou barre verticale.
- **Conversion intelligente** : Nettoyage des chaînes numériques avec espaces insécables et virgules décimales (`1 250,50 €` -> `1250.50`), parsing robuste de dates (`pd.to_datetime`).
- **Typage sémantique** : Classification automatique de chaque variable en `numeric`, `categorical`, `boolean`, `date`, `id` ou `text`.
- **Suggestion de cible et de problème** : Analyse de la position de la dernière colonne, des mots-clés (`target`, `label`, `fraud`, `price`, `statut`) et détection automatique entre classification (faible cardinalité ou booléen) et régression (numérique continue).

### `backend/core/eda.py`
Moteur d'analyse exploratoire des données.
- **KPIs globaux** : Nombre total d'observations, de features, taux global de valeurs manquantes, lignes dupliquées, consommation mémoire estimée et déséquilibre de classes.
- **Métriques univariées complètes** : Moyenne, écart-type, quartiles (Q25, Q50, Q75), asymétrie (*skewness*), aplatissement (*kurtosis*), proportion d'outliers calculée par l'écart interquartile ($1.5 \times \text{IQR}$).
- **Mesures d'association avec la variable cible** :
  - Numérique vs Numérique : Coefficient de Pearson ($|r|$) et Spearman.
  - Numérique vs Catégorielle : Rapport de corrélation $\eta$ (*eta*).
  - Catégorielle vs Catégorielle : V de Cramér dérivé du test du $\chi^2$.
- **Histogrammes adaptatifs** : Bins réguliers calculés avec découpage P1–P99 pour isoler les queues de distribution extrêmes.
- **Matrice de corrélation bivariée** : Support de Pearson, Spearman et Kendall.

### `backend/core/analysis.py`
Système expert de détection d'anomalies et moteur de recommandations.
- Analyse chaque variable et identifie avec un niveau de sévérité (`danger`, `warning`, `info`) :
  - Fort taux de valeurs manquantes ($> 50\% \implies$ recommandation de suppression) ;
  - Variable constante ou quasi-constante ($> 95\%$ valeur modale) ;
  - Colonne identifiant ou cardinalité presque unique ($> 95\%$ uniques) ;
  - Redondance colinéaire ($|r| \ge 0.95$ avec une autre feature) ;
  - Fuite de cible suspectée ($|r| \ge 0.98$ ou $\eta \ge 0.98$) ;
  - Forte proportion d'outliers ($> 5\%$) ;
  - Asymétrie prononcée ($|\text{skew}| > 1.5$) ;
  - Casse ou espaces incohérents dans les chaînes de caractères.
- Génère un plan de transformation recommandé doté d'une **justification statistique chiffrée** pour chaque colonne.
- Fournit la fonction `compute_preview_for_feature` permettant d'afficher en temps réel le graphe et les statistiques comparatives avant / après modification d'un paramètre.

### `backend/core/pipeline.py`
Cœur d'exécution garantissant l'étanchéité stricte contre la fuite de données (**Anti-Data-Leakage**).
L'ordre séquentiel est inaltérable :
1. **Suppression des doublons** sur le dataset brut complet si demandée.
2. **Scission Train / Test avant toute transformation** :
   - `test_size` configurable (par défaut 20%).
   - Stratification automatique en classification pour préserver les proportions de classes rares.
   - Graine aléatoire fixée (`random_state=42`) pour une parfaite reproductibilité.
3. **Apprentissage (*Fit*) exclusif sur l'ensemble d'entraînement (*Train*)** :
   - Moyennes et médianes calculées sur le train uniquement.
   - Fréquences de modalités et listes de One-Hot calculées sur le train uniquement.
   - Paramètres de mise à l'échelle (`StandardScaler`, `MinMaxScaler`, `RobustScaler`) fittés uniquement sur le train.
   - Seuils d'écrêtage d'outliers (bornes IQR ou percentiles) estimés sur le train uniquement.
4. **Application (*Transform*) symétrique sur le Train et le Test** :
   - Remplacement des inconnues ou modalités inédites par `<OTHER>` ou 0.
   - Application rigoureuse des moyennes/écarts-types du train sur le test.
5. **Nettoyage des outliers résiduels** :
   - Si l'action choisie est le retrait des lignes, les lignes contenant des outliers sont supprimées **uniquement de l'ensemble d'entraînement**. L'ensemble de test n'est jamais amputé de ses outliers afin d'évaluer les modèles en conditions réelles.
6. **Préservation absolue de la variable cible** :
   - La cible n'est jamais mise à l'échelle ni altérée, garantissant l'interprétabilité des métriques métier d'évaluation.

### `backend/core/report.py`
Générateur de rapport d'audit exhaustif au format Microsoft Excel (`.xlsx`) via la librairie **XlsxWriter**.
Produit un classeur soigné comportant une palette sombre et professionnelle découpée en 5 feuilles de calcul :
1. **Résumé** : Métriques clés du dataset, configuration ML, compteurs de lignes avant/après nettoyage et synthèse du split.
2. **Features** : Tableau exhaustif listant chaque colonne, son type, ses anomalies détectées, sa recommandation, la décision finale de l'utilisateur et les transformations appliquées.
3. **Détail transformations** : Journal technique précis répertoriant les valeurs exactes imputées, les bornes numériques d'écrêtage calculées et les dimensions matricielles résultantes.
4. **Colonnes finales** : Liste de toutes les colonnes présentes dans le jeu de données d'entraînement prêt pour l'entraînement d'algorithmes (Random Forest, XGBoost, LightGBM, Réseaux de neurones, etc.).
5. **Corrélations** : Matrice de corrélation complète avec mise en forme conditionnelle visuelle des associations fortes.

### `backend/data/generate_datasets.py`
Génère deux jeux de données de démonstration réalistes :
- `fraude_bancaire.csv` : 1005 transactions avec déséquilibre de classe (3.1% de fraudes), montants asymétriques, dates, identifiants de terminaux, statuts catégoriels et valeurs manquantes.
- `prix_immobilier.csv` : 854 transactions immobilières pour modélisation de régression avec surfaces, nombres de pièces, dates de construction, types de chauffage et prix asymétriques.

---

## 3. Rôle Détaillé des Fichiers Frontend

### `frontend/src/App.jsx`
Composant racine de l'application :
- Conserve l'état global (`sessionData`, `activeTab`, `executionDone`).
- Pilote le bandeau d'en-tête (Branding avec badge de certification anti-leakage, résumé du dataset actif, bouton de réinitialisation).
- Intègre la barre de progression (*Stepper*) à 4 étapes indiquant les états actifs, complétés et verrouillés.
- Affiche dynamiquement le composant associé à l'étape en cours.

### `frontend/src/styles.css`
Feuille de style centralisée :
- Variables CSS pour les couleurs (fonds sombres `#0a0d14`, teal accent `#2dd4bf`, bleu ciel `#38bdf8`, violet `#818cf8`, alertes ambre `#fbbf24` et rouge `#f87171`).
- Effets de cartes en verre dépoli (*glassmorphism*) avec `backdrop-filter: blur(16px)` et bordures semi-transparentes.
- Responsive design, stylisation des tableaux avec en-têtes collants (*sticky headers*), barres d'actions ancrées et animations fluides.

### `frontend/src/ui.jsx`
Composants atomiques réutilisables :
- `TypeBadge` : Pastilles colorées selon le type sémantique (`NUM`, `CAT`, `BOOL`, `DATE`, `ID`, `TEXT`).
- `SeverityBadge` : Badges visuels pour les anomalies (`DANGER`, `WARNING`, `INFO`, `OK`).
- `KpiCard` : Cartes d'indicateurs clés de performance avec icône et libellé.
- `Modal` : Fenêtre modale accessible avec fond assombri pour les prévisualisations.
- `LoadingSkeleton` : Barres de chargement animées pulsées pendant les calculs backend.

### `frontend/src/charts.jsx`
Composants de data visualisation interactifs utilisant **Recharts** :
- `HistogramChart` : Affiche les distributions de fréquences avec infobulles personnalisées.
- `CategoryBarChart` : Histogramme horizontal des fréquences de modalités pour variables qualitatives.
- `BoxplotVisual` : Rendu des quartiles (Min, Q25, Médiane, Q75, Max) avec bornes IQR.
- `CorrelationHeatmap` : Grille matricielle de corrélation avec coloration en fonction de l'intensité de liaison.
- `ScatterPlotView` : Nuage de points interactif bivarié avec coloration optionnelle selon la classe cible.

### `frontend/src/components/ImportTab.jsx`
Étape 1 : Ingestion des données.
- Zone de glisser-déposer de fichiers CSV ou Excel.
- Cartes de chargement instantané en un clic pour les deux datasets de démonstration.
- Formulaire de configuration du problème ML :
  - Bascule Supervisé / Non supervisé ;
  - Sélecteur de variable cible avec suggestion intelligente mise en avant ;
  - Sélecteur du type de problème (Classification vs Régression) ;
  - Curseur du ratio de l'ensemble de test ($10\%$ à $40\%$) ;
  - Case à cocher pour la déduplication automatique.
- Tableau d'aperçu des premières lignes avec pastilles de typage par colonne.

### `frontend/src/components/EdaTab.jsx`
Étape 2 : Analyse exploratoire approfondie.
- Cartes KPI synthétiques (observations, features, taux de complétude, mémoire, etc.).
- Tableau statistique complet et triable par n'importe quelle colonne (nom, type, % manquants, % uniques, moyenne, écart-type, quartiles, skewness, kurtosis, % outliers, association cible).
- Panneau d'inspection univariée de la colonne sélectionnée avec zoom P1–P99.
- Rapport de qualité des données identifiant les colonnes critiques.
- Panneau de corrélations bivariées et nuage de points avec sélecteur d'axes X et Y.

### `frontend/src/components/PrepTab.jsx`
Étape 3 : Plan de nettoyage et prévisualisation.
- Liste extensible de toutes les variables avec pastilles de sévérité d'anomalies détectées.
- Recommandation formulée en langage clair accompagnée d'une justification statistique chiffrée.
- Contrôles granulaires pour chaque variable permettant à l'utilisateur de modifier ou surcharger :
  - Suppression de la colonne ;
  - Nettoyage textuel (espaces, mise en minuscules) ;
  - Méthode d'imputation (médiane, moyenne, mode, valeur constante) ;
  - Traitement des outliers (winsorisation IQR, percentiles, suppression des lignes train, ou conservation) ;
  - Transformation de puissance (Log1p, Racine carrée) ;
  - Mise à l'échelle (StandardScaler, MinMaxScaler, RobustScaler) ;
  - Encodage (One-Hot avec regroupement des catégories rares, Ordinal, Fréquence, Label).
- Bouton « Tout accepter » appliquant d'un coup l'ensemble des recommandations du système expert.
- Modal de prévisualisation Avant / Après en temps réel permettant de vérifier l'effet sur la distribution avant de lancer le pipeline.
- Barre d'action fixe en bas d'écran avec compteurs d'état et bouton d'exécution.

### `frontend/src/components/ExportTab.jsx`
Étape 4 : Résultats finaux et téléchargements.
- Bilan chiffré des dimensions initiales et des dimensions finales post-encodage.
- Journal d'audit certifiant le respect rigoureux des bonnes pratiques anti-data-leakage.
- Boutons de téléchargement direct en un clic :
  - Dataset complet préparé (CSV ou Excel avec colonne indicatrice `_split`) ;
  - Ensemble d'entraînement (`train.csv`) prêt pour modélisation ;
  - Ensemble de test (`test.csv`) strictement isolé ;
  - Rapport Excel d'audit technique à 5 onglets généré avec XlsxWriter.
- Tableau récapitulatif détaillant exactement ce qui a été imputé, écrêté et transformé pour chaque feature.

---

## 4. Garantie Scientifique Anti-Data-Leakage

Le script de test unitaire automatisé `tests/test_leakage.py` valide de manière mathématique et déterministe l'absence de fuite d'information :
1. **Écart de mise à l'échelle** : La moyenne $\mu_{\text{train}}$ et l'écart-type $\sigma_{\text{train}}$ du `StandardScaler` sont calculés uniquement sur le train. Après transformation, le train a une moyenne strictement nulle ($\approx 0.0$), tandis que la moyenne du test est non nulle, démontrant que les statistiques du test n'ont jamais contaminé le scaler.
2. **Isolation des imputations** : La médiane utilisée pour combler les valeurs manquantes du test est la médiane calculée sur le train uniquement.
3. **Protection de la cible** : Aucune mise à l'échelle n'est appliquée à la variable dépendante $y$.
