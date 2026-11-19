# Structure du projet — étape par étape

> Note : conformément au TP, **React est le frontend** (ce que voit l'utilisateur) et **FastAPI/Python est le backend**
> (là où se font les calculs). React seul ne peut pas lire/traiter un gros dataset de façon fiable avec Pandas/Scikit-learn.

## Arborescence
```
dataprep-studio/
├── README.md                 Lancement rapide
├── docs/STRUCTURE.md         Ce document
├── backend/                  PYTHON (cerveau de l'application)
│   ├── run.py                Étape A – démarre le serveur sur le port 8501
│   ├── main.py               Étape B – routes API (/api/...) + sert l'interface React
│   ├── make_samples.py       Génère 2 datasets de démo (fraude + immobilier)
│   ├── requirements.txt      Dépendances Python
│   ├── data/                 Datasets de démonstration
│   └── core/                 Logique métier, un fichier = une responsabilité
│       ├── common.py         Types, options de transformation, JSON propre
│       ├── loader.py         Lecture CSV/Excel + conversions automatiques
│       ├── eda.py            Analyse exploratoire
│       ├── analysis.py       Anomalies + recommandations justifiées
│       ├── pipeline.py       Exécution SANS data leakage
│       └── report.py         Rapport Excel par feature
└── frontend/                 REACT (interface)
    ├── index.html, vite.config.js, package.json
    └── src/
        ├── main.jsx          Point d'entrée React
        ├── App.jsx           État global + navigation 4 onglets
        ├── api.js            Appels HTTP vers le backend
        ├── ui.jsx            Composants UI réutilisables (cartes, badges…)
        ├── charts.jsx        Histogrammes, boxplot, heatmap, nuage de points
        ├── styles.css        Thème sombre « data lab »
        └── components/
            ├── ImportTab.jsx   Onglet 1
            ├── EdaTab.jsx      Onglet 2
            ├── PrepTab.jsx     Onglet 3
            └── ExportTab.jsx   Onglet 4
```

## Le parcours utilisateur, étape par étape

### Étape 1 — Importation et configuration (`ImportTab.jsx` → `POST /api/upload`, `/config`)
1. Vous déposez un CSV/Excel. `loader.py` détecte le séparateur et l'encodage, remplace `NA, null, ?, vide` par « manquant »,
   convertit les nombres écrits en texte et les dates en vrais types.
2. Vous choisissez **supervisé** (cible + split train/test) ou **non supervisé** (pas de cible, pas de split).
3. La cible est typée automatiquement : **classification** (≤ 10 valeurs ou texte) ou **régression** (modifiable).

### Étape 2 — Analyse exploratoire (`EdaTab.jsx` → `GET /api/.../eda`, `eda.py`)
- *Vue d'ensemble* : lignes, colonnes, types, % manquants, doublons, déséquilibre des classes.
- *Features & statistiques* : moyenne, σ, quartiles, asymétrie (skew), outliers, lien avec la cible (|r|, η, Cramér V).
- *Distributions* : histogramme (zoom P1–P99, échelle log), boxplot, top modalités, comparaison par classe.
- *Manquants · doublons · outliers* : barres par colonne, exemples de doublons (règle de Tukey, IQR × 1,5).
- *Corrélations* : heatmap Pearson / Spearman / Kendall, paires les plus liées, nuage de points interactif.

### Étape 3 — Preprocessing & Cleaning (`PrepTab.jsx` → `/recommendations`, `/preview`, `analysis.py`)
Pour chaque feature : **anomalies détectées** → **plan recommandé** → **justification chiffrée**. Vous pouvez
*accepter*, *changer chaque étape* ou *ne rien appliquer*. Un aperçu **avant/après** se met à jour en direct.

| Étape | Options | Règle de recommandation |
|---|---|---|
| Suppression | oui/non | constante, > 60 % manquants, identifiant, redondante (\|r\| ≥ 0,95) |
| Nettoyage texte | strip + minuscules | « Yes/yes/ YES » fusionnés |
| Imputation | moyenne, médiane, mode, constante / « Missing » | médiane si \|skew\| > 1 ou outliers > 5 %, sinon moyenne ; catégoriel : mode (≤ 5 %) ou « Missing » |
| Outliers | winsorisation IQR / Z-score / percentiles, suppression de lignes (train) | winsorisation si ≤ 5 % hors bornes (on garde l'information utile à la fraude) |
| Transformation | log(1+x), racine, Yeo-Johnson | skew > 1 : log si valeurs ≥ 0, sinon Yeo-Johnson |
| Échelle | Standard, Min-Max, Robuste | Robuste si outliers conservés, sinon Standard |
| Encodage | One-Hot, Ordinal, Fréquence, Label (cible), dates → année/mois/jour… | ≤ 2 modalités : 0/1 ; ≤ 15 : One-Hot ; > 15 : fréquence |

### Étape 4 — Exécution et export (`ExportTab.jsx` → `/apply`, `/download`, `/report`, `pipeline.py`, `report.py`)
Ordre appliqué (**prévention du data leakage**) :
1. suppression des doublons **avant** le split ;
2. split train/test (stratifié en classification, graine 42) **avant** toute transformation ;
3. `fit` (moyennes, médianes, bornes, scalers, encodeurs) sur le **train uniquement** ;
4. `transform` du train **et** du test avec ces paramètres ; les lignes outliers ne sont retirées que du train ;
5. la cible n'est jamais mise à l'échelle sans demande explicite.

Sorties : dataset complet (colonne `_split`), `train.csv`, `test.csv`, XLSX, et **rapport Excel** :
`Résumé` · `Features` (une ligne par feature : stats, anomalies, recommandation, justification, décision, transformation
réellement appliquée) · `Détail transformations` · `Colonnes finales` · `Corrélations`.

## Routes API
`POST /api/upload` · `POST /api/samples/{f}` · `POST /api/session/{id}/config` · `GET .../eda` · `GET .../corr` ·
`GET .../scatter` · `GET .../groupby` · `GET .../recommendations` · `POST .../preview` · `POST .../apply` ·
`GET .../download?part=all|train|test&fmt=csv|xlsx` · `GET .../report`
