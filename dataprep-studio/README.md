# DataPrep Studio — Preprocessing & Cleaning pour le Machine Learning

Application Web générique : **React** (interface) + **Python / FastAPI** (serveur) + **Pandas / Scikit-learn** (traitements).

## Lancer l'application  →  http://localhost:8501

```bash
# 1) Backend + interface compilée (une seule commande ensuite)
cd backend
pip install -r requirements.txt
python make_samples.py        # (optionnel) régénère les 2 datasets de démo
python run.py                 # sert l'API ET l'interface sur http://localhost:8501

# 2) Seulement si vous modifiez le code React
cd frontend
npm install
npm run build                 # produit frontend/dist (servi par FastAPI)
npm run dev                   # ou mode développement : http://localhost:5173 (proxy vers :8501)
```

Utilisation : importez votre CSV/Excel (ex. dataset de fraude bancaire), choisissez supervisé + cible,
parcourez l'EDA, acceptez/modifiez les recommandations, exécutez, téléchargez dataset + rapport Excel.
La structure détaillée est dans `docs/STRUCTURE.md`.
