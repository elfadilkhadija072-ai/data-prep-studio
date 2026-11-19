"""Point d'entrée racine : lance DataPrep Studio en une seule commande 'python run.py'."""
import os
import sys

# Ajouter le répertoire racine au PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from backend.run import main

if __name__ == "__main__":
    main()
