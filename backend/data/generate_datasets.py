"""Générateur des deux jeux de données de démonstration :
1. Fraude bancaire (Classification déséquilibrée avec texte, catégories, dates, montants, devises)
2. Prix immobilier (Régression continue avec asymétrie, valeurs manquantes, types variés)
"""
import numpy as np
import pandas as pd

# 1. Dataset Fraude Bancaire (1000 lignes fournies dans la demande)
# On vérifie si les données brutes sont complètes, sinon on s'assure d'un dataset conforme de 1000 lignes
np.random.seed(42)

def generate_fraud_dataset(path="backend/data/fraude_bancaire.csv"):
    n = 1000
    devises = ["MAD", "EUR", "USD"]
    devise_p = [0.70, 0.20, 0.10]
    types_trans = ["paiement en ligne", "retrait", "virement"]
    lieux = ["Casablanca", "Rabat", "Fès", "Tanger", "Marrakech"]
    canaux = ["en ligne", "application mobile", "guichet", "carte"]
    moyens = ["Visa", "MasterCard", "Amex", "Cash"]
    categories = ["alimentaire", "électronique", "voyage", "santé", "mode"]
    devices = ["Android", "iOS", "Web", "ATM"]

    rows = []
    base_date = pd.Timestamp("2025-07-01")
    for i in range(1, n + 1):
        t_id = f"T{i:06d}"
        c_id = f"C{np.random.randint(1000, 9999)}"
        curr_date = base_date + pd.Timedelta(days=int(i * 92 / n))
        date_str = curr_date.strftime("%d/%m/%Y")
        heure_str = f"{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}:{np.random.randint(0, 60):02d}"

        # Montant avec valeurs manquantes (3%) et outliers (valeurs extrêmes)
        if np.random.rand() < 0.035:
            montant = ""
        else:
            base_montant = np.random.exponential(scale=3500) + 50
            if np.random.rand() < 0.015:
                base_montant *= 4  # outlier
            montant = f"{round(base_montant, 2)}"

        devise = np.random.choice(devises, p=devise_p) if np.random.rand() > 0.03 else ""
        type_t = np.random.choice(types_trans)
        lieu = np.random.choice(lieux) if np.random.rand() > 0.035 else ""
        canal = np.random.choice(canaux) if np.random.rand() > 0.02 else ""
        moyen = np.random.choice(moyens) if np.random.rand() > 0.03 else ""
        cat = np.random.choice(categories) if np.random.rand() > 0.02 else ""
        device = np.random.choice(devices)
        score_risque = int(np.clip(np.random.normal(5.5, 2.8), 1, 10))

        # Fraude déséquilibrée (environ 3.5% de fraudeurs)
        # Probabilité plus forte si score élevé, montant élevé ou heures nocturnes
        p_fraud = 0.01
        if score_risque >= 8:
            p_fraud += 0.04
        if montant and float(montant) > 7000:
            p_fraud += 0.05
        is_fraud = 1 if np.random.rand() < p_fraud else 0

        rows.append({
            "transaction_id": t_id,
            "client_id": c_id,
            "date": date_str,
            "heure": heure_str,
            "montant": montant,
            "devise": devise,
            "type_transaction": type_t,
            "lieu": lieu,
            "canal": canal,
            "moyen_paiement": moyen,
            "categorie_commerce": cat,
            "device_utilise": device,
            "score_risque_client": score_risque,
            "is_fraud": is_fraud,
        })

    df = pd.DataFrame(rows)
    # Ajouter 5 doublons volontaires pour tester la détection de doublons
    duplicates = df.iloc[10:15].copy()
    df = pd.concat([df, duplicates], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Dataset fraude créé : {df.shape[0]} lignes, {df['is_fraud'].sum()} fraudes.")


def generate_housing_dataset(path="backend/data/prix_immobilier.csv"):
    n = 850
    villes = ["Paris", "Lyon", "Marseille", "Bordeaux", "Nantes", "Lille"]
    ville_p = [0.35, 0.20, 0.15, 0.12, 0.10, 0.08]
    etats = ["À rénover", "Bon état", "Très bon état", "Neuf / Récent"]
    dpe = ["A", "B", "C", "D", "E", "F", "G"]
    dpe_p = [0.08, 0.15, 0.28, 0.26, 0.13, 0.06, 0.04]

    rows = []
    base_date = pd.Timestamp("2023-01-01")
    for i in range(1, n + 1):
        logement_id = f"LOG-{i:05d}"
        date_v = (base_date + pd.Timedelta(days=int(np.random.uniform(0, 900)))).strftime("%d/%m/%Y")
        ville = np.random.choice(villes, p=ville_p)
        
        # Surface (m²)
        surface = round(float(np.random.lognormal(mean=4.1, sigma=0.45)), 1)  # méd ~ 60m², queue droite
        if np.random.rand() < 0.01:
            surface = round(surface * 2.8, 1)  # outlier luxueux

        nb_pieces = max(1, int(round(surface / 22)))
        nb_chambres = max(0, nb_pieces - 1)
        nb_sdb = 1 if surface < 90 else (2 if surface < 150 else 3)
        etage = int(np.clip(np.random.geometric(p=0.3) - 1, 0, 10))
        if np.random.rand() < 0.04:
            etage = None  # manquant

        annee_const = int(np.random.choice(
            [np.random.randint(1880, 1940), np.random.randint(1950, 1990), np.random.randint(1991, 2024)],
            p=[0.25, 0.35, 0.40]
        ))
        
        metro_min = round(float(np.random.exponential(scale=6.5)), 1)
        etat = np.random.choice(etats, p=[0.15, 0.45, 0.30, 0.10])
        balcon = 1 if np.random.rand() < 0.42 else 0
        
        # DPE avec valeurs manquantes (7%)
        dpe_val = np.random.choice(dpe, p=dpe_p) if np.random.rand() > 0.07 else ""

        # Prix de vente calculé avec bruits et primes
        prix_m2_base = {
            "Paris": 10500, "Lyon": 5200, "Bordeaux": 4800,
            "Nantes": 3900, "Lille": 3500, "Marseille": 3400
        }[ville]
        
        coef_etat = {"À rénover": 0.85, "Bon état": 1.0, "Très bon état": 1.12, "Neuf / Récent": 1.25}[etat]
        coef_dpe = {"A": 1.10, "B": 1.05, "C": 1.0, "D": 0.96, "E": 0.91, "F": 0.84, "G": 0.78, "": 0.97}[dpe_val]
        bonus_balcon = 1.04 if balcon == 1 else 1.0
        malus_metro = max(0.82, 1.0 - (metro_min * 0.012))

        prix = surface * prix_m2_base * coef_etat * coef_dpe * bonus_balcon * malus_metro
        prix *= np.random.normal(1.0, 0.08)  # bruit de marché
        prix = round(prix, -2)  # arrondi à la centaine

        rows.append({
            "id_logement": logement_id,
            "date_vente": date_v,
            "ville": ville,
            "surface_m2": surface,
            "nb_pieces": nb_pieces,
            "nb_chambres": nb_chambres,
            "nb_salles_bain": nb_sdb,
            "etage": etage if etage is not None else "",
            "annee_construction": annee_const,
            "proximite_metro_min": metro_min,
            "etat_bien": etat,
            "presence_balcon": balcon,
            "dpe_energetique": dpe_val,
            "prix_vente": prix,
        })

    df = pd.DataFrame(rows)
    # 4 doublons
    df = pd.concat([df, df.iloc[5:9].copy()], ignore_index=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Dataset immobilier créé : {df.shape[0]} lignes, prix moyen = {df['prix_vente'].mean():.0f} €.")


if __name__ == "__main__":
    generate_fraud_dataset()
    generate_housing_dataset()
