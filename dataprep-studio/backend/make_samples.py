"""Génère 2 datasets synthétiques de démonstration (le TP fournit son propre dataset de fraude bancaire :
importez-le directement dans l'application)."""
import numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(7)
out = Path(__file__).parent / "data"; out.mkdir(exist_ok=True)

# ── 1) Fraude bancaire (classification déséquilibrée) ──
n = 20000
typ = rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"], n, p=[.45, .2, .28, .07])
amount = np.round(rng.lognormal(5.2, 1.4, n), 2)
hour = rng.integers(0, 24, n)
foreign = rng.choice(["Yes", "No"], n, p=[.12, .88])
logit = -6.2 + 0.0004 * np.minimum(amount, 8000) + 1.4 * np.isin(typ, ["TRANSFER", "CASH_OUT"]) + 1.0 * (foreign == "Yes") + 0.9 * (hour < 5)
fraud = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
old_org = np.round(rng.lognormal(7.5, 1.6, n), 2)
age = rng.integers(18, 85, n).astype(float)
d = pd.DataFrame({
    "transaction_id": np.arange(100000, 100000 + n),
    "txn_date": (pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n), unit="m")).strftime("%Y-%m-%d %H:%M:%S"),
    "type": typ, "amount": amount, "hour": hour,
    "oldbalance_org": old_org, "newbalance_org": np.maximum(old_org - amount, 0) + rng.normal(0, 0.5, n),
    "oldbalance_dest": np.round(rng.lognormal(6.5, 2.0, n), 2),
    "customer_age": age, "account_age_days": np.round(rng.gamma(2.0, 600, n)),
    "country": rng.choice(["FR", "fr", "DE", "ES", " FR ", "MA", "US", "us"], n, p=[.3, .1, .2, .15, .05, .08, .07, .05]),
    "device": rng.choice(["Mobile", "mobile", "Web", "ATM", "POS"], n, p=[.35, .1, .3, .1, .15]),
    "is_foreign": foreign, "currency": "EUR", "isFraud": fraud})
d.loc[rng.random(n) < .06, "customer_age"] = np.nan
d.loc[rng.random(n) < .12, "device"] = np.nan
d.loc[rng.random(n) < .03, "oldbalance_dest"] = np.nan
d.loc[rng.random(n) < .02, "amount"] = d["amount"] * 40        # outliers extrêmes
d.loc[d.sample(frac=.01, random_state=1).index, "account_age_days"] = -1  # valeurs incohérentes
d["amount"] = d["amount"].astype(object); d.loc[rng.random(n) < .01, "amount"] = "n/a"   # numérique pollué par du texte
d = pd.concat([d, d.sample(150, random_state=3)])           # doublons
d.to_csv(out / "bank_fraud_demo.csv", index=False)

# ── 2) Prix immobiliers (régression, autre domaine → généricité) ──
m = 3000
sq = np.round(rng.lognormal(4.6, .45, m)); rooms = np.clip(np.round(sq / 28 + rng.normal(0, 1, m)), 1, 12)
hood = rng.choice(["Centre", "Nord", "Sud", "Périphérie", "Littoral"], m, p=[.2, .25, .2, .2, .15])
price = (1800 * sq + 25000 * rooms + pd.Series(hood).map({"Centre": 90000, "Nord": 30000, "Sud": 20000, "Périphérie": -20000, "Littoral": 120000}).values
         + rng.normal(0, 30000, m)); price = np.round(np.maximum(price, 30000))
h = pd.DataFrame({"house_id": np.arange(m), "surface_m2": sq, "rooms": rooms, "neighborhood": hood,
                  "year_built": rng.integers(1900, 2024, m).astype(float), "has_garage": rng.choice(["yes", "no", "Yes", "NO"], m),
                  "distance_center_km": np.round(rng.exponential(6, m), 1), "surface_sqft": np.round(sq * 10.764),
                  "listing_date": (pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 700, m), unit="D")).strftime("%d/%m/%Y"),
                  "price": price})
h.loc[rng.random(m) < .08, "year_built"] = np.nan
h.loc[rng.random(m) < .15, "neighborhood"] = np.nan
h.loc[rng.random(m) < .03, "price"] = np.nan
h.loc[rng.random(m) < .01, "surface_m2"] = h["surface_m2"] * 15
h.to_csv(out / "house_prices_demo.csv", index=False)
print("ok", d.shape, h.shape)
