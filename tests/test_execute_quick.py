"""Test rapide de l'endpoint execute."""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
import json

# 1. Upload
r = requests.post("http://localhost:8501/api/upload?demo=housing")
sid = r.json()["session_id"]
print(f"Session: {sid}")

# 2. Configure
requests.post("http://localhost:8501/api/configure", json={
    "session_id": sid, "is_supervised": True, "target_col": "prix_vente",
    "problem_type": "regression", "test_size": 0.2, "random_state": 42, "drop_duplicates": True
})
print("Configure OK")

# 3. Plan
plan_r = requests.get(f"http://localhost:8501/api/prep/plan/{sid}")
plan_data = plan_r.json()
features = plan_data["features"]
print(f"Plan: {len(features)} features")

# 4. Build execute payload
plan_payload = {}
for col, info in features.items():
    plan_payload[col] = info["user_config"]

# 5. Execute
exec_r = requests.post("http://localhost:8501/api/prep/execute", json={
    "session_id": sid, "plan": plan_payload, "drop_duplicates": True
})
print(f"Execute status: {exec_r.status_code}")
if exec_r.status_code == 200:
    res = exec_r.json()
    train_shape = res["train_shape"]
    test_shape = res["test_shape"]
    final_feat = res["final_features_count"]
    print(f"Train shape: {train_shape}, Test shape: {test_shape}")
    print(f"Final features: {final_feat}")
    print("=== EXECUTE ENDPOINT WORKS! ===")
else:
    print(f"Error: {exec_r.text[:500]}")
