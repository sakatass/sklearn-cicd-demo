import os, time, requests, numpy as np
URL = os.getenv("STAGE_URL", "http://localhost:8080")

def test_healthz():
    r = requests.get(f"{URL}/healthz", timeout=5)
    assert r.status_code == 200 and r.json().get("status") == "ok"

def test_predict_contract_and_latency():
    features = np.zeros(20).tolist()
    t0 = time.time()
    r = requests.post(f"{URL}/predict", json={"features": features}, timeout=5)
    dt = (time.time()-t0)*1000
    j = r.json()
    assert "proba" in j and "label" in j and 0.0 <= j["proba"] <= 1.0
    assert dt < 200
