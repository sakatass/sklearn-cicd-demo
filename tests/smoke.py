import os, requests, numpy as np
URL = os.getenv("URL", "http://localhost:8080")
def main():
    r = requests.get(f"{URL}/healthz", timeout=5); r.raise_for_status()
    features = np.zeros(20).tolist()
    r = requests.post(f"{URL}/predict", json={"features": features}, timeout=5); r.raise_for_status()
    print("SMOKE OK:", r.json())
if __name__ == "__main__":
    main()
