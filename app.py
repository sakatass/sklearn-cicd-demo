import os, joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist

N_FEATURES = 20
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
GIT_SHA = os.getenv("GIT_SHA", "dev")

class PredictIn(BaseModel):
    features: conlist(float, min_length=N_FEATURES, max_length=N_FEATURES)

class PredictOut(BaseModel):
    proba: float
    label: int

app = FastAPI()

@app.on_event("startup")
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    app.state.model = joblib.load(MODEL_PATH)

@app.get("/healthz")
def healthz():
    return {"status": "ok", "git_sha": GIT_SHA}

@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    try:
        proba1 = app.state.model.predict_proba([inp.features])[0][1]
        return {"proba": float(proba1), "label": int(proba1 >= 0.5)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
