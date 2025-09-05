# train.py
import json, pathlib
from typing import Dict, Any

import joblib
import mlflow
from mlflow.models import infer_signature
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def train_and_save(params: Dict[str, Any] | None = None,
                   model_path: str = "models/model.pkl") -> Dict[str, Any]:
    """
    Обучает простую модель, сохраняет model.pkl и metrics.json, логирует ран в MLflow.
    Возвращает dict с auc и путём к модели.
    """
    cfg = {"n_estimators": 100, "random_state": 42, "test_size": 0.2}
    if params:
        cfg.update(params)

    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=12, n_redundant=2,
        random_state=cfg["random_state"]
    )
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["random_state"]
    )

    clf = RandomForestClassifier(
        n_estimators=cfg["n_estimators"], random_state=cfg["random_state"]
    )
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, proba))

    # локальные артефакты
    pathlib.Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, model_path)
    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump({"auc": auc}, f)

    # MLflow -> файловый backend ./mlruns
    mlruns = (pathlib.Path(".").resolve() / "mlruns")
    mlflow.set_tracking_uri(f"file:///{str(mlruns).replace('\\', '/')}")
    mlflow.set_experiment("sklearn-demo")

    with mlflow.start_run(run_name="train-rf"):
        mlflow.log_params(cfg)
        mlflow.log_metric("auc", auc)
        # подпись + input_example, чтобы убрать WARNING'и
        input_example = Xtr[:5]
        signature = infer_signature(input_example, clf.predict_proba(input_example)[:, 1])
        # важно: в MLflow 3 artifact_path — deprecated, используем name="model"
        mlflow.sklearn.log_model(clf, "model", input_example=input_example, signature=signature)  # :contentReference[oaicite:1]{index=1}
        mlflow.log_artifact("metrics.json")

    return {"auc": auc, "model_path": str(pathlib.Path(model_path).resolve())}


if __name__ == "__main__":
    res = train_and_save()
    print(f"AUC={res['auc']:.4f}")
