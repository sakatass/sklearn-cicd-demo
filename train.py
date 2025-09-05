import os, json, yaml, joblib
import mlflow, mlflow.sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

with open("params.yaml") as f:
    P = yaml.safe_load(f)["train"]

X, y = make_classification(n_samples=2000, n_features=20,
                           n_informative=8, random_state=P["random_state"])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=P["test_size"],
                                      random_state=P["random_state"])

clf = RandomForestClassifier(n_estimators=P["n_estimators"],
                             random_state=P["random_state"]).fit(Xtr, ytr)
auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])

os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/model.pkl")
with open("metrics.json", "w") as f:
    json.dump({"auc": float(auc)}, f)

# MLflow: параметры, метрики, модель
with mlflow.start_run(run_name="train-rf") as run:
    mlflow.log_params(P)
    mlflow.log_metrics({"auc": auc})
    mlflow.sklearn.log_model(clf, artifact_path="model")
    print("MLFLOW_RUN_ID", run.info.run_id)
