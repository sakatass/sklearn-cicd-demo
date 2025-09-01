import os, json, joblib
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def train_and_save(C=0.7, random_state=42):
    X, y = make_classification(n_samples=5000, n_features=20, random_state=random_state)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)
    clf = LogisticRegression(max_iter=500, C=C, random_state=random_state).fit(Xtr, ytr)
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/model.pkl")
    auc = roc_auc_score(yte, clf.predict_proba(Xte)[:,1])
    with open("metrics.json","w") as f: json.dump({"auc": float(auc)}, f, indent=2)
    print("AUC:", round(auc, 6))

if __name__ == "__main__":
    train_and_save()
