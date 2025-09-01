from train import train_and_save
import json, os

def test_training_auc_good(tmp_path, monkeypatch):
    # обучаем в temp и проверяем метрику
    train_and_save(C=0.7, random_state=42)
    assert os.path.exists("models/model.pkl")
    with open("metrics.json") as f:
        auc = json.load(f)["auc"]
    assert auc >= 0.85
