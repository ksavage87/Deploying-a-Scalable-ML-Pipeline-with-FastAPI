import pickle
import pytest
from sklearn.ensemble import RandomForestClassifier
from train_model import X_test, y_test, compute_metrics
import os


project_path = os.path.dirname(os.path.abspath(__file__))


model_path = os.path.join(project_path, "model", "model.pkl")


with open(model_path, "rb") as f:
    model = pickle.load(f)


data_path = os.path.join(project_path, "data", "census.csv")


def test_prediction_type():
    prediction = model.predict(X_test)
    assert isinstance(prediction, list)


def test_model_algorithm():
    assert isinstance(model, RandomForestClassifier)


def test_compute_metrics():
    prediction = model.predict(X_test)
    metrics = compute_metrics(y_test, prediction)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics


if __name__ == "__main__":
    pytest.main()
