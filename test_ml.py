import pickle
import pytest
from sklearn.ensemble import RandomForestClassifier
from train_model import X_test, y_test, compute_metrics
import os
import numpy as np
import pandas as pd
import csv


if os.getenv('GITHUB_ACTIONS') == 'true':
    data_path = 'census.csv'
    model_path = 'model/model.pkl'
else:
    project_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_path, 'census.csv')
    model_path = os.path.join(project_path, "model", "model.pkl")


with open(data_path, 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    data = list(csv_reader)


def test_prediction_type():
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict(X_test)
    assert isinstance(prediction, (list, np.ndarray))


def test_model_algorithm():
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, RandomForestClassifier)


def test_compute_metrics():
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    prediction = model.predict(X_test)
    metrics = compute_metrics(y_test, prediction)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics


if __name__ == "__main__":
    pytest.main()
