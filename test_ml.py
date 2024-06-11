import pickle
import pytest
from sklearn.ensemble import RandomForestClassifier
from train_model import X_test, y_test, compute_metrics
import os
import numpy as np
import pandas as pd


project_path = os.path.dirname(os.path.abspath(__file__))


model_path = os.path.join(project_path, "model", "model.pkl")


data_path = os.path.join(project_path, 'data', 'census.csv')


data = pd.read_csv(data_path)


def test_prediction_type():
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    data = pd.read_csv(data_path)
    X_test = data.drop(columns=['salary'])
    prediction = model.predict(X_test)
    assert isinstance(prediction, (list, np.ndarray))


def test_model_algorithm():
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, RandomForestClassifier)


def test_compute_metrics
     with open(model_path, "rb") as f:
        model = pickle.load(f)

    data = pd.read_csv(data_path)
    X_test = data.drop(columns=['salary'])
    y_test = data['salary']
    prediction = model.predict(X_test)
    metrics = compute_metrics(y_test, prediction)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics


if __name__ == "__main__":
    pytest.main()
