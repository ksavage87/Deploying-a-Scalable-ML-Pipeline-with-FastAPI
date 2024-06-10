# TODO: add necessary import
import pytest
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.append('Users/kaleymayer/Deploying-a-Scalable-ML-Pipeline-with-FastAPI')
from train_model import X_test, y_test, compute_metrics
from sklearn.metrics import fbeta_score, precision_score, recall_score, accuracy_score


# Load the model from model.pkl
model_path = "/Users/kaleymayer/Deploying-a-Scalable-ML-Pipeline-with-FastAPI/model/model.pkl"
print(f"Loading model from {model_path}")
with open(model_path, "rb") as f:
    model = pickle.load(f)


# TODO: implement the first test. Change the function name and input as needed
def test_ml_function_type():
    """
    # Tests if the ML function returns the expected type of result.
    """
    prediction = model.predict(X_test)
    prediction_array = list(prediction)
    assert isinstance(prediction_array, list)
    pass


# TODO: implement the second test. Change the function name and input as needed
def test_ml_algorithm():
    """
    # Tests if the ML model uses the expected algorithm.
    """
    assert isinstance(model, RandomForestClassifier)
    pass



# TODO: implement the third test. Change the function name and input as needed
def test_compute_metrics():
    """
    # Tests if the computing metrics function returns the expected values.
    """
    prediction = model.predict(X_test)
    metrics = compute_metrics(y_test, prediction)
    assert isinstance(metrics, dict)
    pass
