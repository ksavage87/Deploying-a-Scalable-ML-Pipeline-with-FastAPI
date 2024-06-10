import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from train_model import X_test, y_test, compute_metrics

# Get the absolute path to the directory containing this script
project_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the model file relative to the project directory
model_path = os.path.join(project_dir, "model", "model.pkl")
print(f"Loading model from {model_path}")

# Load the model from model.pkl
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Implement the first test
def test_ml_function_type():
    """
    Tests if the ML function returns the expected type of result.
    """
    prediction = model.predict(X_test)
    prediction_array = list(prediction)
    assert isinstance(prediction_array, list)

# Implement the second test
def test_ml_algorithm():
    """
    Tests if the ML model uses the expected algorithm.
    """
    assert isinstance(model, RandomForestClassifier)

# Implement the third test
def test_compute_metrics():
    """
    Tests if the computing metrics function returns the expected values.
    """
    prediction = model.predict(X_test)
    metrics = compute_metrics(y_test, prediction)
    assert isinstance(metrics, dict)
