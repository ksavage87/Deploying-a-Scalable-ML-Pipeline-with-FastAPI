import os
import pickle
import pytest
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

# Test if the ML function returns the expected type of result
def test_prediction_type():

    
    """
    Tests if the ML function returns the prediction as a list.
    """
    prediction = model.predict(X_test)
    assert isinstance(prediction, list)

    
    
    
# Test if the ML model uses the expected algorithm
def test_model_algorithm():
    """
    Tests if the ML model is an instance of RandomForestClassifier.
    """
    assert isinstance(model, RandomForestClassifier)

    
    
    
# Test if the compute_metrics function returns the expected values
def test_compute_metrics():
    """
    Tests if the compute_metrics function returns a dictionary.
    """
    prediction = model.predict(X_test)
    metrics = compute_metrics(y_test, prediction)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics

    
if __name__ == "__main__":
    
    
    pytest.main()
    
    