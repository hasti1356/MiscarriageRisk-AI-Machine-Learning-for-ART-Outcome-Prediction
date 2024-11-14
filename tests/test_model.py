# tests/test_model.py
import pytest
import numpy as np
import tensorflow as tf
from src.model import MiscarriageRiskModel

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.normal(0, 1, (n_samples, n_features))
    y = np.random.randint(0, 2, n_samples)
    
    # Split into train, validation, and test sets
    train_idx = int(0.7 * n_samples)
    val_idx = int(0.85 * n_samples)
    
    X_train = X[:train_idx]
    y_train = y[:train_idx]
    X_val = X[train_idx:val_idx]
    y_val = y[train_idx:val_idx]
    X_test = X[val_idx:]
    y_test = y[val_idx:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test

@pytest.fixture
def config():
    """Create sample configuration."""
    return {
        'learning_rate': 0.001,
        'epochs': 5,
        'batch_size': 32
    }

def test_model_initialization(config):
    """Test model initialization."""
    model = MiscarriageRiskModel(config)
    assert model.config == config
    assert model.model is None
    assert model.history is None

def test_build_model(config):
    """Test model building."""
    model = MiscarriageRiskModel(config)
    model.build_model(input_dim=20)
    
    assert isinstance(model.model, tf.keras.Model)
    assert len(model.model.layers) > 0
    assert model.model.input_shape == (None, 20)
    assert model.model.output_shape == (None, 1)

def test_train_model(config, sample_data):
    """Test model training."""
    X_train, y_train, X_val, y_val, _, _ = sample_data
    
    model = MiscarriageRiskModel(config)
    model.build_model(X_train.shape[1])
    model.train(X_train, y_train, X_val, y_val)
    
    assert model.history is not None
    assert 'loss' in model.history.history
    assert 'val_loss' in model.history.history
    assert len(model.history.history['loss']) == len(model.history.history['val_loss'])

def test_evaluate_model(config, sample_data):
    """Test model evaluation."""
    X_train, y_train, X_val, y_val, X_test, y_test = sample_data
    
    model = MiscarriageRiskModel(config)
    model.build_model(X_train.shape[1])
    model.train(X_train, y_train, X_val, y_val)
    
    metrics, predictions = model.evaluate(X_test, y_test)
    
    assert isinstance(metrics, dict)
    assert all(metric in metrics for metric in ['accuracy', 'auc', 'precision', 'recall', 'f1'])
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(y_test)
    assert all(0 <= pred <= 1 for pred in predictions)

def test_cross_validation(config, sample_data):
    """Test cross-validation."""
    X_train, y_train, _, _, _, _ = sample_data
    
    model = MiscarriageRiskModel(config)
    cv_results = model.cross_validate(X_train, y_train, n_splits=3)
    
    assert isinstance(cv_results, dict)
    assert all(metric in cv_results for metric in ['accuracy', 'auc', 'precision', 'recall', 'f1'])
    assert all(isinstance(cv_results[metric], dict) for metric in cv_results)
    assert all('mean' in cv_results[metric] and 'std' in cv_results[metric] 
              for metric in cv_results)

def test_predict_proba(config, sample_data):
    """Test probability predictions."""
    X_train, y_train, X_val, y_val, X_test, _ = sample_data
    
    model = MiscarriageRiskModel(config)
    model.build_model(X_train.shape[1])
    model.train(X_train, y_train, X_val, y_val)
    
    predictions = model.predict_proba(X_test)
    
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape[1] == 1
    assert all(0 <= pred <= 1 for pred in predictions.flatten())

def test_save_load_model(config, sample_data, tmp_path):
    """Test model saving and loading."""
    X_train, y_train, X_val, y_val, X_test, _ = sample_data
    
    # Create and train model
    model = MiscarriageRiskModel(config)
    model.build_model(X_train.shape[1])
    model.train(X_train, y_train, X_val, y_val)
    
    # Get predictions before saving
    predictions_before = model.predict_proba(X_test)
    
    # Save model
    model_path = tmp_path / "test_model.h5"
    model.save_model(str(model_path))
    
    # Create new model instance and load saved model
    new_model = MiscarriageRiskModel(config)
    new_model.load_model(str(model_path))
    
    # Get predictions after loading
    predictions_after = new_model.predict_proba(X_test)
    
    # Compare predictions
    np.testing.assert_array_almost_equal(predictions_before, predictions_after)
