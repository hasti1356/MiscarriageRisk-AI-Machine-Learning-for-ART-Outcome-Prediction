# tests/test_features.py
import pytest
import pandas as pd
import numpy as np
from src.features import FeatureEngineer

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = pd.DataFrame({
        'age': np.random.normal(33, 5, n_samples),
        'height': np.random.normal(165, 10, n_samples),
        'weight': np.random.normal(65, 10, n_samples),
        'previous_miscarriages': np.random.randint(0, 4, n_samples),
        'fsh': np.random.normal(7, 2, n_samples),
        'lh': np.random.normal(5, 1.5, n_samples),
        'amh': np.random.normal(3, 1, n_samples),
        'estradiol': np.random.normal(150, 30, n_samples),
        'follicle_count': np.random.randint(5, 20, n_samples),
        'medication_dose': np.random.normal(200, 50, n_samples)
    })
    
    return data

@pytest.fixture
def config():
    """Create sample configuration."""
    return {
        'interaction_features': ['age', 'bmi', 'fsh_lh_ratio'],
        'use_pca': True,
        'pca_variance_ratio': 0.95
    }

def test_feature_engineer_initialization(config):
    """Test FeatureEngineer initialization."""
    fe = FeatureEngineer(config)
    assert fe.config == config
    assert fe.feature_importances == {}
    assert fe.selected_features == []
    assert fe.pca is None

def test_create_medical_features(sample_data, config):
    """Test medical feature creation."""
    fe = FeatureEngineer(config)
    result = fe.create_medical_features(sample_data)
    
    # Check new features are created
    assert 'bmi' in result.columns
    assert 'age_risk' in result.columns
    assert 'recurrent_loss' in result.columns
    assert 'fsh_lh_ratio' in result.columns
    assert 'hormone_balance_score' in result.columns
    assert 'response_ratio' in result.columns
    
    # Check calculations are correct
    assert all(result['bmi'] == result['weight'] / (result['height'] / 100) ** 2)
    assert all(result['recurrent_loss'] == (result['previous_miscarriages'] >= 2))
    assert all(result['fsh_lh_ratio'] == result['fsh'] / result['lh'])

def test_create_interaction_features(sample_data, config):
    """Test interaction feature creation."""
    fe = FeatureEngineer(config)
    data_with_medical = fe.create_medical_features(sample_data)
    result = fe.create_interaction_features(data_with_medical)
    
    # Check interaction features are created
    interaction_cols = [col for col in result.columns if col.startswith('interaction_')]
    assert len(interaction_cols) > 0

def test_select_features(sample_data, config):
    """Test feature selection."""
    fe = FeatureEngineer(config)
    data_processed = fe.create_medical_features(sample_data)
    y = np.random.randint(0, 2, len(sample_data))  # Binary target
    
    X_selected, selected_features = fe.select_features(data_processed, y, n_features=5)
    
    assert len(selected_features) == 5
    assert X_selected.shape[1] == 5
    assert len(fe.feature_importances) == len(data_processed.columns)

def test_reduce_dimensions(sample_data, config):
    """Test dimension reduction."""
    fe = FeatureEngineer(config)
    result = fe.reduce_dimensions(sample_data, n_components=5)
    
    assert result.shape[1] <= 5
    assert all(col.startswith('PC') for col in result.columns)
    assert fe.pca is not None

def test_process_pipeline(sample_data, config):
    """Test complete feature engineering pipeline."""
    fe = FeatureEngineer(config)
    y = np.random.randint(0, 2, len(sample_data))
    
    X_processed, selected_features = fe.process_pipeline(sample_data, y, n_features=10)
    
    assert isinstance(X_processed, pd.DataFrame)
    assert isinstance(selected_features, list)
    assert len(selected_features) > 0
