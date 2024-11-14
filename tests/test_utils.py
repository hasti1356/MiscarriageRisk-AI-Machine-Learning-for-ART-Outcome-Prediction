# tests/test_utils.py
import pytest
import numpy as np
from src.utils import Logger, MetricsCalculator
import logging
import os

def test_logger_setup():
    """Test logger setup."""
    logger = Logger.setup_logger('test_logger', 'logs/test.log')
    
    assert isinstance(logger, logging.Logger)
    assert logger.name == 'test_logger'
    assert os.path.exists('logs/test.log')
    
    # Test logging
    test_message = "Test log message"
    logger.info(test_message)
    
    with open('logs/test.log', 'r') as f:
        log_content = f.read()
        assert test_message in log_content

def test_metrics_calculator():
    """Test metrics calculation."""
    calculator = MetricsCalculator()
    
    # Create sample predictions
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
    y_prob = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.6, 0.7, 0.3])
    
    metrics = calculator.calculate_metrics(y_true, y_pred, y_prob)
    
    assert isinstance(metrics, dict)
    assert all(metric in metrics for metric in 
              ['accuracy', 'precision', 'recall', 'f1', 'auc'])
    assert all(0 <= metrics[metric] <= 1 for metric in metrics)
    
    # Test specific metrics
    assert metrics['accuracy'] == 0.75  # 6 correct out of 8
    assert metrics['auc'] > 0  # AUC should be positive
    
    # Test handling of edge cases
    edge_cases = [
        (np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0.1, 0.1, 0.1])),  # All negative
        (np.array([1, 1, 1]), np.array([1, 1, 1]), np.array([0.9, 0.9, 0.9])),  # All positive
        (np.array([0]), np.array([1]), np.array([0.6])),  # Single sample
    ]
    
    for y_true, y_pred, y_prob in edge_cases:
        metrics = calculator.calculate_metrics(y_true, y_pred, y_prob)
        assert all(0 <= metrics[metric] <= 1 for metric in metrics)

def test_metrics_calculator_input_validation():
    """Test input validation in metrics calculation."""
    calculator = MetricsCalculator()
    
    with pytest.raises(ValueError):
        # Different lengths
        calculator.calculate_metrics(
            np.array([0, 1]),
            np.array([0, 1, 0]),
            np.array([0.1, 0.9, 0.2])
        )
    
    with pytest.raises(ValueError):
        # Invalid probabilities
        calculator.calculate_metrics(
            np.array([0, 1]),
            np.array([0, 1]),
            np.array([0.1, 1.2])  # probability > 1
        )
    
    with pytest.raises(ValueError):
        # Invalid binary values
        calculator.calculate_metrics(
            np.array([0, 2]),  # should be 0 or 1
            np.array([0, 1]),
            np.array([0.1, 0.9])
        )
