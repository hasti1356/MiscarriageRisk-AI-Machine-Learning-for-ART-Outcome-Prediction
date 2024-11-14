# src/utils.py
import logging
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Tuple
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix

class ConfigManager:
    """Manages configuration settings for the project."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

class Logger:
    """Custom logger for the project."""
    
    @staticmethod
    def setup_logger(name: str, log_path: str) -> logging.Logger:
        """Set up logger with file and console handlers."""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger

class MetricsCalculator:
    """Calculate and store model performance metrics."""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Calculate comprehensive model performance metrics."""
        metrics = {}
        
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # ROC AUC
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        metrics['avg_precision'] = np.mean(precision)
        
        # F1 Score
        metrics['f1_score'] = 2 * (metrics['precision'] * metrics['sensitivity']) / \
                             (metrics['precision'] + metrics['sensitivity']) \
                             if (metrics['precision'] + metrics['sensitivity']) > 0 else 0
        
        return metrics

class DataValidator:
    """Validate data quality and integrity."""
    
    @staticmethod
    def validate_input_features(
        data: pd.DataFrame,
        required_columns: List[str],
        numerical_columns: List[str],
        categorical_columns: List[str]
    ) -> Tuple[bool, str]:
        """Validate input data format and content."""
        try:
            # Check required columns
            missing_cols = set(required_columns) - set(data.columns)
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}"
            
            # Validate numerical columns
            for col in numerical_columns:
                if not np.issubdtype(data[col].dtype, np.number):
                    return False, f"Column {col} should be numeric"
            
            # Validate categorical columns
            for col in categorical_columns:
                if not pd.api.types.is_categorical_dtype(data[col]) and \
                   not pd.api.types.is_object_dtype(data[col]):
                    return False, f"Column {col} should be categorical"
            
            return True, "Data validation successful"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"

class PathManager:
    """Manage project paths and directories."""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent
        self.data_dir = self.base_dir / 'data'
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        self.models_dir = self.base_dir / 'models'
        self.logs_dir = self.base_dir / 'logs'
        
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        for dir_path in [self.data_dir, self.raw_dir, self.processed_dir,
                        self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def get_data_path(self, filename: str, data_type: str = 'processed') -> Path:
        """Get path for data file."""
        if data_type == 'raw':
            return self.raw_dir / filename
        return self.processed_dir / filename
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path for model file."""
        return self.models_dir / model_name
    
    def get_log_path(self, log_name: str) -> Path:
        """Get path for log file."""
        return self.logs_dir / log_name
