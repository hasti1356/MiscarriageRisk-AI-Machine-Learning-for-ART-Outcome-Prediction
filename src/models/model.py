# src/model.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve
import joblib
from .utils import Logger, MetricsCalculator

class MiscarriageRiskModel:
    """Deep learning model for miscarriage risk prediction."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = Logger.setup_logger('MiscarriageRiskModel', 'logs/model.log')
        self.model = None
        self.metrics_calculator = MetricsCalculator()
        self.history = None
        
    def build_model(self, input_dim: int) -> None:
        """Build the neural network architecture."""
        try:
            model = models.Sequential([
                layers.Input(shape=(input_dim,)),
                
                layers.Dense(
                    256,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.01)
                ),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(
                    128,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.01)
                ),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                
                layers.Dense(
                    64,
                    activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(0.01)
                ),
                layers.BatchNormalization(),
                layers.Dropout(0.1),
                
                layers.Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=self.config.get('learning_rate', 0.001)
                ),
                loss='binary_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ]
            )
            
            self.model = model
            self.logger.info(f"Model built successfully: {model.summary()}")
            
        except Exception as e:
            self.logger.error(f"Error building model: {str(e)}")
            raise
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> None:
        """Train the model with early stopping and learning rate reduction."""
        try:
            # Define callbacks
            callbacks_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                ),
                callbacks.ModelCheckpoint(
                    filepath='models/best_model.h5',
                    monitor='val_auc',
                    mode='max',
                    save_best_only=True
                )
            ]
            
            # Train the model
            self.history = self.model.fit(
                X_train,
                y_train,
                epochs=self.config.get('epochs', 100),
                batch_size=self.config.get('batch_size', 32),
                validation_data=(X_val, y_val),
                callbacks=callbacks_list,
                class_weight=self._calculate_class_weights(y_train)
            )
            
            self.logger.info("Model training completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise
    
    def _calculate_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for imbalanced dataset."""
        classes = np.unique(y)
        weights = dict(zip(
            classes,
            1 / (len(classes) * np.bincount(y.astype(int)))
        ))
        return weights
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Dict[str, float], np.ndarray]:
        """Evaluate model performance."""
        try:
            # Get predictions
            y_prob = self.model.predict(X_test)
            y_pred = (y_prob > 0.5).astype(int)
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_metrics(
                y_test, y_pred, y_prob
            )
            
            self.logger.info(f"Model evaluation metrics: {metrics}")
            return metrics, y_prob
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5
    ) -> Dict[str, List[float]]:
        """Perform k-fold cross-validation."""
        try:
            cv_scores = {
                'accuracy': [], 'auc': [], 'precision': [],
                'recall': [], 'f1': []
            }
            
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                # Build and train model
                self.build_model(X.shape[1])
                self.train(X_train, y_train, X_val, y_val)
                
                # Evaluate
                metrics, _ = self.evaluate(X_val, y_val)
                
                for metric in cv_scores:
                    cv_scores[metric].append(metrics[metric])
                
                self.logger.info(f"Fold {fold+1} completed")
            
            # Calculate mean and std for each metric
            cv_results = {
                metric: {
                    'mean': np.mean(scores),
                    'std': np.std(scores)
                }
                for metric, scores in cv_scores.items()
            }
            
            self.logger.info(f"Cross-validation results: {cv_results}")
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Error in cross-validation: {str(e)}")
            raise
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of miscarriage risk."""
        try:
            return self.model.predict(X)
        except Exception as e:
            self.logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model."""
        try:
            self.model.save(filepath)
            self.logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model."""
        try:
            self.model = models.load_model(filepath)
            self.logger.info(f"Model loaded successfully from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
