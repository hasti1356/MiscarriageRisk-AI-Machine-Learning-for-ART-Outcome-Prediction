# src/features.py
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as stats
from .utils import Logger

class FeatureEngineer:
    """Feature engineering and selection for miscarriage risk prediction."""
    
    def __init__(self, config: dict):
        self.config = config
        self.logger = Logger.setup_logger('FeatureEngineer', 'logs/features.log')
        self.feature_importances = {}
        self.selected_features = []
        self.pca = None
    
    def create_medical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create domain-specific medical features."""
        try:
            df = data.copy()
            
            # BMI calculation (if height and weight are available)
            if all(col in df.columns for col in ['height', 'weight']):
                df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
            
            # Age-related risk factors
            if 'age' in df.columns:
                df['age_risk'] = df['age'].apply(
                    lambda x: 1 if x > 35 else (2 if x > 40 else 0)
                )
            
            # Previous pregnancy outcomes
            if 'previous_miscarriages' in df.columns:
                df['recurrent_loss'] = df['previous_miscarriages'] >= 2
            
            # Hormone level ratios (if available)
            hormones = ['fsh', 'lh', 'amh', 'estradiol']
            if all(hormone in df.columns for hormone in hormones):
                df['fsh_lh_ratio'] = df['fsh'] / df['lh']
                df['hormone_balance_score'] = df.apply(
                    self._calculate_hormone_score, axis=1
                )
            
            # Treatment response indicators
            if 'follicle_count' in df.columns and 'medication_dose' in df.columns:
                df['response_ratio'] = df['follicle_count'] / df['medication_dose']
            
            self.logger.info(f"Created {len(df.columns) - len(data.columns)} new medical features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating medical features: {str(e)}")
            raise
    
    def _calculate_hormone_score(self, row: pd.Series) -> float:
        """Calculate hormone balance score based on multiple parameters."""
        try:
            weights = {
                'fsh': 0.3,
                'lh': 0.2,
                'amh': 0.3,
                'estradiol': 0.2
            }
            
            score = sum(
                weights[hormone] * stats.zscore([row[hormone]])[0]
                for hormone in weights.keys()
            )
            
            return np.clip(score, -1, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating hormone score: {str(e)}")
            return 0.0
    
    def create_interaction_features(
        self,
        data: pd.DataFrame,
        degree: int = 2
    ) -> pd.DataFrame:
        """Create interaction features between important variables."""
        try:
            df = data.copy()
            selected_features = self.config.get('interaction_features', [])
            
            if not selected_features:
                return df
            
            poly = PolynomialFeatures(
                degree=degree,
                include_bias=False,
                interaction_only=True
            )
            
            interactions = poly.fit_transform(df[selected_features])
            feature_names = poly.get_feature_names_out(selected_features)
            
            # Add only interaction terms (skip original features)
            for i, name in enumerate(feature_names[len(selected_features):], len(selected_features)):
                df[f'interaction_{name}'] = interactions[:, i]
            
            self.logger.info(f"Created {len(feature_names) - len(selected_features)} interaction features")
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating interaction features: {str(e)}")
            raise
    
    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_features: int = 20
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Select most important features using statistical tests."""
        try:
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            
            # Get selected feature names
            mask = selector.get_support()
            selected_features = X.columns[mask].tolist()
            
            # Store feature importances
            self.feature_importances = dict(zip(
                X.columns,
                selector.scores_
            ))
            
            self.selected_features = selected_features
            self.logger.info(f"Selected {len(selected_features)} features")
            
            return pd.DataFrame(X_selected, columns=selected_features), selected_features
            
        except Exception as e:
            self.logger.error(f"Error selecting features: {str(e)}")
            raise
    
    def reduce_dimensions(
        self,
        X: pd.DataFrame,
        n_components: Optional[int] = None,
        variance_ratio: float = 0.95
    ) -> pd.DataFrame:
        """Reduce feature dimensions using PCA."""
        try:
            if n_components is None:
                n_components = X.shape[1]
            
            self.pca = PCA(n_components=n_components)
            X_reduced = self.pca.fit_transform(X)
            
            # Find number of components needed to explain variance_ratio of variance
            cumulative_variance_ratio = np.cumsum(self.pca.explained_variance_ratio_)
            n_components_needed = np.argmax(cumulative_variance_ratio >= variance_ratio) + 1
            
            # Keep only necessary components
            X_reduced = X_reduced[:, :n_components_needed]
            
            self.logger.info(
                f"Reduced dimensions from {X.shape[1]} to {n_components_needed} "
                f"components explaining {variance_ratio:.2%} of variance"
            )
            
            return pd.DataFrame(
                X_reduced,
                columns=[f'PC{i+1}' for i in range(n_components_needed)]
            )
            
        except Exception as e:
            self.logger.error(f"Error reducing dimensions: {str(e)}")
            raise
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """Generate a report of feature importances."""
        try:
            importance_df = pd.DataFrame({
                'Feature': list(self.feature_importances.keys()),
                'Importance': list(self.feature_importances.values())
            })
            
            importance_df = importance_df.sort_values(
                'Importance',
                ascending=False
            ).reset_index(drop=True)
            
            return importance_df
            
        except Exception as e:
            self.logger.error(f"Error generating feature importance report: {str(e)}")
            raise
    
    def process_pipeline(
        self,
        data: pd.DataFrame,
        target: pd.Series,
        n_features: int = 20
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Execute the complete feature engineering pipeline."""
        try:
            # Create medical features
            data_engineered = self.create_medical_features(data)
            
            # Create interaction features
            data_engineered = self.create_interaction_features(data_engineered)
            
            # Select features
            X_selected, selected_features = self.select_features(
                data_engineered, target, n_features
            )
            
            # Reduce dimensions if specified in config
            if self.config.get('use_pca', False):
                X_selected = self.reduce_dimensions(
                    X_selected,
                    variance_ratio=self.config.get('pca_variance_ratio', 0.95)
                )
            
            return X_selected, selected_features
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering pipeline: {str(e)}")
            raise
