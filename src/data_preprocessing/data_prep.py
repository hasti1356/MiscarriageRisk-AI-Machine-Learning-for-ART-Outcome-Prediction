# src/data_prep.py
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from .utils import Logger, DataValidator

class DataPreprocessor:
    """Handle data preprocessing for the miscarriage risk prediction model."""
    
    def __init__(self, config: dict):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config (dict): Configuration dictionary containing preprocessing parameters
        """
        self.config = config
        self.logger = Logger.setup_logger('DataPreprocessor', 'logs/preprocessing.log')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.validator = DataValidator()
        
    def load_and_validate(self, filepath: str) -> pd.DataFrame:
        """Load and validate the dataset."""
        try:
            data = pd.read_csv(filepath)
            
            # Validate data structure
            is_valid, message = self.validator.validate_input_features(
                data,
                self.config['required_columns'],
                self.config['numerical_features'],
                self.config['categorical_features']
            )
            
            if not is_valid:
                self.logger.error(f"Data validation failed: {message}")
                raise ValueError(message)
            
            self.logger.info(f"Data loaded successfully: {data.shape}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        try:
            # Log missing value statistics
            missing_stats = data.isnull().sum()
            self.logger.info("Missing value statistics:\n" + 
                           str(missing_stats[missing_stats > 0]))
            
            # Handle numerical features
            num_features = self.config['numerical_features']
            if num_features:
                data[num_features] = self.imputer.fit_transform(data[num_features])
            
            # Handle categorical features
            cat_features = self.config['categorical_features']
            for feature in cat_features:
                data[feature].fillna(data[feature].mode()[0], inplace=True)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error handling missing values: {str(e)}")
            raise
    
    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle outliers using IQR method."""
        try:
            num_features = self.config['numerical_features']
            
            for feature in num_features:
                Q1 = data[feature].quantile(0.25)
                Q3 = data[feature].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Replace outliers with bounds
                data[feature] = np.where(
                    data[feature] < lower_bound,
                    lower_bound,
                    np.where(data[feature] > upper_bound, upper_bound, data[feature])
                )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error handling outliers: {str(e)}")
            raise
    
    def encode_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables."""
        try:
            cat_features = self.config['categorical_features']
            
            for feature in cat_features:
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                
                data[feature] = self.label_encoders[feature].fit_transform(
                    data[feature].astype(str)
                )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error encoding categorical variables: {str(e)}")
            raise
    
    def scale_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features."""
        try:
            num_features = self.config['numerical_features']
            data[num_features] = self.scaler.fit_transform(data[num_features])
            return data
            
        except Exception as e:
            self.logger.error(f"Error scaling features: {str(e)}")
            raise
    
    def prepare_training_data(
        self,
        data: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for model training."""
        try:
            X = data.drop(columns=[self.config['target_column']])
            y = data[self.config['target_column']]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def process_pipeline(
        self,
        filepath: str,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Execute the complete preprocessing pipeline."""
        try:
            # Load and validate data
            data = self.load_and_validate(filepath)
            
            # Handle missing values
            data = self.handle_missing_values(data)
            
            # Handle outliers
            data = self.handle_outliers(data)
            
            # Encode categorical variables
            data = self.encode_categorical(data)
            
            # Scale features
            data = self.scale_features(data)
            
            # Prepare training data
            return self.prepare_training_data(data, test_size, random_state)
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {str(e)}")
            raise
