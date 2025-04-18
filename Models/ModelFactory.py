from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Models.LSTMModel import LSTMModel
from Training.DataPreprocessor import DataPreprocessor
from Training.ModelTrainer import ModelTrainer
import os
import tensorflow as tf
from typing import Optional, Tuple, Dict, Any


class ModelFactory:
    """Factory for creating model-related components."""

    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

    def create_data_preprocessor(self, data_storage) -> DataPreprocessor:
        """Create a data preprocessor instance."""
        return DataPreprocessor(self.config, self.logger, data_storage)

    def create_model_trainer(self, data_preprocessor: DataPreprocessor,
                             model: Optional[LSTMModel] = None) -> ModelTrainer:
        """Create a model trainer instance."""
        return ModelTrainer(self.config, self.logger, data_preprocessor, model)

    def create_lstm_model(self, input_shape: Tuple[int, int], n_features: int) -> LSTMModel:
        """Create an LSTM model instance."""
        return LSTMModel(self.config, input_shape, n_features)

    def load_model(self, model_path: str) -> LSTMModel:
        """Load a trained model from file."""
        try:
            self.logger.info(f"Loading model from {model_path}")

            if not os.path.exists(model_path):
                self.logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")

            # Load Keras model to get input shape
            keras_model = tf.keras.models.load_model(model_path, compile=False)
            input_shape = keras_model.input_shape[1:]  # Remove batch dimension
            n_features = input_shape[1]

            # Create LSTM model with appropriate dimensions
            model = LSTMModel(self.config, input_shape, n_features)

            # Load saved weights
            model.load_model(model_path)

            self.logger.info(f"Model loaded successfully with input shape {input_shape}")
            return model

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def create_ensemble_model(self, n_models: int = 5) -> Dict[str, Any]:
        """Create an ensemble of models (placeholder for now)."""
        # This would be expanded in a full implementation with a proper EnsembleModel class
        self.logger.info(f"Creating ensemble model with {n_models} models")

        # Placeholder implementation
        ensemble = {
            'n_models': n_models,
            'models': []
        }

        return ensemble