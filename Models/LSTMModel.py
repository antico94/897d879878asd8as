import tensorflow as tf
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Bidirectional, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model, register_keras_serializable
import numpy as np
import os
from typing import Dict, List, Any, Tuple, Optional, Union

@register_keras_serializable(package='Custom', name='AttentionLayer')
class AttentionLayer(tf.keras.layers.Layer):
    """Custom attention mechanism to focus on important time steps."""

    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="normal"
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros"
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        # Alignment scores
        e = tf.tanh(tf.matmul(x, self.W) + self.b)

        # Get attention weights
        a = tf.nn.softmax(e, axis=1)

        # Apply attention weights to input
        context = x * a
        context = tf.reduce_sum(context, axis=1)

        return context

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        # no additional args to add here, but if you add hyperparameters in __init__,
        # include them in this dict
        return config


class LSTMModel:
    """LSTM model architecture for multi-output financial prediction."""

    def __init__(self, config, input_shape: Tuple[int, int], n_features: int):
        """Initialize the model.

        Args:
            config: Application configuration
            input_shape: (sequence_length, n_features)
            n_features: Number of input features
        """
        self.config = config
        self.sequence_length = input_shape[0]
        self.n_features = n_features
        self.model = None
        self.history = None

        # Get model configuration from config
        ml_config = self.config.get('GoldTradingSettings', {}).get('MachineLearning', {})
        lstm_config = ml_config.get('Hyperparameters', {}).get('LSTM', {})

        # Model parameters with defaults
        self.units = lstm_config.get('units', 50)
        self.dropout = lstm_config.get('dropout', 0.2)
        self.lr = lstm_config.get('learning_rate', 0.001)
        self.batch_size = lstm_config.get('batch_size', 32)
        self.epochs = lstm_config.get('epochs', 50)

    def build_model(self) -> None:
        """Build the LSTM model architecture."""
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features))

        # LSTM layers
        x = Bidirectional(LSTM(self.units, return_sequences=True))(inputs)
        x = Dropout(self.dropout)(x)
        x = Bidirectional(LSTM(self.units, return_sequences=True))(x)
        x = Dropout(self.dropout)(x)

        # Attention mechanism
        x = AttentionLayer()(x)
        x = Dropout(self.dropout)(x)

        # Common dense layers
        x = Dense(32, activation='relu')(x)

        # Multiple outputs

        # 1. Direction prediction (binary classification)
        direction_output = Dense(1, activation='sigmoid', name='direction')(x)

        # 2. Magnitude prediction (regression)
        magnitude_output = Dense(1, activation='linear', name='magnitude')(x)

        # 3. Volatility prediction (regression, always positive)
        volatility_output = Dense(1, activation='relu', name='volatility')(x)

        # Create model with multiple outputs
        self.model = Model(
            inputs=inputs,
            outputs=[direction_output, magnitude_output, volatility_output]
        )

        # Compile the model
        self.compile_model()

    def compile_model(self) -> None:
        """Compile the model with appropriate loss functions and metrics."""
        # Define losses for each output
        losses = {
            'direction': 'binary_crossentropy',
            'magnitude': 'mse',
            'volatility': 'mse'
        }

        # Define metrics for each output
        metrics = {
            'direction': ['accuracy'],
            'magnitude': ['mae'],
            'volatility': ['mae']
        }

        # Define loss weights to balance the objectives
        loss_weights = {
            'direction': 1.0,
            'magnitude': 0.5,
            'volatility': 0.3
        }

        # Compile model with Adam optimizer and learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss=losses,
            metrics=metrics,
            loss_weights=loss_weights
        )

    def fit(self, dataset: Dict[str, Any], epochs: int = None, batch_size: int = None,
            callbacks: List = None) -> Dict[str, Any]:
        """Train the model.

        Args:
            dataset: Dictionary containing X_train, y_train, X_val, y_val
            epochs: Number of epochs to train
            batch_size: Batch size for training
            callbacks: List of Keras callbacks

        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()

        # Use parameters or defaults
        epochs = epochs or self.epochs
        batch_size = batch_size or self.batch_size

        # Prepare target data format for model
        y_train = {
            'direction': dataset['y_train']['direction'],
            'magnitude': dataset['y_train']['magnitude'].reshape(-1, 1),
            'volatility': dataset['y_train']['volatility'].reshape(-1, 1) if 'volatility' in dataset['y_train'] else
            np.zeros((len(dataset['y_train']['direction']), 1))
        }

        y_val = {
            'direction': dataset['y_val']['direction'],
            'magnitude': dataset['y_val']['magnitude'].reshape(-1, 1),
            'volatility': dataset['y_val']['volatility'].reshape(-1, 1) if 'volatility' in dataset['y_val'] else
            np.zeros((len(dataset['y_val']['direction']), 1))
        }

        # Define default callbacks if none provided
        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor='val_direction_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    mode='max'
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001
                )
            ]

        # Train the model
        self.history = self.model.fit(
            dataset['X_train'],
            y_train,
            validation_data=(dataset['X_val'], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )

        return self.history.history

    def predict(self, X_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Make predictions.

        Args:
            X_data: Input feature sequences

        Returns:
            Dictionary with predictions for direction, magnitude and volatility
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() or load_model() first.")

        # Make predictions
        predictions = self.model.predict(X_data)

        # Return results as dictionary
        return {
            'direction': predictions[0].flatten(),
            'magnitude': predictions[1].flatten(),
            'volatility': predictions[2].flatten()
        }

    def evaluate(self, X_test: np.ndarray, y_test: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Evaluate model performance.

        Args:
            X_test: Test feature sequences
            y_test: Dictionary of test targets

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build_model() or load_model() first.")

        # Format test data
        y_test_formatted = {
            'direction': y_test['direction'],
            'magnitude': y_test['magnitude'].reshape(-1, 1),
            'volatility': y_test['volatility'].reshape(-1, 1) if 'volatility' in y_test else
            np.zeros((len(y_test['direction']), 1))
        }

        # Evaluate the model
        results = self.model.evaluate(X_test, y_test_formatted, verbose=0)

        # Create metrics dictionary
        metrics = {}
        for i, metric_name in enumerate(self.model.metrics_names):
            metrics[metric_name] = results[i]

        return metrics

    def save_model(self, path: str) -> None:
        """Save the model to file."""

        if self.model is None:
            raise ValueError("No model to save. Build or load a model first.")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save the model with custom objects
        self.model.save(path, save_format='h5')
        # keras.saving.save_model(self.model, path, save_format='h5')

    def load_model(self, path: str) -> None:
        """Load a model from file."""
        try:
            if not os.path.exists(path):
                # self.logger.error(f"Model file not found: {path}")
                raise FileNotFoundError(f"Model file not found: {path}")

            # Create a new model instance with the correct architecture first
            # This ensures the AttentionLayer is registered properly

            # Load as a standard TF SavedModel (without custom objects) to extract shape info
            temp_model = None
            try:
                # Try to load just to get input shape
                temp_model = tf.keras.models.load_model(path, compile=False, custom_objects={})
            except Exception as shape_error:
                # self.logger.warning(f"Could not directly load model to get shape: {shape_error}")
                # If that fails, try to manually build a model with the default shape
                # self.logger.info("Using default model architecture")

                # Default architecture settings if we can't load
                input_shape = (self.sequence_length, self.n_features)
                self.build_model()
                return

            # If we got here, we have the shape info from the loaded model
            input_shape = temp_model.input_shape[1:]  # Remove batch dimension
            n_features = input_shape[1]

            # self.logger.info(f"Recreating model architecture with input shape {input_shape}")

            # Build a new model with the exact same architecture
            # This pre-registers the AttentionLayer in the current scope
            inputs = tf.keras.Input(shape=input_shape)
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.units, return_sequences=True))(inputs)
            x = tf.keras.layers.Dropout(self.dropout)(x)
            x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.units, return_sequences=True))(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)

            # Attention mechanism
            # Directly use the registered AttentionLayer here
            # from Models.LSTMModel import AttentionLayer
            x = AttentionLayer()(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)

            # Common dense layers
            x = tf.keras.layers.Dense(32, activation='relu')(x)

            # Multiple outputs
            direction_output = tf.keras.layers.Dense(1, activation='sigmoid', name='direction')(x)
            magnitude_output = tf.keras.layers.Dense(1, activation='linear', name='magnitude')(x)
            volatility_output = tf.keras.layers.Dense(1, activation='relu', name='volatility')(x)

            # Create new model
            self.model = tf.keras.Model(
                inputs=inputs,
                outputs=[direction_output, magnitude_output, volatility_output]
            )

            # Now try to load the weights
            # self.logger.info("Attempting to load weights from saved model")
            try:
                # Use a custom object scope for loading weights
                with tf.keras.utils.custom_object_scope({'AttentionLayer': AttentionLayer}):
                    # Just load the weights from the saved model
                    self.model.load_weights(path)
                    # self.logger.info("Successfully loaded weights")
            except Exception as weights_error:
                print("""
                # self.logger.error(f"Error loading weights: {weights_error}")
                # self.logger.info("Using newly initialized weights instead")
                """)
                # self.logger.error(f"Error loading weights: {weights_error}")
                # self.logger.info("Using newly initialized weights instead")

            # Compile the model
            self.compile_model()
            # self.logger.info(f"Model ready for training with input shape {input_shape}")

        except Exception as e:
            # self.logger.error(f"Failed to load model: {e}")
            # Create a new model with default architecture as fallback
            # self.logger.info("Creating new model with default architecture as fallback")
            self.build_model()
            raise


    def get_model_summary(self) -> str:
        """Get model summary as a string."""

        if self.model is None:
            return "Model not built yet"

        # Capture model summary as string
        stringlist = []
        self.model.summary(print_fn=lambda x: stringlist.append(x))
        return '\n'.join(stringlist)
