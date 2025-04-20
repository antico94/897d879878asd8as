import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List


class DataPreparer:
    """Helper class to prepare data for model prediction during live trading."""

    def __init__(self, logger, feature_list=None):
        self.logger = logger
        self.scalers = {}
        # Explicitly provide only 7 features to match the model's expectation
        self.feature_list = feature_list or [
            "macd_histogram", "stoch_k", "resistance2", "close_pct_change_3",
            "close_pct_change_5", "high_pct_change_3", "rsi"
        ]
        # Log the features this preparer will use
        self.logger.info(f"DataPreparer initialized with 7 features: {self.feature_list}")
    def update_feature_list(self, new_feature_list):
        """Update the feature list to match the model's expected features."""
        self.logger.info(f"Updating feature list from {self.feature_list} to {new_feature_list}")
        self.feature_list = new_feature_list
        # Reset scalers when changing features
        self.scalers = {}

    def prepare_sequence(self, market_data: Dict[str, Any], historical_data: pd.DataFrame) -> np.ndarray:
        """Prepare data sequence for model prediction."""
        try:
            sequence_length = 24  # Get this from model.sequence_length in real implementation

            # Log which features we're preparing for trading
            self.logger.info(f"Preparing sequence for trading with features: {self.feature_list}")

            # Ensure we have enough historical data
            if len(historical_data) < sequence_length:
                self.logger.error(
                    f"Not enough historical data for sequence. Need {sequence_length}, got {len(historical_data)}")
                # Return zeros array as fallback
                return np.zeros((1, sequence_length, len(self.feature_list)))

            # Extract relevant features from historical data
            available_features = [f for f in self.feature_list if f in historical_data.columns]
            if not available_features:
                self.logger.error(f"None of the required features found in historical data")
                return np.zeros((1, sequence_length, len(self.feature_list)))

            # Get the most recent sequence_length data points
            recent_data = historical_data.iloc[-sequence_length:].copy()
            features_df = recent_data[available_features]

            # Scale features
            scaled_features = self._scale_features(features_df)

            # Format as 3D array for LSTM: (1, sequence_length, n_features)
            sequence = scaled_features.values.reshape(1, sequence_length, len(available_features))

            return sequence

        except Exception as e:
            self.logger.error(f"Error preparing sequence for prediction: {e}")
            return np.zeros((1, sequence_length, len(self.feature_list)))

    def _scale_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using StandardScaler."""
        scaled_df = features_df.copy()

        # Fit or use existing scalers for each feature
        for column in features_df.columns:
            if column not in self.scalers:
                self.scalers[column] = StandardScaler()
                self.scalers[column].fit(features_df[[column]])

            # Scale the feature
            scaled_df[column] = self.scalers[column].transform(features_df[[column]])

        return scaled_df
