import numpy as np
from typing import Dict, List, Any, Tuple, Optional, Union
from enum import Enum


class SignalType(Enum):
    """Trading signal types."""
    STRONG_BUY = "STRONG_BUY"
    MODERATE_BUY = "MODERATE_BUY"
    WEAK_BUY = "WEAK_BUY"
    NEUTRAL = "NEUTRAL"
    WEAK_SELL = "WEAK_SELL"
    MODERATE_SELL = "MODERATE_SELL"
    STRONG_SELL = "STRONG_SELL"


class SignalGenerator:
    """Converts model predictions into actionable trading signals."""

    def __init__(self, config, logger):
        """Initialize the signal generator.

        Args:
            config: Application configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger

        # Load signal thresholds from config or use defaults
        self.thresholds = self._load_thresholds()

    def _load_thresholds(self) -> Dict[str, float]:
        """Load signal thresholds from configuration."""
        # Try to load from config
        trading_settings = self.config.get('GoldTradingSettings', {})
        signal_config = trading_settings.get('SignalGeneration', {})

        # Default thresholds
        default_thresholds = {
            'strong_threshold': 0.75,  # Probability threshold for strong signals
            'moderate_threshold': 0.65,  # Probability threshold for moderate signals
            'weak_threshold': 0.55,  # Probability threshold for weak signals
            'min_confidence': 0.6,  # Minimum confidence to generate any signal
            'min_magnitude': 0.1,  # Minimum expected magnitude (%) to generate a signal
        }

        # Update with values from config if they exist
        thresholds = {}
        for key, default_value in default_thresholds.items():
            thresholds[key] = signal_config.get(key, default_value)

        self.logger.info(f"Signal thresholds loaded: {thresholds}")
        return thresholds

    def generate_signals(self, predictions: Dict[str, np.ndarray],
                         confidence: Optional[np.ndarray] = None,
                         current_market_data: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate trading signals from model predictions.

        Args:
            predictions: Dictionary with model predictions (direction, magnitude, volatility)
            confidence: Optional array with prediction confidence scores
            current_market_data: Optional dictionary with current market conditions

        Returns:
            List of signal dictionaries
        """
        try:
            signals = []

            # Get required predictions
            direction_probs = predictions.get('direction')
            magnitudes = predictions.get('magnitude')
            volatilities = predictions.get('volatility', None)

            if direction_probs is None or magnitudes is None:
                self.logger.error("Missing required predictions for signal generation")
                return []

            # If no confidence provided, assume maximum confidence
            if confidence is None:
                confidence = np.ones_like(direction_probs)

            # Process each prediction
            for i in range(len(direction_probs)):
                # Skip if confidence is too low
                if confidence[i] < self.thresholds['min_confidence']:
                    continue

                # Skip if expected magnitude is too small
                if abs(magnitudes[i]) < self.thresholds['min_magnitude']:
                    continue

                # Determine signal type based on direction probability
                direction_prob = direction_probs[i]
                magnitude = magnitudes[i]

                # Determine signal type based on direction probability
                signal_type = self._determine_signal_type(float(direction_prob), float(magnitude))

                # Skip neutral signals
                if signal_type == SignalType.NEUTRAL:
                    continue

                # Create signal dictionary
                signal = {
                    'type': signal_type,
                    'direction_probability': float(direction_prob),
                    'expected_magnitude': float(magnitude),
                    'confidence': float(confidence[i]),
                    'signal_strength': self.calculate_signal_strength(
                        float(direction_prob),
                        float(confidence[i]),
                        float(magnitude)
                    )
                }

                # Add volatility if available
                if volatilities is not None:
                    signal['expected_volatility'] = float(volatilities[i])

                # Add timestamp if available in market data
                if current_market_data and 'timestamp' in current_market_data:
                    signal['timestamp'] = current_market_data['timestamp']

                # Add price information if available
                if current_market_data and 'price' in current_market_data:
                    signal['price'] = current_market_data['price']

                signals.append(signal)

            self.logger.info(f"Generated {len(signals)} trading signals")
            return signals

        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []

    def _determine_signal_type(self, direction_prob: float, magnitude: float) -> SignalType:
        """Determine signal type based on direction probability and magnitude.

        Args:
            direction_prob: Probability of price going up (0-1)
            magnitude: Expected price movement magnitude

        Returns:
            SignalType enum value
        """
        # Ensure direction_prob is a float, not ndarray
        direction_prob = float(direction_prob)
        magnitude = float(magnitude)

        # Determine signal based on direction probability
        if direction_prob >= self.thresholds['strong_threshold']:
            return SignalType.STRONG_BUY
        elif direction_prob >= self.thresholds['moderate_threshold']:
            return SignalType.MODERATE_BUY
        elif direction_prob >= self.thresholds['weak_threshold']:
            return SignalType.WEAK_BUY
        elif direction_prob <= (1 - self.thresholds['strong_threshold']):
            return SignalType.STRONG_SELL
        elif direction_prob <= (1 - self.thresholds['moderate_threshold']):
            return SignalType.MODERATE_SELL
        elif direction_prob <= (1 - self.thresholds['weak_threshold']):
            return SignalType.WEAK_SELL
        else:
            return SignalType.NEUTRAL

    def calculate_signal_strength(self, direction_probability: float,
                                  confidence: float, magnitude: float) -> float:
        """Calculate signal strength for position sizing.

        Args:
            direction_probability: Probability of price going up (0-1)
            confidence: Model confidence in the prediction (0-1)
            magnitude: Expected price movement magnitude

        Returns:
            Signal strength score (0-1)
        """
        # Ensure all inputs are float values, not numpy arrays
        direction_probability = float(direction_probability)
        confidence = float(confidence)
        magnitude = float(magnitude)

        # Convert direction probability to conviction (0 = neutral, 1 = maximum conviction)
        # Max conviction is at 0 or 1, minimum at 0.5
        direction_conviction = abs(direction_probability - 0.5) * 2

        # Normalize magnitude (typical gold moves are 0.1-2.0%)
        # Cap at 2% to avoid overweighting outliers
        norm_magnitude = min(abs(magnitude) / 2.0, 1.0)

        # Calculate weighted signal strength
        signal_strength = (
                0.5 * direction_conviction +  # Direction conviction (50% weight)
                0.3 * confidence +  # Model confidence (30% weight)
                0.2 * norm_magnitude  # Expected magnitude (20% weight)
        )

        return signal_strength

    def filter_signals(self, signals: List[Dict[str, Any]],
                       min_strength: float = 0.5) -> List[Dict[str, Any]]:
        """Filter signals based on strength threshold.

        Args:
            signals: List of signal dictionaries
            min_strength: Minimum signal strength to include

        Returns:
            Filtered list of signals
        """
        filtered_signals = [s for s in signals if s['signal_strength'] >= min_strength]

        self.logger.info(f"Filtered {len(signals)} signals to {len(filtered_signals)} "
                         f"with min_strength={min_strength}")

        return filtered_signals