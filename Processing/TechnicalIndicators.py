import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, List, Tuple


class TechnicalIndicators:
    """Calculate technical indicators for financial market data."""

    @staticmethod
    def calculate_sma(data: pd.DataFrame, column: str = 'close',
                      periods: List[int] = [5, 8, 13, 21, 50, 200]) -> pd.DataFrame:
        """Calculate Simple Moving Averages."""
        result = data.copy()
        for period in periods:
            result[f'sma_{period}'] = result[column].rolling(window=period).mean()
        return result

    @staticmethod
    def calculate_ema(data: pd.DataFrame, column: str = 'close',
                      periods: List[int] = [5, 8, 13, 21, 50, 200]) -> pd.DataFrame:
        """Calculate Exponential Moving Averages."""
        result = data.copy()
        for period in periods:
            result[f'ema_{period}'] = result[column].ewm(span=period, adjust=False).mean()
        return result

    @staticmethod
    def calculate_macd(data: pd.DataFrame, column: str = 'close',
                       fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        result = data.copy()
        # Fast EMA
        fast_ema = result[column].ewm(span=fast_period, adjust=False).mean()
        # Slow EMA
        slow_ema = result[column].ewm(span=slow_period, adjust=False).mean()
        # MACD Line
        result['macd_line'] = fast_ema - slow_ema
        # Signal Line
        result['macd_signal'] = result['macd_line'].ewm(span=signal_period, adjust=False).mean()
        # MACD Histogram
        result['macd_histogram'] = result['macd_line'] - result['macd_signal']
        return result

    @staticmethod
    def calculate_rsi(data: pd.DataFrame, column: str = 'close', period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index."""
        result = data.copy()
        delta = result[column].diff()

        # Make two series: one for gains and one for losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # First value is sum of gains or losses
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        result['rsi'] = 100 - (100 / (1 + rs))
        return result

    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, column: str = 'close',
                                  period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        result = data.copy()
        result['bb_middle'] = result[column].rolling(window=period).mean()
        result['bb_std'] = result[column].rolling(window=period).std()
        result['bb_upper'] = result['bb_middle'] + (result['bb_std'] * num_std)
        result['bb_lower'] = result['bb_middle'] - (result['bb_std'] * num_std)
        result['bb_width'] = (result['bb_upper'] - result['bb_lower']) / result['bb_middle']
        return result

    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        result = data.copy()
        high = result['high']
        low = result['low']
        close = result['close']

        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result['atr'] = tr.rolling(window=period).mean()
        return result

    @staticmethod
    def calculate_stochastic(data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        result = data.copy()
        # %K calculation
        lowest_low = result['low'].rolling(window=k_period).min()
        highest_high = result['high'].rolling(window=k_period).max()
        result['stoch_k'] = 100 * ((result['close'] - lowest_low) / (highest_high - lowest_low))

        # %D calculation (3-period SMA of %K)
        result['stoch_d'] = result['stoch_k'].rolling(window=d_period).mean()
        return result

    @staticmethod
    def calculate_pivot_points(data: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
        """Calculate daily pivot points based on previous day's data.
        Methods: 'standard', 'fibonacci', 'woodie', 'camarilla'
        """
        result = data.copy()

        # Get high, low, close for pivot calculation (previous day)
        high = result['high'].shift(1)
        low = result['low'].shift(1)
        close = result['close'].shift(1)

        if method == 'standard':
            # Standard pivot points
            pp = (high + low + close) / 3
            s1 = (2 * pp) - high
            s2 = pp - (high - low)
            r1 = (2 * pp) - low
            r2 = pp + (high - low)

            result['pivot'] = pp
            result['support1'] = s1
            result['support2'] = s2
            result['resistance1'] = r1
            result['resistance2'] = r2

        elif method == 'fibonacci':
            # Fibonacci pivot points
            pp = (high + low + close) / 3
            r1 = pp + 0.382 * (high - low)
            r2 = pp + 0.618 * (high - low)
            r3 = pp + (high - low)
            s1 = pp - 0.382 * (high - low)
            s2 = pp - 0.618 * (high - low)
            s3 = pp - (high - low)

            result['pivot'] = pp
            result['support1'] = s1
            result['support2'] = s2
            result['support3'] = s3
            result['resistance1'] = r1
            result['resistance2'] = r2
            result['resistance3'] = r3

        return result

    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        result = data.copy()

        # Calculate all indicators
        result = TechnicalIndicators.calculate_sma(result)
        result = TechnicalIndicators.calculate_ema(result)
        result = TechnicalIndicators.calculate_macd(result)
        result = TechnicalIndicators.calculate_rsi(result)
        result = TechnicalIndicators.calculate_bollinger_bands(result)
        result = TechnicalIndicators.calculate_atr(result)
        result = TechnicalIndicators.calculate_stochastic(result)
        result = TechnicalIndicators.calculate_pivot_points(result)

        return result
