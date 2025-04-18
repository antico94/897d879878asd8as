from typing import Dict, Any, Tuple, Optional
from SignalGenerator import SignalType


class RiskManager:
    """Handles position sizing and risk management."""

    def __init__(self, config, logger, account_info: Dict[str, Any] = None):
        """Initialize the risk manager."""
        self.config = config
        self.logger = logger
        self.account_info = account_info or {}

        # Load risk management parameters from config
        self.risk_params = self._load_risk_parameters()

    def _load_risk_parameters(self) -> Dict[str, Any]:
        """Load risk management parameters from configuration."""
        # Try to load from config
        trading_settings = self.config.get('GoldTradingSettings', {})
        risk_config = trading_settings.get('RiskManagement', {})

        # Default risk parameters
        default_params = {
            'max_risk_per_trade': 0.02,  # Maximum account % to risk per trade
            'base_risk_per_trade': 0.01,  # Base risk % for moderate signals
            'max_position_size_factor': 0.1,  # Maximum position size as % of account
            'stop_loss_atr_multiplier': 1.5,  # ATR multiplier for stop loss calculation
            'stop_loss_volatility_factor': 0.5,  # Factor to adjust stop based on predicted volatility
            'take_profit_risk_ratio': 2.0,  # Risk:reward ratio for take profit calculation
            'partial_take_profit_level': 0.5,  # Level at which to take partial profits (R multiple)
            'partial_take_profit_pct': 0.5,  # Percentage of position to close at partial TP
            'breakeven_level': 1.0,  # Move to breakeven at this R multiple
        }

        # Update with values from config if they exist
        params = {}
        for key, default_value in default_params.items():
            params[key] = risk_config.get(key, default_value)

        self.logger.info(f"Risk parameters loaded: {params}")
        return params

    def calculate_position_size(self, signal: Dict[str, Any],
                                stop_loss_pips: float,
                                account_balance: Optional[float] = None) -> float:
        """Calculate appropriate position size based on risk parameters."""
        try:
            # Get account balance from account info if not provided
            if account_balance is None:
                account_balance = self.account_info.get('balance', 10000.0)  # Default to 10,000 if not available

            # Adjust risk based on signal strength
            signal_strength = signal.get('signal_strength', 0.5)

            # Determine risk percentage based on signal type
            signal_type = signal.get('type')

            if signal_type in [SignalType.STRONG_BUY, SignalType.STRONG_SELL]:
                risk_pct = self.risk_params['base_risk_per_trade'] * 1.5
            elif signal_type in [SignalType.MODERATE_BUY, SignalType.MODERATE_SELL]:
                risk_pct = self.risk_params['base_risk_per_trade']
            elif signal_type in [SignalType.WEAK_BUY, SignalType.WEAK_SELL]:
                risk_pct = self.risk_params['base_risk_per_trade'] * 0.7
            else:
                risk_pct = self.risk_params['base_risk_per_trade'] * 0.5

            # Ensure we don't exceed maximum risk
            risk_pct = min(risk_pct, self.risk_params['max_risk_per_trade'])

            # Calculate risk amount in account currency
            risk_amount = account_balance * risk_pct

            # For XAUUSD (Gold), convert pips to dollars
            # 1 pip in XAUUSD is typically 0.1 USD per 0.01 lot (micro lot)
            pip_value_per_lot = 0.1 * 100  # Value per full lot (100 micro lots)

            # Calculate position size in lots
            if stop_loss_pips > 0:
                position_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
            else:
                self.logger.warning("Stop loss is zero or negative, using minimum position size")
                position_size = 0.01  # Minimum position size

            # Cap position size based on account size
            max_position_size = account_balance * self.risk_params[
                'max_position_size_factor'] / 100000  # Convert to lots
            position_size = min(position_size, max_position_size)

            # Round to 2 decimal places (0.01 lot precision)
            position_size = round(position_size, 2)

            # Ensure minimum position size of 0.01 lots
            position_size = max(position_size, 0.01)

            self.logger.info(f"Calculated position size: {position_size} lots based on "
                             f"risk: {risk_pct * 100:.1f}%, stop loss: {stop_loss_pips} pips")

            return position_size

        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.01  # Return minimum position size on error

    def calculate_stop_loss(self, current_price: float, predicted_volatility: float,
                            atr_value: float, signal_type: SignalType) -> float:
        """Calculate optimal stop loss price based on volatility and ATR."""
        try:
            # Adjust ATR multiplier based on predicted volatility
            # Higher predicted volatility = wider stops
            volatility_factor = 1.0
            if predicted_volatility > 0:
                # Normalize volatility (typical range 0.1% - 2%)
                norm_volatility = min(predicted_volatility / 2.0, 1.0)
                volatility_factor = 1.0 + (norm_volatility * self.risk_params['stop_loss_volatility_factor'])

            # Calculate base stop distance
            base_stop_distance = atr_value * self.risk_params['stop_loss_atr_multiplier'] * volatility_factor

            # Calculate stop loss price based on signal type
            if signal_type in [SignalType.STRONG_BUY, SignalType.MODERATE_BUY, SignalType.WEAK_BUY]:
                # Long position: Stop below current price
                stop_loss_price = current_price - base_stop_distance
            elif signal_type in [SignalType.STRONG_SELL, SignalType.MODERATE_SELL, SignalType.WEAK_SELL]:
                # Short position: Stop above current price
                stop_loss_price = current_price + base_stop_distance
            else:
                # Default to ATR-based stop below price
                stop_loss_price = current_price - base_stop_distance

            # Round to appropriate precision (2 decimal places for XAUUSD)
            stop_loss_price = round(stop_loss_price, 2)

            self.logger.info(f"Calculated stop loss: {stop_loss_price} based on ATR: {atr_value}, "
                             f"volatility factor: {volatility_factor:.2f}")

            return stop_loss_price

        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {e}")
            # Fallback to simple stop (1% away from current price)
            if signal_type in [SignalType.STRONG_BUY, SignalType.MODERATE_BUY, SignalType.WEAK_BUY]:
                return round(current_price * 0.99, 2)
            else:
                return round(current_price * 1.01, 2)

    def calculate_take_profit(self, current_price: float, entry_price: float, stop_loss_price: float,
                              predicted_magnitude: float, signal_type: SignalType) -> float:
        """Calculate take profit level based on risk:reward and predicted magnitude."""
        try:
            # Calculate risk in price terms
            if signal_type in [SignalType.STRONG_BUY, SignalType.MODERATE_BUY, SignalType.WEAK_BUY]:
                risk = entry_price - stop_loss_price
                # Adjust direction based on predicted magnitude
                direction = 1
            else:
                risk = stop_loss_price - entry_price
                # For short positions
                direction = -1

            # Base take profit on risk:reward ratio
            base_tp_distance = risk * self.risk_params['take_profit_risk_ratio']

            # Adjust based on predicted magnitude if available
            if predicted_magnitude > 0:
                # Convert percentage magnitude to price
                predicted_price_move = current_price * (predicted_magnitude / 100) * direction

                # Use weighted average of risk:reward and predicted move
                tp_distance = (0.7 * base_tp_distance) + (0.3 * predicted_price_move)
            else:
                tp_distance = base_tp_distance

            # Calculate take profit price
            if signal_type in [SignalType.STRONG_BUY, SignalType.MODERATE_BUY, SignalType.WEAK_BUY]:
                take_profit_price = entry_price + tp_distance
            else:
                take_profit_price = entry_price - tp_distance

            # Round to appropriate precision
            take_profit_price = round(take_profit_price, 2)

            self.logger.info(f"Calculated take profit: {take_profit_price} with risk:reward ratio "
                             f"{self.risk_params['take_profit_risk_ratio']:.1f}")

            return take_profit_price

        except Exception as e:
            self.logger.error(f"Error calculating take profit: {e}")
            # Fallback to simple take profit (2% from current price)
            if signal_type in [SignalType.STRONG_BUY, SignalType.MODERATE_BUY, SignalType.WEAK_BUY]:
                return round(current_price * 1.02, 2)
            else:
                return round(current_price * 0.98, 2)

    def calculate_breakeven_level(self, entry_price: float, stop_loss_price: float,
                                  signal_type: SignalType) -> float:
        """Calculate level at which to move stop loss to breakeven."""
        try:
            # Calculate risk in price terms
            if signal_type in [SignalType.STRONG_BUY, SignalType.MODERATE_BUY, SignalType.WEAK_BUY]:
                risk = entry_price - stop_loss_price
                # Level = entry + (risk * breakeven_level)
                breakeven_level = entry_price + (risk * self.risk_params['breakeven_level'])
            else:
                risk = stop_loss_price - entry_price
                # Level = entry - (risk * breakeven_level)
                breakeven_level = entry_price - (risk * self.risk_params['breakeven_level'])

            # Round to appropriate precision
            breakeven_level = round(breakeven_level, 2)

            return breakeven_level

        except Exception as e:
            self.logger.error(f"Error calculating breakeven level: {e}")
            # Fallback to simple calculation (0.5% from entry)
            if signal_type in [SignalType.STRONG_BUY, SignalType.MODERATE_BUY, SignalType.WEAK_BUY]:
                return round(entry_price * 1.005, 2)
            else:
                return round(entry_price * 0.995, 2)

    def calculate_partial_profit_level(self, entry_price: float, stop_loss_price: float,
                                       take_profit_price: float, signal_type: SignalType) -> float:
        """Calculate level at which to take partial profits."""
        try:
            # Use partial take profit level setting
            partial_level = self.risk_params['partial_take_profit_level']

            # Calculate partial profit level based on entry, stop, and take profit
            if signal_type in [SignalType.STRONG_BUY, SignalType.MODERATE_BUY, SignalType.WEAK_BUY]:
                risk = entry_price - stop_loss_price
                reward = take_profit_price - entry_price
                partial_profit = entry_price + (reward * partial_level)
            else:
                risk = stop_loss_price - entry_price
                reward = entry_price - take_profit_price
                partial_profit = entry_price - (reward * partial_level)

            # Round to appropriate precision
            partial_profit = round(partial_profit, 2)

            return partial_profit

        except Exception as e:
            self.logger.error(f"Error calculating partial profit level: {e}")
            # Fallback to midpoint between entry and take profit
            if signal_type in [SignalType.STRONG_BUY, SignalType.MODERATE_BUY, SignalType.WEAK_BUY]:
                return round(entry_price + ((take_profit_price - entry_price) * 0.5), 2)
            else:
                return round(entry_price - ((entry_price - take_profit_price) * 0.5), 2)

    def get_risk_reward_metrics(self, entry_price: float, stop_loss_price: float,
                                take_profit_price: float) -> Dict[str, float]:
        """Calculate risk:reward metrics for a trade."""
        try:
            # Calculate risk and reward in price terms
            risk = abs(entry_price - stop_loss_price)
            reward = abs(entry_price - take_profit_price)

            # Calculate R:R ratio
            risk_reward_ratio = reward / risk if risk > 0 else 0

            # Calculate R-multiple (how many R's the trade is worth)
            r_multiple = risk_reward_ratio

            return {
                'risk_price': risk,
                'reward_price': reward,
                'risk_reward_ratio': risk_reward_ratio,
                'r_multiple': r_multiple
            }

        except Exception as e:
            self.logger.error(f"Error calculating risk:reward metrics: {e}")
            return {
                'risk_price': 0,
                'reward_price': 0,
                'risk_reward_ratio': 0,
                'r_multiple': 0
            }