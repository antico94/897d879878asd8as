from typing import Dict, List, Any, Tuple, Optional, Union
import time
from datetime import datetime
from enum import Enum

from Strategies.SignalGenerator import SignalType
from Strategies.RiskManager import RiskManager


class OrderType(Enum):
    """Types of orders to execute."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderDirection(Enum):
    """Direction of the trade."""
    BUY = "BUY"
    SELL = "SELL"


class TradeExecutor:
    """Handles the execution of trades based on signals and risk parameters."""

    def __init__(self, config, logger, risk_manager: RiskManager, mt5_connector):
        """Initialize the trade executor.

        Args:
            config: Application configuration
            logger: Logger instance
            risk_manager: Risk manager instance
            mt5_connector: Connector to MetaTrader 5
        """
        self.config = config
        self.logger = logger
        self.risk_manager = risk_manager
        self.mt5_connector = mt5_connector

        # Track active trades
        self.active_trades = {}

        # Load trading parameters from config
        self.trade_params = self._load_trade_parameters()

    def _load_trade_parameters(self) -> Dict[str, Any]:
        """Load trading parameters from configuration."""
        # Try to load from config
        trading_settings = self.config.get('GoldTradingSettings', {})
        trade_config = trading_settings.get('Trading', {})

        # Default parameters
        default_params = {
            'slippage_pips': 3,  # Allowed slippage in pips
            'max_open_trades': 5,  # Maximum number of open trades
            'use_market_orders': True,  # Use market orders (True) or limit orders (False)
            'retry_attempts': 3,  # Number of retry attempts on failed orders
            'retry_delay_seconds': 2,  # Delay between retry attempts
            'comment': 'ML Trading Bot',  # Comment for trades
            'magic_number': 12345,  # Magic number for identifying bot trades
            'partial_close_enabled': True,  # Whether to use partial close
            'breakeven_enabled': True,  # Whether to move to breakeven
            'trailing_stop_enabled': True,  # Whether to use trailing stops
            'trailing_stop_activation': 1.0,  # R multiple for trailing stop activation
            'trailing_stop_distance': 0.5  # R multiple distance for trailing stop
        }

        # Update with values from config if they exist
        params = {}
        for key, default_value in default_params.items():
            params[key] = trade_config.get(key, default_value)

        self.logger.info(f"Trade parameters loaded: {params}")
        return params

    def execute_trade(self, signal: Dict[str, Any], current_market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a trade based on signal and current market data.

        Args:
            signal: Trading signal dictionary
            current_market_data: Dictionary with current market conditions

        Returns:
            Dictionary with trade details or empty dict on failure
        """
        try:
            self.logger.info(f"Executing trade for signal: {signal['type'].value}")

            # Check if we are already at max open trades
            if len(self.active_trades) >= self.trade_params['max_open_trades']:
                self.logger.warning(
                    f"Maximum open trades reached ({self.trade_params['max_open_trades']}), not opening new trade")
                return {}

            # Get current price from market data
            current_price = current_market_data.get('price')
            if not current_price:
                self.logger.error("No current price provided in market data")
                return {}

            # Get ATR for stop loss calculation
            atr_value = current_market_data.get('atr', 0)
            if not atr_value:
                self.logger.warning("No ATR value provided, using default")
                atr_value = current_price * 0.01  # Default to 1% of price

            # Get signal details
            signal_type = signal['type']
            expected_magnitude = signal.get('expected_magnitude', 0)
            expected_volatility = signal.get('expected_volatility', 0)

            # Determine trade direction
            order_direction = OrderDirection.BUY if signal_type in [
                SignalType.STRONG_BUY, SignalType.MODERATE_BUY, SignalType.WEAK_BUY
            ] else OrderDirection.SELL

            # Calculate stop loss price
            stop_loss_price = self.risk_manager.calculate_stop_loss(
                current_price, expected_volatility, atr_value, signal_type
            )

            # Calculate stop loss distance in pips
            stop_loss_pips = abs(current_price - stop_loss_price) * 10  # Convert to pips (0.1 increments)

            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                signal, stop_loss_pips, current_market_data.get('account_balance')
            )

            # Calculate take profit price
            take_profit_price = self.risk_manager.calculate_take_profit(
                current_price, current_price, stop_loss_price, expected_magnitude, signal_type
            )

            # Determine order type (market or limit)
            order_type = OrderType.MARKET
            entry_price = current_price  # For market orders, use current price

            # Execute the trade
            trade_result = self._send_order(
                order_direction,
                order_type,
                current_market_data['symbol'],
                position_size,
                entry_price,
                stop_loss_price,
                take_profit_price
            )

            if not trade_result:
                self.logger.error("Failed to execute trade")
                return {}

            # Create trade record
            trade_id = trade_result.get('ticket', int(time.time()))

            trade_record = {
                'id': trade_id,
                'symbol': current_market_data['symbol'],
                'direction': order_direction.value,
                'entry_price': entry_price,
                'position_size': position_size,
                'stop_loss': stop_loss_price,
                'take_profit': take_profit_price,
                'signal_type': signal_type.value,
                'open_time': datetime.now().isoformat(),
                'status': 'OPEN',
                'breakeven_level': self.risk_manager.calculate_breakeven_level(
                    entry_price, stop_loss_price, signal_type
                ),
                'partial_profit_level': self.risk_manager.calculate_partial_profit_level(
                    entry_price, stop_loss_price, take_profit_price, signal_type
                ),
                'partial_close_done': False,
                'moved_to_breakeven': False,
                'risk_metrics': self.risk_manager.get_risk_reward_metrics(
                    entry_price, stop_loss_price, take_profit_price
                ),
                'trade_result': trade_result
            }

            # Add to active trades
            self.active_trades[trade_id] = trade_record

            self.logger.info(f"Trade executed successfully: {order_direction.value} {position_size} lots "
                             f"at {entry_price}, SL: {stop_loss_price}, TP: {take_profit_price}")

            return trade_record

        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return {}

    def _send_order(self, direction: OrderDirection, order_type: OrderType,
                    symbol: str, lot_size: float, price: float,
                    stop_loss: float, take_profit: float) -> Dict[str, Any]:
        """Send order to MetaTrader 5.

        Args:
            direction: Buy or Sell
            order_type: Market, Limit, or Stop
            symbol: Trading symbol (e.g., "XAUUSD")
            lot_size: Position size in lots
            price: Entry price (for limit/stop orders)
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Dictionary with order result or empty dict on failure
        """
        try:
            # Try to connect to MT5 if not connected
            if not self.mt5_connector.is_connected():
                self.logger.warning("MT5 not connected, attempting to connect")
                if not self.mt5_connector.connect():
                    self.logger.error("Failed to connect to MT5")
                    return {}

            # Prepare order parameters
            order_params = {
                'action': direction.value,
                'symbol': symbol,
                'volume': lot_size,
                'type': order_type.value,
                'price': price,
                'sl': stop_loss,
                'tp': take_profit,
                'magic': self.trade_params['magic_number'],
                'comment': self.trade_params['comment'],
                'type_time': 'GTC',  # Good Till Cancelled
                'type_filling': 'IOC'  # Immediate or Cancel
            }

            # Add slippage for market orders
            if order_type == OrderType.MARKET:
                order_params['deviation'] = self.trade_params['slippage_pips']

            # Send order to MT5
            retry_count = 0
            result = {}

            # Retry logic for order execution
            while retry_count < self.trade_params['retry_attempts']:
                result = self.mt5_connector.send_order(order_params)

                if result and result.get('retcode') == 10009:  # Success code
                    break

                retry_count += 1
                self.logger.warning(f"Order failed, retrying {retry_count}/{self.trade_params['retry_attempts']}")
                time.sleep(self.trade_params['retry_delay_seconds'])

            # Check if order was successful
            if not result or result.get('retcode') != 10009:
                error_code = result.get('retcode', 'unknown')
                error_message = result.get('message', 'No error message')
                self.logger.error(f"Order failed with code {error_code}: {error_message}")
                return {}

            return result

        except Exception as e:
            self.logger.error(f"Error sending order: {e}")
            return {}

    def manage_open_positions(self, current_market_data: Dict[str, Any],
                              new_predictions: Optional[Dict[str, Any]] = None) -> None:
        """Manage existing positions based on new market data and predictions.

        Args:
            current_market_data: Dictionary with current market conditions
            new_predictions: Optional new model predictions
        """
        try:
            if not self.active_trades:
                return

            self.logger.info(f"Managing {len(self.active_trades)} open positions")

            # Get current price from market data
            current_price = current_market_data.get('price')
            if not current_price:
                self.logger.error("No current price provided in market data")
                return

            # Get current positions from MT5 to ensure our records match reality
            mt5_positions = self.mt5_connector.get_open_positions(
                self.trade_params['magic_number']
            )

            # Create mapping of position tickets
            position_map = {str(p['ticket']): p for p in mt5_positions} if mt5_positions else {}

            # Process each active trade
            for trade_id, trade in list(self.active_trades.items()):
                # Check if trade still exists in MT5
                if str(trade_id) not in position_map:
                    # Trade is closed, remove from active trades
                    self.logger.info(f"Trade {trade_id} is closed, removing from active trades")
                    self.active_trades.pop(trade_id, None)
                    continue

                # Get original entry data
                entry_price = trade['entry_price']
                stop_loss = trade['stop_loss']
                take_profit = trade['take_profit']
                direction = trade['direction']
                position_size = trade['position_size']

                # Check if we need to update the position
                need_update = False
                new_stop_loss = stop_loss
                new_take_profit = take_profit
                close_partial = False

                # Check partial profit level
                if (self.trade_params['partial_close_enabled'] and
                        not trade['partial_close_done'] and
                        trade['partial_profit_level'] > 0):

                    if direction == OrderDirection.BUY.value:
                        # For buy positions, check if price reached partial profit level
                        if current_price >= trade['partial_profit_level']:
                            close_partial = True
                    else:
                        # For sell positions, check if price reached partial profit level
                        if current_price <= trade['partial_profit_level']:
                            close_partial = True

                    if close_partial:
                        self.logger.info(f"Closing partial position for trade {trade_id} at {current_price}")
                        # Close partial position
                        partial_size = position_size * self.risk_params['partial_take_profit_pct']
                        self._close_partial_position(trade_id, partial_size)

                        # Update trade record
                        trade['partial_close_done'] = True
                        trade['position_size'] -= partial_size

                # Check breakeven level
                if (self.trade_params['breakeven_enabled'] and
                        not trade['moved_to_breakeven'] and
                        trade['breakeven_level'] > 0):

                    if direction == OrderDirection.BUY.value:
                        # For buy positions, move to breakeven when price rises above breakeven level
                        if current_price >= trade['breakeven_level']:
                            new_stop_loss = entry_price
                            need_update = True
                            trade['moved_to_breakeven'] = True
                    else:
                        # For sell positions, move to breakeven when price falls below breakeven level
                        if current_price <= trade['breakeven_level']:
                            new_stop_loss = entry_price
                            need_update = True
                            trade['moved_to_breakeven'] = True

                    if trade['moved_to_breakeven']:
                        self.logger.info(f"Moving trade {trade_id} to breakeven, new SL: {new_stop_loss}")

                # Check trailing stop conditions
                if (self.trade_params['trailing_stop_enabled'] and
                        trade['moved_to_breakeven']):

                    # Calculate R multiple achieved so far
                    risk = abs(entry_price - stop_loss)

                    if direction == OrderDirection.BUY.value:
                        # For buy positions
                        current_profit = current_price - entry_price
                        r_multiple_achieved = current_profit / risk if risk > 0 else 0

                        # If we've reached trailing activation level, move stop up
                        if r_multiple_achieved >= self.trade_params['trailing_stop_activation']:
                            # Calculate new stop loss based on trailing distance
                            trailing_distance = risk * self.trade_params['trailing_stop_distance']
                            potential_new_stop = current_price - trailing_distance

                            # Only move stop up, never down
                            if potential_new_stop > new_stop_loss:
                                new_stop_loss = potential_new_stop
                                need_update = True
                                self.logger.info(f"Updating trailing stop for trade {trade_id} to {new_stop_loss}")
                    else:
                        # For sell positions
                        current_profit = entry_price - current_price
                        r_multiple_achieved = current_profit / risk if risk > 0 else 0

                        # If we've reached trailing activation level, move stop down
                        if r_multiple_achieved >= self.trade_params['trailing_stop_activation']:
                            # Calculate new stop loss based on trailing distance
                            trailing_distance = risk * self.trade_params['trailing_stop_distance']
                            potential_new_stop = current_price + trailing_distance

                            # Only move stop down, never up
                            if potential_new_stop < new_stop_loss:
                                new_stop_loss = potential_new_stop
                                need_update = True
                                self.logger.info(f"Updating trailing stop for trade {trade_id} to {new_stop_loss}")

                # Update the trade if needed
                if need_update:
                    self._modify_position(trade_id, new_stop_loss, new_take_profit)
                    trade['stop_loss'] = new_stop_loss
                    trade['take_profit'] = new_take_profit

            # Check for new predictions that might warrant closing positions
            if new_predictions:
                self._update_based_on_predictions(new_predictions, current_market_data)

        except Exception as e:
            self.logger.error(f"Error managing open positions: {e}")

    def _close_partial_position(self, trade_id: int, partial_size: float) -> bool:
        """Close a partial position.

        Args:
            trade_id: Trade identifier
            partial_size: Size to close in lots

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the trade
            trade = self.active_trades.get(trade_id)
            if not trade:
                self.logger.error(f"Trade {trade_id} not found for partial close")
                return False

            # Prepare close parameters
            close_params = {
                'ticket': trade_id,
                'volume': partial_size
            }

            # Send close request to MT5
            result = self.mt5_connector.close_position(close_params)

            # Check if close was successful
            if not result or result.get('retcode') != 10009:
                error_code = result.get('retcode', 'unknown')
                error_message = result.get('message', 'No error message')
                self.logger.error(f"Partial close failed with code {error_code}: {error_message}")
                return False

            self.logger.info(f"Partial position {trade_id} closed successfully: {partial_size} lots")
            return True

        except Exception as e:
            self.logger.error(f"Error closing partial position: {e}")
            return False

    def _modify_position(self, trade_id: int, stop_loss: float, take_profit: float) -> bool:
        """Modify an existing position's stop loss and take profit.

        Args:
            trade_id: Trade identifier
            stop_loss: New stop loss price
            take_profit: New take profit price

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare modification parameters
            modify_params = {
                'ticket': trade_id,
                'sl': stop_loss,
                'tp': take_profit
            }

            # Send modification request to MT5
            result = self.mt5_connector.modify_position(modify_params)

            # Check if modification was successful
            if not result or result.get('retcode') != 10009:
                error_code = result.get('retcode', 'unknown')
                error_message = result.get('message', 'No error message')
                self.logger.error(f"Position modification failed with code {error_code}: {error_message}")
                return False

            self.logger.info(f"Position {trade_id} modified successfully: SL={stop_loss}, TP={take_profit}")
            return True

        except Exception as e:
            self.logger.error(f"Error modifying position: {e}")
            return False

    def _update_based_on_predictions(self, predictions: Dict[str, Any],
                                     current_market_data: Dict[str, Any]) -> None:
        """Update open positions based on new predictions.

        Args:
            predictions: New model predictions
            current_market_data: Current market data
        """
        try:
            # Check if predictions strongly contradict current positions
            for trade_id, trade in list(self.active_trades.items()):
                direction = trade['direction']
                symbol = trade['symbol']

                # Get prediction for this symbol
                if 'direction' not in predictions:
                    continue

                direction_pred = predictions['direction']
                # Convert to float if it's a numpy array
                if hasattr(direction_pred, 'item'):
                    direction_pred = direction_pred.item()

                # Check if prediction strongly contradicts position
                if (direction == OrderDirection.BUY.value and direction_pred < 0.3) or \
                        (direction == OrderDirection.SELL.value and direction_pred > 0.7):

                    self.logger.info(f"Prediction ({direction_pred:.2f}) contradicts position {trade_id} "
                                     f"direction ({direction}), closing position")

                    # Close the full position
                    close_params = {
                        'ticket': trade_id,
                        'volume': trade['position_size']
                    }

                    result = self.mt5_connector.close_position(close_params)

                    if result and result.get('retcode') == 10009:
                        self.logger.info(f"Position {trade_id} closed due to contradicting prediction")
                        self.active_trades.pop(trade_id, None)
                    else:
                        error_code = result.get('retcode', 'unknown')
                        error_message = result.get('message', 'No error message')
                        self.logger.error(f"Failed to close position {trade_id}: {error_code} - {error_message}")

        except Exception as e:
            self.logger.error(f"Error updating based on predictions: {e}")

    def get_trade_statistics(self) -> Dict[str, Any]:
        """Get statistics on all active trades.

        Returns:
            Dictionary with trade statistics
        """
        try:
            stats = {
                'total_trades': len(self.active_trades),
                'buy_positions': 0,
                'sell_positions': 0,
                'total_position_size': 0,
                'unrealized_profit': 0
            }

            # Get current positions from MT5
            mt5_positions = self.mt5_connector.get_open_positions(
                self.trade_params['magic_number']
            )

            # Create mapping of position tickets
            position_map = {str(p['ticket']): p for p in mt5_positions} if mt5_positions else {}

            # Calculate statistics
            for trade_id, trade in self.active_trades.items():
                direction = trade['direction']
                position_size = trade['position_size']

                if direction == OrderDirection.BUY.value:
                    stats['buy_positions'] += 1
                else:
                    stats['sell_positions'] += 1

                stats['total_position_size'] += position_size

                # Get unrealized profit if position exists in MT5
                if str(trade_id) in position_map:
                    stats['unrealized_profit'] += position_map[str(trade_id)].get('profit', 0)

            return stats

        except Exception as e:
            self.logger.error(f"Error getting trade statistics: {e}")
            return {
                'total_trades': 0,
                'buy_positions': 0,
                'sell_positions': 0,
                'total_position_size': 0,
                'unrealized_profit': 0,
                'error': str(e)
            }