from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Strategies.SignalGenerator import SignalGenerator
from Strategies.RiskManager import RiskManager
from Strategies.TradeExecutor import TradeExecutor
from typing import Optional, Dict, Any
import datetime
import threading
import time
import random


class TradingSession:
    """Manages an active trading session."""

    def __init__(self, config, logger, model, signal_generator, risk_manager, trade_executor, mt5_connector=None):
        self.config = config
        self.logger = logger
        self.model = model
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.trade_executor = trade_executor
        self.mt5_connector = mt5_connector

        # Trading session state
        self.trading_thread = None
        self.stop_event = threading.Event()
        self.is_running_flag = False

        # Trading statistics
        self.stats = {
            'start_time': None,
            'symbol': None,
            'timeframe': None,
            'update_interval': None,
            'open_positions': 0,
            'completed_trades': 0,
            'current_pl': 0,
            'duration': None,
            'total_trades': 0,
            'win_rate': 0,
            'net_pl': 0
        }

        # Recent signals and trades
        self.recent_signals = []
        self.active_trades = {}
        self.completed_trades = []

    def start(self, pair: str, timeframe: str, update_interval: int = 15) -> bool:
        """Start a trading session.

        Args:
            pair: Currency pair to trade
            timeframe: Timeframe to use
            update_interval: Update interval in minutes

        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self.is_running_flag:
                self.logger.warning("Trading session already running")
                return False

            self.logger.info(f"Starting trading session for {pair} {timeframe}")

            # Connect to MT5 if available
            if self.mt5_connector and not self.mt5_connector.is_connected():
                if not self.mt5_connector.connect():
                    self.logger.error("Failed to connect to MT5")
                    return False

            self.stop_event.clear()
            self.is_running_flag = True

            # Initialize statistics
            self.stats['start_time'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.stats['symbol'] = pair
            self.stats['timeframe'] = timeframe
            self.stats['update_interval'] = update_interval

            # Start background thread
            self.trading_thread = threading.Thread(
                target=self._trading_loop,
                args=(pair, timeframe, update_interval),
                daemon=True
            )
            self.trading_thread.start()

            self.logger.info(f"Trading session started for {pair} {timeframe}")
            return True

        except Exception as e:
            self.logger.error(f"Error starting trading session: {e}")
            self.is_running_flag = False
            return False

    def stop(self) -> bool:
        """Stop the trading session.

        Returns:
            True if stopped successfully, False otherwise
        """
        try:
            if not self.is_running_flag:
                self.logger.warning("Trading session not running")
                return False

            self.logger.info("Stopping trading session")
            self.stop_event.set()

            # Wait for trading thread to complete
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=30)

            self.is_running_flag = False

            # Disconnect from MT5 if connected
            if self.mt5_connector and self.mt5_connector.is_connected():
                self.mt5_connector.disconnect()

            # Update final statistics
            end_time = datetime.datetime.now()
            start_time = datetime.datetime.strptime(self.stats['start_time'], "%Y-%m-%d %H:%M:%S")
            duration_seconds = (end_time - start_time).total_seconds()
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            self.stats['duration'] = f"{hours}h {minutes}m"

            # Calculate final statistics
            self.stats['total_trades'] = len(self.completed_trades)
            if self.stats['total_trades'] > 0:
                winning_trades = sum(1 for trade in self.completed_trades if trade.get('profit_loss', 0) > 0)
                self.stats['win_rate'] = winning_trades / self.stats['total_trades']
                self.stats['net_pl'] = sum(trade.get('profit_loss', 0) for trade in self.completed_trades)

            self.logger.info("Trading session stopped")
            return True

        except Exception as e:
            self.logger.error(f"Error stopping trading session: {e}")
            return False

    def is_running(self) -> bool:
        """Check if the trading session is running."""
        return self.is_running_flag

    def get_statistics(self) -> Dict[str, Any]:
        """Get current trading session statistics."""
        # Update some real-time stats
        if self.is_running_flag:
            # Calculate duration so far
            start_time = datetime.datetime.strptime(self.stats['start_time'], "%Y-%m-%d %H:%M:%S")
            current_time = datetime.datetime.now()
            duration_seconds = (current_time - start_time).total_seconds()
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            self.stats['duration'] = f"{hours}h {minutes}m"

        return self.stats

    def get_recent_signals(self) -> list:
        """Get recent trading signals."""
        return self.recent_signals

    def join(self, timeout: Optional[float] = None) -> None:
        """Wait for the trading session to complete.

        Args:
            timeout: Optional timeout in seconds
        """
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=timeout)

    def _trading_loop(self, pair: str, timeframe: str, update_interval: int) -> None:
        """Main trading loop that runs in a background thread.

        Args:
            pair: Currency pair to trade
            timeframe: Timeframe to use
            update_interval: Update interval in minutes
        """
        try:
            self.logger.info(f"Trading loop started for {pair} {timeframe}")

            # Main trading loop
            while not self.stop_event.is_set():
                try:
                    # 1. Fetch latest market data
                    market_data = self._fetch_market_data(pair, timeframe)

                    # 2. Generate model predictions
                    predictions = self._generate_predictions(market_data)

                    # 3. Generate trading signals
                    signals = self._generate_signals(predictions, market_data)

                    # 4. Execute trades based on signals
                    self._execute_trades(signals, market_data)

                    # 5. Update position management
                    self._manage_positions(market_data, predictions)

                    # 6. Update statistics
                    self._update_statistics()

                    # Log activity
                    self.logger.info(f"Trading loop iteration completed for {pair}")

                except Exception as e:
                    self.logger.error(f"Error in trading loop: {e}")

                # Wait for next update interval or until stopped
                self.stop_event.wait(update_interval * 60)

            self.logger.info("Trading loop stopped")

        except Exception as e:
            self.logger.error(f"Fatal error in trading loop: {e}")
            self.is_running_flag = False

    def _fetch_market_data(self, pair: str, timeframe: str) -> Dict[str, Any]:
        """Fetch latest market data.

        In a real implementation, this would fetch data from MT5.
        For demo purposes, we generate random data if MT5 is not available.

        Args:
            pair: Currency pair
            timeframe: Timeframe

        Returns:
            Dictionary with market data
        """
        # Try to get data from MT5 if available
        if self.mt5_connector and self.mt5_connector.is_connected():
            try:
                market_data = self.mt5_connector.get_market_data(pair, timeframe, 100)
                if market_data:
                    return market_data
            except Exception as e:
                self.logger.error(f"Error fetching market data from MT5: {e}")

        # Fallback to simulated data
        last_price = 1800 + random.uniform(-10, 10)  # Simulating XAUUSD price

        return {
            'symbol': pair,
            'timeframe': timeframe,
            'time': datetime.datetime.now(),
            'open': last_price - random.uniform(0, 2),
            'high': last_price + random.uniform(0, 3),
            'low': last_price - random.uniform(0, 3),
            'close': last_price,
            'volume': random.randint(100, 1000),
            'price': last_price,
            'atr': random.uniform(1, 5),
            'account_balance': 10000  # Default account balance for risk calculations
        }

    def _generate_predictions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model predictions.

        In a real implementation, this would prepare data and run the model.

        Args:
            market_data: Current market data

        Returns:
            Dictionary with predictions
        """
        try:
            # In a real implementation, this would prepare sequences from market data
            # and run the model.prediction method

            # For now, if we have a model, we'll simulate predictions
            if self.model:
                # Create a dummy sequence for prediction
                # This is just a placeholder - in a real implementation you would
                # prepare proper sequences from market data
                import numpy as np
                dummy_sequence = np.random.random((1, 24, 10))  # 1 sample, 24 timesteps, 10 features

                # Try to get real predictions from model
                try:
                    predictions = self.model.predict(dummy_sequence)
                    return predictions
                except Exception as e:
                    self.logger.error(f"Error generating model predictions: {e}")

            # Fallback to random predictions
            return {
                'direction': random.uniform(0, 1),  # Probability of up movement
                'magnitude': random.uniform(0.1, 2.0),  # Expected price movement in %
                'volatility': random.uniform(0.1, 1.0)  # Expected volatility
            }

        except Exception as e:
            self.logger.error(f"Error generating predictions: {e}")
            # Return default predictions
            return {
                'direction': 0.5,  # Neutral
                'magnitude': 0.5,
                'volatility': 0.5
            }

    def _generate_signals(self, predictions: Dict[str, Any], market_data: Dict[str, Any]) -> list:
        """Generate trading signals from predictions.

        Args:
            predictions: Model predictions
            market_data: Current market data

        Returns:
            List of signal dictionaries
        """
        try:
            # Use the signal generator to create trading signals
            signals = self.signal_generator.generate_signals(
                predictions,
                confidence=1.0,  # For demo purposes
                current_market_data=market_data
            )

            # Filter signals to remove weak ones
            strong_signals = self.signal_generator.filter_signals(signals, min_strength=0.6)

            # Store recent signals (keep only last 10)
            if strong_signals:
                # Add timestamp to signals
                for signal in strong_signals:
                    signal['timestamp'] = datetime.datetime.now().strftime("%H:%M:%S")

                self.recent_signals = strong_signals + self.recent_signals
                if len(self.recent_signals) > 10:
                    self.recent_signals = self.recent_signals[:10]

            return strong_signals

        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            return []

    def _execute_trades(self, signals: list, market_data: Dict[str, Any]) -> None:
        """Execute trades based on signals.

        Args:
            signals: List of trading signals
            market_data: Current market data
        """
        try:
            # Check if we have signals and if we can open more positions
            max_positions = self.config.get_nested('GoldTradingSettings', 'RiskManagement', 'max_open_trades', 5)
            if not signals or len(self.active_trades) >= max_positions:
                return

            for signal in signals:
                # Let the trade executor handle the trade
                if self.trade_executor:
                    trade_result = self.trade_executor.execute_trade(signal, market_data)

                    if trade_result:
                        # Add to active trades
                        trade_id = trade_result.get('id', f"T{int(time.time())}")
                        self.active_trades[trade_id] = trade_result

                        # Update statistics
                        self.stats['open_positions'] += 1

                        self.logger.info(
                            f"Executed trade: {trade_id} {trade_result.get('direction')} at {trade_result.get('entry_price')}")
                else:
                    # Simulate trade execution without the executor
                    trade_id = f"T{int(time.time())}"

                    # Create a simple trade record
                    trade = {
                        'id': trade_id,
                        'symbol': market_data['symbol'],
                        'direction': signal['type'].name,
                        'entry_price': market_data['price'],
                        'position_size': 0.1,  # Fixed size for demo
                        'stop_loss': market_data['price'] * 0.99 if 'BUY' in signal['type'].name else market_data[
                                                                                                          'price'] * 1.01,
                        'take_profit': market_data['price'] * 1.02 if 'BUY' in signal['type'].name else market_data[
                                                                                                            'price'] * 0.98,
                        'signal_type': signal['type'].name,
                        'open_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'signal_strength': signal.get('signal_strength', 0)
                    }

                    # Add to active trades
                    self.active_trades[trade_id] = trade

                    # Update statistics
                    self.stats['open_positions'] += 1

                    self.logger.info(f"Simulated trade: {trade_id} {trade['direction']} at {trade['entry_price']}")

        except Exception as e:
            self.logger.error(f"Error executing trades: {e}")

    def _manage_positions(self, market_data: Dict[str, Any], predictions: Dict[str, Any]) -> None:
        """Manage open positions.

        Args:
            market_data: Current market data
            predictions: Latest model predictions
        """
        try:
            # If we have an executor, let it manage positions
            if self.trade_executor:
                self.trade_executor.manage_open_positions(market_data, predictions)

                # Update active trades list from executor
                # This would be implemented properly in a real system
                pass

            # Otherwise, simulate position management
            else:
                for trade_id, trade in list(self.active_trades.items()):
                    # Simulate trade outcome
                    # In a real implementation, this would check for stop loss/take profit hits

                    # For demo, randomly close some positions (20% chance each update)
                    if random.random() < 0.2:
                        # Simulate trade result
                        close_price = market_data['price']

                        # Calculate profit/loss based on direction
                        profit_loss = 0
                        if 'BUY' in trade['direction']:
                            profit_loss = (close_price - trade['entry_price']) * 1000 * trade['position_size']
                        else:
                            profit_loss = (trade['entry_price'] - close_price) * 1000 * trade['position_size']

                        # Create completed trade record
                        completed_trade = {
                            **trade,
                            'exit_price': close_price,
                            'exit_time': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'profit_loss': profit_loss,
                            'win': profit_loss > 0
                        }

                        # Remove from active trades
                        del self.active_trades[trade_id]

                        # Add to completed trades
                        self.completed_trades.append(completed_trade)

                        # Update statistics
                        self.stats['open_positions'] -= 1
                        self.stats['completed_trades'] += 1
                        self.stats['current_pl'] += profit_loss

                        self.logger.info(f"Closed trade: {trade_id} with P/L: ${profit_loss:.2f}")

            # Get account info if available
            if self.mt5_connector and self.mt5_connector.is_connected():
                try:
                    account_info = self.mt5_connector.get_account_info()
                    if account_info:
                        # Update stats with real account info
                        self.stats['current_pl'] = account_info.get('profit', 0)
                except Exception as e:
                    self.logger.error(f"Error getting account info: {e}")

        except Exception as e:
            self.logger.error(f"Error managing positions: {e}")

    def _update_statistics(self) -> None:
        """Update trading statistics."""
        try:
            # Update current P/L from completed trades
            current_pl = sum(trade.get('profit_loss', 0) for trade in self.completed_trades)

            # Update stats
            self.stats['current_pl'] = current_pl
            self.stats['open_positions'] = len(self.active_trades)
            self.stats['completed_trades'] = len(self.completed_trades)

            # Calculate win rate
            if self.completed_trades:
                winning_trades = sum(1 for trade in self.completed_trades if trade.get('win', False))
                self.stats['win_rate'] = winning_trades / len(self.completed_trades)

            # Calculate net P/L
            self.stats['net_pl'] = current_pl

        except Exception as e:
            self.logger.error(f"Error updating statistics: {e}")


class StrategyFactory:
    """Factory for creating trading strategy components."""

    def __init__(self, config: Config, logger: Logger, model_factory):
        self.config = config
        self.logger = logger
        self.model_factory = model_factory
        self.active_trading_session = None
        self.mt5_connector = None

        # Initialize MT5 connector
        # self._initialize_mt5_connector()

    # def _initialize_mt5_connector(self) -> None:
    #     """Initialize MT5 connector."""
    #     try:
    #         self.mt5_connector = MT5Connector(self.config, self.logger)
    #         # Don't connect immediately, we'll connect when needed
    #     except Exception as e:
    #         self.logger.error(f"Error initializing MT5 connector: {e}")

    def create_signal_generator(self) -> SignalGenerator:
        """Create a signal generator instance."""
        return SignalGenerator(self.config, self.logger)

    def create_risk_manager(self, account_info: Optional[Dict[str, Any]] = None) -> RiskManager:
        """Create a risk manager instance."""
        return RiskManager(self.config, self.logger, account_info)

    def create_trade_executor(self, risk_manager: RiskManager) -> TradeExecutor:
        """Create a trade executor instance."""
        return TradeExecutor(self.config, self.logger, risk_manager, self.mt5_connector)

    def create_trading_session(self, model, signal_generator, risk_manager, trade_executor) -> TradingSession:
        """Create a trading session."""
        session = TradingSession(
            self.config,
            self.logger,
            model,
            signal_generator,
            risk_manager,
            trade_executor,
            self.mt5_connector
        )
        self.active_trading_session = session
        return session

    def get_active_trading_session(self) -> Optional[TradingSession]:
        """Get the active trading session if one exists."""
        return self.active_trading_session

    # def get_mt5_connector(self) -> Optional[MT5Connector]:
    #     """Get the MT5 connector."""
    #     return self.mt5_connector

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information from MT5."""
        if self.mt5_connector and self.mt5_connector.is_connected():
            return self.mt5_connector.get_account_info()
        return {}

    def test_mt5_connection(self) -> bool:
        """Test connection to MT5.

        Returns:
            True if connection successful, False otherwise
        """
        if not self.mt5_connector:
            return False

        if not self.mt5_connector.is_connected():
            return self.mt5_connector.connect()

        return True