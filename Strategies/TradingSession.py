import time
import threading
from datetime import datetime
from typing import Optional, Dict, Any, List
import random  # Just for demo purposes

from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Models.LSTMModel import LSTMModel
from Strategies.SignalGenerator import SignalGenerator, SignalType
from Strategies.RiskManager import RiskManager
from Strategies.TradeExecutor import TradeExecutor


class TradingSession:
    """Manages a live trading session with a background thread for market monitoring."""

    def __init__(
            self,
            config: Config,
            logger: Logger,
            model: LSTMModel,
            signal_generator: SignalGenerator,
            risk_manager: RiskManager,
            trade_executor: TradeExecutor
    ):
        self.config = config
        self.logger = logger
        self.model = model
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.trade_executor = trade_executor

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
        """Start the trading session.

        Args:
            pair: Currency pair to trade
            timeframe: Timeframe to use
            update_interval: Data update interval in minutes

        Returns:
            True if started successfully, False otherwise
        """
        try:
            if self.is_running_flag:
                self.logger.warning("Trading session already running")
                return False

            self.logger.info(f"Starting trading session for {pair} {timeframe}")
            self.stop_event.clear()
            self.is_running_flag = True

            # Initialize statistics
            self.stats['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

            # Update final statistics
            end_time = datetime.now()
            start_time = datetime.strptime(self.stats['start_time'], "%Y-%m-%d %H:%M:%S")
            duration_seconds = (end_time - start_time).total_seconds()
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            self.stats['duration'] = f"{hours}h {minutes}m"

            # In a real implementation, these would be calculated from actual trade data
            self.stats['total_trades'] = len(self.completed_trades)
            if self.stats['total_trades'] > 0:
                winning_trades = sum(1 for trade in self.completed_trades if trade.get('profit_loss', 0) > 0)
                self.stats['win_rate'] = winning_trades / self.stats['total_trades']
                self.stats['net_pl'] = sum(trade.get('profit_loss', 0) for trade in self.completed_trades)

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
            start_time = datetime.strptime(self.stats['start_time'], "%Y-%m-%d %H:%M:%S")
            current_time = datetime.now()
            duration_seconds = (current_time - start_time).total_seconds()
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            self.stats['duration'] = f"{hours}h {minutes}m"

        return self.stats

    def get_recent_signals(self) -> List[Dict[str, Any]]:
        """Get recent trading signals."""
        return self.recent_signals

    def get_active_trades(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active trades."""
        return self.active_trades

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
            update_interval: Data update interval in minutes
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

        In a real implementation, this would fetch data from MetaTrader or another source.
        For demo purposes, we generate random data.

        Args:
            pair: Currency pair
            timeframe: Timeframe

        Returns:
            Dictionary with market data
        """
        # For demo purposes, generate random price data
        last_price = 1800 + random.uniform(-10, 10)  # Simulating XAUUSD price

        return {
            'symbol': pair,
            'timeframe': timeframe,
            'time': datetime.now(),
            'open': last_price - random.uniform(0, 2),
            'high': last_price + random.uniform(0, 3),
            'low': last_price - random.uniform(0, 3),
            'close': last_price,
            'volume': random.randint(100, 1000),
            'price': last_price,
            'atr': random.uniform(1, 5)
        }

    def _generate_predictions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model predictions.

        In a real implementation, this would prepare data and run the model.
        For demo purposes, we generate random predictions.

        Args:
            market_data: Current market data

        Returns:
            Dictionary with predictions
        """
        # For demo purposes, generate random predictions
        return {
            'direction': random.uniform(0, 1),  # Probability of price going up
            'magnitude': random.uniform(0.1, 2.0),  # Expected price movement in %
            'volatility': random.uniform(0.1, 1.0)  # Expected volatility
        }

    def _generate_signals(self, predictions: Dict[str, Any], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate trading signals from predictions.

        Args:
            predictions: Model predictions
            market_data: Current market data

        Returns:
            List of signal dictionaries
        """
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
                signal['timestamp'] = datetime.now().strftime("%H:%M:%S")

            self.recent_signals = strong_signals + self.recent_signals
            if len(self.recent_signals) > 10:
                self.recent_signals = self.recent_signals[:10]

        return strong_signals

    def _execute_trades(self, signals: List[Dict[str, Any]], market_data: Dict[str, Any]) -> None:
        """Execute trades based on signals.

        Args:
            signals: List of trading signals
            market_data: Current market data
        """
        # Check if we have signals and if we can open more positions
        if not signals or len(self.active_trades) >= 5:  # Limit to 5 open positions
            return

        # In a real implementation, this would execute trades via MT5
        # For demo purposes, we'll simulate trade execution
        for signal in signals:
            # Create a trade object
            trade_id = f"T{int(time.time())}"

            # Simulate trade execution
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
                'open_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'signal_strength': signal.get('signal_strength', 0)
            }

            # Add to active trades
            self.active_trades[trade_id] = trade

            # Update statistics
            self.stats['open_positions'] += 1

            self.logger.info(f"Executed trade: {trade_id} {trade['direction']} at {trade['entry_price']}")

    def _manage_positions(self, market_data: Dict[str, Any], predictions: Dict[str, Any]) -> None:
        """Manage open positions.

        Args:
            market_data: Current market data
            predictions: Latest model predictions
        """
        # In a real implementation, this would check for stop loss/take profit hits
        # For demo purposes, we'll randomly close some positions

        for trade_id, trade in list(self.active_trades.items()):
            # Randomly decide if the trade closed
            if random.random() < 0.2:  # 20% chance of closing each update
                # Simulate trade result
                close_price = market_data['price']
                profit_loss = 0

                if trade['direction'] == 'STRONG_BUY' or trade['direction'] == 'MODERATE_BUY':
                    profit_loss = (close_price - trade['entry_price']) * 1000 * trade['position_size']
                else:
                    profit_loss = (trade['entry_price'] - close_price) * 1000 * trade['position_size']

                # Create completed trade
                completed_trade = {
                    **trade,
                    'exit_price': close_price,
                    'exit_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

    def _update_statistics(self) -> None:
        """Update trading statistics."""
        # In a real implementation, this would calculate actual statistics
        # For demo purposes, we'll just update a few key metrics

        # Update current P/L
        current_pl = 0
        for trade in self.completed_trades:
            current_pl += trade.get('profit_loss', 0)

        # Add unrealized P/L (not implemented in this demo)

        self.stats['current_pl'] = current_pl
        self.stats['open_positions'] = len(self.active_trades)
        self.stats['completed_trades'] = len(self.completed_trades)

        # Calculate win rate
        if self.completed_trades:
            winning_trades = sum(1 for trade in self.completed_trades if trade.get('win', False))
            self.stats['win_rate'] = winning_trades / len(self.completed_trades)

        # Calculate net P/L
        self.stats['net_pl'] = current_pl