import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import os

from Models.LSTMModel import LSTMModel
from Training.DataPreprocessor import DataPreprocessor
from Strategies.SignalGenerator import SignalGenerator, SignalType
from Strategies.RiskManager import RiskManager


class BacktestEngine:
    """Backtests trading strategies on historical data."""

    def __init__(self, config, logger, data_storage, model, signal_generator, risk_manager):
        """Initialize the backtest engine.

        Args:
            config: Application configuration
            logger: Logger instance
            data_storage: DataStorage instance for retrieving historical data
            model: Trained model for predictions
            signal_generator: SignalGenerator instance
            risk_manager: RiskManager instance
        """
        self.config = config
        self.logger = logger
        self.data_storage = data_storage
        self.model = model
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager

        # Initialize backtest results storage
        self.results = {
            'equity_curve': [],
            'trades': [],
            'metrics': {},
            'daily_returns': []
        }

        # Initialize backtest parameters from config
        self.params = self._load_backtest_parameters()

    def _load_backtest_parameters(self) -> Dict[str, Any]:
        """Load backtest parameters from configuration."""
        # Try to load from config
        trading_settings = self.config.get('GoldTradingSettings', {})
        backtest_config = trading_settings.get('Backtesting', {})

        # Default parameters
        default_params = {
            'initial_balance': 10000,  # Initial account balance
            'position_size_mode': 'risk',  # 'fixed' or 'risk'
            'fixed_position_size': 0.1,  # Fixed position size if using fixed mode
            'risk_per_trade': 0.01,  # Risk per trade if using risk mode
            'spread_pips': 1.5,  # Spread in pips
            'commission_per_lot': 7,  # Commission per lot (round turn)
            'slippage_pips': 0.5,  # Slippage in pips
            'use_sl_tp': True,  # Whether to use stop loss and take profit
            'enable_partial_close': True,  # Enable partial position closing
            'enable_breakeven': True,  # Enable moving to breakeven
            'max_open_trades': 5,  # Maximum number of simultaneous open trades
            'signal_threshold': 0.6,  # Minimum signal strength to open a trade
        }

        # Update with values from config if they exist
        params = {}
        for key, default_value in default_params.items():
            params[key] = backtest_config.get(key, default_value)

        self.logger.info(f"Backtest parameters loaded: {params}")
        return params

    def run_backtest(self, pair: str, timeframe: str, start_date: str, end_date: str) -> Dict[str, Any]:
        """Run backtest on historical data.

        Args:
            pair: Currency pair to backtest
            timeframe: Timeframe to use
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            Dictionary with backtest results
        """
        try:
            self.logger.info(f"Starting backtest for {pair} {timeframe} "
                             f"from {start_date} to {end_date}")

            # Reset results
            self.results = {
                'equity_curve': [],
                'trades': [],
                'metrics': {},
                'daily_returns': []
            }

            # Load historical data
            data = self._load_historical_data(pair, timeframe, start_date, end_date)
            if data.empty:
                self.logger.error("No data available for backtest")
                return self.results

            # Initialize account
            account = {
                'balance': self.params['initial_balance'],
                'equity': self.params['initial_balance'],
                'open_positions': {},
                'closed_trades': []
            }

            # Add initial equity point
            self.results['equity_curve'].append({
                'timestamp': data.iloc[0]['time'],
                'balance': account['balance'],
                'equity': account['equity']
            })

            # Create DataPreprocessor for data preparation
            data_preprocessor = DataPreprocessor(self.config, self.logger, self.data_storage)
            data_preprocessor.load_feature_importance()

            # Get sequence length from model
            sequence_length = self.model.sequence_length if hasattr(self.model, 'sequence_length') else 24

            # Process data for prediction
            processed_data = self._prepare_data_for_backtest(data, data_preprocessor, sequence_length)

            # Run the backtest
            for i in range(sequence_length, len(data)):
                # Current data point
                current_row = data.iloc[i]
                current_time = current_row['time']
                current_price = current_row['close']
                current_high = current_row['high']
                current_low = current_row['low']
                atr_value = current_row.get('atr', current_price * 0.01)  # Default to 1% if ATR not available

                # Create market data dict
                market_data = {
                    'symbol': pair,
                    'price': current_price,
                    'high': current_high,
                    'low': current_low,
                    'time': current_time,
                    'atr': atr_value,
                    'account_balance': account['balance']
                }

                # Get prediction for this data point
                X_sequence = processed_data['X'][i - sequence_length]
                prediction = self.model.predict(np.array([X_sequence]))

                # Generate signal
                signals = self.signal_generator.generate_signals(
                    predictions=prediction,
                    confidence=np.array([0.8]),  # Default confidence for backtest
                    current_market_data=market_data
                )

                # Filter signals based on threshold
                valid_signals = self.signal_generator.filter_signals(
                    signals, min_strength=self.params['signal_threshold']
                )

                # Update existing positions
                self._update_positions(account, current_row, market_data)

                # Process new signals if we have capacity
                if valid_signals and len(account['open_positions']) < self.params['max_open_trades']:
                    for signal in valid_signals:
                        if len(account['open_positions']) >= self.params['max_open_trades']:
                            break

                        # Open new position based on signal
                        self._open_position(account, signal, current_row, market_data)

                # Update equity curve at end of bar
                self.results['equity_curve'].append({
                    'timestamp': current_time,
                    'balance': account['balance'],
                    'equity': self._calculate_equity(account, current_price)
                })

                # Record daily return if this is end of day
                if i > 0 and (current_time.day != data.iloc[i - 1]['time'].day or i == len(data) - 1):
                    daily_return = {
                        'date': current_time.date().isoformat(),
                        'return': (self.results['equity_curve'][-1]['equity'] /
                                   self.results['equity_curve'][-2]['equity'] - 1)
                    }
                    self.results['daily_returns'].append(daily_return)

            # Close any remaining open positions at end of backtest
            self._close_all_positions(account, data.iloc[-1])

            # Calculate final metrics
            self.results['metrics'] = self.calculate_performance_metrics(
                account['closed_trades'], self.results['equity_curve'], self.results['daily_returns']
            )

            self.logger.info(f"Backtest completed for {pair} {timeframe}. "
                             f"Final balance: {account['balance']:.2f}, "
                             f"Total trades: {len(account['closed_trades'])}, "
                             f"Win rate: {self.results['metrics'].get('win_rate', 0):.2f}")

            return self.results

        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            raise

    def _load_historical_data(self, pair: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load historical data for backtest.

        Args:
            pair: Currency pair
            timeframe: Timeframe
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)

        Returns:
            DataFrame with historical data
        """
        try:
            # Get data from storage
            data_type = "training"  # Use training data set
            X, y = self.data_storage.load_processed_data(pair, timeframe, data_type)

            if X.empty:
                self.logger.error(f"No data found for {pair} {timeframe}")
                return pd.DataFrame()

            # Combine features and price data
            df = X.copy()

            # Add target variables if available
            if not y.empty:
                for col in y.columns:
                    df[col] = y[col]

            # Filter by date range if provided
            if start_date and end_date:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)

                df = df[(df['time'] >= start_dt) & (df['time'] <= end_dt)]

            # Ensure required columns are present
            required_cols = ['time', 'open', 'high', 'low', 'close']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Missing required columns in data. Required: {required_cols}, "
                                  f"Available: {list(df.columns)}")
                return pd.DataFrame()

            # Ensure time column is datetime
            df['time'] = pd.to_datetime(df['time'])

            # Sort by time
            df = df.sort_values('time')

            self.logger.info(f"Loaded {len(df)} data points for backtest from {df['time'].min()} to {df['time'].max()}")
            return df

        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()

    def _prepare_data_for_backtest(self, data: pd.DataFrame,
                                   data_preprocessor: DataPreprocessor,
                                   sequence_length: int) -> Dict[str, np.ndarray]:
        """Prepare data for backtest.

        Args:
            data: DataFrame with historical data
            data_preprocessor: DataPreprocessor instance
            sequence_length: Sequence length for LSTM

        Returns:
            Dictionary with processed data
        """
        try:
            # Create a copy to avoid modifying the original
            df = data.copy()

            # Get selected features
            selected_features = data_preprocessor.get_selected_features()
            available_features = [f for f in selected_features if f in df.columns]

            # Extract time and required columns
            time_col = df['time']
            price_cols = ['open', 'high', 'low', 'close']
            price_data = df[price_cols]

            # Extract features (excluding time and price columns)
            feature_df = df[available_features]

            # Scale features
            scaled_features = data_preprocessor.scale_features(feature_df)

            # Create sequences
            X_sequences, _ = data_preprocessor.create_sequences(scaled_features, sequence_length)

            processed_data = {
                'X': X_sequences,
                'times': time_col.values[sequence_length - 1:],
                'prices': price_data.values[sequence_length - 1:]
            }

            self.logger.info(f"Prepared {len(X_sequences)} sequences for backtest")
            return processed_data

        except Exception as e:
            self.logger.error(f"Error preparing data for backtest: {e}")
            raise

    def _open_position(self, account: Dict[str, Any], signal: Dict[str, Any],
                       bar_data: pd.Series, market_data: Dict[str, Any]) -> None:
        """Open a new position based on signal.

        Args:
            account: Account dictionary
            signal: Signal dictionary
            bar_data: Current bar data
            market_data: Market data dictionary
        """
        try:
            # Determine trade direction
            signal_type = signal['type']
            is_buy = signal_type.value.endswith('BUY')

            # Get current price with spread adjustment
            spread_adjustment = self.params['spread_pips'] / 10.0  # Convert pips to price

            if is_buy:
                entry_price = bar_data['close'] + spread_adjustment
                direction = 'BUY'
            else:
                entry_price = bar_data['close'] - spread_adjustment
                direction = 'SELL'

            # Calculate stop loss and take profit
            if is_buy:
                stop_loss = self.risk_manager.calculate_stop_loss(
                    entry_price, signal.get('expected_volatility', 0),
                    market_data.get('atr', entry_price * 0.01), signal_type
                )
                take_profit = self.risk_manager.calculate_take_profit(
                    entry_price, entry_price, stop_loss,
                    signal.get('expected_magnitude', 0), signal_type
                )
            else:
                stop_loss = self.risk_manager.calculate_stop_loss(
                    entry_price, signal.get('expected_volatility', 0),
                    market_data.get('atr', entry_price * 0.01), signal_type
                )
                take_profit = self.risk_manager.calculate_take_profit(
                    entry_price, entry_price, stop_loss,
                    signal.get('expected_magnitude', 0), signal_type
                )

            # Calculate stop loss distance in pips
            stop_loss_pips = abs(entry_price - stop_loss) * 10  # Convert price to pips

            # Calculate position size
            if self.params['position_size_mode'] == 'risk':
                # Risk-based position sizing
                risk_amount = account['balance'] * self.params['risk_per_trade']

                # Convert pips to dollars (for gold, 1 pip = $0.1 per 0.01 lot)
                pip_value = 10.0  # $10 per pip for 1 lot

                position_size = risk_amount / (stop_loss_pips * pip_value)
                position_size = max(min(position_size, 10.0), 0.01)  # Cap between 0.01 and 10 lots
                position_size = round(position_size, 2)  # Round to 2 decimal places
            else:
                # Fixed position sizing
                position_size = self.params['fixed_position_size']

            # Calculate breakeven and partial profit levels
            breakeven_level = self.risk_manager.calculate_breakeven_level(
                entry_price, stop_loss, signal_type
            )

            partial_profit_level = self.risk_manager.calculate_partial_profit_level(
                entry_price, stop_loss, take_profit, signal_type
            )

            # Generate trade ID
            trade_id = f"{len(account['closed_trades']) + len(account['open_positions']) + 1}"

            # Create position record
            position = {
                'id': trade_id,
                'symbol': market_data['symbol'],
                'direction': direction,
                'entry_price': entry_price,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'breakeven_level': breakeven_level,
                'partial_profit_level': partial_profit_level,
                'signal_type': signal_type.value,
                'entry_time': bar_data['time'],
                'entry_bar': len(account['closed_trades']) + len(account['open_positions']),
                'partial_close_done': False,
                'moved_to_breakeven': False,
                'signal_strength': signal['signal_strength'],
                'expected_magnitude': signal.get('expected_magnitude', 0),
                'commission': position_size * self.params['commission_per_lot']
            }

            # Add to open positions
            account['open_positions'][trade_id] = position

            # Deduct commission from balance
            account['balance'] -= position['commission']

            self.logger.debug(f"Opened {direction} position {trade_id} at {entry_price} "
                              f"with size {position_size} lots, SL: {stop_loss}, TP: {take_profit}")

        except Exception as e:
            self.logger.error(f"Error opening position: {e}")

    def _update_positions(self, account: Dict[str, Any], bar_data: pd.Series,
                          market_data: Dict[str, Any]) -> None:
        """Update open positions.

        Args:
            account: Account dictionary
            bar_data: Current bar data
            market_data: Market data dictionary
        """
        try:
            if not account['open_positions']:
                return

            current_price = bar_data['close']
            high = bar_data['high']
            low = bar_data['low']

            # Process each open position
            for trade_id, position in list(account['open_positions'].items()):
                # Check if position should be closed
                position_closed = False

                # Direction-specific logic
                if position['direction'] == 'BUY':
                    # Check stop loss hit
                    if low <= position['stop_loss']:
                        # Close at stop loss
                        self._close_position(account, trade_id, position['stop_loss'],
                                             bar_data['time'], "Stop Loss")
                        position_closed = True

                    # Check take profit hit
                    elif high >= position['take_profit']:
                        # Close at take profit
                        self._close_position(account, trade_id, position['take_profit'],
                                             bar_data['time'], "Take Profit")
                        position_closed = True

                    # Check partial profit
                    elif (self.params['enable_partial_close'] and
                          not position['partial_close_done'] and
                          high >= position['partial_profit_level']):
                        # Close half position at partial profit
                        partial_size = position['position_size'] * 0.5
                        self._close_partial_position(account, trade_id, partial_size,
                                                     position['partial_profit_level'],
                                                     bar_data['time'], "Partial Profit")
                        position['partial_close_done'] = True
                        position['position_size'] -= partial_size

                    # Check move to breakeven
                    if (self.params['enable_breakeven'] and
                            not position['moved_to_breakeven'] and
                            high >= position['breakeven_level']):
                        # Move stop loss to breakeven
                        position['stop_loss'] = position['entry_price']
                        position['moved_to_breakeven'] = True
                        self.logger.debug(f"Moved position {trade_id} to breakeven")

                else:  # SELL position
                    # Check stop loss hit
                    if high >= position['stop_loss']:
                        # Close at stop loss
                        self._close_position(account, trade_id, position['stop_loss'],
                                             bar_data['time'], "Stop Loss")
                        position_closed = True

                    # Check take profit hit
                    elif low <= position['take_profit']:
                        # Close at take profit
                        self._close_position(account, trade_id, position['take_profit'],
                                             bar_data['time'], "Take Profit")
                        position_closed = True

                    # Check partial profit
                    elif (self.params['enable_partial_close'] and
                          not position['partial_close_done'] and
                          low <= position['partial_profit_level']):
                        # Close half position at partial profit
                        partial_size = position['position_size'] * 0.5
                        self._close_partial_position(account, trade_id, partial_size,
                                                     position['partial_profit_level'],
                                                     bar_data['time'], "Partial Profit")
                        position['partial_close_done'] = True
                        position['position_size'] -= partial_size

                    # Check move to breakeven
                    if (self.params['enable_breakeven'] and
                            not position['moved_to_breakeven'] and
                            low <= position['breakeven_level']):
                        # Move stop loss to breakeven
                        position['stop_loss'] = position['entry_price']
                        position['moved_to_breakeven'] = True
                        self.logger.debug(f"Moved position {trade_id} to breakeven")


        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")

    def _close_position(self, account: Dict[str, Any], trade_id: str, exit_price: float,
                        exit_time: datetime, exit_reason: str) -> None:
        """Close a position.

        Args:
            account: Account dictionary
            trade_id: Trade ID
            exit_price: Exit price
            exit_time: Exit time
            exit_reason: Exit reason
        """
        try:
            # Get position
            position = account['open_positions'].get(trade_id)
            if not position:
                self.logger.warning(f"Position {trade_id} not found for closing")
                return

            # Calculate profit/loss in pips
            if position['direction'] == 'BUY':
                pips_change = (exit_price - position['entry_price']) * 10.0  # Convert to pips
            else:
                pips_change = (position['entry_price'] - exit_price) * 10.0  # Convert to pips

            # Calculate profit/loss in dollars
            pip_value = 10.0 * position['position_size']  # $10 per pip per lot
            profit_loss = pips_change * pip_value

            # Update account balance
            account['balance'] += profit_loss

            # Complete the trade record
            closed_trade = {
                **position,
                'exit_price': exit_price,
                'exit_time': exit_time,
                'exit_reason': exit_reason,
                'profit_loss': profit_loss,
                'pips_change': pips_change,
                'trade_duration': (exit_time - position['entry_time']).total_seconds() / 3600.0,  # hours
                'win': profit_loss > 0
            }

            # Remove from open positions
            account['open_positions'].pop(trade_id)

            # Add to closed trades
            account['closed_trades'].append(closed_trade)

            # Add to results trades list
            self.results['trades'].append(closed_trade)

            self.logger.debug(f"Closed position {trade_id} at {exit_price} with P/L: ${profit_loss:.2f} "
                              f"({pips_change:.1f} pips), reason: {exit_reason}")

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")

    def _close_partial_position(self, account: Dict[str, Any], trade_id: str,
                                size: float, exit_price: float,
                                exit_time: datetime, exit_reason: str) -> None:
        """Close part of a position.

        Args:
            account: Account dictionary
            trade_id: Trade ID
            size: Position size to close
            exit_price: Exit price
            exit_time: Exit time
            exit_reason: Exit reason
        """
        try:
            # Get position
            position = account['open_positions'].get(trade_id)
            if not position:
                self.logger.warning(f"Position {trade_id} not found for partial closing")
                return

            # Calculate profit/loss in pips
            if position['direction'] == 'BUY':
                pips_change = (exit_price - position['entry_price']) * 10.0  # Convert to pips
            else:
                pips_change = (position['entry_price'] - exit_price) * 10.0  # Convert to pips

            # Calculate profit/loss in dollars
            pip_value = 10.0 * size  # $10 per pip per lot
            profit_loss = pips_change * pip_value

            # Update account balance
            account['balance'] += profit_loss

            # Create partial trade record
            partial_trade = {
                **position,
                'position_size': size,  # Only the closed portion
                'exit_price': exit_price,
                'exit_time': exit_time,
                'exit_reason': exit_reason,
                'profit_loss': profit_loss,
                'pips_change': pips_change,
                'trade_duration': (exit_time - position['entry_time']).total_seconds() / 3600.0,  # hours
                'win': profit_loss > 0,
                'partial_close': True
            }

            # Add to closed trades
            account['closed_trades'].append(partial_trade)

            # Add to results trades list
            self.results['trades'].append(partial_trade)

            self.logger.debug(f"Partially closed position {trade_id} ({size} lots) at {exit_price} "
                              f"with P/L: ${profit_loss:.2f} ({pips_change:.1f} pips), reason: {exit_reason}")

        except Exception as e:
            self.logger.error(f"Error partially closing position: {e}")

    def _close_all_positions(self, account: Dict[str, Any], last_bar: pd.Series) -> None:
        """Close all open positions at the end of backtest.

        Args:
            account: Account dictionary
            last_bar: Last data bar
        """
        try:
            if not account['open_positions']:
                return

            self.logger.info(f"Closing {len(account['open_positions'])} open positions at end of backtest")

            # Get close price from last bar
            close_price = last_bar['close']
            exit_time = last_bar['time']

            # Close each position
            for trade_id, position in list(account['open_positions'].items()):
                self._close_position(account, trade_id, close_price, exit_time, "End of Backtest")

        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")

    def _calculate_equity(self, account: Dict[str, Any], current_price: float) -> float:
        """Calculate current equity including unrealized P/L.

        Args:
            account: Account dictionary
            current_price: Current price

        Returns:
            Current equity
        """
        equity = account['balance']

        # Add unrealized P/L of open positions
        for position in account['open_positions'].values():
            # Calculate unrealized P/L
            if position['direction'] == 'BUY':
                pips_change = (current_price - position['entry_price']) * 10.0  # Convert to pips
            else:
                pips_change = (position['entry_price'] - current_price) * 10.0  # Convert to pips

            # Calculate profit/loss in dollars
            pip_value = 10.0 * position['position_size']  # $10 per pip per lot
            unrealized_pl = pips_change * pip_value

            # Add to equity
            equity += unrealized_pl

        return equity

    def calculate_performance_metrics(self, trades: List[Dict[str, Any]],
                                      equity_curve: List[Dict[str, Any]],
                                      daily_returns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate performance metrics for the backtest.

        Args:
            trades: List of closed trades
            equity_curve: Equity curve
            daily_returns: Daily returns

        Returns:
            Dictionary with performance metrics
        """
        try:
            if not trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'avg_profit_per_trade': 0,
                    'max_drawdown': 0,
                    'sharpe_ratio': 0,
                    'sortino_ratio': 0,
                    'return_pct': 0,
                    'annualized_return': 0
                }

            # Basic trade metrics
            total_trades = len(trades)
            winning_trades = [t for t in trades if t['profit_loss'] > 0]
            losing_trades = [t for t in trades if t['profit_loss'] <= 0]

            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0

            # Profit metrics
            gross_profit = sum(t['profit_loss'] for t in winning_trades)
            gross_loss = abs(sum(t['profit_loss'] for t in losing_trades))
            net_profit = gross_profit - gross_loss

            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            avg_profit_per_trade = net_profit / total_trades if total_trades > 0 else 0

            # Calculate drawdown
            peak = equity_curve[0]['equity']
            max_drawdown = 0
            max_drawdown_pct = 0

            for point in equity_curve:
                if point['equity'] > peak:
                    peak = point['equity']

                drawdown = peak - point['equity']
                drawdown_pct = drawdown / peak if peak > 0 else 0

                if drawdown_pct > max_drawdown_pct:
                    max_drawdown = drawdown
                    max_drawdown_pct = drawdown_pct

            # Calculate returns
            initial_equity = equity_curve[0]['equity']
            final_equity = equity_curve[-1]['equity']

            return_pct = (final_equity / initial_equity - 1) * 100 if initial_equity > 0 else 0

            # Calculate annualized return
            start_date = equity_curve[0]['timestamp']
            end_date = equity_curve[-1]['timestamp']
            years = (end_date - start_date).days / 365.25

            annualized_return = (1 + return_pct / 100) ** (1 / years) - 1 if years > 0 else 0
            annualized_return *= 100  # Convert to percentage

            # Calculate risk metrics
            if daily_returns:
                daily_return_values = [r['return'] for r in daily_returns]

                # Sharpe ratio (assuming risk-free rate of 0)
                mean_daily_return = np.mean(daily_return_values)
                std_daily_return = np.std(daily_return_values)

                sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252) if std_daily_return > 0 else 0

                # Sortino ratio (only considering negative returns)
                negative_returns = [r for r in daily_return_values if r < 0]
                std_negative_returns = np.std(negative_returns) if negative_returns else 0

                sortino_ratio = (mean_daily_return / std_negative_returns) * np.sqrt(
                    252) if std_negative_returns > 0 else 0
            else:
                sharpe_ratio = 0
                sortino_ratio = 0

            # Calculate trade statistics
            avg_win = np.mean([t['profit_loss'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['profit_loss'] for t in losing_trades]) if losing_trades else 0

            # Calculate expectancy
            win_prob = win_rate
            loss_prob = 1 - win_rate

            expectancy = (win_prob * avg_win - loss_prob * abs(avg_loss)) if total_trades > 0 else 0
            expectancy_per_dollar = expectancy / abs(avg_loss) if avg_loss != 0 else 0

            # Compile all metrics
            metrics = {
                'total_trades': total_trades,
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'gross_profit': gross_profit,
                'gross_loss': gross_loss,
                'net_profit': net_profit,
                'avg_profit_per_trade': avg_profit_per_trade,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'expectancy': expectancy,
                'expectancy_per_dollar': expectancy_per_dollar,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown_pct * 100,  # Convert to percentage
                'return_pct': return_pct,
                'annualized_return': annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {
                'total_trades': len(trades),
                'win_rate': 0,
                'profit_factor': 0,
                'avg_profit_per_trade': 0,
                'max_drawdown': 0,
                'return_pct': 0,
                'error': str(e)
            }

    def generate_performance_report(self, output_dir: str = "BacktestResults") -> None:
        """Generate performance report with visualizations.

        Args:
            output_dir: Directory to save reports
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Check if we have results
            if not self.results.get('equity_curve') or not self.results.get('trades'):
                self.logger.error("No backtest results to generate report")
                return

            # 1. Equity curve
            self._plot_equity_curve(output_dir)

            # 2. Drawdown chart
            self._plot_drawdown(output_dir)

            # 3. Trade distribution
            self._plot_trade_distribution(output_dir)

            # 4. Monthly returns
            self._plot_monthly_returns(output_dir)

            # 5. Generate summary text report
            self._generate_summary_report(output_dir)

            self.logger.info(f"Performance report generated in {output_dir}")

        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")

    def _plot_equity_curve(self, output_dir: str) -> None:
        """Plot equity curve.

        Args:
            output_dir: Directory to save plot
        """
        try:
            # Extract data
            timestamps = [point['timestamp'] for point in self.results['equity_curve']]
            balance = [point['balance'] for point in self.results['equity_curve']]
            equity = [point['equity'] for point in self.results['equity_curve']]

            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, balance, label='Balance')
            plt.plot(timestamps, equity, label='Equity')

            # Add labels and title
            plt.title('Equity Curve')
            plt.xlabel('Date')
            plt.ylabel('Account Value ($)')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot
            plt.savefig(os.path.join(output_dir, 'equity_curve.png'))
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {e}")

    def _plot_drawdown(self, output_dir: str) -> None:
        """Plot drawdown chart.

        Args:
            output_dir: Directory to save plot
        """
        try:
            # Extract data
            timestamps = [point['timestamp'] for point in self.results['equity_curve']]
            equity = [point['equity'] for point in self.results['equity_curve']]

            # Calculate drawdown
            peak = equity[0]
            drawdown = []
            drawdown_pct = []

            for eq in equity:
                if eq > peak:
                    peak = eq

                dd = peak - eq
                dd_pct = (dd / peak) * 100 if peak > 0 else 0

                drawdown.append(dd)
                drawdown_pct.append(dd_pct)

            # Create plot
            plt.figure(figsize=(12, 6))
            plt.plot(timestamps, drawdown_pct, color='red')

            # Fill area
            plt.fill_between(timestamps, drawdown_pct, 0, color='red', alpha=0.3)

            # Add labels and title
            plt.title('Drawdown (%)')
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)

            # Invert y-axis to show drawdowns as negative
            plt.gca().invert_yaxis()

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save plot
            plt.savefig(os.path.join(output_dir, 'drawdown.png'))
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting drawdown: {e}")

    def _plot_trade_distribution(self, output_dir: str) -> None:
        """Plot trade distribution.

        Args:
            output_dir: Directory to save plot
        """
        try:
            # Extract data
            profits = [trade['profit_loss'] for trade in self.results['trades']]

            # Create plot
            plt.figure(figsize=(12, 6))
            plt.hist(profits, bins=20, alpha=0.7, color='blue')

            # Add vertical line at zero
            plt.axvline(x=0, color='red', linestyle='--')

            # Add labels and title
            plt.title('Trade Profit/Loss Distribution')
            plt.xlabel('Profit/Loss ($)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot
            plt.savefig(os.path.join(output_dir, 'trade_distribution.png'))
            plt.close()

            # Create second plot for cumulative P/L
            cum_profits = np.cumsum(profits)

            plt.figure(figsize=(12, 6))
            plt.plot(cum_profits)

            # Add labels and title
            plt.title('Cumulative Profit/Loss')
            plt.xlabel('Trade #')
            plt.ylabel('Cumulative P/L ($)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            # Save plot
            plt.savefig(os.path.join(output_dir, 'cumulative_pl.png'))
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting trade distribution: {e}")

    def _plot_monthly_returns(self, output_dir: str) -> None:
        """Plot monthly returns.

        Args:
            output_dir: Directory to save plot
        """
        try:
            # Extract data from daily returns and aggregate by month
            monthly_returns = {}

            for daily_return in self.results['daily_returns']:
                date = daily_return['date']
                month_key = date[:7]  # YYYY-MM format

                if month_key not in monthly_returns:
                    monthly_returns[month_key] = []

                monthly_returns[month_key].append(daily_return['return'])

            # Calculate compounded monthly returns
            months = []
            returns = []

            for month, daily_rets in sorted(monthly_returns.items()):
                months.append(month)

                # Compound daily returns for the month
                month_return = (np.prod([1 + r for r in daily_rets]) - 1) * 100
                returns.append(month_return)

            # Create plot
            plt.figure(figsize=(14, 6))
            bars = plt.bar(months, returns, color=['green' if r >= 0 else 'red' for r in returns])

            # Add labels and title
            plt.title('Monthly Returns (%)')
            plt.xlabel('Month')
            plt.ylabel('Return (%)')
            plt.grid(True, alpha=0.3, axis='y')

            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                if height >= 0:
                    va = 'bottom'
                    y_pos = height + 0.5
                else:
                    va = 'top'
                    y_pos = height - 0.5

                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    y_pos,
                    f'{height:.1f}%',
                    ha='center',
                    va=va,
                    fontsize=8
                )

            # Save plot
            plt.savefig(os.path.join(output_dir, 'monthly_returns.png'))
            plt.close()

        except Exception as e:
            self.logger.error(f"Error plotting monthly returns: {e}")

    def _generate_summary_report(self, output_dir: str) -> None:
        """Generate summary text report.

        Args:
            output_dir: Directory to save report
        """
        try:
            # Get metrics
            metrics = self.results['metrics']

            # Create report content
            report = [
                "BACKTEST SUMMARY REPORT",
                "======================="
            ]

            # Basic information
            report.extend([
                "",
                "BASIC INFORMATION",
                "-----------------",
                f"Initial Balance: ${self.params['initial_balance']:.2f}",
                f"Final Balance: ${metrics['net_profit'] + self.params['initial_balance']:.2f}",
                f"Net Profit: ${metrics['net_profit']:.2f}",
                f"Return: {metrics['return_pct']:.2f}%",
                f"Annualized Return: {metrics['annualized_return']:.2f}%",
                f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%"
            ])

            # Trade statistics
            report.extend([
                "",
                "TRADE STATISTICS",
                "----------------",
                f"Total Trades: {metrics['total_trades']}",
                f"Winning Trades: {metrics['winning_trades']}",
                f"Losing Trades: {metrics['losing_trades']}",
                f"Win Rate: {metrics['win_rate'] * 100:.2f}%",
                f"Profit Factor: {metrics['profit_factor']:.2f}",
                f"Average Profit per Trade: ${metrics['avg_profit_per_trade']:.2f}",
                f"Average Win: ${metrics['avg_win']:.2f}",
                f"Average Loss: ${metrics['avg_loss']:.2f}",
                f"Expectancy: ${metrics['expectancy']:.2f}",
                f"Expectancy per Dollar Risked: ${metrics['expectancy_per_dollar']:.2f}"
            ])

            # Risk metrics
            report.extend([
                "",
                "RISK METRICS",
                "------------",
                f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
                f"Sortino Ratio: {metrics['sortino_ratio']:.2f}",
                f"Max Drawdown: ${metrics['max_drawdown']:.2f} ({metrics['max_drawdown_pct']:.2f}%)"
            ])

            # Backtest parameters
            report.extend([
                "",
                "BACKTEST PARAMETERS",
                "-------------------",
                f"Initial Balance: ${self.params['initial_balance']:.2f}",
                f"Position Sizing: {self.params['position_size_mode']}",
                f"Risk per Trade: {self.params['risk_per_trade'] * 100:.2f}%" if self.params[
                                                                                     'position_size_mode'] == 'risk' else f"Fixed Position Size: {self.params['fixed_position_size']} lots",
                f"Spread: {self.params['spread_pips']} pips",
                f"Commission: ${self.params['commission_per_lot']:.2f} per lot",
                f"Slippage: {self.params['slippage_pips']} pips",
                f"Use Stop Loss/Take Profit: {'Yes' if self.params['use_sl_tp'] else 'No'}",
                f"Partial Close Enabled: {'Yes' if self.params['enable_partial_close'] else 'No'}",
                f"Breakeven Enabled: {'Yes' if self.params['enable_breakeven'] else 'No'}",
                f"Maximum Open Trades: {self.params['max_open_trades']}",
                f"Signal Threshold: {self.params['signal_threshold']:.2f}"
            ])

            # Write report to file
            with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
                f.write('\n'.join(report))

        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")

    def export_trades_to_csv(self, output_dir: str) -> None:
        """Export trades to CSV file.

        Args:
            output_dir: Directory to save CSV
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Check if we have trades
            if not self.results.get('trades'):
                self.logger.error("No trades to export")
                return

            # Create DataFrame from trades
            trades_df = pd.DataFrame(self.results['trades'])

            # Convert timestamps to strings
            if 'entry_time' in trades_df.columns:
                trades_df['entry_time'] = trades_df['entry_time'].apply(str)

            if 'exit_time' in trades_df.columns:
                trades_df['exit_time'] = trades_df['exit_time'].apply(str)

            # Save to CSV
            csv_path = os.path.join(output_dir, 'trades.csv')
            trades_df.to_csv(csv_path, index=False)

            self.logger.info(f"Trades exported to {csv_path}")

        except Exception as e:
            self.logger.error(f"Error exporting trades to CSV: {e}")

    def export_equity_curve_to_csv(self, output_dir: str) -> None:
        """Export equity curve to CSV file.

        Args:
            output_dir: Directory to save CSV
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Check if we have equity curve
            if not self.results.get('equity_curve'):
                self.logger.error("No equity curve to export")
                return

            # Create DataFrame from equity curve
            equity_df = pd.DataFrame(self.results['equity_curve'])

            # Convert timestamps to strings
            if 'timestamp' in equity_df.columns:
                equity_df['timestamp'] = equity_df['timestamp'].apply(str)

            # Save to CSV
            csv_path = os.path.join(output_dir, 'equity_curve.csv')
            equity_df.to_csv(csv_path, index=False)

            self.logger.info(f"Equity curve exported to {csv_path}")

        except Exception as e:
            self.logger.error(f"Error exporting equity curve to CSV: {e}")