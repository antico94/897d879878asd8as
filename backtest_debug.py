import pandas as pd
import numpy as np
import logging
import sys
import os
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger('backtest_debugger')

# Add project root to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Now import project-specific modules
from dependency_injector.wiring import inject, Provide
from Utilities.Container import Container
from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Evaluation.BacktestEngine import BacktestEngine
from Fetching.FetcherFactory import FetcherFactory
from Processing.ProcessorFactory import ProcessorFactory
from Models.ModelFactory import ModelFactory
from Strategies.StrategyFactory import StrategyFactory
from Evaluation.BacktestFactory import BacktestFactory


class BacktestDebugger:
    """A tool to debug backtesting issues by analyzing different components."""

    def __init__(self, backtest_engine):
        self.backtest_engine = backtest_engine
        self.logger = logger
        self.debugging_results = {
            'data_preparation': {},
            'signal_generation': {},
            'trade_execution': {},
            'model_prediction': {}
        }

    def debug_data_preparation(self, data: pd.DataFrame) -> dict:
        """Analyze the data preparation process."""
        result = {}

        try:
            # Check if data is empty
            if data.empty:
                result['status'] = 'FAIL'
                result['message'] = 'Data is empty'
                return result

            # Check for required columns
            required_cols = ['time', 'open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in data.columns]

            if missing_cols:
                result['status'] = 'FAIL'
                result['message'] = f'Missing required columns: {missing_cols}'
                return result

            # Check data timestamps
            result['date_range'] = {
                'start': data['time'].min(),
                'end': data['time'].max(),
                'total_bars': len(data)
            }

            # Check for missing values in key columns
            null_counts = {col: data[col].isnull().sum() for col in required_cols}
            if any(null_counts.values()):
                result['status'] = 'WARNING'
                result['message'] = f'Found null values in columns: {null_counts}'
            else:
                result['status'] = 'OK'
                result['message'] = 'Data preparation looks good'

            # Check if there's enough data for the model's sequence length
            seq_length = getattr(self.backtest_engine.model, 'sequence_length', 24)
            if len(data) <= seq_length:
                result['status'] = 'FAIL'
                result['message'] = f'Not enough data ({len(data)} bars) for sequence length ({seq_length})'

            # Sample some data rows for inspection
            result['data_sample'] = data.head(3).to_dict('records')

            return result

        except Exception as e:
            self.logger.error(f"Error in data preparation debugging: {str(e)}")
            result['status'] = 'ERROR'
            result['message'] = f'Debug error: {str(e)}'
            return result

    def debug_model_prediction(self, processed_data: dict) -> dict:
        """Debug model prediction issues."""
        result = {}

        try:
            # Check if processed data is valid
            if not processed_data or 'X' not in processed_data or len(processed_data['X']) == 0:
                result['status'] = 'FAIL'
                result['message'] = 'No processed data for prediction'
                return result

            # Try to make a prediction with a sample
            sample_X = processed_data['X'][0]

            # Ensure sample has correct shape for the model
            expected_shape = (self.backtest_engine.model.sequence_length, self.backtest_engine.model.n_features)
            actual_shape = sample_X.shape

            if actual_shape != expected_shape:
                result['status'] = 'FAIL'
                result['message'] = f'Shape mismatch: model expects {expected_shape}, got {actual_shape}'
                return result

            # Attempt prediction
            try:
                prediction = self.backtest_engine.model.predict(np.array([sample_X]))

                # Check prediction format
                if isinstance(prediction, dict):
                    result['prediction_keys'] = list(prediction.keys())

                    # Check for direction predictions
                    if 'direction' in prediction:
                        dir_values = prediction['direction']
                        result['direction_stats'] = {
                            'min': float(np.min(dir_values)),
                            'max': float(np.max(dir_values)),
                            'mean': float(np.mean(dir_values)),
                            'shape': dir_values.shape
                        }

                    result['status'] = 'OK'
                    result['message'] = 'Model prediction successful'
                else:
                    result['status'] = 'WARNING'
                    result['message'] = f'Unexpected prediction format: {type(prediction)}'

            except Exception as e:
                self.logger.error(f"Error making prediction: {str(e)}")
                result['status'] = 'FAIL'
                result['message'] = f'Prediction error: {str(e)}'

            return result

        except Exception as e:
            self.logger.error(f"Error in model prediction debugging: {str(e)}")
            result['status'] = 'ERROR'
            result['message'] = f'Debug error: {str(e)}'
            return result

    def debug_signal_generation(self, prediction: dict, market_data: dict) -> dict:
        """Debug signal generation issues."""
        result = {}

        try:
            # Check prediction format
            if not prediction or not isinstance(prediction, dict):
                result['status'] = 'FAIL'
                result['message'] = f'Invalid prediction format: {type(prediction)}'
                return result

            # Check market data
            if not market_data or not isinstance(market_data, dict):
                result['status'] = 'FAIL'
                result['message'] = f'Invalid market data format: {type(market_data)}'
                return result

            # Log the prediction and market data for inspection
            result['prediction'] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in prediction.items()
            }
            result['market_data'] = market_data

            # Generate signals using signal generator
            try:
                signals = self.backtest_engine.signal_generator.generate_signals(
                    predictions=prediction,
                    confidence=np.array([0.8]),  # Default confidence for backtest
                    current_market_data=market_data
                )

                if not signals:
                    result['status'] = 'WARNING'
                    result['message'] = 'No signals generated from prediction'
                else:
                    result['status'] = 'OK'
                    result['message'] = f'Generated {len(signals)} signals'

                # Check signal threshold filter
                threshold = self.backtest_engine.params['signal_threshold']
                valid_signals = self.backtest_engine.signal_generator.filter_signals(
                    signals, min_strength=threshold
                )

                result['signals'] = {
                    'total': len(signals),
                    'valid_after_threshold': len(valid_signals),
                    'threshold': threshold
                }

                if len(signals) > 0 and len(valid_signals) == 0:
                    result['status'] = 'WARNING'
                    result['message'] = f'All signals filtered out by threshold ({threshold})'

                    # Check signal strengths
                    if signals:
                        strengths = [s.get('signal_strength', 0) for s in signals]
                        result['signal_strengths'] = {
                            'min': min(strengths),
                            'max': max(strengths),
                            'mean': sum(strengths) / len(strengths)
                        }

            except Exception as e:
                self.logger.error(f"Error generating signals: {str(e)}")
                result['status'] = 'FAIL'
                result['message'] = f'Signal generation error: {str(e)}'

            return result

        except Exception as e:
            self.logger.error(f"Error in signal generation debugging: {str(e)}")
            result['status'] = 'ERROR'
            result['message'] = f'Debug error: {str(e)}'
            return result

    def debug_trade_execution(self, account: dict, signal: dict, bar_data: pd.Series, market_data: dict) -> dict:
        """Debug trade execution issues."""
        result = {}

        try:
            # Check account state
            result['account'] = {
                'balance': account['balance'],
                'open_positions': len(account['open_positions']),
                'max_open_positions': self.backtest_engine.params['max_open_trades']
            }

            # Check if we can open new positions
            if len(account['open_positions']) >= self.backtest_engine.params['max_open_trades']:
                result['status'] = 'WARNING'
                result['message'] = 'Max open positions reached'
                return result

            # Simulate position opening without actually modifying the account
            try:
                # Determine trade direction
                signal_type = signal['type']
                is_buy = signal_type.value.endswith('BUY')
                direction = 'BUY' if is_buy else 'SELL'

                # Calculate entry price (with spread)
                spread_adjustment = self.backtest_engine.params['spread_pips'] / 10.0
                entry_price = bar_data['close'] + spread_adjustment if is_buy else bar_data['close'] - spread_adjustment

                # Calculate stop loss and take profit
                stop_loss = self.backtest_engine.risk_manager.calculate_stop_loss(
                    entry_price, signal.get('expected_volatility', 0),
                    market_data.get('atr', entry_price * 0.01), signal_type
                )

                take_profit = self.backtest_engine.risk_manager.calculate_take_profit(
                    entry_price, entry_price, stop_loss,
                    signal.get('expected_magnitude', 0), signal_type
                )

                # Calculate position size
                stop_loss_pips = abs(entry_price - stop_loss) * 10

                if self.backtest_engine.params['position_size_mode'] == 'risk':
                    risk_amount = account['balance'] * self.backtest_engine.params['risk_per_trade']
                    pip_value = 10.0  # $10 per pip for 1 lot
                    position_size = risk_amount / (stop_loss_pips * pip_value)
                    position_size = max(min(position_size, 10.0), 0.01)
                    position_size = round(position_size, 2)
                else:
                    position_size = self.backtest_engine.params['fixed_position_size']

                result['simulated_trade'] = {
                    'direction': direction,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'position_size': position_size,
                    'stop_loss_pips': stop_loss_pips
                }

                # Calculate risk-reward ratio
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit - entry_price)
                risk_reward = reward / risk if risk > 0 else 0

                result['risk_reward'] = risk_reward

                result['status'] = 'OK'
                result['message'] = 'Trade execution simulation successful'

            except Exception as e:
                self.logger.error(f"Error simulating trade execution: {str(e)}")
                result['status'] = 'FAIL'
                result['message'] = f'Trade execution error: {str(e)}'

            return result

        except Exception as e:
            self.logger.error(f"Error in trade execution debugging: {str(e)}")
            result['status'] = 'ERROR'
            result['message'] = f'Debug error: {str(e)}'
            return result

    def run_full_debug(self, pair: str, timeframe: str, start_date: str, end_date: str) -> dict:
        """Run a full debug cycle on the backtest engine."""
        debug_results = {}

        # 1. Debug data loading
        data = self.backtest_engine._load_historical_data(pair, timeframe, start_date, end_date)
        debug_results['data_preparation'] = self.debug_data_preparation(data)

        if debug_results['data_preparation']['status'] == 'FAIL':
            self.logger.error("Data preparation failed, stopping debug")
            return debug_results

        # 2. Create DataPreprocessor for data preparation
        try:
            from Training.DataPreprocessor import DataPreprocessor
            data_preprocessor = DataPreprocessor(
                self.backtest_engine.config,
                self.backtest_engine.logger,
                self.backtest_engine.data_storage
            )
            data_preprocessor.load_feature_importance()
        except Exception as e:
            self.logger.error(f"Error creating DataPreprocessor: {str(e)}")
            debug_results['data_processing'] = {
                'status': 'FAIL',
                'message': f'Error creating DataPreprocessor: {str(e)}'
            }
            return debug_results

        # 3. Get sequence length from model
        sequence_length = self.backtest_engine.model.sequence_length if hasattr(self.backtest_engine.model,
                                                                                'sequence_length') else 24

        # 4. Process data for prediction
        processed_data = self.backtest_engine._prepare_data_for_backtest(
            data, data_preprocessor, sequence_length
        )

        debug_results['processed_data'] = {
            'sequences_shape': processed_data['X'].shape if 'X' in processed_data else None,
            'status': 'OK' if 'X' in processed_data and len(processed_data['X']) > 0 else 'FAIL'
        }

        if debug_results['processed_data']['status'] == 'FAIL':
            self.logger.error("Data processing failed, stopping debug")
            return debug_results

        # 5. Debug model prediction
        debug_results['model_prediction'] = self.debug_model_prediction(processed_data)

        if debug_results['model_prediction']['status'] == 'FAIL':
            self.logger.error("Model prediction failed, stopping debug")
            return debug_results

        # 6. Debug signal generation with a sample
        sample_idx = 5  # Use a sample point that's not at the beginning
        if len(processed_data['X']) > sample_idx:
            X_sequence = processed_data['X'][sample_idx]
            prediction = self.backtest_engine.model.predict(np.array([X_sequence]))

            # Create market data dict for the sample
            if len(processed_data['prices']) > sample_idx:
                idx = sample_idx + sequence_length - 1  # Account for sequence offset
                sample_time = processed_data['times'][sample_idx] if sample_idx < len(processed_data['times']) else None

                if idx < len(data):
                    sample_row = data.iloc[idx]

                    market_data = {
                        'symbol': pair,
                        'price': sample_row['close'],
                        'high': sample_row['high'],
                        'low': sample_row['low'],
                        'time': sample_time or sample_row['time'],
                        'atr': sample_row.get('atr', sample_row['close'] * 0.01),
                        'account_balance': self.backtest_engine.params['initial_balance']
                    }

                    debug_results['signal_generation'] = self.debug_signal_generation(
                        prediction, market_data
                    )

                    # 7. Debug trade execution if a signal was generated
                    if ('signals' in debug_results['signal_generation'] and
                            debug_results['signal_generation']['signals']['valid_after_threshold'] > 0):

                        # Find the valid signals
                        signals = self.backtest_engine.signal_generator.generate_signals(
                            predictions=prediction,
                            confidence=np.array([0.8]),
                            current_market_data=market_data
                        )

                        valid_signals = self.backtest_engine.signal_generator.filter_signals(
                            signals, min_strength=self.backtest_engine.params['signal_threshold']
                        )

                        if valid_signals:
                            # Mock account
                            mock_account = {
                                'balance': self.backtest_engine.params['initial_balance'],
                                'equity': self.backtest_engine.params['initial_balance'],
                                'open_positions': {},
                                'closed_trades': []
                            }

                            debug_results['trade_execution'] = self.debug_trade_execution(
                                mock_account, valid_signals[0], sample_row, market_data
                            )

        return debug_results

    def summarize_results(self, debug_results: dict) -> str:
        """Create a human-readable summary of the debug results."""
        summary = []
        summary.append("BACKTEST DEBUGGING SUMMARY")
        summary.append("=========================")
        summary.append("")

        # Data preparation summary
        if 'data_preparation' in debug_results:
            dp = debug_results['data_preparation']
            summary.append(f"DATA PREPARATION: {dp['status']}")
            summary.append(f"Message: {dp.get('message', 'N/A')}")

            if 'date_range' in dp:
                dr = dp['date_range']
                summary.append(f"Date range: {dr['start']} to {dr['end']} ({dr['total_bars']} bars)")

            summary.append("")

        # Processed data summary
        if 'processed_data' in debug_results:
            pd = debug_results['processed_data']
            summary.append(f"DATA PROCESSING: {pd['status']}")

            if pd['sequences_shape']:
                summary.append(f"Processed data shape: {pd['sequences_shape']}")
            else:
                summary.append("No processed data was generated")

            summary.append("")

        # Model prediction summary
        if 'model_prediction' in debug_results:
            mp = debug_results['model_prediction']
            summary.append(f"MODEL PREDICTION: {mp['status']}")
            summary.append(f"Message: {mp.get('message', 'N/A')}")

            if 'prediction_keys' in mp:
                summary.append(f"Prediction outputs: {', '.join(mp['prediction_keys'])}")

            if 'direction_stats' in mp:
                ds = mp['direction_stats']
                summary.append(f"Direction values range: {ds['min']:.2f} to {ds['max']:.2f} (mean: {ds['mean']:.2f})")

            summary.append("")

        # Signal generation summary
        if 'signal_generation' in debug_results:
            sg = debug_results['signal_generation']
            summary.append(f"SIGNAL GENERATION: {sg['status']}")
            summary.append(f"Message: {sg.get('message', 'N/A')}")

            if 'signals' in sg:
                s = sg['signals']
                summary.append(f"Total signals: {s['total']}")
                summary.append(f"Valid signals after threshold ({s['threshold']}): {s['valid_after_threshold']}")

                if 'signal_strengths' in sg:
                    ss = sg['signal_strengths']
                    summary.append(
                        f"Signal strengths range: {ss['min']:.2f} to {ss['max']:.2f} (mean: {ss['mean']:.2f})")

            summary.append("")

        # Trade execution summary
        if 'trade_execution' in debug_results:
            te = debug_results['trade_execution']
            summary.append(f"TRADE EXECUTION: {te['status']}")
            summary.append(f"Message: {te.get('message', 'N/A')}")

            if 'account' in te:
                a = te['account']
                summary.append(f"Account balance: ${a['balance']:.2f}")
                summary.append(f"Open positions: {a['open_positions']}/{a['max_open_positions']}")

            if 'simulated_trade' in te:
                st = te['simulated_trade']
                summary.append(f"Simulated {st['direction']} trade:")
                summary.append(
                    f"  Entry: {st['entry_price']:.2f}, Stop Loss: {st['stop_loss']:.2f}, Take Profit: {st['take_profit']:.2f}")
                summary.append(
                    f"  Position size: {st['position_size']} lots, SL distance: {st['stop_loss_pips']:.1f} pips")

                if 'risk_reward' in te:
                    summary.append(f"  Risk-Reward ratio: {te['risk_reward']:.2f}")

            summary.append("")

        # Final conclusion
        summary.append("CONCLUSION")
        summary.append("==========")

        # Identify the most likely issue
        if 'data_preparation' in debug_results and debug_results['data_preparation']['status'] == 'FAIL':
            summary.append("ISSUE IDENTIFIED: Data preparation failure")
            summary.append(f"Details: {debug_results['data_preparation'].get('message', 'Unknown')}")

        elif 'processed_data' in debug_results and debug_results['processed_data']['status'] == 'FAIL':
            summary.append("ISSUE IDENTIFIED: Data processing failure")

        elif 'model_prediction' in debug_results and debug_results['model_prediction']['status'] == 'FAIL':
            summary.append("ISSUE IDENTIFIED: Model prediction failure")
            summary.append(f"Details: {debug_results['model_prediction'].get('message', 'Unknown')}")

        elif ('signal_generation' in debug_results and
              ('signals' not in debug_results['signal_generation'] or
               debug_results['signal_generation']['signals']['valid_after_threshold'] == 0)):

            summary.append("ISSUE IDENTIFIED: No valid trading signals generated")
            if 'signals' in debug_results['signal_generation']:
                signals = debug_results['signal_generation']['signals']
                if signals['total'] > 0 and signals['valid_after_threshold'] == 0:
                    summary.append(
                        f"Details: All {signals['total']} signals were filtered out by threshold ({signals['threshold']})")

                    if 'signal_strengths' in debug_results['signal_generation']:
                        ss = debug_results['signal_generation']['signal_strengths']
                        summary.append(
                            f"Signal strengths range: {ss['min']:.2f} to {ss['max']:.2f} (mean: {ss['mean']:.2f})")
                        summary.append(f"Consider lowering the signal threshold below {ss['max']:.2f}")

        elif 'trade_execution' in debug_results and debug_results['trade_execution']['status'] == 'FAIL':
            summary.append("ISSUE IDENTIFIED: Trade execution failure")
            summary.append(f"Details: {debug_results['trade_execution'].get('message', 'Unknown')}")

        else:
            summary.append("No clear issue identified. The backtesting components appear to be functioning correctly.")
            summary.append("Possible issues to check:")
            summary.append("1. Signal threshold may be too high")
            summary.append("2. Model predictions may not be strong enough to generate signals")
            summary.append("3. Check the date range to ensure you're testing on a period with market movement")

        return "\n".join(summary)


def fix_signal_threshold(backtest_engine, new_threshold=0.3):
    """Apply a common fix - lower the signal threshold."""
    original_threshold = backtest_engine.params['signal_threshold']
    backtest_engine.params['signal_threshold'] = new_threshold
    logger.info(f"Lowered signal threshold from {original_threshold} to {new_threshold}")
    return original_threshold


@inject
def create_backtest_components(
        config: Config = Provide[Container.config],
        logger: Logger = Provide[Container.logger],
        processor_factory: ProcessorFactory = Provide[Container.processor_factory],
        model_factory: ModelFactory = Provide[Container.model_factory],
        strategy_factory: StrategyFactory = Provide[Container.strategy_factory],
        backtest_factory: BacktestFactory = Provide[Container.backtest_factory]
):
    # Get components for backtesting
    data_storage = processor_factory.create_data_storage()

    # Get model path from args or use a default
    model_path = args.model_path if args.model_path else "TrainedModels/best_model.h5"

    try:
        model = model_factory.load_model(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        print(f"ERROR: Failed to load model from {model_path}: {e}")
        sys.exit(1)

    signal_generator = strategy_factory.create_signal_generator()
    risk_manager = strategy_factory.create_risk_manager()

    # Create backtest engine
    backtest_engine = backtest_factory.create_backtest_engine(
        data_storage, model, signal_generator, risk_manager
    )

    return backtest_engine


def main():
    logger.info("Starting backtest debugger")

    # Initialize dependency injection container
    container = Container()
    container.wire(modules=[__name__])

    # Create backtest components
    backtest_engine = create_backtest_components()

    # Create debugger
    debugger = BacktestDebugger(backtest_engine)

    # Run full debug
    pair = args.pair if args.pair else "XAUUSD"
    timeframe = args.timeframe if args.timeframe else "H1"
    start_date = args.start_date if args.start_date else "2023-01-01"
    end_date = args.end_date if args.end_date else "2023-03-01"

    print(f"Running debug for {pair} {timeframe} from {start_date} to {end_date}")
    debug_results = debugger.run_full_debug(pair, timeframe, start_date, end_date)

    # Print summary
    summary = debugger.summarize_results(debug_results)
    print("\n" + summary)

    # Apply fix if needed and if auto-fix is enabled
    if args.auto_fix and 'signal_generation' in debug_results:
        if 'signal_strengths' in debug_results['signal_generation']:
            strengths = debug_results['signal_generation']['signal_strengths']
            if 'max' in strengths:
                # Set threshold just below the max signal strength
                new_threshold = max(0.2, strengths['max'] * 0.9)
                original = fix_signal_threshold(backtest_engine, new_threshold)
                print(f"\nAUTO-FIX: Lowered signal threshold from {original} to {new_threshold}")

                # Run a quick test backtest
                print("\nRunning quick test backtest with new threshold...")
                try:
                    backtest_engine.run_backtest(pair, timeframe, start_date, end_date)
                    trades = len(backtest_engine.results.get('trades', []))
                    print(f"Test backtest generated {trades} trades")
                except Exception as e:
                    print(f"Test backtest failed: {e}")

    logger.info("Backtest debugging completed")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Debug backtesting issues')
    parser.add_argument('--pair', help='Currency pair to test (default: XAUUSD)')
    parser.add_argument('--timeframe', help='Timeframe to test (default: H1)')
    parser.add_argument('--start-date', help='Start date in YYYY-MM-DD format (default: 2023-01-01)')
    parser.add_argument('--end-date', help='End date in YYYY-MM-DD format (default: 2023-03-01)')
    parser.add_argument('--model-path', help='Path to the model file (default: TrainedModels/best_model.h5)')
    parser.add_argument('--auto-fix', action='store_true', help='Automatically apply fixes for common issues')

    args = parser.parse_args()

    main()