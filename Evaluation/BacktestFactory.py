from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Evaluation.BacktestEngine import BacktestEngine
import os
import json
from typing import Optional, Dict, Any


class BacktestFactory:
    """Factory for creating backtesting components."""

    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

    def create_backtest_engine(
            self,
            data_storage,
            model,
            signal_generator,
            risk_manager
    ) -> BacktestEngine:
        """Create a backtest engine instance."""
        return BacktestEngine(
            self.config,
            self.logger,
            data_storage,
            model,
            signal_generator,
            risk_manager
        )

    def load_backtest_results(self, results_path: str) -> Dict[str, Any]:
        """Load backtest results from file."""
        try:
            self.logger.info(f"Loading backtest results from {results_path}")

            if not os.path.exists(results_path):
                self.logger.error(f"Results path not found: {results_path}")
                raise FileNotFoundError(f"Results path not found: {results_path}")

            # Look for summary report
            summary_path = os.path.join(results_path, 'summary_report.txt')
            if not os.path.exists(summary_path):
                self.logger.warning(f"Summary report not found at {summary_path}")

            # Load trades CSV if available
            trades_path = os.path.join(results_path, 'trades.csv')
            trades_data = None
            if os.path.exists(trades_path):
                import pandas as pd
                trades_data = pd.read_csv(trades_path)

            # Load equity curve CSV if available
            equity_path = os.path.join(results_path, 'equity_curve.csv')
            equity_data = None
            if os.path.exists(equity_path):
                import pandas as pd
                equity_data = pd.read_csv(equity_path)

            # Return combined results
            results = {
                'path': results_path,
                'summary_path': summary_path,
                'trades': trades_data,
                'equity_curve': equity_data
            }

            return results

        except Exception as e:
            self.logger.error(f"Error loading backtest results: {e}")
            raise
