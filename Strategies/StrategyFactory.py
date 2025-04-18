from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Strategies.SignalGenerator import SignalGenerator
from Strategies.RiskManager import RiskManager
from Strategies.TradeExecutor import TradeExecutor
from Trading.MT5Connector import MT5Connector
from typing import Optional, Dict, Any, List
from Trading.TradingSession import TradingSession


class StrategyFactory:
    """Factory for creating trading strategy components."""

    def __init__(self, config: Config, logger: Logger, model_factory):
        self.config = config
        self.logger = logger
        self.model_factory = model_factory
        self.active_trading_session = None
        self.mt5_connector = None

        # Initialize MT5 connector
        self._initialize_mt5_connector()

    def _initialize_mt5_connector(self) -> None:
        """Initialize MT5 connector."""
        try:
            self.mt5_connector = MT5Connector(self.config, self.logger)
            # Don't connect immediately, we'll connect when needed
            self.logger.info("MT5 connector initialized (not yet connected)")
        except Exception as e:
            self.logger.error(f"Error initializing MT5 connector: {e}")

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

    def get_mt5_connector(self) -> Optional[MT5Connector]:
        """Get the MT5 connector."""
        return self.mt5_connector

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