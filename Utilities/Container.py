from dependency_injector import containers, providers
from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Processing.ProcessorFactory import ProcessorFactory
from Fetching.FetcherFactory import FetcherFactory
from Models.ModelFactory import ModelFactory
from Strategies.StrategyFactory import StrategyFactory
from Evaluation.BacktestFactory import BacktestFactory
from Utilities.PathResolver import PathResolver
import logging


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=[__name__])

    config = providers.Singleton(Config, file_path='Configuration/Configuration.yaml')

    # Logger with warnings and errors only for console, full logging to database
    logger = providers.Singleton(
        Logger,
        name='MT5App',
        level=logging.INFO,
        use_console=True,
        console_level=logging.WARNING,  # Only warnings and errors to console
        db_config=providers.Callable(
            lambda c: c['Database'],
            c=config
        )
    )

    # Fetcher factory
    fetcher_factory = providers.Singleton(
        FetcherFactory,
        config=config,
        logger=logger
    )

    # Data processor factory
    processor_factory = providers.Singleton(
        ProcessorFactory,
        config=config,
        logger=logger
    )

    # Model factory for ML components
    model_factory = providers.Singleton(
        ModelFactory,
        config=config,
        logger=logger
    )

    # Strategy factory for trading components
    strategy_factory = providers.Singleton(
        StrategyFactory,
        config=config,
        logger=logger,
        model_factory=model_factory
    )

    # Backtest factory
    backtest_factory = providers.Singleton(
        BacktestFactory,
        config=config,
        logger=logger
    )

    path_resolver = providers.Singleton(
        PathResolver,
        config=config
    )