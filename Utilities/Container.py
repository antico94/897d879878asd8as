from dependency_injector import containers, providers
from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Fetching.FetcherFactory import FetcherFactory
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
