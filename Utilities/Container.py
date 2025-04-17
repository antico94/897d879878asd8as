from dependency_injector import containers, providers
from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
import logging

class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=[__name__])
    config = providers.Singleton(Config, file_path='Configuration/Configuration.yaml')
    logger = providers.Singleton(Logger, name='MT5App', level=logging.INFO)