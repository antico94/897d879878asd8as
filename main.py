# file: main.py
from dependency_injector.wiring import inject, Provide
from Utilities.Container import Container
from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from UI.cli import TradingBotCLI


@inject
def main(
        config: Config = Provide[Container.config],
        logger: Logger = Provide[Container.logger]
) -> None:
    logger.info('Application started')
    mt5_login = config.get_nested('MetaTrader5', 'Login')
    logger.debug(f'MT5 login: {mt5_login}')
    mt5_password = config.get_nested('MetaTrader5', 'Password')
    mt5_server = config.get_nested('MetaTrader5', 'Server')
    mt5_timeout = config.get_nested('MetaTrader5', 'Timeout')
    fetch_settings = config.get('FetchingSettings')
    db_host = config.get_nested('Database', 'Host')
    db_port = config.get_nested('Database', 'Port')
    db_user = config.get_nested('Database', 'User')
    db_password = config.get_nested('Database', 'Password')
    db_name = config.get_nested('Database', 'Database')
    logger.info(f'MT5 Server: {mt5_server} with timeout {mt5_timeout}')
    logger.info(f'Database: {db_user}@{db_host}:{db_port}/{db_name}')
    logger.info('Application finished')


if __name__ == '__main__':
    container = Container()
    container.wire(modules=[__name__])
    main()
    cli = TradingBotCLI()
    cli.main_menu()
