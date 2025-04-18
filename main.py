from dependency_injector.wiring import inject, Provide
from Utilities.Container import Container
from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Fetching.FetcherFactory import FetcherFactory
from Processing.ProcessorFactory import ProcessorFactory
from Models.ModelFactory import ModelFactory
from Strategies.StrategyFactory import StrategyFactory
from Evaluation.BacktestFactory import BacktestFactory
from UI.cli import TradingBotCLI
from UI.Constants import AppMode


@inject
def main(
        config: Config = Provide[Container.config],
        logger: Logger = Provide[Container.logger],
        fetcher_factory: FetcherFactory = Provide[Container.fetcher_factory],
        processor_factory: ProcessorFactory = Provide[Container.processor_factory],
        model_factory: ModelFactory = Provide[Container.model_factory],
        strategy_factory: StrategyFactory = Provide[Container.strategy_factory],
        backtest_factory: BacktestFactory = Provide[Container.backtest_factory]
) -> None:
    logger.info('Application started')

    # Log configuration values
    mt5_login = config.get_nested('MetaTrader5', 'Login')
    logger.debug(f'MT5 login: {mt5_login}')
    mt5_server = config.get_nested('MetaTrader5', 'Server')
    mt5_timeout = config.get_nested('MetaTrader5', 'Timeout')

    # Log database connection info
    db_host = config.get_nested('Database', 'Host')
    db_port = config.get_nested('Database', 'Port')
    db_user = config.get_nested('Database', 'User')
    db_name = config.get_nested('Database', 'Database')

    logger.info(f'MT5 Server: {mt5_server} with timeout {mt5_timeout}')
    logger.info(f'Database: {db_user}@{db_host}:{db_port}/{db_name}')

    # Run CLI
    cli = TradingBotCLI(config)

    while True:
        action = cli.main_menu()

        if action == "exit":
            logger.info('Application exiting')
            break
        elif action == AppMode.FETCH_DATA.value:
            handle_fetch_data(cli, logger, fetcher_factory)
        elif action == AppMode.PROCESS_DATA.value:
            handle_process_data(cli, logger, processor_factory)
        elif action == AppMode.VALIDATE_DATA.value:
            handle_validate_data(cli, logger, processor_factory)
        elif action == AppMode.ANALYZE_FEATURES.value:
            handle_analyze_features(cli, logger, processor_factory)
        elif action == AppMode.TRAIN_MODEL.value:
            handle_train_model(cli, logger, processor_factory, model_factory)
        elif action == AppMode.BACKTEST.value:
            handle_backtest(cli, logger, processor_factory, model_factory, strategy_factory, backtest_factory)
        elif action == AppMode.LIVE_TRADING.value:
            handle_live_trading(cli, logger, processor_factory, model_factory, strategy_factory)
        else:
            logger.info(f'Selected action: {action}')

    logger.info('Application finished')


def handle_fetch_data(cli: TradingBotCLI, logger: Logger, fetcher_factory: FetcherFactory) -> None:
    """Handle fetch data flow"""
    while True:
        fetch_action = cli.fetch_data_menu()

        if fetch_action == "back":
            logger.info("Returning to main menu")
            break

        elif fetch_action == "fetch_current":
            logger.info("Fetching data with current configuration")
            print("Starting data fetch with current configuration...")

            fetcher = fetcher_factory.create_mt5_fetcher()
            success = fetcher.fetch_data()

            if success:
                logger.info("Data fetching completed successfully")
                print("✓ Data fetching completed successfully")
            else:
                logger.error("Data fetching failed")
                print("✗ Data fetching failed")

        elif fetch_action == "change_config":
            logger.info("Changing fetching configuration")
            new_config = cli.change_config_menu()

            if new_config:
                logger.info(f"New configuration: {new_config}")
                print(f"Starting data fetch with new configuration...")

                fetcher = fetcher_factory.create_mt5_fetcher()
                success = fetcher.fetch_data(
                    pair=new_config['pair'],
                    days=new_config['days'],
                    timeframe=new_config['timeframe']
                )

                if success:
                    logger.info("Data fetching with new config completed successfully")
                    print("✓ Data fetching completed successfully")
                else:
                    logger.error("Data fetching with new config failed")
                    print("✗ Data fetching failed")
            else:
                logger.info("Configuration change cancelled")
                print("Configuration change cancelled")


def handle_process_data(cli: TradingBotCLI, logger: Logger, processor_factory: ProcessorFactory) -> None:
    """Handle data processing flow"""
    while True:
        process_action = cli.process_data_menu()

        if process_action == "back":
            logger.info("Returning to main menu")
            break

        elif process_action == "process_all":
            logger.info("Processing all datasets")
            print("Starting to process all datasets...")

            processor = processor_factory.create_data_processor()
            storage = processor_factory.create_data_storage()

            # Process training, validation, and testing datasets for XAUUSD
            process_datasets(processor, storage, logger, "XAUUSD", "H1", ["training", "validation", "testing"])

        elif process_action == "process_specific":
            logger.info("Processing specific dataset")
            dataset_config = cli.select_dataset_menu()

            if dataset_config:
                logger.info(f"Selected dataset: {dataset_config}")
                print(f"Processing {dataset_config['dataset_type']} data for "
                      f"{dataset_config['pair']} {dataset_config['timeframe']}...")

                processor = processor_factory.create_data_processor()
                storage = processor_factory.create_data_storage()

                process_datasets(
                    processor,
                    storage,
                    logger,
                    dataset_config['pair'],
                    dataset_config['timeframe'],
                    [dataset_config['dataset_type']]
                )
            else:
                logger.info("Dataset selection cancelled")
                print("Dataset selection cancelled")


def handle_validate_data(cli: TradingBotCLI, logger: Logger, processor_factory: ProcessorFactory) -> None:
    """Handle data validation flow"""
    while True:
        validate_action = cli.validate_data_menu()

        if validate_action == "back":
            logger.info("Returning to main menu")
            break

        elif validate_action == "validate_all":
            logger.info("Validating all datasets")
            print("Starting to validate all datasets...")

            processor = processor_factory.create_data_processor()
            validator = processor_factory.create_indicator_validator()

            # Validate training, validation, and testing datasets for XAUUSD
            validate_datasets(processor, validator, logger, "XAUUSD", "H1", ["training", "validation", "testing"])

        elif validate_action == "validate_specific":
            logger.info("Validating specific dataset")
            dataset_config = cli.select_dataset_menu()

            if dataset_config:
                logger.info(f"Selected dataset for validation: {dataset_config}")
                print(f"Validating {dataset_config['dataset_type']} data for "
                      f"{dataset_config['pair']} {dataset_config['timeframe']}...")

                processor = processor_factory.create_data_processor()
                validator = processor_factory.create_indicator_validator()

                validate_datasets(
                    processor,
                    validator,
                    logger,
                    dataset_config['pair'],
                    dataset_config['timeframe'],
                    [dataset_config['dataset_type']]
                )
            else:
                logger.info("Dataset selection cancelled")
                print("Dataset selection cancelled")

        elif validate_action == "generate_visualizations":
            logger.info("Generating indicator visualizations")
            print("Generating visualizations of technical indicators...")

            processor = processor_factory.create_data_processor()
            validator = processor_factory.create_indicator_validator()

            try:
                # Use training data for visualizations
                df = processor.get_data_from_db("XAUUSD", "H1", "training")
                processed_df = processor.process_raw_data(df)

                if not processed_df.empty:
                    validator.generate_indicator_visualizations(processed_df)
                    print("✓ Visualizations generated successfully in ValidationReports/Visualizations")
                else:
                    logger.error("No data available for visualization")
                    print("✗ No data available for visualization")

            except Exception as e:
                logger.error(f"Error generating visualizations: {e}")
                print(f"✗ Error generating visualizations: {str(e)}")


def handle_analyze_features(cli: TradingBotCLI, logger: Logger, processor_factory: ProcessorFactory) -> None:
    """Handle feature analysis flow"""
    while True:
        analyze_action = cli.analyze_features_menu()

        if analyze_action == "back":
            logger.info("Returning to main menu")
            break

        elif analyze_action == "run_analysis":
            logger.info("Running feature analysis")
            print("Starting feature analysis...")

            feature_analyzer = processor_factory.create_feature_analyzer()

            # Run analysis for default settings
            try:
                selected_features = feature_analyzer.run_complete_analysis(
                    pair="XAUUSD",
                    timeframe="H1",
                    dataset_type="training",
                    target_col="future_price_1"
                )

                if selected_features:
                    print(f"✓ Feature analysis completed successfully")
                    print(f"  - Selected {len(selected_features)} optimal features")
                    print(f"  - Results saved to FeatureAnalysis directory")
                else:
                    print("✗ Feature analysis failed to select features")
            except Exception as e:
                logger.error(f"Error running feature analysis: {e}")
                print(f"✗ Error running feature analysis: {str(e)}")

        elif analyze_action == "custom_analysis":
            logger.info("Running custom feature analysis")
            analysis_config = cli.feature_analysis_config_menu()

            if analysis_config:
                logger.info(f"Selected analysis config: {analysis_config}")
                print(f"Running feature analysis for {analysis_config['pair']} {analysis_config['timeframe']} "
                      f"with target {analysis_config['target']}...")

                feature_analyzer = processor_factory.create_feature_analyzer()

                try:
                    selected_features = feature_analyzer.run_complete_analysis(
                        pair=analysis_config['pair'],
                        timeframe=analysis_config['timeframe'],
                        dataset_type="training",
                        target_col=analysis_config['target']
                    )

                    if selected_features:
                        print(f"✓ Feature analysis completed successfully")
                        print(f"  - Selected {len(selected_features)} optimal features")
                        print(f"  - Results saved to FeatureAnalysis directory")
                    else:
                        print("✗ Feature analysis failed to select features")
                except Exception as e:
                    logger.error(f"Error running feature analysis: {e}")
                    print(f"✗ Error running feature analysis: {str(e)}")
            else:
                logger.info("Feature analysis configuration cancelled")
                print("Feature analysis configuration cancelled")


def handle_train_model(cli: TradingBotCLI, logger: Logger,
                       processor_factory: ProcessorFactory,
                       model_factory: ModelFactory) -> None:
    """Handle model training flow"""
    while True:
        train_action = cli.train_model_menu()

        if train_action == "back":
            logger.info("Returning to main menu")
            break

        elif train_action == "train_new":
            logger.info("Training new model")
            training_config = cli.model_training_config_menu()

            if training_config:
                logger.info(f"Selected training config: {training_config}")
                print(f"Training new model for {training_config['pair']} {training_config['timeframe']}...")

                # Get components needed for training
                data_storage = processor_factory.create_data_storage()
                data_preprocessor = model_factory.create_data_preprocessor(data_storage)
                model_trainer = model_factory.create_model_trainer(data_preprocessor)

                try:
                    # Prepare data
                    dataset = model_trainer.prepare_training_data(
                        pair=training_config['pair'],
                        timeframe=training_config['timeframe'],
                        sequence_length=training_config['sequence_length']
                    )

                    if not dataset:
                        logger.error("Failed to prepare dataset for training")
                        print("✗ Failed to prepare dataset for training")
                        continue

                    # Train model
                    history = model_trainer.train_model(
                        epochs=training_config['epochs'],
                        batch_size=training_config['batch_size']
                    )

                    if history:
                        print(f"✓ Model training completed successfully")
                        print(f"  - Model and training results saved to ModelTraining directory")

                        # Evaluate model
                        metrics = model_trainer.evaluate_model()
                        print(f"✓ Model evaluation:")
                        print(f"  - Direction accuracy: {metrics.get('direction_accuracy', 0):.2f}")
                        print(f"  - Win rate: {metrics.get('win_rate', 0):.2f}")
                        print(f"  - Profit factor: {metrics.get('profit_factor', 0):.2f}")
                    else:
                        print("✗ Model training failed")
                except Exception as e:
                    logger.error(f"Error training model: {e}")
                    print(f"✗ Error training model: {str(e)}")
            else:
                logger.info("Model training configuration cancelled")
                print("Model training configuration cancelled")

        elif train_action == "continue_training":
            logger.info("Continuing training for existing model")
            continue_config = cli.continue_training_menu()

            if continue_config:
                logger.info(f"Selected continue config: {continue_config}")
                print(f"Continuing training for model at {continue_config['model_path']}...")

                # Get components needed for training
                data_storage = processor_factory.create_data_storage()
                data_preprocessor = model_factory.create_data_preprocessor(data_storage)
                model_trainer = model_factory.create_model_trainer(data_preprocessor)

                try:
                    # Load existing model
                    model_trainer.load_model(continue_config['model_path'])

                    # Prepare data
                    dataset = model_trainer.prepare_training_data(
                        pair=continue_config['pair'],
                        timeframe=continue_config['timeframe'],
                        sequence_length=continue_config['sequence_length']
                    )

                    if not dataset:
                        logger.error("Failed to prepare dataset for training")
                        print("✗ Failed to prepare dataset for training")
                        continue

                    # Continue training
                    history = model_trainer.train_model(
                        epochs=continue_config['epochs'],
                        batch_size=continue_config['batch_size'],
                        continued_training=True
                    )

                    if history:
                        print(f"✓ Model training continued successfully")
                        print(f"  - Updated model saved to ModelTraining directory")
                    else:
                        print("✗ Model training failed")
                except Exception as e:
                    logger.error(f"Error continuing model training: {e}")
                    print(f"✗ Error continuing model training: {str(e)}")
            else:
                logger.info("Continue training configuration cancelled")
                print("Continue training configuration cancelled")


def handle_backtest(cli: TradingBotCLI, logger: Logger,
                    processor_factory: ProcessorFactory,
                    model_factory: ModelFactory,
                    strategy_factory: StrategyFactory,
                    backtest_factory: BacktestFactory) -> None:
    """Handle backtesting flow"""
    while True:
        backtest_action = cli.backtest_menu()

        if backtest_action == "back":
            logger.info("Returning to main menu")
            break

        elif backtest_action == "run_backtest":
            logger.info("Running backtest")
            backtest_config = cli.backtest_config_menu()

            if backtest_config:
                logger.info(f"Selected backtest config: {backtest_config}")
                print(f"Running backtest for {backtest_config['pair']} {backtest_config['timeframe']} "
                      f"from {backtest_config['start_date']} to {backtest_config['end_date']}...")

                # Get components for backtesting
                data_storage = processor_factory.create_data_storage()
                model = model_factory.load_model(backtest_config['model_path'])
                signal_generator = strategy_factory.create_signal_generator()
                risk_manager = strategy_factory.create_risk_manager()
                backtest_engine = backtest_factory.create_backtest_engine(
                    data_storage, model, signal_generator, risk_manager
                )

                try:
                    # Run backtest
                    results = backtest_engine.run_backtest(
                        pair=backtest_config['pair'],
                        timeframe=backtest_config['timeframe'],
                        start_date=backtest_config['start_date'],
                        end_date=backtest_config['end_date']
                    )

                    if results:
                        # Generate performance report
                        backtest_engine.generate_performance_report()

                        # Display results summary
                        metrics = results['metrics']
                        print(f"✓ Backtest completed successfully:")
                        print(f"  - Net profit: ${metrics.get('net_profit', 0):.2f}")
                        print(f"  - Return: {metrics.get('return_pct', 0):.2f}%")
                        print(f"  - Win rate: {metrics.get('win_rate', 0) * 100:.2f}%")
                        print(f"  - Profit factor: {metrics.get('profit_factor', 0):.2f}")
                        print(f"  - Max drawdown: {metrics.get('max_drawdown_pct', 0):.2f}%")
                        print(f"  - Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                        print(f"  - Total trades: {metrics.get('total_trades', 0)}")
                        print(f"  - Detailed report saved to BacktestResults directory")
                    else:
                        print("✗ Backtest failed")
                except Exception as e:
                    logger.error(f"Error running backtest: {e}")
                    print(f"✗ Error running backtest: {str(e)}")
            else:
                logger.info("Backtest configuration cancelled")
                print("Backtest configuration cancelled")

        elif backtest_action == "view_results":
            logger.info("Viewing backtest results")
            results_path = cli.select_backtest_results_menu()

            if results_path:
                logger.info(f"Selected results path: {results_path}")
                print(f"Loading backtest results from {results_path}...")

                try:
                    # Load and display results
                    backtest_engine = backtest_factory.create_backtest_engine(None, None, None, None)
                    backtest_engine.load_results(results_path)
                    backtest_engine.display_results_summary()

                    print(f"✓ Results loaded successfully")
                    print(f"  - Use the generated visualizations in {results_path} to analyze performance")
                except Exception as e:
                    logger.error(f"Error loading backtest results: {e}")
                    print(f"✗ Error loading backtest results: {str(e)}")
            else:
                logger.info("Results selection cancelled")
                print("Results selection cancelled")


def handle_live_trading(cli: TradingBotCLI, logger: Logger,
                        processor_factory: ProcessorFactory,
                        model_factory: ModelFactory,
                        strategy_factory: StrategyFactory) -> None:
    """Handle live trading flow"""
    while True:
        trading_action = cli.live_trading_menu()

        if trading_action == "back":
            logger.info("Returning to main menu")
            break

        elif trading_action == "start_trading":
            logger.info("Starting live trading")
            trading_config = cli.trading_config_menu()

            if trading_config:
                logger.info(f"Selected trading config: {trading_config}")
                print(f"Starting live trading for {trading_config['pair']} "
                      f"with model {trading_config['model_path']}...")

                # Create trading components
                model = model_factory.load_model(trading_config['model_path'])
                signal_generator = strategy_factory.create_signal_generator()
                risk_manager = strategy_factory.create_risk_manager()
                trade_executor = strategy_factory.create_trade_executor(risk_manager)

                # Create trading session
                trading_session = strategy_factory.create_trading_session(
                    model, signal_generator, risk_manager, trade_executor
                )

                try:
                    # Start trading session
                    trading_session.start(
                        pair=trading_config['pair'],
                        timeframe=trading_config['timeframe'],
                        update_interval=trading_config['update_interval']
                    )

                    print(f"✓ Trading session started successfully")
                    print(f"  - Monitoring market data and generating signals")
                    print(f"  - Press Ctrl+C to stop the trading session")

                    # This would typically involve a long-running process
                    # that we'd monitor until the user stops it
                    try:
                        trading_session.join()  # Wait for the trading session to complete
                    except KeyboardInterrupt:
                        trading_session.stop()
                        print("Trading session stopped by user")
                except Exception as e:
                    logger.error(f"Error in trading session: {e}")
                    print(f"✗ Error in trading session: {str(e)}")
            else:
                logger.info("Trading configuration cancelled")
                print("Trading configuration cancelled")

        elif trading_action == "monitor_trades":
            logger.info("Monitoring active trades")

            try:
                # Get trading session if available
                trading_session = strategy_factory.get_active_trading_session()

                if trading_session and trading_session.is_running():
                    # Display current trading stats
                    stats = trading_session.get_statistics()

                    print("Current Trading Session Statistics:")
                    print(f"  - Active since: {stats.get('start_time', 'Unknown')}")
                    print(f"  - Symbol: {stats.get('symbol', 'Unknown')}")
                    print(f"  - Open positions: {stats.get('open_positions', 0)}")
                    print(f"  - Completed trades: {stats.get('completed_trades', 0)}")
                    print(f"  - Current P/L: ${stats.get('current_pl', 0):.2f}")

                    # Display recent signals
                    signals = trading_session.get_recent_signals()
                    if signals:
                        print("\nRecent Signals:")
                        for i, signal in enumerate(signals[:5]):  # Show last 5 signals
                            print(f"  {i + 1}. {signal['type'].value} at {signal.get('timestamp', 'Unknown')} "
                                  f"with strength {signal.get('signal_strength', 0):.2f}")
                else:
                    print("No active trading session found")
                    print("Start a trading session first using 'Start Trading'")
            except Exception as e:
                logger.error(f"Error monitoring trades: {e}")
                print(f"✗ Error monitoring trades: {str(e)}")

        elif trading_action == "stop_trading":
            logger.info("Stopping live trading")

            try:
                # Get trading session if available
                trading_session = strategy_factory.get_active_trading_session()

                if trading_session and trading_session.is_running():
                    # Stop the trading session
                    trading_session.stop()

                    print("✓ Trading session stopped successfully")

                    # Display summary of session
                    stats = trading_session.get_statistics()
                    print("\nTrading Session Summary:")
                    print(f"  - Duration: {stats.get('duration', 'Unknown')}")
                    print(f"  - Total trades: {stats.get('total_trades', 0)}")
                    print(f"  - Win rate: {stats.get('win_rate', 0) * 100:.2f}%")
                    print(f"  - Net P/L: ${stats.get('net_pl', 0):.2f}")
                else:
                    print("No active trading session found")
            except Exception as e:
                logger.error(f"Error stopping trading session: {e}")
                print(f"✗ Error stopping trading session: {str(e)}")


def process_datasets(processor, storage, logger, pair, timeframe, dataset_types):
    """Process multiple datasets and save to database"""
    for dataset_type in dataset_types:
        try:
            print(f"Processing {dataset_type} data...")
            X, y = processor.prepare_dataset(pair, timeframe, dataset_type)

            if X.empty:
                logger.warning(f"No data found for {pair} {timeframe} {dataset_type}")
                print(f"✗ No data found for {dataset_type}")
                continue

            # Log information about the processed data
            logger.info(f"Processed {len(X)} rows for {pair} {timeframe} {dataset_type}")
            logger.info(f"Features: {list(X.columns)}")
            if not y.empty:
                logger.info(f"Targets: {list(y.columns)}")

            # Save processed data to database
            print(f"Saving processed data to database...")
            table_name = f"{pair}_{timeframe}_{dataset_type}_processed"
            db_success = storage.save_processed_data(X, y, pair, timeframe, dataset_type)

            if db_success:
                print(f"✓ Successfully saved {len(X)} rows to {table_name}")
                print(f"  - Dataset includes {len(X.columns)} features")
                if not y.empty:
                    print(f"  - Created {len(y.columns)} target variables")
            else:
                print(f"✗ Failed to save data to database")

        except Exception as e:
            logger.error(f"Error processing {dataset_type} data: {e}")
            print(f"✗ Error processing {dataset_type} data: {str(e)}")


def validate_datasets(processor, validator, logger, pair, timeframe, dataset_types):
    """Validate indicators across multiple datasets"""
    for dataset_type in dataset_types:
        try:
            print(f"Validating {dataset_type} data...")

            # Get the raw data and process it
            df = processor.get_data_from_db(pair, timeframe, dataset_type)

            # Log the raw data columns
            logger.info(f"Raw data columns: {list(df.columns)}")

            # Process the data (add indicators)
            processed_df = processor.process_raw_data(df)
            logger.info(f"After processing, columns: {list(processed_df.columns)}")

            # Create features
            feature_df = processor.create_features(processed_df)
            logger.info(f"After feature engineering, columns: {list(feature_df.columns)}")

            if feature_df.empty:
                logger.warning(f"No data found for {pair} {timeframe} {dataset_type}")
                print(f"✗ No data found for {dataset_type}")
                continue

            # Run validation
            print(f"Running indicator validation on {len(feature_df)} rows...")
            results = validator.validate_technical_indicators(feature_df)

            # Display results
            all_valid = all(results.values())

            if all_valid:
                print(f"✓ All indicators validated successfully for {dataset_type} data")
                print("Generating visualizations for visual confirmation...")
                validator.generate_indicator_visualizations(feature_df)
                print("✓ Visualizations generated in ValidationReports/Visualizations")
            else:
                print(f"✗ Validation failed for some indicators in {dataset_type} data:")
                for indicator, is_valid in results.items():
                    status = "✓" if is_valid else "✗"
                    print(f"  {status} {indicator}")

                # Generate visualizations anyway to help diagnose issues
                print("Generating visualizations to help diagnose issues...")
                validator.generate_indicator_visualizations(feature_df)
                print("✓ Visualizations generated in ValidationReports/Visualizations")

                print("\nPlease check ValidationReports for detailed validation report.")

        except Exception as e:
            logger.error(f"Error validating {dataset_type} data: {e}")
            print(f"✗ Error validating {dataset_type} data: {str(e)}")


if __name__ == '__main__':
    container = Container()
    container.wire(modules=[__name__])
    main()