from dependency_injector.wiring import inject, Provide
from Utilities.Container import Container
from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Processing.ProcessorFactory import ProcessorFactory
from Analisys.FeatureAnalyzer import FeatureAnalyzer
from Processing.DataStorage import DataStorage


@inject
def main(
        config: Config = Provide[Container.config],
        logger: Logger = Provide[Container.logger],
        processor_factory: ProcessorFactory = Provide[Container.processor_factory]
) -> None:
    logger.info('Feature Analysis started')

    # Create instances
    data_storage = processor_factory.create_data_storage()
    feature_analyzer = FeatureAnalyzer(config, logger, data_storage)

    # Run feature analysis for different targets
    try:
        # Default target (1-period ahead price prediction)
        default_target = "future_price_1"
        logger.info(f"Running feature analysis for target: {default_target}")
        selected_features = feature_analyzer.run_complete_analysis(
            pair="XAUUSD",
            timeframe="H1",
            dataset_type="training",
            target_col=default_target
        )

        if selected_features:
            logger.info(f"Analysis complete! Selected {len(selected_features)} optimal features")
            print(f"✓ Feature analysis complete for {default_target}")
            print(f"  - Selected {len(selected_features)} optimal features")
            print(f"  - Results saved to FeatureAnalysis directory")
        else:
            logger.error("Feature analysis failed to return selected features")
            print("✗ Feature analysis failed")

        # Optional: analyze directional prediction targets too
        direction_target = "direction_1"
        if direction_target in data_storage.load_processed_data("XAUUSD", "H1", "training")[1].columns:
            logger.info(f"Running feature analysis for target: {direction_target}")
            direction_features = feature_analyzer.run_complete_analysis(
                pair="XAUUSD",
                timeframe="H1",
                dataset_type="training",
                target_col=direction_target
            )

            if direction_features:
                logger.info(
                    f"Analysis complete! Selected {len(direction_features)} optimal features for direction prediction")
                print(f"✓ Feature analysis complete for {direction_target}")
                print(f"  - Selected {len(direction_features)} optimal features")

    except Exception as e:
        logger.error(f"Error in feature analysis: {e}")
        print(f"✗ Error in feature analysis: {str(e)}")

    logger.info('Feature Analysis finished')


if __name__ == '__main__':
    container = Container()
    container.wire(modules=[__name__])
    main()