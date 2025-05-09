from Utilities.ConfigurationUtils import Config
from Utilities.LoggingUtils import Logger
from Processing.DataProcessor import DataProcessor
from Processing.DataStorage import DataStorage
from Processing.IndicatorValidator import IndicatorValidator
from Analisys.FeatureAnalyzer import FeatureAnalyzer


class ProcessorFactory:
    def __init__(self, config: Config, logger: Logger):
        self.config = config
        self.logger = logger

    def create_data_processor(self) -> DataProcessor:
        return DataProcessor(self.config, self.logger)

    def create_data_storage(self) -> DataStorage:
        return DataStorage(self.config, self.logger)

    def create_indicator_validator(self) -> IndicatorValidator:
        return IndicatorValidator(self.config, self.logger)

    def create_feature_analyzer(self) -> FeatureAnalyzer:
        storage = self.create_data_storage()
        return FeatureAnalyzer(self.config, self.logger, storage)