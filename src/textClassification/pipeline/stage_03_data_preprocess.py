from textClassification.config.configuration import ConfigurationManager
from textClassification.components.data_preprocess import DataPreprocessor
from textClassification.logging import logging


class DataPreprocessTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_preprocess_config = config.get_data_preprocess_config()
        data_preprocess = DataPreprocessor(config=data_preprocess_config)
        data_preprocess.convert()