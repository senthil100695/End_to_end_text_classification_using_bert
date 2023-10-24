from textClassification.config.configuration import ConfigurationManager
from textClassification.components.prepare_base_model import BaseModel
from textClassification.logging import logging


class BaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        base_model_config = config.get_base_model_config()
        base_model = BaseModel(config=base_model_config)
        base_model.download_base_model()