from textClassification.config.configuration import ConfigurationManager
from textClassification.components.model_trainer_and_evaluation import ModelTrainer
from textClassification.logging import logging


class ModelTrainingEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.convert_text_vector()