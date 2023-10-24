from textClassification.logging import logging
from textClassification.exception import AppException
import sys

from textClassification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textClassification.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from textClassification.pipeline.stage_03_data_preprocess import DataPreprocessTrainingPipeline
#from textClassification.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
#from textClassification.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline


STAGE_NAME = "Data Ingestion stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        AppException(e,sys)
        raise e

STAGE_NAME = "Data Validation stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        AppException(e,sys)
        raise e

STAGE_NAME = "Data PreProcessing stage"
try:
   logging.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_preprocess = DataPreprocessTrainingPipeline()
   data_preprocess.main()
   logging.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        AppException(e,sys)
        raise e

