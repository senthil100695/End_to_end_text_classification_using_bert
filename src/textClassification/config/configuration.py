from textClassification.constants import *
from textClassification.utils.common import read_yaml, create_directories
from textClassification.entity import (DataIngestionConfig,
                                       DataValidationConfig,
                                       DataPreprocessConfig,
                                       BaseModelConfig,
                                       ModeltrainerConfig)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES,
        )

        return data_validation_config
    
    def get_data_preprocess_config(self) -> DataPreprocessConfig:
        config = self.config.data_preprocess

        create_directories([config.root_dir])

        data_preprocess_config = DataPreprocessConfig(
            root_dir= config.root_dir,
            data_path =config.data_path,
            indepent_feature= config.indepent_feature,
            drop_feature= config.drop_feature,
            target_feature= config.target_feature,
            preprocess_data_path= config.preprocess_data_path,
        )

        return data_preprocess_config
    
    def get_base_model_config(self) -> BaseModelConfig:
        config = self.config.base_model

        create_directories([config.root_dir])

        base_model_config = BaseModelConfig(
            root_dir= config.root_dir,
            model_name =config.model_name,
            base_model_path= config.base_model_path,
        )

        return base_model_config
    
    def get_model_trainer_config(self) -> ModeltrainerConfig:
        config = self.config.model_trainer

        create_directories([config.root_dir])

        model_trainer_config = ModeltrainerConfig(
            root_dir= config.root_dir,
            base_model_path =config.base_model_path,
            model_path= config.model_path,
            preprocess_data_path = config.preprocess_data_path,
            targert_feature = config.targert_feature,
            input_feature = config.input_feature,
        )

        return model_trainer_config
