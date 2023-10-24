from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list


@dataclass(frozen=True)
class DataPreprocessConfig:
    root_dir: Path
    data_path: Path
    indepent_feature: list
    drop_feature: list
    target_feature: str
    preprocess_data_path: Path

@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    model_name: str
    base_model_path: Path

@dataclass(frozen=True)
class ModeltrainerConfig:
    root_dir: Path
    base_model_path: Path
    model_path: Path
    preprocess_data_path: Path 
    targert_feature : str
    input_feature : list