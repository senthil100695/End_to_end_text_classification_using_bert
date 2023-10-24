from sentence_transformers import SentenceTransformer, util
import os
from textClassification.entity import BaseModelConfig


class BaseModel:
    def __init__(self, config: BaseModelConfig):
        self.config = config

    def download_model(self,model_name,base_model_path):
        model = SentenceTransformer(model_name)
        model.save(base_model_path)

    def download_base_model(self):
        model_name = self.config.model_name
        base_model_path = self.config.base_model_path
        self.download_model(model_name,base_model_path)


