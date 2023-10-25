import os
import sys
from box.exceptions import BoxValueError
import yaml
from textClassification.logging import logging
from textClassification.exception import AppException
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any

import json

@ensure_annotations
def read_yaml(path_to_yaml : Path) -> ConfigBox:

    try:
        with open(path_to_yaml)as yaml_file:
            content = yaml.safe_load(yaml_file)
            logging.info(f'yaml file : {path_to_yaml} loaded sucessfully')
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError('yaml file is empty')
    except Exception as e:
        raise AppException(e,sys)
    
@ensure_annotations
def create_directories(path_to_directories : list, verbose = True):
    for path in path_to_directories:
        os.makedirs(path,exist_ok=True)
        if verbose:
            logging.info(f'created directory at : {path}')

@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path)/1024)
    return f'~ {size_in_kb} KB'

def save_json(data,path):
        #path = os.path.join(path,'article_type.json')
        with open(path,'w') as out:
            json.dump(data,out)