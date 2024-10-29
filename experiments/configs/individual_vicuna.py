import os

from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    config.model_paths = [
        "/path/to/model/vicuna-7b-v1.3",
        # more models
    ]
    config.tokenizer_paths = [
        "/path/to/model/vicuna-7b-v1.3",
        # more tokenizers
    ]
    return config