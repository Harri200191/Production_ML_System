import os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

import ast

load_dotenv()
def get(item, default=None):
    if os.environ.get(item) != None:
        return os.environ.get(item)
    else:
        return default

def get_int(item, default=None):
    return int(get(item, default))

def get_bool(item, default=True):
    value = get(item, default)
    return value.lower() == "true" if isinstance(value, str) else bool(value)

def get_float(item, default=None):
    return float(get(item, default))

def get_list(item, default=None):
    if os.environ.get(item) == None:
        return get(item, default)
    else:
        return ast.literal_eval(get(item, default))

class Config():
    '''Load variables from .env file'''
    DATASET_CONFIGS = {
        "LABELS_TXT_PATH" : get("LABELS_TXT_PATH", "../data/tiny-imagenet-200/words.txt"),
        "LABELS_DICT_JSON_PATH" : get("LABELS_DICT_JSON_PATH", "../assets/labels_dict.json"),
        "IMAGES_DIR" : get("IMAGES_DIR", "../data/tiny-imagenet-200/train"),
    }

    MODEL_CONFIGS = {
        "CLIP_MODEL_NAME" : get("CLIP_MODEL_NAME", "ViT-B/32"),
        "MODEL_CHOICES_JSON": get("MODEL_CHOICES_JSON", "../assets/model_choices.json"),
        "DEVICE" : get("DEVICE", "cuda"),
        "CHROMA_VERSION" : get("CHROMA_VERSION", "clip_v0.0"),
        "SIMILARITY_ALGO" : get("SIMILARITY_ALGO", "l2"), 
        "IMAGE_SIZE" : get_int("IMAGE_SIZE", 64) 
    }