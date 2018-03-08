import json

from models import unet

MODELS = {
    "unet": unet.UNet
}

CONFIG_JSON = "training/model_configurations.json"
with open(CONFIG_JSON, 'r') as config_json_file:
    CONFIGS = json.load(config_json_file)


def build_model(model_name, **kwargs):
    return MODELS[model_name](**kwargs)


def load_model(model_id):
    return build_model(**CONFIGS[model_id])