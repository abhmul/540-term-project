import json

from models import keras_unet
from models import pytorch_unet_rnn

MODELS = {
    "unet": keras_unet.unet,
    "unet-rnn": pytorch_unet_rnn.UnetRNN
}

CONFIG_JSON = "training/model_configurations.json"
with open(CONFIG_JSON, 'r') as config_json_file:
    CONFIGS = json.load(config_json_file)


def build_model(name, **kwargs):
    return MODELS[name](**kwargs)


def load_model(model_id, **kwargs):
    return build_model(**CONFIGS[model_id], **kwargs)