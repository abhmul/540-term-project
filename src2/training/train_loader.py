import json
from functools import partial

import torch.optim as optim

OPTIMIZERS = {
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
    "adam": optim.Adam
}

CONFIG_JSON = "training/training_configurations.json"
with open(CONFIG_JSON, 'r') as config_json_file:
    CONFIGS = json.load(config_json_file)


def build_optimizer(name, **kwargs):
    return partial(OPTIMIZERS[name], **kwargs)


def load_train_setup(train_id):
    config = CONFIGS[train_id]
    model = config["model_id"]
    optimizer = build_optimizer(**config["optimizer"])

    batch_size = config["batch_size"]
    epochs = config["epochs"]

    seed = config["seed"]

    return {"model": model,
            "optimizer": optimizer,
            "batch_size": batch_size,
            "epochs": epochs,
            "seed": seed,
            "kfold": config.get("kfold"),
            "img_size": config.get("img_size")}
