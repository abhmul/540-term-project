import json
# from functools import partial
#
# import torch.optim as optim
#
# OPTIMIZERS = {
#     "sgd": optim.SGD,
#     "rmsprop": optim.RMSprop,
#     "adam": optim.Adam
# }

CONFIG_JSON = "training/training_configurations.json"
with open(CONFIG_JSON, 'r') as config_json_file:
    CONFIGS = json.load(config_json_file)


# def build_optimizer(name, **kwargs):
#     return partial(OPTIMIZERS[name], **kwargs)


def load_train_setup(train_id):
    config = CONFIGS[train_id]

    model = config["model_id"]

    batch_size = config["batch_size"]
    epochs = config["epochs"]

    seed = config["seed"]

    new_config = {"model": model,
                  "batch_size": batch_size,
                  "epochs": epochs,
                  "seed": seed,
                  "kfold": config.get("kfold"),
                  "split": config.get("split"),
                  "img_size": config.get("img_size"),
                  "augment": config.get("augment"),
                  "augment_times": config.get("augment_times"),
                  "img_mode": config.setdefault("img_mode", "rgb"),
                  "segments": config.setdefault("segments", False)}

    return new_config
