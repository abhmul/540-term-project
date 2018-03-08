import argparse
import logging
import os

import numpy as np
from scipy.special import expit
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

import pyjet.backend as J
from pyjet.callbacks import ModelCheckpoint, Plotter, MetricLogger, ReduceLROnPlateau
from pyjet.data import DatasetGenerator, NpDataset

from training import load_model, load_train_setup
import data_utils as dsb
from utils import safe_open_dir

parser = argparse.ArgumentParser(description='Run the models.')
parser.add_argument('train_id', help='ID of the train configuration')
parser.add_argument('--train', action="store_true", help="Whether to run this script to train a model")
parser.add_argument('--test', action="store_true", help="Whether to run this script to generate submissions")
parser.add_argument('--num_completed', type=int, default=0, help="How many completed folds")


def create_filenames(train_id):
    model_file = safe_open_dir("../models/") + train_id + ".state"
    log_file = safe_open_dir("../logs/") + train_id + ".txt"
    plot_file = os.path.join(safe_open_dir('../plots/'), 'loss_' + train_id + ".png")
    submission_file = os.path.join(safe_open_dir('../submissions/'), train_id + ".csv")
    return model_file, log_file, plot_file, submission_file


def train_model(model, train_id, optimizer, train_data, val_data, epochs=10, batch_size=32, plot=False, load_model=False):
    logging.info("Train Data: %s samples" % len(train_data))
    logging.info("Val Data: %s samples" % len(val_data))

    traingen = DatasetGenerator(train_data, batch_size=batch_size, shuffle=True, seed=np.random.randint(2 ** 32))
    valgen = DatasetGenerator(val_data, batch_size=batch_size, shuffle=False, seed=np.random.randint(2 ** 32))

    model_file, log_file, plot_file, _ = create_filenames(train_id)

    # callbacks
    best_model = ModelCheckpoint(model_file, monitor="loss", verbose=1, save_best_only=True)
    log_to_file = MetricLogger(log_file)
    callbacks = [best_model, log_to_file]
    # This will plot the losses while training
    if plot:
        loss_plotter = Plotter(monitor='loss', scale='log', save_to_file=plot_file, block_on_end=False)
        callbacks.append(loss_plotter)

    # Setup the weights
    if load_model:
        logging.info("Loading the model from %s to resume training" % model_file)
        model.load_state(model_file)
    else:
        logging.info("Resetting model parameters")
        model.reset_parameters()

    # And the optimizer
    optimizer = optimizer([param for param in model.parameters() if param.requires_grad])

    loss = binary_cross_entropy_with_logits

    # And finally train
    tr_logs, val_logs = model.fit_generator(traingen, steps_per_epoch=traingen.steps_per_epoch,
                                            epochs=epochs, callbacks=callbacks, optimizer=[optimizer],
                                            loss_fn=loss, validation_generator=valgen,
                                            validation_steps=valgen.steps_per_epoch)

    # Clear the memory associated with models and optimizers
    del optimizer
    del callbacks
    if J.use_cuda:
        torch.cuda.empty_cache()

    return model, tr_logs, val_logs


# TODO Add test augmentation to this
def test_model(model, train_id, test_data, batch_size, augmenter=None):
    testgen = DatasetGenerator(test_data, batch_size=batch_size, shuffle=False)
    model_file, _, _, _ = create_filenames(train_id)
    # Initialize the model
    model.load_state(model_file)

    # Get the predictions
    return model.predict_generator(testgen, testgen.steps_per_epoch, verbose=1)


def kfold(dataset, config, train_id, num_completed=0):
    # Initialize the model
    model = load_model(config["model"])
    logging.info("Total Data: %s samples" % len(dataset))
    logging.info("Running %sfold validation" % config["kfold"])
    best_vals = []

    completed = set(range(num_completed))
    for i, (train_data, val_data) in enumerate(dataset.kfold(k=args.kfold, shuffle=True, seed=np.random.randint(2 ** 32))):
        if i in completed:
            continue
        logging.info("Training Fold%s" % (i + 1))
        train_id = train_id + "_fold%s" % i
        # TODO add load model functinoality
        model, tr_logs, val_logs = train_model(model, train_id, config["optimizer"],
                                               train_data, val_data,
                                               epochs=config["epochs"], batch_size=config["batch_size"],
                                               plot=config["plot"], load_model=False)
        best_vals.append(min(val_logs['loss']))

    logging.info("Average val loss: %s" % (sum(best_vals) / len(best_vals)))
    return model


def test(dataset, test_img_sizes, config, train_id, model=None):
    if model is None:
        model = load_model(config["model"])

    if not config["kfold"]:
        raise NotImplementedError("Non-kfold testing is not implemented")

    predictions = 0.
    for i in range(config["kfold"]):
        logging.info("Predicting fold %s/%s" % (i+1, config["kfold"]))
        predictions = predictions + test_model(model, train_id + "_fold%s" % i, dataset, config["batch_size"])
    predictions = predictions / config["kfold"]

    _, _, _, submission_file = create_filenames(train_id)
    # Make the submission
    dsb.save_submission(dataset.ids, expit(predictions), test_img_sizes, submission_file)


if __name__ == "__main__":
    args = parser.parse_args()
    # Load the train_config
    train_config = load_train_setup(args.train_id)
    trained_model = None
    if args.train:
        # Load the train data
        train_ids, x_train, y_train = dsb.load_train_data(path_to_train="../input/train",
                                                          img_size=train_config["img_size"], num_channels=3)
        train_dataset = NpDataset(x=x_train, y=y_train, ids=train_ids)
        # train the models
        if not train_config["kfold"]:
            raise NotImplementedError("Non-kfold training is not implemented")
        trained_model = kfold(x_train, y_train, train_config)

    if args.test:
        # Load the test data
        test_ids, x_test, sizes_test = dsb.load_test_data(path_to_test="../input/test",
                                                          img_size=train_config["img_size"], num_channels=3)
        test_dataset = NpDataset(x=x_test, ids=test_ids)
        test(test_dataset, sizes_test, train_config, args.train_id, model=trained_model)



