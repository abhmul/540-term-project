import argparse
import logging
import os

import numpy as np
from keras.callbacks import ModelCheckpoint
from pyjet.data import NpDataset, DatasetGenerator
from pyjet.preprocessing.image import ImageDataGenerator

from training import load_model, load_train_setup
from models.model_utils import reset_model
import data_utils as dsb
from utils import safe_open_dir

parser = argparse.ArgumentParser(description='Run the models.')
parser.add_argument('train_id', help='ID of the train configuration')
parser.add_argument('--train', action="store_true", help="Whether to run this script to train a model")
parser.add_argument('--test', action="store_true", help="Whether to run this script to generate submissions")
parser.add_argument('--plot', action="store_true", help="Whether to plot the training loss")
parser.add_argument('--num_completed', type=int, default=0, help="How many completed folds")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

PLOT = False


def create_filenames(train_id):
    model_file = safe_open_dir("../models/") + train_id + ".state"
    log_file = safe_open_dir("../logs/") + train_id + ".txt"
    plot_file = os.path.join(safe_open_dir('../plots/'), 'loss_' + train_id + ".png")
    submission_file = os.path.join(safe_open_dir('../submissions/'), train_id + ".csv")
    return model_file, log_file, plot_file, submission_file


def train_model(model, train_id, train_data, val_data, epochs=10, batch_size=32, plot=False, load_model=False):
    logging.info("Train Data: %s samples" % len(train_data))
    logging.info("Val Data: %s samples" % len(val_data))

    traingen = DatasetGenerator(train_data, batch_size=batch_size, shuffle=True, seed=np.random.randint(2 ** 32))
    valgen = DatasetGenerator(val_data, batch_size=batch_size, shuffle=False, seed=np.random.randint(2 ** 32))

    model_file, log_file, plot_file, _ = create_filenames(train_id)

    # callbacks
    best_model = ModelCheckpoint(model_file, monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)
    callbacks = [best_model]
    # This will plot the losses while training
    if plot:
        raise NotImplementedError("Plotting")

    # Setup the weights
    if load_model:
        logging.info("Loading the model from %s to resume training" % model_file)
        model.load_weights(model_file)
    else:
        logging.info("Resetting model parameters")
        reset_model(model)

    # And finally train
    history = model.fit_generator(traingen, steps_per_epoch=traingen.steps_per_epoch,
                                  epochs=epochs, callbacks=callbacks,
                                  validation_data=valgen,
                                  validation_steps=valgen.steps_per_epoch)

    return model, history


# TODO Add test augmentation to this
def test_model(model, train_id, test_data, batch_size):
    testgen = DatasetGenerator(test_data, batch_size=batch_size, shuffle=False)
    model_file, _, _, _ = create_filenames(train_id)
    # Initialize the model
    model.load_weights(model_file)

    # Get the predictions
    return model.predict_generator(testgen, testgen.steps_per_epoch, verbose=1)


def kfold(dataset, config, train_id, num_completed=0):
    # Initialize the model
    model = load_model(config["model"])
    logging.info("Total Data: %s samples" % len(dataset))
    logging.info("Running %sfold validation" % config["kfold"])
    best_vals = []
    completed = set(range(num_completed))
    for i, (train_data, val_data) in enumerate(dataset.kfold(k=config["kfold"], shuffle=True, seed=np.random.randint(2 ** 32))):
        if i in completed:
            continue
        logging.info("Training Fold%s" % (i + 1))
        train_id = train_id + "_fold%s" % i
        # TODO add load model functinoality
        model, history = train_model(model, train_id,
                                     train_data, val_data,
                                     epochs=config["epochs"], batch_size=config["batch_size"],
                                     plot=PLOT, load_model=False)
        best_vals.append(min(history.history['val_loss']))

    logging.info("Average val loss: %s" % (sum(best_vals) / len(best_vals)))
    return model


def test(dataset, test_img_sizes, config, train_id, model=None):
    if model is None:
        model = load_model(config["model"])

    if not config["kfold"]:
        predictions = test_model(model, train_id, dataset, config["batch_size"])
    else:
        predictions = 0.
        for i in range(config["kfold"]):
            logging.info("Predicting fold %s/%s" % (i+1, config["kfold"]))
            predictions = predictions + test_model(model, train_id + "_fold%s" % i, dataset, config["batch_size"])
        predictions = predictions / config["kfold"]

    _, _, _, submission_file = create_filenames(train_id)
    # Make the submission
    dsb.save_submission(dataset.ids, predictions, test_img_sizes, submission_file,
                        resize_img=config["img_size"] is not None)


if __name__ == "__main__":
    args = parser.parse_args()
    # Load the train_config
    train_config = load_train_setup(args.train_id)
    # Seed the random generator
    np.random.seed(train_config["seed"])
    trained_model = None
    PLOT = args.plot
    if args.train:
        # Load the train data
        train_ids, x_train, y_train = dsb.load_train_data(path_to_train="../input/train/",
                                                          img_size=train_config["img_size"], num_channels=3)
        train_dataset = NpDataset(x=x_train, y=y_train, ids=train_ids)
        # train the models
        if not train_config["kfold"]:
            raise NotImplementedError("Non-kfold training is not implemented")
        trained_model = kfold(train_dataset, train_config, args.train_id, num_completed=args.num_completed)

    if args.test:
        # Load the test data
        test_ids, x_test, sizes_test = dsb.load_test_data(path_to_test="../input/test/",
                                                          img_size=train_config["img_size"], num_channels=3)
        test_dataset = NpDataset(x=x_test, ids=test_ids)
        test(test_dataset, sizes_test, train_config, args.train_id, model=trained_model)



