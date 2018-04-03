import argparse
import logging
import os
from pprint import pprint

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.models import load_model as reload_model_checkpoint
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
parser.add_argument('--reload_model', action='store_true', help="Continues training from a saved model")
parser.add_argument('--cutoff', type=float, default=0, help="Cutoff to use for producing submission")
parser.add_argument('--use_iou', action='store_true', help="creates test predictions with iou checkpointed model.")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

PLOT = False
CHECKPOINT = "../models/checkpoint.state"


def create_filenames(train_id):
    model_file = os.path.join(safe_open_dir("../models/"), train_id + ".state")
    model_iou_file = os.path.join(safe_open_dir("../models/"), train_id + "_iou" + ".state")
    log_file = os.path.join(safe_open_dir("../logs/"), train_id + ".txt")
    plot_file = os.path.join(safe_open_dir('../plots/'), 'loss_' + train_id + ".png")
    submission_file = os.path.join(safe_open_dir('../submissions/'), train_id + ".csv")
    return model_file, model_iou_file, log_file, plot_file, submission_file


def train_model(model, train_id, train_data, val_data, epochs=10, batch_size=32, plot=False, reload_model=False,
                augment=None):
    logging.info("Train Data: %s samples" % len(train_data))
    logging.info("Val Data: %s samples" % len(val_data))

    traingen = DatasetGenerator(train_data, batch_size=batch_size, shuffle=True, seed=np.random.randint(2 ** 32))
    valgen = DatasetGenerator(val_data, batch_size=batch_size, shuffle=False, seed=np.random.randint(2 ** 32))

    # Set up the augmentation
    if augment is not None:
        logging.info("Using augmentation with settings:")
        pprint(augment)

        traingen = ImageDataGenerator(traingen, labels=True, augment_masks=True, **augment)

        # For debugging purposes
        # import matplotlib.pyplot as plt
        # for batch_x, batch_y in traingen:
        #     plt.imshow(batch_x[0])
        #     plt.show()
        #     plt.imshow(batch_y[0][..., 0], cmap="gray")
        #     plt.show()
        #     plt.imshow(batch_x[6])
        #     plt.show()
        #     plt.imshow(batch_y[6][..., 0], cmap="gray")
        #     plt.show()
        #     raise ValueError()

    model_file, model_iou_file, log_file, plot_file, _ = create_filenames(train_id)

    # callbacks
    best_model = ModelCheckpoint(model_file, monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)
    best_model_iou = ModelCheckpoint(model_iou_file, monitor="val_mean_iou", verbose=1, save_best_only=True,
                                     save_weights_only=True, mode='max')
    # This one will allow us to resume training
    checkpoint = ModelCheckpoint(CHECKPOINT, verbose=1)
    callbacks = [best_model, best_model_iou, checkpoint]
    # This will plot the losses while training
    if plot:
        callbacks.append(Plotter("loss", scale="log", save_to_file=plot_file, block_on_end=False))
        callbacks.append(Plotter("mean_iou", scale="linear", save_to_file=plot_file, block_on_end=False))

    # Setup the weights
    if reload_model:
        logging.info("Loading the model from %s to resume training" % model_file)
        model = reload_model_checkpoint(CHECKPOINT)
    else:
        logging.info("Resetting model parameters")
        reset_model(model)

    # And finally train
    history = model.fit_generator(traingen, steps_per_epoch=traingen.steps_per_epoch,
                                  epochs=epochs, callbacks=callbacks,
                                  validation_data=valgen,
                                  validation_steps=valgen.steps_per_epoch)

    return model, history


def test_model(model, train_id, test_data, batch_size, augment=None, augment_times=None, use_iou=False):
    testgen = DatasetGenerator(test_data, batch_size=batch_size, shuffle=False)
    model_file, model_iou_file, _, _, _ = create_filenames(train_id)
    if use_iou:
        logging.info("Creating test predictions with model checkpointed with mean iou.")
        model_file = model_iou_file
    # Initialize the model
    model.load_weights(model_file)

    # Get the predictions
    if augment is None or augment_times is None:
        return model.predict_generator(testgen, testgen.steps_per_epoch, verbose=1)
    else:
        logging.info("Using test augmentation with settings:")
        pprint(augment)
        logging.info("Evaluating %s times" % augment_times)
        testgen = ImageDataGenerator(testgen, labels=False, augment_masks=False, **augment)
        preds = 0
        for i in range(augment_times):
            preds += model.predict_generator(testgen, testgen.steps_per_epoch, verbose=1)
        return preds / augment_times


def train(dataset, config, train_id, reload_model=False):
    # Initialize the model
    model = load_model(config["model"], img_size=config["img_size"], max_val=config["max_val"],
                       num_channels=config["num_channels"])
    logging.info("Total Data: %s samples" % len(dataset))
    logging.info("Splitting data into %s validation" % config["split"])

    train_data, val_data = dataset.validation_split(split=config["split"], shuffle=True, seed=np.random.randint(2 ** 32))

    model, history = train_model(model, train_id,
                                 train_data, val_data,
                                 epochs=config["epochs"], batch_size=config["batch_size"],
                                 plot=PLOT, augment=config["augment"],
                                 reload_model=reload_model)
    best_val = min(history.history['val_loss'])
    best_mean_iou = min(history.history['val_mean_iou'])

    logging.info("Best val loss: %s" % best_val)
    logging.info("Best val mean iou: %s" % best_mean_iou)
    return model


def kfold(dataset, config, train_id, num_completed=0, reload_model=False):
    # Initialize the model
    model = load_model(config["model"], img_size=config["img_size"], max_val=config["max_val"],
                       num_channels=config["num_channels"])
    logging.info("Total Data: %s samples" % len(dataset))
    logging.info("Running %sfold validation" % config["kfold"])
    best_vals = []
    best_mean_ious = []
    completed = set(range(num_completed))
    for i, (train_data, val_data) in enumerate(dataset.kfold(k=config["kfold"], shuffle=True, seed=np.random.randint(2 ** 32))):
        if i in completed:
            continue
        logging.info("Training Fold%s" % (i + 1))
        fold_train_id = train_id + "_fold%s" % i
        model, history = train_model(model, fold_train_id,
                                     train_data, val_data,
                                     epochs=config["epochs"], batch_size=config["batch_size"],
                                     plot=PLOT, augment=config["augment"],
                                     reload_model=reload_model)
        # Set to false for the next time
        reload_model = False
        best_vals.append(min(history.history['val_loss']))
        best_mean_ious.append(min(history.history['val_mean_iou']))

    logging.info("Average val loss: %s" % (sum(best_vals) / len(best_vals)))
    logging.info("Average val mean iou: %s" % (sum(best_mean_ious) / len(best_mean_ious)))
    return model


def test(dataset, test_img_sizes, config, train_id, model=None, use_iou=False):
    if model is None:
        model = load_model(config["model"], img_size=config["img_size"], max_val=config["max_val"],
                           num_channels=config["num_channels"])

    if not config["kfold"]:
        predictions = test_model(model, train_id, dataset, config["batch_size"], augment=config["augment"],
                                 augment_times=config["augment_times"])
    else:
        predictions = 0.
        for i in range(config["kfold"]):
            logging.info("Predicting fold %s/%s" % (i+1, config["kfold"]))
            predictions = predictions + test_model(model, train_id + "_fold%s" % i, dataset, config["batch_size"],
                                                   augment=config["augment"], augment_times=config["augment_times"],
                                                   use_iou=use_iou)
        predictions = predictions / config["kfold"]

    _, _, _, _, submission_file = create_filenames(train_id)
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

    # Set up plotting
    PLOT = args.plot
    if PLOT:
        from callbacks import plotter
        plotter.plt.switch_backend('TkAgg')
        Plotter = plotter.Plotter
    else:
        Plotter = None

    # Image loading and other setup values
    train_config["num_channels"] = 1 if train_config["img_mode"] == "gray" else 3
    train_config["max_val"] = 235. if train_config["img_mode"] == "ycbcr" else 255.

    logging.info("Training configuration is:")
    pprint(train_config)

    if args.train:
        # Load the train data
        train_ids, x_train, y_train = dsb.load_train_data(path_to_train="../input/train/",
                                                          img_size=train_config["img_size"],
                                                          num_channels=train_config["num_channels"],
                                                          mode=train_config["img_mode"])
        train_dataset = NpDataset(x=x_train, y=y_train, ids=train_ids)
        # train the models
        if not train_config["kfold"]:
            trained_model = train(train_dataset, train_config, args.train_id, reload_model=args.reload_model)
        else:
            trained_model = kfold(train_dataset, train_config, args.train_id, num_completed=args.num_completed,
                                  reload_model=args.reload_model)

    if args.test:
        # Load the test data
        test_ids, x_test, sizes_test = dsb.load_test_data(path_to_test="../input/test/",
                                                          img_size=train_config["img_size"],
                                                          num_channels=train_config["num_channels"],
                                                          mode=train_config["img_mode"])
        test_dataset = NpDataset(x=x_test, ids=test_ids)
        test(test_dataset, sizes_test, train_config, args.train_id, model=trained_model)



