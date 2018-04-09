import argparse
import logging
import os
from pprint import pprint
import pickle as pkl

import numpy as np
from sklearn.metrics import jaccard_similarity_score
from keras.callbacks import ModelCheckpoint
from keras.models import load_model as reload_model_checkpoint
import keras.backend as K
from pyjet.data import NpDataset, DatasetGenerator
from pyjet.preprocessing.image import ImageDataGenerator
import pyjet.backend as J

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
parser.add_argument('--initial_epoch', type=int, default=0, help="Continues training from specified epoch")
parser.add_argument('--cutoff', type=float, default=0.5, help="Cutoff to use for producing submission")
parser.add_argument('--use_iou', action='store_true', help="creates test predictions with iou checkpointed model.")
parser.add_argument('--test_debug', action='store_true', help="debugs the test output.")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

PLOT = False
CHECKPOINT = "../models/checkpoint.state"


def limit_mem():
    cfg = K.tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    K.set_session(K.tf.Session(config=cfg))


def clear_mem():
    logging.info("Clearing keras model and tf session.")
    sess = K.get_session()
    sess.close()
    limit_mem()
    return


def create_filenames(train_id):
    model_file = os.path.join(safe_open_dir("../models/"), train_id + ".state")
    model_iou_file = os.path.join(safe_open_dir("../models/"), train_id + "_iou" + ".state")
    log_file = os.path.join(safe_open_dir("../logs/"), train_id + ".txt")
    plot_file = os.path.join(safe_open_dir('../plots/'), 'loss_' + train_id + ".png")
    submission_file = os.path.join(safe_open_dir('../submissions/'), train_id + ".csv")
    cache_file = os.path.join(safe_open_dir('../caches/'), train_id + ".pkl")
    return model_file, model_iou_file, log_file, plot_file, submission_file, cache_file


def save_cache(cache, cache_filename):
    with open(cache_filename, "wb") as cache_file:
        logging.info("Saving cache to %s" % cache_filename)
        pkl.dump(cache, cache_file)


def build_cache(train_id, model=None, config=None, val_data: NpDataset=None):
    # First try to load the cache
    model_file, model_iou_file, log_file, plot_file, submission_file, cache_file = create_filenames(train_id)
    try:
        with open(cache_file, "rb") as cache_file_obj:
            cache = pkl.load(cache_file_obj)
        logging.info("Loaded cache from %s" % cache_file)
    except FileNotFoundError:
        logging.info("Could not load cache, constructing")
        orig_output_labels = val_data.output_labels
        val_data.output_labels = False
        cache = {}
        cache["preds"] = test_model(model, train_id, val_data, batch_size=config["batch_size"])
        cache["iou_preds"] = test_model(model, train_id, val_data, batch_size=config["batch_size"], use_iou=True)
        cache["labels"] = val_data.y
        save_cache(cache, cache_file)
        val_data.output_labels = orig_output_labels
    return cache


def train_model(model, train_id, train_data, val_data, epochs=10, batch_size=32, plot=False, reload_model=False,
                augment=None, initial_epoch=0):
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

    model_file, model_iou_file, log_file, plot_file, _, cache_file, *other_filenames = create_filenames(train_id)

    # callbacks
    best_model = ModelCheckpoint(model_file, monitor="val_loss", verbose=1, save_best_only=True, save_weights_only=True)
    best_model_iou = ModelCheckpoint(model_iou_file, monitor="val_mean_iou", verbose=1, save_best_only=True,
                                     save_weights_only=True, mode='max')
    # This one will allow us to resume training
    # checkpoint = ModelCheckpoint(CHECKPOINT, verbose=1, save_weights_only=False)
    callbacks = [best_model, best_model_iou] #, checkpoint]
    # This will plot the losses while training
    if plot:
        callbacks.append(Plotter("loss", scale="log", save_to_file=plot_file, block_on_end=False))
        callbacks.append(Plotter("mean_iou", scale="linear", save_to_file=plot_file, block_on_end=False))

    # Setup the weights
    if reload_model:
        raise NotImplementedError("Reloading models is nor working yet")
        logging.info("Loading the model from %s to resume training" % model_file)
        loss = model.loss
        metrics = model.metrics
        model = None
        clear_mem()
        model = reload_model_checkpoint(CHECKPOINT,
                                        custom_objects={custom.__name__: custom for custom in [loss] + metrics})

    # And finally train
    history = model.fit_generator(traingen, steps_per_epoch=traingen.steps_per_epoch,
                                  epochs=epochs, callbacks=callbacks,
                                  validation_data=valgen,
                                  validation_steps=valgen.steps_per_epoch, initial_epoch=initial_epoch)

    # Get the validation weights
    logging.info("Calculating validation predictions")
    cache = {}
    model.load_weights(model_file)
    cache["preds"] = model.predict_generator(valgen, steps=valgen.steps_per_epoch)
    model.load_weights(model_iou_file)
    cache["iou_preds"] = model.predict_generator(valgen, steps=valgen.steps_per_epoch)
    cache["labels"] = val_data.y
    cache["history"] = history.history

    # Save the cache
    save_cache(cache, cache_file)

    return model, history, cache


def test_model(model, train_id, test_data, batch_size, augment=None, augment_times=None, use_iou=False):
    testgen = DatasetGenerator(test_data, batch_size=batch_size, shuffle=False)
    model_file, model_iou_file, *args = create_filenames(train_id)
    if use_iou:
        logging.info("Creating test predictions with model checkpointed with mean iou.")
        model_file = model_iou_file
    # Initialize the model
    model.load_weights(model_file)

    # Get the predictions
    if augment is None or augment_times == 0:
        return model.predict_generator(testgen, testgen.steps_per_epoch, verbose=1)
    else:
        logging.info("Using test augmentation with settings:")
        pprint(augment)
        logging.info("Evaluating %s times" % augment_times)
        testgen = ImageDataGenerator(testgen, labels=False, augment_masks=False, save_inverses=True, **augment)
        preds = 0
        averaging_array = None
        for i in range(augment_times):
            # Need to use max_q_size as 1 to not sample too much from testgen
            aug_preds = model.predict_generator(testgen, testgen.steps_per_epoch, verbose=1)
            orig_preds = testgen.invert_images(aug_preds)
            # Use this to not drive padded regions down
            if averaging_array is None:
                averaging_array = np.zeros(orig_preds.shape)
            averaging_array += (orig_preds > 0.005).astype(np.float)
            preds += orig_preds
            assert preds.shape[1:] == (256, 256, 1), "Preds shape is incorrect {}".format(preds.shape)
        # Set everywhere where the averaging constant is 0 to 1, so no divide by zero
        averaging_array[averaging_array == 0] = 1.
        return preds / averaging_array


def cross_validate_cutoff(preds, labels, train_id):
    cutoffs = np.linspace(0., 1., 100)
    scores = []
    for cutoff in cutoffs:
        thresholded = (preds > cutoff).squeeze()
        assert thresholded.ndim == 3
        labels = labels.reshape(thresholded.shape)
        scores.append(np.average([jaccard_similarity_score(l, t) for l, t in zip(labels, thresholded)]))
    scores = np.asarray(scores, dtype=np.float)
    if PLOT:
        import matplotlib.pyplot as plt
        plt.plot(cutoffs, scores)
        plt.savefig("../plots/cutoffs_" + train_id + ".png")

    # Maximize the score
    cutoff_ind = np.argmax(scores)
    logging.info("Best score achieved: %s" % scores[cutoff_ind])
    return cutoffs[cutoff_ind]


def train(dataset, config, train_id, reload_model=False, initial_epoch=0):
    # Initialize the model
    clear_mem()
    model = load_model(config["model"], img_size=config["img_size"], max_val=config["max_val"],
                       num_channels=config["num_channels"])
    logging.info("Total Data: %s samples" % len(dataset))
    logging.info("Splitting data into %s validation" % config["split"])

    train_data, val_data = dataset.validation_split(split=config["split"], shuffle=True, seed=np.random.randint(2 ** 32))

    model, history, cache = train_model(model, train_id,
                                        train_data, val_data,
                                        epochs=config["epochs"], batch_size=config["batch_size"],
                                        plot=PLOT, augment=config["augment"],
                                        reload_model=reload_model, initial_epoch=initial_epoch)
    best_val = min(history.history['val_loss'])
    best_mean_iou = min(history.history['val_mean_iou'])

    logging.info("Best val loss: %s" % best_val)
    logging.info("Best val mean iou: %s" % best_mean_iou)

    # Cross validate cutoffs
    best_cutoff = cross_validate_cutoff(cache["preds"], cache["labels"], train_id)
    best_iou_cutoff = cross_validate_cutoff(cache["iou_preds"], cache["labels"], train_id + "_iou")

    logging.info("Best cutoff for loss checkpointed model: %s" % best_cutoff)
    logging.info("Best cutoff for iou checkpointed model: %s" % best_iou_cutoff)

    return model, {"cutoff": best_cutoff, "iou_cutoff": best_iou_cutoff}


def kfold(dataset, config, train_id, num_completed=0, reload_model=False, initial_epoch=0):
    # Initialize the model
    clear_mem()
    model = load_model(config["model"], img_size=config["img_size"], max_val=config["max_val"],
                       num_channels=config["num_channels"])
    logging.info("Total Data: %s samples" % len(dataset))
    logging.info("Running %sfold validation" % config["kfold"])
    best_vals = []
    best_mean_ious = []
    completed = set(range(num_completed))
    caches = []
    for i, (train_data, val_data) in enumerate(dataset.kfold(k=config["kfold"], shuffle=True, seed=np.random.randint(2 ** 32))):
        logging.info("Training Fold%s" % (i + 1))
        fold_train_id = train_id + "_fold%s" % i
        if i in completed:
            # Build the cache
            caches.append(build_cache(fold_train_id, model, config, val_data))
            continue
        model, history, cache = train_model(model, fold_train_id,
                                            train_data, val_data,
                                            epochs=config["epochs"], batch_size=config["batch_size"],
                                            plot=PLOT, augment=config["augment"],
                                            reload_model=reload_model, initial_epoch=initial_epoch)
        # Set to false for the next time
        reload_model = False
        initial_epoch = 0
        best_vals.append(min(history.history['val_loss']))
        best_mean_ious.append(min(history.history['val_mean_iou']))
        caches.append(cache)

        # Reset the model
        model = None
        clear_mem()
        model = load_model(config["model"], img_size=config["img_size"], max_val=config["max_val"],
                           num_channels=config["num_channels"])

    if len(best_vals) != 0:
        logging.info("Average val loss: %s" % (sum(best_vals) / len(best_vals)))
        logging.info("Average val mean iou: %s" % (sum(best_mean_ious) / len(best_mean_ious)))

    cache = {"preds": np.concatenate([c["preds"] for c in caches], axis=0),
             "iou_preds": np.concatenate([c["iou_preds"] for c in caches], axis=0),
             "labels": np.concatenate([c["labels"] for c in caches], axis=0), }

    best_cutoff = cross_validate_cutoff(cache["preds"], cache["labels"], train_id)
    best_iou_cutoff = cross_validate_cutoff(cache["iou_preds"], cache["labels"], train_id + "_iou")

    logging.info("Best cutoff for loss checkpointed model: %s" % best_cutoff)
    logging.info("Best cutoff for iou checkpointed model: %s" % best_iou_cutoff)

    return model, {"cutoff": best_cutoff, "iou_cutoff": best_iou_cutoff}


def test(dataset, test_img_sizes, config, train_id, model=None, use_iou=False, cutoff=0.5,
         test_debug=False):
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

    if test_debug:
        try:
            import matplotlib.pyplot as plt
            for i, pred in enumerate(predictions.squeeze()):
                plt.imshow(pred, cmap="gray")
                plt.title(dataset.ids[i])
                plt.show()
        except KeyboardInterrupt:
            pass

    _, _, _, _, submission_file, *args = create_filenames(train_id)
    # Make the submission
    logging.info("Using cutoff %s" % cutoff)
    dsb.save_submission(dataset.ids, predictions, test_img_sizes, submission_file,
                        resize_img=config["img_size"] is not None, cutoff=cutoff)


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
    train_cache = None


    logging.info("Training configuration is:")
    pprint(train_config)

    if args.train:
        # Load the train data
        train_ids, x_train, y_train = dsb.load_train_data(path_to_train="../input/train/",
                                                          img_size=train_config["img_size"],
                                                          num_channels=train_config["num_channels"],
                                                          mode=train_config["img_mode"],
                                                          return_segments=train_config["segments"])
        if train_config["segments"]:
            train_dataset = dsb.MaskSegmentDataset(x=x_train, y=y_train, ids=train_ids)
        else:
            train_dataset = NpDataset(x=x_train, y=y_train, ids=train_ids)
        # train the models
        if not train_config["kfold"]:
            trained_model, train_cache = train(train_dataset, train_config, args.train_id,
                                               reload_model=args.reload_model, initial_epoch=args.initial_epoch)
        else:
            trained_model, train_cache = kfold(train_dataset, train_config, args.train_id, num_completed=args.num_completed,
                                               reload_model=args.reload_model, initial_epoch=args.initial_epoch)

    if args.test:
        # Load the test data
        test_ids, x_test, sizes_test = dsb.load_test_data(path_to_test="../input/test/",
                                                          img_size=train_config["img_size"],
                                                          num_channels=train_config["num_channels"],
                                                          mode=train_config["img_mode"])
        if train_config["segments"]:
            test_dataset = dsb.MaskSegmentDataset(x=x_test, ids=test_ids)
        else:
            test_dataset = NpDataset(x=x_test, ids=test_ids)
        if train_cache is None:
            cutoff = args.cutoff
        else:
            cutoff = train_cache["iou_cutoff"] if args.use_iou else train_cache["cutoff"]

        test(test_dataset, sizes_test, train_config, args.train_id, model=trained_model, use_iou=args.use_iou,
             cutoff=cutoff, test_debug=args.test_debug)



