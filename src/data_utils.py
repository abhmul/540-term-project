import os
import sys
import random
import warnings
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.io import imread
from skimage.transform import resize
from skimage.morphology import label


# Much of this code was based on the kernel https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277

def get_data_ids(path_to_data):
    return next(os.walk(path_to_data))[1]


def load_train_data(path_to_train='../input/train/', img_size=(128, 128), num_channels=3):
    img_height, img_width = img_size
    train_ids = get_data_ids(path_to_train)
    x_train = np.zeros((len(train_ids), img_height, img_width, num_channels), dtype=np.uint8)
    # These are the masks
    y_train = np.zeros((len(train_ids), img_height, img_width, 1), dtype=np.bool)
    logging.info("Loading %s train images" % len(train_ids))
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        # Get the path and read it
        path = path_to_train + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :num_channels]
        # Resize to the input size
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        x_train[n] = img
        mask = np.zeros((img_height, img_width, 1), dtype=np.bool)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = imread(path + '/masks/' + mask_file)
            mask_ = np.expand_dims(resize(mask_, (img_height, img_width), mode='constant',
                                          preserve_range=True), axis=-1)
            # This will combine the masks
            mask = np.maximum(mask, mask_)
        y_train[n] = mask
    return train_ids, x_train, y_train


def load_test_data(path_to_test='../input/test/', img_size=(128, 128), num_channels=3):
    img_height, img_width = img_size
    test_ids = get_data_ids(path_to_test)
    X_test = np.zeros((len(test_ids), img_height, img_width, num_channels), dtype=np.uint8)
    sizes_test = []
    print('Getting and resizing test images ... ')
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = path_to_test + id_
        img = imread(path + '/images/' + id_ + '.png')[:, :, :num_channels]
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (img_height, img_width), mode='constant', preserve_range=True)
        X_test[n] = img


# Run-length encoding stolen from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)


def save_submission(test_ids, preds, save_name, save_path='../submissions/'):
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(tqdm(test_ids)):
        rle = list(prob_to_rles(preds[n]))
        rles += rle
        new_test_ids += ([id_] * len(rle))

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(os.path.join(save_path, save_name + ".csv"), index=False)


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed