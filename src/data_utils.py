import os
import sys
import random
import warnings
import logging

import numpy as np
import pandas as pd

from tqdm import tqdm
from skimage.io import imread
from skimage.color import rgb2ycbcr, rgb2gray
from skimage.transform import resize
from skimage.morphology import label


# Much of this code was based on the kernel https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277

def get_data_ids(path_to_data):
    return next(os.walk(path_to_data))[1]


def normalize(img, max_val=255):
    mean = np.average(img)
    if mean > (max_val / 2):
        return max_val - img
    return img


def load_img(path_to_img, img_size=None, num_channels=3, mode="ycbcr", normalize_img=True):
    img = imread(path_to_img)[:, :, :num_channels]
    if normalize_img:
        img = normalize(img)
    if mode == "rgb":
        pass
    elif mode == "ycbcr":
        img = rgb2ycbcr(img)
    elif mode == "gray":
        assert (num_channels == 1)
        img = rgb2gray(img)[:, :, np.newaxis]
    else:
        raise NotImplementedError("Image mode %s" % mode)


    orig_img_shape = img.shape[:2]
    # Resize to the input size
    if img_size is not None:
        img = resize(img, img_size, mode='constant', preserve_range=True).astype(np.uint8)
    return img, orig_img_shape


def load_mask(path_to_masks, img_size=None):
    if img_size is None:
        mask = None
    else:
        mask = np.zeros(img_size, dtype=np.bool)[..., np.newaxis]
    mask_folder = os.path.join(path_to_masks, 'masks/')
    for mask_file in next(os.walk(mask_folder))[2]:
        mask_ = imread(os.path.join(mask_folder,  mask_file))
        # Initialize mask if we haven't yet
        if mask is None:
            mask = np.zeros(mask_.shape, dtype=np.bool)[..., np.newaxis]
        if img_size is not None:
            mask_ = resize(mask_, img_size, mode='constant', preserve_range=True)
        mask_ = np.expand_dims(mask_, axis=-1)
        # This will combine the masks
        mask = np.maximum(mask, mask_).astype(np.bool)
    return mask


def load_train_data(path_to_train='../input/train/', img_size=None, num_channels=3, mode="ycbcr"):
    train_ids = get_data_ids(path_to_train)
    x_train = [None for _ in train_ids]
    y_train = [None for _ in train_ids]

    logging.info("Loading %s train images with mode %s" % (len(train_ids), mode))
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        # Get the path and read it
        path = os.path.join(path_to_train, id_)
        path_to_img = os.path.join(path, 'images/', id_ + '.png')
        x_train[n], _ = load_img(path_to_img, img_size=img_size, num_channels=num_channels, mode=mode)
        y_train[n] = load_mask(path, img_size=img_size)
        assert x_train[n].shape[:2] == y_train[n].shape[:2]
    # If we have a fixed image size, cast it to numpy
    if img_size is not None:
        x_train = np.stack(x_train)
        y_train = np.stack(y_train)
    return np.array(train_ids), np.array(x_train), np.array(y_train)


def load_test_data(path_to_test='../input/test/', img_size=None, num_channels=3, mode="ycbcr"):
    test_ids = get_data_ids(path_to_test)
    x_test = [None for _ in test_ids]
    sizes_test = [None for _ in test_ids]
    logging.info("Loading %s test images with mode %s" % (len(test_ids), mode))
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = os.path.join(path_to_test, id_)
        path_to_img = os.path.join(path, 'images/', id_ + '.png')
        x_test[n], sizes_test[n] = load_img(path_to_img, img_size=img_size, num_channels=num_channels, mode=mode)

    assert all(i is not None for i in x_test)
    assert all(i is not None for i in sizes_test)
    return np.array(test_ids), np.array(x_test), np.array(sizes_test)


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


def save_submission(test_ids, preds, sizes_test, save_name, resize_img=False, cutoff=0.5):
    new_test_ids = []
    rles = []
    for n, id_ in enumerate(tqdm(test_ids)):
        orig_height, orig_width = sizes_test[n]
        if resize_img:
            pred = resize(preds[n], (orig_height, orig_width), preserve_range=True)
        else:
            pred = preds[n][:orig_height, :orig_width]
        rle = list(prob_to_rles(pred, cutoff=cutoff))
        rles += rle
        new_test_ids += ([id_] * len(rle))

    # Create submission DataFrame
    sub = pd.DataFrame()
    sub['ImageId'] = new_test_ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(save_name, index=False)


warnings.filterwarnings('ignore', category=UserWarning, module='skimage')