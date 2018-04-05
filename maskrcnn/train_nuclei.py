import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

from config import Config
import utils
import model as modellib
import visualize
from model import log



# Root directory of the project
ROOT_DIR = ''' fill in your own ''' #os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

class NucleiConfig(Config):
    """Configuration for training on the nuclei dataset.
    Derives from the base Config class and overrides values specific
    to the nuclei dataset.
    """
    # Give the configuration a recognizable name
    NAME = "nuclei"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 nuclei

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 100

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5



class NucleiDataset(utils.Dataset):
    def load_nuclei(self,count,data_path):
        # add classes
        self.add_class('nuclei',1,'nucleus')
        ## add images
        paths = #### this is a list of count number length of paths for (images,masks) pairings
        for i in range(count):
            self.add_image('nuclei',id = i,path = paths[i])
        return paths
    def load_mask(self,image_id):
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        info = self.image_info[image_id]

        mask = skimage.io.imread(info['path'][1])
        return mask,class_ids
    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nuclei":
            return info["nuclei"]
        else:
            super(self.__class__).image_reference(self, image_id)



# Training dataset
dataset_train = NucleiDataset()
dataset_train.load_nuclei(500, data_path)
dataset_train.prepare()

# Validation dataset
dataset_val = NucleiDataset()
dataset_val.load_nuclei(50, data_path)
dataset_val.prepare()


def create_maskrcnn(init_with = 'coco',mode='training',config = NucleiConfig(),model_dir = MODEL_DIR):
    model = modellib.MaskRCNN(mode=mode, config=config,
                          model_dir=MODEL_DIR)
     # imagenet, coco, or last (init_with)

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
    # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
    return model

def train_maskrcnn(model,train,val,lr,epochs,layers='heads'):
    model.train(train,val,lr,epochs,layers=layers)
    return model


class InferenceConfig(NucleiConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def inference_maskrcnn(inference_config = InferenceConfig(),model_path,mode='inference'):
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    return model


def evaluate_maskrcnn(dataset_val,inference_config = InferenceConfig()):
    image_ids = np.random.choice(dataset_val.image_ids, 10)
    APs = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask =\
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps =\
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        APs.append(AP)


    print("mAP: ", np.mean(APs))
    return APs
