import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pyjet.backend as J


# This code is borrowed from https://github.com/petrosgk/Kaggle-Carvana-Image-Masking-Challenge

def dice_coeff(y_pred, y_true):
    y_true, y_pred = F.sigmoid(y_true), F.sigmoid(y_pred)
    smooth = 1.
    y_true_f = J.flatten(y_true)
    y_pred_f = J.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_pred, y_true):
    loss = 1 - dice_coeff(y_pred, y_true)
    return loss


def bce_dice_loss(y_pred, y_true):
    loss = F.binary_cross_entropy_with_logits(y_pred, y_true) + dice_loss(y_pred, y_true)
    return loss


def weighted_dice_coeff(y_pred, y_true, weight):
    y_true, y_pred = F.sigmoid(y_true), F.sigmoid(y_pred)
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * torch.sum(w * intersection) + smooth) / (torch.sum(w * m1) + torch.sum(w * m2) + smooth)
    return score


def weighted_dice_loss(y_pred, y_true, mask_weight=None):
    y_true = (y_true.float())
    y_pred = (y_pred.float())
    # if we want to get same size of output, kernel size must be odd number
    if y_pred.size(1) == 128:
        kernel_size = 7
    elif y_pred.size(1) == 256:
        kernel_size = 11
    elif y_pred.size(1) == 512:
        kernel_size = 11
    elif y_pred.size(1) == 1024:
        kernel_size = 21
    else:
        raise ValueError('Unexpected image size')

    padding = (kernel_size - 1) // 2
    averaged_mask = F.avg_pool2d(y_true, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=padding)
    border = (averaged_mask > 0.005).float() * (averaged_mask < 0.995).float()
    if mask_weight is None:
        weight = J.ones(*averaged_mask.size())
    else:
        weight = mask_weight
    w0 = torch.sum(weight)
    weight += border * 2
    w1 = torch.sum(weight)
    weight *= (w0 / w1)
    loss = 1 - weighted_dice_coeff(y_pred, y_true, weight)
    return loss


def weighted_bce_dice_loss(y_pred, y_true, mask_weight=None):
    y_true = y_true.float()
    y_pred = y_pred.float()
    # if we want to get same size of output, kernel size must be odd number
    if y_pred.size(1) == 128:
        kernel_size = 7
    elif y_pred.size(1) == 256:
        kernel_size = 11
    elif y_pred.size(1) == 512:
        kernel_size = 11
    elif y_pred.size(1) == 1024:
        kernel_size = 21
    else:
        raise ValueError('Unexpected image size')

    padding = (kernel_size - 1) // 2
    averaged_mask = F.avg_pool2d(y_true, kernel_size=(kernel_size, kernel_size), stride=(1, 1), padding=padding)
    border = (averaged_mask > 0.005).float() * (averaged_mask < 0.995).float()
    if mask_weight is None:
        weight = Variable(J.ones(*averaged_mask.size()))
    else:
        weight = mask_weight.expand_as(averaged_mask)
    w0 = torch.sum(weight)
    border *= 2
    border += weight
    weight = border
    w1 = torch.sum(weight)
    weight *= (w0 / w1)
    loss = F.binary_cross_entropy_with_logits(y_pred, y_true, weight) + \
           (1 - weighted_dice_coeff(y_pred, y_true, weight))
    return loss


loss_dict = {"binary_crossentropy": "binary_crossentropy",
             "dice_loss": dice_loss,
             "weighted_dice_loss": weighted_dice_loss,
             "weighted_bce_dice_loss": weighted_bce_dice_loss}