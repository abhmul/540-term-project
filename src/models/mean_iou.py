import tensorflow as tf
import torch
import torch.nn.functional as F

import pyjet.backend as J


# Define IoU metric
def mean_iou(y_pred, y_true):
    t_range = torch.arange(0.5, 1.0, 0.05)
    prec = J.zeros(*t_range.size())
    y_true = y_true.float()
    for i, t in enumerate(t_range):
        y_pred_ = (y_pred > t).float()

        # Calculate the iou
        y_true_f = J.flatten(y_true)
        y_pred_f = J.flatten(y_pred_)
        intersection = torch.sum(y_true_f * y_pred_f).data[0]
        union = (torch.sum(y_true_f) + torch.sum(y_pred_f)).data[0]
        if union == 0.:
            # this means there's no intersection either
            score = 1.
        else:
            score = intersection / union
        prec[i] = score

    return torch.mean(prec)
