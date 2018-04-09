import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from pyjet.layers import LSTM, GRU, Conv2D, MaxPooling2D, UpSampling2D, MaskedInput
import pyjet.backend as J
from pyjet.models import SLModel


from . import pytorch_losses
from . import mean_iou


class UNet(nn.Module):

    def __init__(self, num_filters=16, factor=2, num_channels=3, max_val=255.):
        super(UNet, self).__init__()
        self.num_filters = num_filters
        self.factor = factor
        self.num_channels = num_channels
        self.max_val = max_val

        settings_str = "\tnum_filters: {num_filters}\n" \
                       "\tfactor: {factor}\n" \
                       "\tnum_channels: {num_channels}\n" \
                       "\tmax_val: {max_val}\n".format(num_filters=num_filters, factor=factor,
                                                       num_channels=num_channels,
                                                       max_val=max_val)

        logging.info("Creating pytorch unet model with settings:\n" + settings_str)

        # Initialize the unet
        num_filters1 = self.num_filters
        self.resample = Conv2D(self.num_channels, num_filters1, 3, batchnorm=True, fix_inputs=False)
        self.c1_1 = Conv2D(num_filters1, num_filters1, 3, batchnorm=True, activation='relu', padding='same', fix_inputs=False, dropout=0.1)
        self.c1_2 = Conv2D(num_filters1, num_filters1, 3, batchnorm=True, activation='relu', padding='same', fix_inputs=False)
        self.p = MaxPooling2D(2, fix_inputs=False)

        num_filters2 = self.factor * num_filters1
        self.c2_1 = Conv2D(num_filters1, num_filters2, 3, batchnorm=True, activation='relu', padding='same', fix_inputs=False, dropout=0.1)
        self.c2_2 = Conv2D(num_filters2, num_filters2, 3, batchnorm=True, activation='relu', padding='same', fix_inputs=False)

        num_filters3 = self.factor * num_filters2
        self.c3_1 = Conv2D(num_filters2, num_filters3, 3, batchnorm=True, activation='relu', padding='same', fix_inputs=False, dropout=0.2)
        self.c3_2 = Conv2D(num_filters3, num_filters3, 3, batchnorm=True, activation='relu', padding='same', fix_inputs=False)

        num_filters4 = self.factor * num_filters3
        self.c4_1 = Conv2D(num_filters3, num_filters4, 3, batchnorm=True, activation='relu', padding='same', fix_inputs=False, dropout=0.2)
        self.c4_2 = Conv2D(num_filters4, num_filters4, 3, batchnorm=True, activation='relu', padding='same', fix_inputs=False)


        num_filters5 = self.factor * num_filters4
        self.c5_1 = Conv2D(num_filters4, num_filters5, 3, batchnorm=True, activation='relu', padding='same', fix_inputs=False, dropout=0.3)
        self.c5_2 = Conv2D(num_filters5, num_filters5, 3, batchnorm=True, activation='relu', padding='same', fix_inputs=False)

        num_filters6 = num_filters5 // self.factor
        self.u = UpSampling2D(2, fix_inputs=False)
        self.c6_1 = Conv2D(num_filters5 + num_filters4, num_filters6, 3, batchnorm=True, activation='relu',
                           padding='same', fix_inputs=False, dropout=0.2)
        self.c6_2 = Conv2D(num_filters6, num_filters6, 3, batchnorm=True, activation='relu', padding='same', fix_inputs=False)

        num_filters7 = num_filters6 // self.factor
        self.c7_1 = Conv2D(num_filters6 + num_filters3, num_filters7, 3, batchnorm=True, activation='relu',
                           padding='same', fix_inputs=False, dropout=0.2)
        self.c7_2 = Conv2D(num_filters7, num_filters7, 3, batchnorm=True, activation='relu', padding='same', fix_inputs=False)

        num_filters8 = num_filters7 // self.factor
        self.c8_1 = Conv2D(num_filters7 + num_filters2, num_filters8, 3, batchnorm=True, activation='relu',
                           padding='same', fix_inputs=False, dropout=0.1)
        self.c8_2 = Conv2D(num_filters8, num_filters8, 3, batchnorm=True, activation='relu', padding='same', fix_inputs=False)

        num_filters9 = num_filters8 // self.factor
        self.c9_1 = Conv2D(num_filters8 + num_filters1, num_filters9, 3, batchnorm=True, activation='relu',
                           padding='same', fix_inputs=False, dropout=0.1)
        self.c9_2 = Conv2D(num_filters9, num_filters9, 3, batchnorm=True, activation='relu', padding='same', fix_inputs=False)

        self.out = Conv2D(num_filters9, 1, kernel_size=1, activation='linear', fix_inputs=False)

    def forward(self, x):

        x = self.resample(x.permute(0, 3, 2, 1))
        x = self.c1_1(x)
        x = self.c1_2(x)
        c1 = x
        x = self.p(x)

        x = self.c2_1(x)
        x = self.c2_2(x)
        c2 = x
        x = self.p(x)

        x = self.c3_1(x)
        x = self.c3_2(x)
        c3 = x
        x = self.p(x)

        x = self.c4_1(x)
        x = self.c4_2(x)
        c4 = x
        x = self.p(x)

        x = self.c5_1(x)
        x = self.c5_2(x)

        x = torch.cat([self.u(x), c4], dim=1)
        x = self.c6_1(x)
        x = self.c6_2(x)

        x = torch.cat([self.u(x), c3], dim=1)
        x = self.c7_1(x)
        x = self.c7_2(x)

        x = torch.cat([self.u(x), c2], dim=1)
        x = self.c8_1(x)
        x = self.c8_2(x)

        x = torch.cat([self.u(x), c1], dim=1)
        x = self.c9_1(x)
        x = self.c9_2(x)

        loss_in = self.out(x).permute(0, 2, 3, 1).contiguous()
        return F.sigmoid(loss_in), loss_in


class UnetRNNModule(nn.Module):

    def __init__(self, rnn_type='lstm', num_filters=16, factor=2, num_channels=3, max_val=255., img_size=(256, 256),
                 stop_criterion=0.99):
        super(UnetRNNModule, self).__init__()

        # Create the unet
        self.unet = UNet(num_filters, factor, num_channels, max_val)
        if rnn_type == 'lstm':
            rnn_func = LSTM
        elif rnn_type == 'gru':
            rnn_func = GRU
        else:
            raise NotImplementedError(rnn_type)
        self.img_size = img_size
        self.height = img_size[0]
        self.width = img_size[1]
        self.num_channels = num_channels
        self.rnn1_output = Variable(J.zeros(1, self.height, self.width))
        self.rnn2_output = Variable(J.zeros(1, self.width, self.height))

        self.rnn1 = rnn_func(self.width * (num_channels + 2), self.width, return_sequences=True)
        self.rnn2 = rnn_func(self.height * (num_channels + 2), self.height, return_sequences=True)

        self.out = Conv2D(1, 1, kernel_size=1, activation='linear')

        self.stop_criterion = stop_criterion

        self.masker = MaskedInput(mask_value=0.)

    def stopping_criterion(self, total_mask_output, unet_mask, smoothing=1e-9):
        # Stop if the instersection over union is greater than threshold
        batch_size = total_mask_output.size(0)
        assert total_mask_output.size(0) == unet_mask.size(0) == batch_size
        assert total_mask_output.size(1) == unet_mask.size(1) == self.height
        assert total_mask_output.size(2) == unet_mask.size(2) == self.width
        assert total_mask_output.dim() == unet_mask.dim() == 3

        intersection = torch.bmm(total_mask_output.view(batch_size, -1), unet_mask.view(batch_size, -1))
        union = torch.sum(total_mask_output) + torch.sum(unet_mask)
        soft_iou = (intersection + smoothing) / (union + smoothing)
        return soft_iou >= self.stop_criterion

    def forward(self, inputs, num_nuclei=None):
        # num_nuclei is a longtensor of size B that holds number of images in the image
        batch_size = inputs.size(0)
        # inputs is B x H x W x 3
        # mask is B x H x W x 1
        mask, unet_loss_in = self.unet(inputs)

        # Create the rnn input as batch, seq, features
        # features is inputs + mask + rnn out
        rnn1_image = inputs.view(batch_size, self.height, self.width * self.num_channels)
        rnn1_mask = mask.view(batch_size, self.height, self.width)

        rnn2_image = inputs.transpose(1, 2).contiguous().view(batch_size, self.width, self.height * self.num_channels)

        self.rnn1_output = self.rnn1_output.expand(batch_size, self.height, self.width)
        self.rnn2_output = self.rnn2_output.expand(batch_size, self.width, self.height)

        out_masks = []
        rnn_loss_in = []
        if num_nuclei is not None:
            max_num_times = torch.max(num_nuclei)
            for i in range(int(max_num_times)):
                # TODO: Try using ground truth mask instead
                # TODO: Wrap this piece into a method to use for test as well
                # TODO: Validate for the appropriate stopping criterion
                rnn1_input = torch.cat([rnn1_image, rnn1_mask, self.rnn1_output], dim=-1)
                # Take the max to get the output from previous segmasks
                self.rnn1_output = torch.max(self.rnn1(rnn1_input), self.rnn1_output)

                # B x W x H
                rnn2_mask = self.rnn1_output.transpose(1, 2)
                rnn2_input = torch.cat([rnn2_image, rnn2_mask, self.rnn2_output], dim=-1)
                pre_mask_output = self.rnn2(rnn2_input)
                self.rnn2_output = torch.max(pre_mask_output, self.rnn2_output)
                # reshape, sigmoid, and save the output masks N x (B x H x W)
                # Enters rnn_loss_in as B x H x W
                rnn_loss_in.append(self.out(pre_mask_output.transpose(1, 2).unsqueeze(-1)).squeeze(-1))
                out_masks.append(F.sigmoid(rnn_loss_in[-1]))
            # Stack the outputs together and mask the ones after
            # B x N x H x W
            rnn_loss_in = torch.stack(rnn_loss_in, dim=1)
            out_masks = torch.stack(out_masks, dim=1)

        # We have to dynamically generate until we reach the stopping criterion
        else:
            num_nuclei = J.zeros(batch_size).long()
            stop = J.zeros(batch_size).byte()
            # Keep on generating until 1 all of the samples are done
            while not stop.all():
                # Get the batches stopping criteria
                stop = torch.max(stop, self.stopping_criterion(self.rnn2_output.transpose(1, 2), rnn1_mask))
                # If we don't stop add 1 to the number of nuclei
                num_nuclei += (1 - stop)
                # TODO: Run the network
            max_num_times = torch.max(num_nuclei)


        # Mask the produced nuclei beyond num nuclei
        rnn_loss_weights = self.masker(Variable(J.ones(batch_size, max_num_times, 1, 1)), num_nuclei)
        out_masks = self.masker(out_masks, num_nuclei)
        return out_masks, unet_loss_in, rnn_loss_in, rnn_loss_weights


# TODO: create the SLModel
class UnetRNN(SLModel):

    def __init__(self, rnn_type='lstm', num_filters=16, factor=2, num_channels=3, max_val=255., img_size=(256, 256),
                 stop_criterion=0.99):
        super(UnetRNN, self).__init__()
        self.unet_rnn = UnetRNNModule(rnn_type='lstm', num_filters=16, factor=2, num_channels=3, max_val=255.,
                                      img_size=(256, 256), stop_criterion=0.99)

        self.height = self.unet_rnn.height
        self.width = self.unet_rnn.width
        # Create the loss function
        self.register_loss_function(self.unet_bce_dice)
        self.register_loss_function(self.rnn_bce_dice)

        self.register_metric_function(self.unet_mean_iou)
        self.register_metric_function(self.rnn_mean_iou)
        self.out_masks, self.unet_loss, self.rnn_loss, self.rnn_loss_weights = [None]*4

        self.add_optimizer(optim.Adam(param for param in self.parameters() if param.requires_grad), "adam")

    def cast_input_to_torch(self, inputs, volatile=False):
        x, num_nuclei = inputs
        # Fix x
        x = (x / 255.).astype(np.float32)

        x = super().cast_input_to_torch(x, volatile=volatile)
        num_nuclei = J.from_numpy(num_nuclei).long()
        return x, num_nuclei

    def cast_target_to_torch(self, targets, volatile=False):
        # keep these sparse
        # targets["mask"] = Variable((torch.from_numpy(targets["mask"])),
        #                            requires_grad=False, volatile=volatile).contiguous()
        # targets["segment"] = Variable((torch.from_numpy(targets["segment"])),
        #                               requires_grad=False, volatile=volatile).contiguous()
        # if J.use_cuda:
        #     targets["mask"] = targets["mask"].cuda()
        #     targets["segment"] = targets["segment"].cuda()
        targets["mask"] = super().cast_target_to_torch(targets["mask"], volatile=volatile).contiguous()
        targets["segment"] = super().cast_target_to_torch(targets["segment"], volatile=volatile).contiguous()
        return targets

    def call(self, inputs):
        x, num_nuclei = inputs
        self.out_masks, self.unet_loss, self.rnn_loss, self.rnn_loss_weights = self.unet_rnn(x, num_nuclei)
        return self.out_masks

    def unet_bce_dice(self, targets):
        # Unet loss is B x H x W x 1
        return pytorch_losses.weighted_bce_dice_loss(self.unet_loss, targets["mask"])

    def rnn_bce_dice(self, targets):
        # rnn loss is B x N x H x W
        # We'll reshape combine the batch and N dimension
        rnn_loss = self.rnn_loss.view(-1, self.height, self.width)
        rnn_loss_weights = self.rnn_loss_weights.view(-1, 1, 1)
        rnn_target = targets["segment"].view(-1, self.height, self.width)
        return pytorch_losses.weighted_bce_dice_loss(rnn_loss, rnn_target, mask_weight=rnn_loss_weights)

    def unet_mean_iou(self, targets):
        return mean_iou.mean_iou(F.sigmoid(self.unet_loss), targets["mask"])

    def rnn_mean_iou(self, targets):
        # rnn loss is B x N x H x W
        # We'll reshape combine the batch and N dimension
        rnn_loss = self.rnn_loss.view(-1, self.height, self.width)
        rnn_loss_weights = self.rnn_loss_weights.view(-1, 1, 1)
        rnn_target = targets["segment"].view(-1, self.height, self.width)
        return mean_iou.mean_iou(F.sigmoid(rnn_loss) * rnn_loss_weights, rnn_target)
