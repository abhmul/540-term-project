import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from pyjet.models import SLModel
from pyjet.layers import MaskedLayer, Layer
import pyjet.backend as J
import pyjet.layers.functions as L

from . import layer_loader


class EncoderBlock(Layer):

    def __init__(self, convs, pool):
        super(EncoderBlock, self).__init__()
        prepool_mask_value = 'min' if "max" in pool[0]["name"] else 0.0
        self.convs = nn.ModuleList(
            [MaskedLayer(layer_loader.load_layer(**conv),
                         dim=2,
                         mask_value=prepool_mask_value) for conv in convs])
        self.pool = MaskedLayer(layer_loader.load_layer(**pool), dim=2, mask_value=0.0)

    def calc_input_size(self, output_size):
        output_size = self.pool.layer.calc_input_size(output_size)
        for conv in self.convs:
            output_size = conv.layer.calc_input_size(output_size)
        return output_size

    def forward(self, x, seq_lens):
        for conv in self.convs:
            x, seq_lens = conv(x, seq_lens)
        residual = x
        x, seq_lens = self.pool(x, seq_lens)
        return x, seq_lens, residual

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.pool.reset_parameters()


class Neck(nn.Module):

    def __init__(self, layers):
        super(Neck, self).__init__()
        self.layers = nn.ModuleList(
            [MaskedLayer(layer_loader.load_layer(**layer),
                         dim=2,
                         mask_value=0.0) for layer in layers])

    def forward(self, x, seq_lens):
        for layer in self.layers:
            x, seq_lens = layer(x, seq_lens)
        return x, seq_lens


class DecoderBlock(nn.Module):

    def __init__(self, convs, upsample):
        super(DecoderBlock, self).__init__()
        self.convs = nn.ModuleList(
            [MaskedLayer(layer_loader.load_layer(**conv),
                         dim=2,
                         mask_value=0.0) for conv in convs])
        self.upsample = MaskedLayer(layer_loader.load_layer(**upsample),
                                    dim=2,
                                    mask_value=0.0)

    def forward(self, x, seq_lens, residual):
        x, seq_lens = self.upsample(x, seq_lens)
        x = torch.cat([x, residual], dim=-1)
        for conv in self.convs:
            x, seq_lens = conv(x, seq_lens)
        return x, seq_lens

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.upsample.reset_parameters()


class UNet(SLModel):

    def __init__(self, encoder, neck, decoder):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList([EncoderBlock(**encoder_block) for encoder_block in encoder])
        self.neck = Neck(neck)
        self.decoder = nn.ModuleList([DecoderBlock(**decoder_block) for decoder_block in decoder])
        self.min_size = self.calc_input_size(1)

    def calc_input_size(self, min_size):
        for encoder_block in self.encoder:
            min_size = encoder_block.calc_input_size(min_size)
        return min_size

    def cast_input_to_torch(self, x, volatile=False):
        # Get the seq lens and pad it
        seq_lens = J.LongTensor(
            [[max(sample.shape[0], self.min_len), max(sample.shape[1], self.min_len)] for sample in x])
        pad_shape, _ = seq_lens.max(dim=0)

        x = np.stack([L.pad_numpy_to_shape(sample, shape=tuple(pad_shape)) for sample in x])
        return Variable(J.from_numpy(x).float(), volatile=volatile), seq_lens

    def cast_target_to_torch(self, y, volatile=False):
        return self.cast_input_to_torch(y, volatile=volatile)[0]

    def forward(self, inputs):
        x, seq_lens = inputs

        residuals = []
        for encoder_block in self.encoder:
            x, seq_lens, residual = encoder_block(x, seq_lens)
            residuals.append(residual)

        x, seq_lens = self.neck(x, seq_lens)

        for decoder_block, residual in zip(self.decoder, reversed(residuals)):
            x, seq_lens = decoder_block(x, seq_lens, residual)

        self.loss_in = x
        self.loss_kwargs["weight"] = L.create2d_mask(x, seq_lens)
        return x
