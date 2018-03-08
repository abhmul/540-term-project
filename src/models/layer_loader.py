from pyjet.layers import Conv2D, UpSampling2D, MaxPooling2D

LAYERS = {
    "conv2d": Conv2D,
    "upsample2d": UpSampling2D,
    "maxpool2d": MaxPooling2D
}


def load_layer(layer_name, **kwargs):
    return LAYERS[layer_name](**kwargs)