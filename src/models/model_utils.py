import logging

import numpy as np
import tensorflow as tf
import keras.backend as K


def reset_model(model):
    session = K.get_session()
    for layer in model.layers:
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg, 'initializer'):
                initializer_method = getattr(v_arg, 'initializer')
                initializer_method.run(session=session)
                logging.info('reinitializing layer {}.{}'.format(layer.name, v))


# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)