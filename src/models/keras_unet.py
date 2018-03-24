from keras.layers import Conv2D, Dropout, MaxPooling2D, Input, UpSampling2D, concatenate, Activation, BatchNormalization
from keras.models import Model
from . import model_utils


def unet(num_filters=16, factor=2, optimizer="adam", loss="binary_crossentropy"):
    inputs = Input((None, None, 3), name="input")
    s = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (1, 1), padding='same')(inputs)))
    c1 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(s)))
    c1 = Dropout(0.1)(c1)
    c1 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(c1)))
    p1 = MaxPooling2D((2, 2))(c1)
    
    num_filters *= factor   
    c2 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(p1)))
    c2 = Dropout(0.1)(c2)
    c2 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(c2)))
    p2 = MaxPooling2D((2, 2))(c2)

    num_filters *= factor
    c3 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(p2)))
    c3 = Dropout(0.2)(c3)
    c3 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(c3)))
    p3 = MaxPooling2D((2, 2))(c3)

    num_filters *= factor
    c4 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(p3)))
    c4 = Dropout(0.2)(c4)
    c4 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(c4)))
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    num_filters *= factor
    c5 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(p4)))
    c5 = Dropout(0.3)(c5)
    c5 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(c5)))

    num_filters //= factor
    u6 = UpSampling2D(size=(2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(u6)))
    c6 = Dropout(0.2)(c6)
    c6 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(c6)))

    num_filters //= factor
    u7 = UpSampling2D(size=(2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(u7)))
    c7 = Dropout(0.2)(c7)
    c7 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(c7)))

    num_filters //= factor
    u8 = UpSampling2D(size=(2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(u8)))
    c8 = Dropout(0.1)(c8)
    c8 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(c8)))

    num_filters //= factor
    u9 = UpSampling2D(size=(2, 2))(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(u9)))
    c9 = Dropout(0.1)(c9)
    c9 = Activation('relu')(BatchNormalization()(Conv2D(num_filters, (3, 3), padding='same')(c9)))

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=optimizer, loss=loss, metrics=[model_utils.mean_iou])
    model.summary()
    return model
