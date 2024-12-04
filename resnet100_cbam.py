import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, MaxPooling2D,
    GlobalAveragePooling2D, Dense, Add, Multiply
)
from tensorflow.keras.models import Model


def convolutional_block(x, filters, kernel_size=3, stride=1, padding='same'):
    """
    Convolutional block with BatchNormalization and ReLU activation.
    """
    x = Conv2D(filters, kernel_size, strides=stride, padding=padding)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def identity_block(x, filters, kernel_size=3, stride=1, padding='same'):
    """
    Identity block with BatchNormalization and ReLU activation.
    """
    shortcut = x

    x = convolutional_block(x, filters, kernel_size, stride, padding)
    x = Conv2D(filters, kernel_size, padding=padding)(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x


def cbam_block(x, ratio=8):
    """
    Convolutional Block Attention Module (CBAM).
    """
    # Channel attention
    channel_attention = GlobalAveragePooling2D()(x)
    channel_attention = Dense(x.shape[-1] // ratio, activation='relu')(channel_attention)
    channel_attention = Dense(x.shape[-1], activation='sigmoid')(channel_attention)
    x = Multiply()([x, channel_attention])

    # Spatial attention
    spatial_attention = tf.reduce_max(x, axis=-1, keepdims=True)
    spatial_attention = tf.reduce_mean(x, axis=-1, keepdims=True)
    spatial_attention = tf.expand_dims(spatial_attention, axis=-1)
    spatial_attention = Conv2D(1, 7, padding='same', activation='sigmoid')(spatial_attention)
    x = Multiply()([x, spatial_attention])

    return x


def resnet100_cbam(input_shape, num_classes):
    """
    ResNet100 with CBAM attention module.
    """
    inputs = Input(shape=input_shape)

    # Initial convolutional layer
    x = convolutional_block(inputs, 64, 7, 2, 'same')
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # ResNet blocks
    x = identity_block(x, 64)
    x = identity_block(x, 64)
    x = identity_block(x, 64)

    x = identity_block(x, 128, stride=2)
    x = identity_block(x, 128)
    x = identity_block(x, 128)
    x = identity_block(x, 128)

    x = identity_block(x, 256, stride=2)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)
    x = identity_block(x, 256)

    x = identity_block(x, 512, stride=2)
    x = identity_block(x, 512)
    x = identity_block(x, 512)

    # CBAM block
    x = cbam_block(x)
    print(f"CBAM output shape: {x.shape}")

    # Global average pooling and dense layer
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
