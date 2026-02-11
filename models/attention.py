import tensorflow as tf
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Dense, Multiply, Add, Concatenate, Layer

class SpatialAttention(Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.conv = Conv2D(1, kernel_size, padding='same', activation='sigmoid')

    def call(self, inputs):
        avg = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_val = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = Concatenate(axis=-1)([avg, max_val])
        return self.conv(concat)

def cbam_block(input_feature, ratio=8):
    avg_pool = GlobalAveragePooling2D()(input_feature)
    max_pool = GlobalMaxPooling2D()(input_feature)

    concat = Concatenate()([avg_pool, max_pool])
    dense1 = Dense(input_feature.shape[-1] // ratio, activation='relu')(concat)
    dense2 = Dense(input_feature.shape[-1], activation='sigmoid')(dense1)

    channel_attention = Multiply()([input_feature, dense2])

    spatial = SpatialAttention()(channel_attention)
    spatial_attention = Multiply()([channel_attention, spatial])

    return Add()([input_feature, spatial_attention])
