# Defines a custom neural network architecture called CASPIAN, tailored for flood depth prediction.
# Integrates domain-specific inputs like SLR (Sea Level Rise).
# Uses a UNet-like encoder-decoder structure enhanced with:
    # ResNeXt bottlenecks
    # Attention (CBAM)
    # Spatial supervision
    # Optional SLR injection


from models.keras_vision_transformer import swin_layers
from models.keras_vision_transformer import transformer_layers
from tensorflow.keras.layers import (Input, Dropout, Lambda, Conv2D, Conv2DTranspose, 
                                     MaxPooling2D, Concatenate, Activation, Add, multiply, 
                                     LeakyReLU, ZeroPadding2D, UpSampling2D, Dense, 
                                     AveragePooling2D)


from tensorflow.keras.models import Model
import tensorflow.keras.layers as layers
import tensorflow as tf
from keras import backend as K

class SLRExpandLayer(tf.keras.layers.Layer):
    def __init__(self, target_channels, **kwargs):
        super(SLRExpandLayer, self).__init__(**kwargs)
        self.target_channels = target_channels
        self.dense = layers.Dense(target_channels, activation='relu')

    def call(self, inputs):
        slr, x_in = inputs
        slr_adjusted = self.dense(slr)
        shape = tf.shape(x_in)
        height = shape[1]
        width = shape[2]
        slr_expanded = tf.expand_dims(tf.expand_dims(slr_adjusted, axis=1), axis=1)
        slr_tiled = tf.tile(slr_expanded, [1, height, width, 1])
        return slr_tiled

    def get_config(self):
        config = super(SLRExpandLayer, self).get_config()
        config.update({"target_channels": self.target_channels})
        return config

class CenteringLayer(tf.keras.layers.Layer):
    def call(self, x):
        return tf.where(tf.equal(x, 0.5), tf.constant(-1.0, dtype=x.dtype), x)

def cbam_block(input_tensor, reduction=16):
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Dense(channels // reduction, activation='relu')(se)
    se = layers.Dense(channels, activation='sigmoid')(se)
    se = layers.Reshape((1, 1, channels))(se)
    x = layers.multiply([input_tensor, se])

    se_avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    se_max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    se = layers.Concatenate(axis=-1)([se_avg_pool, se_max_pool])
    se = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(se)
    x = layers.multiply([x, se])
    return x

def resnetxt_block(x, filters, cardinality=32, bottleneck_width=4, 
                   strides=1, activation='relu', init="glorot_normal", kernel=3):
    shortcut = x
    group_width = bottleneck_width * cardinality

    x = layers.Conv2D(group_width, (1, 1), padding='same', strides=strides, 
                      activation=activation, kernel_initializer=init)(x)
    x = layers.Conv2D(group_width, (kernel, kernel), padding='same', strides=1, 
                      groups=cardinality, activation=activation, kernel_initializer=init)(x)
    x = layers.Conv2D(filters, (1, 1), padding='same', strides=1, kernel_initializer=init)(x)

    if strides != 1 or x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides)(shortcut)

    x = layers.add([shortcut, x])
    x = layers.Activation(activation)(x)
    return x

def CASPIAN(input_shape=(1024, 1024, 1), filters=81, cardinality=32, 
            activation='relu', init="glorot_normal",
            input_centering=True, depth=4, bottleneck_depth=6, 
            compression_factor=0.9, gr_level=4, sup_level=1,
            slr_input=False):
    
    input_grid = layers.Input(shape=input_shape, name='input_grid')
    if slr_input:
        slr_value = layers.Input(shape=(1,), name='slr_input')
        slr_denseX = layers.Dense(1, activation=activation)(slr_value)
        slr_dense = layers.Dense(64, activation=activation)(slr_value)
        slr_dense = layers.Dense(128, activation=activation)(slr_dense)
    else:
        slr_value = None

    if input_centering:
        x = CenteringLayer()(input_grid)
    else:
        x = input_grid

    pools = []
    for i in range(depth):
        if i == 0:
            pool_size = 2
            s = layers.Conv2D(filters=filters, kernel_size=(2, 2), strides=(2, 2), 
                              padding='same', activation=activation, kernel_initializer=init)(x)
            mask = tf.where(tf.equal(x, -1 if input_centering else 0.5), x, tf.zeros_like(x))
            pool_1 = mask
            mask = tf.where(tf.equal(x, 1), x, tf.zeros_like(x))
            pool_2 = mask
            p = layers.concatenate([pool_1, pool_2], axis=-1)
            p = layers.MaxPooling2D(pool_size=(pool_size, pool_size), 
                                    strides=(2, 2), padding='same')(p)
            pools.append(p)
            x = layers.concatenate([s, p], axis=-1)
            x = layers.Conv2D(filters, (1, 1), padding="same", 
                              activation=None, kernel_initializer=init)(x)
            x = layers.add([s, x])
            x = layers.Activation(activation)(x)
        else:
            s = x
            d = layers.DepthwiseConv2D(depth_multiplier=1, kernel_size=(2, 2), 
                                       strides=(2, 2), padding="same", 
                                       activation=activation, kernel_initializer=init)(x)
            d = layers.Conv2D(filters, (1, 1), padding="same", 
                              activation=activation, kernel_initializer=init)(d)
            p = layers.MaxPooling2D(pool_size=(pool_size, pool_size), 
                                    strides=(2, 2), padding='same')(pools[-1])
            pools.append(p)
            x = layers.concatenate([d, p], axis=-1)
            x = layers.Conv2D(filters, (1, 1), padding="same", 
                              activation=None, kernel_initializer=init)(x)
            b = layers.Conv2D(filters, (1, 1), strides=(2, 2), 
                              padding="same", activation=None, kernel_initializer=init)(s)
            x = layers.add([b, x])
            x = layers.Activation(activation)(x)
    
    for i in range(bottleneck_depth):
        x = resnetxt_block(x, filters=filters, cardinality=cardinality, 
                           init=init, activation=activation)
        if i % 2 == 0:
            x = cbam_block(x)
        
    the_weights = None
    for i in range(depth-1, -1, -1):
        x = layers.Concatenate(axis=-1)([x, pools[i]])
        x = layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), 
                                   strides=(2, 2), padding='same', 
                                   activation=activation, kernel_initializer=init)(x)

        if sup_level != 0:
            p = tf.reduce_sum(pools[i], axis=-1, keepdims=True)
            p = layers.AveragePooling2D(pool_size=(gr_level, gr_level), 
                                        strides=(gr_level, gr_level), 
                                        padding='same')(p)
            spatial_encoding_flattened = layers.Flatten()(p)
            weights = layers.Dense(int(compression_factor*filters), 
                                   activation=activation, kernel_initializer=init)(spatial_encoding_flattened)
            if slr_input:
                combined = layers.Concatenate()([slr_denseX, weights])
                weights = layers.Dense(filters, activation='sigmoid', kernel_initializer=init)(combined)
                weights_reshaped = layers.Reshape((1, 1, filters))(weights)
                the_weights = weights_reshaped
            sup_level = sup_level - 1
        else:
            weights_reshaped = the_weights

        unifying_convs = layers.Conv2D(filters, (1, 1), padding='same', 
                                       kernel_initializer=init, activation=activation)(x)
        x = tf.keras.layers.Multiply()([unifying_convs, weights_reshaped])
    
    out = layers.Conv2D(1, kernel_size=(1, 1), strides=(1, 1), 
                        activation=None, padding='same', kernel_initializer=init)(x)
    summed_features = tf.reduce_sum(x, axis=-1, keepdims=True)
    if slr_input:
        from models.keras_vision_transformer import transformer_layers  # example if needed
        slr_tiled = SLRExpandLayer(target_channels=1)([slr_denseX, summed_features])
        summed_features = layers.Multiply()([summed_features, slr_tiled])
    out = layers.add([summed_features, out])
    out = layers.Activation('relu')(out)

    if slr_input:
        model = tf.keras.models.Model(inputs=[input_grid, slr_value], outputs=out)
    else:
        model = tf.keras.models.Model(inputs=input_grid, outputs=out)

    return model
