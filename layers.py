import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class FMLayer(keras.layers.Layer):
    def __init__(self, embed_size=128):
        super(FMLayer, self).__init__()
        self.embed_size = 128

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.W = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer='he_uniform',
                                 regularizer=keras.regularizers.l2(),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.k, input_shape[-1]),
                                 initializer='he_uniform',
                                 regularizer=keras.regularizers.l2(),
                                 trainable=True)

    def call(self, inputs):
        first_order = self.w0 + tf.matmul(inputs, self.W)
        second_order = 0.5 * tf.reduce_sum(
            tf.pow(tf.matmul(inputs, tf.transpose(self.V)), 2) -
            tf.matmul(tf.pow(inputs, 2), tf.pow(tf.transpose(self.V), 2)),
            axis=1, keepdims=True)

        return first_order + second_order


class ResidualLayer(keras.layers.Layer):
    def __init__(self, unit1, unit2):
        super(ResidualLayer, self).__init__()
        self.layer1 = keras.layers.Dense(units=unit1)
        self.layer2 = keras.layers.Dense(units=unit2)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = keras.layers.LeakyReLU(0.2)(x)
        x = self.layer2(x)
        outputs = keras.layers.LeakyReLU(0.2)(x + inputs)


