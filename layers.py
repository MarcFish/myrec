import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


class FMLayer(keras.layers.Layer):
    def __init__(self, k=10, w_lr=1e-2, v_lr=1e-2):
        super(FMLayer, self).__init__()
        self.k = k
        self.w_lr = w_lr
        self.v_lr = v_lr

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.W = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer='he_uniform',
                                 regularizer=keras.regularizers.l2(self.w_lr),
                                 trainable=True)
        self.V = self.add_weight(name='V', shape=(self.k, input_shape[-1]),
                                 initializer='he_uniform',
                                 regularizer=keras.regularizers.l2(self.v_lr),
                                 trainable=True)

    def call(self, inputs):
        first_order = self.w0 + tf.matmul(inputs, self.W)
        second_order = 0.5 * tf.reduce_sum(
            tf.pow(tf.matmul(inputs, tf.transpose(self.V)), 2) -
            tf.matmul(tf.pow(inputs, 2), tf.pow(tf.transpose(self.V), 2)),
            axis=1, keepdims=True)

        return tf.squeeze(first_order + second_order)

