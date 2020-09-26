import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from .model import Model


class UserAutoRec(Model):
    def __init__(self, data, hidden_size=128, lr=0.01, batch_size=1024, epochs=1000):
        self.data = data
        self.model = keras.Sequential()
        self.model.add(keras.layers.InputLayer(input_shape=self.data.user_size, batch_size=batch_size))
        self.model.add(keras.layers.Dense(hidden_size, activation='relu',
                                          kernel_regularizer=keras.regularizers.L2))
        self.model.add(keras.layers.Dense(self.data.user_size,activation="sigmoid",
                                          kernel_regularizer=keras.regularizers.L2))
        self.opt = keras.optimizers.Nadam(lr)

        self.batch_size = batch_size
        self.epochs = epochs

    def train(self):
        for user_batch, epoch in self.get_batch():
            loss = self.train_step(user_batch)
            if epoch % 100 == 0:
                print("epoch:{} loss:{}".format(epoch, tf.reduce_mean(loss)))

    def make_rec(self, u, item_cand):
        pre = self.model(self.data.matrix[u].toarray()).numpy()(item_cand)
        return pre

    @tf.function
    def train_step(self, user_batch):
        with tf.GradientTape() as tape:
            pre = self.model(user_batch)
            loss = keras.losses.MSE(user_batch, pre)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def get_batch(self):
        m = self.data.matrix
        sample_array = np.arange(start=0, stop=self.data.user_size)
        for epoch in range(self.epochs):
            sample_batch = np.random.choice(sample_array, size=self.batch_size)
            user_batch = m[sample_batch].toarray()
            yield np.asarray(user_batch).astype(np.float32), epoch


class ItemAutoRec(Model):
    def __init__(self, data, hidden_size=128, lr=0.01, batch_size=1024, epochs=1000):
        self.data = data
        self.model = keras.Sequential()
        self.model.add(keras.layers.InputLayer(input_shape=self.data.item_size, batch_size=batch_size))
        self.model.add(keras.layers.Dense(hidden_size, activation='relu',
                                          kernel_regularizer=keras.regularizers.L2))
        self.model.add(keras.layers.Dense(self.data.item_size,activation="sigmoid",
                                          kernel_regularizer=keras.regularizers.L2))
        self.opt = keras.optimizers.Nadam(lr)

        self.batch_size = batch_size
        self.epochs = epochs

        self._m = self.data.matrix.transpose(copy=True)

    def train(self):
        for item_batch, epoch in self.get_batch():
            loss = self.train_step(item_batch)
            if epoch % 100 == 0:
                print("epoch:{} loss:{}".format(epoch, tf.reduce_mean(loss)))

    def make_rec(self, u, item_cand):
        m = self._m
        pre = self.model(m[item_cand].toarray())[:, u]
        return pre

    @tf.function
    def train_step(self, item_batch):
        with tf.GradientTape() as tape:
            pre = self.model(item_batch)
            loss = keras.losses.MSE(item_batch, pre)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def get_batch(self):
        m = self._m
        sample_array = np.arange(start=0, stop=self.data.item_size)
        for epoch in range(self.epochs):
            sample_batch = np.random.choice(sample_array, size=self.batch_size)
            item_batch = m[sample_batch].toarray()
            yield np.asarray(item_batch).astype(np.float32), epoch

