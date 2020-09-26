import tensorflow as tf
import tensorflow.keras as keras
import numpy as np


from .model import Model


class _MF(keras.Model):
    def __init__(self, user_size, item_size, embed_size=128):
        super(_MF, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.embed_size = embed_size

        self.user_embeddings = keras.layers.Embedding(input_dim=self.user_size, output_dim=self.embed_size,
                                                      embeddings_regularizer=keras.regularizers.L2(),
                                                      embeddings_initializer='he_uniform')
        self.item_embeddings = keras.layers.Embedding(input_dim=self.item_size, output_dim=self.embed_size,
                                                      embeddings_regularizer=keras.regularizers.L2(),
                                                      embeddings_initializer='he_uniform')

    def call(self, inputs):
        user_batch = inputs[0]
        item_batch = inputs[1]
        user_embeddings = self.user_embeddings(user_batch)
        item_embeddings = self.item_embeddings(item_batch)
        return tf.reduce_sum(tf.multiply(user_embeddings, item_embeddings), axis=-1)


class MF(Model):
    def __init__(self, data, embed_size=128, lr=0.01, batch_size=512, epochs=10):
        self.data = data
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.model = _MF(self.data.user_size, self.data.item_size, self.embed_size)
        self.model.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.Nadam(self.lr), metrics=[keras.metrics.RootMeanSquaredError()])

    def train(self):
        auc_list = list()
        for train, test in self.data.train_test_split():
            user = train.user.values
            movie = train.movie.values
            rating = train.rating.values
            self.model.fit(x=[user, movie], y=rating, epochs=self.epochs, batch_size=self.batch_size)
            auc = self.model.evaluate([test.user.values, test.movie.values], test.rating.values)[1]
            auc_list.append(auc)
            print("test AUC:{}".format(auc))
        print("all AUC:{}".format(np.mean(auc_list)))

    def make_rec(self, u, item_cand):  # TODO
        pass
