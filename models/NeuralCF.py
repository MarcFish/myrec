import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from .model import Model


class _GMF(keras.Model):
    def __init__(self, user_size, item_size, embed_size=128):
        super(_GMF, self).__init__()
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
        return tf.multiply(user_embeddings, item_embeddings)


class _NMF(keras.Model):
    def __init__(self, user_size, item_size, embed_size=128):
        super(_NMF, self).__init__()
        self.user_embeddings = keras.layers.Embedding(input_dim=user_size,output_dim=embed_size,
                                                      embeddings_regularizer=keras.regularizers.L2(),
                                                      embeddings_initializer='he_uniform')
        self.item_embeddings = keras.layers.Embedding(input_dim=item_size,output_dim=embed_size,
                                                      embeddings_regularizer=keras.regularizers.L2(),
                                                      embeddings_initializer='he_uniform')
        self.concat = keras.layers.Concatenate()

        self.mlp = keras.Sequential()
        self.mlp.add(keras.layers.Dense(1024, activation='relu',
                                            kernel_regularizer=keras.regularizers.L2()))
        self.mlp.add(keras.layers.Dense(512, activation='relu',
                                            kernel_regularizer=keras.regularizers.L2()))
        self.mlp.add(keras.layers.Dense(256, activation='relu',
                                            kernel_regularizer=keras.regularizers.L2()))
        self.mlp.add(keras.layers.Dense(128, activation='relu',
                                            kernel_regularizer=keras.regularizers.L2()))

    def call(self, inputs):
        user_batch = inputs[0]
        item_batch = inputs[1]
        user_embed = self.user_embeddings(user_batch)
        item_embed = self.item_embeddings(item_batch)
        embed = self.concat([user_embed, item_embed])
        out = self.mlp(embed)
        return out


class _NCF(keras.Model):
    def __init__(self, user_size, item_size, embed_size=128):
        super(_NCF, self).__init__()
        self.gmf = _GMF(user_size, item_size, embed_size)
        self.nmf = _NMF(user_size, item_size, embed_size)

        self.pre_layer = keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_batch = inputs[0]
        item_batch = inputs[1]
        o1 = self.gmf([user_batch, item_batch])
        o2 = self.nmf([user_batch, item_batch])
        o = tf.concat([o1, o2], axis=-1)
        o = self.pre_layer(o)
        return o


class NeuralCF(Model):
    def __init__(self, data, lr=0.001, embed_size=128, batch_size=2048, epochs=10):
        self.data = data
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.model = _NCF(self.data.user_size, self.data.item_size, self.embed_size)
        self.model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Nadam(self.lr), metrics=[keras.metrics.RootMeanSquaredError()])

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

    def make_rec(self, u, item_cand):
        us = np.repeat(np.asarray([u], dtype=np.int32), len(item_cand), axis=0)
        o = self.model(us, item_cand)
        return o


