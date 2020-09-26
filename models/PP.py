import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


from .model import Model
from ..layers import ResidualLayer


class _PP(keras.Model):
    def __init__(self, user_size, event_size, feature_nums, embed_size=128):
        super(_PP, self).__init__()
        self.user_embeddings = keras.layers.Embedding(input_dim=user_size, output_dim=embed_size,
                                                      embeddings_regularizer=keras.regularizers.L2(),
                                                      embeddings_initializer='he_uniform')
        self.embeddings_dict = dict()
        for i, num in enumerate(feature_nums):
            self.embeddings_dict[i] = keras.layers.Embedding(input_dim=num, output_dim=embed_size,
                                                             embeddings_regularizer=keras.regularizers.L2(),
                                                             embeddings_initializer="he_uniform")
        self.rec_layer = keras.Sequential()
        self.mha = tfa.layers.MultiHeadAttention(head_size=embed_size, num_heads=8)
        for i in range(6):
            self.rec_layer.add(keras.layers.Conv2D(8, kernel_size=2, strides=2))
            self.rec_layer.add(keras.layers.BatchNormalization())
        # self.rec_layer.add(keras.layers.Dense(event_size, activation="softmax"))
        self.flat = keras.layers.Flatten()
        self.embed_size = embed_size
        self.out = keras.layers.Dense(self.embed_size, activation='tanh')

    def call(self, inputs):
        user_batch = inputs[0]
        user_embed = self.user_embeddings(user_batch)  # batch, embed_size
        for i in range(len(self.embeddings_dict)):
            if i == 0:
                stack = self.embeddings_dict[i](inputs[1][:, :, i])
            else:
                embed_i = self.embeddings_dict[i](inputs[1][:, :, i])
                stack = tf.concat([stack, embed_i], axis=-1)
        stack = tf.concat([stack, inputs[1][:, :, 3:]], axis=-1)  # batch, 20, 388
        stack = tf.reshape(tf.repeat(stack, self.embed_size, axis=-1), (stack.shape[0], stack.shape[1], -1, self.embed_size))
        user_embed = tf.expand_dims(tf.expand_dims(user_embed, axis=1), axis=1)
        stack = stack * user_embed
        stack = tf.transpose(stack, (0, 2, 3, 1))
        out = self.rec_layer(stack)
        out = self.flat(out)
        out = self.out(out)
        return out
        

class PP(Model):
    def train(self):
        pass

    def make_rec(self, u, item_cand):
        pass

    def __init__(self, data, lr=0.001, epochs=10, batch_size=1024, embed_size=128, k=20):
        self.data = data
        self.k = k
        self.model = _PP(self.data.user_size, self.data.item_size, [self.data.item_size, self.data.organizer_size, self.data.category_size])
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=epochs,
            decay_rate=0.96,
            staircase=True)
        optimizer = tfa.optimizers.ConditionalGradient(lr_schedule)
        optimizer = tfa.optimizers.Lookahead(optimizer)
        self.model.compile(loss=keras.losses.sparse_categorical_crossentropy, optimizer=optimizer, metrics=keras.metrics.Accuracy())

    def get_data(self, k=10):
        tss = TimeSeriesSplit(n_splits=k)
        for train, test in tss.split(self.data.event):
            user = self.data.rsvp[self.data.rsvp['event'].isin(train)]['user'].values
            event = self.data.rsvp[self.data.rsvp['event'].isin(train)]['event'].values
            feature = self.data.event.loc[event][['event', 'organzier', 'category', 'start time', 'end time', 'lon', 'lat']].values
            feature_list = [feature]
            for i in range(self.k-1):
                features = self.data.event.sample(len(user))[
                    ['event', 'organzier', 'category', 'start time', 'end time', 'lon', 'lat']].values
                feature_list.append(features)

            test_user = self.data.rsvp[self.data.rsvp['event'].isin(test)]['user'].values
            event = self.data.rsvp[self.data.rsvp['event'].isin(test)]['event'].values
            feature = self.data.event.loc[event][['event', 'organzier', 'category', 'start time', 'end time', 'lon', 'lat']].values
            test_feature_list = [feature]
            for i in range(self.k-1):
                features = self.data.event.sample(len(test_user))[
                    ['event', 'organzier', 'category', 'start time', 'end time', 'lon', 'lat']].values
                test_feature_list.append(features)
            feature_array = np.stack(feature_list).transpose(1, 0, 2).astype(np.float32)
            test_feature_array = np.stack(test_feature_list).transpose(1, 0, 2).astype(np.float32)

            yield [user, feature_array], [test_user, test_feature_array]
