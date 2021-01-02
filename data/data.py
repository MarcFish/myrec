import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Data:
    def __init__(self, filepath='./', batch_size=512, only_rating=False, buffer_size=1024):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.train = pd.read_csv(filepath+'train.csv')
        self.test = pd.read_csv(filepath+'test.csv')
        self.only_rating = only_rating

        self.feature_list = []
        all = pd.concat([self.train, self.test])
        for column in all.columns[:-1]:
            encoder = LabelEncoder()
            encoder.fit(all[column])
            self.feature_list.append(len(encoder.classes_))
            self.train[column] = encoder.transform(self.train[column])
            self.test[column] = encoder.transform(self.test[column])
        self.user_size = self.feature_list[0]
        self.movie_size = self.feature_list[1]

    def get_train(self):
        target = self.train.pop("rating")
        if self.only_rating:
            return tf.data.Dataset.from_tensor_slices((self.train[["user_id", "movie_id"]].values, target.values)).batch(self.batch_size).prefetch(self.buffer_size)
        else:
            return tf.data.Dataset.from_tensor_slices((self.train.values, target.values)).batch(self.batch_size).prefetch(self.buffer_size)

    def get_test(self):
        target = self.test.pop("rating")
        if self.only_rating:
            return tf.data.Dataset.from_tensor_slices(
                (self.test[["user_id", "movie_id"]].values, target.values)).batch(self.batch_size).prefetch(
                self.buffer_size)
        else:
            return tf.data.Dataset.from_tensor_slices((self.test.values, target.values)).batch(
                self.batch_size).prefetch(self.buffer_size)
