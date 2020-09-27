import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class Data:
    def __init__(self, filepath='./', batch_size=512, only_rating=False, buffer_size=1024):
        self.user_size = 7000
        self.movie_size = 4000
        self.zip_size = 3439
        self.occ_size = 21
        self.cat_size = 18
        self.feature_list = [self.user_size,self.movie_size,2,self.occ_size, self.zip_size]
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.train = tf.data.TFRecordDataset(filepath+'train.tfrecord')
        self.test = tf.data.TFRecordDataset(filepath+'test.tfrecord')
        self.only_rating = only_rating

    def get_train(self):
        if self.only_rating:
            return self.train.map(self.test_parse, AUTOTUNE).batch(self.batch_size).prefetch(self.buffer_size)
        else:
            return self.train.map(self.train_parse, AUTOTUNE).batch(self.batch_size).prefetch(self.buffer_size)

    def get_test(self):
        if self.only_rating:
            return self.test.map(self.test_parse, AUTOTUNE).batch(self.batch_size).prefetch(self.buffer_size)
        else:
            return self.test.map(self.train_parse, AUTOTUNE).batch(self.batch_size).prefetch(self.buffer_size)


    def test_parse(self, example):
        feature_description = {
            'user': tf.io.FixedLenFeature([], tf.int64),
            'movie': tf.io.FixedLenFeature([], tf.int64),
            'rating': tf.io.FixedLenFeature([], tf.float32)
        }
        feature = tf.io.parse_single_example(example, feature_description)
        return (feature['user'], feature['movie']), feature['rating']

    def train_parse(self, example):
        feature_description = {
            'user': tf.io.FixedLenFeature([], tf.int64),
            'movie': tf.io.FixedLenFeature([], tf.int64),
            'rating': tf.io.FixedLenFeature([], tf.float32),
            'gender':tf.io.FixedLenFeature([], tf.int64),
            'age':tf.io.FixedLenFeature([], tf.float32),
            'occupation':tf.io.FixedLenFeature([], tf.int64),
            'zip_code':tf.io.FixedLenFeature([], tf.int64),
            'cats':tf.io.VarLenFeature(tf.int64),
        }

        feature = tf.io.parse_single_example(example, feature_description)
        return (feature['user'], feature['movie'], feature['gender'], feature['occupation'], feature['zip_code']), feature['rating']
