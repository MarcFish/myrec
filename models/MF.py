import tensorflow as tf
import tensorflow.keras as keras
import argparse
from data import Data

parser = argparse.ArgumentParser()
parser.add_argument("--result",type=str,default='../results/result.txt')
parser.add_argument("--lr_u",type=float,default=0.01)
parser.add_argument("--lr_i",type=float,default=0.01)
parser.add_argument("--embed_size",type=int,default=128)
parser.add_argument("--lr", type=float,default=1e-3)
parser.add_argument("--batch",type=int,default=1024)
parser.add_argument("--epochs",type=int,default=10)

arg = parser.parse_args()


class MF(keras.Model):
    def __init__(self, user_size, item_size, embed_size=128):
        super(MF, self).__init__()
        self.user_size = user_size
        self.item_size = item_size
        self.embed_size = embed_size

        self.user_embeddings = keras.layers.Embedding(input_dim=self.user_size, output_dim=self.embed_size,
                                                      embeddings_regularizer=keras.regularizers.L2(arg.lr_u),
                                                      embeddings_initializer='he_uniform')
        self.item_embeddings = keras.layers.Embedding(input_dim=self.item_size, output_dim=self.embed_size,
                                                      embeddings_regularizer=keras.regularizers.L2(arg.lr_i),
                                                      embeddings_initializer='he_uniform')

    def call(self, inputs):
        user_batch = inputs[0]
        item_batch = inputs[1]
        user_embeddings = self.user_embeddings(user_batch)
        item_embeddings = self.item_embeddings(item_batch)
        return tf.reduce_sum(tf.multiply(user_embeddings, item_embeddings), axis=-1)


data = Data(batch_size=arg.batch, only_rating=True)
mf = MF(user_size=data.user_size, item_size=data.movie_size, embed_size=arg.embed_size)
mf.compile(loss=keras.losses.MSE, optimizer=keras.optimizers.SGD(arg.lr), metrics=[keras.metrics.RootMeanSquaredError()])
mf.fit(data.get_train(), epochs=arg.epochs)
loss, rmse = mf.evaluate(data.get_test())
print("RMSE:{:.4f}".format(rmse))
with open(arg.result, 'a', encoding='utf-8', newline='') as f:
    f.write("MF,{:.4f}\n".format(rmse))