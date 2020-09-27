import tensorflow as tf
import tensorflow.keras as keras
import argparse
from data import Data
from layers import FMLayer, ResidualLayer

parser = argparse.ArgumentParser()
parser.add_argument("--result",type=str,default='../results/result.txt')
parser.add_argument("--file",type=str,default='E:/project/rec_movielens/data/')
parser.add_argument("--embed_size",type=int,default=128)
parser.add_argument("--lr", type=float,default=1e-3)
parser.add_argument("--batch",type=int,default=1024)
parser.add_argument("--epochs",type=int,default=10)

arg = parser.parse_args()


class DeepFM(keras.Model):
    def __init__(self, feature_list, k=10, hidden_unit=128, hidden_number=3):
        super(DeepFM, self).__init__()
        self.fm = FMLayer(k)
        self.deep = keras.Sequential()
        for unit in range(hidden_number):
            self.deep.add(ResidualLayer(hidden_unit, hidden_unit*len(feature_list)))
        self.embed_layers = {
            'embed_'+str(j):keras.layers.Embedding(input_dim=i,
                                                   output_dim=hidden_unit,
                                                   embeddings_regularizer=keras.regularizers.L2(0.01))
            for j,i in enumerate(feature_list)
        }
        self.dense = keras.layers.Dense(1, activation=None)
        self.w1 = self.add_weight(name='wide_weight',
                                  shape=(1,),
                                  trainable=True)
        self.w2 = self.add_weight(name='deep_weight',
                                  shape=(1,),
                                  trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(1,),
                                    trainable=True)

    def call(self, inputs):
        stack = list()
        for i, v in enumerate(inputs):
            stack.append(self.embed_layers['embed_'+str(i)](v))
        stack = tf.concat(stack,axis=-1)
        wide_outputs = self.fm(stack)
        deep_outputs = self.deep(stack)
        deep_outputs = self.dense(deep_outputs)
        outputs = tf.nn.sigmoid(tf.add(tf.add(self.w1*wide_outputs, self.w2*deep_outputs), self.bias))
        return outputs


data = Data(filepath=arg.file, batch_size=arg.batch)
dfm = DeepFM(data.feature_list)
dfm.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.SGD(arg.lr), metrics=[keras.metrics.RootMeanSquaredError()])
dfm.fit(data.get_train(), epochs=arg.epochs)
loss, rmse = dfm.evaluate(data.get_test())
print("RMSE:{:.4f}".format(rmse))
