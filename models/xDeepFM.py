import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import argparse
from data import Data
from layers import CINLayer

parser = argparse.ArgumentParser()
parser.add_argument("--result",type=str,default='../results/result.txt')
parser.add_argument("--file",type=str,default='E:/project/rec_movielens/data/')
parser.add_argument("--embed_size",type=int,default=32)
parser.add_argument("--lr", type=float,default=1e-3)
parser.add_argument("--l2", type=float,default=1e-4)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--batch",type=int,default=1024)
parser.add_argument("--epochs",type=int,default=10)

arg = parser.parse_args()


class xDeepFM(keras.Model):
    def __init__(self, feature_list, cin_layer_num=8, hidden_unit=64, hidden_number=3):
        super(xDeepFM, self).__init__()
        self.cin = CINLayer([hidden_unit]*cin_layer_num)
        self.deep = keras.Sequential()
        for unit in range(hidden_number):
            self.deep.add(keras.layers.Dense(hidden_unit))
            self.deep.add(keras.layers.Dropout(arg.dropout))
            self.deep.add(keras.layers.LayerNormalization())
            self.deep.add(keras.layers.LeakyReLU(0.2))
        self.embed_layers = {
            'embed_'+str(j):keras.layers.Embedding(input_dim=i,
                                                   output_dim=hidden_unit)
            for j,i in enumerate(feature_list)
        }
        self.deep_dense = keras.layers.Dense(1, activation=None)
        self.cin_dense = keras.layers.Dense(1, activation=None)
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
            stack.append(tf.squeeze(self.embed_layers['embed_'+str(i)](v), axis=1))
        concat = tf.concat(stack, axis=-1)  # batch, feature_num * embed_size
        stack = tf.stack(stack, axis=1)  # batch, feature_num, embed_size
        wide_outputs = self.cin(stack)
        wide_outputs = self.cin_dense(wide_outputs)
        deep_outputs = self.deep(concat)
        deep_outputs = self.deep_dense(deep_outputs)
        outputs = tf.nn.sigmoid(tf.add(tf.add(self.w1*wide_outputs, self.w2*deep_outputs), self.bias))
        return outputs


data = Data(filepath=arg.file, batch_size=arg.batch)
xdfm = xDeepFM(data.feature_list)
xdfm.compile(loss=keras.losses.binary_crossentropy, optimizer=tfa.optimizers.Lookahead(tfa.optimizers.AdamW(learning_rate=arg.lr, weight_decay=arg.l2)), metrics=[keras.metrics.MeanSquaredError()])
xdfm.fit(data.get_train(), epochs=arg.epochs)
loss, metric = xdfm.evaluate(data.get_test())
print("mse:{:.4f}".format(metric))
