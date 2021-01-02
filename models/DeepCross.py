import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_addons as tfa
import argparse
from data import Data
from layers import CrossLayer

parser = argparse.ArgumentParser()
parser.add_argument("--result",type=str,default='../results/result.txt')
parser.add_argument("--file",type=str,default='E:/project/myrec/data/')
parser.add_argument("--embed_size",type=int,default=32)
parser.add_argument("--lr", type=float,default=1e-3)
parser.add_argument("--l2", type=float,default=1e-4)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--batch",type=int,default=1024)
parser.add_argument("--epochs",type=int,default=10)

arg = parser.parse_args()


class DeepCross(keras.Model):
    def __init__(self, feature_list, cross_layer_num=6, hidden_unit=32, hidden_number=3):
        super(DeepCross, self).__init__()
        self.cross = CrossLayer(cross_layer_num)
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
        self.w = keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        stack = list()
        for i, v in enumerate(tf.split(inputs, inputs.shape[-1], axis=-1)):
            stack.append(tf.squeeze(self.embed_layers['embed_'+str(i)](v), axis=1))
        concat = tf.concat(stack, axis=-1)  # batch, feature_num * embed_size

        wide_outputs = self.cross(concat)
        deep_outputs = self.deep(concat)
        outputs = self.w(tf.concat([wide_outputs, deep_outputs], axis=-1))
        return outputs


data = Data(filepath=arg.file, batch_size=arg.batch)
dcn = DeepCross(data.feature_list)
dcn.compile(loss=keras.losses.MSE, optimizer=tfa.optimizers.Lookahead(tfa.optimizers.AdamW(learning_rate=arg.lr, weight_decay=arg.l2)), metrics=[keras.metrics.MeanSquaredError()])
dcn.fit(data.get_train(), epochs=arg.epochs)
loss, metric = dcn.evaluate(data.get_test())
print("mse:{:.4f}".format(metric))
