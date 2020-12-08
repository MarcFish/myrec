import tensorflow as tf
import tensorflow.keras as keras


class CINLayer(keras.layers.Layer):
    def __init__(self, units=[32, 32, 32, 32]):
        self.units = units
        super(CINLayer, self).__init__()

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise Exception("dim wrong")
        self.ws = list()
        self.bs = list()
        self.field_nums = [input_shape[1]]
        for unit in self.units:
            w = self.add_weight(shape=(unit, self.field_nums[0], self.field_nums[-1]))
            b = self.add_weight(shape=(unit,))
            self.ws.append(w)
            self.bs.append(b)
            self.field_nums.append(unit)
        super(CINLayer, self).__init__(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape[0], sum(self.units)

    def call(self, inputs):
        hidden_inputs = [inputs]
        for w, b in zip(self.ws, self.bs):
            out = tf.einsum("bid, bmd, jim->bjd", hidden_inputs[0], hidden_inputs[-1], w)
            out = tf.nn.bias_add(out, b, data_format="NC...")
            out = tf.nn.leaky_relu(out)
            hidden_inputs.append(out)
        out = tf.concat(hidden_inputs[1:], axis=1)
        out = tf.reduce_sum(out, axis=-1, keepdims=False)
        return out


class FMLayer(keras.layers.Layer):
    def __init__(self):
        super(FMLayer, self).__init__()

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise Exception("dim wrong")
        super(FMLayer, self).__init__(input_shape)

    def call(self, inputs):
        square_of_sum = tf.square(tf.reduce_sum(
            inputs, axis=1, keepdims=True))
        sum_of_square = tf.reduce_sum(
            inputs * inputs, axis=1, keepdims=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * tf.reduce_sum(cross_term, axis=2, keepdims=False)

        return cross_term

