import tensorflow as tf

from .param import ConvTasNetParam
from typing import List


class Encoder(tf.keras.layers.Layer):
    """Encodes the input (`x`, p.1258) into a mixture representation (`w`, p.1258).

    Attributes:
        U: Encoder basis functions (p. 1258)
    """

    def __init__(self, param: ConvTasNetParam):
        super(Encoder, self).__init__()
        self.param = param
        self.U = tf.keras.layers.Dense(
            self.param.N, activation=self.param.Ha)

    def call(self, inputs):
        return self.U(inputs)

    def compute_output_shape(self, input_shape):
        return self.U.compute_output_shape(input_shape)

    def compute_output_signature(self, input_signature: tf.TensorSpec) -> tf.TensorSpec:
        return self.U.compute_output_signature(input_signature)

    def get_config(self):
        return self.param.get_config()


class Decoder(tf.keras.layers.Layer):
    """Restores the waveform from the mixture representation.

    Attributes:
        V: Decoder basis functions (p. 1258)
    """

    def __init__(self, param: ConvTasNetParam):
        super(Decoder, self).__init__()
        self.param = param
        self.V = tf.keras.layers.Dense(
            self.param.L, activation="linear")

    def call(self, inputs):
        return self.V(inputs)

    def compute_output_shape(self, input_shape):
        return self.V.compute_output_shape(input_shape)

    def compute_output_signature(self, input_signature: tf.TensorSpec) -> tf.TensorSpec:
        return self.V.compute_output_signature(input_signature)

    def get_config(self):
        return self.param.get_config()


class GlobalNormalization(tf.keras.layers.Layer):
    """Implements `gLN(F)` defined in page 1259.

    Attributes:
        epsilon: Small constant for numerical stability (p. 1259)
        gamma: Trainable parameters (p. 1259)
        beta: Trainable parameters (p. 1259)
    """

    def __init__(self, epsilon: float):
        super(GlobalNormalization, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape: tf.TensorShape):
        new_shape = (int(input_shape[-1]), )
        self.gamma = self.add_weight(
            "gamma", shape=new_shape, initializer="glorot_uniform")
        self.beta = self.add_weight(
            "beta", shape=new_shape, initializer="glorot_uniform")

    def compute_output_signature(self, input_signature: tf.TensorSpec) -> tf.TensorSpec:
        return input_signature

    def call(self, inputs):
        mean = tf.math.reduce_mean(inputs)
        var = tf.math.reduce_variance(inputs)
        return ((inputs - mean) / tf.math.sqrt(var + self.epsilon)) * self.gamma + self.beta


class ConvBlock(tf.keras.layers.Layer):
    """Implmenets 1-D Conv block design (p. 1258)

    Attributes:
        conv_bottle: Bottleneck convolution followed by `prelu_bottle`
        prelu_bottle: PReLU followed by `norm_bottle`
        norm_bottle: gLN or cLN followed by the D-conv block
        dconv: Depthwise convolution
        prelu_sconv: PReLU after `dconv`
        norm_sconv: gLN or cLN after `prelu_sconv`
        conv_skip: 1x1-conv block on the skip-connection path
        conv_output: 1x1-conv block on the residual path
        merge_output: A merge layer on the residual path
    """

    def __init__(self, param: ConvTasNetParam, dilation: int):
        super(ConvBlock, self).__init__()
        self.param = param
        self.dilation = dilation
        self.conv_bottle = tf.keras.layers.Conv1D(
            filters=self.param.H, kernel_size=1, activation="linear")
        self.prelu_bottle = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.norm_bottle = GlobalNormalization(self.param.epsilon)
        self.dconv = tf.keras.layers.Conv1D(
            filters=self.param.H,
            kernel_size=self.param.P,
            padding="same",
            dilation_rate=dilation,
            activation="linear",
            groups=self.param.H)
        self.prelu_sconv = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.norm_sconv = GlobalNormalization(self.param.epsilon)
        self.conv_skip = tf.keras.layers.Conv1D(
            filters=self.param.Sc, kernel_size=1, activation="linear")
        self.conv_output = tf.keras.layers.Conv1D(
            filters=self.param.B, kernel_size=1, activation="linear")
        self.merge_output = tf.keras.layers.Add()

    def call(self, inputs):
        outputs = self.conv_bottle(inputs)
        outputs = self.prelu_bottle(outputs)
        outputs = self.norm_bottle(outputs)
        outputs = self.dconv(outputs)
        outputs = self.prelu_sconv(outputs)
        outputs = self.norm_sconv(outputs)
        result_skip = self.conv_skip(outputs)
        result_output = self.conv_output(outputs)
        result_output = self.merge_output([inputs, result_output])
        return [result_skip, result_output]

    def compute_output_signature(self, input_signature: tf.TensorSpec) -> List[tf.TensorSpec]:
        output = self.conv_bottle.compute_output_signature(input_signature)
        output = self.prelu_bottle.compute_output_signature(output)
        output = self.norm_bottle.compute_output_signature(output)
        output = self.prelu_sconv.compute_output_signature(output)
        output = self.norm_sconv.compute_output_signature(output)
        result_skip = self.conv_skip.compute_output_signature(output)
        result_output = self.conv_output.compute_output_signature(output)
        return [result_skip, result_output]

    def get_config(self):
        d = self.param.get_config()
        d["dilation"] = self.dilation
        return d


class Separation(tf.keras.layers.Layer):
    """Implmements temporal convolutional network separation module (p. 1258)

    Attributes:
        layer_norm: LayerNormalization before `initial_conv`
        initial_conv: Initial 1x1 convolution
        conv_blocks: 1-D Conv blocks
        skip_conn: Merge layer for skip-connection
        prelu: PReLU before the final 1x1 convolution
        final_conv: Final 1x1 convolution
        final_reshape: Splits the result by channels
    """

    def __init__(self, param: ConvTasNetParam):
        super(Separation, self).__init__()
        self.param = param
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.initial_conv = tf.keras.layers.Conv1D(
            filters=self.param.B, kernel_size=1, activation="linear")
        self.conv_blocks = []
        for r in range(self.param.R):
            for x in range(self.param.X):
                self.conv_blocks.append(ConvBlock(param, 2 ** x))
        self.skip_conn = tf.keras.layers.Add()
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.final_conv = tf.keras.layers.Conv1D(
            filters=self.param.N * self.param.C, kernel_size=1, activation="sigmoid")
        self.final_reshape = tf.keras.layers.Reshape(
            target_shape=(self.param.THat, self.param.C, self.param.N)
        )

    def compute_output_signature(self, input_signature: tf.TensorSpec):
        shape = input_signature.shape
        new_shape = [*shape[:-1], self.param.C, shape[-1]]
        return tf.TensorSpec(shape=new_shape, dtype=input_signature.dtype)

    def call(self, inputs):
        output = self.layer_norm(inputs)
        output = self.initial_conv(output)
        skip_outputs = []
        for block in self.conv_blocks:
            skip, output = block(output)
            skip_outputs.append(skip)
        output = self.skip_conn(skip_outputs)
        output = self.prelu(output)
        output = self.final_conv(output)
        output = self.final_reshape(output)
        return output

    def get_config(self):
        return self.param.get_config()
