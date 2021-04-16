import tensorflow as tf

from .param import ConvTasNetParam
from .layers import Encoder, Decoder, Separation
from .loss import SiSNR


class ConvTasNet(tf.keras.Model):
    """Implements Conv-TasNet.

    Attributes:
        encoder: The encoder module (p. 1258)
        separation: The separation module (p. 1258)
        encoded_reshape: Reshapes the repeated mixture representation
        mask_apply: Multiplies the mixture representation and the mask
        decoder: The decoder module (p. 1258)
    """

    @staticmethod
    def make(param: ConvTasNetParam,
             optimizer: tf.keras.optimizers.Optimizer = "adam",
             loss: tf.keras.losses.Loss = SiSNR()):
        """Instantiate and compile a new Conv-TasNet model instance.

        Args:
            param: Hyperparameters of Conv-TasNet
            optimizer: The optimizer function to use
            loss: The loss function to use

        Returns:
            A compiled Conv-TasNet model instance.
        """
        model = ConvTasNet(param)
        model.compile(optimizer=optimizer, loss=loss)
        model.build(input_shape=(None, param.THat, param.L))
        return model

    def __init__(self, param: ConvTasNetParam):
        super(ConvTasNet, self).__init__()
        self.param = param
        self.encoder = Encoder(param)
        self.separation = Separation(param)
        self.encoded_reshape = tf.keras.layers.Reshape(
            target_shape=(self.param.C, self.param.THat, self.param.N))
        self.mask_apply = tf.keras.layers.Multiply()
        self.decoder = Decoder(param)

    def call(self, inputs):
        encoded = self.encoder(inputs)
        separated = self.separation(encoded)
        encoded = tf.keras.layers.concatenate(
            [encoded for i in range(self.param.C)], axis=1)
        encoded = self.encoded_reshape(encoded)
        applied = self.mask_apply([encoded, separated])
        decoded = self.decoder(applied)
        return decoded
