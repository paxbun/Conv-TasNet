# Copyright (c) 2021 Chanjung Kim. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import math


class SiSNR(tf.keras.losses.Loss):
    """Implements the SiSNR function.

    Attributes:
        epsilon: A small constant for numerical stability
    """

    def __init__(self, epsilon: float = 1e-10):
        super(SiSNR, self).__init__()
        self.epsilon = epsilon

    def call(self, s, s_hat):
        s_target = s * (tf.reduce_sum(tf.multiply(s, s_hat)) /
                        tf.reduce_sum(tf.multiply(s, s)))
        e_noise = s_hat - s_target
        result = 20 * tf.math.log(tf.norm(e_noise) /
                                  (tf.norm(s_target + self.epsilon) + self.epsilon)) / math.log(10)
        return result


class SDR(tf.keras.losses.Loss):
    """Implements the SDR function.

    Attributes:
        epsilon: A small constant for enumerical stability
    """

    def __init__(self, epsilon: float = 1e-10, **kwargs):
        super(SDR, self).__init__(**kwargs)
        self.epsilon = epsilon

    def call(self, s, s_hat):
        return 20 * tf.math.log(tf.norm(s_hat - s) / (tf.norm(s) + self.epsilon) + self.epsilon) / math.log(10)
