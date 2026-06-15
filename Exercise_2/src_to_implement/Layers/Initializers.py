import numpy as np


class Constant():
    def __init__(self, const = 0.1):
        self.const = const

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.const)


class UniformRandom():
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.random(weights_shape)


class Xavier():
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/(fan_in+fan_out))
        return np.random.normal(0, sigma, weights_shape)


class He():
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/fan_in)
        return np.random.normal(0, sigma, weights_shape)