import numpy as np
import Layers.Base as Base

class ReLU(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input = input_tensor
        return np.where(self.input > 0, self.input, 0)
    
    def backward(self, error_tensor):
        return np.multiply(np.where(self.input > 0, 1, 0), error_tensor)