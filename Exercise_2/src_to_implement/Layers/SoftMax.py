import numpy as np
import Layers.Base as Base

class SoftMax(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input = input_tensor
        max_row = np.max(self.input, axis = 1)
        input_tensor = input_tensor - max_row.reshape((-1,1))
        input_tensor = np.exp(input_tensor)
        self.output = input_tensor = input_tensor/np.sum(input_tensor, axis=1).reshape((-1,1))
        return self.output
    
    def backward(self, error_tensor):
        error_tensor = error_tensor - np.sum(np.multiply(error_tensor, self.output), axis = 1).reshape((-1,1))
        return np.multiply(self.output, error_tensor)