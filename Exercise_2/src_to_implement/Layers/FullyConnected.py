import numpy as np
import Layers.Base as Base

class FullyConnected(Base.BaseLayer):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(self.input_size+1, self.output_size) # creating w.T with parameters of each neuron on each column
        # print(self.w)
        self.optimizer = None  # Setter getter property !!!
        self._optimizer = None # corresponding inner variable
        self.input = None
        self.output = None
        self.error = None
        self.gradient_weights = None

    ########## Newly added ##########
    def initialize(self, weights_initializer, bias_initializer):
        w = weights_initializer.initialize(self.weights[:-1,:].shape, self.input_size, self.output_size)
        b = bias_initializer.initialize(self.weights[-1,:].shape, self.input_size, self.output_size).reshape(1,-1)
        # print(w.shape, b.shape)
        self.weights = np.concatenate([w,b], axis=0)
    ########## Newly added ##########

    def forward(self, input_tensor):
        # self.batch_size = input_tensor.shape[0]
        self.input = np.concatenate([input_tensor, np.ones((input_tensor.shape[0],1))], axis=1) # Addding a column of ones for bias of neurons
        self.output = np.dot(self.input, self.weights)
        return self.output
    
    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def backward(self, error_tensor):
        self.error = error_tensor
        error_previous = np.dot(self.error, self.weights[:-1,:].T)
        # self.gradient_weights = np.dot(self.input.T, np.tile(self.error, reps=[1,self.batch_size]))
        self.gradient_weights = np.dot(self.input.T, self.error)
        self.weights = self.calculate_update(self.weights, self.gradient_weights)
        return error_previous
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self._optimizer is not None:
            new_weights = self._optimizer.calculate_update(weight_tensor, gradient_tensor)
            return new_weights
        else:
            return self.weights