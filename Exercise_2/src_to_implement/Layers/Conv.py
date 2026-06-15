import numpy as np
from scipy.signal import convolve, correlate
import Layers.Base as Base
import copy

class Conv(Base.BaseLayer):
    
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.stride_shape = stride_shape if (len(convolution_shape) == 2 or isinstance(stride_shape, tuple)) else (stride_shape, stride_shape)
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.ndim = len(convolution_shape) # either 2 or 3
        self.weights = np.random.rand(self.num_kernels * np.prod(self.convolution_shape)).reshape((self.num_kernels,) + self.convolution_shape)
        self.bias = np.random.rand(self.num_kernels)
        self.gradient_weights = np.zeros_like(self.weights) 
        self.gradient_bias = np.zeros_like(self.bias)
        self.in_shape = 0
        self.optimizer = None  # Setter getter property !!!
        self._optimizer = None # corresponding inner variable
        self._optimizer_b = None # corresponding inner variable for bias

    def forward(self, input_tensor):
        self.in_shape = input_tensor.shape
        self.input = input_tensor
        pad1 = input_tensor.shape[:-1] + (self.convolution_shape[-1]//2,)
        pad2 = input_tensor.shape[:-1] + (int((self.convolution_shape[-1]-0.5)//2),)
        # print(pad1,pad2)
        input_tensor = np.concatenate([np.zeros(pad1), input_tensor, np.zeros(pad2)], axis=self.ndim)
        if self.ndim == 3:
            pad1 = input_tensor.shape[:-2] + (self.convolution_shape[-2]//2,) + (input_tensor.shape[-1],)
            pad2 = input_tensor.shape[:-2] + (int((self.convolution_shape[-2]-0.5)//2),) + (input_tensor.shape[-1],)
            input_tensor = np.concatenate([np.zeros(pad1), input_tensor, np.zeros(pad2)], axis=self.ndim-1)
        self.input_pad = input_tensor
        # print("final input shape: ",input_tensor.shape)
        
        # out = -1*convolve(input_tensor, self.weights[0].reshape((1,) + self.weights[0].shape), mode='valid', method='direct')
        # for k in self.weights[1:]:
        #     out = np.concatenate([out, -1*convolve(input_tensor, k.reshape((1,) + k.shape), mode='valid', method='direct')], axis = 1)
        if self.ndim == 2:
            out_dim = (self.in_shape[0], self.num_kernels) + (int(np.ceil(self.in_shape[2]/self.stride_shape[0])), )
        else:
            out_dim = (self.in_shape[0], self.num_kernels) + (int(np.ceil(self.in_shape[2]/self.stride_shape[0])), int(np.ceil(self.in_shape[3]/self.stride_shape[1])))
        out = np.zeros(out_dim)
        for s,sample in enumerate(input_tensor):
            for f,filter in enumerate(self.weights):
                if self.ndim == 2:
                    # temp = convolve(sample, filter[:,::-1], mode='valid', method='direct').reshape(self.in_shape[2:])
                    temp = convolve(sample, filter, mode='valid', method='direct').reshape(self.in_shape[2:])
                    out[s,f] = temp[::self.stride_shape[0]]
                else:
                    # temp = convolve(sample, filter[:,::-1,::-1], mode='valid', method='direct').reshape(self.in_shape[2:])
                    temp = correlate(sample, filter, mode='valid', method='direct').reshape(self.in_shape[2:])
                    out[s,f] = temp[::self.stride_shape[0],::self.stride_shape[1]]
        # print("After weight multiplication: ", out.shape)

        if self.ndim == 2:
            out = out + np.tile(self.bias, (input_tensor.shape[0], 1)).reshape((input_tensor.shape[0], self.num_kernels, 1))
        else:
            out = out + np.tile(self.bias, (input_tensor.shape[0], 1)).reshape((input_tensor.shape[0], self.num_kernels, 1, 1))
        # print("After bias addition: ", out.shape)
        return out
        

    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer_b = copy.deepcopy(optimizer)


    def backward(self, error_tensor):
        self.error = error_tensor
        Grad_ll = np.zeros((error_tensor.shape[0],) + self.in_shape[1:])
        
        if self.ndim == 2:
            temp = np.zeros(error_tensor.shape[:2] + (self.in_shape[2],))
            temp[:,:,::self.stride_shape[0]] = error_tensor
        else:
            temp = np.zeros(error_tensor.shape[:2] + self.in_shape[2:])
            temp[:,:,::self.stride_shape[0],::self.stride_shape[1]] = error_tensor
        error_tensor = temp
        self.error_stride = error_tensor

        pad1 = error_tensor.shape[:-1] + (self.convolution_shape[-1]//2,)
        pad2 = error_tensor.shape[:-1] + (int((self.convolution_shape[-1]-0.5)//2),)
        error_tensor = np.concatenate([np.zeros(pad1), error_tensor, np.zeros(pad2)], axis=self.ndim)
        if self.ndim == 3:
            pad1 = error_tensor.shape[:-2] + (self.convolution_shape[-2]//2,) + (error_tensor.shape[-1],)
            pad2 = error_tensor.shape[:-2] + (int((self.convolution_shape[-2]-0.5)//2),) + (error_tensor.shape[-1],)
            error_tensor = np.concatenate([np.zeros(pad1), error_tensor, np.zeros(pad2)], axis=self.ndim-1)
        # print("final error shape (padding): ",error_tensor.shape)

        for s, error_s in enumerate(error_tensor): #(self.error_stride) or (error_tensor):
            for k in range(self.convolution_shape[0]):
                if self.ndim == 2:
                    # print(error_tensor.shape, self.in_shape)
                    # print(error_s.shape, self.weights[:,k,::-1].shape, Grad_ll[s,k].shape)
                    # Grad_ll[s,k] = convolve(error_s, self.weights[:,k,::-1], mode='valid', method='direct')
                    Grad_ll[s,k] = correlate(error_s, self.weights[:,k,::-1], mode='valid', method='direct')
                else:
                    # Grad_ll[s,k] = convolve(error_s, self.weights[:,k,::-1,::-1], mode='valid', method='direct')
                    Grad_ll[s,k] = correlate(error_s, self.weights[:,k,::-1,::-1], mode='valid', method='direct')

                    # for n in range(self.num_kernels):  ## mode=same implementation
                    #     Grad_ll[s,k] += convolve(error_s[n], self.weights[n,k,:,:], mode='same', method='direct')
                    #     Grad_ll[s,k] += correlate(error_s[n], self.weights[n,k,::-1,::-1], mode='same', method='direct')
                    
                    # for i in range(Grad_ll[s,k].shape[0]):
                    #     for j in range(Grad_ll[s,k].shape[1]):
                    #         # print(s,k,i,j, error_s[:,i:self.convolution_shape[1], j:self.convolution_shape[2]].shape, self.weights[:,k,::-1,::-1].shape)
                    #         Grad_ll[s,k,i,j] = np.sum(np.multiply(error_s[:,i:i+self.convolution_shape[1], j:j+self.convolution_shape[2]], self.weights[:,k,::-1,::-1]), axis=(0,1,2))
                    
        # print("final grad_ll shape: ",Grad_ll.shape)

        Grad_w = np.zeros((error_tensor.shape[0],)+self.weights.shape)
        # print(self.input_pad.shape, self.error.shape)
        for s, (input_s, error_s) in enumerate(zip(self.input_pad, self.error_stride)):
            for f in range(self.error.shape[1]):    #(self.convolution_shape[0]):
                # print(input_s.shape, error_s[f:f+1].shape)
                # Grad_w[s,f] = convolve(input_s, error_s[f:f+1], mode='valid', method='direct')
                Grad_w[s,f] = correlate(input_s, error_s[f:f+1], mode='valid', method='direct')
        self.gradient_weights = np.sum(Grad_w, axis=0)
        # print("final grad_w shape: ",self.gradient_weights.shape)
        self.weights = self.calculate_update(self.weights, self.gradient_weights)
        
        if self.ndim == 2:
            self.gradient_bias = np.sum(self.error, axis=(0,2))
        else:
            self.gradient_bias = np.sum(self.error, axis=(0,2,3))
        # self.bias = self._optimizer_b.calculate_update(self.bias, self.gradient_bias)
        # print('Bias gradient: ', self.gradient_bias.shape)
        self.bias = self.calculate_update(self.bias, self.gradient_bias, True)
        # print('Bias shape: ', self.bias.shape)

        return Grad_ll

    def initialize(self, weights_initializer, bias_initializer):
        # print(self.weights.shape, self.in_shape, self.convolution_shape, np.prod(self.convolution_shape))
        self.weights = weights_initializer.initialize(self.weights.shape, np.prod(self.convolution_shape), self.num_kernels*np.prod(self.convolution_shape[1:]))
        self.bias = bias_initializer.initialize(self.bias.shape, np.prod(self.convolution_shape), self.num_kernels*np.prod(self.convolution_shape[1:]))


    def calculate_update(self, weight_tensor, gradient_tensor, isbias=False):
        if isbias and self._optimizer_b is not None:
            return self._optimizer_b.calculate_update(weight_tensor, gradient_tensor)
        elif (not isbias) and self._optimizer is not None:
            return self._optimizer.calculate_update(weight_tensor, gradient_tensor)
        else:
            return weight_tensor