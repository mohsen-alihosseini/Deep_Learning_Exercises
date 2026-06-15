import numpy as np
import Layers.Base as Base

class Pooling(Base.BaseLayer):
    
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape 
        self.pooling_shape = pooling_shape

    def forward(self, input_tensor):
        self.in_shape = input_tensor.shape
        
        #  Output Size = [Input Size - pooling size]/Stride   + 1
        out_shape = input_tensor.shape[:2] + ((input_tensor.shape[2]-self.pooling_shape[0])//self.stride_shape[0]+1, (input_tensor.shape[3]-self.pooling_shape[1])//self.stride_shape[1]+1)
        out_f = np.zeros(out_shape)
        location = np.zeros(out_shape, dtype=tuple)
        # print(self.stride_shape, self.pooling_shape)
        # print(input_tensor.shape)
        # print(out_shape)


        # tensor [  b,      k,     m,     n]
        #         image, channel, row, column
        for b in range(input_tensor.shape[0]):
            for k in range(input_tensor.shape[1]):
                for m in range(out_shape[2]):
                    for n in range(out_shape[3]):
                        temp = input_tensor[b,k, m*self.stride_shape[0]:m*self.stride_shape[0]+self.pooling_shape[0], n*self.stride_shape[1]:n*self.stride_shape[1]+self.pooling_shape[1]]
                        out_f[b,k,m,n] = np.max(temp)
                        location[b,k,m,n] = np.unravel_index(np.argmax(temp),temp.shape)
        self.location = location
        return out_f


    def backward(self, error_tensor):
        out_b = np.zeros(self.in_shape)
        for b in range(error_tensor.shape[0]):
            for k in range(error_tensor.shape[1]):
                for m in range(error_tensor.shape[2]):
                    for n in range(error_tensor.shape[3]):
                        # print("***", self.location[b,k,m,n][0], self.location[b,k,m,n][1])
                        out_b[b,k,self.location[b,k,m,n][0]+m*self.stride_shape[0],self.location[b,k,m,n][1]+n*self.stride_shape[1]] += error_tensor[b,k,m,n]
        return out_b