import numpy as np

class CrossEntropyLoss():
    def __init__(self):
        pass

    def forward(self, prediction_tensor, label_tensor):
        self.prediction = prediction_tensor
        temp = np.sum(np.multiply(prediction_tensor, label_tensor), axis = 1) + np.finfo(float).eps
        temp = np.sum(-1*np.log(temp))
        return temp

    def backward(self, label_tensor):
        return np.divide(-1 * label_tensor, self.prediction + np.finfo(float).eps)