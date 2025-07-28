import nn
import numpy as np

class Loss:
    class BinaryCrossentropy:
        @staticmethod
        def forward(y_true, y_pred):
            return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
        @staticmethod
        def backward(y_true, y_pred):
            return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)

    class MSELoss:      
        @staticmethod
        def forward(y_true, y_pred):
            return np.mean(np.power(y_true-y_pred, 2))
        
        @staticmethod
        def backward(y_true, y_pred):
            return 2 * (y_pred-y_true) / np.size(y_true)
    
