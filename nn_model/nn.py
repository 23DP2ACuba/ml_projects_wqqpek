import numpy as np

class nn:
    class Layer:
        def __init__(self):
            self.input = None
            self.output = None
            
    class Dense(Layer):
        def __init__(self, input_size, output_size):
            super().__init__()
            self.weights = np.random.randn(output_size, input_size)
            self.bias =  np.random.randn(output_size, 1)
            
        def forward(self, input):
            self.input = input
            return np.dot(self.weights, self.input) + self.bias
        
        def backward(self, output_gradient, learning_rate):
            weights_gradient = np.dot(output_gradient, self.input.T)
            self.weights -= learning_rate * weights_gradient
            self.bias -= learning_rate * output_gradient
            return np.dot(self.weights.T, output_gradient)
    
    class Activation(Layer):
        def __init__(self, activation, activation_derivative):
            self.activation = activation
            self.activation_derivative = activation_derivative
            
        def forward(self, input):
            self.input = input
            return self.activation(self.input)
        
        def backward(self, output_gradient, lr):
            return np.multiply(output_gradient, self.activation_derivative(self.input))
        
    class Tanh(Activation):
        def __init__(self):
            tanh = lambda x: np.tanh(x)
            tanh_prime = lambda x: 1 - np.tanh(x) ** 2
            super().__init__(tanh, tanh_prime)
            
    class ReLU(Activation):
        def __init__(self):
            relu = lambda x: np.maximum(0, x)
            relu_derivative = lambda x: (x > 0).astype(float)
            super().__init__(relu, relu_derivative)
    
    class Sigmoid(Activation):
        def __init__(self):
            sigmoid = lambda x: 1 / (1 + np.exp(-x))
            def sigmoid_derivative(x):
                s = sigmoid(x)
                return s * (1 - s)
            super().__init__(sigmoid, sigmoid_derivative)
    
    class WSigmoid(Activation):
        def __init__(self):
            sigmoid = lambda x: 1 / (1 + np.exp(-x))*x
            def sigmoid_derivative(x):
                s = sigmoid(x)
                return s * (1 + x * (1 - s))
            super().__init__(sigmoid, sigmoid_derivative)
            
