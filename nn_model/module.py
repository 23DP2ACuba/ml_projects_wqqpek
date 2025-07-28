from nn import nn
from losses import Loss
import numpy as np

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [0], [1]], (4, 1, 1))
model = [
        nn.Dense(2, 3),
        nn.WSigmoid(),
        nn.Dense(3, 1),
        nn.WSigmoid()
    ]

epochs = 15
lr = 0.1

for e in range(epochs):
    error = 0
    
    for x, y in zip(X, Y):
        output = x
        for layer in model:
            output = layer.forward(output)
        
        error += Loss.BinaryCrossentropy.forward(y, output)
        grad = Loss.BinaryCrossentropy.backward(y, output)
        
        for layer in reversed(model):
            grad = layer.backward(grad, lr)
            
    error /= len(X)
    print(f"error: {error}")
        
        