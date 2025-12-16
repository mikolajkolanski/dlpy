import math
import numpy as np

class Linear:
    def __init__(self, in_n, n_neurons):
        self.ws = np.random.rand(n_neurons, in_n) # [n_neurons, in_n]
        self.bias = np.random.rand(n_neurons, 1)

        # print(self.bias)
    
    def forward(self, x):
        # Assume x is of size [in_n]

        x = np.matmul(self.ws, x) + self.bias
        return x
    
    def grads(self, inp: np.matrix, target: np.matrix):
        # Target is of size [n_neurons]
        d_y = self.forward(inp) - target

        grads = np.matmul(d_y, inp.transpose())

        return grads

# Test

l = Linear(3,5)

targ = np.matrix([0,0,0,1,5]).transpose()
x = np.matrix([1,2,3]).transpose()
for i in range(100):
    print(l.forward(x))
    g = l.grads(x, targ)
    l.ws = l.ws - g*0.001