import math
import numpy as np

class LeakyReLU:
    def __init__(self, alpha=0):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)
    def deriv(self, x):
        return np.where(x > 0, 1.0, self.alpha)

class Linear:
    def __init__(self, in_features, n_neurons, activation=None):
        self.ws: np.matrix = np.random.randn(n_neurons, in_features) * np.sqrt(2.0/in_features) # [n_neurons, in_n]
        self.bias: np.matrix = np.zeros((n_neurons, 1)) # [n_neurons, 1]
        self.activation = activation
    
    def set_activation(self, act):
        self.activation = act

        return self
    
    def forward(self, x):
        # Assume x is of size [N, in_n]
        # N = x.shape[0]

        self.inp = x
        self.z = np.matmul(x, self.ws.T) + self.bias.T # [N, n_neurons]
        
        if self.activation:
            return self.activation.forward(self.z)
        return self.z
    
    def grads(self, d_y: np.matrix):
        # Assume d_y is of size [N, n_neurons]

        if self.activation:
            d_y = np.multiply(d_y, self.activation.deriv(self.z))

        dL_dW = np.matmul(d_y.T, self.inp) / d_y.shape[0]
        dL_dB = np.sum(d_y, axis=0).T / d_y.shape[0] # [n_neurons, 1]

        d_x = np.matmul(d_y, self.ws) # [N, in_features] 
        return dL_dW, dL_dB, d_x 

class SGD:
    def __init__(self, lr, momentum=0):
        self.lr = lr
        self.beta = momentum # beta
        self.m = None
    def step(self, grads: np.matrix) -> np.matrix:
        # Grads is list of (dW, dB)
        # Returns change in params

        if self.m == None:
            self.m = []
            for i in grads:
                self.m.append((np.zeros_like(i[0]),
                               np.zeros_like(i[1])))
        else:
            for c, (dW, dB) in enumerate(self.m):
                g_dW, g_dB = grads[c]
                self.m[c] = (self.beta*dW-self.lr*g_dW*(1-self.beta),
                             self.beta*dB-self.lr*g_dB*(1-self.beta))
        # SGD(lr=0.005, momentum=0.5)
        # Update params
        return self.m


class MLP:
    def __init__(self, opt):
        self.layers = []
        self.opt = opt
    
    def add(self, layer):
        self.layers.append(layer)
    
    def ff(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x
    
    def ff_dist(self, x):
        out = [x]
        
        for l in self.layers:
            x = l.forward(x)
            out.append(x)
        return out
    
    def calc_grads(self, x, y):
        xs = self.ff_dist(x)
        
        grads_rev = []

        dY = xs[-1] - y
        
        for i in range(len(self.layers)-1, -1, -1):
            # print(i)
            dW, dB, dY = self.layers[i].grads(dY)
            # print('Layer', i)
            # print(g)
            # print(self.layers[i].ws)
            grads_rev.append((dW, dB))

        grads_rev.reverse()
        self.grads = grads_rev
        # print(grads_rev)
        
        return self

    def calc_loss(self, x, y):
        return np.mean(np.square(self.ff(x) - y))

    def grad_cap(self, cap=1):
        for g in range(len(self.grads)):
            self.grads[g] = (np.clip(self.grads[g][0], -cap, cap),
                             np.clip(self.grads[g][1], -cap, cap)) 

        return self

    def step(self):
        if self.opt:
            opt_grads = self.opt.step(self.grads)

            for (dW, dB),l in zip(opt_grads, self.layers):
                l.ws = l.ws + dW
                l.bias = l.bias + dB
        else:
            for (dW, dB),l in zip(self.grads, self.layers):
                l.ws = l.ws - dW*0.005
                l.bias = l.bias - dB*0.005
        

# Test
if __name__=='__main__':
    mlp = MLP(SGD(lr=0.005, momentum=0.1))
    mlp.add(Linear(3,10, LeakyReLU(0.2)))
    mlp.add(Linear(10,2))

    targ = np.matrix([[0,5], [5,0]])
    # [5,0]
    print(targ.shape)
    x = np.matrix([[1,2,3], [5,6,7]])


    for i in range(1000):
        if i%100==0:
            # print(mlp.ff(x))
            print(mlp.calc_loss(x, targ))
        mlp.calc_grads(x, targ).grad_cap(1)
        # mlp.grad_cap(0)
        mlp.step()