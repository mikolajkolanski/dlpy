import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from nn import *

digs = load_digits()
X: np.matrix = digs['data']
y: np.matrix = digs['target']

y = OneHotEncoder().fit_transform(y.reshape(-1, 1))
# print(y)
# exit()
print(X.shape)
print(y.shape)

mlp = MLP(opt=SGD(0.01, 0.99))
mlp.add(Linear(64,15, LeakyReLU(0.2)))
mlp.add(Linear(15,10))

for i in range(100 * 100):
    if i%100==0:
        print('Loss:', mlp.calc_loss(X, y))
    mlp.calc_grads(X, y).grad_cap(1)
    # mlp.grad_cap(0)
    mlp.step()

pred = mlp.ff(X)
tt = 0
ff = 0
for p,t in zip(pred,digs['target']):
    if np.argmin(np.abs(p-1)) == t:
        tt += 1
    else:
        ff += 1
print(tt, ff, tt+ff)
print('Correct:', tt/(tt+ff))
print('Incorect:', ff/(tt+ff))

