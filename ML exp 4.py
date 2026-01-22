import numpy as np
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])
np.random.seed(1)
W1 = np.random.rand(2,2)
W2 = np.random.rand(2,1)
def sigmoid(x):
    return 1/(1+np.exp(-x))
for _ in range(5000):
    h = sigmoid(np.dot(X, W1))
    o = sigmoid(np.dot(h, W2))

    error = y - o
    d_o = error * o * (1-o)
    d_h = d_o.dot(W2.T) * h * (1-h)

    W2 += h.T.dot(d_o) * 0.1
    W1 += X.T.dot(d_h) * 0.1
print("ANN Output after Training:")
print(np.round(o, 2))
