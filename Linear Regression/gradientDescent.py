
import numpy as np

# X - A training set matrix represented as ndarray of shape (N,D)
# y - A target vector of ndarray of shape (N, )
# scaler lr - learning rate value
# num_iter - number of iterations to run the algorithm
# delta - hyperparameter of huber loss 

def gradient_descent(X, y, lr, num_iter, delta):
    w = np.zeros((len(y),1))
    b = np.zeros((len(y),1))
    N = len(y)

    for i in range(0, num_iter):
        f = np.dot(X, w) + b
        for a in np.where(np.abs(w) <= delta):
            if len(a) > 0:
                w[a] = w[a] - (lr/N)*np.dot((f[a]-y[a]),X[a])
                b[a] = f[a] - y[a]
        for c in np.where(np.abs(w) > delta):
            if len(c) > 0:
                w[c] = w[c] - np.dot((lr/N), np.dot(delta, X[c]))
                b[c] = delta
    return (w, b)


