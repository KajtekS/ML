import numpy as np
from requests.packages import target
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = datasets.load_breast_cancer()
x, y = data.data, data.data

xtrain, ytrain, xtest, ytest = train_test_split(x, y, test_size=0.2, random_state=0)

n_samples, n_features = x.shape

#initialization of weights and biases
weights = np.zeros(n_features)
bias = 0
epochs = 1000
#singmod
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_backward_propagation(w, b, x_train, y_head):
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)
    cost = (np.sum(loss)) / x_train.shape[1]

    # backward propogation
    derivative_weight = (np.dot(x_train, ((y_head - y_train).T))) / x_train.shape[1]
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1]
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost, gradients