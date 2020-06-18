from scipy.io import loadmat
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import time


def getData():
    data = loadmat('ex3data1.mat')
    X, y = np.array(data['X']), np.array(data['y'])
    X = np.hstack([np.ones([len(X), 1]), X])
    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient(theta, X, y, Lambda):
    # 注意深拷贝否则为引用
    theta_1 = theta.copy()
    theta_1[0] = 0
    m, n = X.shape
    theta_1 = theta_1.reshape(n, 1)
    theta = theta.reshape(n, 1)
    h = sigmoid(X @ theta)
    grad = 1 / m * (X.T @ (h - y)) + Lambda / m * theta_1
    return grad


def cost(theta, X, y, Lambda):
    theta = theta.reshape([-1, 1])

    # 正则化时要对theta做深拷贝copy(),否则出错
    temp = theta.copy()
    temp[0] = 0

    m = X.shape[0]
    h = sigmoid(X @ theta)
    first = (y.T @ np.log(h) + (1 - y.T) @ np.log(1 - h)) / (-m)
    reg = temp.T @ temp * Lambda / (2 * m)
    return reg + first


#%%
X, y = getData()
theta = np.zeros([X.shape[1], 1])
allTheta = np.zeros([10, len(theta)])
for i in range(1, 11):
    fmi = minimize(fun=cost, x0=theta, args=(X, y == i, 0.02),
                   method="TNC", jac=gradient)
    allTheta[i-1, :] = fmi.x
    print("iterations:", i, "success:", fmi.success)

#%%
h = sigmoid(X @ allTheta.T)

# 读出每行值最大的列号，即每个图片的最大预测值,手写数字从1开始，而列号从0开始需要加1
h_argmax = np.argmax(h, axis=1) + 1
h_argmax = h_argmax.reshape([-1, 1])

m, n = X.shape
x = X[:, 1:n]
for i in range(0, m, 100):
    temp = x[i, :]
    temp = temp.reshape([20, 20])
    plt.imshow(temp.T, cmap="gray")
    plt.show()
    print("predict:", h_argmax[i, 0])
    time.sleep(1)


accuracy = np.mean(h_argmax == y)
print(accuracy * 100)