import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import tensorflow


def getData(dir: str) -> np.ndarray:
    data = np.loadtxt(dir, delimiter=',')
    x = data[:, [0, 1]]
    y = data[:, [-1]]
    return x, y


def initData(x):
    m = x.shape[0]
    X = np.ones((m, 1))
    degree = 6
    for i in range(1, degree + 1):
        for j in range(i + 1):
            temp = x[:, [0]] ** (i - j) * x[:, [1]] ** j
            X = np.hstack((X, temp))
    n = X.shape[1]
    theta = np.zeros((n, 1))

    return X, theta



def sigmoid(z):
    z = z.reshape([len(z), 1])
    sigma = 1 / (1 + np.exp(-z))
    return sigma


def costFunction(theta, X, y, Lambda):
    m = X.shape[0]

    # 将theta_0 不参与正则化置为0
    theta_1 = np.copy(theta)
    theta_1[0] = 0
    theta_1 = theta_1.reshape(len(theta), 1)

    h = sigmoid(X @ theta)
    J = (y.T @ np.log(h) + (1 - y.T) @ np.log(1 - h)) / (-m) + \
        Lambda / (2 * m) * theta_1.T @ theta_1
    return J


def gradient(theta, X, y, Lambda):
    theta_1 = theta.copy()
    theta_1[0] = 0
    m, n = X.shape
    theta_1 = theta_1.reshape(n, 1)
    theta = theta.reshape(n, 1)
    h = sigmoid(X @ theta)
    grad = 1 / m * (X.T @ (h - y)) + Lambda / (2 * m) * theta_1 ** 2
    return grad


def plotData(x, y):
    plt.scatter(x[np.where(y == 1), 0], x[np.where(y == 1), 1], marker='o')
    plt.scatter(x[np.where(y == 0), 0], x[np.where(y == 0), 1], marker='x')
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.show()


def plotDecisionBoundary(x, y, theta):
    # 等高线的绘制
    plt.figure()

    theta = theta.reshape(len(theta), 1)
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros([len(u), len(v)])
    for i in range(len(u)):
        for j in range(len(v)):
            temp = mapData(u[i], v[j])
            z[i, j] = temp @ theta
    z = z.T
    plt.contour(u, v, z)

    # 样本绘制
    plt.scatter(x[np.where(y == 1), 0], x[np.where(y == 1), 1], marker='o')
    plt.scatter(x[np.where(y == 0), 0], x[np.where(y == 0), 1], marker='x')
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.ylim(-1, 1)
    plt.xlim(-1, 1)

    plt.show()


def mapData(a, b):
    degree = 6
    c = np.ones([1, 1])
    for i in range(1, degree + 1):
        for j in range(i + 1):
            temp = np.array([[a**(i - j) * b**j]])
            c = np.hstack((c, temp))
    return c


def main():
    dir = "ex2data2.txt"
    Lambda = 1
    x, y = getData(dir)
    X, theta = initData(x)
    result = op.minimize(costFunction, theta, (X, y, Lambda), method="TNC", jac=gradient)
    plotDecisionBoundary(x, y, result['x'])

    # 计算预测的准确率
    p = np.zeros([len(y), 1])
    p[np.where(sigmoid(X @ result['x']) >= 0.5)] = True
    accuracy = np.mean(p == y)
    print("accuracy = ", accuracy * 100, "%")


main()


