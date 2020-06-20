import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from scipy.optimize import minimize
import time


def load_data(path):
    data = loadmat(path)
    X = data['X']
    y = data['y']
    return X, y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    # input layer
    a1 = np.hstack([np.ones([m, 1]), X])

    # hidden layer
    z2 = a1 @ theta1.T
    a2 = np.hstack([np.ones([m, 1]), sigmoid(z2)])

    # output layer
    z3 = a2 @ theta2.T
    h = a3 = sigmoid(z3)

    return a1, z2, a2, z3, h


def cost_function(params, inputSize, hiddenSize, numLabels, X, y):
    m = X.shape[0]

    # 从params中取出对用层得参数矩阵
    theta1 = np.reshape(params[:hiddenSize * (inputSize + 1)],
                        [hiddenSize, inputSize + 1])
    theta2 = np.reshape(params[hiddenSize * (inputSize + 1):],
                        [numLabels, hiddenSize + 1])

    # 运行前向传播算法
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # 计算代价函数
    J = 0
    for i in range(m):
        temp = (y[i, :] @ np.log(h[i, :].T) +
                (1 - y[i, :]) @ np.log(1 - h[i, :].T))
        J += temp

    J = J / (-m)

    return J


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def cost_reg(params, inputSize, hiddenSize, numLabels, X, y, Lambda):
    m = X.shape[0]

    # 从params中取出对用层得参数矩阵
    theta1 = np.reshape(params[:hiddenSize * (inputSize + 1)],
                        [hiddenSize, -1])
    theta2 = np.reshape(params[hiddenSize * (inputSize + 1):],
                        [numLabels, -1])

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # cost function
    J = 0
    for i in range(m):
        J += y[i, :] @ np.log(h[i, :].T) + (1 - y[i, :]) @ np.log(1 - h[i, :].T)
    J = J / (-m)

    # 正则化项，就是对神经网络各个参数值（权重值）做平方项后求和，不包含偏置单元的权重值！
    reg = (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))\
        * Lambda / (2 * m)
    J += reg
    return J


def back_propagate(params, inputSize, hiddenSize, numLabels, X, y, Lambda):
    m = X.shape[0]

    # 从params中取出theta1、theta2.
    theta1 = np.reshape(params[:hiddenSize * (inputSize + 1)],
                        [hiddenSize, -1])
    theta2 = np.reshape(params[hiddenSize * (inputSize + 1):],
                        [numLabels, -1])

    # 执行前向传播
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

    # 初始化
    J = 0
    delta1 = np.zeros(theta1.shape)
    delta2 = np.zeros(theta2.shape)

    # cost function
    J = 0
    for i in range(m):
        J += y[i, :] @ np.log(h[i, :].T) + (1 - y[i, :]) @ np.log(1 - h[i, :].T)
    J = J / (-m)

    # 正则化项，就是对神经网络各个参数值（权重值）做平方项后求和，不包含偏置单元的权重值！
    reg = (np.sum(theta1[:, 1:] ** 2) + np.sum(theta2[:, 1:] ** 2))\
        * Lambda / (2 * m)
    J += reg

    # 实现反向传播
    for t in range(m):
        a1t = a1[t, :].reshape([1, -1])  # (1, 401)
        z2t = z2[t, :].reshape([1, -1])  # (1, 25)
        a2t = a2[t, :].reshape([1, -1])  # (1, 26)
        ht = h[t, :].reshape([1, -1])    # (1, 10)
        yt = y[t, :].reshape([1, -1])    # (1, 10)

        d3t = ht - yt   # (1, 10)

        z2t = np.hstack([np.ones([1, 1]), z2t])             # 加入一列不会偏置单元(bias unit) (1, 26)
        d2t = (theta2.T @ d3t.T).T * sigmoid_gradient(z2t)  # (26, 10) x (10, 11) = (1, 26)

        delta1 = delta1 + d2t[:, 1:].T @ a1t
        delta2 = delta2 + d3t.T @ a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))

    return J, grad


#%%
X, y = load_data("ex4data1.mat")
print("X:%s\ty:%s" % (X.shape, y.shape))

# one-hot编码将类标签n(k类)转换为长度为k的矢量
encoder = OneHotEncoder(sparse=False)   # 不返回稀疏矩阵,稀疏矩阵：非零元素值得上下左右相邻元素均为0
yOneHot = encoder.fit_transform(y)      # 将y进行One-Hot编码然后转置
print("yOneHot shape:", yOneHot.shape)


# 初始化设置
inputSize = 400
hiddenSize = 25
numLabels = 10
Lambda = 1

# 初始化完整网络参数大小的参数数组,使其失去对称性
params = (np.random.random(hiddenSize * (inputSize + 1) + numLabels * (hiddenSize + 1)) - 0.5) * 0.25  # 产生[0, 1)内服从均匀分布的浮点数
print("parameter shape: ", params.shape)

m = X.shape[0]
theta1 = np.reshape(params[:hiddenSize * (inputSize + 1)],
                    [hiddenSize, inputSize + 1])
theta2 = np.reshape(params[hiddenSize * (inputSize + 1):],
                    [numLabels, hiddenSize + 1])
print("theta1 shape:", theta1.shape)
print("theta2 shape:", theta2.shape)

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
print(a1.shape, z2.shape, a2.shape, z3.shape, h.shape)
J = cost_function(params, inputSize, hiddenSize, numLabels, X, yOneHot)
print(J)

JReg = cost_reg(params, inputSize, hiddenSize, numLabels, X, yOneHot, Lambda)
print(JReg)


J, grad = back_propagate(params, inputSize, hiddenSize, numLabels, X, yOneHot, Lambda)
print(J, grad.shape)

fmin = minimize(fun=back_propagate,
                x0=params,
                args=(inputSize, hiddenSize, numLabels, X, yOneHot, Lambda),
                method="TNC",
                jac=True,
                options={"maxiter": 1000})
print(fmin)


#%%
theta1 = np.reshape(fmin.x[:hiddenSize * (inputSize + 1)],
                    [hiddenSize, -1])
theta2 = np.reshape(fmin.x[hiddenSize * (inputSize + 1):],
                    [numLabels, -1])
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
predict = np.argmax(h, axis=1) + 1
predict[predict == 10] = 0
y[y == 10] = 0


#%%
predict = predict.reshape([-1, 1])
correct = np.zeros(predict.shape)
correct[y == predict] = 1
print(correct.shape)

accuary = np.mean(correct)
print("accuary:", accuary * 100, "%")
print(classification_report(y, predict))


#%%
for _ in range(30):
    i = np.random.randint(0, 5001)
    temp = X[i, :]
    temp = temp.reshape([20, 20])
    plt.imshow(temp.T, cmap=plt.cm.binary)
    plt.show()
    print("predict:", predict[i, 0])
    time.sleep(1)

