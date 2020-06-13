import numpy as np
from matplotlib import pyplot as plt
import scipy.optimize as op


def getData(dir: str) -> np.ndarray:
    data = np.loadtxt(dir, delimiter=",")
    x = np.hstack((np.ones((data.shape[0], 1)), data[:, [0, 1]]))
    y = data[:, [2]]
    return x, y


def sigmoid(z: np.ndarray) -> np.ndarray:
    '''
    启动函数
    :param z:z = x @ theta
    :return: sigmoid函数
    '''
    sigma = 1 / (1 + np.exp(-z))
    return sigma


def costFunction(theta, x, y):
    m = x.shape[0]
    h = sigmoid(x @ theta)
    J = (y.T @ np.log(h) + (1 - y.T) @ np.log(1 - h)) / (-m)
    return J


def gradientDescent(alpha, iterations, x, y):
    m = x.shape[0]
    theta = np.zeros((3, 1))
    J_History = np.zeros((iterations, 1))
    for i in range(iterations):
        h = sigmoid(x @ theta)
        theta -= alpha * x.T @ (h - y) / m
        J_History[i] = costFunction(theta, x, y)

    return theta, J_History


def plotData(x, y):
    plt.scatter(x[np.where(y == 1), 1], x[np.where(y == 1), 2], marker='o')
    plt.scatter(x[np.where(y == 0), 1], x[np.where(y == 0), 2], marker='x')
    plt.xlabel("exam1 score")
    plt.ylabel("exam2 score")
    plt.show()


def gradient(theta, x, y):
    '''
    计算当前取值时的costFunction的梯度值
    :param theta: 当前参数矩阵
    :param x: 特征x的矩阵
    :param y: 结果y的矩阵
    :return: 返回当前梯度值
    '''
    m, n = x.shape
    theta = theta.reshape((n, 1))
    grad = np.dot(x.T, sigmoid(x.dot(theta))-y)/m
    return grad.flatten()       # flatten 将数组变为一维的形式


def plotBoundary(theta, x, y, title: str):
    '''
    绘制决策边界
    :param theta:参数矩阵
    :param x: 特征值矩阵
    :param y: 结果
    :param title:图名
    :return: None
    '''
    plt.scatter(x[np.where(y == 1), 1], x[np.where(y == 1), 2], marker='o')
    plt.scatter(x[np.where(y == 0), 1], x[np.where(y == 0), 2], marker='x')
    plt.title(title)
    plt.xlabel("exam1 score")
    plt.ylabel("exam2 score")
    plot_x = np.array([min(x[:, 2]), max(x[:, 2])])
    plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0])
    plt.plot(plot_x, plot_y, 'r')
    plt.show()


def featureNormalize(x):
    '''
    对特征值，以"标准化"的形式进行特征缩放
    :param x: 未缩放的特征矩阵
    :return: 缩放后的特征矩阵
    '''
    x = x[:, 1:x.shape[1]]
    meanValue = np.mean(x, axis=0)
    stdValue = np.std(x, axis=0)
    x_nor = (x - meanValue) / stdValue
    x_nor = np.hstack((np.ones((x.shape[0], 1)), x_nor))
    return x_nor, meanValue, stdValue


def main():
    dir = "ex2data1.txt"
    x, y = getData(dir)

    # 绘制数据散点图
    plotData(x, y)

    # 梯度下降算法
    theta, J_History = gradientDescent(0.001, 600000, x, y)     # 学习率过高costFunction\iteration曲线会震荡或者不收敛
    plt.plot(range(600000), J_History)
    plt.show()
    print(theta)
    plotBoundary(theta, x, y, "gradient descent")

    # 无约束最小化算法fminunc的替代minimize ## help(op.minimize)查询
    # 能够自动选择学习率,迭代次数相较于传统梯度下降算法较少
    # 参数   1.fun:       算法的损耗函数(自己写), 需要的优化的目标函数
    #       2.x0:        参数矩阵theta
    #       3.arg:       训练数据(x, y) 其中x为初始化之后的，即前面加了一排1的
    #       4.method:    "TNC",“BFGS”等等算法
    #       5.jac:       求损耗函数梯度的函数返回梯度值(自己写)
    theta = np.zeros((3, 1))
    result = op.minimize(costFunction, theta, (x, y), method='TNC', jac=gradient)
    print("TNC get theta = ", result["x"], end='\n')
    theta = np.array([-25.16131858,   0.20623159,   0.20147149]).reshape((3, 1))
    plotBoundary(theta, x, y, "TNC method")

    # 对特征进行缩放后，调用TNC算法
    x_nor, meanValue, stdValue = featureNormalize(x)
    theta = np.zeros((3, 1))
    result_nor = op.minimize(costFunction, theta, (x_nor, y), method="TNC", jac=gradient)
    print("feature normalize TNC get theta = ", result_nor["x"])
    plotBoundary(result_nor["x"], x_nor, y, "feature normalize TNC method")


main()

