import os
import numpy as np
from matplotlib import pyplot as plt


def getData(dir:str)->np.ndarray:
    if os.path.isfile(dir):
        data = np.loadtxt(dir, delimiter=',')
        x = data[:, [0, 1]]
        y = data[:, [2]]
        return x, y


def featureNormalize(x:np.ndarray):
    '''
    特征缩放：只对输入数据x进行标准化，不需要对输出进行y进行标准化
    标准化： 不含x_0, x_nor = （x - 均值）/标准差
    :param x:  不含x_0的数据矩阵
    :return:   x_nor， mean std
    '''
    meanValue = np.mean(x, axis=0)      # 求均值
    stdValue = np.std(x, axis=0)        # 求标准差
    x_nor = (x - meanValue) / stdValue
    return x_nor, meanValue, stdValue

def computeCost(k, m, alpha):
    J = alpha * sum(k**2) / (2 * m)
    return J


def gradientDescent(x, y, alpha, iterations):
    '''
    多元梯度下降和单元梯度下降算法，在使用矩阵进行实现时没有区别
    :param x: 训练数据集的输入部分
    :param y: 对应的输出部分
    :param alpha: 学习率（梯度下降步进）
    :param iterations: 迭代次数
    :return: 参数数组， 迭代过程中代价函数的值
    '''
    m = x.shape[0]
    theta = np.zeros((3, 1))
    J_History = np.zeros(iterations)
    x = np.hstack((np.ones((m, 1)), x))
    for i in range(iterations):
        h = x @ theta
        k = np.transpose(x) @ (h-y)
        theta -= alpha * k / m
        J_History[i] = computeCost(k, m, alpha)

    return theta, J_History


def normalEquation(x, y):
    '''
    正规方程法，数据量小于1w时首先考虑
    :param x: 训练数据集的输入部分
    :param y: 对应的输出值
    :return: 正规方程法求得的参数矩阵
    '''
    m = x.shape[0]
    x = np.hstack((np.ones((m, 1)), x))
    x_T = np.transpose(x)
    theta = np.linalg.pinv(x_T @ x) @ x_T @ y
    return theta


def main():
    dir = "ex1data2.txt"
    x, y = getData(dir)
    x_nor, meanValue, stdValue = featureNormalize(x)

    # gradient descent method
    theta_gradientDescent, J_History = gradientDescent(x_nor, y, 0.03, 400)
    plt.plot(np.arange(400), J_History)
    plt.show()

    # normal equation method
    theta_NormalEquation = normalEquation(x, y)

    # 通过标准化得到的theta，当做预测时也需要将数据进行表准化
    test = ((np.array([1650, 3]) - meanValue) / stdValue)
    test = np.hstack((1, test))

    print("gradient descent method = ", test @ theta_gradientDescent)
    print("normal equation method = ", np.array([1, 1650, 3]) @ theta_NormalEquation)


main()
