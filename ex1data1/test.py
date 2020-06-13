import os
import numpy as np
from matplotlib import pyplot as plt


def getData(dir:str):
    '''
    获取txt文本训练数据
    :param dir: txt文件目录
    :return:    文本中读取的数据
    '''
    if os.path.isfile(dir):
        data = np.loadtxt(dir, delimiter=',')
        x = data[:, 0]
        y = data[:, 1]
        return x, y


def gradientDescent(x, y, alpha, iterations):
    '''
    梯度下降算法
    :param x: 训练数据输入值
    :param y: 训练数据输出值
    :param alpha: 学习率
    :param iterations: 迭代次数
    :return: 假设函数的设置值
    '''
    theta = np.zeros((2, 1))
    m = len(y)
    x = np.hstack((np.ones((m, 1)), x))  # (97, 2)
    costHistory = np.zeros((iterations, 1))

    for i in range(iterations):
        # print(i)
        h = x @ theta
        k = np.transpose(x) @ (h - y)
        theta = theta - alpha * k / m
        costHistory[i] = computeLoss(m, h, y)

    plt.plot(range(0, iterations), costHistory)
    plt.title('cost function')
    plt.xlabel('iterations')
    plt.ylabel('cost')
    plt.show()
    return theta


def computeLoss(m, h, y):
    cost = sum((h-y)**2) / (2 * m)
    return cost


def main():
    # get training dataset
    dir = "ex1data1.txt"
    x, y = getData(dir)
    x = np.reshape(x, (97, 1))
    y = np.reshape(y, (97, 1))

    # gradient descent
    theta = gradientDescent(x, y, 0.01, 1200)       # 学习率(步进)太高会使得计算溢出
    print(theta)

    plt.scatter(x, y, marker='x', color='k')
    plt.plot(x, theta[1] * x + theta[0], color='r')
    plt.show()


main()
