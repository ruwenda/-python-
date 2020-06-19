
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import scipy.optimize as opt
from sklearn.metrics import classification_report


#%%
def load_data(path, transpose = True):
    data = sio.loadmat(path)
    y = data.get('y')
    y = y.reshape([y.shape[0], 1])

    X = data.get('X')

    # 图片的方向旋转正常
    if transpose:
        # 取出每行的图片并进行转置
        X = np.array([im.reshape([20, 20]).T for im in X])
        # 将20x20的图片变为400，然后变为原来的5000x400
        X = np.array([im.reshape([400]).T for im in X])
    return X, y


def plot_an_image(image):
    plt.imshow(image.reshape([20, 20]), cmap=plt.cm.binary)
    # 去除x轴和y轴的刻度, 刻度为像素的刻度20x20的
    # plt.xticks(np.array([]))
    # plt.yticks(np.array([]))
    plt.show()


def plot_100_image(X):
    size = int(np.sqrt(X.shape[1]))

    # 从5000张图片中抽出100张来
    sample_idx = np.random.choice(np.arange(X.shape[0]), 100)
    sample_img = X[sample_idx, :]

    # 注意subplots和subplot是不同的
    fig, ax_array = plt.subplots(nrows=10, ncols=10,
                                 sharey=True, sharex=True, figsize=(8, 8))
    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(sample_img[10 * r + c].reshape([size, size]),
                                   cmap=plt.cm.binary)
            # 去除图片的坐标轴刻度
            plt.xticks([])
            plt.yticks([])
    plt.show()


X, y = load_data('ex3data1.mat')
print(X.shape)
print(y.shape)

pick_one = np.random.randint(0, 5000)
plot_an_image(X[pick_one, :])
plt.show()
print("this should be %d" %(y[pick_one]))

plot_100_image(X)


#%%
def init_data(X, y):
    X = np.hstack([np.ones([X.shape[0], 1]), X])
    yInit = np.zeros([10, X.shape[0]])
    for i in range(10):
        yInit[i, np.where((i + 1) == y)[0]] = 1

    temp = yInit[-1, :].copy()
    yInit = yInit[0:-1, :].copy()
    yInit = np.vstack([temp, yInit])

    return X, yInit


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def costFunction(theta, X, y, Lambda):
    y = y.reshape([-1, 1])
    theta = theta.reshape([-1, 1])
    m = X.shape[0]
    theta1 = theta.copy()
    theta1[0] = 0
    h = sigmoid(X @ theta)
    J = (y.T @ np.log(h) + (1 - y.T) @ np.log(1 - h)) / (-m)
    reg = Lambda * theta1.T @ theta1 / (2*m)
    return J + reg


def gradient(theta, X, y, Lambda):
    theta = theta.reshape([-1, 1])
    y = y.reshape([-1, 1])
    theta1 = theta.copy()
    theta1[0] = 0
    m = X.shape[0]
    h = sigmoid(X @ theta)
    grad = X.T @ (h - y) / m
    reg = Lambda / m * theta1
    return grad + reg


#%%
X, yInit = init_data(X, y)
Lambda = 0.03
allTheta = np.zeros([10, X.shape[1]])
for i in range(10):
    theta = np.zeros([X.shape[1], 1])
    y_i = yInit[i, :].reshape([-1, 1])
    result = opt.minimize(costFunction,
                          theta,
                          (X, y_i, Lambda),
                          method="TNC",
                          jac=gradient)
    allTheta[i, :] = result.x
    print("success:", result.success)


#%%
predict = sigmoid(X @ allTheta.T)
predict = np.argmax(predict, axis=1)
y1 = y.copy()
y1[y1 == 10] = 0
print(classification_report(y1, predict))


#%% 本部分开始为神经网络
def load_weight(path):
    data = sio.loadmat(path)
    return data["Theta1"], data["Theta2"]


theta1, theta2 = load_weight('ex3weights.mat')
print("theta1:%s \ntheta2:%s" % (theta1.shape, theta2.shape))

X, y = load_data('ex3data1.mat', transpose=False)
print("X:%s \ny:%s" % (X.shape, y.shape))


#%% 神经网络的前向传播
# input layer
X = np.hstack([np.ones([X.shape[0], 1]), X])
a1 = X

# hidden layer
z2 = a1 @ theta1.T
a2 = sigmoid(z2)
a2 = np.hstack([np.ones([a2.shape[0], 1]), a2])

# output layer
z3 = a2 @ theta2.T
a3 = sigmoid(z3)

predict = np.argmax(a3, axis=1) + 1
print(classification_report(y, predict))