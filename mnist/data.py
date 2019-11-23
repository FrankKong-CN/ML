import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt

RESHAPED = 784
CLASS_NUM = 10


def get_data():
    """
    获取mnist数据
    :return:
    """
    # 载入数据并划分
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    return (X_train, y_train), (X_test, y_test)


def process_for_fully_connected_network(X_train, y_train, X_test, y_test):
    """
    reshape & 归一化 & one-hot编码,用于全连接网络
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    X_train = X_train.reshape(60000, RESHAPED).astype('float32')
    X_test = X_test.reshape(10000, RESHAPED).astype('float32')

    X_train /= 255
    X_test /= 255

    Y_train = np_utils.to_categorical(y_train, CLASS_NUM)
    Y_test = np_utils.to_categorical(y_test, CLASS_NUM)

    return X_train, Y_train, X_test, Y_test


def process_for_CNN(X_train, y_train, X_test, y_test):
    """

    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = X_train[:, :, :, np.newaxis]
    X_test = X_test[:, :, :, np.newaxis]

    Y_train = np_utils.to_categorical(y_train, CLASS_NUM)
    Y_test = np_utils.to_categorical(y_test, CLASS_NUM)

    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = get_data()
    # print(X_train.shape)
    img = X_test[300][:, :]
    plt.imshow(img, cmap='Greys')
    plt.show()
