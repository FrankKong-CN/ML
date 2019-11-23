import os
# GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# CPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten


def fully_connected_network_model(input_dim, CLASS_NUM):
    """
    :param CLASS_NUM
    :param input_dim:
    :return: model
    """
    model = Sequential()
    model.add(Dense(units=128, input_dim=input_dim, activation='relu'))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=CLASS_NUM, activation='softmax'))

    return model


def CNN_model(input_dim, CLASS_NUM):
    """

    :param input_dim:
    :param CLASS_NUM:
    :return:
    """
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, input_shape=(input_dim, input_dim, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    # model.add((Conv2D(32, kernel_size=3)))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    model.add(Dense(CLASS_NUM, activation="softmax"))

    return model


if __name__ == '__main__':
    model = CNN_model(28, 10)
    model.summary()
