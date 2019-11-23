import os
# GPU
# 包依赖 tensorflow-gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# CPU
# 包依赖 tensorflow
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.optimizers import SGD, Adam
from keras.utils import np_utils
import numpy as np

from data import *
from model import *

OPTIMIZER = Adam()
BATCH_SIZE = 128
EPOCH = 10

(X_train, y_train), (X_test, y_test) = get_data()

# # 全连接数据处理
# X_train, Y_train, X_test, Y_test = process_for_fully_connected_network(X_train, y_train, X_test, y_test)
#
# # 全连接层训练
# model = fully_connected_network_model(784, 10)
#
# model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
#
# history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCH)
#
# model.save('fully_connected_model.h5')

# CNN数据处理
X_train, Y_train, X_test, Y_test = process_for_CNN(X_train, y_train, X_test, y_test)

# CNN训练
model = CNN_model(28, 10)

model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCH)

model.save('CNN_model.h5')
