from keras.models import load_model

from data import *

model = load_model('fully_connected_model.h5')

(X_train, y_train), (X_test, y_test) = get_data()


# 全连接数据处理
X_train, Y_train, X_test, Y_test = process_for_fully_connected_network(X_train, y_train, X_test, y_test)

# score = model.evaluate(X_test, Y_test)

# print(score)

# img = X_test[300, :]
# img.reshape(1, 784)
prediction = model.predict_classes(X_test[300].reshape(1, 784))
print(prediction)

print(X_test.shape)
