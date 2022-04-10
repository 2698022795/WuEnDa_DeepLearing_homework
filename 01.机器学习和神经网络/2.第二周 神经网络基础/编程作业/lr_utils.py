import matplotlib.pyplot as plt
import numpy as np
import h5py
import scipy
from scipy import ndimage
#-*- coding:utf-8 -*-

# ctrl + shift + "+":所有函数折叠

# 加载数据
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


# 第一步：检查训练集和测试集的尺寸
def size_example():
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]

    return m_train, m_test, num_px


# 第一步：读取测试集和训练集数据，并检查其大小是否符合要求
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_train, m_test, num_px = size_example()
print("第一步：读取测试集和训练集数据，并检查其大小是否符合要求")
print("Number of training examples: m_train = " + str(m_train))
print("Number of testing examples: m_test = " + str(m_test))
print("Height/Width of each image: num_px = " + str(num_px))
print("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("train_set_x_orig shape: " + str(train_set_x_orig.shape))
print("train_set_y shape: " + str(train_set_y.shape))
print("test_set_x_orig shape: " + str(test_set_x_orig.shape))
print("test_set_y shape: " + str(test_set_y.shape))
print("")


# 第二步：将多维数据压平成一维数据
def reshape_example(train_set_x_orig, test_set_x_orig):
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    return train_set_x_flatten, test_set_x_flatten


# 第二步：将数据压平
train_set_x_flatten, test_set_x_flatten = reshape_example(train_set_x_orig, test_set_x_orig)
print("第二步：将数据压平")
print("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print("")


# 第三步：数据归一化
def normalization(data_set, mean_value, variance):
    return (data_set - mean_value) / variance


# 第三步：数据归一化
train_set_x = normalization(train_set_x_flatten, 0, 255)
test_set_x = normalization(test_set_x_flatten, 0, 255)
print("第三步：数据归一化")
print("train_set_x shape:" + str(train_set_x.shape))
print("test_set_x shape:" + str(test_set_x.shape))
print("")


# 第四步：构建sigmoid函数

def sigmoid(z):
    a = 1 / (1 + np.exp(-z))
    return a


# 第四步：测试sigmoid函数是否符合要求
print("第四步：构建sigmoid函数")
print("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))
print("")


# 第五步：初始化参数(单层神经网络）

def initialize_with_zeros(dim):  # 输入维度，输出(dim,1)维度的w和单个b
    w = np.zeros((dim, 1))
    b = 0

    return w, b


# 第五步：初始化参数（单层神经网络）
print("第五步：初始化参数（单层神经网络）")
(train_set_x.shape[0])
dim = 2
w, b = initialize_with_zeros(dim)
print("w = " + str(w))
print("b = " + str(b))
print("")


# 第六步：前向传播和梯度（单层神经网络）
def forward_propagation(w, b, X, Y):
    m = X.shape[1]

    Z = np.dot(w.T, X) + b
    A = sigmoid(Z)
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dZ = A - Y
    dW = 1 / m * np.dot(X, dZ.T)
    dB = 1 / m * np.sum(dZ)

    grads = {
        "dW": dW,
        "dB": dB
    }
    return grads, cost


w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
grads, cost = forward_propagation(w, b, X, Y)
print("第六步：前向传播和梯度（单层神经网络）")
print("dw = " + str(grads["dW"]))
print("db = " + str(grads["dB"]))
print("cost = " + str(cost))
print("")


# 第七步：优化函数（单层神经网络）
def backward_propagation(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []

    for i in range(num_iterations):
        grads, cost = forward_propagation(w, b, X, Y)
        dW = grads["dW"]
        dB = grads["dB"]

        w = w - learning_rate * dW
        b = b - learning_rate * dB

        if i % 100 == 0:
            costs.append(cost)
        if print_cost == True and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dW,
             "db": dB}

    return params, grads, costs


params, grads, costs = backward_propagation(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)

print("第七步：反向传播（单层神经网络）")
print("w = " + str(params["w"]))
print("b = " + str(params["b"]))
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print(costs)
print()


# 第八步：测试集测试（单层神经网络）
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(m):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction


print("第八步：测试集测试（单层神经网络）")
print("predictions = " + str(predict(w, b, X)))
print("")


# 第九步：整合所有功能（单层神经网络）
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    # 1.训练
    # 1.1 生成参数w,b
    w, b = initialize_with_zeros(X_train.shape[0])
    # 1.2 正向传播
    grads, cost = forward_propagation(w, b, X_train, Y_train)
    # 1.3 反向传播
    params, grads, costs = backward_propagation(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # 2.测试
    # 2.1 取出参数w,b
    w = params["w"]
    b = params["b"]
    # 2.2 测试训练集和测试集
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)
    # 2.3 输出结果
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d


# 第九步：建立模型（单层神经网络）
print("第九步：建立模型（单层神经网络）")
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
print("")

# 第十步：绘制损失函数曲线（单层神经网络）
print("第十步：绘制损失函数曲线（单层神经网络）")
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations(per hundreds)')
plt.title("Learning rate = " + str(d["learning_rate"]))
plt.show()
print("")

# 第十一步：调整学习率（单层神经网络）
print("第十一步：调整学习率（单层神经网络）")
learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i,
                           print_cost=False)
    print('\n' + "-------------------------------------------------------" + '\n')
for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))
plt.ylabel('cost')
plt.xlabel('iterations')
legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()

print("")

# # 第十二步：测试自己的图片（单层神经网络）
print("第十二步：测试自己的图片（单层神经网络）(未完成）")
# fname = r'C:\Users\Lenovo\Desktop\1.jpg'
# image = np.array(plt.imread(fname))
# my_image = scipy.misc.imresize(image, size=(num_px, num_px)).reshape((1, num_px * num_px * 3)).T
# my_predicted_image = predict(d["w"], d["b"], my_image)
# print("第十二步：测试自己的图片（单层神经网络）")
#
# print("")
