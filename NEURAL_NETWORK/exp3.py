from keras.datasets import imdb
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
# 参数num_words = dimension 的意思是仅保留训练数据的前dimension个最常见出现的单词，低频单词将被舍弃。
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
import warnings
dimension = 10000
warnings.filterwarnings("ignore")
# 2、preprocess data
# 定义数据集向量化的函数（转换为one hot编码）


def vectorize_sequences(sequences, dimension=dimension):
    results = np.zeros((len(sequences), dimension))  # 数据集长度
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1  # one-hot
    return results


def vec_seq(numSeq):
    Vec = [0] * dimension
    for num in numSeq:
        Vec[int(num)] += 1
    return Vec


def train():
    # 1、load data
    time1 = datetime.datetime.now()
    # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=dimension)
    trainlines = open("./task3/train/train_data.txt").readlines()
    x_data = np.array([vec_seq(line.split()) for line in trainlines])
    trainlabels = open("./task3/train/train_labels.txt").readlines()
    y = list(map(int, [line.strip() for line in trainlabels]))
    y_data = np.array(to_categorical(y))
    testlines = open("./task3/test/test_data.txt").readlines()
    X_pred = np.array([vec_seq(line.split()) for line in testlines])
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=0)
    # 取出测试数据集
    # with open("./task3/test/test_data.txt", "rb") as fr:
    #     test_data_n = [inst.decode().strip().split(' ') for inst in fr.readlines()]
    #     test_data = [[int(element) for element in line] for line in test_data_n]
    # test_data = np.array(test_data)

    # 数据预处理：转化为one hot编码
    # X_train = vectorize_sequences(X_train)
    # X_test = vectorize_sequences(X_test)
    # x_test_local = vectorize_sequences(test_data)

    time2 = datetime.datetime.now()
    print("data load and preprocess takes " + str((time2 - time1).seconds) + " s")

    time1 = datetime.datetime.now()

    # 定义Sequential类
    model = models.Sequential()
    # 全连接层，128个节点
    model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
    model.add(layers.Dropout(0.2))
    # 全连接层，64个节点
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    # 全连接层，得到输出
    model.add(layers.Dense(2, activation='softmax'))
    # loss
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train,
                        Y_train,
                        epochs=2,
                        batch_size=64,
                        validation_data=(X_test, Y_test),
                        validation_freq=1)
    model.summary()
    results = model.evaluate(X_test, Y_test)
    print(results)

    time2 = datetime.datetime.now()
    print("model train takes " + str((time2 - time1).seconds) + " s")
    # 4、model predict
    time1 = datetime.datetime.now()
    y_pred = model.predict(X_test)
    y_pred_local = model.predict(X_pred)
    time2 = datetime.datetime.now()
    print("model predict takes " + str((time2 - time1).seconds) + " s")
    # 5、model evaluation
    des = np.argmax(y_pred_local, axis=1)
    np.savetxt('exp3_result.txt', des, fmt='%d', delimiter='\n')
    print('测试集分类结果保存完毕')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def max_itr():
    # 1、load data
    time1 = datetime.datetime.now()
    # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=dimension)
    trainlines = open("./task3/train/train_data.txt").readlines()
    x_data = [vec_seq(line.split()) for line in trainlines]
    trainlabels = open("./task3/train/train_labels.txt").readlines()
    y_data = list(map(int, [line.strip() for line in trainlabels]))
    testlines = open("./task3/test/test_data.txt").readlines()
    X_pred = [vec_seq(line.split()) for line in testlines]
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=0)
    time2 = datetime.datetime.now()
    print("data load and preprocess takes " + str((time2 - time1).seconds) + " s")
    n = 80
    ans = []
    x = range(1, n + 1, 2)

    plt.plot(x, ans, "g", marker='D', markersize=5, label="F-measure")
    # 绘制坐标轴标签
    plt.xlabel("Max_iter")
    plt.ylabel("F-measure")
    plt.show()


if __name__ == "__main__":
    train()
