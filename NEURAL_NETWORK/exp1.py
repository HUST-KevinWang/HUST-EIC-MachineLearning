# -*- coding:UTF-8 -*-
import numpy as np
from tensorflow.keras import layers, models
import warnings
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore")


def colicLR():
    # 使用np读取数据
    trainFiled = np.loadtxt('./Task1/horseColicTraining.txt', delimiter="\t")
    trainSet = trainFiled[:, :-1]
    trainLables = trainFiled[:, -1:]

    testFiled = np.loadtxt('./Task1/horseColicTest.txt', delimiter="\t")
    testSet = testFiled[:, :-1]
    testLables = testFiled[:, -1:]
    classifier = LogisticRegression(solver='lbfgs', max_iter=25).fit(trainSet, trainLables)
    test_accurcy = classifier.score(testSet, testLables) * 100
    print('测试集ACCURACY')
    print(str(test_accurcy))


def colicNN():
    # 使用np读取数据
    trainFiled = np.loadtxt('./Task1/horseColicTraining.txt', delimiter="\t")
    X_train = np.array(trainFiled[:, :-1])
    Y_train = np.array(to_categorical(trainFiled[:, -1:]))

    testFiled = np.loadtxt('./Task1/horseColicTest.txt', delimiter="\t")
    X_test = np.array(testFiled[:, :-1])
    Y_test = np.array(to_categorical(testFiled[:, -1:]))
    # 定义Sequential类
    model = models.Sequential()
    # 全连接层，128个节点
    model.add(layers.Dense(128, activation='relu', input_shape=(21,)))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    # 全连接层，64个节点
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    # 全连接层，16个节点
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    # 全连接层，得到输出
    model.add(layers.Dense(2, activation='softmax'))
    # loss
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train,
                        Y_train,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_test, Y_test),
                        validation_freq=1)
    model.summary()
    results = model.evaluate(X_test, Y_test)
    acc = results[1]*100
    print('正确率：%f%%' % acc)

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


if __name__ == '__main__':
    colicLR()
