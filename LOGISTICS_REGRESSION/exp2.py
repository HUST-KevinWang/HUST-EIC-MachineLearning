from keras.datasets import imdb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# 参数num_words = dimension 的意思是仅保留训练数据的前dimension个最常见出现的单词，低频单词将被舍弃。
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
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
    trainlines = open("./task2/train/train_data.txt").readlines()
    x_data = [vec_seq(line.split()) for line in trainlines]
    trainlabels = open("./task2/train/train_labels.txt").readlines()
    y_data = list(map(int, [line.strip() for line in trainlabels]))
    testlines = open("./task2/test/test_data.txt").readlines()
    X_pred = [vec_seq(line.split()) for line in testlines]
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=0)
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

    model = LogisticRegression(max_iter=50)
    model.fit(X_train, Y_train)

    time2 = datetime.datetime.now()
    print("model train takes " + str((time2 - time1).seconds) + " s")
    # 4、model predict
    time1 = datetime.datetime.now()
    y_pred = model.predict(X_test)
    y_pred_local = model.predict(X_pred)
    print('训练集分数：' + str(accuracy_score(Y_train, model.predict(X_train))))
    print("验证集分数:", model.score(X_test, Y_test))
    print("model accuracy is " + str(accuracy_score(Y_test, y_pred)))
    print("model precision is " + str(precision_score(Y_test, y_pred, average='macro')))
    print("model recall is " + str(recall_score(Y_test, y_pred, average='macro')))
    print("model f1_score is " + str(f1_score(Y_test, y_pred, average='macro')))
    score = cross_val_score(model, x_data, y_data, cv=5)
    print('交叉验证分数：' + str(sum(score) / 5))
    print(score)
    time2 = datetime.datetime.now()
    print("model predict takes " + str((time2 - time1).seconds) + " s")
    # 5、model evaluation

    des = y_pred_local.astype(int)
    np.savetxt('exp2_result.txt', des, fmt='%d', delimiter='\n')
    print('测试集分类结果保存完毕')


def max_itr():
    # 1、load data
    time1 = datetime.datetime.now()
    # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=dimension)
    trainlines = open("./task2/train/train_data.txt").readlines()
    x_data = [vec_seq(line.split()) for line in trainlines]
    trainlabels = open("./task2/train/train_labels.txt").readlines()
    y_data = list(map(int, [line.strip() for line in trainlabels]))
    testlines = open("./task2/test/test_data.txt").readlines()
    X_pred = [vec_seq(line.split()) for line in testlines]
    X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=0)
    time2 = datetime.datetime.now()
    print("data load and preprocess takes " + str((time2 - time1).seconds) + " s")
    n = 80
    ans = []
    x = range(1, n + 1, 2)
    for i in x:
        model = LogisticRegression(max_iter=i)
        a = sum(cross_val_score(model, x_data, y_data, cv=5)) / 5
        ans.append(a)
        print(a)
    plt.plot(x, ans, "g", marker='D', markersize=5, label="F-measure")
    # 绘制坐标轴标签
    plt.xlabel("Max_iter")
    plt.ylabel("F-measure")
    plt.show()


if __name__ == "__main__":
    train()
