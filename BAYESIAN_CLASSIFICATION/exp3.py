# from keras.datasets import imdb
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import datetime

# 参数num_words = dimension 的意思是仅保留训练数据的前dimension个最常见出现的单词，低频单词将被舍弃。
dimension = 10000


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


if __name__ == "__main__":
    # 1、load data
    time1 = datetime.datetime.now()
    # (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=dimension)
    trainlines = open("./task3/train/train_data.txt").readlines()
    X_train = [vec_seq(line.split()) for line in trainlines]
    trainlabels = open("./task3/train/train_labels.txt").readlines()
    Y_train = list(map(int, [line.strip() for line in trainlabels]))
    testlines = open("./task3/test/test_data.txt").readlines()
    X_test = [vec_seq(line.split()) for line in testlines]
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

    # 3、model train
    # 多项式分布、伯努利分布：https://blog.csdn.net/qq_27009517/article/details/80044431
    # 二者的计算粒度不一样，多项式模型以单词为粒度，伯努利模型以文件为粒度，因此二者的先验概率和类条件概率的计算方法都不同。
    # 计算后验概率时，对于一个文档d，多项式模型中，只有在d中出现过的单词，才会参与后验概率计算，伯努利模型中，没有在d中出现，但是在全局单词表中出现的单词，也会参与计算，不过是作为“反方”参与的。
    # 当训练集文档较短，也就说不太会出现很多重复词的时候，多项式和伯努利模型公式的分子相等，多项式分母值大于伯努利分子值，因此多项式的似然估计值会小于伯努利的似然估计值。
    # 所以，当训练集文本较短时，我们更倾向于使用伯努利模型。而文本较长时，我们更倾向于多项式模型，因为，在一篇文档中的高频词，会使该词的似然概率值相对较大。
    # 高斯分布：
    # 适合连续变量

    time1 = datetime.datetime.now()
    # model = MultinomialNB()
    model = BernoulliNB()
    model.fit(X_train, Y_train)

    time2 = datetime.datetime.now()
    print("model train takes " + str((time2 - time1).seconds) + " s")
    # 4、model predict
    time1 = datetime.datetime.now()
    # y_pred = model.predict(X_test)
    y_pred_local = model.predict(X_test)
    time2 = datetime.datetime.now()
    print("model predict takes " + str((time2 - time1).seconds) + " s")
    # 5、model evaluation
    # print("model accuracy is " + str(accuracy_score(y_test, y_pred_local)))
    # print("model precision is " + str(precision_score(y_test, y_pred_local, average='macro')))
    # print("model recall is " + str(recall_score(y_test, y_pred_local, average='macro')))
    # print("model f1_score is " + str(f1_score(y_test, y_pred_local, average='macro')))
    print('训练集分数：'+str(accuracy_score(Y_train, model.predict(X_train))))
    des = y_pred_local.astype(int)
    np.savetxt('exp3_result.txt', des, fmt='%d', delimiter='\n')
    print('测试集分类结果保存完毕')
