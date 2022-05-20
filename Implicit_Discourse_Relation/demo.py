import pandas as pd
import numpy as np
import re
import pickle
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
import warnings
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, regularizers, activations
from tensorflow.keras.utils import to_categorical
from tensorflow import optimizers
import seaborn as sns


# 提取论元特征
def get_feature_average(data):
    feature = np.zeros((len(data), 300), dtype=np.float32)  # 初始化所有论元的特征向量
    for i, arg in enumerate(data):
        arg = re.sub(r'[^A-Za-z0-9 ]+', '', arg)  # 删去论元中除 A-Z,a-z,0-9,空格 之外的字符
        arg = arg.split(' ')  # 根据空格将论元划分为list
        vector = np.zeros(300, dtype=np.float32)  # 初始化当前论元的特征向量
        for word in arg:
            vector += word_vec[word]  # 对论元中所有词的词向量求和
        feature[i] = vector / len(arg)  # 取平均，作为当前论元的特征向量
    return feature


# 提取论元特征
def get_feature_max(data):
    feature = np.zeros((len(data), 300), dtype=np.float32)  # 初始化所有论元的特征向量
    for i, arg in enumerate(data):
        arg = re.sub(r'[^A-Za-z0-9 ]+', '', arg)  # 删去论元中除 A-Z,a-z,0-9,空格 之外的字符
        arg = arg.split(' ')  # 根据空格将论元划分为list
        vector = np.zeros(300, dtype=np.float32)  # 初始化当前论元的特征向量
        for word in arg:
            for j in range(len(vector)):
                vector[j] = max(vector[j], word_vec[word][j])  # 对论元中所有词的词向量求和
        feature[i] = vector  # 取平均，作为当前论元的特征向量
    return feature


# 提取论元特征
def get_feature_min(data):
    feature = np.zeros((len(data), 300), dtype=np.float32)  # 初始化所有论元的特征向量
    for i, arg in enumerate(data):
        arg = re.sub(r'[^A-Za-z0-9 ]+', '', arg)  # 删去论元中除 A-Z,a-z,0-9,空格 之外的字符
        arg = arg.split(' ')  # 根据空格将论元划分为list
        vector = np.zeros(300, dtype=np.float32)  # 初始化当前论元的特征向量
        for word in arg:
            for j in range(len(vector)):
                vector[j] = min(vector[j], word_vec[word][j])  # 对论元中所有词的词向量求和
        feature[i] = vector  # 取平均，作为当前论元的特征向量
    return feature


# 提取论元特征
def get_feature_first2(data):
    feature = np.zeros((len(data), 600), dtype=np.float32)  # 初始化所有论元的特征向量
    for i, arg in enumerate(data):
        arg = re.sub(r'[^A-Za-z0-9 ]+', '', arg)  # 删去论元中除 A-Z,a-z,0-9,空格 之外的字符
        arg = arg.split(' ')  # 根据空格将论元划分为list
        str0 = arg[0]
        if len(arg) == 1:
            feature[i] += np.concatenate((word_vec[str0], word_vec[str0]))
        else:
            str1 = arg[1]
            feature[i] += np.concatenate((word_vec[str0], word_vec[str1]))
    return feature


# 提取论元特征
def get_feature_last2(data):
    feature = np.zeros((len(data), 600), dtype=np.float32)  # 初始化所有论元的特征向量
    for i, arg in enumerate(data):
        arg = re.sub(r'[^A-Za-z0-9 ]+', '', arg)  # 删去论元中除 A-Z,a-z,0-9,空格 之外的字符
        arg = arg.split(' ')  # 根据空格将论元划分为list
        str0 = arg[-1]
        if len(arg) == 1:
            feature[i] += np.concatenate((word_vec[str0], word_vec[str0]))
        else:
            str1 = arg[-2]
            feature[i] += np.concatenate((word_vec[str1], word_vec[str0]))
    return feature


def Find_Best_Param_SVM(X, y):
    print('Find Best Parameter...')
    y = y.flatten()
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=3)
    C_best = 0
    sigma_best = 0
    Cparalist = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1]
    Sparalist = [2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4]
    predict_best = 0.0
    dlist = pd.DataFrame(columns=Cparalist, index=Sparalist[::-1])
    dlist.index.name = 'sigma'
    dlist.columns.name = 'C'
    for C_now in Cparalist:
        for sigma_now in Sparalist[::-1]:
            classifier = svm.SVC(C=C_now, kernel='rbf', gamma=np.power(sigma_now, -2.0) / 2,
                                 decision_function_shape='ovo')
            classifier.fit(X_train, Y_train)
            predict = sum(cross_val_score(classifier, X, y,
                                          cv=5)) / 5  # Returns the mean accuracy on the given test data and labels
            print('C, Sigma, Validation Score:')
            print(C_now, sigma_now, predict)
            dlist.loc[sigma_now, C_now] = predict
            if predict > predict_best:
                predict_best, C_best, sigma_best = predict, C_now, sigma_now
    print("Accuracy List:")
    print(dlist)
    fig, ax = plt.subplots(figsize=(10, 10))
    df = pd.DataFrame((np.array(dlist) * 100 // 1).astype(np.int))
    df.index.name = 'sigma'
    df.columns.name = 'C'
    sns.heatmap(df, vmin=35, vmax=50, cmap='RdBu_r',
                xticklabels=Cparalist, yticklabels=Sparalist[::-1])
    ax.set_title('Parameter Scan')
    plt.show()
    print('Best Parameter(C, sigma, accuracy):')
    print(str(C_best) + ' ' + str(sigma_best) + ' ' + str(predict_best))
    return C_best, sigma_best, predict_best


def train_SVM(C, sigma, X, Y, X_test):
    print('Training...')
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2)
    model = svm.SVC(C=C, kernel='rbf', gamma=np.power(sigma, -2.0) / 2, decision_function_shape='ovo')
    model.fit(X_train, Y_train)
    print('Predicting...')
    val_acc = model.score(X_val, Y_val)
    print(f'Validation Set: Acc={val_acc:.4f}')
    # SVM分类
    clf = svm.SVC(C=C, kernel='rbf', gamma=np.power(sigma, -2.0) / 2, decision_function_shape='ovo')
    clf.fit(X, Y)

    # 计算训练集上的Acc和F1
    train_pred = clf.predict(X)
    train_acc = accuracy_score(Y, train_pred)
    train_f1 = f1_score(Y, train_pred, average='macro')
    print(f'Train Set: Acc={train_acc:.4f}, F1={train_f1:.4f}')

    # 计算测试集预测结果并保存
    test_pred = clf.predict(X_test)
    with open('IDRR实验结果_信卓1901_U201913504_王浩芃.txt', 'w') as f:
        for label in test_pred:
            f.write(str(label) + '\n')
    f.close()
    return model.predict(X_val)


def train_NN(X, Y, X_test, category, inputsize):
    y = to_categorical(Y)
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)
    # 定义Sequential类
    model = models.Sequential()
    # 全连接层，1024个节点
    model.add(layers.Dense(1024, activation='relu', input_shape=(inputsize,),
                           kernel_regularizer=regularizers.l2(0.001),
                           bias_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.2))
    # 全连接层，256个节点
    model.add(layers.Dense(256, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001),
                           bias_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.2))
    # 全连接层，64个节点
    model.add(layers.Dense(64, activation='relu',
                           kernel_regularizer=regularizers.l2(0.001),
                           bias_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dropout(0.2))
    # 全连接层，得到输出
    model.add(layers.Dense(category, activation='softmax'))
    # loss
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(X_train,
                        Y_train,
                        epochs=100,
                        batch_size=128,
                        validation_data=(X_val, Y_val),
                        validation_freq=1)
    model.summary()
    results = model.evaluate(X_val, Y_val)
    print(results)

    # 4、model predict
    y_pred = model.predict(X_test)
    # 5、model evaluation
    des = np.argmax(y_pred, axis=1)
    np.savetxt('IDRR实验结果_信卓1901_U201913504_王浩芃.txt', des, fmt='%d', delimiter='\n')
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
    return model.predict(X_val)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    print('Data Processing...')
    # 读取数据
    train_data = pd.read_csv('train.tsv', delimiter='\t', header=None)
    test_data = pd.read_csv('test.tsv', delimiter='\t', header=None)
    print(train_data[1].value_counts())
    train_arg1, train_arg2, train_label = train_data.iloc[:, 2], train_data.iloc[:, 3], train_data.iloc[:, 1]
    test_arg1, test_arg2 = test_data.iloc[:, 2], test_data.iloc[:, 3]

    # 加载词向量文件
    word_vec = pickle.load(open('glove_300.pickle', 'rb'))
    train_arg1_feature = get_feature_average(train_arg1)  # 提取训练集中所有论元1的特征
    train_arg2_feature = get_feature_average(train_arg2)
    test_arg1_feature = get_feature_average(test_arg1)
    test_arg2_feature = get_feature_average(test_arg2)
    # train_arg1_feature_first2 = get_feature_first2(train_arg1)  # 提取训练集中所有论元1的特征
    # train_arg2_feature_first2 = get_feature_first2(train_arg2)
    # test_arg1_feature_first2 = get_feature_first2(test_arg1)
    # test_arg2_feature_first2 = get_feature_first2(test_arg2)
    # train_arg1_feature_last2 = get_feature_last2(train_arg1)  # 提取训练集中所有论元1的特征
    # test_arg1_feature_last2 = get_feature_last2(test_arg1)
    train_feature = np.concatenate((train_arg1_feature, train_arg2_feature), axis=1)  # 将论元1和论元2的特征拼接
    test_feature = np.concatenate((test_arg1_feature, test_arg2_feature), axis=1)
    # Cfinal, sigmafinal, predict_final = Find_Best_Param_SVM(train_feature, train_label)

    # 对类别标签编码
    class_dict = {'Comparison': 0, 'Contingency': 1, 'Expansion': 2, 'Temporal': 3}
    class_dict_com = {'Comparison': 1, 'Contingency': 0, 'Expansion': 0, 'Temporal': 0}
    class_dict_con = {'Comparison': 0, 'Contingency': 1, 'Expansion': 0, 'Temporal': 0}
    class_dict_exp = {'Comparison': 0, 'Contingency': 0, 'Expansion': 1, 'Temporal': 0}
    class_dict_tem = {'Comparison': 0, 'Contingency': 0, 'Expansion': 0, 'Temporal': 1}

    train_label_total = np.array([class_dict[label] for label in train_label])
    train_SVM(0.8, 3, train_feature, train_label_total, test_feature)
    # train_label_com = np.array([class_dict_com[label] for label in train_label])
    # train_label_con = np.array([class_dict_con[label] for label in train_label])
    # train_label_exp = np.array([class_dict_exp[label] for label in train_label])
    # train_label_tem = np.array([class_dict_tem[label] for label in train_label])
    # # result_SVM = train_SVM(1, 4, train_feature, train_label_total, test_feature)
    # result0 = train_NN(train_feature, train_label_total, test_feature, 4, len(train_feature[0]))
    # category0 = np.argmax(result0, axis=1)
    # result = []
    # result.append(train_NN(train_feature, train_label_com, test_feature, 2, len(train_feature[0])))
    # result.append(train_NN(train_feature, train_label_con, test_feature, 2, len(train_feature[0])))
    # result.append(train_NN(train_feature, train_label_exp, test_feature, 2, len(train_feature[0])))
    # result.append(train_NN(train_feature, train_label_tem, test_feature, 2, len(train_feature[0])))
    # X_train, X_val, Y_train, Y_val = train_test_split(train_feature, train_label_total, test_size=0.2, random_state=0,
    #                                                   shuffle=False)
    # print(result0)
    # for i in range(len(result0)):
    #     for j in range(4):
    #         if j == 0:
    #             result0[i][j] += 0.1 * result[j][i][1]
    #         elif j == 1:
    #             result0[i][j] += 0.1 * result[j][i][1]
    #         elif j == 2:
    #             result0[i][j] += 0.05 * result[j][i][1]
    #         else:
    #             result0[i][j] += 0.15 * result[j][i][1]
    # print(result)
    #
    # category = np.argmax(result0, axis=1)
    # print(category0)
    # print(category)
    # print(Y_val)
    # ans = 0
    # for i in range(len(category)):
    #     if category0[i] == Y_val[i]:
    #         ans += 1
    # print(ans)
