# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


# /usr/bin/python
# encoding:utf-8


# 对原始数据进行分为训练数据和测试数据
import numpy as np
import pandas as pd
from sklearn import tree, impute, metrics
import graphviz

Train_data_f = pd.read_csv('./Task2/TrainDT.csv', encoding='gb2312')
Test_data_f = pd.read_csv('./Task2/TestDT.csv', encoding='gb2312')
imputer = impute.SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-100)
Train_data = pd.DataFrame(imputer.fit_transform(Train_data_f))
Test_data = pd.DataFrame(imputer.fit_transform(Test_data_f))
Train_data.columns = Train_data_f.columns
Test_data.columns = Test_data_f.columns
feature_train = Train_data_f[['finLabel', 'BSSIDLabel', 'RoomLabel']]
feature_test = Test_data_f[['finLabel', 'BSSIDLabel', 'RoomLabel']]


BSSID_v = list(set(Train_data_f['BSSIDLabel']))
BSSID_l = len(BSSID_v)
train_bssid = feature_train.groupby('finLabel')
train_input = []
Train_data_classes = []


for i, v in train_bssid:
    tmp = np.array(v['BSSIDLabel'])
    tmpa = BSSID_l * [0]
    for bssidv in BSSID_v:
        if bssidv in tmp:
            tmpa[BSSID_v.index(bssidv)] = 1
    train_input.append(tmpa)
    roomid = np.array(v['RoomLabel'])
    Train_data_classes.append(roomid[1])
Train_data_inputs = np.array(train_input)


test_bssid = feature_test.groupby('finLabel')
test_input = []
Test_data_classes = []


for i, v in test_bssid:
    tmp = np.array(v['BSSIDLabel'])
    tmpa = BSSID_l * [0]
    for bssidv in BSSID_v:
        if bssidv in tmp:
            tmpa[BSSID_v.index(bssidv)] = 1
    test_input.append(tmpa)
    roomid = np.array(v['RoomLabel'])
    Test_data_classes.append(roomid[1])
Test_data_inputs = np.array(test_input)


decision_tree_classifier = tree.DecisionTreeClassifier()
decision_tree_classifier.fit(Train_data_inputs, Train_data_classes)
decision_tree_output = decision_tree_classifier.predict(Test_data_inputs)
dot_data = tree.export_graphviz(decision_tree_classifier, out_file=None,
                                feature_names=BSSID_v,
                                class_names=['ROOM1', 'ROOM2', 'ROOM3', 'ROOM4'],
                                filled=True,
                                impurity=True,
                                rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("WiFi")

print('真实值是：')
print(Test_data_classes)
print('预测值是:')
print(decision_tree_output)
print('ACCURACY')
print(str(len(decision_tree_output))+'/'+str(len(Test_data_classes)))
print(metrics.accuracy_score(Test_data_classes, decision_tree_output))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
