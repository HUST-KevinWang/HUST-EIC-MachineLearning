from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import cross_val_score
warnings.filterwarnings("ignore")
# 加载数据
df = pd.read_csv(r'./task1/钞票训练集.txt', header=None)
X_data = np.array(df.loc[:][[0, 1, 2, 3]])
y_data = df.loc[:][4].values
# 拆分测试集、训练集
X_train, X_test, Y_train, Y_test = train_test_split(X_data, y_data, test_size=0.20, random_state=0)
dpred = pd.read_csv(r'./task1/钞票测试集.txt', header=None)
X_pred = np.array(dpred.loc[:][[0, 1, 2, 3]])
# # 标准化特征值（观察结果使准确度下降了0.3%，所以不再使用）
# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)

# 训练回归模型
model = LogisticRegression(max_iter=20)
model.fit(X_train, Y_train)


# 预测
y_pred_local = model.predict(X_pred)
y_pred = model.predict(X_test)
result = pd.DataFrame(X_pred)
result['result'] = y_pred_local
num = range(1, len(X_pred) + 1)
result.insert(loc=0, column='num', value=num)
# np.savetxt("result.csv", result, delimiter=",")
result.to_csv('exp1_result.csv', index=False, header=False)
print('训练集分数：' + str(accuracy_score(Y_train, model.predict(X_train))))
print("验证集分数:", model.score(X_test, Y_test))
print("model accuracy is " + str(accuracy_score(Y_test, y_pred)))
print("model precision is " + str(precision_score(Y_test, y_pred, average='macro')))
print("model recall is " + str(recall_score(Y_test, y_pred, average='macro')))
print("model f1_score is " + str(f1_score(Y_test, y_pred, average='macro')))
print("4-fold cross val score is " + str(sum(cross_val_score(model, X_data, y_data, cv=5)) / 5))

# b = model.coef_
# a = model.intercept_
# print(a)
# print(b)
# x = range(1, n+1)
# plt.plot(x, ans, "g", marker='D', markersize=5, label="F-measure")
# # 绘制坐标轴标签
# plt.xlabel("max_iter")
# plt.ylabel("F-measure")
# plt.show()
