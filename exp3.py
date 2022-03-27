import numpy as np
import pandas as pd
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
import graphviz
import matplotlib.pyplot as plt


data = pd.read_csv("./Task3/titanic/train.csv")
data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis="columns", inplace=True)
X_train = data.drop(["Survived"], axis="columns")
# For the variable Y I can simply select the column Survived
Y_train = data["Survived"]  # Another way to declare: Y = data.Survived
X_train.Sex = X_train.Sex.map({"male": 0, "female": 1})
X_train.Age = X_train.Age.fillna(-100)
test = pd.read_csv("./Task3/titanic/test.csv")
test.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis="columns", inplace=True)
X_test = test.drop([], axis="columns")
X_test.Sex = X_test.Sex.map({"male": 0, "female": 1})
X_test.Age = X_test.Age.fillna(-100)
X_test.Fare = X_test.Fare.fillna(0)


model = tree.DecisionTreeClassifier(max_depth=6,
                                    min_samples_leaf=6,
                                    min_samples_split=10
                                    )
model.fit(X_train, Y_train)
pd.DataFrame({'PassengerID': range(892, 892+len(model.predict(X_test))),
              'Survived': model.predict(X_test)}).to_csv('gender_submission.csv', index=None)
# test.append(score)
# plt.plot(range(1, size+1), test, color='red')
# plt.ylabel('score')
# plt.xlabel('max_depth')
# plt.xticks(range(1, size+1))
# plt.show()
# pre = metrics.precision_score(Y_real, model.predict(X_test))
# recall = metrics.recall_score(Y_real, model.predict(X_test))
# print('ACCURACY')
# print(model.score(X_test, Y_real))
# print('PRECISION')
# print(pre)
# print('Recall')
# print(recall)
# print('F-measure')
# print(2*pre*recall/(pre+recall))
dot_data = tree.export_graphviz(model, out_file=None,
                                feature_names=X_train.columns.tolist(),
                                class_names=['DIED', 'SURVIVED'],
                                filled=True,
                                impurity=True,
                                rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("TITANIC")


