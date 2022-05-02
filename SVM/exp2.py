import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
import SVM_Functions as svmF
from exp1 import visualize_boundary
import warnings
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)


def svm_svmF():
    X, y = svmF.loadData('./task2/task2.mat')
    svmF.plotData(X, y)

    model = svmF.svmTrain_SMO(X, y, C=1, kernelFunction='gaussian', K_matrix=svmF.gaussianKernel(X, sigma=0.1))
    svmF.visualizeBoundaryGaussian(X, y, model, sigma=0.1)


def svm_sklearn():
    c = 1
    sigma = 0.1
    X, y = svmF.loadData('./task2/task2.mat')
    svmF.plotData(X, y)
    clf = svm.SVC(C=c, kernel='rbf', gamma=np.power(sigma, -2))
    clf.fit(X, y)
    visualize_boundary(clf, X, y, -0.65, 0.3, -0.7, 0.6)


def Find_Best_Param():
    X, y = svmF.loadData('./task2/task2.mat')
    y = y.flatten()
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    C_best = 0
    sigma_best = 0
    Clist = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    slist = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    res = []
    predict_best = 0.0
    dlist = pd.DataFrame(columns=Clist, index=slist)
    dlist.index.name = 'sigma'
    dlist.columns.name = 'C'
    for C_now in Clist:
        res1 = []
        for sigma_now in slist:
            classifier = svm.SVC(C=C_now, kernel='rbf', gamma=np.power(sigma_now, -2.0) / 2, decision_function_shape='ovr')
            classifier.fit(X_train, Y_train)
            predict = classifier.score(X_val, Y_val)  # Returns the mean accuracy on the given test data and labels
            dlist.loc[sigma_now, C_now] = predict
            res1.append(predict)
            if predict > predict_best:
                predict_best, C_best, sigma_best = predict, C_now, sigma_now
        res.append(res1)
    print("Accuracy List:")
    print(dlist)
    print('Best Parameter(C, sigma, accuracy):')
    print(str(C_best)+' '+str(sigma_best)+' '+str(predict_best))
    return C_best, sigma_best, predict_best


def scan_parameter():
    X, y = svmF.loadData('./task2/task2.mat')
    y = y.flatten()
    Clist = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    slist = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    dlist = pd.DataFrame(columns=Clist, index=slist)
    dlist.index.name = 'sigma'
    dlist.columns.name = 'C'
    for C_now in Clist:
        res1 = []
        for sigma_now in slist:
            classifier = svm.SVC(C=C_now, kernel='rbf', gamma=np.power(sigma_now, -2.0) / 2,
                                 decision_function_shape='ovr')
            classifier.fit(X, y)
            predict = classifier.score(X, y)  # Returns the mean accuracy on the given test data and labels
            dlist.loc[sigma_now, C_now] = predict
            res1.append(predict)
    print("Accuracy List:")
    print(dlist)


if __name__ == '__main__':
    svm_svmF()

# Clist = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
# slist = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
# dlist = pd.DataFrame(columns=Clist, index=slist)
# dlist.index.name = 'sigma'
# dlist.columns.name = 'C'
