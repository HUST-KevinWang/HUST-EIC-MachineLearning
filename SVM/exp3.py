import numpy as np
import scipy.io as scio
from sklearn import svm
import SVM_Functions as svmF
from sklearn.model_selection import train_test_split
import pandas as pd


def svm_sklearn():
    data = scio.loadmat('./task3/task3_train.mat')
    data_t = scio.loadmat('./task3/task3_test.mat')
    X = data['X']
    y = data['y'].flatten()
    X_t = data_t['X']
    shape = np.shape(X)
    shape_t = np.shape(X_t)
    print('训练集样本数:%d,特征维度:%d' % (shape[0], shape[1]))
    print('测试样本数:%d,特征维度:%d' % (shape_t[0], shape_t[1]))
    print('Training Linear SVM (Spam Classification)')

    c = 10
    sigma = 10
    clf = svm.SVC(C=c, kernel='rbf', gamma=np.power(sigma, -2.0) / 2, decision_function_shape='ovr')
    clf.fit(X, y)
    p = clf.predict(X)
    print('Training Accuracy: {}'.format(np.mean(p == y) * 100))

    result = clf.predict(X_t)
    np.savetxt('exp3_result.txt', result, fmt='%d', delimiter='\n')


def Find_Best_Param():
    X, y = svmF.loadData('./task3/task3_train.mat')
    y = y.flatten()
    X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=0)
    C_best = 0
    sigma_best = 0
    Clist = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    slist = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    predict_best = 0.0
    dlist = pd.DataFrame(columns=Clist, index=slist)
    dlist.index.name = 'sigma'
    dlist.columns.name = 'C'
    for C_now in Clist:
        for sigma_now in slist:
            classifier = svm.SVC(C=C_now, kernel='rbf', gamma=np.power(sigma_now, -2.0) / 2, decision_function_shape='ovr')
            classifier.fit(X_train, Y_train)
            predict = classifier.score(X_val, Y_val)  # Returns the mean accuracy on the given test data and labels
            dlist.loc[sigma_now, C_now] = predict
            if predict > predict_best:
                predict_best, C_best, sigma_best = predict, C_now, sigma_now
    print("Accuracy List:")
    print(dlist)
    print('Best Parameter(C, sigma, accuracy):')
    print(str(C_best)+' '+str(sigma_best)+' '+str(predict_best))
    return C_best, sigma_best, predict_best


if __name__ == '__main__':
    svm_sklearn()
# c = 0.1
# sigma = 0.1
# clf = svm.SVC(c, kernel='rbf', gamma=np.power(sigma, -2))
