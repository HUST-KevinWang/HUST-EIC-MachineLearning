import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
import SVM_Functions as svmF


def visualize_boundary(clf, X, y, x_min, x_max, y_min, y_max, title=None):
    X_pos = []
    X_neg = []
    sampleArray = np.concatenate((X, y), axis=1)
    for array in list(sampleArray):
        if array[-1]:
            X_pos.append(array)
        else:
            X_neg.append(array)
    X_pos = np.array(X_pos)
    X_neg = np.array(X_neg)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if title:
        ax.set_title(title)
    pos = plt.scatter(X_pos[:, 0], X_pos[:, 1], marker='+', c='b')
    neg = plt.scatter(X_neg[:, 0], X_neg[:, 1], marker='o', c='y')
    plt.legend((pos, neg), ('postive', 'negtive'), loc=2)
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='r', levels=[0])
    plt.show()


def linear():
    X, y = svmF.loadData('./task1/task1_linear.mat')
    svmF.plotData(X, y, 'Distribution')

    model = svmF.svmTrain_SMO(X, y, C=1, max_iter=20)
    svmF.visualizeBoundaryLinear(X, y, model)

    clf = svm.SVC(C=1, kernel='linear', tol=1e-3)
    clf.fit(X, y)
    visualize_boundary(clf, X, y, 0, 4.5, 1.5, 5)


def gaussian():
    Xg, yg = svmF.loadData('./task1/task1_gaussian.mat')
    svmF.plotData(Xg, yg)
    modelg = svmF.svmTrain_SMO(Xg, yg, C=1, kernelFunction='gaussian', K_matrix=svmF.gaussianKernel(Xg, sigma=0.1))
    svmF.visualizeBoundaryGaussian(Xg, yg, modelg, sigma=0.1)


def gaussian_sklearn():
    c = 1
    sigma = 0.1
    X, y = svmF.loadData('./task1/task1_gaussian.mat')
    svmF.plotData(X, y)
    clf = svm.SVC(C=c, kernel='rbf', gamma=np.power(sigma, -2))
    clf.fit(X, y)
    visualize_boundary(clf, X, y, 0, 1.02, 0.38, 1.05)


if __name__ == '__main__':
    gaussian_sklearn()
