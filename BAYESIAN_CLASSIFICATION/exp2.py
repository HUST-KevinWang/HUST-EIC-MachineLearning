import numpy as np
from struct import unpack
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
import os
# 配置文件
config = {
    # 训练集文件
    'train_images_idx3_ubyte_file_path': './task2/MNIST/train-images.idx3-ubyte',
    # 训练集标签文件
    'train_labels_idx1_ubyte_file_path': './task2/MNIST/train-labels.idx1-ubyte',

    # 测试集文件
    'test_images_idx3_ubyte_file_path': './task2/MNIST/t10k-images.idx3-ubyte',
    # 测试集标签文件
    'test_labels_idx1_ubyte_file_path': './task2/MNIST/t10k-labels.idx1-ubyte',

    # numpy 文件路径
    'x_train_savepath': './task2/MNIST/x_train_savepath.npy',
    'x_test_savepath': './task2/MNIST/x_test_savepath.npy',
    'y_train_savepath': './task2/MNIST/y_train_savepath.npy',
    'y_test_savepath': './task2/MNIST/y_test_savepath.npy',
    # 特征提取阙值
    'binarization_limit_value': 0.14,

    # 特征提取后的边长
    'side_length': 28
}


def decode_idx3_ubyte(path):
    '''
    解析idx3-ubyte文件，即解析MNIST图像文件
    '''

    '''
    也可不解压，直接打开.gz文件。path是.gz文件的路径
    import gzip
    with gzip.open(path, 'rb') as f:
    '''
    print('loading %s' % path)
    with open(path, 'rb') as f:
        # 前16位为附加数据，每4位为一个整数，分别为幻数，图片数量，每张图片像素行数，列数。
        magic, num, rows, cols = unpack('>4I', f.read(16))
        print('magic:%d num:%d rows:%d cols:%d' % (magic, num, rows, cols))
        mnistImage = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
    print('done')
    return mnistImage


def decode_idx1_ubyte(path):
    '''
    解析idx1-ubyte文件，即解析MNIST标签文件
    '''
    print('loading %s' % path)
    with open(path, 'rb') as f:
        # 前8位为附加数据，每4位为一个整数，分别为幻数，标签数量。
        magic, num = unpack('>2I', f.read(8))
        print('magic:%d num:%d' % (magic, num))
        mnistLabel = np.fromfile(f, dtype=np.uint8)
    print('done')
    return mnistLabel


def normalizeImage(image):
    '''
    将图像的像素值正规化为0.0 ~ 1.0
    '''
    res = image.astype(np.float32) / 255.0
    return res


def load_train_images(path=config['train_images_idx3_ubyte_file_path']):
    return normalizeImage(decode_idx3_ubyte(path))


def load_train_labels(path=config['train_labels_idx1_ubyte_file_path']):
    return decode_idx1_ubyte(path)


def load_test_images(path=config['test_images_idx3_ubyte_file_path']):
    return normalizeImage(decode_idx3_ubyte(path))


def load_test_labels(path=config['test_labels_idx1_ubyte_file_path']):
    return decode_idx1_ubyte(path)


def oneImagesFeatureExtraction(image):
    '''
    对单张图片进行特征提取
    '''
    res = np.empty((config['side_length'], config['side_length']))
    num = 28//config['side_length']
    for i in range(0, config['side_length']):
        for j in range(0, config['side_length']):
            # tempMean = (image[2*i:2*(i+1),2*j:2*(j+1)] != 0).sum()/(2 * 2)
            tempMean = image[num*i:num*(i+1), num*j:num*(j+1)].mean()  # 根据配置图片大小进行多像素点均值运算，缩小图片尺寸
            if tempMean > config['binarization_limit_value']:  # 如果像素点均值大于某个阈值，则置1；反之，置0
                res[i, j] = 1
            else:
                res[i, j] = 0
    return res


def featureExtraction(images):
    res = np.empty((images.shape[0], config['side_length'],
                    config['side_length']), dtype=np.float32)  # 数据集数组
    for i in range(images.shape[0]):
        print(i)
        res[i] = oneImagesFeatureExtraction(images[i])  # 对每张图片进行特征提取
    return res


if __name__ == '__main__':
    if os.path.exists(config['x_train_savepath']) and os.path.exists(config['y_train_savepath']) and os.path.exists(
            config['x_test_savepath']) and os.path.exists(config['y_test_savepath']):
        print('-------------Load Datasets-----------------')
        X_train = np.load(config['x_train_savepath'])
        Y_train = np.load(config['y_train_savepath'])
        X_test = np.load(config['x_test_savepath'])
        Y_test = np.load(config['y_test_savepath'])
    else:
        print('-------------Generate Datasets-----------------')
        train_data = load_train_images()
        Y_train = load_train_labels()
        test_data = load_test_images()
        Y_test = load_test_labels()
        X_train = featureExtraction(train_data)
        X_test = featureExtraction(test_data)

        print('-------------Save Datasets-----------------')
        np.save(config['x_train_savepath'], X_train)
        np.save(config['y_train_savepath'], Y_train)
        np.save(config['x_test_savepath'], X_test)
        np.save(config['y_test_savepath'], Y_test)

    print('Fit the model')
    X_train = X_train.reshape(X_train.shape[0], config['side_length']*config['side_length'])
    X_test = X_test.reshape(X_test.shape[0], config['side_length']*config['side_length'])
    # 重构数据集维度为2匹配贝叶斯模型数据输入
    clf = BernoulliNB()  # 建立贝叶斯模型
    clf = clf.fit(X_train, Y_train)  # 训练集数据送入模型
    print('RESULT:')
    y_pred = clf.predict(X_test)
    print("训练集分数： %f 测试集分数: %f" % (clf.score(X_train, Y_train), clf.score(X_test, Y_test)))
    print("朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (X_test.shape[0], (Y_test != y_pred).sum()))
    print("model accuracy is " + str(accuracy_score(Y_test, y_pred)))
    print("model precision is " + str(precision_score(Y_test, y_pred, average='macro')))
    print("model recall is " + str(recall_score(Y_test, y_pred, average='macro')))
    print("model f1_score is " + str(f1_score(Y_test, y_pred, average='macro')))