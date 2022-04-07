import numpy as np
from numpy import array, ones, log
import random


def textParse(bigString):  # input is big string, #output is word list
    """
        接受一个大字符串并将其解析为字符串列表。该函数去掉少于两个字符的字符串，并将所有字符串转换为小写。
    """
    import re
    listOfTokens = re.split(r'\W', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def createVocabList(dataSet):
    """
        创建一个包含在所有文档中出现的不重复的词的列表。
    """
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)


def bagOfWords2VecMN(vocabList, inputSet):
    """
        获得文档向量，向量中的数值代表词汇表中的某个单词在一篇文档中的出现次数
    """
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)                                    # 创建一个其中所含元素都为0的向量
    for word in inputSet:                                                # 遍历每个词条
        if word in vocabList:                                            # 如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec                                                    # 返回文档向量


def trainNB0(trainMatrix, trainCategory):  # 输入参数为文档矩阵trainMatrix,和每篇文档类别标签所构成的向量trainCategory
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)  # 正例占比
    #p0Num = zeros(numWords); p1Num = zeros(numWords)     #初始化概率
    #p0Denom = 0.0; p1Denom = 0.0
    p0Num = ones(numWords)  # 防止0概率出现
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for j in range(numTrainDocs):
        if trainCategory[j] == 1:
            p1Num += trainMatrix[j]  # 正例中每个词出现次数
            p1Denom += sum(trainMatrix[j])  # 正例总词数
        else:
            p0Num += trainMatrix[j]  # 反例中每个词出现次数
            p0Denom += sum(trainMatrix[j])  # 反例总词数
    #p1Vect = p1Num/p1Denom         #change to log()
    #p0Vect = p0Num/p0Denom         #change to log()
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    return p0Vect, p1Vect, pAbusive  # 返回反例每个词出现概率取对数，正例每个词出现概率取对数，正例占比


#朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass):   # 参数分别为：要分类的向量以及使用trainNB0()计算得到的三个概率
    p1 = sum(vec2Classify*p1Vec)+np.log(pClass)
    p0 = sum(vec2Classify*p0Vec)+np.log(1-pClass)
    if p1 > p0:
        return 1
    else:
        return 0


def spamTest():
    # 初始化数据列表
    docList = []
    classList = []
    fullText = []
    # spam和ham文件夹里的邮件是25封,所以用for循环25次
    for i in range(1, 26):
        # 切分文本
        wordList = textParse(open('./task1/spam/%d.txt' % i, encoding='utf-8').read())
        # 切分后的文本以原始列表形式加入文档列表
        docList.append(wordList)
        # 切分后的文本直接合并到词汇列表
        fullText.extend(wordList)
        # 标签列表更新
        classList.append(1)

        wordList = textParse(open('./task1/ham/%d.txt' % i, encoding='utf-8').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    # 创建一个包含所有文档中出现的不重复词的列表
    vocabList = createVocabList(docList)
    # 初始化训练集和测试集列表
    trainingSet = list(range(50))
    testSet = []
    # 随机构建测试集，随机选取10个样本作为测试样本，并从训练样本中剔除
    for i in range(10):
        # randIndex=random.uniform(a,b)用于生成指定范围内的随机浮点数
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 将该样本加入测试集中
        testSet.append(trainingSet[randIndex])
        # 同时将该样本从训练集中剔除
        del (trainingSet[randIndex])

    # 初始化训练集数据列表和标签列表
    trainMat = []
    trainClasses = []

    # 遍历训练集
    for docIndex in trainingSet:
        # 词表转换为向量，并加入到训练数据列表中
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        # 相应的标签也加入训练标签列表中
        trainClasses.append(classList[docIndex])
    # 朴素贝叶斯分类器训练函数
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    # 初始化错误计数
    errorCount = 0

    # 遍历测试集来测试
    for docIndex in testSet:
        # 词表转换为向量
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        # 判断分类结果与原标签是否一致
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            # 如果不一致则错误计数加1
            errorCount += 1
            # 并且输出出错的文档
            print("classification error", docList[docIndex])
    # 打印输出信息
    print('the error rate is: ', float(errorCount) / len(testSet))
    # 返回词汇表和全部单词列表
    return float(errorCount) / len(testSet)


if __name__ == '__main__':
    ave = 0
    n = 1000
    for i in range(n):
        ave += spamTest()
    print('AVERAGE ERROR RATE')
    print(ave/n)
    print('ACCURACY')
    print(1-ave / n)
