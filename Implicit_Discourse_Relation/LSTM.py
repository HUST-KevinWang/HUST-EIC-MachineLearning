import pandas as pd
import numpy as np
import re
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
import warnings
import matplotlib.pyplot as plt
from tensorflow.keras import models, regularizers, activations
from tensorflow.keras.layers import Conv1D, Bidirectional, LSTM, Dense, Input, Dropout, Embedding
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.utils import to_categorical
from tensorflow import optimizers
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

MAX_WORD = 100000
MAX_SEQ_LENGTH = 200
EMBEDDING_DIM = 300
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3


# 提取论元特征
def to_list(data1, data2):
    feature = []
    feature1 = []
    feature2 = []
    for i, arg in enumerate(data1):
        arg = re.sub(r'[^A-Za-z0-9 ]+', '', str(arg).lower().strip())  # 删去论元中除 A-Z,a-z,0-9,空格 之外的字符
        arg = arg.split(' ')
        if len(arg) < MAX_SEQ_LENGTH // 2:
            exls = (MAX_SEQ_LENGTH // 2 - len(arg)) * [' ']
            arg.extend(exls)
        elif len(arg) > MAX_SEQ_LENGTH // 2:
            arg = arg[0:MAX_SEQ_LENGTH // 2]
        feature1.append(arg)
    for i, arg in enumerate(data2):
        arg = re.sub(r'[^A-Za-z0-9 ]+', '', str(arg).lower().strip())  # 删去论元中除 A-Z,a-z,0-9,空格 之外的字符
        arg = arg.split(' ')
        if len(arg) < MAX_SEQ_LENGTH // 2:
            exls = (MAX_SEQ_LENGTH // 2 - len(arg)) * [' ']
            arg.extend(exls)
        elif len(arg) > MAX_SEQ_LENGTH // 2:
            arg = arg[0:MAX_SEQ_LENGTH // 2]
        feature2.append(arg)
    for i in range(len(feature1)):
        feature.append(feature1[i] + feature2[i])
    return feature


if __name__ == '__main__':
    print("Tensorflow version : ", tf.__version__)
    train_data = pd.read_csv('train.tsv', delimiter='\t', header=None)
    test_data = pd.read_csv('test.tsv', delimiter='\t', header=None)
    train_data.columns = ['sentiment', 'category', 'text1', 'text2']
    test_data.columns = ['sentiment', 'category', 'text1', 'text2']
    print(train_data.head())
    print(train_data['category'].value_counts())
    train_arg1, train_arg2, train_label = train_data.iloc[:, 2], train_data.iloc[:, 3], train_data.iloc[:, 1]
    train_ls = to_list(train_arg1, train_arg2)
    test_arg1, test_arg2, test_label = test_data.iloc[:, 2], test_data.iloc[:, 3], test_data.iloc[:, 1]
    test_ls = to_list(test_arg1, test_arg2)
    X_train, X_test, Y_train, Y_test = train_test_split(train_ls, train_label, test_size=0.2, random_state=0)
    Y_train.columns = ['category']
    Y_test.columns = ['category']

    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(X_train.text)
    # word_index = tokenizer.word_index
    # vocab_size = len(word_index) + 1
    # print(vocab_size)
    #
    # x_train = pad_sequences(tokenizer.texts_to_sequences(X_train.text),
    #                         maxlen=MAX_SEQ_LENGTH)
    # x_test = pad_sequences(tokenizer.texts_to_sequences(X_test.text),
    #                        maxlen=MAX_SEQ_LENGTH)
    # print(x_train)
    class_dict = {'Comparison': 0, 'Contingency': 1, 'Expansion': 2, 'Temporal': 3}
    y_train = to_categorical(np.array([class_dict[label] for label in Y_train]))
    y_test = to_categorical(np.array([class_dict[label] for label in Y_test]))
    print(y_train.shape)

    # 加载词向量文件
    embedding_index = pickle.load(open('glove_300.pickle', 'rb'))
    embedding_index[' '] = [0] * 300
    vocab_size = len(embedding_index)
    vocab = [x.lower() for x in list(embedding_index.keys())]
    train_len = len(X_train)
    test_len = len(X_test)
    test_local_len = len(test_ls)
    x_train = np.zeros((train_len, MAX_SEQ_LENGTH))
    x_test = np.zeros((test_len, MAX_SEQ_LENGTH))
    x_test_local = np.zeros((test_local_len, MAX_SEQ_LENGTH))
    for i in range(train_len):
        for j in range(len(X_train[i])):
            x_train[i][j] = vocab.index(X_train[i][j])
    for i in range(test_len):
        for j in range(len(X_test[i])):
            x_test[i][j] = vocab.index(X_test[i][j])
    for i in range(test_local_len):
        for j in range(len(test_ls[i])):
            x_test_local[i][j] = vocab.index(test_ls[i][j])
    print(x_train)

    warnings.filterwarnings("ignore")
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    num = 0

    for word, values in embedding_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            if num < vocab_size:
                embedding_matrix[num, :] = embedding_vector
            num += 1
    print(embedding_matrix.shape)
    sequence_input = Input(shape=(MAX_SEQ_LENGTH,), dtype='int32')
    embedding_layer = Embedding(vocab_size,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQ_LENGTH,
                                trainable=False)
    embedding_sequence = embedding_layer(sequence_input)

    x = SpatialDropout1D(0.2)(embedding_sequence)
    print(x.shape)
    #     x = Conv1D(64, 5, activation='relu')(x)
    # print(x.shape)
    x1 = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x[:, 0:MAX_SEQ_LENGTH // 2])
    x2 = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))(x[:, MAX_SEQ_LENGTH // 2:])
    x = tf.concat([x1, x2], axis=1)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.05), bias_regularizer=regularizers.l2(0.05))(
        x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.05), bias_regularizer=regularizers.l2(0.05))(
        x)
    outputs = Dense(4, activation='softmax')(x)
    print(outputs.shape)
    model = tf.keras.Model(sequence_input, outputs)
    model.compile(optimizer=optimizers.RMSprop(learning_rate=LR),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    ReduceLR = ReduceLROnPlateau(factor=0.1, min_lr=0.01, monitor='val_loss', verbose=1)
    history = model.fit(x_train,
                        y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(x_test, y_test),
                        validation_freq=1,
                        callbacks=[ReduceLR])
    model.summary()
    results = model.evaluate(x_test, y_test)
    print(results)
    # 4、model predict
    y_pred = model.predict(x_test_local)
    # 5、model evaluation
    des = np.argmax(y_pred, axis=1)
    np.savetxt('./IDRR实验结果_信卓1901_U201913504_王浩芃.txt', des, fmt='%d', delimiter='\n')
    print('测试集分类结果保存完毕')
