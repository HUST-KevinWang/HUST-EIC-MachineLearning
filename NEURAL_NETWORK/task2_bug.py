import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras import models
from keras.utils import to_categorical
import numpy as np

import numpy as np
import os
import sys
import random
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Activation

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 读入数据
file_name = './Task2/train/train_texts.dat'
with open(file_name, 'rb') as f:
    train_texts = pickle.load(f)
file_name = './Task2/test/test_texts.dat'
with open(file_name, 'rb') as f:
    test_texts = pickle.load(f)

train_labals = []
fl = open('./Task2/train/train_labels.txt')
for line in fl.readlines():
    train_labals.append(line)

# TFIDF向量化
vectorizer = TfidfVectorizer(max_features=10000)
train_vector = vectorizer.fit_transform(train_texts)
print(train_vector.shape)
test_vector = vectorizer.transform(test_texts)
print(test_vector.shape)

one_hot_train_labels = to_categorical(train_labals)


model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=10, dropout=0.2))
model.add(LSTM(10, dropout_W=0.2, dropout_U=0.2))  # 输出维度 :100
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
model.add(Dense(20, activation='softmax'))

# # 定义Sequential类
# model = models.Sequential()
# # 全连接层，128个节点
# model.add(layers.Dense(128, activation='relu', input_shape=(10000,)))
# # 全连接层，64个节点
# model.add(layers.Dense(64, activation='relu'))
# # 全连接层，得到输出
# model.add(layers.Dense(20, activation='softmax'))
# loss
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# 拆分测试集与训练集
X_train, X_test, y_train, y_test = train_test_split(train_vector, one_hot_train_labels, test_size=0.2, random_state=0)
x_test = X_test.toarray()
partial_x_train = X_train.toarray()

history = model.fit(partial_x_train,
                    y_train,
                    epochs=1,
                    batch_size=64,
                    validation_data=(x_test, y_test),
                    validation_freq=1)
model.summary()
results = model.evaluate(x_test, y_test)

# 结果输出：
x_input = test_vector.toarray()
predictions = model.predict(x_input)
out_put = np.argmax(predictions, axis=1)
np.savetxt('exp2LSTM_result.txt', out_put, fmt='%d', delimiter='\n')


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