import json
import random
import re
import loading
import numpy as np
from keras.layers import Conv1D, Conv2D, Flatten, LSTM, MaxPooling1D, \
    MaxPooling2D, TimeDistributed
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Sequential
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

import textaug

inputs, labels = loading.load_data_binary()
assert len(inputs) == len(labels)
labels2 = labels[:]
data = np.loadtxt('simple_swap_augdata.txt', delimiter='\n', dtype=str)[2199:]
print('done')


X_train, y_train, test_data, labels, embedding_matrix, \
    vocab_size, maxlen, t = loading.set_up_data(data, inputs, labels)

print(type(X_train), type(y_train), type(test_data), type(labels))
model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix],
                            input_length=maxlen, trainable=False)
model.add(embedding_layer)
# model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
model.add(LSTM(512, return_sequences=True))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
# model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
# model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
# model.add(TimeDistributed(Flatten()))
# define LSTM model
model.add(LSTM(512))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)])
history = model.fit(X_train, y_train, batch_size=512, epochs=6, verbose=1,
                    validation_split=0.2)
model.save('binary simple swap - LSTM-512 -v2.h5')
score = model.evaluate(test_data, labels, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
