from keras import activations
import textaug
import loading
import mmsdk
from mmsdk import mmdatasdk
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from keras.preprocessing.text import one_hot
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer


inputs, labels = loading.load_data_binary()
temp_input, temp_lab = inputs, labels
print('done loading')


# simple_aug_data, simple_aug_labels = textaug.simple_augment(inputs, labels)
# np.savetxt(X=simple_aug_data, fname='simple_augdata.txt', fmt='%s')
# np.savetxt(X=simple_aug_labels, fname='simple_auglabs.txt', fmt='%d')
# inputs, labels = temp_input, temp_lab


# simple_swap_data, simple_swap_labels = textaug.simple_with_swap(inputs, labels)
# np.savetxt(X=simple_swap_data, fname='simple_swap_augdata.txt', fmt='%s')
# np.savetxt(X=simple_swap_labels, fname='simple_swap_auglabs.txt', fmt='%d')
# inputs, labels = temp_input, temp_lab


simple_context_data, simple_context_labs = textaug.simple_with_context_sub(inputs, labels)
np.savetxt(X=simple_context_data, fname='simple_context_augdata.txt', fmt='%s')
np.savetxt(X=simple_context_labs, fname='simple_context_auglabs.txt', fmt='%d')
inputs, labels = temp_input, temp_lab
#
#
# complex_data, complex_labs = textaug.context_sub(inputs, labels)
# np.savetxt(X=simple_aug_data, fname='complex_augdata.txt', fmt='%s')
# np.savetxt(X=simple_aug_labels, fname='complex_auglabs.txt', fmt='%d')
# inputs, labels = temp_input, temp_lab

print("done!!!!!!!!!!!!!!!!")































































































# maxLength = 15
#
# X_train, X_test, y_train, y_test = train_test_split(aug_data, aug_lab,
#                                                     test_size=0.20,
#                                                     random_state=42)
#
# tokenizer = Tokenizer(num_words=5000)
# tokenizer.fit_on_texts(X_train)
#
# X_train = tokenizer.texts_to_sequences(X_train)
# X_test = tokenizer.texts_to_sequences(X_test)
#
# # Adding 1 because of reserved 0 index
# vocab_size = len(tokenizer.word_index) + 1
#
# maxlen = 14
#
# X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
# X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
#
#
# embeddings_dictionary = dict()
# glove_file = open('../../glove.6B.100d.txt',
#                   encoding="utf8")
#
# for line in glove_file:
#     records = line.split()
#     word = records[0]
#     vector_dimensions = np.asarray(records[1:], dtype='float32')
#     embeddings_dictionary[word] = vector_dimensions
# glove_file.close()
#
# embedding_matrix = np.zeros((vocab_size, 100))
# for word, index in tokenizer.word_index.items():
#     embedding_vector = embeddings_dictionary.get(word)
#     if embedding_vector is not None:
#         embedding_matrix[index] = embedding_vector
#
# model = Sequential()
# embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
# model.add(embedding_layer)
# # model.add(LSTM(256, return_sequences=True))
# # model.add(Activation(activations.tanh))
# model.add(LSTM(128))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
#
# model.save('model.h5')
# #m = keras.models.load_model('model.h5')
# #m.evaluate(X_test, y_test)
# score = model.evaluate(X_test, y_test, verbose=1)
# print("Test Score:", score[0])
# print("Test Accuracy:", score[1])
#
# import matplotlib.pyplot as plt
#
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
#
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train','test'], loc='upper left')
# plt.show()
#
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
#
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train','test'], loc='upper left')
# plt.show()
# plt.savefig("test.pdf")
