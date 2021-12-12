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
negative = []
positive = []

binary_model = keras.models.load_model("binary simple swap - LSTM-512.h5")
score = binary_model.evaluate(test_data, labels, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])
count_positives = 0
count_negatives = 0
correct = 0
for i in range(len(test_data)):
    example = test_data[i]
    # example = t.texts_to_sequences(example)
    # flat_list = []
    # for sublist in example:
    #     for item in sublist:
    #         flat_list.append(item)
    #
    # flat_list = [flat_list]
    #
    # example = pad_sequences(flat_list, padding='post', maxlen=maxlen)
    x = binary_model.predict(example)[0]
    print(i, x, labels[i])
    if labels[i] == 1:
        count_positives += 1
    else:
        count_negatives += 1
    if x > 0.5:
        positive.append(example)
    else:
        negative.append(example)
    if x > 0.5:
        if labels[i] == 1:
            correct += 1
    else:
        if labels[i] == 0:
            correct += 1


print(len(positive) / count_positives)
print(len(negative) / count_negatives)
print(correct / len(labels))





