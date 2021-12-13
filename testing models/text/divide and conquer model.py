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
import tensorflow as tf
import textaug

inputs_7_cat, labels_7_cat = loading.load_data_7()
inputs, labels = loading.load_data_binary()
assert len(inputs) == len(labels)
labels2 = labels[:]
data = np.loadtxt('simple_swap_augdata.txt', delimiter='\n', dtype=str)[2199:]
print('done')


X_train, y_train, test_data, labels, embedding_matrix, \
    vocab_size, maxlen, t, unpadded, untokenized = loading.set_up_data(data, inputs, labels)
negative = []
positive = []

binary_model = keras.models.load_model("binary simple swap - LSTM-512.h5", compile=True)
#binary_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)])
# binary_model.fit(X_train, y_train, batch_size=512, epochs=6, verbose=1,
#                     validation_split=0.2)
score = binary_model.evaluate(test_data, labels, verbose=1)
print(test_data)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])
count_positives = 0
count_negatives = 0
correct = 0
closecount = 0
predicted_labs = []
for i in range(len(test_data)):
    example = test_data[i]
    example = tf.expand_dims(example, axis=0)
    #print(example)
    # example = t.texts_to_sequences(example)
    # flat_list = []
    # for sublist in example:
    #     for item in sublist:
    #         flat_list.append(item)
    #
    # flat_list = [flat_list]
    #
    # example = pad_sequences(flat_list, padding='post', maxlen=maxlen)
    # try:
    #     assert example.shape == (14,)
    # except AssertionError:
    #     print("bad shape", example.shape)
    x = binary_model.predict(example)[0]
    if i == 0:
        print('test 2')
    if labels[i] == 1:
        count_positives += 1
    else:
        count_negatives += 1
    if x > 0.5:
        predicted_labs.append(1)
        positive.append(example)
    else:
        predicted_labs.append(0)
        negative.append(example)
    if x > 0.5:
        if labels[i] == 1:
            correct += 1
    else:
        if labels[i] == 0:
            correct += 1
    if abs(x - 0.5) <= 0.1:
        closecount += 1


print(count_positives, len(positive) / count_positives)
print(count_negatives, len(negative) / count_negatives)
print(correct, len(labels))
print(correct / len(labels))
print(closecount)

negative_model_1 = keras.models.load_model("binary simple swap - LSTM-512x2 - (n3,n2) cat.h5", compile=True)
negative_model_2 = keras.models.load_model("binary simple swap - LSTM-512x2 - (n2,n1) cat.h5", compile=True)

positive_model_1 = keras.models.load_model("binary simple swap - LSTM-512x2 - (1,2) cat.h5", compile=True)
positive_model_2 = keras.models.load_model("binary simple swap - LSTM-512x2 - (2,3) cat.h5", compile=True)


correctcount = 0
for i in range(min(len(positive), len(negative))):
    print(i)

    p1 = positive_model_1.predict(positive[i])[0]
    p2 = positive_model_2.predict(positive[i])[0]
    n1 = negative_model_1.predict(negative[i])[0]
    n2 = negative_model_2.predict(negative[i])[0]

    p1_lab = 0
    p2_lab = 0
    n1_lab = 0
    n2_lab = 0

    p_lab = 0
    n_lab = 0

    total_lab = 0

    if p1 > 0.5:
        p1_lab = 2
    else:
        p1_lab = 1

    if p2 > 0.5:
        p2_lab = 3
    else:
        p2_lab = 2

    if n1 > 0.5:
        n1_lab = -2
    else:
        n1_lab = -3

    if n2 > 0.5:
        n2_lab = -1
    else:
        n2_lab = -2

    if p1_lab == p2_lab:
        p_lab = 2
    elif p1_lab == 1 and p2_lab == 2:
        p_lab = 1
    elif p1_lab == 2 and p2_lab == 3:
        p_lab = 2

    if n1_lab == n2_lab:
        n_lab = -2
    elif n1_lab == -3 and n2_lab == -2:
        n_lab = -3
    elif n1_lab == -2 and n2_lab == -1:
        n_lab = -1

    if n_lab == -1 and p_lab == 1:
        lab = 0
    else:
        if predicted_labs[i] == 1:
            lab = p_lab
        else:
            lab = n_lab

    if lab == labels_7_cat[i]:
        correctcount += 1

print(correctcount / min(len(positive), len(negative)))
