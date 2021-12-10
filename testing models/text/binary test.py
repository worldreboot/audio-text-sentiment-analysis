import csv
import json
import random

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array, os
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
from mmsdk import mmdatasdk


mydictLabels = {'myfeaturesLabels': '../../cmumosi/CMU_MOSI_Opinion_Labels.csd'}
# cmumosi_highlevel=mmdatasdk.mmdataset(mmdatasdk.cmu_mosi.highlevel['glove_vectors'],'cmumosi/')
# mydictText = {'myfeaturesText':'cmumosi/CMU_MOSI_TimestampedWordVectors.csd'}
mydatasetLabels = mmdatasdk.mmdataset(mydictLabels)
# mydatasetText = mmdatasdk.mmdataset(mydictText)
# print(mydataset.computational_sequences['myfeatures'].data)


# Get text with labels
totalSegments = 0
for key in mydatasetLabels.computational_sequences[
    'myfeaturesLabels'].data.keys():
    totalSegments += len(
        mydatasetLabels.computational_sequences['myfeaturesLabels'].data[key][
            'features'])

textInput = np.zeros(totalSegments, dtype=object)
labelInput = np.zeros(totalSegments)
segmentCounter = 0
for key in mydatasetLabels.computational_sequences[
    'myfeaturesLabels'].data.keys():
    textPath = '../../raw/Raw/Transcript/Segmented/%s.annotprocessed' % (key)
    with open(textPath) as file:  # Use file to refer to the file object
        text = file.read()
        text = text.replace("_DELIM_", "")
        text = text.split("\n")
        for segment in range(len(mydatasetLabels.computational_sequences[
                                     'myfeaturesLabels'].data[key][
                                     'features'])):
            labelInput[segmentCounter] = \
            mydatasetLabels.computational_sequences['myfeaturesLabels'].data[
                key]['features'][segment]
            text[segment] = ''.join(
                [i for i in text[segment] if not i.isdigit()])
            textInput[segmentCounter] = text[segment]
            segmentCounter += 1

maxLength = 13
newInput = np.array([sentence.lower() for sentence in textInput])
labels = np.array([1 if item > 0 else 0 for item in labelInput])

X_train, X_test, y_train, y_test = train_test_split(newInput, labels,
                                                    test_size=0.95,
                                                    random_state=42)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 14

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


embeddings_dictionary = dict()
glove_file = open('../../glove.6B.100d.txt',
                  encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = np.zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

model = keras.models.load_model('binary - 128.h5')
score = model.evaluate(X_test, y_test, verbose=1)
