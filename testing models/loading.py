import numpy as np
from numpy.core.multiarray import ndarray

import textaug
from mmsdk import mmdatasdk
import json
import random
import re
import loading
from keras.layers import Conv1D, Conv2D, Flatten, LSTM, MaxPooling1D, \
    MaxPooling2D, TimeDistributed
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split



def load_data_normal():
    mydictLabels = {
        'myfeaturesLabels': '../../cmumosi/CMU_MOSI_Opinion_Labels.csd'}
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
            mydatasetLabels.computational_sequences['myfeaturesLabels'].data[
                key][
                'features'])

    textInput = np.zeros(totalSegments, dtype=object)
    labelInput = np.zeros(totalSegments)
    segmentCounter = 0
    for key in mydatasetLabels.computational_sequences[
        'myfeaturesLabels'].data.keys():
        textPath = '../../raw/Raw/Transcript/Segmented/%s.annotprocessed' % (
            key)
        with open(textPath) as file:  # Use file to refer to the file object
            text = file.read()
            text = text.replace("_DELIM_", "")
            text = text.split("\n")
            for segment in range(len(mydatasetLabels.computational_sequences[
                                         'myfeaturesLabels'].data[key][
                                         'features'])):
                labelInput[segmentCounter] = \
                    mydatasetLabels.computational_sequences[
                        'myfeaturesLabels'].data[
                        key]['features'][segment]
                text[segment] = ''.join(
                    [i for i in text[segment] if not i.isdigit()])
                textInput[segmentCounter] = text[segment]
                segmentCounter += 1

    newInput = [sentence.lower() for sentence in textInput]
    return newInput, labelInput


def load_data_binary():
    newInput, labelInput = load_data_normal()
    labels = [1 if labelInput[i] > 0 else 0 for i in range(len(labelInput))]

    return newInput, labels


def load_data_7():
    newInput, labelInput = load_data_normal()
    labels = [round(labelInput[i]) for i in range(len(labelInput))]

    return newInput, labels


def load_data_from_categories(categories: list, data: list, labels: list):
    """
    assumed that data and labels are the same length
    and will be performed only after "set_up_data" is used

    :param categories:  list, length 2
    :param data: data to be sorted
    :param labels: labels correspoinding to data
    :return: sorted data with corresponding labels
    """
    #newInput, labelInput = load_data_7()
    data2 = []
    labels2 = []
    i = 0
    for i in range(len(data)):
        if labels[i] in categories:
            data2.append(data[i])
            labels2.append(labels[i])
    labels2 = [0 if item == categories[0] else 1 for item in labels2]

    return np.array(data2), np.array(labels2)


def preprocess_text(sen):
    """
    taken from https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/
    :param sen:
    :return:
    """
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


TAG_RE = re.compile(r'<[^>]+>')


def set_up_data(data: ndarray, inputs: list, labels: list):
    """
    tokenization, paddings, and embedding setup steps taken from
    https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/


    :param data:
    :param inputs:
    :param labels:
    :return:
    """
    labels2 = labels[:]
    X = []

    labels2 = np.array(labels2)
    data, labels2 = textaug.remove_duplicates(data.tolist(), labels2.tolist())
    print(len(data), len(labels2), 'done!')

    for sen in data:
        X.append(preprocess_text(sen))

    test_data = []
    for sen in inputs:
        test_data.append(preprocess_text(sen))

    textaug.clean_test_train_data(X, test_data, labels2)

    test_data = np.array(test_data)
    labels = np.array(labels)

    y = labels2

    X = np.array(X)
    inputs = np.array(inputs)

    assert len(X) == len(y)
    assert len(inputs) == len(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01,
                                                        random_state=42)

    # _, test_data, _2, test_labels = train_test_split(inputs, labels, test_size=0.99, random_state=42)
    # assert len(test_data) == len(test_labels)
    # print(test_data)
    # print(test_labels)

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    test_data = tokenizer.texts_to_sequences(test_data)
    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    maxlen = 14

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    test_data = pad_sequences(test_data, padding='post', maxlen=maxlen)

    embeddings_dictionary = dict()
    glove_file = open('../../glove.6B/glove.6B.100d.txt', encoding="utf8")

    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        if vector_dimensions.shape == (100,):
            embeddings_dictionary[word] = vector_dimensions
    glove_file.close()

    embedding_matrix = np.zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            if embedding_vector.shape != (100,):
                print(embedding_vector.shape)
            assert embedding_vector.shape == (100,)
            embedding_matrix[index] = embedding_vector

    return X_train, np.array(
        y_train), test_data, labels, embedding_matrix, vocab_size, maxlen, tokenizer




