import json
import random
import re

import numpy as np
from keras.layers import Conv1D, Conv2D, Flatten, LSTM, MaxPooling1D, \
    MaxPooling2D, TimeDistributed
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from numpy import os
from sklearn.model_selection import train_test_split

X_data = []
labels = []
binary_labels = []
exists = False
counts = [0, 0, 0, 0, 0]
if not os.path.isfile('data.txt'):
    exists = True
    with open('../../Books_5.json') as f:
        for line in f:
            entry = json.loads(line)
            rating = int(entry['overall'])
            if int(entry['overall']) != 3:
                if counts[rating - 1] < 25000:
                    X_data.append(entry['reviewText'])
                    labels.append(int(entry['overall']))
                    counts[rating - 1] += 1
            if len(X_data) > 100000:
                break
    f.close()

    X_data = np.array(X_data)
    binary_labels = []
    random.seed(10)
    for item in labels:
        if item < 3:
            binary_labels.append(0)
        elif item > 3:
            binary_labels.append(1)
        elif item == 3:
            x = random.randint(0, 1)
            if x == 1:
                binary_labels.append(1)
            elif x == 0:
                binary_labels.append(0)

    binary_labels = np.array(binary_labels)
    np.savetxt('data.txt', X_data, fmt='%s')
    np.savetxt('binary_labels.txt', binary_labels, fmt='%d')


if not exists:
    X_data = np.loadtxt('data.txt', dtype=str, delimiter='\n')
    binary_labels = np.loadtxt('binary_labels.txt', dtype=int, delimiter='\n')
    binary_labels = binary_labels[:99991]
    assert X_data.shape == binary_labels.shape


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):

    return TAG_RE.sub('', text)

X = []
sentences = list(X_data)
for sen in sentences:
    X.append(preprocess_text(sen))

y = binary_labels


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

maxlen = 100

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

embeddings_dictionary = dict()
glove_file = open('../../glove.6B.100d/glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    if vector_dimensions.shape == (100, ):
        embeddings_dictionary[word] = vector_dimensions
glove_file.close()

embedding_matrix = np.zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        if embedding_vector.shape != (100, ):
            print(embedding_vector.shape)
        assert embedding_vector.shape == (100, )
        embedding_matrix[index] = embedding_vector

model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
model.add(embedding_layer)
# model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
# model.add(LSTM(128))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
# model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
# model.add(TimeDistributed(Flatten()))
# define LSTM model
model.add(LSTM(128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, batch_size=256, epochs=6, verbose=1, validation_split=0.2)
model.save('binary conv recc - 128.h5')
score = model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
