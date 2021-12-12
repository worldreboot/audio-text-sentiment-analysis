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
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

import textaug

inputs, labels = loading.load_data()
assert len(inputs) == len(labels)
labels2 = labels[:]
data = np.loadtxt('simple_swap_augdata.txt', delimiter='\n', dtype=str)[2199:]
print('done')
data, labels2 = textaug.remove_duplicates(data.tolist(), labels2)
print(len(data))

data = np.array(data)
labels2 = np.array(labels2)

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
sentences = list(data)
for sen in sentences:
    X.append(preprocess_text(sen))

test_data = []
for sen in inputs:
    test_data.append(preprocess_text(sen))
    
textaug.clean_test_train_data(X, test_data, labels)

test_data = np.array(test_data)
labels = np.array(labels)

y = labels2



X = np.array(X)
inputs = np.array(inputs)
labels = np.array(labels)
assert len(X) == len(y)
assert len(inputs) == len(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#_, test_data, _2, test_labels = train_test_split(inputs, labels, test_size=0.99, random_state=42)
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
model.add(LSTM(512, return_sequences=True))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Flatten())
# model.add(Dense(1, activation='sigmoid'))
#model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
# model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
# model.add(TimeDistributed(Flatten()))
# define LSTM model
model.add(LSTM(512))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, batch_size=512, epochs=6, verbose=1, validation_split=0.2)
model.save('binary simple swap - LSTM-512.h5')
score = model.evaluate(test_data, labels, verbose=1)
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
