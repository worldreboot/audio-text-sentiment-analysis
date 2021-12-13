import json
import random
import re
# import loading
import numpy as np
# from keras.layers import Conv1D, Conv2D, Flatten, LSTM, MaxPooling1D, \
    # MaxPooling2D, TimeDistributed
# from keras.layers.core import Dense
# from keras.layers.embeddings import Embedding
# from keras.models import Sequential
# import keras
# from keras.preprocessing.sequence import pad_sequences
# from keras.preprocessing.text import Tokenizer
# from sklearn.model_selection import train_test_split

import os
import numpy as np
import pandas as pd
from audio import convert_audio_to_log_mel_spectrogram
import model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import sys

from tensorflow.keras.utils import to_categorical

# import textaug

from mmsdk import mmdatasdk

import os

import model_2

def numpy_fillna(data):
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])

    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]

    # Setup output array and put elements from data into masked positions
    out = np.zeros(mask.shape)
    out[mask] = np.concatenate(data)
    return out

mydictLabels={'myfeaturesLabels':'cmumosi/CMU_MOSI_Opinion_Labels.csd'}
mydatasetLabels = mmdatasdk.mmdataset(mydictLabels)

audio = 'datasets/CMU_MOSI/Audio/WAV_16000/Segmented'
actor_folders = os.listdir(audio) # list files in audio directory
actor_folders.sort()

emotion = []
file_path = []

filename = os.listdir(audio) # iterate over Actor folders
for f in filename: # go through files in Actor folder
    part = f.split('.')[0].split('_')
    print(part)
    clip = ""
    for i in range(len(part) - 1):
        if i > 0:
            clip += "_"
        clip += part[i]
    segment = int(part[len(part) - 1]) - 1
    print(clip + " " + str(segment))
    if int(mydatasetLabels.computational_sequences['myfeaturesLabels'].data[clip]['features'][segment][0]) < 0:
        emotion.append(0)
    else:
        emotion.append(1)
    # emotion.append(if int(mydatasetLabels.computational_sequences['myfeaturesLabels'].data[clip]['features'][segment][0]) < 0: 0 else: 1)
    print(mydatasetLabels.computational_sequences['myfeaturesLabels'].data[clip]['features'][segment][0])
    file_path.append(audio + '/' + f)

print("Loading CMU dataset...")
log_mels = []
for index, path in enumerate(file_path):
    # print(path)
    log_spectrogram = convert_audio_to_log_mel_spectrogram(path)
    log_mels.append(log_spectrogram)

log_mels = numpy_fillna(log_mels)
print("Finished loading CMU dataset")

lb = LabelEncoder()
emotion_one_hot = to_categorical(lb.fit_transform(emotion))

# def gen():
#     for i in range(len(log_mels)):
#         yield log_mels[i], emotion_one_hot[i]

# dataset = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(128, None), dtype=tf.float32, name=None), tf.TensorSpec(shape=(7,), dtype=tf.float32)))

# print(dataset)
# print(dataset.element_spec)

# train_dataset = dataset.take(1650)
# test_dataset = dataset.skip(1650)
# batches = train_dataset.padded_batch(8)
# test_batches = test_dataset.padded_batch(8)

# cnn = model.CNN(0, 1)

# log_mels = np.array(log_mels)
# log_mels = log_mels[:, :, np.newaxis]

combined = [[log_mels[i], emotion_one_hot[i]] for i in range(len(log_mels))]

train, test = train_test_split(combined, test_size=0.2, random_state=0)

x_train = [sample[0] for sample in train]
y_train = [sample[1] for sample in train]

x_test = [sample[0] for sample in test]
y_test = [sample[1] for sample in test]

cnn = model_2.CNN(x_train.shape[1], 8)
cnn.model.load_weights('best_initial_model_3.hdf5')

# load data here
X_data =[]
y_data = []
X_test = []
y_test = []

model = Sequential()
model.add(Dense(9))
model.add(LSTM(128, activation='relu'))
model.add(Dense(7, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_data, y_data, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

model.save('mutlimodal.h5')
score = model.evaluate(X_test, y_test, verbose=1)
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
