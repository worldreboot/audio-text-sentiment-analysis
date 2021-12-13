import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

import utils.audio as audio
import utils.dataset as dataset

from tensorflow import keras
from tensorflow.keras import layers

import sys

import model

filename = 'datasets/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav'
# filename = 'datasets/RAVDESS/Actor_01/03-01-03-01-01-02-01.wav'
dataset_path = 'datasets/RAVDESS/audio_speech_actors_01-24/'

cmu_dataset = 'datasets/CMU_MOSI/Audio/WAV_16000/Segmented'

# dataset.load_ravdess_dataset(dataset_path)
dataset.load_cmu_dataset(cmu_dataset)

# np.set_printoptions(threshold=sys.maxsize)
# with open('test.txt', 'w') as f:
#     sys.stdout = f # Change the standard output to the file we created.
#     print(audio.convert_audio_to_log_mel_spectrogram(filename))
#     sys.stdout = sys.original_stdout # Reset the standard output to its original value
# print(audio.convert_audio_to_log_mel_spectrogram(filename))

# plt.colorbar(format='%+2.0f dB')
# plt.savefig('plot.png')

# model.tfb()

# cnn = model.CNN(0, 4)
