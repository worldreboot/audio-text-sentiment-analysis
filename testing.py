import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

import utils.audio as audio
import utils.dataset as dataset

from tensorflow import keras
from tensorflow.keras import layers

import model

filename = 'datasets/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav'
# filename = 'datasets/RAVDESS/Actor_01/03-01-03-01-01-02-01.wav'
# dataset_path = 'datasets/RAVDESS/audio_speech_actors_01-24/'

# dataset.load_ravdess_dataset(dataset_path)

print(audio.convert_audio_to_log_mel_spectrogram(filename))

# plt.colorbar(format='%+2.0f dB')
# plt.savefig('plot.png')

# model.tfb()

cnn = model.CNN(0, 4)
