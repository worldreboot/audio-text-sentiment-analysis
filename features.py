import sklearn
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile as sp
import scipy.fft as fourier
import os
import numpy as np


def load_train_data(directory):
    data = []
    for filename in os.listdir(directory):
        row = []
        rate, audio = sp.read(filename)
        # gets rid of silent start and end sections of files
        # plus gets absolute value of amplitute (what matters for loudness)
        audio = np.abs(np.trim_zeros(audio))
        # amplitute (loudness) of speaker
        minamp = min(audio)
        maxamp = max(audio)
        row.append(maxamp - minamp)
        # the fourier transform
        freqamp = np.abs(fourier.rfft(audio))
        #frequency bins
        frequencies = fourier.rfftfreq(len(audio),  1/rate)
        amp = freqamp / sum(freqamp)
        mean = sum(frequencies * amp)
        row.append(mean)
        # most promenent frequency
        row.append(np.argmax(freqamp))


        data.append(row)
    data = np.array(data)
    return data






