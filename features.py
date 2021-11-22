import sklearn
import matplotlib.pyplot as plt
import librosa
import scipy.io.wavfile as sp
import scipy.fft as fourier
import os
import numpy as np
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

def trim_and_abs(arr):
    index = 0
    while index < len(arr) and arr[index] == 0:

        index += 1
    index2 = len(arr) - 1
    while index2 >= 0 and arr[index] == 0:
        index2 -= 1

    r = arr[index: index2 + 1]
    for i in range(len(r)):
        r[i] = abs(r[i])
    return r

def spectral_properties(y: np.ndarray, fs: int):
    arr = []
    spec = np.abs(np.fft.rfft(y))
    freq = np.fft.rfftfreq(len(y), d=1 / fs)
    spec = np.abs(spec)
    amp = spec / spec.sum()
    mean = (freq * amp).sum()
    arr.append(mean)
    sd = np.sqrt(np.sum(amp * ((freq - mean) ** 2)))
    arr.append(sd)
    amp_cumsum = np.cumsum(amp)
    median = freq[len(amp_cumsum[amp_cumsum <= 0.5]) + 1]
    arr.append(median)
    mode = freq[amp.argmax()]
    arr.append(mode)
    Q25 = freq[len(amp_cumsum[amp_cumsum <= 0.25]) + 1]
    arr.append(Q25)
    Q75 = freq[len(amp_cumsum[amp_cumsum <= 0.75]) + 1]
    arr.append(Q75)
    IQR = Q75 - Q25
    arr.append(IQR)
    z = amp - amp.mean()
    w = amp.std()
    skew = ((z ** 3).sum() / (len(spec) - 1)) / w ** 3
    arr.append(skew)
    kurt = ((z ** 4).sum() / (len(spec) - 1)) / w ** 4
    arr.append(kurt)
    return arr

def shift_pitch(soundFile, shift):
    audio, sr = soundFile
    return librosa.effects.pitch_shift(audio, sr, shift)


def load_train_data(dir):
    data = []
    y = []
    for directory in os.listdir(dir):
        for filename in os.listdir(dir + "/" + directory):
            parameters = filename.split("-")
            emotion = int(parameters[2])
            row = []
            file = dir + "/" + directory + "/" + filename

            audio, rate = sp.read(file)

            # some files are mono and some are stereo for some reason
            if len(audio.shape) > 1:
                if audio.shape[1] == 2:
                    audio = audio[:, 0]

            i = 0.5
            while i < 8:
                row.append(shift_pitch((aduio, rate), i))
            #row = spectral_properties(audio, rate)
            # gets rid of silent start and end sections of files
            # plus gets absolute value of amplitute (what matters for loudness)
            #audio = trim_and_abs(audio)
            # amplitute (loudness) of speaker
            # minamp = min(audio)
           # maxamp = max(audio)
            #row.append(maxamp - minamp)
            # the fourier transform
           # freqamp = np.abs(fourier.rfft(audio))
            #frequency bins
            # most promenent frequency
            #row.append(np.argmax(freqamp))
           # y.append(emotion)

            data.append(row)
    data = np.array(data)
    y = np.array(y)
    return data, y




# path = 'datasets/archive/audio_speech_actors_01-24'
# X, y = load_train_data(path)
#
# model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
# # define the model evaluation procedure
# cv = RepeatedStratifiedKFold()
# # evaluate the model and collect the scores
# n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# # report the model performance
#
# # 23
# print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))
