import librosa
import tensorflow as tf
import numpy as np
# import librosa.display

def convert_audio_to_log_mel_spectrogram(filename, n_mels=128, hop_length=345, n_fft=1380):
    y, sr = librosa.load(filename, sr=44100)
    
    # trim silent edges
    audio, _ = librosa.effects.trim(y)

    S = librosa.feature.melspectrogram(audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    # return tf.constant(S_DB)
    log_spectrogram = np.mean(S_DB, axis = 0)

    return tf.constant(log_spectrogram)
    
    