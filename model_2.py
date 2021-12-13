import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, load_model

class CNN:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = None
        self.initialize_layers()
    
    def initialize_layers(self):
        self.model = tf.keras.Sequential()
        self.model.add(layers.Conv1D(64, kernel_size=(10), activation='relu', input_shape=(self.input_size, 1)))
        self.model.add(layers.Conv1D(128, kernel_size=(10),activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        self.model.add(layers.MaxPooling1D(pool_size=(8)))
        self.model.add(layers.Dropout(0.4))
        self.model.add(layers.Conv1D(128, kernel_size=(10),activation='relu'))
        self.model.add(layers.MaxPooling1D(pool_size=(8)))
        self.model.add(layers.Dropout(0.4))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(256, activation='relu'))
        self.model.add(layers.Dropout(0.4))
        self.model.add(layers.Dense(self.output_size, activation='softmax'))
        opt = keras.optimizers.Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt,metrics=['accuracy'])
        print(self.model.summary())