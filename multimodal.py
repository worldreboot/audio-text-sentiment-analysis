import numpy as np



class Perceptron:

    """
    contains multiple regression techniques on multimodal data
    """


    def __init__(self, audio, text, labels):
        self.audioInput = audio
        self.textInput = text
        self.labels = labels
        self.theta = np.zeros((2,))

        # combines audio and text input into a single feature vector
        self.features = np.vstack((self.audioInput, self.textInput)).T

    def gradient_descent(self):




